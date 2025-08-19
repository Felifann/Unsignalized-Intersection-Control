import time
import math
from typing import Dict, List, Set, Any, Tuple
from env.simulation_config import SimulationConfig

class TrafficController:
    """
    基于拍卖结果的统一交通控制器 - 支持车队和单车
    核心思想：所有控制都基于拍卖获胜者的优先级排序
    """
    
    def __init__(self, carla_wrapper, state_extractor, max_go_agents: int = None):
        self.carla = carla_wrapper
        self.state_extractor = state_extractor
        self.world = carla_wrapper.world
        self.traffic_manager = carla_wrapper.client.get_trafficmanager()
        
        # 添加交叉口中心和检测区域配置
        self.intersection_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        self.intersection_half_size = SimulationConfig.INTERSECTION_HALF_SIZE
        
        # 控制参数
        self.default_speed_diff = -40.0  # 默认速度差异
        self.default_follow_distance = 1.5  # 默认跟车距离

        # 控制状态跟踪
        self.controlled_vehicles: Dict[str, Dict] = {}
        self.current_controlled_vehicles: Set[str] = set()
        self.platoon_manager = None
        
        # Add configurable max go agents limit (can be None)
        self.max_go_agents = max_go_agents
        
        # Statistics tracking
        self.total_vehicles_controlled = 0  # Total number of vehicles ever controlled
        self.vehicles_exited_intersection = 0  # Number of vehicles that exited intersection
        self.control_history = {}  # Track when vehicles entered/exited control
        
        # Enhanced: Separate positive/negative acceleration tracking with simulation time
        self.acceleration_data = {
            'positive': {},  # {vehicle_id: [positive_acceleration_values]}
            'negative': {},  # {vehicle_id: [negative_acceleration_values]}
            'absolute': {}   # {vehicle_id: [absolute_acceleration_values]} for backward compatibility
        }
        # Archive completed vehicles' acceleration history so final stats keep samples
        self.archived_acceleration_data = {
            'positive': {},
            'negative': {},
            'absolute': {}
        }
        self.previous_velocities = {}  # {vehicle_id: previous_velocity_vector}
        self.previous_sim_timestamps = {}  # {vehicle_id: previous_simulation_timestamp}
        
        # Acceleration filtering parameters
        self.accel_filter_config = {
            'min_time_delta': 0.01,  # Skip samples with very small time differences
            'max_acceleration': 15.0,  # Truncate extreme acceleration values (m/s²)
            'use_median_filter': True,  # Apply median filtering
            'median_window_size': 5    # Window size for median filter
        }
        
        limit_text = "unlimited" if max_go_agents is None else str(max_go_agents)
        print(f"🎮 增强交通控制器初始化完成 - 支持车队、单车 (max go agents: {limit_text})")

    def set_platoon_manager(self, platoon_manager):
        """Set platoon manager reference"""
        self.platoon_manager = platoon_manager
        print("🔗 车队管理器已连接到交通控制器")

    # Add method to update configuration
    def update_max_go_agents(self, max_go_agents: int = None):
        """Update the maximum go agents limit"""
        self.max_go_agents = max_go_agents
        limit_text = "unlimited" if max_go_agents is None else str(max_go_agents)
        print(f"🔄 Traffic controller: Updated MAX_GO_AGENTS to {limit_text}")

    def update_control(self, platoon_manager=None, auction_engine=None, direct_winners=None):
        """主控制更新函数"""
        if platoon_manager:
            self.platoon_manager = platoon_manager
        
        # 1. Maintain intersection vehicle control
        current_controlled = self._maintain_intersection_vehicle_control()
        
        # 2. Apply auction-based control - use direct winners if provided
        auction_winners = direct_winners or (auction_engine.get_current_priority_order() if auction_engine else [])
        if auction_winners:
            auction_controlled = self._apply_auction_based_control(
                auction_winners, platoon_manager
            )
        else:
            auction_controlled = set()
        
        current_controlled.update(auction_controlled)
        
        # 3. Update acceleration data for currently controlled vehicles
        self._update_acceleration_data(current_controlled)
        
        # 4. 恢复不再被控制的车辆 (using expanded vehicle state detection)
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 5. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _update_acceleration_data(self, controlled_vehicles: Set[str]):
        """Update acceleration data for controlled vehicles using simulation time and separate positive/negative tracking"""
        # Get current simulation timestamp
        current_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        vehicle_states = self.state_extractor.get_vehicle_states()
        vehicle_lookup = {str(v['id']): v for v in vehicle_states}
        
        for vehicle_id in controlled_vehicles:
            if vehicle_id in vehicle_lookup:
                vehicle_state = vehicle_lookup[vehicle_id]
                
                try:
                    # Get current velocity - handle both dict and tuple formats
                    velocity_data = vehicle_state.get('velocity', (0, 0, 0))
                    
                    if isinstance(velocity_data, dict):
                        # Dictionary format: {'x': val, 'y': val, 'z': val}
                        current_velocity_x = velocity_data.get('x', 0)
                        current_velocity_y = velocity_data.get('y', 0) 
                        current_velocity_z = velocity_data.get('z', 0)
                    elif isinstance(velocity_data, (tuple, list)) and len(velocity_data) >= 3:
                        # Tuple/list format: (x, y, z)
                        current_velocity_x = velocity_data[0]
                        current_velocity_y = velocity_data[1]
                        current_velocity_z = velocity_data[2]
                    else:
                        # Fallback for unexpected format
                        current_velocity_x = current_velocity_y = current_velocity_z = 0
                    
                    current_speed = math.sqrt(current_velocity_x**2 + current_velocity_y**2 + current_velocity_z**2)
                    
                    # Calculate acceleration if we have previous data
                    if vehicle_id in self.previous_velocities and vehicle_id in self.previous_sim_timestamps:
                        prev_velocity = self.previous_velocities[vehicle_id]
                        prev_sim_timestamp = self.previous_sim_timestamps[vehicle_id]
                        
                        # Handle previous velocity format consistently
                        if isinstance(prev_velocity, dict):
                            prev_speed = math.sqrt(prev_velocity['x']**2 + prev_velocity['y']**2 + prev_velocity['z']**2)
                        else:
                            prev_speed = math.sqrt(prev_velocity[0]**2 + prev_velocity[1]**2 + prev_velocity[2]**2)
                        
                        time_delta = current_sim_time - prev_sim_timestamp
                        
                        # Filter: Skip if time delta is too small
                        if time_delta >= self.accel_filter_config['min_time_delta']:
                            # Calculate raw acceleration (can be positive or negative)
                            raw_acceleration = (current_speed - prev_speed) / time_delta
                            
                            # Truncate extreme values
                            max_accel = self.accel_filter_config['max_acceleration']
                            truncated_acceleration = max(-max_accel, min(max_accel, raw_acceleration))
                            
                            # Initialize acceleration lists if needed
                            for accel_type in ['positive', 'negative', 'absolute']:
                                if vehicle_id not in self.acceleration_data[accel_type]:
                                    self.acceleration_data[accel_type][vehicle_id] = []
                            
                            # Store acceleration data separately by sign
                            if truncated_acceleration > 0:
                                self.acceleration_data['positive'][vehicle_id].append(truncated_acceleration)
                            elif truncated_acceleration < 0:
                                # store negative accelerations as negative values so sign is preserved
                                self.acceleration_data['negative'][vehicle_id].append(truncated_acceleration)
                            
                            # Also store absolute value for backward compatibility
                            self.acceleration_data['absolute'][vehicle_id].append(abs(truncated_acceleration))
                            
                            # Apply median filtering if enabled
                            if self.accel_filter_config['use_median_filter']:
                                self._apply_median_filter(vehicle_id)
                    
                    # Update previous data - store as tuple for consistency
                    self.previous_velocities[vehicle_id] = (current_velocity_x, current_velocity_y, current_velocity_z)
                    self.previous_sim_timestamps[vehicle_id] = current_sim_time
                    
                except Exception as e:
                    print(f"[Warning] 计算车辆 {vehicle_id} 加速度失败: {e}")
                    # Debug: Print velocity data format for troubleshooting
                    try:
                        velocity_debug = vehicle_state.get('velocity', 'NOT_FOUND')
                        print(f"[Debug] 车辆 {vehicle_id} 速度数据格式: {type(velocity_debug)} = {velocity_debug}")
                    except:
                        pass

    def _apply_median_filter(self, vehicle_id: str):
        """Apply median filtering to the most recent acceleration samples"""
        window_size = self.accel_filter_config['median_window_size']
        
        for accel_type in ['positive', 'negative', 'absolute']:
            if vehicle_id in self.acceleration_data[accel_type]:
                accel_list = self.acceleration_data[accel_type][vehicle_id]
                
                # Apply median filter only if we have enough samples
                if len(accel_list) >= window_size:
                    # Get the most recent window
                    recent_window = accel_list[-window_size:]
                    
                    # Calculate median of the window
                    sorted_window = sorted(recent_window)
                    n = len(sorted_window)
                    if n % 2 == 0:
                        median_value = (sorted_window[n//2 - 1] + sorted_window[n//2]) / 2
                    else:
                        median_value = sorted_window[n//2]
                    
                    # Replace the most recent value with the median
                    accel_list[-1] = median_value

    def _calculate_average_acceleration(self) -> Dict[str, float]:
        """Calculate average acceleration for positive, negative, and absolute values with separate absolute averages"""
        results = {}
        
        for accel_type in ['positive', 'negative', 'absolute']:
            all_accelerations = []
            
            # include both currently tracked and archived samples
            for vehicle_id, accelerations in self.acceleration_data[accel_type].items():
                all_accelerations.extend(accelerations)
            for vehicle_id, accelerations in self.archived_acceleration_data[accel_type].items():
                all_accelerations.extend(accelerations)
            
            if all_accelerations:
                # Calculate average of absolute values for positive and negative separately
                if accel_type in ['positive', 'negative']:
                    # For positive/negative, calculate average of absolute values
                    results[f'average_absolute_{accel_type}_acceleration'] = sum(abs(a) for a in all_accelerations) / len(all_accelerations)
                    results[f'average_{accel_type}_acceleration'] = sum(all_accelerations) / len(all_accelerations)
                else:
                    # For absolute, keep original behavior
                    results[f'average_{accel_type}_acceleration'] = sum(all_accelerations) / len(all_accelerations)
                
                results[f'{accel_type}_acceleration_samples'] = len(all_accelerations)
                results[f'{accel_type}_acceleration_vehicles'] = len(self.acceleration_data[accel_type])
            else:
                if accel_type in ['positive', 'negative']:
                    results[f'average_absolute_{accel_type}_acceleration'] = 0.0
                    results[f'average_{accel_type}_acceleration'] = 0.0
                else:
                    results[f'average_{accel_type}_acceleration'] = 0.0
                
                results[f'{accel_type}_acceleration_samples'] = 0
                results[f'{accel_type}_acceleration_vehicles'] = 0
        
        return results

    def _maintain_intersection_vehicle_control(self) -> Set[str]:
        """维持路口内车辆的控制"""
        maintained_vehicles = set()
        vehicle_states = self.state_extractor.get_vehicle_states()
        
        for vehicle_state in vehicle_states:
            vehicle_id = str(vehicle_state['id'])
            
            # 如果车辆在路口内且之前被控制，继续维持控制
            if (vehicle_state.get('is_junction', False) and 
                vehicle_id in self.controlled_vehicles):
                
                # 确保控制仍然有效
                if self._apply_single_vehicle_control(
                    vehicle_id, 
                    self.controlled_vehicles[vehicle_id]['rank'],
                    0.0,  # bid_value
                    'go'  # 路口内车辆应该继续通行
                ):
                    maintained_vehicles.add(vehicle_id)
        
        return maintained_vehicles

    def _get_control_action_by_rank(self, rank: int) -> str:
        """根据排名获取控制动作"""
        if self.max_go_agents is None:
            return 'go'  # No limit, everyone can go
        elif rank <= self.max_go_agents:
            return 'go'  # Within limit, can go
        else:
            return 'wait'  # Beyond limit, must wait

    def _apply_auction_based_control(self, auction_winners: List, platoon_manager=None) -> Set[str]:
        """Apply control based on auction results with traffic flow control awareness"""
        controlled_vehicles = set()
        
        if not auction_winners:
            return controlled_vehicles
        
        print(f"🚦 Applying auction control to {len(auction_winners)} winners:")
        
        # First pass: Identify and protect vehicles already in transit
        in_transit_vehicles = set()
        vehicle_states = self.state_extractor.get_vehicle_states()
        vehicle_lookup = {str(v['id']): v for v in vehicle_states}
        
        for vehicle_id, vehicle_state in vehicle_lookup.items():
            if vehicle_state.get('is_junction', False) and vehicle_id in self.controlled_vehicles:
                # Vehicle is in intersection and was previously controlled - protect it
                if self.controlled_vehicles[vehicle_id].get('action') == 'go':
                    in_transit_vehicles.add(vehicle_id)
                    # Force continue with 'go' action
                    if self._apply_single_vehicle_control(vehicle_id, 
                                                        self.controlled_vehicles[vehicle_id]['rank'],
                                                        self.controlled_vehicles[vehicle_id]['bid_value'], 
                                                        'go'):
                        controlled_vehicles.add(vehicle_id)
                        print(f"   🔒 Vehicle {vehicle_id}: PROTECTED (in transit)")
        
        # Second pass: Apply new auction controls
        for i, winner in enumerate(auction_winners):
            participant = winner.participant
            
            # Use the conflict_action determined by Nash solver (includes traffic flow control)
            control_action = getattr(winner, 'conflict_action', 'go')  # Safe default
            
            # Validate action
            if control_action not in ['go', 'wait']:
                print(f"⚠️ Invalid conflict_action '{control_action}' for {participant.id}, defaulting to 'go'")
                control_action = 'go'
            
            # Apply control
            if participant.type == 'vehicle':
                vehicle_id = str(participant.id)
                
                # Skip if vehicle is already protected as in-transit
                if vehicle_id in in_transit_vehicles:
                    continue
                    
                action_emoji = "🟢" if control_action == 'go' else "🔴"
                reason = f"rank #{winner.rank}"
                print(f"   🚗 Vehicle {vehicle_id}: {control_action} {action_emoji} ({reason})")
                
                if self._apply_single_vehicle_control(vehicle_id, winner.rank, 
                                                    winner.bid.value, control_action):
                    controlled_vehicles.add(vehicle_id)
                    
            elif participant.type == 'platoon':
                vehicles = participant.data.get('vehicles', [])
                if vehicles:
                    leader_id = str(vehicles[0]['id'])
                    
                    # Check if any platoon vehicle is in transit
                    platoon_in_transit = any(str(v['id']) in in_transit_vehicles for v in vehicles)
                    if platoon_in_transit:
                        # Force 'go' for entire platoon if any member is in transit
                        control_action = 'go'
                        print(f"   🔒 Platoon {participant.id}: PROTECTED (member in transit)")
                    
                    action_emoji = "🟢" if control_action == 'go' else "🔴"
                    reason = f"rank #{winner.rank}"
                    print(f"   🚛 Platoon {participant.id} (leader {leader_id}): {control_action} {action_emoji} ({reason})")
                    
                    platoon_vehicles = self._apply_platoon_control(
                        participant, winner.rank, winner.bid.value, control_action
                    )
                    controlled_vehicles.update(platoon_vehicles)
        
        return controlled_vehicles

    def _get_control_params_by_rank_and_action(self, rank: int, action: str, 
                                         is_platoon_member: bool = False,
                                         is_leader: bool = False) -> Dict[str, float]:
        """根据排名、动作和车队状态获取控制参数"""
        if action == 'wait':
            if is_platoon_member and not is_leader:
                # Followers should wait more aggressively to maintain formation
                return {
                    'speed_diff': -75.0,      # Stronger speed reduction for platoon followers
                    'follow_distance': 1.0,   # Very tight following for formation
                    'ignore_lights': 0.0,     
                    'ignore_signs': 0.0,      
                    'ignore_vehicles': 20.0   # Allow some vehicle ignoring to follow leader
                }
            else:
                return {
                    'speed_diff': -70.0,      # Strong speed reduction for waiting
                    'follow_distance': 2.5 if not is_platoon_member else 2.0,
                    'ignore_lights': 0.0,     
                    'ignore_signs': 0.0,      
                    'ignore_vehicles': 0.0    
                }
        elif action == 'go':
            if is_platoon_member and not is_leader:
                # Followers should be very aggressive in following the leader
                return {
                    'speed_diff': -45.0,      # Less speed reduction to keep up with leader
                    'follow_distance': 0.8,   # Very tight following distance
                    'ignore_lights': 100.0,   
                    'ignore_signs': 100.0,    
                    'ignore_vehicles': 90.0   # Higher vehicle ignoring for aggressive following
                }
            elif is_platoon_member and is_leader:
                # Leaders should move smoothly but not too aggressively
                return {
                    'speed_diff': -50.0,      
                    'follow_distance': 1.5,   # Normal following distance for leader
                    'ignore_lights': 100.0,   
                    'ignore_signs': 100.0,    
                    'ignore_vehicles': 50.0   # Limited vehicle ignoring for leader
                }
            else:
                return {
                    'speed_diff': -55.0,      
                    'follow_distance': 1.2,   
                    'ignore_lights': 100.0,   
                    'ignore_signs': 100.0,    
                    'ignore_vehicles': 50.0
                }

    def _restore_uncontrolled_vehicles(self, current_controlled: Set[str]):
        """恢复不再被控制的车辆，包括已离开路口的车辆"""
        previously_controlled = set(self.controlled_vehicles.keys())
        vehicles_to_restore = previously_controlled - current_controlled
        
        # 检查是否有车辆已完全离开路口区域
        vehicle_states = self.state_extractor.get_vehicle_states()
        vehicle_lookup = {str(v['id']): v for v in vehicle_states}
        
        for vehicle_id in list(self.controlled_vehicles.keys()):
            if vehicle_id in vehicle_lookup:
                vehicle_state = vehicle_lookup[vehicle_id]
                
                # 如果车辆已离开路口且不在当前控制列表中，移除控制
                if (not vehicle_state.get('is_junction', False) and 
                    vehicle_id not in current_controlled and
                    self._vehicle_has_exited_intersection(vehicle_state)):
                    vehicles_to_restore.add(vehicle_id)
                    print(f"✅ 车辆 {vehicle_id} 已离开路口，移除控制")
            else:
                # Vehicle no longer exists in simulation
                vehicles_to_restore.add(vehicle_id)
                print(f"🗑️ 车辆 {vehicle_id} 已从仿真中移除，清理控制")
        
        for vehicle_id in vehicles_to_restore:
            try:
                carla_vehicle = self.world.get_actor(int(vehicle_id))
                if carla_vehicle and carla_vehicle.is_alive:
                    # 恢复默认控制参数
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        carla_vehicle, self.default_speed_diff
                    )
                    self.traffic_manager.distance_to_leading_vehicle(
                        carla_vehicle, self.default_follow_distance
                    )
                    self.traffic_manager.ignore_lights_percentage(carla_vehicle, 0.0)
                    self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 0.0)
                
                # Track exit statistics
                if vehicle_id in self.controlled_vehicles:
                    # Mark exit time for statistics - USE SIMULATION TIME
                    current_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
                    self.control_history[vehicle_id] = {
                        'enter_time': self.controlled_vehicles[vehicle_id].get('sim_timestamp', current_sim_time),
                        'exit_time': current_sim_time,  # Use simulation time instead of wall-clock time
                        'action': self.controlled_vehicles[vehicle_id].get('action', 'unknown')
                    }
                    self.vehicles_exited_intersection += 1
                
                # Clean up acceleration tracking data for exited vehicles
                self.previous_velocities.pop(vehicle_id, None)
                self.previous_sim_timestamps.pop(vehicle_id, None)  # Updated to use sim timestamps
                
                # Clean up acceleration data
                # move the samples to archive so final stats include them
                for accel_type in ['positive', 'negative', 'absolute']:
                    if vehicle_id in self.acceleration_data[accel_type]:
                        self.archived_acceleration_data[accel_type].setdefault(vehicle_id, []).extend(
                            self.acceleration_data[accel_type][vehicle_id]
                        )
                        del self.acceleration_data[accel_type][vehicle_id]
                
                # 移除控制记录
                self.controlled_vehicles.pop(vehicle_id, None)
                
            except Exception as e:
                print(f"[Warning] 恢复车辆控制失败 {vehicle_id}: {e}")

    def _vehicle_has_exited_intersection(self, vehicle_state: Dict) -> bool:
        """检查车辆是否已完全离开路口区域"""
        vehicle_location = vehicle_state['location']
        distance_to_center = SimulationConfig.distance_to_intersection_center(vehicle_location)
        
        # 如果车辆距离路口中心超过一定距离，认为已离开
        exit_threshold = self.intersection_half_size/ 2
        return distance_to_center > exit_threshold

    def get_control_stats(self) -> Dict[str, Any]:
        """Get control statistics"""
        go_vehicles = 0
        waiting_vehicles = 0
        platoon_members = 0
        leaders = 0
        
        for vehicle_id, control_info in self.controlled_vehicles.items():
            if control_info.get('action') == 'go':
                go_vehicles += 1
            else:
                waiting_vehicles += 1
            
            if control_info.get('is_platoon_member', False):
                platoon_members += 1
                if control_info.get('is_leader', False):
                    leaders += 1
        
        return {
            'total_controlled': len(self.controlled_vehicles),
            'go_vehicles': go_vehicles,
            'waiting_vehicles': waiting_vehicles,
            'platoon_members': platoon_members,
            'platoon_leaders': leaders,
            'active_controls': list(self.controlled_vehicles.keys()),
            # New statistics
            'total_vehicles_ever_controlled': self.total_vehicles_controlled,
            'vehicles_exited_intersection': self.vehicles_exited_intersection
        }

    def get_final_statistics(self) -> Dict[str, Any]:
        """Get final simulation statistics with enhanced acceleration metrics"""
        accel_stats = self._calculate_average_acceleration()
        
        base_stats = {
            'total_vehicles_controlled': self.total_vehicles_controlled,
            'vehicles_exited_intersection': self.vehicles_exited_intersection,
            'vehicles_still_controlled': len(self.controlled_vehicles),
            'control_history_count': len(self.control_history),
        }
        
        # Merge acceleration statistics
        base_stats.update(accel_stats)
        
        # Add legacy field for backward compatibility
        base_stats['average_absolute_acceleration'] = accel_stats['average_absolute_acceleration']
        base_stats['total_acceleration_samples'] = accel_stats['absolute_acceleration_samples']
        base_stats['vehicles_with_acceleration_data'] = accel_stats['absolute_acceleration_vehicles']
        
        return base_stats

    def _apply_single_vehicle_control(self, vehicle_id: str, rank: int, bid_value: float, 
                                    action: str) -> bool:
        """Apply control to a single vehicle"""
        try:
            carla_vehicle = self.world.get_actor(int(vehicle_id))
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False
            
            # Get control parameters based on action
            params = self._get_control_params_by_rank_and_action(rank, action)
            
            # Apply traffic manager settings
            self.traffic_manager.vehicle_percentage_speed_difference(
                carla_vehicle, params['speed_diff']
            )
            self.traffic_manager.distance_to_leading_vehicle(
                carla_vehicle, params['follow_distance']
            )
            self.traffic_manager.ignore_lights_percentage(
                carla_vehicle, params['ignore_lights']
            )
            self.traffic_manager.ignore_signs_percentage(
                carla_vehicle, params['ignore_signs']
            )
            self.traffic_manager.ignore_vehicles_percentage(
                carla_vehicle, params['ignore_vehicles']
            )
            
            # If first time controlling this vehicle, increment total counter
            if vehicle_id not in self.controlled_vehicles:
                self.total_vehicles_controlled += 1
            # Record control state - USE SIMULATION TIME
            current_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self.controlled_vehicles[vehicle_id] = {
                 'rank': rank,
                 'bid_value': bid_value,
                 'action': action,
                 'params': params,
                 'is_platoon_member': False,
                 'is_leader': False,
                 'timestamp': time.time(),  # Keep wall-clock time for other purposes
                 'sim_timestamp': current_sim_time  # Add simulation time
             }
            
            return True
            
        except Exception as e:
            print(f"[Warning] 应用车辆控制失败 {vehicle_id}: {e}")
            return False

    def _apply_platoon_control(self, participant, rank: int, bid_value: float, 
                             action: str) -> Set[str]:
        """Apply control to all vehicles in a platoon"""
        controlled_vehicles = set()
        
        try:
            vehicles = participant.data.get('vehicles', [])
            if not vehicles:
                return controlled_vehicles
            
            for i, vehicle_data in enumerate(vehicles):
                vehicle_id = str(vehicle_data['id'])
                is_leader = (i == 0)
                
                # Apply control to each vehicle in platoon
                if self._apply_single_platoon_vehicle_control(
                    vehicle_id, rank, bid_value, action, is_leader
                ):
                    controlled_vehicles.add(vehicle_id)
            
            return controlled_vehicles
            
        except Exception as e:
            print(f"[Warning] 应用车队控制失败 {participant.id}: {e}")
            return controlled_vehicles

    def _apply_single_platoon_vehicle_control(self, vehicle_id: str, rank: int, 
                                            bid_value: float, action: str, 
                                            is_leader: bool) -> bool:
        """Apply control to a single vehicle within a platoon with enhanced follower aggression"""
        try:
            carla_vehicle = self.world.get_actor(int(vehicle_id))
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False
            
            # Get control parameters for platoon member
            params = self._get_control_params_by_rank_and_action(
                rank, action, is_platoon_member=True, is_leader=is_leader
            )
            
            # Apply traffic manager settings
            self.traffic_manager.vehicle_percentage_speed_difference(
                carla_vehicle, params['speed_diff']
            )
            self.traffic_manager.distance_to_leading_vehicle(
                carla_vehicle, params['follow_distance']
            )
            self.traffic_manager.ignore_lights_percentage(
                carla_vehicle, params['ignore_lights']
            )
            self.traffic_manager.ignore_signs_percentage(
                carla_vehicle, params['ignore_signs']
            )
            self.traffic_manager.ignore_vehicles_percentage(
                carla_vehicle, params['ignore_vehicles']
            )
        
            
            # If first time controlling this vehicle, increment total counter
            if vehicle_id not in self.controlled_vehicles:
                self.total_vehicles_controlled += 1
            # Record control state - USE SIMULATION TIME
            current_sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self.controlled_vehicles[vehicle_id] = {
                 'rank': rank,
                 'bid_value': bid_value,
                 'action': action,
                 'params': params,
                 'is_platoon_member': True,
                 'is_leader': is_leader,
                 'timestamp': time.time(),  # Keep wall-clock time for other purposes  
                 'sim_timestamp': current_sim_time  # Add simulation time
             }
            
            return True
            
        except Exception as e:
            print(f"[Warning] 应用车队车辆控制失败 {vehicle_id}: {e}")
            return False