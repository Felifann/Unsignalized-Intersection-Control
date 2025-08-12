import time
import math
from typing import Dict, List, Set, Any, Tuple
from env.simulation_config import SimulationConfig
from nash.deadlock_nash_solver import DeadlockNashController, SimpleAgent

class TrafficController:
    """
    基于拍卖结果的统一交通控制器 - 支持车队和单车
    核心思想：所有控制都基于拍卖获胜者的优先级排序
    """
    
    def __init__(self, carla_wrapper, state_extractor):
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
        # self.default_ignore_vehicles = 100.0  # 默认忽略信号灯
        # self.default_ignore_signs = 100.0  # 默认忽略标志

        # 控制状态跟踪
        self.controlled_vehicles: Dict[str, Dict] = {}
        self.current_controlled_vehicles: Set[str] = set()
        self.platoon_manager = None
        
        # Nash deadlock resolution
        intersection_bbox = (
            self.intersection_center[0] - self.intersection_half_size/2,
            self.intersection_center[0] + self.intersection_half_size/2,
            self.intersection_center[1] - self.intersection_half_size/2,
            self.intersection_center[1] + self.intersection_half_size/2
        )
        self.nash_controller = DeadlockNashController(
            intersection_polygon=intersection_bbox,
            deadlock_time_window=3.0,
            min_agents_for_deadlock=3,
            progress_eps=0.5,
            collision_penalty=1000.0,
            wait_penalty_allwait=10.0,
            w_wait_inv=1.0,
            w_bid=1.0
        )
        
        print("🎮 增强交通控制器初始化完成 - 支持车队、单车和Nash deadlock解决")

    def set_platoon_manager(self, platoon_manager):
        """Set platoon manager reference"""
        self.platoon_manager = platoon_manager
        print("🔗 车队管理器已连接到交通控制器")

    def update_control(self, platoon_manager=None, auction_engine=None):
        """主控制更新函数 - 增加Nash deadlock resolution"""
        if platoon_manager:
            self.platoon_manager = platoon_manager
        
        # 1. Check for deadlock and apply Nash resolution
        nash_actions = self._handle_deadlock_resolution(auction_engine)
        
        # 2. Maintain intersection vehicle control
        current_controlled = self._maintain_intersection_vehicle_control()
        
        # 3. Apply auction-based control with Nash override
        auction_winners = auction_engine.get_current_priority_order()
        
        # 3. 基于拍卖结果应用控制 (supports platoons and vehicles)
        if auction_winners:
            auction_controlled = self._apply_auction_based_control(
                auction_winners, platoon_manager, nash_override=nash_actions
            )
            current_controlled.update(auction_controlled)
        
        # 4. 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 5. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _handle_deadlock_resolution(self, auction_engine) -> Dict[str, str]:
        """Handle deadlock detection and Nash resolution with system pause"""
        try:
            # Convert auction agents to Nash agents
            nash_agents = self._convert_to_nash_agents(auction_engine)
            if not nash_agents:
                return {}
            
            # Apply Nash deadlock resolution (now with system pause)
            nash_actions = self.nash_controller.handle_deadlock(nash_agents, time.time())
            
            if nash_actions:
                # Check if system is paused due to deadlock
                if self.nash_controller.deadlock_state.is_active:
                    print(f"🔒 SYSTEM PAUSED - Deadlock resolution in progress")
                    print(f"   Active participants: {self.nash_controller.deadlock_state.participants}")
                    print(f"   Current group: {self.nash_controller.deadlock_state.current_group_index + 1}/{len(self.nash_controller.deadlock_state.resolution_order)}")
                else:
                    print(f"🎯 Nash resolution applied: {nash_actions}")
            
            return nash_actions
            
        except Exception as e:
            print(f"[Warning] Nash deadlock resolution failed: {e}")
            return {}

    def _convert_to_nash_agents(self, auction_engine) -> List[SimpleAgent]:
        """Convert auction system agents to Nash SimpleAgent format"""
        nash_agents = []
        
        try:
            # Get current auction winners/participants
            auction_winners = auction_engine.get_current_priority_order()
            if not auction_winners:
                return []
            
            vehicle_states = self.state_extractor.get_vehicle_states()
            vehicle_lookup = {str(v['id']): v for v in vehicle_states}
            
            for winner in auction_winners:
                participant = winner.participant
                
                if participant.type == 'vehicle':
                    vehicle_id = str(participant.id)
                    if vehicle_id in vehicle_lookup:
                        v_state = vehicle_lookup[vehicle_id]
                        nash_agent = self._create_nash_agent_from_vehicle(
                            v_state, winner.bid.value
                        )
                        if nash_agent:
                            nash_agents.append(nash_agent)
                            
                elif participant.type == 'platoon':
                    # Handle platoon - create agent for leader
                    vehicles = participant.data.get('vehicles', [])
                    if vehicles:
                        leader_id = str(vehicles[0]['id'])
                        if leader_id in vehicle_lookup:
                            v_state = vehicle_lookup[leader_id]
                            nash_agent = self._create_nash_agent_from_vehicle(
                                v_state, winner.bid.value, is_platoon_leader=True
                            )
                            if nash_agent:
                                nash_agents.append(nash_agent)
            
            return nash_agents
            
        except Exception as e:
            print(f"[Warning] Converting to Nash agents failed: {e}")
            return []

    def _create_nash_agent_from_vehicle(self, vehicle_state: Dict, bid_value: float, 
                                      is_platoon_leader: bool = False) -> SimpleAgent:
        """Create Nash SimpleAgent from vehicle state"""
        try:
            location = vehicle_state['location']
            velocity = vehicle_state.get('velocity', [0, 0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
            
            # Estimate wait time from speed (simple heuristic)
            wait_time = max(0.1, 5.0 - speed)  # Lower speed = longer wait
            
            # Create simple intended path (straight line for now)
            current_pos = (location[0], location[1])
            heading = vehicle_state.get('rotation', [0, 0, 0])[2]  # yaw in degrees
            heading_rad = math.radians(heading)
            
            # Project path forward through intersection
            path_length = 20.0  # meters
            end_x = current_pos[0] + path_length * math.cos(heading_rad)
            end_y = current_pos[1] + path_length * math.sin(heading_rad)
            intended_path = [current_pos, (end_x, end_y)]
            
            return SimpleAgent(
                id=str(vehicle_state['id']),
                position=current_pos,
                speed=speed,
                heading=heading_rad,
                intended_path=intended_path,
                bid=bid_value,
                wait_time=wait_time
            )
            
        except Exception as e:
            print(f"[Warning] Creating Nash agent failed for vehicle {vehicle_state.get('id')}: {e}")
            return None

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
        """根据排名获取控制动作 - fallback仅用于Nash deadlock解决"""
        if rank <= 4:
            return 'go'  # 最高优先级，直接通行
        else:
            return 'wait'  # 其他优先级都等待，不使用fallback

    def _apply_auction_based_control(self, auction_winners: List, platoon_manager=None, 
                                   nash_override: Dict[str, str] = None) -> Set[str]:
        """Apply control with Nash system pause override and fallback support"""
        controlled_vehicles = set()
        
        if not auction_winners:
            return controlled_vehicles
        
        # Check if system is paused due to deadlock
        system_paused = (nash_override and 
                        self.nash_controller.deadlock_state.is_active)
        
        if system_paused:
            print(f"🔒 Applying system pause - only deadlock participants controlled")
            print(f"🎮 Nash actions: {nash_override}")
            
            # During system pause, only apply controls specified in nash_override
            for winner in auction_winners:
                participant = winner.participant
                if participant.type == 'vehicle':
                    vehicle_id = str(participant.id)
                    if vehicle_id in nash_override:
                        control_action = nash_override[vehicle_id]
                        print(f"   🚗 Vehicle {vehicle_id}: {control_action} (Nash)")
                        if self._apply_single_vehicle_control(vehicle_id, winner.rank, 
                                                            winner.bid.value, control_action):
                            controlled_vehicles.add(vehicle_id)
                elif participant.type == 'platoon':
                    vehicles = participant.data.get('vehicles', [])
                    if vehicles:
                        leader_id = str(vehicles[0]['id'])
                        if leader_id in nash_override:
                            control_action = nash_override[leader_id]
                            print(f"   🚛 Platoon {participant.id} (leader {leader_id}): {control_action} (Nash)")
                            platoon_vehicles = self._apply_platoon_control(
                                participant, winner.rank, winner.bid.value, control_action
                            )
                            controlled_vehicles.update(platoon_vehicles)
            
            return controlled_vehicles
        
        # Normal operation (no system pause) - NO FALLBACK in normal auction
        print(f"🚦 Normal auction control (no deadlock)")
        for winner in auction_winners:
            participant = winner.participant
            
            # Determine control action (go/wait only in normal operation)
            control_action = self._get_control_action_by_rank(winner.rank)
            
            # Apply Nash overrides only if they're not fallback (safety check)
            if nash_override:
                if participant.type == 'vehicle':
                    vehicle_id = str(participant.id)
                    if vehicle_id in nash_override:
                        nash_action = nash_override[vehicle_id]
                        if nash_action in ['go', 'wait']:  # Only allow go/wait overrides in normal mode
                            control_action = nash_action
                            print(f"   🎯 Nash override for vehicle {vehicle_id}: {control_action}")
                elif participant.type == 'platoon':
                    vehicles = participant.data.get('vehicles', [])
                    if vehicles:
                        leader_id = str(vehicles[0]['id'])
                        if leader_id in nash_override:
                            nash_action = nash_override[leader_id]
                            if nash_action in ['go', 'wait']:  # Only allow go/wait overrides in normal mode
                                control_action = nash_action
                                print(f"   🎯 Nash override for platoon {participant.id}: {control_action}")
            
            # Apply control
            if participant.type == 'vehicle':
                vehicle_id = str(participant.id)
                print(f"   🚗 Vehicle {vehicle_id}: {control_action}")
                if self._apply_single_vehicle_control(vehicle_id, winner.rank, 
                                                    winner.bid.value, control_action):
                    controlled_vehicles.add(vehicle_id)
                    
            elif participant.type == 'platoon':
                vehicles = participant.data.get('vehicles', [])
                if vehicles:
                    leader_id = str(vehicles[0]['id'])
                    print(f"   🚛 Platoon {participant.id} (leader {leader_id}): {control_action}")
                    platoon_vehicles = self._apply_platoon_control(
                        participant, winner.rank, winner.bid.value, control_action
                    )
                    controlled_vehicles.update(platoon_vehicles)
        
        return controlled_vehicles

    def _determine_agent_control_status(self, auction_winners: List) -> Dict[str, str]:
        """根据拍卖排名和当前状态确定代理控制状态"""
        control_status = {}
        
        for winner in auction_winners:
            participant = winner.participant
            
            if participant.type == 'vehicle':
                vehicle_id = str(participant.id)
                # 基于排名和当前动作确定控制状态
                control_status[vehicle_id] = self._get_control_action_by_rank(winner.rank)
                
            elif participant.type == 'platoon':
                vehicles = participant.data.get('vehicles', [])
                if vehicles:
                    leader_id = str(vehicles[0]['id'])
                    # 基于排名和当前动作确定控制状态 (使用车队首领的排名)
                    control_status[participant.id] = self._get_control_action_by_rank(winner.rank)
        
        return control_status

    def _get_control_params_by_rank_and_action(self, rank: int, action: str, 
                                         is_platoon_member: bool = False,
                                         is_leader: bool = False) -> Dict[str, float]:
        """根据排名、动作和车队状态获取控制参数 - 支持fallback动作"""
        if action == 'wait':
            return {
                'speed_diff': -70.0,      # Strong speed reduction for waiting
                'follow_distance': 2.5 if not is_platoon_member else 2.0,
                'ignore_lights': 0.0,     
                'ignore_signs': 0.0,      
                'ignore_vehicles': 0.0    
            }
        elif action == 'fallback':  # NEW: Fallback action for deadlock resolution
            return {
                'speed_diff': -80.0,      # Very strong speed reduction
                'follow_distance': 6.0,   # Large following distance to create space
                'ignore_lights': 100.0,   # Ignore lights to allow reverse movement
                'ignore_signs': 100.0,    # Ignore signs to allow reverse movement
                'ignore_vehicles': 80.0   # Mostly ignore vehicles behind (for backing up)
            }
        elif action == 'go':
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
        """Get control statistics including deadlock state"""
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
        
        # Add deadlock state information
        deadlock_info = {
            'deadlock_active': self.nash_controller.deadlock_state.is_active,
            'deadlock_participants': len(self.nash_controller.deadlock_state.participants),
            'system_paused': self.nash_controller.deadlock_state.is_active
        }
        
        return {
            'total_controlled': len(self.controlled_vehicles),
            'go_vehicles': go_vehicles,
            'waiting_vehicles': waiting_vehicles,
            'platoon_members': platoon_members,
            'platoon_leaders': leaders,
            'active_controls': list(self.controlled_vehicles.keys()),
            'deadlock_state': deadlock_info
        }

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
            
            # Handle fallback action (reverse movement)
            if action == 'fallback':
                self._apply_fallback_movement(carla_vehicle, vehicle_id)
            
            # Record control state
            self.controlled_vehicles[vehicle_id] = {
                'rank': rank,
                'bid_value': bid_value,
                'action': action,
                'params': params,
                'is_platoon_member': False,
                'is_leader': False,
                'timestamp': time.time()
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
        """Apply control to a single vehicle within a platoon"""
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
            
            # Handle fallback action for platoon vehicles
            if action == 'fallback':
                self._apply_fallback_movement(carla_vehicle, vehicle_id)
            
            # Record control state
            self.controlled_vehicles[vehicle_id] = {
                'rank': rank,
                'bid_value': bid_value,
                'action': action,
                'params': params,
                'is_platoon_member': True,
                'is_leader': is_leader,
                'timestamp': time.time()
            }
            
            return True
            
        except Exception as e:
            print(f"[Warning] 应用车队车辆控制失败 {vehicle_id}: {e}")
            return False

    def _apply_fallback_movement(self, carla_vehicle, vehicle_id: str):
        """Apply fallback movement (reverse) to a vehicle during deadlock resolution"""
        try:
            print(f"🔙 Applying fallback control to vehicle {vehicle_id}")
            
            # Get current vehicle state
            vehicle_transform = carla_vehicle.get_transform()
            current_location = vehicle_transform.location
            current_rotation = vehicle_transform.rotation
            
            # Calculate reverse direction (180 degrees from current heading)
            reverse_yaw = (current_rotation.yaw + 180) % 360
            reverse_rotation = carla_vehicle.get_world().get_blueprint_library().find('static.prop.streetbarrier').get_attribute('size')
            
            # Apply reverse movement by setting a waypoint behind the vehicle
            import carla
            reverse_direction = carla.Rotation(
                pitch=current_rotation.pitch,
                yaw=reverse_yaw,
                roll=current_rotation.roll
            )
            
            # Move vehicle backward by 5 meters
            import math
            reverse_distance = 5.0
            reverse_x = current_location.x - reverse_distance * math.cos(math.radians(reverse_yaw))
            reverse_y = current_location.y - reverse_distance * math.sin(math.radians(reverse_yaw))
            
            reverse_location = carla.Location(
                x=reverse_x,
                y=reverse_y,
                z=current_location.z
            )
            
            # Set destination for traffic manager to reverse location
            if self.traffic_manager:
                # Force vehicle to move backward using traffic manager
                self.traffic_manager.set_desired_speed(carla_vehicle, 5.0)  # Slow reverse speed
                # Note: Traffic manager doesn't directly support reverse, 
                # so we use very aggressive ignore settings to allow backing up
                self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 90.0)
                self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
                self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
            
            print(f"📍 Moving vehicle {vehicle_id} backward")
            
        except Exception as e:
            print(f"[Warning] Failed to apply fallback movement to {vehicle_id}: {e}")

