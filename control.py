import math
import time
from typing import Dict, List, Set, Any
from env.simulation_config import SimulationConfig

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
        
        # 控制状态跟踪
        self.controlled_vehicles: Dict[str, Dict] = {}
        self.current_controlled_vehicles: Set[str] = set()
    
        # REACTIVATED: Platoon manager reference
        self.platoon_manager = None
    
        print("🎮 增强交通控制器初始化完成 - 支持车队和单车")

    def set_platoon_manager(self, platoon_manager):
        """Set platoon manager reference"""
        self.platoon_manager = platoon_manager
        print("🔗 车队管理器已连接到交通控制器")

    def update_control(self, platoon_manager=None, auction_engine=None):
        """
        主控制更新函数 - 支持车队和单车混合控制
        """
        # Update local platoon manager reference
        if platoon_manager:
            self.platoon_manager = platoon_manager
        
        # 1. 维持已在路口内车辆的控制
        current_controlled = self._maintain_intersection_vehicle_control()
        
        # 2. 获取拍卖优先级排序（包含车队和单车）
        auction_winners = auction_engine.get_current_priority_order()
        
        # 3. 基于拍卖结果应用控制 (supports platoons and vehicles)
        if auction_winners:
            auction_controlled = self._apply_auction_based_control(auction_winners, platoon_manager)
            current_controlled.update(auction_controlled)
        
        # 4. 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 5. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

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

    def _apply_auction_based_control(self, auction_winners: List, platoon_manager=None) -> Set[str]:
        """基于拍卖结果应用统一控制 - 支持车队和单车"""
        controlled_vehicles = set()
        
        if not auction_winners:
            return controlled_vehicles
        
        platoon_count = len([w for w in auction_winners if w.participant.type == 'platoon'])
        vehicle_count = len([w for w in auction_winners if w.participant.type == 'vehicle'])
        print(f"🎯 基于竞价排序应用混合控制: {platoon_count}个车队, {vehicle_count}个单车")
        
        # 确定agent控制状态
        agent_control_status = self._determine_agent_control_status(auction_winners)
        
        # Apply 'go' controls first, then 'wait' controls
        go_winners = [w for w in auction_winners if agent_control_status.get(w.participant.id) == 'go']
        wait_winners = [w for w in auction_winners if agent_control_status.get(w.participant.id) == 'wait']
        
        # Process 'go' agents first
        for winner in go_winners:
            participant = winner.participant
            bid_value = winner.bid.value
            rank = winner.rank
            control_action = 'go'
            
            agent_type = "车队" if participant.type == 'platoon' else "车辆"
            print(f"🎮 {agent_type} {participant.id}: rank={rank}, action={control_action}")
            
            try:
                if participant.type == 'vehicle':
                    vehicle_id = participant.id
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_action):
                        controlled_vehicles.add(vehicle_id)
                elif participant.type == 'platoon':
                    platoon_vehicles = self._apply_platoon_control(participant, rank, bid_value, control_action)
                    controlled_vehicles.update(platoon_vehicles)
            except Exception as e:
                print(f"[Warning] {agent_type} {participant.id} 控制应用失败: {e}")
        
        # Then process 'wait' agents
        for winner in wait_winners:
            participant = winner.participant
            bid_value = winner.bid.value
            rank = winner.rank
            control_action = 'wait'
            
            agent_type = "车队" if participant.type == 'platoon' else "车辆"
            print(f"🎮 {agent_type} {participant.id}: rank={rank}, action={control_action}")
            
            try:
                if participant.type == 'vehicle':
                    vehicle_id = participant.id
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_action):
                        controlled_vehicles.add(vehicle_id)
                elif participant.type == 'platoon':
                    platoon_vehicles = self._apply_platoon_control(participant, rank, bid_value, control_action)
                    controlled_vehicles.update(platoon_vehicles)
            except Exception as e:
                print(f"[Warning] {agent_type} {participant.id} 控制应用失败: {e}")
    
        # Ensure all vehicles currently in the junction are controlled
        vehicle_states = self.state_extractor.get_vehicle_states()
        for vehicle_state in vehicle_states:
            vehicle_id = str(vehicle_state['id'])
            if vehicle_state.get('is_junction', False) and vehicle_id not in self.controlled_vehicles:
                if self._apply_single_vehicle_control(vehicle_id, rank=0, bid_value=0.0, control_action='go'):
                    controlled_vehicles.add(vehicle_id)

        return controlled_vehicles

    def _apply_platoon_control(self, participant, rank: int, bid_value: float, 
                         control_action: str = 'go') -> Set[str]:
        """为车队agent应用统一控制，使成员同步行动"""
        controlled_vehicles = set()
        try:
            vehicles = participant.data.get('vehicles', [])
            if not vehicles:
                return controlled_vehicles

            print(f"🚛 控制车队 {participant.id}: {len(vehicles)}辆车, 动作={control_action}")

            # --- IMPROVED: Better coordinated platoon parameters ---
            if control_action == 'go':
                # Leader: smooth, less aggressive
                leader_params = {
                    'speed_diff': -20.0,      # Less speed reduction, smoother
                    'follow_distance': 2.5,   # Slightly larger gap
                    'ignore_lights': 100.0,
                    'ignore_signs': 100.0,
                    'ignore_vehicles': 60.0
                }
                # Followers: aggressive, close following
                follower_params = {
                    'speed_diff': -35.0,      # More speed reduction, keeps close
                    'follow_distance': 1.0,   # Very tight following
                    'ignore_lights': 100.0,
                    'ignore_signs': 100.0,
                    'ignore_vehicles': 100.0   # Almost ignore others, focus on leader
                }
            else:  # wait
                # All platoon members wait together
                wait_params = {
                    'speed_diff': -70.0,
                    'follow_distance': 2.0,
                    'ignore_lights': 0.0,
                    'ignore_signs': 0.0,
                    'ignore_vehicles': 0.0
                }
                leader_params = follower_params = wait_params

            # Apply control to each vehicle with role-specific parameters
            for idx, vehicle_data in enumerate(vehicles):
                vehicle_id = str(vehicle_data['id'])
                is_leader = (idx == 0)
                
                # Use appropriate parameters based on role
                params = leader_params if is_leader else follower_params
                
                if self._apply_single_vehicle_control(
                    vehicle_id,
                    rank,
                    bid_value,
                    control_action,
                    is_platoon_member=True,
                    is_leader=is_leader,
                    custom_params=params
                ):
                    controlled_vehicles.add(vehicle_id)
                    print(f"   ✅ {'Leader' if is_leader else 'Follower'} {vehicle_id} 控制应用成功")
                else:
                    print(f"   ❌ {'Leader' if is_leader else 'Follower'} {vehicle_id} 控制失败")

            return controlled_vehicles

        except Exception as e:
            print(f"[Warning] 车队控制失败 {participant.id}: {e}")
            return controlled_vehicles

    def _determine_agent_control_status(self, auction_winners: List) -> Dict[str, str]:
        """确定agent控制状态 - 简化：按优先级最多允许3辆go，其余wait，不做冲突检测"""
        agent_control_status = {}
        agents = [w.participant for w in auction_winners]
        max_concurrent_agents = 3  # 或根据需要调整
        for idx, agent in enumerate(agents):
            if idx < max_concurrent_agents:
                agent_control_status[agent.id] = 'go'
            else:
                agent_control_status[agent.id] = 'wait'
        return agent_control_status

    def _is_agent_in_intersection(self, participant) -> bool:
        """检查agent是否在路口内 - 单车版本"""
        # SIMPLIFIED: Only handle single vehicles
        if participant.type == 'vehicle':
            return participant.data.get('is_junction', False)
        # DISABLED: Platoon logic removed
        return False

    def _apply_single_vehicle_control(self, vehicle_id: str, rank: int, bid_value: float, 
                                    control_action: str = 'go', is_platoon_member: bool = False,
                                    is_leader: bool = False, custom_params: dict = None) -> bool:
        """为单车agent应用控制 - 支持自定义参数用于车队同步"""
        try:
            carla_vehicle = self.world.get_actor(int(vehicle_id))
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False

            # Use custom_params if provided (for platoon sync), else default logic
            if custom_params is not None:
                control_params = custom_params
            else:
                control_params = self._get_control_params_by_rank_and_action(
                    rank, control_action, is_platoon_member, is_leader
                )

            self.traffic_manager.set_hybrid_physics_mode(False)

            # ENHANCED: Apply platoon-specific settings with valid CARLA methods only
            if is_platoon_member:
                # Additional platoon coordination settings
                if is_leader:
                    # Leader: Steady, predictable movement
                    self.traffic_manager.auto_lane_change(carla_vehicle, False)
                    self.traffic_manager.collision_detection(carla_vehicle, carla_vehicle, True)
                else:
                    # Follower: Focus on following the leader/predecessor
                    self.traffic_manager.auto_lane_change(carla_vehicle, False)
                    self.traffic_manager.collision_detection(carla_vehicle, carla_vehicle, True)
                    # Use aggressive following behavior for tight formation
                    # This is achieved through the follow_distance parameter below

            # Apply standard traffic manager settings
            self.traffic_manager.vehicle_percentage_speed_difference(
                carla_vehicle, control_params['speed_diff']
            )
            self.traffic_manager.distance_to_leading_vehicle(
                carla_vehicle, control_params['follow_distance']
            )
            self.traffic_manager.ignore_lights_percentage(
                carla_vehicle, control_params['ignore_lights']
            )
            self.traffic_manager.ignore_signs_percentage(
                carla_vehicle, control_params['ignore_signs']
            )
            self.traffic_manager.ignore_vehicles_percentage(
                carla_vehicle, control_params['ignore_vehicles']
            )

            # Store control information
            self.controlled_vehicles[vehicle_id] = {
                'rank': rank,
                'action': control_action,
                'params': control_params,
                'control_time': time.time(),
                'is_platoon_member': is_platoon_member,
                'is_leader': is_leader
            }

            return True

        except Exception as e:
            print(f"[Warning] 单车控制失败 {vehicle_id}: {e}")
            return False

    def _get_control_params_by_rank_and_action(self, rank: int, action: str, 
                                             is_platoon_member: bool = False,
                                             is_leader: bool = False) -> Dict[str, float]:
        """根据排名、动作和车队状态获取控制参数 - 调整为更温和的参数"""
        if action == 'wait':
            return {
                'speed_diff': -60.0,      # 减少降速强度 (从-80.0)
                'follow_distance': 2.5 if not is_platoon_member else 2.0,   # 车队成员更紧密
                'ignore_lights': 0.0,     # 遵守信号灯
                'ignore_signs': 0.0,      # 遵守标志
                'ignore_vehicles': 0.0    # 遵守其他车辆
            }
        elif action == 'go':
            # Platoon members get more moderate coordination - LESS AGGRESSIVE
            if is_platoon_member:
                return {
                    'speed_diff': -45.0 if is_leader else -50.0,     # 更温和的速度控制
                    'follow_distance': 1.2 if not is_leader else 1.5,  # 增加跟车距离
                    'ignore_lights': 100.0,   # 忽略信号灯
                    'ignore_signs': 100.0,    # 忽略标志
                    'ignore_vehicles': 90.0 if not is_leader else 80.0  # 跟随车更积极
                }
            else:
                return {
                    'speed_diff': -55.0,      # 更温和的单车控制
                    'follow_distance': 1.2,   # 增加跟车距离
                    'ignore_lights': 100.0,   # 忽略信号灯
                    'ignore_signs': 100.0,    # 忽略标志
                    'ignore_vehicles': 100.0  # 部分忽略其他车辆
                }

        # 默认参数
        return {
            'speed_diff': self.default_speed_diff,
            'follow_distance': self.default_follow_distance,
            'ignore_lights': 0.0,
            'ignore_signs': 0.0,
            'ignore_vehicles': 0.0
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
        """获取控制器统计信息 - 增强版包含车队信息"""
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
            'active_controls': list(self.controlled_vehicles.keys())
        }