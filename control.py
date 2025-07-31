import carla
import math
import time
from typing import Dict, List, Set, Any
from env.simulation_config import SimulationConfig

class TrafficController:
    """
    基于拍卖结果的统一交通控制器
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
    
        # 新增：车队管理器引用（将在主程序中设置）
        self.platoon_manager = None
    
        # 新增：路口容量限制
        self.max_concurrent_agents = 4  # 最多同时通过4个agent
    
        print("🎮 基于拍卖的交通控制器初始化完成 - 集成安全控制和冲突解决")

    def set_platoon_manager(self, platoon_manager):
        """设置车队管理器引用"""
        self.platoon_manager = platoon_manager
    
    def update_control(self, platoon_manager, auction_engine):
        """
        主控制更新函数 - 基于新的拍卖引擎结构
        """
        # 1. 获取拍卖优先级排序
        auction_winners = auction_engine.get_current_priority_order()
        
        # 2. 基于拍卖结果应用控制
        current_controlled = set()
        if auction_winners:
            current_controlled = self._apply_auction_based_control(auction_winners, platoon_manager)
        
        # 3. 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 4. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _apply_auction_based_control(self, auction_winners: List, platoon_manager=None) -> Set[str]:
        """基于拍卖结果应用统一控制"""
        controlled_vehicles = set()
        
        if not auction_winners:
            return controlled_vehicles
        
        # 🔍 详细调试platoon_manager状态
        print(f"🎯 基于竞价排序应用控制，共{len(auction_winners)}个参与agents")
        
        # 确定agent控制状态
        agent_control_status = self._determine_agent_control_status(auction_winners)
        
        # 应用控制参数
        for winner in auction_winners:
            participant = winner.participant
            bid_value = winner.bid.value
            rank = winner.rank
            control_action = agent_control_status.get(participant.id, 'wait')
            
            print(f"🎮 Agent {participant.id}: rank={rank}, action={control_action}")
            
            try:
                if participant.type == 'vehicle':
                    vehicle_id = participant.id
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_action):
                        controlled_vehicles.add(vehicle_id)
                
                elif participant.type == 'platoon':
                    # 🔍 为车队添加更详细的调试信息
                    platoon_vehicles = participant.vehicles
                    direction = participant.data.get('goal_direction', 'unknown')
                    
                    print(f"🚛 处理车队 {participant.id}:")
                    print(f"   📊 车队大小: {len(platoon_vehicles)}")
                    print(f"   🎯 方向: {direction}")
                    print(f"   🎬 动作: {control_action}")
                    print(f"   🚗 车辆列表: {[v.get('id', 'unknown') for v in platoon_vehicles]}")
                    
                    # 🔍 验证车队车辆数据完整性
                    valid_vehicles = []
                    for i, v in enumerate(platoon_vehicles):
                        if 'id' in v:
                            carla_vehicle = self.world.get_actor(v['id'])
                            if carla_vehicle and carla_vehicle.is_alive:
                                valid_vehicles.append(v)
                            else:
                                print(f"   ⚠️ 车辆 {v.get('id', 'unknown')} 无效或已销毁")
                        else:
                            print(f"   ❌ 车队中第{i}辆车缺少ID信息")
                    
                    if len(valid_vehicles) != len(platoon_vehicles):
                        print(f"   🔄 车队车辆数据不完整: {len(valid_vehicles)}/{len(platoon_vehicles)} 有效")
                    
                    controlled_in_platoon = self._apply_platoon_agent_control(
                        valid_vehicles, rank, bid_value, direction, control_action
                    )
                    controlled_vehicles.update(controlled_in_platoon)
                    
                    print(f"   ✅ 车队控制结果: {len(controlled_in_platoon)}/{len(valid_vehicles)} 车辆被控制")
        
            except Exception as e:
                print(f"[Warning] agent {participant.id} 控制应用失败: {e}")
                import traceback
                traceback.print_exc()  # 打印详细错误信息
    
        return controlled_vehicles

    def _determine_agent_control_status(self, auction_winners: List) -> Dict[str, str]:
        """确定agent控制状态"""
        agent_control_status = {}
        
        # 统计当前路口内的agent
        current_agents_in_intersection = 0
        agents_in_intersection = []
        approaching_agents = []
        
        for winner in auction_winners:
            participant = winner.participant
            if self._is_agent_in_intersection(participant):
                current_agents_in_intersection += 1
                agents_in_intersection.append(winner)
            else:
                approaching_agents.append(winner)

        print(f"🏢 路口状态: {current_agents_in_intersection}个agent在路口内, {len(approaching_agents)}个agent接近中")
        
        # 默认所有agent都等待
        for winner in auction_winners:
            agent_control_status[winner.participant.id] = 'wait'

        # 1. 路口内的agent优先通行
        for winner in agents_in_intersection:
            if winner.protected:
                agent_control_status[winner.participant.id] = 'go'

        # 2. 如果路口容量允许，让接近的车道领头者进入
        available_capacity = self.max_concurrent_agents - current_agents_in_intersection
        
        if available_capacity > 0:
            allowed_count = 0
            
            for winner in approaching_agents:
                if allowed_count >= available_capacity:
                    break
                
                # 允许排名靠前的agent通行
                if winner.rank <= 2:  # 前两名可以通行
                    agent_control_status[winner.participant.id] = 'go'
                    allowed_count += 1

        return agent_control_status

    def _is_agent_in_intersection(self, participant) -> bool:
        """检查agent是否在路口内"""
        if participant.type == 'vehicle':
            return participant.data.get('is_junction', False)
        elif participant.type == 'platoon':
            # 车队中任何一辆车在路口内就认为整个车队在路口内
            return any(v.get('is_junction', False) for v in participant.vehicles)
        return False

    def _apply_single_vehicle_control(self, vehicle_id: str, rank: int, bid_value: float, 
                                    control_action: str = 'go') -> bool:
        """为单车agent应用控制"""
        try:
            carla_vehicle = self.world.get_actor(vehicle_id)
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False

            # 根据排名和动作调整控制强度
            control_params = self._get_control_params_by_rank_and_action(rank, control_action)

            # 应用控制参数
            self.traffic_manager.vehicle_percentage_speed_difference(
                carla_vehicle, control_params['speed_diff']
            )
            self.traffic_manager.distance_to_leading_vehicle(
                carla_vehicle, control_params['follow_distance']
            )
            self.traffic_manager.ignore_lights_percentage(
                carla_vehicle, control_params['ignore_lights']
            )
            self.traffic_manager.ignore_vehicles_percentage(
                carla_vehicle, control_params['ignore_vehicles']
            )

            # 记录控制状态
            self.controlled_vehicles[vehicle_id] = {
                'rank': rank,
                'action': control_action,
                'params': control_params,
                'control_time': time.time()
            }

            return True

        except Exception as e:
            print(f"[Warning] 单车控制失败 {vehicle_id}: {e}")
            return False

    def _get_control_params_by_rank_and_action(self, rank: int, action: str) -> Dict[str, float]:
        """根据排名和动作获取控制参数"""
        if action == 'wait':
            return {
                'speed_diff': -80.0,  # 大幅降速
                'follow_distance': 3.0,  # 增加跟车距离
                'ignore_lights': 0.0,   # 遵守信号灯
                'ignore_vehicles': 0.0  # 遵守其他车辆
            }
        elif action == 'go':
            if rank == 1:
                return {
                    'speed_diff': 10.0,   # 略微提速
                    'follow_distance': 1.0,  # 紧密跟车
                    'ignore_lights': 100.0,  # 忽略信号灯
                    'ignore_vehicles': 30.0  # 部分忽略其他车辆
                }
            elif rank <= 3:
                return {
                    'speed_diff': 0.0,    # 正常速度
                    'follow_distance': 1.5,  # 正常跟车距离
                    'ignore_lights': 80.0,   # 大部分忽略信号灯
                    'ignore_vehicles': 20.0  # 少量忽略其他车辆
                }
            else:
                return {
                    'speed_diff': -20.0,  # 略微降速
                    'follow_distance': 2.0,  # 增加跟车距离
                    'ignore_lights': 60.0,   # 部分忽略信号灯
                    'ignore_vehicles': 10.0  # 少量忽略其他车辆
                }
        
        # 默认参数
        return {
            'speed_diff': self.default_speed_diff,
            'follow_distance': self.default_follow_distance,
            'ignore_lights': 0.0,
            'ignore_vehicles': 0.0
        }

    def _get_platoon_leader_params(self, rank: int, action: str) -> Dict[str, float]:
        """获取车队队长控制参数 - 更保守的策略"""
        base_params = self._get_control_params_by_rank_and_action(rank, action)
        
        if action == 'go':
            # 🚛 队长采用保守策略，但不能太慢
            base_params['speed_diff'] = max(-5.0, base_params['speed_diff'])  # 改为最多降速5%
            base_params['follow_distance'] = max(1.5, base_params['follow_distance'] + 0.5)
            base_params['ignore_vehicles'] = 50.0  # 固定50%
            
            print(f"🚛 保守队长参数: speed_diff={base_params['speed_diff']}, "
                  f"follow_distance={base_params['follow_distance']}")
        
        return base_params

    def _get_platoon_follower_params(self, rank: int, action: str) -> Dict[str, float]:
        """获取车队跟随者控制参数 - 更激进的跟随策略"""
        base_params = self._get_control_params_by_rank_and_action(rank, action)
        
        if action == 'go':
            # 🔥 跟随者需要更激进以紧跟队长
            base_params['follow_distance'] = 0.5  # 改为0.5米，避免过于紧密
            base_params['ignore_lights'] = 0.0
            base_params['ignore_vehicles'] = 100.0
            base_params['speed_diff'] = 30.0  # 固定30%加速
            
            print(f"🚗 激进跟随者参数: ignore_vehicles={base_params['ignore_vehicles']}, "
                  f"follow_distance={base_params['follow_distance']}, speed_diff={base_params['speed_diff']}")
        
        return base_params

    def _apply_platoon_agent_control(self, platoon_vehicles: List[Dict], rank: int, 
                                   bid_value: float, direction: str, 
                                   control_action: str = 'go') -> Set[str]:
        """为车队agent应用控制 - 简化参数设置，避免冲突"""
        controlled_vehicles = set()

        try:
            platoon_size = len(platoon_vehicles)
            print(f"🚛 车队控制策略: 队长适度保守 + 跟随者激进, {platoon_size}辆车, 动作={control_action}")
            
            for i, vehicle_state in enumerate(platoon_vehicles):
                vehicle_id = vehicle_state['id']
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    print(f"⚠️ 车辆 {vehicle_id} 不存在或已销毁，跳过控制")
                    continue

                # 车队内角色：队长 vs 跟随者
                if i == 0:  # 队长
                    control_params = self._get_platoon_leader_params(rank, control_action)
                    role = 'platoon_leader'
                else:  # 跟随者
                    control_params = self._get_platoon_follower_params(rank, control_action)
                    role = 'platoon_follower'

                # 🔥 一次性应用所有参数，避免重复设置
                try:
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        carla_vehicle, control_params['speed_diff']
                    )
                    self.traffic_manager.distance_to_leading_vehicle(
                        carla_vehicle, control_params['follow_distance']
                    )
                    self.traffic_manager.ignore_lights_percentage(carla_vehicle, 0.0)
                    self.traffic_manager.ignore_vehicles_percentage(
                        carla_vehicle, control_params['ignore_vehicles']
                    )
                    
                    # 🔥 路口通用设置（不再重复设置上面的参数）
                    if control_action == 'go':
                        self.traffic_manager.auto_lane_change(carla_vehicle, False)
                        
                        # 只设置额外的参数，不重复设置已有的
                        if i > 0:  # 只对跟随者设置额外参数
                            self.traffic_manager.ignore_walkers_percentage(carla_vehicle, 100.0)
                            self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
                        
                        print(f"   {'🚛' if i == 0 else '🚗'} {role} {vehicle_id}: "
                              f"speed_diff={control_params['speed_diff']}, "
                              f"follow_distance={control_params['follow_distance']}, "
                              f"ignore_vehicles={control_params['ignore_vehicles']}")
                            
                except Exception as e:
                    print(f"[Warning] 车队车辆控制失败 {vehicle_id}: {e}")
                
                # 记录控制状态
                self.controlled_vehicles[vehicle_id] = {
                    'rank': rank,
                    'action': control_action,
                    'params': control_params,
                    'role': role,
                    'platoon_position': i,
                    'platoon_size': platoon_size,
                    'strategy': 'unified_params_no_conflict',
                    'control_time': time.time()
                }

                controlled_vehicles.add(vehicle_id)
                
            print(f"✅ 车队控制完成: {len(controlled_vehicles)}/{platoon_size}辆车被控制")

        except Exception as e:
            print(f"[Warning] 车队控制失败: {e}")

        return controlled_vehicles

    def _restore_uncontrolled_vehicles(self, current_controlled: Set[str]):
        """恢复不再被控制的车辆"""
        previously_controlled = set(self.controlled_vehicles.keys())
        vehicles_to_restore = previously_controlled - current_controlled
        
        for vehicle_id in vehicles_to_restore:
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
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

    def get_control_stats(self) -> Dict[str, Any]:
        """获取控制器统计信息"""
        go_vehicles = 0
        waiting_vehicles = 0
        
        for vehicle_id, control_info in self.controlled_vehicles.items():
            if control_info.get('action') == 'go':
                go_vehicles += 1
            else:
                waiting_vehicles += 1
        
        return {
            'total_controlled': len(self.controlled_vehicles),
            'go_vehicles': go_vehicles,
            'waiting_vehicles': waiting_vehicles,
            'active_controls': list(self.controlled_vehicles.keys())
        }

    def _apply_follower_intersection_override(self, follower_vehicle, leader_vehicle):
        """为跟随者应用特殊的路口穿越设置"""
        try:
            # 禁用自动变道
            self.traffic_manager.auto_lane_change(follower_vehicle, False)
            
            # 设置更激进的速度（使用正确的方法）
            self.traffic_manager.vehicle_percentage_speed_difference(follower_vehicle, 20.0)  # 比目标速度快20%
            
            # 设置更紧密的跟车距离
            self.traffic_manager.distance_to_leading_vehicle(follower_vehicle, 0.5)
            
            # 强制忽略安全检查
            self.traffic_manager.ignore_vehicles_percentage(follower_vehicle, 80.0)
            self.traffic_manager.ignore_lights_percentage(follower_vehicle, 100.0)
            
            print(f"🔧 跟随者 {follower_vehicle.id} 应用路口穿越覆盖设置")
            
        except Exception as e:
            print(f"⚠️ 跟随者路口设置失败: {e}")