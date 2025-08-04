import carla
import math
import time
from typing import Dict, List, Set, Any
from env.simulation_config import SimulationConfig

class TrafficController:
    """
    基于拍卖结果的统一交通控制器 - 单车版本
    核心思想：所有控制都基于拍卖获胜者的优先级排序
    车队逻辑已暂时禁用
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
    
        # DISABLED: Platoon manager reference removed
        # self.platoon_manager = None
    
        # 新增：路口容量限制
        # self.max_concurrent_agents = 4  # 最多同时通过4个agent
    
        print("🎮 单车专用交通控制器初始化完成 - 车队逻辑已禁用")

    # DISABLED: Platoon manager setter removed
    # def set_platoon_manager(self, platoon_manager):

    def update_control(self, platoon_manager=None, auction_engine=None):
        """
        主控制更新函数 - 单车版本（忽略platoon_manager）
        """
        # 1. 获取拍卖优先级排序
        auction_winners = auction_engine.get_current_priority_order()
        
        # 2. 基于拍卖结果应用控制 (single vehicles only)
        current_controlled = set()
        if auction_winners:
            current_controlled = self._apply_auction_based_control(auction_winners, None)
        
        # 3. 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 4. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _apply_auction_based_control(self, auction_winners: List, platoon_manager=None) -> Set[str]:
        """基于拍卖结果应用统一控制 - 单车版本"""
        controlled_vehicles = set()
        
        if not auction_winners:
            return controlled_vehicles
        
        print(f"🎯 基于竞价排序应用单车控制，共{len(auction_winners)}个车辆")
        
        # 确定agent控制状态
        agent_control_status = self._determine_agent_control_status(auction_winners)
        
        # 应用控制参数 (single vehicles only)
        for winner in auction_winners:
            participant = winner.participant
            bid_value = winner.bid.value
            rank = winner.rank
            control_action = agent_control_status.get(participant.id, 'wait')
            
            print(f"🎮 Vehicle {participant.id}: rank={rank}, action={control_action}")
            
            try:
                # SIMPLIFIED: Only handle single vehicles
                if participant.type == 'vehicle':
                    vehicle_id = participant.id
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_action):
                        controlled_vehicles.add(vehicle_id)
                
                # DISABLED: Platoon control logic removed
                # elif participant.type == 'platoon':
        
            except Exception as e:
                print(f"[Warning] vehicle {participant.id} 控制应用失败: {e}")
    
        return controlled_vehicles

    def _determine_agent_control_status(self, auction_winners: List) -> Dict[str, str]:
        """确定agent控制状态 - 单车版本"""
        agent_control_status = {}
        
        # 统计当前路口内的agent (single vehicles only)
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

        print(f"🏢 路口状态: {current_agents_in_intersection}个车辆在路口内, {len(approaching_agents)}个车辆接近中")
        
        # 默认所有agent都等待
        for winner in auction_winners:
            agent_control_status[winner.participant.id] = 'wait'

        # 1. 路口内的agent优先通行
        for winner in agents_in_intersection:
            # if winner.protected:
            agent_control_status[winner.participant.id] = 'go'

        # 2. 如果路口容量允许，让接近的车道领头者进入
        # available_capacity = self.max_concurrent_agents - current_agents_in_intersection
        
        # if available_capacity > 0:
        #     allowed_count = 0
            
        #     for winner in approaching_agents:
        #         if allowed_count >= available_capacity:
        #             break
        #
        #         # 允许所有有空位的agent通行（不再限制rank）
        #         agent_control_status[winner.participant.id] = 'go'
        #         allowed_count += 1
        
        # if available_capacity > 0:
        #     for winner in approaching_agents[:available_capacity]:
        #         agent_control_status[winner.participant.id] = 'go'
        if approaching_agents:
            # 方案A: 允许前3名同时通行
            for winner in approaching_agents[:3]:
                agent_control_status[winner.participant.id] = 'go'

        return agent_control_status

    def _is_agent_in_intersection(self, participant) -> bool:
        """检查agent是否在路口内 - 单车版本"""
        # SIMPLIFIED: Only handle single vehicles
        if participant.type == 'vehicle':
            return participant.data.get('is_junction', False)
        # DISABLED: Platoon logic removed
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
            return {
                    'speed_diff': -30.0,   # 略微提速
                    'follow_distance': 1.0,  # 紧密跟车
                    'ignore_lights': 100.0,  # 忽略信号灯
                    'ignore_vehicles': 100.0  # 部分忽略其他车辆
                }

        # 默认参数
        return {
            'speed_diff': self.default_speed_diff,
            'follow_distance': self.default_follow_distance,
            'ignore_lights': 0.0,
            'ignore_vehicles': 0.0
        }

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