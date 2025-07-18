import math
import random
import time
from env.simulation_config import SimulationConfig

class AgentBidPolicy:
    def __init__(self, agent, intersection_center=(-188.9, -89.7, 0.0), state_extractor=None):
        self.agent = agent
        self.intersection_center = intersection_center
        self.state_extractor = state_extractor
        
    def compute_bid(self):

        """
        路口竞价策略：针对路口通行优化 - 车队优先版
        返回出价值（越高优先级越高）
        """
        # 🔥 车队获得巨大的基础优势
        if self._is_platoon():
            platoon_base_advantage = 200.0  # 车队基础优势大幅提升
        else:
            platoon_base_advantage = 0.0
    
        # 基础因子
        urgency = self._estimate_urgency()
        position_advantage = self._calculate_position_advantage()
        speed_factor = self._calculate_speed_factor()
        safety_factor = self._calculate_safety_factor()
        conflict_penalty = self._calculate_conflict_penalty()
        
        # 车队优势因子 - 大幅加强
        platoon_bonus = self._get_platoon_bonus()
        
        # 路口状态奖励/惩罚
        junction_factor = self._get_junction_factor()
        
        # 等待时间奖励
        wait_time_bonus = self._calculate_wait_time_bonus()
        
        # 加权计算最终出价 - 调整权重以突出车队优势
        base_bid = (platoon_base_advantage +        # 🔥 车队基础优势
                   urgency * 15 +                   # 方向紧急性权重（降低）
                   position_advantage * 12 +        # 位置优势权重（降低）
                   speed_factor * 8 +               # 速度因子权重（降低）
                   safety_factor * 10 +             # 安全因子权重（降低）
                   platoon_bonus * 50 +             # 🔥 车队奖励权重大幅提升
                   junction_factor * 25 +           # 路口状态因子（提升）
                   wait_time_bonus * 20)            # 等待时间奖励（降低）
    
        # 冲突惩罚
        final_bid = base_bid - conflict_penalty
    
        return max(0.0, final_bid)

    def _calculate_position_advantage(self):
        """计算位置优势：车队在路口内获得更高优势 - 使用正方形检测"""
        if self._is_platoon():
            leader = self.agent['vehicles'][0]
            at_junction = self.agent.get('at_junction', False)
            
            # 🔥 车队在路口内获得巨大位置优势
            if at_junction:
                return 100.0  # 从30.0提升到100.0
            elif SimulationConfig.is_in_intersection_area(leader['location']):
                # 在正方形区域内，距离中心越近优势越大
                distance = SimulationConfig.distance_to_intersection_center(leader['location'])
                half_size = SimulationConfig.INTERSECTION_HALF_SIZE
                # 归一化距离并计算优势
                normalized_distance = min(1.0, distance / half_size)
                return 50.0 * (1 - normalized_distance)  # 距离越近优势越大
            else:
                return 0.0
        else:
            at_junction = self.agent.get('at_junction', False)
            
            if at_junction:
                return 30.0  # 单车路口优势保持不变
            elif SimulationConfig.is_in_intersection_area(self.agent['data']['location']):
                distance = SimulationConfig.distance_to_intersection_center(self.agent['data']['location'])
                half_size = SimulationConfig.INTERSECTION_HALF_SIZE
                normalized_distance = min(1.0, distance / half_size)
                return 15.0 * (1 - normalized_distance)
            else:
                return 0.0

    def _get_junction_factor(self):
        """路口状态因子：车队在路口内的紧迫性更高"""
        if self._is_platoon():
            at_junction = self.agent.get('at_junction', False)
            if at_junction:
                return 80.0  # 🔥 车队在路口内获得更高优先级
            else:
                return 0.0
        else:
            at_junction = self.agent.get('at_junction', False)
            if at_junction:
                return 25.0  # 单车保持原有优先级
            else:
                return 0.0

    def _calculate_conflict_penalty(self):
        """计算冲突惩罚：左转与直行/右转的冲突"""
        direction = self._get_goal_direction()
        
        # 左转与其他方向冲突更多
        if direction == 'left':
            return 5.0  # 左转惩罚
        elif direction == 'right':
            return 0.0  # 右转最少冲突
        else:  # straight
            return 2.0  # 直行中等冲突

    def _estimate_urgency(self):
        """估计方向紧急性：右转 > 直行 > 左转"""
        direction = self._get_goal_direction()
        
        urgency_map = {
            'right': 5.0,    # 右转最容易，优先级高
            'straight': 3.0, # 直行中等
            'left': 1.5      # 左转最复杂，优先级低
        }
        
        return urgency_map.get(direction, 2.0)

    def _calculate_speed_factor(self):
        """计算速度因子：合理速度有优势"""
        if self._is_platoon():
            leader = self.agent['vehicles'][0]
            speed = self._get_current_speed(leader)
        else:
            speed = self._get_current_speed(self.agent['data'])
        
        # 路口适宜的速度范围
        if 2.0 <= speed <= 8.0:  # 合理通行速度
            return 8.0
        elif speed < 2.0:  # 速度过慢
            return 3.0
        else:  # 速度过快
            return 5.0

    def _calculate_safety_factor(self):
        """计算安全因子：确保安全通行的车辆有更高优先级"""
        if self._is_platoon():
            platoon_size = len(self.agent['vehicles'])
            if platoon_size <= 3:  # 小车队更安全
                return 10.0
            else:  # 大车队需要更多时间
                return 6.0
        else:
            # 单车安全因子
            speed = self._get_current_speed(self.agent['data'])
            if 2.0 <= speed <= 10.0:  # 合理速度范围
                return 8.0
            else:
                return 4.0

    def _get_platoon_bonus(self):
        """车队奖励：大幅鼓励车队协调通行"""
        if self._is_platoon():
            platoon_size = len(self.agent['vehicles'])
            
            # 🔥 大幅提升车队协调通行效益
            if platoon_size == 2:
                return 80.0   # 从10.0提升到80.0
            elif platoon_size == 3:
                return 120.0  # 从15.0提升到120.0
            elif platoon_size >= 4:
                return 150.0  # 从18.0提升到150.0
            else:
                return 0.0
        return 0.0

    def _get_goal_direction(self):
        """从导航系统获取目标方向"""
        if self._is_platoon():
            return self.agent.get('goal_direction', 'straight')
        else:
            # 单车从导航系统获取方向
            return self._get_navigation_direction_for_vehicle()

    def _get_navigation_direction_for_vehicle(self):
        """为单车从导航系统获取行驶方向"""
        vehicle_data = self.agent['data']
        
        # 检查车辆是否有目的地
        if not vehicle_data.get('destination'):
            print(f"[Warning] 车辆 {vehicle_data['id']} 没有目的地，使用默认方向")
            return 'straight'

        # 检查state_extractor是否初始化
        if not self.state_extractor:
            print(f"[Warning] StateExtractor未初始化，车辆 {vehicle_data['id']} 使用默认方向")
            return 'straight'

        vehicle_location = vehicle_data['location']
        destination = vehicle_data['destination']
        
        try:
            # 转换为carla.Location对象
            import carla
            carla_location = carla.Location(
                x=vehicle_location[0],
                y=vehicle_location[1], 
                z=vehicle_location[2]
            )
            
            # 使用state_extractor获取路径方向
            direction = self.state_extractor.get_route_direction(carla_location, destination)
            return direction if direction else 'straight'
            
        except Exception as e:
            print(f"[Warning] 车辆 {vehicle_data['id']} 导航方向获取失败：{e}，使用默认方向")
            return 'straight'

    def _is_platoon(self):
        """判断是否为车队"""
        return 'vehicles' in self.agent and len(self.agent['vehicles']) > 1

    def _get_current_speed(self, vehicle_state):
        """获取当前速度"""
        velocity = vehicle_state.get('velocity', (0, 0, 0))
        return math.sqrt(velocity[0]**2 + velocity[1]**2)

    # def _distance_to_intersection(self, vehicle_state):
    #     """计算到交叉口的距离"""
    #     location = vehicle_state.get('location', (0, 0, 0))
    #     return SimulationConfig.distance_to_intersection_center(location)

    def _calculate_wait_time_bonus(self):
        """计算等待时间奖励：等待越久，出价越高"""
        wait_time = self.agent.get('wait_time', 0.0)
        
        if wait_time <= 2.0:
            return 0.0
        elif wait_time <= 5.0:
            return (wait_time - 2.0) * 5.0
        elif wait_time <= 10.0:
            return 15.0 + (wait_time - 5.0) * 8.0
        else:
            return 55.0 + (wait_time - 10.0) * 10.0
