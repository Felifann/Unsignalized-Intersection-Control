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
        路口竞价策略：单车版本 - 车队逻辑已禁用
        返回出价值（越高优先级越高）
        """
        # DISABLED: Platoon logic temporarily removed
        # 🚫 车队逻辑已暂时禁用，所有参与者都作为独立车辆处理
        
        # 基础因子
        urgency = self._estimate_urgency()
        position_advantage = self._calculate_position_advantage()
        speed_factor = self._calculate_speed_factor()
        
        # 路口状态奖励/惩罚
        junction_factor = self._get_junction_factor()
        
        # 等待时间奖励
        wait_time_bonus = self._calculate_wait_time_bonus()
        
        # 加权计算最终出价 - 单车优化权重
        base_bid = (urgency * 20 +                   # 方向紧急性权重
                   position_advantage * 15 +         # 位置优势权重
                   speed_factor * 10 +               # 速度因子权重
                   junction_factor * 25 +            # 路口状态因子
                   wait_time_bonus * 15)             # 等待时间奖励
    
        return max(0.0, base_bid)

    def _calculate_position_advantage(self):
        """计算位置优势：路口内 > 接近路口 > 远离路口 - 单车版本"""
        # DISABLED: Platoon-specific logic removed
        at_junction = self.agent.get('at_junction', False)
        
        if at_junction:
            return 60.0  # 路口内单车高优势
        else:
            # 计算距离优势
            vehicle_location = self._get_vehicle_location()
            if vehicle_location:
                distance = SimulationConfig.distance_to_intersection_center(vehicle_location)
                if distance <= 50.0:
                    return 30.0 - distance * 0.3
                else:
                    return 5.0
            else:
                return 5.0  # Fallback if location unavailable

    def _get_vehicle_location(self):
        """Helper method to get vehicle location from agent dict - 单车版本"""
        # DISABLED: Platoon location logic removed
        # Only handle individual vehicles
        if 'data' in self.agent and 'location' in self.agent['data']:
            return self.agent['data']['location']
        elif 'location' in self.agent:
            return self.agent['location']
        
        return None

    def _get_junction_factor(self):
        """路口状态因子：考虑距离的紧迫性 - 单车版本"""
        # DISABLED: Platoon-specific logic removed
        at_junction = self.agent.get('at_junction', False)
        if at_junction:
            return 40.0
        else:
            vehicle_location = self._get_vehicle_location()
            if vehicle_location:
                distance = SimulationConfig.distance_to_intersection_center(vehicle_location)
                return max(0.0, 25.0 - distance * 0.25)
            else:
                return 10.0  # Fallback

    def _calculate_speed_factor(self):
        """计算速度因子 - 单车版本"""
        try:
            # DISABLED: Platoon speed logic removed
            # Get vehicle data properly for single vehicle
            vehicle_data = self._get_vehicle_data()
            if vehicle_data:
                speed = self._get_current_speed(vehicle_data)
            else:
                return 5.0  # Fallback
            
            # Reasonable speed gets bonus
            if 3.0 <= speed <= 10.0:
                return 10.0
            elif speed < 3.0:
                return 5.0
            else:
                return 7.0
                
        except Exception as e:
            print(f"[Warning] Speed factor calculation failed: {e}")
            return 5.0  # Default value

    def _get_vehicle_data(self):
        """Helper method to get vehicle data from agent dict - 单车版本"""
        # DISABLED: Platoon data logic removed
        # Only handle individual vehicles
        if 'data' in self.agent:
            return self.agent['data']
        else:
            # Fallback: treat the agent dict itself as vehicle data
            return self.agent

    def _get_goal_direction(self):
        """从导航系统获取目标方向 - 单车版本"""
        # DISABLED: Platoon direction logic removed
        # Only handle individual vehicles
        return self._get_navigation_direction_for_vehicle()

    def _get_navigation_direction_for_vehicle(self):
        """通过导航系统获取单车方向"""
        # Check if this is a vehicle participant with data
        if self.agent['type'] == 'vehicle' and 'data' in self.agent:
            vehicle_data = self.agent['data']
        elif self.agent['type'] == 'vehicle':
            # Fallback: treat the agent dict itself as vehicle data
            vehicle_data = self.agent
        else:
            return None
        
        if not vehicle_data.get('destination'):
            return None
        
        try:
            import carla
            vehicle_location = carla.Location(
                x=vehicle_data['location'][0],
                y=vehicle_data['location'][1], 
                z=vehicle_data['location'][2]
            )
            
            return self.state_extractor.get_route_direction(
                vehicle_location, vehicle_data['destination']
            )
        except Exception as e:
            print(f"[Warning] Navigation direction failed: {e}")
            return None

    def _is_platoon(self):
        """判断是否为车队 - 暂时禁用，总是返回False"""
        # DISABLED: Always return False since platoons are disabled
        return False

    def _get_current_speed(self, vehicle_state):
        """获取当前速度"""
        velocity = vehicle_state.get('velocity', (0, 0, 0))
        return math.sqrt(velocity[0]**2 + velocity[1]**2)

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

    def _estimate_urgency(self):
        """估算紧急性：基于方向和距离 - 单车版本"""
        direction = self._get_goal_direction()
        
        # 基础紧急性
        base_urgency = 10.0
        
        # 方向奖励
        direction_bonus = {
            'straight': 15.0,  # 直行最优先
            'left': 10.0,      # 左转次优先
            'right': 12.0      # 右转中等优先
        }.get(direction, 8.0)
        
        # 距离因子
        vehicle_location = self._get_vehicle_location()
        if vehicle_location:
            distance = SimulationConfig.distance_to_intersection_center(vehicle_location)
            if distance <= 30.0:
                distance_urgency = 20.0 - distance * 0.5
            else:
                distance_urgency = 5.0
        else:
            distance_urgency = 5.0
        
        return base_urgency + direction_bonus + distance_urgency

    def _get_platoon_bonus(self):
        """获取车队奖励 - 暂时禁用，总是返回0"""
        # DISABLED: Always return 0 since platoons are disabled
        return 0.0
