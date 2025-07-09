import math
import time

class AgentBidPolicy:
    def __init__(self, agent, intersection_center=(-188.9, -89.7, 0.0)):
        self.agent = agent  # 传入单车或车队状态信息
        self.intersection_center = intersection_center
        
    def compute_bid(self):
        """
        分布式竞价策略：综合考虑多个因素
        返回出价值（越高优先级越高）
        """
        # 基础因子
        urgency = self._estimate_urgency()
        time_waiting = self._get_wait_time()
        distance_factor = self._calculate_distance_factor()
        speed_factor = self._calculate_speed_factor()
        congestion_factor = self._calculate_congestion_factor()
        safety_factor = self._calculate_safety_factor()
        
        # 车队优势因子
        platoon_bonus = self._get_platoon_bonus()
        
        # 加权计算最终出价
        base_bid = (urgency * 15 +           # 方向紧急性权重
                   time_waiting * 5 +        # 等待时间权重  
                   distance_factor * 10 +    # 距离因子权重
                   speed_factor * 8 +        # 速度因子权重
                   safety_factor * 12 +      # 安全因子权重
                   platoon_bonus)            # 车队奖励
        
        # 拥堵惩罚
        final_bid = base_bid * congestion_factor
        
        return max(0.0, final_bid)  # 确保出价非负

    def _estimate_urgency(self):
        """估计方向紧急性：右转 > 直行 > 左转"""
        direction = self._get_goal_direction()
        
        urgency_map = {
            'right': 4.0,    # 右转最容易，优先级高
            'straight': 2.5, # 直行中等
            'left': 1.0      # 左转最复杂，优先级低
        }
        
        return urgency_map.get(direction, 2.0)

    def _get_wait_time(self):
        """计算等待时间（简化版本）"""
        if self._is_platoon():
            # 车队使用队长的等待时间
            leader = self.agent['vehicles'][0] if 'vehicles' in self.agent else self.agent
            current_speed = self._get_current_speed(leader)
        else:
            current_speed = self._get_current_speed(self.agent)
        
        # 如果速度很低，认为在等待
        if current_speed < 2.0:  # 2 m/s 以下认为在等待
            return 10.0  # 模拟等待时间
        return 0.0

    def _calculate_distance_factor(self):
        """计算距离因子：距离路口越近，出价越高"""
        if self._is_platoon():
            leader = self.agent['vehicles'][0]
            distance = self._distance_to_intersection(leader)
        else:
            distance = self._distance_to_intersection(self.agent)
        
        # 距离越近，因子越高（最大20米范围内）
        max_distance = 20.0
        if distance >= max_distance:
            return 0.0
        
        return (max_distance - distance) / max_distance * 10.0

    def _calculate_speed_factor(self):
        """计算速度因子：速度越高，出价越高（体现冲量优势）"""
        if self._is_platoon():
            leader = self.agent['vehicles'][0]
            speed = self._get_current_speed(leader)
        else:
            speed = self._get_current_speed(self.agent)
        
        # 速度范围0-20 m/s，标准化到0-5分
        return min(speed / 4.0, 5.0)

    def _calculate_congestion_factor(self):
        """计算拥堵因子：周围车辆越少，因子越高"""
        # 简化版本：基于前车距离判断拥堵程度
        if self._is_platoon():
            leader = self.agent['vehicles'][0]
            leading_dist = leader.get('leading_vehicle_dist', -1)
        else:
            leading_dist = self.agent.get('leading_vehicle_dist', -1)
        
        if leading_dist < 0:  # 没有前车
            return 1.2  # 奖励因子
        elif leading_dist < 5:  # 前车很近，拥堵
            return 0.6  # 惩罚因子
        elif leading_dist < 15:  # 中等距离
            return 0.9
        else:  # 距离较远，不拥堵
            return 1.1

    def _calculate_safety_factor(self):
        """计算安全因子：确保安全通行的车辆有更高优先级"""
        if self._is_platoon():
            # 车队安全性更高（协调通行）
            platoon_size = len(self.agent['vehicles'])
            if platoon_size <= 3:  # 小车队更安全
                return 8.0
            else:  # 大车队需要更多时间
                return 5.0
        else:
            # 单车安全因子基于速度稳定性
            speed = self._get_current_speed(self.agent)
            if 3.0 <= speed <= 12.0:  # 合理速度范围
                return 6.0
            else:  # 速度过快或过慢
                return 3.0

    def _get_platoon_bonus(self):
        """车队奖励：鼓励车队协调通行"""
        if self._is_platoon():
            platoon_size = len(self.agent['vehicles'])
            
            # 车队越大，协调效益越高，但也越复杂
            if platoon_size == 2:
                return 8.0
            elif platoon_size == 3:
                return 12.0
            elif platoon_size >= 4:
                return 15.0
            else:
                return 0.0
        return 0.0  # 单车无奖励

    def _get_goal_direction(self):
        """获取目标方向"""
        if self._is_platoon():
            return self.agent.get('goal_direction', 'straight')
        else:
            # 单车需要从状态推断方向
            return self._infer_direction_from_state()

    def _infer_direction_from_state(self):
        """从车辆状态推断行驶方向（简化版本）"""
        # 这里可以基于车辆位置、朝向等信息推断
        # 简化版本：随机分配
        import random
        return random.choice(['left', 'straight', 'right'])

    def _is_platoon(self):
        """判断是否为车队"""
        return 'vehicles' in self.agent and len(self.agent['vehicles']) > 1

    def _get_current_speed(self, vehicle_state):
        """获取当前速度"""
        velocity = vehicle_state.get('velocity', (0, 0, 0))
        return math.sqrt(velocity[0]**2 + velocity[1]**2)

    def _distance_to_intersection(self, vehicle_state):
        """计算到交叉口的距离"""
        location = vehicle_state.get('location', (0, 0, 0))
        dx = location[0] - self.intersection_center[0]
        dy = location[1] - self.intersection_center[1]
        return math.sqrt(dx*dx + dy*dy)
