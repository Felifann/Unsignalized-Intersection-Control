import math

class Platoon:
    def __init__(self, vehicle_list):
        self.vehicles = vehicle_list  # 每个元素是 dict 状态
        self.leader = self.vehicles[0] if self.vehicles else None
        self.goal_direction = self._infer_goal_direction()

    def _infer_goal_direction(self):
        # 从leader推断整个车队目标方向
        if not self.leader:
            return 'straight'
        
        # 获取当前yaw角度（弧度）
        current_yaw = math.radians(self.leader['rotation'][1])
        
        # 获取车道信息来推断方向
        road_id = self.leader['road_id']
        lane_id = self.leader['lane_id']
        
        # 基于yaw角度和车道位置推断转向意图
        # 这里使用简化的角度判断逻辑
        yaw_degrees = self.leader['rotation'][1]
        
        # 标准化角度到[-180, 180]
        while yaw_degrees > 180:
            yaw_degrees -= 360
        while yaw_degrees < -180:
            yaw_degrees += 360
        
        # 基于角度范围判断方向（可根据具体地图调整）
        if -45 <= yaw_degrees <= 45:
            return 'straight'
        elif 45 < yaw_degrees <= 135:
            return 'left'
        elif -135 <= yaw_degrees < -45:
            return 'right'
        else:
            return 'straight'  # 默认直行

    def get_vehicle_ids(self):
        return [v['id'] for v in self.vehicles]

    def get_leader(self):
        return self.leader

    def get_goal_direction(self):
        return self.goal_direction

    def is_valid(self):
        return len(self.vehicles) > 0
    
    def get_size(self):
        return len(self.vehicles)
    
    def get_average_speed(self):
        if not self.vehicles:
            return 0.0
        
        total_speed = 0.0
        for vehicle in self.vehicles:
            velocity = vehicle['velocity']
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            total_speed += speed
        
        return total_speed / len(self.vehicles)
    
    def get_leader_position(self):
        if self.leader:
            return self.leader['location']
        return None
    
    def get_lane_info(self):
        if self.leader:
            return (self.leader['road_id'], self.leader['lane_id'])
        return None
