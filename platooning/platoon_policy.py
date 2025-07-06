import math

class Platoon:
    def __init__(self, vehicle_list, intersection_center=(-188.9, -89.7, 0.0)):
        self.vehicles = vehicle_list  # 每个元素是 dict 状态
        self.leader = self.vehicles[0] if self.vehicles else None
        self.intersection_center = intersection_center
        self.goal_direction = self._infer_goal_direction()

    def _infer_goal_direction(self):
        # 从leader推断整个车队目标方向
        if not self.leader:
            return 'straight'
        
        # 获取车辆当前位置和朝向
        vehicle_pos = self.leader['location']
        yaw_degrees = self.leader['rotation'][1]
        
        # 计算车辆到交叉口中心的向量
        to_intersection = [
            self.intersection_center[0] - vehicle_pos[0],
            self.intersection_center[1] - vehicle_pos[1]
        ]
        
        # 计算车辆当前朝向向量
        yaw_rad = math.radians(yaw_degrees)
        forward_vector = [math.cos(yaw_rad), math.sin(yaw_rad)]
        
        # 计算叉积来判断转向方向
        cross_product = forward_vector[0] * to_intersection[1] - forward_vector[1] * to_intersection[0]
        
        # 计算点积来判断是否朝向交叉口
        dot_product = forward_vector[0] * to_intersection[0] + forward_vector[1] * to_intersection[1]
        
        # 结合角度和位置关系判断方向
        if abs(cross_product) < 0.3 and dot_product > 0:  # 朝向交叉口且基本直行
            return 'straight'
        elif cross_product > 0.3:  # 需要左转到达交叉口
            return 'left'
        elif cross_product < -0.3:  # 需要右转到达交叉口
            return 'right'
        else:
            # 回退到基于角度的判断
            yaw_normalized = yaw_degrees
            while yaw_normalized > 180:
                yaw_normalized -= 360
            while yaw_normalized < -180:
                yaw_normalized += 360
            
            if -45 <= yaw_normalized <= 45:
                return 'straight'
            elif 45 < yaw_normalized <= 135:
                return 'left'
            elif -135 <= yaw_normalized < -45:
                return 'right'
            else:
                return 'straight'

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
