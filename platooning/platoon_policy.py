import math

class Platoon:
    def __init__(self, vehicle_list, intersection_center=(-188.9, -89.7, 0.0), goal_direction=None):
        self.vehicles = vehicle_list  # 每个元素是 dict 状态
        self.leader = self.vehicles[0] if self.vehicles else None
        self.intersection_center = intersection_center
        # 直接使用外部传入的方向（由全局路径推断），无则默认'straight'
        self.goal_direction = goal_direction or 'straight'

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
