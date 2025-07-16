import math

class Platoon:
    def __init__(self, vehicle_list, intersection_center=(-188.9, -89.7, 0.0), goal_direction=None, state_extractor=None):
        self.vehicles = vehicle_list  # 每个元素是 dict 状态
        self.leader = self.vehicles[0] if self.vehicles else None
        self.intersection_center = intersection_center
        self.state_extractor = state_extractor  # 添加state_extractor参数
        # 使用导航系统获取方向，无则默认'straight'
        self.goal_direction = self._get_navigation_direction() or goal_direction or 'straight'

    def _get_navigation_direction(self):
        """从导航系统获取车队领导车辆的行驶方向"""
        if not self.leader or not self.state_extractor:
            return None
        
        # 检查领导车辆是否有目的地
        if not self.leader.get('destination'):
            return None
        
        try:
            # 转换为carla.Location对象
            import carla
            vehicle_location = self.leader['location']
            carla_location = carla.Location(
                x=vehicle_location[0],
                y=vehicle_location[1], 
                z=vehicle_location[2]
            )
            
            # 使用state_extractor获取路径方向
            direction = self.state_extractor.get_route_direction(carla_location, self.leader['destination'])
            return direction
            
        except Exception as e:
            print(f"[Warning] 车队{self.leader['id']}路径规划方向获取失败：{e}")
            return None

    def get_vehicle_ids(self):
        return [v['id'] for v in self.vehicles]

    def get_leader(self):
        return self.leader

    def get_goal_direction(self):
        # 实时获取最新的导航方向
        navigation_direction = self._get_navigation_direction()
        if navigation_direction:
            self.goal_direction = navigation_direction
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
