import math
from .platoon_policy import Platoon

class PlatoonManager:
    def __init__(self, state_extractor, intersection_center=(-188.9, -89.7, 0.0)):
        self.state_extractor = state_extractor
        self.platoons = []  # List of Platoon objects
        self.intersection_center = intersection_center

    def update(self):
        # Step 1: 获取所有车辆状态
        vehicle_states = self.state_extractor.get_vehicle_states()

        # Step 2: 筛选出交叉口 30m 范围内的车辆
        intersection_vehicles = self._filter_near_intersection(vehicle_states)

        # Step 3: 对这些车辆按车道 + 目的方向聚类
        groups = self._group_by_lane_and_goal(intersection_vehicles)

        # Step 4: 将每个 group 建立为一个 Platoon
        self.platoons = []
        for group in groups:
            platoon = self._form_platoon(group)
            if platoon and platoon.is_valid():
                self.platoons.append(platoon)

    def _filter_near_intersection(self, vehicle_states):
        # 对每辆车计算与交叉口中心点的距离（欧氏距离）
        # 返回 30 米以内的车辆
        return [v for v in vehicle_states if self._distance_to_intersection(v) < 30]

    def _group_by_lane_and_goal(self, vehicles):
        # 按照"所在车道ID + 目的方向（left/straight/right）"为key进行分组
        groups = {}
        for v in vehicles:
            lane_id = self._get_lane_id(v)
            direction = self._estimate_goal_direction(v)
            key = (lane_id, direction)
            groups.setdefault(key, []).append(v)
        return list(groups.values())

    def _form_platoon(self, vehicle_group):
        # 将一组车辆构建为一个 Platoon 对象（可能限制最多3辆）
        if not vehicle_group:
            return None
        sorted_group = self._sort_by_distance(vehicle_group)
        return Platoon(sorted_group[:3])  # 最多取3辆

    def _get_lane_id(self, vehicle):
        # 使用CARLA map接口获取所在车道的ID
        road_id = vehicle['road_id']
        lane_id = vehicle['lane_id']
        return f"{road_id}_{lane_id}"

    def _estimate_goal_direction(self, vehicle):
        # 基于当前yaw和目标位置推测是左转、右转还是直行
        yaw_degrees = vehicle['rotation'][1]
        
        # 标准化角度到[-180, 180]
        while yaw_degrees > 180:
            yaw_degrees -= 360
        while yaw_degrees < -180:
            yaw_degrees += 360
        
        # 基于角度范围判断方向（可根据Town05地图特点调整）
        if -45 <= yaw_degrees <= 45:
            return 'straight'
        elif 45 < yaw_degrees <= 135:
            return 'left'
        elif -135 <= yaw_degrees < -45:
            return 'right'
        else:
            return 'straight'  # 默认直行

    def _distance_to_intersection(self, vehicle):
        # 返回车与交叉口中心的距离
        x, y, z = vehicle['location']
        center_x, center_y, center_z = self.intersection_center
        return math.sqrt((x - center_x)**2 + (y - center_y)**2)

    def _sort_by_distance(self, group):
        # 按照车辆到路口的距离从近到远排序
        return sorted(group, key=lambda v: self._distance_to_intersection(v))

    def get_all_platoons(self):
        return self.platoons
    
    def get_platoon_stats(self):
        """获取车队统计信息"""
        if not self.platoons:
            return {
                'num_platoons': 0,
                'vehicles_in_platoons': 0,
                'avg_platoon_size': 0.0,
                'direction_distribution': {}
            }
        
        total_vehicles = sum(p.get_size() for p in self.platoons)
        avg_size = total_vehicles / len(self.platoons) if self.platoons else 0.0
        
        # 统计各方向的车队数量
        direction_dist = {}
        for platoon in self.platoons:
            direction = platoon.get_goal_direction()
            direction_dist[direction] = direction_dist.get(direction, 0) + 1
        
        return {
            'num_platoons': len(self.platoons),
            'vehicles_in_platoons': total_vehicles,
            'avg_platoon_size': avg_size,
            'direction_distribution': direction_dist
        }
    
    def get_platoons_by_direction(self, direction):
        """获取指定方向的所有车队"""
        return [p for p in self.platoons if p.get_goal_direction() == direction]
    
    def print_platoon_info(self):
        """打印车队详细信息（用于调试）"""
        stats = self.get_platoon_stats()
        print(f"\n=== 车队信息 ===")
        print(f"车队总数: {stats['num_platoons']}")
        print(f"编队车辆数: {stats['vehicles_in_platoons']}")
        print(f"平均车队大小: {stats['avg_platoon_size']:.1f}")
        print(f"方向分布: {stats['direction_distribution']}")
        
        for i, platoon in enumerate(self.platoons):
            lane_info = platoon.get_lane_info()
            print(f"车队 {i+1}: {platoon.get_goal_direction()}, "
                  f"车辆数: {platoon.get_size()}, "
                  f"车道: {lane_info}, "
                  f"平均速度: {platoon.get_average_speed()*3.6:.1f} km/h")
