import math
from collections import defaultdict

class PlatoonManager:
    def __init__(self, state_extractor, max_platoon_size=3):
        self.state_extractor = state_extractor  # 提取车辆状态信息的对象
        self.max_platoon_size = max_platoon_size
        self.platoons = []  # 存储所有车队，每个车队是一个列表
        self.vehicle_platoon_map = {}  # 车辆ID到车队的映射

    def form_platoons(self):
        """
        动态生成车队：根据方向/目的地等规则将车辆分为小组。
        """
        # Step 1: 获取所有车辆的状态
        vehicle_states = self.state_extractor.get_vehicle_states()
        
        if not vehicle_states:
            self.platoons = []
            return self.platoons

        # Step 2: 根据相似方向和位置进行聚类
        self.platoons = []
        self.vehicle_platoon_map = {}
        used_vehicles = set()
        
        # 按方向分组 (使用yaw角度)
        direction_groups = defaultdict(list)
        for vehicle in vehicle_states:
            yaw = vehicle['rotation'][1]  # yaw角度
            # 将角度归一化到8个方向 (每45度一个方向)
            direction_key = round(yaw / 45) * 45
            direction_groups[direction_key].append(vehicle)
        
        # 在每个方向组内，根据距离形成车队
        for direction, vehicles in direction_groups.items():
            if len(vehicles) < 2:  # 至少需要2辆车才能形成车队
                continue
                
            # 按位置排序（假设按x坐标排序）
            vehicles.sort(key=lambda v: v['location'][0])
            
            i = 0
            while i < len(vehicles):
                if vehicles[i]['id'] in used_vehicles:
                    i += 1
                    continue
                    
                # 创建新车队
                current_platoon = [vehicles[i]]
                used_vehicles.add(vehicles[i]['id'])
                
                # 寻找附近的车辆加入车队
                for j in range(i + 1, len(vehicles)):
                    if len(current_platoon) >= self.max_platoon_size:
                        break
                    if vehicles[j]['id'] in used_vehicles:
                        continue
                        
                    # 检查距离是否合适 (小于30米)
                    if self._calculate_distance(vehicles[i]['location'], vehicles[j]['location']) < 30:
                        current_platoon.append(vehicles[j])
                        used_vehicles.add(vehicles[j]['id'])
                
                # 如果车队有足够车辆，则保存
                if len(current_platoon) >= 2:
                    self.platoons.append(current_platoon)
                    for vehicle in current_platoon:
                        self.vehicle_platoon_map[vehicle['id']] = len(self.platoons) - 1
                
                i += 1

        return self.platoons
    
    def _calculate_distance(self, pos1, pos2):
        """计算两个位置之间的欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_platoon_by_vehicle(self, vehicle_id):
        """根据车辆ID获取其所属车队"""
        platoon_idx = self.vehicle_platoon_map.get(vehicle_id)
        if platoon_idx is not None:
            return self.platoons[platoon_idx]
        return None
    
    def get_platoon_stats(self):
        """获取车队统计信息"""
        total_vehicles_in_platoons = sum(len(platoon) for platoon in self.platoons)
        return {
            'num_platoons': len(self.platoons),
            'vehicles_in_platoons': total_vehicles_in_platoons,
            'avg_platoon_size': total_vehicles_in_platoons / len(self.platoons) if self.platoons else 0
        }
