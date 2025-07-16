import math
from .platoon_policy import Platoon
import carla

class PlatoonManager:
    def __init__(self, state_extractor, intersection_center=(-188.9, -89.7, 0.0)):
        self.state_extractor = state_extractor
        self.platoons = []  # List of Platoon objects
        self.intersection_center = intersection_center
        self.max_platoon_size = 3  # 可配置的最大车队大小
        self.min_platoon_size = 2  # 最小车队大小改为2，单车不成队
        self.max_following_distance = 15.0  # 车队内最大跟车距离（米）

    def update(self):
        # Step 1: 获取所有车辆状态
        vehicle_states = self.state_extractor.get_vehicle_states()

        # Step 2: 筛选出交叉口 30m 范围内的车辆
        intersection_vehicles = self._filter_near_intersection(vehicle_states)

        # Step 3: 对这些车辆按车道 + 目的方向聚类
        groups = self._group_by_lane_and_goal(intersection_vehicles)

        # Step 4: 将每个 group 建立为多个 Platoon（支持多车队）
        self.platoons = []
        for group in groups:
            platoons_from_group = self._form_multiple_platoons(group)
            self.platoons.extend(platoons_from_group)

    def _filter_near_intersection(self, vehicle_states):
        # 对每辆车计算与交叉口中心点的距离（欧氏距离）
        # 返回 30 米以内的车辆
        return [v for v in vehicle_states if self._distance_to_intersection(v) < 30]

    def _group_by_lane_and_goal(self, vehicles):
        """按照车道ID + 目的方向分组，并确保车队内车辆相邻"""
        # 先按车道分组
        lane_groups = {}
        for v in vehicles:
            lane_id = self._get_lane_id(v)
            direction = self._estimate_goal_direction(v)
            
            # 只处理有明确方向的车辆
            if direction is None:
                continue
            
            if lane_id not in lane_groups:
                lane_groups[lane_id] = []
            lane_groups[lane_id].append((v, direction))
        
        # 对每个车道内的车辆按距离排序，然后检查相邻性
        final_groups = []
        for lane_id, vehicles_with_direction in lane_groups.items():
            # 按距离交叉口排序
            sorted_vehicles = sorted(vehicles_with_direction, 
                                   key=lambda x: self._distance_to_intersection(x[0]))
            
            # 找出相邻且目标方向相同的车辆组
            adjacent_groups = self._find_adjacent_groups(sorted_vehicles)
            final_groups.extend(adjacent_groups)
        
        return final_groups

    def _find_adjacent_groups(self, sorted_vehicles_with_direction):
        """找出相邻且目标方向相同的车辆组"""
        if not sorted_vehicles_with_direction:
            return []
        
        groups = []
        current_group = [sorted_vehicles_with_direction[0][0]]  # 只存储车辆对象
        current_direction = sorted_vehicles_with_direction[0][1]
        
        for i in range(1, len(sorted_vehicles_with_direction)):
            vehicle, direction = sorted_vehicles_with_direction[i]
            prev_vehicle = sorted_vehicles_with_direction[i-1][0]
            
            # 检查方向是否相同
            if direction != current_direction:
                # 方向不同，结束当前组，开始新组
                if len(current_group) >= self.min_platoon_size:
                    groups.append(current_group)
                current_group = [vehicle]
                current_direction = direction
                continue
            
            # 检查是否相邻（距离小于阈值）
            distance_between = self._calculate_vehicle_distance(prev_vehicle, vehicle)
            
            if distance_between <= self.max_following_distance:  # 相邻
                current_group.append(vehicle)
            else:
                # 不相邻，结束当前组，开始新组
                if len(current_group) >= self.min_platoon_size:
                    groups.append(current_group)
                current_group = [vehicle]
        
        # 处理最后一组
        if len(current_group) >= self.min_platoon_size:
            groups.append(current_group)
        
        return groups

    def _calculate_vehicle_distance(self, vehicle1, vehicle2):
        """计算两车之间的距离"""
        x1, y1, _ = vehicle1['location']
        x2, y2, _ = vehicle2['location']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _form_multiple_platoons(self, vehicle_group):
        """将一组相邻车辆构建为一个 Platoon 对象"""
        if not vehicle_group or len(vehicle_group) < self.min_platoon_size:
            return []
        
        # 限制车队大小
        if len(vehicle_group) > self.max_platoon_size:
            vehicle_group = vehicle_group[:self.max_platoon_size]
        
        # 验证车队内所有车辆方向一致
        directions = [self._estimate_goal_direction(v) for v in vehicle_group]
        if len(set(filter(None, directions))) != 1:
            print(f"[Warning] 车队内车辆方向不一致，跳过编队")
            return []
        
        platoon = Platoon(vehicle_group, self.intersection_center, goal_direction=directions[0])
        if platoon and platoon.is_valid():
            return [platoon]
        else:
            return []

    def _get_lane_id(self, vehicle):
        # 使用CARLA map接口获取所在车道的ID
        road_id = vehicle['road_id']
        lane_id = vehicle['lane_id']
        return f"{road_id}_{lane_id}"

    def _estimate_goal_direction(self, vehicle):
        """使用GlobalRoutePlanner估计车辆目标方向"""
        # 只使用路线规划分析方向，删除备用方法
        if not vehicle.get('destination'):
            return None  # 没有目的地的车辆不参与编队
        
        vehicle_location = carla.Location(
            x=vehicle['location'][0],
            y=vehicle['location'][1],
            z=vehicle['location'][2]
        )
        
        try:
            direction = self.state_extractor.get_route_direction(
                vehicle_location, vehicle['destination']
            )
            return direction
        except Exception as e:
            print(f"[Warning] 车辆 {vehicle['id']} 路线方向估计失败: {e}")
            return None  # 估计失败的车辆不参与编队

    def _distance_to_intersection(self, vehicle):
        # 返回车与交叉口中心的距离
        x, y, z = vehicle['location']
        center_x, center_y, center_z = self.intersection_center
        return math.sqrt((x - center_x)**2 + (y - center_y)**2)

    def _sort_by_distance(self, group):
        # 按照车辆到路口的距离从近到远排序
        return sorted(group, key=lambda v: self._distance_to_intersection(v))

    def get_all_platoons(self):
        """获取所有车队"""
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
        unplatoon_count = self.get_unplatoon_vehicles_count()
        
        print(f"\n{'='*60}")
        print(f"🚗 相邻车队管理系统状态报告")
        print(f"{'='*60}")
        print(f"📊 总体统计:")
        print(f"   - 相邻车队总数: {stats['num_platoons']}")
        print(f"   - 编队车辆数: {stats['vehicles_in_platoons']}")
        print(f"   - 独行车辆数: {unplatoon_count}")
        print(f"   - 平均车队大小: {stats['avg_platoon_size']:.1f}")
        print(f"   - 方向分布: {stats['direction_distribution']}")
        print(f"\n🔍 详细车队信息:")
        
        if not self.platoons:
            print("   暂无活跃相邻车队")
            return
        
        for i, platoon in enumerate(self.platoons):
            lane_info = platoon.get_lane_info()
            direction = platoon.get_goal_direction()
            avg_speed = platoon.get_average_speed() * 3.6  # 转换为km/h
            leader_pos = platoon.get_leader_position()
            
            # 方向emoji映射
            direction_emoji = {
                'left': '⬅️',
                'right': '➡️', 
                'straight': '⬆️'
            }
            
            print(f"\n   🚙 相邻车队 {i+1}: {direction_emoji.get(direction, '❓')} {direction.upper()}")
            print(f"      📍 车道: Road {lane_info[0]}/Lane {lane_info[1]}" if lane_info else "      📍 车道: 未知")
            print(f"      👥 成员数: {platoon.get_size()}")
            print(f"      🏃 平均速度: {avg_speed:.1f} km/h")
            if leader_pos:
                print(f"      🎯 队长位置: ({leader_pos[0]:.1f}, {leader_pos[1]:.1f})")
            
            # 验证车队相邻性
            adjacency_status = self._verify_platoon_adjacency(platoon)
            print(f"      🔗 相邻性验证: {adjacency_status}")
            
            # 打印车队成员详细信息及间距
            print(f"      👨‍👩‍👧‍👦 成员详情及间距:")
            for j, vehicle in enumerate(platoon.vehicles):
                role = "🔰队长" if j == 0 else f"🚗成员{j}"
                speed = math.sqrt(vehicle['velocity'][0]**2 + vehicle['velocity'][1]**2) * 3.6
                dist_to_center = self._distance_to_intersection(vehicle)
                junction_status = "🏢路口内" if vehicle['is_junction'] else "🛣️路段上"
                
                # 计算与前车距离
                if j > 0:
                    distance_to_prev = self._calculate_vehicle_distance(platoon.vehicles[j-1], vehicle)
                    distance_info = f"距前车:{distance_to_prev:.1f}m"
                else:
                    distance_info = "领头车"
                
                print(f"         {role} [ID:{vehicle['id']}] "
                      f"速度:{speed:.1f}km/h "
                      f"距中心:{dist_to_center:.1f}m "
                      f"{junction_status} "
                      f"({distance_info})")
            
            # 显示车队计划行动
            action_plan = self._get_platoon_action_plan(platoon)
            print(f"      📋 行动计划: {action_plan}")
        
        print(f"{'='*60}\n")

    def _get_platoon_action_plan(self, platoon):
        """获取车队的行动计划描述"""
        direction = platoon.get_goal_direction()
        size = platoon.get_size()
        leader = platoon.get_leader()
        
        if not leader:
            return "⚠️ 无效车队"
        
        # 分析当前状态
        is_in_junction = leader['is_junction']
        dist_to_center = self._distance_to_intersection(leader)
        avg_speed = platoon.get_average_speed() * 3.6
        
        # 检查车队是否准备好同时通过路口
        ready_to_pass = self._is_platoon_ready_to_pass(platoon)
        
        # 基于距离和位置制定行动计划
        if is_in_junction:
            if direction == 'left':
                return f"🔄 {size}车编队正在同时左转 (速度:{avg_speed:.1f}km/h)"
            elif direction == 'right':
                return f"🔄 {size}车编队正在同时右转 (速度:{avg_speed:.1f}km/h)"
            else:
                return f"🔄 {size}车编队正在同时直行 (速度:{avg_speed:.1f}km/h)"
        else:
            if dist_to_center < 15:  # 接近路口
                if ready_to_pass:
                    if direction == 'left':
                        return f"🚦 {size}车编队准备同时左转进入路口 ✅"
                    elif direction == 'right':
                        return f"🚦 {size}车编队准备同时右转进入路口 ✅"
                    else:
                        return f"🚦 {size}车编队准备同时直行进入路口 ✅"
                else:
                    return f"⏳ {size}车编队等待最佳时机进入路口 (目标:{direction})"
            else:  # 距离路口较远
                return f"🛣️ {size}车编队保持队形向路口行进 (目标:{direction})"

    def update_and_print_stats(self):
        """更新车队并打印统计信息（新增方法）"""
        self.update()
        
        # 获取基本统计
        stats = self.get_platoon_stats()
        unplatoon_count = self.get_unplatoon_vehicles_count()
        
        print(f"🚗 车队快报: {stats['num_platoons']}队/{stats['vehicles_in_platoons']}编队车/{unplatoon_count}独行车 | "
              f"方向: {stats['direction_distribution']}")

    def get_unplatoon_vehicles_count(self):
        """获取未编队车辆数量"""
        # 获取所有交叉口附近车辆
        vehicle_states = self.state_extractor.get_vehicle_states()
        intersection_vehicles = self._filter_near_intersection(vehicle_states)
        
        # 获取已编队车辆ID
        platoon_vehicle_ids = set()
        for platoon in self.platoons:
            for vehicle in platoon.vehicles:
                platoon_vehicle_ids.add(vehicle['id'])
        
        # 只统计有明确目的地的未编队车辆
        unplatoon_count = 0
        for vehicle in intersection_vehicles:
            if (vehicle['id'] not in platoon_vehicle_ids and 
                self._estimate_goal_direction(vehicle) is not None):
                unplatoon_count += 1
        
        return unplatoon_count

    def _is_platoon_ready_to_pass(self, platoon):
        """判断车队是否准备好同时通过路口"""
        if platoon.get_size() < 2:
            return True  # 单车总是准备好的
        
        vehicles = platoon.vehicles
        
        # 检查车队内车辆间距是否合适
        for i in range(len(vehicles) - 1):
            distance = self._calculate_vehicle_distance(vehicles[i], vehicles[i+1])
            if distance > self.max_following_distance:
                return False  # 车距太大，不适合同时通过
        
        # 检查车队速度是否同步
        speeds = [math.sqrt(v['velocity'][0]**2 + v['velocity'][1]**2) for v in vehicles]
        speed_variance = max(speeds) - min(speeds)
        if speed_variance > 5.0:  # 速度差超过5m/s
            return False
        
        # 检查是否有足够的通行时间窗口
        # 这里可以添加更复杂的冲突检测逻辑
        
        return True

    def _verify_platoon_adjacency(self, platoon):
        """验证车队的相邻性"""
        vehicles = platoon.vehicles
        if len(vehicles) < 2:
            return "✅ 单车无需验证"
        
        max_distance = 0
        for i in range(len(vehicles) - 1):
            distance = self._calculate_vehicle_distance(vehicles[i], vehicles[i+1])
            max_distance = max(max_distance, distance)
        
        if max_distance <= self.max_following_distance:
            return f"✅ 相邻 (最大间距:{max_distance:.1f}m)"
        else:
            return f"❌ 间距过大 (最大间距:{max_distance:.1f}m)"

