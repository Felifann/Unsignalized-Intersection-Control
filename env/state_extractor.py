from .simulation_config import SimulationConfig
import math
import carla
import time
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner

class StateExtractor:
    def __init__(self, carla_wrapper):
        self.carla = carla_wrapper
        self.world_map = self.carla.world.get_map()  # 缓存地图对象

        # 初始化GlobalRoutePlannerDAO
        dao = GlobalRoutePlannerDAO(self.world_map, 2.0)  # 2.0米采样距离

        # 初始化GlobalRoutePlanner
        self.global_route_planner = GlobalRoutePlanner(dao)
        self.global_route_planner.setup()  # 设置拓扑结构
        
        # 缓存相关属性
        self._cached_actors = []
        self._cache_counter = 0
        self._cache_interval = max(1, SimulationConfig.ACTOR_CACHE_INTERVAL // 2)
        
        # 新增：状态缓存机制
        self._vehicle_states_cache = []
        self._states_cache_timestamp = 0
        self._states_cache_duration = 0.05  # 缓存持续时间（秒）
        
        # 新增：waypoint缓存
        self._waypoint_cache = {}
        self._waypoint_cache_timestamp = 0
        self._waypoint_cache_duration = 0.1  # waypoint缓存持续时间
        
        # 新增：车辆目标点缓存
        self._vehicle_destinations = {}
        self._destination_cache_timestamp = 0
        self._destination_cache_duration = 5.0  # 目标点缓存时间较长
        
        self.radius_sq = SimulationConfig.INTERSECTION_RADIUS ** 2

    def get_vehicle_states(self, force_update=False):
        """获取车辆状态，支持缓存机制"""
        current_time = time.time()
        
        # 检查是否可以使用缓存的状态
        if (not force_update and 
            self._vehicle_states_cache and 
            current_time - self._states_cache_timestamp < self._states_cache_duration):
            return self._vehicle_states_cache
        
        # 更新状态缓存
        self._vehicle_states_cache = self._extract_vehicle_states()
        self._states_cache_timestamp = current_time
        
        return self._vehicle_states_cache

    def _extract_vehicle_states(self):
        """实际提取车辆状态的方法"""
        # 更频繁地更新 actor 列表以捕获新车辆
        if self._cache_counter % self._cache_interval == 0:
            self._cached_actors = list(self.carla.world.get_actors().filter('vehicle.*'))
        self._cache_counter += 1
        
        # 获取或更新waypoint缓存
        vehicle_waypoints = self._get_cached_waypoints()
        
        # 更新车辆目标点
        self._update_vehicle_destinations()
        
        valid_vehicles = [
            vehicle for vehicle in self._cached_actors
            if vehicle.is_alive and vehicle.id in vehicle_waypoints
        ]
        
        target_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        vehicle_states = []

        for vehicle in valid_vehicles:
            try:
                transform = vehicle.get_transform()
                location = transform.location
                
                # 检查车辆是否在目标交叉口半径内
                dist_sq = (location.x - target_center[0])**2 + (location.y - target_center[1])**2
                if dist_sq > self.radius_sq:
                    continue
                
                # 新增：剔除驶离路口的车辆
                if self._is_vehicle_leaving_intersection(vehicle, location, transform):
                    continue
                
                velocity = vehicle.get_velocity()
                current_waypoint = vehicle_waypoints[vehicle.id]

                # 计算到前方车辆的距离（优化版本）
                leading_vehicle_dist = self._calculate_leading_distance(
                    vehicle, transform, valid_vehicles, vehicle_waypoints, current_waypoint
                )

                state = {
                    'id': vehicle.id,
                    'location': (location.x, location.y, location.z),
                    'rotation': (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                    'velocity': (velocity.x, velocity.y, velocity.z),
                    'type': vehicle.type_id,
                    'road_id': current_waypoint.road_id,  # 从waypoint获取道路ID
                    'lane_id': current_waypoint.lane_id,  # 从waypoint获取车道ID
                    'is_junction': current_waypoint.is_junction,
                    'leading_vehicle_dist': leading_vehicle_dist,
                    'distance_to_center': math.sqrt(dist_sq),
                    'destination': self._vehicle_destinations.get(vehicle.id),  # 添加目标点信息
                }
                vehicle_states.append(state)
                
            except Exception as e:
                print(f"[Warning] 处理车辆 {vehicle.id} 状态失败: {e}")
                continue
    
        return vehicle_states

    def _update_vehicle_destinations(self):
        """更新车辆目标点缓存"""
        current_time = time.time()
        
        # 检查目标点缓存是否需要更新
        if (current_time - self._destination_cache_timestamp > self._destination_cache_duration):
            
            for vehicle in self._cached_actors:
                if vehicle.is_alive and vehicle.id not in self._vehicle_destinations:
                    # 为新车辆分配随机目标点
                    try:
                        spawn_points = self.world_map.get_spawn_points()
                        if spawn_points:
                            import random
                            destination = random.choice(spawn_points).location
                            self._vehicle_destinations[vehicle.id] = destination
                    except:
                        continue
            
            # 清理已销毁车辆的目标点
            active_vehicle_ids = {v.id for v in self._cached_actors if v.is_alive}
            self._vehicle_destinations = {
                vid: dest for vid, dest in self._vehicle_destinations.items() 
                if vid in active_vehicle_ids
            }
            
            self._destination_cache_timestamp = current_time

    def get_route_direction(self, vehicle_location, destination):
        """使用GlobalRoutePlanner分析路线方向"""
        try:
            # 获取起点和终点的waypoint
            start_waypoint = self.world_map.get_waypoint(vehicle_location)
            end_waypoint = self.world_map.get_waypoint(destination)
            
            if not start_waypoint or not end_waypoint:
                return 'straight'
            
            # 计算路线
            route = self.global_route_planner.trace_route(
                vehicle_location, destination
            )
            
            if len(route) < 3:
                return 'straight'
            
            # 分析路线中的转向
            return self._analyze_route_direction(route, vehicle_location)
            
        except Exception as e:
            print(f"[Warning] 路线方向分析失败: {e}")
            return 'straight'

    def _analyze_route_direction(self, route, current_location):
        """分析路线方向"""
        intersection_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        
        # 找到进入交叉口附近的waypoint
        intersection_waypoints = []
        post_intersection_waypoints = []
        
        in_intersection_area = False
        
        for waypoint, _ in route:
            wp_location = waypoint.transform.location
            distance_to_center = math.sqrt(
                (wp_location.x - intersection_center[0])**2 + 
                (wp_location.y - intersection_center[1])**2
            )
            
            if distance_to_center <= 25:  # 25米范围内认为是交叉口附近
                intersection_waypoints.append(waypoint)
                in_intersection_area = True
            elif in_intersection_area and distance_to_center > 25:
                # 刚离开交叉口区域
                post_intersection_waypoints.append(waypoint)
                if len(post_intersection_waypoints) >= 3:
                    break
        
        # 如果没有足够的waypoint进行分析，返回直行
        if len(intersection_waypoints) < 2:
            return 'straight'
        
        # 计算进入交叉口和离开交叉口的方向变化
        entry_yaw = intersection_waypoints[0].transform.rotation.yaw
        
        if post_intersection_waypoints:
            exit_yaw = post_intersection_waypoints[-1].transform.rotation.yaw
        elif len(intersection_waypoints) > 2:
            exit_yaw = intersection_waypoints[-1].transform.rotation.yaw
        else:
            return 'straight'
        
        # 计算角度差
        yaw_diff = self._normalize_angle(exit_yaw - entry_yaw)
        
        # 根据角度差判断方向
        if yaw_diff > 45:
            return 'left'
        elif yaw_diff < -45:
            return 'right'
        else:
            return 'straight'

    def _normalize_angle(self, angle):
        """标准化角度到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _get_cached_waypoints(self):
        """获取缓存的waypoint信息"""
        current_time = time.time()
        
        # 检查waypoint缓存是否过期
        if (current_time - self._waypoint_cache_timestamp > self._waypoint_cache_duration or
            not self._waypoint_cache):
            
            self._waypoint_cache = {}
            for vehicle in self._cached_actors:
                if vehicle.is_alive:
                    try:
                        waypoint = self.world_map.get_waypoint(vehicle.get_location())
                        if waypoint is not None:
                            self._waypoint_cache[vehicle.id] = waypoint
                    except:
                        continue
            
            self._waypoint_cache_timestamp = current_time
        
        return self._waypoint_cache

    def _calculate_leading_distance(self, vehicle, transform, valid_vehicles, vehicle_waypoints, current_waypoint):
        """优化的前车距离计算"""
        min_dist = float('inf')
        
        for other_vehicle in valid_vehicles:
            if vehicle.id == other_vehicle.id:
                continue
            
            other_wp = vehicle_waypoints.get(other_vehicle.id)
            if other_wp is None:
                continue
            
            # 检查是否在同一车道
            if (other_wp.road_id == current_waypoint.road_id and 
                other_wp.lane_id == current_waypoint.lane_id):
                
                other_location = other_vehicle.get_location()
                vec_to_other = carla.Vector3D(
                    other_location.x - transform.location.x,
                    other_location.y - transform.location.y,
                    other_location.z - transform.location.z
                )
                
                # 手动计算点积
                forward_vector = transform.get_forward_vector()
                dot_product = (forward_vector.x * vec_to_other.x + 
                             forward_vector.y * vec_to_other.y + 
                             forward_vector.z * vec_to_other.z)
                
                if dot_product > 0:  # 判断是否在前方
                    dist = math.sqrt(vec_to_other.x**2 + vec_to_other.y**2 + vec_to_other.z**2)
                    if dist < min_dist:
                        min_dist = dist
        
        return min_dist if min_dist != float('inf') else -1.0

    def clear_cache(self):
        """清除所有缓存"""
        self._vehicle_states_cache = []
        self._waypoint_cache = {}
        self._cached_actors = []
        self._vehicle_destinations = {}
        self._states_cache_timestamp = 0
        self._waypoint_cache_timestamp = 0
        self._destination_cache_timestamp = 0

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return {
            'cached_vehicles': len(self._vehicle_states_cache),
            'cached_waypoints': len(self._waypoint_cache),
            'cached_actors': len(self._cached_actors),
            'cached_destinations': len(self._vehicle_destinations),
            'states_cache_age': time.time() - self._states_cache_timestamp,
            'waypoint_cache_age': time.time() - self._waypoint_cache_timestamp
        }

    def _is_vehicle_leaving_intersection(self, vehicle, location, transform):
        """判断车辆是否正在驶离交叉口"""
        target_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        
        # 计算车辆到交叉口中心的方向向量
        to_center_x = target_center[0] - location.x
        to_center_y = target_center[1] - location.y
        
        # 获取车辆前进方向
        forward_vector = transform.get_forward_vector()
        
        # 计算车辆前进方向与朝向交叉口方向的夹角
        dot_product = forward_vector.x * to_center_x + forward_vector.y * to_center_y
        
        # 如果点积为负，说明车辆正在远离交叉口
        return dot_product < 0