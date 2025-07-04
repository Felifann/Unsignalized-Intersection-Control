from .simulation_config import SimulationConfig
import math
import carla

class StateExtractor:
    def __init__(self, carla_wrapper):
        self.carla = carla_wrapper
        self.world_map = self.carla.world.get_map()  # 缓存地图对象
        self._cached_actors = []
        self._cache_counter = 0
        self._cache_interval = max(1, SimulationConfig.ACTOR_CACHE_INTERVAL // 2)  # 减少缓存间隔
        self.radius_sq = SimulationConfig.INTERSECTION_RADIUS ** 2  # 缓存半径平方

    def get_vehicle_states(self):
        # 更频繁地更新 actor 列表以捕获新车辆
        if self._cache_counter % self._cache_interval == 0:
            self._cached_actors = list(self.carla.world.get_actors().filter('vehicle.*'))
        self._cache_counter += 1
        
        vehicle_waypoints = {}
        valid_vehicles = [
            vehicle for vehicle in self._cached_actors
            if vehicle.is_alive and self.world_map.get_waypoint(vehicle.get_location()) is not None
        ]
        
        for vehicle in valid_vehicles:
            waypoint = self.world_map.get_waypoint(vehicle.get_location())
            vehicle_waypoints[vehicle.id] = waypoint

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

                velocity = vehicle.get_velocity()
                current_waypoint = vehicle_waypoints[vehicle.id]

                # 计算到前方车辆的距离
                leading_vehicle_dist = -1.0
                min_dist = float('inf')
                
                for other_vehicle in valid_vehicles:
                    if vehicle.id == other_vehicle.id:
                        continue
                    
                    other_wp = vehicle_waypoints.get(other_vehicle.id)
                    if other_wp is None:
                        continue
                    
                    # 检查是否在同一车道
                    if other_wp.road_id == current_waypoint.road_id and \
                       other_wp.lane_id == current_waypoint.lane_id:
                        
                        other_location = other_vehicle.get_location()
                        vec_to_other = carla.Vector3D(
                            other_location.x - transform.location.x,
                            other_location.y - transform.location.y,
                            other_location.z - transform.location.z
                        )
                        
                        # 手动计算点积
                        forward_vector = transform.get_forward_vector()
                        dot_product = forward_vector.x * vec_to_other.x + \
                                      forward_vector.y * vec_to_other.y + \
                                      forward_vector.z * vec_to_other.z
                        
                        if dot_product > 0:  # 判断是否在前方
                            dist = math.sqrt(vec_to_other.x**2 + vec_to_other.y**2 + vec_to_other.z**2)
                            if dist < min_dist:
                                min_dist = dist
                
                if min_dist != float('inf'):
                    leading_vehicle_dist = min_dist

                state = {
                    'id': vehicle.id,
                    'location': (location.x, location.y, location.z),
                    'rotation': (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                    'velocity': (velocity.x, velocity.y, velocity.z),
                    'type': vehicle.type_id,
                    'road_id': current_waypoint.road_id,
                    'lane_id': current_waypoint.lane_id,
                    'is_junction': current_waypoint.is_junction,
                    'leading_vehicle_dist': leading_vehicle_dist,
                    'distance_to_center': math.sqrt(dist_sq),
                }
                vehicle_states.append(state)
                
            except Exception as e:
                print(f"[Warning] 处理车辆 {vehicle.id} 状态失败: {e}")
                continue
        return vehicle_states