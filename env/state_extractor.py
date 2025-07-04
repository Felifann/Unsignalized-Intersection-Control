from .simulation_config import SimulationConfig

class StateExtractor:
    def __init__(self, carla_wrapper):
        self.carla = carla_wrapper
        self._cached_actors = []
        self._cache_counter = 0
        self._cache_interval = SimulationConfig.ACTOR_CACHE_INTERVAL  # 使用配置文件中的缓存间隔设置

    def get_vehicle_states(self):
        # 缓存 actor 列表以减少查询开销
        if self._cache_counter % self._cache_interval == 0:
            self._cached_actors = list(self.carla.world.get_actors().filter('vehicle.*'))
        self._cache_counter += 1
        
        vehicle_states = []
        for vehicle in self._cached_actors:
            try:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                state = {
                    'id': vehicle.id,
                    'location': (transform.location.x, transform.location.y, transform.location.z),
                    'rotation': (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                    'velocity': (velocity.x, velocity.y, velocity.z),
                    'type': vehicle.type_id
                }
                vehicle_states.append(state)
            except:
                # 处理已销毁的车辆
                continue
        return vehicle_states