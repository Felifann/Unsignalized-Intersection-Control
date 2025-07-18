import carla
import random
import time
from .simulation_config import SimulationConfig

class CarlaWrapper:
    def __init__(self, host=None, port=None, timeout=None, town=None):
        # 使用配置文件中的默认值
        self.client = carla.Client(
            host or SimulationConfig.CARLA_HOST, 
            port or SimulationConfig.CARLA_PORT
        )
        self.client.set_timeout(timeout or SimulationConfig.CARLA_TIMEOUT)
        
        # 使用配置文件中的地图设置
        map_name = town or SimulationConfig.MAP_NAME
        self.client.load_world(map_name)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # 使用配置文件中的仿真设置
        settings = self.world.get_settings()
        settings.synchronous_mode = SimulationConfig.SYNCHRONOUS_MODE
        settings.fixed_delta_seconds = SimulationConfig.FIXED_DELTA_SECONDS
        settings.max_substep_delta_time = 0.05  # 例如 0.05 秒
        settings.max_substeps = 10              # 例如 10 步
        self.world.apply_settings(settings)
        
        # 设置全局俯瞰视角
        self.setup_global_overview()

    def setup_global_overview(self):
        spectator = self.world.get_spectator()
        
        # 从配置文件获取俯瞰设置
        overview_config = SimulationConfig.get_overview_setting()
        overview_location = carla.Location(*overview_config['location'])
        overview_rotation = carla.Rotation(*overview_config['rotation'])
        
        # 应用俯瞰视角
        spectator.set_transform(carla.Transform(overview_location, overview_rotation))

    def spawn_vehicle(self, blueprint_filter='vehicle.*', transform=None):
        blueprint = random.choice(self.blueprint_library.filter(blueprint_filter))
        if transform is None:
            transform = random.choice(self.world.get_map().get_spawn_points())
        vehicle = self.world.spawn_actor(blueprint, transform)
        return vehicle

    def destroy_all_vehicles(self):
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            actor.destroy()