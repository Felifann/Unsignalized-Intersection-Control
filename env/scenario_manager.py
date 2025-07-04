from .carla_wrapper import CarlaWrapper
from .traffic_generator import TrafficGenerator
from .simulation_config import SimulationConfig

class ScenarioManager:
    def __init__(self, town=None):
        # 使用配置文件中的地图设置
        map_name = town or SimulationConfig.MAP_NAME
        self.carla = CarlaWrapper(town=map_name)
        self.traffic_gen = TrafficGenerator(self.carla)

    def reset_scenario(self):
        self.carla.destroy_all_vehicles()
        self.traffic_gen.generate_traffic()