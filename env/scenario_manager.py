import carla
import math
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
    
    def update_vehicle_labels(self):
        """更新车辆标签显示"""
        self.traffic_gen.update_vehicle_labels()

    def show_intersection_area(self):
        """在地图上显示目标交叉口中心点和半径"""
        world = self.carla.world
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        radius = SimulationConfig.INTERSECTION_RADIUS

        # 显示中心点
        world.debug.draw_point(
            carla.Location(x=center[0], y=center[1], z=center[2]+1.0),
            size=0.3,
            color=carla.Color(255, 0, 0),
            life_time=10.0
        )
        
        # 使用多条线段绘制圆形
        num_segments = 32  # 圆形分段数
        for i in range(num_segments):
            angle1 = 2 * math.pi * i / num_segments
            angle2 = 2 * math.pi * (i + 1) / num_segments
            
            x1 = center[0] + radius * math.cos(angle1)
            y1 = center[1] + radius * math.sin(angle1)
            x2 = center[0] + radius * math.cos(angle2)
            y2 = center[1] + radius * math.sin(angle2)
            
            world.debug.draw_line(
                carla.Location(x=x1, y=y1, z=center[2]+0.5),
                carla.Location(x=x2, y=y2, z=center[2]+0.5),
                thickness=0.2,
                color=carla.Color(0, 255, 0),
                life_time=10.0,
                persistent_lines=False
            )