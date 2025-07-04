import random
import carla
from carla import Transform, Location, Rotation
from .simulation_config import SimulationConfig

class TrafficGenerator:
    def __init__(self, carla_wrapper, max_vehicles=None):
        self.carla = carla_wrapper
        self.max_vehicles = max_vehicles or SimulationConfig.MAX_VEHICLES
        self.vehicle_labels = {}  # 存储车辆标签的引用

    def _create_vehicle_label(self, vehicle):
        """为车辆创建ID标签"""
        try:
            # 获取车辆位置并在上方创建文字标签
            vehicle_location = vehicle.get_location()
            label_location = carla.Location(
                vehicle_location.x, 
                vehicle_location.y, 
                vehicle_location.z + 3.0  # 在车辆上方3米
            )
            
            # 创建文字标签 - 使用debug功能显示ID
            self.carla.world.debug.draw_string(
                label_location,
                str(vehicle.id),
                draw_shadow=False,
                color=carla.Color(255, 255, 255),  # 白色文字
                life_time=0.1,  # 短暂显示，需要持续更新
                persistent_lines=False
            )
            
            return True
        except:
            return False

    def generate_traffic(self):
        spawn_points = self.carla.world.get_map().get_spawn_points()
        num_vehicles = min(self.max_vehicles, len(spawn_points))
        random.shuffle(spawn_points)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        
        # 全局设置
        traffic_manager.global_percentage_speed_difference(-40.0)  # 全局提速40%
        traffic_manager.set_global_distance_to_leading_vehicle(1.5)  # 跟车距离1.5米
        traffic_manager.set_random_device_seed(42)  # 固定随机种子
        
        self.vehicles = []
        for i in range(num_vehicles):
            transform = spawn_points[i]
            vehicle = self.carla.spawn_vehicle(transform=transform)
            if vehicle is not None:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                
                # 随机化每辆车的行为
                if random.random() < 0.3:  # 30%的车更激进
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, -60.0)
                    traffic_manager.distance_to_leading_vehicle(vehicle, 0.8)
                
                self.vehicles.append(vehicle)
                print(f"车辆 {vehicle.id} 已生成并启用高速自动驾驶")

    def update_vehicle_labels(self):
        """更新所有车辆的ID标签位置"""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                self._create_vehicle_label(vehicle)