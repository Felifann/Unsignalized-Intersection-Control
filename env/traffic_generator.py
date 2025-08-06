import random
import carla
from .simulation_config import SimulationConfig

class TrafficGenerator:
    def __init__(self, carla_wrapper, max_vehicles=None):
        self.carla = carla_wrapper
        self.max_vehicles = max_vehicles or SimulationConfig.MAX_VEHICLES
        self.vehicle_labels = {}
        self.collision_sensors = {}  # 新增：存储每辆车的碰撞传感器

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

                # 新增：为每辆车添加碰撞传感器
                collision_sensor = self.carla.world.spawn_actor(
                    self.carla.blueprint_library.find('sensor.other.collision'),
                    carla.Transform(),
                    attach_to=vehicle
                )
                self.collision_sensors[vehicle.id] = collision_sensor
                collision_sensor.listen(lambda event, vid=vehicle.id: self._on_collision(event, vid))

    def _on_collision(self, event, vehicle_id):
        """碰撞事件回调，记录碰撞状态"""
        if not hasattr(self, 'collision_status'):
            self.collision_status = {}
        self.collision_status[vehicle_id] = True

    def reset_collision_status(self, vehicle_id):
        """重置车辆碰撞状态（可在恢复后调用）"""
        if hasattr(self, 'collision_status'):
            self.collision_status[vehicle_id] = False

    def get_collision_status(self, vehicle_id):
        """获取车辆是否发生碰撞"""
        if hasattr(self, 'collision_status'):
            return self.collision_status.get(vehicle_id, False)
        return False

    def update_vehicle_labels(self):
        """更新所有车辆的ID标签位置"""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                self._create_vehicle_label(vehicle)