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
        """在地图上显示目标交叉口中心点和正方形检测区域"""
        world = self.carla.world
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        half_size = SimulationConfig.INTERSECTION_HALF_SIZE

        # 显示中心点
        world.debug.draw_point(
            carla.Location(x=center[0], y=center[1], z=center[2]+1.0),
            size=0.1,
            color=carla.Color(255, 0, 0),
            life_time=99999.0
        )
        
        # 绘制正方形边界
        # 计算正方形四个角的坐标
        corners = [
            (center[0] - half_size, center[1] - half_size),  # 左下角
            (center[0] + half_size, center[1] - half_size),  # 右下角
            (center[0] + half_size, center[1] + half_size),  # 右上角
            (center[0] - half_size, center[1] + half_size),  # 左上角
        ]
        
        # 绘制正方形的四条边
        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]  # 下一个角点（循环到第一个）
            
            world.debug.draw_line(
                carla.Location(x=start_corner[0], y=start_corner[1], z=center[2]+0.5),
                carla.Location(x=end_corner[0], y=end_corner[1], z=center[2]+0.5),
                thickness=0.3,
                color=carla.Color(0, 255, 0),
                life_time=99999.0,
                persistent_lines=False
            )
        
        print(f"✅ 已显示正方形检测区域：中心({center[0]:.1f}, {center[1]:.1f})，边长{half_size*2}米")

    def show_intersection_area1(self):
        """在地图上显示目标交叉口中心点和正方形检测区域"""
        world = self.carla.world
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        half_size = SimulationConfig.INTERSECTION_HALF_SIZE

        # 显示中心点
        world.debug.draw_point(
            carla.Location(x=center[0], y=center[1], z=center[2]+1.0),
            size=0.1,
            color=carla.Color(255, 0, 0),
            life_time=99999.0
        )
        
        # 绘制正方形边界
        # 计算正方形四个角的坐标
        corners = [
            (center[0] - half_size/5, center[1] - half_size/5),  # 左下角
            (center[0] + half_size/5, center[1] - half_size/5),  # 右下角
            (center[0] + half_size/5, center[1] + half_size/5),  # 右上角
            (center[0] - half_size/5, center[1] + half_size/5),  # 左上角
        ]
        
        # 绘制正方形的四条边
        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]  # 下一个角点（循环到第一个）
            
            world.debug.draw_line(
                carla.Location(x=start_corner[0], y=start_corner[1], z=center[2]+0.5),
                carla.Location(x=end_corner[0], y=end_corner[1], z=center[2]+0.5),
                thickness=0.3,
                color=carla.Color(0, 0, 255),
                life_time=99999.0,
                persistent_lines=False
            )

    def show_road_lane_ids(self, display_radius=40):
        """在交叉口附近显示道路和车道ID"""
        world = self.carla.world
        world_map = world.get_map()
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        
        # 获取交叉口附近的waypoints
        waypoints_in_area = []
        
        # 在指定半径内采样waypoints
        for x in range(int(center[0] - display_radius), int(center[0] + display_radius), 5):
            for y in range(int(center[1] - display_radius), int(center[1] + display_radius), 5):
                location = carla.Location(x=float(x), y=float(y), z=center[2])
                distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                
                if distance <= display_radius:
                    try:
                        waypoint = world_map.get_waypoint(location)
                        if waypoint:
                            waypoints_in_area.append(waypoint)
                    except:
                        continue
        
        # 去重：基于road_id和lane_id组合
        unique_waypoints = {}
        for wp in waypoints_in_area:
            key = (wp.road_id, wp.lane_id)
            if key not in unique_waypoints:
                unique_waypoints[key] = wp
        
        # 显示道路和车道ID标签
        for (road_id, lane_id), waypoint in unique_waypoints.items():
            try:
                # 计算标签位置（在waypoint上方）
                label_location = carla.Location(
                    waypoint.transform.location.x,
                    waypoint.transform.location.y,
                    waypoint.transform.location.z + 4.0  # 在道路上方4米
                )
                
                # 创建标签文本
                label_text = f"R:{road_id}/L:{lane_id}"
                
                # 根据是否在交叉口使用不同颜色
                if waypoint.is_junction:
                    color = carla.Color(255, 255, 0)  # 黄色表示交叉口
                else:
                    color = carla.Color(0, 255, 255)  # 青色表示普通道路
                
                # 显示文字标签
                world.debug.draw_string(
                    label_location,
                    label_text,
                    draw_shadow=True,
                    color=color,
                    life_time=99999.0,  # 标签持续显示
                    persistent_lines=False
                )
                
                # 在waypoint位置画一个小点作为参考
                world.debug.draw_point(
                    waypoint.transform.location,
                    size=0.1,
                    color=color,
                    life_time=99999.0
                )
                
            except Exception as e:
                print(f"[Warning] 显示road/lane标签失败: {e}")
                continue