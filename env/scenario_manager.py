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

        # 初始化时显示道路和车道ID
        self.show_road_lane_ids()

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
            life_time=99999.0
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
                life_time=99999.0,
                persistent_lines=False
            )

    def show_road_lane_ids(self, display_radius=50):
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

    def show_detailed_lane_info(self):
        """显示更详细的车道信息，包括车道类型和方向"""
        world = self.carla.world
        world_map = world.get_map()
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        radius = 30
        
        # 获取交叉口附近的所有waypoints
        for x in range(int(center[0] - radius), int(center[0] + radius), 3):
            for y in range(int(center[1] - radius), int(center[1] + radius), 3):
                location = carla.Location(x=float(x), y=float(y), z=center[2])
                distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                
                if distance <= radius:
                    try:
                        waypoint = world_map.get_waypoint(location)
                        if waypoint:
                            # 绘制waypoint方向箭头
                            forward_vector = waypoint.transform.get_forward_vector()
                            start_point = waypoint.transform.location
                            end_point = carla.Location(
                                start_point.x + forward_vector.x * 3,
                                start_point.y + forward_vector.y * 3,
                                start_point.z + 0.5
                            )
                            
                            # 根据车道类型选择颜色
                            if waypoint.is_junction:
                                arrow_color = carla.Color(255, 0, 255)  # 紫色：交叉口
                            elif waypoint.lane_type == carla.LaneType.Driving:
                                arrow_color = carla.Color(0, 255, 0)    # 绿色：行车道
                            elif waypoint.lane_type == carla.LaneType.Parking:
                                arrow_color = carla.Color(255, 255, 0)  # 黄色：停车位
                            else:
                                arrow_color = carla.Color(128, 128, 128) # 灰色：其他
                            
                            # 绘制方向箭头
                            world.debug.draw_arrow(
                                start_point,
                                end_point,
                                thickness=0.1,
                                arrow_size=0.2,
                                color=arrow_color,
                                life_time=10.0
                            )
                            
                    except:
                        continue

    def show_all_lane_ids(self):
        """静态显示模拟环境中所有车道的ID（包括路口内）"""
        world = self.carla.world
        world_map = world.get_map()
        
        # 获取地图的所有拓扑结构
        topology = world_map.get_topology()
        
        # 存储已处理的车道，避免重复显示
        processed_lanes = set()
        
        print(f"开始显示所有车道ID，共发现 {len(topology)} 个道路段...")
        
        for road_segment in topology:
            start_waypoint = road_segment[0]
            end_waypoint = road_segment[1]
            
            # 从起始waypoint开始，遍历这条道路段的所有车道
            current_wp = start_waypoint
            
            # 获取当前waypoint所在车道的所有相邻车道
            lane_waypoints = [current_wp]
            
            # 向左查找相邻车道
            left_wp = current_wp
            while True:
                try:
                    left_wp = left_wp.get_left_lane()
                    if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                        lane_waypoints.append(left_wp)
                    else:
                        break
                except:
                    break
            
            # 向右查找相邻车道
            right_wp = current_wp
            while True:
                try:
                    right_wp = right_wp.get_right_lane()
                    if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                        lane_waypoints.append(right_wp)
                    else:
                        break
                except:
                    break
            
            # 为每条车道显示ID
            for wp in lane_waypoints:
                lane_key = (wp.road_id, wp.lane_id)
                
                if lane_key not in processed_lanes:
                    processed_lanes.add(lane_key)
                    
                    try:
                        # 沿着车道方向采样多个点来显示车道ID
                        distance = 0
                        sample_wp = wp
                        
                        while distance < 20:  # 在20米范围内采样
                            # 计算标签位置
                            label_location = carla.Location(
                                sample_wp.transform.location.x,
                                sample_wp.transform.location.y,
                                sample_wp.transform.location.z + 3.0
                            )
                            
                            # 创建标签文本
                            label_text = f"R{wp.road_id}L{wp.lane_id}"
                            
                            # 根据是否在交叉口使用不同颜色
                            if sample_wp.is_junction:
                                color = carla.Color(255, 100, 0)  # 橙色表示交叉口车道
                                size = 0.15
                            else:
                                color = carla.Color(0, 200, 255)  # 蓝色表示普通车道
                                size = 0.1
                            
                            # 显示文字标签
                            world.debug.draw_string(
                                label_location,
                                label_text,
                                draw_shadow=True,
                                color=color,
                                life_time=99999.0,
                                persistent_lines=False
                            )
                            
                            # 在车道中心画点
                            world.debug.draw_point(
                                sample_wp.transform.location,
                                size=size,
                                color=color,
                                life_time=99999.0
                            )
                            
                            # 移动到下一个采样点
                            next_wps = sample_wp.next(5.0)  # 每5米采样一次
                            if next_wps:
                                sample_wp = next_wps[0]
                                distance += 5.0
                            else:
                                break
                                
                    except Exception as e:
                        print(f"[Warning] 显示车道 R{wp.road_id}L{wp.lane_id} 标签失败: {e}")
                        continue
        
        print(f"车道ID显示完成，共处理 {len(processed_lanes)} 条车道")
        print("图例：蓝色=普通车道，橙色=交叉口车道")