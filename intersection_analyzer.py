import sys
import os
import glob
import math
import time

# 添加CARLA egg路径
egg_path = glob.glob(os.path.join("carla", "carla-*.egg"))
if egg_path:
    sys.path.append(egg_path[0])
    print(f"Added CARLA egg path: {egg_path[0]}")
else:
    # 尝试其他可能的路径
    possible_paths = [
        r"C:\carla\PythonAPI\carla\dist\carla-*.egg",
        r"D:\carla\PythonAPI\carla\dist\carla-*.egg",
        os.path.join(os.path.dirname(__file__), "PythonAPI", "carla", "dist", "carla-*.egg")
    ]
    
    found = False
    for path_pattern in possible_paths:
        eggs = glob.glob(path_pattern)
        if eggs:
            sys.path.append(eggs[0])
            print(f"Found CARLA egg at: {eggs[0]}")
            found = True
            break
    
    if not found:
        print("CARLA egg not found. Please ensure CARLA is installed and the path is correct.")
        print("Looking for carla-*.egg in the following locations:")
        for path in possible_paths:
            print(f"  {path}")
        sys.exit(1)

# 现在导入 carla
try:
    import carla
    print("CARLA module imported successfully")
except ImportError as e:
    print(f"Failed to import carla: {e}")
    print("Please check your CARLA installation and Python API setup")
    sys.exit(1)

from env.simulation_config import SimulationConfig

class IntersectionAnalyzer:
    def __init__(self):
        try:
            self.client = carla.Client(SimulationConfig.CARLA_HOST, SimulationConfig.CARLA_PORT)
            self.client.set_timeout(SimulationConfig.CARLA_TIMEOUT)
            print(f"Connected to CARLA server at {SimulationConfig.CARLA_HOST}:{SimulationConfig.CARLA_PORT}")
            
            # 创建截图保存目录
            self.screenshot_dir = os.path.join(os.path.dirname(__file__), "intersection_screenshots")
            os.makedirs(self.screenshot_dir, exist_ok=True)
            print(f"Screenshots will be saved to: {self.screenshot_dir}")
            
        except Exception as e:
            print(f"Failed to connect to CARLA server: {e}")
            print("Please make sure CARLA server is running")
            raise
        
    def analyze_map_intersections(self, map_name):
        """分析指定地图的十字路口"""
        print(f"\n=== 分析地图: {map_name} ===")
        
        try:
            # 加载地图
            self.client.load_world(map_name)
            world = self.client.get_world()
            carla_map = world.get_map()
            
            # 获取所有路点
            waypoints = carla_map.generate_waypoints(distance=2.0)
            
            # 查找交叉路口
            intersections = self._find_intersections(waypoints, carla_map)
            
            # 分析并打印结果
            self._print_intersection_analysis(map_name, intersections, carla_map)
            
            return intersections
            
        except Exception as e:
            print(f"无法加载地图 {map_name}: {e}")
            return []
    
    def _find_intersections(self, waypoints, carla_map):
        """查找无信号灯十字路口"""
        intersections = []
        processed_locations = set()
        
        for wp in waypoints:
            # 避免重复处理相近的位置
            location_key = (round(wp.transform.location.x, 1), round(wp.transform.location.y, 1))
            if location_key in processed_locations:
                continue
            processed_locations.add(location_key)
            
            # 检查是否为交叉路口
            if wp.is_junction:
                junction = wp.get_junction()
                if junction:
                    # 检查是否有信号灯
                    if self._has_traffic_lights(junction):
                        continue  # 跳过有信号灯的路口
                    
                    # 获取交叉路口的路点
                    junction_waypoints = junction.get_waypoints(carla.LaneType.Driving)
                    
                    # 计算进入和离开的道路数量
                    entry_roads = set()
                    exit_roads = set()
                    
                    for entry_wp, exit_wp in junction_waypoints:
                        entry_roads.add(entry_wp.road_id)
                        exit_roads.add(exit_wp.road_id)
                    
                    # 只考虑有4条或更多道路的交叉路口（十字路口）
                    total_roads = len(entry_roads | exit_roads)
                    if total_roads >= 4:
                        # 计算路口宽度指标
                        width_metric = self._calculate_intersection_width(junction_waypoints, junction)
                        
                        # 计算交叉路口中心位置
                        center_x = sum(wp.transform.location.x for wp, _ in junction_waypoints) / len(junction_waypoints)
                        center_y = sum(wp.transform.location.y for wp, _ in junction_waypoints) / len(junction_waypoints)
                        center_z = sum(wp.transform.location.z for wp, _ in junction_waypoints) / len(junction_waypoints)
                        
                        intersection_info = {
                            'id': junction.id,
                            'center': (center_x, center_y, center_z),
                            'num_roads': total_roads,
                            'num_lanes': len(junction_waypoints),
                            'entry_roads': list(entry_roads),
                            'exit_roads': list(exit_roads),
                            'bounding_box': junction.bounding_box,
                            'width_metric': width_metric,
                            'has_traffic_lights': False,
                            'waypoints': junction_waypoints[:10]  # 保存前10个路点作为示例
                        }
                        intersections.append(intersection_info)
        
        # 去重（基于位置相近性）
        unique_intersections = []
        for intersection in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                distance = math.sqrt(
                    (intersection['center'][0] - existing['center'][0])**2 + 
                    (intersection['center'][1] - existing['center'][1])**2
                )
                if distance < 20:  # 20米内认为是同一个路口
                    # 保留宽度指标更高的路口
                    if intersection['width_metric'] > existing['width_metric']:
                        unique_intersections.remove(existing)
                    else:
                        is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(intersection)
        
        # 按宽度指标排序，选择最宽阔的三个
        unique_intersections.sort(key=lambda x: x['width_metric'], reverse=True)
        return unique_intersections[:3]  # 只返回最好的三个
    
    def _has_traffic_lights(self, junction):
        """检查交叉路口是否有信号灯"""
        try:
            # 获取路口范围内的所有交通灯
            world = self.client.get_world()
            traffic_lights = world.get_actors().filter('traffic.traffic_light')
            
            junction_bbox = junction.bounding_box
            junction_center = junction_bbox.location
            junction_extent = junction_bbox.extent
            
            for traffic_light in traffic_lights:
                tl_location = traffic_light.get_location()
                
                # 检查信号灯是否在路口范围内
                if (abs(tl_location.x - junction_center.x) <= junction_extent.x + 10 and
                    abs(tl_location.y - junction_center.y) <= junction_extent.y + 10):
                    return True
            
            return False
        except Exception:
            # 如果无法检测信号灯，假设为无信号灯路口
            return False
    
    def _calculate_intersection_width(self, junction_waypoints, junction):
        """计算路口宽度指标"""
        if not junction_waypoints:
            return 0
        
        # 计算边界框面积作为宽度指标
        bbox = junction.bounding_box
        area = bbox.extent.x * bbox.extent.y * 4  # 边界框面积
        
        # 考虑车道数量
        lane_bonus = len(junction_waypoints) * 10
        
        # 考虑道路数量
        entry_roads = set()
        for entry_wp, _ in junction_waypoints:
            entry_roads.add(entry_wp.road_id)
        road_bonus = len(entry_roads) * 50
        
        return area + lane_bonus + road_bonus

    def _print_intersection_analysis(self, map_name, intersections, carla_map):
        """打印交叉路口分析结果并截图"""
        print(f"地图 {map_name} 发现 {len(intersections)} 个最佳无信号灯十字路口:")
        
        for i, intersection in enumerate(intersections, 1):
            print(f"\n--- 路口 {i} (推荐度: {i}) ---")
            print(f"ID: {intersection['id']}")
            print(f"中心位置: ({intersection['center'][0]:.1f}, {intersection['center'][1]:.1f}, {intersection['center'][2]:.1f})")
            print(f"连接道路数: {intersection['num_roads']}")
            print(f"车道数: {intersection['num_lanes']}")
            print(f"宽度指标: {intersection['width_metric']:.1f}")
            print(f"有信号灯: {intersection['has_traffic_lights']}")
            print(f"进入道路ID: {intersection['entry_roads']}")
            print(f"离开道路ID: {intersection['exit_roads']}")
            
            # 计算适合的俯瞰视角
            center = intersection['center']
            overview_pos = (center[0], center[1], center[2] + 100)
            print(f"建议俯瞰位置: {overview_pos}")
            
            # 评估路口复杂度
            complexity = self._evaluate_complexity(intersection)
            print(f"复杂度评级: {complexity}")
            
            # 推荐理由
            reasons = []
            if intersection['width_metric'] > 2000:
                reasons.append("宽阔度高")
            if intersection['num_roads'] == 4:
                reasons.append("标准十字路口")
            elif intersection['num_roads'] > 4:
                reasons.append("多路交汇")
            if intersection['num_lanes'] >= 8:
                reasons.append("车道充足")
            
            print(f"推荐理由: {', '.join(reasons)}")
            
            # 截图
            try:
                screenshot_path = self._capture_intersection_screenshot(map_name, intersection, i)
                print(f"截图已保存: {screenshot_path}")
            except Exception as e:
                print(f"截图失败: {e}")

    def _capture_intersection_screenshot(self, map_name, intersection, rank):
        """为指定路口拍摄俯视图截图"""
        world = self.client.get_world()
        
        # 设置相机位置（俯视角度）
        center = intersection['center']
        camera_height = 80  # 相机高度
        camera_location = carla.Location(
            x=center[0],
            y=center[1], 
            z=center[2] + camera_height
        )
        
        # 相机朝下看
        camera_rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        
        # 创建相机蓝图
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        
        # 生成相机
        camera = world.spawn_actor(camera_bp, camera_transform)
        
        # 设置图像保存路径
        filename = f"{map_name}_intersection_{intersection['id']}_rank{rank}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        # 存储截图标志
        screenshot_saved = [False]
        
        def save_image(image):
            image.save_to_disk(filepath)
            screenshot_saved[0] = True
        
        # 监听图像数据
        camera.listen(save_image)
        
        # 等待截图完成
        timeout = 5.0
        start_time = time.time()
        while not screenshot_saved[0] and (time.time() - start_time) < timeout:
            world.tick()
            time.sleep(0.1)
        
        # 清理相机
        camera.stop()
        camera.destroy()
        
        if screenshot_saved[0]:
            return filepath
        else:
            raise Exception("Screenshot timeout")

    def capture_predefined_intersections(self):
        """为预定义的最佳路口截图"""
        predefined_intersections = {
            'Town03': [
                {'center': (-16.2, -13.5, 0.0), 'id': 1469, 'rank': 1},
                {'center': (-81.9, -138.2, 0.0), 'id': 730, 'rank': 2},
                {'center': (1.6, 194.3, 0.0), 'id': 82, 'rank': 3},
            ],
            'Town04': [
                {'center': (-25.7, 320.3, 0.0), 'id': 134, 'rank': 1},
                {'center': (6.5, -270.8, 0.0), 'id': 1159, 'rank': 2},
                {'center': (1.7, 97.1, 0.0), 'id': 1061, 'rank': 3},
            ],
            'Town05': [
                {'center': (-267.0, 0.5, 0.0), 'id': 1930, 'rank': 1},
                {'center': (-188.9, -89.7, 0.0), 'id': 396, 'rank': 2},
                {'center': (-189.4, 89.1, 0.0), 'id': 562, 'rank': 3},
            ],
        }
        
        print("\n开始为预定义的最佳路口截图...")
        
        for map_name, intersections in predefined_intersections.items():
            print(f"\n=== 为 {map_name} 截图 ===")
            
            try:
                # 加载地图
                self.client.load_world(map_name)
                world = self.client.get_world()
                
                # 等待地图加载完成
                time.sleep(2)
                
                for intersection in intersections:
                    try:
                        screenshot_path = self._capture_intersection_screenshot(
                            map_name, intersection, intersection['rank']
                        )
                        print(f"✓ 路口ID {intersection['id']} (排名{intersection['rank']}) 截图完成: {os.path.basename(screenshot_path)}")
                    except Exception as e:
                        print(f"✗ 路口ID {intersection['id']} 截图失败: {e}")
                        
            except Exception as e:
                print(f"无法加载地图 {map_name}: {e}")
        
        print(f"\n所有截图已保存到: {self.screenshot_dir}")

    def _evaluate_complexity(self, intersection):
        """评估路口复杂度"""
        score = 0
        
        # 基于道路数量
        if intersection['num_roads'] == 4:
            score += 10  # 标准十字路口
        elif intersection['num_roads'] > 4:
            score += 15  # 复杂多路交叉
        
        # 基于车道数量
        if intersection['num_lanes'] <= 8:
            score += 5   # 简单
        elif intersection['num_lanes'] <= 16:
            score += 10  # 中等
        else:
            score += 15  # 复杂
        
        if score <= 15:
            return "简单 ⭐"
        elif score <= 25:
            return "中等 ⭐⭐"
        else:
            return "复杂 ⭐⭐⭐"

def main():
    analyzer = IntersectionAnalyzer()
    
    # 添加一个选项来决定是否进行完整分析还是只截图
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--screenshot-only":
        # 只为预定义的路口截图
        analyzer.capture_predefined_intersections()
        return
    
    # 分析所有CARLA标准地图
    maps_to_analyze = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
    
    all_results = {}
    for map_name in maps_to_analyze:
        try:
            intersections = analyzer.analyze_map_intersections(map_name)
            all_results[map_name] = intersections
        except Exception as e:
            print(f"分析地图 {map_name} 时出错: {e}")
    
    # 生成总结报告
    print("\n" + "="*80)
    print("最佳无信号灯十字路口分析总结")
    print("="*80)
    
    best_overall = []
    for map_name, intersections in all_results.items():
        if intersections:
            print(f"\n{map_name}: {len(intersections)} 个推荐路口")
            for i, intersection in enumerate(intersections, 1):
                complexity = analyzer._evaluate_complexity(intersection)
                width = intersection['width_metric']
                print(f"  推荐{i}: 位置({intersection['center'][0]:.0f}, {intersection['center'][1]:.0f}) - {complexity} - 宽度{width:.0f}")
                best_overall.append((map_name, intersection, width))
        else:
            print(f"\n{map_name}: 未发现合适的无信号灯十字路口")
    
    # 全局最佳路口排名
    print("\n" + "="*60)
    print("全局最佳无信号灯十字路口 TOP 5")
    print("="*60)
    best_overall.sort(key=lambda x: x[2], reverse=True)
    for i, (map_name, intersection, width) in enumerate(best_overall[:5], 1):
        center = intersection['center']
        complexity = analyzer._evaluate_complexity(intersection)
        print(f"{i}. {map_name} - 路口ID{intersection['id']}")
        print(f"   位置: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        print(f"   宽度指标: {width:.1f} | 复杂度: {complexity}")
        print(f"   道路数: {intersection['num_roads']} | 车道数: {intersection['num_lanes']}")
    
    print(f"\n推荐配置更新:")
    print("BEST_UNSIGNALIZED_INTERSECTIONS = {")
    for map_name, intersections in all_results.items():
        if intersections:
            print(f"    '{map_name}': [")
            for i, intersection in enumerate(intersections, 1):
                center = intersection['center']
                complexity = analyzer._evaluate_complexity(intersection)
                print(f"        # 推荐{i}: {complexity}, 宽度{intersection['width_metric']:.0f}")
                print(f"        {{'center': ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), 'id': {intersection['id']}, 'rank': {i}}},")
            print("    ],")
    print("}")

if __name__ == "__main__":
    main()