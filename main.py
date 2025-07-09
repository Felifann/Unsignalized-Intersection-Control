import sys
import os
import glob


egg_path = glob.glob(os.path.join("carla", "carla-*.egg"))
if egg_path:
    sys.path.append(egg_path[0])
else:
    raise RuntimeError("CARLA egg not found in ./carla/ folder.")

import carla

# ===== 环境相关模块 =====
from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from env.simulation_config import SimulationConfig

# ===== 车队管理模块 =====
from platooning.platoon_manager import PlatoonManager

# ===== 拍卖系统模块 =====
from auction.auction_engine import DecentralizedAuctionEngine

# ===== 交通控制模块 =====
from control import TrafficController

# 初始化环境模块
scenario = ScenarioManager()
state_extractor = StateExtractor(scenario.carla)

# 初始化车队管理
platoon_manager = PlatoonManager(state_extractor)

# 初始化分布式拍卖引擎
auction_engine = DecentralizedAuctionEngine()

# 初始化交通控制器
traffic_controller = TrafficController(scenario.carla, state_extractor)

# 显示地图信息
spawn_points = scenario.carla.world.get_map().get_spawn_points()
print(f"=== 无信号灯交叉路口仿真 (集成拍卖系统) ===")
print(f"当前地图: {SimulationConfig.MAP_NAME}")
print(f"spawn点数量: {len(spawn_points)}")
print(f"预计车辆数: {len(spawn_points)}")
print("=============================")

# 生成交通流
scenario.reset_scenario()
scenario.show_intersection_area()

# 主仿真循环
try:
    step = 0
    print_interval = SimulationConfig.PRINT_INTERVAL
    platoon_update_interval = 10
    control_update_interval = 5
    auction_update_interval = 8  # 拍卖更新间隔
    
    while True:
        scenario.carla.world.tick()
        vehicle_states = state_extractor.get_vehicle_states()
        
        # 定期更新车队分组
        if step % platoon_update_interval == 0:
            platoon_manager.update_and_print_stats()
        
        # 定期更新拍卖系统
        if step % auction_update_interval == 0:
            auction_engine.update(vehicle_states, platoon_manager)
        
        # 定期更新交通控制（现在包含拍卖控制）
        if step % control_update_interval == 0:
            traffic_controller.update_control(platoon_manager, auction_engine)
        
        # 减少打印频率
        if step % print_interval == 0:
            actual_fps = 1 / SimulationConfig.FIXED_DELTA_SECONDS
            print(f"[Step {step}] Vehicle Total: {len(vehicle_states)} | FPS: {actual_fps:.1f}")

            vehicles_in_radius = vehicle_states
            vehicles_in_junction = [v for v in vehicle_states if v['is_junction']]
            
            print(f"--- Vehicles in {SimulationConfig.INTERSECTION_RADIUS}m Radius: {len(vehicles_in_radius)} ---")
            print(f"--- Vehicles in Junction Area: {len(vehicles_in_junction)} ---")

            # 打印控制状态（现在包含拍卖信息）
            traffic_controller.print_control_status()
            
            # 打印拍卖状态
            auction_engine.print_auction_status()

            # 打印车队信息
            if step % (print_interval * 3) == 0:
                platoon_manager.print_platoon_info()

            # for v in vehicles_in_radius[:3]:  # 显示半径内的前10辆车
            #     speed_kmh = (v['velocity'][0]**2 + v['velocity'][1]**2)**0.5 * 3.6
            #     dist_to_center = v.get('distance_to_center', 0)
            #     junction_status = "Junction" if v['is_junction'] else "Road"
            #     print(
            #         f"  [ID: {v['id']}] "
            #         f"Pos: ({v['location'][0]:.1f}, {v['location'][1]:.1f}) | "
            #         f"Speed: {speed_kmh:.1f} km/h | "
            #         f"Road/Lane: {v['road_id']}/{v['lane_id']} | "
            #         f"Status: {junction_status} | "
            #         f"LeadDist: {v['leading_vehicle_dist']:.1f} m | "
            #         f"CenterDist: {dist_to_center:.1f} m"
            #     )
        
        # 更新车辆ID标签显示
        scenario.update_vehicle_labels()
                
        step += 1
        
except KeyboardInterrupt:
    print("\n仿真已手动终止。")
    traffic_controller.emergency_reset_all_controls()
