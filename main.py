import sys
import os
import glob
import math  # 用于数学计算
import time  # 用于时间相关操作


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

# 初始化车队管理 - 传入state_extractor用于导航
platoon_manager = PlatoonManager(state_extractor)

# 初始化分布式拍卖引擎 - 传入state_extractor
auction_engine = DecentralizedAuctionEngine(state_extractor=state_extractor)

# 初始化交通控制器
traffic_controller = TrafficController(scenario.carla, state_extractor)

# 🔥 设置车队管理器引用，用于车队协调控制
traffic_controller.set_platoon_manager(platoon_manager)

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

# 在仿真开始前添加
from traffic_light_override import force_vehicles_run_lights, freeze_lights_green

# 选择一种方法
# force_vehicles_run_lights(scenario.carla.world, scenario.carla.traffic_manager)
# 或者
# freeze_lights_green(scenario.carla.world)

# 主仿真循环
try:
    step = 0
    # 统一所有更新和打印频率
    unified_update_interval = 5  # 统一更新间隔：每5步更新一次
    unified_print_interval = SimulationConfig.PRINT_INTERVAL  # 统一打印间隔
    
    while True:
        scenario.carla.world.tick()
        vehicle_states = state_extractor.get_vehicle_states()
        
        # 统一更新频率：所有子系统同时更新
        if step % unified_update_interval == 0:
            # 1. 更新车队分组
            platoon_manager.update()
            
            # 2. 更新拍卖系统
            auction_engine.update(vehicle_states, platoon_manager)
            
            # 3. 更新交通控制
            traffic_controller.update_control(platoon_manager, auction_engine)
        
        # 统一打印频率：所有状态信息同时输出
        if step % unified_print_interval == 0:
            # 清屏（可选，让输出更清晰）
            os.system('cls' if os.name == 'nt' else 'clear')  # 取消注释以启用清屏
            
            print(f"\n{'='*80}")
            print(f"[Step {step}] 无信号灯交叉路口仿真状态报告")
            print(f"{'='*80}")
            
            # 基础仿真信息
            actual_fps = 1 / SimulationConfig.FIXED_DELTA_SECONDS
            vehicles_in_radius = vehicle_states
            vehicles_in_junction = [v for v in vehicle_states if v['is_junction']]
            
            print(f"📊 基础信息: 总车辆:{len(vehicle_states)} | 路口内:{len(vehicles_in_junction)} | FPS:{actual_fps:.1f}")
            
            # 新增：安全控制状态
            safety_stats = traffic_controller.get_safety_stats()
            if safety_stats['intersection_pass_vehicles'] > 0:
                print(f"🚧 路口通过状态: {safety_stats['intersection_pass_vehicles']}辆正在强制通过路口")
            
            # 1. 车队管理状态
            print(f"\n🚗 车队管理状态:")
            platoon_stats = platoon_manager.get_platoon_stats()
            unplatoon_count = platoon_manager.get_unplatoon_vehicles_count()
            print(f"   车队数:{platoon_stats['num_platoons']} | "
                  f"编队车辆:{platoon_stats['vehicles_in_platoons']} | "
                  f"独行车辆:{unplatoon_count} | "
                  f"平均队长:{platoon_stats['avg_platoon_size']:.1f} | "
                  f"方向分布:{platoon_stats['direction_distribution']}")
            
            # 2. 拍卖系统状态
            print(f"\n🎯 拍卖系统状态:")
            auction_stats = auction_engine.get_auction_stats()
            conflict_stats = auction_engine.conflict_resolver.get_conflict_stats()
            print(f"   活跃竞价:{auction_stats['active_auctions']} | "
                  f"已完成:{auction_stats['completed_auctions']} | "
                  f"参与者:{auction_stats['platoon_participants']}车队+{auction_stats['vehicle_participants']}单车")
            
            # 显示当前优先级排序（前5名）
            priority_order = auction_engine._get_current_priority_order()
            if priority_order:
                print(f"   🏆 当前通行优先级（前5名）:")
                for i, winner in enumerate(priority_order[:5]):
                    agent = winner['agent']
                    bid_value = winner['bid_value']
                    rank = winner['rank']
                    conflict_action = winner.get('conflict_action', 'go')
                    action_emoji = "🟢" if conflict_action == 'go' else "🔴"
                    
                    if agent['type'] == 'platoon':
                        print(f"      #{rank}: {action_emoji}🚛车队{agent['id']} "
                              f"({agent['size']}车-{agent['goal_direction']}) 出价:{bid_value:.1f}")
                    else:
                        print(f"      #{rank}: {action_emoji}🚗单车{agent['id']} "
                              f"({agent.get('goal_direction', 'unknown')}) 出价:{bid_value:.1f}")
            
            # 只在统一打印时显示拍卖状态，避免重复输出
            # auction_engine.print_auction_status()  # 注释掉，减少重复信息

            # 只在统一打印时显示车队信息，避免重复输出
            # platoon_manager.print_platoon_info()  # 注释掉，减少重复信息

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
        
        # 更新车辆ID标签显示（保持原频率）
        scenario.update_vehicle_labels()
                
        step += 1
        
except KeyboardInterrupt:
    print("\n仿真已手动终止。")
    traffic_controller.emergency_reset_all_controls()
