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

# 新增导入
from controls.vehicle_enforcer import VehicleControlEnforcer

# 初始化环境模块
scenario = ScenarioManager()
state_extractor = StateExtractor(scenario.carla)

# 初始化车队管理 - 传入state_extractor用于导航
platoon_manager = PlatoonManager(state_extractor)

# 初始化分布式拍卖引擎 - 传入state_extractor
auction_engine = DecentralizedAuctionEngine(state_extractor=state_extractor)

# 初始化交通控制器
traffic_controller = TrafficController(scenario.carla, state_extractor)

# 初始化车辆控制强制器
vehicle_enforcer = VehicleControlEnforcer(scenario.carla, state_extractor)

# 设置引用关系
traffic_controller.set_platoon_manager(platoon_manager)
vehicle_enforcer.set_platoon_manager(platoon_manager)
auction_engine.set_vehicle_enforcer(vehicle_enforcer)

# 显示地图信息
spawn_points = scenario.carla.world.get_map().get_spawn_points()
print(f"=== 无信号灯交叉路口仿真 (集成拍卖系统) ===")
# print(f"当前地图: {SimulationConfig.MAP_NAME}")
# print(f"spawn点数量: {len(spawn_points)}")
# print(f"预计车辆数: {len(spawn_points)}")
# print("=============================")

# 生成交通流
scenario.reset_scenario()
scenario.show_intersection_area()

# 在仿真开始前添加
# from traffic_light_override import force_vehicles_run_lights, freeze_lights_green

# 选择一种方法
# force_vehicles_run_lights(scenario.carla.world, scenario.carla.traffic_manager)
# 或者
# freeze_lights_green(scenario.carla.world)

# 主仿真循环
try:
    step = 0
    unified_update_interval = 5
    unified_print_interval = SimulationConfig.PRINT_INTERVAL
    
    while True:
        scenario.carla.world.tick()
        vehicle_states = state_extractor.get_vehicle_states()
        
        if step % unified_update_interval == 0:
            # 1. 更新车队分组
            platoon_manager.update()
            
            # 2. 更新拍卖系统（包含强制控制）
            auction_engine.update(vehicle_states, platoon_manager)
            
            # 3. 更新交通控制
            traffic_controller.update_control(platoon_manager, auction_engine)
            
            # 4. 🔥 新增：清理过期的强制控制
            vehicle_enforcer.cleanup_expired_controls()
        
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
            
            print(f"📊 基础信息: FPS:{actual_fps:.1f}")
            
            # 新增：安全控制状态
            safety_stats = traffic_controller.get_safety_stats()
            if safety_stats['intersection_pass_vehicles'] > 0:
                print(f"🚧 路口通过状态: {safety_stats['intersection_pass_vehicles']}辆正在强制通过路口")
            
            # 1. 车队管理状态
            platoon_manager.print_platoon_info()
            
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
            
            # 🔥 新增：显示强制控制状态
            enforcer_stats = vehicle_enforcer.get_enforcement_stats()
            if enforcer_stats['total_enforced'] > 0:
                print(f"🎮 强制控制状态: 总控制:{enforcer_stats['total_enforced']} | "
                      f"等待:{enforcer_stats['waiting_vehicles']} | "
                      f"通行:{enforcer_stats['go_vehicles']}")
        
        # 更新车辆ID标签显示（保持原频率）
        scenario.update_vehicle_labels()
                
        step += 1
        
except KeyboardInterrupt:
    print("\n仿真已手动终止。")
    # 🔥 新增：紧急释放所有控制
    vehicle_enforcer.emergency_release_all()
    traffic_controller.emergency_reset_all_controls()
