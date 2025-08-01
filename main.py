import sys
import os
import glob
import gc  # Add garbage collection
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
egg_path = []

if sys.platform.startswith('linux'):
    egg_path = glob.glob(os.path.join(base_dir, "carla_l", "carla-*linux-x86_64.egg"))

if egg_path:
    sys.path.insert(0, egg_path[0])  # 更鲁棒地优先导入
else:
    raise RuntimeError(
        "CARLA egg not found.\n"
    )

import carla

# ===== 环境相关模块 =====
from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from env.simulation_config import SimulationConfig

# ===== 车队管理模块 =====
# from platooning.platoon_manager import PlatoonManager

# ===== 拍卖系统模块 =====
from auction.auction_engine import DecentralizedAuctionEngine

# ===== 交通控制模块 =====
from control import TrafficController

# 初始化环境模块
scenario = ScenarioManager()
state_extractor = StateExtractor(scenario.carla)

# 初始化车队管理 - 传入state_extractor用于导航
# platoon_manager = PlatoonManager(state_extractor)

# 初始化分布式拍卖引擎 - 传入state_extractor
auction_engine = DecentralizedAuctionEngine(state_extractor=state_extractor)

# 初始化交通控制器
traffic_controller = TrafficController(scenario.carla, state_extractor)

# DISABLED: Platoon manager reference removed
# traffic_controller.set_platoon_manager(platoon_manager)

# 显示地图信息
spawn_points = scenario.carla.world.get_map().get_spawn_points()
print(f"=== 无信号灯交叉路口仿真 (集成拍卖系统) ===")

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
    unified_update_interval = 10
    unified_print_interval = SimulationConfig.PRINT_INTERVAL
    
    while True:
        scenario.carla.world.tick()
        vehicle_states = state_extractor.get_vehicle_states()
        
        if step % unified_update_interval == 0:
            # DISABLED: Platoon update removed
            # 1. 更新车队分组
            # platoon_manager.update()
            
            # 2. 更新拍卖系统 (single vehicles only)
            auction_winners = auction_engine.update(vehicle_states, None)  # Pass None instead of platoon_manager
            
            # 3. 更新交通控制 (single vehicles only)
            traffic_controller.update_control(None, auction_engine)  # Pass None for platoon_manager
        
        # 统一打印频率：所有状态信息同时输出
        if step % unified_print_interval == 0:
            # 清屏（可选，让输出更清晰）
            os.system('clear')  # Linux: use 'clear' to clear the terminal
            
            print(f"\n{'='*80}")
            print(f"[Step {step}] 无信号灯交叉路口仿真状态报告 - 单车模式")
            print(f"{'='*80}")
            
            # 基础仿真信息
            actual_fps = 1 / SimulationConfig.FIXED_DELTA_SECONDS
            vehicles_in_radius = vehicle_states
            vehicles_in_junction = [v for v in vehicle_states if v['is_junction']]
            
            print(f"📊 基础信息: FPS:{actual_fps:.1f}, 车辆总数:{len(vehicles_in_radius)}, 路口内:{len(vehicles_in_junction)}")
        
            # DISABLED: Platoon status reporting removed
            # 1. 车队管理状态
            # platoon_manager.print_platoon_info()
            print(f"🚫 车队管理: 已暂时禁用（专注单车行为）")
            
            # 2. 拍卖系统状态
            print(f"\n🎯 拍卖系统状态:")
            auction_stats = auction_engine.get_auction_stats()
            print(f"   活跃竞价: {'是' if auction_stats['active_auction'] else '否'} | "
                  f"已完成: {auction_stats['completed_auctions']} | "
                  f"参与者: {auction_stats['vehicle_participants']}独立车辆")  # Removed platoon count
            
            # 显示当前优先级排序（前5名）
            priority_order = auction_engine.get_current_priority_order()
            if priority_order:
                print(f"   🏆 当前通行优先级（前5名）:")
                for winner in priority_order[:5]:
                    participant = winner.participant
                    bid_value = winner.bid.value
                    rank = winner.rank
                    conflict_action = winner.conflict_action
                    action_emoji = "🟢" if conflict_action == 'go' else "🔴"
                    protection_emoji = "🛡️" if winner.protected else ""
                    
                    # SIMPLIFIED: Only show vehicle info
                    print(f"      #{rank}: {action_emoji}{protection_emoji}🚗车辆{participant.id} "
                          f"出价:{bid_value:.1f}")
            
            # 3. 控制器状态
            control_stats = traffic_controller.get_control_stats()
            if control_stats['total_controlled'] > 0:
                print(f"🎮 控制器状态: 总控制:{control_stats['total_controlled']} | "
                      f"等待:{control_stats['waiting_vehicles']} | "
                      f"通行:{control_stats['go_vehicles']}")
        
        # 更新车辆ID标签显示（保持原频率）
        scenario.update_vehicle_labels()
                
        step += 1

except KeyboardInterrupt:
    print("\n仿真已手动终止。")


