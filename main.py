import sys
import os
import glob


egg_path = glob.glob(os.path.join("carla", "carla-*.egg"))
if egg_path:
    sys.path.append(egg_path[0])
else:
    raise RuntimeError("CARLA egg not found in ./carla/ folder.")

# ===== 环境相关模块 =====
from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from env.simulation_config import SimulationConfig

# ===== 车队管理模块 =====
from platooning.platoon_manager import PlatoonManager
# from platooning.platoon_policy import PlatoonPolicy

# ===== 保留但暂不启用的模块 =====
# from auction.auction_engine import AuctionEngine
# from nash.nash_solver import NashSolver
# from rl.agents.ppo import PPOAgent
# from policies.composite_policy import CompositePolicy

# 初始化环境模块（使用配置文件中的地图）
scenario = ScenarioManager()
state_extractor = StateExtractor(scenario.carla)

# 初始化车队管理
platoon_manager = PlatoonManager(state_extractor)

# 显示地图信息
spawn_points = scenario.carla.world.get_map().get_spawn_points()
print(f"=== 无信号灯交叉路口仿真 ===")
print(f"当前地图: {SimulationConfig.MAP_NAME}")
print(f"spawn点数量: {len(spawn_points)}")
print(f"预计车辆数: {len(spawn_points)}")

print("=============================")

# 生成交通流
scenario.reset_scenario()
scenario.show_intersection_area()  # 新增：显示检测点和半径

# 主仿真循环（持续运行，Ctrl+C可中断）
try:
    step = 0
    print_interval = SimulationConfig.PRINT_INTERVAL
    platoon_update_interval = 10  # 每10步更新一次车队
    
    while True:
        # 先让仿真前进
        scenario.carla.world.tick()
        
        # 获取车辆状态
        vehicle_states = state_extractor.get_vehicle_states()
        
        # 定期更新车队分组
        if step % platoon_update_interval == 0:
            platoon_manager.update()
            if platoon_manager.get_all_platoons():
                platoon_stats = platoon_manager.get_platoon_stats()
                print(f"车队更新 - 车队数: {platoon_stats['num_platoons']}, "
                      f"编队车辆: {platoon_stats['vehicles_in_platoons']}, "
                      f"平均车队大小: {platoon_stats['avg_platoon_size']:.1f}")
        
        # 减少打印频率以提升性能
        if step % print_interval == 0:
            actual_fps = 1 / SimulationConfig.FIXED_DELTA_SECONDS
            print(f"[Step {step}] Vehicle Total: {len(vehicle_states)} | FPS: {actual_fps:.1f}")

            # 统计并打印交叉口附近车辆信息
            vehicles_in_radius = vehicle_states
            vehicles_in_junction = [v for v in vehicle_states if v['is_junction']]
            
            print(f"--- Vehicles in {SimulationConfig.INTERSECTION_RADIUS}m Radius: {len(vehicles_in_radius)} ---")
            print(f"--- Vehicles in Junction Area: {len(vehicles_in_junction)} ---")

            # 打印车队信息
            if step % (print_interval * 3) == 0:  # 每90步打印一次详细车队信息
                platoon_manager.print_platoon_info()

            for v in vehicles_in_radius[:10]:  # 显示半径内的前10辆车
                speed_kmh = (v['velocity'][0]**2 + v['velocity'][1]**2)**0.5 * 3.6
                dist_to_center = v.get('distance_to_center', 0)
                junction_status = "Junction" if v['is_junction'] else "Road"
                print(
                    f"  [ID: {v['id']}] "
                    f"Pos: ({v['location'][0]:.1f}, {v['location'][1]:.1f}) | "
                    f"Speed: {speed_kmh:.1f} km/h | "
                    f"Road/Lane: {v['road_id']}/{v['lane_id']} | "
                    f"Status: {junction_status} | "
                    f"LeadDist: {v['leading_vehicle_dist']:.1f} m | "
                    f"CenterDist: {dist_to_center:.1f} m"
                )
        
        # 更新车辆ID标签显示
        scenario.update_vehicle_labels()
                
        step += 1
        
except KeyboardInterrupt:
    print("\n仿真已手动终止。")

# 原计划主循环结构（暂注释）
"""
for episode in range(num_episodes):
    env.reset()
    while not env.done():
        state = env.get_state()

        # 1. 编队分组
        platoon_mgr.update(state)

        # 2. 拍卖竞价
        bids = auction_engine.collect_bids(state)

        # 3. 拍卖冲突 → 纳什均衡解决
        order = nash_solver.resolve_conflict(bids)

        # 4. PPO策略输出行为（基于分布式/邻域信息）
        actions = policy.get_actions(state, order)

        # 5. 车辆执行动作
        env.step(actions)

        # 6. 更新DRL训练
        ppo_agent.observe(state, actions, reward, next_state, done)
"""
