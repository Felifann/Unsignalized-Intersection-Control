# main.py
from env.carla_wrapper import CarlaEnv
from platooning.platoon_manager import PlatoonManager
from auction.auction_engine import AuctionEngine
from nash.nash_solver import NashSolver
from rl.agents.ppo import PPOAgent
from policies.composite_policy import CompositePolicy

# 初始化环境
env = CarlaEnv()
ppo_agent = PPOAgent()
platoon_mgr = PlatoonManager()
auction_engine = AuctionEngine()
nash_solver = NashSolver()
policy = CompositePolicy(platoon_mgr, auction_engine, nash_solver, ppo_agent)

# 主仿真循环
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
