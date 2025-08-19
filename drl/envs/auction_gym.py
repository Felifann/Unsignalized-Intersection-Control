import gym
import numpy as np
from gym import spaces
from typing import Optional, Dict, Any, List

from drl.envs.sim_wrapper import SimulationEnv

class AuctionGymEnv(gym.Env):
    """Enhanced Gym environment for traffic intersection auction system"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, sim_cfg: Dict = None):
        super().__init__()
        
        self.sim_cfg = sim_cfg or {}
        self.sim = SimulationEnv(self.sim_cfg)
        
        # Define observation space - 确保与sim_wrapper一致
        obs_dim = self.sim.observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # 扩展动作空间 - 包含所有14个可训练参数
        self.action_space = spaces.Box(
            low=np.array([
                0.1,  # bid_scale
                0.5,  # eta_weight
                0.0,  # speed_weight
                0.0,  # congestion_sensitivity
                0.0,  # platoon_bonus
                0.0,  # junction_penalty
                0.0,  # fairness_factor
                1.0,  # urgency_threshold
                0.0,  # proximity_bonus_weight
                -30.0, # speed_diff_modifier
                -2.0, # follow_distance_modifier
                0.0,  # ignore_vehicles_go
                0.0,  # ignore_vehicles_wait
                0.0   # avg_ignore_vehicles_platoon (leader+follower)/2
            ], dtype=np.float32),
            high=np.array([
                5.0,  # bid_scale
                3.0,  # eta_weight
                1.0,  # speed_weight
                1.0,  # congestion_sensitivity
                2.0,  # platoon_bonus
                1.0,  # junction_penalty
                0.5,  # fairness_factor
                10.0, # urgency_threshold
                3.0,  # proximity_bonus_weight
                30.0, # speed_diff_modifier
                3.0,  # follow_distance_modifier
                100.0, # ignore_vehicles_go
                50.0, # ignore_vehicles_wait
                100.0 # avg_ignore_vehicles_platoon
            ], dtype=np.float32),
            shape=(14,), 
            dtype=np.float32
        )
        
        self.current_obs = None
        self.render_mode = None
        
        print("🎮 完全扩展的Auction Gym Environment初始化")
        print(f"   观察空间: {self.observation_space.shape} (确保209维)")
        print(f"   动作空间: {self.action_space.shape} (14个可训练参数)")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """Reset environment"""
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed)
        self.current_obs = obs
        
        # 验证观察维度
        expected_shape = self.observation_space.shape[0]
        if obs.shape[0] != expected_shape:
            print(f"⚠️ Reset观察维度不匹配: 期望 {expected_shape}, 实际 {obs.shape[0]}")
            # 确保维度正确
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:expected_shape]
        
        return obs

    def step(self, action: np.ndarray) -> tuple:
        """Enhanced step with all trainable parameters"""
        # 解析所有14个参数
        action_params = {
            'bid_scale': float(action[0]),
            'eta_weight': float(action[1]),
            'speed_weight': float(action[2]),
            'congestion_sensitivity': float(action[3]),
            'platoon_bonus': float(action[4]),
            'junction_penalty': float(action[5]),
            'fairness_factor': float(action[6]),
            'urgency_threshold': float(action[7]),
            'proximity_bonus_weight': float(action[8]),
            'speed_diff_modifier': float(action[9]),
            'follow_distance_modifier': float(action[10]),
            'ignore_vehicles_go': float(action[11]),
            'ignore_vehicles_wait': float(action[12]),
        }
        
        # 处理车队的ignore_vehicles参数 (从平均值计算)
        avg_platoon_ignore = float(action[13])
        action_params['ignore_vehicles_platoon_leader'] = max(0.0, avg_platoon_ignore - 20.0)
        action_params['ignore_vehicles_platoon_follower'] = min(100.0, avg_platoon_ignore + 20.0)
        
        # 更新仿真
        obs, reward, done, info = self.sim.step_with_all_params(action_params)
        
        # 验证观察维度
        expected_shape = self.observation_space.shape[0]
        if obs.shape[0] != expected_shape:
            print(f"⚠️ Step观察维度不匹配: 期望 {expected_shape}, 实际 {obs.shape[0]}")
            # 确保维度正确
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:expected_shape]
        
        self.current_obs = obs
        
        # 增强信息包含所有参数
        info.update({
            'action_params': action_params,
            'total_trainable_params': 14,
            'observation_shape': obs.shape[0]
        })
        
        return obs, float(reward), bool(done), info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Enhanced render with visualization options"""
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            print(f"Unsupported render mode: {mode}")

    def _render_human(self):
        """Human-readable console rendering"""
        if hasattr(self.sim, 'metrics'):
            print(f"\n🎮 Simulation State:")
            print(f"   Throughput: {self.sim.metrics['throughput']:.1f} vehicles/h")
            print(f"   Avg Acceleration: {self.sim.metrics['avg_acceleration']:.3f} m/s²")
            print(f"   Collisions: {self.sim.metrics['collision_count']}")
            print(f"   Step: {self.sim.current_step}/{self.sim.max_steps}")
            
            # Policy information
            if hasattr(self.sim, 'bid_policy'):
                policy_stats = self.sim.bid_policy.get_policy_stats()
                print(f"   Bid Scale: {policy_stats.get('current_bid_scale', 0):.2f}")
                print(f"   Success Rate: {policy_stats.get('success_rate', 0):.1%}")

    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for video recording"""
        # This would require implementing a visual renderer
        # For now, return a placeholder
        return np.zeros((600, 800, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close environment"""
        if hasattr(self, 'sim'):
            self.sim.close()
        print("🏁 Enhanced Auction Gym Environment closed")

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action descriptions for all 14 parameters"""
        return [
            "Bid Scale (0.1-5.0): 总体出价缩放因子",
            "ETA Weight (0.5-3.0): ETA到达时间权重", 
            "Speed Weight (0.0-1.0): 车辆速度权重",
            "Congestion Sensitivity (0.0-1.0): 拥堵敏感度",
            "Platoon Bonus (0.0-2.0): 车队奖励系数",
            "Junction Penalty (0.0-1.0): 路口位置惩罚",
            "Fairness Factor (0.0-0.5): 公平性调节因子",
            "Urgency Threshold (1.0-10.0): 紧急度阈值",
            "Proximity Bonus Weight (0.0-3.0): 邻近性奖励权重",
            "Speed Diff Modifier (-30 to +30): 速度控制修正",
            "Follow Distance Modifier (-2 to +3): 跟车距离修正",
            "Ignore Vehicles Go (0-100): GO状态ignore_vehicles%",
            "Ignore Vehicles Wait (0-50): WAIT状态ignore_vehicles%",
            "Avg Platoon Ignore Vehicles (0-100): 车队平均ignore_vehicles%"
        ]

    def get_reward_info(self) -> Dict[str, str]:
        """Get information about reward components"""
        return {
            "throughput": "Vehicles successfully exiting intersection (+10 per vehicle)",
            "safety": "Collision avoidance (-100 per collision)",
            "efficiency": "Smooth acceleration patterns (+5 for low jerk)",
            "utilization": "Optimal intersection usage (+5 for good ratios)",
            "deadlock_penalty": "Deadlock avoidance (-50 per deadlock)",
            "step_penalty": "Encourage efficiency (-0.1 per step)"
        }

    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety-related metrics"""
        if hasattr(self.sim, 'metrics'):
            return {
                'collision_count': self.sim.metrics.get('collision_count', 0),
                'deadlock_detections': getattr(self.sim, 'deadlock_detector', None).get_stats().get('deadlocks_detected', 0) if hasattr(self.sim, 'deadlock_detector') else 0,
                'safety_score': self._calculate_safety_score()
            }
        return {}

    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score (0-1, higher is safer)"""
        if not hasattr(self.sim, 'metrics'):
            return 1.0
        
        collision_penalty = min(self.sim.metrics.get('collision_count', 0) * 0.1, 0.5)
        deadlock_penalty = 0.0
        
        if hasattr(self.sim, 'deadlock_detector'):
            deadlock_stats = self.sim.deadlock_detector.get_stats()
            deadlock_penalty = min(deadlock_stats.get('deadlocks_detected', 0) * 0.2, 0.3)
        
        safety_score = max(0.0, 1.0 - collision_penalty - deadlock_penalty)
        return safety_score
