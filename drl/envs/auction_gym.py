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
        
        # FIXED observation space - ensure exact consistency
        obs_dim = 195  # Fixed dimension matching sim_wrapper
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0,  # Reasonable bounds
            shape=(obs_dim,), dtype=np.float32
        )
        
        # VALIDATED action space
        self.action_space = spaces.Box(
            low=np.array([
                0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -30.0, -2.0, 0.0, 0.0, 0.0
            ], dtype=np.float32),
            high=np.array([
                5.0, 3.0, 1.0, 1.0, 2.0, 1.0, 0.5, 10.0, 3.0, 30.0, 3.0, 100.0, 50.0, 100.0
            ], dtype=np.float32),
            shape=(14,), 
            dtype=np.float32
        )
        
        self.current_obs = None
        self.render_mode = None
        
        print("🎮 FIXED Auction Gym Environment initialized")
        print(f"   观察空间: {self.observation_space.shape} (FIXED 195维)")
        print(f"   动作空间: {self.action_space.shape} (14个参数)")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """FIXED reset with proper validation"""
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed)
        
        # ENSURE exact dimension match
        expected_shape = 195
        if obs.shape[0] != expected_shape:
            print(f"⚠️ FIXING observation dimension: {obs.shape[0]} -> {expected_shape}")
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:expected_shape]
        
        # Validate data
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        self.current_obs = obs
        
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple:
        """FIXED step with validation"""
        # VALIDATE action bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Parse parameters with validation
        action_params = {
            'bid_scale': float(np.clip(action[0], 0.1, 5.0)),
            'eta_weight': float(np.clip(action[1], 0.5, 3.0)),
            'speed_weight': float(np.clip(action[2], 0.0, 1.0)),
            'congestion_sensitivity': float(np.clip(action[3], 0.0, 1.0)),
            'platoon_bonus': float(np.clip(action[4], 0.0, 2.0)),
            'junction_penalty': float(np.clip(action[5], 0.0, 1.0)),
            'fairness_factor': float(np.clip(action[6], 0.0, 0.5)),
            'urgency_threshold': float(np.clip(action[7], 1.0, 10.0)),
            'proximity_bonus_weight': float(np.clip(action[8], 0.0, 3.0)),
            'speed_diff_modifier': float(np.clip(action[9], -30.0, 30.0)),
            'follow_distance_modifier': float(np.clip(action[10], -2.0, 3.0)),
            'ignore_vehicles_go': float(np.clip(action[11], 0.0, 100.0)),
            'ignore_vehicles_wait': float(np.clip(action[12], 0.0, 50.0)),
        }
        
        # FIXED platoon ignore vehicles calculation
        avg_platoon_ignore = float(np.clip(action[13], 0.0, 100.0))
        action_params['ignore_vehicles_platoon_leader'] = np.clip(avg_platoon_ignore - 20.0, 0.0, 80.0)
        action_params['ignore_vehicles_platoon_follower'] = np.clip(avg_platoon_ignore + 20.0, 50.0, 100.0)
        
        # Update simulation
        obs, reward, done, info = self.sim.step_with_all_params(action_params)
        
        # ENSURE exact dimension match
        expected_shape = 195
        if obs.shape[0] != expected_shape:
            print(f"⚠️ FIXING step observation: {obs.shape[0]} -> {expected_shape}")
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:expected_shape]
        
        # Validate all outputs
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)
        reward = np.clip(float(reward), -1000.0, 1000.0)
        done = bool(done)
        
        self.current_obs = obs
        
        # Enhanced info
        info.update({
            'action_params': action_params,
            'observation_validated': True,
            'observation_shape': obs.shape[0]
        })
        
        return obs, reward, done, info

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
