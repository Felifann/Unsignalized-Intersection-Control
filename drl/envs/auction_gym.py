import gym
import numpy as np
from gym import spaces
from typing import Optional, Dict, Any, List

from drl.envs.sim_wrapper import SimulationEnv
from config.unified_config import UnifiedConfig, get_config

class AuctionGymEnv(gym.Env):
    """Enhanced Gym environment for traffic intersection auction system"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, sim_cfg: Dict = None, unified_config: UnifiedConfig = None):
        super().__init__()
        
        # Use unified config or get global config
        if unified_config is None:
            unified_config = get_config()
        
        self.unified_config = unified_config
        self.sim_cfg = sim_cfg or {}
        
        # Pass unified config to simulation environment
        self.sim = SimulationEnv(self.sim_cfg, unified_config=unified_config)
        
        # FIXED observation space - ensure exact consistency
        obs_dim = 195  # Fixed dimension matching sim_wrapper
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0,  # Reasonable bounds
            shape=(obs_dim,), dtype=np.float32
        )
        
        # OPTIMIZED action space - quantized ranges for efficiency
        self.action_space = spaces.Box(
            low=np.array([
                0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -30.0, -2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 5.0
            ], dtype=np.float32),
            high=np.array([
                5.0, 3.0, 1.0, 1.0, 2.0, 1.0, 0.5, 10.0, 3.0, 30.0, 3.0, 100.0, 50.0, 100.0, 8.0, 5.0, 20.0
            ], dtype=np.float32),
            shape=(17,), 
            dtype=np.float32
        )
        
        self.current_obs = None
        self.render_mode = None
        
        print("🎮 OPTIMIZED Auction Gym Environment initialized")
        print(f"   观察空间: {self.observation_space.shape} (FIXED 195维)")
        print(f"   动作空间: {self.action_space.shape} (17个参数 - 量化优化版本)")
        print(f"   🔧 Config - Conflict window: {unified_config.conflict.conflict_time_window}s")
        print(f"   🔧 Config - Deadlock threshold: {unified_config.deadlock.deadlock_speed_threshold} m/s")
        print(f"   🎯 NEW - Path intersection threshold: {unified_config.conflict.path_intersection_threshold}m")
        print(f"   🎯 NEW - Platoon conflict distance: {unified_config.conflict.platoon_conflict_distance}m")
        print(f"   ⚡ OPTIMIZATION - Parameters quantized for faster training")

    def _quantize_param(self, value: float, min_val: float, max_val: float, step_size: float) -> float:
        """Quantize parameter to discrete steps for efficient training"""
        # Clamp to bounds
        value = np.clip(value, min_val, max_val)
        # Quantize to nearest step
        steps = round((value - min_val) / step_size)
        quantized = min_val + steps * step_size
        # Ensure still within bounds after quantization
        return float(np.clip(quantized, min_val, max_val))

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
        """OPTIMIZED step with efficient parameter handling"""
        # VALIDATE action bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # OPTIMIZED parameter parsing - quantized for faster training
        action_params = {
            # Core bidding parameters (reduced precision)
            'bid_scale': self._quantize_param(action[0], 0.1, 5.0, 0.2),  # Steps of 0.2
            'eta_weight': self._quantize_param(action[1], 0.5, 3.0, 0.25),  # Steps of 0.25
            'speed_weight': self._quantize_param(action[2], 0.0, 1.0, 0.1),  # Steps of 0.1
            'congestion_sensitivity': self._quantize_param(action[3], 0.0, 1.0, 0.1),  # Steps of 0.1
            'platoon_bonus': self._quantize_param(action[4], 0.0, 2.0, 0.2),  # Steps of 0.2
            'junction_penalty': self._quantize_param(action[5], 0.0, 1.0, 0.1),  # Steps of 0.1
            'fairness_factor': self._quantize_param(action[6], 0.0, 0.5, 0.05),  # Steps of 0.05
            'urgency_threshold': self._quantize_param(action[7], 1.0, 10.0, 1.0),  # Integer steps
            'proximity_bonus_weight': self._quantize_param(action[8], 0.0, 3.0, 0.25),  # Steps of 0.25
            
            # Control parameters (coarser quantization)
            'speed_diff_modifier': self._quantize_param(action[9], -30.0, 30.0, 5.0),  # Steps of 5
            'follow_distance_modifier': self._quantize_param(action[10], -2.0, 3.0, 0.5),  # Steps of 0.5
            
            # Percentage parameters (10% steps for efficiency)
            'ignore_vehicles_go': self._quantize_param(action[11], 0.0, 100.0, 10.0),  # 10% steps
            'ignore_vehicles_wait': self._quantize_param(action[12], 0.0, 50.0, 10.0),  # 10% steps
            
            # Discrete integer parameter
            'max_participants_per_auction': int(np.clip(np.round(action[14]), 4, 8)),  # Exact integers 4-8
            
            # Nash parameters (coarser for stability)
            'path_intersection_threshold': self._quantize_param(action[15], 1.0, 5.0, 0.5),  # Steps of 0.5m
            'platoon_conflict_distance': self._quantize_param(action[16], 5.0, 20.0, 2.5),   # Steps of 2.5m
        }
        
        # OPTIMIZED platoon ignore vehicles (10% step quantization)
        avg_platoon_ignore = self._quantize_param(action[13], 0.0, 100.0, 10.0)
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
        """Get human-readable action descriptions for all 17 QUANTIZED parameters"""
        return [
            "Bid Scale (0.1-5.0, steps=0.2): 总体出价缩放因子 ⚡",
            "ETA Weight (0.5-3.0, steps=0.25): ETA到达时间权重 ⚡", 
            "Speed Weight (0.0-1.0, steps=0.1): 车辆速度权重 ⚡",
            "Congestion Sensitivity (0.0-1.0, steps=0.1): 拥堵敏感度 ⚡",
            "Platoon Bonus (0.0-2.0, steps=0.2): 车队奖励系数 ⚡",
            "Junction Penalty (0.0-1.0, steps=0.1): 路口位置惩罚 ⚡",
            "Fairness Factor (0.0-0.5, steps=0.05): 公平性调节因子 ⚡",
            "Urgency Threshold (1-10, integer): 紧急度阈值 🔢",
            "Proximity Bonus Weight (0.0-3.0, steps=0.25): 邻近性奖励权重 ⚡",
            "Speed Diff Modifier (-30 to +30, steps=5): 速度控制修正 ⚡",
            "Follow Distance Modifier (-2 to +3, steps=0.5): 跟车距离修正 ⚡",
            "Ignore Vehicles Go (0-100%, steps=10%): GO状态ignore_vehicles% ⚡",
            "Ignore Vehicles Wait (0-50%, steps=10%): WAIT状态ignore_vehicles% ⚡",
            "Avg Platoon Ignore Vehicles (0-100%, steps=10%): 车队平均ignore_vehicles% ⚡",
            "Max Participants Per Auction (4-8, integer): 每轮拍卖最大参与者数量 🔢",
            "Path Intersection Threshold (1.0-5.0, steps=0.5m): Nash路径交叉检测敏感度 ⚡",
            "Platoon Conflict Distance (5.0-20.0, steps=2.5m): Nash车队冲突检测距离 ⚡"
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
                'deadlock_detections': self._get_deadlock_stats(),
                'deadlock_severity': self._get_deadlock_severity(),
                'safety_score': self._calculate_safety_score()
            }
        return {}

    def _get_deadlock_stats(self) -> int:
        """Get deadlock detection count from the simulation environment"""
        try:
            if (hasattr(self.sim, 'nash_solver') and 
                hasattr(self.sim.nash_solver, 'deadlock_detector') and
                hasattr(self.sim.nash_solver.deadlock_detector, 'stats')):
                return self.sim.nash_solver.deadlock_detector.stats.get('deadlocks_detected', 0)
            elif hasattr(self.sim, 'deadlock_detector'):
                # Fallback for direct deadlock detector access
                deadlock_stats = self.sim.deadlock_detector.get_stats()
                return deadlock_stats.get('deadlocks_detected', 0)
            return 0
        except Exception as e:
            print(f"⚠️ Error getting deadlock stats: {str(e)}")
            return 0
    
    def _get_deadlock_severity(self) -> float:
        """Get current deadlock severity level (0-1)"""
        try:
            if (hasattr(self.sim, 'nash_solver') and 
                hasattr(self.sim.nash_solver, 'deadlock_detector') and
                hasattr(self.sim.nash_solver.deadlock_detector, 'get_deadlock_severity')):
                return self.sim.nash_solver.deadlock_detector.get_deadlock_severity()
            return 0.0
        except Exception as e:
            print(f"⚠️ Error getting deadlock severity: {str(e)}")
            return 0.0

    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score (0-1, higher is safer) with severity consideration"""
        if not hasattr(self.sim, 'metrics'):
            return 1.0
        
        collision_penalty = min(self.sim.metrics.get('collision_count', 0) * 0.1, 0.5)
        
        # Enhanced deadlock penalty with severity consideration
        deadlock_count = self._get_deadlock_stats()
        deadlock_severity = self._get_deadlock_severity()
        
        # Graduated penalty based on both count and severity
        if deadlock_count > 0:
            deadlock_penalty = min(deadlock_count * 0.3, 0.4)  # Base penalty for deadlock occurrence
        else:
            deadlock_penalty = 0.0
            
        # Additional penalty for near-deadlock situations (severity-based)
        severity_penalty = min(deadlock_severity * 0.3, 0.3)  # Up to 30% penalty for high severity
        
        total_deadlock_penalty = deadlock_penalty + severity_penalty
        safety_score = max(0.0, 1.0 - collision_penalty - total_deadlock_penalty)
        return safety_score
