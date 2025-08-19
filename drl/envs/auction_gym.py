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
        
        # Initialize simulation
        self.sim_cfg = sim_cfg or {}
        self.sim = SimulationEnv(self.sim_cfg)
        
        # Define observation space
        obs_dim = self.sim.observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Enhanced action space - multiple parameters for fine control
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.5, 0.0, 0.0, -20.0, -1.0]),  # [bid_scale, eta_weight, speed_weight, congestion_sens, speed_diff_mod, follow_dist_mod]
            high=np.array([5.0, 3.0, 1.0, 1.0, 20.0, 2.0]),
            shape=(6,), 
            dtype=np.float32
        )
        
        self.current_obs = None
        self.render_mode = None
        
        print("🎮 Enhanced Auction Gym Environment initialized")
        print(f"   Observation space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> np.ndarray:
        """Reset environment"""
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed)
        self.current_obs = obs
        return obs

    def step(self, action: np.ndarray) -> tuple:
        """Enhanced step with multi-parameter control"""
        # Extract parameters from action
        bid_scale = float(action[0])
        eta_weight = float(action[1]) if len(action) > 1 else 1.0
        speed_weight = float(action[2]) if len(action) > 2 else 0.3
        congestion_sensitivity = float(action[3]) if len(action) > 3 else 0.4
        speed_diff_modifier = float(action[4]) if len(action) > 4 else 0.0
        follow_distance_modifier = float(action[5]) if len(action) > 5 else 0.0
        
        # Update simulation with all parameters
        obs, reward, done, info = self.sim.step_with_enhanced_params(
            bid_scale=bid_scale,
            eta_weight=eta_weight,
            speed_weight=speed_weight,
            congestion_sensitivity=congestion_sensitivity,
            speed_diff_modifier=speed_diff_modifier,
            follow_distance_modifier=follow_distance_modifier
        )
        
        self.current_obs = obs
        
        # Enhanced info with action details
        info.update({
            'action_bid_scale': bid_scale,
            'action_eta_weight': eta_weight,
            'action_speed_weight': speed_weight,
            'action_congestion_sensitivity': congestion_sensitivity
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
        """Get human-readable action descriptions"""
        return [
            "Bid Scale (0.1-5.0): Overall bidding aggression",
            "ETA Weight (0.5-3.0): Importance of estimated time to intersection", 
            "Speed Weight (0.0-1.0): Importance of current vehicle speed",
            "Congestion Sensitivity (0.0-1.0): Response to traffic congestion",
            "Speed Diff Modifier (-20 to +20): Adjustment to speed control",
            "Follow Distance Modifier (-1 to +2): Adjustment to following distance"
        ]

    def get_reward_info(self) -> Dict[str, str]:
        """Get information about reward components"""
        return {
            "throughput": "Vehicles successfully exiting intersection (+20 per vehicle)",
            "safety": "Collision avoidance (-200 per collision)",
            "efficiency": "Smooth acceleration patterns (+5 for low jerk)",
            "utilization": "Optimal intersection usage (+8 for good ratios)",
            "coordination": "Platoon coordination bonus (+2 per coordinated platoon)",
            "balance": "Traffic flow balance across lanes (+5 for balanced flow)"
        }
