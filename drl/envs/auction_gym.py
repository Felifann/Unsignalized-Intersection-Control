import gym
import numpy as np
from gym import spaces
from typing import Optional, Dict, Any, List

from drl.envs.sim_wrapper import SimulationEnv
from config.unified_config import UnifiedConfig, get_config

class AuctionGymEnv(gym.Env):
    """Enhanced Gym environment for traffic intersection auction system - OPTIMIZED VERSION"""
    
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
        
        # OPTIMIZED observation space - REDUCED from 169 to 50 dimensions
        obs_dim = 50  # OPTIMIZED: 50 dimensions with 8 vehicles
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0,  # Reasonable bounds
            shape=(obs_dim,), dtype=np.float32
        )
        
        # OPTIMIZED action space - REDUCED from 18 to 8 core parameters
        self.action_space = spaces.Box(
            low=np.array([
                0.1, 0.5, 0.0, 0.0, -30.0, 4.0, 0.0, 0.0  # 8 core parameters
            ], dtype=np.float32),
            high=np.array([
                5.0, 3.0, 2.0, 1.0, 30.0, 8.0, 80.0, 40.0  # 8 core parameters
            ], dtype=np.float32),
            shape=(8,), 
            dtype=np.float32
        )
        
        # DISCRETE action space for max_participants_per_auction
        self.discrete_action_space = spaces.Discrete(5)  # 5 discrete values: 4,5,6,7,8
        
        self.current_obs = None
        self.render_mode = None
        
        print("🎮 OPTIMIZED Auction Gym Environment initialized")
        print(f"   观察空间: {self.observation_space.shape} (OPTIMIZED 50维)")
        print(f"   动作空间: {self.action_space.shape} (8个核心参数 - 精简优化版本)")
        print(f"   🔧 Config - Conflict window: {unified_config.conflict.conflict_time_window}s")
        print(f"   🔧 Config - Deadlock threshold: {unified_config.deadlock.deadlock_speed_threshold} m/s")
        print(f"   🎯 OPTIMIZATION - Reduced from 18 to 8 trainable parameters")
        print(f"   🚀 OBSERVATION SPACE OPTIMIZED: 169 → 50 dimensions (8 vehicles × 5 features)")
        print(f"   🚀 ACTION SPACE OPTIMIZED: 18 → 8 parameters (kept only core parameters)")
        print(f"   🎯 CORE TRAINABLE PARAMETERS:")
        print(f"      • Bidding: bid_scale (sigmoid), eta_weight, platoon_bonus, junction_penalty")
        print(f"      • Control: speed_diff_modifier, max_participants_per_auction (discrete)")
        print(f"      • Safety: ignore_vehicles_go, ignore_vehicles_platoon_leader")
        print(f"   🔧 PARAMETER PROCESSING:")
        print(f"      • bid_scale: Sigmoid mapping (0.1-5.0) for smooth training")
        print(f"      • max_participants: Discrete mapping (4,5,6,7,8) with smooth training boundaries")

    def _quantize_param(self, value: float, min_val: float, max_val: float, step_size: float) -> float:
        """Quantize parameter to discrete steps for efficient training"""
        # Clamp to bounds
        value = np.clip(value, min_val, max_val)
        # Quantize to nearest step
        steps = round((value - min_val) / step_size)
        quantized = min_val + steps * step_size
        # Ensure still within bounds after quantization
        return np.clip(quantized, min_val, max_val).astype(np.float32)

    def _sigmoid_map_param(self, raw_value: float, min_val: float, max_val: float) -> float:
        """Map raw network output to parameter range using sigmoid for smooth training"""
        # Apply sigmoid to raw value (unbounded input)
        sigmoid_value = 1.0 / (1.0 + np.exp(-raw_value))
        # Map to [min_val, max_val] range
        mapped_value = min_val + sigmoid_value * (max_val - min_val)
        return np.clip(mapped_value, min_val, max_val).astype(np.float32)

    def _discrete_participants(self, raw_value: float) -> int:
        """Convert continuous action to discrete participants (4,5,6,7,8) with smooth training"""
        # Map raw value to [0, 1] using sigmoid for smooth training
        sigmoid_value = 1.0 / (1.0 + np.exp(-raw_value))
        
        # Map to discrete values with smooth boundaries for training
        if sigmoid_value < 0.2:
            return 4
        elif sigmoid_value < 0.4:
            return 5
        elif sigmoid_value < 0.6:
            return 6
        elif sigmoid_value < 0.8:
            return 7
        else:
            return 8

    def _continuous_participants(self, raw_value: float) -> float:
        """Legacy method - now calls discrete version"""
        return float(self._discrete_participants(raw_value))

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """FIXED reset with proper validation and gym compatibility"""
        print(f"🔄 AuctionGymEnv reset called (seed={seed}, options={options})")
        
        try:
            # CRITICAL: Pass seed to simulation wrapper for proper initialization
            print("🔄 Calling simulation reset...")
            obs = self.sim.reset(seed=seed)
            print(f"✅ Simulation reset successful, obs type: {type(obs)}, shape: {getattr(obs, 'shape', 'no shape')}")
            
        except Exception as sim_reset_error:
            print(f"❌ CRITICAL: Simulation reset failed: {sim_reset_error}")
            import traceback
            traceback.print_exc()
            # Return emergency fallback - proper gym format
            obs = np.zeros(50, dtype=np.float32)  # OPTIMIZED: 50 dimensions with 8 vehicles
            info = {'reset_error': str(sim_reset_error), 'emergency_fallback': True}
            print(f"🔄 Returning emergency fallback observation: {obs.shape}")
            return obs, info
        
        # ENHANCED observation validation with error recovery
        print("🔄 Validating observation...")
        obs = self._validate_and_fix_observation(obs)
        
        # Store current observation
        self.current_obs = obs
        
        # Generate info dict
        info = {
            'reset_successful': True,
            'observation_shape': obs.shape[0],
            'observation_validated': True,
            'seed_used': seed
        }
        
        print(f"✅ AuctionGymEnv reset completed successfully (obs_shape={obs.shape[0]})")
        
        # Return in proper format for gym compatibility
        result = self._format_reset_return(obs, info)
        print(f"🔄 Reset returning: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'not tuple'}")
        return result
    
    def _validate_and_fix_observation(self, obs: np.ndarray) -> np.ndarray:
        """Validate and fix observation with comprehensive error handling"""
        if obs is None:
            print("❌ CRITICAL: Received None observation")
            return np.zeros(50, dtype=np.float32)  # OPTIMIZED: 50 dimensions with 8 vehicles
        
        if not isinstance(obs, np.ndarray):
            print(f"❌ CRITICAL: Observation is not numpy array: {type(obs)}")
            return np.zeros(50, dtype=np.float32)  # OPTIMIZED: 50 dimensions with 8 vehicles
        
        # Check for expected shape
        expected_shape = 50  # OPTIMIZED: 50 dimensions with 8 vehicles
        if obs.shape[0] != expected_shape:
            print(f"⚠️ SHAPE MISMATCH: Got {obs.shape[0]}, expected {expected_shape}")
            
            if obs.shape[0] < expected_shape:
                # Pad with zeros
                padding_size = expected_shape - obs.shape[0]
                padding = np.zeros(padding_size, dtype=np.float32)
                obs = np.concatenate([obs, padding])
                print(f"🔧 FIXED: Padded with {padding_size} zeros")
            else:
                # Truncate
                obs = obs[:expected_shape]
                print(f"🔧 FIXED: Truncated to {expected_shape} dimensions")
        
        # Validate and clean data
        obs = np.asarray(obs, dtype=np.float32)
        
        # Check for invalid values
        nan_count = np.isnan(obs).sum()
        inf_count = np.inf(obs).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️ CLEANING: {nan_count} NaN, {inf_count} Inf values found")
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Final validation
        if obs.shape[0] != expected_shape:
            print(f"❌ CRITICAL: Could not fix observation shape: {obs.shape[0]} != {expected_shape}")
            return np.zeros(expected_shape, dtype=np.float32)  # OPTIMIZED: 60 dimensions
        
        # Check for reasonable value ranges (optional warning)
        extreme_values = np.abs(obs) > 1000
        if extreme_values.any():
            extreme_count = extreme_values.sum()
            print(f"⚠️ WARNING: {extreme_count} extreme values (>1000) in observation")
        
        return obs.astype(np.float32)
    
    def _format_reset_return(self, obs: np.ndarray, info: Dict):
        """Format reset return value for gym compatibility - SIMPLIFIED VERSION"""
        # SIMPLIFIED: Always return gymnasium format (obs, info) for compatibility
        # This avoids the complex inspection logic that was causing errors
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        """FIXED: Consistent parameter handling - ONLY 8 trainable parameters"""
        # VALIDATE action bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # FIXED: ONLY 8 trainable parameters - no extra parameters
        action_params = {
            # Core bidding parameters (4 parameters)
            'bid_scale': self._sigmoid_map_param(action[0], 0.1, 5.0),  # Smooth sigmoid mapping
            'eta_weight': self._quantize_param(action[1], 0.5, 3.0, 0.25),  # Steps of 0.25
            'platoon_bonus': self._quantize_param(action[2], 0.0, 2.0, 0.2),  # Steps of 0.2
            'junction_penalty': self._quantize_param(action[3], 0.0, 1.0, 0.1),  # Steps of 0.1
            
            # Control parameter (1 parameter)
            'speed_diff_modifier': self._quantize_param(action[4], -30.0, 30.0, 5.0),  # Steps of 5
            
            # Auction efficiency parameter (1 parameter) - DISCRETE mapping
            'max_participants_per_auction': self._discrete_participants(action[5]),  # Discrete: 4,5,6,7,8
            
            # Safety parameters (2 parameters)
            'ignore_vehicles_go': self._quantize_param(action[6], 0.0, 80.0, 10.0),  # GO state: 0-80%, 10% steps
            'ignore_vehicles_platoon_leader': self._quantize_param(action[7], 0.0, 40.0, 10.0),  # 0-40%, 10% steps
        }
        
        # FIXED: NO extra parameters - only the 8 trainable ones
        # The sim_wrapper will handle setting fixed values for non-trainable parameters
        
        # Update simulation
        obs, reward, done, info = self.sim.step_with_all_params(action_params)
        
        # ENSURE exact dimension match - OPTIMIZED: 50 dimensions with 8 vehicles
        expected_shape = 50
        if obs.shape[0] != expected_shape:
            print(f"⚠️ FIXING step observation: {obs.shape[0]} -> {expected_shape}")
            if obs.shape[0] < expected_shape:
                padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            else:
                obs = obs[:expected_shape]
        
        # Validate all outputs
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)
        reward = np.clip(np.asarray(reward, dtype=np.float32), -1000.0, 1000.0)
        done = bool(done)
        
        self.current_obs = obs
        
        # Enhanced info with ONLY trainable parameters
        info.update({
            'action_params': action_params,
            'observation_validated': True,
            'observation_shape': obs.shape[0],
            'trainable_params_count': 8  # Confirm we only have 8 trainable parameters
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

    def update_simulation_config(self, fixed_delta_seconds=None, steps_per_action=None):
        """
        Update simulation configuration dynamically
        
        Args:
            fixed_delta_seconds: CARLA tick rate (lower = faster, higher = slower)
                                e.g., 0.05 = 20 FPS, 0.1 = 10 FPS, 0.2 = 5 FPS
            steps_per_action: Number of simulation steps per DRL action
                             (higher = smoother simulation, lower = faster training)
        """
        if hasattr(self.sim, 'update_simulation_config'):
            self.sim.update_simulation_config(
                fixed_delta_seconds=fixed_delta_seconds,
                steps_per_action=steps_per_action
            )
        else:
            print("⚠️ Simulation wrapper does not support dynamic configuration updates")
    
    def get_simulation_config(self):
        """Get current simulation configuration"""
        if hasattr(self.sim, 'get_simulation_config'):
            return self.sim.get_simulation_config()
        else:
            return None

    def close(self) -> None:
        """Close environment"""
        if hasattr(self, 'sim'):
            self.sim.close()
        print("🏁 Optimized Auction Gym Environment closed")

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action descriptions for 8 OPTIMIZED core parameters"""
        return [
            # Core bidding parameters (4 parameters)
            "Bid Scale (0.1-5.0, sigmoid mapping): 总体出价缩放因子 ⚡",
            "ETA Weight (0.5-3.0, steps=0.25): ETA到达时间权重 ⚡", 
            "Platoon Bonus (0.0-2.0, steps=0.2): 车队奖励系数 ⚡",
            "Junction Penalty (0.0-1.0, steps=0.1): 路口位置惩罚 ⚡",
            
            # Control parameter (1 parameter)
            "Speed Diff Modifier (-30 to +30, steps=5): 速度控制修正 ⚡",
            
            # Auction efficiency parameter (1 parameter) - DISCRETE
            "Max Participants Per Auction (4-8, discrete): 每轮拍卖最大参与者数量 🔢",
            
            # Safety parameters (2 parameters)
            "Ignore Vehicles Go (0-80%, steps=10%): GO状态ignore_vehicles% ⚡",
            "Ignore Vehicles Platoon Leader (0-40%, steps=10%): 车队领队ignore_vehicles% ⚡"
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
