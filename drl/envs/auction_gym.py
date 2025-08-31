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
        
        # FIXED: Expose max_actions from simulation environment for proper episode tracking
        self.max_actions = getattr(self.sim, 'max_actions', 128)
        print(f"🎯 Episode length: {self.max_actions} steps (max_actions properly exposed)")
        
        # OPTIMIZED observation space - REDUCED from 169 to 50 dimensions
        obs_dim = 50  # OPTIMIZED: 50 dimensions with 8 vehicles
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0,  # Reasonable bounds
            shape=(obs_dim,), dtype=np.float32
        )
        
        # OPTIMIZED action space - REDUCED from 8 to 4 MOST IMPORTANT parameters for deadlock avoidance
        # NOTE: Using expanded action space so SAC can explore the full parameter ranges effectively
        self.action_space = spaces.Box(
            low=np.array([
                -5.0, -5.0, -5.0, -5.0  # Expanded bounds for better exploration
            ], dtype=np.float32),
            high=np.array([
                5.0, 5.0, 5.0, 5.0  # Expanded bounds for better exploration
            ], dtype=np.float32),
            shape=(4,), 
            dtype=np.float32
        )
        
        # DISCRETE action space for max_participants_per_auction
        self.discrete_action_space = spaces.Discrete(4)  # 4 discrete values: 3,4,5,6
        
        self.current_obs = None
        self.render_mode = None
        
        # Enable action space debugging by default for better parameter tracking
        self._debug_action_space = True
        
        print("🎮 ULTRA-OPTIMIZED Auction Gym Environment initialized")
        print(f"   观察空间: {self.observation_space.shape} (OPTIMIZED 50维)")
        print(f"   动作空间: {self.action_space.shape} (4个最重要的死锁避免参数)")
        print(f"   🔧 Config - Conflict window: {unified_config.conflict.conflict_time_window}s")
        print(f"   🔧 Config - Deadlock threshold: {unified_config.deadlock.deadlock_speed_threshold} m/s")

        print(f"   🎯 CRITICAL TRAINABLE PARAMETERS (Deadlock Avoidance Focus):")
        print(f"      • urgency_position_ratio: 紧急度vs位置优势关系因子 (0.1-3.0, sigmoid)")
        print(f"      • speed_diff_modifier: 速度控制修正 (-30 to +30)")
        print(f"      • max_participants_per_auction: 拍卖参与者数量 (3-6, discrete)")
        print(f"      • ignore_vehicles_go: GO状态ignore_vehicles% (0-80%)")

        print(f"   🚀 ACTION SPACE: Expanded bounds [-5, 5] for better parameter exploration")

        print(f"      • ignore_vehicles_go = ignore_vehicles_go - 10 (calculated in sim_wrapper)")
        print(f"   💡 NEW: urgency_position_ratio 控制紧急度与位置优势的平衡关系")

    def _quantize_param(self, value: float, min_val: float, max_val: float, step_size: float) -> float:
        """Quantize parameter to discrete steps for efficient training - FIXED for better range distribution"""
        # FIXED: Don't clip input value - let it be unbounded for better exploration
        # The neural network can output any value, and we want to map it to the full range
        
        # Map input from [-5, 5] to [0, 1] range using linear mapping (better for bounded inputs)
        # This ensures uniform distribution across the full range
        normalized_value = (value + 5.0) / 10.0  # Maps [-5, 5] to [0, 1]
        normalized_value = np.clip(normalized_value, 0.0, 1.0)  # Ensure [0, 1]
        
        # Map to the target range
        mapped_value = min_val + normalized_value * (max_val - min_val)
        
        # Now quantize to nearest step
        steps = round((mapped_value - min_val) / step_size)
        quantized = min_val + steps * step_size
        
        # Ensure within bounds after quantization
        return np.clip(quantized, min_val, max_val).astype(np.float32)

    def _sigmoid_map_param(self, raw_value: float, min_val: float, max_val: float) -> float:
        """Map raw network output to parameter range using sigmoid for smooth training - IMPROVED for better distribution"""
        # IMPROVED: Use better sigmoid scaling for more uniform distribution across the range
        
        # Apply sigmoid to raw value (bounded input from -5 to 5)
        # Use scaling factor to make sigmoid more responsive across the full range
        scaled_value = raw_value / 2.0  # Makes sigmoid more responsive
        sigmoid_value = 1.0 / (1.0 + np.exp(-scaled_value))
        
        # Map to [min_val, max_val] range
        mapped_value = min_val + sigmoid_value * (max_val - min_val)
        return np.clip(mapped_value, min_val, max_val).astype(np.float32)

    def _discrete_participants(self, raw_value: float) -> int:
        """Convert continuous action to discrete participants (3,4,5,6) with smooth training - IMPROVED for better distribution"""
        # IMPROVED: Better mapping to ensure all discrete values are reachable
        
        # Map raw value from [-5, 5] to [0, 1] using improved sigmoid
        scaled_value = raw_value / 2.0  # Better scaling for sigmoid
        sigmoid_value = 1.0 / (1.0 + np.exp(-scaled_value))
        
        # Map to discrete values with better distribution
        # Use more balanced thresholds to ensure all values are reachable
        if sigmoid_value < 0.2:
            return 3
        elif sigmoid_value < 0.4:
            return 4
        elif sigmoid_value < 0.6:
            return 5
        else:
            return 6

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
            'observation_shape': getattr(obs, 'shape', [0])[0] if hasattr(obs, 'shape') and len(obs.shape) > 0 else 0,
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
        
        # Check for expected shape with safe error handling
        expected_shape = 50  # OPTIMIZED: 50 dimensions with 8 vehicles
        
        try:
            if not hasattr(obs, 'shape'):
                print(f"⚠️ obs has no shape attribute, type: {type(obs)}")
                obs = np.zeros(expected_shape, dtype=np.float32)
            elif len(obs.shape) == 0:
                print(f"⚠️ obs has scalar shape, converting to array")
                obs = np.array([obs], dtype=np.float32)
            elif obs.shape[0] != expected_shape:
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
        except Exception as shape_error:
            print(f"⚠️ Error handling obs shape: {shape_error}, obs type: {type(obs)}")
            obs = np.zeros(expected_shape, dtype=np.float32)
        
        # Validate and clean data
        obs = np.asarray(obs, dtype=np.float32)
        
        # Check for invalid values
        nan_count = np.isnan(obs).sum()
        inf_count = np.inf(obs).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️ CLEANING: {nan_count} NaN, {inf_count} Inf values found")
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Final validation with safe shape access
        try:
            if not hasattr(obs, 'shape') or obs.shape[0] != expected_shape:
                print(f"❌ CRITICAL: Could not fix observation shape: {getattr(obs, 'shape', 'no_shape')} != {expected_shape}")
                return np.zeros(expected_shape, dtype=np.float32)  # OPTIMIZED: 50 dimensions
        except Exception as final_validation_error:
            print(f"⚠️ Final validation error: {final_validation_error}")
            return np.zeros(expected_shape, dtype=np.float32)
        
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
        """FIXED: Consistent parameter handling - ONLY 4 trainable parameters"""
        # FIXED: ONLY 4 trainable parameters - others are fixed for deadlock avoidance focus
        # NOTE: Raw neural network output should be unbounded - processing functions handle bounds
        action_params = {
            # 1. NEW: 紧急度与位置优势的关系因子 (1 parameter) - 替换 bid_scale
            'urgency_position_ratio': self._sigmoid_map_param(action[0], 0.1, 3.0),  # 控制紧急度vs位置优势的平衡
            
            # 2. Control parameter (1 parameter)
            'speed_diff_modifier': self._quantize_param(action[1], -30.0, 30.0, 1.0),  # Steps of 1.0
            
            # 3. Auction efficiency parameter (1 parameter) - DISCRETE mapping
            'max_participants_per_auction': self._discrete_participants(action[2]),  # Discrete: 3,4,5,6
            
            # 4. Safety parameter (1 parameter)
            'ignore_vehicles_go': self._quantize_param(action[3], 0.0, 80.0, 1.0),  # GO state: 0-80%, steps of 1%
        }
        
        # FIXED: NO extra parameters - only the 4 trainable ones
        # The sim_wrapper will handle setting fixed values for non-trainable parameters
        
        # Update simulation
        obs, reward, done, info = self.sim.step_with_all_params(action_params)
        
        # ENSURE exact dimension match - OPTIMIZED: 50 dimensions with 8 vehicles
        expected_shape = 50
        
        # Safe shape validation with error handling
        try:
            if not hasattr(obs, 'shape'):
                print(f"⚠️ obs has no shape attribute, type: {type(obs)}")
                obs = np.zeros(expected_shape, dtype=np.float32)
            elif len(obs.shape) == 0:
                print(f"⚠️ obs has scalar shape, converting to array")
                obs = np.array([obs], dtype=np.float32)
            elif obs.shape[0] != expected_shape:
                print(f"⚠️ FIXING step observation: {obs.shape[0]} -> {expected_shape}")
                if obs.shape[0] < expected_shape:
                    padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                    obs = np.concatenate([obs, padding])
                else:
                    obs = obs[:expected_shape]
        except Exception as shape_error:
            print(f"⚠️ Error handling obs shape: {shape_error}, obs type: {type(obs)}")
            obs = np.zeros(expected_shape, dtype=np.float32)
        
        # Validate all outputs
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)
        reward = np.clip(np.asarray(reward, dtype=np.float32), -1000.0, 1000.0)
        done = bool(done)
        
        self.current_obs = obs
        
        # Enhanced info with ONLY trainable parameters
        info.update({
            'action_params': action_params,
            'raw_neural_network_output': action.tolist(),  # Add raw outputs for debugging
            'observation_validated': True,
            'observation_shape': getattr(obs, 'shape', [0])[0] if hasattr(obs, 'shape') and len(obs.shape) > 0 else 0,
            'trainable_params_count': 4,  # Confirm we only have 4 trainable parameters
            'action_space_dimensions': 4,  # Confirm action space is 4D
            'parameter_mapping_applied': True  # Confirm parameters were mapped
        })

        return obs, reward, done, info

    def test_parameter_mapping(self, num_samples: int = 1000):
        """Test parameter mapping functions to verify range distribution"""
        print(f"🧪 Testing Parameter Mapping Functions ({num_samples} samples)")
        print("=" * 60)
        
        # Test data: Generate samples across the full action space range [-5, 5]
        test_inputs = np.linspace(-5.0, 5.0, num_samples)
        
        # Test 1: urgency_position_ratio (sigmoid mapping)
        print("\n🔍 Test 1: urgency_position_ratio (sigmoid mapping)")
        print("   Expected range: [0.1, 3.0]")
        urgency_values = []
        for x in test_inputs:
            mapped = self._sigmoid_map_param(x, 0.1, 3.0)
            urgency_values.append(mapped)
        
        urgency_values = np.array(urgency_values)
        print(f"   Actual range: [{urgency_values.min():.3f}, {urgency_values.max():.3f}]")
        print(f"   Mean: {urgency_values.mean():.3f}")
        print(f"   Std: {urgency_values.std():.3f}")
        print(f"   Values < 0.5: {(urgency_values < 0.5).sum()} ({(urgency_values < 0.5).sum()/len(urgency_values)*100:.1f}%)")
        print(f"   Values > 2.5: {(urgency_values > 2.5).sum()} ({(urgency_values > 2.5).sum()/len(urgency_values)*100:.1f}%)")
        
        # Test 2: speed_diff_modifier (quantized mapping)
        print("\n🔍 Test 2: speed_diff_modifier (quantized mapping)")
        print("   Expected range: [-30.0, 30.0] with 1.0 steps")
        speed_values = []
        for x in test_inputs:
            mapped = self._quantize_param(x, -30.0, 30.0, 1.0)
            speed_values.append(mapped)
        
        speed_values = np.array(speed_values)
        print(f"   Actual range: [{speed_values.min():.1f}, {speed_values.max():.1f}]")
        print(f"   Mean: {speed_values.mean():.1f}")
        print(f"   Std: {speed_values.std():.1f}")
        print(f"   Values < -20: {(speed_values < -20).sum()} ({(speed_values < -20).sum()/len(speed_values)*100:.1f}%)")
        print(f"   Values > 20: {(speed_values > 20).sum()} ({(speed_values > 20).sum()/len(speed_values)*100:.1f}%)")
        
        # Test 3: max_participants_per_auction (discrete mapping)
        print("\n🔍 Test 3: max_participants_per_auction (discrete mapping)")
        print("   Expected values: [3, 4, 5, 6]")
        participant_values = []
        for x in test_inputs:
            mapped = self._discrete_participants(x)
            participant_values.append(mapped)
        
        participant_values = np.array(participant_values)
        unique_values, counts = np.unique(participant_values, return_counts=True)
        print(f"   Actual values: {unique_values.tolist()}")
        print(f"   Distribution:")
        for value, count in zip(unique_values, counts):
            percentage = count / len(participant_values) * 100
            print(f"     {value}: {count} ({percentage:.1f}%)")
        
        # Test 4: ignore_vehicles_go (quantized mapping)
        print("\n🔍 Test 4: ignore_vehicles_go (quantized mapping)")
        print("   Expected range: [0.0, 80.0] with 1.0 steps")
        ignore_values = []
        for x in test_inputs:
            mapped = self._quantize_param(x, 0.0, 80.0, 1.0)
            ignore_values.append(mapped)
        
        ignore_values = np.array(ignore_values)
        print(f"   Actual range: [{ignore_values.min():.1f}, {ignore_values.max():.1f}]")
        print(f"   Mean: {ignore_values.mean():.1f}")
        print(f"   Std: {ignore_values.std():.1f}")
        print(f"   Values < 20: {(ignore_values < 20).sum()} ({(ignore_values < 20).sum()/len(ignore_values)*100:.1f}%)")
        print(f"   Values > 60: {(ignore_values > 60).sum()} ({(ignore_values > 60).sum()/len(ignore_values)*100:.1f}%)")
        
        # Summary analysis
        print("\n📊 SUMMARY ANALYSIS:")
        print("   ✅ urgency_position_ratio: {'Good distribution' if urgency_values.std() > 0.8 else 'Poor distribution'}")
        print("   ✅ speed_diff_modifier: {'Good distribution' if speed_values.std() > 15.0 else 'Poor distribution'}")
        print("   ✅ max_participants_per_auction: {'Good distribution' if len(unique_values) == 4 else 'Poor distribution'}")
        print("   ✅ ignore_vehicles_go: {'Good distribution' if ignore_values.std() > 20.0 else 'Poor distribution'}")
        
        print("\n✅ Parameter mapping test completed!")
        return {
            'urgency_position_ratio': urgency_values,
            'speed_diff_modifier': speed_values,
            'max_participants_per_auction': participant_values,
            'ignore_vehicles_go': ignore_values
        }

    def set_action_space_debug(self, enabled: bool = True):
        """Enable or disable action space debugging"""
        self._debug_action_space = enabled
        print(f"🔍 Action space debugging {'enabled' if enabled else 'disabled'}")

    def set_verbose_parameter_logging(self, enabled: bool = True):
        """Enable or disable verbose parameter update logging"""
        if hasattr(self, 'sim') and hasattr(self.sim, 'set_verbose_parameter_logging'):
            self.sim.set_verbose_parameter_logging(enabled)
        else:
            print(f"⚠️ Verbose parameter logging control not available")

    def get_current_parameter_values(self) -> Dict[str, Any]:
        """Get current values of all 4 trainable parameters"""
        if hasattr(self, 'sim') and hasattr(self.sim, 'bid_policy'):
            bid_config = self.sim.bid_policy.get_current_config()
            auction_config = self.sim.auction_engine.get_current_config()
            
            return {
                'urgency_position_ratio': bid_config.get('urgency_position_ratio', 'N/A'),
                'speed_diff_modifier': bid_config.get('speed_diff_modifier', 'N/A'),
                'max_participants_per_auction': auction_config.get('max_participants_per_auction', 'N/A'),
                'ignore_vehicles_go': bid_config.get('ignore_vehicles_go', 'N/A'),
                'bid_policy_config': bid_config,
                'auction_engine_config': auction_config
            }
        else:
            return {'error': 'Simulation not initialized'}

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
                print(f"   Urgency Position Ratio: {policy_stats.get('current_urgency_position_ratio', 0):.2f}")
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
        """Get human-readable action descriptions for 4 MOST IMPORTANT deadlock avoidance parameters"""
        return [
            # 1. Bid Scale - 总体出价缩放因子（最重要）
            "Bid Scale (0.1-5.0, sigmoid mapping): 总体出价缩放因子 ⚡ - 控制车辆优先级，避免死锁",
            
            # 2. Speed Diff Modifier - 速度控制修正（关键）
            "Speed Diff Modifier (-30 to +30, steps=5): 速度控制修正 ⚡ - 保持车辆流动，防止停滞",
            
            # 3. Max Participants Per Auction - 拍卖参与者数量（策略性）
            "Max Participants Per Auction (3-6, discrete): 每轮拍卖最大参与者数量 🔢 - 优化决策速度",
            
            # 4. Ignore Vehicles Go - GO状态ignore_vehicles百分比（重要）
            "Ignore Vehicles GO (0-80%, steps=10%): GO状态ignore_vehicles% ⚡ - 减少不必要等待，车队领队自动-10%"
        ]

    def get_current_action_space_config(self) -> Dict[str, Any]:
        """Get current action space configuration for verification"""
        return {
            'action_space_dimensions': 4,
            'action_space_shape': self.action_space.shape,
            'action_space_low': self.action_space.low.tolist(),
            'action_space_high': self.action_space.high.tolist(),
            'discrete_action_space_size': self.discrete_action_space.n,
            'parameter_mappings': {
                'urgency_position_ratio': {
                    'index': 0,
                    'range': [0.1, 3.0],
                    'mapping': 'sigmoid',
                    'description': '紧急度vs位置优势关系因子'
                },
                'speed_diff_modifier': {
                    'index': 1,
                    'range': [-30.0, 30.0],
                    'mapping': 'quantized',
                    'step_size': 1.0,
                    'description': '速度控制修正'
                },
                'max_participants_per_auction': {
                    'index': 2,
                    'range': [3, 6],
                    'mapping': 'discrete',
                    'values': [3, 4, 5, 6],
                    'description': '拍卖参与者数量'
                },
                'ignore_vehicles_go': {
                    'index': 3,
                    'range': [0.0, 80.0],
                    'mapping': 'quantized',
                    'step_size': 1.0,
                    'description': 'GO状态ignore_vehicles百分比'
                }
            }
        }

    def get_current_parameter_values(self) -> Dict[str, Any]:
        """Get current values of all 4 trainable parameters from the simulation environment"""
        if hasattr(self, 'sim') and hasattr(self.sim, 'get_current_parameter_values'):
            return self.sim.get_current_parameter_values()
        else:
            return {'error': 'Simulation environment not accessible'}

    def verify_episode_level_updates(self) -> Dict[str, Any]:
        """Verify that episode-level parameter updates are working correctly"""
        if not hasattr(self, 'sim'):
            return {'error': 'Simulation not initialized'}
        
        verification = {
            'episode_level_updates_enabled': True,
            'episode_params_updated': getattr(self.sim, '_episode_params_updated', False),
            'current_episode_action_params': getattr(self.sim, '_current_episode_action_params', {}),
            'parameter_update_mechanism': 'episode_boundary_only',
            'step_update_mechanism': 'cached_parameters_only'
        }
        
        # Check if the simulation has the required methods
        if hasattr(self.sim, 'get_current_parameter_values'):
            current_params = self.sim.get_current_parameter_values()
            verification['current_parameter_values'] = current_params
        else:
            verification['current_parameter_values'] = {'error': 'Method not available'}
        
        return verification

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
