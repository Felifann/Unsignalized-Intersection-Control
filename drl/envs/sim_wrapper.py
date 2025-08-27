import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

# Import unified configuration
from config.unified_config import UnifiedConfig, get_config

# Import CARLA after adding egg to path
try:
    import carla  # type: ignore
except ImportError:
    print("⚠️ CARLA not found - make sure CARLA egg is properly installed")
    carla = None

# Core imports
from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from platooning.platoon_manager import PlatoonManager
from auction.auction_engine import DecentralizedAuctionEngine
from control import TrafficController
from nash.deadlock_nash_solver import DeadlockNashSolver
from nash.deadlock_detector import DeadlockException
from drl.policies.bid_policy import TrainableBidPolicy
from drl.envs.metrics_manager import SimulationMetricsManager

class SimulationEnv:
    """Streamlined simulation environment wrapper"""
    
    def __init__(self, sim_cfg: dict = None, unified_config: UnifiedConfig = None):
        # Use unified config or get global config
        if unified_config is None:
            unified_config = get_config()
        
        self.unified_config = unified_config
        self.sim_cfg = sim_cfg or {}
        
        # Update unified config with sim_cfg if provided
        if 'training_mode' in self.sim_cfg:
            self.unified_config.system.training_mode = self.sim_cfg['training_mode']
        
        # Generate sim config from unified config
        unified_sim_cfg = self.unified_config.to_sim_config()
        # Override with any explicit sim_cfg values
        unified_sim_cfg.update(self.sim_cfg)
        self.sim_cfg = unified_sim_cfg
        
        # CORRECTED: Track actions vs simulation steps separately 
        self.max_actions = self.sim_cfg.get('max_steps', 2000)  # This is actually max actions
        self.current_action = 0  # Track actions taken by DRL agent
        self.current_step = 0    # Track simulation steps (for internal use)
        
        # BALANCED Performance settings for DRL training - from unified config
        training_mode = self.unified_config.system.training_mode
        # Auto-calc steps_per_action from seconds-based logic interval and fixed delta
        self.steps_per_action = self._compute_steps_per_action()
        # Persist back so downstream reads stay consistent
        self.unified_config.system.steps_per_action = self.steps_per_action
        self.observation_cache_steps = 20 if training_mode else 15  # ULTRA-FAST: Longer caching
        self.last_observation = None
        self.last_obs_step = -1
        
        # Pre-allocated observation array - OPTIMIZED from 60 to 50 dimensions (8 vehicles)
        self._obs_array = np.zeros(50, dtype=np.float32)
        
        # Dedicated metrics manager
        self.metrics_manager = SimulationMetricsManager(unified_config=self.unified_config)
        
        # DISABLED: Prevent unnecessary mid-episode resets for DRL training
        # DRL training should handle episode termination, not mid-episode resets
        self.deadlock_reset_enabled = False  # FORCE DISABLE for clean episodes
        self.deadlock_timeout_duration = self.unified_config.deadlock.deadlock_timeout_duration
        self.deadlock_first_detected_time = None
        self.deadlock_consecutive_detections = 0
        self.deadlock_reset_count = 0
        self.max_deadlock_resets = 0  # FORCE DISABLE
        
        # DISABLED: Severe deadlock resets should terminate episode, not reset mid-episode
        self.severe_deadlock_reset_enabled = False  # FORCE DISABLE for DRL
        self.severe_deadlock_punishment = self.unified_config.system.severe_deadlock_punishment
        self.severe_deadlock_reset_count = 0
        
        # Initialize simulation components
        self._init_simulation()
        
        # Trainable policy
        self.bid_policy = TrainableBidPolicy()
        
        # Connect bid_policy ONCE during initialization (not during reset or every step)
        self.traffic_controller.set_bid_policy(self.bid_policy)
        self.auction_engine.set_bid_policy(self.bid_policy)
        
        print(f"🤖 Streamlined Simulation Environment initialized with UNIFIED CONFIG")
        print(f"   🔧 Config - Conflict window: {self.unified_config.conflict.conflict_time_window}s")
        print(f"   🔧 Config - Safe distance: {self.unified_config.conflict.min_safe_distance}m")
        print(f"   🔧 Config - Deadlock threshold: {self.unified_config.deadlock.deadlock_speed_threshold} m/s (FURTHER LOOSENED)")
        print(f"   🔧 Config - Deadlock duration: {self.unified_config.deadlock.deadlock_duration_threshold}s (FURTHER LOOSENED)")
        print(f"   🔧 Config - Deadlock check interval: {self.unified_config.deadlock.deadlock_check_interval}s (FURTHER LOOSENED)")
        print(f"   🚫 Mid-episode deadlock resets: DISABLED (clean DRL episodes)")
        print(f"   🚫 Severe deadlock resets: DISABLED (episodes terminate cleanly)")
        print(f"   ✅ DRL Training Mode: Episodes end on deadlock rather than mid-reset")
        print(f"   🚀 OBSERVATION SPACE OPTIMIZED: 60 → 50 dimensions (8 vehicles × 5 features)")
        print(f"   🎯 NEW OPTIMIZED STRUCTURE: 10 + 40 = 50 dimensions")
 
    def observation_dim(self) -> int:
        """Return optimized observation space dimension"""
        return 50  # 10 + 40 (optimized structure with 8 vehicles × 5 features)

    def _init_simulation(self):
        """Initialize core simulation components"""
        try:
            # Core components - PASS UNIFIED CONFIG TO SCENARIO MANAGER
            self.scenario = ScenarioManager(unified_config=self.unified_config)
            training_mode = self.unified_config.system.training_mode
            self.state_extractor = StateExtractor(self.scenario.carla, training_mode=training_mode)
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Clean and reset
            self._cleanup_existing_vehicles()
            time.sleep(0.5)
            self.scenario.reset_scenario()
            time.sleep(1.0)
            
            # Verify initial setup
            initial_vehicles = self.state_extractor.get_vehicle_states()
            print(f"🚗 Initial setup: {len(initial_vehicles)} vehicles")
            
            # Create other components
            self.auction_engine = DecentralizedAuctionEngine(
                state_extractor=self.state_extractor,
                max_go_agents=None,
                max_participants_per_auction=self.unified_config.auction.max_participants_per_auction
            )
            
            # Set unified auction interval
            self.auction_engine.set_auction_interval_from_config(
                self.unified_config.auction.auction_interval
            )
            
            self.nash_solver = DeadlockNashSolver(
                unified_config=self.unified_config,
                intersection_center=self.unified_config.system.intersection_center,
                max_go_agents=self.unified_config.mwis.max_go_agents
            )
            
            self.traffic_controller = TrafficController(
                self.scenario.carla, 
                self.state_extractor, 
                max_go_agents=None
            )
            
            # Connect components
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            
            print("✅ Core simulation components initialized")
            
        except Exception as e:
            print(f"❌ Simulation initialization failed: {str(e)}")
            raise

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """FIXED: Reset environment with proper seed handling and validation"""
        print(f"🔄 Environment reset starting (seed={seed})...")
        
        # CRITICAL: Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            # TODO: Also set other relevant random seeds (CARLA, etc.)
        
        try:
            # Phase 1: Reset internal state
            self._reset_internal_state()
            
            # Phase 2: Reset simulation environment with retries
            reset_success = self._safe_reset_scenario_with_retries()
            if not reset_success:
                raise RuntimeError("Scenario reset failed after multiple attempts")
            
            # Phase 3: Initialize/reset components with proper cleanup
            self._initialize_components_safely()
            
            # Phase 4: Wait for simulation stabilization with proper validation
            self._wait_for_stabilization()
            
            # Phase 5: Validate reset state before proceeding
            validation_success = self._validate_reset_state()
            if not validation_success:
                raise RuntimeError("Reset state validation failed")
            
            # Phase 6: Get and validate initial observation
            obs = self._get_validated_observation()
            
            print(f"✅ Reset completed successfully with {obs.shape[0]}-dim observation")
            return obs
            
        except Exception as e:
            print(f"❌ FATAL: Reset failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # CRITICAL: Try emergency recovery
            try:
                print("🚨 Attempting emergency recovery...")
                emergency_obs = self._emergency_recovery()
                print("⚠️ Emergency recovery succeeded - training may be unstable")
                return emergency_obs
            except Exception as recovery_error:
                print(f"❌ Emergency recovery also failed: {recovery_error}")
                # This is a fatal error - should not continue training
                raise RuntimeError(f"Complete reset failure: {str(e)}, recovery failed: {recovery_error}")

    def step_with_all_params(self, action_params: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute simulation step with all trainable parameters"""
        step_start = time.time()
        
        try:
            # Increment action counter
            self.current_action += 1
            
            # Update trainable parameters
            self._update_policy_parameters(action_params)
            
            # OPTIMIZED: Don't reconnect bid_policy every step - it's the same object
            # self.traffic_controller.set_bid_policy(self.bid_policy)  # REMOVED: Unnecessary
            # self.auction_engine.set_bid_policy(self.bid_policy)     # REMOVED: Unnecessary
            
            # Run simulation steps
            initial_vehicle_count = len(self.state_extractor.get_vehicle_states())
            severe_deadlock_occurred = False
            
            # EXACT main.py pattern: Only update on final frame (unified_update_interval behavior)
            for i in range(self.steps_per_action):
                # Advance simulation (every frame like main.py)
                self.scenario.carla.world.tick()
                self.current_step += 1
                
                # Update system components ONLY on final frame (exactly like main.py)
                if i == self.steps_per_action - 1:  # Only on final frame
                    vehicle_states = self.state_extractor.get_vehicle_states()
                    
                    if vehicle_states:
                        try:
                            if self.current_action % 3 == 0:  # Only every 3rd action
                                self.platoon_manager.update()
                            
                            auction_winners = self.auction_engine.update(vehicle_states, self.platoon_manager)
                            self.traffic_controller.update_control(
                                self.platoon_manager, self.auction_engine, auction_winners
                            )
                        except Exception as update_error:
                            print(f"⚠️ Update error: {update_error}")
                            continue
                
                # Only check for deadlocks/collisions on final frame (performance optimization)
                if i == self.steps_per_action - 1:  # Only on final frame
                    # SIMPLIFIED: Just check for termination conditions - NO MID-EPISODE RESETS
                    # Check for severe deadlock (severity 1.0) - terminate episode
                    if self._check_severe_deadlock():
                        print(f"🚨 SEVERE DEADLOCK (severity 1.0) detected at step {self.current_step} - TERMINATING EPISODE")
                        severe_deadlock_occurred = True
                        break  # Terminate episode cleanly
                    
                    # Check for regular deadlock - terminate episode
                    deadlock_detected = self._check_deadlock()
                    if deadlock_detected:
                        print(f"🚨 Deadlock detected at step {self.current_step} - TERMINATING EPISODE")
                        break  # Terminate episode cleanly
                    
                    # Check for collisions - terminate episode 
                    collision_detected = self._check_collision()
                    if collision_detected:
                        print(f"💥 Collision detected at step {self.current_step} - TERMINATING EPISODE")
                        break  # Terminate episode cleanly
            
            # Calculate reward using metrics manager with action tracking
            reward = self.metrics_manager.calculate_reward(
                self.traffic_controller, self.state_extractor, 
                self.scenario, self.nash_solver, self.current_step,
                actions_since_reset=self.current_action
            )
            
            # Apply severe deadlock punishment if occurred (WITHIN THIS EPISODE)
            if severe_deadlock_occurred:
                reward += self.severe_deadlock_punishment
                print(f"⚡ Applied EPISODE severe deadlock punishment: {self.severe_deadlock_punishment} (step reward: {reward})")
                print(f"   Note: This punishment applies ONLY to this step, not carried to next episode")
            
            # Get observation
            obs = self._get_observation_cached()
            
            # SIMPLIFIED: Check episode termination based on ACTIONS, not simulation steps
            # Terminate on any deadlock or collision - no mid-episode resets
            done = (self.current_action >= self.max_actions or 
                   self._check_collision() or 
                   self._check_deadlock() or
                   severe_deadlock_occurred)
            
            # Generate info using metrics manager
            info = self.metrics_manager.get_info_dict(
                self.traffic_controller, self.auction_engine, self.nash_solver,
                self.scenario, self.state_extractor, self.bid_policy,
                self.current_action, self.max_actions  # Report actions, not sim steps
            )
            
            # Add validation info
            current_vehicles = len(self.state_extractor.get_vehicle_states())
            info.update({
                'step_validation': {
                    'vehicles_stable': abs(current_vehicles - initial_vehicle_count) <= 3,
                    'reward_realistic': abs(reward) <= 50.0,
                    'observation_valid': len(obs) == 50  # OPTIMIZED: 50 dimensions with 8 vehicles
                },
                'action_info': {
                    'current_action': self.current_action,
                    'max_actions': self.max_actions,
                    'sim_steps_taken': self.current_step,
                    'steps_per_action': self.steps_per_action
                },
                'termination_info': {
                    'deadlock_detected': self._check_deadlock(),
                    'collision_detected': self._check_collision(),
                    'severe_deadlock_detected': severe_deadlock_occurred,
                    'max_actions_reached': self.current_action >= self.max_actions,
                    'termination_reason': 'max_actions' if self.current_action >= self.max_actions else
                                        'severe_deadlock' if severe_deadlock_occurred else
                                        'deadlock' if self._check_deadlock() else
                                        'collision' if self._check_collision() else 'none'
                }
            })
            
            # Record performance
            step_time = time.time() - step_start
            self.metrics_manager.record_performance(step_time)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"❌ Step failed: {str(e)}")
            return (np.zeros(50, dtype=np.float32), -10.0, True,  # FIXED: 50 dimensions for 8 vehicles
                   {'error': str(e), 'using_real_data': False})

    def _update_policy_parameters(self, action_params: Dict):
        """Update ONLY the 8 trainable parameters from the action space"""
        # FIXED: Only update parameters that are actually in the 8-dimensional action space
        
        # 1. Core bidding parameters (4 parameters)
        if 'bid_scale' in action_params:
            self.bid_policy.bid_scale = action_params['bid_scale']
        if 'eta_weight' in action_params:
            self.bid_policy.eta_weight = action_params['eta_weight']
        if 'platoon_bonus' in action_params:
            self.bid_policy.platoon_bonus = action_params['platoon_bonus']
        if 'junction_penalty' in action_params:
            self.bid_policy.junction_penalty = action_params['junction_penalty']
        
        # 2. Control parameter (1 parameter)
        if 'speed_diff_modifier' in action_params:
            self.bid_policy.speed_diff_modifier = action_params['speed_diff_modifier']
        
        # 3. Auction efficiency parameter (1 parameter)
        if 'max_participants_per_auction' in action_params:
            self.auction_engine.update_max_participants_per_auction(
                action_params['max_participants_per_auction']
            )
        
        # 4. Safety parameters (2 parameters)
        if 'ignore_vehicles_go' in action_params:
            self.bid_policy.ignore_vehicles_go = action_params['ignore_vehicles_go']
        if 'ignore_vehicles_platoon_leader' in action_params:
            self.bid_policy.ignore_vehicles_platoon_leader = action_params['ignore_vehicles_platoon_leader']
        
        # FIXED: Set FIXED values for non-trainable parameters (not in action space)
        # These parameters are NOT trainable and should remain constant
        self.bid_policy.speed_weight = 0.3  # Fixed at 0.3 (not trainable)
        self.bid_policy.congestion_sensitivity = 0.4  # Fixed at 0.4 (not trainable)
        self.bid_policy.fairness_factor = 0.1  # Fixed at 0.1 (not trainable)
        self.bid_policy.urgency_threshold = 5.0  # Fixed at 5.0 (not trainable)
        self.bid_policy.proximity_bonus_weight = 1.0  # Fixed at 1.0 (not trainable)
        self.bid_policy.follow_distance_modifier = 0.0  # Fixed at 0.0 (not trainable)
        self.bid_policy.ignore_vehicles_wait = 0.0  # Fixed at 0 (not trainable)
        self.bid_policy.ignore_vehicles_platoon_follower = 90.0  # Fixed at 90% (not trainable)
        
        # FIXED: Set FIXED values for reward function parameters (not trainable)
        # These are NOT in the action space and should remain constant
        if hasattr(self.unified_config, 'drl'):
            self.unified_config.drl.vehicle_exit_reward = 10.0  # Fixed (not trainable)
            self.unified_config.drl.collision_penalty = 100.0  # Fixed (not trainable)
            self.unified_config.drl.deadlock_penalty = 800.0  # Fixed (not trainable)
            self.unified_config.drl.throughput_bonus = 0.01  # Fixed (not trainable)
        
        # FIXED: Set FIXED values for conflict detection parameters (not trainable)
        # These are NOT in the action space and should remain constant
        if hasattr(self.unified_config, 'conflict'):
            self.unified_config.conflict.conflict_time_window = 2.5  # Fixed at 2.5s (not trainable)
            self.unified_config.conflict.min_safe_distance = 3.0  # Fixed at 3.0m (not trainable)
            self.unified_config.conflict.collision_threshold = 2.0  # Fixed at 2.0m (not trainable)
        
        # FIXED: Set FIXED values for Nash parameters (not trainable)
        # These are NOT in the action space and should remain constant
        if hasattr(self.unified_config, 'nash'):
            self.unified_config.nash.path_intersection_threshold = 2.5  # Fixed at 2.5m (not trainable)
            self.unified_config.nash.platoon_conflict_distance = 15.0  # Fixed at 15m (not trainable)
        
        # Debug logging for parameter updates
        print(f"🔧 Updated trainable parameters:")
        print(f"   Bid: scale={action_params.get('bid_scale', 'N/A'):.3f}, eta={action_params.get('eta_weight', 'N/A'):.3f}")
        print(f"   Platoon: bonus={action_params.get('platoon_bonus', 'N/A'):.3f}, penalty={action_params.get('junction_penalty', 'N/A'):.3f}")
        print(f"   Control: speed_mod={action_params.get('speed_diff_modifier', 'N/A'):.3f}")
        print(f"   Auction: max_participants={action_params.get('max_participants_per_auction', 'N/A')}")
        print(f"   Safety: ignore_go={action_params.get('ignore_vehicles_go', 'N/A'):.1f}%, ignore_leader={action_params.get('ignore_vehicles_platoon_leader', 'N/A'):.1f}%")

    def _get_observation_cached(self) -> np.ndarray:
        """Get observation with caching"""
        if (self.last_observation is not None and 
            self.current_step - self.last_obs_step < self.observation_cache_steps):
            return self.last_observation.copy()
        
        obs = self._get_observation()
        self.last_observation = obs.copy()
        self.last_obs_step = self.current_step
        return obs

    def _get_observation(self) -> np.ndarray:
        """Generate observation array with OPTIMIZED 50 dimensions - REDESIGNED for better DRL training"""
        try:
            # Reset array - OPTIMIZED to 50 dimensions (10 + 40 for 8 vehicles × 5 features)
            self._obs_array = np.zeros(50, dtype=np.float32)
            
            # Get current state
            vehicle_states = self.state_extractor.get_vehicle_states()
            control_stats = self.traffic_controller.get_control_stats()
            auction_stats = self.auction_engine.get_auction_stats()
            
            # ===== ESSENTIAL CONTROL METRICS (indices 0-9) - 10 dimensions =====
            # Only the most meaningful control and performance metrics
            total_controlled = control_stats.get('total_controlled', 0)
            go_vehicles = control_stats.get('go_vehicles', 0)
            waiting_vehicles = control_stats.get('waiting_vehicles', 0)
            
            self._obs_array[0] = min(total_controlled, 50) / 50.0  # Controlled vehicles
            self._obs_array[1] = min(go_vehicles, 20) / 20.0  # GO vehicles
            self._obs_array[2] = min(waiting_vehicles, 30) / 30.0  # Waiting vehicles
            
            # Performance metrics - only the most meaningful
            throughput = self.metrics_manager.metrics.get('throughput', 0)
            avg_accel = self.metrics_manager.metrics.get('avg_acceleration', 0)
            
            self._obs_array[3] = np.clip(throughput / 1000.0, 0.0, 1.0)  # Throughput (0-1000 vehicles/h)
            self._obs_array[4] = np.clip((avg_accel + 10.0) / 20.0, 0.0, 1.0)  # Acceleration (-10 to +10 m/s²)
            
            # Safety metrics - direct and meaningful
            collision_count = self.metrics_manager.metrics.get('collision_count', 0)
            deadlock_count = 0
            if hasattr(self.nash_solver, 'deadlock_detector') and hasattr(self.nash_solver.deadlock_detector, 'stats'):
                deadlock_count = self.nash_solver.deadlock_detector.stats.get('deadlocks_detected', 0)
            
            self._obs_array[5] = np.clip(collision_count / 10.0, 0.0, 1.0)  # Collision count (0-10)
            self._obs_array[6] = np.clip(deadlock_count / 5.0, 0.0, 1.0)  # Deadlock count (0-5)
            
            # Current bid policy state - meaningful for decision making
            current_bid_scale = self.bid_policy.get_current_bid_scale()
            self._obs_array[7] = np.clip((current_bid_scale - 0.1) / 4.9, 0.0, 1.0)  # Bid scale
            
            # Average waiting time - meaningful for efficiency
            total_waiting_time = 0
            waiting_vehicles_count = 0
            for v in vehicle_states:
                if v.get('is_junction', False) and self._get_vehicle_speed(v) < 0.5:
                    waiting_vehicles_count += 1
                    if waiting_vehicles_count > 0:
                        total_waiting_time += 1
            
            avg_waiting_time = total_waiting_time / max(waiting_vehicles_count, 1)
            self._obs_array[8] = np.clip(avg_waiting_time / 20.0, 0.0, 1.0)  # Waiting time
            
            # Traffic congestion level - meaningful for decision making
            total_vehicles = len(vehicle_states)
            vehicles_in_junction = sum(1 for v in vehicle_states if v.get('is_junction', False))
            if total_vehicles > 0:
                avg_speed = sum(self._get_vehicle_speed(v) for v in vehicle_states) / total_vehicles
                congestion_level = (1.0 - avg_speed / 20.0) * (vehicles_in_junction / max(total_vehicles, 1))
                self._obs_array[9] = np.clip(congestion_level, 0.0, 1.0)  # Congestion
            else:
                self._obs_array[9] = 0.0
            
            # ===== VEHICLE STATES (indices 10-49, 8 vehicles × 5 features each) - 40 dimensions =====
            try:
                active_controls = control_stats.get('active_controls', [])
                active_controls_set = set(str(control_id) for control_id in active_controls)
                
                for i, vehicle_state in enumerate(vehicle_states[:8]):  # Top 8 vehicles by priority (FIXED: was 5)
                    try:
                        base_idx = 10 + i * 5
                        
                        # Feature 1: Distance to intersection center (normalized)
                        loc = vehicle_state.get('location', {})
                        if isinstance(loc, dict):
                            x, y = loc.get('x', 0.0), loc.get('y', 0.0)
                        elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                            x, y = np.asarray(loc[0], dtype=np.float32), np.asarray(loc[1], dtype=np.float32)
                        else:
                            x, y = 0.0, 0.0
                        
                        distance_to_center = np.sqrt((x + 188.9)**2 + (y + 89.7)**2)
                        self._obs_array[base_idx] = np.clip(distance_to_center / 100.0, 0.0, 1.0)
                        
                        # Feature 2: Speed (normalized)
                        vel = vehicle_state.get('velocity', {})
                        if isinstance(vel, dict):
                            speed = np.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2)
                        elif isinstance(vel, (list, tuple)) and len(vel) >= 3:
                            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                        else:
                            speed = 0.0
                        self._obs_array[base_idx + 1] = np.clip(speed / 20.0, 0.0, 1.0)
                        
                        # Feature 3: ETA to intersection (normalized)
                        eta = vehicle_state.get('eta_to_intersection', 0)
                        self._obs_array[base_idx + 2] = np.clip(eta / 60.0, 0.0, 1.0)
                        
                        # Feature 4: Junction status (binary)
                        self._obs_array[base_idx + 3] = np.asarray(vehicle_state.get('is_junction', False), dtype=np.float32)
                        
                        # Feature 5: Control status (binary)
                        vehicle_id = vehicle_state.get('id', 0)
                        self._obs_array[base_idx + 4] = np.asarray(str(vehicle_id) in active_controls_set, dtype=np.float32)
                        
                    except Exception as vehicle_error:
                        print(f"⚠️ Vehicle {i} observation error: {str(vehicle_error)}")
                        base_idx = 10 + i * 5
                        self._obs_array[base_idx:base_idx + 5] = 0.0
                        
            except Exception as vehicles_error:
                print(f"⚠️ Vehicle states observation error: {str(vehicles_error)}")
                self._obs_array[10:50] = 0.0  # FIXED: Properly handle 50 dimensions for 8 vehicles
            
            # Final validation and normalization
            self._obs_array = np.nan_to_num(self._obs_array, nan=0.0, posinf=1.0, neginf=0.0)
            self._obs_array = np.clip(self._obs_array, 0.0, 1.0)
            
            return self._obs_array.copy()
            
        except Exception as e:
            print(f"❌ Observation generation failed: {str(e)}")
            return np.zeros(50, dtype=np.float32)  # FIXED: Return 50 dimensions to match expected size
    
    def _get_vehicle_speed(self, vehicle_state):
        """Helper method to extract vehicle speed"""
        try:
            vel = vehicle_state.get('velocity', {})
            if isinstance(vel, dict):
                return np.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2)
            elif isinstance(vel, (list, tuple)) and len(vel) >= 3:
                return np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            else:
                return 0.0
        except:
            return 0.0

    def _cleanup_existing_vehicles(self):
        """Clean up existing vehicles - COMPLETELY REWRITTEN with robust error handling"""
        try:
            # Safeguard when CARLA is not available
            if carla is None or not hasattr(self, 'scenario') or not hasattr(self.scenario, 'carla'):
                return
                
            world = self.scenario.carla.world
            if not world:
                return
                
            # Get all vehicle actors
            try:
                vehicles = world.get_actors().filter('vehicle.*')
            except Exception as e:
                print(f"⚠️ Failed to get vehicle actors: {e}")
                return
            
            if len(vehicles) > 0:
                print(f"🧹 Cleaning {len(vehicles)} vehicles")
                
                # COMPLETELY REWRITTEN: Individual vehicle cleanup with robust error handling
                destroyed_count = 0
                error_count = 0
                
                for vehicle in vehicles:
                    try:
                        # Check if vehicle is still valid and alive
                        if vehicle and hasattr(vehicle, 'is_alive') and vehicle.is_alive:
                            # Try to destroy the vehicle
                            try:
                                vehicle.destroy()
                                destroyed_count += 1
                            except Exception as destroy_error:
                                # Vehicle might have been destroyed by another process
                                error_count += 1
                                continue
                        else:
                            # Vehicle is already invalid/dead
                            error_count += 1
                            continue
                            
                    except Exception as vehicle_error:
                        # Skip this vehicle if any error occurs
                        error_count += 1
                        continue
                
                # Report cleanup results
                if destroyed_count > 0:
                    print(f"✅ Successfully destroyed {destroyed_count} vehicles")
                if error_count > 0:
                    print(f"⚠️ Skipped {error_count} invalid/dead vehicles")
                
                # Wait for CARLA to process the destruction
                try:
                    world.tick()
                    time.sleep(0.1)  # Slightly longer wait for better cleanup
                except Exception as tick_error:
                    print(f"⚠️ World tick error during cleanup: {tick_error}")
            
        except Exception as e:
            print(f"⚠️ Vehicle cleanup failed: {str(e)}")
            # Don't raise - continue with simulation

    def _reset_internal_state(self):
        """Phase 1: Reset all internal state variables"""
        self.current_step = 0      # Reset simulation steps
        self.current_action = 0    # Reset action counter
        
        # Reset deadlock tracking
        self.deadlock_first_detected_time = None
        self.deadlock_consecutive_detections = 0
        self.deadlock_reset_count = 0
        self.severe_deadlock_reset_count = 0
        
        # Reset observation cache
        self.last_observation = None
        self.last_obs_step = -1
        
        # Clear pre-allocated observation array
        self._obs_array.fill(0.0)
        
        # OPTIMIZED: Clear performance caches to prevent stale data
        if hasattr(self, '_cached_vehicle_states'):
            self._cached_vehicle_states = None
            self._cached_vehicle_states_time = None
        
        if hasattr(self, '_cached_severity_time'):
            self._cached_severity_time = None
            self._cached_severity_value = None
        
        # OPTIMIZED: Initialize Nash solver parameter cache for new episode
        if not hasattr(self, '_last_nash_params'):
            self._last_nash_params = {}
        else:
            # Clear cache for fresh episode start
            self._last_nash_params.clear()
        
        print("✅ Internal state reset completed")

    def _safe_reset_scenario_with_retries(self) -> bool:
        """Phase 2: Reset scenario with multiple attempts and proper error handling - OPTIMIZED"""
        max_attempts = 2  # Reduced from 3 to 2 for faster training
        for attempt in range(max_attempts):
            try:
                print(f"🎯 Scenario reset attempt {attempt + 1}/{max_attempts}...")
                
                # Clean up existing vehicles first
                self._cleanup_existing_vehicles()
                time.sleep(0.1)  # Reduced from 0.2s to 0.1s
                
                # Reset scenario
                self.scenario.reset_scenario()
                
                # Wait for scenario reset to complete - OPTIMIZED timing
                wait_time = 0.1 if self.sim_cfg.get('training_mode', False) else 0.3  # Reduced from 0.2/0.8
                time.sleep(wait_time)
                
                # Tick the world to ensure everything is synchronized
                self.scenario.carla.world.tick()
                
                # Verify reset success by checking for vehicles
                vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
                if len(vehicles) > 0:
                    print(f"✅ Scenario reset successful: {len(vehicles)} vehicles spawned")
                    return True
                else:
                    print(f"⚠️ Attempt {attempt + 1}: No vehicles after reset")
                    if attempt < max_attempts - 1:
                        time.sleep(0.2)  # Reduced from 0.5s to 0.2s
                        continue
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(0.2)  # Reduced from 0.5s to 0.2s
                    continue
                else:
                    print(f"💥 All {max_attempts} scenario reset attempts failed")
                    return False
        
        return False

    def _initialize_components_safely(self):
        """Phase 3: Initialize/reset components with proper cleanup and validation - OPTIMIZED"""
        print("🔧 Initializing components safely...")
        
        # Start simulation timers
        if hasattr(self.scenario, 'start_time_counters'):
            self.scenario.start_time_counters()
        
        # FIXED: Properly cleanup and recreate platoon manager
        if hasattr(self, 'platoon_manager') and self.platoon_manager is not None:
            # Clean up existing platoon manager if it has cleanup methods
            if hasattr(self.platoon_manager, 'cleanup'):
                self.platoon_manager.cleanup()
        self.platoon_manager = PlatoonManager(self.state_extractor)
        
        # FIXED: Safely handle auction engine and nash solver
        # Only create if they don't exist - but ensure they're properly reset
        if not hasattr(self, 'auction_engine') or self.auction_engine is None:
            self.auction_engine = DecentralizedAuctionEngine(
                state_extractor=self.state_extractor,
                max_go_agents=None,
                max_participants_per_auction=self.unified_config.auction.max_participants_per_auction
            )
            self.auction_engine.set_auction_interval_from_config(
                self.unified_config.auction.auction_interval
            )
        
        if not hasattr(self, 'nash_solver') or self.nash_solver is None:
            self.nash_solver = DeadlockNashSolver(
                unified_config=self.unified_config,
                intersection_center=self.unified_config.system.intersection_center,
                max_go_agents=self.unified_config.mwis.max_go_agents
            )
        
        # Reconnect all components
        self.traffic_controller.set_platoon_manager(self.platoon_manager)
        self.auction_engine.set_nash_controller(self.nash_solver)
        # OPTIMIZED: Don't reconnect bid_policy during reset - it's the same object
        # self.traffic_controller.set_bid_policy(self.bid_policy)  # REMOVED: Unnecessary
        # self.auction_engine.set_bid_policy(self.bid_policy)     # REMOVED: Unnecessary
        
        # OPTIMIZED: Reset episode state for all components - no unnecessary waits
        self.traffic_controller.reset_episode_state()
        
        if hasattr(self.auction_engine, 'reset_episode_state'):
            self.auction_engine.reset_episode_state()
        
        if hasattr(self.nash_solver, 'reset_stats'):
            self.nash_solver.reset_stats()
        
        # Reset metrics AFTER all components are reset
        self.metrics_manager.reset_metrics(
            nash_solver=self.nash_solver,
            traffic_controller=self.traffic_controller
        )
        
        print("✅ Component initialization completed")

    def _wait_for_stabilization(self):
        """Phase 4: Wait for simulation to stabilize with proper validation - OPTIMIZED"""
        print("⏱️ Waiting for simulation stabilization...")
        
        # OPTIMIZED: Reduced timeout for faster training
        stabilization_timeout = 3.0  # Reduced from 10.0s to 3.0s for faster training
        check_interval = 0.1  # Check every 100ms
        start_time = time.time()
        
        while time.time() - start_time < stabilization_timeout:
            # Tick the world to advance simulation
            self.scenario.carla.world.tick()
            
            # Check if vehicles are properly spawned and stable
            vehicles = self.state_extractor.get_vehicle_states()
            if len(vehicles) > 0:
                # OPTIMIZED: Quick validation - just check if vehicles exist and have moved from origin
                valid_vehicles = 0
                for vehicle in vehicles[:3]:  # Check only first 3 vehicles for speed
                    pos = vehicle.get('location', {})
                    if isinstance(pos, dict):
                        x, y = pos.get('x', 0), pos.get('y', 0)
                    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        x, y = pos[0], pos[1]
                    else:
                        continue
                    
                    # Check if vehicle is not at origin (indicating proper spawn)
                    if abs(x) > 1.0 or abs(y) > 1.0:
                        valid_vehicles += 1
                        if valid_vehicles >= 2:  # Only need 2 valid vehicles to proceed
                            print(f"✅ Simulation stabilized with {len(vehicles)} vehicles ({valid_vehicles} valid)")
                            return
            
            time.sleep(check_interval)
        
        print(f"⚠️ Stabilization timeout reached ({stabilization_timeout}s) - proceeding anyway")

    def _validate_reset_state(self) -> bool:
        """Phase 5: Validate that reset was successful - OPTIMIZED for speed"""
        print("🔍 Validating reset state...")
        
        try:
            # Check 1: Vehicles exist and are valid - FAST CHECK
            vehicles = self.state_extractor.get_vehicle_states()
            if len(vehicles) == 0:
                print("❌ Validation failed: No vehicles found")
                return False
            
            # Check 2: Component connections are valid - FAST CHECK
            if not hasattr(self, 'platoon_manager') or self.platoon_manager is None:
                print("❌ Validation failed: Platoon manager not initialized")
                return False
            
            if not hasattr(self, 'auction_engine') or self.auction_engine is None:
                print("❌ Validation failed: Auction engine not initialized")
                return False
            
            if not hasattr(self, 'nash_solver') or self.nash_solver is None:
                print("❌ Validation failed: Nash solver not initialized")
                return False
            
            # OPTIMIZED: Skip expensive observation generation test during validation
            # This will be tested naturally during the first step() call
            print(f"✅ Reset state validation passed: {len(vehicles)} vehicles, all components operational")
            return True
            
        except Exception as e:
            print(f"❌ Validation failed with exception: {str(e)}")
            return False

    def _get_validated_observation(self) -> np.ndarray:
        """Phase 6: Get observation with comprehensive validation"""
        try:
            obs = self._get_observation()
            
            # ENSURE exact dimension match - OPTIMIZED: 50 dimensions with 8 vehicles
            expected_shape = 50
            if obs.shape[0] != expected_shape:
                print(f"⚠️ FIXING step observation: {obs.shape[0]} -> {expected_shape}")
                if obs.shape[0] < expected_shape:
                    padding = np.zeros(expected_shape - obs.shape[0], dtype=np.float32)
                    obs = np.concatenate([obs, padding])
                else:
                    obs = obs[:expected_shape]
            
            # Validate observation content
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Final validation
            if obs.shape[0] != expected_shape:
                print(f"❌ CRITICAL: Could not fix observation shape: {obs.shape[0]} != {expected_shape}")
                return np.zeros(expected_shape, dtype=np.float32)  # FIXED: 50 dimensions for 8 vehicles
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Observation generation failed: {str(e)}")
            # Return safe fallback observation
            return np.zeros(50, dtype=np.float32)  # FIXED: 50 dimensions for 8 vehicles

    def _emergency_recovery(self) -> np.ndarray:
        """Emergency recovery when normal reset fails - OPTIMIZED"""
        print("🚨 Running emergency recovery protocol...")
        
        try:
            # Step 1: Force cleanup everything
            self._force_cleanup_all()
            
            # Step 2: Reinitialize basic simulation
            self._init_simulation()
            
            # Step 3: Quick scenario reset - OPTIMIZED timing
            self.scenario.reset_scenario()
            time.sleep(0.3)  # Reduced from 1.0s to 0.3s for faster recovery
            
            # Step 4: Basic component setup
            self.platoon_manager = PlatoonManager(self.state_extractor)
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            
            # Step 5: Return minimal valid observation
            vehicles = self.state_extractor.get_vehicle_states()
            if len(vehicles) > 0:
                print(f"✅ Emergency recovery successful: {len(vehicles)} vehicles")
                return self._get_validated_observation()
            else:
                print("⚠️ Emergency recovery: No vehicles, returning zero observation")
                return np.zeros(50, dtype=np.float32)  # FIXED: 50 dimensions for 8 vehicles
                
        except Exception as e:
            print(f"❌ Emergency recovery failed: {str(e)}")
            raise

    def _force_cleanup_all(self):
        """Force cleanup of all resources"""
        try:
            # Cleanup vehicles
            self._cleanup_existing_vehicles()
            
            # Reset component references
            self.platoon_manager = None
            self.auction_engine = None
            self.nash_solver = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("🧹 Force cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Force cleanup error: {e}")

    def _safe_reset_scenario(self) -> bool:
        """OPTIMIZED single-attempt scenario reset - FAST TRAINING VERSION"""
        try:
            print("🎯 Quick scenario reset...")
            
            # Reset scenario and generate new traffic
            self.scenario.reset_scenario()
            
            # OPTIMIZED: Much shorter wait time for training
            if self.sim_cfg.get('training_mode', False):
                time.sleep(0.1)  # Reduced from 0.2s to 0.1s for faster training
            else:
                time.sleep(0.4)  # Reduced from 0.8s to 0.4s
            
            # Single world tick - no extra sleep
            self.scenario.carla.world.tick()
            
            # Quick verification without extensive debugging
            vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
            if len(vehicles) > 0:
                print(f"✅ Reset successful: {len(vehicles)} vehicles")
                return True
            else:
                print(f"⚠️ No vehicles after reset - retrying once")
                # One quick retry - OPTIMIZED timing
                time.sleep(0.1)  # Reduced from 0.3s to 0.1s
                self.scenario.carla.world.tick()
                vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
                return len(vehicles) > 0
                
        except Exception as e:
            print(f"❌ Reset failed: {str(e)}")
            return False

    def _check_collision(self) -> bool:
        """Check for collisions"""
        try:
            if (hasattr(self.scenario, 'traffic_generator') and 
                hasattr(self.scenario.traffic_generator, 'collision_count')):
                stats = self.scenario.traffic_generator.get_collision_statistics()
                current = stats.get('total_collisions', 0)
                prev = getattr(self, '_prev_collision_count', 0)
                
                if current > prev:
                    print(f"🚨 Collision detected: {current}")
                    self._prev_collision_count = current
                    return True
                    
                self._prev_collision_count = current
            return False
            
        except Exception:
            return False

    def _check_deadlock(self) -> bool:
        """Check for deadlocks with proper error handling and timeout management - OPTIMIZED"""
        try:
            if (hasattr(self, 'nash_solver') and 
                hasattr(self.nash_solver, 'deadlock_detector')):
                
                # OPTIMIZED: Cache vehicle states to avoid repeated calls
                if not hasattr(self, '_cached_vehicle_states') or self._cached_vehicle_states is None:
                    self._cached_vehicle_states = self.state_extractor.get_vehicle_states()
                    self._cached_vehicle_states_time = self.current_step
                
                # Only refresh vehicle states if simulation has advanced significantly
                # OPTIMIZED: Increased cache time to match expanded deadlock detection interval
                if self.current_step - self._cached_vehicle_states_time > 15:  # Increased from 10 to 15
                    self._cached_vehicle_states = self.state_extractor.get_vehicle_states()
                    self._cached_vehicle_states_time = self.current_step
                
                # OPTIMIZED: Create vehicle_dict only when needed
                vehicle_dict = {str(v['id']): v for v in self._cached_vehicle_states}
                current_time = self.scenario.carla.world.get_snapshot().timestamp.elapsed_seconds
                
                return self.nash_solver.deadlock_detector.detect_deadlock(vehicle_dict, current_time)
            
            return False
            
        except DeadlockException as e:
            # Deadlock detected - handle with timeout logic
            return self._handle_deadlock_detection(e)
        except Exception as e:
            # Other errors - log but don't fail
            print(f"⚠️ Deadlock detection error: {str(e)}")
            return False

    def _handle_deadlock_detection(self, deadlock_exception: DeadlockException) -> bool:
        """SIMPLIFIED: Handle deadlock detection - just track and terminate episode"""
        current_time = self.scenario.carla.world.get_snapshot().timestamp.elapsed_seconds
        
        # Track deadlock duration for metrics only
        if self.deadlock_first_detected_time is None:
            self.deadlock_first_detected_time = current_time
            self.deadlock_consecutive_detections = 1
            print(f"🚨 DEADLOCK DETECTED: {deadlock_exception.deadlock_type} affecting {deadlock_exception.affected_vehicles} vehicles")
            print(f"   🏁 Episode will terminate (no mid-episode resets in DRL training)")
        else:
            self.deadlock_consecutive_detections += 1
            deadlock_duration = current_time - self.deadlock_first_detected_time
            print(f"⏳ Deadlock persisting: {deadlock_duration:.1f}s - episode terminating")
        
        # Always return True to terminate episode (no resets)
        return True

    def _check_severe_deadlock(self) -> bool:
        """SIMPLIFIED: Check for severe deadlock (severity 1.0) - just detection, no reset - OPTIMIZED"""
        try:
            if (hasattr(self, 'nash_solver') and 
                hasattr(self.nash_solver, 'deadlock_detector')):
                
                # OPTIMIZED: Cache severity check to avoid repeated calls
                if not hasattr(self, '_cached_severity_time') or self._cached_severity_time is None:
                    self._cached_severity_time = -1
                    self._cached_severity_value = 0.0
                
                # Only check severity every few steps to avoid performance bottleneck
                # OPTIMIZED: Increased cache time to match expanded deadlock detection interval
                if self.current_step - self._cached_severity_time > 9:  # Increased from 6 to 9
                    self._cached_severity_value = self.nash_solver.deadlock_detector.get_deadlock_severity()
                    self._cached_severity_time = self.current_step
                
                # Check if severity is 1.0 (complete deadlock)
                if self._cached_severity_value >= 0.99:  # Use 0.99 to account for floating point precision
                    print(f"⚡ SEVERE DEADLOCK: severity {self._cached_severity_value:.2f} >= 0.99 - episode terminating")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"⚠️ Severe deadlock detection error: {str(e)}")
            return False

    def update_simulation_config(self, fixed_delta_seconds=None, steps_per_action=None):
        """
        Dynamically update simulation configuration during runtime
        
        Args:
            fixed_delta_seconds: New CARLA tick rate (e.g., 0.05 for 20 FPS, 0.1 for 10 FPS)
            steps_per_action: New number of simulation steps per DRL action
        """
        print("🔧 Updating simulation configuration...")
        
        # Update unified config
        if fixed_delta_seconds is not None:
            old_delta = self.unified_config.system.fixed_delta_seconds
            self.unified_config.system.fixed_delta_seconds = fixed_delta_seconds
            print(f"   Fixed Delta Seconds: {old_delta} → {fixed_delta_seconds}")
            
            # Update CARLA world settings immediately
            if hasattr(self.scenario, 'update_carla_settings'):
                self.scenario.update_carla_settings(fixed_delta_seconds=fixed_delta_seconds)
            else:
                print("   ⚠️ ScenarioManager does not support dynamic CARLA settings updates")
            # If caller didn't explicitly set steps_per_action, recalc to keep seconds-based logic interval
            if steps_per_action is None:
                recalculated = self._compute_steps_per_action()
                print(f"   Recalculated steps_per_action from logic interval: {self.steps_per_action} → {recalculated}")
                self.steps_per_action = recalculated
                self.unified_config.system.steps_per_action = recalculated
        
        if steps_per_action is not None:
            old_steps = self.steps_per_action
            self.steps_per_action = steps_per_action
            self.unified_config.system.steps_per_action = steps_per_action
            print(f"   Steps Per Action: {old_steps} → {steps_per_action}")
    
    def get_simulation_config(self):
        """Get current simulation configuration"""
        config = {
            'fixed_delta_seconds': self.unified_config.system.fixed_delta_seconds,
            'steps_per_action': self.steps_per_action,
            'training_mode': self.unified_config.system.training_mode,
        }
        
        # Add CARLA settings if available
        if hasattr(self.scenario, 'get_carla_settings'):
            carla_settings = self.scenario.get_carla_settings()
            if carla_settings:
                config['carla_settings'] = carla_settings
        
        return config

    def close(self):
        """Clean up environment"""
        try:
            # CRITICAL: Clean up collision sensors to prevent file handle leaks
            if hasattr(self, 'scenario') and hasattr(self.scenario, 'traffic_gen'):
                if hasattr(self.scenario.traffic_gen, 'cleanup_sensors'):
                    self.scenario.traffic_gen.cleanup_sensors()
            
            if hasattr(self.scenario, 'stop_time_counters'):
                self.scenario.stop_time_counters()
            print("🏁 Environment closed")
        except Exception as e:
            print(f"❌ Close error: {str(e)}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.metrics_manager.get_performance_stats()

    def set_performance_mode(self, fast_mode: bool = True):
        """Adjust performance settings for balanced speed vs quality"""
        if fast_mode:
            self.steps_per_action = 2  # Very fast for quick training
            self.observation_cache_steps = 10
        else:
            self.steps_per_action = 5  # Balanced approach
            self.observation_cache_steps = 5
        
        print(f"🔧 Performance mode: {'fast' if fast_mode else 'normal'} (steps_per_action: {self.steps_per_action})")

    def _compute_steps_per_action(self) -> int:
        """Compute steps per DRL action from seconds-based logic interval.
        Uses system.logic_update_interval_seconds and system.fixed_delta_seconds.
        """
        try:
            logic_seconds = getattr(self.unified_config.system, 'logic_update_interval_seconds', 0.5)
            fixed_delta = max(1e-6, np.asarray(self.unified_config.system.fixed_delta_seconds, dtype=np.float32))
            steps = int(round(np.asarray(logic_seconds, dtype=np.float32) / fixed_delta))
            if steps < 1:
                steps = 1
            return steps
        except Exception:
            return max(1, int(self.unified_config.system.steps_per_action))