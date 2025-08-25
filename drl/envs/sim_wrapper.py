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
        self.steps_per_action = self.unified_config.system.steps_per_action
        self.observation_cache_steps = 20 if training_mode else 15  # ULTRA-FAST: Longer caching
        self.last_observation = None
        self.last_obs_step = -1
        
        # Pre-allocated observation array
        self._obs_array = np.zeros(195, dtype=np.float32)
        
        # Dedicated metrics manager
        self.metrics_manager = SimulationMetricsManager()
        
        # Deadlock timeout and reset configuration - from unified config
        self.deadlock_reset_enabled = self.sim_cfg.get('deadlock_reset_enabled', True)
        self.deadlock_timeout_duration = self.unified_config.deadlock.deadlock_timeout_duration
        self.deadlock_first_detected_time = None
        self.deadlock_consecutive_detections = 0
        self.deadlock_reset_count = 0
        self.max_deadlock_resets = self.unified_config.deadlock.max_deadlock_resets
        
        # Severe deadlock (severity 1.0) immediate reset configuration - from unified config
        self.severe_deadlock_reset_enabled = self.unified_config.system.severe_deadlock_reset_enabled
        self.severe_deadlock_punishment = self.unified_config.system.severe_deadlock_punishment
        self.severe_deadlock_reset_count = 0
        
        # Initialize simulation components
        self._init_simulation()
        
        # Trainable policy
        self.bid_policy = TrainableBidPolicy()
        
        print(f"🤖 Streamlined Simulation Environment initialized with UNIFIED CONFIG")
        print(f"   🔧 Config - Conflict window: {self.unified_config.conflict.conflict_time_window}s")
        print(f"   🔧 Config - Safe distance: {self.unified_config.conflict.min_safe_distance}m")
        print(f"   🔧 Config - Deadlock threshold: {self.unified_config.deadlock.deadlock_speed_threshold} m/s")
        if self.deadlock_reset_enabled:
            print(f"   🔄 Deadlock auto-reset: ON (timeout: {self.deadlock_timeout_duration}s, max resets: {self.max_deadlock_resets})")
        if self.severe_deadlock_reset_enabled:
            print(f"   ⚡ Severe deadlock reset: ON (punishment: {self.severe_deadlock_punishment})")

    def observation_dim(self) -> int:
        """Return fixed observation space dimension"""
        return 195  # 10 + 160 + 20 + 5

    def _init_simulation(self):
        """Initialize core simulation components"""
        try:
            # Core components - SPEED UP: Pass training mode to state extractor
            self.scenario = ScenarioManager()
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
        """Reset environment with improved stability"""
        try:
            self.current_step = 0      # Reset simulation steps
            self.current_action = 0    # Reset action counter
            # Note: We'll sync metrics after components are ready
            
            # Reset deadlock tracking
            self.deadlock_first_detected_time = None
            self.deadlock_consecutive_detections = 0
            self.deadlock_reset_count = 0
            self.severe_deadlock_reset_count = 0
            
            # Reset observation cache
            self.last_observation = None
            self.last_obs_step = -1
            
            print(f"🔄 Environment reset starting...")
            
            # Safe scenario reset
            reset_success = self._safe_reset_scenario()
            if not reset_success:
                print("❌ Reset failed - returning zero observation")
                return np.zeros(195, dtype=np.float32)
            
            # Start simulation timers
            if hasattr(self.scenario, 'start_time_counters'):
                self.scenario.start_time_counters()
            
            # OPTIMIZED: Minimal component recreation - reuse existing objects
            # Only create platoon manager if it doesn't exist
            if not hasattr(self, 'platoon_manager') or self.platoon_manager is None:
                self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Keep existing auction engine and nash solver - avoid recreation
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
            
            # Quick reconnect
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            
            # Connect trainable policy
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # CRITICAL: Reset episode state but preserve cumulative statistics
            self.traffic_controller.reset_episode_state()
            
            # Reset auction engine state
            if hasattr(self.auction_engine, 'reset_episode_state'):
                self.auction_engine.reset_episode_state()
            
            # Reset Nash solver statistics  
            if hasattr(self.nash_solver, 'reset_stats'):
                self.nash_solver.reset_stats()
            
            # IMPORTANT: Reset metrics AFTER all component statistics are reset
            self.metrics_manager.reset_metrics(
                nash_solver=self.nash_solver,
                traffic_controller=self.traffic_controller
            )
            
            # CRITICAL: Allow time for vehicle stabilization before first update
            time.sleep(0.1)  # Small delay to ensure vehicle physics are stable
            
            # Initial system update - now with properly registered vehicles
            initial_vehicles = self.state_extractor.get_vehicle_states()
            print(f"🚗 First update after reset: {len(initial_vehicles)} vehicles detected")
            
            if initial_vehicles:
                self.platoon_manager.update()
                winners = self.auction_engine.update(initial_vehicles, self.platoon_manager)
                # First update after reset - TrafficController will skip exit tracking
                self.traffic_controller.update_control(
                    self.platoon_manager, self.auction_engine, winners
                )
            
            # Get initial observation
            obs = self._get_observation()
            
            print(f"✅ Reset complete - {len(initial_vehicles)} vehicles active")
            return obs
            
        except Exception as e:
            print(f"❌ Reset failed: {str(e)}")
            return np.zeros(195, dtype=np.float32)

    def step_with_all_params(self, action_params: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute simulation step with all trainable parameters"""
        step_start = time.time()
        
        try:
            # Increment action counter
            self.current_action += 1
            
            # Update trainable parameters
            self._update_policy_parameters(action_params)
            
            # Connect policies
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
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
                    # Check for severe deadlock (severity 1.0) first - immediate action needed
                    if self._check_severe_deadlock():
                        print(f"🚨 SEVERE DEADLOCK (severity 1.0) detected at step {self.current_step}")
                        severe_deadlock_occurred = True
                        if self.severe_deadlock_reset_enabled:
                            print(f"⚡ Performing immediate severe deadlock reset")
                            if self._perform_severe_deadlock_reset():
                                break  # Exit loop to apply punishment and continue
                            else:
                                print(f"❌ Severe deadlock reset failed - terminating episode")
                                break
                        else:
                            break
                    
                    # Check for regular deadlock
                    deadlock_detected = self._check_deadlock()
                    collision_detected = self._check_collision()
                    
                    if collision_detected:
                        print(f"💥 Collision detected at step {self.current_step}")
                        break
                        
                    # For regular deadlock: only break if not using auto-reset or reset failed
                    if deadlock_detected:
                        # If deadlock reset is disabled or we're out of resets, break
                        if (not self.deadlock_reset_enabled or 
                            self.deadlock_reset_count >= self.max_deadlock_resets):
                            print(f"🚨 Deadlock detected at step {self.current_step} - episode terminating")
                            break
                        # Otherwise, deadlock handling may have performed a reset, continue
            
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
            
            # Check episode termination based on ACTIONS, not simulation steps
            # For deadlock: only terminate if reset is disabled or max resets exceeded
            deadlock_should_terminate = (self._check_deadlock() and 
                                       (not self.deadlock_reset_enabled or 
                                        self.deadlock_reset_count >= self.max_deadlock_resets))
            
            done = (self.current_action >= self.max_actions or 
                   self._check_collision() or 
                   deadlock_should_terminate)
            
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
                    'observation_valid': len(obs) == 195
                },
                'action_info': {
                    'current_action': self.current_action,
                    'max_actions': self.max_actions,
                    'sim_steps_taken': self.current_step,
                    'steps_per_action': self.steps_per_action
                },
                'deadlock_reset_info': {
                    'deadlock_reset_enabled': self.deadlock_reset_enabled,
                    'deadlock_reset_count': self.deadlock_reset_count,
                    'max_deadlock_resets': self.max_deadlock_resets,
                    'deadlock_timeout_duration': self.deadlock_timeout_duration,
                    'deadlock_currently_detected': self.deadlock_first_detected_time is not None,
                    'deadlock_consecutive_detections': self.deadlock_consecutive_detections,
                    'severe_deadlock_reset_count': self.severe_deadlock_reset_count,
                    'severe_deadlock_occurred_this_step': severe_deadlock_occurred
                }
            })
            
            # Record performance
            step_time = time.time() - step_start
            self.metrics_manager.record_performance(step_time)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"❌ Step failed: {str(e)}")
            return (np.zeros(195, dtype=np.float32), -10.0, True, 
                   {'error': str(e), 'using_real_data': False})

    def _update_policy_parameters(self, action_params: Dict):
        """Update all trainable policy parameters"""
        self.bid_policy.update_all_bid_params(
            bid_scale=action_params.get('bid_scale'),
            eta_weight=action_params.get('eta_weight'),
            speed_weight=action_params.get('speed_weight'),
            congestion_sensitivity=action_params.get('congestion_sensitivity'),
            platoon_bonus=action_params.get('platoon_bonus'),
            junction_penalty=action_params.get('junction_penalty'),
            fairness_factor=action_params.get('fairness_factor'),
            urgency_threshold=action_params.get('urgency_threshold'),
            proximity_bonus_weight=action_params.get('proximity_bonus_weight')
        )
        
        self.bid_policy.update_control_params(
            speed_diff_modifier=action_params.get('speed_diff_modifier'),
            follow_distance_modifier=action_params.get('follow_distance_modifier')
        )
        
        self.bid_policy.update_ignore_vehicles_params(
            ignore_vehicles_go=action_params.get('ignore_vehicles_go'),
            ignore_vehicles_wait=action_params.get('ignore_vehicles_wait'),
            ignore_vehicles_platoon_leader=action_params.get('ignore_vehicles_platoon_leader'),
            ignore_vehicles_platoon_follower=action_params.get('ignore_vehicles_platoon_follower')
        )
        
        # Update auction engine parameters
        if 'max_participants_per_auction' in action_params:
            self.auction_engine.update_max_participants_per_auction(
                action_params['max_participants_per_auction']
            )
        
        # Update Nash solver parameters via unified config
        nash_params = {}
        if 'path_intersection_threshold' in action_params:
            nash_params['path_intersection_threshold'] = action_params['path_intersection_threshold']
        if 'platoon_conflict_distance' in action_params:
            nash_params['platoon_conflict_distance'] = action_params['platoon_conflict_distance']
        
        if nash_params:
            # Update unified config and recreate Nash solver for new parameters
            self.unified_config.update_from_drl_params(**nash_params)
            
            # Recreate Nash solver with updated config
            self.nash_solver = DeadlockNashSolver(
                unified_config=self.unified_config,
                intersection_center=self.unified_config.system.intersection_center,
                max_go_agents=self.unified_config.mwis.max_go_agents
            )
            
            # Reconnect to systems
            self.auction_engine.set_nash_controller(self.nash_solver)

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
        """Generate observation array with fixed 195 dimensions"""
        try:
            # Reset array
            self._obs_array.fill(0.0)
            
            # Get current state
            vehicle_states = self.state_extractor.get_vehicle_states()
            control_stats = self.traffic_controller.get_control_stats()
            auction_stats = self.auction_engine.get_auction_stats()
            
            # Base metrics (indices 0-9)
            vehicles_in_junction = sum(1 for v in vehicle_states if v.get('is_junction', False))
            self._obs_array[0] = min(len(vehicle_states), 999)
            self._obs_array[1] = min(vehicles_in_junction, 50)
            self._obs_array[2] = min(control_stats.get('total_controlled', 0), 100)
            self._obs_array[3] = min(control_stats.get('go_vehicles', 0), 50)
            self._obs_array[4] = min(control_stats.get('waiting_vehicles', 0), 50)
            self._obs_array[5] = min(auction_stats.get('current_agents', 0), 50)
            self._obs_array[6] = min(auction_stats.get('current_go_count', 0), 20)
            self._obs_array[7] = np.clip(self.metrics_manager.metrics.get('throughput', 0), 0, 10000)
            self._obs_array[8] = np.clip(self.metrics_manager.metrics.get('avg_acceleration', 0), -10, 10)
            self._obs_array[9] = min(self.metrics_manager.metrics.get('collision_count', 0), 100)
            
            # Vehicle states (indices 10-169, 20 vehicles × 8 features each)
            try:
                active_controls = control_stats.get('active_controls', [])
                for i, vehicle_state in enumerate(vehicle_states[:20]):
                    try:
                        base_idx = 10 + i * 8
                        
                        # Location
                        loc = vehicle_state.get('location', {})
                        if isinstance(loc, dict):
                            self._obs_array[base_idx] = float(loc.get('x', 0.0))
                            self._obs_array[base_idx + 1] = float(loc.get('y', 0.0))
                        elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                            self._obs_array[base_idx] = float(loc[0])
                            self._obs_array[base_idx + 1] = float(loc[1])
                        
                        # Velocity
                        vel = vehicle_state.get('velocity', {})
                        if isinstance(vel, dict):
                            speed = np.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2)
                            self._obs_array[base_idx + 2] = min(speed, 50.0)
                        elif isinstance(vel, (list, tuple)) and len(vel) >= 3:
                            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                            self._obs_array[base_idx + 2] = min(speed, 50.0)
                        
                        # Other features
                        self._obs_array[base_idx + 3] = np.clip(vehicle_state.get('eta_to_intersection', 0), 0, 100)
                        self._obs_array[base_idx + 4] = float(vehicle_state.get('is_junction', False))
                        vehicle_id = vehicle_state.get('id', 0)
                        self._obs_array[base_idx + 5] = float(str(vehicle_id) in active_controls)
                        self._obs_array[base_idx + 6] = np.clip(vehicle_state.get('lane_id', 0), 0, 10)
                        # index 7 reserved
                    except Exception:

                        base_idx = 10 + i * 8
                        self._obs_array[base_idx:base_idx + 8] = 0.0
            except Exception as vehicles_error:
                print(f"⚠️ Vehicle states observation error: {str(vehicles_error)}")
                # Fill entire vehicle section with zeros
                self._obs_array[10:170] = 0.0
            
            # Platoon info (indices 170-189, 5 platoons × 4 features each)
            try:
                platoons = self.platoon_manager.get_all_platoons()
                for i, platoon in enumerate(platoons[:5]):
                    base_idx = 170 + i * 4
                    self._obs_array[base_idx] = min(platoon.get_size(), 20)
                    leader_id = platoon.get_leader_id()
                    if leader_id is not None and isinstance(leader_id, (int, float)):
                        self._obs_array[base_idx + 1] = float(leader_id % 10000)
                    else:
                        self._obs_array[base_idx + 1] = 0.0
                    self._obs_array[base_idx + 2] = min(len(platoon.get_follower_ids()), 15)
                    # index 3 reserved
            except Exception as platoon_error:
                print(f"⚠️ Platoon observation error: {str(platoon_error)}")
                # Fill platoon section with zeros if error occurs
                for i in range(5):
                    base_idx = 170 + i * 4
                    self._obs_array[base_idx:base_idx + 4] = 0.0
            
            # Auction info (indices 190-194)
            priority_order = self.auction_engine.get_current_priority_order()
            self._obs_array[190] = min(len(priority_order), 50)
            self._obs_array[191] = np.clip(self.bid_policy.get_current_bid_scale(), 0.1, 5.0)
            self._obs_array[192] = min(auction_stats.get('platoon_agents', 0), 20)
            self._obs_array[193] = min(auction_stats.get('vehicle_agents', 0), 50)
            # index 194 reserved
            
            # Final validation
            self._obs_array = np.nan_to_num(self._obs_array, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            return self._obs_array.copy()
            
        except Exception as e:
            print(f"❌ Observation generation failed: {str(e)}")
            return np.zeros(195, dtype=np.float32)

    def _cleanup_existing_vehicles(self):
        """Clean up existing vehicles"""
        try:
            # Safeguard when CARLA is not available
            if carla is None or not hasattr(self, 'scenario') or not hasattr(self.scenario, 'carla'):
                return
            world = self.scenario.carla.world
            vehicles = world.get_actors().filter('vehicle.*')
            
            if len(vehicles) > 0:
                print(f"🧹 Cleaning {len(vehicles)} vehicles")
                client = self.scenario.carla.client
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
                world.tick()
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {str(e)}")

    def _safe_reset_scenario(self) -> bool:
        """OPTIMIZED single-attempt scenario reset"""
        try:
            print("🎯 Quick scenario reset...")
            
            # Reset scenario and generate new traffic
            self.scenario.reset_scenario()
            
            # OPTIMIZED: Much shorter wait time for training
            if self.sim_cfg.get('training_mode', False):
                time.sleep(0.2)  # Reduced from 0.5
            else:
                time.sleep(0.8)  # Reduced from 1.5
            
            # Single world tick - no extra sleep
            self.scenario.carla.world.tick()
            
            # Quick verification without extensive debugging
            vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
            if len(vehicles) > 0:
                print(f"✅ Reset successful: {len(vehicles)} vehicles")
                return True
            else:
                print(f"⚠️ No vehicles after reset - retrying once")
                # One quick retry
                time.sleep(0.3)
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
        """Check for deadlocks with proper error handling and timeout management"""
        try:
            if (hasattr(self, 'nash_solver') and 
                hasattr(self.nash_solver, 'deadlock_detector')):
                vehicle_states = self.state_extractor.get_vehicle_states()
                current_time = self.scenario.carla.world.get_snapshot().timestamp.elapsed_seconds
                
                vehicle_dict = {str(v['id']): v for v in vehicle_states}
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
        """Handle deadlock detection with timeout and auto-reset logic"""
        current_time = self.scenario.carla.world.get_snapshot().timestamp.elapsed_seconds
        
        # Track deadlock duration
        if self.deadlock_first_detected_time is None:
            self.deadlock_first_detected_time = current_time
            self.deadlock_consecutive_detections = 1
            print(f"🚨 DEADLOCK FIRST DETECTED: {deadlock_exception.deadlock_type} affecting {deadlock_exception.affected_vehicles} vehicles")
            print(f"   ⏱️ Starting deadlock timer - will auto-reset in {self.deadlock_timeout_duration}s if persists")
        else:
            self.deadlock_consecutive_detections += 1
            deadlock_duration = current_time - self.deadlock_first_detected_time
            
            # Check if deadlock has persisted for timeout duration
            if (self.deadlock_reset_enabled and 
                deadlock_duration >= self.deadlock_timeout_duration and 
                self.deadlock_reset_count < self.max_deadlock_resets):
                
                print(f"⏰ DEADLOCK TIMEOUT: {deadlock_duration:.1f}s >= {self.deadlock_timeout_duration}s")
                print(f"🔄 Initiating automatic scenario reset ({self.deadlock_reset_count + 1}/{self.max_deadlock_resets})")
                
                # Perform immediate scenario reset
                if self._perform_deadlock_reset():
                    self.deadlock_reset_count += 1
                    self.deadlock_first_detected_time = None
                    self.deadlock_consecutive_detections = 0
                    return False  # Reset successful, continue without episode termination
                else:
                    print("❌ Deadlock reset failed - terminating episode")
                    return True  # Reset failed, terminate episode
            
            elif deadlock_duration >= self.deadlock_timeout_duration:
                # Max resets exceeded
                print(f"🚫 Max deadlock resets exceeded ({self.max_deadlock_resets}) - terminating episode")
                return True
            
            else:
                # Still within timeout period
                print(f"⏳ Deadlock persisting: {deadlock_duration:.1f}s / {self.deadlock_timeout_duration}s")
        
        # Return True to indicate deadlock (will cause negative reward but not episode termination if reset is available)
        return True

    def _perform_deadlock_reset(self) -> bool:
        """Perform scenario reset to resolve deadlock situation"""
        try:
            print("🔄 Performing deadlock resolution reset...")
            
            # Clean up current state
            self._cleanup_existing_vehicles()
            time.sleep(0.3)  # Reduced wait time
            
            # Reset scenario
            self.scenario.reset_scenario()
            
            # OPTIMIZED: Much faster deadlock reset
            wait_time = 0.3 if self.sim_cfg.get('training_mode', False) else 0.8
            time.sleep(wait_time)
            
            # OPTIMIZED: Reuse existing components to prevent resource leaks
            # Only recreate platoon manager (lightweight)
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Keep existing engines - DO NOT recreate to prevent file handle leaks
            # Just ensure they exist, but don't recreate them unnecessarily
            
            # Quick reconnect
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # CRITICAL: Reset episode state after deadlock reset  
            self.traffic_controller.reset_episode_state()
            if hasattr(self.auction_engine, 'reset_episode_state'):
                self.auction_engine.reset_episode_state()
            if hasattr(self.nash_solver, 'reset_stats'):
                self.nash_solver.reset_stats()
            
            # Verify reset success - use include_all_vehicles for comprehensive check
            new_vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
            if len(new_vehicles) > 0:
                # Initial system update
                self.platoon_manager.update()
                winners = self.auction_engine.update(new_vehicles, self.platoon_manager)
                self.traffic_controller.update_control(
                    self.platoon_manager, self.auction_engine, winners
                )
                
                print(f"✅ Deadlock reset successful - {len(new_vehicles)} new vehicles spawned")
                return True
            else:
                print("⚠️ No vehicles after deadlock reset - scenario may not have reset properly")
                return False
                
        except Exception as e:
            print(f"❌ Deadlock reset failed: {str(e)}")
            return False

    def _check_severe_deadlock(self) -> bool:
        """Check for severe deadlock (severity 1.0) requiring immediate reset"""
        try:
            if (hasattr(self, 'nash_solver') and 
                hasattr(self.nash_solver, 'deadlock_detector')):
                
                # Get current deadlock severity
                severity = self.nash_solver.deadlock_detector.get_deadlock_severity()
                
                # Check if severity is 1.0 (complete deadlock)
                if severity >= 0.99:  # Use 0.99 to account for floating point precision
                    print(f"⚡ SEVERE DEADLOCK: severity {severity:.2f} >= 0.99")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"⚠️ Severe deadlock detection error: {str(e)}")
            return False

    def _perform_severe_deadlock_reset(self) -> bool:
        """Perform immediate scenario reset for severe deadlock (severity 1.0)"""
        try:
            print("⚡ Performing SEVERE deadlock resolution reset...")
            self.severe_deadlock_reset_count += 1
            
            # Clean up current state immediately
            self._cleanup_existing_vehicles()
            time.sleep(0.2)  # Even shorter wait for immediate response
            
            # Reset scenario
            self.scenario.reset_scenario()
            
            # OPTIMIZED: Ultra-fast reset for severe deadlock
            wait_time = 0.2 if self.sim_cfg.get('training_mode', False) else 0.5
            time.sleep(wait_time)
            
            # OPTIMIZED: Minimal recreation to prevent resource leaks  
            # Only recreate platoon manager (lightweight)
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Keep existing engines - DO NOT recreate to prevent file handle accumulation
            
            # Ultra-quick reconnect
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # CRITICAL: Reset episode state after severe deadlock reset
            self.traffic_controller.reset_episode_state()
            if hasattr(self.auction_engine, 'reset_episode_state'):
                self.auction_engine.reset_episode_state()
            if hasattr(self.nash_solver, 'reset_stats'):
                self.nash_solver.reset_stats()
            
            # Reset deadlock tracking since we have a fresh scenario
            self.deadlock_first_detected_time = None
            self.deadlock_consecutive_detections = 0
            
            # Verify reset success - use include_all_vehicles for comprehensive check
            new_vehicles = self.state_extractor.get_vehicle_states(include_all_vehicles=True)
            if len(new_vehicles) > 0:
                # Initial system update
                self.platoon_manager.update()
                winners = self.auction_engine.update(new_vehicles, self.platoon_manager)
                self.traffic_controller.update_control(
                    self.platoon_manager, self.auction_engine, winners
                )
                
                print(f"✅ SEVERE deadlock reset successful - {len(new_vehicles)} new vehicles spawned")
                print(f"⚡ Total severe deadlock resets this episode: {self.severe_deadlock_reset_count}")
                return True
            else:
                print("⚠️ No vehicles after severe deadlock reset - scenario may not have reset properly")
                return False
                
        except Exception as e:
            print(f"❌ Severe deadlock reset failed: {str(e)}")
            return False

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