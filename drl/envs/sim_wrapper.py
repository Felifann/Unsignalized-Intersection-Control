import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

# Import CARLA after adding egg to path
try:
    import carla
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
from drl.policies.bid_policy import TrainableBidPolicy
from drl.envs.metrics_manager import SimulationMetricsManager

class SimulationEnv:
    """Streamlined simulation environment wrapper"""
    
    def __init__(self, sim_cfg: dict = None):
        self.sim_cfg = sim_cfg or {}
        self.max_steps = self.sim_cfg.get('max_steps', 2000)
        self.current_step = 0
        
        # Performance settings
        self.steps_per_action = self.sim_cfg.get('steps_per_action', 3)
        self.observation_cache_steps = 5
        self.last_observation = None
        self.last_obs_step = -1
        
        # Pre-allocated observation array
        self._obs_array = np.zeros(195, dtype=np.float32)
        
        # Dedicated metrics manager
        self.metrics_manager = SimulationMetricsManager()
        
        # Initialize simulation components
        self._init_simulation()
        
        # Trainable policy
        self.bid_policy = TrainableBidPolicy()
        
        print(f"🤖 Streamlined Simulation Environment initialized")

    def observation_dim(self) -> int:
        """Return fixed observation space dimension"""
        return 195  # 10 + 160 + 20 + 5

    def _init_simulation(self):
        """Initialize core simulation components"""
        try:
            # Core components
            self.scenario = ScenarioManager()
            self.state_extractor = StateExtractor(self.scenario.carla)
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
                max_go_agents=None
            )
            
            self.nash_solver = DeadlockNashSolver(
                max_exact=15,
                conflict_time_window=3.0,
                intersection_center=(-188.9, -89.7, 0.0),
                max_go_agents=None
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
            print(f"❌ Simulation initialization failed: {e}")
            raise

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment with improved stability"""
        try:
            self.current_step = 0
            self.metrics_manager.reset_metrics()
            
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
            
            # Reinitialize components for clean state
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            self.auction_engine = DecentralizedAuctionEngine(
                state_extractor=self.state_extractor,
                max_go_agents=None
            )
            
            self.nash_solver = DeadlockNashSolver(
                max_exact=15,
                conflict_time_window=3.0,
                intersection_center=(-188.9, -89.7, 0.0),
                max_go_agents=None
            )
            
            # Reconnect components
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            
            # Connect trainable policy
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # Initial system update
            initial_vehicles = self.state_extractor.get_vehicle_states()
            if initial_vehicles:
                self.platoon_manager.update()
                winners = self.auction_engine.update(initial_vehicles, self.platoon_manager)
                self.traffic_controller.update_control(
                    self.platoon_manager, self.auction_engine, winners
                )
            
            # Get initial observation
            obs = self._get_observation()
            
            print(f"✅ Reset complete - {len(initial_vehicles)} vehicles active")
            return obs
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            return np.zeros(195, dtype=np.float32)

    def step_with_all_params(self, action_params: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute simulation step with all trainable parameters"""
        step_start = time.time()
        
        try:
            # Update trainable parameters
            self._update_policy_parameters(action_params)
            
            # Connect policies
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # Run simulation steps
            initial_vehicle_count = len(self.state_extractor.get_vehicle_states())
            
            for i in range(self.steps_per_action):
                # Advance simulation
                self.scenario.carla.world.tick()
                self.current_step += 1
                
                # Update system components
                vehicle_states = self.state_extractor.get_vehicle_states()
                
                if vehicle_states:
                    try:
                        self.platoon_manager.update()
                        auction_winners = self.auction_engine.update(vehicle_states, self.platoon_manager)
                        self.traffic_controller.update_control(
                            self.platoon_manager, self.auction_engine, auction_winners
                        )
                    except Exception as update_error:
                        print(f"⚠️ Update error: {update_error}")
                        continue
                
                # Early termination check
                if self._check_collision() or self._check_deadlock():
                    break
            
            # Calculate reward using metrics manager
            reward = self.metrics_manager.calculate_reward(
                self.traffic_controller, self.state_extractor, 
                self.scenario, self.nash_solver, self.current_step
            )
            
            # Get observation
            obs = self._get_observation_cached()
            
            # Check episode termination
            done = (self.current_step >= self.max_steps or 
                   self._check_collision() or 
                   self._check_deadlock())
            
            # Generate info using metrics manager
            info = self.metrics_manager.get_info_dict(
                self.traffic_controller, self.auction_engine, self.nash_solver,
                self.scenario, self.state_extractor, self.bid_policy,
                self.current_step, self.max_steps
            )
            
            # Add validation info
            current_vehicles = len(self.state_extractor.get_vehicle_states())
            info.update({
                'step_validation': {
                    'vehicles_stable': abs(current_vehicles - initial_vehicle_count) <= 3,
                    'reward_realistic': abs(reward) <= 50.0,
                    'observation_valid': len(obs) == 195
                }
            })
            
            # Record performance
            step_time = time.time() - step_start
            self.metrics_manager.record_performance(step_time)
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
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
            active_controls = control_stats.get('active_controls', [])
            for i, vehicle_state in enumerate(vehicle_states[:20]):
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
                self._obs_array[base_idx + 5] = float(str(vehicle_state['id']) in active_controls)
                self._obs_array[base_idx + 6] = np.clip(vehicle_state.get('lane_id', 0), 0, 10)
                # index 7 reserved
            
            # Platoon info (indices 170-189, 5 platoons × 4 features each)
            platoons = self.platoon_manager.get_all_platoons()
            for i, platoon in enumerate(platoons[:5]):
                base_idx = 170 + i * 4
                self._obs_array[base_idx] = min(platoon.get_size(), 20)
                self._obs_array[base_idx + 1] = float(platoon.get_leader_id() % 10000)
                self._obs_array[base_idx + 2] = min(len(platoon.get_follower_ids()), 15)
                # index 3 reserved
            
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
            print(f"❌ Observation generation failed: {e}")
            return np.zeros(195, dtype=np.float32)

    def _cleanup_existing_vehicles(self):
        """Clean up existing vehicles"""
        try:
            world = self.scenario.carla.world
            vehicles = world.get_actors().filter('vehicle.*')
            
            if len(vehicles) > 0:
                print(f"🧹 Cleaning {len(vehicles)} vehicles")
                client = self.scenario.carla.client
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
                world.tick()
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

    def _safe_reset_scenario(self, max_attempts: int = 3) -> bool:
        """Safe scenario reset with stability checks"""
        for attempt in range(max_attempts):
            try:
                print(f"🎯 Reset attempt {attempt + 1}/{max_attempts}")
                
                if attempt == 0:
                    self._cleanup_existing_vehicles()
                    time.sleep(0.5)
                
                self.scenario.reset_scenario()
                time.sleep(3.0 + attempt * 1.0)
                
                # Verify stability
                vehicles = self.state_extractor.get_vehicle_states()
                if len(vehicles) > 0:
                    time.sleep(2.0)
                    vehicles_after = self.state_extractor.get_vehicle_states()
                    
                    if len(vehicles_after) > 0:
                        print(f"✅ Stable reset: {len(vehicles_after)} vehicles")
                        return True
                        
            except Exception as e:
                print(f"⚠️ Reset attempt {attempt + 1} failed: {e}")
                continue
        
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
        """Check for deadlocks"""
        try:
            if (hasattr(self, 'nash_solver') and 
                hasattr(self.nash_solver, 'deadlock_detector')):
                vehicle_states = self.state_extractor.get_vehicle_states()
                current_time = self.scenario.carla.world.get_snapshot().timestamp.elapsed_seconds
                
                vehicle_dict = {str(v['id']): v for v in vehicle_states}
                return self.nash_solver.deadlock_detector.detect_deadlock(vehicle_dict, current_time)
            
            return False
            
        except Exception:
            return False

    def close(self):
        """Clean up environment"""
        try:
            if hasattr(self.scenario, 'stop_time_counters'):
                self.scenario.stop_time_counters()
            print("🏁 Environment closed")
        except Exception as e:
            print(f"❌ Close error: {e}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.metrics_manager.get_performance_stats()

    def set_performance_mode(self, fast_mode: bool = True):
        """Adjust performance settings"""
        if fast_mode:
            self.steps_per_action = 2
            self.observation_cache_steps = 10
        else:
            self.steps_per_action = 5
            self.observation_cache_steps = 5
        
        print(f"🔧 Performance mode: {'fast' if fast_mode else 'normal'}")