import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from platooning.platoon_manager import PlatoonManager
from auction.auction_engine import DecentralizedAuctionEngine
from control import TrafficController
from nash.deadlock_nash_solver import DeadlockNashSolver
from drl.policies.bid_policy import TrainableBidPolicy

class SimulationEnv:
    """Wrapper for the traffic intersection simulation"""
    
    def __init__(self, sim_cfg: dict = None):
        self.sim_cfg = sim_cfg or {}
        self.max_steps = self.sim_cfg.get('max_steps', 2000)
        self.current_step = 0
        
        # Performance optimization settings
        self.steps_per_action = self.sim_cfg.get('steps_per_action', 3)  # Reduced from 10
        self.observation_cache_steps = 5  # Cache observation for N steps
        self.last_observation = None
        self.last_obs_step = -1
        
        # Pre-allocated arrays for performance
        self._obs_array = np.zeros(self.observation_dim(), dtype=np.float32)
        self._vehicle_features = np.zeros(160, dtype=np.float32)
        self._platoon_features = np.zeros(20, dtype=np.float32)
        
        # Performance tracking
        self.perf_stats = {
            'step_times': [],
            'obs_times': [],
            'reward_times': [],
            'total_ticks': 0
        }
        
        # Initialize simulation components
        self._init_simulation()
        
        # Trainable policies
        self.bid_policy = TrainableBidPolicy()
        
        # Metrics tracking
        self.reset_metrics()
        
        print(f"🤖 DRL Simulation Environment initialized (steps_per_action: {self.steps_per_action})")

    def _init_simulation(self):
        """Initialize all simulation components"""
        try:
            # Core simulation components
            self.scenario = ScenarioManager()
            self.state_extractor = StateExtractor(self.scenario.carla)
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Auction and control systems
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
            
        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            raise

    def observation_dim(self) -> int:
        """Return observation space dimension"""
        # Base metrics: 10 features
        # Vehicle states: up to 20 vehicles × 8 features = 160
        # Platoon info: up to 5 platoons × 4 features = 20
        # Auction info: 5 features
        return 195

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the simulation environment"""
        try:
            self.current_step = 0
            self.reset_metrics()
            
            # Reset observation cache
            self.last_observation = None
            self.last_obs_step = -1
            
            # Reset simulation
            self.scenario.reset_scenario()
            self.scenario.start_time_counters()
            
            # Reset components
            self.platoon_manager = PlatoonManager(self.state_extractor)
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            
            # Get initial observation using the optimized method
            obs = self._get_observation_optimized()
            return obs
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.observation_dim())

    def step_with_bid_scale(self, bid_scale: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step simulation with DRL action - optimized version"""
        step_start = time.time()
        
        try:
            # Update trainable parameters
            self.bid_policy.update_bid_scale(bid_scale)
            
            # Connect bid policy to auction engine
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # Run fewer simulation steps per action for speed
            rewards = []
            for i in range(self.steps_per_action):
                tick_start = time.time()
                
                self.scenario.carla.world.tick()
                self.current_step += 1
                self.perf_stats['total_ticks'] += 1
                
                # Only update components every other tick for performance
                if i % 2 == 0 or i == self.steps_per_action - 1:
                    # Update simulation components
                    vehicle_states = self.state_extractor.get_vehicle_states()
                    self.platoon_manager.update()
                    
                    # Apply trainable bid policy
                    auction_winners = self.auction_engine.update(vehicle_states, self.platoon_manager)
                    
                    # Update traffic control
                    self.traffic_controller.update_control(
                        self.platoon_manager, self.auction_engine, auction_winners
                    )
                
                # Calculate step reward less frequently
                if i == self.steps_per_action - 1:  # Only on last step
                    reward_start = time.time()
                    step_reward = self._calculate_reward()
                    self.perf_stats['reward_times'].append(time.time() - reward_start)
                    rewards.append(step_reward)
                
                # Check for early termination
                if self._check_collision() or self._check_deadlock():
                    break
            
            # Aggregate reward
            total_reward = sum(rewards) if rewards else 0.0
            
            # Get new observation (with caching)
            obs_start = time.time()
            obs = self._get_observation_cached()
            self.perf_stats['obs_times'].append(time.time() - obs_start)
            
            # Check if episode is done
            done = (self.current_step >= self.max_steps or 
                   self._check_collision() or 
                   self._check_deadlock())
            
            # Prepare info with performance stats
            info = self._get_info()
            if len(self.perf_stats['step_times']) > 0:
                info['perf'] = {
                    'avg_step_time': np.mean(self.perf_stats['step_times'][-100:]),
                    'avg_obs_time': np.mean(self.perf_stats['obs_times'][-100:]),
                    'avg_reward_time': np.mean(self.perf_stats['reward_times'][-100:]) if self.perf_stats['reward_times'] else 0,
                    'total_ticks': self.perf_stats['total_ticks']
                }
            
            step_time = time.time() - step_start
            self.perf_stats['step_times'].append(step_time)
            
            return obs, total_reward, done, info
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
            obs = np.zeros(self.observation_dim())
            return obs, -100.0, True, {'error': str(e)}

    def step_with_all_params(self, action_params: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step simulation with all trainable parameters - optimized version"""
        step_start = time.time()
        
        try:
            # Update all bid policy parameters
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
            
            # Update control parameters
            self.bid_policy.update_control_params(
                speed_diff_modifier=action_params.get('speed_diff_modifier'),
                follow_distance_modifier=action_params.get('follow_distance_modifier')
            )
            
            # Update ignore_vehicles parameters
            self.bid_policy.update_ignore_vehicles_params(
                ignore_vehicles_go=action_params.get('ignore_vehicles_go'),
                ignore_vehicles_wait=action_params.get('ignore_vehicles_wait'),
                ignore_vehicles_platoon_leader=action_params.get('ignore_vehicles_platoon_leader'),
                ignore_vehicles_platoon_follower=action_params.get('ignore_vehicles_platoon_follower')
            )
            
            # Connect bid policy to both systems
            self.traffic_controller.set_bid_policy(self.bid_policy)
            self.auction_engine.set_bid_policy(self.bid_policy)
            
            # Run fewer simulation steps per action for speed
            rewards = []
            cached_vehicle_states = None
            
            for i in range(self.steps_per_action):
                self.scenario.carla.world.tick()
                self.current_step += 1
                self.perf_stats['total_ticks'] += 1
                
                # Cache and reuse vehicle states for performance
                if i % 2 == 0 or cached_vehicle_states is None:
                    cached_vehicle_states = self.state_extractor.get_vehicle_states()
                    self.platoon_manager.update()
                
                # Update auction engine with cached vehicle states
                auction_winners = self.auction_engine.update(cached_vehicle_states, self.platoon_manager)
                
                # Update traffic control less frequently
                if i % 2 == 0:
                    self.traffic_controller.update_control(
                        self.platoon_manager, self.auction_engine, auction_winners
                    )
                
                # Calculate step reward less frequently
                if i == self.steps_per_action - 1:  # Only on last step
                    step_reward = self._calculate_reward()
                    rewards.append(step_reward)
                
                # Check for early termination
                if self._check_collision() or self._check_deadlock():
                    break
            
            # Aggregate reward
            total_reward = sum(rewards) if rewards else 0.0
            
            # Get new observation (with caching)
            obs = self._get_observation_cached()
            
            # Check if episode is done
            done = (self.current_step >= self.max_steps or 
                   self._check_collision() or 
                   self._check_deadlock())
            
            # Prepare enhanced info with all parameters and performance stats
            info = self._get_info()
            info.update({
                # Add all trainable parameters to info for logging
                'bid_scale': self.bid_policy.bid_scale,
                'eta_weight': self.bid_policy.eta_weight,
                'speed_weight': self.bid_policy.speed_weight,
                'congestion_sensitivity': self.bid_policy.congestion_sensitivity,
                'platoon_bonus': self.bid_policy.platoon_bonus,
                'junction_penalty': self.bid_policy.junction_penalty,
                'fairness_factor': self.bid_policy.fairness_factor,
                'urgency_threshold': self.bid_policy.urgency_threshold,
                'proximity_bonus_weight': self.bid_policy.proximity_bonus_weight,
                'speed_diff_modifier': self.bid_policy.speed_diff_modifier,
                'follow_distance_modifier': self.bid_policy.follow_distance_modifier,
                'ignore_vehicles_go': self.bid_policy.ignore_vehicles_go,
                'ignore_vehicles_wait': self.bid_policy.ignore_vehicles_wait,
                'ignore_vehicles_platoon_leader': self.bid_policy.ignore_vehicles_platoon_leader,
                'ignore_vehicles_platoon_follower': self.bid_policy.ignore_vehicles_platoon_follower
            })
            
            # Add performance info
            step_time = time.time() - step_start
            self.perf_stats['step_times'].append(step_time)
            if len(self.perf_stats['step_times']) > 0:
                info['perf'] = {
                    'avg_step_time': np.mean(self.perf_stats['step_times'][-100:]),
                    'current_step_time': step_time,
                    'total_ticks': self.perf_stats['total_ticks']
                }
            
            return obs, total_reward, done, info
            
        except Exception as e:
            print(f"❌ Step with all params failed: {e}")
            import traceback
            traceback.print_exc()
            obs = np.zeros(self.observation_dim())
            return obs, -100.0, True, {'error': str(e)}

    def _get_observation_cached(self) -> np.ndarray:
        """Get observation with caching for performance"""
        # Use cached observation if recent enough
        if (self.last_observation is not None and 
            self.current_step - self.last_obs_step < self.observation_cache_steps):
            return self.last_observation.copy()
        
        # Generate new observation
        obs = self._get_observation_optimized()
        self.last_observation = obs.copy()
        self.last_obs_step = self.current_step
        return obs

    def _get_observation_optimized(self) -> np.ndarray:
        """Optimized observation collection with pre-allocated arrays"""
        try:
            # Reset pre-allocated arrays
            self._obs_array.fill(0.0)
            self._vehicle_features.fill(0.0)
            self._platoon_features.fill(0.0)
            
            # Get vehicle states once
            vehicle_states = self.state_extractor.get_vehicle_states()
            control_stats = self.traffic_controller.get_control_stats()
            auction_stats = self.auction_engine.get_auction_stats()
            
            # Base metrics (10 features) - direct assignment
            vehicles_in_junction = sum(1 for v in vehicle_states if v.get('is_junction', False))
            
            self._obs_array[0] = len(vehicle_states)
            self._obs_array[1] = vehicles_in_junction
            self._obs_array[2] = control_stats.get('total_controlled', 0)
            self._obs_array[3] = control_stats.get('go_vehicles', 0)
            self._obs_array[4] = control_stats.get('waiting_vehicles', 0)
            self._obs_array[5] = auction_stats.get('current_agents', 0)
            self._obs_array[6] = auction_stats.get('current_go_count', 0)
            self._obs_array[7] = self.metrics['throughput']
            self._obs_array[8] = self.metrics['avg_acceleration']
            self._obs_array[9] = self.metrics['collision_count']
            
            # Vehicle states (optimized loop)
            active_controls = control_stats.get('active_controls', [])
            for i, vehicle_state in enumerate(vehicle_states[:20]):
                base_idx = i * 8
                
                # Extract location efficiently
                loc = vehicle_state['location']
                if isinstance(loc, dict):
                    self._vehicle_features[base_idx] = loc.get('x', 0.0)
                    self._vehicle_features[base_idx + 1] = loc.get('y', 0.0)
                elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    self._vehicle_features[base_idx] = float(loc[0])
                    self._vehicle_features[base_idx + 1] = float(loc[1])
                
                # Extract velocity efficiently
                vel = vehicle_state.get('velocity', {})
                if isinstance(vel, dict):
                    speed_sq = vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2
                    self._vehicle_features[base_idx + 2] = np.sqrt(speed_sq)
                elif isinstance(vel, (list, tuple)) and len(vel) >= 3:
                    speed_sq = vel[0]**2 + vel[1]**2 + vel[2]**2
                    self._vehicle_features[base_idx + 2] = np.sqrt(speed_sq)
                
                # Other features
                self._vehicle_features[base_idx + 3] = vehicle_state.get('eta_to_intersection', 0)
                self._vehicle_features[base_idx + 4] = float(vehicle_state.get('is_junction', False))
                self._vehicle_features[base_idx + 5] = float(str(vehicle_state['id']) in active_controls)
                self._vehicle_features[base_idx + 6] = vehicle_state.get('lane_id', 0)
                # base_idx + 7 is reserved (already 0)
            
            # Copy vehicle features to main array
            self._obs_array[10:170] = self._vehicle_features
            
            # Platoon info (optimized)
            platoons = self.platoon_manager.get_all_platoons()
            for i, platoon in enumerate(platoons[:5]):
                base_idx = i * 4
                self._platoon_features[base_idx] = platoon.get_size()
                self._platoon_features[base_idx + 1] = float(platoon.get_leader_id())
                self._platoon_features[base_idx + 2] = len(platoon.get_follower_ids())
                # base_idx + 3 is reserved (already 0)
            
            # Copy platoon features to main array
            self._obs_array[170:190] = self._platoon_features
            
            # Auction info (5 features)
            priority_order = self.auction_engine.get_current_priority_order()
            self._obs_array[190] = len(priority_order)
            self._obs_array[191] = self.bid_policy.get_current_bid_scale()
            self._obs_array[192] = auction_stats.get('platoon_agents', 0)
            self._obs_array[193] = auction_stats.get('vehicle_agents', 0)
            # obs_array[194] is reserved (already 0)
            
            return self._obs_array.copy()
            
        except Exception as e:
            print(f"❌ Failed to get optimized observation: {e}")
            return np.zeros(self.observation_dim(), dtype=np.float32)

    def _calculate_reward(self) -> float:
        """Calculate step reward - optimized version"""
        reward = 0.0
        
        try:
            # Get current statistics (cached if possible)
            control_stats = self.traffic_controller.get_control_stats()
            
            # Throughput reward (vehicles exiting intersection)
            vehicles_exited = control_stats.get('vehicles_exited_intersection', 0)
            prev_exited = self.metrics.get('prev_vehicles_exited', 0)
            new_exits = vehicles_exited - prev_exited
            reward += new_exits * 10.0
            self.metrics['prev_vehicles_exited'] = vehicles_exited
            
            # Update throughput less frequently for performance
            if self.current_step % 10 == 0:  # Only every 10 steps
                sim_time = self.current_step * 0.1
                self.metrics['throughput'] = vehicles_exited / max(sim_time, 1.0) * 3600
                reward += self.metrics['throughput'] * 0.01
            
            # Simplified acceleration calculation
            if self.current_step % 20 == 0:  # Only every 20 steps
                final_stats = self.traffic_controller.get_final_statistics()
                avg_abs_accel = final_stats.get('average_absolute_acceleration', 0.0)
                self.metrics['avg_acceleration'] = avg_abs_accel
                if avg_abs_accel > 3.0:
                    reward -= (avg_abs_accel - 3.0) * 2.0
            
            # Efficiency reward (simplified)
            vehicle_count = len(self.state_extractor.get_vehicle_states()) if self.current_step % 5 == 0 else getattr(self, '_cached_vehicle_count', 1)
            if self.current_step % 5 == 0:
                self._cached_vehicle_count = vehicle_count
            
            controlled_ratio = control_stats.get('total_controlled', 0) / max(vehicle_count, 1)
            reward += controlled_ratio * 5.0
            
            # Collision and deadlock checks (simplified)
            if self._check_collision():
                reward -= 100.0
                self.metrics['collision_count'] += 1
            
            if self._check_deadlock():
                reward -= 50.0
            
            # Small step penalty
            reward -= 0.1
            
        except Exception as e:
            print(f"❌ Reward calculation failed: {e}")
            reward = -10.0
        
        return reward

    def _check_collision(self) -> bool:
        """Check for collisions"""
        try:
            # This would need to be implemented based on your collision detection
            # For now, return False
            return False
        except:
            return False

    def _check_deadlock(self) -> bool:
        """Check for deadlock conditions"""
        try:
            # Simple deadlock detection: too many vehicles waiting for too long
            control_stats = self.traffic_controller.get_control_stats()
            waiting_vehicles = control_stats.get('waiting_vehicles', 0)
            total_controlled = control_stats.get('total_controlled', 0)
            
            if total_controlled > 10 and waiting_vehicles / max(total_controlled, 1) > 0.8:
                return True
            return False
        except:
            return False

    def reset_metrics(self):
        """Reset metrics tracking"""
        self.metrics = {
            'throughput': 0.0,
            'avg_acceleration': 0.0,
            'collision_count': 0,
            'prev_vehicles_exited': 0
        }

    def _get_info(self) -> Dict:
        """Get additional information"""
        try:
            control_stats = self.traffic_controller.get_control_stats()
            auction_stats = self.auction_engine.get_auction_stats()
            
            return {
                'step': self.current_step,
                'throughput': self.metrics['throughput'],
                'avg_acceleration': self.metrics['avg_acceleration'],
                'collision_count': self.metrics['collision_count'],
                'total_controlled': control_stats.get('total_controlled', 0),
                'vehicles_exited': control_stats.get('vehicles_exited_intersection', 0),
                'auction_agents': auction_stats.get('current_agents', 0),
                'bid_scale': self.bid_policy.get_current_bid_scale()
            }
        except Exception as e:
            return {'error': str(e)}

    def close(self):
        """Clean up simulation"""
        try:
            if hasattr(self, 'scenario'):
                self.scenario.stop_time_counters()
            print("🏁 Simulation environment closed")
        except Exception as e:
            print(f"❌ Error closing simulation: {e}")

    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        if not self.perf_stats['step_times']:
            return {}
        
        return {
            'avg_step_time': np.mean(self.perf_stats['step_times']),
            'min_step_time': np.min(self.perf_stats['step_times']),
            'max_step_time': np.max(self.perf_stats['step_times']),
            'avg_obs_time': np.mean(self.perf_stats['obs_times']) if self.perf_stats['obs_times'] else 0,
            'avg_reward_time': np.mean(self.perf_stats['reward_times']) if self.perf_stats['reward_times'] else 0,
            'total_ticks': self.perf_stats['total_ticks'],
            'steps_per_action': self.steps_per_action,
            'observation_cache_steps': self.observation_cache_steps
        }

    def set_performance_mode(self, fast_mode: bool = True):
        """Adjust performance settings"""
        if fast_mode:
            self.steps_per_action = 2
            self.observation_cache_steps = 10
            print("🚀 Fast mode enabled: 2 steps per action, 10-step observation cache")
        else:
            self.steps_per_action = 5
            self.observation_cache_steps = 5
            print("🐌 Normal mode enabled: 5 steps per action, 5-step observation cache")
    
    def _get_observation(self) -> np.ndarray:
        """Backward compatibility method - delegates to optimized version"""
        return self._get_observation_optimized()
