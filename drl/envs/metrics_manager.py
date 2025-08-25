import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import os
import atexit

class SimulationMetricsManager:
    """Dedicated manager for simulation metrics tracking and validation"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Core metrics with enhanced deadlock tracking
        self.metrics = {
            'avg_acceleration': 0.0,
            'collision_count': 0,
            'prev_vehicles_exited': 0,
            'prev_collision_count': 0,
            'prev_deadlock_count': 0,
            'current_deadlock_severity': 0.0,
            'deadlock_threat_level': 'none',
            'max_severity_seen': 0.0,
            'severity_warnings': 0,
            'using_real_data': True,
            'throughput': 0.0,
            'last_throughput_calc_step': -1
        }
        
        # Performance tracking with memory management
        self.perf_stats = {
            'step_times': deque(maxlen=200),
            'obs_times': deque(maxlen=200),
            'reward_times': deque(maxlen=200),
            'total_ticks': 0
        }
        
        # Reward validation
        self.reward_history = deque(maxlen=100)
        self.suspicious_rewards = 0
        
        # File handling optimization - prevent "Too many open files" error
        self._csv_file_handle = None
        self._csv_file_path = None
        self._csv_buffer = []
        self._max_buffer_size = 50  # Buffer size before writing to disk
        self._last_write_time = 0
        self._write_interval = 5.0  # Write every 5 seconds at most
        
        # Register cleanup on exit
        atexit.register(self._cleanup_resources)
        
        print("📊 Metrics Manager initialized with memory-bounded tracking and optimized file handling")

    def _cleanup_resources(self):
        """Clean up file handles and resources"""
        try:
            if self._csv_file_handle is not None:
                self._csv_file_handle.close()
                self._csv_file_handle = None
                print("🧹 Metrics Manager file handles cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    def _safe_write_csv(self, data: List[Dict], file_path: str):
        """Safely write CSV data with proper file handle management"""
        try:
            # Use pandas for efficient CSV writing
            import pandas as pd
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to DataFrame and write efficiently
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, mode='w', float_format='%.6f')
            
            print(f"📊 Saved {len(data)} metrics records to {file_path}")
            
        except Exception as e:
            print(f"⚠️ CSV write failed: {str(e)}")
            # Fallback: try simple text writing
            try:
                with open(file_path, 'w') as f:
                    if data:
                        # Write header
                        headers = list(data[0].keys())
                        f.write(','.join(headers) + '\n')
                        
                        # Write data
                        for row in data:
                            values = [str(row.get(h, '')) for h in headers]
                            f.write(','.join(values) + '\n')
                        
                        print(f"📊 Fallback CSV write successful: {len(data)} records")
            except Exception as fallback_e:
                print(f"❌ Both CSV write methods failed: {fallback_e}")

    def _write_buffered_metrics(self):
        """Write buffered metrics to disk with rate limiting"""
        current_time = time.time()
        
        # Only write if buffer is full or enough time has passed
        if (len(self._csv_buffer) >= self._max_buffer_size or 
            current_time - self._last_write_time >= self._write_interval):
            
            if self._csv_buffer:
                # Write to disk
                self._safe_write_csv(self._csv_buffer, self._csv_file_path)
                
                # Clear buffer and update timing
                self._csv_buffer.clear()
                self._last_write_time = current_time

    def reset_metrics(self, nash_solver=None, traffic_controller=None):
        """Reset all metrics for new episode with proper baseline sync"""
        # FIXED: Reset deadlock baseline to ZERO for new episode
        # Each episode should start fresh without carrying over deadlock counts
        print("🔄 Resetting deadlock baseline to 0 for fresh episode start")
        
        # CORRECTED: Use cumulative exit count as baseline for new episode
        initial_vehicles_exited = 0
        if traffic_controller:
            try:
                # Get current cumulative exit count as new episode baseline
                final_stats = traffic_controller.get_final_statistics()
                initial_vehicles_exited = final_stats.get('vehicles_exited_intersection', 0)
                print(f"✅ Setting episode baseline from cumulative exits: {initial_vehicles_exited}")
            except Exception as e:
                print(f"⚠️ Could not get cumulative baseline: {e}")
                initial_vehicles_exited = 0
        
        # IMPORTANT: Store Nash solver reference to get fresh deadlock count when needed
        self._nash_solver_ref = nash_solver
        
        self.metrics = {
            'avg_acceleration': 0.0,
            'collision_count': 0,
            'prev_vehicles_exited': initial_vehicles_exited,  # Initialize with current baseline
            'prev_collision_count': 0,
            'prev_deadlock_count': 0,  # FIXED: Always start at 0 for new episode
            'episode_deadlock_baseline': None,  # Will be set on first deadlock check
            'current_deadlock_severity': 0.0,
            'deadlock_threat_level': 'none',
            'max_severity_seen': 0.0,
            'severity_warnings': 0,
            'using_real_data': True,
            'throughput': 0.0,
            'last_throughput_calc_step': -1
        }
        
        # Clear performance stats but keep structure
        for key in self.perf_stats:
            if key != 'total_ticks':
                self.perf_stats[key].clear()
        
        self.reward_history.clear()
        self.suspicious_rewards = 0
        
        # Clear CSV buffer for new episode
        self._csv_buffer.clear()
        
        print("🔄 Metrics reset with proper baseline sync")

    def calculate_reward(self, traffic_controller, state_extractor, scenario, 
                        nash_solver, current_step: int, actions_since_reset: int = 0) -> float:
        """Calculate STEP-WISE reward (NOT cumulative) with proper episode isolation"""
        # IMPORTANT: This function calculates reward for THIS STEP only
        # Reward should NOT carry over between episodes - each step is independent
        reward = 0.0
        
        # DEBUG: Extra logging for first few actions after reset
        debug_early_actions = actions_since_reset <= 5
        
        try:
            # Get REAL statistics
            control_stats = traffic_controller.get_control_stats()
            final_stats = traffic_controller.get_final_statistics()
            
            # VALIDATED exit reward calculation
            current_exited = final_stats.get('vehicles_exited_intersection', 0)
            prev_exited = self.metrics.get('prev_vehicles_exited', 0)
            new_exits = max(0, current_exited - prev_exited)
            
            # Cross-validation with current vehicle count
            current_vehicles = state_extractor.get_vehicle_states()
            current_vehicle_count = len(current_vehicles)
            
            # CRITICAL: Check if TrafficController is in post-reset grace period
            if (hasattr(traffic_controller, '_just_reset') and 
                traffic_controller._just_reset and 
                traffic_controller._reset_update_count <= 3):
                print(f"🔄 GRACE PERIOD: Skipping exit rewards during reset stabilization")
                new_exits = 0  # Force to 0 during grace period
            elif new_exits > 10:  # More than 10 vehicles exiting in one step is suspicious
                print(f"⚠️ SUSPICIOUS: {new_exits} vehicles exit in one step (current: {current_exited}, prev: {prev_exited})")
                new_exits = min(new_exits, 3)  # Cap at 3 to prevent false rewards
            
            if new_exits > 0:
                # ENHANCED exit reward with efficiency bonus
                base_exit_reward = new_exits * 8.0  # Increased base reward
                
                # Efficiency bonus based on traffic throughput
                current_vehicle_count = len(current_vehicles)
                if current_vehicle_count > 5:  # High traffic scenario
                    throughput_bonus = min(new_exits * 2.0, 10.0)  # Cap bonus
                    exit_reward = base_exit_reward + throughput_bonus
                    print(f"✅ +{exit_reward:.1f} for {new_exits} exits (base: {base_exit_reward:.1f}, throughput bonus: {throughput_bonus:.1f})")
                else:
                    exit_reward = base_exit_reward
                    print(f"✅ +{exit_reward:.1f} for {new_exits} verified exits")
                
                reward += exit_reward
                self.metrics['prev_vehicles_exited'] = current_exited
            else:
                # Update baseline even if no new exits
                self.metrics['prev_vehicles_exited'] = current_exited
                if debug_early_actions and actions_since_reset <= 2:
                    print(f"🔍 DEBUG (action {actions_since_reset}): No new exits (current: {current_exited}, prev: {prev_exited})")
            
            # ENHANCED: Smoother acceleration reward/penalty
            real_avg_accel = final_stats.get('average_absolute_acceleration', 0.0)
            self.metrics['avg_acceleration'] = real_avg_accel
            
            # Graduated acceleration penalty/reward system
            if real_avg_accel < 1.0:  # Very smooth traffic
                reward += 2.0
            elif real_avg_accel < 1.5:  # Smooth traffic
                reward += 1.0
            elif real_avg_accel > 3.0:  # Aggressive driving
                reward -= (real_avg_accel - 3.0) * 2.0
            elif real_avg_accel > 2.0:  # Moderate penalty
                reward -= (real_avg_accel - 2.0) * 1.0
            
            # ENHANCED control effectiveness with activity bonus
            if current_vehicles:
                controlled_vehicles = control_stats.get('total_controlled', 0)
                control_ratio = controlled_vehicles / len(current_vehicles)
                
                # Base control effectiveness reward
                control_reward = control_ratio * 2.0
                
                # Activity bonus for managing more vehicles
                if len(current_vehicles) > 8:
                    activity_bonus = min(1.0, (len(current_vehicles) - 8) * 0.1)
                    control_reward += activity_bonus
                    
                reward += control_reward
            
            # ENHANCED collision penalty with context
            if hasattr(scenario, 'traffic_generator') and hasattr(scenario.traffic_generator, 'collision_count'):
                current_collisions = scenario.traffic_generator.collision_count
                prev_collisions = self.metrics.get('prev_collision_count', 0)
                new_collisions = current_collisions - prev_collisions
                
                if new_collisions > 0:
                    # Base collision penalty
                    base_collision_penalty = new_collisions * 25.0  # Increased from 20.0
                    
                    # Context multiplier - higher penalty in high traffic
                    traffic_multiplier = 1.0
                    if current_vehicle_count > 10:
                        traffic_multiplier = 1.3  # 30% higher penalty in dense traffic
                    elif current_vehicle_count > 6:
                        traffic_multiplier = 1.15  # 15% higher penalty in moderate traffic
                    
                    total_collision_penalty = base_collision_penalty * traffic_multiplier
                    reward -= total_collision_penalty
                    self.metrics['prev_collision_count'] = current_collisions
                    
                    print(f"💥 COLLISION PENALTY: -{total_collision_penalty:.1f} ({new_collisions} collisions, traffic: {current_vehicle_count})")
            
            # Enhanced deadlock penalty system with severity-based punishment
            # FIXED: Skip deadlock penalties during grace period after reset
            if actions_since_reset > 10:  # Grace period of 10 actions
                deadlock_penalty = self._calculate_deadlock_penalty(nash_solver)
                reward += deadlock_penalty  # deadlock_penalty is negative
                
                # Near-deadlock warning system - penalize approaching deadlock situations
                severity_penalty = self._calculate_severity_penalty(nash_solver)
                reward += severity_penalty  # severity_penalty is negative
            else:
                # During grace period, no deadlock penalties
                if actions_since_reset == 1:  # Only log once at start
                    print(f"🕐 Grace period: Skipping deadlock penalties for first 10 actions")
            
            # IMPROVED step penalty and bonuses
            if current_vehicle_count < 3:  # Low traffic - smaller penalty
                reward -= 0.005  
            else:
                reward -= 0.02  # Higher penalty for inactive periods in high traffic
            
            # Enhanced activity bonus
            if current_vehicle_count > 2 and control_stats.get('total_controlled', 0) > 0:
                base_activity = 0.2
                # Scale with vehicle density
                density_bonus = min(0.3, current_vehicle_count * 0.02)
                reward += base_activity + density_bonus
            
            # Bound reward to realistic range
            reward = np.clip(reward, -50.0, 50.0)
            
            # VALIDATION: Check for suspicious reward patterns
            self.reward_history.append(reward)
            if len(self.reward_history) > 10:
                recent_rewards = list(self.reward_history)[-10:]
                if all(r > 20 for r in recent_rewards):
                    print(f"⚠️ SUSPICIOUS: Consistently high rewards in last 10 steps (avg: {np.mean(recent_rewards):.1f})")
                    self.suspicious_rewards += 1
                elif all(r < -20 for r in recent_rewards):
                    print(f"⚠️ SUSPICIOUS: Consistently low rewards in last 10 steps (avg: {np.mean(recent_rewards):.1f})")
                    self.suspicious_rewards += 1
            
            # DEBUG logging for early actions
            if debug_early_actions:
                print(f"🔍 DEBUG (action {actions_since_reset}): Step reward = {reward:.2f}")
            
            return reward
            
        except Exception as e:
            print(f"❌ Reward calculation failed: {str(e)}")
            return -1.0

    def calculate_throughput(self, scenario, current_step: int, 
                           vehicles_exited: int) -> float:
        """Calculate throughput with caching and validation"""
        try:
            recalc_interval = 100  # OPTIMIZED: Less frequent throughput calculation
            should_recalc = (current_step % recalc_interval == 0) or (
                self.metrics.get('last_throughput_calc_step', -1) < 0)
            
            if should_recalc:
                sim_elapsed = (scenario.get_sim_elapsed() 
                             if hasattr(scenario, 'get_sim_elapsed') else None)
                
                if sim_elapsed is not None and sim_elapsed > 0.1 and vehicles_exited >= 0:
                    computed = (vehicles_exited / sim_elapsed) * 3600.0
                    real_throughput = max(0.0, min(float(computed), 3600.0))
                    
                    self.metrics['throughput'] = real_throughput
                    self.metrics['last_throughput_calc_step'] = current_step
                    
                    return real_throughput
            
            # Return cached value
            return float(self.metrics.get('throughput', 0.0))
            
        except Exception:
            return float(self.metrics.get('throughput', 0.0))

    def get_info_dict(self, traffic_controller, auction_engine, nash_solver,
                     scenario, state_extractor, bid_policy, current_step: int,
                     max_steps: int) -> Dict:
        """Generate comprehensive info dictionary with validation"""
        try:
            # Get real statistics
            control_stats = traffic_controller.get_control_stats()
            final_stats = traffic_controller.get_final_statistics()
            auction_stats = auction_engine.get_auction_stats()
            current_vehicles = state_extractor.get_vehicle_states()
            
            # Calculate throughput
            vehicles_exited = final_stats['vehicles_exited_intersection']
            real_throughput = self.calculate_throughput(scenario, current_step, vehicles_exited)
            
            # Basic validation
            sim_elapsed = (scenario.get_sim_elapsed() 
                         if hasattr(scenario, 'get_sim_elapsed') else None)
            data_valid = sim_elapsed is not None and vehicles_exited >= 0
            
            # Get collision and deadlock data
            collision_count = 0
            if (hasattr(scenario, 'traffic_generator') and 
                hasattr(scenario.traffic_generator, 'collision_count')):
                collision_count = scenario.traffic_generator.collision_count
            
            deadlocks_detected = 0
            if (hasattr(nash_solver, 'deadlock_detector') and 
                hasattr(nash_solver.deadlock_detector, 'stats')):
                deadlocks_detected = nash_solver.deadlock_detector.stats.get('deadlocks_detected', 0)
            elif hasattr(nash_solver, 'stats'):
                # Fallback to direct nash_solver stats if available
                deadlocks_detected = nash_solver.stats.get('deadlocks_detected', 0)
            
            return {
                # Core simulation metrics
                'throughput': float(real_throughput),
                'avg_acceleration': float(final_stats.get('average_absolute_acceleration', 0.0)),
                'collision_count': int(collision_count),
                'total_controlled': int(final_stats['total_vehicles_controlled']),
                'vehicles_exited': int(vehicles_exited),
                'auction_agents': int(auction_stats.get('current_agents', 0)),
                'deadlocks_detected': int(deadlocks_detected),
                
                # Enhanced deadlock severity metrics
                'deadlock_severity': float(self.metrics.get('current_deadlock_severity', 0.0)),
                'deadlock_threat_level': str(self.metrics.get('deadlock_threat_level', 'none')),
                'max_severity_seen': float(self.metrics.get('max_severity_seen', 0.0)),
                'severity_warnings': int(self.metrics.get('severity_warnings', 0)),
                
                # Real-time state
                'vehicles_detected': len(current_vehicles),
                'vehicles_in_junction': sum(1 for v in current_vehicles 
                                          if v.get('is_junction', False)),
                'go_vehicles': int(control_stats.get('go_vehicles', 0)),
                'waiting_vehicles': int(control_stats.get('waiting_vehicles', 0)),
                
                # Training parameters
                'bid_scale': float(bid_policy.bid_scale),
                'eta_weight': float(bid_policy.eta_weight),
                'speed_weight': float(bid_policy.speed_weight),
                'congestion_sensitivity': float(bid_policy.congestion_sensitivity),
                'platoon_bonus': float(bid_policy.platoon_bonus),
                'junction_penalty': float(bid_policy.junction_penalty),
                'fairness_factor': float(bid_policy.fairness_factor),
                'urgency_threshold': float(bid_policy.urgency_threshold),
                'proximity_bonus_weight': float(bid_policy.proximity_bonus_weight),
                'speed_diff_modifier': float(bid_policy.speed_diff_modifier),
                'follow_distance_modifier': float(bid_policy.follow_distance_modifier),
                'ignore_vehicles_go': float(bid_policy.ignore_vehicles_go),
                'ignore_vehicles_wait': float(bid_policy.ignore_vehicles_wait),
                'ignore_vehicles_platoon_leader': float(bid_policy.ignore_vehicles_platoon_leader),
                'ignore_vehicles_platoon_follower': float(bid_policy.ignore_vehicles_platoon_follower),
                
                # Metadata
                'sim_time_elapsed': float(sim_elapsed) if sim_elapsed is not None else 0.0,
                'current_step': int(current_step),
                'max_steps': int(max_steps),
                'using_real_data': True,
                'data_source': 'actual_carla_simulation',
                'data_validated': data_valid,
                
                # Quality metrics
                'reward_validation': {
                    'suspicious_rewards': self.suspicious_rewards,
                    'avg_recent_reward': np.mean(list(self.reward_history)) if self.reward_history else 0.0,
                    'reward_stability': np.std(list(self.reward_history)) if len(self.reward_history) > 1 else 0.0
                }
            }
            
        except Exception as e:
            print(f"❌ Failed to get info: {str(e)}")
            return {
                'using_real_data': False,
                'data_source': 'error_fallback',
                'error': str(e)
            }

    def record_performance(self, step_time: float, obs_time: float = None, 
                          reward_time: float = None):
        """Record performance metrics"""
        self.perf_stats['step_times'].append(step_time)
        if obs_time is not None:
            self.perf_stats['obs_times'].append(obs_time)
        if reward_time is not None:
            self.perf_stats['reward_times'].append(reward_time)
        self.perf_stats['total_ticks'] += 1

    def _calculate_deadlock_penalty(self, nash_solver) -> float:
        """Calculate penalty based on actual deadlocks detected with severity - FIXED EPISODE ISOLATION"""
        penalty = 0.0
        
        try:
            if (hasattr(nash_solver, 'deadlock_detector') and 
                hasattr(nash_solver.deadlock_detector, 'stats')):
                current_deadlocks = nash_solver.deadlock_detector.stats.get('deadlocks_detected', 0)
                
                # FIXED: Establish episode baseline on first check
                if self.metrics.get('episode_deadlock_baseline') is None:
                    self.metrics['episode_deadlock_baseline'] = current_deadlocks
                    self.metrics['prev_deadlock_count'] = 0  # Episode starts at 0
                    print(f"🔄 Established deadlock baseline for episode: {current_deadlocks}")
                    return 0.0  # No penalty on baseline establishment
                
                # Calculate deadlocks WITHIN THIS EPISODE only
                episode_baseline = self.metrics.get('episode_deadlock_baseline', 0)
                episode_deadlocks = current_deadlocks - episode_baseline
                prev_episode_deadlocks = self.metrics.get('prev_deadlock_count', 0)
                new_deadlocks = episode_deadlocks - prev_episode_deadlocks
                
                if new_deadlocks > 0:
                    # Get severity for graduated penalty
                    severity = nash_solver.deadlock_detector.get_deadlock_severity()
                    
                    # IMPROVED deadlock penalty system
                    # Base penalty scales with occurrence frequency
                    base_penalty = new_deadlocks * 30.0  # Reduced from 50.0
                    
                    # Smart severity multiplier (1.0 to 2.5 based on severity)
                    severity_multiplier = 1.0 + (severity * 1.5)
                    
                    # Context-aware penalty adjustment
                    affected_vehicles = nash_solver.deadlock_detector.stats.get('total_affected_vehicles', 0)
                    if affected_vehicles <= 3:  # Minor deadlock
                        vehicle_factor = 1.0
                    elif affected_vehicles <= 6:  # Moderate deadlock
                        vehicle_factor = 1.3
                    else:  # Major deadlock
                        vehicle_factor = 1.8
                    
                    total_penalty = base_penalty * severity_multiplier * vehicle_factor
                    penalty = -min(total_penalty, 300.0)  # Reduced cap from 500.0
                    
                    # Update EPISODE deadlock count (not absolute count)
                    self.metrics['prev_deadlock_count'] = episode_deadlocks
                    print(f"🚨 BALANCED deadlock penalty: {penalty:.1f} (deadlocks: {new_deadlocks}, severity: {severity:.2f}, affected: {affected_vehicles})")
                    
        except Exception as e:
            print(f"⚠️ Deadlock penalty calculation error: {str(e)}")
            
        return penalty
    
    def _calculate_severity_penalty(self, nash_solver) -> float:
        """Calculate penalty for approaching deadlock situations (severity-based early warning)"""
        penalty = 0.0
        
        try:
            if (hasattr(nash_solver, 'deadlock_detector') and 
                hasattr(nash_solver.deadlock_detector, 'get_deadlock_severity')):
                current_severity = nash_solver.deadlock_detector.get_deadlock_severity()
                
                # IMPROVED graduated penalties with smoother progression
                if current_severity > 0.15:  # Raised threshold to 15%
                    if current_severity >= 0.9:  # Critical: 90%+ stalled (stricter)
                        penalty = -15.0  # Reduced from -20.0
                        self.metrics['deadlock_threat_level'] = 'critical'
                    elif current_severity >= 0.7:  # High: 70-89% stalled
                        penalty = -8.0  # Reduced from -10.0
                        self.metrics['deadlock_threat_level'] = 'high'
                    elif current_severity >= 0.5:  # Medium: 50-69% stalled
                        penalty = -4.0  # Reduced from -5.0
                        self.metrics['deadlock_threat_level'] = 'medium'
                    elif current_severity >= 0.3:  # Low: 30-49% stalled
                        penalty = -1.5  # Reduced from -2.0
                        self.metrics['deadlock_threat_level'] = 'low'
                    else:  # Very low: 15-29% stalled
                        penalty = -0.3  # Reduced from -0.5
                        self.metrics['deadlock_threat_level'] = 'very_low'
                        
                    # Store current severity for monitoring
                    self.metrics['current_deadlock_severity'] = current_severity
                    
                    # Track maximum severity seen
                    if current_severity > self.metrics.get('max_severity_seen', 0.0):
                        self.metrics['max_severity_seen'] = current_severity
                    
                    # Count severity warnings
                    if penalty < -1.0:
                        self.metrics['severity_warnings'] = self.metrics.get('severity_warnings', 0) + 1
                        print(f"⚠️ Near-deadlock penalty: {penalty:.1f} (severity: {current_severity:.2f}, level: {self.metrics['deadlock_threat_level']})")
                else:
                    self.metrics['deadlock_threat_level'] = 'none'
                    self.metrics['current_deadlock_severity'] = 0.0
                    
        except Exception as e:
            print(f"⚠️ Severity penalty calculation error: {str(e)}")
            
        return penalty
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.perf_stats['step_times']:
            return {}
        
        return {
            'avg_step_time': np.mean(list(self.perf_stats['step_times'])),
            'min_step_time': np.min(list(self.perf_stats['step_times'])),
            'max_step_time': np.max(list(self.perf_stats['step_times'])),
            'avg_obs_time': np.mean(list(self.perf_stats['obs_times'])) if self.perf_stats['obs_times'] else 0,
            'total_ticks': self.perf_stats['total_ticks'],
            'memory_usage': {
                'step_times_len': len(self.perf_stats['step_times']),
                'obs_times_len': len(self.perf_stats['obs_times']),
                'reward_history_len': len(self.reward_history)
            }
        }

    def close(self):
        """Clean up resources"""
        self._cleanup_resources()
