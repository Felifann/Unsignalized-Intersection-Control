#!/usr/bin/env python3
"""
Enhanced DRL training script with comprehensive metrics collection and analysis
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import json
import resource
import gc
from typing import List, Dict
from datetime import datetime

# --- Prefer gymnasium if available, and make it available as 'gym' for legacy imports ---
try:
    import gymnasium as gym  # type: ignore
    print("✅ Using gymnasium")
    sys.modules['gym'] = gym
except Exception:
    try:
        import gym  # type: ignore
        print("✅ Using legacy gym")
    except Exception:
        print("❌ No gym or gymnasium found")
        sys.exit(1)

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# Ensure CARLA Python egg is on sys.path
egg_candidates = []
egg_candidates += glob.glob(os.path.join(base_dir, "carla_l", "carla-*.egg"))
egg_candidates += glob.glob(os.path.join(base_dir, "carla_w", "carla-*.egg"))
egg_candidates += glob.glob(os.path.join(base_dir, "carla", "carla-*.egg"))
if egg_candidates:
    egg_path = egg_candidates[0]
    if egg_path not in sys.path:
        sys.path.insert(0, egg_path)
else:
    print("Warning: CARLA egg not found. Ensure CARLA PythonAPI egg is available.")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from drl.envs.auction_gym import AuctionGymEnv
from drl.utils.analysis import TrainingAnalyzer

class SimpleMetricsCallback(BaseCallback):
    """Enhanced callback to log training metrics with action space parameter tracking and per-episode statistics"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode-level tracking
        self.episode_actions = []  # Store actions for current episode
        self.episode_metrics = []  # Store metrics for current episode
        self.episode_count = 0
        self.episode_start_step = 0
        
        # CSV file paths
        self.step_metrics_path = os.path.join(log_dir, 'step_metrics.csv')
        self.episode_metrics_path = os.path.join(log_dir, 'episode_metrics.csv')
        
        # File handle management
        self._last_write_timestamp = 0
        self._write_interval = 10.0  # Write every 10 seconds at most
        
        # Register cleanup function
        import atexit
        atexit.register(self._cleanup_resources)
        
        print(f"📊 Enhanced Metrics Callback initialized:")
        print(f"   Step metrics: {self.step_metrics_path}")
        print(f"   Episode metrics: {self.episode_metrics_path}")

    def _on_step(self) -> bool:
        """Log metrics every step and track episode boundaries"""
        try:
            # Get current info and actions
            infos = self.locals.get('infos', [{}])
            actions = self.locals.get('actions', [])
            
            # FIXED: Use proper length checks instead of boolean checks for arrays
            if len(infos) == 0 or len(actions) == 0:
                return True
            
            info = infos[0] if isinstance(infos[0], dict) else {}
            action = actions[0] if isinstance(actions[0], (np.ndarray, list)) else []
            
            # Check if episode ended (reset occurred)
            if self._is_episode_reset():
                self._finalize_episode()
                self._start_new_episode()
            
            # Store action for current episode
            if isinstance(action, (np.ndarray, list)) and len(action) == 4:
                self.episode_actions.append(action)
            
            # Store step metrics
            step_metrics = {
                'timestep': self.num_timesteps,
                'episode': self.episode_count,
                'throughput': info.get('throughput', 0.0),
                'avg_acceleration': info.get('avg_acceleration', 0.0),
                'collision_count': info.get('collision_count', 0),
                'total_controlled': info.get('total_controlled', 0),
                'vehicles_exited': info.get('vehicles_exited', 0),
                'urgency_position_ratio': info.get('urgency_position_ratio', 1.0),
                'speed_diff_modifier': info.get('speed_diff_modifier', 0.0),
                'max_participants_per_auction': info.get('max_participants_per_auction', 4),
                'ignore_vehicles_go': info.get('ignore_vehicles_go', 50.0),
                'deadlocks_detected': info.get('deadlocks_detected', 0),
                'deadlock_severity': info.get('deadlock_severity', 0.0)
            }
            
            # Add simulation time information if available
            simulation_time_info = info.get('simulation_time', {})
            if simulation_time_info:
                step_metrics.update({
                    'episode_simulation_time': simulation_time_info.get('episode_simulation_time', 0.0),
                    'total_simulation_time': simulation_time_info.get('total_simulation_time', 0.0),
                    'episode_start_time': simulation_time_info.get('episode_start_time', ''),
                    'simulation_start_time': simulation_time_info.get('simulation_start_time', '')
                })
            
            # SAFETY CHECK: Detect suspiciously high collision counts
            collision_count = step_metrics['collision_count']
            if collision_count > 100:
                print(f"🚨 SAFETY CHECK: Suspiciously high collision count in training: {collision_count}")
                print(f"   This suggests collision counter was not properly reset between episodes")
                print(f"   Training may be unstable due to massive negative rewards")
                print(f"   Check if traffic_generator.reset_episode_state() is working properly")
                
                # ENHANCED: Provide more diagnostic information
                if hasattr(self, 'episode_count') and self.episode_count > 0:
                    print(f"   Current episode: {self.episode_count}")
                    print(f"   Episode start step: {self.episode_start_step}")
                    print(f"   Current timestep: {self.num_timesteps}")
                
                # Check if this is a new episode issue
                if collision_count > 1000:
                    print(f"   🚨 CRITICAL: Collision count {collision_count} is extremely high!")
                    print(f"   This episode should be terminated immediately")
                    print(f"   Consider checking environment reset logic")
            
            self.episode_metrics.append(step_metrics)
            
            # Save step metrics every 1000 steps
            if self.num_timesteps % 1000 == 0:
                self._save_step_metrics()
                
        except Exception as e:
            print(f"⚠️ Metrics callback error: {e}")
        
        return True

    def _is_episode_reset(self) -> bool:
        """Check if episode reset occurred by looking for environment reset signals"""
        # ENHANCED: More reliable episode boundary detection
        # Check if we have a significant gap in timesteps or if episode length is reasonable
        if len(self.episode_actions) == 0:
            return False
        
        # Method 1: Check for reasonable episode length (most reliable)
        # Normal episodes should be around 128 steps based on config
        if len(self.episode_actions) > 200:  # Episode too long, likely needs reset
            return True
        
        # Method 2: Check for timestep gaps (backup method)
        if hasattr(self, 'episode_start_step') and self.episode_start_step > 0:
            current_gap = self.num_timesteps - self.episode_start_step
            if current_gap > 1000:  # Large gap suggests episode boundary
                return True
        
        # Method 3: Check for action pattern changes (heuristic)
        if len(self.episode_actions) > 10:
            recent_actions = self.episode_actions[-10:]
            early_actions = self.episode_actions[:10]
            
            # If recent actions are very different from early actions, might be new episode
            recent_mean = np.mean(recent_actions, axis=0)
            early_mean = np.mean(early_actions, axis=0)
            action_change = np.mean(np.abs(recent_mean - early_mean))
            
            if action_change > 2.0:  # Significant change in action patterns
                return True
        
        return False

    def _start_new_episode(self):
        """Start tracking a new episode"""
        self.episode_count += 1
        self.episode_start_step = self.num_timesteps
        self.episode_actions = []
        self.episode_metrics = []
        print(f"🔄 Starting episode {self.episode_count} at step {self.episode_start_step}")

    def _finalize_episode(self):
        """Calculate and save episode-level statistics"""
        if not self.episode_actions or not self.episode_metrics:
            return
        
        try:
            # Convert actions to numpy array for calculations
            actions_array = np.array(self.episode_actions)
            
            # Calculate action space parameter statistics
            action_means = np.mean(actions_array, axis=0)
            action_vars = np.var(actions_array, axis=0)
            action_stds = np.std(actions_array, axis=0)
            
            # Get episode-level metrics
            episode_stats = self._calculate_episode_stats()
            
            # Create episode summary
            episode_summary = {
                'episode': self.episode_count,
                'episode_start_step': self.episode_start_step,
                'episode_end_step': self.num_timesteps,
                'episode_length': len(self.episode_actions),
                
                # TRUE EXACT parameter values (actual values applied in environment)
                'urgency_position_ratio_exact': 0.0,  # Will be filled by environment
                'speed_diff_modifier_exact': 0.0,     # Will be filled by environment
                'max_participants_exact': 4.0,        # Will be filled by environment
                'ignore_vehicles_go_exact': 50.0,     # Will be filled by environment
                
                # Episode performance metrics
                'total_vehicles_exited': episode_stats['total_exits'],
                'total_collisions': episode_stats['total_collisions'],
                'total_deadlocks': episode_stats['total_deadlocks'],
                'max_deadlock_severity': episode_stats['max_deadlock_severity'],
                'avg_throughput': episode_stats['avg_throughput'],
                'avg_acceleration': episode_stats['avg_acceleration'],
                'total_controlled_vehicles': episode_stats['total_controlled'],
                
                # Simulation time statistics
                'episode_simulation_time': episode_stats.get('episode_simulation_time', 0.0),
                'total_simulation_time': episode_stats.get('total_simulation_time', 0.0),
                'episode_start_time': episode_stats.get('episode_start_time', ''),
                'simulation_start_time': episode_stats.get('simulation_start_time', ''),
                'episode_duration_hours': episode_stats.get('episode_duration_hours', 0.0),
                'total_duration_hours': episode_stats.get('total_duration_hours', 0.0)
            }
            
            # Save episode summary
            self._save_episode_metrics(episode_summary)
            
            # Print episode summary
            print(f"📊 Episode {self.episode_count} Summary:")
            print(f"   Length: {episode_summary['episode_length']} steps")
            print(f"   Vehicles exited: {episode_summary['total_vehicles_exited']}")
            print(f"   Collisions: {episode_summary['total_collisions']}")
            print(f"   Deadlocks: {episode_summary['total_deadlocks']}")
            print(f"   Avg throughput: {episode_summary['avg_throughput']:.1f} vehicles/h")
            print(f"   Action means: [{action_means[0]:.3f}, {action_means[1]:.1f}, {action_means[2]:.1f}, {action_means[3]:.1f}]")
            print(f"   ⏰ Simulation time: {episode_summary['episode_simulation_time']:.1f}s ({episode_summary['episode_duration_hours']:.3f}h)")
            print(f"   ⏰ Total simulation time: {episode_summary['total_simulation_time']:.1f}s ({episode_summary['total_duration_hours']:.3f}h)")
            
        except Exception as e:
            print(f"⚠️ Episode finalization error: {e}")

    def _calculate_episode_stats(self) -> dict:
        """Calculate episode-level statistics from step metrics"""
        if not self.episode_metrics:
            return {
                'total_exits': 0, 'total_collisions': 0, 'total_deadlocks': 0,
                'max_deadlock_severity': 0.0, 'avg_throughput': 0.0,
                'avg_acceleration': 0.0, 'total_controlled': 0
            }
        
        # Calculate cumulative statistics
        total_exits = max(0, self.episode_metrics[-1].get('vehicles_exited', 0) - 
                         self.episode_metrics[0].get('vehicles_exited', 0))
        
        total_collisions = max(0, self.episode_metrics[-1].get('collision_count', 0) - 
                              self.episode_metrics[0].get('collision_count', 0))
        
        total_deadlocks = max(0, self.episode_metrics[-1].get('deadlocks_detected', 0) - 
                             self.episode_metrics[0].get('deadlocks_detected', 0))
        
        # Calculate averages
        throughputs = [m.get('throughput', 0.0) for m in self.episode_metrics]
        accelerations = [m.get('avg_acceleration', 0.0) for m in self.episode_metrics]
        controlled = [m.get('total_controlled', 0) for m in self.episode_metrics]
        
        # Get max deadlock severity
        max_severity = max([m.get('deadlock_severity', 0.0) for m in self.episode_metrics])
        
        # Calculate simulation time statistics
        episode_simulation_time = 0.0
        total_simulation_time = 0.0
        episode_start_time = ''
        simulation_start_time = ''
        
        if self.episode_metrics:
            # Get the latest simulation time info
            latest_metrics = self.episode_metrics[-1]
            episode_simulation_time = latest_metrics.get('episode_simulation_time', 0.0)
            total_simulation_time = latest_metrics.get('total_simulation_time', 0.0)
            episode_start_time = latest_metrics.get('episode_start_time', '')
            simulation_start_time = latest_metrics.get('simulation_start_time', '')
        
        return {
            'total_exits': total_exits,
            'total_collisions': total_collisions,
            'total_deadlocks': total_deadlocks,
            'max_deadlock_severity': max_severity,
            'avg_throughput': np.mean(throughputs) if throughputs else 0.0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'total_controlled': max(controlled) if controlled else 0,
            'episode_simulation_time': episode_simulation_time,
            'total_simulation_time': total_simulation_time,
            'episode_start_time': episode_start_time,
            'simulation_start_time': simulation_start_time,
            'episode_duration_hours': round(episode_simulation_time / 3600, 3) if episode_simulation_time else 0.0,
            'total_duration_hours': round(total_simulation_time / 3600, 3) if total_simulation_time else 0.0
        }

    def _save_step_metrics(self):
        """Save step-level metrics to CSV"""
        if not self.episode_metrics:
            return
            
        current_time = time.time()
        if current_time - self._last_write_timestamp < self._write_interval:
            return
            
        try:
            df = pd.DataFrame(self.episode_metrics)
            df.to_csv(self.step_metrics_path, index=False)
            self._last_write_timestamp = current_time
            print(f"📊 Step metrics saved: {len(self.episode_metrics)} records")
        except Exception as e:
            print(f"⚠️ Step metrics save failed: {e}")

    def _save_episode_metrics(self, episode_summary: dict):
        """Save episode-level metrics to CSV"""
        try:
            # Load existing data or create new file
            csv_path = self.episode_metrics_path
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                new_df = pd.concat([existing_df, pd.DataFrame([episode_summary])], ignore_index=True)
            else:
                new_df = pd.DataFrame([episode_summary])
            
            new_df.to_csv(csv_path, index=False)
            print(f"📊 Episode {episode_summary['episode']} metrics saved")
            
        except Exception as e:
            print(f"⚠️ Episode metrics save failed: {e}")

    def _cleanup_resources(self):
        """Clean up resources on exit"""
        try:
            # Finalize current episode if training ends
            if self.episode_actions:
                self._finalize_episode()
            
            # Save final step metrics
            self._save_step_metrics()
            
            print("🧹 Metrics callback cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

    def _on_training_end(self):
        """Called when training ends"""
        self._cleanup_resources()

# System resource monitoring functions removed - no longer needed

def create_timestamped_directories(instance_id: int = 0) -> Dict[str, str]:
    """Create timestamped directories for each training run"""
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base timestamped directory
    timestamped_base = f"drl/training_runs/{timestamp}"
    
    # Add instance suffix if multiple instances
    if instance_id > 0:
        timestamped_base += f"_instance_{instance_id}"
    
    # Create all subdirectories
    dirs = {
        'base_dir': timestamped_base,
        'log_dir': f"{timestamped_base}/logs",
        'checkpoint_dir': f"{timestamped_base}/checkpoints", 
        'results_dir': f"{timestamped_base}/results",
        'plots_dir': f"{timestamped_base}/plots",
        'csv_dir': f"{timestamped_base}/csv",
        'config_dir': f"{timestamped_base}/config"
    }
    
    # Create all directories
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"📁 {name}: {path}")
    
    # Save training configuration to config directory
    config_info = {
        'timestamp': timestamp,
        'instance_id': instance_id,
        'created_at': datetime.now().isoformat(),
        'training_parameters': {
            'total_timesteps': '5000',  # Default value
            'learning_rate': '3e-4',
            'n_steps': '256',
            'batch_size': '64',
            'n_epochs': '4'
        }
    }
    
    config_file = os.path.join(dirs['config_dir'], 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    print(f"📋 Training configuration saved to: {config_file}")
    
    return dirs

def main():
    print("🚀 Starting DRL Training for Traffic Intersection Control")
    print("=" * 70)
    
    # Parse command line arguments for multi-instance support
    import argparse
    parser = argparse.ArgumentParser(description='DRL Training with multi-CARLA instance support')
    parser.add_argument('--carla-port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--carla-host', type=str, default='localhost', help='CARLA server host (default: localhost)')
    parser.add_argument('--instance-id', type=int, default=0, help='CARLA instance ID for logging (default: 0)')
    parser.add_argument('--total-timesteps', type=int, default=5000, help='Total training timesteps (default: 5000)')
    
    args = parser.parse_args()
    
    # Create timestamped directories for this training run
    print(f"🕐 Creating timestamped directories for training run...")
    dirs = create_timestamped_directories(args.instance_id)
    
    # DRL parameters
    config = {
        'total_timesteps': args.total_timesteps,
        'learning_rate': 3e-4,
        'n_steps': 256,
        'batch_size': 64,
        'n_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'checkpoint_freq': 1000
        }
    
    # Update config file with actual training parameters
    config_file = os.path.join(dirs['config_dir'], 'training_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_info = json.load(f)
        config_info['training_parameters'].update({
            'total_timesteps': str(args.total_timesteps),
            'learning_rate': str(config['learning_rate']),
            'n_steps': str(config['n_steps']),
            'batch_size': str(config['batch_size']),
            'n_epochs': str(config['n_epochs'])
        })
        with open(config_file, 'w') as f:
            json.dump(config_info, f, indent=2)
    
    training_success = False
    model = None
    
    try:
        print(f"🎯 Creating optimized training environment for CARLA instance {args.instance_id}...")
        print(f"   🌐 CARLA Server: {args.carla_host}:{args.carla_port}")
        print(f"   📁 Training run directory: {dirs['base_dir']}")
        
        env = AuctionGymEnv(sim_cfg={
            'max_steps': 128,  # OPTIMAL: Aligned with n_steps=512 for 2 episodes per update
            'training_mode': True,  # Enable performance optimizations
            # Multi-instance CARLA configuration
            'carla_port': args.carla_port,
            'carla_host': args.carla_host,
            'carla_instance_id': args.instance_id,
            'fixed_delta_seconds': 0.1,  # 10 FPS simulation
            'logic_update_interval_seconds': 1.0,  # 1s decision intervals (REDUCED from 2.0s)
            'auction_interval': 4.0,  # 4s auction cycles (REDUCED from 6.0s)
            'bidding_duration': 2.0,  # 2s bidding phase (REDUCED from 3.0s)
            'deadlock_check_interval': 8.0,  # 8s system checks (REDUCED from 12.0s)
            'deadlock_reset_enabled': False,  # Episodes terminate on deadlock
            'severe_deadlock_reset_enabled': False,  # Episodes terminate on severe deadlock
            'severe_deadlock_punishment': -200.0  # Punishment applied to final step only
        })
        print("✅ Environment created successfully")

        # FIXED: Verify that max_steps configuration was properly applied
        print("\n🔍 VERIFYING EPISODE LENGTH CONFIGURATION:")
        if hasattr(env, 'max_actions'):
            print(f"   ✅ Environment max_actions: {env.max_actions}")
            if env.max_actions == 128:
                print(f"   🎯 SUCCESS: Episode length correctly set to 128 steps")
            else:
                print(f"   ❌ FAILED: Expected 128 steps, got {env.max_actions}")
        else:
            print(f"   ⚠️ Environment has no max_actions attribute")
        
        # Also check sim_cfg if available
        if hasattr(env, 'sim_cfg'):
            print(f"   Environment sim_cfg max_steps: {env.sim_cfg.get('max_steps', 'NOT_SET')}")
        else:
            print(f"   Environment has no sim_cfg attribute")
        
        # Show action space configuration for verification
        print("\n🔍 VERIFYING ACTION SPACE CONFIGURATION:")
        action_space_config = env.get_current_action_space_config()
        print(f"   Action space dimensions: {action_space_config['action_space_dimensions']}")
        print(f"   Action space shape: {action_space_config['action_space_shape']}")
        print(f"   Parameter mappings: {len(action_space_config['parameter_mappings'])} parameters")
        
        # Show initial parameter values
        initial_params = env.get_current_parameter_values()
        if 'error' not in initial_params:
            print(f"   Initial parameter values:")
            for param_name in ['urgency_position_ratio', 'speed_diff_modifier', 'max_participants_per_auction', 'ignore_vehicles_go']:
                if param_name in initial_params:
                    print(f"     {param_name}: {initial_params[param_name]}")
        else:
            print(f"   Could not get initial parameter values: {initial_params['error']}")
        
        # Test parameter mapping to verify ranges
        print("\n🔍 TESTING PARAMETER MAPPING RANGES:")
        env.test_parameter_mapping(num_samples=100)

        # SIMPLIFIED: Robust compatibility wrapper without complex logic
        class SimpleCompatWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                """Simple reset handling that works with both gym and gymnasium"""
                try:
                    # Call the environment's reset method
                    result = self.env.reset(**kwargs)
                    
                    # Handle different return formats safely
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            # gymnasium format: (obs, info) or more
                            obs = result[0]
                            info = result[1] if len(result) > 1 else {}
                        else:
                            # Single element tuple
                            obs = result[0]
                            info = {}
                    else:
                        # Single value (old gym format)
                        obs = result
                        info = {}
                    
                    # Ensure obs is numpy array with correct shape
                    if not isinstance(obs, np.ndarray):
                        try:
                            obs = np.array(obs, dtype=np.float32)
                        except Exception as array_error:
                            print(f"⚠️ Failed to convert obs to numpy array: {array_error}")
                            obs = np.zeros(50, dtype=np.float32)
                    
                    # Ensure correct dimensions with proper error handling
                    try:
                        if not hasattr(obs, 'shape'):
                            print(f"⚠️ obs has no shape attribute, type: {type(obs)}")
                            obs = np.zeros(50, dtype=np.float32)
                        elif len(obs.shape) == 0:
                            print(f"⚠️ obs has scalar shape, converting to array")
                            obs = np.array([obs], dtype=np.float32)
                        elif obs.shape[0] != 50:
                            if obs.shape[0] < 50:
                                # Pad with zeros
                                padding = np.zeros(50 - obs.shape[0], dtype=np.float32)
                                obs = np.concatenate([obs, padding])
                            else:
                                # Truncate
                                obs = obs[:50]
                    except Exception as shape_error:
                        print(f"⚠️ Error handling obs shape: {shape_error}, obs type: {type(obs)}")
                        obs = np.zeros(50, dtype=np.float32)
                    
                    return obs, info
                    
                except Exception as e:
                    print(f"⚠️ Reset wrapper error: {str(e)}")
                    # Return safe fallback
                    fallback_obs = np.zeros(50, dtype=np.float32)
                    fallback_info = {'reset_error': str(e), 'fallback': True}
                    return fallback_obs, fallback_info

            def step(self, action):
                """Simple step handling that works with both gym and gymnasium"""
                try:
                    result = self.env.step(action)
                    
                    # Handle different return formats
                    if isinstance(result, tuple):
                        if len(result) == 4:
                            # Old gym: obs, reward, done, info
                            obs, reward, done, info = result
                            return obs, reward, done, False, info  # Add truncated=False
                        elif len(result) == 5:
                            # New gymnasium: obs, reward, terminated, truncated, info
                            return result
                        else:
                            # Unexpected format - try to handle gracefully
                            print(f"⚠️ Unexpected step output length: {len(result)}")
                            return result
                    else:
                        print(f"⚠️ Step output is not tuple: {type(result)}")
                        return result
                        
                except Exception as e:
                    print(f"⚠️ Step wrapper error: {str(e)}")
                    # Return safe fallback
                    fallback_obs = np.zeros(50, dtype=np.float32)
                    fallback_info = {'step_error': str(e), 'fallback': True}
                    return fallback_obs, -10.0, True, True, fallback_info

        env = SimpleCompatWrapper(env)
        
        # Setup logging WITHOUT TensorBoard
        logger = configure(dirs['log_dir'], ["csv"])  # REMOVED: "tensorboard"
        
        # Create PPO model with OPTIMAL parameters
        print("🤖 Creating PPO model...")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            verbose=1
            # REMOVED: tensorboard_log=dirs['log_dir']
        )
        
        model.set_logger(logger)
        print("✅ PPO model created successfully")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=dirs['checkpoint_dir'],
            name_prefix="ppo_traffic"
        )
        
        # Simple metrics callback
        metrics_callback = SimpleMetricsCallback(
            log_dir=dirs['results_dir'],
            verbose=0
        )
        
        # Start training
        print(f"\n🎓 Starting DEBUG training for {config['total_timesteps']} timesteps...")
        print("Press Ctrl+C to stop training early")
        print("=" * 60)
        
        start_time = time.time()
        
        # FIXED: Add explicit training termination check
        print(f"🎓 Training will stop automatically at {config['total_timesteps']} timesteps")
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True
        )
        
        print(f"🏁 Training COMPLETED - reached {config['total_timesteps']} timesteps")
        
        # Save final model
        final_model_path = os.path.join(dirs['checkpoint_dir'], "final_model.zip")
        model.save(final_model_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed_time:.2f} seconds")
        print(f"📁 Final model saved to: {final_model_path}")
        
        training_success = True
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        try:
            if model is not None:
                interrupted_path = os.path.join(dirs['checkpoint_dir'], "interrupted_model.zip")
                model.save(interrupted_path)
                print(f"💾 Model saved to: {interrupted_path}")
        except Exception as save_error:
            print(f"❌ Could not save interrupted model: {save_error}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if hasattr(env, 'close'):
                env.close()
            print("🏁 Environment closed")
            
            # Force garbage collection to clean up resources
            import gc
            gc.collect()
        except:
            pass
        
        # ALWAYS generate analysis plots (whether successful or interrupted)
        print("\n" + "=" * 70)
        print("📊 GENERATING TRAINING ANALYSIS")
        print("=" * 70)
        
        try:
            # Import and use the new plotting utility
            from drl.utils.plot_generator import plot_training_metrics, generate_summary_report
            
            print("🎨 Generating training plots using new plotting utility...")
            plot_training_metrics(dirs['results_dir'], dirs['plots_dir'], save_plots=True)
            generate_summary_report(dirs['results_dir'], dirs['plots_dir'])
            
            print(f"\n✅ Analysis complete! Check these locations:")
            print(f"   📊 Plots: {dirs['plots_dir']}")
            print(f"   📋 Summary: {os.path.join(dirs['plots_dir'], 'training_summary.txt')}")
            print(f"   📈 Metrics: {os.path.join(dirs['results_dir'], 'episode_metrics.csv')}")
            
        except ImportError:
            print("⚠️ New plotting utility not available, using legacy analysis...")
            try:
                # Fallback to legacy analysis
                analyzer = TrainingAnalyzer(dirs['results_dir'], dirs['plots_dir'])
                analyzer.generate_all_plots()
                analyzer.generate_report()
                analyzer.save_summary_json()
                
                print(f"\n✅ Legacy analysis completed. Check {dirs['plots_dir']} for plots and reports")
            except Exception as legacy_error:
                print(f"❌ Legacy analysis also failed: {legacy_error}")
        except Exception as analysis_error:
            print(f"❌ Analysis failed: {analysis_error}")
            print(f"   You can still run analysis manually:")
            print(f"   python -m drl.utils.plot_generator --results-dir {dirs['results_dir']} --plots-dir {dirs['plots_dir']}")
        
        # Copy CSV files to dedicated CSV directory for easy access
        try:
            print(f"\n📁 Copying CSV files to dedicated directory...")
            import shutil
            
            # Copy episode metrics CSV
            episode_csv_src = os.path.join(dirs['results_dir'], 'episode_metrics.csv')
            episode_csv_dst = os.path.join(dirs['csv_dir'], 'episode_metrics.csv')
            if os.path.exists(episode_csv_src):
                shutil.copy2(episode_csv_src, episode_csv_dst)
                print(f"   ✅ Copied episode_metrics.csv to {dirs['csv_dir']}")
            
            # Copy step metrics CSV
            step_csv_src = os.path.join(dirs['results_dir'], 'step_metrics.csv')
            step_csv_dst = os.path.join(dirs['csv_dir'], 'step_metrics.csv')
            if os.path.exists(step_csv_src):
                shutil.copy2(step_csv_src, step_csv_dst)
                print(f"   ✅ Copied step_metrics.csv to {dirs['csv_dir']}")
            
            # Create a summary CSV with training run information
            summary_csv_path = os.path.join(dirs['csv_dir'], 'training_summary.csv')
            summary_data = {
                'training_run': [dirs['base_dir'].split('/')[-1]], # Use the timestamped base directory name
                'instance_id': [args.instance_id],
                'total_timesteps': [args.total_timesteps],
                'training_success': [training_success],
                'elapsed_time_seconds': [elapsed_time if 'elapsed_time' in locals() else 0],
                'base_directory': [dirs['base_dir']],
                'created_at': [datetime.now().isoformat()]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"   ✅ Created training_summary.csv in {dirs['csv_dir']}")
            
        except Exception as csv_error:
            print(f"⚠️ CSV copying failed: {csv_error}")
        
        print(f"\n🏁 Training session complete!")
        print(f"📁 All results stored in: {dirs['base_dir']}")
        print(f"📊 CSV files available in: {dirs['csv_dir']}")

if __name__ == "__main__":
    main()
