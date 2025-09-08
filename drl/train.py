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
import shutil
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
    
    def __init__(self, log_dir: str, verbose: int = 0, continue_training: bool = False):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.continue_training = continue_training
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode-level tracking
        self.episode_actions = []  # Store actions for current episode
        self.episode_metrics = []  # Store metrics for current episode
        self.episode_count = 0
        self.episode_start_step = 0
        self.current_episode_termination_reason = "Unknown"  # Store termination reason for current episode
        self.current_episode_params = {}  # Store current episode parameters
        
        # CSV file paths
        self.step_metrics_path = os.path.join(log_dir, 'step_metrics.csv')
        self.episode_metrics_path = os.path.join(log_dir, 'episode_metrics.csv')
        
        # File handle management
        self._last_write_timestamp = 0
        self._write_interval = 10.0  # Write every 10 seconds at most
        
        # Register cleanup function
        import atexit
        atexit.register(self._cleanup_resources)
        
        # If continuing training, load existing metrics
        if self.continue_training:
            self._load_existing_metrics()
        else:
            # Auto-detect and copy past_train CSV files if they exist
            self._auto_copy_past_train_csv()
        
        print(f"📊 Enhanced Metrics Callback initialized:")
        print(f"   Step metrics: {self.step_metrics_path}")
        print(f"   Episode metrics: {self.episode_metrics_path}")
        if self.continue_training:
            print(f"   🔄 Continue training mode: Loaded existing metrics")
        else:
            print(f"   🆕 New training mode: Ready to start")

    def _load_existing_metrics(self):
        """Load existing metrics data to continue recording"""
        try:
            # Load existing step metrics
            if os.path.exists(self.step_metrics_path):
                existing_step_df = pd.read_csv(self.step_metrics_path)
                if len(existing_step_df) > 0:
                    # Get the last timestep
                    last_timestep = existing_step_df['timestep'].max()
                    self.num_timesteps = last_timestep
                    print(f"📊 Loaded existing step metrics: {len(existing_step_df)} rows, last timestep: {last_timestep}")
                else:
                    print(f"📊 Existing step metrics file is empty")
            else:
                print(f"📊 No existing step metrics file found")
            
            # Load existing episode metrics
            if os.path.exists(self.episode_metrics_path):
                existing_episode_df = pd.read_csv(self.episode_metrics_path)
                if len(existing_episode_df) > 0:
                    # Get the last episode number
                    last_episode = existing_episode_df['episode'].max()
                    self.episode_count = last_episode
                    print(f"📊 Loaded existing episode metrics: {len(existing_episode_df)} rows, last episode: {last_episode}")
                else:
                    print(f"📊 Existing episode metrics file is empty")
            else:
                print(f"📊 No existing episode metrics file found")
                
        except Exception as e:
            print(f"⚠️ Failed to load existing metrics: {e}")
            # Continue with default values
            print(f"   🔄 Continuing with default values (episode_count: 0, timestep: 0)")

    def _auto_copy_past_train_csv(self):
        """Auto-detect and copy past_train CSV files if they exist."""
        # Use absolute path: past_train is in same directory as train.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        past_train_dir = os.path.join(current_dir, "past_train")
        results_dir = os.path.join(past_train_dir, "results")
        
        if not os.path.exists(results_dir):
            print(f"📁 past_train/results directory does not exist, no CSV files to copy.")
            return
        
        print(f"🔍 Auto-detecting and copying CSV files from past_train/results directory...")
        
        # Find step_metrics.csv and episode_metrics.csv
        step_metrics_src = os.path.join(results_dir, "step_metrics.csv")
        episode_metrics_src = os.path.join(results_dir, "episode_metrics.csv")
        
        # Copy step_metrics.csv
        if os.path.exists(step_metrics_src):
            try:
                shutil.copy2(step_metrics_src, self.step_metrics_path)
                print(f"   ✅ Copied step_metrics.csv to {self.step_metrics_path}")
                
                # Read and display file information
                df = pd.read_csv(self.step_metrics_path)
                print(f"   📊 step_metrics.csv contains {len(df)} rows of data")
                if 'timestep' in df.columns and len(df) > 0:
                    last_timestep = df['timestep'].max()
                    print(f"   🎯 Last timestep: {last_timestep}")
            except Exception as copy_error:
                print(f"⚠️ Failed to copy step_metrics.csv: {copy_error}")
        else:
            print(f"   📁 step_metrics.csv does not exist")
        
        # Copy episode_metrics.csv
        if os.path.exists(episode_metrics_src):
            try:
                shutil.copy2(episode_metrics_src, self.episode_metrics_path)
                print(f"   ✅ Copied episode_metrics.csv to {self.episode_metrics_path}")
                
                # Read and display file information
                df = pd.read_csv(self.episode_metrics_path)
                print(f"   📊 episode_metrics.csv contains {len(df)} rows of data")
                if 'episode' in df.columns and len(df) > 0:
                    last_episode = df['episode'].max()
                    print(f"   🎯 Last episode: {last_episode}")
            except Exception as copy_error:
                print(f"⚠️ Failed to copy episode_metrics.csv: {copy_error}")
        else:
            print(f"   📁 episode_metrics.csv does not exist")
        
        print(f"   🔄 CSV file copying complete, new training will continue based on this data")

    def auto_detect_past_train_checkpoint(self):
        """Auto-detect latest checkpoint in past_train folder"""
        # Use absolute path: past_train is in same directory as train.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        past_train_dir = os.path.join(current_dir, "past_train")
        checkpoints_dir = os.path.join(past_train_dir, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            print(f"📁 past_train/checkpoints directory does not exist")
            return None
        
        # Find all checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoints_dir):
            if file.endswith('.zip') and 'ppo_traffic_' in file and '_steps.zip' in file:
                try:
                    # Extract step count
                    steps_str = file.replace('ppo_traffic_', '').replace('_steps.zip', '')
                    steps = int(steps_str)
                    checkpoint_files.append((steps, file))
                except ValueError:
                    continue
        
        if not checkpoint_files:
            print(f"📁 No valid checkpoint files found")
            return None
        
        # Sort by step count and find the latest
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        latest_steps, latest_file = checkpoint_files[0]
        latest_path = os.path.join(checkpoints_dir, latest_file)
        
        print(f"🔍 Auto-detected latest checkpoint: {latest_file}")
        print(f"   📍 Path: {latest_path}")
        print(f"   📊 Training steps: {latest_steps}")
        
        return latest_path, latest_steps

    def _on_step(self) -> bool:
        """Log metrics every step and track episode boundaries - FIXED deadlock detection"""
        try:
            # Get current info and actions
            infos = self.locals.get('infos', [{}])
            actions = self.locals.get('actions', [])
            
            # FIXED: Use proper length checks instead of boolean checks for arrays
            if len(infos) == 0 or len(actions) == 0:
                return True
            
            info = infos[0] if isinstance(infos[0], dict) else {}
            action = actions[0] if isinstance(actions[0], (np.ndarray, list)) else []
            
            # FIXED: Check for episode termination based on IMMEDIATE deadlock detection
            # This ensures episodes end immediately when deadlock occurs, not delayed
            episode_should_end = False
            termination_reason = ""
            
            # Check if episode reached max steps (128)
            if len(self.episode_actions) >= 128:
                episode_should_end = True
                termination_reason = f"Reached exactly {len(self.episode_actions)} steps"
                print(f"🎯 Episode {self.episode_count} completed: {termination_reason}")
            
            # FIXED: Check for immediate deadlock detection from environment
            # Use termination_info from the current step, not delayed
            elif info.get('termination_info', {}).get('deadlock_detected', False):
                episode_should_end = True
                termination_reason = "Deadlock detected by environment (IMMEDIATE)"
                print(f"🚨 Episode {self.episode_count} terminated immediately: {termination_reason}")
                print(f"   📊 Deadlock info: {info.get('termination_info', {})}")
                print(f"   📊 Current episode length: {len(self.episode_actions)} steps")
                print(f"   ✅ Reward will be applied to current episode, not delayed")
            
            elif info.get('termination_info', {}).get('severe_deadlock_detected', False):
                episode_should_end = True
                termination_reason = "Severe deadlock detected by environment (IMMEDIATE)"
                print(f"⚡ Episode {self.episode_count} terminated immediately: {termination_reason}")
                print(f"   📊 Severe deadlock info: {info.get('termination_info', {})}")
                print(f"   📊 Current episode length: {len(self.episode_actions)} steps")
                print(f"   ✅ Reward will be applied to current episode, not delayed")
            
            # End episode if any termination condition is met
            if episode_should_end:
                print(f"📊 Episode {self.episode_count} finalizing: {termination_reason}")
                # Store termination reason for this episode
                self.current_episode_termination_reason = termination_reason
                
                # CRITICAL: Add current step's action and metrics BEFORE finalizing episode
                # This ensures the final step is included in the episode data
                if isinstance(action, (np.ndarray, list)) and len(action) == 4:
                    self.episode_actions.append(action)
                
                # Create and add current step metrics to current episode
                current_step_metrics = {
                    'timestep': self.num_timesteps,
                    'episode': self.episode_count,
                    'throughput': info.get('throughput', 0.0),
                    'avg_acceleration': info.get('avg_acceleration', 0.0),
                    'collision_count': info.get('collision_count', 0),
                    'total_controlled': info.get('total_controlled', 0),
                    'vehicles_exited': info.get('vehicles_exited', 0),
                    'deadlocks_detected': info.get('deadlocks_detected', 0),
                    'deadlock_severity': info.get('deadlock_severity', 0.0),
                    'reward': info.get('reward', 0.0),  # FIXED: Include reward from current step
                    'termination_reason': termination_reason,  # NEW: Track why episode ended
                    'deadlock_detected': info.get('termination_info', {}).get('deadlock_detected', False),  # NEW: Track deadlock status
                    'severe_deadlock_detected': info.get('termination_info', {}).get('severe_deadlock_detected', False)  # NEW: Track severe deadlock status
                }
                
                # Add simulation time information if available
                simulation_time_info = info.get('simulation_time', {})
                if simulation_time_info:
                    current_step_metrics.update({
                        'episode_simulation_time': simulation_time_info.get('episode_simulation_time', 0.0),
                        'total_simulation_time': simulation_time_info.get('total_simulation_time', 0.0),
                        'episode_start_time': simulation_time_info.get('episode_start_time', ''),
                        'simulation_start_time': simulation_time_info.get('simulation_start_time', '')
                    })
                
                self.episode_metrics.append(current_step_metrics)
                print(f"📊 Added final step metrics to episode {self.episode_count}: {len(self.episode_metrics)} total steps")
                print(f"   💰 Final step reward: {current_step_metrics['reward']:.2f}")
                print(f"   🚨 Deadlock detected: {current_step_metrics['deadlock_detected']}")
                print(f"   ⚡ Severe deadlock: {current_step_metrics['severe_deadlock_detected']}")
                
                # Now finalize the episode with complete data
                self._finalize_episode()
                self._start_new_episode()
                
                # CRITICAL: Skip further processing since episode has ended
                return True
            
            # FIXED: Capture new episode parameters at the beginning of each episode
            # This ensures we store the correct parameters that were applied for this episode
            if len(self.episode_actions) == 1:  # First action of new episode
                try:
                    print(f"🔍 Episode {self.episode_count}: Capturing episode parameters...")
                    
                    # Method 1: Get parameters from action_params in info (most reliable)
                    if 'action_params' in info:
                        action_params = info['action_params']
                        self.current_episode_params = {
                            'urgency_position_ratio': action_params.get('urgency_position_ratio', 1.0),
                            'speed_diff_modifier': action_params.get('speed_diff_modifier', 0.0),
                            'max_participants_per_auction': action_params.get('max_participants_per_auction', 4.0),
                            'ignore_vehicles_go': action_params.get('ignore_vehicles_go', 50.0)
                        }
                        print(f"🎯 Episode {self.episode_count} parameters captured from action_params:")
                        print(f"   Urgency Position Ratio: {self.current_episode_params['urgency_position_ratio']:.3f}")
                        print(f"   Speed Diff Modifier: {self.current_episode_params['speed_diff_modifier']:.1f}")
                        print(f"   Max Participants: {self.current_episode_params['max_participants_per_auction']}")
                        print(f"   Ignore Vehicles GO: {self.current_episode_params['ignore_vehicles_go']:.1f}%")
                    
                    # Method 2: Fallback to environment method if action_params not available
                    elif hasattr(self, 'training_env') and hasattr(self.training_env, 'get_current_parameter_values'):
                        self.current_episode_params = self.training_env.get_current_parameter_values()
                        print(f"🎯 Episode {self.episode_count} parameters captured from environment:")
                        print(f"   Urgency Position Ratio: {self.current_episode_params.get('urgency_position_ratio', 'N/A')}")
                        print(f"   Speed Diff Modifier: {self.current_episode_params.get('speed_diff_modifier', 'N/A')}")
                        print(f"   Max Participants: {self.current_episode_params.get('max_participants_per_auction', 'N/A')}")
                        print(f"   Ignore Vehicles GO: {self.current_episode_params.get('ignore_vehicles_go', 'N/A')}")
                    
                    # Method 3: Use fallback values if no other method works
                    else:
                        print(f"⚠️ No parameter access method available, using fallback values")
                        self.current_episode_params = {
                            'urgency_position_ratio': 1.0,
                            'speed_diff_modifier': 0.0,
                            'max_participants_per_auction': 4.0,
                            'ignore_vehicles_go': 50.0
                        }
                        print(f"   Using default parameters for episode {self.episode_count}")
                        
                except Exception as e:
                    print(f"⚠️ Warning: Could not retrieve episode parameters: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use fallback values
                    self.current_episode_params = {
                        'urgency_position_ratio': 1.0,
                        'speed_diff_modifier': 0.0,
                        'max_participants_per_auction': 4.0,
                        'ignore_vehicles_go': 50.0
                    }
            
            # FIXED: Enhanced debugging for deadlock detection
            if info.get('termination_info'):
                term_info = info.get('termination_info', {})
                if term_info.get('deadlock_detected') or term_info.get('severe_deadlock_detected'):
                    print(f"🚨 DEADLOCK DETECTED at step {self.num_timesteps}:")
                    print(f"   Episode: {self.episode_count}, Actions: {len(self.episode_actions)}")
                    print(f"   Deadlock: {term_info.get('deadlock_detected', False)}")
                    print(f"   Severe deadlock: {term_info.get('severe_deadlock_detected', False)}")
                    print(f"   Current reward: {info.get('reward', 'N/A')}")
            
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
                'deadlocks_detected': info.get('deadlocks_detected', 0),
                'deadlock_severity': info.get('deadlock_severity', 0.0),
                'reward': info.get('reward', 0.0),  # FIXED: Add exact reward value
                'deadlock_detected': info.get('termination_info', {}).get('deadlock_detected', False),  # NEW: Track deadlock status
                'severe_deadlock_detected': info.get('termination_info', {}).get('severe_deadlock_detected', False)  # NEW: Track severe deadlock status
            }
            
            # ENHANCED DEBUG: Log collision and deadlock counts for debugging
            collision_count = step_metrics['collision_count']
            deadlocks_detected = step_metrics['deadlocks_detected']
            
            if collision_count > 0 or deadlocks_detected > 0:
                print(f"🔍 Step {self.num_timesteps} Metrics: collisions={collision_count}, deadlocks={deadlocks_detected}")
            
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
            
            # Debug: Log episode metrics collection every 20 steps
            if len(self.episode_actions) % 20 == 0:
                print(f"📊 Episode {self.episode_count}: Collected {len(self.episode_metrics)} step metrics, {len(self.episode_actions)} actions")
                # Show current episode totals
                if self.episode_metrics:
                    first_step = self.episode_metrics[0]
                    current_step = self.episode_metrics[-1]
                    episode_collisions = current_step.get('collision_count', 0) - first_step.get('collision_count', 0)
                    episode_deadlocks = current_step.get('deadlocks_detected', 0) - first_step.get('deadlocks_detected', 0)
                    print(f"   📊 Episode totals so far: collisions={episode_collisions}, deadlocks={episode_deadlocks}")
            
            # Save step metrics every 10 steps (increased frequency for reward tracking)
            if self.num_timesteps % 10 == 0:
                self._save_step_metrics()
                
        except Exception as e:
            print(f"⚠️ Metrics callback error: {e}")
        
        return True



    def _start_new_episode(self):
        """Start tracking a new episode"""
        self.episode_count += 1
        self.episode_start_step = self.num_timesteps
        self.episode_actions = []
        self.episode_metrics = []
        self.current_episode_termination_reason = "Unknown"  # Reset termination reason
        self.current_episode_params = {}  # Reset episode parameters
        print(f"🔄 Starting episode {self.episode_count} at step {self.episode_start_step}")

    def _finalize_episode(self):
        """Calculate and save episode-level statistics"""
        if not self.episode_actions or not self.episode_metrics:
            return
        
        try:
            # Convert actions to numpy array for reference (not for calculations)
            actions_array = np.array(self.episode_actions)
            
            # Get episode-level metrics
            episode_stats = self._calculate_episode_stats()
            
            # Use stored episode parameters (captured at episode start)
            exact_params = self.current_episode_params
            if not exact_params:
                print(f"⚠️ Warning: No episode parameters stored for episode {self.episode_count}")
                # Use fallback values
                exact_params = {
                    'urgency_position_ratio': 1.0,
                    'speed_diff_modifier': 0.0,
                    'max_participants_per_auction': 4.0,
                    'ignore_vehicles_go': 50.0
                }
            
            # Use stored termination reason or determine based on episode length and metrics
            termination_reason = self.current_episode_termination_reason
            if termination_reason == "Unknown":
                # Fallback logic if termination reason wasn't set
                if len(self.episode_actions) >= 128:
                    termination_reason = f"Reached exactly {len(self.episode_actions)} steps"
                elif episode_stats.get('total_deadlocks', 0) > 0:
                    termination_reason = "Deadlock detected during episode"
                elif len(self.episode_actions) < 128:
                    termination_reason = f"Early termination at {len(self.episode_actions)} steps"
            
            # Create episode summary
            episode_summary = {
                'episode': self.episode_count,
                'episode_start_step': self.episode_start_step,
                'episode_end_step': self.num_timesteps,
                'episode_length': len(self.episode_actions),
                'termination_reason': termination_reason,  # Record why episode ended
                
                # TRUE EXACT parameter values (actual values applied in environment)
                'urgency_position_ratio_exact': exact_params.get('urgency_position_ratio', 0.0),
                'speed_diff_modifier_exact': exact_params.get('speed_diff_modifier', 0.0),
                'max_participants_exact': exact_params.get('max_participants_per_auction', 4.0),
                'ignore_vehicles_go_exact': exact_params.get('ignore_vehicles_go', 50.0),
                
                # Episode performance metrics
                'total_vehicles_exited': episode_stats['total_exits'],
                'total_collisions': episode_stats['total_collisions'],
                'total_deadlocks': episode_stats['total_deadlocks'],
                'max_deadlock_severity': episode_stats['max_deadlock_severity'],
                'avg_throughput': episode_stats['avg_throughput'],
                'avg_acceleration': episode_stats['avg_acceleration'],
                'total_controlled_vehicles': episode_stats['total_controlled'],
                
                # NEW: Reward statistics (exact values only, no calculations)
                'total_reward': episode_stats['total_reward'],
                
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
            print(f"   TRUE EXACT params: [{episode_summary['urgency_position_ratio_exact']:.3f}, {episode_summary['speed_diff_modifier_exact']:.1f}, {episode_summary['max_participants_exact']:.0f}, {episode_summary['ignore_vehicles_go_exact']:.1f}]")
            print(f"   💰 Reward: Total={episode_summary['total_reward']:.1f}")
            print(f"   ⏰ Simulation time: {episode_summary['episode_simulation_time']:.1f}s ({episode_summary['episode_duration_hours']:.3f}h)")
            print(f"   ⏰ Total simulation time: {episode_summary['total_simulation_time']:.1f}s ({episode_summary['total_duration_hours']:.3f}h)")
            
        except Exception as e:
            print(f"⚠️ Episode finalization error: {e}")

    def _calculate_episode_stats(self) -> dict:
        """Calculate episode-level statistics from step metrics"""
        if not self.episode_metrics:
            print(f"⚠️ No episode metrics available for episode {self.episode_count}")
            return {
                'total_exits': 0, 'total_collisions': 0, 'total_deadlocks': 0,
                'max_deadlock_severity': 0.0, 'avg_throughput': 0.0,
                'avg_acceleration': 0.0, 'total_controlled': 0
            }
        
        print(f"🔍 Calculating stats for episode {self.episode_count}: {len(self.episode_metrics)} step metrics")
        
        # Calculate cumulative statistics
        total_exits = max(0, self.episode_metrics[-1].get('vehicles_exited', 0) - 
                         self.episode_metrics[0].get('vehicles_exited', 0))
        
        # FIXED: Collision count calculation - use absolute values, not differences
        # Each step's collision_count is already cumulative, so we need to get the actual
        # collisions that occurred during this episode
        first_step_collisions = self.episode_metrics[0].get('collision_count', 0)
        last_step_collisions = self.episode_metrics[-1].get('collision_count', 0)
        
        # Calculate collisions that occurred during this episode
        total_collisions = max(0, last_step_collisions - first_step_collisions)
        
        print(f"   🔍 Collision Debug: first_step={first_step_collisions}, last_step={last_step_collisions}, episode_total={total_collisions}")
        
        # FIXED: Deadlock count calculation - also use absolute values
        first_step_deadlocks = self.episode_metrics[0].get('deadlocks_detected', 0)
        last_step_deadlocks = self.episode_metrics[-1].get('deadlocks_detected', 0)
        total_deadlocks = max(0, last_step_deadlocks - first_step_deadlocks)
        
        print(f"   📊 Stats calculated: exits={total_exits}, collisions={total_collisions}, deadlocks={total_deadlocks}")
        print(f"   🔍 Deadlock Debug: first_step={first_step_deadlocks}, last_step={last_step_deadlocks}, episode_total={total_deadlocks}")
        
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
        
        # Calculate reward statistics (exact values only, no calculations)
        rewards = [m.get('reward', 0.0) for m in self.episode_metrics]
        total_reward = sum(rewards) if rewards else 0.0
        
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
            'total_duration_hours': round(total_simulation_time / 3600, 3) if total_simulation_time else 0.0,
            'total_reward': total_reward
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
            # Debug: Print episode summary before saving
            print(f"🔍 Saving episode {episode_summary['episode']} to CSV:")
            print(f"   Length: {episode_summary['episode_length']} steps")
            print(f"   Termination reason: {episode_summary['termination_reason']}")
            print(f"   CSV path: {self.episode_metrics_path}")
            
            # Load existing data or create new file
            csv_path = self.episode_metrics_path
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                print(f"   📁 Existing CSV has {len(existing_df)} episodes")
                new_df = pd.concat([existing_df, pd.DataFrame([episode_summary])], ignore_index=True)
            else:
                print(f"   📁 Creating new CSV file")
                new_df = pd.DataFrame([episode_summary])
            
            new_df.to_csv(csv_path, index=False)
            print(f"📊 Episode {episode_summary['episode']} metrics saved successfully")
            print(f"   📊 CSV now contains {len(new_df)} episodes")
            
        except Exception as e:
            print(f"⚠️ Episode metrics save failed: {e}")
            import traceback
            traceback.print_exc()

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
            'total_timesteps': '100000',  # Default value
            'learning_rate': '1e-4',
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
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Total training timesteps (default: 100000)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to continue training from')
    parser.add_argument('--continue-training', action='store_true', help='Continue training from checkpoint')
    
    args = parser.parse_args()
    
    # Create timestamped directories for this training run
    print(f"🕐 Creating timestamped directories for training run...")
    dirs = create_timestamped_directories(args.instance_id)
    
    # DRL parameters - Auto-detect from past_train if available, but expand to 100000 steps
    past_train_config = None
    # Use absolute path: past_train is in same directory as train.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    past_train_config_path = os.path.join(current_dir, "past_train", "config", "training_config.json")
    
    if os.path.exists(past_train_config_path):
        try:
            with open(past_train_config_path, 'r') as f:
                past_train_config = json.load(f)
            print(f"📋 Detected past_train configuration: {past_train_config_path}")
            
            # Extract past_train training parameters
            past_params = past_train_config.get('training_parameters', {})
            past_learning_rate = float(past_params.get('learning_rate', 0.0003))
            past_n_epochs = int(past_params.get('n_epochs', 4))
            
            print(f"   🎯 Past Train Settings:")
            print(f"      • Original target steps: {past_params.get('total_timesteps', 40000)}")
            print(f"      • Learning rate: {past_learning_rate}")
            print(f"      • Training epochs: {past_n_epochs}")
            print(f"   🚀 Extended target: 100,000 steps (keeping other parameters consistent)")
            
            # Use past_train settings, but extend to 100,000 steps
            config = {
                'total_timesteps': 100000,              # Extended to 100,000 steps
                'learning_rate': past_learning_rate,   # Keep past_train learning rate
                'n_steps': 256,
                'batch_size': 64,
                'n_epochs': past_n_epochs,             # Keep past_train training epochs
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'checkpoint_freq': 1000
            }
            
            print(f"   ✅ Using past_train settings + extended steps:")
            print(f"      • Total training steps: {config['total_timesteps']} (extended)")
            print(f"      • Learning rate: {config['learning_rate']} (maintained)")
            print(f"      • Training epochs: {config['n_epochs']} (maintained)")
            
        except Exception as e:
            print(f"⚠️ Failed to read past_train configuration: {e}")
            print(f"   🔄 Using default settings")
            # Use default settings
            config = {
                'total_timesteps': args.total_timesteps,
                'learning_rate': 1e-4,
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
    else:
        print(f"📋 No past_train configuration detected, using default settings")
        # Use default settings
        config = {
            'total_timesteps': args.total_timesteps,
            'learning_rate': 1e-4,
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
            'max_steps': 128,  
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
        
        # Auto-detect past_train checkpoint if no explicit checkpoint specified
        auto_checkpoint_path = None
        auto_checkpoint_steps = 0
        
        if not args.checkpoint:
            print("🔍 Auto-detecting checkpoints in past_train folder...")
            # Create temporary metrics callback to detect checkpoints
            temp_metrics_callback = SimpleMetricsCallback(
                log_dir=dirs['results_dir'],
                verbose=0,
                continue_training=False
            )
            
            auto_checkpoint_result = temp_metrics_callback.auto_detect_past_train_checkpoint()
            if auto_checkpoint_result:
                auto_checkpoint_path, auto_checkpoint_steps = auto_checkpoint_result
                print(f"✅ Auto-detected checkpoint: {auto_checkpoint_path}")
                print(f"   🎯 Suggested to continue training, current progress: {auto_checkpoint_steps} steps")
                
                # Calculate remaining steps using extended target steps
                target_timesteps = config['total_timesteps']  # 100,000
                remaining_steps = target_timesteps - auto_checkpoint_steps
                
                print(f"   📊 Training progress analysis:")
                print(f"      • Current steps: {auto_checkpoint_steps}")
                print(f"      • Target steps: {target_timesteps} (extended target)")
                print(f"      • Remaining steps: {remaining_steps}")
                print(f"      • Completion percentage: {auto_checkpoint_steps/target_timesteps*100:.1f}%")
                
                if remaining_steps > 0:
                    print(f"   🚀 Can continue training for {remaining_steps} steps to complete extended target")
                else:
                    print(f"   ✅ Training already completed, no need to continue")
                    return
                
                # Ask user whether to continue training
                print(f"🔄 Continue training from detected checkpoint?")
                print(f"   📁 Checkpoint: {os.path.basename(auto_checkpoint_path)}")
                print(f"   📊 Current steps: {auto_checkpoint_steps}")
                print(f"   🎯 Target steps: {target_timesteps}")
                
                # Automatically decide to continue training (can be modified to manual confirmation if needed)
                should_continue = True  # Auto-continue training
                if should_continue:
                    print(f"🚀 Auto-selected continue training mode")
                    args.checkpoint = auto_checkpoint_path
                    args.continue_training = True
                else:
                    print(f"🆕 Selected to start new training")
            else:
                print(f"📁 No past_train checkpoint detected, will start new training")
        
        # Check if we should load from checkpoint
        if args.checkpoint and args.continue_training:
            print(f"🔄 Loading model from checkpoint: {args.checkpoint}")
            try:
                # Load model from checkpoint
                model = PPO.load(args.checkpoint, env=env)
                model.set_logger(logger)
                print(f"✅ Model loaded successfully from checkpoint")
                print(f"   Current training timesteps: {model.num_timesteps}")
                
                # Calculate remaining training steps
                remaining_steps = config['total_timesteps'] - model.num_timesteps
                if remaining_steps > 0:
                    print(f"   Remaining training steps: {remaining_steps}")
                    print(f"   Will continue training for {remaining_steps} more steps")
                else:
                    print(f"   Training already completed ({model.num_timesteps} >= {config['total_timesteps']})")
                    return
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print(f"   Starting new training instead")
                model = None
        
        # Create PPO model if not loaded from checkpoint
        if model is None:
            print("🤖 Creating new PPO model...")
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
            verbose=0,
            continue_training=args.continue_training
        )
        
        # Start training
        if args.checkpoint and args.continue_training:
            # Continue training from checkpoint
            remaining_steps = config['total_timesteps'] - model.num_timesteps
            print(f"\n🔄 Continuing training from checkpoint for {remaining_steps} more timesteps...")
            print(f"   Current progress: {model.num_timesteps}/{config['total_timesteps']} ({model.num_timesteps/config['total_timesteps']*100:.1f}%)")
            print("Press Ctrl+C to stop training early")
            print("=" * 60)
            
            start_time = time.time()
            
            model.learn(
                total_timesteps=remaining_steps,
                callback=[checkpoint_callback, metrics_callback],
                progress_bar=True,
                reset_num_timesteps=False  # Key: Don't reset timestep counter
            )
            
            print(f"🏁 Continued training COMPLETED - reached {config['total_timesteps']} total timesteps")
        else:
            # Start new training
            print(f"\n🎓 Starting new training for {config['total_timesteps']} timesteps...")
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
        if args.checkpoint and args.continue_training:
            print(f"\n✅ Continued training completed in {elapsed_time:.2f} seconds")
            print(f"📁 Final model saved to: {final_model_path}")
            print(f"🎯 Total training progress: {model.num_timesteps}/{config['total_timesteps']} timesteps")
        else:
            print(f"\n✅ Training completed in {elapsed_time:.2f} seconds")
            print(f"📁 Final model saved to: {final_model_path}")
        
        training_success = True
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        try:
            if model is not None:
                if args.checkpoint and args.continue_training:
                    interrupted_path = os.path.join(dirs['checkpoint_dir'], "interrupted_continued_model.zip")
                    print(f"💾 Saving continued training model to: {interrupted_path}")
                else:
                    interrupted_path = os.path.join(dirs['checkpoint_dir'], "interrupted_model.zip")
                    print(f"💾 Saving interrupted model to: {interrupted_path}")
                model.save(interrupted_path)
                print(f"✅ Model saved successfully")
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
