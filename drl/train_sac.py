#!/usr/bin/env python3
"""
Enhanced DRL training script with SAC for traffic intersection control
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

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

from drl.envs.auction_gym import AuctionGymEnv
from drl.utils.analysis import TrainingAnalyzer

class SimpleSACMetricsCallback(BaseCallback):
    """Simplified callback for SAC data collection"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.metrics_log = []
        self.collection_interval = 100  # Collect every 100 steps
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'sac_training_metrics.csv')
        
        print(f"📊 Simple SAC metrics collection initialized:")
        print(f"   Collection interval: every {self.collection_interval} steps")

    def _save_metrics(self):
        """Simple save metrics to CSV"""
        if not self.metrics_log:
            return
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.metrics_log)
            df.to_csv(self.csv_path, index=False)
            print(f"📊 Saved {len(self.metrics_log)} SAC metrics records")
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
            


    def _on_step(self) -> bool:
        """Simple SAC data collection every N steps"""
        if self.num_timesteps % self.collection_interval != 0:
            return True
            
        try:
            infos = self.locals.get('infos', [])
            if not infos:
                return True
                
            info = infos[0] if isinstance(infos[0], dict) else {}
            reward_value = float(self.locals.get('rewards', [0.0])[0])
            
            # Collect basic metrics
            metrics = {
                'timestep': int(self.num_timesteps),
                'reward': reward_value,
                'algorithm': 'SAC',
                'throughput': float(info.get('throughput', 0.0)),
                'avg_acceleration': float(info.get('avg_acceleration', 0.0)),
                'collision_count': int(info.get('collision_count', 0)),
                'total_controlled': int(info.get('total_controlled', 0)),
                'vehicles_exited': int(info.get('vehicles_exited', 0)),
                'vehicles_detected': int(info.get('vehicles_detected', 0)),
                'deadlocks_detected': int(info.get('deadlocks_detected', 0)),
                'deadlock_severity': float(info.get('deadlock_severity', 0.0)),
                
                # Core bidding parameters (4 parameters)
                'bid_scale': float(info.get('bid_scale', 1.0)),
                'eta_weight': float(info.get('eta_weight', 1.0)),
                'platoon_bonus': float(info.get('platoon_bonus', 0.5)),
                'junction_penalty': float(info.get('junction_penalty', 0.2)),
                
                # Control parameter (1 parameter)
                'speed_diff_modifier': float(info.get('speed_diff_modifier', 0.0)),
                
                # Auction efficiency parameter (1 parameter)
                'max_participants_per_auction': int(info.get('max_participants_per_auction', 6)),
                
                # Safety parameters (2 parameters)
                'ignore_vehicles_go': float(info.get('ignore_vehicles_go', 25.0)),
                'ignore_vehicles_platoon_leader': float(info.get('ignore_vehicles_platoon_leader', 20.0)),
                
                # FIXED: Reward function parameters (not trainable)
                'vehicle_exit_reward': float(info.get('vehicle_exit_reward', 10.0)),
                'collision_penalty': float(info.get('collision_penalty', 100.0)),
                'deadlock_penalty': float(info.get('deadlock_penalty', 800.0)),
                'throughput_bonus': float(info.get('throughput_bonus', 0.01)),
                
                # FIXED: Conflict detection parameters (not trainable)
                'conflict_time_window': float(info.get('conflict_time_window', 2.5)),
                'min_safe_distance': float(info.get('min_safe_distance', 3.0)),
                'collision_threshold': float(info.get('collision_threshold', 2.0))
            }
            
            self.metrics_log.append(metrics)
            
            # Save every 1000 records
            if len(self.metrics_log) >= 1000:
                self._save_metrics()
                self.metrics_log = []
                
        except Exception as e:
            print(f"⚠️ SAC Metrics error: {e}")
    
        return True



    def _on_training_end(self):
        """Final save when SAC training ends"""
        self._save_metrics()
        print(f"✅ Final SAC metrics saved")

# System resource monitoring functions removed - no longer needed

def create_directories():
    """Create all necessary directories"""
    dirs = {
        'log_dir': "drl/logs",
        'checkpoint_dir': "drl/checkpoints", 
        'results_dir': "drl/results",
        'plots_dir': "drl/plots"
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"📁 {name}: {path}")
    
    return dirs

def main():
    print("🚀 Starting SAC Training for Traffic Intersection Control")
    print("=" * 70)
    
    # Create directories
    dirs = create_directories()
    
    # SAC parameters
    config = {
        'total_timesteps': 4000,
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 4,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'learning_starts': 1000,
        'use_sde': False,
        'policy_kwargs': dict(log_std_init=-3, net_arch=[256, 256]),
        'checkpoint_freq': 1000
        }
    
    training_success = False
    model = None
    
    try:
        print("🎯 Creating optimized SAC training environment...")
        env = AuctionGymEnv(sim_cfg={
            'max_steps': 400,  # Shorter episodes for SAC (off-policy can handle this)
            'training_mode': True,  # Enable performance optimizations
            # DISABLED: No mid-episode resets - clean episode termination for SAC DRL
            'deadlock_reset_enabled': False,  # Episodes terminate on deadlock
            'severe_deadlock_reset_enabled': False,  # Episodes terminate on severe deadlock
            'severe_deadlock_punishment': -300.0  # Punishment applied to final step only
        })
        print("✅ SAC Environment created successfully")

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
                        obs = np.array(obs, dtype=np.float32)
                    
                    # Ensure correct dimensions
                    if obs.shape[0] != 50:
                        if obs.shape[0] < 50:
                            # Pad with zeros
                            padding = np.zeros(50 - obs.shape[0], dtype=np.float32)
                            obs = np.concatenate([obs, padding])
                        else:
                            # Truncate
                            obs = obs[:50]
                    
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
        
        # Create optional action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = None  # SAC has built-in exploration, noise not usually needed
        
        # Create SAC model with OPTIMAL parameters
        print("🤖 Creating SAC model...")
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            tau=config['tau'],
            gamma=config['gamma'],
            train_freq=config['train_freq'],
            gradient_steps=config['gradient_steps'],
            ent_coef=config['ent_coef'],
            target_update_interval=config['target_update_interval'],
            learning_starts=config['learning_starts'],
            use_sde=config['use_sde'],
            policy_kwargs=config['policy_kwargs'],
            action_noise=action_noise,
            verbose=1
            # REMOVED: tensorboard_log=dirs['log_dir']
        )
        
        model.set_logger(logger)
        print("✅ SAC model created successfully")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=dirs['checkpoint_dir'],
            name_prefix="sac_traffic"
        )
        
        # Simple metrics callback
        metrics_callback = SimpleSACMetricsCallback(
            log_dir=dirs['results_dir'],
            verbose=0
        )
        
        # Start SAC training
        print(f"\n🎓 Starting SAC training for {config['total_timesteps']} timesteps...")
        print("Press Ctrl+C to stop training early")
        print("=" * 60)
        
        start_time = time.time()
        
        # FIXED: Add explicit SAC training termination check
        print(f"🎓 SAC Training will stop automatically at {config['total_timesteps']} timesteps")
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True
        )
        
        print(f"🏁 SAC Training COMPLETED - reached {config['total_timesteps']} timesteps")
        
        # Save final model
        final_model_path = os.path.join(dirs['checkpoint_dir'], "final_sac_model.zip")
        model.save(final_model_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ SAC Training completed in {elapsed_time:.2f} seconds")
        print(f"📁 Final SAC model saved to: {final_model_path}")
        
        training_success = True
        
    except KeyboardInterrupt:
        print("\n⚠️ SAC Training interrupted by user")
        try:
            if model is not None:
                interrupted_path = os.path.join(dirs['checkpoint_dir'], "interrupted_sac_model.zip")
                model.save(interrupted_path)
                print(f"💾 SAC Model saved to: {interrupted_path}")
        except Exception as save_error:
            print(f"❌ Could not save interrupted SAC model: {save_error}")
    
    except Exception as e:
        print(f"\n❌ SAC Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if hasattr(env, 'close'):
                env.close()
            print("🏁 SAC Environment closed")
            
            # Force garbage collection to clean up resources
            import gc
            gc.collect()
        except:
            pass
        
        # ALWAYS generate analysis plots (whether successful or interrupted)
        print("\n" + "=" * 70)
        print("📊 GENERATING SAC TRAINING ANALYSIS")
        print("=" * 70)
        
        try:
            # Create analyzer and generate plots
            analyzer = TrainingAnalyzer(dirs['results_dir'], dirs['plots_dir'])
            analyzer.generate_all_plots()
            analyzer.generate_report()
            analyzer.save_summary_json()
            
            print(f"\n✅ SAC Analysis complete! Check these locations:")
            print(f"   📊 Plots: {dirs['plots_dir']}")
            print(f"   📋 Report: {os.path.join(dirs['plots_dir'], 'training_report.txt')}")
            print(f"   📈 Metrics: {os.path.join(dirs['results_dir'], 'sac_training_metrics.csv')}")
            
            # Display summary statistics if available
            if analyzer.metrics_df is not None and len(analyzer.metrics_df) > 0:
                print(f"\n📈 SAC Training Summary:")
                print(f"   Steps completed: {analyzer.metrics_df['timestep'].max():,}")
                print(f"   Average throughput: {analyzer.metrics_df['throughput'].mean():.1f} vehicles/h")
                print(f"   Final bid scale: {analyzer.metrics_df['bid_scale'].iloc[-1]:.3f}")
                print(f"   Total data points: {len(analyzer.metrics_df)}")
            
        except Exception as analysis_error:
            print(f"❌ SAC Analysis failed: {analysis_error}")
            print(f"   You can still run analysis manually:")
            print(f"   python -c \"from drl.utils.analysis import quick_analysis; quick_analysis('{dirs['results_dir']}', '{dirs['plots_dir']}')\"")
        
        print(f"\n🏁 SAC Training session complete!")

if __name__ == "__main__":
    main()
