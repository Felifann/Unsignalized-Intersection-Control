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

# --- Prefer gymnasium if available, and make it available as 'gym' for legacy imports ---
try:
    import gymnasium as gym  # type: ignore
    sys.modules['gym'] = gym
except Exception:
    try:
        import gym  # type: ignore
    except Exception:
        pass

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

class EnhancedMetricsCallback(BaseCallback):
    """Enhanced callback to collect comprehensive training metrics"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.metrics_log = []
        self.checkpoint_interval = 10  # Save more frequently for debugging
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file
        self.csv_path = os.path.join(log_dir, 'training_metrics.csv')
        self.csv_initialized = False
        print(f"📊 Enhanced metrics collection initialized, logging to {log_dir}")

    def _init_csv_headers(self):
        """Initialize CSV file with headers when we have actual data"""
        if not self.csv_initialized and self.metrics_log:
            try:
                # Use actual data to create headers
                headers = list(self.metrics_log[0].keys())
                df_header = pd.DataFrame(columns=headers)
                df_header.to_csv(self.csv_path, index=False)
                self.csv_initialized = True
                print(f"✅ Initialized CSV file with headers: {self.csv_path}")
            except Exception as e:
                print(f"⚠️ Failed to initialize CSV headers: {e}")

    def _on_step(self) -> bool:
        """Collect metrics on each step"""
        try:
            # Get info from the last step
            infos = self.locals.get('infos', [])
            if len(infos) > 0:
                info = infos[0] if isinstance(infos[0], dict) else {}
                
                # Get reward from the last step
                rewards = self.locals.get('rewards', [0.0])
                reward = float(rewards[0]) if len(rewards) > 0 else 0.0
                
                # Collect comprehensive metrics with safe access and type conversion
                metrics = {
                    'timestep': int(self.num_timesteps),
                    'reward': float(reward),
                    'throughput': float(info.get('throughput', 0.0)),
                    'avg_acceleration': float(info.get('avg_acceleration', 0.0)),
                    'collision_count': int(info.get('collision_count', 0)),
                    'total_controlled': int(info.get('total_controlled', 0)),
                    'vehicles_exited': int(info.get('vehicles_exited', 0)),
                    'auction_agents': int(info.get('auction_agents', 0)),
                    'deadlocks_detected': int(info.get('deadlocks_detected', 0)),
                    
                    # Policy parameters (with safe defaults)
                    'bid_scale': float(info.get('bid_scale', 1.0)),
                    'eta_weight': float(info.get('eta_weight', 1.0)),
                    'speed_weight': float(info.get('speed_weight', 0.3)),
                    'congestion_sensitivity': float(info.get('congestion_sensitivity', 0.4)),
                    'platoon_bonus': float(info.get('platoon_bonus', 0.5)),
                    'junction_penalty': float(info.get('junction_penalty', 0.2)),
                    'fairness_factor': float(info.get('fairness_factor', 0.1)),
                    'urgency_threshold': float(info.get('urgency_threshold', 5.0)),
                    'proximity_bonus_weight': float(info.get('proximity_bonus_weight', 1.0)),
                    'speed_diff_modifier': float(info.get('speed_diff_modifier', 0.0)),
                    'follow_distance_modifier': float(info.get('follow_distance_modifier', 0.0)),
                    'ignore_vehicles_go': float(info.get('ignore_vehicles_go', 50.0)),
                    'ignore_vehicles_wait': float(info.get('ignore_vehicles_wait', 0.0)),
                    'ignore_vehicles_platoon_leader': float(info.get('ignore_vehicles_platoon_leader', 50.0)),
                    'ignore_vehicles_platoon_follower': float(info.get('ignore_vehicles_platoon_follower', 90.0))
                }
                
                self.metrics_log.append(metrics)
                
                # Initialize CSV headers on first data
                if not self.csv_initialized:
                    self._init_csv_headers()
                
                # Save metrics more frequently for debugging
                if self.num_timesteps % self.checkpoint_interval == 0:
                    self._save_metrics()
                    if self.verbose > 0:
                        print(f"📊 Step {self.num_timesteps}: Throughput={metrics['throughput']:.1f}, "
                              f"Reward={metrics['reward']:.2f}, Controlled={metrics['total_controlled']}")
            
        except Exception as e:
            print(f"⚠️ Metrics collection error at step {self.num_timesteps}: {e}")
            import traceback
            traceback.print_exc()
        
        return True

    def _save_metrics(self):
        """Save metrics to CSV file with better error handling"""
        if not self.metrics_log:
            return
            
        try:
            df = pd.DataFrame(self.metrics_log)
            
            # Ensure all columns are properly typed
            numeric_columns = ['timestep', 'reward', 'throughput', 'avg_acceleration', 
                             'collision_count', 'total_controlled', 'vehicles_exited',
                             'auction_agents', 'deadlocks_detected', 'bid_scale', 
                             'eta_weight', 'speed_weight', 'congestion_sensitivity',
                             'platoon_bonus', 'junction_penalty', 'fairness_factor',
                             'urgency_threshold', 'proximity_bonus_weight', 
                             'speed_diff_modifier', 'follow_distance_modifier',
                             'ignore_vehicles_go', 'ignore_vehicles_wait',
                             'ignore_vehicles_platoon_leader', 'ignore_vehicles_platoon_follower']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Save to CSV (overwrite entire file to ensure consistency)
            df.to_csv(self.csv_path, index=False, float_format='%.6f')
            
            # Also save JSON backup for robustness
            json_path = os.path.join(self.log_dir, 'training_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(self.metrics_log, f, indent=2)
            
            print(f"📊 Saved {len(self.metrics_log)} metrics records to {self.csv_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save metrics: {e}")
            import traceback
            traceback.print_exc()

    def _on_training_end(self):
        """Final save when training ends"""
        self._save_metrics()
        print(f"✅ Final metrics saved with {len(self.metrics_log)} data points")

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
    print("🚀 Starting Enhanced DRL Training for Traffic Intersection Control")
    print("=" * 70)
    
    # Create directories
    dirs = create_directories()
    
    # Training configuration - 减少到很小用于调试
    config = {
        'total_timesteps': 400,  # 大幅减少用于调试
        'learning_rate': 3e-4,
        'n_steps': 64,   # 减少
        'batch_size': 16,  # 减少
        'n_epochs': 3,     # 减少
        'gamma': 0.99,
        'checkpoint_freq': 100
    }
    
    print(f"Configuration (DEBUG MODE):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    training_success = False
    model = None
    
    try:
        # Create environment
        print("🎯 Creating enhanced training environment...")
        env = AuctionGymEnv(sim_cfg={'max_steps': 200})  # 减少最大步数
        print("✅ Environment created successfully")

        # Compatibility wrapper
        class ResetStepCompatWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                out = self.env.reset(**kwargs)
                if isinstance(out, tuple):
                    if len(out) == 2:
                        return out
                    if len(out) == 1:
                        return out[0], {}
                    obs = out[0]
                    info = out[-1] if isinstance(out[-1], dict) else {}
                    return obs, info
                else:
                    return out, {}

            def step(self, action):
                out = self.env.step(action)
                if isinstance(out, tuple):
                    if len(out) == 4:
                        obs, reward, done, info = out
                        return obs, reward, done, False, info
                return out

        env = ResetStepCompatWrapper(env)
        
        # Setup enhanced logging
        logger = configure(dirs['log_dir'], ["csv", "tensorboard"])
        
        # Create PPO model
        print("🤖 Creating PPO model...")
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            verbose=1,
            tensorboard_log=dirs['log_dir']
        )
        
        model.set_logger(logger)
        print("✅ PPO model created successfully")
        
        # Setup enhanced callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=dirs['checkpoint_dir'],
            name_prefix="ppo_traffic"
        )
        
        # Enhanced metrics callback
        metrics_callback = EnhancedMetricsCallback(
            log_dir=dirs['results_dir'],
            verbose=1
        )
        
        # Start training
        print(f"\n🎓 Starting DEBUG training for {config['total_timesteps']} timesteps...")
        print("Press Ctrl+C to stop training early")
        print("=" * 60)
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[checkpoint_callback, metrics_callback],
            progress_bar=True
        )
        
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
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
            print("🏁 Environment closed")
        except:
            pass
        
        # ALWAYS generate analysis plots (whether successful or interrupted)
        print("\n" + "=" * 70)
        print("📊 GENERATING TRAINING ANALYSIS")
        print("=" * 70)
        
        try:
            # Create analyzer and generate plots
            analyzer = TrainingAnalyzer(dirs['results_dir'], dirs['plots_dir'])
            analyzer.generate_all_plots()
            analyzer.generate_report()
            analyzer.save_summary_json()
            
            print(f"\n✅ Analysis complete! Check these locations:")
            print(f"   📊 Plots: {dirs['plots_dir']}")
            print(f"   📋 Report: {os.path.join(dirs['plots_dir'], 'training_report.txt')}")
            print(f"   📈 Metrics: {os.path.join(dirs['results_dir'], 'training_metrics.csv')}")
            
            # Display summary statistics if available
            if analyzer.metrics_df is not None and len(analyzer.metrics_df) > 0:
                print(f"\n📈 Training Summary:")
                print(f"   Steps completed: {analyzer.metrics_df['timestep'].max():,}")
                print(f"   Average throughput: {analyzer.metrics_df['throughput'].mean():.1f} vehicles/h")
                print(f"   Final bid scale: {analyzer.metrics_df['bid_scale'].iloc[-1]:.3f}")
                print(f"   Total data points: {len(analyzer.metrics_df)}")
            
        except Exception as analysis_error:
            print(f"❌ Analysis failed: {analysis_error}")
            print(f"   You can still run analysis manually:")
            print(f"   python -c \"from drl.utils.analysis import quick_analysis; quick_analysis('{dirs['results_dir']}', '{dirs['plots_dir']}')\"")
        
        print(f"\n🏁 Training session complete!")

if __name__ == "__main__":
    main()
