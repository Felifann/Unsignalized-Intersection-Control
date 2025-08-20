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
    """Enhanced callback with STRICT real-only data collection - NO FAKE DATA"""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.metrics_log = []
        self.checkpoint_interval = 10
        self.max_metrics_memory = 1000
        self.last_recorded_timestep = -1  # Track actual recorded steps
        
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'training_metrics.csv')
        self.csv_initialized = False
        print(f"📊 STRICT real-only metrics collection initialized")

        # FIXED throughput caching
        self.throughput_interval = 50
        self.cached_throughput = 0.0
        self.last_throughput_calc_step = -1

    def _on_step(self) -> bool:
        """Collect ONLY actual step data - NO INTERPOLATION OR FAKE DATA"""
        try:
            # CRITICAL: Only record if this is an actual new timestep
            if self.num_timesteps <= self.last_recorded_timestep:
                return True  # Skip duplicate or out-of-order steps
            
            infos = self.locals.get('infos', [])
            if len(infos) == 0:
                return True  # No info to record
                
            info = infos[0] if isinstance(infos[0], dict) else {}
            
            # STRICT validation - reject all non-real data
            using_real_data = info.get('using_real_data', False)
            data_source = info.get('data_source', 'unknown')
            
            if not using_real_data:
                if self.verbose > 0:
                    print(f"❌ STEP {self.num_timesteps}: REJECTING NON-REAL DATA! Source: {data_source}")
                return True
            
            # Extract ONLY validated real metrics
            vehicles_exited = max(0, min(int(info.get('vehicles_exited', 0)), 10000))
            sim_time_elapsed = max(0.0, min(float(info.get('sim_time_elapsed', 0.0)), 86400.0))
            vehicles_detected = max(0, min(int(info.get('vehicles_detected', 0)), 1000))
            
            # ZERO-TOLERANCE for fake rewards
            reward_value = float(self.locals.get('rewards', [0.0])[0])
            if abs(reward_value) > 500:  # Sanity check for realistic rewards
                print(f"⚠️ STEP {self.num_timesteps}: Suspicious reward value {reward_value} - capping")
                reward_value = np.clip(reward_value, -100.0, 100.0)
            
            # Calculate throughput ONLY from real data
            real_throughput = 0.0
            try:
                should_recalc = (self.num_timesteps % self.throughput_interval == 0) or (self.last_throughput_calc_step < 0)
                if should_recalc and sim_time_elapsed > 0.1 and vehicles_exited >= 0:  # Require meaningful sim time
                    computed = (vehicles_exited / sim_time_elapsed) * 3600.0
                    # STRICT bounds for realistic throughput
                    real_throughput = max(0.0, min(float(computed), 3600.0))  # Max 1 vehicle/second
                    self.cached_throughput = real_throughput
                    self.last_throughput_calc_step = int(self.num_timesteps)
                else:
                    real_throughput = float(self.cached_throughput)
            except Exception:
                real_throughput = float(self.cached_throughput)
            
            # Build ONLY validated metrics record
            metrics = {
                'timestep': int(self.num_timesteps),  # Actual timestep from training
                'reward': np.clip(reward_value, -1000.0, 1000.0),
                
                # VALIDATED simulation metrics with strict bounds
                'throughput': real_throughput,
                'avg_acceleration': np.clip(float(info.get('avg_acceleration', 0.0)), -20.0, 20.0),
                'collision_count': max(0, min(int(info.get('collision_count', 0)), 50)),  # Fixed wrong variable
                'total_controlled': max(0, min(int(info.get('total_controlled', 0)), 1000)),
                'vehicles_exited': vehicles_exited,
                'vehicles_detected': vehicles_detected,
                'vehicles_in_junction': max(0, min(int(info.get('vehicles_in_junction', 0)), 100)),
                'auction_agents': max(0, min(int(info.get('auction_agents', 0)), 100)),
                'deadlocks_detected': max(0, min(int(info.get('deadlocks_detected', 0)), 50)),
                'go_vehicles': max(0, min(int(info.get('go_vehicles', 0)), 100)),
                'waiting_vehicles': max(0, min(int(info.get('waiting_vehicles', 0)), 100)),
                
                'sim_time_elapsed': sim_time_elapsed,
                'using_real_data': using_real_data,
                'data_source': data_source,
                
                # Training parameters with validation
                'bid_scale': np.clip(float(info.get('bid_scale', 1.0)), 0.1, 5.0),
                'eta_weight': np.clip(float(info.get('eta_weight', 1.0)), 0.5, 3.0),
                'speed_weight': np.clip(float(info.get('speed_weight', 0.3)), 0.0, 1.0),
                'congestion_sensitivity': np.clip(float(info.get('congestion_sensitivity', 0.4)), 0.0, 1.0),
                'platoon_bonus': np.clip(float(info.get('platoon_bonus', 0.5)), 0.0, 2.0),
                'junction_penalty': np.clip(float(info.get('junction_penalty', 0.2)), 0.0, 1.0),
                'fairness_factor': np.clip(float(info.get('fairness_factor', 0.1)), 0.0, 0.5),
                'urgency_threshold': np.clip(float(info.get('urgency_threshold', 5.0)), 1.0, 10.0),
                'proximity_bonus_weight': np.clip(float(info.get('proximity_bonus_weight', 1.0)), 0.0, 3.0),
                'speed_diff_modifier': np.clip(float(info.get('speed_diff_modifier', 0.0)), -30.0, 30.0),
                'follow_distance_modifier': np.clip(float(info.get('follow_distance_modifier', 0.0)), -2.0, 3.0),
                'ignore_vehicles_go': np.clip(float(info.get('ignore_vehicles_go', 50.0)), 0.0, 100.0),
                'ignore_vehicles_wait': np.clip(float(info.get('ignore_vehicles_wait', 0.0)), 0.0, 50.0),
                'ignore_vehicles_platoon_leader': np.clip(float(info.get('ignore_vehicles_platoon_leader', 50.0)), 0.0, 80.0),
                'ignore_vehicles_platoon_follower': np.clip(float(info.get('ignore_vehicles_platoon_follower', 90.0)), 50.0, 100.0)
            }
            
            # RECORD the actual step
            self.metrics_log.append(metrics)
            self.last_recorded_timestep = int(self.num_timesteps)
            
            # Memory management without data loss
            if len(self.metrics_log) > self.max_metrics_memory:
                overflow = self.metrics_log[:-self.max_metrics_memory//2]
                self.metrics_log = self.metrics_log[-self.max_metrics_memory//2:]
                self._save_overflow_metrics(overflow)
            
            # Save frequently
            if self.num_timesteps % 10 == 0:  # More frequent saves for debugging
                self._save_metrics()
                    
            if self.verbose > 0 and self.num_timesteps % 5 == 0:  # More frequent logging for debugging
                print(f"📊 REAL STEP {self.num_timesteps}: "
                      f"Vehicles={vehicles_detected}, "
                      f"Exits={vehicles_exited}, "
                      f"Reward={reward_value:.2f}")
        
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

    def _save_overflow_metrics(self, overflow_data):
        """Save overflow metrics to prevent memory issues"""
        try:
            overflow_path = os.path.join(self.log_dir, f'overflow_metrics_{self.num_timesteps}.json')
            with open(overflow_path, 'w') as f:
                json.dump(overflow_data, f)
            print(f"💾 Saved {len(overflow_data)} overflow metrics to {overflow_path}")
        except Exception as e:
            print(f"⚠️ Failed to save overflow metrics: {e}")

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
        'total_timesteps': 4000,  # 大幅减少用于调试
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
        
        # Setup logging WITHOUT TensorBoard
        logger = configure(dirs['log_dir'], ["csv"])  # REMOVED: "tensorboard"
        
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
            verbose=1
            # REMOVED: tensorboard_log=dirs['log_dir']
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
