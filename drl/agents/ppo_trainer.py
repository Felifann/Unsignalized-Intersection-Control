# rl/agents/ppo_trainer.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

from drl.envs.auction_gym import AuctionGymEnv
from drl.utils.analysis import TrainingAnalyzer

class MetricsCallback(BaseCallback):
    """Enhanced callback to log training metrics with action space parameter tracking and per-episode statistics"""
    
    def __init__(self, eval_env, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
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
            
            self.episode_metrics.append(step_metrics)
            
            # Save step metrics every 1000 steps
            if self.num_timesteps % 1000 == 0:
                self._save_step_metrics()
                
        except Exception as e:
            print(f"⚠️ Metrics callback error: {e}")
        
        return True

    def _is_episode_reset(self) -> bool:
        """Check if episode reset occurred by looking for environment reset signals"""
        # This is a simple heuristic - in practice, you might want to use
        # more sophisticated episode boundary detection
        return len(self.episode_actions) > 0 and len(self.episode_actions) > 200

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
                
                # Action space parameter statistics
                'urgency_position_ratio_mean': float(action_means[0]),
                'urgency_position_ratio_var': float(action_vars[0]),
                'urgency_position_ratio_std': float(action_stds[0]),
                
                'speed_diff_modifier_mean': float(action_means[1]),
                'speed_diff_modifier_var': float(action_vars[1]),
                'speed_diff_modifier_std': float(action_stds[1]),
                
                'max_participants_mean': float(action_means[2]),
                'max_participants_var': float(action_vars[2]),
                'max_participants_std': float(action_stds[2]),
                
                'ignore_vehicles_go_mean': float(action_means[3]),
                'ignore_vehicles_go_var': float(action_vars[3]),
                'ignore_vehicles_go_std': float(action_stds[3]),
                
                # Episode performance metrics
                'total_vehicles_exited': episode_stats['total_exits'],
                'total_collisions': episode_stats['total_collisions'],
                'total_deadlocks': episode_stats['total_deadlocks'],
                'max_deadlock_severity': episode_stats['max_deadlock_severity'],
                'avg_throughput': episode_stats['avg_throughput'],
                'avg_acceleration': episode_stats['avg_acceleration'],
                'total_controlled_vehicles': episode_stats['total_controlled']
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
        
        return {
            'total_exits': total_exits,
            'total_collisions': total_collisions,
            'total_deadlocks': total_deadlocks,
            'max_deadlock_severity': max_severity,
            'avg_throughput': np.mean(throughputs) if throughputs else 0.0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'total_controlled': max(controlled) if controlled else 0
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

class PPOTrainer:
    """PPO trainer for traffic intersection environment"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.setup_directories()
        
        # Training parameters
        self.total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.eval_freq = self.config.get('eval_freq', 1000)
        self.checkpoint_freq = self.config.get('checkpoint_freq', 1000)
        
        print("🚀 PPO Trainer initialized")
        print("🔒 EPISODE-LEVEL PARAMETER UPDATES:")
        print("   • Action space parameters are updated ONLY at episode boundaries")
        print("   • During an episode, parameters remain constant for consistent behavior")
        print("   • This allows the agent to explore the consequences of its parameter choices")
        print("   • Parameters are cached and reused throughout each episode")

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        default_config = {
            'total_timesteps': 1_000_000,
            'eval_freq': 5000,
            'checkpoint_freq': 1000,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            # FIXED: Updated timing configuration for better vehicle control
            'sim_config': {
                'map': 'Town05', 
                'max_steps': 128,  # Reduced for faster training
                'fixed_delta_seconds': 0.1,  # 10 FPS simulation
                'logic_update_interval_seconds': 1.0,  # 1s decision intervals
                'auction_interval': 4.0,  # 4s auction cycles
                'bidding_duration': 2.0,  # 2s bidding phase
                'deadlock_check_interval': 8.0  # 8s system checks
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Merge with defaults
            default_config.update(config)
        
        return default_config

    def setup_directories(self):
        """Setup training directories"""
        self.base_dir = "drl"
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.best_model_dir = os.path.join(self.base_dir, "best_models")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        
        for directory in [self.log_dir, self.checkpoint_dir, self.best_model_dir, 
                         self.results_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)

    def create_env(self):
        """Create training environment"""
        return AuctionGymEnv(sim_cfg=self.config.get('sim_config', {}))

    def train(self):
        """Train PPO agent"""
        print("🎯 Starting PPO training...")
        
        # Create environments
        train_env = self.create_env()
        eval_env = self.create_env()
        
        # Setup logging WITHOUT TensorBoard
        logger = configure(self.log_dir, ["csv"])  # REMOVED: "tensorboard"
        
        # Create PPO model
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            verbose=1
            # REMOVED: tensorboard_log=self.log_dir
        )
        
        model.set_logger(logger)
        
        # Setup callbacks with reduced frequency to prevent file handle exhaustion
        # 减少检查点保存频率以避免文件句柄耗尽
        safe_checkpoint_freq = self.checkpoint_freq  # 最少2000步保存一次
        checkpoint_callback = CheckpointCallback(
            save_freq=safe_checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix="ppo_auction"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.results_dir,
            eval_freq=self.eval_freq,
            n_eval_episodes=5,
            deterministic=True
        )
        
        metrics_callback = MetricsCallback(eval_env, self.results_dir)
        
        # Train model
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=[checkpoint_callback, eval_callback, metrics_callback]
            )
            
            # Save final model
            final_model_path = os.path.join(self.checkpoint_dir, "final_ppo_model.zip")
            model.save(final_model_path)
            print(f"✅ Training completed. Final model saved to {final_model_path}")
            
            # Generate analysis ONLY at the end
            self.analyze_training()
            
        except KeyboardInterrupt:
            print("⚠️ Training interrupted by user")
            model.save(os.path.join(self.checkpoint_dir, "interrupted_model.zip"))
            # Generate analysis even if interrupted
            self.analyze_training()
        except Exception as e:
            print(f"❌ Training failed: {e}")
        finally:
            train_env.close()
            eval_env.close()

    def analyze_training(self):
        """Analyze training results and generate plots"""
        print("📊 Generating training analysis...")
        
        try:
            analyzer = TrainingAnalyzer(self.results_dir, self.plots_dir)
            analyzer.generate_all_plots()
            analyzer.generate_report()
            analyzer.save_summary_json()
            print(f"✅ Analysis completed. Check {self.plots_dir} for plots and reports")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")

    def load_and_test(self, model_path: str, num_episodes: int = 5):
        """Load trained model and test performance"""
        print(f"🧪 Testing model: {model_path}")
        
        try:
            model = PPO.load(model_path)
            test_env = self.create_env()
            
            episode_rewards = []
            episode_metrics = []
            
            for episode in range(num_episodes):
                obs = test_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                episode_metrics.append(info)
                
                print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                      f"Throughput={info.get('throughput', 0):.2f}")
            
            # Print summary
            avg_reward = np.mean(episode_rewards)
            avg_throughput = np.mean([m.get('throughput', 0) for m in episode_metrics])
            avg_acceleration = np.mean([m.get('avg_acceleration', 0) for m in episode_metrics])
            
            print(f"\n📈 Test Results Summary:")
            print(f"   Average Reward: {avg_reward:.2f}")
            print(f"   Average Throughput: {avg_throughput:.2f} vehicles/h")
            print(f"   Average Acceleration: {avg_acceleration:.3f} m/s²")
            
            test_env.close()
            return episode_rewards, episode_metrics
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return [], []

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO for traffic intersection')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--test', type=str, help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    
    args = parser.parse_args()
    
    trainer = PPOTrainer(args.config)
    
    if args.test:
        trainer.load_and_test(args.test, args.episodes)
    else:
        trainer.train()

if __name__ == "__main__":
    main()
