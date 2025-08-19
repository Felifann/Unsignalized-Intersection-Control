# rl/agents/ppo_trainer.py
import os
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
    """Custom callback to log training metrics"""
    
    def __init__(self, eval_env, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.metrics_log = []
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Log metrics every 100 steps
        if self.num_timesteps % 100 == 0:
            info = self.locals.get('infos', [{}])[0]
            
            metrics = {
                'timestep': self.num_timesteps,
                'throughput': info.get('throughput', 0.0),
                'avg_acceleration': info.get('avg_acceleration', 0.0),
                'collision_count': info.get('collision_count', 0),
                'total_controlled': info.get('total_controlled', 0),
                'vehicles_exited': info.get('vehicles_exited', 0),
                'bid_scale': info.get('bid_scale', 1.0)
            }
            
            self.metrics_log.append(metrics)
            
            # Save metrics every 1000 steps
            if self.num_timesteps % 1000 == 0:
                self._save_metrics()
        
        return True

    def _save_metrics(self):
        """Save metrics to CSV"""
        if self.metrics_log:
            df = pd.DataFrame(self.metrics_log)
            df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)

class PPOTrainer:
    """PPO trainer for traffic intersection environment"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.setup_directories()
        
        # Training parameters
        self.total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.eval_freq = self.config.get('eval_freq', 5000)
        self.checkpoint_freq = self.config.get('checkpoint_freq', 10000)
        
        print("🚀 PPO Trainer initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        default_config = {
            'total_timesteps': 1_000_000,
            'eval_freq': 5000,
            'checkpoint_freq': 10000,
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
            'sim_config': {'map': 'Town05', 'max_steps': 2000}
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
        
        # Setup logging
        logger = configure(self.log_dir, ["csv", "tensorboard"])
        
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
            verbose=1,
            tensorboard_log=self.log_dir
        )
        
        model.set_logger(logger)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_freq,
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
            
            # Generate analysis
            self.analyze_training()
            
        except KeyboardInterrupt:
            print("⚠️ Training interrupted by user")
            model.save(os.path.join(self.checkpoint_dir, "interrupted_model.zip"))
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
            print(f"✅ Analysis completed. Check {self.plots_dir} for plots")
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
