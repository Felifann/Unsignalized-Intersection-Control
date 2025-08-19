#!/usr/bin/env python3
"""
Simple DRL training script for traffic intersection control
Run this to start training immediately
"""

import os
import sys
import time
import glob
import numpy as np

# --- Prefer gymnasium if available, and make it available as 'gym' for legacy imports ---
try:
    import gymnasium as gym  # type: ignore
    sys.modules['gym'] = gym
except Exception:
    # Fall back to installed gym if gymnasium not present
    try:
        import gym  # type: ignore
    except Exception:
        pass

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

# Ensure CARLA Python egg is on sys.path before importing any env that imports carla
egg_candidates = []
egg_candidates += glob.glob(os.path.join(base_dir, "carla_l", "carla-*.egg"))
egg_candidates += glob.glob(os.path.join(base_dir, "carla_w", "carla-*.egg"))
# also allow top-level 'carla' folder (if present)
egg_candidates += glob.glob(os.path.join(base_dir, "carla", "carla-*.egg"))
if egg_candidates:
    egg_path = egg_candidates[0]
    if egg_path not in sys.path:
        sys.path.insert(0, egg_path)
else:
    print("Warning: CARLA egg not found under carla_l/carla_w/caral. Ensure CARLA PythonAPI egg is available if using real simulator envs.")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from drl.envs.auction_gym import AuctionGymEnv

def main():
    print("🚀 Starting DRL Training for Traffic Intersection Control")
    print("=" * 60)
    
    # Create directories
    log_dir = "drl/logs"
    checkpoint_dir = "drl/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training configuration - REDUCED FOR DEBUGGING
    config = {
        'total_timesteps': 5000,  # Reduced from 100000 for faster debugging
        'learning_rate': 3e-4,
        'n_steps': 256,  # Reduced from 512 for faster updates
        'batch_size': 32,  # Reduced from 64
        'n_epochs': 5,  # Reduced from 10
        'gamma': 0.99,
        'checkpoint_freq': 500  # Reduced from 5000 for more frequent saves
    }
    
    print(f"Configuration (DEBUG MODE - REDUCED TIMESTEPS):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Create environment
        print("🎯 Creating training environment...")
        env = AuctionGymEnv(sim_cfg={'max_steps': 1000})
        print("✅ Environment created successfully")

        # --- Compatibility wrapper: normalize reset/step outputs for Gym/Gymnasium differences ---
        class ResetStepCompatWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                out = self.env.reset(**kwargs)
                # Normalize to (obs, info)
                if isinstance(out, tuple):
                    if len(out) == 2:
                        return out
                    if len(out) == 1:
                        return out[0], {}
                    # collapse extra values: take first as obs, last dict-like as info if possible
                    obs = out[0]
                    info = out[-1] if isinstance(out[-1], dict) else {}
                    return obs, info
                else:
                    return out, {}

            def step(self, action):
                out = self.env.step(action)
                # If env.step returns 4-tuple (obs, reward, done, info) convert to Gymnasium style:
                if isinstance(out, tuple):
                    if len(out) == 4:
                        obs, reward, done, info = out
                        # Treat 'done' as terminated, set truncated=False
                        return obs, reward, done, False, info
                    # If already gymnasium-style (5-tuple), pass through
                return out

        env = ResetStepCompatWrapper(env)
        
        # Setup logging
        logger = configure(log_dir, ["csv", "tensorboard"])
        
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
            tensorboard_log=log_dir
        )
        
        model.set_logger(logger)
        print("✅ PPO model created successfully")
        
        # Setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=config['checkpoint_freq'],
            save_path=checkpoint_dir,
            name_prefix="ppo_traffic"
        )
        
        # Start training
        print(f"\n🎓 Starting training for {config['total_timesteps']} timesteps...")
        print("Press Ctrl+C to stop training early")
        print("=" * 60)
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
        model.save(final_model_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed_time:.2f} seconds")
        print(f"📁 Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        try:
            interrupted_path = os.path.join(checkpoint_dir, "interrupted_model.zip")
            model.save(interrupted_path)
            print(f"💾 Model saved to: {interrupted_path}")
        except:
            print("❌ Could not save interrupted model")
    
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

if __name__ == "__main__":
    main()
