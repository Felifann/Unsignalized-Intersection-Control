#!/usr/bin/env python3
"""
Simple script to run SAC training for traffic intersection control
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SAC training for traffic intersection')
    parser.add_argument('--config', type=str, default='drl/configs/sac_config.yaml',
                       help='Path to SAC config file')
    parser.add_argument('--timesteps', type=int, default=5000,
                       help='Number of training timesteps')
    parser.add_argument('--test', type=str, help='Path to trained model for testing')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes for testing')
    
    args = parser.parse_args()
    
    print("🚀 SAC Training for Traffic Intersection Control")
    print("=" * 50)
    
    if args.test:
        print(f"🧪 Testing SAC model: {args.test}")
        print(f"   Episodes: {args.episodes}")
        
        # Run testing using SAC trainer
        cmd = [
            sys.executable, 
            "drl/agents/sac_trainer.py", 
            "--test", args.test,
            "--episodes", str(args.episodes)
        ]
        
        if args.config:
            cmd.extend(["--config", args.config])
            
    else:
        print(f"🎯 Starting SAC training...")
        print(f"   Config: {args.config}")
        print(f"   Timesteps: {args.timesteps:,}")
        
        # Check if config file exists
        if not os.path.exists(args.config):
            print(f"⚠️ Config file not found: {args.config}")
            print("   Using default parameters...")
        
        # Run training using the new SAC training script
        cmd = [sys.executable, "drl/train_sac.py"]
    
    print(f"\n🔧 Running command:")
    print(f"   {' '.join(cmd)}")
    print("\n" + "=" * 50)
    
    try:
        # Run the training/testing
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Process completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Process failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
