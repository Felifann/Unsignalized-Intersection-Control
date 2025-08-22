#!/usr/bin/env python3
"""
Quick start script for DRL training
Usage: python run_drl_training.py
"""

import subprocess
import sys

def main():
    print("🚀 Starting DRL Training for Traffic Intersection Control")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 
            "drl/train.py"
        ], check=True)
        
        print("✅ Training completed successfully")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
