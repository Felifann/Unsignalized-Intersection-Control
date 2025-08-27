#!/usr/bin/env python3
"""
Simple test script to verify DRL environment reset works
"""

import os
import sys
import numpy as np

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

# Test gym import
try:
    import gymnasium as gym
    print("✅ Using gymnasium")
    sys.modules['gym'] = gym
except Exception:
    try:
        import gym
        print("✅ Using legacy gym")
    except Exception:
        print("❌ No gym or gymnasium found")
        sys.exit(1)

def test_environment_creation():
    """Test if environment can be created without errors"""
    try:
        print("🔧 Testing environment creation...")
        
        # Import the environment
        from drl.envs.auction_gym import AuctionGymEnv
        
        # Create environment with minimal config
        env = AuctionGymEnv(sim_cfg={
            'max_steps': 100,
            'training_mode': True,
            'deadlock_reset_enabled': False,
            'severe_deadlock_reset_enabled': False
        })
        
        print("✅ Environment created successfully")
        return env
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment_reset(env):
    """Test if environment reset works"""
    if env is None:
        return False
        
    try:
        print("🔄 Testing environment reset...")
        
        # Test reset
        obs, info = env.reset(seed=42)
        
        print(f"✅ Reset successful!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_step(env):
    """Test if environment step works"""
    if env is None:
        return False
        
    try:
        print("🚶 Testing environment step...")
        
        # Create a random action
        action = np.random.uniform(
            low=env.action_space.low,
            high=env.action_space.high,
            size=env.action_space.shape
        )
        
        # Test step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"✅ Step successful!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        print(f"   Info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Testing DRL Environment Fixes")
    print("=" * 50)
    
    # Test 1: Environment creation
    env = test_environment_creation()
    if env is None:
        print("❌ Cannot proceed without environment")
        return
    
    # Test 2: Environment reset
    reset_success = test_environment_reset(env)
    if not reset_success:
        print("❌ Reset test failed")
        return
    
    # Test 3: Environment step
    step_success = test_environment_step(env)
    if not step_success:
        print("❌ Step test failed")
        return
    
    print("\n🎉 All tests passed! DRL environment is working correctly.")
    
    # Clean up
    try:
        env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"⚠️ Environment close warning: {e}")

if __name__ == "__main__":
    main()

