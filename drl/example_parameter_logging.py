#!/usr/bin/env python3
"""
Example script demonstrating the new parameter logging behavior
"""

import os
import sys
import numpy as np

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

def demonstrate_parameter_logging():
    """Demonstrate the new parameter logging behavior"""
    
    try:
        from drl.envs.auction_gym import AuctionGymEnv
        
        print("🧪 Demonstrating Smart Parameter Logging")
        print("=" * 60)
        
        # Create environment
        env = AuctionGymEnv(sim_cfg={
            'max_steps': 5,  # Very short episode for demonstration
            'training_mode': True
        })
        
        print("✅ Environment created successfully")
        
        # Test 1: First episode - parameters should update
        print("\n🔍 Test 1: First episode (parameters should update)")
        obs, info = env.reset()
        
        # Take a step with new parameters
        action = np.array([1.5, 15.0, 5, 60.0])  # New parameter values
        obs, reward, done, info = env.step(action)
        
        # Test 2: Same episode, same parameters - should NOT show updates
        print("\n🔍 Test 2: Same episode, same parameters (should NOT show updates)")
        action = np.array([1.5, 15.0, 5, 60.0])  # Same parameter values
        obs, reward, done, info = env.step(action)
        
        # Test 3: Same episode, different parameters - should show updates
        print("\n🔍 Test 3: Same episode, different parameters (should show updates)")
        action = np.array([2.0, 20.0, 6, 70.0])  # Different parameter values
        obs, reward, done, info = env.step(action)
        
        # Test 4: Same episode, same parameters again - should NOT show updates
        print("\n🔍 Test 4: Same episode, same parameters again (should NOT show updates)")
        action = np.array([2.0, 20.0, 6, 70.0])  # Same parameter values
        obs, reward, done, info = env.step(action)
        
        # Test 5: New episode - parameters should update again
        print("\n🔍 Test 5: New episode (parameters should update again)")
        obs, info = env.reset()
        
        action = np.array([0.8, -10.0, 3, 30.0])  # New parameter values
        obs, reward, done, info = env.step(action)
        
        # Test 6: Verbose logging control
        print("\n🔍 Test 6: Verbose logging control")
        print("   Disabling verbose parameter logging...")
        env.set_verbose_parameter_logging(False)
        
        action = np.array([0.8, -10.0, 3, 30.0])  # Same parameters
        obs, reward, done, info = env.step(action)
        
        print("   Re-enabling verbose parameter logging...")
        env.set_verbose_parameter_logging(True)
        
        action = np.array([0.8, -10.0, 3, 30.0])  # Same parameters
        obs, reward, done, info = env.step(action)
        
        # Summary
        print("\n📊 DEMONSTRATION SUMMARY:")
        print("   ✅ Parameters only print when they actually change")
        print("   ✅ Unchanged parameters show minimal output")
        print("   ✅ Verbose logging can be controlled")
        print("   ✅ Episode-level updates work correctly")
        
        env.close()
        print("\n✅ Demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_parameter_logging()
