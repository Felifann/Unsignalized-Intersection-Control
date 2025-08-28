#!/usr/bin/env python3
"""
Test script to demonstrate enhanced DRL metrics logging
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

def test_metrics_logging():
    """Test the enhanced metrics logging functionality"""
    
    print("🧪 Testing Enhanced DRL Metrics Logging")
    print("=" * 60)
    
    try:
        from drl.utils.plot_generator import plot_training_metrics, generate_summary_report
        
        # Create test data directory
        test_results_dir = "test_metrics"
        test_plots_dir = "test_plots"
        os.makedirs(test_results_dir, exist_ok=True)
        os.makedirs(test_plots_dir, exist_ok=True)
        
        # Generate sample episode metrics
        print("📊 Generating sample episode metrics...")
        sample_episode_data = []
        
        for episode in range(1, 11):  # 10 episodes
            # Simulate action parameter evolution
            base_urgency = 1.0 + 0.1 * episode
            base_speed = -5.0 + episode * 2.0
            base_participants = 3 + (episode % 4)
            base_ignore = 30.0 + episode * 3.0
            
            # Add some noise and variance
            urgency_mean = base_urgency + np.random.normal(0, 0.1)
            urgency_std = 0.05 + np.random.uniform(0, 0.1)
            urgency_var = urgency_std ** 2
            
            speed_mean = base_speed + np.random.normal(0, 1.0)
            speed_std = 1.0 + np.random.uniform(0, 0.5)
            speed_var = speed_std ** 2
            
            participants_mean = base_participants + np.random.normal(0, 0.3)
            participants_std = 0.2 + np.random.uniform(0, 0.3)
            participants_var = participants_std ** 2
            
            ignore_mean = base_ignore + np.random.normal(0, 2.0)
            ignore_std = 1.0 + np.random.uniform(0, 1.0)
            ignore_var = ignore_std ** 2
            
            # Simulate performance metrics
            vehicles_exited = max(0, int(5 + episode * 2 + np.random.normal(0, 1)))
            collisions = int(np.random.poisson(0.1 * episode))  # More collisions in later episodes
            deadlocks = int(np.random.poisson(0.05 * episode))  # Some deadlocks
            throughput = 100 + episode * 20 + np.random.normal(0, 10)
            
            episode_data = {
                'episode': episode,
                'episode_start_step': (episode - 1) * 128,
                'episode_end_step': episode * 128,
                'episode_length': 128,
                
                # Action space parameter statistics
                'urgency_position_ratio_mean': round(urgency_mean, 3),
                'urgency_position_ratio_var': round(urgency_var, 6),
                'urgency_position_ratio_std': round(urgency_std, 3),
                
                'speed_diff_modifier_mean': round(speed_mean, 1),
                'speed_diff_modifier_var': round(speed_var, 2),
                'speed_diff_modifier_std': round(speed_std, 1),
                
                'max_participants_mean': round(participants_mean, 1),
                'max_participants_var': round(participants_var, 2),
                'max_participants_std': round(participants_std, 1),
                
                'ignore_vehicles_go_mean': round(ignore_mean, 1),
                'ignore_vehicles_go_var': round(ignore_var, 2),
                'ignore_vehicles_go_std': round(ignore_std, 1),
                
                # Episode performance metrics
                'total_vehicles_exited': vehicles_exited,
                'total_collisions': collisions,
                'total_deadlocks': deadlocks,
                'max_deadlock_severity': round(np.random.uniform(0, 0.3), 2),
                'avg_throughput': round(throughput, 1),
                'avg_acceleration': round(1.5 + np.random.normal(0, 0.5), 2),
                'total_controlled_vehicles': int(8 + np.random.normal(0, 2))
            }
            
            sample_episode_data.append(episode_data)
        
        # Save sample episode metrics
        episode_df = pd.DataFrame(sample_episode_data)
        episode_metrics_path = os.path.join(test_results_dir, 'episode_metrics.csv')
        episode_df.to_csv(episode_metrics_path, index=False)
        print(f"✅ Sample episode metrics saved: {episode_metrics_path}")
        
        # Generate sample step metrics
        print("📊 Generating sample step metrics...")
        sample_step_data = []
        
        for step in range(0, 1280, 10):  # Every 10 steps
            episode = (step // 128) + 1
            
            step_data = {
                'timestep': step,
                'episode': episode,
                'throughput': 100 + episode * 20 + np.random.normal(0, 10),
                'avg_acceleration': 1.5 + np.random.normal(0, 0.5),
                'collision_count': int(np.random.poisson(0.01)),
                'total_controlled': int(8 + np.random.normal(0, 2)),
                'vehicles_exited': int(step / 20 + np.random.normal(0, 1)),
                'urgency_position_ratio': 1.0 + 0.1 * episode + np.random.normal(0, 0.05),
                'speed_diff_modifier': -5.0 + episode * 2.0 + np.random.normal(0, 0.5),
                'max_participants_per_auction': 3 + (episode % 4),
                'ignore_vehicles_go': 30.0 + episode * 3.0 + np.random.normal(0, 1.0),
                'deadlocks_detected': int(np.random.poisson(0.005)),
                'deadlock_severity': round(np.random.uniform(0, 0.2), 3)
            }
            
            sample_step_data.append(step_data)
        
        # Save sample step metrics
        step_df = pd.DataFrame(sample_step_data)
        step_metrics_path = os.path.join(test_results_dir, 'step_metrics.csv')
        step_df.to_csv(step_metrics_path, index=False)
        print(f"✅ Sample step metrics saved: {step_metrics_path}")
        
        # Test plotting functionality
        print("\n🎨 Testing plotting functionality...")
        plot_training_metrics(test_results_dir, test_plots_dir, save_plots=True)
        generate_summary_report(test_results_dir, test_plots_dir)
        
        print(f"\n✅ Test completed successfully!")
        print(f"   📊 Sample data: {test_results_dir}")
        print(f"   🎨 Generated plots: {test_plots_dir}")
        print(f"   📋 Summary report: {os.path.join(test_plots_dir, 'training_summary.txt')}")
        
        # Clean up test files
        import shutil
        if os.path.exists(test_results_dir):
            shutil.rmtree(test_results_dir)
        if os.path.exists(test_plots_dir):
            shutil.rmtree(test_plots_dir)
        print("🧹 Test files cleaned up")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def show_metrics_structure():
    """Show the structure of the metrics being logged"""
    
    print("\n📋 Enhanced Metrics Structure")
    print("=" * 60)
    
    print("🎯 Action Space Parameters (4 trainable parameters):")
    print("   1. urgency_position_ratio: 紧急度vs位置优势关系因子 (0.1-3.0)")
    print("   2. speed_diff_modifier: 速度控制修正 (-30 to +30)")
    print("   3. max_participants_per_auction: 拍卖参与者数量 (3-6)")
    print("   4. ignore_vehicles_go: GO状态ignore_vehicles% (0-80%)")
    
    print("\n📊 Episode-Level Metrics:")
    print("   • Episode performance: vehicles_exited, collisions, deadlocks, throughput")
    print("   • Action parameter statistics: mean, variance, standard deviation")
    print("   • Episode metadata: length, start/end steps")
    
    print("\n📈 Step-Level Metrics:")
    print("   • Real-time performance: throughput, acceleration, collision_count")
    print("   • Current parameter values: all 4 trainable parameters")
    print("   • Safety metrics: deadlock detection and severity")
    
    print("\n💾 Output Files:")
    print("   • episode_metrics.csv: Episode-level summaries with action statistics")
    print("   • step_metrics.csv: Step-level detailed metrics")
    print("   • Generated plots: Performance trends and parameter evolution")
    print("   • training_summary.txt: Human-readable training summary")

if __name__ == "__main__":
    show_metrics_structure()
    test_metrics_logging()
