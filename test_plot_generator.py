#!/usr/bin/env python3
"""
Test script to verify plot generator works without font warnings
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

def create_test_data():
    """Create test data to verify plot generation"""
    print("🔧 Creating test data...")
    
    # Create test directories
    test_results_dir = "test_results"
    test_plots_dir = "test_plots"
    
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(test_plots_dir, exist_ok=True)
    
    # Create mock episode metrics data
    episodes = list(range(10))  # 10 episodes
    
    # Mock 4 action space parameters
    urgency_ratios = [1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 2.9, 2.6, 2.3]
    speed_modifiers = [-5.0, -2.0, 1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]
    max_participants = [4, 4, 5, 5, 6, 6, 5, 4, 5, 6]
    ignore_vehicles_go = [50, 52, 55, 58, 60, 62, 65, 68, 70, 72]
    
    # Create DataFrame
    test_data = {
        'episode': episodes,
        'urgency_position_ratio_mean': urgency_ratios,
        'speed_diff_modifier_mean': speed_modifiers,
        'max_participants_mean': max_participants,
        'ignore_vehicles_go_mean': ignore_vehicles_go,
        'total_vehicles_exited': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'total_collisions': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'avg_throughput': [120, 135, 150, 165, 180, 195, 210, 225, 240, 255],
        'avg_acceleration': [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to CSV
    csv_path = os.path.join(test_results_dir, 'episode_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Test data saved: {csv_path}")
    
    return test_results_dir, test_plots_dir

def test_plot_generation():
    """Test plot generation without font warnings"""
    print("🧪 Testing plot generation...")
    
    try:
        # Import plot generator
        from drl.utils.plot_generator import plot_training_metrics
        
        # Create test data
        test_results_dir, test_plots_dir = create_test_data()
        
        # Generate plots
        print("🎨 Generating test plots...")
        plot_training_metrics(test_results_dir, test_plots_dir, save_plots=True)
        
        print("✅ Plot generation test completed successfully!")
        print("📁 Check test_plots directory for generated files")
        
        # Clean up test files
        import shutil
        if os.path.exists("test_results"):
            shutil.rmtree("test_results")
        if os.path.exists("test_plots"):
            shutil.rmtree("test_plots")
        print("🧹 Test files cleaned up")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plot_generation()
