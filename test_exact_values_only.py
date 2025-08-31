#!/usr/bin/env python3
"""
测试脚本：验证只使用精确值列的功能
"""

import os
import pandas as pd
import numpy as np

def create_test_data():
    """创建测试数据，只包含精确值列"""
    
    # 创建测试数据目录
    test_dir = "test_exact_values_only"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试数据
    episodes = list(range(1, 11))  # 10个episodes
    
    # 模拟真正的精确参数值（这些应该是环境中实际应用的参数）
    exact_params = {
        'urgency_position_ratio_exact': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        'speed_diff_modifier_exact': [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
        'max_participants_exact': [4, 4, 5, 5, 6, 6, 6, 5, 5, 4],
        'ignore_vehicles_go_exact': [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    }
    
    # 创建完整的测试数据
    test_data = []
    for i, episode in enumerate(episodes):
        row = {
            'episode': episode,
            'episode_start_step': i * 1000,
            'episode_end_step': (i + 1) * 1000,
            'episode_length': 1000,
            
            # 真正的精确值（环境中实际应用的参数）
            'urgency_position_ratio_exact': exact_params['urgency_position_ratio_exact'][i],
            'speed_diff_modifier_exact': exact_params['speed_diff_modifier_exact'][i],
            'max_participants_exact': exact_params['max_participants_exact'][i],
            'ignore_vehicles_go_exact': exact_params['ignore_vehicles_go_exact'][i],
            
            # 其他指标
            'total_vehicles_exited': 50 + i * 10,
            'total_collisions': max(0, 2 - i // 3),
            'avg_throughput': 200 + i * 20,
            'avg_acceleration': 1.5 + i * 0.1
        }
        test_data.append(row)
    
    # 保存测试数据
    test_csv_path = os.path.join(test_dir, 'episode_metrics.csv')
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"✅ 测试数据已创建: {test_csv_path}")
    print(f"   包含 {len(test_data)} 个episodes")
    print(f"   列数: {len(test_data[0])}")
    
    # 显示列名
    print(f"\n📊 数据列名:")
    for col in test_df.columns:
        print(f"   - {col}")
    
    # 验证精确值列
    print(f"\n🔍 验证精确值列:")
    for param in ['urgency_position_ratio_exact', 'speed_diff_modifier_exact', 'max_participants_exact', 'ignore_vehicles_go_exact']:
        if param in test_df.columns:
            values = test_df[param].values
            print(f"   {param}:")
            print(f"     范围: [{values.min():.3f}, {values.max():.3f}]")
            print(f"     平均值: {values.mean():.3f}")
            print(f"     标准差: {values.std():.3f}")
    
    return test_dir

def test_plot_generator():
    """测试修复后的plot_generator"""
    
    print("\n🧪 测试只使用精确值列的plot_generator...")
    
    try:
        # 导入修复后的plot_generator
        from drl.utils.plot_generator import plot_training_metrics
        
        # 创建测试目录
        test_dir = create_test_data()
        
        # 设置输出目录
        plots_dir = os.path.join(test_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 测试plot_training_metrics函数
        print(f"\n🎨 调用 plot_training_metrics...")
        plot_training_metrics(
            results_dir=test_dir,
            plots_dir=plots_dir,
            save_plots=True
        )
        
        print(f"\n✅ 测试完成！")
        print(f"   检查 {plots_dir} 目录中的生成文件")
        
        # 列出生成的文件
        if os.path.exists(plots_dir):
            files = os.listdir(plots_dir)
            print(f"\n📁 生成的文件:")
            for file in files:
                print(f"   - {file}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plot_generator()
