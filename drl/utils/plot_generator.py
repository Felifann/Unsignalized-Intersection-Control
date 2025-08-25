"""
简化的图表生成工具 - 专注于参数趋势和安全指标可视化
可独立运行，从现有训练数据生成分析图表
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))

from drl.utils.analysis import TrainingAnalyzer, quick_analysis

def find_results_directories():
    """Find all possible results directories"""
    possible_dirs = [
        "drl/results",
        "drl/logs", 
        "logs",
        "results",
        "drl/checkpoints",
        "checkpoints"
    ]
    
    found_dirs = []
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            # Check if directory contains relevant files
            files = os.listdir(dir_path)
            if any(f.endswith('.csv') or f.endswith('.json') for f in files):
                found_dirs.append(dir_path)
    
    return found_dirs

def main():
    """Main function - Generate DRL training analysis charts"""
    parser = argparse.ArgumentParser(
        description='Generate DRL Training Analysis Charts - Focus on Parameter Trends and Safety Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python plot_generator.py --auto-find          # Auto-find and process all training results
  python plot_generator.py --results-dir drl/results --plots-dir my_plots
        """
    )
    parser.add_argument('--results-dir', type=str, 
                       help='Directory path containing training results')
    parser.add_argument('--plots-dir', type=str, 
                       help='Directory path to save charts')
    parser.add_argument('--auto-find', action='store_true', 
                       help='Automatically find all training results directories')
    
    args = parser.parse_args()
    
    print("🚀 DRL Training Analysis Chart Generator")
    print("=" * 50)
    print("Focus on: Parameter Trends, Collision Count, Deadlock Count")
    print("=" * 50)
    
    if args.auto_find:
        print("🔍 Auto-searching training results directories...")
        found_dirs = find_results_directories()
        
        if not found_dirs:
            print("❌ No directories containing training data found")
            print("   Please ensure training has been run and CSV data files generated")
            return
        
        print(f"✅ Found {len(found_dirs)} directories with data:")
        for i, dir_path in enumerate(found_dirs):
            print(f"   {i+1}. {dir_path}")
        
        # Generate charts for all found directories
        for dir_path in found_dirs:
            plots_dir = args.plots_dir or f"{dir_path}_analysis"
            print(f"\n📊 Processing: {dir_path} -> {plots_dir}")
            
            try:
                analyzer = TrainingAnalyzer(dir_path, plots_dir)
                analyzer.generate_all_plots()
                print(f"   ✅ Charts generated successfully")
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
    
    else:
        results_dir = args.results_dir or "drl/results"
        plots_dir = args.plots_dir or "drl/plots"
        
        print(f"📂 Training Results Directory: {results_dir}")
        print(f"📁 Charts Output Directory: {plots_dir}")
        
        if not os.path.exists(results_dir):
            print(f"❌ Directory does not exist: {results_dir}")
            print("   Tip: Use --auto-find to automatically search for training results")
            return
        
        try:
            analyzer = TrainingAnalyzer(results_dir, plots_dir)
            analyzer.generate_all_plots()
            print(f"\n✅ Chart generation completed!")
            print(f"   Please check: {plots_dir}")
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()