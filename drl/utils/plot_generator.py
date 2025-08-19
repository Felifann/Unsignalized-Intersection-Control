"""
Standalone plot generator for training analysis
Can be run independently to generate plots from existing data
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
    parser = argparse.ArgumentParser(description='Generate training analysis plots')
    parser.add_argument('--results-dir', type=str, help='Directory containing training results')
    parser.add_argument('--plots-dir', type=str, help='Directory to save plots')
    parser.add_argument('--auto-find', action='store_true', 
                       help='Automatically find results directories')
    
    args = parser.parse_args()
    
    if args.auto_find:
        print("🔍 Searching for results directories...")
        found_dirs = find_results_directories()
        
        if not found_dirs:
            print("❌ No results directories found")
            return
        
        print(f"📁 Found {len(found_dirs)} directories with data:")
        for i, dir_path in enumerate(found_dirs):
            print(f"   {i+1}. {dir_path}")
        
        # Generate plots for all found directories
        for dir_path in found_dirs:
            plots_dir = args.plots_dir or f"{dir_path}_plots"
            print(f"\n📊 Generating plots for {dir_path} -> {plots_dir}")
            
            try:
                quick_analysis(dir_path, plots_dir)
                print(f"✅ Plots generated successfully for {dir_path}")
            except Exception as e:
                print(f"❌ Failed to generate plots for {dir_path}: {e}")
    
    else:
        results_dir = args.results_dir or "drl/results"
        plots_dir = args.plots_dir or "drl/plots"
        
        print(f"📊 Generating plots: {results_dir} -> {plots_dir}")
        
        try:
            quick_analysis(results_dir, plots_dir)
            print("✅ Plots generated successfully")
        except Exception as e:
            print(f"❌ Failed to generate plots: {e}")

if __name__ == "__main__":
    main()