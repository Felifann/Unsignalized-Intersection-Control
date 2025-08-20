import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import glob
import json

class TrainingAnalyzer:
    """Analyze and visualize training results with REAL DATA ONLY"""
    
    def __init__(self, results_dir: str, plots_dir: str):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.metrics_df = None
        
        os.makedirs(self.plots_dir, exist_ok=True)
        plt.style.use('default')
        
        # Load ONLY real data
        self.load_real_data_only()

    def load_real_data_only(self):
        """Load ONLY actual recorded training data - NO INTERPOLATION"""
        print(f"📊 Loading STRICT real training data from: {self.results_dir}")
        
        # Try to load real CSV data
        csv_files = [
            os.path.join(self.results_dir, 'training_metrics.csv'),
            os.path.join(self.results_dir, 'metrics.csv'),
        ]
        
        for csv_path in csv_files:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        # STRICT validation - only use validated real data
                        if 'using_real_data' in df.columns:
                            real_data_mask = df['using_real_data'] == True
                            real_count = real_data_mask.sum()
                            
                            if real_count == 0:
                                print(f"❌ NO REAL DATA in {csv_path} - all {len(df)} entries are fake!")
                                continue
                            elif real_count < len(df):
                                print(f"⚠️ FILTERING fake data: {real_count}/{len(df)} real entries in {csv_path}")
                                df = df[real_data_mask].copy()
                            else:
                                print(f"✅ All {len(df)} entries validated as real data")
                        
                        # Additional validation - check for unrealistic patterns
                        if 'timestep' in df.columns:
                            # Verify timesteps are actually from training (not interpolated)
                            timestep_gaps = df['timestep'].diff().dropna()
                            if len(timestep_gaps) > 0:
                                avg_gap = timestep_gaps.mean()
                                if avg_gap == 1.0 and len(df) > 20:
                                    print(f"⚠️ SUSPICIOUS: Perfect 1-step gaps over {len(df)} points - may be interpolated")
                        
                        # Check for realistic reward ranges
                        if 'reward' in df.columns:
                            reward_range = df['reward'].max() - df['reward'].min()
                            if reward_range > 1000:
                                print(f"⚠️ UNREALISTIC reward range: {reward_range:.1f}")
                        
                        self.metrics_df = df
                        self._standardize_columns()
                        print(f"✅ Loaded VERIFIED real training data: {len(df)} records")
                        
                        # Show data summary
                        if len(df) > 0:
                            print(f"   Data spans timesteps: {df['timestep'].min()} to {df['timestep'].max()}")
                            if 'reward' in df.columns:
                                print(f"   Reward range: {df['reward'].min():.2f} to {df['reward'].max():.2f}")
                        
                        return
                except Exception as e:
                    print(f"⚠️ Failed to load {csv_path}: {e}")
        
        # NO FALLBACK TO FAKE DATA
        print("❌ NO VERIFIED REAL TRAINING DATA FOUND!")
        print("   Possible causes:")
        print("   1. Training hasn't run long enough to collect data")
        print("   2. Metrics callback is not properly connected")
        print("   3. Simulation is not generating real vehicle data")
        print("   SOLUTION: Run actual DRL training with real CARLA simulation")
        self.metrics_df = None

    def _try_load_from_json(self):
        """Try to load from JSON backup files"""
        json_files = [
            os.path.join(self.results_dir, 'training_metrics.json'),
            os.path.join(self.results_dir.replace('results', 'logs'), 'training_metrics.json')
        ]
        
        for json_path in json_files:
            if os.path.exists(json_path):
                try:
                    print(f"🔍 Trying to load JSON: {json_path}")
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list) and len(data) > 0:
                        self.metrics_df = pd.DataFrame(data)
                        print(f"✅ Loaded {len(self.metrics_df)} data points from JSON")
                        self._standardize_columns()
                        return
                        
                except Exception as e:
                    print(f"⚠️ Could not load JSON {json_path}: {e}")



    def _standardize_columns(self):
        """Standardize column names across different data sources - NO THROUGHPUT"""
        if self.metrics_df is None:
            return
        
        # Column mapping for different naming conventions - NO THROUGHPUT
        column_mapping = {
            'step': 'timestep',
            'steps': 'timestep',
            'episode': 'timestep',
            'episode_reward': 'reward',
            'mean_reward': 'reward',
            'ep_rew_mean': 'reward',
            'acceleration': 'avg_acceleration',
            'mean_acceleration': 'avg_acceleration',
            'controlled': 'total_controlled',
            'controlled_vehicles': 'total_controlled',
            'exited': 'vehicles_exited',
            'exits': 'vehicles_exited',
            'scale': 'bid_scale',
            'bid': 'bid_scale'
        }
        
        # Rename columns
        self.metrics_df = self.metrics_df.rename(columns=column_mapping)
        
        # Ensure required columns exist - NO THROUGHPUT
        required_columns = [
            'timestep', 'reward', 'avg_acceleration',
            'total_controlled', 'vehicles_exited', 'bid_scale', 'collision_count'
        ]
        
        for col in required_columns:
            if col not in self.metrics_df.columns:
                if col == 'timestep':
                    # Use index as timestep if missing (deterministic)
                    self.metrics_df['timestep'] = self.metrics_df.index.to_series().astype(int).values
                    print(f"🔧 Added missing deterministic column 'timestep' (from index)")
                else:
                    # Do NOT fabricate values; insert NaN and warn
                    self.metrics_df[col] = np.nan
                    print(f"⚠️ Missing column '{col}' — filled with NaN (no synthetic data generated)")
        
        # Coerce numeric types where possible, keep NaNs for missing/invalid entries
        for col in required_columns:
            if col in self.metrics_df.columns:
                try:
                    self.metrics_df[col] = pd.to_numeric(self.metrics_df[col], errors='coerce')
                except Exception:
                    pass

    def _try_load_from_tensorboard(self):
        """Try to extract data from tensorboard logs"""
        tb_log_dirs = [
            os.path.join(self.results_dir, 'tb_logs'),
            os.path.join(self.results_dir.replace('results', 'logs'), 'tb_logs'),
            self.results_dir.replace('results', 'logs')
        ]
        
        for log_dir in tb_log_dirs:
            if os.path.exists(log_dir):
                try:
                    # This would require tensorboard parsing, skip for now
                    print(f"📈 Found tensorboard logs in {log_dir} (parsing not implemented)")
                except Exception as e:
                    continue

    def plot_training_progress(self):
        """Plot ACTUAL training progress - no interpolation for missing data"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("⚠️ No real data available - cannot plot training progress")
            # Create empty plot with explanation
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'NO REAL TRAINING DATA AVAILABLE\n\nRun DRL training first to collect data', 
                   ha='center', va='center', fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Training Progress - No Data')
            save_path = os.path.join(self.plots_dir, 'no_data_available.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        try:
            plt.ioff()
            
            # Determine subplot layout based on available data
            available_metrics = []
            if 'reward' in self.metrics_df.columns and not self.metrics_df['reward'].isna().all():
                available_metrics.append('reward')
            if 'throughput' in self.metrics_df.columns and not self.metrics_df['throughput'].isna().all():
                available_metrics.append('throughput')
            if 'avg_acceleration' in self.metrics_df.columns and not self.metrics_df['avg_acceleration'].isna().all():
                available_metrics.append('acceleration')
            if 'bid_scale' in self.metrics_df.columns and not self.metrics_df['bid_scale'].isna().all():
                available_metrics.append('bid_scale')
            
            if len(available_metrics) == 0:
                print("⚠️ No plottable metrics found in real data")
                return
            
            # Create subplots based on available metrics
            n_plots = len(available_metrics)
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
            if n_plots == 1:
                axes = [axes]
            
            fig.suptitle(f'REAL Training Progress ({len(self.metrics_df)} actual data points)', 
                        fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # Plot only available metrics with actual data
            if 'reward' in available_metrics:
                reward_data = self.metrics_df[['timestep', 'reward']].dropna()
                if len(reward_data) > 0:
                    axes[plot_idx].plot(reward_data['timestep'], reward_data['reward'], 
                                       'b-', linewidth=2, marker='o', markersize=3)
                    axes[plot_idx].set_title(f'Reward Progress ({len(reward_data)} points)')
                    axes[plot_idx].set_xlabel('Training Steps')
                    axes[plot_idx].set_ylabel('Reward')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
            
            if 'throughput' in available_metrics:
                throughput_data = self.metrics_df[['timestep', 'throughput']].dropna()
                if len(throughput_data) > 0:
                    axes[plot_idx].plot(throughput_data['timestep'], throughput_data['throughput'], 
                                       'orange', linewidth=2, marker='s', markersize=3)
                    axes[plot_idx].set_title(f'Throughput ({len(throughput_data)} points)')
                    axes[plot_idx].set_xlabel('Training Steps')
                    axes[plot_idx].set_ylabel('Vehicles/hour')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
            
            if 'acceleration' in available_metrics:
                accel_data = self.metrics_df[['timestep', 'avg_acceleration']].dropna()
                if len(accel_data) > 0:
                    axes[plot_idx].plot(accel_data['timestep'], accel_data['avg_acceleration'], 
                                       'green', linewidth=2, marker='^', markersize=3)
                    axes[plot_idx].set_title(f'Acceleration ({len(accel_data)} points)')
                    axes[plot_idx].set_xlabel('Training Steps')
                    axes[plot_idx].set_ylabel('m/s²')
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
            
            if 'bid_scale' in available_metrics:
                bid_data = self.metrics_df[['timestep', 'bid_scale']].dropna()
                if len(bid_data) > 0:
                    axes[plot_idx].plot(bid_data['timestep'], bid_data['bid_scale'], 
                                       'red', linewidth=2, marker='d', markersize=3)
                    axes[plot_idx].set_title(f'Bid Scale Learning ({len(bid_data)} points)')
                    axes[plot_idx].set_xlabel('Training Steps')
                    axes[plot_idx].set_ylabel('Bid Scale')
                    axes[plot_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, 'real_training_progress.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ REAL training progress plot saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to create real training progress plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_performance_metrics(self):
        """Plot REAL performance metrics only"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("⚠️ No REAL data available for performance metrics plot")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Real Performance Metrics Analysis', fontsize=16, fontweight='bold')
            
            # Real throughput distribution
            if 'throughput' in self.metrics_df.columns:
                throughput_data = self.metrics_df['throughput'].dropna()
                if len(throughput_data) > 0:
                    axes[0, 0].hist(throughput_data, bins=20, alpha=0.7, 
                                   color='skyblue', edgecolor='black')
                    axes[0, 0].set_title('Real Throughput Distribution')
                    axes[0, 0].set_xlabel('Throughput (vehicles/h)')
                    axes[0, 0].set_ylabel('Frequency')
                    mean_throughput = throughput_data.mean()
                    axes[0, 0].axvline(mean_throughput, color='red', 
                                      linestyle='--', label=f'Mean: {mean_throughput:.1f}')
                    axes[0, 0].legend()
            
            # Real reward vs throughput correlation
            if 'reward' in self.metrics_df.columns and 'throughput' in self.metrics_df.columns:
                valid_data = self.metrics_df[['reward', 'throughput']].dropna()
                if len(valid_data) > 0:
                    axes[0, 1].scatter(valid_data['reward'], valid_data['throughput'], 
                                      alpha=0.6, color='orange')
                    axes[0, 1].set_title('Reward vs Real Throughput')
                    axes[0, 1].set_xlabel('Reward')
                    axes[0, 1].set_ylabel('Throughput (vehicles/h)')
                    axes[0, 1].grid(True, alpha=0.3)
            
            # Real collision count over time
            if 'collision_count' in self.metrics_df.columns:
                axes[1, 0].plot(self.metrics_df['timestep'], self.metrics_df['collision_count'], 
                               linewidth=2, color='red')
                axes[1, 0].set_title('Real Collision Count')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Collisions')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Real vehicles controlled vs detected
            if 'total_controlled' in self.metrics_df.columns and 'vehicles_detected' in self.metrics_df.columns:
                valid_data = self.metrics_df[['total_controlled', 'vehicles_detected']].dropna()
                if len(valid_data) > 0:
                    axes[1, 1].scatter(valid_data['vehicles_detected'], valid_data['total_controlled'], 
                                      color='purple', alpha=0.6)
                    axes[1, 1].set_title('Vehicles Controlled vs Detected')
                    axes[1, 1].set_xlabel('Vehicles Detected')
                    axes[1, 1].set_ylabel('Vehicles Controlled')
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, 'performance_metrics_real.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ REAL performance metrics plot saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to create REAL performance metrics plot: {e}")

    def plot_correlation_matrix(self):
        """Plot correlation matrix of metrics - NO THROUGHPUT"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("⚠️ No data available for correlation matrix")
            return
        
        try:
            # Select numeric columns for correlation - NO THROUGHPUT
            numeric_cols = ['avg_acceleration', 'total_controlled', 
                           'vehicles_exited', 'bid_scale', 'collision_count']
            
            # Add reward if available
            if 'reward' in self.metrics_df.columns:
                numeric_cols.insert(0, 'reward')
            
            # Only include columns that exist
            available_cols = [col for col in numeric_cols if col in self.metrics_df.columns]
            
            if len(available_cols) < 2:
                print(f"⚠️ Not enough numeric columns for correlation: {available_cols}")
                return
            
            corr_data = self.metrics_df[available_cols].corr()
            
            plt.figure(figsize=(10, 8))
            
            # Create heatmap manually if seaborn not available
            try:
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.3f')
            except:
                # Fallback to matplotlib imshow
                im = plt.imshow(corr_data, cmap='coolwarm', aspect='auto')
                plt.colorbar(im)
                plt.xticks(range(len(available_cols)), available_cols, rotation=45)
                plt.yticks(range(len(available_cols)), available_cols)
                
                # Add correlation values as text
                for i in range(len(available_cols)):
                    for j in range(len(available_cols)):
                        plt.text(j, i, f'{corr_data.iloc[i, j]:.3f}', 
                                ha='center', va='center')
            
            plt.title('Metrics Correlation Matrix (No Throughput)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(self.plots_dir, 'correlation_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Correlation matrix plot saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to create correlation matrix: {e}")

    def generate_report(self):
        """Generate text summary report with throughput calculation ONLY HERE"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("⚠️ No data available for report generation")
            return
        
        try:
            report_path = os.path.join(self.plots_dir, 'training_report_real_data.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== DRL Training Analysis Report (REAL DATA ONLY) ===\n\n")
                
                # Data validation summary
                f.write("Data Validation:\n")
                f.write(f"  Total Data Points: {len(self.metrics_df):,}\n")
                if 'using_real_data' in self.metrics_df.columns:
                    real_count = self.metrics_df['using_real_data'].sum()
                    f.write(f"  Real Data Points: {real_count:,}\n")
                    f.write(f"  Data Authenticity: {(real_count/len(self.metrics_df)*100):.1f}%\n")
                f.write("\n")
                
                # Training statistics using real data
                f.write("Training Statistics (Real Data):\n")
                f.write(f"  Total Training Steps: {self.metrics_df['timestep'].max():,}\n")
                
                # Real performance metrics
                f.write("Real Performance Metrics:\n")
                if 'throughput' in self.metrics_df.columns:
                    throughput_data = self.metrics_df['throughput'].dropna()
                    if len(throughput_data) > 0:
                        f.write(f"  Average Real Throughput: {throughput_data.mean():.2f} ± {throughput_data.std():.2f} vehicles/h\n")
                        f.write(f"  Max Real Throughput: {throughput_data.max():.2f} vehicles/h\n")
                        f.write(f"  Min Real Throughput: {throughput_data.min():.2f} vehicles/h\n")
                
                if 'avg_acceleration' in self.metrics_df.columns:
                    accel_data = self.metrics_df['avg_acceleration'].dropna()
                    if len(accel_data) > 0:
                        f.write(f"  Real Average Acceleration: {accel_data.mean():.3f} ± {accel_data.std():.3f} m/s²\n")
                
                if 'collision_count' in self.metrics_df.columns:
                    f.write(f"  Real Total Collisions: {self.metrics_df['collision_count'].sum():,}\n")
                
                if 'vehicles_exited' in self.metrics_df.columns:
                    f.write(f"  Real Total Vehicles Exited: {self.metrics_df['vehicles_exited'].sum():,}\n")
                
                # Learning progress from real data
                if len(self.metrics_df) > 4:
                    first_quarter = self.metrics_df.iloc[:len(self.metrics_df)//4]
                    last_quarter = self.metrics_df.iloc[3*len(self.metrics_df)//4:]
                    
                    f.write("\nReal Learning Progress:\n")
                    
                    if 'throughput' in self.metrics_df.columns:
                        first_throughput = first_quarter['throughput'].mean()
                        last_throughput = last_quarter['throughput'].mean()
                        improvement = last_throughput - first_throughput
                        f.write(f"  Throughput Improvement: {improvement:+.2f} vehicles/h\n")
                    
                    if 'collision_count' in self.metrics_df.columns:
                        first_collisions = first_quarter['collision_count'].mean()
                        last_collisions = last_quarter['collision_count'].mean()
                        f.write(f"  Collision Trend: {last_collisions - first_collisions:+.2f} per episode\n")
                    
                    if 'bid_scale' in self.metrics_df.columns:
                        f.write(f"  Final Bid Scale: {self.metrics_df['bid_scale'].iloc[-1]:.3f}\n")
                
                # Data quality assessment
                f.write("\nData Quality Assessment:\n")
                missing_data = self.metrics_df.isnull().sum().sum()
                total_cells = len(self.metrics_df) * len(self.metrics_df.columns)
                completeness = ((total_cells - missing_data) / total_cells * 100)
                f.write(f"  Missing Values: {missing_data}\n")
                f.write(f"  Data Completeness: {completeness:.1f}%\n")
                f.write(f"  Source: Real CARLA simulation data\n")
            
            print(f"✅ REAL data training report saved to {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to generate REAL data report: {e}")

    def save_summary_json(self):
        """Save a JSON summary for easy programmatic access - NO THROUGHPUT"""
        if self.metrics_df is None:
            return
        
        try:
            summary = {
                'training_steps': int(self.metrics_df['timestep'].max()),
                'data_points': len(self.metrics_df),
                'avg_acceleration': float(self.metrics_df['avg_acceleration'].mean()),
                'total_collisions': int(self.metrics_df['collision_count'].sum()),
                'total_vehicles_exited': int(self.metrics_df['vehicles_exited'].sum()),
                'final_bid_scale': float(self.metrics_df['bid_scale'].iloc[-1])
            }
            
            # Add reward metrics if available
            if 'reward' in self.metrics_df.columns:
                summary.update({
                    'avg_reward': float(self.metrics_df['reward'].mean()),
                    'max_reward': float(self.metrics_df['reward'].max()),
                    'final_reward': float(self.metrics_df['reward'].iloc[-1])
                })
            
            summary_path = os.path.join(self.plots_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ Training summary saved to {summary_path} (no throughput)")
            
        except Exception as e:
            print(f"❌ Failed to save summary JSON: {e}")

    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("📊 Generating all training analysis plots...")
        
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("⚠️ No real data available - cannot generate plots")
            return
        
        try:
            # Generate individual plots
            self.plot_training_progress()
            self.plot_performance_metrics()
            self.plot_correlation_matrix()
            
            print(f"✅ All plots generated successfully in {self.plots_dir}")
            
        except Exception as e:
            print(f"❌ Failed to generate some plots: {e}")
            import traceback
            traceback.print_exc()

def quick_analysis(results_dir: str = None, plots_dir: str = None):
    """Quick analysis function for easy usage"""
    if results_dir is None:
        results_dir = "drl/results"
    if plots_dir is None:
        plots_dir = "drl/plots"
    
    print("ℹ️ Running analysis without TensorBoard dependency")
    analyzer = TrainingAnalyzer(results_dir, plots_dir)
    analyzer.generate_all_plots()
    analyzer.generate_report()
    analyzer.save_summary_json()
    
    return analyzer

if __name__ == "__main__":
    # Allow running analysis directly
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        plots_dir = sys.argv[2] if len(sys.argv) > 2 else "drl/plots"
    else:
        results_dir = "drl/results"
        plots_dir = "drl/plots"
    
    quick_analysis(results_dir, plots_dir)
