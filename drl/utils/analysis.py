"""
Simplified Training Analyzer - Focus on Parameter Trends and Safety Metrics Visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import glob

class TrainingAnalyzer:
    """Simplified training analyzer focused on parameter trends and safety metrics"""
    
    def __init__(self, results_dir: str, plots_dir: str):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
        # Create directories if they don't exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize data containers
        self.metrics_df = None
        
        # Set matplotlib style - use simple built-in style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"📊 Training Analyzer initialized")
        print(f"   Results directory: {results_dir}")
        print(f"   Plots directory: {plots_dir}")

    def _find_csv_files(self) -> List[str]:
        """Find all training data CSV files"""
        csv_patterns = [
            os.path.join(self.results_dir, 'training_metrics.csv'),
            os.path.join(self.results_dir, 'sac_training_metrics.csv'),
            os.path.join(self.results_dir, '*.csv')
        ]
        
        csv_files = []
        for pattern in csv_patterns:
            csv_files.extend(glob.glob(pattern))
        
        # Remove duplicates and return
        return list(set(csv_files))

    def load_data(self) -> bool:
        """Load training data"""
        csv_files = self._find_csv_files()
        
        if not csv_files:
            print("❌ No training data CSV files found")
            return False
        
        print(f"📂 Found {len(csv_files)} CSV files")
        
        # Read and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    dataframes.append(df)
                    print(f"   ✅ Loaded: {os.path.basename(csv_file)} ({len(df)} rows)")
                else:
                    print(f"   ⚠️ Empty file: {os.path.basename(csv_file)}")
            except Exception as e:
                print(f"   ❌ Failed to load: {os.path.basename(csv_file)} - {e}")
        
        if not dataframes:
            print("❌ All CSV files failed to load")
            return False
        
        # Merge data
        self.metrics_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates and sort by timestep
        if 'timestep' in self.metrics_df.columns:
            self.metrics_df = self.metrics_df.drop_duplicates(subset=['timestep'])
            self.metrics_df = self.metrics_df.sort_values('timestep')
        
        print(f"✅ Data loading completed: {len(self.metrics_df)} rows, {len(self.metrics_df.columns)} columns")
        return True

    def plot_parameter_trends(self):
        """Plot training parameter trend charts"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("❌ No data available for parameter trend plotting")
            return
        
        # Define parameter groups to plot
        param_groups = {
            'Bidding Strategy Parameters': {
                'bid_scale': 'Bid Scale Factor',
                'eta_weight': 'ETA Weight',
                'speed_weight': 'Speed Weight',
                'congestion_sensitivity': 'Congestion Sensitivity'
            },
            'Platoon and Fairness Parameters': {
                'platoon_bonus': 'Platoon Bonus',
                'junction_penalty': 'Junction Penalty',
                'fairness_factor': 'Fairness Factor',
                'urgency_threshold': 'Urgency Threshold'
            },
            'Control Parameters': {
                'speed_diff_modifier': 'Speed Diff Modifier',
                'follow_distance_modifier': 'Follow Distance Modifier',
                'ignore_vehicles_go': 'Ignore Vehicles GO %',
                'ignore_vehicles_wait': 'Ignore Vehicles WAIT %'
            },
            'Platoon Ignore Vehicle Parameters': {
                'ignore_vehicles_platoon_leader': 'Platoon Leader Ignore Vehicles %',
                'ignore_vehicles_platoon_follower': 'Platoon Follower Ignore Vehicles %'
            }
        }
        
        for group_name, params in param_groups.items():
            self._plot_parameter_group(group_name, params)
    
    def _plot_parameter_group(self, group_name: str, params: Dict[str, str]):
        """Plot parameter group trend chart"""
        # Check which parameters exist in the data
        available_params = {key: name for key, name in params.items() 
                          if key in self.metrics_df.columns and not self.metrics_df[key].isna().all()}
        
        if not available_params:
            print(f"⚠️ {group_name}: No available parameter data")
            return
        
        # Create subplots
        n_params = len(available_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        if n_params == 1:
            axes = [axes]
        
        fig.suptitle(f'{group_name} Trends', fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", n_params)
        
        for idx, (param_key, param_name) in enumerate(available_params.items()):
            # Get valid data points
            valid_data = self.metrics_df[['timestep', param_key]].dropna()
            
            if len(valid_data) > 0:
                axes[idx].plot(valid_data['timestep'], valid_data[param_key], 
                             color=colors[idx], linewidth=2, marker='o', markersize=3)
                axes[idx].set_title(f'{param_name} ({len(valid_data)} data points)')
                axes[idx].set_xlabel('Training Steps')
                axes[idx].set_ylabel('Parameter Value')
                axes[idx].grid(True, alpha=0.3)
                
                # Add trend line
                if len(valid_data) > 2:
                    z = np.polyfit(valid_data['timestep'], valid_data[param_key], 1)
                    p = np.poly1d(z)
                    axes[idx].plot(valid_data['timestep'], p(valid_data['timestep']), 
                                 "--", color='red', alpha=0.8, linewidth=1)
        
        plt.tight_layout()
        
        # Save chart
        safe_group_name = group_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(self.plots_dir, f'parameter_trends_{safe_group_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {group_name} trend chart saved: {save_path}")

    def plot_safety_metrics(self):
        """Plot ENHANCED safety metrics chart with detailed deadlock statistics"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("❌ No data available for safety metrics plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Safety Metrics with Deadlock Analysis', fontsize=16, fontweight='bold')
        
        # 1. Collision count trend  
        if 'collision_count' in self.metrics_df.columns:
            collision_data = self.metrics_df[['timestep', 'collision_count']].dropna()
            if len(collision_data) > 0:
                axes[0, 0].plot(collision_data['timestep'], collision_data['collision_count'], 
                               'r-', linewidth=2, marker='o', markersize=3)
                axes[0, 0].set_title(f'Collision Count Trend ({len(collision_data)} data points)')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Cumulative Collisions')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Enhanced statistics info
                total_collisions = collision_data['collision_count'].max()
                collision_rate = total_collisions / len(collision_data) if len(collision_data) > 0 else 0
                axes[0, 0].text(0.02, 0.98, 
                               f'Total Collisions: {total_collisions}\nCollision Rate: {collision_rate:.4f}/step', 
                               transform=axes[0, 0].transAxes, fontsize=10,
                               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.7))
        
        # 2. ENHANCED Deadlock count trend with detailed stats
        if 'deadlocks_detected' in self.metrics_df.columns:
            deadlock_data = self.metrics_df[['timestep', 'deadlocks_detected']].dropna()
            if len(deadlock_data) > 0:
                axes[0, 1].plot(deadlock_data['timestep'], deadlock_data['deadlocks_detected'], 
                               'orange', linewidth=2, marker='s', markersize=3)
                
                # Calculate deadlock statistics
                total_deadlocks = deadlock_data['deadlocks_detected'].max()
                deadlock_rate = total_deadlocks / len(deadlock_data) if len(deadlock_data) > 0 else 0
                
                # Calculate deadlock increments (new deadlocks per step)
                deadlock_increments = deadlock_data['deadlocks_detected'].diff().fillna(0)
                new_deadlocks_count = (deadlock_increments > 0).sum()
                avg_deadlock_gap = len(deadlock_data) / max(1, new_deadlocks_count)
                
                axes[0, 1].set_title(f'Deadlock Analysis ({len(deadlock_data)} data points)')
                axes[0, 1].set_xlabel('Training Steps') 
                axes[0, 1].set_ylabel('Cumulative Deadlocks')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Enhanced deadlock statistics
                stats_text = f"""Total Deadlocks: {total_deadlocks}
Deadlock Rate: {deadlock_rate:.4f}/step
Episodes with Deadlocks: {new_deadlocks_count}
Avg Steps Between: {avg_deadlock_gap:.1f}"""
                
                axes[0, 1].text(0.02, 0.98, stats_text,
                               transform=axes[0, 1].transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle="round", facecolor='orange', alpha=0.7))
        
        # 3. ENHANCED Deadlock severity analysis
        if 'deadlock_severity' in self.metrics_df.columns:
            severity_data = self.metrics_df['deadlock_severity'].dropna()
            if len(severity_data) > 0:
                # Create histogram with enhanced analysis
                n, bins, patches = axes[1, 0].hist(severity_data, bins=20, alpha=0.7, color='purple', edgecolor='black')
                
                # Color code the histogram bars by severity level
                for i, patch in enumerate(patches):
                    bin_center = (bins[i] + bins[i+1]) / 2
                    if bin_center >= 0.8:
                        patch.set_color('darkred')  # Critical
                    elif bin_center >= 0.6:
                        patch.set_color('red')      # High
                    elif bin_center >= 0.4:
                        patch.set_color('orange')   # Medium
                    elif bin_center >= 0.2:
                        patch.set_color('yellow')   # Low
                    else:
                        patch.set_color('lightgreen')  # Very low
                
                axes[1, 0].set_title('Deadlock Severity Distribution (Color Coded)')
                axes[1, 0].set_xlabel('Severity (0-1)')
                axes[1, 0].set_ylabel('Frequency')
                
                # Enhanced statistics
                mean_severity = severity_data.mean()
                max_severity = severity_data.max()
                critical_episodes = (severity_data >= 0.8).sum()
                
                axes[1, 0].axvline(mean_severity, color='black', linestyle='--', linewidth=2,
                                  label=f'Mean: {mean_severity:.3f}')
                axes[1, 0].axvline(max_severity, color='darkred', linestyle=':', linewidth=2,
                                  label=f'Max: {max_severity:.3f}')
                
                # Add severity level legend
                severity_text = f"""Critical (≥0.8): {critical_episodes} episodes
High (0.6-0.8): {((severity_data >= 0.6) & (severity_data < 0.8)).sum()}
Medium (0.4-0.6): {((severity_data >= 0.4) & (severity_data < 0.6)).sum()}
Low (0.2-0.4): {((severity_data >= 0.2) & (severity_data < 0.4)).sum()}"""
                
                axes[1, 0].text(0.98, 0.98, severity_text,
                               transform=axes[1, 0].transAxes, fontsize=8,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
                axes[1, 0].legend()
        
        # 4. ENHANCED Safety events analysis with rates
        if 'collision_count' in self.metrics_df.columns and 'deadlocks_detected' in self.metrics_df.columns:
            collision_data = self.metrics_df[['timestep', 'collision_count']].dropna()
            deadlock_data = self.metrics_df[['timestep', 'deadlocks_detected']].dropna()
            
            if len(collision_data) > 0 and len(deadlock_data) > 0:
                # Calculate rates (events per 1000 steps)
                collision_increments = collision_data['collision_count'].diff().fillna(0)
                deadlock_increments = deadlock_data['deadlocks_detected'].diff().fillna(0)
                
                # Rolling window for rate calculation (per 1000 steps)
                window_size = min(50, len(collision_data) // 10)
                if window_size > 5:
                    collision_rate = collision_increments.rolling(window=window_size).sum() * (1000 / window_size)
                    deadlock_rate = deadlock_increments.rolling(window=window_size).sum() * (1000 / window_size)
                    
                    axes[1, 1].plot(collision_data['timestep'], collision_rate, 
                                   'r-', linewidth=2, label=f'Collision Rate (per 1000 steps)', alpha=0.8)
                    axes[1, 1].plot(deadlock_data['timestep'], deadlock_rate, 
                                   'orange', linewidth=2, label=f'Deadlock Rate (per 1000 steps)', alpha=0.8)
                    
                    # Add trend lines
                    if len(collision_rate.dropna()) > 2:
                        valid_collision = collision_rate.dropna()
                        if len(valid_collision) > 0:
                            collision_trend = np.polyfit(range(len(valid_collision)), valid_collision, 1)[0]
                            trend_color = 'green' if collision_trend <= 0 else 'red'
                            axes[1, 1].text(0.02, 0.15, f'Collision Trend: {collision_trend:.3f}/step',
                                           transform=axes[1, 1].transAxes, color=trend_color, fontweight='bold')
                    
                    if len(deadlock_rate.dropna()) > 2:
                        valid_deadlock = deadlock_rate.dropna()
                        if len(valid_deadlock) > 0:
                            deadlock_trend = np.polyfit(range(len(valid_deadlock)), valid_deadlock, 1)[0]
                            trend_color = 'green' if deadlock_trend <= 0 else 'red'
                            axes[1, 1].text(0.02, 0.08, f'Deadlock Trend: {deadlock_trend:.3f}/step',
                                           transform=axes[1, 1].transAxes, color=trend_color, fontweight='bold')
                    
                    axes[1, 1].set_title(f'Safety Event Rates (Rolling {window_size}-step window)')
                else:
                    # Fallback to normalized comparison
                    if collision_data['collision_count'].max() > 0:
                        collision_norm = collision_data['collision_count'] / collision_data['collision_count'].max()
                        axes[1, 1].plot(collision_data['timestep'], collision_norm, 
                                       'r-', linewidth=2, label='Collisions (normalized)', marker='o', markersize=2)
                    
                    if deadlock_data['deadlocks_detected'].max() > 0:
                        deadlock_norm = deadlock_data['deadlocks_detected'] / deadlock_data['deadlocks_detected'].max()
                        axes[1, 1].plot(deadlock_data['timestep'], deadlock_norm, 
                                       'orange', linewidth=2, label='Deadlocks (normalized)', marker='s', markersize=2)
                    
                    axes[1, 1].set_title('Safety Events Trend Comparison (Normalized)')
                
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Rate/Normalized Count')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, 'safety_metrics_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Safety metrics chart saved: {save_path}")

    def plot_reward_and_performance(self):
        """Plot reward and performance metrics chart"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("❌ No data available for performance metrics plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Reward and Performance Metrics', fontsize=16, fontweight='bold')
        
        # 1. Reward trend
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df[['timestep', 'reward']].dropna()
            if len(reward_data) > 0:
                axes[0, 0].plot(reward_data['timestep'], reward_data['reward'], 
                               'b-', linewidth=2, marker='o', markersize=3)
                axes[0, 0].set_title(f'Reward Trend ({len(reward_data)} data points)')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Reward Value')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add moving average line
                if len(reward_data) > 10:
                    window = min(50, len(reward_data) // 10)
                    reward_ma = reward_data['reward'].rolling(window=window).mean()
                    axes[0, 0].plot(reward_data['timestep'], reward_ma, 
                                   'r--', linewidth=2, label=f'{window}-step Moving Average')
                    axes[0, 0].legend()
            
        # 2. Throughput trend
        if 'throughput' in self.metrics_df.columns:
            throughput_data = self.metrics_df[['timestep', 'throughput']].dropna()
            if len(throughput_data) > 0:
                axes[0, 1].plot(throughput_data['timestep'], throughput_data['throughput'], 
                               'g-', linewidth=2, marker='s', markersize=3)
                axes[0, 1].set_title(f'Throughput Trend ({len(throughput_data)} data points)')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Vehicles/Hour')
                axes[0, 1].grid(True, alpha=0.3)
            
        # 3. Vehicle control effectiveness
        if 'total_controlled' in self.metrics_df.columns and 'vehicles_detected' in self.metrics_df.columns:
            controlled_data = self.metrics_df[['timestep', 'total_controlled', 'vehicles_detected']].dropna()
            if len(controlled_data) > 0:
                axes[1, 0].plot(controlled_data['timestep'], controlled_data['total_controlled'], 
                               'purple', linewidth=2, label='Controlled Vehicles', marker='o', markersize=2)
                axes[1, 0].plot(controlled_data['timestep'], controlled_data['vehicles_detected'], 
                               'cyan', linewidth=2, label='Detected Vehicles', marker='s', markersize=2)
                axes[1, 0].set_title('Vehicle Control Effectiveness')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Number of Vehicles')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
        # 4. Reward distribution histogram
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df['reward'].dropna()
            if len(reward_data) > 0:
                axes[1, 1].hist(reward_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].set_title('Reward Distribution')
                axes[1, 1].set_xlabel('Reward Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(reward_data.mean(), color='red', linestyle='--', 
                                  label=f'Mean: {reward_data.mean():.2f}')
                axes[1, 1].legend()
            
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, 'reward_and_performance_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Reward and performance metrics chart saved: {save_path}")

    def generate_summary_report(self):
        """Generate summary report"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("❌ No data available for report generation")
            return
        
        report_path = os.path.join(self.plots_dir, 'training_analysis_report.txt')
        
        report_lines = [
            "=" * 50,
            "DRL Training Analysis Report",
            "=" * 50,
            "",
            f"Data Overview:",
            f"  Total Training Steps: {self.metrics_df['timestep'].max():,}",
            f"  Data Records Count: {len(self.metrics_df):,}",
            "",
        ]
        
        # Safety metrics statistics
        if 'collision_count' in self.metrics_df.columns:
            total_collisions = self.metrics_df['collision_count'].max()
            report_lines.extend([
                f"Safety Metrics:",
                f"  Total Collisions: {total_collisions}",
            ])
        
        if 'deadlocks_detected' in self.metrics_df.columns:
            total_deadlocks = self.metrics_df['deadlocks_detected'].max()
            report_lines.append(f"  Total Deadlocks: {total_deadlocks}")
        
        if 'deadlock_severity' in self.metrics_df.columns:
            avg_severity = self.metrics_df['deadlock_severity'].mean()
            max_severity = self.metrics_df['deadlock_severity'].max()
            report_lines.extend([
                f"  Average Deadlock Severity: {avg_severity:.3f}",
                f"  Maximum Deadlock Severity: {max_severity:.3f}",
            ])
        
        report_lines.append("")
        
        # Performance metrics
        if 'reward' in self.metrics_df.columns:
            avg_reward = self.metrics_df['reward'].mean()
            final_reward = self.metrics_df['reward'].iloc[-1]
            report_lines.extend([
                f"Performance Metrics:",
                f"  Average Reward: {avg_reward:.2f}",
                f"  Final Reward: {final_reward:.2f}",
            ])
        
        if 'throughput' in self.metrics_df.columns:
            avg_throughput = self.metrics_df['throughput'].mean()
            report_lines.append(f"  Average Throughput: {avg_throughput:.1f} vehicles/hour")
        
        # Final parameter values
        param_keys = ['bid_scale', 'eta_weight', 'speed_weight', 'congestion_sensitivity']
        final_params = {}
        for key in param_keys:
            if key in self.metrics_df.columns and not self.metrics_df[key].isna().all():
                final_params[key] = self.metrics_df[key].iloc[-1]
        
        if final_params:
            report_lines.extend(["", "Final Parameter Values:"])
            for key, value in final_params.items():
                report_lines.append(f"  {key}: {value:.4f}")
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Analysis report saved: {report_path}")

    def generate_all_plots(self):
        """Generate all analysis charts"""
        print("📊 Starting to generate all analysis charts...")
        
        # First load data
        if not self.load_data():
            print("❌ Unable to load data, exiting")
            return
        
        try:
            # Generate various charts
            print("   Generating parameter trend charts...")
            self.plot_parameter_trends()
            
            print("   Generating safety metrics charts...")
            self.plot_safety_metrics()
            
            print("   Generating reward and performance metrics charts...")
            self.plot_reward_and_performance()
            
            print("   Generating analysis report...")
            self.generate_summary_report()
            
            print(f"✅ All charts generated successfully, saved in: {self.plots_dir}")
            
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
            import traceback
            traceback.print_exc()

    def generate_report(self):
        """Compatibility method - generate report"""
        self.generate_summary_report()

    def save_summary_json(self):
        """Save JSON summary"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            return
        
        import json
        
        summary = {
            'training_steps': int(self.metrics_df['timestep'].max()),
            'data_points': len(self.metrics_df),
            'total_collisions': int(self.metrics_df.get('collision_count', pd.Series([0])).max()),
            'total_deadlocks': int(self.metrics_df.get('deadlocks_detected', pd.Series([0])).max()),
        }
        
        # Add available metrics
        if 'reward' in self.metrics_df.columns:
            summary['avg_reward'] = float(self.metrics_df['reward'].mean())
            summary['final_reward'] = float(self.metrics_df['reward'].iloc[-1])
        
        if 'throughput' in self.metrics_df.columns:
            summary['avg_throughput'] = float(self.metrics_df['throughput'].mean())
        
        if 'bid_scale' in self.metrics_df.columns:
            summary['final_bid_scale'] = float(self.metrics_df['bid_scale'].iloc[-1])
        
        summary_path = os.path.join(self.plots_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON summary saved: {summary_path}")

def quick_analysis(results_dir: str = None, plots_dir: str = None):
    """Quick analysis function"""
    if results_dir is None:
        results_dir = "drl/results"
    if plots_dir is None:
        plots_dir = "drl/plots"
    
    analyzer = TrainingAnalyzer(results_dir, plots_dir)
    analyzer.generate_all_plots()
    analyzer.save_summary_json()
    
    return analyzer

if __name__ == "__main__":
    # Run analysis
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        plots_dir = sys.argv[2] if len(sys.argv) > 2 else "drl/plots"
    else:
        results_dir = "drl/results"
        plots_dir = "drl/plots"
    
    quick_analysis(results_dir, plots_dir)