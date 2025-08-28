"""
ULTRA-OPTIMIZED Training Analyzer - Enhanced for Deadlock Avoidance System
================================================================================

This analyzer has been completely optimized for the ULTRA-OPTIMIZED DRL system with:

KEY IMPROVEMENTS:
   • Action Space: Reduced from 8 to 4 critical parameters (50% reduction)
   • Observation Space: Optimized from 169 to 50 dimensions (70% reduction)
   • Focus: Deadlock avoidance and collision prevention
   • Enhanced visualizations with emojis and color coding
   • Parameter-specific analysis and recommendations

ENHANCED OUTPUTS:
   1. parameter_trends_[param_group_name].png - ULTRA-OPTIMIZED parameter trends
      • 4 critical trainable parameters with bounds visualization
      • Trend analysis with direction indicators (UP/DOWN/STABLE)
      • Parameter-specific insights and recommendations
      • Color-coded trainable vs fixed parameters

   2. ULTRA_OPTIMIZED_safety_metrics_deadlock_avoidance.png - Enhanced safety analysis
      • Collision and deadlock trends with ULTRA-OPTIMIZED focus
      • Deadlock severity distribution with color coding
      • Safety event rates with trend analysis
      • Enhanced statistics and status indicators

   3. ULTRA_OPTIMIZED_reward_and_performance_metrics.png - Performance analysis
      • Reward trends with moving averages and trend analysis
      • Throughput analysis with efficiency metrics
      • Vehicle control effectiveness with efficiency calculations
      • Reward distribution with performance color coding

   4. ULTRA_OPTIMIZED_training_analysis_report.txt - Comprehensive report
      • System overview and ULTRA-OPTIMIZATION summary
      • Enhanced safety and performance analysis
      • Parameter analysis with trainable vs fixed distinction
      • AI-generated recommendations based on performance

ANALYSIS FEATURES:
   • Automatic data validation and cleaning
   • Trend analysis with visual indicators
   • Performance status assessment (EXCELLENT/GOOD/WARNING/CRITICAL)
   • Parameter bounds visualization
   • Enhanced color schemes and typography
   • Comprehensive error handling and fallbacks

TECHNICAL ENHANCEMENTS:
   • 300 DPI high-quality output
   • Enhanced matplotlib styling
   • Robust data handling with fallbacks
   • Performance-optimized calculations
   • Memory-efficient data processing
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
        
        print(f"Training Analyzer initialized")
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
            print("No training data CSV files found")
            return False
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Read and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    # FIXED: Clean the data during loading
                    df_cleaned = self._clean_dataframe(df)
                    if len(df_cleaned) > 0:
                        dataframes.append(df_cleaned)
                        print(f"   Loaded: {os.path.basename(csv_file)} ({len(df)} rows -> {len(df_cleaned)} clean rows)")
                    else:
                        print(f"   No clean data after cleaning: {os.path.basename(csv_file)}")
                else:
                    print(f"   Empty file: {os.path.basename(csv_file)}")
            except Exception as e:
                print(f"   Failed to load: {os.path.basename(csv_file)} - {e}")
        
        if not dataframes:
            print("All CSV files failed to load")
            return False
        
        # Merge data
        self.metrics_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates and sort by timestep
        if 'timestep' in self.metrics_df.columns:
            self.metrics_df = self.metrics_df.drop_duplicates(subset=['timestep'])
            self.metrics_df = self.metrics_df.sort_values('timestep')
        
        # FIXED: Final data validation and cleaning
        self.metrics_df = self._final_data_cleanup(self.metrics_df)
        
        print(f"Data loading completed: {len(self.metrics_df)} rows, {len(self.metrics_df.columns)} columns")
        
        # Report data quality
        self._report_data_quality()
        
        return True

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean individual dataframe during loading"""
        try:
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Remove rows with invalid timesteps
            if 'timestep' in df.columns:
                df = df[df['timestep'].notna()]
                df = df[df['timestep'] >= 0]  # Only positive timesteps
            
            # Clean safety metrics - ensure they're non-negative
            safety_columns = ['collision_count', 'deadlocks_detected', 'cumulative_collisions', 
                             'cumulative_deadlocks', 'new_collisions_this_step', 'new_deadlocks_this_step']
            
            for col in safety_columns:
                if col in df.columns:
                    # Replace negative values with 0
                    df[col] = df[col].clip(lower=0)
                    # Replace NaN with 0 for safety metrics
                    df[col] = df[col].fillna(0)
            
            # Clean deadlock severity - ensure it's in valid range [0, 1]
            if 'deadlock_severity' in df.columns:
                df['deadlock_severity'] = df['deadlock_severity'].clip(lower=0, upper=1)
                df['deadlock_severity'] = df['deadlock_severity'].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"   Data cleaning failed: {e}")
            return df  # Return original if cleaning fails

    def _final_data_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup after merging all dataframes"""
        try:
            # Ensure all numeric columns are properly typed
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns:
                    # Replace infinite values with NaN, then fill with appropriate defaults
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Set appropriate defaults based on column type
                    if 'collision' in col.lower() or 'deadlock' in col.lower():
                        df[col] = df[col].fillna(0)  # Safety metrics default to 0
                    elif 'reward' in col.lower():
                        df[col] = df[col].fillna(0.0)  # Rewards default to 0.0
                    elif 'throughput' in col.lower():
                        df[col] = df[col].fillna(0.0)  # Throughput default to 0.0
                    else:
                        df[col] = df[col].fillna(0.0)  # Other numeric columns default to 0.0
            
            # Ensure timestep is integer
            if 'timestep' in df.columns:
                df['timestep'] = df['timestep'].astype(int)
            
            return df
            
        except Exception as e:
            print(f"Final data cleanup failed: {e}")
            return df

    def _report_data_quality(self):
        """Report data quality metrics"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            return
        
        print(f"\nData Quality Report:")
        print(f"   Total rows: {len(self.metrics_df):,}")
        print(f"   Total columns: {len(self.metrics_df.columns)}")
        
        # Check for safety metrics
        safety_columns = ['collision_count', 'deadlocks_detected', 'cumulative_collisions', 
                         'cumulative_deadlocks', 'new_collisions_this_step', 'new_deadlocks_this_step']
        
        available_safety_cols = [col for col in safety_columns if col in self.metrics_df.columns]
        print(f"   Safety metrics available: {len(available_safety_cols)}/{len(safety_columns)}")
        
        if available_safety_cols:
            print(f"   Available safety columns: {', '.join(available_safety_cols)}")
        
        # Check for missing data
        missing_data = self.metrics_df.isnull().sum()
        total_cells = len(self.metrics_df) * len(self.metrics_df.columns)
        missing_percentage = (missing_data.sum() / total_cells) * 100
        
        print(f"   Missing data: {missing_percentage:.1f}%")
        
        # Report specific column statistics
        if 'timestep' in self.metrics_df.columns:
            timestep_range = f"{self.metrics_df['timestep'].min():,} - {self.metrics_df['timestep'].max():,}"
            print(f"   Timestep range: {timestep_range}")
        
        if 'reward' in self.metrics_df.columns:
            reward_stats = self.metrics_df['reward'].describe()
            print(f"   Reward stats: mean={reward_stats['mean']:.2f}, std={reward_stats['std']:.2f}")
        
        print()  # Empty line for readability

    def plot_parameter_trends(self):
        """Plot training parameter trend charts - OPTIMIZED for ULTRA-OPTIMIZED system"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("No data available for parameter trend plotting")
            return
        
        # ULTRA-OPTIMIZED: Only 4 trainable parameters for deadlock avoidance
        param_groups = {
            'ULTRA-OPTIMIZED Trainable Parameters (Deadlock Avoidance Focus)': {
                'urgency_position_ratio': 'Urgency Position Ratio Factor (0.1-3.0, sigmoid)',
                'speed_diff_modifier': 'Speed Diff Modifier (-30 to +30, steps=5)',
                'max_participants_per_auction': 'Max Participants Per Auction (3-6, discrete)',
                'ignore_vehicles_go': 'Ignore Vehicles GO % (0-80%, steps=10%)'
            },
            'Fixed Parameters (Not Trainable - System Stability)': {
                'eta_weight': 'ETA Weight (Fixed: 1.0)',
                'platoon_bonus': 'Platoon Bonus (Fixed: 0.5)',
                'junction_penalty': 'Junction Penalty (Fixed: 0.2)',
                'ignore_vehicles_platoon_leader': 'Platoon Leader Ignore % (Auto: GO-10%)'
            }
        }
        
        for group_name, params in param_groups.items():
            self._plot_parameter_group_optimized(group_name, params)
    
    def _plot_parameter_group_optimized(self, group_name: str, params: Dict[str, str]):
        """Plot parameter group trend chart with ULTRA-OPTIMIZED enhancements"""
        # Check which parameters exist in the data
        available_params = {key: name for key, name in params.items() 
                          if key in self.metrics_df.columns and not self.metrics_df[key].isna().all()}
        
        if not available_params:
            print(f"{group_name}: No available parameter data")
            return
        
        # Create subplots with optimized layout
        n_params = len(available_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(14, 5*n_params))
        if n_params == 1:
            axes = [axes]
        
        # Enhanced title with system info
        is_trainable = "ULTRA-OPTIMIZED Trainable Parameters" in group_name
        title_color = 'darkgreen' if is_trainable else 'darkblue'
        fig.suptitle(f'{group_name}', fontsize=16, fontweight='bold', color=title_color)
        
        # Use distinct color schemes for trainable vs fixed parameters
        if is_trainable:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Bright colors for trainable
        else:
            colors = ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5']  # Muted colors for fixed
        
        for idx, (param_key, param_name) in enumerate(available_params.items()):
            # Get valid data points
            valid_data = self.metrics_df[['timestep', param_key]].dropna()
            
            if len(valid_data) > 0:
                # Main parameter line
                axes[idx].plot(valid_data['timestep'], valid_data[param_key], 
                             color=colors[idx % len(colors)], linewidth=3, marker='o', markersize=4)
                
                # Enhanced title with parameter info
                param_type = "TRAINABLE" if is_trainable else "FIXED"
                axes[idx].set_title(f'{param_name} - {param_type} ({len(valid_data)} data points)', 
                                  fontweight='bold', fontsize=12)
                axes[idx].set_xlabel('Training Steps', fontsize=10)
                axes[idx].set_ylabel('Parameter Value', fontsize=10)
                axes[idx].grid(True, alpha=0.4, linestyle='--')
                
                # Enhanced trend analysis
                if len(valid_data) > 2:
                    # Linear trend
                    z = np.polyfit(valid_data['timestep'], valid_data[param_key], 1)
                    p = np.poly1d(z)
                    trend_line = p(valid_data['timestep'])
                    axes[idx].plot(valid_data['timestep'], trend_line, 
                                 "--", color='red', alpha=0.8, linewidth=2, label='Linear Trend')
                    
                    # Trend analysis text
                    trend_slope = z[0]
                    trend_direction = "UP Increasing" if trend_slope > 0.001 else "DOWN Decreasing" if trend_slope < -0.001 else "STABLE Stable"
                    trend_strength = "Strong" if abs(trend_slope) > 0.01 else "Weak" if abs(trend_slope) > 0.001 else "Very Weak"
                    
                    # Parameter-specific analysis
                    param_analysis = self._get_parameter_analysis(param_key, valid_data[param_key])
                    
                    analysis_text = f"""Trend: {trend_direction} ({trend_strength})
Slope: {trend_slope:.6f}
{param_analysis}"""
                    
                    axes[idx].text(0.02, 0.98, analysis_text,
                                 transform=axes[idx].transAxes, fontsize=9,
                                 verticalalignment='top', bbox=dict(boxstyle="round", 
                                 facecolor='lightyellow', alpha=0.8))
                    
                    axes[idx].legend(loc='upper right')
                
                # Add parameter bounds if available
                self._add_parameter_bounds(axes[idx], param_key)
        
        plt.tight_layout()
        
        # Save chart with descriptive filename
        safe_group_name = group_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(self.plots_dir, f'parameter_trends_{safe_group_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{group_name} trend chart saved: {save_path}")
        if is_trainable:
            print(f"   ULTRA-OPTIMIZED: Focused on 4 critical deadlock avoidance parameters")
        else:
            print(f"   Fixed parameters: System stability maintained")

    def _get_parameter_analysis(self, param_key: str, param_values: pd.Series) -> str:
        """Get parameter-specific analysis for enhanced insights"""
        if param_key == 'urgency_position_ratio':
            avg_bid = param_values.mean()
            if avg_bid < 1.0:
                return f"Low bidding strategy (avg: {avg_bid:.2f})"
            elif avg_bid > 3.0:
                return f"Aggressive bidding (avg: {avg_bid:.2f})"
            else:
                return f"Balanced bidding (avg: {avg_bid:.2f})"
        
        elif param_key == 'speed_diff_modifier':
            avg_speed = param_values.mean()
            if avg_speed < -10:
                return f"Conservative speed (avg: {avg_speed:.1f})"
            elif avg_speed > 10:
                return f"Aggressive speed (avg: {avg_speed:.1f})"
            else:
                return f"Balanced speed (avg: {avg_speed:.1f})"
        
        elif param_key == 'max_participants_per_auction':
            most_common = param_values.mode().iloc[0] if len(param_values.mode()) > 0 else param_values.mean()
            return f"Most common: {most_common} participants"
        
        elif param_key == 'ignore_vehicles_go':
            avg_ignore = param_values.mean()
            if avg_ignore < 20:
                return f"Low ignore rate (avg: {avg_ignore:.1f}%)"
            elif avg_ignore > 60:
                return f"High ignore rate (avg: {avg_ignore:.1f}%)"
            else:
                return f"Medium ignore rate (avg: {avg_ignore:.1f}%)"
        
        else:
            return f"Mean: {param_values.mean():.3f}, Std: {param_values.std():.3f}"

    def _add_parameter_bounds(self, ax, param_key: str):
        """Add parameter bounds visualization"""
        bounds = {
            'urgency_position_ratio': (0.1, 3.0),
            'speed_diff_modifier': (-30.0, 30.0),
            'max_participants_per_auction': (3.0, 6.0),
            'ignore_vehicles_go': (0.0, 80.0)
        }
        
        if param_key in bounds:
            min_val, max_val = bounds[param_key]
            ax.axhline(y=min_val, color='red', linestyle=':', alpha=0.5, label=f'Min: {min_val}')
            ax.axhline(y=max_val, color='red', linestyle=':', alpha=0.5, label=f'Max: {max_val}')
            ax.fill_between(ax.get_xlim(), min_val, max_val, alpha=0.1, color='green', label='Valid Range')

    def plot_safety_metrics(self):
        """Plot ENHANCED safety metrics chart - OPTIMIZED for ULTRA-OPTIMIZED deadlock avoidance system"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("No data available for safety metrics plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ULTRA-OPTIMIZED Safety Metrics - Deadlock Avoidance Focus', fontsize=18, fontweight='bold', color='darkred')
        
        # 1. ENHANCED: Collision count trend with ULTRA-OPTIMIZED analysis
        collision_columns = ['collision_count', 'cumulative_collisions', 'new_collisions_this_step']
        available_collision_cols = [col for col in collision_columns if col in self.metrics_df.columns]
        
        if available_collision_cols:
            best_collision_col = available_collision_cols[0]
            collision_data = self.metrics_df[['timestep', best_collision_col]].dropna()
            
            if len(collision_data) > 0:
                # Clean the data - remove negative values and NaN
                collision_data = collision_data[collision_data[best_collision_col] >= 0]
                
                if len(collision_data) > 0:
                    # Enhanced collision visualization
                    axes[0, 0].plot(collision_data['timestep'], collision_data[best_collision_col], 
                                   'r-', linewidth=3, marker='o', markersize=4, label='Collisions')
                    axes[0, 0].set_title(f'Collision Analysis - {best_collision_col}', fontsize=14, fontweight='bold', color='darkred')
                    axes[0, 0].set_xlabel('Training Steps', fontsize=11)
                    axes[0, 0].set_ylabel('Collision Count', fontsize=11)
                    axes[0, 0].grid(True, alpha=0.4, linestyle='--')
                    
                    # ULTRA-OPTIMIZED collision statistics
                    total_collisions = collision_data[best_collision_col].max()
                    collision_rate = total_collisions / len(collision_data) if len(collision_data) > 0 else 0
                    
                    # Collision trend analysis
                    if len(collision_data) > 10:
                        collision_trend = np.polyfit(collision_data['timestep'], collision_data[best_collision_col], 1)[0]
                        trend_direction = "DOWN Decreasing" if collision_trend < -0.001 else "UP Increasing" if collision_trend > 0.001 else "STABLE Stable"
                        
                        stats_text = f"""Total Collisions: {total_collisions}
Collision Rate: {collision_rate:.4f}/step
Trend: {trend_direction}
Data Source: {best_collision_col}"""
                    else:
                        stats_text = f"""Total Collisions: {total_collisions}
Collision Rate: {collision_rate:.4f}/step
Data Source: {best_collision_col}"""
                    
                    axes[0, 0].text(0.02, 0.98, stats_text,
                                   transform=axes[0, 0].transAxes, fontsize=10,
                                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
                    
                    axes[0, 0].legend()
                else:
                                    axes[0, 0].text(0.5, 0.5, 'No collisions detected', 
                               transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=14, color='green')
                axes[0, 0].set_title('Collision Analysis - No Collisions', fontsize=14, fontweight='bold', color='green')
            else:
                axes[0, 0].text(0.5, 0.5, 'No collision data available', 
                               transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=12)
                axes[0, 0].set_title('Collision Analysis - No Data', fontsize=14, fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No collision columns found', 
                           transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Collision Analysis - No Data', fontsize=14, fontweight='bold')
        
        # 2. ENHANCED: Deadlock analysis with ULTRA-OPTIMIZED focus
        deadlock_columns = ['deadlocks_detected', 'cumulative_deadlocks', 'new_deadlocks_this_step']
        available_deadlock_cols = [col for col in deadlock_columns if col in self.metrics_df.columns]
        
        if available_deadlock_cols:
            best_deadlock_col = available_deadlock_cols[0]
            deadlock_data = self.metrics_df[['timestep', best_deadlock_col]].dropna()
            
            if len(deadlock_data) > 0:
                # Clean the data - remove negative values and NaN
                deadlock_data = deadlock_data[deadlock_data[best_deadlock_col] >= 0]
                
                if len(deadlock_data) > 0:
                    # Enhanced deadlock visualization
                    axes[0, 1].plot(deadlock_data['timestep'], deadlock_data[best_deadlock_col], 
                                   'orange', linewidth=3, marker='s', markersize=4, label='Deadlocks')
                    
                    # ULTRA-OPTIMIZED deadlock statistics
                    total_deadlocks = deadlock_data[best_deadlock_col].max()
                    deadlock_rate = total_deadlocks / len(deadlock_data) if len(deadlock_data) > 0 else 0
                    
                    # Deadlock trend analysis
                    if len(deadlock_data) > 10:
                        deadlock_trend = np.polyfit(deadlock_data['timestep'], deadlock_data[best_deadlock_col], 1)[0]
                        trend_direction = "DOWN Decreasing" if deadlock_trend < -0.001 else "UP Increasing" if deadlock_trend > 0.001 else "STABLE Stable"
                        
                        # Calculate deadlock episodes
                        deadlock_increments = deadlock_data[best_deadlock_col].diff().fillna(0)
                        new_deadlocks_count = (deadlock_increments > 0).sum()
                        avg_deadlock_gap = len(deadlock_data) / max(1, new_deadlocks_count)
                        
                        stats_text = f"""Total Deadlocks: {total_deadlocks}
Deadlock Rate: {deadlock_rate:.4f}/step
Trend: {trend_direction}
Episodes with Deadlocks: {new_deadlocks_count}
Avg Steps Between: {avg_deadlock_gap:.1f}"""
                    else:
                        stats_text = f"""Total Deadlocks: {total_deadlocks}
Deadlock Rate: {deadlock_rate:.4f}/step
Data Source: {best_deadlock_col}"""
                    
                    axes[0, 1].set_title(f'Deadlock Analysis - {best_deadlock_col}', fontsize=14, fontweight='bold', color='darkorange')
                    axes[0, 1].set_xlabel('Training Steps', fontsize=11)
                    axes[0, 1].set_ylabel('Deadlock Count', fontsize=11)
                    axes[0, 1].grid(True, alpha=0.4, linestyle='--')
                    
                    axes[0, 1].text(0.02, 0.98, stats_text,
                                   transform=axes[0, 1].transAxes, fontsize=9,
                                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor='moccasin', alpha=0.8))
                    
                    axes[0, 1].legend()
                else:
                    axes[0, 1].text(0.5, 0.5, 'No deadlocks detected', 
                                   transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=14, color='green')
                    axes[0, 1].set_title('Deadlock Analysis - No Deadlocks', fontsize=14, fontweight='bold', color='green')
            else:
                axes[0, 1].text(0.5, 0.5, 'No deadlock data available', 
                               transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=12)
                axes[0, 1].set_title('Deadlock Analysis - No Data', fontsize=14, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No deadlock columns found', 
                           transform=axes[0, 1].transAxes, ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Deadlock Analysis - No Data', fontsize=14, fontweight='bold')
        
        # 3. ENHANCED: Deadlock severity analysis with ULTRA-OPTIMIZED insights
        if 'deadlock_severity' in self.metrics_df.columns:
            severity_data = self.metrics_df['deadlock_severity'].dropna()
            
            if len(severity_data) > 0:
                # Clean the data - remove NaN and infinite values
                severity_data = severity_data[np.isfinite(severity_data)]
                severity_data = severity_data[(severity_data >= 0) & (severity_data <= 1)]  # Valid range
                
                if len(severity_data) > 0:
                    # Create enhanced histogram with ULTRA-OPTIMIZED color coding
                    n, bins, patches = axes[1, 0].hist(severity_data, bins=20, alpha=0.8, edgecolor='black', linewidth=1)
                    
                    # ULTRA-OPTIMIZED color coding for deadlock severity
                    for i, patch in enumerate(patches):
                        bin_center = (bins[i] + bins[i+1]) / 2
                        if bin_center >= 0.8:
                            patch.set_color('darkred')      # Critical: Severe deadlock
                            patch.set_label('Critical (>=0.8)' if i == 0 else "")
                        elif bin_center >= 0.6:
                            patch.set_color('red')          # High: High severity
                            patch.set_label('High (0.6-0.8)' if i == 0 else "")
                        elif bin_center >= 0.4:
                            patch.set_color('orange')       # Medium: Medium severity
                            patch.set_label('Medium (0.4-0.6)' if i == 0 else "")
                        elif bin_center >= 0.2:
                            patch.set_color('yellow')       # Low: Low severity
                            patch.set_label('Low (0.2-0.4)' if i == 0 else "")
                        else:
                            patch.set_color('lightgreen')   # Very low: Very low severity
                            patch.set_label('Very Low (0-0.2)' if i == 0 else "")
                    
                    axes[1, 0].set_title('Deadlock Severity Distribution - ULTRA-OPTIMIZED Analysis', 
                                        fontsize=14, fontweight='bold', color='darkred')
                    axes[1, 0].set_xlabel('Severity Level (0-1)', fontsize=11)
                    axes[1, 0].set_ylabel('Frequency', fontsize=11)
                    
                    # ULTRA-OPTIMIZED severity statistics
                    mean_severity = severity_data.mean()
                    max_severity = severity_data.max()
                    critical_episodes = (severity_data >= 0.8).sum()
                    high_severity_episodes = ((severity_data >= 0.6) & (severity_data < 0.8)).sum()
                    
                    # Add severity level statistics
                    axes[1, 0].axvline(mean_severity, color='black', linestyle='--', linewidth=2,
                                      label=f'Mean: {mean_severity:.3f}')
                    axes[1, 0].axvline(max_severity, color='darkred', linestyle=':', linewidth=2,
                                      label=f'Max: {max_severity:.3f}')
                    
                    # Enhanced severity analysis text
                    severity_text = f"""Critical (>=0.8): {critical_episodes} episodes
High (0.6-0.8): {high_severity_episodes} episodes
Total Episodes: {len(severity_data)}
Mean Severity: {mean_severity:.3f}"""
                    
                    axes[1, 0].text(0.98, 0.98, severity_text,
                                   transform=axes[1, 0].transAxes, fontsize=9,
                                   verticalalignment='top', horizontalalignment='right',
                                   bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.9))
                    
                    axes[1, 0].legend(loc='upper left')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No severity data available', 
                                   transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
                    axes[1, 0].set_title('Deadlock Severity - No Data', fontsize=14, fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No severity data available', 
                               transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('Deadlock Severity - No Data', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No severity column found', 
                           transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Deadlock Severity - No Data', fontsize=14, fontweight='bold')
        
        # 4. ENHANCED: ULTRA-OPTIMIZED safety events analysis with focus on deadlock avoidance
        collision_available = any(col in self.metrics_df.columns for col in ['collision_count', 'cumulative_collisions', 'new_collisions_this_step'])
        deadlock_available = any(col in self.metrics_df.columns for col in ['deadlocks_detected', 'cumulative_deadlocks', 'new_deadlocks_this_step'])
        
        if collision_available and deadlock_available:
            # Get the best available collision and deadlock data
            best_collision_col = next(col for col in ['collision_count', 'cumulative_collisions', 'new_collisions_this_step'] 
                                    if col in self.metrics_df.columns)
            best_deadlock_col = next(col for col in ['deadlocks_detected', 'cumulative_deadlocks', 'new_deadlocks_this_step'] 
                                   if col in self.metrics_df.columns)
            
            collision_data = self.metrics_df[['timestep', best_collision_col]].dropna()
            deadlock_data = self.metrics_df[['timestep', best_deadlock_col]].dropna()
            
            # Clean the data
            collision_data = collision_data[collision_data[best_collision_col] >= 0]
            deadlock_data = deadlock_data[deadlock_data[best_deadlock_col] >= 0]
            
            if len(collision_data) > 0 and len(deadlock_data) > 0:
                # Calculate ULTRA-OPTIMIZED safety event rates
                collision_increments = collision_data[best_collision_col].diff().fillna(0)
                deadlock_increments = deadlock_data[best_deadlock_col].diff().fillna(0)
                
                # Enhanced rolling window analysis for deadlock avoidance
                window_size = min(50, len(collision_data) // 10)
                if window_size > 5:
                    collision_rate = collision_increments.rolling(window=window_size).sum() * (1000 / window_size)
                    deadlock_rate = deadlock_increments.rolling(window=window_size).sum() * (1000 / window_size)
                    
                    # Enhanced visualization with ULTRA-OPTIMIZED focus
                    axes[1, 1].plot(collision_data['timestep'], collision_rate, 
                                   'r-', linewidth=3, label=f'Collision Rate (per 1000 steps)', alpha=0.9, marker='o', markersize=2)
                    axes[1, 1].plot(deadlock_data['timestep'], deadlock_rate, 
                                   'orange', linewidth=3, label=f'Deadlock Rate (per 1000 steps)', alpha=0.9, marker='s', markersize=2)
                    
                    # ULTRA-OPTIMIZED trend analysis for deadlock avoidance
                    if len(collision_rate.dropna()) > 2:
                        valid_collision = collision_rate.dropna()
                        if len(valid_collision) > 0:
                            collision_trend = np.polyfit(range(len(valid_collision)), valid_collision, 1)[0]
                            trend_color = 'green' if collision_trend <= 0 else 'red'
                            trend_symbol = "DOWN" if collision_trend <= 0 else "UP"
                            axes[1, 1].text(0.02, 0.15, f'Collision Trend: {trend_symbol} {collision_trend:.3f}/step',
                                           transform=axes[1, 1].transAxes, color=trend_color, fontweight='bold', fontsize=10)
                    
                    if len(deadlock_rate.dropna()) > 2:
                        valid_deadlock = deadlock_rate.dropna()
                        if len(valid_deadlock) > 0:
                            deadlock_trend = np.polyfit(range(len(valid_deadlock)), valid_deadlock, 1)[0]
                            trend_color = 'green' if deadlock_trend <= 0 else 'red'
                            trend_symbol = "DOWN" if deadlock_trend <= 0 else "UP"
                            axes[1, 1].text(0.02, 0.08, f'Deadlock Trend: {trend_symbol} {deadlock_trend:.3f}/step',
                                           transform=axes[1, 1].transAxes, color=trend_color, fontweight='bold', fontsize=10)
                    
                    axes[1, 1].set_title(f'ULTRA-OPTIMIZED Safety Event Rates - Deadlock Avoidance Focus\nRolling {window_size}-step window analysis', 
                                        fontsize=14, fontweight='bold', color='darkred')
                else:
                    # Fallback to normalized comparison for ULTRA-OPTIMIZED analysis
                    if collision_data[best_collision_col].max() > 0:
                        collision_norm = collision_data[best_collision_col] / collision_data[best_collision_col].max()
                        axes[1, 1].plot(collision_data['timestep'], collision_norm, 
                                       'r-', linewidth=3, label='Collisions (normalized)', marker='o', markersize=3)
                    
                    if deadlock_data[best_deadlock_col].max() > 0:
                        deadlock_norm = deadlock_data[best_deadlock_col] / deadlock_data[best_deadlock_col].max()
                        axes[1, 1].plot(deadlock_data['timestep'], deadlock_norm, 
                                       'orange', linewidth=3, label='Deadlocks (normalized)', marker='s', markersize=3)
                    
                    axes[1, 1].set_title('ULTRA-OPTIMIZED Safety Events - Normalized Comparison\nDeadlock Avoidance Analysis', 
                                        fontsize=14, fontweight='bold', color='darkred')
                
                axes[1, 1].set_xlabel('Training Steps', fontsize=11)
                axes[1, 1].set_ylabel('Rate/Normalized Count', fontsize=11)
                axes[1, 1].legend(loc='upper right', fontsize=10)
                axes[1, 1].grid(True, alpha=0.4, linestyle='--')
            else:
                axes[1, 1].text(0.5, 0.5, 'No safety events detected', 
                               transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=14, color='green')
                axes[1, 1].set_title('Safety Events - No Events', fontsize=14, fontweight='bold', color='green')
        else:
            axes[1, 1].text(0.5, 0.5, 'No safety event data available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Safety Events - No Data', fontsize=14, fontweight='bold')
            
        plt.tight_layout()
        
        # Save chart with ULTRA-OPTIMIZED naming
        save_path = os.path.join(self.plots_dir, 'ULTRA_OPTIMIZED_safety_metrics_deadlock_avoidance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ULTRA-OPTIMIZED Safety Metrics chart saved: {save_path}")
        print(f"   Focus: Deadlock avoidance and collision prevention")
        print(f"   Enhanced analysis for 4 critical trainable parameters")

    def plot_reward_and_performance(self):
        """Plot reward and performance metrics chart - OPTIMIZED for ULTRA-OPTIMIZED system"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("No data available for performance metrics plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ULTRA-OPTIMIZED Reward and Performance Metrics - Deadlock Avoidance Focus', 
                    fontsize=18, fontweight='bold', color='darkblue')
        
        # 1. ENHANCED: Reward trend with ULTRA-OPTIMIZED analysis
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df[['timestep', 'reward']].dropna()
            if len(reward_data) > 0:
                # Enhanced reward visualization
                axes[0, 0].plot(reward_data['timestep'], reward_data['reward'], 
                               'b-', linewidth=3, marker='o', markersize=4, label='Reward', alpha=0.8)
                axes[0, 0].set_title(f'Reward Trend Analysis - ULTRA-OPTIMIZED System', 
                                    fontsize=14, fontweight='bold', color='darkblue')
                axes[0, 0].set_xlabel('Training Steps', fontsize=11)
                axes[0, 0].set_ylabel('Reward Value', fontsize=11)
                axes[0, 0].grid(True, alpha=0.4, linestyle='--')
                
                # ULTRA-OPTIMIZED reward analysis
                if len(reward_data) > 10:
                    window = min(50, len(reward_data) // 10)
                    reward_ma = reward_data['reward'].rolling(window=window).mean()
                    axes[0, 0].plot(reward_data['timestep'], reward_ma, 
                                   'r--', linewidth=3, label=f'{window}-step Moving Average', alpha=0.9)
                    
                    # Reward trend analysis
                    reward_trend = np.polyfit(reward_data['timestep'], reward_data['reward'], 1)[0]
                    trend_direction = "UP Improving" if reward_trend > 0.001 else "DOWN Declining" if reward_trend < -0.001 else "STABLE Stable"
                    trend_strength = "Strong" if abs(reward_trend) > 0.01 else "Weak" if abs(reward_trend) > 0.001 else "Very Weak"
                    
                    # Reward statistics
                    avg_reward = reward_data['reward'].mean()
                    final_reward = reward_data['reward'].iloc[-1]
                    reward_std = reward_data['reward'].std()
                    
                    analysis_text = f"""Reward Trend: {trend_direction} ({trend_strength})
Slope: {reward_trend:.6f}
Average: {avg_reward:.2f}
Final: {final_reward:.2f}
Std Dev: {reward_std:.2f}"""
                    
                    axes[0, 0].text(0.02, 0.98, analysis_text,
                                   transform=axes[0, 0].transAxes, fontsize=9,
                                   verticalalignment='top', bbox=dict(boxstyle="round", 
                                   facecolor='lightblue', alpha=0.8))
                    
                    axes[0, 0].legend(loc='upper right', fontsize=10)
                else:
                    axes[0, 0].legend(loc='upper right', fontsize=10)
            
        # 2. ENHANCED: Throughput trend with ULTRA-OPTIMIZED focus
        if 'throughput' in self.metrics_df.columns:
            throughput_data = self.metrics_df[['timestep', 'throughput']].dropna()
            if len(throughput_data) > 0:
                # Enhanced throughput visualization
                axes[0, 1].plot(throughput_data['timestep'], throughput_data['throughput'], 
                               'g-', linewidth=3, marker='s', markersize=4, label='Throughput', alpha=0.8)
                axes[0, 1].set_title(f'Throughput Analysis - Vehicle Flow Efficiency', 
                                    fontsize=14, fontweight='bold', color='darkgreen')
                axes[0, 1].set_xlabel('Training Steps', fontsize=11)
                axes[0, 1].set_ylabel('Vehicles/Hour', fontsize=11)
                axes[0, 1].grid(True, alpha=0.4, linestyle='--')
                
                # ULTRA-OPTIMIZED throughput analysis
                if len(throughput_data) > 10:
                    # Throughput trend analysis
                    throughput_trend = np.polyfit(throughput_data['timestep'], throughput_data['throughput'], 1)[0]
                    trend_direction = "UP Improving" if throughput_trend > 0.1 else "DOWN Declining" if throughput_trend < -0.1 else "STABLE Stable"
                    
                    # Throughput statistics
                    avg_throughput = throughput_data['throughput'].mean()
                    max_throughput = throughput_data['throughput'].max()
                    final_throughput = throughput_data['throughput'].iloc[-1]
                    
                    analysis_text = f"""Throughput Trend: {trend_direction}
Slope: {throughput_trend:.2f} vehicles/step^2
Average: {avg_throughput:.1f} vehicles/h
Maximum: {max_throughput:.1f} vehicles/h
Final: {final_throughput:.1f} vehicles/h"""
                    
                    axes[0, 1].text(0.02, 0.98, analysis_text,
                                   transform=axes[0, 1].transAxes, fontsize=9,
                                   verticalalignment='top', bbox=dict(boxstyle="round", 
                                   facecolor='lightgreen', alpha=0.8))
                    
                    axes[0, 1].legend(loc='upper right', fontsize=10)
                else:
                    axes[0, 1].legend(loc='upper right', fontsize=10)
            
        # 3. ENHANCED: Vehicle control effectiveness with ULTRA-OPTIMIZED metrics
        if 'total_controlled' in self.metrics_df.columns and 'vehicles_detected' in self.metrics_df.columns:
            controlled_data = self.metrics_df[['timestep', 'total_controlled', 'vehicles_detected']].dropna()
            if len(controlled_data) > 0:
                # Enhanced control visualization
                axes[1, 0].plot(controlled_data['timestep'], controlled_data['total_controlled'], 
                               'purple', linewidth=3, label='Controlled Vehicles', marker='o', markersize=3, alpha=0.8)
                axes[1, 0].plot(controlled_data['timestep'], controlled_data['vehicles_detected'], 
                               'cyan', linewidth=3, label='Detected Vehicles', marker='s', markersize=3, alpha=0.8)
                
                # Calculate control efficiency
                control_efficiency = controlled_data['total_controlled'] / controlled_data['vehicles_detected'].replace(0, 1)
                axes[1, 0].plot(controlled_data['timestep'], control_efficiency * 10, 
                               'orange', linewidth=2, label='Control Efficiency x10', marker='^', markersize=2, alpha=0.7)
                
                axes[1, 0].set_title('ULTRA-OPTIMIZED Vehicle Control Effectiveness', 
                                    fontsize=14, fontweight='bold', color='purple')
                axes[1, 0].set_xlabel('Training Steps', fontsize=11)
                axes[1, 0].set_ylabel('Number of Vehicles', fontsize=11)
                axes[1, 0].grid(True, alpha=0.4, linestyle='--')
                
                # Control efficiency analysis
                avg_efficiency = control_efficiency.mean()
                final_efficiency = control_efficiency.iloc[-1]
                
                analysis_text = f"""Control Efficiency Analysis:
Average: {avg_efficiency:.1%}
Final: {final_efficiency:.1%}
Controlled: {controlled_data['total_controlled'].iloc[-1]}
Detected: {controlled_data['vehicles_detected'].iloc[-1]}"""
                
                axes[1, 0].text(0.02, 0.98, analysis_text,
                               transform=axes[1, 0].transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle="round", 
                               facecolor='lavender', alpha=0.8))
                
                axes[1, 0].legend(loc='upper right', fontsize=10)
            
        # 4. ENHANCED: Reward distribution histogram with ULTRA-OPTIMIZED insights
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df['reward'].dropna()
            if len(reward_data) > 0:
                # Enhanced histogram with ULTRA-OPTIMIZED analysis
                n, bins, patches = axes[1, 1].hist(reward_data, bins=30, alpha=0.7, color='skyblue', 
                                                   edgecolor='black', linewidth=1)
                
                # Color code histogram by reward performance
                for i, patch in enumerate(patches):
                    bin_center = (bins[i] + bins[i+1]) / 2
                    if bin_center >= 20:
                        patch.set_color('darkgreen')      # Excellent performance
                    elif bin_center >= 10:
                        patch.set_color('green')          # Good performance
                    elif bin_center >= 0:
                        patch.set_color('lightgreen')     # Positive performance
                    elif bin_center >= -10:
                        patch.set_color('yellow')         # Neutral performance
                    else:
                        patch.set_color('red')            # Poor performance
                
                axes[1, 1].set_title('ULTRA-OPTIMIZED Reward Distribution Analysis', 
                                    fontsize=14, fontweight='bold', color='darkblue')
                axes[1, 1].set_xlabel('Reward Value', fontsize=11)
                axes[1, 1].set_ylabel('Frequency', fontsize=11)
                
                # Enhanced reward statistics
                mean_reward = reward_data.mean()
                median_reward = reward_data.median()
                std_reward = reward_data.std()
                positive_rewards = (reward_data > 0).sum()
                negative_rewards = (reward_data < 0).sum()
                
                # Add statistical lines
                axes[1, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2,
                                  label=f'Mean: {mean_reward:.2f}')
                axes[1, 1].axvline(median_reward, color='orange', linestyle=':', linewidth=2,
                                  label=f'Median: {median_reward:.2f}')
                
                # Reward performance analysis
                performance_text = f"""Reward Performance Analysis:
Mean: {mean_reward:.2f}
Median: {median_reward:.2f}
Std Dev: {std_reward:.2f}
Positive: {positive_rewards} ({positive_rewards/len(reward_data)*100:.1f}%)
Negative: {negative_rewards} ({negative_rewards/len(reward_data)*100:.1f}%)"""
                
                axes[1, 1].text(0.98, 0.98, performance_text,
                               transform=axes[1, 1].transAxes, fontsize=8,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle="round", facecolor='lightcyan', alpha=0.9))
                
                axes[1, 1].legend(loc='upper left', fontsize=9)
            
        plt.tight_layout()
        
        # Save chart with ULTRA-OPTIMIZED naming
        save_path = os.path.join(self.plots_dir, 'ULTRA_OPTIMIZED_reward_and_performance_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ULTRA-OPTIMIZED Reward and Performance Metrics chart saved: {save_path}")
        print(f"   Focus: Deadlock avoidance performance and system efficiency")
        print(f"   Enhanced analysis for 4 critical trainable parameters")

    def generate_summary_report(self):
        """Generate ULTRA-OPTIMIZED summary report for deadlock avoidance system"""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("No data available for report generation")
            return
        
        report_path = os.path.join(self.plots_dir, 'ULTRA_OPTIMIZED_training_analysis_report.txt')
        
        report_lines = [
            "=" * 70,
            "ULTRA-OPTIMIZED DRL Training Analysis Report",
            "Deadlock Avoidance Focus - 4 Critical Trainable Parameters",
            "=" * 70,
            "",
            "SYSTEM OVERVIEW:",
            "   ULTRA-OPTIMIZED Mode: Reduced from 8 to 4 trainable parameters",
            "   Focus: Deadlock avoidance and collision prevention",
            "   Observation Space: 50 dimensions (8 vehicles x 5 features)",
            "   Action Space: 4 critical parameters for deadlock avoidance",
            "",
            f"DATA OVERVIEW:",
            f"   Total Training Steps: {self.metrics_df['timestep'].max():,}",
            f"   Data Records Count: {len(self.metrics_df):,}",
            f"   Training Episodes: {len(self.metrics_df) // 100 if len(self.metrics_df) > 100 else 'Unknown'}",
            "",
        ]
        
        # ULTRA-OPTIMIZED: Safety metrics statistics with enhanced analysis
        # Check for collision data with fallback options
        collision_columns = ['collision_count', 'cumulative_collisions', 'new_collisions_this_step']
        available_collision_cols = [col for col in collision_columns if col in self.metrics_df.columns]
        
        if available_collision_cols:
            best_collision_col = available_collision_cols[0]
            collision_data = self.metrics_df[best_collision_col].dropna()
            collision_data = collision_data[collision_data >= 0]  # Remove negative values
            
            if len(collision_data) > 0:
                total_collisions = int(collision_data.max())
                collision_rate = total_collisions / len(collision_data) if len(collision_data) > 0 else 0
                
                # Enhanced collision analysis
                if len(collision_data) > 10:
                    collision_trend = np.polyfit(range(len(collision_data)), collision_data, 1)[0]
                    trend_direction = "DOWN Decreasing" if collision_trend < -0.001 else "UP Increasing" if collision_trend > 0.001 else "STABLE Stable"
                    report_lines.extend([
                        f"ULTRA-OPTIMIZED SAFETY METRICS - COLLISION ANALYSIS:",
                        f"   Total Collisions: {total_collisions} (from {best_collision_col})",
                        f"   Collision Rate: {collision_rate:.4f} per step",
                        f"   Collision Trend: {trend_direction} (slope: {collision_trend:.6f})",
                        f"   Safety Status: {'EXCELLENT' if total_collisions == 0 else 'NEEDS IMPROVEMENT' if total_collisions < 5 else 'CRITICAL'}",
                    ])
                else:
                    report_lines.extend([
                        f"ULTRA-OPTIMIZED SAFETY METRICS - COLLISION ANALYSIS:",
                        f"   Total Collisions: {total_collisions} (from {best_collision_col})",
                        f"   Collision Rate: {collision_rate:.4f} per step",
                        f"   Safety Status: {'EXCELLENT' if total_collisions == 0 else 'NEEDS IMPROVEMENT' if total_collisions < 5 else 'CRITICAL'}",
                    ])
            else:
                report_lines.extend([
                        f"ULTRA-OPTIMIZED SAFETY METRICS - COLLISION ANALYSIS:",
                        f"   Total Collisions: 0 (no valid data in {best_collision_col})",
                        f"   Safety Status: EXCELLENT - No collisions detected",
                    ])
        else:
            report_lines.extend([
                f"ULTRA-OPTIMIZED SAFETY METRICS - COLLISION ANALYSIS:",
                f"   Total Collisions: Unknown (no collision data columns found)",
                f"   Safety Status: UNKNOWN - No collision data available",
            ])
        
        # Check for deadlock data with ULTRA-OPTIMIZED analysis
        deadlock_columns = ['deadlocks_detected', 'cumulative_deadlocks', 'new_deadlocks_this_step']
        available_deadlock_cols = [col for col in deadlock_columns if col in self.metrics_df.columns]
        
        if available_deadlock_cols:
            best_deadlock_col = available_deadlock_cols[0]
            deadlock_data = self.metrics_df[best_deadlock_col].dropna()
            deadlock_data = deadlock_data[deadlock_data >= 0]  # Remove negative values
            
            if len(deadlock_data) > 0:
                total_deadlocks = int(deadlock_data.max())
                deadlock_rate = total_deadlocks / len(deadlock_data) if len(deadlock_data) > 0 else 0
                
                # Enhanced deadlock analysis
                if len(deadlock_data) > 10:
                    deadlock_trend = np.polyfit(range(len(deadlock_data)), deadlock_data, 1)[0]
                    trend_direction = "DOWN Decreasing" if deadlock_trend < -0.001 else "UP Increasing" if deadlock_trend > 0.001 else "STABLE Stable"
                    
                    # Calculate deadlock episodes
                    deadlock_increments = deadlock_data.diff().fillna(0)
                    new_deadlocks_count = (deadlock_increments > 0).sum()
                    avg_deadlock_gap = len(deadlock_data) / max(1, new_deadlocks_count)
                    
                    report_lines.extend([
                        f"   Total Deadlocks: {total_deadlocks} (from {best_deadlock_col})",
                        f"   Deadlock Rate: {deadlock_rate:.4f} per step",
                        f"   Deadlock Trend: {trend_direction} (slope: {deadlock_trend:.6f})",
                        f"   Episodes with Deadlocks: {new_deadlocks_count}",
                        f"   Average Steps Between Deadlocks: {avg_deadlock_gap:.1f}",
                        f"   Deadlock Status: {'EXCELLENT' if total_deadlocks == 0 else 'NEEDS IMPROVEMENT' if total_deadlocks < 3 else 'CRITICAL'}",
                    ])
                else:
                    report_lines.extend([
                        f"   Total Deadlocks: {total_deadlocks} (from {best_deadlock_col})",
                        f"   Deadlock Rate: {deadlock_rate:.4f} per step",
                        f"   Deadlock Status: {'EXCELLENT' if total_deadlocks == 0 else 'NEEDS IMPROVEMENT' if total_deadlocks < 3 else 'CRITICAL'}",
                    ])
            else:
                report_lines.extend([
                    f"   Total Deadlocks: 0 (no valid data in {best_deadlock_col})",
                    f"   Deadlock Status: EXCELLENT - No deadlocks detected",
                ])
        else:
            report_lines.extend([
                f"   Total Deadlocks: Unknown (no deadlock data columns found)",
                f"   Deadlock Status: UNKNOWN - No deadlock data available",
            ])
        
        # Check for deadlock severity data with ULTRA-OPTIMIZED insights
        if 'deadlock_severity' in self.metrics_df.columns:
            severity_data = self.metrics_df['deadlock_severity'].dropna()
            severity_data = severity_data[np.isfinite(severity_data)]  # Remove NaN and inf
            severity_data = severity_data[(severity_data >= 0) & (severity_data <= 1)]  # Valid range
            
            if len(severity_data) > 0:
                avg_severity = severity_data.mean()
                max_severity = severity_data.max()
                critical_episodes = (severity_data >= 0.8).sum()
                high_severity_episodes = ((severity_data >= 0.6) & (severity_data < 0.8)).sum()
                
                report_lines.extend([
                    f"   Average Deadlock Severity: {avg_severity:.3f}",
                    f"   Maximum Deadlock Severity: {max_severity:.3f}",
                    f"   Critical Episodes (>=0.8): {critical_episodes}",
                    f"   High Severity Episodes (0.6-0.8): {high_severity_episodes}",
                    f"   Severity Status: {'EXCELLENT' if avg_severity < 0.3 else 'MODERATE' if avg_severity < 0.6 else 'CRITICAL'}",
                ])
            else:
                report_lines.append(f"   Deadlock Severity: No valid data available")
        else:
            report_lines.append(f"   Deadlock Severity: No severity data column found")
        
        report_lines.append("")
        
        # ULTRA-OPTIMIZED: Performance metrics with enhanced analysis
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df['reward'].dropna()
            if len(reward_data) > 0:
                avg_reward = reward_data.mean()
                final_reward = reward_data.iloc[-1]
                reward_std = reward_data.std()
                
                # Enhanced reward analysis
                if len(reward_data) > 10:
                    reward_trend = np.polyfit(range(len(reward_data)), reward_data, 1)[0]
                    trend_direction = "UP Improving" if reward_trend > 0.001 else "DOWN Declining" if reward_trend < -0.001 else "STABLE Stable"
                    trend_strength = "Strong" if abs(reward_trend) > 0.01 else "Weak" if abs(reward_trend) > 0.001 else "Very Weak"
                    
                    positive_rewards = (reward_data > 0).sum()
                    negative_rewards = (reward_data < 0).sum()
                    
                    report_lines.extend([
                        f"ULTRA-OPTIMIZED PERFORMANCE METRICS - REWARD ANALYSIS:",
                        f"   Average Reward: {avg_reward:.2f}",
                        f"   Final Reward: {final_reward:.2f}",
                        f"   Reward Standard Deviation: {reward_std:.2f}",
                        f"   Reward Trend: {trend_direction} ({trend_strength})",
                        f"   Positive Rewards: {positive_rewards} ({positive_rewards/len(reward_data)*100:.1f}%)",
                        f"   Negative Rewards: {negative_rewards} ({negative_rewards/len(reward_data)*100:.1f}%)",
                        f"   Performance Status: {'EXCELLENT' if avg_reward > 15 else 'GOOD' if avg_reward > 5 else 'NEEDS IMPROVEMENT' if avg_reward > -5 else 'POOR'}",
                    ])
                else:
                    report_lines.extend([
                        f"ULTRA-OPTIMIZED PERFORMANCE METRICS - REWARD ANALYSIS:",
                        f"   Average Reward: {avg_reward:.2f}",
                        f"   Final Reward: {final_reward:.2f}",
                        f"   Reward Standard Deviation: {reward_std:.2f}",
                        f"   Performance Status: {'EXCELLENT' if avg_reward > 15 else 'GOOD' if avg_reward > 5 else 'NEEDS IMPROVEMENT' if avg_reward > -5 else 'POOR'}",
                    ])
            else:
                report_lines.extend([
                    f"ULTRA-OPTIMIZED PERFORMANCE METRICS - REWARD ANALYSIS:",
                    f"   Average Reward: No valid data",
                    f"   Final Reward: No valid data",
                    f"   Performance Status: UNKNOWN - No reward data available",
                ])
        
        if 'throughput' in self.metrics_df.columns:
            throughput_data = self.metrics_df['throughput'].dropna()
            if len(throughput_data) > 0:
                avg_throughput = throughput_data.mean()
                max_throughput = throughput_data.max()
                final_throughput = throughput_data.iloc[-1]
                
                # Enhanced throughput analysis
                if len(throughput_data) > 10:
                    throughput_trend = np.polyfit(range(len(throughput_data)), throughput_data, 1)[0]
                    trend_direction = "UP Improving" if throughput_trend > 0.1 else "DOWN Declining" if throughput_trend < -0.1 else "STABLE Stable"
                    
                    report_lines.extend([
                        f"   Average Throughput: {avg_throughput:.1f} vehicles/hour",
                        f"   Maximum Throughput: {max_throughput:.1f} vehicles/hour",
                        f"   Final Throughput: {final_throughput:.1f} vehicles/hour",
                        f"   Throughput Trend: {trend_direction} (slope: {throughput_trend:.2f})",
                        f"   Throughput Status: {'EXCELLENT' if avg_throughput > 800 else 'GOOD' if avg_throughput > 500 else 'NEEDS IMPROVEMENT' if avg_throughput > 200 else 'POOR'}",
                    ])
                else:
                    report_lines.extend([
                        f"   Average Throughput: {avg_throughput:.1f} vehicles/hour",
                        f"   Maximum Throughput: {max_throughput:.1f} vehicles/hour",
                        f"   Final Throughput: {final_throughput:.1f} vehicles/hour",
                        f"   Throughput Status: {'EXCELLENT' if avg_throughput > 800 else 'GOOD' if avg_throughput > 500 else 'NEEDS IMPROVEMENT' if avg_throughput > 200 else 'POOR'}",
                    ])
            else:
                report_lines.append(f"   Average Throughput: No valid data")
        
        report_lines.append("")
        
        # ULTRA-OPTIMIZED: Final parameter values with enhanced analysis
        trainable_params = ['urgency_position_ratio', 'speed_diff_modifier', 'max_participants_per_auction', 'ignore_vehicles_go']
        fixed_params = ['eta_weight', 'platoon_bonus', 'junction_penalty']
        
        final_trainable_params = {}
        final_fixed_params = {}
        
        for key in trainable_params:
            if key in self.metrics_df.columns and not self.metrics_df[key].isna().all():
                param_data = self.metrics_df[key].dropna()
                if len(param_data) > 0:
                    final_trainable_params[key] = param_data.iloc[-1]
        
        for key in fixed_params:
            if key in self.metrics_df.columns and not self.metrics_df[key].isna().all():
                param_data = self.metrics_df[key].dropna()
                if len(param_data) > 0:
                    final_fixed_params[key] = param_data.iloc[-1]
        
        if final_trainable_params:
            report_lines.extend(["ULTRA-OPTIMIZED TRAINABLE PARAMETERS (Deadlock Avoidance Focus):"])
            for key, value in final_trainable_params.items():
                param_name = {
                    'urgency_position_ratio': 'Urgency Position Ratio Factor (0.1-3.0)',
                    'speed_diff_modifier': 'Speed Diff Modifier (-30 to +30)',
                    'max_participants_per_auction': 'Max Participants Per Auction (3-6)',
                    'ignore_vehicles_go': 'Ignore Vehicles GO % (0-80%)'
                }.get(key, key)
                report_lines.append(f"   {param_name}: {value:.4f}")
        
        if final_fixed_params:
            report_lines.extend(["", "FIXED PARAMETERS (System Stability - Not Trainable):"])
            for key, value in final_fixed_params.items():
                param_name = {
                    'eta_weight': 'ETA Weight (Fixed: 1.0)',
                    'platoon_bonus': 'Platoon Bonus (Fixed: 0.5)',
                    'junction_penalty': 'Junction Penalty (Fixed: 0.2)'
                }.get(key, key)
                report_lines.append(f"   {param_name}: {value:.4f}")
        
        report_lines.extend([
            "",
            "ULTRA-OPTIMIZATION SUMMARY:",
            "   Action Space: Reduced from 8 to 4 parameters (50% reduction)",
            "   Observation Space: Optimized from 169 to 50 dimensions (70% reduction)",
            "   Focus: 4 critical parameters for deadlock avoidance",
            "   System: Fixed parameters maintain stability",
            "   Training: Faster convergence with focused learning",
            "",
            "RECOMMENDATIONS:",
        ])
        
        # Generate ULTRA-OPTIMIZED recommendations based on analysis
        if 'reward' in self.metrics_df.columns and 'collision_count' in self.metrics_df.columns:
            reward_data = self.metrics_df['reward'].dropna()
            collision_data = self.metrics_df.get('collision_count', pd.Series([0]))
            
            if len(reward_data) > 0:
                avg_reward = reward_data.mean()
                total_collisions = collision_data.max() if len(collision_data) > 0 else 0
                
                if avg_reward < 0 and total_collisions > 5:
                    report_lines.extend([
                        "   CRITICAL: High collision rate and negative rewards detected",
                        "   ACTION: Review urgency_position_ratio and speed_diff_modifier parameters",
                        "   ACTION: Consider reducing ignore_vehicles_go for better safety",
                        "   ACTION: Check deadlock detection thresholds"
                    ])
                elif avg_reward < 5:
                    report_lines.extend([
                        "   MODERATE: Low rewards but manageable safety",
                        "   ACTION: Optimize max_participants_per_auction for efficiency",
                        "   ACTION: Fine-tune urgency_position_ratio for better vehicle prioritization",
                        "   ACTION: Monitor deadlock severity trends"
                    ])
                else:
                    report_lines.extend([
                        "   EXCELLENT: Good performance and safety achieved",
                        "   ACTION: Fine-tune parameters for even better performance",
                        "   ACTION: Consider reducing training steps if stable",
                        "   ACTION: Monitor for long-term stability"
                    ])
            else:
                report_lines.append("   UNKNOWN: Insufficient data for recommendations")
        else:
            report_lines.append("   UNKNOWN: Missing reward or collision data for recommendations")
        
        report_lines.extend([
            "",
            "=" * 70,
            "Report generated by ULTRA-OPTIMIZED DRL Training Analyzer",
            "Focus: Deadlock Avoidance and System Efficiency",
            "=" * 70
        ])
        
        # Write report
        report_content = "\n".join(report_lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"ULTRA-OPTIMIZED Analysis report saved: {report_path}")
        print(f"   Focus: 4 critical deadlock avoidance parameters")
        print(f"   Enhanced analysis for ULTRA-OPTIMIZED system")

    def generate_all_plots(self):
        """Generate all ULTRA-OPTIMIZED analysis charts for deadlock avoidance system"""
        print("Starting to generate ULTRA-OPTIMIZED analysis charts...")
        print("   Focus: 4 critical deadlock avoidance parameters")
        print("   Enhanced analysis for ULTRA-OPTIMIZED system")
        
        # First load data
        if not self.load_data():
            print("Unable to load data, exiting")
            return
        
        try:
            # Generate various charts with ULTRA-OPTIMIZED enhancements
            print("   Generating ULTRA-OPTIMIZED parameter trend charts...")
            self.plot_parameter_trends()
            
            print("   Generating ULTRA-OPTIMIZED safety metrics charts...")
            self.plot_safety_metrics()
            
            print("   Generating ULTRA-OPTIMIZED reward and performance metrics charts...")
            self.plot_reward_and_performance()
            
            print("   Generating ULTRA-OPTIMIZED analysis report...")
            self.generate_summary_report()
            
            print(f"All ULTRA-OPTIMIZED charts generated successfully!")
            print(f"   Saved in: {self.plots_dir}")
            print(f"   Focus: Deadlock avoidance and collision prevention")
            print(f"   Enhanced analysis for 4 critical trainable parameters")
            
        except Exception as e:
            print(f"Error generating ULTRA-OPTIMIZED charts: {e}")
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
        }
        
        # FIXED: Proper collision data extraction with fallback options
        collision_columns = ['collision_count', 'cumulative_collisions', 'new_collisions_this_step']
        available_collision_cols = [col for col in collision_columns if col in self.metrics_df.columns]
        
        if available_collision_cols:
            best_collision_col = available_collision_cols[0]
            collision_data = self.metrics_df[best_collision_col].dropna()
            collision_data = collision_data[collision_data >= 0]  # Remove negative values
            
            if len(collision_data) > 0:
                summary['total_collisions'] = int(collision_data.max())
                summary['collision_data_source'] = best_collision_col
                summary['collision_rate'] = float(collision_data.max() / len(collision_data))
            else:
                summary['total_collisions'] = 0
                summary['collision_data_source'] = best_collision_col
                summary['collision_rate'] = 0.0
        else:
            summary['total_collisions'] = -1  # Indicates no data available
            summary['collision_data_source'] = 'none'
            summary['collision_rate'] = -1.0
        
        # FIXED: Proper deadlock data extraction with fallback options
        deadlock_columns = ['deadlocks_detected', 'cumulative_deadlocks', 'new_deadlocks_this_step']
        available_deadlock_cols = [col for col in deadlock_columns if col in self.metrics_df.columns]
        
        if available_deadlock_cols:
            best_deadlock_col = available_deadlock_cols[0]
            deadlock_data = self.metrics_df[best_deadlock_col].dropna()
            deadlock_data = deadlock_data[deadlock_data >= 0]  # Remove negative values
            
            if len(deadlock_data) > 0:
                summary['total_deadlocks'] = int(deadlock_data.max())
                summary['deadlock_data_source'] = best_deadlock_col
                summary['deadlock_rate'] = float(deadlock_data.max() / len(deadlock_data))
            else:
                summary['total_deadlocks'] = 0
                summary['deadlock_data_source'] = best_deadlock_col
                summary['deadlock_rate'] = 0.0
        else:
            summary['total_deadlocks'] = -1  # Indicates no data available
            summary['deadlock_data_source'] = 'none'
            summary['deadlock_rate'] = -1.0
        
        # Add available metrics
        if 'reward' in self.metrics_df.columns:
            reward_data = self.metrics_df['reward'].dropna()
            if len(reward_data) > 0:
                summary['avg_reward'] = float(reward_data.mean())
                summary['final_reward'] = float(reward_data.iloc[-1])
            else:
                summary['avg_reward'] = 0.0
                summary['final_reward'] = 0.0
        
        if 'throughput' in self.metrics_df.columns:
            throughput_data = self.metrics_df['throughput'].dropna()
            if len(throughput_data) > 0:
                summary['avg_throughput'] = float(throughput_data.mean())
            else:
                summary['avg_throughput'] = 0.0
        
        if 'urgency_position_ratio' in self.metrics_df.columns:
            urgency_position_ratio_data = self.metrics_df['urgency_position_ratio'].dropna()
            if len(urgency_position_ratio_data) > 0:
                summary['final_urgency_position_ratio'] = float(urgency_position_ratio_data.iloc[-1])
            else:
                summary['final_urgency_position_ratio'] = 1.0
        
        # Add data quality indicators
        summary['data_quality'] = {
            'has_collision_data': available_collision_cols != [],
            'has_deadlock_data': available_deadlock_cols != [],
            'collision_columns_found': available_collision_cols,
            'deadlock_columns_found': available_deadlock_cols,
            'total_columns': len(self.metrics_df.columns),
            'missing_data_percentage': float(self.metrics_df.isnull().sum().sum() / (len(self.metrics_df) * len(self.metrics_df.columns)) * 100)
        }
        
        summary_path = os.path.join(self.plots_dir, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"JSON summary saved: {summary_path}")
        print(f"   Collision data: {summary['collision_data_source']} (total: {summary['total_collisions']})")
        print(f"   Deadlock data: {summary['deadlock_data_source']} (total: {summary['total_deadlocks']})")

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