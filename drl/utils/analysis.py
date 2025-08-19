import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

class TrainingAnalyzer:
    """Analyze and visualize training results"""
    
    def __init__(self, results_dir: str, plots_dir: str):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.metrics_df = None
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load data
        self.load_data()

    def load_data(self):
        """Load training metrics data"""
        metrics_file = os.path.join(self.results_dir, 'training_metrics.csv')
        
        if os.path.exists(metrics_file):
            self.metrics_df = pd.read_csv(metrics_file)
            print(f"📊 Loaded {len(self.metrics_df)} training data points")
        else:
            print("⚠️ No metrics file found")

    def plot_training_progress(self):
        """Plot training progress over time"""
        if self.metrics_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Throughput
        axes[0, 0].plot(self.metrics_df['timestep'], self.metrics_df['throughput'])
        axes[0, 0].set_title('Throughput (vehicles/hour)')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Vehicles/Hour')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average acceleration
        axes[0, 1].plot(self.metrics_df['timestep'], self.metrics_df['avg_acceleration'], 
                       color='orange')
        axes[0, 1].set_title('Average Acceleration')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Acceleration (m/s²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Vehicles controlled
        axes[1, 0].plot(self.metrics_df['timestep'], self.metrics_df['total_controlled'], 
                       color='green')
        axes[1, 0].set_title('Total Controlled Vehicles')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Number of Vehicles')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bid scale
        axes[1, 1].plot(self.metrics_df['timestep'], self.metrics_df['bid_scale'], 
                       color='red')
        axes[1, 1].set_title('Learned Bid Scale')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Bid Scale')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_metrics(self):
        """Plot key performance metrics"""
        if self.metrics_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Throughput distribution
        axes[0, 0].hist(self.metrics_df['throughput'], bins=30, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Throughput Distribution')
        axes[0, 0].set_xlabel('Vehicles/Hour')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.metrics_df['throughput'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.metrics_df["throughput"].mean():.1f}')
        axes[0, 0].legend()
        
        # Acceleration vs Throughput
        axes[0, 1].scatter(self.metrics_df['avg_acceleration'], self.metrics_df['throughput'], 
                          alpha=0.6, color='orange')
        axes[0, 1].set_title('Acceleration vs Throughput')
        axes[0, 1].set_xlabel('Average Acceleration (m/s²)')
        axes[0, 1].set_ylabel('Throughput (vehicles/h)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling average throughput
        window = min(100, len(self.metrics_df) // 10)
        rolling_throughput = self.metrics_df['throughput'].rolling(window=window).mean()
        axes[1, 0].plot(self.metrics_df['timestep'], rolling_throughput, 
                       linewidth=2, color='green')
        axes[1, 0].set_title(f'Rolling Average Throughput (window={window})')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Throughput (vehicles/h)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative vehicles exited
        axes[1, 1].plot(self.metrics_df['timestep'], self.metrics_df['vehicles_exited'].cumsum(), 
                       color='purple')
        axes[1, 1].set_title('Cumulative Vehicles Exited')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Total Vehicles Exited')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_matrix(self):
        """Plot correlation matrix of metrics"""
        if self.metrics_df is None:
            return
        
        # Select numeric columns for correlation
        numeric_cols = ['throughput', 'avg_acceleration', 'total_controlled', 
                       'vehicles_exited', 'bid_scale']
        corr_data = self.metrics_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curves(self):
        """Plot learning curves with confidence intervals"""
        if self.metrics_df is None:
            return
        
        # Smooth the data
        window = min(50, len(self.metrics_df) // 20)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Throughput learning curve
        smooth_throughput = self.metrics_df['throughput'].rolling(window=window, center=True).mean()
        std_throughput = self.metrics_df['throughput'].rolling(window=window, center=True).std()
        
        axes[0].plot(self.metrics_df['timestep'], smooth_throughput, 
                    linewidth=2, label='Mean', color='blue')
        axes[0].fill_between(self.metrics_df['timestep'], 
                           smooth_throughput - std_throughput,
                           smooth_throughput + std_throughput,
                           alpha=0.3, color='blue', label='±1 Std')
        axes[0].set_title('Throughput Learning Curve')
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Throughput (vehicles/h)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Acceleration learning curve
        smooth_accel = self.metrics_df['avg_acceleration'].rolling(window=window, center=True).mean()
        std_accel = self.metrics_df['avg_acceleration'].rolling(window=window, center=True).std()
        
        axes[1].plot(self.metrics_df['timestep'], smooth_accel, 
                    linewidth=2, label='Mean', color='red')
        axes[1].fill_between(self.metrics_df['timestep'], 
                           smooth_accel - std_accel,
                           smooth_accel + std_accel,
                           alpha=0.3, color='red', label='±1 Std')
        axes[1].set_title('Acceleration Learning Curve')
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Acceleration (m/s²)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("📊 Generating training analysis plots...")
        
        self.plot_training_progress()
        print("✅ Training progress plot saved")
        
        self.plot_performance_metrics()
        print("✅ Performance metrics plot saved")
        
        self.plot_correlation_matrix()
        print("✅ Correlation matrix plot saved")
        
        self.plot_learning_curves()
        print("✅ Learning curves plot saved")

    def generate_report(self):
        """Generate text summary report"""
        if self.metrics_df is None:
            return
        
        report_path = os.path.join(self.plots_dir, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== DRL Training Analysis Report ===\n\n")
            
            # Basic statistics
            f.write("Training Statistics:\n")
            f.write(f"  Total Training Steps: {self.metrics_df['timestep'].max():,}\n")
            f.write(f"  Data Points Collected: {len(self.metrics_df):,}\n\n")
            
            # Performance metrics
            f.write("Performance Metrics:\n")
            f.write(f"  Average Throughput: {self.metrics_df['throughput'].mean():.2f} ± {self.metrics_df['throughput'].std():.2f} vehicles/h\n")
            f.write(f"  Max Throughput: {self.metrics_df['throughput'].max():.2f} vehicles/h\n")
            f.write(f"  Average Acceleration: {self.metrics_df['avg_acceleration'].mean():.3f} ± {self.metrics_df['avg_acceleration'].std():.3f} m/s²\n")
            f.write(f"  Total Vehicles Exited: {self.metrics_df['vehicles_exited'].sum():,}\n")
            f.write(f"  Average Controlled Vehicles: {self.metrics_df['total_controlled'].mean():.1f}\n\n")
            
            # Learning progress
            first_quarter = self.metrics_df.iloc[:len(self.metrics_df)//4]
            last_quarter = self.metrics_df.iloc[3*len(self.metrics_df)//4:]
            
            improvement_throughput = last_quarter['throughput'].mean() - first_quarter['throughput'].mean()
            improvement_accel = last_quarter['avg_acceleration'].mean() - first_quarter['avg_acceleration'].mean()
            
            f.write("Learning Progress:\n")
            f.write(f"  Throughput Improvement: {improvement_throughput:+.2f} vehicles/h\n")
            f.write(f"  Acceleration Change: {improvement_accel:+.3f} m/s²\n")
            f.write(f"  Final Bid Scale: {self.metrics_df['bid_scale'].iloc[-1]:.3f}\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            if improvement_throughput > 0:
                f.write("  ✅ Model is learning to improve throughput\n")
            else:
                f.write("  ⚠️ Consider adjusting reward function for throughput\n")
            
            if abs(improvement_accel) < 0.5:
                f.write("  ✅ Acceleration is stable\n")
            else:
                f.write("  ⚠️ Large acceleration changes detected\n")
        
        print(f"✅ Training report saved to {report_path}")
