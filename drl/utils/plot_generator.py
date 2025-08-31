#!/usr/bin/env python3
"""
Simple plotting utility for DRL training metrics
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

def plot_training_metrics(results_dir: str, plots_dir: str, save_plots: bool = True):
    """Plot all training metrics with English labels only"""
    
    print("🎨 Generating training plots using new plotting utility...")
    
    # Find CSV files
    episode_csv = os.path.join(results_dir, 'episode_metrics.csv')
    step_csv = os.path.join(results_dir, 'step_metrics.csv')
    
    if not os.path.exists(episode_csv):
        print(f"❌ Episode metrics CSV not found: {episode_csv}")
        return
    
    # Load data
    episode_df = pd.read_csv(episode_csv)
    step_df = None
    if os.path.exists(step_csv):
        step_df = pd.read_csv(step_csv)
    
    print(f"📊 Loaded {len(episode_df)} episode metrics")
    
    # Clean data
    episode_df = episode_df.dropna(subset=['episode'])
    if step_df is not None:
        step_df = step_df.dropna(subset=['timestep'])
    
    # Generate all plots
    print("📈 Generating episode performance plot...")
    _plot_episode_performance(episode_df, plots_dir, save_plots)
    
    print("⚙️ Generating action parameters plot...")
    _plot_action_parameters(episode_df, plots_dir, save_plots)
    
    print("🎯 Generating action space exact values plot...")
    plot_action_space_exact_values(results_dir, plots_dir, save_plots)
    
    if step_df is not None and len(step_df) > 0:
        print("📊 Generating step metrics plot...")
        _plot_step_metrics(step_df, plots_dir, save_plots)
    
    # Check for simulation time data
    time_columns = ['episode_simulation_time', 'total_simulation_time', 
                   'episode_duration_hours', 'total_duration_hours']
    has_time_data = any(col in episode_df.columns for col in time_columns)
    
    if has_time_data:
        print("⏱️ Generating simulation time plot...")
        _plot_simulation_time(episode_df, plots_dir, save_plots)
    
    # Generate summary report
    print("📝 Generating summary report...")
    generate_summary_report(results_dir, plots_dir)
    
    print("✅ All training plots and reports generated successfully!")
    print(f"📁 Check {plots_dir} for generated files")

def _plot_episode_performance(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot episode performance metrics with English labels only"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Episode Performance Metrics', fontsize=16)
    
    # Plot 1: Total vehicles exited
    axes[0, 0].plot(episode_df['episode'], episode_df['total_vehicles_exited'], 'g-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Total Vehicles Exited', fontsize=12)
    axes[0, 0].set_xlabel('Episode', fontsize=10)
    axes[0, 0].set_ylabel('Vehicles', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['total_vehicles_exited'])):
        axes[0, 0].annotate(f'{value:.0f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 2: Total collisions
    axes[0, 1].plot(episode_df['episode'], episode_df['total_collisions'], 'r-o', linewidth=2, markersize=6)
    axes[0, 1].set_title('Total Collisions', fontsize=12)
    axes[0, 1].set_xlabel('Episode', fontsize=10)
    axes[0, 1].set_ylabel('Collisions', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['total_collisions'])):
        axes[0, 1].annotate(f'{value:.0f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 3: Average throughput
    axes[1, 0].plot(episode_df['episode'], episode_df['avg_throughput'], 'b-o', linewidth=2, markersize=6)
    axes[1, 0].set_title('Average Throughput', fontsize=12)
    axes[1, 0].set_xlabel('Episode', fontsize=10)
    axes[1, 0].set_ylabel('Throughput (vehicles/h)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['avg_throughput'])):
        axes[1, 0].annotate(f'{value:.1f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 4: Average acceleration
    axes[1, 1].plot(episode_df['episode'], episode_df['avg_acceleration'], 'purple', marker='o', linewidth=2, markersize=6)
    axes[1, 1].set_title('Average Acceleration', fontsize=12)
    axes[1, 1].set_xlabel('Episode', fontsize=10)
    axes[1, 1].set_ylabel('Acceleration (m/s²)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['avg_acceleration'])):
        axes[1, 1].annotate(f'{value:.2f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Set x-axis to show integer episode numbers
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'episode_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Episode performance plot saved: {plot_path}")
    
    plt.show()

def _plot_action_parameters(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot action space parameter EXACT VALUES (actual values applied in environment)"""
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Action Space Parameters - TRUE EXACT VALUES', fontsize=16)
    
    # Use TRUE EXACT value columns only
    urgency_col = 'urgency_position_ratio_exact'
    speed_col = 'speed_diff_modifier_exact'
    participants_col = 'max_participants_exact'
    ignore_col = 'ignore_vehicles_go_exact'
    
    # Verify all required columns exist
    required_cols = [urgency_col, speed_col, participants_col, ignore_col]
    missing_cols = [col for col in required_cols if col not in episode_df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(episode_df.columns)}")
        return
    
    # Plot 1: Urgency Position Ratio - TRUE EXACT VALUE
    axes[0, 0].plot(episode_df['episode'], episode_df[urgency_col], 'b-o', linewidth=2, markersize=6, label='TRUE EXACT')
    axes[0, 0].set_title('Urgency Position Ratio', fontsize=12)
    axes[0, 0].set_xlabel('Episode', fontsize=10)
    axes[0, 0].set_ylabel('Ratio', fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value annotations on the plot
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[urgency_col])):
        axes[0, 0].annotate(f'{value:.3f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 2: Speed Diff Modifier - TRUE EXACT VALUE
    axes[0, 1].plot(episode_df['episode'], episode_df[speed_col], 'r-o', linewidth=2, markersize=6, label='TRUE EXACT')
    axes[0, 1].set_title('Speed Diff Modifier', fontsize=12)
    axes[0, 1].set_xlabel('Episode', fontsize=10)
    axes[0, 1].set_ylabel('Modifier', fontsize=10)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value annotations on the plot
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[speed_col])):
        axes[0, 1].annotate(f'{value:.1f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 3: Max Participants - TRUE EXACT VALUE
    axes[1, 0].plot(episode_df['episode'], episode_df[participants_col], 'g-o', linewidth=2, markersize=6, label='TRUE EXACT')
    axes[1, 0].set_title('Max Participants per Auction', fontsize=12)
    axes[1, 0].set_xlabel('Episode', fontsize=10)
    axes[1, 0].set_ylabel('Participants', fontsize=10)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value annotations on the plot
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[participants_col])):
        axes[1, 0].annotate(f'{value:.0f}', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Plot 4: Ignore Vehicles GO - TRUE EXACT VALUE
    axes[1, 1].plot(episode_df['episode'], episode_df[ignore_col], 'orange', marker='o', linewidth=2, markersize=6, label='TRUE EXACT')
    axes[1, 1].set_title('Ignore Vehicles GO (%)', fontsize=12)
    axes[1, 0].set_xlabel('Episode', fontsize=10)
    axes[1, 1].set_ylabel('Percentage (%)', fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value annotations on the plot
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[ignore_col])):
        axes[1, 1].annotate(f'{value:.1f}%', (episode, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    
    # Set x-axis to show integer episode numbers and ensure proper y-axis scaling
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Force matplotlib to use actual data range for y-axis
        ax.autoscale_view()
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'action_parameters.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Action parameters plot saved: {plot_path}")
    
    plt.show()

def _plot_step_metrics(step_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot step-level metrics with English labels only"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Step-Level Training Metrics', fontsize=16)
    
    # Plot 1: Collision count over time
    axes[0, 0].plot(step_df['timestep'], step_df['collision_count'], 'r-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Collision Count Over Time', fontsize=12)
    axes[0, 0].set_xlabel('Timestep', fontsize=10)
    axes[0, 0].set_ylabel('Collision Count', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Vehicles exited over time
    axes[0, 1].plot(step_df['timestep'], step_df['vehicles_exited'], 'g-', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Vehicles Exited Over Time', fontsize=12)
    axes[0, 1].set_xlabel('Timestep', fontsize=10)
    axes[0, 1].set_ylabel('Vehicles Exited', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average acceleration over time
    axes[1, 0].plot(step_df['timestep'], step_df['avg_acceleration'], 'purple', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Average Acceleration Over Time', fontsize=12)
    axes[1, 0].set_xlabel('Timestep', fontsize=10)
    axes[1, 0].set_ylabel('Acceleration (m/s²)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Throughput over time
    if 'throughput' in step_df.columns:
        axes[1, 1].plot(step_df['timestep'], step_df['throughput'], 'b-', linewidth=1, alpha=0.7)
        axes[1, 1].set_title('Throughput Over Time', fontsize=12)
        axes[1, 1].set_xlabel('Timestep', fontsize=10)
        axes[1, 1].set_ylabel('Throughput (vehicles/h)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Action parameters are now stored as exact values per episode, not per step
        axes[1, 1].text(0.5, 0.5, 'Action parameters stored as exact values per episode\n(see episode_metrics.csv for exact values)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].set_title('Action Parameters (Episode-Level Only)', fontsize=12)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'step_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Step metrics plot saved: {plot_path}")
    
    plt.show()

def _plot_simulation_time(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot simulation time metrics with English labels only"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulation Time Metrics', fontsize=16)
    
    # Plot 1: Episode simulation time
    if 'episode_simulation_time' in episode_df.columns:
        axes[0, 0].plot(episode_df['episode'], episode_df['episode_simulation_time'], 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Episode Simulation Time', fontsize=12)
        axes[0, 0].set_xlabel('Episode', fontsize=10)
        axes[0, 0].set_ylabel('Time (seconds)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['episode_simulation_time'])):
            axes[0, 0].annotate(f'{value:.1f}s', (episode, value), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    else:
        axes[0, 0].text(0.5, 0.5, 'Episode simulation time data not available', 
                        ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Episode Simulation Time (Data Unavailable)', fontsize=12)
    
    # Plot 2: Total simulation time
    if 'total_simulation_time' in episode_df.columns:
        axes[0, 1].plot(episode_df['episode'], episode_df['total_simulation_time'], 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Total Simulation Time', fontsize=12)
        axes[0, 1].set_xlabel('Episode', fontsize=10)
        axes[0, 1].set_ylabel('Time (seconds)', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['total_simulation_time'])):
            axes[0, 1].annotate(f'{value:.1f}s', (episode, value), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    else:
        axes[0, 1].text(0.5, 0.5, 'Total simulation time data not available', 
                        ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Total Simulation Time (Data Unavailable)', fontsize=12)
    
    # Plot 3: Episode duration in hours
    if 'episode_duration_hours' in episode_df.columns:
        axes[1, 0].plot(episode_df['episode'], episode_df['episode_duration_hours'], 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Episode Duration (Hours)', fontsize=12)
        axes[1, 0].set_xlabel('Episode', fontsize=10)
        axes[1, 0].set_ylabel('Duration (hours)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['episode_duration_hours'])):
            axes[1, 0].annotate(f'{value:.3f}h', (episode, value), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    else:
        axes[1, 0].text(0.5, 0.5, 'Episode duration data not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Episode Duration (Data Unavailable)', fontsize=12)
    
    # Plot 4: Total duration in hours
    if 'total_duration_hours' in episode_df.columns:
        axes[1, 1].plot(episode_df['episode'], episode_df['total_duration_hours'], 'orange', marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Total Duration (Hours)', fontsize=12)
        axes[1, 1].set_xlabel('Episode', fontsize=10)
        axes[1, 1].set_ylabel('Duration (hours)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df['total_duration_hours'])):
            axes[1, 1].annotate(f'{value:.3f}h', (episode, value), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=8, alpha=0.8)
    else:
        axes[1, 1].text(0.5, 0.5, 'Total duration data not available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Total Duration (Data Unavailable)', fontsize=12)
    
    # Set x-axis to show integer episode numbers
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'simulation_time.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Simulation time plot saved: {plot_path}")
    
    plt.show()

def generate_summary_report(results_dir: str, plots_dir: str):
    """Generate comprehensive training summary report with English labels only"""
    
    # Find CSV files
    episode_csv = os.path.join(results_dir, 'episode_metrics.csv')
    step_csv = os.path.join(results_dir, 'step_metrics.csv')
    
    if not os.path.exists(episode_csv):
        print(f"❌ Episode metrics CSV not found: {episode_csv}")
        return
    
    # Load episode data
    episode_df = pd.read_csv(episode_csv)
    step_df = None
    if os.path.exists(step_csv):
        step_df = pd.read_csv(step_csv)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DRL TRAINING SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Basic training info
    report_lines.append("📊 TRAINING OVERVIEW:")
    report_lines.append(f"   Total Episodes: {len(episode_df)}")
    report_lines.append(f"   Data Range: Episodes {episode_df['episode'].min()} to {episode_df['episode'].max()}")
    report_lines.append("")
    
    # Performance metrics
    report_lines.append("🚗 PERFORMANCE METRICS:")
    if 'total_vehicles_exited' in episode_df.columns:
        total_exits = episode_df['total_vehicles_exited'].sum()
        avg_exits = episode_df['total_vehicles_exited'].mean()
        report_lines.append(f"   Total Vehicles Exited: {total_exits:.0f}")
        report_lines.append(f"   Average Exits per Episode: {avg_exits:.1f}")
    
    if 'total_collisions' in episode_df.columns:
        total_collisions = episode_df['total_collisions'].sum()
        avg_collisions = episode_df['total_collisions'].mean()
        report_lines.append(f"   Total Collisions: {total_collisions:.0f}")
        report_lines.append(f"   Average Collisions per Episode: {avg_collisions:.1f}")
    
    if 'avg_throughput' in episode_df.columns:
        avg_throughput = episode_df['avg_throughput'].mean()
        max_throughput = episode_df['avg_throughput'].max()
        report_lines.append(f"   Average Throughput: {avg_throughput:.1f} vehicles/h")
        report_lines.append(f"   Maximum Throughput: {max_throughput:.1f} vehicles/h")
    
    report_lines.append("")
    
    # Action space parameters
    report_lines.append("⚙️ ACTION SPACE PARAMETERS:")
    report_lines.append("   📊 Using TRUE EXACT parameter values (actual values applied in environment)")
    
    if 'urgency_position_ratio_exact' in episode_df.columns:
        urgency_ratio = episode_df['urgency_position_ratio_exact'].iloc[-1]
        report_lines.append(f"   Final Urgency Position Ratio (TRUE EXACT): {urgency_ratio:.3f}")
    else:
        report_lines.append(f"   Final Urgency Position Ratio: NOT AVAILABLE")
    
    if 'speed_diff_modifier_exact' in episode_df.columns:
        speed_modifier = episode_df['speed_diff_modifier_exact'].iloc[-1]
        report_lines.append(f"   Final Speed Diff Modifier (TRUE EXACT): {speed_modifier:.1f}")
    else:
        report_lines.append(f"   Final Speed Diff Modifier: NOT AVAILABLE")
    
    if 'max_participants_exact' in episode_df.columns:
        max_participants = episode_df['max_participants_exact'].iloc[-1]
        report_lines.append(f"   Final Max Participants (TRUE EXACT): {max_participants:.0f}")
    else:
        report_lines.append(f"   Final Max Participants: NOT AVAILABLE")
    
    if 'ignore_vehicles_go_exact' in episode_df.columns:
        ignore_vehicles = episode_df['ignore_vehicles_go_exact'].iloc[-1]
        report_lines.append(f"   Final Ignore Vehicles GO (TRUE EXACT): {ignore_vehicles:.1f}%")
    else:
        report_lines.append(f"   Final Ignore Vehicles GO: NOT AVAILABLE")
    
    report_lines.append("")
    
    # Training progress
    report_lines.append("📈 TRAINING PROGRESS:")
    if 'avg_throughput' in episode_df.columns:
        # Calculate improvement
        first_throughput = episode_df['avg_throughput'].iloc[0]
        last_throughput = episode_df['avg_throughput'].iloc[-1]
        improvement = ((last_throughput - first_throughput) / first_throughput * 100) if first_throughput > 0 else 0
        report_lines.append(f"   Throughput Improvement: {improvement:+.1f}%")
    
    if 'total_vehicles_exited' in episode_df.columns:
        first_exits = episode_df['total_vehicles_exited'].iloc[0]
        last_exits = episode_df['total_vehicles_exited'].iloc[-1]
        exit_improvement = ((last_exits - first_exits) / first_exits * 100) if first_exits > 0 else 0
        report_lines.append(f"   Exit Rate Improvement: {exit_improvement:+.1f}%")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("💡 RECOMMENDATIONS:")
    if 'total_collisions' in episode_df.columns and episode_df['total_collisions'].sum() > 0:
        report_lines.append("   ⚠️  Collisions detected - consider adjusting safety parameters")
    
    if 'avg_throughput' in episode_df.columns:
        if episode_df['avg_throughput'].mean() < 100:
            report_lines.append("   📉 Low throughput - consider optimizing traffic flow parameters")
        elif episode_df['avg_throughput'].mean() > 500:
            report_lines.append("   📈 High throughput achieved - good parameter optimization")
    
    report_lines.append("   🔧 Monitor action space parameter convergence")
    report_lines.append("   📊 Continue training for better performance")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(plots_dir, 'training_summary_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"📝 Summary report saved: {report_path}")
    except Exception as e:
        print(f"⚠️ Failed to save report: {e}")
    
    # Print report to console
    print('\n'.join(report_lines))

def plot_action_space_exact_values(results_dir: str, plots_dir: str, save_plots: bool = True):
    """Specialized plot to display TRUE EXACT values of action space parameters"""
    
    episode_csv = os.path.join(results_dir, 'episode_metrics.csv')
    if not os.path.exists(episode_csv):
        print(f"❌ Episode metrics CSV not found: {episode_csv}")
        return
    
    episode_df = pd.read_csv(episode_csv)
    
    # Use TRUE EXACT value columns only
    urgency_col = 'urgency_position_ratio_exact'
    speed_col = 'speed_diff_modifier_exact'
    participants_col = 'max_participants_exact'
    ignore_col = 'ignore_vehicles_go_exact'
    
    # Verify all required columns exist
    required_cols = [urgency_col, speed_col, participants_col, ignore_col]
    missing_cols = [col for col in required_cols if col not in episode_df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(episode_df.columns)}")
        return
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Action Space Parameters - TRUE EXACT VALUES', fontsize=18, fontweight='bold')
    
    # Plot 1: Urgency Position Ratio
    axes[0, 0].plot(episode_df['episode'], episode_df[urgency_col], 'b-o', 
                     linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[0, 0].set_title('Urgency Position Ratio', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Ratio', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value annotations with exact values
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[urgency_col])):
        axes[0, 0].annotate(f'{value:.3f}', (episode, value), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Speed Diff Modifier
    axes[0, 1].plot(episode_df['episode'], episode_df[speed_col], 'r-o', 
                     linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[0, 1].set_title('Speed Diff Modifier', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Modifier', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value annotations with exact values
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[speed_col])):
        axes[0, 1].annotate(f'{value:.1f}', (episode, value), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Max Participants
    axes[1, 0].plot(episode_df['episode'], episode_df[participants_col], 'g-o', 
                     linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[1, 0].set_title('Max Participants per Auction', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Participants', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value annotations with exact values
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[participants_col])):
        axes[1, 0].annotate(f'{value:.0f}', (episode, value), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Ignore Vehicles GO
    axes[1, 1].plot(episode_df['episode'], episode_df[ignore_col], 'orange', marker='o', 
                     linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[1, 1].set_title('Ignore Vehicles GO (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value annotations with exact values
    for i, (episode, value) in enumerate(zip(episode_df['episode'], episode_df[ignore_col])):
        axes[1, 1].annotate(f'{value:.1f}%', (episode, value), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set x-axis to show integer episode numbers
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'action_space_exact_values.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Action space TRUE EXACT VALUES plot saved: {plot_path}")
    
    plt.show()

def main():
    """Main function for standalone plot generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DRL training plots from CSV data')
    parser.add_argument('--results-dir', type=str, default='drl/results', 
                       help='Directory containing training results CSV files')
    parser.add_argument('--plots-dir', type=str, default='drl/plots', 
                       help='Directory to save generated plots')
    parser.add_argument('--no-save', action='store_true', 
                       help='Show plots without saving to disk')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print(f"🎨 DRL Training Plot Generator")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📁 Plots directory: {args.plots_dir}")
    print(f"💾 Save plots: {not args.no_save}")
    print("=" * 50)
    
    # Generate all plots
    plot_training_metrics(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        save_plots=not args.no_save
    )

    # plot_training_metrics(
    #     results_dir="/cs/student/projects2/seiot/2024/xueyifan/drl/training_runs/20250831_053325/results",  # 指定具体的日期文件夹
    #     plots_dir="/cs/student/projects2/seiot/2024/xueyifan/drl/training_runs/20250831_053325/plots",
    #     save_plots=True
    # )
    
    print("🎉 Plot generation completed!")

if __name__ == "__main__":
    main()