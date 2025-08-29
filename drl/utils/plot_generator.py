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
    """
    Generate plots from training metrics CSV files
    
    Args:
        results_dir: Directory containing CSV metrics files
        plots_dir: Directory to save generated plots
        save_plots: Whether to save plots to disk
    """
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check for metrics files
    step_metrics_path = os.path.join(results_dir, 'step_metrics.csv')
    episode_metrics_path = os.path.join(results_dir, 'episode_metrics.csv')
    
    if not os.path.exists(episode_metrics_path):
        print(f"⚠️ Episode metrics file not found: {episode_metrics_path}")
        return
    
    try:
        # Load episode metrics
        episode_df = pd.read_csv(episode_metrics_path)
        print(f"📊 Loaded {len(episode_df)} episode metrics")
        
        # Data validation and cleaning
        if 'episode' in episode_df.columns:
            # Convert episode to numeric and remove invalid values
            episode_df['episode'] = pd.to_numeric(episode_df['episode'], errors='coerce')
            episode_df = episode_df.dropna(subset=['episode'])
            episode_df = episode_df[episode_df['episode'] >= 0]
            
            # Remove duplicates and sort
            episode_df = episode_df.drop_duplicates(subset=['episode']).sort_values('episode')
            
            print(f"📊 After cleaning: {len(episode_df)} valid episode metrics")
            print(f"📊 Episode range: {episode_df['episode'].min()} to {episode_df['episode'].max()}")
            print(f"📊 Episode values: {episode_df['episode'].tolist()}")
        
        # Generate plots
        _plot_episode_performance(episode_df, plots_dir, save_plots)
        _plot_action_parameters(episode_df, plots_dir, save_plots)
        _plot_simulation_time(episode_df, plots_dir, save_plots)
        
        if os.path.exists(step_metrics_path):
            step_df = pd.read_csv(step_metrics_path)
            print(f"📊 Loaded {len(step_df)} step metrics")
            
            # Data validation and cleaning for step metrics
            if 'timestep' in step_df.columns:
                # Convert timestep to numeric and remove invalid values
                step_df['timestep'] = pd.to_numeric(step_df['timestep'], errors='coerce')
                step_df = step_df.dropna(subset=['timestep'])
                step_df = step_df[step_df['timestep'] >= 0]
                
                # Remove duplicates and sort
                step_df = step_df.drop_duplicates(subset=['timestep']).sort_values('timestep')
                
                print(f"📊 After cleaning: {len(step_df)} valid step metrics")
                print(f"📊 Timestep range: {step_df['timestep'].min()} to {step_df['timestep'].max()}")
                print(f"📊 Timestep values (first 10): {step_df['timestep'].head(10).tolist()}")
            
            _plot_step_metrics(step_df, plots_dir, save_plots)
        
        print(f"✅ Plots generated successfully in {plots_dir}")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

def _plot_episode_performance(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot episode performance metrics"""
    
    # Check available columns and provide fallbacks
    available_columns = episode_df.columns.tolist()
    print(f"📊 Available columns: {available_columns}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Episode Performance Metrics', fontsize=16)
    
    # Plot 1: Vehicles exited per episode
    vehicles_exited_col = 'total_vehicles_exited' if 'total_vehicles_exited' in available_columns else 'vehicles_exited'
    if vehicles_exited_col in available_columns:
        axes[0, 0].plot(episode_df['episode'], episode_df[vehicles_exited_col], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Vehicles Exited per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Vehicles Exited')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, f'Column not found:\n{vehicles_exited_col}', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Vehicles Exited per Episode (Data Unavailable)')
    
    # Plot 2: Collisions per episode
    collisions_col = 'total_collisions' if 'total_collisions' in available_columns else 'collision_count'
    if collisions_col in available_columns:
        axes[0, 1].plot(episode_df['episode'], episode_df[collisions_col], 'r-o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Collisions per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Collisions')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, f'Column not found:\n{collisions_col}', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Collisions per Episode (Data Unavailable)')
    
    # Plot 3: Deadlocks per episode
    deadlocks_col = 'total_deadlocks' if 'total_deadlocks' in available_columns else 'deadlocks_detected'
    if deadlocks_col in available_columns:
        axes[1, 0].plot(episode_df['episode'], episode_df[deadlocks_col], 'orange', marker='o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Deadlocks per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Deadlocks')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, f'Column not found:\n{deadlocks_col}', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Deadlocks per Episode (Data Unavailable)')
    
    # Plot 4: Throughput per episode
    throughput_col = 'avg_throughput' if 'avg_throughput' in available_columns else 'throughput'
    if throughput_col in available_columns:
        axes[1, 1].plot(episode_df['episode'], episode_df[throughput_col], 'g-o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Average Throughput per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Throughput (vehicles/h)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f'Column not found:\n{throughput_col}', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Average Throughput per Episode (Data Unavailable)')
    
    # Set x-axis to show integer episode numbers and ensure proper y-axis scaling
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Force matplotlib to use actual data range for y-axis
        ax.autoscale_view()
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'episode_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Episode performance plot saved: {plot_path}")
    
    plt.show()

def _plot_action_parameters(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot action space parameter trends"""
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Action Space Parameter Trends', fontsize=16)
    
    # Plot 1: Urgency Position Ratio
    axes[0, 0].plot(episode_df['episode'], episode_df['urgency_position_ratio_mean'], 'b-o', linewidth=2, markersize=4, label='Mean')
    axes[0, 0].fill_between(episode_df['episode'], 
                           episode_df['urgency_position_ratio_mean'] - episode_df['urgency_position_ratio_std'],
                           episode_df['urgency_position_ratio_mean'] + episode_df['urgency_position_ratio_std'],
                           alpha=0.3, color='blue', label='±1 Std Dev')
    axes[0, 0].set_title('Urgency Position Ratio')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Speed Diff Modifier
    axes[0, 1].plot(episode_df['episode'], episode_df['speed_diff_modifier_mean'], 'r-o', linewidth=2, markersize=4, label='Mean')
    axes[0, 1].fill_between(episode_df['episode'], 
                           episode_df['speed_diff_modifier_mean'] - episode_df['speed_diff_modifier_std'],
                           episode_df['speed_diff_modifier_mean'] + episode_df['speed_diff_modifier_std'],
                           alpha=0.3, color='red', label='±1 Std Dev')
    axes[0, 1].set_title('Speed Diff Modifier')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Modifier')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Max Participants
    axes[1, 0].plot(episode_df['episode'], episode_df['max_participants_mean'], 'g-o', linewidth=2, markersize=4, label='Mean')
    axes[1, 0].fill_between(episode_df['episode'], 
                           episode_df['max_participants_mean'] - episode_df['max_participants_std'],
                           episode_df['max_participants_mean'] + episode_df['max_participants_std'],
                           alpha=0.3, color='green', label='±1 Std Dev')
    axes[1, 0].set_title('Max Participants per Auction')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Participants')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Ignore Vehicles GO
    axes[1, 1].plot(episode_df['episode'], episode_df['ignore_vehicles_go_mean'], 'orange', marker='o', linewidth=2, markersize=4, label='Mean')
    axes[1, 1].fill_between(episode_df['episode'], 
                           episode_df['ignore_vehicles_go_mean'] - episode_df['ignore_vehicles_go_std'],
                           episode_df['ignore_vehicles_go_mean'] + episode_df['ignore_vehicles_go_std'],
                           alpha=0.3, color='orange', label='±1 Std Dev')
    axes[1, 1].set_title('Ignore Vehicles GO (%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
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
    """Plot step-level metrics"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Step-Level Training Metrics', fontsize=16)
    
    # Plot 1: Collision count over time
    axes[0, 0].plot(step_df['timestep'], step_df['collision_count'], 'r-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Collision Count Over Time')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Collision Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Vehicles exited over time
    axes[0, 1].plot(step_df['timestep'], step_df['vehicles_exited'], 'g-', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Vehicles Exited Over Time')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Vehicles Exited')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average acceleration over time
    axes[1, 0].plot(step_df['timestep'], step_df['avg_acceleration'], 'purple', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Average Acceleration Over Time')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Acceleration (m/s²)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Total controlled vehicles over time
    axes[1, 1].plot(step_df['timestep'], step_df['total_controlled'], 'brown', linewidth=1, alpha=0.7)
    axes[1, 1].set_title('Total Controlled Vehicles Over Time')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Vehicles Controlled')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'step_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Step metrics plot saved: {plot_path}")
    
    plt.show()

def _plot_simulation_time(episode_df: pd.DataFrame, plots_dir: str, save_plots: bool):
    """Plot simulation time metrics"""
    
    # Check available columns and provide fallbacks
    available_columns = episode_df.columns.tolist()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulation Time Metrics', fontsize=16)
    
    # Plot 1: Episode simulation time
    episode_time_col = 'episode_simulation_time'
    if episode_time_col in available_columns:
        axes[0, 0].plot(episode_df['episode'], episode_df[episode_time_col], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Episode Simulation Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, f'Column not found:\n{episode_time_col}', 
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Episode Simulation Time (Data Unavailable)')
    
    # Plot 2: Total simulation time
    total_time_col = 'total_simulation_time'
    if total_time_col in available_columns:
        axes[0, 1].plot(episode_df['episode'], episode_df[total_time_col], 'r-o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Total Simulation Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, f'Column not found:\n{total_time_col}', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Total Simulation Time (Data Unavailable)')
    
    # Plot 3: Episode duration in hours
    episode_hours_col = 'episode_duration_hours'
    if episode_hours_col in available_columns:
        axes[1, 0].plot(episode_df['episode'], episode_df[episode_hours_col], 'g-o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Episode Duration (Hours)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Duration (hours)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, f'Column not found:\n{episode_hours_col}', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Episode Duration (Hours) (Data Unavailable)')
    
    # Plot 4: Total duration in hours
    total_hours_col = 'total_duration_hours'
    if total_hours_col in available_columns:
        axes[1, 1].plot(episode_df['episode'], episode_df[total_hours_col], 'orange', marker='o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Total Duration (Hours)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Duration (hours)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f'Column not found:\n{total_hours_col}', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Total Duration (Hours) (Data Unavailable)')
    
    # Set x-axis to show integer episode numbers and ensure proper y-axis scaling
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Force matplotlib to use actual data range for y-axis
        ax.autoscale_view()
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = os.path.join(plots_dir, 'simulation_time.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Simulation time plot saved: {plot_path}")
    
    plt.show()

def generate_summary_report(results_dir: str, plots_dir: str):
    """Generate a summary report of training metrics"""
    
    episode_metrics_path = os.path.join(results_dir, 'episode_metrics.csv')
    
    if not os.path.exists(episode_metrics_path):
        print(f"⚠️ Episode metrics file not found: {episode_metrics_path}")
        return
    
    try:
        episode_df = pd.read_csv(episode_metrics_path)
        
        # Check available columns and provide fallbacks
        available_columns = episode_df.columns.tolist()
        print(f"📊 Available columns for summary: {available_columns}")
        
        # Calculate summary statistics with fallbacks
        total_episodes = len(episode_df)
        
        # Vehicles exited
        vehicles_exited_col = 'total_vehicles_exited' if 'total_vehicles_exited' in available_columns else 'vehicles_exited'
        total_vehicles_exited = episode_df[vehicles_exited_col].sum() if vehicles_exited_col in available_columns else 0
        
        # Collisions
        collisions_col = 'total_collisions' if 'total_collisions' in available_columns else 'collision_count'
        total_collisions = episode_df[collisions_col].sum() if collisions_col in available_columns else 0
        
        # Deadlocks
        deadlocks_col = 'total_deadlocks' if 'total_deadlocks' in available_columns else 'deadlocks_detected'
        total_deadlocks = episode_df[deadlocks_col].sum() if deadlocks_col in available_columns else 0
        
        # Throughput
        throughput_col = 'avg_throughput' if 'avg_throughput' in available_columns else 'throughput'
        avg_throughput = episode_df[throughput_col].mean() if throughput_col in available_columns else 0.0
        
        # Episode length
        episode_length_col = 'episode_length' if 'episode_length' in available_columns else 'episode'
        avg_episode_length = episode_df[episode_length_col].mean() if episode_length_col in available_columns else 0.0
        
        # Action parameter statistics with fallbacks
        final_urgency_ratio = 0.0
        if 'urgency_position_ratio_mean' in available_columns:
            final_urgency_ratio = episode_df['urgency_position_ratio_mean'].iloc[-1]
        elif 'urgency_position_ratio' in available_columns:
            final_urgency_ratio = episode_df['urgency_position_ratio'].iloc[-1]
            
        final_speed_modifier = 0.0
        if 'speed_diff_modifier_mean' in available_columns:
            final_speed_modifier = episode_df['speed_diff_modifier_mean'].iloc[-1]
        elif 'speed_diff_modifier' in available_columns:
            final_speed_modifier = episode_df['speed_diff_modifier'].iloc[-1]
            
        final_max_participants = 0.0
        if 'max_participants_mean' in available_columns:
            final_max_participants = episode_df['max_participants_mean'].iloc[-1]
        elif 'max_participants_per_auction' in available_columns:
            final_max_participants = episode_df['max_participants_per_auction'].iloc[-1]
            
        final_ignore_vehicles = 0.0
        if 'ignore_vehicles_go_mean' in available_columns:
            final_ignore_vehicles = episode_df['ignore_vehicles_go_mean'].iloc[-1]
        elif 'ignore_vehicles_go' in available_columns:
            final_ignore_vehicles = episode_df['ignore_vehicles_go'].iloc[-1]
        
        # Simulation time statistics
        episode_simulation_time = 0.0
        total_simulation_time = 0.0
        episode_duration_hours = 0.0
        total_duration_hours = 0.0
        
        if 'episode_simulation_time' in available_columns:
            episode_simulation_time = episode_df['episode_simulation_time'].iloc[-1]
            episode_duration_hours = episode_df['episode_duration_hours'].iloc[-1] if 'episode_duration_hours' in available_columns else 0.0
        
        if 'total_simulation_time' in available_columns:
            total_simulation_time = episode_df['total_simulation_time'].iloc[-1]
            total_duration_hours = episode_df['total_duration_hours'].iloc[-1] if 'total_duration_hours' in available_columns else 0.0
        
        # Generate report
        report_path = os.path.join(plots_dir, 'training_summary.txt')
        with open(report_path, 'w') as f:
            f.write("DRL Training Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Vehicles Exited: {total_vehicles_exited}\n")
            f.write(f"Total Collisions: {total_collisions}\n")
            f.write(f"Total Deadlocks: {total_deadlocks}\n\n")
            
            f.write(f"Average Throughput: {avg_throughput:.1f} vehicles/h\n")
            f.write(f"Average Episode Length: {avg_episode_length:.1f} steps\n\n")
            
            f.write("Simulation Time Statistics:\n")
            f.write(f"  Episode Simulation Time: {episode_simulation_time:.1f} seconds ({episode_duration_hours:.3f} hours)\n")
            f.write(f"  Total Simulation Time: {total_simulation_time:.1f} seconds ({total_duration_hours:.3f} hours)\n\n")
            
            f.write("Final Action Parameters:\n")
            f.write(f"  Urgency Position Ratio: {final_urgency_ratio:.3f}\n")
            f.write(f"  Speed Diff Modifier: {final_speed_modifier:.1f}\n")
            f.write(f"  Max Participants: {final_max_participants:.1f}\n")
            f.write(f"  Ignore Vehicles GO: {final_ignore_vehicles:.1f}%\n\n")
            
            f.write("Performance Analysis:\n")
            if total_collisions == 0:
                f.write("  ✅ No collisions detected during training\n")
            else:
                f.write(f"  ⚠️ {total_collisions} collisions detected\n")
            
            if total_deadlocks == 0:
                f.write("  ✅ No deadlocks detected during training\n")
            else:
                f.write(f"  ⚠️ {total_deadlocks} deadlocks detected\n")
            
            if avg_throughput > 100:
                f.write("  ✅ Good throughput performance\n")
            else:
                f.write("  ⚠️ Low throughput performance\n")
        
        print(f"📋 Training summary report saved: {report_path}")
        
    except Exception as e:
        print(f"❌ Error generating summary report: {e}")

def main():
    """Main function to generate plots from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DRL training plots')
    parser.add_argument('--results-dir', type=str, default='drl/results', help='Results directory path')
    parser.add_argument('--plots-dir', type=str, default='drl/plots', help='Plots output directory')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save plots to disk')
    
    args = parser.parse_args()
    
    save_plots = not args.no_save
    
    print(f"🎨 Generating DRL training plots...")
    print(f"   Results directory: {args.results_dir}")
    print(f"   Plots directory: {args.plots_dir}")
    print(f"   Save plots: {save_plots}")
    
    plot_training_metrics(args.results_dir, args.plots_dir, save_plots)
    generate_summary_report(args.results_dir, args.plots_dir)

if __name__ == "__main__":
    main()