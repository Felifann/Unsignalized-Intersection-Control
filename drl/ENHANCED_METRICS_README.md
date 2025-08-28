# Enhanced DRL Metrics Logging

This document describes the enhanced metrics logging system for DRL training that tracks action space parameters and per-episode statistics.

## Overview

The enhanced metrics logging system provides comprehensive tracking of:
1. **Action Space Parameters**: Mean, variance, and standard deviation of all 4 trainable parameters per episode
2. **Per-Episode Statistics**: Vehicles exited, collisions, deadlocks, and performance metrics
3. **Step-Level Metrics**: Real-time tracking of training progress
4. **Automatic Plotting**: Generation of visualization plots and summary reports

## Action Space Parameters Tracked

The system tracks the 4 critical trainable parameters for deadlock avoidance:

| Parameter | Description | Range | Mapping |
|-----------|-------------|-------|---------|
| `urgency_position_ratio` | 紧急度vs位置优势关系因子 | 0.1-3.0 | Sigmoid |
| `speed_diff_modifier` | 速度控制修正 | -30 to +30 | Quantized |
| `max_participants_per_auction` | 拍卖参与者数量 | 3-6 | Discrete |
| `ignore_vehicles_go` | GO状态ignore_vehicles% | 0-80% | Quantized |

## Metrics Structure

### Episode-Level Metrics (`episode_metrics.csv`)

Each row represents one episode with the following columns:

#### Action Parameter Statistics
- `urgency_position_ratio_mean/var/std`: Statistics for parameter 1
- `speed_diff_modifier_mean/var/std`: Statistics for parameter 2  
- `max_participants_mean/var/std`: Statistics for parameter 3
- `ignore_vehicles_go_mean/var/std`: Statistics for parameter 4

#### Performance Metrics
- `total_vehicles_exited`: Number of vehicles that successfully exited
- `total_collisions`: Number of collisions detected
- `total_deadlocks`: Number of deadlocks detected
- `max_deadlock_severity`: Maximum deadlock severity during episode
- `avg_throughput`: Average throughput (vehicles/hour)
- `avg_acceleration`: Average acceleration patterns
- `total_controlled_vehicles`: Total vehicles under control

#### Episode Metadata
- `episode`: Episode number
- `episode_length`: Number of steps in episode
- `episode_start_step`: Starting timestep
- `episode_end_step`: Ending timestep

### Step-Level Metrics (`step_metrics.csv`)

Each row represents one training step with real-time metrics:

- `timestep`: Current training step
- `episode`: Current episode number
- `throughput`: Current throughput
- `collision_count`: Cumulative collision count
- `vehicles_exited`: Cumulative vehicles exited
- Current values of all 4 trainable parameters
- Safety metrics (deadlock detection, severity)

## Implementation

### Enhanced Metrics Callback

The system uses an enhanced `MetricsCallback` class that:

1. **Tracks Actions**: Stores all actions taken during each episode
2. **Detects Episode Boundaries**: Automatically identifies when episodes end
3. **Calculates Statistics**: Computes mean, variance, and standard deviation
4. **Saves Data**: Writes metrics to CSV files with proper file handling
5. **Cleanup**: Properly finalizes episodes when training ends

### Integration Points

The enhanced metrics are integrated into:

- **PPO Trainer** (`drl/agents/ppo_trainer.py`)
- **SAC Trainer** (`drl/agents/sac_trainer.py`) 
- **Main Training Script** (`drl/train.py`)

## Usage

### Automatic Logging

Metrics are automatically logged during training. No additional configuration is needed.

### Manual Plot Generation

After training, generate plots manually:

```bash
# Generate plots from existing metrics
python -m drl.utils.plot_generator --results-dir drl/results --plots-dir drl/plots

# Generate plots without saving to disk
python -m drl.utils.plot_generator --results-dir drl/results --plots-dir drl/plots --no-save
```

### Test the System

Test the metrics logging functionality:

```bash
python drl/test_enhanced_metrics.py
```

## Generated Outputs

### CSV Files
- `episode_metrics.csv`: Episode-level summaries with action statistics
- `step_metrics.csv`: Step-level detailed metrics

### Plots
- `episode_performance.png`: Performance metrics per episode
- `action_parameters.png`: Action parameter evolution with variance
- `step_metrics.png`: Step-level metric trends

### Reports
- `training_summary.txt`: Human-readable training summary

## Plot Types

### 1. Episode Performance Metrics
- Vehicles exited per episode
- Collisions per episode  
- Deadlocks per episode
- Average throughput per episode

### 2. Action Parameter Trends
- Parameter evolution over episodes
- Variance and standard deviation bands
- Learning progress visualization

### 3. Step-Level Metrics
- Real-time performance tracking
- Safety metric monitoring
- Training progress analysis

## Benefits

### For Training Analysis
- **Parameter Evolution**: Track how action parameters change during training
- **Performance Correlation**: Correlate parameter changes with performance
- **Convergence Analysis**: Identify when training converges
- **Safety Monitoring**: Track collision and deadlock patterns

### For Debugging
- **Parameter Validation**: Verify parameter ranges and distributions
- **Performance Issues**: Identify episodes with poor performance
- **Training Stability**: Monitor for training instability
- **Resource Usage**: Track episode lengths and computational costs

### For Research
- **Comparative Studies**: Compare different training runs
- **Hyperparameter Tuning**: Analyze parameter sensitivity
- **Algorithm Comparison**: Compare PPO vs SAC performance
- **Reproducibility**: Detailed logging for experiment reproduction

## Technical Details

### Episode Boundary Detection
The system uses a heuristic approach to detect episode boundaries:
- Monitors action sequence length
- Detects when actions exceed expected episode length
- Automatically finalizes and starts new episodes

### File Management
- **Rate Limiting**: Prevents excessive file I/O
- **Proper Cleanup**: Handles file handles correctly
- **Error Recovery**: Graceful handling of file errors
- **Memory Management**: Efficient data storage and processing

### Performance Considerations
- **Minimal Overhead**: Metrics collection adds <1% training time
- **Efficient Storage**: Compressed CSV format
- **Smart Caching**: Reduces redundant calculations
- **Background Processing**: Non-blocking metrics collection

## Troubleshooting

### Common Issues

1. **No Metrics Generated**
   - Check if training completed successfully
   - Verify callback is properly registered
   - Check file permissions for output directories

2. **Missing Episode Boundaries**
   - Verify episode length configuration
   - Check if environment resets are working
   - Monitor console output for episode start messages

3. **File I/O Errors**
   - Check disk space availability
   - Verify directory permissions
   - Monitor for file handle leaks

### Debug Commands

```bash
# Check if metrics files exist
ls -la drl/results/

# Verify CSV file contents
head -5 drl/results/episode_metrics.csv
head -5 drl/results/step_metrics.csv

# Test plotting functionality
python -m drl.utils.plot_generator --results-dir drl/results --plots-dir drl/plots
```

## Future Enhancements

### Planned Features
- **TensorBoard Integration**: Real-time visualization during training
- **Custom Metrics**: User-defined metric collection
- **Advanced Analytics**: Statistical analysis and insights
- **Export Formats**: Support for additional output formats

### Extensibility
The system is designed to be easily extensible:
- Add new metrics by extending the callback
- Customize episode boundary detection
- Implement custom plotting functions
- Add new output formats

## Conclusion

The enhanced DRL metrics logging system provides comprehensive tracking of training progress, enabling better understanding of agent behavior, performance analysis, and research insights. The system is designed to be robust, efficient, and easy to use while providing valuable information for training optimization and debugging.
