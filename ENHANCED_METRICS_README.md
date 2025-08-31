# Enhanced DRL Training Metrics

## 概述

这个增强版本的DRL训练系统提供了更详细的metrics记录和分析功能，满足以下需求：

1. **碰撞数、死锁数、exit车辆数根据timestep的总变化趋势**
2. **四个action space参数和单个episode里reward根据episode的变化趋势**
3. **Training summary内的数据保存原样**
4. **所有数据都是实际数据，不进行归一化**

## 新增功能

### 1. Timestep级别的累积趋势分析

新增了`timestep_trends.csv`文件，记录：
- `cumulative_collisions`: 累积碰撞总数（整个训练过程）
- `cumulative_deadlocks`: 累积死锁总数（整个训练过程）
- `cumulative_vehicles_exited`: 累积退出车辆总数（整个训练过程）
- `episode_cumulative_reward`: 当前episode的累积reward

**重要修复**: 已修复timestep累积计算逻辑，确保：
- **Episode级别重置**: 每个episode开始时，碰撞数、死锁数等会在模拟环境中重置为0（用于reward计算）
- **CSV累积记录**: CSV文件记录的是整个训练过程中的累积总数，不会因为episode重置而丢失
- **双重记录**: 同时记录当前值和全局累积值，满足不同分析需求

### 2. 增强的Episode级别Metrics

每个episode结束时记录：
- **Action Space参数统计**:
  - `urgency_position_ratio_mean/var/std`: 紧急度位置比率
  - `speed_diff_modifier_mean/var/std`: 速度差异修正器
  - `max_participants_mean/var/std`: 最大参与者数量
  - `ignore_vehicles_go_mean/var/std`: 忽略车辆GO百分比

- **Reward统计**:
  - `episode_total_reward`: episode总reward
  - `episode_avg_reward`: episode平均reward
  - `episode_min_reward`: episode最小reward
  - `episode_max_reward`: episode最大reward
  - `episode_reward_std`: episode reward标准差

### 3. 增强的Step级别Metrics

每个timestep记录：
- `current_reward`: 当前step的reward
- `episode_cumulative_reward`: 当前episode的累积reward
- **全局累积值** (新增):
  - `global_cumulative_collisions`: 整个训练过程的累积碰撞数
  - `global_cumulative_deadlocks`: 整个训练过程的累积死锁数
  - `global_cumulative_exits`: 整个训练过程的累积退出车辆数
- 所有原有的metrics保持不变

## 文件结构

```
drl/
├── train.py                    # 增强的训练脚本
├── utils/
│   └── plot_generator.py      # 增强的绘图工具
└── logs/                      # 训练日志目录
    ├── step_metrics.csv       # Step级别metrics (包含全局累积值)
    ├── episode_metrics.csv    # Episode级别metrics
    └── timestep_trends.csv    # Timestep累积趋势 (整个训练过程)
```

## 使用方法

### 1. 运行训练

```bash
python drl/train.py --total-timesteps 10000
```

### 2. 生成分析图表

训练完成后，系统会自动生成以下图表：

- **Action Parameters**: 四个action space参数的变化趋势
- **Episode Rewards**: Reward的各种统计指标
- **Timestep Trends**: 累积碰撞数、死锁数、退出车辆数的变化趋势（整个训练过程）

### 3. 手动生成图表

```bash
python -m drl.utils.plot_generator --results-dir drl/logs --plots-dir drl/plots
```

## 数据格式

### Step Metrics (step_metrics.csv)
```csv
timestep,episode,throughput,avg_acceleration,collision_count,total_controlled,vehicles_exited,urgency_position_ratio,speed_diff_modifier,max_participants_per_auction,ignore_vehicles_go,deadlocks_detected,deadlock_severity,current_reward,episode_cumulative_reward,global_cumulative_collisions,global_cumulative_deadlocks,global_cumulative_exits
```

### Episode Metrics (episode_metrics.csv)
```csv
episode,episode_start_step,episode_end_step,episode_length,urgency_position_ratio_mean,urgency_position_ratio_var,urgency_position_ratio_std,speed_diff_modifier_mean,speed_diff_modifier_var,speed_diff_modifier_std,max_participants_mean,max_participants_var,max_participants_std,ignore_vehicles_go_mean,ignore_vehicles_go_var,ignore_vehicles_go_std,total_vehicles_exited,total_collisions,total_deadlocks,max_deadlock_severity,avg_throughput,avg_acceleration,total_controlled_vehicles,episode_total_reward,episode_avg_reward,episode_min_reward,episode_max_reward,episode_reward_std
```

### Timestep Trends (timestep_trends.csv)
```csv
timestep,episode,cumulative_collisions,cumulative_deadlocks,cumulative_vehicles_exited,current_collision_count,current_deadlocks_detected,current_vehicles_exited,current_reward,episode_cumulative_reward
```

## 关键特性

1. **实际数据**: 所有metrics都是原始值，不进行归一化
2. **累积趋势**: 提供timestep级别的累积统计（整个训练过程）
3. **双重记录**: 同时记录当前值和全局累积值
4. **完整记录**: 保存所有训练过程中的详细数据
5. **自动分析**: 训练完成后自动生成分析图表
6. **错误处理**: 增强的错误处理和数据验证
7. **精确计算**: 修复的timestep累积计算逻辑，确保数据准确性
8. **Episode重置兼容**: 支持episode级别的重置，同时保持全局累积记录

## 重要说明

### Episode重置 vs CSV累积记录

- **Episode重置**: 每个episode开始时，模拟环境会重置碰撞数、死锁数等计数器为0，这是正常的DRL训练行为
- **CSV累积记录**: CSV文件记录的是整个训练过程中的累积总数，不会因为episode重置而丢失
- **双重用途**: 
  - 重置的值用于episode内的reward计算
  - 累积值用于分析整个训练过程的趋势

## 注意事项

1. 确保有足够的磁盘空间存储详细的metrics数据
2. 对于长时间训练，建议定期清理旧的metrics文件
3. 所有数据都以CSV格式保存，便于后续分析
4. 图表生成需要matplotlib库支持
5. 只生成你需要的图表：timestep趋势、action parameters和rewards
6. 理解episode重置和CSV累积记录的区别

## 故障排除

如果遇到问题：

1. 检查CSV文件是否正确生成
2. 验证数据列名是否匹配
3. 确保所有必需的Python库已安装
4. 查看控制台输出的错误信息
5. 检查timestep累积计算是否正确
6. 确认全局累积计数器是否正常工作
7. 理解episode重置和全局累积记录的区别
