# Episode-Level Parameter Updates

## Overview

The DRL training system now uses **episode-level parameter updates** instead of step-level updates. This means that the 4 trainable action space parameters are only updated when a new episode begins (at reset), not on every simulation step.

## Why Episode-Level Updates?

### 1. **Better Learning Dynamics**
- **Consistent behavior**: Parameters remain constant during an episode, allowing the agent to explore the consequences of its choices
- **Stable exploration**: The agent can see how its parameter settings affect the entire episode outcome
- **Reduced noise**: No parameter changes mid-episode that could confuse the learning process

### 2. **Improved Convergence**
- **Episode-level feedback**: The agent learns from complete episode outcomes, not individual step rewards
- **Parameter exploration**: Each episode represents a different parameter combination to explore
- **Better credit assignment**: Clear connection between parameter choices and episode performance

### 3. **More Realistic Training**
- **Real-world analogy**: In practice, traffic control parameters are set and remain constant for extended periods
- **Stable evaluation**: Easier to evaluate the effectiveness of specific parameter combinations

## How It Works

### Parameter Update Flow

```
Episode Start (reset) → Update 4 trainable parameters → Episode runs with constant parameters → Episode ends → Next episode starts → New parameters applied
```

### The 4 Trainable Parameters

1. **`urgency_position_ratio`** (0.1-3.0): Controls emergency vs. position advantage balance
2. **`speed_diff_modifier`** (-30 to +30): Speed control correction factor
3. **`max_participants_per_auction`** (3-6): Maximum participants per auction round
4. **`ignore_vehicles_go`** (0-80%): Percentage of vehicles to ignore in GO state

### Implementation Details

- **Episode tracking**: `_episode_params_updated` flag tracks when parameters were last updated
- **Reset detection**: Parameters are only updated when `_episode_params_updated = False`
- **Parameter persistence**: Once set, parameters remain constant until the next episode
- **Parameter caching**: `_current_episode_action_params` stores parameters for reuse throughout episode
- **Verification**: All parameter updates are verified and logged for debugging

### Parameter Caching Mechanism

```python
# During episode start (reset)
if not self._episode_params_updated:
    self._update_policy_parameters(action_params)  # Update parameters
    self._episode_params_updated = True           # Mark as updated
    self._current_episode_action_params = action_params.copy()  # Cache for episode

# During episode steps
else:
    cached_params = self.get_current_episode_action_params()  # Get cached parameters
    self._apply_cached_parameters(cached_params)             # Apply cached values
```

## Benefits

### For Training
- **Faster convergence**: More stable learning environment
- **Better exploration**: Clear episode-level parameter exploration
- **Reduced variance**: Less parameter noise during episodes

### For Evaluation
- **Clearer results**: Each episode represents a specific parameter combination
- **Better analysis**: Easier to correlate parameters with performance
- **Stable metrics**: No mid-episode parameter changes affecting measurements

## Configuration

### Training Parameters
- **`max_steps`**: Episode length (e.g., 128 steps)
- **`n_steps`**: PPO update frequency (e.g., 256 steps)
- **Episode alignment**: `n_steps` should be a multiple of `max_steps` for optimal training

### Example Configuration
```yaml
# PPO Configuration
n_steps: 256          # Update every 256 steps
max_steps: 128        # Episode length

# This means:
# - 2 episodes per PPO update
# - Parameters change every 128 steps (episode boundary)
# - Stable parameter exploration during each episode
```

## Monitoring

### During Training
- **Episode start**: Parameters are updated and logged
- **Episode progress**: Parameters remain constant
- **Episode end**: Episode performance is recorded
- **Next episode**: New parameters are applied

### Logging Output
```
🔄 NEW EPISODE: Updating 4 trainable parameters
🔧 Updated 2 trainable parameters (deadlock avoidance focus):
   ✅ urgency_position_ratio: 1.200 → 1.850
   ✅ speed_diff_modifier: 5.0 → -12.0
🔒 These parameters will remain constant for this episode (until next reset)
```

**Note**: Parameters are only printed when they actually change. If no parameters change, you'll see:
```
🔒 Using existing parameters (no changes needed)
```

### Verbose Logging Control

You can control the verbosity of parameter logging:

```python
# Enable verbose parameter logging (shows all parameter details)
env.set_verbose_parameter_logging(True)

# Disable verbose parameter logging (minimal output)
env.set_verbose_parameter_logging(False)
```

When verbose logging is disabled:
- Only changed parameters are shown
- Unchanged parameters show minimal output
- Reduces console clutter during training

### Parameter Verification
```python
# Get current parameter values
current_params = env.get_current_parameter_values()
print(f"Episode params updated: {current_params['episode_params_updated']}")
print(f"Current urgency_position_ratio: {current_params['urgency_position_ratio']}")
```

## Best Practices

### 1. **Episode Length**
- Keep episodes long enough to see parameter effects
- Balance between exploration and training efficiency
- Consider the complexity of the traffic scenario

### 2. **Update Frequency**
- Align PPO updates with episode boundaries
- Use multiples of episode length for `n_steps`
- Avoid mid-episode parameter changes

### 3. **Parameter Ranges**
- Ensure parameter ranges allow meaningful exploration
- Use appropriate mapping functions (sigmoid, quantized, discrete)
- Test parameter distribution across intended ranges

## Troubleshooting

### Common Issues

1. **Parameters not updating**: Check if `_episode_params_updated` flag is being reset
2. **Frequent parameter changes**: Verify that updates only happen at episode boundaries
3. **Poor exploration**: Check parameter mapping functions and ranges

### Debug Commands

```python
# Check current parameter values
current_params = env.get_current_parameter_values()

# Test parameter mapping
env.test_parameter_mapping(num_samples=100)

# Verify episode boundary updates
print(f"Episode params updated: {env.sim._episode_params_updated}")

# Check cached episode parameters
print(f"Cached episode params: {env.sim._current_episode_action_params}")
```

## Conclusion

Episode-level parameter updates provide a more stable and effective training environment for the DRL agent. By maintaining consistent parameters throughout each episode, the agent can better explore the parameter space and learn more effectively from its experiences.

This approach aligns with real-world traffic control scenarios where parameters are set and remain constant for extended periods, leading to more practical and robust learning outcomes.

### Key Implementation Features

- **Automatic episode detection**: Uses `_episode_params_updated` flag
- **Parameter caching**: Stores parameters in `_current_episode_action_params`
- **Consistent application**: Applies cached parameters throughout episode
- **Reset synchronization**: Automatically resets flag during environment reset
- **Debug logging**: Comprehensive logging for parameter update verification
- **Smart logging**: Only prints parameter updates when values actually change
- **Verbose control**: Configurable logging verbosity for training vs. debugging
