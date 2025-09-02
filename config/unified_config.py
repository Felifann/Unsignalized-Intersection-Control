"""
Unified Configuration Management for Unisignalized Intersection Control System

This module provides centralized configuration management for all system components:
- System-wide settings
- Conflict detection parameters  
- MWIS solver parameters
- Deadlock detection and prevention
- DRL training parameters
- Simulation environment settings

FIXED TIME HIERARCHY DESIGN:
===========================
The system uses a 4-level time hierarchy for optimal performance and proper vehicle control:

Level 1: Simulation Step (0.1s)
  - fixed_delta_seconds = 0.1s
  - Basic physics and vehicle movement updates
  - Smooth, responsive simulation

Level 2: Decision Step (1.0s) 
  - logic_update_interval_seconds = 1.0s (REDUCED from 2.0s for better responsiveness)
  - Vehicle behavior decisions and route planning
  - 10 simulation steps per decision (REDUCED from 20 for better control)

Level 3: Auction Cycle (4.0s) (REDUCED from 6.0s for better synchronization)
  - auction_interval = 4.0s (total cycle)
  - bidding_duration = 2.0s (50% bidding, 50% execution)
  - 4 decision steps per auction cycle (PERFECT synchronization)
  - Allows proper bidding and execution phases with clear timing

Level 4: System Check (8.0s) (REDUCED from 12.0s)
  - deadlock_check_interval = 8.0s
  - 2 auction cycles per system check
  - High-level system health monitoring

This FIXED hierarchy ensures:
- Smooth simulation (0.1s)
- Responsive decisions (1.0s) - TWICE as responsive as before
- Proper auction timing (4.0s) - PERFECT synchronization with decision steps
- Efficient system monitoring (8.0s)
- Vehicles properly respect 'wait' commands due to synchronized timing
- No more "all vehicles moving together" issue
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import copy


@dataclass
class SystemConfig:
    """System-wide configuration parameters"""
    # Core intersection settings
    intersection_center: Tuple[float, float, float] = (-188.9, -89.7, 0.0)
    intersection_half_size: float = 40.0
    
    # Training and simulation modes
    training_mode: bool = False
    steps_per_action: int = 1
    observation_cache_steps: int = 5
    
    # FIXED Time hierarchy design:
    # fixed_delta_seconds (0.1s) -> logic_update_interval (1.0s) -> auction_cycle (4.0s)
    # This creates a PERFECT 3-level hierarchy: simulation step -> decision step -> auction cycle
    # 10 sim steps per decision, 4 decisions per auction cycle = PERFECT synchronization
    logic_update_interval_seconds: float = 1.0  # Decision making interval (10 simulation steps) - REDUCED for better control
    
    # Deadlock handling
    severe_deadlock_reset_enabled: bool = True
    severe_deadlock_punishment: float = -800.0
    
    # Map and CARLA settings
    map_name: str = 'Town05'
    carla_host: str = 'localhost'
    carla_port: int = 2000
    carla_timeout: float = 10.0
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.1  # Simulation step size - keep small for smooth simulation
    
    # Traffic generation
    max_vehicles: int = 500
    spawn_rate: float = 1.0


@dataclass
class ConflictConfig:
    """Conflict detection and analysis parameters"""
    # FIXED Time hierarchy: conflict_time_window should be 2-3x logic_update_interval
    # This allows for proper conflict prediction and resolution planning
    conflict_time_window: float = 2.5  # seconds to predict conflicts (2.5x logic_update_interval) - REDUCED for better responsiveness
    min_safe_distance: float = 3.0     # minimum safe distance between vehicles (meters)
    collision_threshold: float = 2.0   # distance below which collision is imminent
    prediction_steps: int = 50          # number of steps to predict ahead (increased for better coverage)
    velocity_threshold: float = 0.1     # minimum velocity to consider for prediction
    
    # TRAINABLE NASH PARAMETERS - optimizable via DRL
    path_intersection_threshold: float = 2.5   # path intersection sensitivity (meters)
    platoon_conflict_distance: float = 15.0   # platoon interaction distance (meters)


@dataclass
class MWISConfig:
    """Maximum Weight Independent Set solver parameters"""
    max_go_agents: Optional[int] = None  # None = unlimited, int = max agents that can go
    max_exact: int = 15                  # threshold for exact vs heuristic algorithm
    weight_factor: float = 1.0           # weight factor for vehicle priorities
    timeout_seconds: float = 5.0         # maximum solving time
    prefer_exact: bool = True            # prefer exact solution when possible


@dataclass
class AuctionConfig:
    """Auction system configuration parameters"""
    max_participants_per_auction: int = 4  # Maximum agents per auction round (prevents mass movement)
    
    # FIXED Auction cycle design:
    # Total auction cycle = bidding_duration + auction_interval = 4.0 seconds (REDUCED from 6.0s)
    # This creates PERFECT synchronization: 2s bidding + 2s execution = 4s total cycle
    # Logic updates every 1s, so we get 4 logic updates per auction cycle (PERFECT alignment)
    # This ensures vehicles properly respect 'wait' commands and don't move together
    auction_interval: float = 4.0          # seconds between auction cycles (total cycle time) - REDUCED for better sync
    bidding_duration: float = 2.0          # duration of bidding phase (50% of cycle) - REDUCED for better sync
    
    priority_in_transit_weight: float = 2.0  # weight multiplier for agents already in transit
    priority_distance_weight: float = 1.5    # weight multiplier for proximity to intersection


@dataclass
class DeadlockConfig:
    """Deadlock detection and prevention parameters"""
    # Deadlock detection and resolution - OPTIMIZED for new time hierarchy
    deadlock_speed_threshold: float = 0.2      # m/s - vehicles below this are considered stopped
    deadlock_detection_window: float = 30.0    # seconds to track for deadlock detection
    deadlock_min_vehicles: int = 6             # minimum vehicles for deadlock detection
    
    # FIXED Time hierarchy: deadlock_check_interval should be 2x auction_cycle for perfect sync
    # This allows for proper deadlock detection without interfering with auction cycles
    deadlock_check_interval: float = 8.0       # check every 8 seconds (2x auction cycle) - REDUCED for better sync
    
    deadlock_severity_threshold: float = 0.95  # 95% of vehicles stalled
    deadlock_duration_threshold: float = 45.0  # 45 seconds continuous stalling
    deadlock_timeout_duration: float = 90.0    # seconds before timeout reset
    deadlock_core_half_size: float = 5.0       # core region half size for deadlock detection
    
    # Timeout and reset settings
    max_deadlock_resets: int = 3               # maximum auto-resets per episode


@dataclass
class DRLConfig:
    """Deep Reinforcement Learning training parameters"""
    # PPO parameters
    learning_rate: float = 1e-4
    n_steps: int = 128
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training schedule
    total_timesteps: int = 100000
    eval_freq: int = 1000
    checkpoint_freq: int = 1000
    warmup_steps: int = 32000
    decay_steps: int = 100000
    min_learning_rate: float = 1e-5
    
    # Simulation for training
    max_steps: int = 128
    n_eval_episodes: int = 5
    eval_deterministic: bool = True
    
    # Reward structure
    vehicle_exit_reward: float = 10.0
    throughput_bonus: float = 0.01
    acceleration_penalty_threshold: float = 3.0
    acceleration_penalty_factor: float = 2.0
    efficiency_bonus: float = 5.0
    collision_penalty: float = 100.0
    deadlock_penalty: float = 800.0
    step_penalty: float = 0.1


@dataclass
class UnifiedConfig:
    """
    Unified configuration container for all system components.
    
    This class centralizes all configuration parameters and provides
    methods to convert to component-specific configurations.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    mwis: MWISConfig = field(default_factory=MWISConfig)
    auction: AuctionConfig = field(default_factory=AuctionConfig)
    deadlock: DeadlockConfig = field(default_factory=DeadlockConfig)
    drl: DRLConfig = field(default_factory=DRLConfig)
    
    def update_from_drl_params(self, **kwargs):
        """Update configuration from DRL training parameters"""
        # Map DRL parameters to unified config fields
        param_mapping = {
            'conflict_time_window': ('conflict', 'conflict_time_window'),
            'min_safe_distance': ('conflict', 'min_safe_distance'),
            'path_intersection_threshold': ('conflict', 'path_intersection_threshold'),
            'platoon_conflict_distance': ('conflict', 'platoon_conflict_distance'),
            'max_go_agents': ('mwis', 'max_go_agents'),
            'max_participants_per_auction': ('auction', 'max_participants_per_auction'),
            'auction_interval': ('auction', 'auction_interval'),
            'bidding_duration': ('auction', 'bidding_duration'),
            'deadlock_speed_threshold': ('deadlock', 'deadlock_speed_threshold'),
            'deadlock_timeout_duration': ('deadlock', 'deadlock_timeout_duration'),
            'learning_rate': ('drl', 'learning_rate'),
            'max_steps': ('drl', 'max_steps'),
            'batch_size': ('drl', 'batch_size'),
            'intersection_center': ('system', 'intersection_center'),
            'training_mode': ('system', 'training_mode'),
        }
        
        updated_params = []
        for param_name, value in kwargs.items():
            if param_name in param_mapping:
                section, field_name = param_mapping[param_name]
                section_obj = getattr(self, section)
                setattr(section_obj, field_name, value)
                updated_params.append(param_name)
    
    def to_solver_config(self) -> Dict[str, Any]:
        """Convert to solver configuration dictionary"""
        return {
            # System parameters
            'intersection_center': self.system.intersection_center,
            'intersection_half_size': self.system.intersection_half_size,
            
            # Conflict parameters
            'conflict_time_window': self.conflict.conflict_time_window,
            'min_safe_distance': self.conflict.min_safe_distance,
            'collision_threshold': self.conflict.collision_threshold,
            'prediction_steps': self.conflict.prediction_steps,
            'velocity_threshold': self.conflict.velocity_threshold,
            'path_intersection_threshold': self.conflict.path_intersection_threshold,
            'platoon_conflict_distance': self.conflict.platoon_conflict_distance,
            
            # MWIS parameters
            'max_go_agents': self.mwis.max_go_agents,
            'max_exact': self.mwis.max_exact,
            'weight_factor': self.mwis.weight_factor,
            'timeout_seconds': self.mwis.timeout_seconds,
            'prefer_exact': self.mwis.prefer_exact,
            
            # Auction parameters
            'max_participants_per_auction': self.auction.max_participants_per_auction,
            'auction_interval': self.auction.auction_interval,
            'bidding_duration': self.auction.bidding_duration,
            'priority_in_transit_weight': self.auction.priority_in_transit_weight,
            'priority_distance_weight': self.auction.priority_distance_weight,
            
            # Deadlock parameters
            'deadlock_speed_threshold': self.deadlock.deadlock_speed_threshold,
            'deadlock_detection_window': self.deadlock.deadlock_detection_window,
            'deadlock_min_vehicles': self.deadlock.deadlock_min_vehicles,
            'deadlock_check_interval': self.deadlock.deadlock_check_interval,
            'deadlock_severity_threshold': self.deadlock.deadlock_severity_threshold,
            'deadlock_duration_threshold': self.deadlock.deadlock_duration_threshold,
            'deadlock_core_half_size': self.deadlock.deadlock_core_half_size,
        }
    
    def to_sim_config(self) -> Dict[str, Any]:
        """Convert to simulation configuration dictionary"""
        return {
            # Map and CARLA
            'map': self.system.map_name,
            'carla_host': self.system.carla_host,
            'carla_port': self.system.carla_port,
            'carla_timeout': self.system.carla_timeout,
            'synchronous_mode': self.system.synchronous_mode,
            'fixed_delta_seconds': self.system.fixed_delta_seconds,
            'logic_update_interval_seconds': self.system.logic_update_interval_seconds,
            
            # Traffic and intersection
            'max_vehicles': self.system.max_vehicles,
            'spawn_rate': self.system.spawn_rate,
            'intersection_center': self.system.intersection_center,
            'intersection_half_size': self.system.intersection_half_size,
            
            # Training
            'training_mode': self.system.training_mode,
            'max_steps': self.drl.max_steps,
            'steps_per_action': self.system.steps_per_action,
            'observation_cache_steps': self.system.observation_cache_steps,
            
            # Auction parameters
            'max_participants_per_auction': self.auction.max_participants_per_auction,
            'auction_interval': self.auction.auction_interval,
            'bidding_duration': self.auction.bidding_duration,
            
            # Deadlock handling
            'deadlock_reset_enabled': True,
            'deadlock_timeout_duration': self.deadlock.deadlock_timeout_duration,
            'max_deadlock_resets': self.deadlock.max_deadlock_resets,
            'severe_deadlock_reset_enabled': self.system.severe_deadlock_reset_enabled,
            'severe_deadlock_punishment': self.system.severe_deadlock_punishment,
        }
    
    def to_drl_config(self) -> Dict[str, Any]:
        """Convert to DRL training configuration dictionary"""
        return {
            # PPO parameters
            'learning_rate': self.drl.learning_rate,
            'n_steps': self.drl.n_steps,
            'batch_size': self.drl.batch_size,
            'n_epochs': self.drl.n_epochs,
            'gamma': self.drl.gamma,
            'gae_lambda': self.drl.gae_lambda,
            'clip_range': self.drl.clip_range,
            'ent_coef': self.drl.ent_coef,
            'vf_coef': self.drl.vf_coef,
            'max_grad_norm': self.drl.max_grad_norm,
            
            # Training schedule
            'total_timesteps': self.drl.total_timesteps,
            'eval_freq': self.drl.eval_freq,
            'checkpoint_freq': self.drl.checkpoint_freq,
            'warmup_steps': self.drl.warmup_steps,
            'decay_steps': self.drl.decay_steps,
            'min_learning_rate': self.drl.min_learning_rate,
            
            # Simulation settings
            'max_steps': self.drl.max_steps,
            'n_eval_episodes': self.drl.n_eval_episodes,
            'eval_deterministic': self.drl.eval_deterministic,
            
            # Reward structure
            'vehicle_exit_reward': self.drl.vehicle_exit_reward,
            'throughput_bonus': self.drl.throughput_bonus,
            'acceleration_penalty_threshold': self.drl.acceleration_penalty_threshold,
            'acceleration_penalty_factor': self.drl.acceleration_penalty_factor,
            'efficiency_bonus': self.drl.efficiency_bonus,
            'collision_penalty': self.drl.collision_penalty,
            'deadlock_penalty': self.drl.deadlock_penalty,
            'step_penalty': self.drl.step_penalty,
        }
    
    def copy(self) -> 'UnifiedConfig':
        """Create a deep copy of the configuration"""
        return copy.deepcopy(self)

# Global configuration instance
_global_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global unified configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = UnifiedConfig()
    return _global_config


def set_config(config: UnifiedConfig):
    """Set the global unified configuration instance"""
    global _global_config
    _global_config = config


def reset_config():
    """Reset the global configuration to default values"""
    global _global_config
    _global_config = UnifiedConfig()


def print_config_summary(config: Optional[UnifiedConfig] = None):
    """Print a summary of the configuration"""
    if config is None:
        config = get_config()
    print(config.summary())


def load_config_from_yaml(yaml_path: str) -> UnifiedConfig:
    """Load configuration from a YAML file (placeholder for future implementation)"""
    # TODO: Implement YAML loading if needed
    print(f"📄 Loading config from {yaml_path} (not yet implemented)")
    return get_config()


def save_config_to_yaml(config: UnifiedConfig, yaml_path: str):
    """Save configuration to a YAML file (placeholder for future implementation)"""
    # TODO: Implement YAML saving if needed
    print(f"💾 Saving config to {yaml_path} (not yet implemented)")


# Initialize default configuration on import
if _global_config is None:
    _global_config = UnifiedConfig()
