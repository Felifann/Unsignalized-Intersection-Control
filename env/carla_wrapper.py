import carla
import random
from .simulation_config import SimulationConfig

class CarlaWrapper:
    def __init__(self, host=None, port=None, timeout=None, town=None, unified_config=None):
        """
        Initialize CarlaWrapper with optional unified configuration
        
        Args:
            unified_config: UnifiedConfig object containing dynamic simulation parameters
        """
        self.unified_config = unified_config
        
        # Use unified config if available, otherwise fall back to legacy config
        if unified_config:
            carla_host = host or unified_config.system.carla_host
            carla_port = port or unified_config.system.carla_port
            carla_timeout = timeout or unified_config.system.carla_timeout
            map_name = town or unified_config.system.map_name
            synchronous_mode = unified_config.system.synchronous_mode
            fixed_delta_seconds = unified_config.system.fixed_delta_seconds
        else:
            # Legacy fallback
            carla_host = host or SimulationConfig.CARLA_HOST
            carla_port = port or SimulationConfig.CARLA_PORT
            carla_timeout = timeout or SimulationConfig.CARLA_TIMEOUT
            map_name = town or SimulationConfig.MAP_NAME
            synchronous_mode = SimulationConfig.SYNCHRONOUS_MODE
            fixed_delta_seconds = SimulationConfig.FIXED_DELTA_SECONDS
        
        # Initialize CARLA client
        self.client = carla.Client(carla_host, carla_port)
        self.client.set_timeout(carla_timeout)
        
        # Load world
        self.client.load_world(map_name)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Apply world settings with unified config values
        self._apply_world_settings(synchronous_mode, fixed_delta_seconds)
        
        # 设置全局俯瞰视角
        self.setup_global_overview()
    
    def _apply_world_settings(self, synchronous_mode, fixed_delta_seconds):
        """Apply CARLA world settings"""
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = fixed_delta_seconds
        
        # Set reasonable substep parameters based on fixed_delta_seconds
        if fixed_delta_seconds <= 0.05:
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
        elif fixed_delta_seconds <= 0.1:
            settings.max_substep_delta_time = 0.02
            settings.max_substeps = 8
        else:
            settings.max_substep_delta_time = 0.05
            settings.max_substeps = 15
        
        self.world.apply_settings(settings)
        print(f"✅ CARLA world settings applied:")
        print(f"   Fixed Delta: {fixed_delta_seconds}s")
        print(f"   Max Substep Delta: {settings.max_substep_delta_time}s")
        print(f"   Max Substeps: {settings.max_substeps}")
    
    def update_world_settings(self, fixed_delta_seconds=None, synchronous_mode=None):
        """Dynamically update CARLA world settings"""
        settings = self.world.get_settings()
        
        if synchronous_mode is not None:
            settings.synchronous_mode = synchronous_mode
        
        if fixed_delta_seconds is not None:
            settings.fixed_delta_seconds = fixed_delta_seconds
            # Adjust substep parameters
            if fixed_delta_seconds <= 0.05:
                settings.max_substep_delta_time = 0.01
                settings.max_substeps = 10
            elif fixed_delta_seconds <= 0.1:
                settings.max_substep_delta_time = 0.02
                settings.max_substeps = 8
            else:
                settings.max_substep_delta_time = 0.05
                settings.max_substeps = 15
        
        self.world.apply_settings(settings)
        
    def get_current_settings(self):
        """Get current CARLA world settings"""
        settings = self.world.get_settings()
        return {
            'synchronous_mode': settings.synchronous_mode,
            'fixed_delta_seconds': settings.fixed_delta_seconds,
            'max_substep_delta_time': settings.max_substep_delta_time,
            'max_substeps': settings.max_substeps
        }

    def setup_global_overview(self):
        spectator = self.world.get_spectator()
        
        # 从配置文件获取俯瞰设置
        overview_config = SimulationConfig.get_overview_setting()
        overview_location = carla.Location(*overview_config['location'])
        overview_rotation = carla.Rotation(*overview_config['rotation'])
        
        # 应用俯瞰视角
        spectator.set_transform(carla.Transform(overview_location, overview_rotation))

    def spawn_vehicle(self, blueprint_filter='vehicle.*', transform=None):
        blueprint = random.choice(self.blueprint_library.filter(blueprint_filter))
        if transform is None:
            transform = random.choice(self.world.get_map().get_spawn_points())
        vehicle = self.world.spawn_actor(blueprint, transform)
        return vehicle

    def destroy_all_vehicles(self):
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            actor.destroy()