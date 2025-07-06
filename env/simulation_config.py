import carla
class SimulationConfig:
    # ===== 地图设置 =====
    MAP_NAME = 'Town05'
    
    # ===== CARLA连接设置 =====
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    # ===== 仿真设置 =====
    SYNCHRONOUS_MODE = True
    FIXED_DELTA_SECONDS = 0.01
    
    # ===== 交通生成设置 =====
    MAX_VEHICLES = 500
    
    # ===== 俯瞰视角设置（根据不同地图可调整） =====
    OVERVIEW_SETTINGS = {
        'Town01': {'location': (100, 100, 150), 'rotation': (-90, 0, 0)},
        'Town02': {'location': (100, 100, 150), 'rotation': (-90, 0, 0)},
        'Town03': {'location': (-81.91795076642718, -138.1560788835798, 100.02285810453552), 'rotation': (-90, 0, 0)},
        'Town04': {'location': (0, 0, 200), 'rotation': (-90, 0, 0)},
        'Town05': {'location': (-188.9327573776245, -89.66813325881958, 75.02772521972656), 'rotation': (-90, 0, 0)},
        # -188.9327573776245, -89.66813325881958, 100.02772521972656
        # -189.42076110839844, 89.12556982040405, 100.02772521972656
    }
    
    # ===== 目标交叉口设置 =====
    # Town05 的主十字路口中心坐标 (确认为无信号灯路口)
    TARGET_INTERSECTION_CENTER = (-188.9, -89.7, 0.0)
    # 监控半径（米）
    INTERSECTION_RADIUS = 40  # 增加到40米，确保能捕获接近的车辆
    
    # 新增：明确标识这是无信号灯路口
    INTERSECTION_TYPE = 'unsignalized'  # 'signalized' 或 'unsignalized'
    
    # 新增：控制策略仅应用于目标路口
    CONTROL_TARGET_INTERSECTION_ONLY = True
    
    # ===== 状态提取设置 =====
    ACTOR_CACHE_INTERVAL = 1  # 改为1，每帧都更新
    
    # ===== 打印设置 =====
    PRINT_INTERVAL = 30
    
    @classmethod
    def get_overview_setting(cls):
        """获取当前地图的俯瞰视角设置"""
        return cls.OVERVIEW_SETTINGS.get(cls.MAP_NAME, 
                                        {'location': (100, 100, 150), 'rotation': (-90, 0, 0)})
