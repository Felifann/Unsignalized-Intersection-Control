class SimulationConfig:
    # ===== 地图设置 =====
    MAP_NAME = 'Town05'
    
    # ===== CARLA连接设置 =====
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    # ===== 仿真设置 =====
    SYNCHRONOUS_MODE = True
    FIXED_DELTA_SECONDS = 0.2
    
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
    # 改为正方形检测区域 - 边长80米（半边长40米）
    INTERSECTION_HALF_SIZE = 40.0  # 正方形半边长（米）
    
    # 新增：明确标识这是无信号灯路口
    INTERSECTION_TYPE = 'unsignalized'  # 'signalized' 或 'unsignalized'
    
    # 新增：控制策略仅应用于目标路口
    CONTROL_TARGET_INTERSECTION_ONLY = True
    
    # ===== 状态提取设置 =====
    ACTOR_CACHE_INTERVAL = 1  # 改为1，每帧都更新
    
    # ===== 打印设置 =====
    PRINT_INTERVAL = 30
    
    # ===== 路口容量控制设置 =====
    # MAX_CONCURRENT_AGENTS = 4      # 最多同时通过的agent数
    # INTERSECTION_CAPACITY_ENABLED = True  # 是否启用容量限制
    
    @classmethod
    def get_overview_setting(cls):
        """获取当前地图的俯瞰视角设置"""
        return cls.OVERVIEW_SETTINGS.get(cls.MAP_NAME, 
                                        {'location': (100, 100, 150), 'rotation': (-90, 0, 0)})
    
    @classmethod
    def is_in_intersection_area(cls, location):
        """检查位置是否在正方形交叉口区域内"""
        center = cls.TARGET_INTERSECTION_CENTER
        half_size = cls.INTERSECTION_HALF_SIZE
        
        # 检查x和y坐标是否都在正方形范围内
        if hasattr(location, 'x'):
            # CARLA Location对象
            x, y = location.x, location.y
        else:
            # 元组或列表
            x, y = location[0], location[1]
        
        return (abs(x - center[0]) <= half_size and 
                abs(y - center[1]) <= half_size)
    
    @classmethod
    def distance_to_intersection_center(cls, location):
        """计算到交叉口中心的距离（保持兼容性）"""
        center = cls.TARGET_INTERSECTION_CENTER
        if hasattr(location, 'x'):
            # CARLA Location对象
            dx = location.x - center[0]
            dy = location.y - center[1]
        else:
            # 元组或列表
            dx = location[0] - center[0]
            dy = location[1] - center[1]
        
        return (dx * dx + dy * dy) ** 0.5
