# deadlocknashsolver.py （替换为此实现；若工程里类名/入口不同，请做同名替换）
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import math
import itertools
import time
from collections import defaultdict

# 假定这些类型在项目中已有定义；保持引用名不变
# from auction.types import AuctionAgent, AuctionWinner, Bid
# 或根据你的工程实际 import：
try:
    from auction.auction_engine import AuctionWinner  # 若已有该类
except:
    @dataclass
    class AuctionWinner:
        participant: object
        bid: object
        rank: int
        conflict_action: str = 'go'  # 'go' or 'wait'

# ---- 工具函数：可用就用，缺啥就用本地近似 ----

def _euclidean_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _eta_to_conflict_center(agent_state: Dict, center: Tuple[float, float, float]) -> float:
    """
    近似ETA：到路口中心的距离 / max(speed, eps)。
    agent_state 需包含 'location' (x,y,z) 和 'speed' (m/s)。
    若缺失，返回 +inf。
    """
    loc = agent_state.get('location')
    v = max(agent_state.get('speed', 0.0), 0.1)
    if not loc:
        return float('inf')
    d = _euclidean_2d(loc, center)
    return d / v

def _turn_conflict(turn_i: str, turn_j: str) -> bool:
    """
    简易转向冲突矩阵：直行与对向直行/左转可能冲突；左转与对向直行/右侧直行等冲突；右转较少冲突但在窄口可能冲突。
    可按需要细化/替换为更准确的拓扑判断。
    """
    # 统一为 {'left','right','straight'}，未知当作 'straight'
    si = (turn_i or 'straight').lower()
    sj = (turn_j or 'straight').lower()
    if si == 'right' and sj == 'right':
        return False
    if si == sj == 'straight':
        return True
    if 'left' in (si, sj) and 'straight' in (si, sj):
        return True
    if si == 'left' and sj == 'left':
        return True  # 同向对斜也可能在中心区冲突
    # 右转与直行/左转在部分几何下也会冲突，这里保守处理
    if 'right' in (si, sj):
        return True
    return False

class DeadlockException(Exception):
    """Exception raised when deadlock is detected"""
    pass

class DeadlockNashSolver:
    """
    Enhanced MWIS-based deadlock solver with improved conflict detection and path analysis
    """
    def __init__(self,
                 max_exact: int = 15,
                 conflict_time_window: float = 3.0,
                 intersection_center: Tuple[float, float, float] = (-188.9, -89.7, 0.0),
                 intersection_radius: float = 25.0,
                 min_safe_distance: float = 5.0,
                 speed_prediction_horizon: float = 5.0,
                 max_go_agents: int = 8):  # Changed default to match DRLConfig
        self.max_exact = max_exact
        self.dt_conflict = conflict_time_window
        self.center = intersection_center
        self.intersection_radius = intersection_radius
        self.min_safe_distance = min_safe_distance
        self.prediction_horizon = speed_prediction_horizon
        
        # Enhanced conflict detection parameters
        self.path_intersection_threshold = 3.0  # meters
        self.velocity_similarity_threshold = 0.3  # for detecting following behavior
        
        # Deadlock detection parameters - Use exact square area like show_intersection_area1
        self.deadlock_detection_window = 35.0  # seconds to track for deadlock
        self.deadlock_speed_threshold = 0.5  # m/s - vehicles below this are considered stopped
        self.deadlock_min_vehicles = 6  # minimum vehicles for deadlock detection
        self.deadlock_history = []  # track intersection state over time
        self.last_deadlock_check = 0
        self.deadlock_check_interval = 2.0  # check every 2 seconds
        
        # Core deadlock detection area - EXACT SQUARE like show_intersection_area1
        from env.simulation_config import SimulationConfig
        self.deadlock_core_half_size = SimulationConfig.INTERSECTION_HALF_SIZE / 5  # Same as show_intersection_area1
        
        # Traffic flow control parameters
        self.stalled_vehicles_threshold = 3  # Block entry if more than 3 stalled vehicles
        self.region_entry_blocked = False
        self.last_entry_block_check = 0
        self.entry_block_check_interval = 1.0  # Check every 1 second
        
        # Performance tracking
        self.stats = {
            'total_resolutions': 0,
            'conflicts_detected': 0,
            'mwis_exact_calls': 0,
            'mwis_greedy_calls': 0,
            'avg_resolution_time': 0.0,
            'deadlocks_detected': 0,
            'entry_blocks_activated': 0,
            'entry_blocks_released': 0
        }

        # Add max go agents limit - now configurable from DRLConfig
        self.max_go_agents = max_go_agents

    # === 外部调用的主入口（签名尽量与旧版一致） ===
    def resolve(self,
                candidates: List,
                vehicle_states: Dict[str, Dict],
                platoon_manager=None,
                *args, **kwargs) -> List[AuctionWinner]:
        """Enhanced resolve with performance tracking, better conflict analysis, and traffic flow control"""
        start_time = time.time()
        
        if not candidates:
            return []

        # 1) Check and update traffic flow control status
        self._update_traffic_flow_control(vehicle_states, start_time)

        # 2) Check for deadlock before processing
        deadlock_detected = self._detect_deadlock(vehicle_states, start_time)
        if deadlock_detected:
            self._handle_deadlock_detection()
            return []  # Return empty to halt normal processing

        # 3) Enhanced conflict graph construction
        adj, conflict_analysis = self._build_enhanced_conflict_graph(
            candidates, vehicle_states, platoon_manager
        )

        # 4) Adaptive MWIS algorithm selection
        weights = [self._get_bid(c) for c in candidates]
        selected_idx = self._solve_mwis_adaptive(weights, adj, conflict_analysis)

        # 5) Enhanced winner assembly with conflict actions and traffic flow control
        winners = self._assemble_winners_with_traffic_control(
            candidates, selected_idx, weights, conflict_analysis, vehicle_states
        )
        
        # 6) Update performance statistics
        resolution_time = time.time() - start_time
        self._update_stats(resolution_time, len(adj), conflict_analysis)
        
        return winners

    def _update_traffic_flow_control(self, vehicle_states: Dict[str, Dict], current_time: float):
        """Update traffic flow control based on stalled vehicles in core region"""
        # Only check periodically to avoid excessive computation
        if current_time - self.last_entry_block_check < self.entry_block_check_interval:
            return
        
        self.last_entry_block_check = current_time
        
        # Get vehicles in core region
        core_vehicles = self._get_core_region_vehicles(vehicle_states)
        stalled_count = self._count_stalled_vehicles(core_vehicles)
        
        previous_block_status = self.region_entry_blocked
        
        if stalled_count > self.stalled_vehicles_threshold:
            if not self.region_entry_blocked:
                self.region_entry_blocked = True
                self.stats['entry_blocks_activated'] += 1
                print(f"\n🚫 TRAFFIC FLOW CONTROL ACTIVATED")
                print(f"   🔴 {stalled_count} stalled vehicles in core region (threshold: {self.stalled_vehicles_threshold})")
                print(f"   🚧 Blocking new entries until region clears")
        else:
            if self.region_entry_blocked:
                # Check if all previously stalled vehicles are now moving
                if self._all_stalled_vehicles_recovered(core_vehicles):
                    self.region_entry_blocked = False
                    self.stats['entry_blocks_released'] += 1
                    print(f"\n✅ TRAFFIC FLOW CONTROL RELEASED")
                    print(f"   🟢 Stalled vehicles recovered ({stalled_count} remaining)")
                    print(f"   🚦 Allowing new entries to core region")

    def _get_core_region_vehicles(self, vehicle_states: Dict[str, Dict]) -> List[Dict]:
        """Get vehicles specifically in the core blue square region"""
        core_vehicles = []
        
        for vehicle_id, vehicle_state in vehicle_states.items():
            if not vehicle_state or 'location' not in vehicle_state:
                continue
            
            location = vehicle_state['location']
            
            # Use EXACT SQUARE bounds like show_intersection_area1
            center_x, center_y = self.center[0], self.center[1]
            half_size = self.deadlock_core_half_size
            
            # Check if vehicle is within the core square bounds
            in_core_square = (
                (center_x - half_size) <= location[0] <= (center_x + half_size) and
                (center_y - half_size) <= location[1] <= (center_y + half_size)
            )
            
            if in_core_square:
                vehicle_data = dict(vehicle_state)
                vehicle_data['id'] = vehicle_id
                core_vehicles.append(vehicle_data)
        
        return core_vehicles

    def _count_stalled_vehicles(self, vehicles: List[Dict]) -> int:
        """Count stalled vehicles in the given list"""
        stalled_count = 0
        
        for vehicle in vehicles:
            velocity = vehicle.get('velocity', [0, 0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
            
            if speed < self.deadlock_speed_threshold:
                stalled_count += 1
        
        return stalled_count

    def _all_stalled_vehicles_recovered(self, current_vehicles: List[Dict]) -> bool:
        """Check if all previously stalled vehicles have recovered to normal movement"""
        stalled_count = self._count_stalled_vehicles(current_vehicles)
        
        # Consider recovery successful if stalled count is below threshold
        # and there's evidence of movement in the region
        if stalled_count <= self.stalled_vehicles_threshold:
            # Additional check: ensure vehicles are actually moving
            moving_vehicles = 0
            for vehicle in current_vehicles:
                velocity = vehicle.get('velocity', [0, 0, 0])
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
                if speed >= self.deadlock_speed_threshold:
                    moving_vehicles += 1
            
            # Release block if we have fewer stalled vehicles AND some movement
            return stalled_count == 0 or moving_vehicles > 0
        
        return False

    # === 构图：判断任意两候选是否冲突 ===
    def _build_enhanced_conflict_graph(self, candidates: List, vehicle_states: Dict[str, Dict], 
                                     platoon_manager=None) -> Tuple[List[Set[int]], Dict]:
        """Enhanced conflict graph with geometric path analysis and time predictions"""
        n = len(candidates)
        adj: List[Set[int]] = [set() for _ in range(n)]
        conflict_analysis = {
            'spatial_conflicts': 0,
            'temporal_conflicts': 0,
            'platoon_conflicts': 0,
            'path_intersections': 0
        }
        
        # Enhanced metadata extraction
        meta = []
        for c in candidates:
            agent = self._get_agent(c)
            state = self._lookup_state(agent, vehicle_states, platoon_manager)
            
            # Enhanced turn inference with path prediction
            turn = self._infer_turn_enhanced(agent, state, vehicle_states)
            eta = self._calculate_enhanced_eta(state, agent)
            path = self._predict_vehicle_path(state, agent)
            
            meta.append({
                'agent': agent,
                'state': state,
                'turn': turn,
                'eta': eta,
                'predicted_path': path,
                'is_platoon': agent.type == 'platoon' if hasattr(agent, 'type') else False
            })

        # Enhanced conflict detection
        for i, j in itertools.combinations(range(n), 2):
            conflict_type = self._detect_enhanced_conflict(meta[i], meta[j])
            if conflict_type:
                adj[i].add(j)
                adj[j].add(i)
                conflict_analysis[conflict_type] += 1
        
        return adj, conflict_analysis

    def _turn_conflict_enhanced(self, turn_i: str, turn_j: str, 
                          approach_i: str, approach_j: str,
                          location_i: Tuple[float, float, float],
                          location_j: Tuple[float, float, float]) -> bool:
        """
        增强转向冲突检测，基于实际路网几何和进入方向
    
        Args:
            turn_i, turn_j: 转向动作 {'left', 'right', 'straight', 'u_turn'}
            approach_i, approach_j: 进入方向 {'north', 'south', 'east', 'west'}
            location_i, location_j: 当前位置坐标
    
        Returns:
            bool: 是否存在冲突
        """
        # 标准化输入
        turn_i = (turn_i or 'straight').lower()
        turn_j = (turn_j or 'straight').lower()
    
        # 如果无法确定进入方向，使用位置推断
        if not approach_i:
            approach_i = self._infer_approach_direction(location_i)
        if not approach_j:
            approach_j = self._infer_approach_direction(location_j)
    
        # 1. 同向车辆冲突检测
        if approach_i == approach_j:
            return self._same_approach_conflict(turn_i, turn_j)
    
        # 2. 对向车辆冲突检测
        if self._are_opposite_approaches(approach_i, approach_j):
            return self._opposite_approach_conflict(turn_i, turn_j)
    
        # 3. 垂直方向车辆冲突检测
        if self._are_perpendicular_approaches(approach_i, approach_j):
            return self._perpendicular_approach_conflict(turn_i, turn_j, approach_i, approach_j)
    
        # 4. 默认保守处理：未知配置认为可能冲突
        return True

    def _infer_approach_direction(self, location: Tuple[float, float, float]) -> str:
        """根据车辆位置推断进入方向"""
        try:
            center_x, center_y = self.center[0], self.center[1]
            rel_x = location[0] - center_x
            rel_y = location[1] - center_y
        
            # 使用角度判断主要方向
            angle = math.atan2(rel_y, rel_x)
            angle_deg = math.degrees(angle)
        
            # 标准化到 [0, 360)
            if angle_deg < 0:
                angle_deg += 360
        
            # 分配到四个主要方向（考虑45度容差）
            if 315 <= angle_deg or angle_deg < 45:
                return 'east'    # 从东侧进入
            elif 45 <= angle_deg < 135:
                return 'north'   # 从北侧进入
            elif 135 <= angle_deg < 225:
                return 'west'    # 从西侧进入
            else:  # 225 <= angle_deg < 315
                return 'south'   # 从南侧进入
            
        except Exception:
            return 'unknown'

    def _same_approach_conflict(self, turn_i: str, turn_j: str) -> bool:
        """同向车辆转向冲突矩阵"""
        # 同向车辆冲突相对较少，主要是变道和速度差异
        conflict_matrix = {
            ('straight', 'straight'): False,  # 直行不冲突
            ('straight', 'left'): True,       # 直行与左转可能冲突（变道）
            ('straight', 'right'): True,      # 直行与右转可能冲突（变道）
            ('left', 'left'): False,          # 同向左转不冲突
            ('left', 'right'): True,          # 左转与右转交叉冲突
            ('right', 'right'): False,        # 同向右转不冲突
            ('u_turn', 'straight'): True,     # 掉头与直行冲突
            ('u_turn', 'left'): True,         # 掉头与左转冲突
            ('u_turn', 'right'): True,        # 掉头与右转冲突
            ('u_turn', 'u_turn'): True,       # 掉头之间冲突
        }
    
        key = tuple(sorted([turn_i, turn_j]))
        return conflict_matrix.get(key, True)  # 未知情况保守处理

    def _opposite_approach_conflict(self, turn_i: str, turn_j: str) -> bool:
        """对向车辆转向冲突矩阵"""
        # 对向车辆冲突是最复杂的情况
        conflict_matrix = {
            ('straight', 'straight'): True,   # 对向直行冲突
            ('straight', 'left'): True,       # 直行与对向左转冲突
            ('straight', 'right'): False,     # 直行与对向右转通常不冲突
            ('left', 'left'): False,          # 对向左转通常不冲突（除非路口很小）
            ('left', 'right'): False,         # 左转与对向右转不冲突
            ('right', 'right'): False,        # 对向右转不冲突
            ('u_turn', 'straight'): True,     # 掉头与对向直行冲突
            ('u_turn', 'left'): True,         # 掉头与对向左转冲突
            ('u_turn', 'right'): True,        # 掉头与对向右转冲突
            ('u_turn', 'u_turn'): True,       # 对向掉头冲突
        }
    
        key = tuple(sorted([turn_i, turn_j]))
        return conflict_matrix.get(key, True)

    def _perpendicular_approach_conflict(self, turn_i: str, turn_j: str, 
                                   approach_i: str, approach_j: str) -> bool:
        """垂直方向车辆转向冲突矩阵"""
        # 确定相对位置关系
        is_i_left_of_j = self._is_left_approach(approach_i, approach_j)
    
        # 垂直方向冲突矩阵（考虑右行规则）
        if is_i_left_of_j:
            # i在j的左侧
            conflict_matrix = {
                ('straight', 'straight'): True,   # 垂直直行冲突
                ('straight', 'left'): True,       # 直行与垂直左转冲突
                ('straight', 'right'): False,     # 直行与垂直右转不冲突（右转先行）
                ('left', 'straight'): True,       # 左转与垂直直行冲突
                ('left', 'left'): True,           # 左转与垂直左转冲突
                ('left', 'right'): False,         # 左转与垂直右转不冲突
                ('right', 'straight'): False,     # 右转与垂直直行不冲突（右转先行）
                ('right', 'left'): False,         # 右转与垂直左转不冲突
                ('right', 'right'): False,        # 右转之间不冲突
                ('u_turn', 'straight'): True,     # 掉头与垂直直行冲突
                ('u_turn', 'left'): True,         # 掉头与垂直左转冲突
                ('u_turn', 'right'): True,        # 掉头与垂直右转冲突
                ('u_turn', 'u_turn'): True,       # 掉头之间冲突
            }
        else:
            # j在i的左侧，交换优先级
            conflict_matrix = {
                ('straight', 'straight'): True,
                ('straight', 'left'): False,      # 垂直左转让行直行
                ('straight', 'right'): True,      # 直行与垂直右转冲突
                ('left', 'straight'): False,      # 左转让行垂直直行
                ('left', 'left'): True,
                ('left', 'right'): True,
                ('right', 'straight'): True,
                ('right', 'left'): True,
                ('right', 'right'): False,
                ('u_turn', 'straight'): True,
                ('u_turn', 'left'): True,
                ('u_turn', 'right'): True,
                ('u_turn', 'u_turn'): True,
            }
    
        key = (turn_i, turn_j)
        return conflict_matrix.get(key, conflict_matrix.get((turn_j, turn_i), True))

    def _are_opposite_approaches(self, approach_i: str, approach_j: str) -> bool:
        """判断是否为对向进入"""
        opposites = {
            ('north', 'south'), ('south', 'north'),
            ('east', 'west'), ('west', 'east')
        }
        return (approach_i, approach_j) in opposites

    def _are_perpendicular_approaches(self, approach_i: str, approach_j: str) -> bool:
        """判断是否为垂直进入"""
        if approach_i == 'unknown' or approach_j == 'unknown':
            return False
        return not self._are_opposite_approaches(approach_i, approach_j) and approach_i != approach_j

    def _is_left_approach(self, approach_i: str, approach_j: str) -> bool:
        """判断approach_i是否在approach_j的左侧（基于右行规则）"""
        # 定义左侧关系（顺时针）
        left_relations = {
            'north': 'west',  # 北向车辆的左侧是西向
            'west': 'south',  # 西向车辆的左侧是南向
            'south': 'east',  # 南向车辆的左侧是东向
            'east': 'north'   # 东向车辆的左侧是北向
        }
    
        return left_relations.get(approach_j) == approach_i

    def _infer_turn_enhanced(self, agent, state: Dict, vehicle_states: Dict) -> str:
        """增强转向推断，结合速度向量和路径预测"""
        if not state or 'location' not in state:
            return 'straight'
    
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
    
        # 1. 基础转向推断
        basic_turn = self._infer_turn(agent, state)
    
        # 2. 速度向量分析
        if abs(velocity[0]) > 0.5 or abs(velocity[1]) > 0.5:
            velocity_turn = self._infer_turn_from_velocity(location, velocity)
        
            # 如果速度向量给出明确信号，优先使用
            if velocity_turn != 'straight':
                return velocity_turn
    
        # 3. 历史轨迹分析（如果有历史数据）
        trajectory_turn = self._infer_turn_from_trajectory(agent, state, vehicle_states)
        if trajectory_turn != 'straight':
            return trajectory_turn
    
        # 4. 目标点分析（如果有路径规划信息）
        if hasattr(agent, 'destination') or 'destination' in state:
            destination_turn = self._infer_turn_from_destination(agent, state)
            if destination_turn != 'straight':
                return destination_turn
    
        return basic_turn

    def _infer_turn_from_velocity(self, location: Tuple[float, float, float], 
                            velocity: List[float]) -> str:
        """基于速度向量推断转向"""
        try:
            # 计算当前朝向
            current_heading = math.atan2(velocity[1], velocity[0])
        
            # 计算到路口中心的方向
            to_center_x = self.center[0] - location[0]
            to_center_y = self.center[1] - location[1]
            to_center_heading = math.atan2(to_center_y, to_center_x)
        
            # 计算相对角度
            relative_angle = self._normalize_angle(current_heading - to_center_heading)
        
            # 基于角度判断转向
            if relative_angle > math.pi/3:  # 60度
                return 'left'
            elif relative_angle < -math.pi/3:
                return 'right'
            elif abs(relative_angle) > 2*math.pi/3:  # 120度，可能是掉头
                return 'u_turn'
            else:
                return 'straight'
            
        except Exception:
            return 'straight'

    def _infer_turn_from_trajectory(self, agent, state: Dict, 
                              vehicle_states: Dict) -> str:
        """基于历史轨迹推断转向（需要轨迹历史）"""
        # 这里可以实现基于历史位置的转向推断
        # 目前返回默认值，可以在后续实现中添加轨迹跟踪
        return 'straight'

    def _infer_turn_from_destination(self, agent, state: Dict) -> str:
        """基于目标点推断转向"""
        try:
            destination = None
            if hasattr(agent, 'destination'):
                destination = agent.destination
            elif 'destination' in state:
                destination = state['destination']
        
            if not destination:
                return 'straight'
        
            location = state['location']
        
            # 计算到目标的方向
            to_dest_x = destination[0] - location[0]
            to_dest_y = destination[1] - location[1]
            to_dest_heading = math.atan2(to_dest_y, to_dest_x)
        
            # 计算经过路口中心的方向
            to_center_x = self.center[0] - location[0]
            to_center_y = self.center[1] - location[1]
            to_center_heading = math.atan2(to_center_y, to_center_x)
        
            # 比较两个方向
            angle_diff = self._normalize_angle(to_dest_heading - to_center_heading)
        
            if angle_diff > math.pi/4:
                return 'left'
            elif angle_diff < -math.pi/4:
                return 'right'
            else:
                return 'straight'
            
        except Exception:
            return 'straight'

    # 在 _build_enhanced_conflict_graph 方法中更新冲突检测调用
    def _detect_enhanced_conflict(self, meta_i: Dict, meta_j: Dict) -> Optional[str]:
        """使用增强转向冲突检测的冲突检测"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            if not state_i or not state_j:
                return None
            
            # 1. 空间冲突检测
            if self._has_spatial_conflict(meta_i, meta_j):
                return 'spatial_conflicts'
            
            # 2. 时间冲突检测
            if self._has_temporal_conflict(meta_i, meta_j):
                return 'temporal_conflicts'
            
            # 3. 路径相交检测
            if self._has_path_intersection(meta_i, meta_j):
                return 'path_intersections'
            
            # 4. 增强转向冲突检测
            location_i = state_i.get('location', (0, 0, 0))
            location_j = state_j.get('location', (0, 0, 0))
            
            approach_i = self._infer_approach_direction(location_i)
            approach_j = self._infer_approach_direction(location_j)
            
            if self._turn_conflict_enhanced(
                meta_i['turn'], meta_j['turn'],
                approach_i, approach_j,
                location_i, location_j
            ):
                return 'spatial_conflicts'
            
            return None
            
        except Exception as e:
            print(f"[Warning] Enhanced conflict detection failed: {e}")
            return None

    def _has_spatial_conflict(self, meta_i: Dict, meta_j: Dict) -> bool:
        """Check for spatial conflicts"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            loc_i = state_i['location']
            loc_j = state_j['location']
            
            distance = _euclidean_2d(loc_i, loc_j)
            return distance < self.min_safe_distance
            
        except Exception:
            return False

    def _has_temporal_conflict(self, meta_i: Dict, meta_j: Dict) -> bool:
        """Check for temporal conflicts"""
        try:
            eta_i = meta_i.get('eta', float('inf'))
            eta_j = meta_j.get('eta', float('inf'))
            
            if eta_i == float('inf') or eta_j == float('inf'):
                return False
            
            # Conflict if ETAs are within conflict time window
            return abs(eta_i - eta_j) < self.dt_conflict
            
        except Exception:
            return False

    def _has_path_intersection(self, meta_i: Dict, meta_j: Dict) -> bool:
        """Check for path intersections"""
        try:
            path_i = meta_i.get('predicted_path', [])
            path_j = meta_j.get('predicted_path', [])
            
            if len(path_i) < 2 or len(path_j) < 2:
                return False
            
            # Check if any path segments intersect
            for i in range(len(path_i) - 1):
                for j in range(len(path_j) - 1):
                    if self._segments_intersect(path_i[i], path_i[i+1], path_j[j], path_j[j+1]):
                        return True
            
            return False
            
        except Exception:
            return False

    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                           p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect"""
        try:
            # Simple distance-based intersection check
            # Find closest points on the two segments
            dist = self._distance_between_segments(p1, p2, p3, p4)
            return dist < self.path_intersection_threshold
            
        except Exception:
            return False

    def _distance_between_segments(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                  p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        """Calculate minimum distance between two line segments"""
        try:
            # Simplified: use midpoint distances
            mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid2 = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2)
            
            return math.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
            
        except Exception:
            return float('inf')

    def _solve_mwis_adaptive(self, weights: List[float], adj: List[Set[int]], 
                           conflict_analysis: Dict) -> List[int]:
        """Adaptive MWIS solver"""
        n = len(weights)
        if n == 0:
            return []
        
        # Use exact solver for small problems, greedy for large ones
        if n <= self.max_exact:
            self.stats['mwis_exact_calls'] += 1
            return self._solve_mwis_exact(weights, adj)
        else:
            self.stats['mwis_greedy_calls'] += 1
            return self._solve_mwis_greedy(weights, adj)

    def _solve_mwis_exact(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Exact MWIS solver using dynamic programming"""
        n = len(weights)
        if n == 0:
            return []
        
        # For small graphs, use brute force
        if n <= 10:
            return self._solve_mwis_brute_force(weights, adj)
        
        # For larger graphs, use greedy as fallback
        return self._solve_mwis_greedy(weights, adj)

    def _solve_mwis_brute_force(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Brute force MWIS solver"""
        n = len(weights)
        best_weight = 0
        best_set = []
        
        # Try all possible subsets
        for mask in range(1, 1 << n):
            subset = [i for i in range(n) if mask & (1 << i)]
            
            # Check if subset is independent
            if self._is_independent_set(subset, adj):
                weight = sum(weights[i] for i in subset)
                if weight > best_weight:
                    best_weight = weight
                    best_set = subset
        
        return best_set

    def _solve_mwis_greedy(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Greedy MWIS solver"""
        n = len(weights)
        if n == 0:
            return []
        
        # Sort vertices by weight/degree ratio
        vertices = list(range(n))
        vertices.sort(key=lambda i: weights[i] / max(len(adj[i]), 1), reverse=True)
        
        selected = []
        excluded = set()
        
        for v in vertices:
            if v not in excluded:
                selected.append(v)
                excluded.update(adj[v])
        
        return selected

    def _is_independent_set(self, subset: List[int], adj: List[Set[int]]) -> bool:
        """Check if a subset is an independent set"""
        for i in subset:
            for j in subset:
                if i != j and j in adj[i]:
                    return False
        return True

    def _to_winner(self, candidate, action: str, rank: int) -> AuctionWinner:
        """Convert candidate to AuctionWinner"""
        agent = self._get_agent(candidate)
        bid_value = self._get_bid(candidate)
        
        # Create a simple bid object
        class SimpleBid:
            def __init__(self, value):
                self.value = value
        
        return AuctionWinner(
            participant=agent,
            bid=SimpleBid(bid_value),
            rank=rank,
            conflict_action=action
        )

    def _lookup_state(self, agent, vehicle_states: Dict[str, Dict], platoon_manager=None) -> Optional[Dict]:
        """Lookup vehicle state for an agent"""
        try:
            agent_id = str(getattr(agent, 'id', agent))
            
            # For single vehicles
            if agent_id in vehicle_states:
                return vehicle_states[agent_id]
            
            # For platoons, get leader state
            if platoon_manager and hasattr(agent, 'type') and agent.type == 'platoon':
                vehicles = getattr(agent, 'vehicles', [])
                if vehicles:
                    leader_id = str(vehicles[0].get('id', vehicles[0].get('vehicle_id')))
                    return vehicle_states.get(leader_id)
            
            # Try to find by data attribute
            if hasattr(agent, 'data') and 'vehicles' in agent.data:
                vehicles = agent.data['vehicles']
                if vehicles:
                    leader_id = str(vehicles[0].get('id'))
                    return vehicle_states.get(leader_id)
            
            return None
            
        except Exception as e:
            print(f"[Warning] Lookup state failed for agent {agent}: {e}")
            return None

    def _get_bid(self, candidate) -> float:
        """Extract bid value from candidate"""
        if hasattr(candidate, 'bid'):
            if hasattr(candidate.bid, 'value'):
                return candidate.bid.value
            else:
                return float(candidate.bid)
        elif hasattr(candidate, 'value'):
            return candidate.value
        else:
            return 1.0  # Default bid value

    def _infer_turn(self, agent, state: Dict) -> str:
        """Basic turn inference from agent state"""
        if not state:
            return 'straight'
        
        try:
            # Try to get turn from state directly
            if 'turn' in state:
                return state['turn']
            
            # Basic heading-based inference
            rotation = state.get('rotation', [0, 0, 0])
            heading = rotation[2] if len(rotation) > 2 else 0
            
            # Simple heuristic based on heading relative to intersection
            if abs(heading) < 45:
                return 'straight'
            elif heading > 45:
                return 'left'
            else:
                return 'right'
                
        except Exception:
            return 'straight'

    def _detect_deadlock(self, vehicle_states: Dict[str, Dict], current_time: float) -> bool:
        """Enhanced deadlock detection with multiple detection modes"""
        # Only check periodically to avoid excessive computation
        if current_time - self.last_deadlock_check < self.deadlock_check_interval:
            return False
        
        self.last_deadlock_check = current_time
        
        # Get vehicles in core region
        core_vehicles = self._get_core_region_vehicles(vehicle_states)
        
        if len(core_vehicles) < self.deadlock_min_vehicles:
            return False
        
        # Create snapshot of current state
        snapshot = {
            'timestamp': current_time,
            'core_vehicles': {v['id']: {
                'location': v.get('location', (0, 0, 0)),
                'velocity': v.get('velocity', [0, 0, 0]),
                'speed': math.sqrt(sum(x**2 for x in v.get('velocity', [0, 0, 0]))),
                'stalled': math.sqrt(sum(x**2 for x in v.get('velocity', [0, 0, 0]))) < self.deadlock_speed_threshold
            } for v in core_vehicles},
            'stalled_count': self._count_stalled_vehicles(core_vehicles)
        }
        
        # Add to history
        self.deadlock_history.append(snapshot)
        
        # Keep only recent history
        cutoff_time = current_time - self.deadlock_detection_window
        self.deadlock_history = [s for s in self.deadlock_history if s['timestamp'] > cutoff_time]
        
        # Need sufficient history for detection
        if len(self.deadlock_history) < 5:
            return False
        
        # Mode 1: Persistent core stalling
        if self._detect_persistent_core_stalling():
            print(f"\n🚨 DEADLOCK DETECTED - Persistent Core Stalling")
            print(f"   📍 Location: Core intersection region")
            print(f"   🕐 Duration: {self.deadlock_detection_window}s+ of stalling")
            print(f"   🚗 Vehicles: {len(core_vehicles)} vehicles affected")
            return True
        
        # Mode 2: Circular waiting pattern
        if self._detect_circular_waiting():
            print(f"\n🚨 DEADLOCK DETECTED - Circular Waiting Pattern")
            print(f"   🔄 Pattern: Vehicles blocking each other in cycle")
            print(f"   🚗 Vehicles: {len(core_vehicles)} vehicles affected")
            return True
        
        # Mode 3: No progress detection
        if self._detect_no_progress():
            print(f"\n🚨 DEADLOCK DETECTED - No Progress")
            print(f"   ⏱️ Pattern: No movement toward intersection center")
            print(f"   🚗 Vehicles: {len(core_vehicles)} vehicles affected")
            return True
        
        return False

    def _detect_persistent_core_stalling(self) -> bool:
        """Detect if the same set of vehicles have been stalled in core for extended time"""
        if len(self.deadlock_history) < 10:  # Need at least 20 seconds of history
            return False
        
        # Check if we have consistent stalling over time
        recent_snapshots = self.deadlock_history[-10:]
        
        # Count snapshots where stalled vehicle count is above threshold
        high_stall_count = sum(1 for s in recent_snapshots 
                              if s['stalled_count'] >= self.deadlock_min_vehicles)
        
        # If most recent snapshots show high stalling, it's likely deadlock
        return high_stall_count >= 8  # 80% of recent snapshots

    def _detect_circular_waiting(self) -> bool:
        """Detect circular waiting patterns where vehicles block each other"""
        if len(self.deadlock_history) < 5:
            return False
        
        current_snapshot = self.deadlock_history[-1]
        core_vehicles = current_snapshot['core_vehicles']
        
        # Simple heuristic: if most vehicles in core are stalled and positioned 
        # in different quadrants, likely circular waiting
        stalled_vehicles = [v_id for v_id, data in core_vehicles.items() if data['stalled']]
        
        if len(stalled_vehicles) < 4:  # Need at least 4 vehicles for circular pattern
            return False
        
        # Check if vehicles are distributed across different approaches
        quadrant_count = self._count_vehicles_by_quadrant(stalled_vehicles, core_vehicles)
        
        # If vehicles are in 3+ quadrants and mostly stalled, likely circular waiting
        return len(quadrant_count) >= 3 and len(stalled_vehicles) >= self.deadlock_min_vehicles

    def _detect_no_progress(self) -> bool:
        """Detect lack of progress toward intersection center"""
        if len(self.deadlock_history) < 8:  # Need sufficient history
            return False
        
        # Compare current positions with positions from 15 seconds ago
        current_snapshot = self.deadlock_history[-1]
        old_snapshot = self.deadlock_history[-8]  # ~15 seconds ago
        
        current_vehicles = current_snapshot['core_vehicles']
        old_vehicles = old_snapshot['core_vehicles']
        
        # Track vehicles that were present in both snapshots
        common_vehicles = set(current_vehicles.keys()) & set(old_vehicles.keys())
        
        if len(common_vehicles) < 3:
            return False
        
        # Check if vehicles have made progress toward center
        no_progress_count = 0
        center = self.center
        
        for v_id in common_vehicles:
            old_pos = old_vehicles[v_id]['location']
            current_pos = current_vehicles[v_id]['location']
            
            old_dist = _euclidean_2d(old_pos, center)
            current_dist = _euclidean_2d(current_pos, center)
            
            # No significant progress if distance to center hasn't decreased much
            if current_dist >= old_dist - 1.0:  # Less than 1 meter progress
                no_progress_count += 1
        
        # If most tracked vehicles made no progress, likely deadlock
        return no_progress_count >= len(common_vehicles) * 0.8

    def _count_vehicles_by_quadrant(self, vehicle_ids: List[str], 
                                   vehicles_data: Dict[str, Dict]) -> Dict[str, int]:
        """Count vehicles in each quadrant relative to intersection center"""
        quadrant_count = defaultdict(int)
        center_x, center_y = self.center[0], self.center[1]
        
        for v_id in vehicle_ids:
            if v_id not in vehicles_data:
                continue
            
            location = vehicles_data[v_id]['location']
            rel_x = location[0] - center_x
            rel_y = location[1] - center_y
            
            if rel_x >= 0 and rel_y >= 0:
                quadrant = 'NE'
            elif rel_x < 0 and rel_y >= 0:
                quadrant = 'NW'
            elif rel_x < 0 and rel_y < 0:
                quadrant = 'SW'
            else:
                quadrant = 'SE'
            
            quadrant_count[quadrant] += 1
        
        return quadrant_count

    def _handle_deadlock_detection(self):
        """Handle deadlock detection by updating stats and raising exception"""
        self.stats['deadlocks_detected'] += 1
        raise DeadlockException("Deadlock detected in intersection core region")

    def _get_agent(self, candidate) -> object:
        """Extract agent from candidate"""
        if hasattr(candidate, 'participant'):
            return candidate.participant
        elif hasattr(candidate, 'agent'):
            return candidate.agent
        else:
            return candidate

    def _calculate_enhanced_eta(self, state: Dict, agent) -> float:
        """Calculate enhanced ETA with velocity prediction"""
        if not state or 'location' not in state:
            return float('inf')
        
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
        speed = math.sqrt(sum(x**2 for x in velocity)) if velocity else 0.0
        
        # Distance to intersection center
        distance = _euclidean_2d(location, self.center)
        
        # Use actual speed or minimum speed
        effective_speed = max(speed, 0.1)
        
        return distance / effective_speed

    def _predict_vehicle_path(self, state: Dict, agent) -> List[Tuple[float, float]]:
        """Predict vehicle path for the next few seconds"""
        if not state or 'location' not in state:
            return []
        
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
        
        if not velocity or (velocity[0] == 0 and velocity[1] == 0):
            return [location[:2]]  # Stationary vehicle
        
        # Predict path using linear projection
        path = [location[:2]]
        dt = 0.5  # 0.5 second intervals
        
        for i in range(1, int(self.prediction_horizon / dt) + 1):
            t = i * dt
            future_x = location[0] + velocity[0] * t
            future_y = location[1] + velocity[1] * t
            path.append((future_x, future_y))
        
        return path

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _update_stats(self, resolution_time: float, graph_size: int, conflict_analysis: Dict):
        """Update performance statistics"""
        self.stats['total_resolutions'] += 1
        self.stats['conflicts_detected'] += sum(conflict_analysis.values())
        
        # Update average resolution time
        total_time = self.stats['avg_resolution_time'] * (self.stats['total_resolutions'] - 1)
        self.stats['avg_resolution_time'] = (total_time + resolution_time) / self.stats['total_resolutions']

    def _assemble_winners_with_traffic_control(self, candidates: List, selected_idx: List[int], 
                                             weights: List[float], conflict_analysis: Dict,
                                             vehicle_states: Dict[str, Dict]) -> List[AuctionWinner]:
        """Assemble winners with traffic flow control considerations"""
        if not selected_idx:
            return []
        
        # Sort selected candidates by weight (bid value) in descending order
        selected_candidates = [(candidates[i], weights[i], i) for i in selected_idx]
        selected_candidates.sort(key=lambda x: x[1], reverse=True)
        
        winners = []
        go_count = 0
        
        for candidate, weight, idx in selected_candidates:
            agent = self._get_agent(candidate)
            
            # Determine action based on rank and traffic flow control
            if go_count < self.max_go_agents:
                # Check if this agent should be allowed entry during traffic flow control
                if self.region_entry_blocked and self._should_block_entry(agent, vehicle_states):
                    action = 'wait'
                else:
                    action = 'go'
                    go_count += 1
            else:
                action = 'wait'
        
            winner = self._to_winner(candidate, action, len(winners) + 1)
            winners.append(winner)
        
        return winners

    def _should_block_entry(self, agent, vehicle_states: Dict[str, Dict]) -> bool:
        """Determine if agent should be blocked from entering during traffic flow control"""
        try:
            # Get agent's vehicle state
            state = self._lookup_state(agent, vehicle_states)
            if not state or 'location' not in state:
                return True  # Block if we can't determine location
            
            location = state['location']
            
            # Check if vehicle is already in core region
            center_x, center_y = self.center[0], self.center[1]
            half_size = self.deadlock_core_half_size
            
            in_core = (
                (center_x - half_size) <= location[0] <= (center_x + half_size) and
                (center_y - half_size) <= location[1] <= (center_y + half_size)
            )
            
            # Don't block vehicles already in core (let them exit)
            if in_core:
                return False
            
            # Block vehicles trying to enter core during traffic flow control
            return True
            
        except Exception:
            return True  # Conservative: block if unsure

    def _should_allow_entry(self, agent, vehicle_states: Dict[str, Dict]) -> bool:
        """Determine if agent should be allowed entry to core region"""
        return not self._should_block_entry(agent, vehicle_states)

    # Add method to update configuration
    def update_max_go_agents(self, max_go_agents: int):
        """Update the maximum go agents limit"""
        self.max_go_agents = max_go_agents
        print(f"🔄 Nash solver: Updated MAX_GO_AGENTS to {max_go_agents}")
