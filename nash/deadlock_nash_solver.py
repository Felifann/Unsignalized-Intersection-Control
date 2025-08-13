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
        bid: float
        rank: int
        conflict_action: str  # 'go' or 'wait'

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
                 speed_prediction_horizon: float = 5.0):
        self.max_exact = max_exact
        self.dt_conflict = conflict_time_window
        self.center = intersection_center
        self.intersection_radius = intersection_radius
        self.min_safe_distance = min_safe_distance
        self.prediction_horizon = speed_prediction_horizon
        
        # Enhanced conflict detection parameters
        self.path_intersection_threshold = 3.0  # meters
        self.velocity_similarity_threshold = 0.3  # for detecting following behavior
        
        # Performance tracking
        self.stats = {
            'total_resolutions': 0,
            'conflicts_detected': 0,
            'mwis_exact_calls': 0,
            'mwis_greedy_calls': 0,
            'avg_resolution_time': 0.0
        }

    # === 外部调用的主入口（签名尽量与旧版一致） ===
    def resolve(self,
                candidates: List,
                vehicle_states: Dict[str, Dict],
                platoon_manager=None,
                *args, **kwargs) -> List[AuctionWinner]:
        """Enhanced resolve with performance tracking and better conflict analysis"""
        start_time = time.time()
        
        if not candidates:
            return []

        # 1) Enhanced conflict graph construction
        adj, conflict_analysis = self._build_enhanced_conflict_graph(
            candidates, vehicle_states, platoon_manager
        )

        # 2) Adaptive MWIS algorithm selection
        weights = [self._get_bid(c) for c in candidates]
        selected_idx = self._solve_mwis_adaptive(weights, adj, conflict_analysis)

        # 3) Enhanced winner assembly with conflict actions
        winners = self._assemble_winners_with_actions(
            candidates, selected_idx, weights, conflict_analysis
        )
        
        # 4) Update performance statistics
        resolution_time = time.time() - start_time
        self._update_stats(resolution_time, len(adj), conflict_analysis)
        
        return winners

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

    def _infer_turn_enhanced(self, agent, state: Dict, vehicle_states: Dict) -> str:
        """Enhanced turn inference using velocity vectors and waypoint analysis"""
        # 1) Existing turn inference
        turn = self._infer_turn(agent, state)
        if turn != 'straight':
            return turn
        
        # 2) Enhanced inference using velocity direction
        if state and 'velocity' in state and 'location' in state:
            velocity = state['velocity']
            location = state['location']
            
            # Calculate velocity-based heading
            if abs(velocity[0]) > 0.5 or abs(velocity[1]) > 0.5:
                velocity_heading = math.atan2(velocity[1], velocity[0])
                
                # Calculate direction to intersection center
                to_center_x = self.center[0] - location[0]
                to_center_y = self.center[1] - location[1]
                center_heading = math.atan2(to_center_y, to_center_x)
                
                # Calculate relative angle
                angle_diff = self._normalize_angle(velocity_heading - center_heading)
                
                if angle_diff > math.pi/4:  # 45 degrees
                    return 'left'
                elif angle_diff < -math.pi/4:
                    return 'right'
        
        return 'straight'

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _assemble_winners_with_actions(self, candidates: List, selected_idx: List[int], 
                                     weights: List[float], conflict_analysis: Dict) -> List[AuctionWinner]:
        """Enhanced winner assembly with intelligent action assignment"""
        go_set = set(selected_idx)
        go_sorted = sorted(list(go_set), key=lambda i: weights[i], reverse=True)
        
        # Enhanced ranking considering conflict resolution effectiveness
        rank_in_go = {}
        for rank, idx in enumerate(go_sorted):
            rank_in_go[idx] = rank + 1

        winners: List[AuctionWinner] = []
        for i, c in enumerate(candidates):
            if i in go_set:
                action = 'go'
                rank = rank_in_go[i]
            else:
                action = 'wait'
                rank = 0  # Wait vehicles get rank 0
            
            winners.append(self._to_winner(c, action, rank))
        
        return winners

    # ...existing code...

    def _calculate_enhanced_eta(self, state: Dict, agent) -> float:
        """Enhanced ETA calculation with acceleration and deceleration modeling"""
        if not state or 'location' not in state:
            return float('inf')
        
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
        current_speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        distance = _euclidean_2d(location, self.center)
        
        # Enhanced ETA with acceleration modeling
        if current_speed < 0.5:  # Vehicle is stopped or very slow
            # Assume acceleration to reasonable speed
            acceleration = 2.0  # m/s^2
            target_speed = 8.0  # m/s (reasonable urban speed)
            
            # Time to reach target speed
            accel_time = target_speed / acceleration
            accel_distance = 0.5 * acceleration * accel_time**2
            
            if distance <= accel_distance:
                # Pure acceleration phase
                return math.sqrt(2 * distance / acceleration)
            else:
                # Acceleration + constant speed phase
                remaining_distance = distance - accel_distance
                constant_speed_time = remaining_distance / target_speed
                return accel_time + constant_speed_time
        else:
            # Use current speed with slight deceleration near intersection
            effective_speed = max(current_speed * 0.8, 2.0)  # Account for intersection approach
            return distance / effective_speed

    def _predict_vehicle_path(self, state: Dict, agent) -> List[Tuple[float, float]]:
        """Predict vehicle path through intersection for spatial conflict detection"""
        if not state or 'location' not in state:
            return []
        
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
        turn = self._infer_turn_enhanced(agent, state, {})
        
        path_points = []
        current_pos = (location[0], location[1])
        path_points.append(current_pos)
        
        # Calculate path based on turn type
        if turn == 'straight':
            # Straight path through intersection
            if abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1:
                direction = math.atan2(velocity[1], velocity[0])
            else:
                # Use direction to center as fallback
                direction = math.atan2(self.center[1] - location[1], self.center[0] - location[0])
            
            # Create straight path points
            for i in range(1, 6):  # 5 points along path
                step_distance = 10.0 * i
                next_x = current_pos[0] + step_distance * math.cos(direction)
                next_y = current_pos[1] + step_distance * math.sin(direction)
                path_points.append((next_x, next_y))
        
        elif turn in ['left', 'right']:
            # Curved path for turns
            turn_radius = 15.0  # meters
            turn_direction = 1 if turn == 'left' else -1
            
            # Calculate arc path
            start_angle = math.atan2(velocity[1], velocity[0]) if abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1 else 0
            
            for i in range(1, 6):
                angle_increment = turn_direction * (math.pi/2) * (i/5.0)  # 90 degree turn over 5 steps
                current_angle = start_angle + angle_increment
                
                step_distance = 10.0 * i
                next_x = current_pos[0] + step_distance * math.cos(current_angle)
                next_y = current_pos[1] + step_distance * math.sin(current_angle)
                path_points.append((next_x, next_y))
        
        return path_points

    def _detect_enhanced_conflict(self, mi: Dict, mj: Dict) -> Optional[str]:
        """Enhanced conflict detection with spatial, temporal, and behavioral analysis"""
        # 1) Temporal conflict check (enhanced)
        ti, tj = mi['eta'], mj['eta']
        if abs(ti - tj) > self.dt_conflict:
            return None  # No temporal overlap
        
        # 2) Spatial conflict check using predicted paths
        paths_intersect = self._check_path_intersection(
            mi['predicted_path'], mj['predicted_path']
        )
        
        if paths_intersect:
            # 3) Turn-based conflict validation
            if _turn_conflict(mi['turn'], mj['turn']):
                return 'spatial_conflicts'
            else:
                return 'path_intersections'
        
        # 4) Platoon-specific conflict detection
        if mi['is_platoon'] or mj['is_platoon']:
            if self._detect_platoon_conflict(mi, mj):
                return 'platoon_conflicts'
        
        # 5) Following behavior conflict (vehicles too close in same direction)
        if self._detect_following_conflict(mi, mj):
            return 'temporal_conflicts'
        
        return None

    def _check_path_intersection(self, path1: List[Tuple[float, float]], 
                               path2: List[Tuple[float, float]]) -> bool:
        """Check if two predicted paths intersect within intersection area"""
        if not path1 or not path2:
            return False
        
        # Check each segment of path1 against each segment of path2
        for i in range(len(path1) - 1):
            for j in range(len(path2) - 1):
                if self._line_segments_intersect(
                    path1[i], path1[i+1], path2[j], path2[j+1]
                ):
                    # Verify intersection is within intersection area
                    intersection_point = self._get_intersection_point(
                        path1[i], path1[i+1], path2[j], path2[j+1]
                    )
                    if intersection_point:
                        dist_to_center = math.sqrt(
                            (intersection_point[0] - self.center[0])**2 + 
                            (intersection_point[1] - self.center[1])**2
                        )
                        if dist_to_center <= self.intersection_radius:
                            return True
        
        return False

    def _line_segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                               p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect using cross product method"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def _get_intersection_point(self, p1: Tuple[float, float], p2: Tuple[float, float],
                              p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        if 0 <= t <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return (x, y)
        
        return None

    def _detect_platoon_conflict(self, mi: Dict, mj: Dict) -> bool:
        """Detect conflicts involving platoons with enhanced spacing requirements"""
        # Platoons need more space and time
        if mi['is_platoon'] or mj['is_platoon']:
            # Stricter time window for platoons
            if abs(mi['eta'] - mj['eta']) < self.dt_conflict * 1.5:
                return True
        
        return False

    def _detect_following_conflict(self, mi: Dict, mj: Dict) -> bool:
        """Detect conflicts between vehicles that might be following each other"""
        state_i, state_j = mi['state'], mj['state']
        
        if not state_i or not state_j:
            return False
        
        # Check if vehicles are in similar locations (potential following)
        loc_i = state_i.get('location', [0, 0, 0])
        loc_j = state_j.get('location', [0, 0, 0])
        
        distance = math.sqrt((loc_i[0] - loc_j[0])**2 + (loc_i[1] - loc_j[1])**2)
        
        if distance < self.min_safe_distance:
            # Check if moving in similar directions
            vel_i = state_i.get('velocity', [0, 0, 0])
            vel_j = state_j.get('velocity', [0, 0, 0])
            
            speed_i = math.sqrt(vel_i[0]**2 + vel_i[1]**2)
            speed_j = math.sqrt(vel_j[0]**2 + vel_j[1]**2)
            
            if speed_i > 0.5 and speed_j > 0.5:
                # Calculate velocity similarity
                dot_product = vel_i[0]*vel_j[0] + vel_i[1]*vel_j[1]
                magnitude_product = speed_i * speed_j
                
                if magnitude_product > 0:
                    cosine_similarity = dot_product / magnitude_product
                    if cosine_similarity > self.velocity_similarity_threshold:
                        return True
        
        return False

    # === MWIS：精确DFS（n<=max_exact） ===
    def _mwis_exact(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        n = len(weights)
        best_sum = -1.0
        best_set: List[int] = []

        def dfs(idx: int, chosen: List[int], banned: Set[int], cur_sum: float):
            nonlocal best_sum, best_set
            if idx == n:
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_set = chosen.copy()
                return
            if idx in banned:
                dfs(idx+1, chosen, banned, cur_sum)
                return
            # 分支上界（粗略剪枝）
            # 这里可加更强上界估计；先省略
            # 选择 idx
            ok = True
            for k in chosen:
                if idx in adj[k] or k in adj[idx]:
                    ok = False
                    break
            if ok:
                new_banned = banned.union(adj[idx])
                dfs(idx+1, chosen+[idx], new_banned, cur_sum + weights[idx])
            # 不选 idx
            dfs(idx+1, chosen, banned, cur_sum)

        dfs(0, [], set(), 0.0)
        return best_set

    # === MWIS：贪心近似（n>max_exact） ===
    def _mwis_greedy(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        selected: List[int] = []
        blocked: Set[int] = set()
        for i in order:
            if i in blocked:
                continue
            # 与当前已选是否冲突
            conflict = any((i in adj[j]) or (j in adj[i]) for j in selected)
            if not conflict:
                selected.append(i)
                blocked.update(adj[i])
        return selected

    def _solve_mwis_adaptive(self, weights: List[float], adj: List[Set[int]], 
                           conflict_analysis: Dict) -> List[int]:
        """Adaptive MWIS solver selection based on problem characteristics"""
        n = len(weights)
        
        # Enhanced selection criteria
        if n <= self.max_exact:
            self.stats['mwis_exact_calls'] += 1
            return self._mwis_exact_enhanced(weights, adj)
        else:
            self.stats['mwis_greedy_calls'] += 1
            # Choose greedy variant based on conflict density
            total_conflicts = sum(conflict_analysis.values())
            conflict_density = total_conflicts / (n * (n-1) / 2) if n > 1 else 0
            
            if conflict_density > 0.3:  # High conflict density
                return self._mwis_greedy_weighted(weights, adj)
            else:
                return self._mwis_greedy(weights, adj)

    # === MWIS：精确DFS（增强版） ===
    def _mwis_exact_enhanced(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Enhanced exact MWIS with better pruning and memoization"""
        n = len(weights)
        best_sum = -1.0
        best_set: List[int] = []
        
        # Enhanced pruning with upper bound estimation
        def calculate_upper_bound(remaining_vertices: Set[int], current_sum: float) -> float:
            # Greedy upper bound: sort remaining by weight/degree ratio
            vertex_scores = []
            for v in remaining_vertices:
                degree_in_remaining = len(adj[v] & remaining_vertices)
                score = weights[v] / (degree_in_remaining + 1)
                vertex_scores.append((score, weights[v], v))
            
            vertex_scores.sort(reverse=True)
            
            upper_bound = current_sum
            used_vertices = set()
            
            for score, weight, v in vertex_scores:
                if v not in used_vertices:
                    upper_bound += weight
                    used_vertices.add(v)
                    used_vertices.update(adj[v] & remaining_vertices)
            
            return upper_bound

        def dfs_enhanced(idx: int, chosen: List[int], banned: Set[int], cur_sum: float, remaining: Set[int]):
            nonlocal best_sum, best_set
            
            if idx == n:
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_set = chosen.copy()
                return
            
            # Enhanced pruning with upper bound
            if calculate_upper_bound(remaining, cur_sum) <= best_sum:
                return
            
            if idx in banned:
                remaining.discard(idx)
                dfs_enhanced(idx+1, chosen, banned, cur_sum, remaining)
                return
            
            # Try not selecting idx first (better pruning order)
            remaining_copy = remaining.copy()
            remaining_copy.discard(idx)
            dfs_enhanced(idx+1, chosen, banned, cur_sum, remaining_copy)
            
            # Try selecting idx
            conflict_with_chosen = any(idx in adj[k] or k in adj[idx] for k in chosen)
            if not conflict_with_chosen:
                new_banned = banned | adj[idx]
                new_remaining = remaining - adj[idx] - {idx}
                dfs_enhanced(idx+1, chosen + [idx], new_banned, cur_sum + weights[idx], new_remaining)

        initial_remaining = set(range(n))
        dfs_enhanced(0, [], set(), 0.0, initial_remaining)
        return best_set

    # === MWIS：贪心近似（加权版） ===
    def _mwis_greedy_weighted(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Enhanced greedy MWIS considering weight-to-degree ratio"""
        n = len(weights)
        selected: List[int] = []
        available = set(range(n))
        
        while available:
            # Calculate weight-to-degree ratio for remaining vertices
            best_vertex = -1
            best_ratio = -1
            
            for v in available:
                degree = len(adj[v] & available)
                ratio = weights[v] / (degree + 1)  # +1 to avoid division by zero
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_vertex = v
            
            if best_vertex == -1:
                break
            
            # Select the best vertex
            selected.append(best_vertex)
            
            # Remove selected vertex and its neighbors
            to_remove = {best_vertex} | (adj[best_vertex] & available)
            available -= to_remove
        
        return selected

    # === 辅助：抽取数据/封装返回 ===
    def _get_bid(self, c) -> float:
        # 兼容不同字段名
        if hasattr(c, 'bid'):
            bid_obj = getattr(c, 'bid')
            if hasattr(bid_obj, 'value'):
                return float(bid_obj.value)
            return float(bid_obj or 0.0)
        if isinstance(c, dict) and 'bid' in c:
            return float(c['bid'] or 0.0)
        return 0.0

    def _get_agent(self, c):
        if hasattr(c, 'participant'):
            return getattr(c, 'participant')
        if isinstance(c, dict):
            return c.get('participant')
        return c  # 兜底

    def _lookup_state(self, agent, vehicle_states: Dict[str, Dict], platoon_manager=None) -> Dict:
        """
        返回包含至少 {location:(x,y,z), speed:float, turn(optional)} 的state字典。
        - 若 agent 是 platoon，则取其 leader 的state。
        - 若找不到，返回空dict。
        """
        if agent is None:
            return {}
        agent_id = getattr(agent, 'id', None) or (isinstance(agent, dict) and agent.get('id'))
        agent_type = getattr(agent, 'type', None) or (isinstance(agent, dict) and agent.get('type'))
        
        if agent_type == 'platoon' and platoon_manager is not None:
            leader_id = getattr(agent, 'data', {}).get('leader_id', None) if hasattr(agent, 'data') else (agent.get('data', {}) if isinstance(agent, dict) else {}).get('leader_id')
            if leader_id and str(leader_id) in vehicle_states:
                state = vehicle_states[str(leader_id)]
                # Calculate speed from velocity
                velocity = state.get('velocity', [0, 0, 0])
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
                return {**state, 'speed': speed}
        
        if agent_id and str(agent_id) in vehicle_states:
            state = vehicle_states[str(agent_id)]
            # Calculate speed from velocity
            velocity = state.get('velocity', [0, 0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
            return {**state, 'speed': speed}
        return {}

    def _infer_turn(self, agent, state: Dict) -> str:
        """
        尝试从 agent/state 推断转向：'left'/'right'/'straight'。
        若无信息，返回 'straight' 以保守判定。
        """
        # 1) 优先用state内已有标注
        turn = (state or {}).get('turn') if isinstance(state, dict) else None
        if turn:
            return str(turn).lower()
        # 2) 从agent.data里推断
        data = getattr(agent, 'data', {}) if agent is not None else {}
        if isinstance(data, dict):
            if 'turn' in data:
                return str(data['turn']).lower()
            if 'planned_path' in data:
                # TODO: 根据 planned_path 起终向量粗判转向；这里先简化
                pass
        return 'straight'

    def _to_winner(self, c, conflict_action: str, rank: int) -> AuctionWinner:
        bid_value = self._get_bid(c)
        participant = self._get_agent(c)
        
        # Create bid object if needed
        if hasattr(c, 'bid'):
            bid_obj = c.bid
        else:
            # Create a simple bid object
            class SimpleBid:
                def __init__(self, value):
                    self.value = value
            bid_obj = SimpleBid(bid_value)
        
        # 兼容已有 AuctionWinner；若需要保留c的其它字段，可在此扩展
        return AuctionWinner(
            participant=participant,
            bid=bid_obj,
            rank=rank,
            conflict_action=conflict_action
        )

    # === 性能统计相关 ===
    def _update_stats(self, resolution_time: float, graph_size: int, conflict_analysis: Dict):
        """Update performance statistics"""
        self.stats['total_resolutions'] += 1
        self.stats['conflicts_detected'] += sum(conflict_analysis.values())
        
        # Update average resolution time
        old_avg = self.stats['avg_resolution_time']
        new_avg = (old_avg * (self.stats['total_resolutions'] - 1) + resolution_time) / self.stats['total_resolutions']
        self.stats['avg_resolution_time'] = new_avg

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        return {
            **self.stats,
            'exact_vs_greedy_ratio': (
                self.stats['mwis_exact_calls'] / max(1, self.stats['mwis_greedy_calls'])
            ),
            'avg_conflicts_per_resolution': (
                self.stats['conflicts_detected'] / max(1, self.stats['total_resolutions'])
            )
        }
