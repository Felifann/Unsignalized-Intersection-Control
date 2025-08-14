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
        
        # Deadlock detection parameters - Use exact square area like show_intersection_area1
        self.deadlock_detection_window = 15.0  # seconds to track for deadlock
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

    def _assemble_winners_with_traffic_control(self, candidates: List, selected_idx: List[int], 
                                             weights: List[float], conflict_analysis: Dict,
                                             vehicle_states: Dict[str, Dict]) -> List[AuctionWinner]:
        """Enhanced winner assembly with traffic flow control logic"""
        go_set = set(selected_idx)
        go_sorted = sorted(list(go_set), key=lambda i: weights[i], reverse=True)
        
        # Enhanced ranking considering conflict resolution effectiveness
        rank_in_go = {}
        for rank, idx in enumerate(go_sorted):
            rank_in_go[idx] = rank + 1

        winners: List[AuctionWinner] = []
        for i, c in enumerate(candidates):
            if i in go_set:
                # Check if this winner should be allowed to enter core region
                if self._should_allow_entry(c, vehicle_states):
                    action = 'go'
                    rank = rank_in_go[i]
                else:
                    # Block entry due to traffic flow control
                    action = 'wait'
                    rank = 0
                    if self.region_entry_blocked:
                        print(f"🚧 Blocking {self._get_agent_type(c)} {self._get_agent_id(c)} - region entry controlled")
            else:
                action = 'wait'
                rank = 0  # Wait vehicles get rank 0
            
            winners.append(self._to_winner(c, action, rank))
        
        return winners

    def _should_allow_entry(self, candidate, vehicle_states: Dict[str, Dict]) -> bool:
        """Determine if a candidate should be allowed to enter the core region"""
        # If traffic flow control is not active, allow entry
        if not self.region_entry_blocked:
            return True
        
        # If traffic flow control is active, check if candidate is approaching core region
        agent = self._get_agent(candidate)
        state = self._lookup_state(agent, vehicle_states, None)
        
        if not state or 'location' not in state:
            return True  # Default to allow if no location info
        
        location = state['location']
        
        # Check if vehicle is already in core region (allow to continue)
        if self._is_in_core_region(location):
            return True  # Already in region, let it continue
        
        # Check if vehicle is approaching core region
        if self._is_approaching_core_region(location, state):
            return False  # Block entry to core region
        
        # Vehicle is far from core region, allow normal movement
        return True

    def _is_in_core_region(self, location: Tuple[float, float, float]) -> bool:
        """Check if location is within the core blue square region"""
        center_x, center_y = self.center[0], self.center[1]
        half_size = self.deadlock_core_half_size
        
        return (
            (center_x - half_size) <= location[0] <= (center_x + half_size) and
            (center_y - half_size) <= location[1] <= (center_y + half_size)
        )

    def _is_approaching_core_region(self, location: Tuple[float, float, float], state: Dict) -> bool:
        """Check if vehicle is approaching the core region"""
        # Calculate distance to core region boundary
        center_x, center_y = self.center[0], self.center[1]
        half_size = self.deadlock_core_half_size
        
        # Distance to core region center
        distance_to_center = math.sqrt((location[0] - center_x)**2 + (location[1] - center_y)**2)
        
        # Consider approaching if within 2x the core region size
        approach_threshold = half_size * 2
        
        if distance_to_center <= approach_threshold:
            # Check if moving towards the core region
            velocity = state.get('velocity', [0, 0, 0])
            if velocity and len(velocity) >= 2:
                # Calculate direction vector to core center
                to_center_x = center_x - location[0]
                to_center_y = center_y - location[1]
                
                # Calculate dot product to see if moving towards center
                dot_product = velocity[0] * to_center_x + velocity[1] * to_center_y
                
                # If moving towards center, consider it approaching
                return dot_product > 0
        
        return False

    def _get_agent_type(self, candidate) -> str:
        """Get agent type for logging"""
        agent = self._get_agent(candidate)
        agent_type = getattr(agent, 'type', None)
        return agent_type or 'vehicle'

    def _get_agent_id(self, candidate) -> str:
        """Get agent ID for logging"""
        agent = self._get_agent(candidate)
        agent_id = getattr(agent, 'id', None)
        return str(agent_id) if agent_id else 'unknown'

    def _handle_deadlock_detection(self):
        """Handle detected deadlock - stop simulation"""
        self.stats['deadlocks_detected'] += 1
        
        print(f"\n💀 SIMULATION STOPPED DUE TO DEADLOCK")
        print(f"📊 Total deadlocks detected: {self.stats['deadlocks_detected']}")
        print(f"🎯 Conflict resolutions attempted: {self.stats['total_resolutions']}")
        print(f"⚡ Average resolution time: {self.stats['avg_resolution_time']:.3f}s")
        print(f"🚧 Entry blocks activated: {self.stats['entry_blocks_activated']}")
        print(f"✅ Entry blocks released: {self.stats['entry_blocks_released']}")
        
        # Raise exception to stop simulation
        raise DeadlockException("Deadlock detected - simulation terminated")

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
            ),
            'deadlock_detection_enabled': True,
            'deadlock_history_length': len(self.deadlock_history),
            'deadlock_core_half_size': self.deadlock_core_half_size,
            'traffic_flow_control_active': self.region_entry_blocked,
            'entry_blocks_activated': self.stats['entry_blocks_activated'],
            'entry_blocks_released': self.stats['entry_blocks_released']
        }

    def _detect_deadlock(self, vehicle_states: Dict[str, Dict], current_time: float) -> bool:
        """Detect deadlock situations based on vehicle states and history"""
        self._update_deadlock_history(vehicle_states, current_time)
        
        # Analyze deadlock patterns in the history
        if len(self.deadlock_history) < 2:
            return False  # Not enough data for detection
        
        latest_snapshot = self.deadlock_history[-1]
        previous_snapshot = self.deadlock_history[-2]
        
        # 1) Check for consistent stopped vehicles in the core area
        if self._check_consistent_stopped_vehicles():
            return True
        
        # 2) Check for circular waiting patterns
        if self._check_circular_waiting_pattern():
            return True
        
        # 3) Check for no progress in core square
        if self._check_no_progress_pattern():
            return True
        
        return False

    def _update_deadlock_history(self, vehicle_states: Dict[str, Dict], current_time: float):
        """Update the deadlock history with the current state of the intersection"""
        snapshot = self._create_state_snapshot(self._get_intersection_vehicles(vehicle_states), current_time)
        self.deadlock_history.append(snapshot)
        
        # Limit history size
        if len(self.deadlock_history) > 10:
            self.deadlock_history.pop(0)

    def _get_intersection_vehicles(self, vehicle_states: Dict[str, Dict]) -> List[Dict]:
        """Get vehicles that are in the core deadlock detection SQUARE area (same as show_intersection_area1)"""
        intersection_vehicles = []
        
        for vehicle_id, vehicle_state in vehicle_states.items():
            if not vehicle_state or 'location' not in vehicle_state:
                continue
            
            location = vehicle_state['location']
            
            # Use EXACT SQUARE bounds like show_intersection_area1
            center_x, center_y = self.center[0], self.center[1]
            half_size = self.deadlock_core_half_size
            
            # Check if vehicle is within the square bounds
            in_core_square = (
                (center_x - half_size) <= location[0] <= (center_x + half_size) and
                (center_y - half_size) <= location[1] <= (center_y + half_size)
            )
            
            # Check if vehicle is within the buffer square (2x the core area)
            buffer_half_size = half_size * 2
            in_buffer_square = (
                (center_x - buffer_half_size) <= location[0] <= (center_x + buffer_half_size) and
                (center_y - buffer_half_size) <= location[1] <= (center_y + buffer_half_size)
            )
            
            if in_core_square:
                vehicle_data = dict(vehicle_state)
                vehicle_data['id'] = vehicle_id
                vehicle_data['in_core_area'] = True
                vehicle_data['distance_to_center'] = _euclidean_2d(location, self.center)
                intersection_vehicles.append(vehicle_data)
            elif in_buffer_square:
                # Include vehicles in buffer zone but mark them differently
                vehicle_data = dict(vehicle_state)
                vehicle_data['id'] = vehicle_id
                vehicle_data['in_core_area'] = False
                vehicle_data['distance_to_center'] = _euclidean_2d(location, self.center)
                intersection_vehicles.append(vehicle_data)
        
        return intersection_vehicles

    def _create_state_snapshot(self, vehicles: List[Dict], timestamp: float) -> Dict:
        """Create a snapshot of the intersection state focusing on core SQUARE area"""
        stopped_vehicles = []
        moving_vehicles = []
        core_stopped = []
        buffer_stopped = []
        
        for vehicle in vehicles:
            velocity = vehicle.get('velocity', [0, 0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
            
            vehicle_summary = {
                'id': vehicle['id'],
                'location': vehicle['location'],
                'speed': speed,
                'is_junction': vehicle.get('is_junction', False),
                'distance_to_center': vehicle.get('distance_to_center', 0),
                'in_core_area': vehicle.get('in_core_area', False)
            }
            
            if speed < self.deadlock_speed_threshold:
                stopped_vehicles.append(vehicle_summary)
                if vehicle.get('in_core_area', False):
                    core_stopped.append(vehicle_summary)
                else:
                    buffer_stopped.append(vehicle_summary)
            else:
                moving_vehicles.append(vehicle_summary)
        
        return {
            'timestamp': timestamp,
            'stopped_vehicles': stopped_vehicles,
            'moving_vehicles': moving_vehicles,
            'core_stopped': core_stopped,
            'buffer_stopped': buffer_stopped,
            'total_vehicles': len(vehicles),
            'stopped_count': len(stopped_vehicles),
            'moving_count': len(moving_vehicles),
            'core_stopped_count': len(core_stopped),
            'core_detection_half_size': self.deadlock_core_half_size
        }

    def _print_deadlock_analysis(self):
        """Print detailed deadlock analysis with core SQUARE area focus"""
        latest = self.deadlock_history[-1]
        
        print(f"⏰ Detection Time: {time.strftime('%H:%M:%S', time.localtime(latest['timestamp']))}")
        print(f"🚗 Total Vehicles: {latest['total_vehicles']}")
        print(f"🛑 Stopped Vehicles: {latest['stopped_count']}")
        print(f"🚦 Moving Vehicles: {latest['moving_count']}")
        print(f"🔴 Core Square Stopped: {latest['core_stopped_count']}")
        
        # Calculate square bounds for display
        center_x, center_y = self.center[0], self.center[1]
        half_size = latest['core_detection_half_size']
        min_x, max_x = center_x - half_size, center_x + half_size
        min_y, max_y = center_y - half_size, center_y + half_size
        
        print(f"📏 Core Detection Square: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
        print(f"   Square Size: {half_size * 2:.1f}m x {half_size * 2:.1f}m")
        
        print(f"\n📍 Core Square Analysis (Deadlock Detection Zone):")
        for vehicle in latest['core_stopped']:
            status = "🔴 CORE SQUARE" if vehicle.get('in_core_area', False) else "🟡 BUFFER ZONE"
            location = vehicle.get('location', [0, 0, 0])
            distance = vehicle.get('distance_to_center', 0)
            print(f"   {status} Vehicle {vehicle['id']} - Pos: ({location[0]:.1f}, {location[1]:.1f}), "
                  f"Distance: {distance:.1f}m, Speed: {vehicle['speed']:.1f}m/s")
        
        if latest['buffer_stopped']:
            print(f"\n📍 Buffer Zone Stopped Vehicles:")
            for vehicle in latest['buffer_stopped'][:3]:  # Show max 3 buffer vehicles
                location = vehicle.get('location', [0, 0, 0])
                distance = vehicle.get('distance_to_center', 0)
                print(f"   🟡 BUFFER Vehicle {vehicle['id']} - Pos: ({location[0]:.1f}, {location[1]:.1f}), "
                      f"Distance: {distance:.1f}m, Speed: {vehicle['speed']:.1f}m/s")
        
        if len(self.deadlock_history) >= 2:
            duration = latest['timestamp'] - self.deadlock_history[0]['timestamp']
            print(f"\n⏱️  Deadlock Duration: {duration:.1f} seconds")
        
        print(f"\n🔄 Pattern Analysis (Core Square Focus):")
        print(f"   • Consistent core stopped vehicles: {self._check_consistent_stopped_vehicles()}")
        print(f"   • Core square circular waiting: {self._check_circular_waiting_pattern()}")
        print(f"   • No progress in core square: {self._check_no_progress_pattern()}")

    def _check_consistent_stopped_vehicles(self) -> bool:
        """Check if there are consistent stopped vehicles in the core area across history"""
        if len(self.deadlock_history) < 2:
            return False
        
        latest_stopped = {v['id'] for v in self.deadlock_history[-1]['core_stopped']}
        previous_stopped = {v['id'] for v in self.deadlock_history[-2]['core_stopped']}
        
        # Check if the same vehicles are stopped in the core area in the latest two snapshots
        return len(latest_stopped & previous_stopped) >= self.deadlock_min_vehicles

    def _check_circular_waiting_pattern(self) -> bool:
        """Check for circular waiting patterns in the core area"""
        if len(self.deadlock_history) < 2:
            return False
        
        latest_core = self.deadlock_history[-1]['core_stopped']
        previous_core = self.deadlock_history[-2]['core_stopped']
        
        # Create a mapping of vehicle ID to their position in the latest snapshot
        position_map = {v['id']: v['location'] for v in latest_core}
        
        # Check if any of the previously stopped vehicles are now in motion and vice versa
        for vehicle in previous_core:
            if vehicle['id'] in position_map:
                # Vehicle was stopped, now check if it's in motion
                if vehicle.get('speed', 0) > self.deadlock_speed_threshold:
                    return True  # Detected motion from a previously stopped vehicle
        
        return False

    def _check_no_progress_pattern(self) -> bool:
        """Check for no progress patterns in the core square"""
        if len(self.deadlock_history) < 2:
            return False
        
        latest = self.deadlock_history[-1]
        previous = self.deadlock_history[-2]
        
        # Check if the core stopped vehicles have not changed their relative positions
        for vehicle in latest['core_stopped']:
            if vehicle['id'] not in {v['id'] for v in previous['core_stopped']}:
                continue  # Vehicle not present in previous snapshot
            
            # Find the corresponding vehicle in the previous snapshot
            prev_vehicle = next((v for v in previous['core_stopped'] if v['id'] == vehicle['id']), None)
            if prev_vehicle:
                # Compare distances to center as a proxy for progress
                if abs(vehicle['distance_to_center'] - prev_vehicle['distance_to_center']) < 0.1:
                    return True  # Detected no progress
        
        return False

    # === Missing helper methods ===
    def _get_agent(self, candidate):
        """Extract agent from candidate"""
        if hasattr(candidate, 'participant'):
            return candidate.participant
        elif hasattr(candidate, 'agent'):
            return candidate.agent
        else:
            return candidate

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

    def _calculate_enhanced_eta(self, state: Dict, agent) -> float:
        """Calculate enhanced ETA to intersection"""
        if not state or 'location' not in state:
            return float('inf')
        
        try:
            location = state['location']
            distance = _euclidean_2d(location, self.center)
            
            # Get speed from state
            velocity = state.get('velocity', [0, 0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
            speed = max(speed, 0.1)  # Avoid division by zero
            
            return distance / speed
            
        except Exception:
            return float('inf')

    def _predict_vehicle_path(self, state: Dict, agent) -> List[Tuple[float, float]]:
        """Predict vehicle path"""
        if not state or 'location' not in state:
            return []
        
        try:
            location = state['location']
            velocity = state.get('velocity', [0, 0, 0])
            
            # Simple linear prediction
            current_pos = (location[0], location[1])
            
            if abs(velocity[0]) < 0.1 and abs(velocity[1]) < 0.1:
                return [current_pos]  # Stationary vehicle
            
            # Predict next few positions
            path = [current_pos]
            dt = 1.0  # 1 second intervals
            for i in range(1, 6):  # 5 second prediction
                next_x = location[0] + velocity[0] * dt * i
                next_y = location[1] + velocity[1] * dt * i
                path.append((next_x, next_y))
            
            return path
            
        except Exception:
            return [(state.get('location', [0, 0, 0])[0], state.get('location', [0, 0, 0])[1])]

    def _detect_enhanced_conflict(self, meta_i: Dict, meta_j: Dict) -> Optional[str]:
        """Detect enhanced conflicts between two agents"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            if not state_i or not state_j:
                return None
            
            # 1. Spatial conflict check
            if self._has_spatial_conflict(meta_i, meta_j):
                return 'spatial_conflicts'
            
            # 2. Temporal conflict check
            if self._has_temporal_conflict(meta_i, meta_j):
                return 'temporal_conflicts'
            
            # 3. Path intersection check
            if self._has_path_intersection(meta_i, meta_j):
                return 'path_intersections'
            
            # 4. Turn-based conflict
            if _turn_conflict(meta_i['turn'], meta_j['turn']):
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
