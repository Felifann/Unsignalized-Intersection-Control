import math
from typing import List, Dict, Tuple, Set, Optional

def _euclidean_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

class ConflictAnalyzer:
    """
    Part 1: Handles all conflict detection and analysis
    - Enhanced conflict graph construction
    - Turn-based conflict detection
    - Path intersection analysis
    - Temporal and spatial conflict detection
    """
    
    def __init__(self, solver_config):
        """Initialize with solver configuration"""
        self.dt_conflict = solver_config['conflict_time_window']
        self.center = solver_config['intersection_center']
        self.deadlock_core_half_size = solver_config.get('deadlock_core_half_size', 8.0)  # Square region half-size
        self.min_safe_distance = solver_config['min_safe_distance']
        self.prediction_horizon = solver_config.get('speed_prediction_horizon', 5.0)  # Default 5 seconds ahead
        
        # TRAINABLE PARAMETERS - now configurable from DRL
        self.path_intersection_threshold = solver_config.get('path_intersection_threshold', 2.5)  # Trainable: path intersection sensitivity
        self.platoon_conflict_distance = solver_config.get('platoon_conflict_distance', 15.0)  # Trainable: platoon interaction distance
        
        self.velocity_similarity_threshold = 0.3

    def build_enhanced_conflict_graph(self, candidates: List, vehicle_states: Dict[str, Dict], 
                                     platoon_manager=None) -> Tuple[List[Set[int]], Dict]:
        """Enhanced conflict graph with geometric path analysis and time predictions"""
        n = len(candidates)
        adj: List[Set[int]] = [set() for _ in range(n)]
        conflict_analysis = {
            'spatial_conflicts': 0,
            'temporal_conflicts': 0,
            'platoon_conflicts': 0,
            'path_intersections': 0,
            'turn_conflicts': 0
        }
        
        # Enhanced metadata extraction
        meta = []
        for i, c in enumerate(candidates):
            agent = self._get_agent(c)
            state = self._lookup_state(agent, vehicle_states, platoon_manager)
            
            # Enhanced turn inference with path prediction
            turn = self._infer_turn_enhanced(agent, state, vehicle_states) if state else 'straight'
            eta = self._calculate_enhanced_eta(state, agent) if state else float('inf')
            path = self._predict_vehicle_path(state, agent) if state else []
            paths = self._predict_vehicle_paths(state, agent) if state else []
            
            meta.append({
                'index': i,
                'agent': agent,
                'state': state,
                'turn': turn,
                'eta': eta,
                'predicted_path': path,
                'predicted_paths': paths,
                'is_platoon': agent.type == 'platoon' if hasattr(agent, 'type') else False
            })

        # Enhanced conflict detection
        conflicts_found = 0
        for i in range(n):
            for j in range(i + 1, n):
                conflict_type = self._detect_enhanced_conflict(meta[i], meta[j])
                if conflict_type:
                    adj[i].add(j)
                    adj[j].add(i)
                    conflict_analysis[conflict_type] += 1
                    conflicts_found += 1
                    
                    # Debug conflict detection
                    agent_i = meta[i]['agent']
                    agent_j = meta[j]['agent']
                    print(f"   ⚡ Conflict {conflicts_found}: {getattr(agent_i, 'id', 'unknown')} <-> {getattr(agent_j, 'id', 'unknown')} ({conflict_type})")
        
        if conflicts_found == 0:
            print("   ✅ No conflicts detected - all agents can proceed")
        
        return adj, conflict_analysis

    def _detect_enhanced_conflict(self, meta_i: Dict, meta_j: Dict) -> Optional[str]:
        """Enhanced conflict detection with multiple conflict types"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            if not state_i or not state_j:
                # CONSERVATIVE: If state can't be determined, assume conflict
                return 'spatial_conflicts'
            
            # 0. SAFETY CHECK: Vehicles within intersection bounds always conflict
            if self._both_vehicles_in_intersection(state_i, state_j):
                return 'spatial_conflicts'
            
            # 1. Distance-based spatial conflict (immediate danger)
            if self._has_spatial_conflict(meta_i, meta_j):
                return 'spatial_conflicts'

            # 1.5 Right-turn exemption: if either agent is turning right, permit to move (no conflict)
            # Still subject to immediate spatial conflict above
            turn_i = (meta_i.get('turn') or 'straight').lower()
            turn_j = (meta_j.get('turn') or 'straight').lower()
            if turn_i == 'right' or turn_j == 'right':
                return None
            
            # 2. Both vehicles approaching intersection at similar times
            if self._has_temporal_conflict(meta_i, meta_j):
                return 'temporal_conflicts'
            
            # 3. Path intersection check
            if self._has_path_intersection(meta_i, meta_j):
                return 'path_intersections'
            
            # 4. Enhanced turn-based conflict detection
            if self._has_turn_conflict(meta_i, meta_j):
                return 'turn_conflicts'
            
            # 5. Platoon-specific conflicts
            if (meta_i['is_platoon'] or meta_j['is_platoon']) and self._has_proximity_conflict(meta_i, meta_j, self.platoon_conflict_distance):
                return 'platoon_conflicts'
            
            return None
            
        except Exception as e:
            print(f"[Warning] Enhanced conflict detection failed: {e}")
            return 'spatial_conflicts'  # Conservative fallback

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
            
            # If ETA can't be calculated for either, do not infer temporal conflict
            if eta_i == float('inf') or eta_j == float('inf'):
                return False
            
            # Conflict if ETAs are within conflict time window
            return abs(eta_i - eta_j) < self.dt_conflict
            
        except Exception:
            # If temporal information is unreliable, don't use it to assert conflict
            return False

    def _has_path_intersection(self, meta_i: Dict, meta_j: Dict) -> bool:
        """Check for path intersections"""
        try:
            # Support multiple candidate polylines if available
            paths_i = meta_i.get('predicted_paths')
            paths_j = meta_j.get('predicted_paths')

            if paths_i is None:
                pi = meta_i.get('predicted_path', [])
                paths_i = [pi] if pi else []
            if paths_j is None:
                pj = meta_j.get('predicted_path', [])
                paths_j = [pj] if pj else []

            if not paths_i or not paths_j:
                return False

            # Check intersection or near-miss for any path pair
            for path_i in paths_i:
                for path_j in paths_j:
                    if len(path_i) < 2 or len(path_j) < 2:
                        continue
                    for i in range(len(path_i) - 1):
                        for j in range(len(path_j) - 1):
                            p1, p2 = path_i[i], path_i[i+1]
                            p3, p4 = path_j[j], path_j[j+1]
                            if self._segments_intersect(p1, p2, p3, p4):
                                return True
                            # Near miss
                            if self._segment_min_distance(p1, p2, p3, p4) < self.path_intersection_threshold:
                                return True

            return False
            
        except Exception:
            # If path information is unreliable, don't use it to assert conflict
            return False

    def _has_turn_conflict(self, meta_i: Dict, meta_j: Dict) -> bool:
        """Check for turn-based conflicts using enhanced logic"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            location_i = state_i.get('location', (0, 0, 0))
            location_j = state_j.get('location', (0, 0, 0))
            
            approach_i = self._infer_approach_direction(location_i)
            approach_j = self._infer_approach_direction(location_j)
            
            # Use enhanced turn conflict detection
            return self._turn_conflict_enhanced(
                meta_i['turn'], meta_j['turn'],
                approach_i, approach_j,
                location_i, location_j
            )
            
        except Exception:
            return False

    def _has_proximity_conflict(self, meta_i: Dict, meta_j: Dict, threshold: float) -> bool:
        """Check for proximity conflicts"""
        try:
            state_i = meta_i['state']
            state_j = meta_j['state']
            
            loc_i = state_i['location']
            loc_j = state_j['location']
            
            distance = _euclidean_2d(loc_i, loc_j)
            return distance < threshold
            
        except Exception:
            return False

    def _both_vehicles_in_intersection(self, state_i: Dict, state_j: Dict) -> bool:
        """Check if both vehicles are within the intersection square bounds"""
        try:
            loc_i = state_i['location']
            loc_j = state_j['location']
            
            center_x, center_y = self.center[0], self.center[1]
            half_size = self.deadlock_core_half_size
            
            # Check if both vehicles are within intersection square
            in_intersection_i = (
                (center_x - half_size) <= loc_i[0] <= (center_x + half_size) and
                (center_y - half_size) <= loc_i[1] <= (center_y + half_size)
            )
            
            in_intersection_j = (
                (center_x - half_size) <= loc_j[0] <= (center_x + half_size) and
                (center_y - half_size) <= loc_j[1] <= (center_y + half_size)
            )
            
            return in_intersection_i and in_intersection_j
            
        except Exception:
            return True  # Conservative: assume conflict if can't determine location

    # ...existing conflict detection helper methods...
    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                           p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect using orientation tests (including collinear overlap)"""
        try:
            def orientation(a, b, c):
                val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
                if abs(val) < 1e-9:
                    return 0
                return 1 if val > 0 else 2

            def on_segment(a, b, c):
                return (
                    min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
                    min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
                )

            o1 = orientation(p1, p2, p3)
            o2 = orientation(p1, p2, p4)
            o3 = orientation(p3, p4, p1)
            o4 = orientation(p3, p4, p2)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases - collinear and overlapping
            if o1 == 0 and on_segment(p1, p2, p3):
                return True
            if o2 == 0 and on_segment(p1, p2, p4):
                return True
            if o3 == 0 and on_segment(p3, p4, p1):
                return True
            if o4 == 0 and on_segment(p3, p4, p2):
                return True

            return False
        except Exception:
            return False

    def _distance_between_segments(self, p1: Tuple[float, float], p2: Tuple[float, float],
                                  p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        """Calculate minimum distance between two line segments (legacy alias)"""
        return self._segment_min_distance(p1, p2, p3, p4)

    def _segment_min_distance(self, a1: Tuple[float, float], a2: Tuple[float, float],
                              b1: Tuple[float, float], b2: Tuple[float, float]) -> float:
        """Compute minimal distance between two segments in 2D"""
        try:
            if self._segments_intersect(a1, a2, b1, b2):
                return 0.0

            def dot(u, v):
                return u[0]*v[0] + u[1]*v[1]

            def sub(u, v):
                return (u[0]-v[0], u[1]-v[1])

            def norm(u):
                return math.hypot(u[0], u[1])

            def point_to_segment_distance(p, s1, s2):
                v = sub(s2, s1)
                w = sub(p, s1)
                c1 = dot(w, v)
                if c1 <= 0:
                    return norm(sub(p, s1))
                c2 = dot(v, v)
                if c2 <= c1:
                    return norm(sub(p, s2))
                b = c1 / c2
                pb = (s1[0] + b*v[0], s1[1] + b*v[1])
                return norm(sub(p, pb))

            return min(
                point_to_segment_distance(a1, b1, b2),
                point_to_segment_distance(a2, b1, b2),
                point_to_segment_distance(b1, a1, a2),
                point_to_segment_distance(b2, a1, a2)
            )
        except Exception:
            return float('inf')

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
        """Predict a primary vehicle path through the intersection (turn-aware polyline)"""
        paths = self._predict_vehicle_paths(state, agent)
        return paths[0] if paths else []

    def _predict_vehicle_paths(self, state: Dict, agent) -> List[List[Tuple[float, float]]]:
        """Predict one or more candidate polylines for the vehicle path based on approach and turn"""
        if not state or 'location' not in state:
            return []

        loc = state['location']
        pos = (loc[0], loc[1])

        # If stationary, path is just current position
        velocity = state.get('velocity', [0, 0, 0])
        moving = velocity and (abs(velocity[0]) > 1e-3 or abs(velocity[1]) > 1e-3)

        center_xy = (self.center[0], self.center[1])
        approach = self._infer_approach_direction(state.get('location', (0,0,0)))
        turn = state.get('turn') if 'turn' in state else self._infer_turn_enhanced(agent, state, {})

        def dir_vector(direction: str) -> Tuple[float, float]:
            mapping = {
                'east': (1.0, 0.0),
                'west': (-1.0, 0.0),
                'north': (0.0, 1.0),
                'south': (0.0, -1.0)
            }
            return mapping.get(direction, (0.0, 0.0))

        def dest_direction_for_turn(appr: str, t: str) -> str:
            if t == 'straight':
                m = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
            elif t == 'left':
                m = {'north': 'east', 'east': 'south', 'south': 'west', 'west': 'north'}
            elif t == 'right':
                m = {'north': 'west', 'west': 'south', 'south': 'east', 'east': 'north'}
            elif t == 'u_turn':
                m = {'north': 'north', 'south': 'south', 'east': 'east', 'west': 'west'}
            else:
                return ''
            return m.get(appr, '')

        exit_distance = max(self.deadlock_core_half_size * 2.0, 30.0)

        def build_polyline_for_turn(appr: str, t: str) -> List[Tuple[float, float]]:
            if not appr:
                return [pos]
            dest_dir = dest_direction_for_turn(appr, t)
            if not dest_dir:
                return [pos]
            dv = dir_vector(dest_dir)
            dest_point = (center_xy[0] + dv[0] * exit_distance, center_xy[1] + dv[1] * exit_distance)
            # Simple 3-point polyline: current position -> center -> out along dest direction
            return [pos, center_xy, dest_point]

        candidate_turns = []
        if turn in ('left', 'right', 'straight', 'u_turn'):
            candidate_turns = [turn]
        else:
            # Unknown turn: consider common options conservatively
            candidate_turns = ['straight', 'left', 'right']

        polylines = [build_polyline_for_turn(approach, t) for t in candidate_turns]

        # If not moving, reduce to a single point path
        if not moving:
            return [[pos]]

        return polylines

    def _get_agent(self, candidate) -> object:
        """Extract agent from candidate"""
        if hasattr(candidate, 'participant'):
            return candidate.participant
        elif hasattr(candidate, 'agent'):
            return candidate.agent
        else:
            return candidate

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

    def _infer_turn_enhanced(self, agent, state: Dict, vehicle_states: Dict) -> str:
        """Enhanced turn inference with speed vector and path prediction"""
        if not state or 'location' not in state:
            return 'straight'
    
        location = state['location']
        velocity = state.get('velocity', [0, 0, 0])
    
        # 1. Basic turn inference
        basic_turn = self._infer_turn(agent, state)
    
        # 2. Speed vector + approach cross-product analysis (more robust)
        if abs(velocity[0]) > 0.3 or abs(velocity[1]) > 0.3:
            velocity_turn = self._infer_turn_from_velocity_and_approach(location, velocity)
            if velocity_turn != 'straight':
                return velocity_turn
    
        # 3. Destination-based analysis
        if hasattr(agent, 'destination') or 'destination' in state:
            destination_turn = self._infer_turn_from_destination(agent, state)
            if destination_turn != 'straight':
                return destination_turn
    
        return basic_turn

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

    def _infer_turn_from_velocity(self, location: Tuple[float, float, float], 
                            velocity: List[float]) -> str:
        """Infer turn based on velocity vector"""
        try:
            # Calculate current heading
            current_heading = math.atan2(velocity[1], velocity[0])
        
            # Calculate direction to intersection center
            to_center_x = self.center[0] - location[0]
            to_center_y = self.center[1] - location[1]
            to_center_heading = math.atan2(to_center_y, to_center_x)
        
            # Calculate relative angle
            relative_angle = self._normalize_angle(current_heading - to_center_heading)
        
            # Determine turn based on angle
            if relative_angle > math.pi/3:  # 60 degrees
                return 'left'
            elif relative_angle < -math.pi/3:
                return 'right'
            elif abs(relative_angle) > 2*math.pi/3:  # 120 degrees, possibly u-turn
                return 'u_turn'
            else:
                return 'straight'
            
        except Exception:
            return 'straight'

    def _inbound_unit_vector_for_approach(self, approach: str) -> Tuple[float, float]:
        """Unit vector pointing toward the center from a given approach arm"""
        mapping = {
            'east': (-1.0, 0.0),   # from east side heading west toward center
            'west': (1.0, 0.0),    # from west side heading east toward center
            'north': (0.0, -1.0),  # from north side heading south toward center
            'south': (0.0, 1.0),   # from south side heading north toward center
        }
        return mapping.get(approach, (0.0, 0.0))

    def _infer_turn_from_velocity_and_approach(self, location: Tuple[float, float, float],
                                               velocity: List[float]) -> str:
        """Use cross product of approach inbound vector and velocity to decide left/right/straight"""
        try:
            approach = self._infer_approach_direction(location)
            ax, ay = self._inbound_unit_vector_for_approach(approach)
            vx, vy = float(velocity[0]), float(velocity[1])
            speed = math.hypot(vx, vy)
            if speed < 0.3 or (ax == 0.0 and ay == 0.0):
                return 'straight'

            # Normalize velocity
            vnx, vny = vx / speed, vy / speed
            # Dot: alignment with inbound direction (straight when large)
            dot_av = ax * vnx + ay * vny
            # Cross z: + left of approach, - right of approach
            cross_z = ax * vny - ay * vnx

            if dot_av > math.cos(math.radians(30)):
                return 'straight'
            # Noise threshold for right/left determination
            if cross_z > 0.15:
                return 'left'
            if cross_z < -0.15:
                return 'right'
            return 'straight'
        except Exception:
            return 'straight'

    def _infer_turn_from_destination(self, agent, state: Dict) -> str:
        """Infer turn based on destination"""
        try:
            destination = None
            if hasattr(agent, 'destination'):
                destination = agent.destination
            elif 'destination' in state:
                destination = state['destination']
        
            if not destination:
                return 'straight'
        
            location = state['location']
        
            # Calculate direction to destination
            to_dest_x = destination[0] - location[0]
            to_dest_y = destination[1] - location[1]
            to_dest_heading = math.atan2(to_dest_y, to_dest_x)
        
            # Calculate direction through intersection center
            to_center_x = self.center[0] - location[0]
            to_center_y = self.center[1] - location[1]
            to_center_heading = math.atan2(to_center_y, to_center_x)
        
            # Compare directions
            angle_diff = self._normalize_angle(to_dest_heading - to_center_heading)
        
            if angle_diff > math.pi/4:
                return 'left'
            elif angle_diff < -math.pi/4:
                return 'right'
            else:
                return 'straight'
            
        except Exception:
            return 'straight'

    def _turn_conflict_enhanced(self, turn_i: str, turn_j: str, 
                          approach_i: str, approach_j: str,
                          location_i: Tuple[float, float, float],
                          location_j: Tuple[float, float, float]) -> bool:
        """Enhanced turn conflict detection based on actual road geometry"""
        # Normalize inputs
        turn_i = (turn_i or 'straight').lower()
        turn_j = (turn_j or 'straight').lower()
    
        # If can't determine approach directions, use position inference
        if not approach_i:
            approach_i = self._infer_approach_direction(location_i)
        if not approach_j:
            approach_j = self._infer_approach_direction(location_j)
    
        # 1. Same approach vehicle conflicts
        if approach_i == approach_j:
            return self._same_approach_conflict(turn_i, turn_j)
    
        # 2. Opposite approach vehicle conflicts
        if self._are_opposite_approaches(approach_i, approach_j):
            return self._opposite_approach_conflict(turn_i, turn_j)
    
        # 3. Perpendicular approach vehicle conflicts
        if self._are_perpendicular_approaches(approach_i, approach_j):
            return self._perpendicular_approach_conflict(turn_i, turn_j, approach_i, approach_j)
    
        # 4. Default conservative handling
        return True

    def _infer_approach_direction(self, location: Tuple[float, float, float]) -> str:
        """Infer approach direction based on vehicle position"""
        try:
            center_x, center_y = self.center[0], self.center[1]
            rel_x = location[0] - center_x
            rel_y = location[1] - center_y
        
            # Use angle to determine main direction
            angle = math.atan2(rel_y, rel_x)
            angle_deg = math.degrees(angle)
        
            # Normalize to [0, 360)
            if angle_deg < 0:
                angle_deg += 360
        
            # Assign to four main directions
            if 315 <= angle_deg or angle_deg < 45:
                return 'east'
            elif 45 <= angle_deg < 135:
                return 'north'
            elif 135 <= angle_deg < 225:
                return 'west'
            else:  # 225 <= angle_deg < 315
                return 'south'
            
        except Exception:
            return 'unknown'

    def _same_approach_conflict(self, turn_i: str, turn_j: str) -> bool:
        """Same approach vehicle conflict matrix"""
        conflict_matrix = {
            ('left', 'left'): False,
            ('left', 'right'): True,
            ('left', 'straight'): True,
            ('right', 'right'): False,
            ('right', 'straight'): True,
            ('straight', 'straight'): False,
            ('u_turn', 'left'): True,
            ('u_turn', 'right'): True,
            ('u_turn', 'straight'): True,
            ('u_turn', 'u_turn'): True,
        }
    
        key = tuple(sorted([turn_i, turn_j]))
        return conflict_matrix.get(key, True)

    def _opposite_approach_conflict(self, turn_i: str, turn_j: str) -> bool:
        """Opposite approach vehicle conflict matrix - customized rules"""
        # Rules:
        # - Right turns: always non-conflict
        # - Straight vs straight: non-conflict (opposite directions)
        # - Left turns: conflict with anything
        conflict_matrix = {
            ('right', 'right'): False,
            ('right', 'straight'): False,
            ('left', 'right'): True,
            ('left', 'left'): True,
            ('left', 'straight'): True,
            ('straight', 'straight'): False,
            ('u_turn', 'left'): True,
            ('u_turn', 'right'): True,
            ('u_turn', 'straight'): True,
            ('u_turn', 'u_turn'): True,
        }

        key = tuple(sorted([turn_i, turn_j]))
        return conflict_matrix.get(key, True)

    def _perpendicular_approach_conflict(self, turn_i: str, turn_j: str, 
                                   approach_i: str, approach_j: str) -> bool:
        """Perpendicular approach vehicle conflict matrix - stricter with right-turn exemptions"""
        # If any right turn involved, treat as non-conflict to increase flow
        if turn_i == 'right' or turn_j == 'right':
            return False
        # Left turns should be strict: conflict
        if turn_i == 'left' or turn_j == 'left':
            return True
        # Straight vs straight on perpendicular approaches tends to gridlock: conflict
        return True

    def _are_opposite_approaches(self, approach_i: str, approach_j: str) -> bool:
        """Check if approaches are opposite"""
        opposites = {
            ('north', 'south'), ('south', 'north'),
            ('east', 'west'), ('west', 'east')
        }
        return (approach_i, approach_j) in opposites

    def _are_perpendicular_approaches(self, approach_i: str, approach_j: str) -> bool:
        """Check if approaches are perpendicular"""
        # If approach can't be determined, we cannot assert perpendicularity
        if approach_i == 'unknown' or approach_j == 'unknown':
            return False
        return not self._are_opposite_approaches(approach_i, approach_j) and approach_i != approach_j

    def _is_left_approach(self, approach_i: str, approach_j: str) -> bool:
        """Check if approach_i is to the left of approach_j"""
        left_relations = {
            'north': 'west',
            'west': 'south',
            'south': 'east',
            'east': 'north'
        }
        return left_relations.get(approach_j) == approach_i

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
