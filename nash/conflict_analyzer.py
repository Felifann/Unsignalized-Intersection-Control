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
        self.intersection_radius = solver_config['intersection_radius']
        self.min_safe_distance = solver_config['min_safe_distance']
        self.prediction_horizon = solver_config['speed_prediction_horizon']
        self.path_intersection_threshold = 3.0
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
            
            meta.append({
                'index': i,
                'agent': agent,
                'state': state,
                'turn': turn,
                'eta': eta,
                'predicted_path': path,
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
                return None
            
            # 1. Distance-based spatial conflict (immediate danger)
            if self._has_spatial_conflict(meta_i, meta_j):
                return 'spatial_conflicts'
            
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
            if (meta_i['is_platoon'] or meta_j['is_platoon']) and self._has_proximity_conflict(meta_i, meta_j, 15.0):
                return 'platoon_conflicts'
            
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

    # ...existing conflict detection helper methods...
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
    
        # 2. Speed vector analysis
        if abs(velocity[0]) > 0.5 or abs(velocity[1]) > 0.5:
            velocity_turn = self._infer_turn_from_velocity(location, velocity)
        
            # If velocity vector gives clear signal, use it
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
        """Opposite approach vehicle conflict matrix"""
        conflict_matrix = {
            ('left', 'left'): False,
            ('left', 'right'): False,
            ('left', 'straight'): True,
            ('right', 'right'): False,
            ('right', 'straight'): False,
            ('straight', 'straight'): True,
            ('u_turn', 'left'): True,
            ('u_turn', 'right'): True,
            ('u_turn', 'straight'): True,
            ('u_turn', 'u_turn'): True,
        }
    
        key = tuple(sorted([turn_i, turn_j]))
        return conflict_matrix.get(key, True)

    def _perpendicular_approach_conflict(self, turn_i: str, turn_j: str, 
                                   approach_i: str, approach_j: str) -> bool:
        """Perpendicular approach vehicle conflict matrix"""
        # Determine relative position
        is_i_left_of_j = self._is_left_approach(approach_i, approach_j)
    
        if is_i_left_of_j:
            conflict_matrix = {
                ('left', 'left'): True,
                ('left', 'right'): False,
                ('left', 'straight'): True,
                ('right', 'left'): False,
                ('right', 'right'): False,
                ('right', 'straight'): False,
                ('straight', 'left'): True,
                ('straight', 'right'): False,
                ('straight', 'straight'): True,
                ('u_turn', 'left'): True,
                ('u_turn', 'right'): True,
                ('u_turn', 'straight'): True,
                ('u_turn', 'u_turn'): True,
            }
        else:
            conflict_matrix = {
                ('left', 'left'): True,
                ('left', 'right'): True,
                ('left', 'straight'): False,
                ('right', 'left'): True,
                ('right', 'right'): False,
                ('right', 'straight'): True,
                ('straight', 'left'): False,
                ('straight', 'right'): True,
                ('straight', 'straight'): True,
                ('u_turn', 'left'): True,
                ('u_turn', 'right'): True,
                ('u_turn', 'straight'): True,
                ('u_turn', 'u_turn'): True,
            }
    
        key = (turn_i, turn_j)
        return conflict_matrix.get(key, conflict_matrix.get((turn_j, turn_i), True))

    def _are_opposite_approaches(self, approach_i: str, approach_j: str) -> bool:
        """Check if approaches are opposite"""
        opposites = {
            ('north', 'south'), ('south', 'north'),
            ('east', 'west'), ('west', 'east')
        }
        return (approach_i, approach_j) in opposites

    def _are_perpendicular_approaches(self, approach_i: str, approach_j: str) -> bool:
        """Check if approaches are perpendicular"""
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
