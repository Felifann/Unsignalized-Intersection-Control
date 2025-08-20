import math
import time
from typing import List, Dict, Tuple
from collections import defaultdict

def _euclidean_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

class DeadlockException(Exception):
    """Exception raised when deadlock is detected"""
    def __init__(self, message: str, deadlock_type: str = "unknown", affected_vehicles: int = 0):
        super().__init__(message)
        self.deadlock_type = deadlock_type
        self.affected_vehicles = affected_vehicles

class IntersectionDeadlockDetector:
    """
    Part 3: Handles deadlock detection and prevention
    - Multiple deadlock detection modes
    - Core region traffic monitoring
    - Historical analysis for pattern detection
    """
    
    def __init__(self, solver_config):
        """Initialize with solver configuration"""
        self.center = solver_config['intersection_center']
        self.deadlock_core_half_size = solver_config.get('deadlock_core_half_size', 5.0)
        
        # Deadlock detection parameters
        self.deadlock_detection_window = solver_config.get('deadlock_detection_window', 35.0)  # seconds to track for deadlock
        self.deadlock_speed_threshold = solver_config.get('deadlock_speed_threshold', 0.5)  # m/s - vehicles below this are considered stopped
        self.deadlock_min_vehicles = solver_config.get('deadlock_min_vehicles', 6)  # minimum vehicles for deadlock detection
        self.deadlock_history = []  # track intersection state over time
        self.last_deadlock_check = -math.inf
        self.deadlock_check_interval = solver_config.get('deadlock_check_interval', 2.0)  # check every 2 seconds
        
        # Enhanced deadlock tracking
        self.deadlock_severity_threshold = solver_config.get('deadlock_severity_threshold', 0.8)  # 80% of vehicles stalled
        self.deadlock_duration_threshold = solver_config.get('deadlock_duration_threshold', 15.0)  # 15 seconds continuous stalling
        
        # Performance tracking
        self.stats = {
            'deadlocks_detected': 0,
            'false_positives': 0,
            'detection_time_avg': 0.0,
            'deadlock_types': defaultdict(int),
            'total_affected_vehicles': 0
        }

    def _speed_2d(self, velocity) -> float:
        """Consistent 2D speed calculation"""
        if not velocity:
            return 0.0
        try:
            return math.hypot(float(velocity[0]), float(velocity[1]))
        except Exception:
            # defensive fallback
            return float(sum((v*v for v in velocity[:2]))**0.5)

    def detect_deadlock(self, vehicle_states: Dict[str, Dict], current_time: float) -> bool:
        """Enhanced deadlock detection with multiple detection modes"""
        # Only check periodically to avoid excessive computation
        if current_time - self.last_deadlock_check < self.deadlock_check_interval:
            return False
        
        self.last_deadlock_check = current_time
        
        # Get vehicles in core region
        core_vehicles = self._get_core_region_vehicles(vehicle_states)
        
        # Always record snapshot using consistent speed calculation
        snapshot = {
            'timestamp': current_time,
            'core_vehicles': {v['id']: {
                'location': v.get('location', (0, 0, 0)),
                'velocity': v.get('velocity', [0, 0, 0]),
                'speed': self._speed_2d(v.get('velocity', [0,0,0])),
                'stalled': self._speed_2d(v.get('velocity', [0,0,0])) < self.deadlock_speed_threshold
            } for v in core_vehicles},
            'stalled_count': sum(1 for v in core_vehicles 
                               if self._speed_2d(v.get('velocity', [0,0,0])) < self.deadlock_speed_threshold)
        }
        
        # Add to history
        self.deadlock_history.append(snapshot)
        
        # Keep only recent history
        cutoff_time = current_time - self.deadlock_detection_window
        self.deadlock_history = [s for s in self.deadlock_history if s['timestamp'] >= cutoff_time]
        
        # If not enough vehicles now, skip heavy checks early
        if len(core_vehicles) < self.deadlock_min_vehicles:
            return False
        
        # Need sufficient history for detection
        min_snapshots_for_persistent = max(5, int(self.deadlock_duration_threshold / self.deadlock_check_interval))
        if len(self.deadlock_history) < min_snapshots_for_persistent:
            return False
        
        # Mode 1: Persistent core stalling
        if self._detect_persistent_core_stalling(min_snapshots_for_persistent):
            self._handle_deadlock_detected("Persistent Core Stalling", len(core_vehicles))
            return True
        
        # Mode 2: Circular waiting pattern
        if self._detect_circular_waiting():
            self._handle_deadlock_detected("Circular Waiting", len(core_vehicles))
            return True
        
        # Mode 3: No progress detection
        if self._detect_no_progress():
            self._handle_deadlock_detected("No Progress", len(core_vehicles))
            return True
        
        return False

    def _handle_deadlock_detected(self, deadlock_type: str, affected_vehicles: int):
        """Handle deadlock detection with enhanced tracking"""
        self.stats['deadlocks_detected'] += 1
        self.stats['deadlock_types'][deadlock_type] += 1
        self.stats['total_affected_vehicles'] += affected_vehicles
        
        print(f"\n🚨 DEADLOCK DETECTED - {deadlock_type}")
        print(f"   📍 Location: Core intersection region")
        print(f"   🕐 Duration: {self.deadlock_detection_window}s+ of stalling")
        print(f"   🚗 Vehicles: {affected_vehicles} vehicles affected")
        
        # Raise enhanced exception with details
        raise DeadlockException(
            f"Deadlock detected in intersection core region: {deadlock_type}",
            deadlock_type=deadlock_type,
            affected_vehicles=affected_vehicles
        )

    def handle_deadlock_detection(self):
        """Legacy method for backward compatibility"""
        self.stats['deadlocks_detected'] += 1
        raise DeadlockException("Deadlock detected in intersection core region")

    def _get_core_region_vehicles(self, vehicle_states: Dict[str, Dict]) -> List[Dict]:
        """Get vehicles specifically in the core blue square region"""
        core_vehicles = []
        
        for vehicle_id, vehicle_state in vehicle_states.items():
            if not vehicle_state or 'location' not in vehicle_state:
                continue
            
            location = vehicle_state['location']
            
            # Use EXACT SQUARE bounds
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

    def _detect_persistent_core_stalling(self, required_snapshots: int = 10) -> bool:
        """Detect if the same set of vehicles have been stalled in core for extended time"""
        required = required_snapshots or int(self.deadlock_duration_threshold / self.deadlock_check_interval)
        if len(self.deadlock_history) < required:
            return False
        
        # Take the most recent block
        recent_snapshots = self.deadlock_history[-required:]
        
        # Count snapshots where stalled vehicle ratio is above threshold
        high_stall_count = sum(1 for s in recent_snapshots 
                              if s.get('stalled_count', 0) >= self.deadlock_min_vehicles and 
                              (s.get('stalled_count', 0) / max(len(s.get('core_vehicles', {})), 1)) >= self.deadlock_severity_threshold)
        
        # If most recent snapshots show high stalling, it's likely deadlock
        return high_stall_count >= int(0.8 * required)  # 80% of recent snapshots

    def _detect_circular_waiting(self) -> bool:
        """Detect circular waiting patterns where vehicles block each other"""
        if len(self.deadlock_history) < 3:
            return False
        
        current_snapshot = self.deadlock_history[-1]
        core_vehicles = current_snapshot['core_vehicles']
        
        # Simple heuristic: if most vehicles in core are stalled and positioned 
        # in different quadrants, likely circular waiting
        stalled_vehicles = [v_id for v_id, data in core_vehicles.items() if data.get('stalled')]
        
        if len(stalled_vehicles) < 4:  # Need at least 4 vehicles for circular pattern
            return False
        
        # Check if vehicles are distributed across different approaches
        quadrant_count = self._count_vehicles_by_quadrant(stalled_vehicles, core_vehicles)
        
        # If vehicles are in 3+ quadrants and a sufficient number are stalled, likely circular waiting
        return len(quadrant_count) >= 3 and len(stalled_vehicles) >= self.deadlock_min_vehicles

    def _detect_no_progress(self) -> bool:
        """Detect lack of progress toward intersection center"""
        lookback_seconds = 15.0
        if not self.deadlock_history:
            return False
        
        # Find oldest snapshot within lookback_seconds
        current_snapshot = self.deadlock_history[-1]
        current_time = current_snapshot['timestamp']
        target_time = current_time - lookback_seconds
        old_snapshot = None
        for s in reversed(self.deadlock_history):
            if s['timestamp'] <= target_time:
                old_snapshot = s
                break
        
        # Fallback to earliest available if exact not found
        if old_snapshot is None:
            if len(self.deadlock_history) < 2:
                return False
            old_snapshot = self.deadlock_history[0]
        
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
            old_pos = old_vehicles[v_id].get('location', (0,0,0))
            current_pos = current_vehicles[v_id].get('location', (0,0,0))
            
            old_dist = _euclidean_2d(old_pos, center)
            current_dist = _euclidean_2d(current_pos, center)
            
            # No significant progress if distance to center hasn't decreased much
            if current_dist >= old_dist - 1.0:  # Less than 1 meter progress
                no_progress_count += 1
        
        # If most tracked vehicles made no progress, likely deadlock
        return no_progress_count >= max(1, int(len(common_vehicles) * 0.8))

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

    def get_stats(self) -> Dict:
        """Get enhanced deadlock detection statistics"""
        stats = self.stats.copy()
        
        # Calculate deadlock severity metrics
        if stats['deadlocks_detected'] > 0:
            stats['avg_affected_vehicles'] = stats['total_affected_vehicles'] / stats['deadlocks_detected']
            stats['deadlock_rate'] = stats['deadlocks_detected']  # Per session
        else:
            stats['avg_affected_vehicles'] = 0.0
            stats['deadlock_rate'] = 0.0
        
        return stats

    def reset_history(self):
        """Reset deadlock detection history"""
        self.deadlock_history = []
        print("🔄 Deadlock detector: History reset")

    def get_deadlock_severity(self) -> float:
        """Calculate current deadlock severity (0-1)"""
        if not self.deadlock_history:
            return 0.0
        
        recent_snapshots = self.deadlock_history[-5:]  # Last 5 snapshots
        if not recent_snapshots:
            return 0.0
        
        # Calculate average stall ratio
        stall_ratios = []
        for snapshot in recent_snapshots:
            core_vehicles_count = len(snapshot.get('core_vehicles', {}))
            stalled_count = snapshot.get('stalled_count', 0)
            if core_vehicles_count > 0:
                stall_ratios.append(stalled_count / core_vehicles_count)
        
        if not stall_ratios:
            return 0.0
        
        avg_stall_ratio = sum(stall_ratios) / len(stall_ratios)
        return min(1.0, avg_stall_ratio)
