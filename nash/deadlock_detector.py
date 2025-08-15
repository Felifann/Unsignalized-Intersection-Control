import math
import time
from typing import List, Dict, Tuple
from collections import defaultdict

def _euclidean_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

class DeadlockException(Exception):
    """Exception raised when deadlock is detected"""
    pass

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
        self.deadlock_detection_window = 35.0  # seconds to track for deadlock
        self.deadlock_speed_threshold = 0.5  # m/s - vehicles below this are considered stopped
        self.deadlock_min_vehicles = 6  # minimum vehicles for deadlock detection
        self.deadlock_history = []  # track intersection state over time
        self.last_deadlock_check = 0
        self.deadlock_check_interval = 2.0  # check every 2 seconds
        
        # Performance tracking
        self.stats = {
            'deadlocks_detected': 0,
            'false_positives': 0,
            'detection_time_avg': 0.0
        }

    def detect_deadlock(self, vehicle_states: Dict[str, Dict], current_time: float) -> bool:
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
            print(f"\n🚨 DEADLOCK DETECTED - No Progress detection")
            print(f"   ⏳ Duration: {self.deadlock_detection_window}s+ of no progress")
            print(f"   🚗 Vehicles: {len(core_vehicles)} vehicles affected")
            return True
        
        return False

    def handle_deadlock_detection(self):
        """Handle deadlock detection by updating stats and raising exception"""
        self.stats['deadlocks_detected'] += 1
        raise DeadlockException("Deadlock detected in intersection core region")

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

    def get_stats(self) -> Dict:
        """Get deadlock detection statistics"""
        return self.stats.copy()

    def reset_history(self):
        """Reset deadlock detection history"""
        self.deadlock_history = []
        print("🔄 Deadlock detector: History reset")
