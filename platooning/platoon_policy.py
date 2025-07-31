import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PlatoonMetrics:
    """Platoon performance metrics for monitoring and optimization"""
    avg_speed: float
    cohesion_score: float  # How well the platoon stays together
    efficiency_score: float  # How efficiently the platoon moves
    safety_score: float  # Safety based on following distances
    
class Platoon:
    """Enhanced platoon representation with better state management and auction integration"""
    
    def __init__(self, vehicle_list: List[Dict], intersection_center=(-188.9, -89.7, 0.0), 
                 goal_direction=None, state_extractor=None):
        self.vehicles = vehicle_list  # Vehicle state dictionaries
        self.leader = self.vehicles[0] if self.vehicles else None
        self.intersection_center = intersection_center
        self.state_extractor = state_extractor
        
        # Platoon identity and state
        self.platoon_id = f"platoon_{self.leader['id']}" if self.leader else None
        self.formation_time = time.time()
        self.last_update = time.time()
        
        # Initialize navigation cache BEFORE calling _get_navigation_direction()
        self._cached_direction = None
        self._direction_cache_time = 0
        
        # Navigation and direction (now safe to call _get_navigation_direction)
        self.goal_direction = self._get_navigation_direction() or goal_direction or 'straight'
        
        # Platoon dynamics
        self.target_spacing = 3.0  # Target spacing between vehicles (meters)
        self.max_spacing = 15.0    # Maximum allowed spacing
        self.min_spacing = 1.5     # Minimum safe spacing
        
        # Performance tracking
        self.metrics_history: List[PlatoonMetrics] = []
        self._last_metrics_update = 0
    
    def update_state(self, new_vehicle_states: List[Dict]):
        """Update platoon with new vehicle states"""
        if not new_vehicle_states:
            return False
        
        # Update vehicle states
        vehicle_id_to_state = {str(v['id']): v for v in new_vehicle_states}
        updated_vehicles = []
        
        for vehicle in self.vehicles:
            vehicle_id = str(vehicle['id'])
            if vehicle_id in vehicle_id_to_state:
                updated_vehicles.append(vehicle_id_to_state[vehicle_id])
        
        self.vehicles = updated_vehicles
        self.leader = self.vehicles[0] if self.vehicles else None
        self.last_update = time.time()
        
        # Update cached direction if needed
        if time.time() - self._direction_cache_time > 5.0:  # Refresh every 5 seconds
            self._cached_direction = self._get_navigation_direction()
            self._direction_cache_time = time.time()
        
        # Update metrics
        if time.time() - self._last_metrics_update > 2.0:  # Update every 2 seconds
            self._update_metrics()
            self._last_metrics_update = time.time()
        
        return len(self.vehicles) > 0
    
    def _get_navigation_direction(self) -> Optional[str]:
        """Get platoon direction from navigation system with caching"""
        if not self.leader or not self.state_extractor:
            return None
        
        # Use cached direction if available and recent
        if (self._cached_direction and 
            time.time() - self._direction_cache_time < 5.0):
            return self._cached_direction
        
        # Check if leader has destination
        if not self.leader.get('destination'):
            return None
        
        try:
            import carla
            vehicle_location = self.leader['location']
            carla_location = carla.Location(
                x=vehicle_location[0],
                y=vehicle_location[1], 
                z=vehicle_location[2]
            )
            
            direction = self.state_extractor.get_route_direction(
                carla_location, self.leader['destination']
            )
            
            # Cache the result
            self._cached_direction = direction
            self._direction_cache_time = time.time()
            
            return direction
            
        except Exception as e:
            print(f"[Warning] Platoon {self.platoon_id} navigation direction failed: {e}")
            return None
    
    def _update_metrics(self):
        """Update platoon performance metrics"""
        if len(self.vehicles) < 2:
            return
        
        # Calculate metrics
        avg_speed = self.get_average_speed()
        cohesion_score = self._calculate_cohesion_score()
        efficiency_score = self._calculate_efficiency_score()
        safety_score = self._calculate_safety_score()
        
        metrics = PlatoonMetrics(
            avg_speed=avg_speed,
            cohesion_score=cohesion_score,
            efficiency_score=efficiency_score,
            safety_score=safety_score
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 10 entries)
        if len(self.metrics_history) > 10:
            self.metrics_history = self.metrics_history[-10:]
    
    def _calculate_cohesion_score(self) -> float:
        """Calculate how well the platoon stays together (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        distances = []
        for i in range(len(self.vehicles) - 1):
            dist = self._calculate_vehicle_distance(self.vehicles[i], self.vehicles[i + 1])
            distances.append(dist)
        
        # Score based on how close distances are to target spacing
        score = 0.0
        for dist in distances:
            if dist <= self.max_spacing:
                # Perfect score if within target range
                if self.min_spacing <= dist <= self.target_spacing * 1.5:
                    score += 1.0
                else:
                    # Penalty for being too close or too far
                    score += max(0.0, 1.0 - abs(dist - self.target_spacing) / self.target_spacing)
        
        return score / len(distances) if distances else 0.0
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate platoon movement efficiency (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        # Check speed consistency
        speeds = [self._get_vehicle_speed(v) for v in self.vehicles]
        avg_speed = sum(speeds) / len(speeds)
        
        if avg_speed == 0:
            return 0.0
        
        # Calculate speed variance
        speed_variance = sum(abs(s - avg_speed) for s in speeds) / len(speeds)
        speed_consistency = max(0.0, 1.0 - speed_variance / avg_speed)
        
        # Check if platoon is moving towards intersection efficiently
        leader_distance = self._distance_to_intersection(self.leader)
        movement_efficiency = min(1.0, avg_speed / 10.0)  # Normalize by reasonable speed
        
        return (speed_consistency + movement_efficiency) / 2.0
    
    def _calculate_safety_score(self) -> float:
        """Calculate platoon safety score (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        safety_violations = 0
        total_pairs = len(self.vehicles) - 1
        
        for i in range(len(self.vehicles) - 1):
            dist = self._calculate_vehicle_distance(self.vehicles[i], self.vehicles[i + 1])
            if dist < self.min_spacing:
                safety_violations += 1
        
        return 1.0 - (safety_violations / total_pairs) if total_pairs > 0 else 1.0
    
    def _get_vehicle_speed(self, vehicle: Dict) -> float:
        """Get vehicle speed in m/s"""
        velocity = vehicle.get('velocity', [0, 0, 0])
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
    
    def _calculate_vehicle_distance(self, vehicle1: Dict, vehicle2: Dict) -> float:
        """Calculate distance between two vehicles"""
        loc1 = vehicle1['location']
        loc2 = vehicle2['location']
        return math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)
    
    def _distance_to_intersection(self, vehicle: Dict) -> float:
        """Calculate distance from vehicle to intersection center"""
        loc = vehicle['location']
        return math.sqrt(
            (loc[0] - self.intersection_center[0])**2 + 
            (loc[1] - self.intersection_center[1])**2
        )
    
    # Public interface methods
    def get_vehicle_ids(self) -> List[str]:
        """Get list of vehicle IDs in the platoon"""
        return [str(v['id']) for v in self.vehicles]

    def get_leader(self) -> Optional[Dict]:
        """Get platoon leader vehicle state"""
        return self.leader

    def get_goal_direction(self) -> str:
        """Get platoon's goal direction with real-time updates"""
        navigation_direction = self._get_navigation_direction()
        if navigation_direction:
            self.goal_direction = navigation_direction
        return self.goal_direction

    def is_valid(self) -> bool:
        """Check if platoon is valid and operational"""
        return (len(self.vehicles) > 0 and 
                self.leader is not None and 
                time.time() - self.last_update < 10.0)  # Consider stale if not updated in 10s
    
    def get_size(self) -> int:
        """Get number of vehicles in platoon"""
        return len(self.vehicles)
    
    def get_average_speed(self) -> float:
        """Get average speed of platoon in m/s"""
        if not self.vehicles:
            return 0.0
        
        total_speed = sum(self._get_vehicle_speed(v) for v in self.vehicles)
        return total_speed / len(self.vehicles)
    
    def get_leader_position(self) -> Optional[Tuple[float, float, float]]:
        """Get leader position as tuple"""
        return tuple(self.leader['location']) if self.leader else None
    
    def get_lane_info(self) -> Optional[Tuple[int, int]]:
        """Get road and lane ID of the platoon leader"""
        if self.leader:
            return (self.leader.get('road_id'), self.leader.get('lane_id'))
        return None
    
    def is_ready_for_intersection(self) -> bool:
        """Check if platoon is ready to proceed through intersection"""
        if not self.is_valid():
            return False
        
        # Check if platoon is cohesive enough
        if len(self.vehicles) > 1:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            if latest_metrics:
                return (latest_metrics.cohesion_score > 0.7 and 
                       latest_metrics.safety_score > 0.8)
        
        return True
    
    def get_platoon_bounds(self) -> Tuple[float, float]:
        """Get the front and rear bounds of the platoon (distance to intersection)"""
        if not self.vehicles:
            return (0.0, 0.0)
        
        distances = [self._distance_to_intersection(v) for v in self.vehicles]
        return (min(distances), max(distances))
    
    def has_vehicle_in_intersection(self) -> bool:
        """Check if any vehicle in the platoon is currently in the intersection"""
        return any(v.get('is_junction', False) for v in self.vehicles)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of platoon performance"""
        if not self.metrics_history:
            return {
                'avg_speed_kmh': self.get_average_speed() * 3.6,
                'cohesion': 'N/A',
                'efficiency': 'N/A',
                'safety': 'N/A',
                'ready_for_intersection': self.is_ready_for_intersection()
            }
        
        latest = self.metrics_history[-1]
        return {
            'avg_speed_kmh': latest.avg_speed * 3.6,
            'cohesion': f"{latest.cohesion_score:.2f}",
            'efficiency': f"{latest.efficiency_score:.2f}",
            'safety': f"{latest.safety_score:.2f}",
            'ready_for_intersection': self.is_ready_for_intersection()
        }
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"Platoon({self.platoon_id}, size={self.get_size()}, "
                f"direction={self.get_goal_direction()}, "
                f"ready={self.is_ready_for_intersection()})")
