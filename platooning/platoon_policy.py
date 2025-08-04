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
    """
    Platoon representation with modular design for future integration.
    
    This class manages vehicle groupings and their collective behavior without 
    tight coupling to external systems. Integration points are clearly defined
    through optional parameters and callback interfaces.
    """
    
    def __init__(self, vehicle_list: List[Dict], 
                 intersection_center: Tuple[float, float, float] = (-188.9, -89.7, 0.0),
                 goal_direction: Optional[str] = None, 
                 state_extractor=None):
        """
        Initialize platoon with minimal dependencies.
        
        Args:
            vehicle_list: List of vehicle state dictionaries
            intersection_center: Intersection coordinates for distance calculations
            goal_direction: Optional pre-determined direction
            state_extractor: Optional navigation system interface
        """
        # Core platoon data
        self.vehicles = vehicle_list if vehicle_list else []
        self.leader = self.vehicles[0] if self.vehicles else None
        self.intersection_center = intersection_center
        
        # Platoon identity
        self.platoon_id = f"platoon_{self.leader['id']}" if self.leader else f"platoon_empty_{int(time.time())}"
        self.formation_time = time.time()
        self.last_update = time.time()
        
        # Navigation system interface (optional)
        self._state_extractor = state_extractor
        self._cached_direction = None
        self._direction_cache_time = 0
        self._direction_cache_duration = 5.0  # Cache for 5 seconds
        
        # Platoon configuration
        self.target_spacing = 3.0
        self.max_spacing = 15.0
        self.min_spacing = 1.5
        
        # Determine initial direction
        self.goal_direction = self._determine_initial_direction(goal_direction)
        
        # Performance tracking
        self.metrics_history: List[PlatoonMetrics] = []
        self._last_metrics_update = 0
        self._metrics_update_interval = 2.0
    
    def _determine_initial_direction(self, provided_direction: Optional[str]) -> str:
        """Determine initial platoon direction from various sources"""
        if provided_direction:
            return provided_direction
        
        # Try to get direction from navigation system
        nav_direction = self._query_navigation_direction()
        if nav_direction:
            return nav_direction
        
        # Default fallback
        return 'straight'
    
    def update_vehicles(self, new_vehicle_states: List[Dict]) -> bool:
        """
        Update platoon with new vehicle states.
        
        Returns:
            bool: True if platoon remains valid after update
        """
        if not new_vehicle_states:
            return False
        
        # Update vehicle states by ID matching
        vehicle_lookup = {str(v['id']): v for v in new_vehicle_states}
        updated_vehicles = []
        
        for vehicle in self.vehicles:
            vehicle_id = str(vehicle['id'])
            if vehicle_id in vehicle_lookup:
                updated_vehicles.append(vehicle_lookup[vehicle_id])
        
        # Apply updates
        self.vehicles = updated_vehicles
        self.leader = self.vehicles[0] if self.vehicles else None
        self.last_update = time.time()
        
        # Update navigation direction if needed
        self._refresh_navigation_cache()
        
        # Update performance metrics
        self._update_metrics_if_needed()
        
        return self.is_valid()
    
    def _refresh_navigation_cache(self):
        """Refresh navigation direction cache if needed"""
        current_time = time.time()
        if current_time - self._direction_cache_time > self._direction_cache_duration:
            new_direction = self._query_navigation_direction()
            if new_direction:
                self._cached_direction = new_direction
                self.goal_direction = new_direction
            self._direction_cache_time = current_time
    
    def _query_navigation_direction(self) -> Optional[str]:
        """Query navigation system for direction (safe with optional dependency)"""
        if not self.leader or not self._state_extractor:
            return None
        
        if not self.leader.get('destination'):
            return None
        
        try:
            import carla
            vehicle_location = carla.Location(
                x=self.leader['location'][0],
                y=self.leader['location'][1], 
                z=self.leader['location'][2]
            )
            
            direction = self._state_extractor.get_route_direction(
                vehicle_location, self.leader['destination']
            )
            return direction
            
        except Exception:
            # Silently fail if navigation unavailable
            return None
    
    def _update_metrics_if_needed(self):
        """Update performance metrics if enough time has passed"""
        current_time = time.time()
        if current_time - self._last_metrics_update > self._metrics_update_interval:
            self._compute_and_store_metrics()
            self._last_metrics_update = current_time
    
    def _compute_and_store_metrics(self):
        """Compute and store current performance metrics"""
        if len(self.vehicles) < 2:
            return
        
        metrics = PlatoonMetrics(
            avg_speed=self._calculate_average_speed(),
            cohesion_score=self._calculate_cohesion_score(),
            efficiency_score=self._calculate_efficiency_score(),
            safety_score=self._calculate_safety_score()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 10:
            self.metrics_history = self.metrics_history[-10:]
    
    def _calculate_cohesion_score(self) -> float:
        """Calculate platoon cohesion (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        distances = []
        for i in range(len(self.vehicles) - 1):
            dist = self._vehicle_distance(self.vehicles[i], self.vehicles[i + 1])
            distances.append(dist)
        
        score = 0.0
        for dist in distances:
            if dist <= self.max_spacing:
                if self.min_spacing <= dist <= self.target_spacing * 1.5:
                    score += 1.0
                else:
                    score += max(0.0, 1.0 - abs(dist - self.target_spacing) / self.target_spacing)
        
        return score / len(distances) if distances else 0.0
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate movement efficiency (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        speeds = [self._vehicle_speed(v) for v in self.vehicles]
        avg_speed = sum(speeds) / len(speeds)
        
        if avg_speed == 0:
            return 0.0
        
        # Speed consistency
        speed_variance = sum(abs(s - avg_speed) for s in speeds) / len(speeds)
        speed_consistency = max(0.0, 1.0 - speed_variance / avg_speed)
        
        # Movement progress
        movement_efficiency = min(1.0, avg_speed / 10.0)
        
        return (speed_consistency + movement_efficiency) / 2.0
    
    def _calculate_safety_score(self) -> float:
        """Calculate safety score (0-1, higher is better)"""
        if len(self.vehicles) < 2:
            return 1.0
        
        safety_violations = 0
        total_pairs = len(self.vehicles) - 1
        
        for i in range(len(self.vehicles) - 1):
            dist = self._vehicle_distance(self.vehicles[i], self.vehicles[i + 1])
            if dist < self.min_spacing:
                safety_violations += 1
        
        return 1.0 - (safety_violations / total_pairs) if total_pairs > 0 else 1.0
    
    # Helper methods for calculations
    def _vehicle_speed(self, vehicle: Dict) -> float:
        """Get vehicle speed in m/s"""
        velocity = vehicle.get('velocity', [0, 0, 0])
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
    
    def _vehicle_distance(self, vehicle1: Dict, vehicle2: Dict) -> float:
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
    
    def _calculate_average_speed(self) -> float:
        """Calculate average platoon speed"""
        if not self.vehicles:
            return 0.0
        return sum(self._vehicle_speed(v) for v in self.vehicles) / len(self.vehicles)
    
    # Public interface - core functionality
    def get_vehicle_ids(self) -> List[str]:
        """Get list of vehicle IDs in the platoon"""
        return [str(v['id']) for v in self.vehicles]

    def get_leader(self) -> Optional[Dict]:
        """Get platoon leader vehicle state"""
        return self.leader

    def get_goal_direction(self) -> str:
        """Get platoon's goal direction"""
        # Try to get updated direction from navigation
        current_direction = self._query_navigation_direction()
        if current_direction:
            self.goal_direction = current_direction
        return self.goal_direction

    def is_valid(self) -> bool:
        """Check if platoon is valid and operational"""
        return (len(self.vehicles) > 0 and 
                self.leader is not None and 
                time.time() - self.last_update < 10.0)
    
    def get_size(self) -> int:
        """Get number of vehicles in platoon"""
        return len(self.vehicles)
    
    def get_leader_position(self) -> Optional[Tuple[float, float, float]]:
        """Get leader position as tuple"""
        return tuple(self.leader['location']) if self.leader else None
    
    def get_lane_info(self) -> Optional[Tuple[int, int]]:
        """Get road and lane ID of the platoon leader"""
        if self.leader:
            return (self.leader.get('road_id'), self.leader.get('lane_id'))
        return None
    
    # Status and performance queries
    def is_ready_for_intersection(self) -> bool:
        """Check if platoon is ready to proceed through intersection"""
        if not self.is_valid():
            return False
        
        if len(self.vehicles) > 1 and self.metrics_history:
            latest_metrics = self.metrics_history[-1]
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
                'avg_speed_kmh': self._calculate_average_speed() * 3.6,
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
