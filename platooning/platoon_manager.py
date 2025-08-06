import math
import time
from typing import Dict, List, Set, Optional, Tuple, Callable
from collections import defaultdict

from .platoon_policy import Platoon

class PlatoonConfiguration:
    """Configuration container for platoon parameters"""
    def __init__(self):
        self.max_platoon_size = 4
        self.min_platoon_size = 2
        self.max_following_distance = 25.0
        self.target_following_distance = 8.0
        self.update_interval = 1.0
        self.intersection_center = (-188.9, -89.7, 0.0)

class PlatoonManager:
    """
    Enhanced platoon manager with modular design and loose coupling.
    
    This manager handles platoon formation, maintenance, and dissolution
    without tight dependencies on external systems. Integration points
    are provided through callbacks and optional interfaces.
    """
    
    def __init__(self, state_extractor=None, config: Optional[PlatoonConfiguration] = None):
        """
        Initialize platoon manager with optional dependencies.
        
        Args:
            state_extractor: Optional interface for vehicle state queries
            config: Optional configuration object
        """
        # Configuration
        self.config = config or PlatoonConfiguration()
        
        # External interfaces (optional)
        self._state_extractor = state_extractor
        self._vehicle_filter_callback: Optional[Callable] = None
        self._direction_estimator_callback: Optional[Callable] = None
        
        # Core state
        self.platoons: List[Platoon] = []
        self.platoon_history: Dict[str, Platoon] = {}
        self.last_update_time = 0
        
        # Statistics
        self.formation_stats = {
            'total_formed': 0,
            'total_dissolved': 0,
            'successful_crossings': 0
        }
        
        print("🚗 Modular Platoon Manager initialized")
    
    def set_vehicle_filter(self, filter_callback: Callable[[List[Dict]], List[Dict]]):
        """Set callback for filtering vehicles eligible for platooning"""
        self._vehicle_filter_callback = filter_callback
    
    # def set_direction_estimator(self, estimator_callback: Callable[[Dict], Optional[str]]):
    #     """Set callback for estimating vehicle direction"""
    #     self._direction_estimator_callback = estimator_callback
    
    def update(self, vehicle_states: Optional[List[Dict]] = None):
        """
        Main update method with rate limiting and optional vehicle states.
        
        Args:
            vehicle_states: Optional vehicle states. If None, will query state_extractor
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.config.update_interval:
            return
        
        # Get vehicle states
        if vehicle_states is None:
            vehicle_states = self._get_vehicle_states()
        
        if not vehicle_states:
            return
        
        # Filter eligible vehicles
        eligible_vehicles = self._filter_eligible_vehicles(vehicle_states)
        
        # Update existing platoons
        self._update_existing_platoons(eligible_vehicles)
        
        # Form new platoons
        self._attempt_platoon_formation(eligible_vehicles)
        
        # Clean up invalid platoons
        self._cleanup_invalid_platoons()
        
        self.last_update_time = current_time
    
    def _get_vehicle_states(self) -> List[Dict]:
        """Get vehicle states from state extractor or return empty list"""
        if self._state_extractor:
            try:
                return self._state_extractor.get_vehicle_states()
            except Exception:
                return []
        return []
    
    def _filter_eligible_vehicles(self, vehicle_states: List[Dict]) -> List[Dict]:
        """Filter vehicles eligible for platooning"""
        if self._vehicle_filter_callback:
            return self._vehicle_filter_callback(vehicle_states)
        
        # Default filtering: vehicles near intersection with destinations
        return [v for v in vehicle_states if self._default_vehicle_filter(v)]
    
    def _default_vehicle_filter(self, vehicle: Dict) -> bool:
        """Default vehicle filtering logic - 更宽松的过滤条件"""
        # Check if near intersection (simple distance check)
        location = vehicle.get('location', [0, 0, 0])
        distance = math.sqrt(
            (location[0] - self.config.intersection_center[0])**2 + 
            (location[1] - self.config.intersection_center[1])**2
        )
        
        # More relaxed filtering for easier platoon formation
        near_intersection = distance < 120.0  # 增加距离阈值
        has_destination_or_moving = (vehicle.get('destination') or 
                                   vehicle.get('is_junction', False) or
                                   self._vehicle_speed(vehicle) > 0.5)  # 包含移动中的车辆
        
        return near_intersection and has_destination_or_moving
    
    def _vehicle_speed(self, vehicle: Dict) -> float:
        """Helper to calculate vehicle speed"""
        velocity = vehicle.get('velocity', [0, 0, 0])
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
    
    def _update_existing_platoons(self, vehicle_states: List[Dict]):
        """Update existing platoons with new vehicle states"""
        vehicle_lookup = {str(v['id']): v for v in vehicle_states}
        
        for platoon in self.platoons[:]:
            # Get updated states for platoon vehicles
            updated_vehicles = []
            for vehicle_id in platoon.get_vehicle_ids():
                if vehicle_id in vehicle_lookup:
                    updated_vehicles.append(vehicle_lookup[vehicle_id])
            
            # Update platoon
            if updated_vehicles:
                if not platoon.update_vehicles(updated_vehicles):
                    self._dissolve_platoon(platoon, "Failed to update")
            else:
                self._dissolve_platoon(platoon, "No vehicles found")
    
    def _attempt_platoon_formation(self, vehicle_states: List[Dict]):
        """Attempt to form new platoons from available vehicles"""
        # Get vehicles not already in platoons
        available_vehicles = self._get_available_vehicles(vehicle_states)
        
        if len(available_vehicles) < self.config.min_platoon_size:
            return
        
        # Group vehicles by compatibility
        compatible_groups = self._group_compatible_vehicles(available_vehicles)
        
        # Form platoons from groups
        for group in compatible_groups:
            if len(group) >= self.config.min_platoon_size:
                new_platoons = self._create_platoons_from_group(group)
                self.platoons.extend(new_platoons)
    
    def _get_available_vehicles(self, vehicle_states: List[Dict]) -> List[Dict]:
        """Get vehicles not already in platoons"""
        existing_vehicle_ids = set()
        for platoon in self.platoons:
            existing_vehicle_ids.update(platoon.get_vehicle_ids())
        
        return [v for v in vehicle_states 
                if str(v['id']) not in existing_vehicle_ids and 
                self._can_join_platoon(v)]
    
    def _can_join_platoon(self, vehicle: Dict) -> bool:
        """Check if vehicle can participate in platoon formation"""
        return (vehicle.get('destination') or 
                vehicle.get('is_junction', False))
    
    def _group_compatible_vehicles(self, vehicles: List[Dict]) -> List[List[Dict]]:
        """Group vehicles by compatibility (lane and direction)"""
        # Group by lane
        lane_groups = defaultdict(list)
        
        for vehicle in vehicles:
            lane_id = self._get_vehicle_lane_id(vehicle)
            direction = self._estimate_vehicle_direction(vehicle)
            
            if direction:
                lane_groups[lane_id].append((vehicle, direction))
        
        # Process each lane group
        compatible_groups = []
        for lane_vehicles in lane_groups.values():
            if len(lane_vehicles) < self.config.min_platoon_size:
                continue
            
            # Sort by distance to intersection
            sorted_vehicles = sorted(
                lane_vehicles,
                key=lambda x: self._distance_to_intersection(x[0])
            )
            
            # Find adjacent groups with same direction
            groups = self._find_adjacent_compatible_groups(sorted_vehicles)
            compatible_groups.extend(groups)
        
        return compatible_groups
    
    def _get_vehicle_lane_id(self, vehicle: Dict) -> str:
        """Get lane identifier for vehicle"""
        road_id = vehicle.get('road_id', 'unknown')
        lane_id = vehicle.get('lane_id', 'unknown')
        return f"{road_id}_{lane_id}"
    
    def _estimate_vehicle_direction(self, vehicle: Dict) -> Optional[str]:
        # if self._direction_estimator_callback:
        #     direction = self._direction_estimator_callback(vehicle)
        #     if direction:
        #         return direction

        # Try to use destination if available
        if vehicle.get('destination') and self._state_extractor:
            try:
                import carla
                vehicle_location = carla.Location(
                    x=vehicle['location'][0],
                    y=vehicle['location'][1], 
                    z=vehicle['location'][2]
                )
                direction = self._state_extractor.get_route_direction(
                    vehicle_location, vehicle['destination']
                )
                if direction:
                    return direction
            except Exception as e:
                print(f"[Direction] Failed to get route direction for vehicle {vehicle['id']}: {e}")

        # Fallback: Only use velocity if it's significant, and try to infer direction
        velocity = vehicle.get('velocity', [0, 0, 0])
        if len(velocity) >= 2 and (abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1):
            # You may implement a more sophisticated heading-to-direction mapping here
            return None  # Do not guess, skip if not sure

        print(f"[Direction] No clear direction for vehicle {vehicle['id']}")
        return None
    
    def _find_adjacent_compatible_groups(self, sorted_vehicles_with_direction: List[Tuple]) -> List[List[Dict]]:
        """Find adjacent vehicle groups with same direction - Enhanced validation"""
        if not sorted_vehicles_with_direction:
            return []
        
        groups = []
        current_group = [sorted_vehicles_with_direction[0][0]]
        current_direction = sorted_vehicles_with_direction[0][1]
        
        for i in range(1, len(sorted_vehicles_with_direction)):
            vehicle, direction = sorted_vehicles_with_direction[i]
            prev_vehicle = sorted_vehicles_with_direction[i-1][0]
            
            # ENHANCED: Strict direction matching
            if direction != current_direction or direction is None:
                # Direction changed or invalid - finalize current group
                if len(current_group) >= self.config.min_platoon_size:
                    print(f"🔍 Found compatible group: {len(current_group)} vehicles, direction={current_direction}")
                    groups.append(current_group)
                
                # Start new group only if direction is valid
                if direction is not None:
                    current_group = [vehicle]
                    current_direction = direction
                else:
                    current_group = []
                    current_direction = None
                continue
            
            # Check proximity for same direction vehicles
            distance = self._vehicle_distance(prev_vehicle, vehicle)
            
            if distance <= self.config.max_following_distance:
                current_group.append(vehicle)
                print(f"   Added vehicle {vehicle['id']} to group (distance: {distance:.1f}m)")
            else:
                # Distance too large - finalize current group and start new one
                if len(current_group) >= self.config.min_platoon_size:
                    print(f"🔍 Found compatible group: {len(current_group)} vehicles, direction={current_direction}")
                    groups.append(current_group)
                current_group = [vehicle]
        
        # Add final group if valid
        if len(current_group) >= self.config.min_platoon_size and current_direction is not None:
            print(f"🔍 Found final compatible group: {len(current_group)} vehicles, direction={current_direction}")
            groups.append(current_group)
        
        return groups
    
    def _create_platoons_from_group(self, vehicle_group: List[Dict]) -> List[Platoon]:
        """Create platoons from a vehicle group - Enhanced with stricter validation"""
        if len(vehicle_group) < self.config.min_platoon_size:
            print(f"❌ Group too small: {len(vehicle_group)} vehicles (need {self.config.min_platoon_size})")
            return []
        
        platoons = []
        
        # IMPROVED: Sort vehicles by distance to intersection for proper leader selection
        sorted_vehicles = sorted(
            vehicle_group,
            key=lambda v: self._distance_to_intersection(v)
        )
        
        # Split large groups into multiple platoons
        while len(sorted_vehicles) >= self.config.min_platoon_size:
            platoon_size = min(self.config.max_platoon_size, len(sorted_vehicles))
            platoon_vehicles = sorted_vehicles[:platoon_size]
            sorted_vehicles = sorted_vehicles[platoon_size:]
            
            # ENHANCED: Stricter direction validation
            directions = []
            for v in platoon_vehicles:
                direction = self._estimate_vehicle_direction(v)
                if direction:
                    directions.append(direction)
            
            # Ensure ALL vehicles have the same direction
            unique_directions = set(directions)
            
            if len(directions) == len(platoon_vehicles) and len(unique_directions) == 1:
                # All vehicles have same valid direction
                common_direction = list(unique_directions)[0]
                
                # IMPROVED: Additional formation validation
                if self._validate_platoon_formation(platoon_vehicles):
                    platoon = Platoon(
                        platoon_vehicles, 
                        self.config.intersection_center, 
                        goal_direction=common_direction,
                        state_extractor=self._state_extractor
                    )
                    
                    if platoon.is_valid() and platoon.get_size() >= self.config.min_platoon_size:
                        platoons.append(platoon)
                        self.formation_stats['total_formed'] += 1
                        
                        # DEBUG: Enhanced logging
                        leader_id = platoon.get_leader_id()
                        follower_ids = platoon.get_follower_ids()
                        print(f"✅ Formed valid platoon: {platoon.platoon_id}")
                        print(f"   Size: {platoon.get_size()} vehicles")
                        print(f"   Direction: {common_direction}")
                        print(f"   Leader: {leader_id}")
                        print(f"   Followers: {follower_ids}")
                    else:
                        print(f"❌ Failed validation: size={platoon.get_size()}, valid={platoon.is_valid()}")
                else:
                    print(f"❌ Formation validation failed for group")
            else:
                print(f"❌ Direction mismatch: {len(directions)}/{len(platoon_vehicles)} have directions, unique={unique_directions}")
                break  # Stop trying to form platoons from this group
        
        return platoons
    
    def _validate_platoon_formation(self, vehicles: List[Dict]) -> bool:
        """Validate that vehicles can form a proper platoon"""
        if len(vehicles) < 2:
            return False
        
        # Check that vehicles are reasonably spaced
        for i in range(len(vehicles) - 1):
            distance = self._vehicle_distance(vehicles[i], vehicles[i + 1])
            if distance > self.config.max_following_distance:
                print(f"❌ Vehicles too far apart: {distance:.1f}m > {self.config.max_following_distance}m")
                return False
            if distance < 2.0:  # Too close
                print(f"❌ Vehicles too close: {distance:.1f}m < 2.0m")
                return False
        
        return True
    
    def _dissolve_platoon(self, platoon: Platoon, reason: str):
        """Dissolve a platoon and update statistics"""
        if platoon in self.platoons:
            self.platoons.remove(platoon)
            self.platoon_history[platoon.platoon_id] = platoon
            self.formation_stats['total_dissolved'] += 1
            print(f"❌ Dissolved platoon {platoon.platoon_id}: {reason}")
    
    def _cleanup_invalid_platoons(self):
        """Remove invalid or expired platoons"""
        for platoon in self.platoons[:]:
            if not platoon.is_valid():
                self._dissolve_platoon(platoon, "Invalid state")
    
    def _distance_to_intersection(self, vehicle: Dict) -> float:
        """Calculate distance from vehicle to intersection center"""
        location = vehicle.get('location', [0, 0, 0])
        return math.sqrt(
            (location[0] - self.config.intersection_center[0])**2 + 
            (location[1] - self.config.intersection_center[1])**2
        )
    
    def _vehicle_distance(self, vehicle1: Dict, vehicle2: Dict) -> float:
        """Calculate distance between two vehicles"""
        loc1 = vehicle1.get('location', [0, 0, 0])
        loc2 = vehicle2.get('location', [0, 0, 0])
        return math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)
    
    # Public interface for external integration
    def get_all_platoons(self) -> List[Platoon]:
        """Get all valid platoons"""
        return [p for p in self.platoons if p.is_valid()]

    def get_platoon_by_leader_id(self, leader_id: str) -> Optional[Platoon]:
        """Get platoon by leader vehicle ID"""
        for platoon in self.platoons:
            if platoon.leader and str(platoon.leader['id']) == str(leader_id):
                return platoon
        return None

    def get_platoons_by_direction(self, direction: str) -> List[Platoon]:
        """Get platoons heading in specific direction"""
        return [p for p in self.platoons if p.get_goal_direction() == direction]

    def get_platoon_stats(self) -> Dict:
        """Get comprehensive platoon statistics"""
        if not self.platoons:
            return {
                'num_platoons': 0,
                'vehicles_in_platoons': 0,
                'avg_platoon_size': 0.0,
                'direction_distribution': {},
                'performance_summary': self._empty_performance_summary(),
                'formation_stats': self.formation_stats
            }
        
        total_vehicles = sum(p.get_size() for p in self.platoons)
        avg_size = total_vehicles / len(self.platoons)
        
        # Direction distribution
        direction_dist = defaultdict(int)
        performance_summary = self._calculate_performance_summary()
        
        for platoon in self.platoons:
            direction = platoon.get_goal_direction()
            direction_dist[direction] += 1
        
        return {
            'num_platoons': len(self.platoons),
            'vehicles_in_platoons': total_vehicles,
            'avg_platoon_size': avg_size,
            'direction_distribution': dict(direction_dist),
            'performance_summary': performance_summary,
            'formation_stats': self.formation_stats
        }
    
    def _empty_performance_summary(self) -> Dict:
        """Return empty performance summary"""
        return {
            'avg_cohesion': 0.0,
            'avg_efficiency': 0.0,
            'avg_safety': 0.0,
            'ready_platoons': 0
        }
    
    def _calculate_performance_summary(self) -> Dict:
        """Calculate performance summary from current platoons"""
        if not self.platoons:
            return self._empty_performance_summary()
        
        total_cohesion = 0.0
        total_efficiency = 0.0
        total_safety = 0.0
        ready_count = 0
        
        for platoon in self.platoons:
            if platoon.is_ready_for_intersection():
                ready_count += 1
            
            perf = platoon.get_performance_summary()
            if perf['cohesion'] != 'N/A':
                total_cohesion += float(perf['cohesion'])
            if perf['efficiency'] != 'N/A':
                total_efficiency += float(perf['efficiency'])
            if perf['safety'] != 'N/A':
                total_safety += float(perf['safety'])
        
        return {
            'avg_cohesion': total_cohesion / len(self.platoons),
            'avg_efficiency': total_efficiency / len(self.platoons),
            'avg_safety': total_safety / len(self.platoons),
            'ready_platoons': ready_count
        }

    def print_platoon_info(self):
        """Display platoon information (for debugging/monitoring)"""
        stats = self.get_platoon_stats()
        
        print(f"\n🚗 Platoon Management System Status")
        print(f"📊 Overview:")
        print(f"   Active platoons: {stats['num_platoons']}")
        print(f"   Vehicles in platoons: {stats['vehicles_in_platoons']}")
        print(f"   Average platoon size: {stats['avg_platoon_size']:.1f}")
        print(f"   Ready for intersection: {stats['performance_summary']['ready_platoons']}")
        
        if stats['direction_distribution']:
            print(f"   Direction distribution: {stats['direction_distribution']}")
        
        if stats['formation_stats']['total_formed'] > 0:
            print(f"   Formation history: {stats['formation_stats']['total_formed']} formed, "
                  f"{stats['formation_stats']['total_dissolved']} dissolved")
        
        # Show individual platoon details (limited)
        if self.platoons:
            print(f"\n🔍 Active Platoons:")
            for i, platoon in enumerate(self.platoons[:3]):  # Show top 3
                perf = platoon.get_performance_summary()
                print(f"   {i+1}. {platoon.platoon_id} "
                      f"({platoon.get_size()} vehicles, {platoon.get_goal_direction()}) "
                      f"Ready: {platoon.is_ready_for_intersection()}")

