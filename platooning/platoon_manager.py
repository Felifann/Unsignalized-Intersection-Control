import math
import time
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from env.simulation_config import SimulationConfig
from .platoon_policy import Platoon
import carla

class PlatoonManager:
    """Enhanced platoon manager with better state management and auction integration"""
    
    def __init__(self, state_extractor, intersection_center=(-188.9, -89.7, 0.0)):
        self.state_extractor = state_extractor
        self.intersection_center = intersection_center
        
        # Platoon configuration
        self.max_platoon_size = 4
        self.min_platoon_size = 2
        self.max_following_distance = 25.0  # Increased for better formation
        self.target_following_distance = 8.0  # Target distance for formation
        
        # State management
        self.platoons: List[Platoon] = []
        self.platoon_history: Dict[str, Platoon] = {}
        self.last_update_time = 0
        self.update_interval = 1.0  # Update every second
        
        # Performance tracking
        self.formation_stats = {
            'total_formed': 0,
            'total_dissolved': 0,
            'successful_crossings': 0
        }
        
        print("🚗 Enhanced Platoon Manager initialized")

    def update(self):
        """Main update method with improved state management"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Get current vehicle states
        vehicle_states = self.state_extractor.get_vehicle_states()
        intersection_vehicles = self._filter_near_intersection(vehicle_states)
        
        # Update existing platoons
        self._update_existing_platoons(intersection_vehicles)
        
        # Form new platoons from available vehicles
        self._form_new_platoons(intersection_vehicles)
        
        # Clean up invalid platoons
        self._cleanup_invalid_platoons()
        
        self.last_update_time = current_time

    def _filter_near_intersection(self, vehicle_states: List[Dict]) -> List[Dict]:
        """Filter vehicles near intersection with enhanced criteria"""
        intersection_vehicles = []
        
        for vehicle in vehicle_states:
            # Check if vehicle is in intersection area
            if SimulationConfig.is_in_intersection_area(vehicle['location']):
                # Only include vehicles with valid destinations for platoon formation
                if vehicle.get('destination') or vehicle.get('is_junction', False):
                    intersection_vehicles.append(vehicle)
        
        return intersection_vehicles

    def _update_existing_platoons(self, vehicle_states: List[Dict]):
        """Update existing platoons with new vehicle states"""
        vehicle_lookup = {str(v['id']): v for v in vehicle_states}
        
        for platoon in self.platoons[:]:  # Copy list to allow modification
            # Get updated states for platoon vehicles
            updated_vehicles = []
            for vehicle_id in platoon.get_vehicle_ids():
                if vehicle_id in vehicle_lookup:
                    updated_vehicles.append(vehicle_lookup[vehicle_id])
            
            # Update platoon state
            if updated_vehicles:
                platoon.update_state(updated_vehicles)
                if not platoon.is_valid():
                    self._dissolve_platoon(platoon, "Invalid after update")
            else:
                self._dissolve_platoon(platoon, "No vehicles found")

    def _form_new_platoons(self, vehicle_states: List[Dict]):
        """Form new platoons from available vehicles"""
        # Get vehicles not already in platoons
        existing_vehicle_ids = set()
        for platoon in self.platoons:
            existing_vehicle_ids.update(platoon.get_vehicle_ids())
        
        available_vehicles = [
            v for v in vehicle_states 
            if str(v['id']) not in existing_vehicle_ids and self._can_form_platoon(v)
        ]
        
        if len(available_vehicles) < self.min_platoon_size:
            return
        
        # Group vehicles by lane and direction
        groups = self._group_by_lane_and_goal(available_vehicles)
        
        # Form platoons from groups
        for group in groups:
            if len(group) >= self.min_platoon_size:
                new_platoons = self._create_platoons_from_group(group)
                self.platoons.extend(new_platoons)

    def _can_form_platoon(self, vehicle: Dict) -> bool:
        """Check if vehicle can participate in platoon formation"""
        # Must have destination or be in junction
        if not vehicle.get('destination') and not vehicle.get('is_junction', False):
            return False
        
        # Must have valid direction
        direction = self._estimate_goal_direction(vehicle)
        return direction is not None

    def _group_by_lane_and_goal(self, vehicles: List[Dict]) -> List[List[Dict]]:
        """Group vehicles by lane and goal direction with improved logic"""
        # Group by lane first
        lane_groups = defaultdict(list)
        
        for vehicle in vehicles:
            lane_id = self._get_lane_id(vehicle)
            direction = self._estimate_goal_direction(vehicle)
            
            if direction:
                lane_groups[lane_id].append((vehicle, direction))
        
        # Process each lane group
        final_groups = []
        for lane_id, vehicles_with_direction in lane_groups.items():
            if len(vehicles_with_direction) < self.min_platoon_size:
                continue
            
            # Sort by distance to intersection
            sorted_vehicles = sorted(
                vehicles_with_direction,
                key=lambda x: self._distance_to_intersection(x[0])
            )
            
            # Find adjacent groups with same direction
            adjacent_groups = self._find_adjacent_groups_with_direction(sorted_vehicles)
            final_groups.extend(adjacent_groups)
        
        return final_groups

    def _find_adjacent_groups_with_direction(self, sorted_vehicles_with_direction: List[Tuple]) -> List[List[Dict]]:
        """Find adjacent vehicle groups with same direction"""
        if not sorted_vehicles_with_direction:
            return []
        
        groups = []
        current_group = [sorted_vehicles_with_direction[0][0]]
        current_direction = sorted_vehicles_with_direction[0][1]
        
        for i in range(1, len(sorted_vehicles_with_direction)):
            vehicle, direction = sorted_vehicles_with_direction[i]
            prev_vehicle = sorted_vehicles_with_direction[i-1][0]
            
            # Check direction match
            if direction != current_direction:
                if len(current_group) >= self.min_platoon_size:
                    groups.append(current_group)
                current_group = [vehicle]
                current_direction = direction
                continue
            
            # Check adjacency
            distance = self._calculate_vehicle_distance(prev_vehicle, vehicle)
            
            if distance <= self.max_following_distance:
                current_group.append(vehicle)
            else:
                if len(current_group) >= self.min_platoon_size:
                    groups.append(current_group)
                current_group = [vehicle]
        
        # Add final group
        if len(current_group) >= self.min_platoon_size:
            groups.append(current_group)
        
        return groups

    def _create_platoons_from_group(self, vehicle_group: List[Dict]) -> List[Platoon]:
        """Create platoons from a vehicle group"""
        if len(vehicle_group) < self.min_platoon_size:
            return []
        
        platoons = []
        
        # Split large groups into multiple platoons if needed
        while len(vehicle_group) >= self.min_platoon_size:
            platoon_size = min(self.max_platoon_size, len(vehicle_group))
            platoon_vehicles = vehicle_group[:platoon_size]
            vehicle_group = vehicle_group[platoon_size:]
            
            # Verify direction consistency
            directions = [self._estimate_goal_direction(v) for v in platoon_vehicles]
            unique_directions = set(filter(None, directions))
            
            if len(unique_directions) == 1:
                platoon = Platoon(
                    platoon_vehicles, 
                    self.intersection_center, 
                    goal_direction=directions[0],
                    state_extractor=self.state_extractor
                )
                
                if platoon.is_valid():
                    platoons.append(platoon)
                    self.formation_stats['total_formed'] += 1
                    print(f"✅ Formed new platoon: {platoon.platoon_id} "
                          f"({platoon.get_size()} vehicles, {platoon.get_goal_direction()})")
        
        return platoons

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

    def _get_lane_id(self, vehicle: Dict) -> str:
        """Get lane identifier for vehicle"""
        road_id = vehicle.get('road_id', 'unknown')
        lane_id = vehicle.get('lane_id', 'unknown')
        return f"{road_id}_{lane_id}"

    def _estimate_goal_direction(self, vehicle: Dict) -> Optional[str]:
        """Estimate vehicle goal direction using navigation system"""
        if not vehicle.get('destination'):
            return None
        
        try:
            vehicle_location = carla.Location(
                x=vehicle['location'][0],
                y=vehicle['location'][1],
                z=vehicle['location'][2]
            )
            
            return self.state_extractor.get_route_direction(
                vehicle_location, vehicle['destination']
            )
        except Exception as e:
            return None

    def _distance_to_intersection(self, vehicle: Dict) -> float:
        """Calculate distance from vehicle to intersection center"""
        return SimulationConfig.distance_to_intersection_center(vehicle['location'])

    def _calculate_vehicle_distance(self, vehicle1: Dict, vehicle2: Dict) -> float:
        """Calculate distance between two vehicles"""
        loc1 = vehicle1['location']
        loc2 = vehicle2['location']
        return math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)

    # Public interface methods for auction integration
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
        # Initialize performance summary with default values
        performance_summary = {
            'avg_cohesion': 0.0,
            'avg_efficiency': 0.0,
            'avg_safety': 0.0,
            'ready_platoons': 0
        }
        
        if not self.platoons:
            return {
                'num_platoons': 0,
                'vehicles_in_platoons': 0,
                'avg_platoon_size': 0.0,
                'direction_distribution': {},
                'performance_summary': performance_summary,
                'formation_stats': self.formation_stats
            }
        
        total_vehicles = sum(p.get_size() for p in self.platoons)
        avg_size = total_vehicles / len(self.platoons)
        
        # Direction distribution
        direction_dist = defaultdict(int)
        
        total_cohesion = 0.0
        total_efficiency = 0.0
        total_safety = 0.0
        
        for platoon in self.platoons:
            direction = platoon.get_goal_direction()
            direction_dist[direction] += 1
            
            if platoon.is_ready_for_intersection():
                performance_summary['ready_platoons'] += 1
            
            # Get performance metrics
            perf = platoon.get_performance_summary()
            if 'cohesion' in perf and perf['cohesion'] != 'N/A':
                total_cohesion += float(perf['cohesion'])
            if 'efficiency' in perf and perf['efficiency'] != 'N/A':
                total_efficiency += float(perf['efficiency'])
            if 'safety' in perf and perf['safety'] != 'N/A':
                total_safety += float(perf['safety'])
        
        if self.platoons:
            performance_summary['avg_cohesion'] = total_cohesion / len(self.platoons)
            performance_summary['avg_efficiency'] = total_efficiency / len(self.platoons)
            performance_summary['avg_safety'] = total_safety / len(self.platoons)
        
        return {
            'num_platoons': len(self.platoons),
            'vehicles_in_platoons': total_vehicles,
            'avg_platoon_size': avg_size,
            'direction_distribution': dict(direction_dist),
            'performance_summary': performance_summary,
            'formation_stats': self.formation_stats
        }

    def print_platoon_info(self):
        """Enhanced platoon information display"""
        stats = self.get_platoon_stats()
        unplatoon_count = self._get_unplatoon_vehicles_count()
        
        print(f"\n🚗 智能车队管理系统状态")
        print(f"📊 总体统计:")
        print(f"   活跃车队: {stats['num_platoons']} | "
              f"编队车辆: {stats['vehicles_in_platoons']} | "
              f"独行车辆: {unplatoon_count}")
        print(f"   平均车队规模: {stats['avg_platoon_size']:.1f} | "
              f"准备通行: {stats['performance_summary']['ready_platoons']}")
        print(f"   方向分布: {stats['direction_distribution']}")
        
        if stats['formation_stats']['total_formed'] > 0:
            print(f"   历史统计: 已组建{stats['formation_stats']['total_formed']}队 | "
                  f"已解散{stats['formation_stats']['total_dissolved']}队")
        
        # Performance metrics
        perf = stats['performance_summary']
        if perf['avg_cohesion'] > 0:
            print(f"   性能指标: 团结度{perf['avg_cohesion']:.2f} | "
                  f"效率{perf['avg_efficiency']:.2f} | "
                  f"安全{perf['avg_safety']:.2f}")
        
        # Individual platoon details
        if self.platoons:
            print(f"\n🔍 车队详情:")
            for i, platoon in enumerate(self.platoons[:5]):  # Show top 5
                self._print_platoon_details(i + 1, platoon)

    def _print_platoon_details(self, index: int, platoon: Platoon):
        """Print detailed information for a single platoon"""
        direction_emoji = {'left': '⬅️', 'right': '➡️', 'straight': '⬆️'}
        direction = platoon.get_goal_direction()
        lane_info = platoon.get_lane_info()
        performance = platoon.get_performance_summary()
        
        ready_status = "✅" if platoon.is_ready_for_intersection() else "⏳"
        junction_status = "🏢" if platoon.has_vehicle_in_intersection() else "🛣️"
        
        print(f"   {index}. {direction_emoji.get(direction, '❓')} "
              f"车队{platoon.platoon_id} ({platoon.get_size()}车-{direction.upper()}) "
              f"{ready_status}{junction_status}")
        
        if lane_info and lane_info[0] is not None:
            print(f"      📍 车道: R{lane_info[0]}/L{lane_info[1]} | "
                  f"速度: {performance['avg_speed_kmh']:.1f}km/h")
        
        if performance['cohesion'] != 'N/A':
            print(f"      📊 团结: {performance['cohesion']} | "
                  f"效率: {performance['efficiency']} | "
                  f"安全: {performance['safety']}")

    def _get_unplatoon_vehicles_count(self) -> int:
        """Get count of vehicles not in platoons"""
        vehicle_states = self.state_extractor.get_vehicle_states()
        intersection_vehicles = self._filter_near_intersection(vehicle_states)
        
        platoon_vehicle_ids = set()
        for platoon in self.platoons:
            platoon_vehicle_ids.update(platoon.get_vehicle_ids())
        
        unplatoon_count = 0
        for vehicle in intersection_vehicles:
            if (str(vehicle['id']) not in platoon_vehicle_ids and 
                self._can_form_platoon(vehicle)):
                unplatoon_count += 1
        
        return unplatoon_count

