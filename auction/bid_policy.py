import math
from env.simulation_config import SimulationConfig
from typing import Optional, List, Dict

class AgentBidPolicy:
    """
    Bidding policy for intersection auction participants.
    
    Handles both individual vehicles and platoons (when enabled) through
    a unified interface without tight coupling to specific implementations.
    """
    
    def __init__(self, agent, intersection_center=(-188.9, -89.7, 0.0), state_extractor=None):
        self.agent = agent
        self.intersection_center = intersection_center
        self.state_extractor = state_extractor
        
    def compute_bid(self):
        """
        Compute bid value for intersection access.
        
        Returns:
            float: Bid value (higher = higher priority)
        """
        # Determine agent type and compute appropriate bid
        if self._is_platoon_agent():
            return self._compute_platoon_bid()
        else:
            return self._compute_vehicle_bid()

    def _is_platoon_agent(self) -> bool:
        """Check if agent represents a platoon"""
        return self.agent.get('type') == 'platoon'
    
    def _compute_vehicle_bid(self) -> float:
        """Compute bid for individual vehicle agent"""
        # Core bidding factors
        urgency = self._estimate_vehicle_urgency()
        position_advantage = self._calculate_vehicle_position_advantage()
        speed_factor = self._calculate_vehicle_speed_factor()
        junction_factor = self._get_vehicle_junction_factor()
        wait_time_bonus = self._calculate_wait_time_bonus()
        
        # Weighted combination
        base_bid = (urgency * 20 +
                   position_advantage * 15 +
                   speed_factor * 10 +
                   junction_factor * 25 +
                   wait_time_bonus * 15)
        
        return max(0.0, base_bid)
    
    def _compute_platoon_bid(self) -> float:
        """Compute bid for platoon agent"""
        # Base vehicle bid calculation
        base_bid = self._compute_vehicle_bid()
        
        # Platoon-specific bonuses
        platoon_bonus = self._calculate_platoon_bonus()
        coordination_bonus = self._calculate_coordination_bonus()
        efficiency_bonus = self._calculate_efficiency_bonus()
        
        # Combined platoon bid
        platoon_bid = base_bid + platoon_bonus + coordination_bonus + efficiency_bonus
        
        return max(0.0, platoon_bid)
    
    def _calculate_platoon_bonus(self) -> float:
        """Calculate bonus for platoon size and cohesion"""
        if not self._is_platoon_agent():
            return 0.0
        
        vehicles = self.agent.get('vehicles', [])
        platoon_size = len(vehicles)
        
        # Size bonus (larger platoons get higher priority)
        size_bonus = platoon_size * 10.0
        
        # Cohesion bonus (check if platoon is well-formed)
        cohesion_bonus = self._calculate_platoon_cohesion_bonus(vehicles)
        
        return size_bonus + cohesion_bonus
    
    def _calculate_platoon_cohesion_bonus(self, vehicles: List[Dict]) -> float:
        """Calculate bonus for platoon cohesion"""
        if len(vehicles) < 2:
            return 0.0
        
        # Check inter-vehicle distances
        distances = []
        for i in range(len(vehicles) - 1):
            dist = self._vehicle_distance(vehicles[i], vehicles[i + 1])
            distances.append(dist)
        
        # Good spacing gets bonus
        target_spacing = 8.0
        max_spacing = 15.0
        
        cohesion_score = 0.0
        for dist in distances:
            if dist <= max_spacing:
                if target_spacing * 0.5 <= dist <= target_spacing * 1.5:
                    cohesion_score += 15.0  # Optimal spacing
                else:
                    cohesion_score += max(0.0, 10.0 - abs(dist - target_spacing))
        
        return cohesion_score / len(distances) if distances else 0.0
    
    def _calculate_coordination_bonus(self) -> float:
        """Calculate bonus for platoon coordination"""
        if not self._is_platoon_agent():
            return 0.0
        
        vehicles = self.agent.get('vehicles', [])
        
        # Check if all vehicles have same direction
        directions = []
        for vehicle in vehicles:
            direction = self._get_vehicle_direction_from_data(vehicle)
            if direction:
                directions.append(direction)
        
        # Same direction bonus
        if directions and len(set(directions)) == 1:
            return 20.0  # All vehicles going same direction
        
        return 5.0  # Partial coordination
    
    def _calculate_efficiency_bonus(self) -> float:
        """Calculate efficiency bonus for platoon"""
        if not self._is_platoon_agent():
            return 0.0
        
        vehicles = self.agent.get('vehicles', [])
        
        # Speed consistency bonus
        speeds = [self._get_current_speed(v) for v in vehicles]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        
        if avg_speed > 0:
            speed_variance = sum(abs(s - avg_speed) for s in speeds) / len(speeds)
            speed_consistency = max(0.0, 1.0 - speed_variance / avg_speed)
            return speed_consistency * 15.0
        
        return 0.0
    
    def _vehicle_distance(self, vehicle1: Dict, vehicle2: Dict) -> float:
        """Calculate distance between two vehicles"""
        loc1 = vehicle1.get('location', [0, 0, 0])
        loc2 = vehicle2.get('location', [0, 0, 0])
        return math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)
    
    def _get_vehicle_direction_from_data(self, vehicle_data: Dict) -> Optional[str]:
        """Get direction for vehicle from its data"""
        if not vehicle_data.get('destination'):
            return None
        
        try:
            import carla
            vehicle_location = carla.Location(
                x=vehicle_data['location'][0],
                y=vehicle_data['location'][1], 
                z=vehicle_data['location'][2]
            )
            
            return self.state_extractor.get_route_direction(
                vehicle_location, vehicle_data['destination']
            )
        except Exception:
            return None
    
    def _get_platoon_direction(self) -> Optional[str]:
        """Get direction for platoon using leader's direction"""
        vehicles = self.agent.get('vehicles', [])
        if vehicles:
            return self._get_vehicle_direction_from_data(vehicles[0])
        return None
    
    def _estimate_vehicle_urgency(self) -> float:
        """Estimate urgency for individual vehicle"""
        direction = self._get_agent_direction()
        
        # Base urgency
        base_urgency = 10.0
        
        # Direction priority
        direction_bonus = {
            'straight': 15.0,
            'left': 10.0,
            'right': 12.0
        }.get(direction, 8.0)
        
        # Distance factor
        location = self._get_agent_location()
        if location:
            distance = self._distance_to_intersection(location)
            if distance <= 30.0:
                distance_urgency = 20.0 - distance * 0.5
            else:
                distance_urgency = 5.0
        else:
            distance_urgency = 5.0
        
        return base_urgency + direction_bonus + distance_urgency
    
    def _calculate_vehicle_position_advantage(self) -> float:
        """Calculate position advantage for individual vehicle"""
        at_junction = self._is_agent_at_junction()
        
        if at_junction:
            return 60.0  # High advantage for vehicles in intersection
        
        location = self._get_agent_location()
        if location:
            distance = self._distance_to_intersection(location)
            if distance <= 50.0:
                return 30.0 - distance * 0.3
            else:
                return 5.0
        
        return 5.0  # Fallback
    
    def _calculate_vehicle_speed_factor(self) -> float:
        """Calculate speed factor for individual vehicle"""
        try:
            vehicle_data = self._get_vehicle_data()
            if vehicle_data:
                speed = self._get_current_speed(vehicle_data)
                
                # Reasonable speed gets bonus
                if 3.0 <= speed <= 10.0:
                    return 10.0
                elif speed < 3.0:
                    return 5.0
                else:
                    return 7.0
            else:
                return 5.0
                
        except Exception:
            return 5.0  # Default value
    
    def _get_vehicle_junction_factor(self) -> float:
        """Get junction factor for individual vehicle"""
        at_junction = self._is_agent_at_junction()
        if at_junction:
            return 40.0
        
        location = self._get_agent_location()
        if location:
            distance = self._distance_to_intersection(location)
            return max(0.0, 25.0 - distance * 0.25)
        
        return 10.0  # Fallback
    
    def _calculate_wait_time_bonus(self) -> float:
        """Calculate waiting time bonus"""
        wait_time = self.agent.get('wait_time', 0.0)
        
        if wait_time <= 2.0:
            return 0.0
        elif wait_time <= 5.0:
            return (wait_time - 2.0) * 5.0
        elif wait_time <= 10.0:
            return 15.0 + (wait_time - 5.0) * 8.0
        else:
            return 55.0 + (wait_time - 10.0) * 10.0
    
    # Helper methods for agent data extraction
    def _get_agent_location(self):
        """Get agent location regardless of type"""
        if 'location' in self.agent:
            return self.agent['location']
        elif 'data' in self.agent and 'location' in self.agent['data']:
            return self.agent['data']['location']
        return None
    
    def _is_agent_at_junction(self) -> bool:
        """Check if agent is at junction"""
        if 'at_junction' in self.agent:
            return self.agent['at_junction']
        elif 'data' in self.agent:
            return self.agent['data'].get('is_junction', False)
        return False
    
    def _get_agent_direction(self) -> Optional[str]:
        """Get agent direction through navigation system"""
        if self._is_platoon_agent():
            return self._get_platoon_direction()
        else:
            return self._get_vehicle_direction()
    
    def _get_vehicle_direction(self) -> Optional[str]:
        """Get direction for individual vehicle"""
        vehicle_data = self._get_vehicle_data()
        if not vehicle_data or not vehicle_data.get('destination'):
            return None
        
        try:
            import carla
            vehicle_location = carla.Location(
                x=vehicle_data['location'][0],
                y=vehicle_data['location'][1], 
                z=vehicle_data['location'][2]
            )
            
            return self.state_extractor.get_route_direction(
                vehicle_location, vehicle_data['destination']
            )
        except Exception:
            return None
    
    def _get_platoon_direction(self) -> Optional[str]:
        """Get direction for platoon using leader's direction"""
        vehicles = self.agent.get('vehicles', [])
        if vehicles:
            return self._get_vehicle_direction_from_data(vehicles[0])
        return None
    
    def _get_vehicle_data(self):
        """Get vehicle data from agent"""
        if 'data' in self.agent:
            return self.agent['data']
        elif self.agent.get('type') == 'vehicle':
            return self.agent
        return None
    
    def _get_current_speed(self, vehicle_state):
        """Get current speed from vehicle state"""
        velocity = vehicle_state.get('velocity', (0, 0, 0))
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
    
    def _distance_to_intersection(self, location) -> float:
        """Calculate distance to intersection center"""
        return math.sqrt(
            (location[0] - self.intersection_center[0])**2 + 
            (location[1] - self.intersection_center[1])**2
        )
