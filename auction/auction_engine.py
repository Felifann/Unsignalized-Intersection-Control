import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from env.simulation_config import SimulationConfig
from .bid_policy import AgentBidPolicy

class AuctionStatus(Enum):
    WAITING = "waiting"
    BIDDING = "bidding" 
    EVALUATING = "evaluating"
    COMPLETED = "completed"

@dataclass
class AuctionParticipant:
    """Represents an agent participating in the auction"""
    id: str
    type: str  # 'vehicle' or 'platoon'
    location: Tuple[float, float, float]
    data: Dict[str, Any]
    at_junction: bool = False
    
    @property
    def vehicles(self) -> List[Dict]:
        """Get vehicles associated with this participant"""
        if self.type == 'platoon':
            return self.data.get('vehicles', [])
        return [self.data] if self.type == 'vehicle' else []

@dataclass
class Bid:
    """Represents a bid in the auction"""
    participant_id: str
    value: float
    timestamp: float
    participant: AuctionParticipant

@dataclass
class AuctionWinner:
    """Represents an auction winner with ranking"""
    participant: AuctionParticipant
    bid: Bid
    rank: int
    # protected: bool = False
    conflict_action: str = "go"

class Auction:
    """Manages a single auction round"""
    
    def __init__(self, auction_id: str, participants: List[AuctionParticipant], 
                 bidding_duration: float = 1.0):
        self.id = auction_id
        self.participants = participants
        self.start_time = time.time()
        self.deadline = self.start_time + bidding_duration
        self.status = AuctionStatus.BIDDING
        self.bids: Dict[str, Bid] = {}
        self.winners: List[AuctionWinner] = []
    
    def add_bid(self, bid: Bid) -> bool:
        """Add a bid to the auction"""
        if self.status != AuctionStatus.BIDDING:
            return False
        
        self.bids[bid.participant_id] = bid
        return True
    
    def is_expired(self) -> bool:
        """Check if auction has expired"""
        return time.time() >= self.deadline
    
    def get_participation_rate(self) -> float:
        """Get percentage of participants who have bid"""
        if not self.participants:
            return 0.0
        return len(self.bids) / len(self.participants)

class LaneGrouper:
    """Handles lane-based vehicle grouping logic"""
    
    def __init__(self, state_extractor=None):
        self.state_extractor = state_extractor
    
    def get_lane_leaders(self, vehicle_states: List[Dict]) -> List[Dict]:
        """Get the first vehicle in each lane approaching the intersection"""
        lanes = self._group_vehicles_by_lane(vehicle_states)
        lane_leaders = []
        
        for lane_id, vehicles in lanes.items():
            if vehicles:
                # Find closest vehicle to intersection in this lane
                closest_vehicle = min(
                    vehicles,
                    key=lambda v: SimulationConfig.distance_to_intersection_center(v['location'])
                )
                lane_leaders.append(closest_vehicle)
        
        return lane_leaders
    
    def _group_vehicles_by_lane(self, vehicle_states: List[Dict]) -> Dict[str, List[Dict]]:
        """Group vehicles by lane using CARLA waypoint system"""
        lanes = {}
        
        for vehicle in vehicle_states:
            try:
                if self.state_extractor:
                    import carla
                    location = carla.Location(
                        x=vehicle['location'][0],
                        y=vehicle['location'][1], 
                        z=vehicle['location'][2]
                    )
                    waypoint = self.state_extractor.carla.world.get_map().get_waypoint(location)
                    lane_key = f"road_{waypoint.road_id}_lane_{waypoint.lane_id}"
                    
                    if lane_key not in lanes:
                        lanes[lane_key] = []
                    lanes[lane_key].append(vehicle)
                    
            except Exception as e:
                print(f"[LaneGrouper] Error getting lane info for vehicle {vehicle['id']}: {e}")
        
        return lanes

class ParticipantIdentifier:
    """Identifies auction participants from vehicle states and platoons"""
    
    def __init__(self, lane_grouper: LaneGrouper):
        self.lane_grouper = lane_grouper
    
    def identify_participants(self, vehicle_states: List[Dict], 
                            platoon_manager=None) -> List[AuctionParticipant]:
        """识别拍卖参与者 - 单车版本（已禁用车队逻辑）"""
        participants = []
        
        # 获取车道领头者
        lane_leaders = self.lane_grouper.get_lane_leaders(vehicle_states)
        if not lane_leaders:
            return participants
        
        # DISABLED: Platoon logic temporarily removed for single-agent focus
        # 🚫 车队逻辑已暂时禁用，专注于单车行为
        
        # 添加单独车辆参与者
        for vehicle in lane_leaders:
            if self._vehicle_has_destination(vehicle):
                participant = AuctionParticipant(
                    id=vehicle['id'],
                    type='vehicle',
                    location=vehicle['location'],
                    data=vehicle,
                    at_junction=vehicle.get('is_junction', False)
                )
                participants.append(participant)
        
        print(f"🎯 单车拍卖参与者识别完成: {len(participants)}个独立车辆")
        
        return participants
    
    # DISABLED: Platoon-related methods temporarily removed
    # def _analyze_platoon_transit_status(self, platoon_vehicles: List[Dict]) -> Dict:
    # def _get_vehicle_lane(self, vehicle: Dict) -> str:
    
    def _vehicle_has_destination(self, vehicle: Dict) -> bool:
        """Check if vehicle has a valid destination set"""
        try:
            # Check if vehicle has destination in its data
            if 'destination' in vehicle and vehicle['destination'] is not None:
                return True
            
            # Check if vehicle is moving (has non-zero velocity)
            velocity = vehicle.get('velocity', [0, 0, 0])
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                return speed > 0.1  # Moving vehicles likely have destinations
            
            # Default: assume vehicle has destination if it's in the simulation
            return True
            
        except Exception as e:
            print(f"[Warning] 检查车辆目的地失败 {vehicle.get('id', 'unknown')}: {e}")
            return True  # Default to True to include vehicle in auction

class AuctionEvaluator:
    """Handles auction evaluation and winner determination"""
    
    def __init__(self, intersection_center: Tuple[float, float, float]):
        self.intersection_center = intersection_center
        self.protected_agents: set = set()
        self.agents_in_transit: Dict[str, Dict] = {}
    
    def evaluate_auction(self, auction: Auction) -> List[AuctionWinner]:
        """Evaluate auction and determine winners with priority ranking"""
        if not auction.bids:
            return []
        
        # 1. Identify protected agents (already in transit)
        # protected_winners = self._get_protected_winners(auction.bids)
        
        # 2. Sort remaining bidders by bid value
        remaining_bids = {k: v for k, v in auction.bids.items() }
                         # if k not in [w.participant.id for w in protected_winners]}
        
        regular_winners = self._evaluate_regular_bids(remaining_bids)
        
        # 3. Combine and assign final rankings
        all_winners = regular_winners # + protected_winners
        for i, winner in enumerate(all_winners):
            winner.rank = i + 1
        
        auction.winners = all_winners
        return all_winners

    def _get_protected_winners(self, bids: Dict[str, Bid]) -> List[AuctionWinner]:
        """Get winners that are protected (in transit through intersection)"""
        protected_winners = []
        
        for bid in bids.values():
            if self._is_participant_in_transit(bid.participant):
                # Mark as protected
                self.protected_agents.add(bid.participant_id)
                self.agents_in_transit[bid.participant_id] = {
                    'start_time': time.time(),
                    'original_bid': bid.value
                }
                
                winner = AuctionWinner(
                    participant=bid.participant,
                    bid=bid,
                    rank=0,  # Will be reassigned
                    protected=True
                )
                protected_winners.append(winner)
        
        return protected_winners
    
    def _evaluate_regular_bids(self, bids: Dict[str, Bid]) -> List[AuctionWinner]:
        """Evaluate regular (non-protected) bids"""
        if not bids:
            return []
        
        # Sort by bid value (descending), with tie-breaker by timestamp
        sorted_bids = sorted(
            bids.values(),
            key=lambda b: (b.value, -b.timestamp),  # Higher bid wins, earlier timestamp breaks ties
            reverse=True
        )
        
        winners = []
        for bid in sorted_bids:
            winner = AuctionWinner(
                participant=bid.participant,
                bid=bid,
                rank=0,  # Will be reassigned
                #protected=False
            )
            winners.append(winner)
        
        return winners
    
    def _is_participant_in_transit(self, participant: AuctionParticipant) -> bool:
        """Check if participant is currently in transit through intersection"""
        if participant.type == 'vehicle':
            return participant.data.get('is_junction', False)
        # elif participant.type == 'platoon':
        #     # Platoon is in transit if any vehicle is in junction
        #     for vehicle in participant.vehicles:
        #         if vehicle.get('is_junction', False):
        #             return True
        return False
    
    def cleanup_completed_agents(self, vehicle_states: List[Dict], platoon_manager=None):
        """Clean up agents that have completed transit - 单车版本"""
        current_time = time.time()
        completed_agents = []
        
        for agent_id in list(self.protected_agents):
            # SIMPLIFIED: Only check single vehicles since platoons are disabled
            agent_still_in_transit = self._check_single_vehicle_in_transit(
                agent_id, vehicle_states
            )
            
            # Remove protection if agent completed transit or timed out
            transit_time = current_time - self.agents_in_transit.get(agent_id, {}).get('start_time', current_time)
            
            if not agent_still_in_transit or transit_time > 30.0:
                completed_agents.append(agent_id)
        
        # Clean up completed agents
        for agent_id in completed_agents:
            self.protected_agents.discard(agent_id)
            self.agents_in_transit.pop(agent_id, None)
            print(f"✅ Vehicle {agent_id} completed transit, protection removed")
    
    def _check_single_vehicle_in_transit(self, agent_id: str, vehicle_states: List[Dict]) -> bool:
        """检查单车是否仍在通过路口 - 简化版本"""
        for vehicle_state in vehicle_states:
            vehicle_id = str(vehicle_state['id'])
            if vehicle_id == str(agent_id):
                return vehicle_state.get('is_junction', False)
        return False
    
    # DISABLED: Platoon-specific transit checking
    # def _check_agent_still_in_transit(self, agent_id: str, vehicle_states: List[Dict], platoon_manager) -> bool:

class DecentralizedAuctionEngine:
    """Main auction engine managing the complete auction process - 单车版本"""
    
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0), 
                 communication_range=50.0, state_extractor=None):
        self.intersection_center = intersection_center
        self.communication_range = communication_range
        self.state_extractor = state_extractor
        
        # Core components
        self.lane_grouper = LaneGrouper(state_extractor)
        self.participant_identifier = ParticipantIdentifier(self.lane_grouper)
        self.evaluator = AuctionEvaluator(intersection_center)
        
        # Auction management
        self.current_auction: Optional[Auction] = None
        self.auction_history: Dict[str, Auction] = {}
        self.auction_interval = 2.0  # seconds between auctions
        self.last_auction_time = 0
        
        # Communication simulation
        self.message_queue: List[Dict] = []
        
        # Integration points
        self.vehicle_enforcer = None
        
        print("🎯 单车专用拍卖引擎已初始化 - 车队逻辑已禁用")
    
    def set_vehicle_enforcer(self, vehicle_enforcer):
        """Set vehicle control enforcer for integration"""
        self.vehicle_enforcer = vehicle_enforcer
    
    def update(self, vehicle_states: List[Dict], platoon_manager=None) -> List[AuctionWinner]:
        """Main update loop - 单车版本（忽略platoon_manager）"""
        current_time = time.time()
        
        # 1. Identify potential participants (vehicles only)
        participants = self.participant_identifier.identify_participants(
            vehicle_states, None  # Pass None instead of platoon_manager
        )
        
        # 2. Start new auction if needed
        if participants and not self.current_auction:
            self._start_new_auction(participants, current_time)
        
        # 3. Process current auction
        winners = []
        if self.current_auction:
            winners = self._process_current_auction(current_time)
        
        # 4. Clean up completed protected agents (single vehicles only)
        self.evaluator.cleanup_completed_agents(vehicle_states, None)
        
        # 5. Simulate communication
        self._simulate_v2v_communication()
        
        return winners
    
    def _start_new_auction(self, participants: List[AuctionParticipant], start_time: float):
        """Start a new auction round"""
        auction_id = f"junction_auction_{int(start_time)}"
        self.current_auction = Auction(auction_id, participants)
        
        # Collect bids immediately
        self._collect_bids()
        
        # Broadcast auction start
        self._broadcast_message({
            'type': 'auction_start',
            'auction_id': auction_id,
            'participants': [p.id for p in participants],
            'timestamp': start_time
        })
        
        print(f"🎯 Started auction {auction_id} with {len(participants)} participants")
    
    def _process_current_auction(self, current_time: float) -> List[AuctionWinner]:
        """Process the current active auction"""
        if not self.current_auction:
            return []
        
        auction = self.current_auction
        
        if auction.status == AuctionStatus.BIDDING:
            if auction.is_expired():
                auction.status = AuctionStatus.EVALUATING
        
        elif auction.status == AuctionStatus.EVALUATING:
            winners = self.evaluator.evaluate_auction(auction)
            auction.status = AuctionStatus.COMPLETED
            
            # Broadcast results
            self._broadcast_auction_results(auction.id, winners)
            self._print_auction_results(auction.id, winners)
            
            return winners
        
        elif auction.status == AuctionStatus.COMPLETED:
            # Archive and clean up
            self.auction_history[auction.id] = auction
            self.current_auction = None
            self.last_auction_time = current_time
        
        return auction.winners if auction.winners else []
    
    def _collect_bids(self):
        """Collect bids from all participants"""
        if not self.current_auction:
            return
        
        for participant in self.current_auction.participants:
            # Create bid policy and compute bid
            bid_policy = AgentBidPolicy(
                self._participant_to_agent_dict(participant),
                self.intersection_center,
                self.state_extractor
            )
            bid_value = bid_policy.compute_bid()
            
            # Create and add bid
            bid = Bid(
                participant_id=participant.id,
                value=bid_value,
                timestamp=time.time(),
                participant=participant
            )
            
            self.current_auction.add_bid(bid)
    
    def _participant_to_agent_dict(self, participant: AuctionParticipant) -> Dict:
        """Convert AuctionParticipant to legacy agent dict format for BidPolicy - 单车版本"""
        agent_dict = {
            'id': participant.id,
            'type': participant.type,
            'location': participant.location,
            'at_junction': participant.at_junction
        }
        
        # DISABLED: Platoon logic removed
        # Only handle individual vehicles now
        if participant.type == 'vehicle':
            agent_dict['data'] = participant.data
        # elif participant.type == 'platoon':  # DISABLED
        
        return agent_dict
    
    def _print_auction_results(self, auction_id: str, winners: List[AuctionWinner]):
        """Print detailed auction results - 单车版本"""
        if not winners:
            return
        
        print(f"🏆 Auction {auction_id} completed. Priority order:")
        for winner in winners[:3]:  # Show top 3
            participant = winner.participant
            status_emoji = "🏢" if participant.at_junction else "🚦"
            # protection_emoji = "🛡️" if winner.protected else ""
            
            # SIMPLIFIED: Only show vehicle info since platoons are disabled
            print(f"   #{winner.rank}: {status_emoji}🚗 "#{protection_emoji}
                  f"Vehicle {participant.id} Bid: {winner.bid.value:.1f}")
    
    def _broadcast_auction_results(self, auction_id: str, winners: List[AuctionWinner]):
        """Broadcast auction results"""
        self._broadcast_message({
            'type': 'auction_results',
            'auction_id': auction_id,
            'winners': [(w.participant.id, w.bid.value, w.rank) for w in winners[:3]],
            'timestamp': time.time()
        })
    
    def _broadcast_message(self, message: Dict):
        """Add message to communication queue"""
        self.message_queue.append(message)
    
    def _simulate_v2v_communication(self):
        """Simulate V2V communication with delays and packet loss"""
        current_time = time.time()
        valid_messages = []
        
        for message in self.message_queue:
            # Keep messages valid for 0.5 seconds
            if current_time - message['timestamp'] < 0.5:
                valid_messages.append(message)
        
        self.message_queue = valid_messages
    
    def get_current_priority_order(self) -> List[AuctionWinner]:
        """Get current priority order from active or most recent auction"""
        if self.current_auction and self.current_auction.winners:
            return self.current_auction.winners
        
        # Return most recent completed auction results
        if self.auction_history:
            latest_auction = max(self.auction_history.values(), 
                               key=lambda a: a.start_time)
            return latest_auction.winners
        
        return []
    
    def get_auction_stats(self) -> Dict[str, Any]:
        """Get comprehensive auction statistics - 单车版本"""
        current_participants = 0
        vehicle_count = 0
        
        if self.current_auction:
            current_participants = len(self.current_auction.participants)
            # All participants are vehicles now
            vehicle_count = current_participants
        
        return {
            'active_auction': self.current_auction is not None,
            'current_participants': current_participants,
            'platoon_participants': 0,  # Always 0 when platoons disabled
            'vehicle_participants': vehicle_count,
            'completed_auctions': len(self.auction_history),
            'protected_agents': len(self.evaluator.protected_agents),
            'auction_status': self.current_auction.status.value if self.current_auction else 'none'
        }
    
    # Extension points for future integration
    def apply_conflict_resolution(self, winners: List[AuctionWinner], 
                                conflict_actions: Dict[str, str]) -> List[AuctionWinner]:
        """Apply conflict resolution results (Nash equilibrium integration point)"""
        if not conflict_actions:
            return winners
        
        resolved_winners = []
        waiting_winners = []
        
        for winner in winners:
            action = conflict_actions.get(winner.participant.id, 'go')
            winner.conflict_action = action
            
            if action == 'go':
                resolved_winners.append(winner)
            else:
                waiting_winners.append(winner)
        
        # Reassign rankings
        all_winners = resolved_winners + waiting_winners
        for i, winner in enumerate(all_winners):
            winner.rank = i + 1
        
        if waiting_winners:
            print(f"🎮 Conflict resolution: {len(waiting_winners)} agents waiting")
        
        return all_winners
    
    def integrate_learned_bidding_policy(self, policy_function):
        """Integration point for RL-based bidding policies"""
        # Future implementation for PPO integration
        pass
