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
    protected: bool = False
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
                            platoon_manager) -> List[AuctionParticipant]:
        """识别拍卖参与者 - 改进车队完整通过逻辑"""
        participants = []
        
        # 获取车道领头者
        lane_leaders = self.lane_grouper.get_lane_leaders(vehicle_states)
        if not lane_leaders:
            return participants
        
        # 跟踪车队车辆ID
        platoon_vehicle_ids = set()
        for platoon in platoon_manager.get_all_platoons():
            for vehicle in platoon.vehicles:
                platoon_vehicle_ids.add(vehicle['id'])
        
        # 🔥 新增：跟踪正在通过的车队所占用的车道
        lanes_occupied_by_transit_platoons = set()
        
        # 🔥 改进：车队完整通过状态管理
        for platoon in platoon_manager.get_all_platoons():
            leader = platoon.get_leader()
            if leader:
                # 分析车队通过状态
                transit_status = self._analyze_platoon_transit_status(platoon.vehicles)
                
                # 参与条件：
                # 1. 队长是车道领头者 (接近阶段)
                # 2. 车队正在通过过程中 (通过阶段)
                is_lane_leader = any(lv['id'] == leader['id'] for lv in lane_leaders)
                is_in_transit_process = transit_status['in_transit_process']
                
                # 🔥 新增：如果车队正在通过，标记其车道为被占用
                if is_in_transit_process:
                    leader_lane = self._get_vehicle_lane(leader)
                    if leader_lane:
                        lanes_occupied_by_transit_platoons.add(leader_lane)
                
                if is_lane_leader or is_in_transit_process:
                    participant = AuctionParticipant(
                        id=f"platoon_{leader['id']}",
                        type='platoon',
                        location=leader['location'],
                        data={
                            'vehicles': platoon.vehicles,
                            'goal_direction': platoon.get_goal_direction(),
                            'size': platoon.get_size(),
                            'in_transit': is_in_transit_process,
                            'transit_status': transit_status,  # 🔥 新增详细状态
                            'lane': self._get_vehicle_lane(leader)  # 🔥 新增车道信息
                        },
                        at_junction=is_in_transit_process
                    )
                    participants.append(participant)
                    
                    print(f"🚛 车队 {participant.id} 参与拍卖:")
                    print(f"   📍 lane_leader={is_lane_leader}")
                    print(f"   🚦 in_transit_process={is_in_transit_process}")
                    print(f"   📊 状态: {transit_status['phase']} "
                          f"({transit_status['vehicles_in_junction']}/{transit_status['total_vehicles']})")
                    if is_in_transit_process:
                        print(f"   🚧 车道 {participant.data['lane']} 被车队占用")
        
        # 🔥 改进：添加单独车辆参与者（排除车队成员和被占用车道）
        for vehicle in lane_leaders:
            if (vehicle['id'] not in platoon_vehicle_ids and 
                self._vehicle_has_destination(vehicle)):
                
                # 🔥 新增：检查车辆所在车道是否被通过中的车队占用
                vehicle_lane = self._get_vehicle_lane(vehicle)
                if vehicle_lane in lanes_occupied_by_transit_platoons:
                    print(f"🚧 车辆 {vehicle['id']} 在车道 {vehicle_lane} 被通过中的车队占用，暂不参与拍卖")
                    continue
                    
                participant = AuctionParticipant(
                    id=vehicle['id'],
                    type='vehicle',
                    location=vehicle['location'],
                    data=vehicle,
                    at_junction=vehicle.get('is_junction', False)
                )
                participants.append(participant)
        
        print(f"🎯 拍卖参与者识别完成: {len(participants)}个参与者, "
              f"车道占用: {lanes_occupied_by_transit_platoons}")
        
        return participants
    
    def _analyze_platoon_transit_status(self, platoon_vehicles: List[Dict]) -> Dict:
        """分析车队通过状态 - 简化逻辑，移除不必要的past_junction判断"""
        total_vehicles = len(platoon_vehicles)
        vehicles_in_junction = sum(1 for v in platoon_vehicles if v.get('is_junction', False))
        
        # 计算距离路口的距离（用于判断接近状态）
        vehicles_approaching = 0
        for vehicle in platoon_vehicles:
            distance = SimulationConfig.distance_to_intersection_center(vehicle['location'])
            if distance < 30.0 and not vehicle.get('is_junction', False):  # 30米内且未进入路口
                vehicles_approaching += 1
        
        # 简化状态判断逻辑
        if vehicles_in_junction == 0:
            if vehicles_approaching > 0:
                phase = "approaching"  # 接近路口
            else:
                phase = "distant"      # 距离较远
        elif vehicles_in_junction > 0:
            phase = "crossing"         # 正在通过（关键阶段）
        else:
            phase = "unknown"          # 异常状态
        
        # 🔥 核心逻辑：车队在通过过程中需要持续参与拍卖
        in_transit_process = (vehicles_in_junction > 0)  # 简化为：只要有车在路口就是通过状态
        
        return {
            'phase': phase,
            'total_vehicles': total_vehicles,
            'vehicles_in_junction': vehicles_in_junction,
            'vehicles_approaching': vehicles_approaching,
            'in_transit_process': in_transit_process  # 关键字段
        }
    
    def _get_vehicle_lane(self, vehicle: Dict) -> str:
        """获取车辆所在车道标识"""
        try:
            if self.lane_grouper.state_extractor:
                import carla
                location = carla.Location(
                    x=vehicle['location'][0],
                    y=vehicle['location'][1], 
                    z=vehicle['location'][2]
                )
                waypoint = self.lane_grouper.state_extractor.carla.world.get_map().get_waypoint(location)
                return f"road_{waypoint.road_id}_lane_{waypoint.lane_id}"
        except Exception as e:
            print(f"[Warning] 获取车道信息失败 {vehicle['id']}: {e}")
        
        return f"unknown_lane_{vehicle['id']}"

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
        protected_winners = self._get_protected_winners(auction.bids)
        
        # 2. Sort remaining bidders by bid value
        remaining_bids = {k: v for k, v in auction.bids.items() 
                         if k not in [w.participant.id for w in protected_winners]}
        
        regular_winners = self._evaluate_regular_bids(remaining_bids)
        
        # 3. Combine and assign final rankings
        all_winners = protected_winners + regular_winners
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
                protected=False
            )
            winners.append(winner)
        
        return winners
    
    def _is_participant_in_transit(self, participant: AuctionParticipant) -> bool:
        """Check if participant is currently in transit through intersection"""
        if participant.type == 'vehicle':
            return participant.data.get('is_junction', False)
        elif participant.type == 'platoon':
            # Platoon is in transit if any vehicle is in junction
            for vehicle in participant.vehicles:
                if vehicle.get('is_junction', False):
                    return True
        return False
    
    def cleanup_completed_agents(self, vehicle_states: List[Dict], platoon_manager):
        """Clean up agents that have completed transit"""
        current_time = time.time()
        completed_agents = []
        
        for agent_id in list(self.protected_agents):
            agent_still_in_transit = self._check_agent_still_in_transit(
                agent_id, vehicle_states, platoon_manager
            )
            
            # Remove protection if agent completed transit or timed out
            transit_time = current_time - self.agents_in_transit.get(agent_id, {}).get('start_time', current_time)
            
            if not agent_still_in_transit or transit_time > 30.0:
                completed_agents.append(agent_id)
        
        # Clean up completed agents
        for agent_id in completed_agents:
            self.protected_agents.discard(agent_id)
            self.agents_in_transit.pop(agent_id, None)
            print(f"✅ Agent {agent_id} completed transit, protection removed")
    
    def _check_agent_still_in_transit(self, agent_id: str, vehicle_states: List[Dict], 
                                    platoon_manager) -> bool:
        """检查agent是否仍在通过路口 - 优化车队检查逻辑"""
        
        # 单车检查
        if not str(agent_id).startswith('platoon_'):
            for vehicle_state in vehicle_states:
                vehicle_id = str(vehicle_state['id'])
                if vehicle_id == str(agent_id):
                    return vehicle_state.get('is_junction', False)
            return False
        
        # 车队检查 - 关键优化
        if str(agent_id).startswith('platoon_'):
            leader_id = str(agent_id).replace('platoon_', '')
            
            if platoon_manager:
                # 🔥 遍历所有车队，找到包含该leader的车队
                for platoon in platoon_manager.get_all_platoons():
                    leader = platoon.get_leader()
                    if leader and str(leader['id']) == leader_id:
                        # 🚨 关键：只要车队中任何一辆车在路口，整个车队就保持通过状态
                        for vehicle in platoon.vehicles:
                            vehicle_id = str(vehicle['id'])
                            # 在当前车辆状态中查找该车辆
                            for vehicle_state in vehicle_states:
                                if str(vehicle_state['id']) == vehicle_id:
                                    if vehicle_state.get('is_junction', False):
                                        print(f"🚛 车队 {agent_id} 仍在通过: 车辆 {vehicle_id} 在路口")
                                        return True
                
                # 🎯 如果车队中没有车在路口，说明完全通过了
                print(f"✅ 车队 {agent_id} 完成通过: 所有车辆都已离开路口")
                return False
            
            
            # 备用方案：只检查队长
            for vehicle_state in vehicle_states:
                vehicle_id = str(vehicle_state['id'])
                if vehicle_id == leader_id:
                    return vehicle_state.get('is_junction', False)
        
        return False

class DecentralizedAuctionEngine:
    """Main auction engine managing the complete auction process"""
    
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
        
        print("🎯 Refactored Decentralized Auction Engine initialized")
    
    def set_vehicle_enforcer(self, vehicle_enforcer):
        """Set vehicle control enforcer for integration"""
        self.vehicle_enforcer = vehicle_enforcer
    
    def update(self, vehicle_states: List[Dict], platoon_manager) -> List[AuctionWinner]:
        """Main update loop - manages auction lifecycle"""
        current_time = time.time()
        
        # 1. Identify potential participants
        participants = self.participant_identifier.identify_participants(
            vehicle_states, platoon_manager
        )
        
        # 2. Start new auction if needed
        if participants and not self.current_auction:
            self._start_new_auction(participants, current_time)
        
        # 3. Process current auction
        winners = []
        if self.current_auction:
            winners = self._process_current_auction(current_time)
        
        # 4. Clean up completed protected agents
        self.evaluator.cleanup_completed_agents(vehicle_states, platoon_manager)
        
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
        """Convert AuctionParticipant to legacy agent dict format for BidPolicy"""
        agent_dict = {
            'id': participant.id,
            'type': participant.type,
            'location': participant.location,
            'at_junction': participant.at_junction
        }
        
        if participant.type == 'platoon':
            agent_dict.update({
                'vehicles': participant.data['vehicles'],
                'goal_direction': participant.data.get('goal_direction'),
                'size': participant.data.get('size', len(participant.data['vehicles']))
            })
        else:
            # For individual vehicles, ensure 'data' key exists
            agent_dict['data'] = participant.data
        
        return agent_dict
    
    def _print_auction_results(self, auction_id: str, winners: List[AuctionWinner]):
        """Print detailed auction results"""
        if not winners:
            return
        
        print(f"🏆 Auction {auction_id} completed. Priority order:")
        for winner in winners[:5]:  # Show top 5
            participant = winner.participant
            status_emoji = "🏢" if participant.at_junction else "🚦"
            protection_emoji = "🛡️" if winner.protected else ""
            
            if participant.type == 'platoon':
                size = participant.data.get('size', len(participant.vehicles))
                direction = participant.data.get('goal_direction', 'unknown')
                print(f"   #{winner.rank}: {status_emoji}{protection_emoji}🚛 "
                      f"Platoon {participant.id} ({size} vehicles-{direction}) "
                      f"Bid: {winner.bid.value:.1f}")
            else:
                print(f"   #{winner.rank}: {status_emoji}{protection_emoji}🚗 "
                      f"Vehicle {participant.id} Bid: {winner.bid.value:.1f}")
    
    def _broadcast_auction_results(self, auction_id: str, winners: List[AuctionWinner]):
        """Broadcast auction results"""
        self._broadcast_message({
            'type': 'auction_results',
            'auction_id': auction_id,
            'winners': [(w.participant.id, w.bid.value, w.rank) for w in winners[:5]],
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
        """Get comprehensive auction statistics"""
        current_participants = 0
        platoon_count = 0
        vehicle_count = 0
        
        if self.current_auction:
            current_participants = len(self.current_auction.participants)
            for participant in self.current_auction.participants:
                if participant.type == 'platoon':
                    platoon_count += 1
                else:
                    vehicle_count += 1
        
        return {
            'active_auction': self.current_auction is not None,
            'current_participants': current_participants,
            'platoon_participants': platoon_count,
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
