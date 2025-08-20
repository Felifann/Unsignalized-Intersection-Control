import time
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .conflict_analyzer import ConflictAnalyzer
from .mwis_solver import MWISSolver  
from .deadlock_detector import IntersectionDeadlockDetector, DeadlockException

@dataclass
class SimpleAgent:
    """Simple agent representation for Nash solving"""
    id: str
    position: Tuple[float, float]
    speed: float
    heading: float
    intended_path: List[Tuple[float, float]]
    bid: float
    wait_time: float = 5.0

class DeadlockNashSolver:
    """
    Unified Nash equilibrium solver with deadlock detection and prevention.
    Integrates conflict analysis, MWIS solving, and deadlock detection.
    """
    
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0), 
                 max_go_agents: int = None, **kwargs):
        """Initialize Nash solver with configurable parameters"""
        
        # Core configuration
        self.intersection_center = intersection_center
        self.max_go_agents = max_go_agents  # Can be None for no limit
        
        # Solver configuration with defaults
        self.solver_config = {
            'intersection_center': intersection_center,
            'intersection_radius': kwargs.get('intersection_radius', 15.0),
            'min_safe_distance': kwargs.get('min_safe_distance', 4.0),
            'conflict_time_window': kwargs.get('conflict_time_window', 8.0),
            'speed_prediction_horizon': kwargs.get('speed_prediction_horizon', 10.0),
            'max_exact': kwargs.get('max_exact', 12),
            'max_go_agents': max_go_agents,  # Can be None
            'deadlock_core_half_size': kwargs.get('deadlock_core_half_size', 5.0)
        }
        
        # Initialize components
        self.conflict_analyzer = ConflictAnalyzer(self.solver_config)
        self.mwis_solver = MWISSolver(self.solver_config)
        self.deadlock_detector = IntersectionDeadlockDetector(self.solver_config)
        
        # Performance tracking
        self.stats = {
            'resolutions_completed': 0,
            'conflicts_resolved': 0,
            'deadlocks_prevented': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        print(f"🧠 Nash Deadlock Solver initialized:")
        print(f"   📍 Intersection: {intersection_center}")
        print(f"   🚦 Max go agents: {'unlimited' if max_go_agents is None else max_go_agents}")
        print(f"   ⚡ Conflict time window: {self.solver_config['conflict_time_window']}s")
        print(f"   🎯 MWIS threshold: {self.solver_config['max_exact']} vehicles")

    def update_max_go_agents(self, max_go_agents: int = None):
        """Update the maximum go agents limit"""
        self.max_go_agents = max_go_agents
        self.solver_config['max_go_agents'] = max_go_agents
        limit_text = "unlimited" if max_go_agents is None else str(max_go_agents)
        print(f"🔄 Nash solver: Updated MAX_GO_AGENTS to {limit_text}")

    def resolve(self, auction_winners: List, vehicle_states: Dict[str, Dict], 
                platoon_manager=None) -> List:
        """
        Main resolution method with enhanced deadlock handling
        """
        start_time = time.time()
        current_time = start_time
        
        print(f"\n🧠 Nash Conflict Resolution Starting:")
        print(f"   📊 Input: {len(auction_winners)} auction winners")
        print(f"   🚗 Vehicle states: {len(vehicle_states)} vehicles")
        
        try:
            # 1. Check for deadlock first with enhanced detection
            self._check_deadlock_enhanced(vehicle_states, current_time)
            
            # 2. Convert auction winners to candidates
            candidates = self._convert_winners_to_candidates(auction_winners)
            if not candidates:
                print("❌ No valid candidates for Nash resolution")
                return auction_winners
            
            print(f"   🎯 Converted to {len(candidates)} Nash candidates")
            
            # 3. Build conflict graph
            adj, conflict_analysis = self.conflict_analyzer.build_enhanced_conflict_graph(
                candidates, vehicle_states, platoon_manager
            )
            
            # 4. Extract weights (bid values)
            weights = [self._extract_weight(c) for c in candidates]
            
            # 5. Apply MWIS with traffic flow control
            self.mwis_solver.update_traffic_flow_control(vehicle_states, current_time)
            selected_idx = self.mwis_solver.solve_mwis_adaptive(weights, adj, conflict_analysis)
            
            # 6. Assemble winners with strict conflict resolution
            resolved_winners = self.mwis_solver.assemble_winners_with_traffic_control(
                candidates, selected_idx, weights, conflict_analysis, vehicle_states
            )
            
            # 7. Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(candidates), sum(conflict_analysis.values()), processing_time)
            
            print(f"✅ Nash resolution completed in {processing_time:.3f}s")
            print(f"   🟢 GO: {sum(1 for w in resolved_winners if w.conflict_action == 'go')}")
            print(f"   🔴 WAIT: {sum(1 for w in resolved_winners if w.conflict_action == 'wait')}")
            
            return resolved_winners
            
        except DeadlockException as e:
            print(f"🚨 DEADLOCK DETECTED: {e}")
            print(f"   Type: {getattr(e, 'deadlock_type', 'unknown')}")
            print(f"   Affected vehicles: {getattr(e, 'affected_vehicles', 0)}")
            
            self.stats['deadlocks_prevented'] += 1
            
            # Enhanced deadlock resolution based on type
            return self._create_enhanced_deadlock_resolution(auction_winners, e)
            
        except Exception as e:
            print(f"❌ Nash resolution failed: {e}")
            return self._create_conservative_fallback(auction_winners)

    def _check_deadlock_enhanced(self, vehicle_states: Dict[str, Dict], current_time: float):
        """Enhanced deadlock checking with detailed exception info"""
        try:
            if self.deadlock_detector.detect_deadlock(vehicle_states, current_time):
                self.deadlock_detector.handle_deadlock_detection()
        except DeadlockException:
            # Re-raise with additional context
            raise
        except Exception as e:
            print(f"⚠️ Deadlock detection error: {e}")

    def _convert_winners_to_candidates(self, auction_winners: List) -> List:
        """Convert auction winners to Nash solver candidates"""
        candidates = []
        
        for winner in auction_winners:
            # Wrap the auction winner in a candidate structure
            candidate = NashCandidate(
                participant=winner.participant,
                bid=winner.bid,
                rank=winner.rank,
                original_winner=winner
            )
            candidates.append(candidate)
        
        return candidates

    def _extract_weight(self, candidate) -> float:
        """Extract weight (bid value) from candidate"""
        if hasattr(candidate, 'bid') and candidate.bid:
            return candidate.bid.value
        elif hasattr(candidate, 'original_winner') and candidate.original_winner.bid:
            return candidate.original_winner.bid.value
        else:
            return 1.0  # Default weight

    def _create_deadlock_resolution(self, auction_winners: List) -> List:
        """Create deadlock resolution by forcing all agents to wait"""
        resolved_winners = []
        
        for winner in auction_winners:
            # Create new winner with wait action
            resolved_winner = self._copy_winner_with_action(winner, 'wait')
            resolved_winners.append(resolved_winner)
        
        print(f"🚨 Deadlock resolution: All {len(resolved_winners)} agents set to WAIT")
        return resolved_winners

    def _create_conservative_fallback(self, auction_winners: List) -> List:
        """Create conservative fallback when Nash resolution fails"""
        resolved_winners = []
        
        # Sort by rank
        sorted_winners = sorted(auction_winners, key=lambda w: w.rank)
        
        for i, winner in enumerate(sorted_winners):
            # If no limit, or within limit, allow to go
            if self.max_go_agents is None or i < self.max_go_agents:
                action = 'go'
            else:
                action = 'wait'
            
            resolved_winner = self._copy_winner_with_action(winner, action)
            resolved_winners.append(resolved_winner)
        
        go_count = sum(1 for w in resolved_winners if w.conflict_action == 'go')
        wait_count = len(resolved_winners) - go_count
        limit_text = "unlimited" if self.max_go_agents is None else str(self.max_go_agents)
        print(f"⚠️ Conservative fallback: {go_count} agents GO, {wait_count} WAIT (limit: {limit_text})")
        return resolved_winners

    def _copy_winner_with_action(self, original_winner, action: str):
        """Create a copy of winner with specified action"""
        # Import AuctionWinner if available (try package-relative import first), otherwise create simple copy
        try:
            # When running as a package the sibling auction package can be imported relatively
            try:
                from ..auction.auction_engine import AuctionWinner  # type: ignore
            except Exception:
                from auction.auction_engine import AuctionWinner
            return AuctionWinner(
                participant=original_winner.participant,
                bid=original_winner.bid,
                rank=original_winner.rank,
                conflict_action=action
            )
        except Exception:
            # Fallback: modify original winner in-place
            original_winner.conflict_action = action
            return original_winner

    def _update_stats(self, num_candidates: int, num_conflicts: int, processing_time: float):
        """Update performance statistics"""
        self.stats['resolutions_completed'] += 1
        self.stats['conflicts_resolved'] += num_conflicts
        self.stats['total_processing_time'] += processing_time
        self.stats['avg_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['resolutions_completed']
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        combined_stats = dict(self.stats)
        
        # Add component-specific stats
        combined_stats.update({
            'mwis_stats': self.mwis_solver.stats,
            'deadlock_stats': self.deadlock_detector.get_stats(),
            'solver_config': self.solver_config.copy()
        })
        
        return combined_stats

    def reset_stats(self):
        """Reset all performance statistics"""
        self.stats = {
            'resolutions_completed': 0,
            'conflicts_resolved': 0,
            'deadlocks_prevented': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Reset component stats
        if hasattr(self.mwis_solver, 'stats'):
            for key in self.mwis_solver.stats:
                self.mwis_solver.stats[key] = 0
        
        if hasattr(self.deadlock_detector, 'stats'):
            for key in self.deadlock_detector.stats:
                self.deadlock_detector.stats[key] = 0
        
        self.deadlock_detector.reset_history()
        print("🔄 Nash solver: All statistics reset")

    # Integration methods for external systems
    def handle_deadlock(self, agents: List[SimpleAgent], current_time: float) -> Dict[str, str]:
        """Handle deadlock for simple agents (legacy interface)"""
        try:
            # Convert simple agents to vehicle states format for deadlock detection
            vehicle_states = {}
            for agent in agents:
                vehicle_states[agent.id] = {
                    'location': (agent.position[0], agent.position[1], 0.0),
                    'velocity': [
                        agent.speed * math.cos(agent.heading),
                        agent.speed * math.sin(agent.heading),
                        0.0
                    ]
                }
            
            # Check for deadlock
            if self.deadlock_detector.detect_deadlock(vehicle_states, current_time):
                # Return wait actions for all agents
                return {agent.id: 'wait' for agent in agents}
            
            # No deadlock, return go actions
            return {agent.id: 'go' for agent in agents}
            
        except Exception as e:
            print(f"[Warning] Legacy deadlock handling failed: {e}")
            return {agent.id: 'wait' for agent in agents}

@dataclass  
class NashCandidate:
    """Wrapper for auction winners to work with Nash solver"""
    participant: Any
    bid: Any
    rank: int
    original_winner: Any
