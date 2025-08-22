import math
from typing import List, Dict, Tuple, Set, Optional

def _euclidean_2d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Import AuctionWinner for winner creation (try sibling package import first)
try:
    try:
        from ..auction.auction_engine import AuctionWinner  # type: ignore
    except Exception:
        from auction.auction_engine import AuctionWinner
except:
    from dataclasses import dataclass
    @dataclass
    class AuctionWinner:
        participant: object
        bid: object
        rank: int
        conflict_action: str = 'go'

class MWISSolver:
    """
    Part 2: Handles Maximum Weight Independent Set solving and winner assembly
    - Adaptive MWIS algorithm selection
    - Traffic flow control
    - Winner assembly with conflict resolution
    """
    
    def __init__(self, solver_config):
        """Initialize with solver configuration"""
        self.max_exact = solver_config['max_exact']
        self.max_go_agents = solver_config['max_go_agents']
        self.center = solver_config['intersection_center']
        self.deadlock_core_half_size = solver_config.get('deadlock_core_half_size', 10.0)
        self.region_entry_blocked = False
        self.last_entry_block_check = 0
        self.entry_block_check_interval = 1.0
        self.stalled_vehicles_threshold = 3
        self.deadlock_speed_threshold = 0.5
        
        # Performance tracking
        self.stats = {
            'mwis_exact_calls': 0,
            'mwis_greedy_calls': 0,
            'entry_blocks_activated': 0,
            'entry_blocks_released': 0
        }

    def solve_mwis_adaptive(self, weights: List[float], adj: List[Set[int]], 
                           conflict_analysis: Dict) -> List[int]:
        """Adaptive MWIS solver with STRICT conflict handling"""
        n = len(weights)
        if n == 0:
            return []
        
        # Check if there are any conflicts
        total_conflicts = sum(conflict_analysis.values())
        if total_conflicts == 0:
            print("🚀 No conflicts detected - all candidates can proceed")
            # No conflicts, return all candidates sorted by weight
            indexed_weights = [(i, weights[i]) for i in range(n)]
            indexed_weights.sort(key=lambda x: x[1], reverse=True)
            
            selected = [i for i, _ in indexed_weights]
            print(f"✅ No conflicts: selected all {len(selected)} candidates")
            return selected
        
        print(f"⚡ Conflicts detected ({total_conflicts}) - applying STRICT MWIS resolution")
        
        # Use exact solver for small problems, greedy for large ones
        if n <= self.max_exact:
            self.stats['mwis_exact_calls'] += 1
            selected = self._solve_mwis_exact(weights, adj)
            print(f"🎯 Exact MWIS: selected {len(selected)}/{n} candidates")
        else:
            self.stats['mwis_greedy_calls'] += 1
            selected = self._solve_mwis_greedy(weights, adj)
            print(f"🎯 Greedy MWIS: selected {len(selected)}/{n} candidates")
        
        # Verify the solution is actually independent
        if not self._is_independent_set(selected, adj):
            print("❌ WARNING: MWIS solution is not independent! Falling back to single highest bidder")
            # Emergency fallback: select only the highest bidder
            if weights:
                max_idx = max(range(len(weights)), key=lambda i: weights[i])
                selected = [max_idx]
        
        print(f"🔒 STRICT enforcement: {len(selected)} conflict-free candidates selected")
        return selected

    def assemble_winners_with_traffic_control(self, candidates: List, selected_idx: List[int], 
                                             weights: List[float], conflict_analysis: Dict,
                                             vehicle_states: Dict[str, Dict]) -> List:
        """Assemble winners with STRICT conflict resolution - only MWIS winners can GO"""
        if not selected_idx:
            print("❌ No candidates selected by MWIS")
            return []
        
        # Sort ALL candidates by weight (bid value) in descending order for ranking
        all_candidates_with_weights = [(candidates[i], weights[i], i) for i in range(len(candidates))]
        all_candidates_with_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Create set of MWIS selected indices for fast lookup
        mwis_selected = set(selected_idx)
        
        winners = []
        go_count = 0
        
        print(f"🏆 Assembling winners with STRICT conflict resolution:")
        print(f"   📊 Total candidates: {len(all_candidates_with_weights)}")
        print(f"   🎯 MWIS selected: {len(selected_idx)} candidates")
        print(f"   🚦 NO GO LIMIT - All MWIS winners can proceed")
        
        for rank, (candidate, weight, idx) in enumerate(all_candidates_with_weights, 1):
            agent = self._get_agent(candidate)
            
            # STRICT RULE: Only MWIS-selected candidates can get 'go'
            # Note: Removed "in transit" bypass to ensure ALL conflicts are resolved via MWIS
            if idx in mwis_selected:
                # This candidate was selected by MWIS (no conflicts with other selected)
                # Check traffic flow control
                if self.region_entry_blocked and self._should_block_entry(agent, vehicle_states):
                    action = 'wait'
                    reason = "traffic flow control"
                    print(f"   🚧 #{rank}: {getattr(agent, 'type', 'unknown')} {getattr(agent, 'id', 'unknown')}: WAIT ({reason})")
                else:
                    action = 'go'
                    go_count += 1
                    reason = f"MWIS winner #{go_count} (no limit)"
                    print(f"   🟢 #{rank}: {getattr(agent, 'type', 'unknown')} {getattr(agent, 'id', 'unknown')}: GO ({reason})")
            else:
                # This candidate was NOT selected by MWIS (has conflicts)
                action = 'wait'
                reason = "conflict detected"
                print(f"   🔴 #{rank}: {getattr(agent, 'type', 'unknown')} {getattr(agent, 'id', 'unknown')}: WAIT ({reason})")
            
            winner = self._to_winner(candidate, action, rank)
            winners.append(winner)
        
        # Statistics
        go_winners = [w for w in winners if w.conflict_action == 'go']
        wait_winners = [w for w in winners if w.conflict_action == 'wait']
        
        print(f"✅ STRICT conflict resolution completed:")
        print(f"   🟢 GO: {len(go_winners)} agents (no limit)")
        print(f"   🔴 WAIT: {len(wait_winners)} agents")
        print(f"   📈 Conflict resolution rate: {len(selected_idx)}/{len(candidates)} = {len(selected_idx)/len(candidates)*100:.1f}%")
        
        return winners

    def update_traffic_flow_control(self, vehicle_states: Dict[str, Dict], current_time: float):
        """Update traffic flow control based on stalled vehicles in core region"""
        # Only check periodically to avoid excessive computation
        if current_time - self.last_entry_block_check < self.entry_block_check_interval:
            return
        
        self.last_entry_block_check = current_time
        
        # Get vehicles in core region
        core_vehicles = self._get_core_region_vehicles(vehicle_states)
        stalled_count = self._count_stalled_vehicles(core_vehicles)
        
        previous_block_status = self.region_entry_blocked
        
        if stalled_count > self.stalled_vehicles_threshold:
            if not self.region_entry_blocked:
                self.region_entry_blocked = True
                self.stats['entry_blocks_activated'] += 1
                print(f"\n🚫 TRAFFIC FLOW CONTROL ACTIVATED")
                print(f"   🔴 {stalled_count} stalled vehicles in core region (threshold: {self.stalled_vehicles_threshold})")
                print(f"   🚧 Blocking new entries until region clears")
        else:
            if self.region_entry_blocked:
                # Check if all previously stalled vehicles are now moving
                if self._all_stalled_vehicles_recovered(core_vehicles):
                    self.region_entry_blocked = False
                    self.stats['entry_blocks_released'] += 1
                    print(f"\n✅ TRAFFIC FLOW CONTROL RELEASED")
                    print(f"   🟢 Stalled vehicles recovered ({stalled_count} remaining)")
                    print(f"   🚦 Allowing new entries to core region")

    def _solve_mwis_exact(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Exact MWIS solver using brute force for small problems"""
        return self._solve_mwis_brute_force(weights, adj)

    def _solve_mwis_greedy(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Improved greedy MWIS solver with conflict verification"""
        n = len(weights)
        if n == 0:
            return []
        
        # Calculate degree for each vertex
        degrees = [len(neighbors) for neighbors in adj]
        
        # Sort vertices by weight/degree ratio (prefer high weight, low degree)
        vertices = list(range(n))
        vertices.sort(key=lambda i: weights[i] / max(degrees[i], 1), reverse=True)
        
        selected = []
        excluded = set()
        
        print(f"🧮 Greedy MWIS processing {len(vertices)} candidates:")
        
        for v in vertices:
            if v not in excluded:
                selected.append(v)
                # Exclude all neighbors to maintain independence
                neighbors_to_exclude = adj[v]
                excluded.update(neighbors_to_exclude)
                excluded.add(v)  # Mark as processed
                
                print(f"   ✅ Selected candidate {v} (weight: {weights[v]:.1f}, excluded {len(neighbors_to_exclude)} neighbors)")
            else:
                print(f"   ❌ Skipped candidate {v} (weight: {weights[v]:.1f}, conflicts with selected)")
        
        # Verify independence
        if not self._is_independent_set(selected, adj):
            print("❌ ERROR: Greedy solution is not independent!")
            return []
        
        print(f"✅ Greedy MWIS completed: {len(selected)} independent candidates")
        return selected

    def _solve_mwis_brute_force(self, weights: List[float], adj: List[Set[int]]) -> List[int]:
        """Brute force MWIS solver with strict independence verification"""
        n = len(weights)
        best_weight = -1
        best_set = []
        
        print(f"🔍 Brute force MWIS: checking {2**n} possible combinations")
        
        # Try all possible subsets
        for mask in range(1 << n):
            subset = [i for i in range(n) if mask & (1 << i)]
            
            # Check if subset is independent (no conflicts)
            if self._is_independent_set(subset, adj):
                weight = sum(weights[i] for i in subset)
                if weight > best_weight:
                    best_weight = weight
                    best_set = subset
        
        print(f"🎯 Brute force result: {len(best_set)} candidates, total weight: {best_weight:.1f}")
        return best_set

    def _is_independent_set(self, subset: List[int], adj: List[Set[int]]) -> bool:
        """Strictly verify that a subset is an independent set (no conflicts)"""
        if not subset:
            return True
        
        # Check every pair in the subset for conflicts
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                vertex_i = subset[i]
                vertex_j = subset[j]
                
                # If vertex_j is in the adjacency list of vertex_i, there's a conflict
                if vertex_j in adj[vertex_i]:
                    return False
        
        return True

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

    def _all_stalled_vehicles_recovered(self, current_vehicles: List[Dict]) -> bool:
        """Check if all previously stalled vehicles have recovered to normal movement"""
        stalled_count = self._count_stalled_vehicles(current_vehicles)
        
        # Consider recovery successful if stalled count is below threshold
        # and there's evidence of movement in the region
        if stalled_count <= self.stalled_vehicles_threshold:
            # Additional check: ensure vehicles are actually moving
            moving_vehicles = 0
            for vehicle in current_vehicles:
                velocity = vehicle.get('velocity', [0, 0, 0])
                speed = math.sqrt(velocity[0]**2 + velocity[1]**2) if velocity else 0.0
                if speed >= self.deadlock_speed_threshold:
                    moving_vehicles += 1
            
            # Release block if we have fewer stalled vehicles AND some movement
            return stalled_count == 0 or moving_vehicles > 0
        
        return False

    def _should_block_entry(self, agent, vehicle_states: Dict[str, Dict]) -> bool:
        """Check if agent should be blocked from entering core region"""
        try:
            agent_id = str(getattr(agent, 'id', agent))
            state = vehicle_states.get(agent_id)
            
            if not state or 'location' not in state:
                return False
            
            location = state['location']
            center = self.center
            
            # Calculate distance to intersection center
            distance = _euclidean_2d(location, center)
            
            # Block if agent is outside core region and trying to enter
            return distance > self.deadlock_core_half_size * 1.5
            
        except Exception:
            return False

    def _get_agent(self, candidate) -> object:
        """Extract agent from candidate"""
        if hasattr(candidate, 'participant'):
            return candidate.participant
        elif hasattr(candidate, 'agent'):
            return candidate.agent
        else:
            return candidate

    def _to_winner(self, candidate, action: str, rank: int):
        """Convert candidate to AuctionWinner"""
        agent = self._get_agent(candidate)
        bid = getattr(candidate, 'bid', None)
        
        return AuctionWinner(
            participant=agent,
            bid=bid,
            rank=rank,
            conflict_action=action
        )
    
    def _is_agent_in_transit(self, agent, vehicle_states: Dict[str, Dict]) -> bool:
        """Check if agent is currently in transit through intersection"""
        try:
            agent_id = str(getattr(agent, 'id', agent))
            
            # For vehicle agents
            if hasattr(agent, 'type') and agent.type == 'vehicle':
                state = vehicle_states.get(agent_id)
                return state and state.get('is_junction', False)
            
            # For platoon agents
            elif hasattr(agent, 'type') and agent.type == 'platoon':
                vehicles = getattr(agent, 'vehicles', [])
                if hasattr(agent, 'data') and 'vehicles' in agent.data:
                    vehicles = agent.data['vehicles']
                
                for vehicle in vehicles:
                    vehicle_id = str(vehicle.get('id', vehicle.get('vehicle_id')))
                    state = vehicle_states.get(vehicle_id)
                    if state and state.get('is_junction', False):
                        return True
            
            return False
            
        except Exception as e:
            print(f"[Warning] Transit check failed for agent {agent}: {e}")
            return False
