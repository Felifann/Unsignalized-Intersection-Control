# auction/deadlock_nash.py
import itertools
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math
from env.simulation_config import SimulationConfig

@dataclass
class SimpleAgent:
    id: str
    position: Tuple[float, float]         # (x,y)
    speed: float                          # m/s
    heading: float                        # rad
    intended_path: List[Tuple[float,float]]  # coarse polyline through intersection
    bid: float
    wait_time: float                      # seconds
    last_positions: List[Tuple[float,float]] = field(default_factory=list)

@dataclass
class NashPassingOrder:
    """Represents a sequential passing order determined by Nash equilibrium"""
    sequence: List[List[str]]  # List of groups, each group can go simultaneously
    current_group_index: int = 0
    group_start_time: float = 0.0
    min_group_duration: float = 3.0  # Minimum time each group gets to pass

@dataclass
class DeadlockState:
    """Represents the current deadlock state"""
    is_active: bool = False
    participants: List[str] = field(default_factory=list)
    detection_time: float = 0.0
    resolution_order: List[List[str]] = field(default_factory=list)
    current_group_index: int = 0
    group_start_time: float = 0.0

class DeadlockNashController:
    def __init__(self,
                 intersection_polygon,   
                 deadlock_time_window: float = 2.0,
                 min_agents_for_deadlock: int = 2,
                 progress_eps: float = 1.0,   
                 collision_penalty: float = 1000.0,
                 wait_penalty_allwait: float = 10.0,
                 w_wait_inv: float = 1.0,
                 w_bid: float = 1.0,
                 mutual_blocking_distance: float = 4.0,
                 group_passing_duration: float = 4.0):
        self.intersection_polygon = intersection_polygon
        self.deadlock_time_window = deadlock_time_window
        self.min_agents = min_agents_for_deadlock
        self.progress_eps = progress_eps
        self.collision_penalty = collision_penalty
        self.wait_penalty_allwait = wait_penalty_allwait
        self.w_wait_inv = w_wait_inv
        self.w_bid = w_bid
        self.mutual_blocking_distance = mutual_blocking_distance
        self.group_passing_duration = group_passing_duration
        
        # Store history and current passing order
        self.history = {}
        # Replace current_passing_order with deadlock_state
        self.deadlock_state = DeadlockState()
        self.last_nash_resolution_time = 0.0
        self.nash_resolution_cooldown = 2.0  # Minimum time between Nash resolutions

    # ---------- Utility Functions ----------
    def _in_intersection(self, pos: Tuple[float,float]) -> bool:
        """Check if position is inside intersection"""
        # Use the intersection polygon passed to constructor
        # if isinstance(self.intersection_polygon, tuple) and len(self.intersection_polygon) == 4:
        #     x_min, x_max, y_min, y_max = self.intersection_polygon
        #     x, y = pos
        #     return x_min <= x <= x_max and y_min <= y <= y_max
        # else:
        #     Fallback to SimulationConfig
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        center_x, center_y = center[0], center[1]
        half_side = 8.0  # 16m side, so half is 8m
        x, y = pos
        return (center_x - half_side <= x <= center_x + half_side and
                center_y - half_side <= y <= center_y + half_side)

    def update_vehicle_history(self, agent: SimpleAgent, current_time: float):
        """Update vehicle position history for deadlock detection"""
        h = self.history.setdefault(agent.id, [])
        h.append((current_time, agent.position))
        # keep only window
        cutoff = current_time - self.deadlock_time_window
        self.history[agent.id] = [(t,p) for (t,p) in h if t >= cutoff]

    def _progress_in_window(self, agent_id: str) -> float:
        """Calculate progress distance in time window"""
        h = self.history.get(agent_id, [])
        if len(h) < 2:
            return float('inf')
        start_pos = h[0][1]
        end_pos = h[-1][1]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        return math.hypot(dx, dy)

    def paths_conflict(self, path1: List[Tuple[float,float]], path2: List[Tuple[float,float]]) -> bool:
        """Check if two paths conflict using segment distance analysis"""
        if not path1 or not path2 or len(path1) < 2 or len(path2) < 2:
            return False

        def segments(path):
            return list(zip(path[:-1], path[1:]))
        
        def point_to_segment_distance(pt, seg):
            (x,y) = pt
            (x1,y1),(x2,y2) = seg
            dx = x2-x1; dy=y2-y1
            if dx==0 and dy==0:
                return math.hypot(x-x1,y-y1)
            t = max(0,min(1, ((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy)))
            px = x1 + t*dx; py = y1 + t*dy
            return math.hypot(px-x,py-y)

        segs1 = segments(path1)
        segs2 = segments(path2)
        conflict_threshold = 2.0  # meters
        
        for s1 in segs1:
            for s2 in segs2:
                if (point_to_segment_distance(s1[0], s2) < conflict_threshold or
                    point_to_segment_distance(s1[1], s2) < conflict_threshold or
                    point_to_segment_distance(s2[0], s1) < conflict_threshold or
                    point_to_segment_distance(s2[1], s1) < conflict_threshold):
                    return True
        return False

    # ---------- Unified Deadlock Detection ----------
    def detect_deadlock(self, agents: List[SimpleAgent], current_time: float) -> List[SimpleAgent]:
        """Unified deadlock detection for 2-4 vehicles"""
        # 1) Only consider agents inside intersection
        inside = [a for a in agents if self._in_intersection(a.position)]
        if len(inside) < self.min_agents:
            # Clean history for non-inside agents
            for a in agents:
                if a.id not in [ia.id for ia in inside]:
                    self.history.pop(a.id, None)
            return []

        print(f"DEBUG: {len(inside)} agents inside intersection")

        # 2) Update history for inside agents
        for a in inside:
            self.update_vehicle_history(a, current_time)

        # 3) Check stalled agents with more lenient criteria
        stalled = []
        for a in inside:
            progress = self._progress_in_window(a.id)
            
            # More lenient stalling criteria
            history_length = len(self.history.get(a.id, []))
            
            # Consider stalled if:
            # - Very low speed (< 0.1 m/s) OR
            # - Low progress AND sufficient history
            is_stalled = (
                a.speed < 0.1 or  # Almost stopped
                (progress < self.progress_eps and history_length >= 3)  # Low progress with history
            )
            
            print(f"DEBUG: Agent {a.id} - speed: {a.speed:.3f}, progress: {progress:.3f}, history: {history_length}, stalled: {is_stalled}")
            
            if is_stalled:
                stalled.append(a)

        print(f"DEBUG: {len(stalled)} stalled agents out of {len(inside)} inside")

        # 4) For 2 vehicles, use more lenient mutual blocking check
        if len(stalled) >= 2:
            if len(stalled) == 2:
                # 2-vehicle check
                if self._are_mutually_blocking(stalled[0], stalled[1]):
                    print(f"🔒 2-vehicle deadlock detected: {stalled[0].id} ↔ {stalled[1].id}")
                    return stalled
            else:
                # 3+ vehicles: Check if they form a blocking cycle
                if self._forms_blocking_cycle(stalled):
                    print(f"🔒 Multi-vehicle deadlock: {len(stalled)} agents")
                    return stalled[:4]  # Limit to 4 for computational efficiency

        # If vehicles are very close and both stopped, force deadlock detection
        if len(inside) >= 2:
            close_and_stopped = self._check_close_and_stopped(inside)
            if close_and_stopped:
                print(f"🔒 FORCED deadlock due to close proximity and low speeds")
                return close_and_stopped

        return []

    def _check_close_and_stopped(self, agents: List[SimpleAgent]) -> List[SimpleAgent]:
        """Check if vehicles are very close together and mostly stopped"""
        stopped_agents = [a for a in agents if a.speed < 0.5]  # Very slow or stopped
        
        if len(stopped_agents) < 2:
            return []
        
        # Check if any two agents are very close
        for i in range(len(stopped_agents)):
            for j in range(i+1, len(stopped_agents)):
                agent1, agent2 = stopped_agents[i], stopped_agents[j]
                distance = math.hypot(
                    agent1.position[0] - agent2.position[0],
                    agent1.position[1] - agent2.position[1]
                )
                
                print(f"DEBUG: Distance between {agent1.id} and {agent2.id}: {distance:.2f}m")
                
                # If very close (crash distance), force deadlock
                if distance < 8.0:  # Increased threshold for crashed vehicles
                    return stopped_agents[:2]  # Return the close pair
    
        return []

    def _are_mutually_blocking(self, agent1: SimpleAgent, agent2: SimpleAgent) -> bool:
        """Enhanced mutual blocking check"""
        # Distance check
        dx = agent1.position[0] - agent2.position[0]
        dy = agent1.position[1] - agent2.position[1]
        distance = math.hypot(dx, dy)
        
        print(f"DEBUG: Mutual blocking check - distance: {distance:.2f}m (threshold: {self.mutual_blocking_distance})")
        
        if distance > self.mutual_blocking_distance * 1.5:  # More lenient distance
            return False
        
        # If vehicles are very close, assume they're blocking regardless of other factors
        if distance < 3.0:
            print(f"DEBUG: Vehicles very close ({distance:.2f}m) - assuming mutual blocking")
            return True
        
        # Path conflict check - if paths available
        if (len(agent1.intended_path) >= 2 and len(agent2.intended_path) >= 2):
            paths_conflict = self.paths_conflict(agent1.intended_path, agent2.intended_path)
            print(f"DEBUG: Paths conflict: {paths_conflict}")
            if not paths_conflict:
                # Even without explicit path conflict, close proximity might indicate blocking
                if distance < 5.0:
                    return True
                return False
        else:
            # No valid paths - assume conflict if close
            return distance < 6.0
        
        # Heading check - more tolerant
        heading_diff = abs(agent1.heading - agent2.heading)
        heading_diff = min(heading_diff, 2*math.pi - heading_diff)
        
        head_on = abs(heading_diff - math.pi) < 1.0  # More tolerant
        perpendicular = abs(heading_diff - math.pi/2) < 1.0  # More tolerant
        same_direction = heading_diff < 0.5
        
        print(f"DEBUG: Heading diff: {heading_diff:.2f}, head_on: {head_on}, perpendicular: {perpendicular}")
        
        return head_on or perpendicular or (distance < 4.0 and same_direction)

    def _forms_blocking_cycle(self, agents: List[SimpleAgent]) -> bool:
        """Check if multiple agents form a blocking cycle"""
        if len(agents) < 3:
            return False
        
        # If all agents are close and slow, assume blocking cycle
        max_distance = 0
        min_speed = float('inf')
        
        for i in range(len(agents)):
            min_speed = min(min_speed, agents[i].speed)
            for j in range(i+1, len(agents)):
                distance = math.hypot(
                    agents[i].position[0] - agents[j].position[0],
                    agents[i].position[1] - agents[j].position[1]
                )
                max_distance = max(max_distance, distance)
        
        # If all vehicles are within a small area and moving slowly
        compact_and_slow = max_distance < 10.0 and min_speed < 0.5
        
        if compact_and_slow:
            print(f"DEBUG: Compact group detected - max_distance: {max_distance:.2f}, min_speed: {min_speed:.2f}")
            return True
        
        # Original path-based conflict check as fallback
        conflict_count = 0
        total_pairs = len(agents) * (len(agents) - 1) // 2
        
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                if self.paths_conflict(agents[i].intended_path, agents[j].intended_path):
                    conflict_count += 1
        
        # Threshold: if more than 60% of pairs conflict, consider it a cycle
        conflict_ratio = conflict_count / total_pairs if total_pairs > 0 else 0
        return conflict_ratio > 0.6

    # ---------- Enhanced Payoff Computation ----------
    def compute_payoffs(self, agents: List[SimpleAgent], profile: Tuple[int, ...]) -> Dict[str,float]:
        """Enhanced payoff computation with fairness consideration"""
        n = len(agents)
        payoffs = {a.id: 0.0 for a in agents}
        
        # Precompute conflicts
        conflicts = self._precompute_conflicts(agents)
        
        # Compute individual payoffs
        for i, agent in enumerate(agents):
            action = profile[i]
            
            if action == 0:  # Wait
                # Waiting penalty increases with wait time
                payoffs[agent.id] = -0.1 * agent.wait_time - 0.05 * (agent.wait_time ** 1.2)
            else:  # Go
                # Base reward with diminishing returns for very high bids
                bid_reward = self.w_bid * math.log(1 + agent.bid)
                wait_reward = self.w_wait_inv * (1.0/(1.0 + agent.wait_time))
                base_reward = bid_reward + wait_reward
                
                # Collision penalty
                collision_count = sum(1 for j in range(n) 
                                    if j != i and profile[j] == 1 and conflicts.get((min(i,j), max(i,j)), False))
                
                penalty = self.collision_penalty * collision_count
                payoffs[agent.id] = base_reward - penalty

        # Global penalties
        if all(x == 0 for x in profile):  # Everyone waits
            for agent in agents:
                payoffs[agent.id] -= self.wait_penalty_allwait
        elif sum(profile) == n:  # Everyone goes - discourage if many conflicts
            total_conflicts = sum(conflicts.values())
            if total_conflicts > n//2:  # Too many conflicts
                for i, agent in enumerate(agents):
                    if profile[i] == 1:
                        payoffs[agent.id] -= 50.0  # Additional penalty

        return payoffs

    def _precompute_conflicts(self, agents: List[SimpleAgent]) -> Dict[Tuple[int,int], bool]:
        """Precompute all pairwise conflicts"""
        conflicts = {}
        n = len(agents)
        for i in range(n):
            for j in range(i+1, n):
                conflicts[(i,j)] = self.paths_conflict(agents[i].intended_path, agents[j].intended_path)
        return conflicts

    # ---------- Robust Nash Equilibrium Finding ----------
    def find_pure_nash(self, agents: List[SimpleAgent]) -> List[Tuple[int,...]]:
        """Find pure strategy Nash equilibria with robustness checks"""
        n = len(agents)
        if n == 0:
            return []
        
        # For large n, use heuristic approach
        if n > 4:
            return self._heuristic_nash_for_large_groups(agents)
            
        all_profiles = list(itertools.product([0,1], repeat=n))
        nash_equilibria = []
        
        for profile in all_profiles:
            if self._is_nash_equilibrium(agents, profile):
                nash_equilibria.append(profile)
        
        return nash_equilibria

    def _is_nash_equilibrium(self, agents: List[SimpleAgent], profile: Tuple[int,...]) -> bool:
        """Check if a profile is a Nash equilibrium"""
        payoffs = self.compute_payoffs(agents, profile)
        
        for i in range(len(agents)):
            current_payoff = payoffs[agents[i].id]
            
            # Try deviating
            deviation_profile = list(profile)
            deviation_profile[i] = 1 - profile[i]
            deviation_payoffs = self.compute_payoffs(agents, tuple(deviation_profile))
            deviation_payoff = deviation_payoffs[agents[i].id]
            
            # If deviation improves payoff significantly, not Nash
            if deviation_payoff > current_payoff + 1e-6:
                return False
        
        return True

    def _heuristic_nash_for_large_groups(self, agents: List[SimpleAgent]) -> List[Tuple[int,...]]:
        """Heuristic Nash finding for large groups (n > 4)"""
        n = len(agents)
        
        # Strategy 1: Sort by priority and allow top agents to go
        sorted_agents = sorted(enumerate(agents), 
                             key=lambda x: x[1].bid / (1 + x[1].wait_time), reverse=True)
        
        profiles = []
        
        # Try allowing 1, 2, or 3 top agents to go
        for num_go in range(1, min(4, n+1)):
            profile = [0] * n
            for i in range(num_go):
                agent_idx = sorted_agents[i][0]
                profile[agent_idx] = 1
            
            # Check if this creates too many conflicts
            conflicts = self._precompute_conflicts(agents)
            conflict_count = sum(1 for (i,j), has_conflict in conflicts.items() 
                               if has_conflict and profile[i] == 1 and profile[j] == 1)
            
            if conflict_count <= 1:  # Allow at most 1 conflict
                profiles.append(tuple(profile))
        
        return profiles

    # ---------- Unified Nash Resolution ----------
    def resolve_deadlock(self, agents: List[SimpleAgent]) -> Dict[str,str]:
        """Unified deadlock resolution for 2-4+ vehicles"""
        if not agents:
            return {}
        
        n = len(agents)
        print(f"🎯 Resolving {n}-agent deadlock using Nash equilibrium")
        
        # Find Nash equilibria
        nash_profiles = self.find_pure_nash(agents)
        
        # Selection strategy
        chosen_profile = self._select_best_nash_profile(agents, nash_profiles)
        
        if chosen_profile is None:
            # Fallback: Greedy selection based on priority
            chosen_profile = self._greedy_fallback(agents)
        
        # Convert to action dictionary
        result = {}
        for i, agent in enumerate(agents):
            result[agent.id] = 'go' if chosen_profile[i] == 1 else 'wait'
        
        return result

    def _select_best_nash_profile(self, agents: List[SimpleAgent], 
                                 nash_profiles: List[Tuple[int,...]]) -> Optional[Tuple[int,...]]:
        """Select the best Nash profile based on multiple criteria"""
        if not nash_profiles:
            return None
        
        # Filter collision-free profiles
        safe_profiles = [p for p in nash_profiles if not self._profile_has_collisions(agents, p)]
        
        profiles_to_consider = safe_profiles if safe_profiles else nash_profiles
        
        # Multi-criteria selection
        best_profile = None
        best_score = -float('inf')
        
        for profile in profiles_to_consider:
            score = self._evaluate_profile_quality(agents, profile)
            if score > best_score:
                best_score = score
                best_profile = profile
        
        return best_profile

    def _evaluate_profile_quality(self, agents: List[SimpleAgent], profile: Tuple[int,...]) -> float:
        """Evaluate profile quality using multiple criteria"""
        payoffs = self.compute_payoffs(agents, profile)
        
        # Criteria 1: Social welfare (total payoff)
        social_welfare = sum(payoffs.values())
        
        # Criteria 2: Fairness (negative variance of payoffs)
        payoff_values = list(payoffs.values())
        mean_payoff = sum(payoff_values) / len(payoff_values)
        variance = sum((p - mean_payoff)**2 for p in payoff_values) / len(payoff_values)
        fairness_score = -variance
        
        # Criteria 3: Efficiency (number of agents that can go)
        efficiency_score = sum(profile) * 10
        
        # Criteria 4: Safety (negative collision count)
        conflicts = self._precompute_conflicts(agents)
        collision_count = sum(1 for (i,j), has_conflict in conflicts.items() 
                            if has_conflict and profile[i] == 1 and profile[j] == 1)
        safety_score = -collision_count * 100
        
        # Combined score
        total_score = social_welfare + 0.3 * fairness_score + efficiency_score + safety_score
        return total_score

    def _greedy_fallback(self, agents: List[SimpleAgent]) -> Tuple[int,...]:
        """Greedy fallback when no good Nash equilibrium found"""
        n = len(agents)
        
        # Sort by priority (bid/wait_time ratio)
        sorted_indices = sorted(range(n), 
                              key=lambda i: agents[i].bid / (1 + agents[i].wait_time), reverse=True)
        
        profile = [0] * n
        conflicts = self._precompute_conflicts(agents)
        
        # Greedily assign 'go' to highest priority agents without conflicts
        for i in sorted_indices:
            # Check if agent i conflicts with any already assigned 'go' agents
            has_conflict = any(profile[j] == 1 and conflicts.get((min(i,j), max(i,j)), False) 
                             for j in range(n))
            
            if not has_conflict:
                profile[i] = 1
                
                # Limit concurrent 'go' agents for safety
                if sum(profile) >= min(3, n):
                    break
        
        return tuple(profile)

    def _profile_has_collisions(self, agents: List[SimpleAgent], profile: Tuple[int,...]) -> bool:
        """Check if profile has collisions"""
        conflicts = self._precompute_conflicts(agents)
        
        for (i,j), has_conflict in conflicts.items():
            if has_conflict and profile[i] == 1 and profile[j] == 1:
                return True
        return False

    # ---------- Main Entry Point ----------
    def handle_deadlock(self, agents: List[SimpleAgent], current_time: float) -> Dict[str,str]:
        """Main entry point - pause system during deadlock until all participants pass"""
        
        # If deadlock is active, manage the resolution process
        if self.deadlock_state.is_active:
            print(f"🔒 DEADLOCK ACTIVE - Managing resolution (Group {self.deadlock_state.current_group_index + 1}/{len(self.deadlock_state.resolution_order)})")
            return self._manage_active_deadlock(agents, current_time)
        
        # Check for new deadlock
        deadlocked_agents = self.detect_deadlock(agents, current_time)
        if not deadlocked_agents:
            return {}  # No deadlock, allow normal operation
        
        # Apply cooldown to prevent rapid Nash re-computation
        if current_time - self.last_nash_resolution_time < self.nash_resolution_cooldown:
            print(f"🕐 DEADLOCK COOLDOWN - Waiting {self.nash_resolution_cooldown - (current_time - self.last_nash_resolution_time):.1f}s")
            return {}
            
        print(f"🚨 NEW DEADLOCK DETECTED - PAUSING SYSTEM")
        print(f"   Participants: {[a.id for a in deadlocked_agents]}")
        print(f"   Positions: {[(a.id, f'({a.position[0]:.1f}, {a.position[1]:.1f})') for a in deadlocked_agents]}")
        print(f"   Speeds: {[(a.id, f'{a.speed:.2f}m/s') for a in deadlocked_agents]}")
        
        # Initialize deadlock state and create resolution order
        self._initialize_deadlock_resolution(deadlocked_agents, current_time)
        
        # Return initial actions for deadlock resolution
        return self._manage_active_deadlock(agents, current_time)

    def _initialize_deadlock_resolution(self, deadlocked_agents: List[SimpleAgent], current_time: float):
        """Initialize deadlock resolution state"""
        print(f"🎯 Creating Nash-based resolution order for {len(deadlocked_agents)} agents")
        
        # Find Nash equilibria and create resolution order
        nash_profiles = self.find_pure_nash(deadlocked_agents)
        chosen_profile = self._select_best_nash_profile(deadlocked_agents, nash_profiles)
        if chosen_profile is None:
            chosen_profile = self._greedy_fallback(deadlocked_agents)
        
        # Create sequential resolution order
        resolution_order = self._convert_profile_to_sequence(deadlocked_agents, chosen_profile)
        
        # Initialize deadlock state
        self.deadlock_state = DeadlockState(
            is_active=True,
            participants=[agent.id for agent in deadlocked_agents],
            detection_time=current_time,
            resolution_order=resolution_order,
            current_group_index=0,
            group_start_time=current_time
        )
        
        self.last_nash_resolution_time = current_time
        
        print(f"📋 Deadlock resolution order created:")
        for i, group in enumerate(resolution_order):
            print(f"   Group {i+1}: {group}")
        print(f"🔒 SYSTEM PAUSED - Only deadlock participants can proceed")

    def _manage_active_deadlock(self, agents: List[SimpleAgent], current_time: float) -> Dict[str,str]:
        """Manage active deadlock resolution"""
        if not self.deadlock_state.is_active:
            return {}
        
        # Check if all deadlock participants have passed through
        if self._all_participants_have_passed(agents):
            print("✅ All deadlock participants have passed - RESUMING SYSTEM")
            self.deadlock_state = DeadlockState()  # Reset to inactive
            return {}  # Allow normal operation to resume
        
        # Execute current group in resolution order
        actions = self._execute_deadlock_resolution_step(agents, current_time)
        
        # CRITICAL: Pause all non-participant agents
        return self._apply_system_pause_for_non_participants(agents, actions)

    def _all_participants_have_passed(self, agents: List[SimpleAgent]) -> bool:
        """Check if all deadlock participants have passed through the intersection"""
        # Get current agent IDs in the system
        current_agent_ids = {a.id for a in agents}
        
        # Check each participant
        for participant_id in self.deadlock_state.participants:
            if participant_id in current_agent_ids:
                # Find the agent
                participant_agent = next(a for a in agents if a.id == participant_id)
                
                # If still in intersection, not all have passed
                if self._in_intersection(participant_agent.position):
                    return False
        
        # All participants have either left the intersection or left the system
        return True

    def _execute_deadlock_resolution_step(self, agents: List[SimpleAgent], current_time: float) -> Dict[str,str]:
        """Execute current step of deadlock resolution with fallback for wait agents"""
        if not self.deadlock_state.resolution_order:
            return {}
        
        # Get current group
        if self.deadlock_state.current_group_index >= len(self.deadlock_state.resolution_order):
            return self._wait_for_intersection_clearance(agents)
        
        current_group = self.deadlock_state.resolution_order[self.deadlock_state.current_group_index]
        
        # Check if current group has COMPLETELY cleared the intersection
        current_group_in_intersection = [
            agent_id for agent_id in current_group 
            if any(a.id == agent_id and self._in_intersection(a.position) for a in agents)
        ]
        
        # IMMEDIATE SWITCHING: Advance as soon as current group clears intersection
        group_elapsed = current_time - self.deadlock_state.group_start_time
        should_advance = (
            len(current_group_in_intersection) == 0 or  # All cleared immediately
            group_elapsed >= 2.0  # Safety timeout (reduced from 4.0)
        )
        
        if should_advance and self.deadlock_state.current_group_index < len(self.deadlock_state.resolution_order) - 1:
            self.deadlock_state.current_group_index += 1
            self.deadlock_state.group_start_time = current_time
            current_group = self.deadlock_state.resolution_order[self.deadlock_state.current_group_index]
            print(f"🚦 Deadlock resolution: IMMEDIATELY advancing to group {self.deadlock_state.current_group_index + 1}")
        
        # Generate actions for current group with fallback logic
        actions = {}
        go_agents = []
        wait_agents = []
        
        for agent in agents:
            if agent.id in self.deadlock_state.participants:
                if agent.id in current_group:
                    actions[agent.id] = 'go'
                    go_agents.append(agent.id)
                else:
                    # Check if wait agent needs to fall back
                    needs_fallback = self._agent_needs_fallback(agent, agents, current_group)
                    if needs_fallback:
                        actions[agent.id] = 'fallback'
                        print(f"🔙 Agent {agent.id} falling back to clear path")
                    else:
                        actions[agent.id] = 'wait'
                wait_agents.append(agent.id)
        
        print(f"🎮 Group {self.deadlock_state.current_group_index + 1} actions:")
        print(f"   GO: {go_agents}")
        print(f"   WAIT/FALLBACK: {wait_agents}")
        print(f"   Still in intersection: {current_group_in_intersection}")
        
        return actions

    def _agent_needs_fallback(self, wait_agent: SimpleAgent, all_agents: List[SimpleAgent], 
                         current_go_group: List[str]) -> bool:
        """Determine if a waiting agent needs to fall back to clear the path"""
        # Find agents in current go group
        go_agents = [a for a in all_agents if a.id in current_go_group]
        
        if not go_agents:
            return False
        
        # Check if wait agent is blocking any go agent
        for go_agent in go_agents:
            # Calculate distance between wait agent and go agent
            distance = math.hypot(
                wait_agent.position[0] - go_agent.position[0],
                wait_agent.position[1] - go_agent.position[1]
            )
            
            # If they're close (within blocking distance), wait agent should fall back
            if distance < 8.0:  # 8 meters blocking threshold
                print(f"🚧 Agent {wait_agent.id} blocking {go_agent.id} (distance: {distance:.1f}m)")
                return True
        
        # Check if wait agent is in intersection and could block the path
        if self._in_intersection(wait_agent.position):
            print(f"🚧 Agent {wait_agent.id} in intersection - needs fallback")
            return True
        
        return False

    def _wait_for_intersection_clearance(self, agents: List[SimpleAgent]) -> Dict[str,str]:
        """Wait for all participants to clear intersection after resolution"""
        actions = {}
        
        # Allow all remaining participants to continue
        for agent in agents:
            if agent.id in self.deadlock_state.participants:
                actions[agent.id] = 'go'
        
        return actions

    def _apply_system_pause_for_non_participants(self, agents: List[SimpleAgent], 
                                               participant_actions: Dict[str,str]) -> Dict[str,str]:
        """Apply system-wide pause: only deadlock participants can act"""
        all_actions = {}
        
        for agent in agents:
            if agent.id in self.deadlock_state.participants:
                # Use resolution actions for participants
                all_actions[agent.id] = participant_actions.get(agent.id, 'wait')
            else:
                # PAUSE: All non-participants must wait
                all_actions[agent.id] = 'wait'
        
        return all_actions

    def _convert_profile_to_sequence(self, agents: List[SimpleAgent], profile: Tuple[int,...]) -> List[List[str]]:
        """Convert Nash profile to sequential passing order"""
        if not profile or len(profile) != len(agents):
            # Fallback: sequential order
            return [[agent.id] for agent in agents]
        
        # Group agents by their actions
        go_agents = []
        wait_agents = []
        
        for i, agent in enumerate(agents):
            if profile[i] == 1:  # Go
                go_agents.append(agent.id)
            else:  # Wait
                wait_agents.append(agent.id)
        
        # Create resolution order
        resolution_order = []
        
        # Add go agents as first group (they can proceed immediately)
        if go_agents:
            resolution_order.append(go_agents)
        
        # Add wait agents as subsequent groups
        # For safety, split wait agents into smaller groups if many
        if wait_agents:
            # Split into groups of max 2 agents for safety
            for i in range(0, len(wait_agents), 2):
                group = wait_agents[i:i+2]
                resolution_order.append(group)
        
        # Ensure we have at least one group
        if not resolution_order:
            resolution_order = [[agent.id for agent in agents]]
        
        return resolution_order
