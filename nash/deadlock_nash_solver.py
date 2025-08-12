# auction/deadlock_nash.py
import itertools
import time
from typing import List, Dict, Tuple
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

class DeadlockNashController:
    def __init__(self,
                 intersection_polygon,   # shapely polygon or simple bounding box (xmin,xmax,ymin,ymax)
                 deadlock_time_window: float = 3.0,
                 min_agents_for_deadlock: int = 3,
                 progress_eps: float = 0.5,   # meters
                 collision_penalty: float = 1000.0,
                 wait_penalty_allwait: float = 10.0,
                 w_wait_inv: float = 1.0,
                 w_bid: float = 1.0):
        self.intersection_polygon = intersection_polygon
        self.deadlock_time_window = deadlock_time_window
        self.min_agents = min_agents_for_deadlock
        self.progress_eps = progress_eps
        self.collision_penalty = collision_penalty
        self.wait_penalty_allwait = wait_penalty_allwait
        self.w_wait_inv = w_wait_inv
        self.w_bid = w_bid
        # store history: agent_id -> list of (time, position)
        self.history = {}

    # ---------- Utility ----------
    def _in_intersection(self, pos: Tuple[float,float]) -> bool:
        """Check if position is inside intersection (square, 16m side, centered at TARGET_INTERSECTION_CENTER)"""
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

    # ---------- Deadlock detection ----------
    def detect_deadlock(self, agents: List[SimpleAgent], current_time: float) -> List[SimpleAgent]:
        """Detect deadlock: agents inside intersection with progress < 0.5m in 3s or speed < 0.5m/s"""
        # 1) Only consider agents inside intersection
        inside = [a for a in agents if self._in_intersection(a.position)]
        if len(inside) < self.min_agents:
            # Clean history for non-inside agents
            for a in agents:
                if a.id not in [ia.id for ia in inside]:
                    self.history.pop(a.id, None)
            return []

        # 2) Update history for inside agents
        for a in inside:
            self.update_vehicle_history(a, current_time)

        # 3) Find stalled agents: low progress OR low speed
        stalled = []
        for a in inside:
            progress = self._progress_in_window(a.id)
            if progress < self.progress_eps or a.speed < 0.5:
                stalled.append(a)

        # 4) Return deadlock if enough stalled agents
        if len(stalled) >= self.min_agents:
            return stalled[:4]  # limit to at most 4 for Nash routine
        return []

    # ---------- Collision candidate check ----------
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
                # Check if endpoints are too close to opposite segments
                if (point_to_segment_distance(s1[0], s2) < conflict_threshold or
                    point_to_segment_distance(s1[1], s2) < conflict_threshold or
                    point_to_segment_distance(s2[0], s1) < conflict_threshold or
                    point_to_segment_distance(s2[1], s1) < conflict_threshold):
                    return True
        return False

    # ---------- Payoff computation ----------
    def compute_payoffs(self, agents: List[SimpleAgent], profile: Tuple[int, ...]) -> Dict[str,float]:
        """
        profile: tuple of 0/1 per agent (0: Wait, 1: Go)
        returns dict agent_id -> payoff
        """
        n = len(agents)
        payoffs = {a.id: 0.0 for a in agents}
        
        # Precompute all pairwise conflicts
        conflicts = {}
        for i in range(n):
            for j in range(i+1, n):
                conflicts[(i,j)] = self.paths_conflict(
                    agents[i].intended_path, 
                    agents[j].intended_path
                )

        # Compute payoff for each agent
        for i, agent in enumerate(agents):
            action = profile[i]
            
            if action == 0:  # Wait
                payoffs[agent.id] = -0.1 * agent.wait_time
            else:  # Go
                # Base reward from bid and inverse wait time
                base_reward = (self.w_bid * agent.bid + 
                             self.w_wait_inv * (1.0/(1.0 + agent.wait_time)))
                
                # Count conflicting neighbors who also go
                collision_count = 0
                for j in range(n):
                    if j != i and profile[j] == 1:  # Other agent also goes
                        pair_key = (min(i,j), max(i,j))
                        if conflicts.get(pair_key, False):
                            collision_count += 1
                
                penalty = self.collision_penalty * collision_count
                payoffs[agent.id] = base_reward - penalty

        # If everyone waits, apply additional penalty
        if all(x == 0 for x in profile):
            for agent in agents:
                payoffs[agent.id] -= self.wait_penalty_allwait

        return payoffs

    # ---------- Pure Nash enumeration ----------
    def find_pure_nash(self, agents: List[SimpleAgent]) -> List[Tuple[int,...]]:
        """Find all pure strategy Nash equilibria"""
        n = len(agents)
        if n == 0:
            return []
            
        all_profiles = list(itertools.product([0,1], repeat=n))
        nash_equilibria = []
        
        for profile in all_profiles:
            payoffs = self.compute_payoffs(agents, profile)
            # check unilateral deviations
            is_nash = True
            
            # Check if any player can unilaterally improve
            for i in range(n):
                current_payoff = payoffs[agents[i].id]
                
                # Try deviating to opposite action
                deviation_profile = list(profile)
                deviation_profile[i] = 1 - profile[i]
                deviation_payoffs = self.compute_payoffs(agents, tuple(deviation_profile))
                deviation_payoff = deviation_payoffs[agents[i].id]
                
                # If deviation improves payoff, not a Nash equilibrium
                if deviation_payoff > current_payoff + 1e-6:
                    is_nash = False
                    break
            
            if is_nash:
                nash_equilibria.append(profile)
        
        return nash_equilibria

    def _profile_has_collisions(self, agents: List[SimpleAgent], profile: Tuple[int,...]) -> bool:
        """Check if a profile has any collisions (both agents go on conflicting paths)"""
        n = len(agents)
        for i in range(n):
            for j in range(i+1, n):
                if (profile[i] == 1 and profile[j] == 1 and 
                    self.paths_conflict(agents[i].intended_path, agents[j].intended_path)):
                    return True
        return False

    def resolve_deadlock(self, agents: List[SimpleAgent]) -> Dict[str,str]:
        """Resolve deadlock using Nash equilibrium selection"""
        if not agents:
            return {}
        
        # 1) Find all pure Nash equilibria
        nash_profiles = self.find_pure_nash(agents)
        
        # 2) Filter safe (collision-free) Nash equilibria
        safe_nash = [p for p in nash_profiles if not self._profile_has_collisions(agents, p)]
        
        chosen_profile = None
        
        if safe_nash:
            # 3) Choose safe Nash with highest social welfare
            best_welfare = -float('inf')
            for profile in safe_nash:
                payoffs = self.compute_payoffs(agents, profile)
                total_welfare = sum(payoffs.values())
                if total_welfare > best_welfare:
                    best_welfare = total_welfare
                    chosen_profile = profile
        else:
            # 4) No safe Nash - choose max welfare among all collision-free profiles
            all_profiles = list(itertools.product([0,1], repeat=len(agents)))
            collision_free = [p for p in all_profiles if not self._profile_has_collisions(agents, p)]
            
            if collision_free:
                best_welfare = -float('inf')
                for profile in collision_free:
                    payoffs = self.compute_payoffs(agents, profile)
                    total_welfare = sum(payoffs.values())
                    if total_welfare > best_welfare:
                        best_welfare = total_welfare
                        chosen_profile = profile
            else:
                # 5) Last resort: highest bidder goes, others wait
                max_bid_idx = max(range(len(agents)), key=lambda i: agents[i].bid)
                chosen_profile = tuple(1 if i == max_bid_idx else 0 for i in range(len(agents)))

        # Convert profile to actions
        result = {}
        for i, agent in enumerate(agents):
            result[agent.id] = 'go' if chosen_profile[i] == 1 else 'wait'
        
        return result

    # ---------- Helper wrapper ----------
    def handle_deadlock(self, agents: List[SimpleAgent], current_time: float) -> Dict[str,str]:
        """Main entry point: detect and resolve deadlock"""
        deadlocked_agents = self.detect_deadlock(agents, current_time)
        if not deadlocked_agents:
            return {}
        
        print(f"🚨 Deadlock detected with {len(deadlocked_agents)} agents")
        return self.resolve_deadlock(deadlocked_agents)
