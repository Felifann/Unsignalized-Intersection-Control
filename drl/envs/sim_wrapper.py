import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time

# Add project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)

from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
from env.simulation_config import SimulationConfig
from platooning.platoon_manager import PlatoonManager
from auction.auction_engine import DecentralizedAuctionEngine
from control import TrafficController
from nash.deadlock_nash_solver import DeadlockNashSolver
from drl.policies.bid_policy import TrainableBidPolicy

class SimulationEnv:
    """Enhanced simulation environment wrapper for DRL training"""
    
    def __init__(self, sim_cfg: dict = None):
        self.sim_cfg = sim_cfg or {}
        self.max_steps = self.sim_cfg.get('max_steps', 500)  # Reduced from 2000 for faster episodes
        self.current_step = 0
        self.episode_start_time = None
        
        # Initialize simulation components
        self._init_simulation()
        
        # State tracking
        self.metrics = {
            'throughput': 0.0,
            'avg_acceleration': 0.0,
            'collision_count': 0,
            'total_controlled': 0,
            'vehicles_exited': 0,
            'platoon_coordination_score': 0.0
        }
        
        print("🎯 DRL Simulation Environment initialized")

    def _init_simulation(self):
        """Initialize the complete simulation system"""
        try:
            # Core simulation components
            self.scenario = ScenarioManager()
            self.state_extractor = StateExtractor(self.scenario.carla)
            
            # Platoon management
            self.platoon_manager = PlatoonManager(self.state_extractor)
            
            # Trainable bid policy
            self.bid_policy = TrainableBidPolicy()
            
            # Auction engine
            self.auction_engine = DecentralizedAuctionEngine(
                state_extractor=self.state_extractor,
                max_go_agents=None
            )
            
            # Nash solver
            self.nash_solver = DeadlockNashSolver(
                max_exact=15,
                conflict_time_window=3.0,
                intersection_center=(-188.9, -89.7, 0.0),
                max_go_agents=None
            )
            
            # Traffic controller
            self.traffic_controller = TrafficController(
                self.scenario.carla, 
                self.state_extractor, 
                max_go_agents=None
            )
            
            # Connect components
            self.traffic_controller.set_platoon_manager(self.platoon_manager)
            self.auction_engine.set_nash_controller(self.nash_solver)
            
            # CRITICAL: Inject trainable policy into auction engine
            self.auction_engine.bid_policy = self.bid_policy
            
            print("✅ All simulation components initialized successfully")
            
        except Exception as e:
            print(f"❌ Simulation initialization failed: {e}")
            raise

    def observation_dim(self) -> int:
        """Return observation space dimension"""
        return 30

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the simulation environment"""
        try:
            self.current_step = 0
            self.episode_start_time = time.time()
            
            # Reset simulation
            self.scenario.reset_scenario()
            self.scenario.start_time_counters()
            
            # Reset policy
            self.bid_policy.reset_episode()
            
            # Reset metrics
            self.metrics = {
                'throughput': 0.0,
                'avg_acceleration': 0.0,
                'collision_count': 0,
                'total_controlled': 0,
                'vehicles_exited': 0,
                'platoon_coordination_score': 0.0
            }
            
            # Get initial observation
            obs = self._get_observation()
            
            print(f"🔄 Environment reset completed")
            return obs
            
        except Exception as e:
            print(f"❌ Environment reset failed: {e}")
            return np.zeros(self.observation_dim(), dtype=np.float32)

    def step_with_enhanced_params(self, bid_scale: float, eta_weight: float = 1.0,
                            speed_weight: float = 0.3, congestion_sensitivity: float = 0.4,
                            speed_diff_modifier: float = 0.0, follow_distance_modifier: float = 0.0) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute simulation step with DRL parameters"""
        try:
            # Update bid policy with DRL parameters
            self.bid_policy.update_bid_scale(bid_scale)
            self.bid_policy.update_advanced_params(
                eta_weight=eta_weight,
                speed_weight=speed_weight,
                congestion_sensitivity=congestion_sensitivity
            )
            self.bid_policy.update_control_params(
                speed_diff_modifier=speed_diff_modifier,
                follow_distance_modifier=follow_distance_modifier
            )
            
            # Execute simulation steps - REDUCED FOR FASTER DEBUGGING
            for _ in range(5):  # Reduced from 10 to 5 simulation ticks per DRL step
                self.scenario.carla.world.tick()
                self.current_step += 1
                
                # Get vehicle states
                vehicle_states = self.state_extractor.get_vehicle_states()
                
                # Update control every 5 steps instead of 10
                if self.current_step % 5 == 0:
                    # Update platoons
                    self.platoon_manager.update()
                    
                    # Run auction with trained policy
                    auction_winners = self.auction_engine.update(vehicle_states, self.platoon_manager)
                    
                    # Apply traffic control
                    self.traffic_controller.update_control(self.platoon_manager, self.auction_engine, auction_winners)
                
                # Update vehicle labels
                self.scenario.update_vehicle_labels()
            
            # Calculate observation, reward, done
            obs = self._get_observation()
            reward = self._calculate_reward()
            done = self._check_done()
            info = self._get_info()
            
            return obs, reward, done, info
            
        except Exception as e:
            print(f"⚠️ Simulation step error: {e}")
            obs = np.zeros(self.observation_dim(), dtype=np.float32)
            reward = -100.0
            done = True
            info = {'error': str(e)}
            return obs, reward, done, info

    def _build_context(self, vehicle_states: List[Dict]) -> Dict:
        """Build context for auction system"""
        return {
            'total_vehicles': len(vehicle_states),
            'junction_vehicles': len([v for v in vehicle_states if v.get('is_junction', False)]),
            'avg_speed': np.mean([self._extract_speed(v.get('velocity', 0)) for v in vehicle_states]) if vehicle_states else 0.0,
            'congestion_level': min(len(vehicle_states) / 50.0, 1.0),
            'policy_params': self.bid_policy.get_policy_params()
        }

    def _extract_speed(self, velocity) -> float:
        """Extract speed scalar from velocity"""
        if hasattr(velocity, 'length'):
            return velocity.length()
        elif isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            return np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
        elif isinstance(velocity, (int, float)):
            return abs(velocity)
        return 0.0

    def _get_observation(self) -> np.ndarray:
        """Get normalized observation vector"""
        try:
            vehicle_states = self.state_extractor.get_vehicle_states()
            
            # Basic traffic metrics
            total_vehicles = len(vehicle_states)
            junction_vehicles = len([v for v in vehicle_states if v.get('is_junction', False)])
            avg_speed = np.mean([self._extract_speed(v.get('velocity', 0)) for v in vehicle_states]) if vehicle_states else 0.0
            congestion_ratio = min(total_vehicles / 50.0, 1.0)
            
            # Speed statistics
            speeds = [self._extract_speed(v.get('velocity', 0)) for v in vehicle_states] if vehicle_states else [0]
            speed_std = np.std(speeds)
            max_speed = np.max(speeds)
            min_speed = np.min(speeds)
            
            # Position analysis
            positions = [v.get('position', [0, 0, 0]) for v in vehicle_states]
            if positions:
                center_distances = [np.sqrt((p[0] + 188.9)**2 + (p[1] + 89.7)**2) for p in positions]
                avg_distance_to_center = np.mean(center_distances)
                vehicles_near_center = sum(1 for d in center_distances if d < 30.0)
            else:
                avg_distance_to_center = 0.0
                vehicles_near_center = 0
            
            # Platoon information
            platoons = self.platoon_manager.get_all_platoons()
            platoon_count = len(platoons)
            total_platoon_vehicles = sum(p.get_size() for p in platoons)
            avg_platoon_size = total_platoon_vehicles / max(platoon_count, 1)
            
            # Auction state
            auction_stats = self.auction_engine.get_auction_stats()
            current_agents = auction_stats.get('current_agents', 0)
            platoon_agents = auction_stats.get('platoon_agents', 0) 
            vehicle_agents = auction_stats.get('vehicle_agents', 0)
            current_go_count = auction_stats.get('current_go_count', 0)
            
            # Control state
            control_stats = self.traffic_controller.get_control_stats()
            controlled_vehicles = control_stats.get('total_controlled', 0)
            waiting_vehicles = control_stats.get('waiting_vehicles', 0)
            go_vehicles = control_stats.get('go_vehicles', 0)
            
            # Policy parameters
            policy_params = self.bid_policy.get_policy_params()
            bid_scale = policy_params.get('bid_scale', 1.0)
            eta_weight = policy_params.get('eta_weight', 1.0)
            speed_weight = policy_params.get('speed_weight', 0.3)
            congestion_sensitivity = policy_params.get('congestion_sensitivity', 0.4)
            speed_diff_modifier = policy_params.get('speed_diff_modifier', 0.0)
            follow_distance_modifier = policy_params.get('follow_distance_modifier', 0.0)
            
            # Build observation vector (30 dimensions)
            obs = np.array([
                # Traffic state (12 dims)
                total_vehicles / 100.0,
                junction_vehicles / 20.0,
                avg_speed / 15.0,
                congestion_ratio,
                speed_std / 10.0,
                max_speed / 20.0,
                min_speed / 15.0,
                avg_distance_to_center / 100.0,
                vehicles_near_center / 20.0,
                platoon_count / 10.0,
                total_platoon_vehicles / 50.0,
                avg_platoon_size / 5.0,
                
                # Auction state (8 dims)
                current_agents / 30.0,
                platoon_agents / 15.0,
                vehicle_agents / 30.0,
                current_go_count / 15.0,
                controlled_vehicles / 30.0,
                waiting_vehicles / 20.0,
                go_vehicles / 15.0,
                (controlled_vehicles / max(total_vehicles, 1)),
                
                # Policy parameters (6 dims)
                (bid_scale - 0.1) / 4.9,
                (eta_weight - 0.5) / 2.5,
                speed_weight,
                congestion_sensitivity,
                (speed_diff_modifier + 20.0) / 40.0,
                (follow_distance_modifier + 1.0) / 3.0,
                
                # Performance metrics (4 dims)
                self.metrics['throughput'] / 3000.0,
                np.clip(self.metrics['avg_acceleration'] + 5.0, 0, 10) / 10.0,
                self.metrics['collision_count'] / 10.0,
                self.metrics['platoon_coordination_score']
            ], dtype=np.float32)
            
            # Ensure correct dimensions and clipping
            assert len(obs) == 30, f"Observation dimension mismatch: {len(obs)} vs 30"
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            print(f"⚠️ Observation generation failed: {e}")
            return np.zeros(30, dtype=np.float32)

    def _calculate_reward(self) -> float:
        """Calculate reward based on performance metrics"""
        try:
            reward = 0.0
            
            # Step penalty
            reward -= 0.05
            
            # Throughput reward
            control_stats = self.traffic_controller.get_control_stats()
            vehicles_exited = control_stats.get('vehicles_exited_intersection', 0)
            new_exits = vehicles_exited - self.metrics.get('vehicles_exited', 0)
            if new_exits > 0:
                reward += new_exits * 20.0
                self.metrics['vehicles_exited'] = vehicles_exited
            
            # Calculate throughput
            if hasattr(self.scenario, 'get_sim_elapsed'):
                sim_time = self.scenario.get_sim_elapsed()
                if sim_time and sim_time > 0:
                    throughput = (vehicles_exited / sim_time) * 3600
                    self.metrics['throughput'] = throughput
                    if throughput > 2000:
                        reward += 5.0
            
            # Safety - collision penalty
            if hasattr(self.scenario, 'traffic_generator'):
                try:
                    collision_stats = self.scenario.traffic_generator.get_collision_stats()
                    current_collisions = collision_stats.get('total_collisions', 0)
                    new_collisions = current_collisions - self.metrics.get('collision_count', 0)
                    if new_collisions > 0:
                        reward -= new_collisions * 200.0
                        self.metrics['collision_count'] = current_collisions
                except:
                    pass
            
            # Smooth acceleration reward
            final_stats = self.traffic_controller.get_final_statistics()
            avg_abs_accel = final_stats.get('average_absolute_acceleration', 0.0)
            self.metrics['avg_acceleration'] = avg_abs_accel
            
            if avg_abs_accel > 3.0:
                reward -= (avg_abs_accel - 3.0) * 2.0
            else:
                reward += 5.0 * (1.0 - avg_abs_accel / 3.0)
            
            # Platoon coordination bonus
            platoons = self.platoon_manager.get_all_platoons()
            if platoons:
                controlled_vehicle_ids = set(control_stats.get('active_controls', []))
                coordinated_platoons = 0
                for platoon in platoons:
                    platoon_vehicle_ids = platoon.get_vehicle_ids()
                    controlled_count = sum(1 for vid in platoon_vehicle_ids if vid in controlled_vehicle_ids)
                    if controlled_count == len(platoon_vehicle_ids):
                        coordinated_platoons += 1
                
                coordination_score = coordinated_platoons / len(platoons)
                self.metrics['platoon_coordination_score'] = coordination_score
                reward += coordination_score * 2.0
            
            # Deadlock penalty
            if self._detect_deadlock():
                reward -= 1000.0
            
            return float(reward)
            
        except Exception as e:
            print(f"⚠️ Reward calculation error: {e}")
            return -10.0

    def _detect_deadlock(self) -> bool:
        """Simple deadlock detection"""
        try:
            vehicle_states = self.state_extractor.get_vehicle_states()
            junction_vehicles = [v for v in vehicle_states if v.get('is_junction', False)]
            
            if len(junction_vehicles) > 15:
                speeds = [self._extract_speed(v.get('velocity', 0)) for v in junction_vehicles]
                avg_speed = np.mean(speeds) if speeds else 0.0
                return avg_speed < 0.5
            
            return False
        except:
            return False

    def _check_done(self) -> bool:
        """Check if episode should end"""
        if self.current_step >= self.max_steps:
            return True
        if self._detect_deadlock():
            return True
        if self.metrics.get('collision_count', 0) > 50:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        info = {
            'step': self.current_step,
            'metrics': self.metrics.copy(),
            'policy_stats': self.bid_policy.get_policy_stats(),
            'auction_stats': self.auction_engine.get_auction_stats(),
            'control_stats': self.traffic_controller.get_control_stats()
        }
        
        if self._check_done():
            info['final_stats'] = self.traffic_controller.get_final_statistics()
            if self.episode_start_time:
                episode_duration = time.time() - self.episode_start_time
                info['episode_duration'] = episode_duration
        
        return info

    def close(self):
        """Clean up simulation"""
        try:
            if hasattr(self, 'scenario'):
                self.scenario.stop_time_counters()
            print("🏁 Simulation environment closed")
        except Exception as e:
            print(f"⚠️ Error closing environment: {e}")

    def _detect_deadlock(self) -> bool:
        """Simple deadlock detection"""
        try:
            vehicle_states = self.state_extractor.get_vehicle_states()
            junction_vehicles = [v for v in vehicle_states if v.get('is_junction', False)]
            
            if len(junction_vehicles) > 15:
                speeds = [self._extract_speed(v.get('velocity', 0)) for v in junction_vehicles]
                avg_speed = np.mean(speeds) if speeds else 0.0
                return avg_speed < 0.5
            
            return False
        except:
            return False

    def _check_done(self) -> bool:
        """检查回合是否结束"""
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True
        if self._detect_deadlock():
            return True
        if self.metrics.get('collision_count', 0) > 50:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = {
            'step': self.current_step,
            'metrics': self.metrics.copy(),
            'policy_stats': self.bid_policy.get_policy_stats(),
            'auction_stats': self.auction_engine.get_auction_stats(),
            'control_stats': self.traffic_controller.get_control_stats()
        }
        
        # 如果回合结束，添加最终统计
        if self._check_done():
            info['final_stats'] = self.traffic_controller.get_final_statistics()
            
            # 计算性能指标
            if self.episode_start_time:
                episode_duration = time.time() - self.episode_start_time
                info['episode_duration'] = episode_duration
                info['real_time_factor'] = self.current_step * 0.05 / episode_duration  # 假设每步0.05秒仿真时间
        
        return info

    def close(self):
        """清理资源"""
        try:
            if hasattr(self, 'scenario'):
                self.scenario.stop_time_counters()
            print("🏁 仿真环境已关闭")
        except Exception as e:
            print(f"⚠️ 关闭环境时出错: {e}")
    def close(self):
        """清理资源"""
        try:
            if hasattr(self, 'scenario'):
                self.scenario.stop_time_counters()
            print("🏁 仿真环境已关闭")
        except Exception as e:
            print(f"⚠️ 关闭环境时出错: {e}")
