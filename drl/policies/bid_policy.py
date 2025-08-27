import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
import time

class TrainableBidPolicy:
    """增强的可训练出价策略，完全集成DRL优化"""
    
    def __init__(self):
        # 核心可训练参数 - 扩展版本
        self.bid_scale = 1.0  # 总体出价缩放因子
        self.eta_weight = 1.0  # ETA权重
        self.speed_weight = 0.3  # 速度权重
        self.congestion_sensitivity = 0.4  # 拥堵敏感度
        self.platoon_bonus = 0.5  # 车队奖励
        self.junction_penalty = 0.2  # 路口惩罚
        
        # 新增：更多可训练参数
        self.fairness_factor = 0.1  # 公平性因子
        self.urgency_threshold = 5.0  # 紧急度阈值
        self.adaptation_rate = 0.05  # 适应率
        self.proximity_bonus_weight = 1.0  # 邻近性奖励权重
        
        # 控制参数修正 - 扩展版本
        self.speed_diff_modifier = 0.0  # 速度差异修正
        self.follow_distance_modifier = 0.0  # 跟车距离修正
        
        # 新增：ignore_vehicles参数控制
        self.ignore_vehicles_go = 50.0  # GO状态下的ignore_vehicles百分比
        self.ignore_vehicles_wait = 0.0  # WAIT状态下的ignore_vehicles百分比
        self.ignore_vehicles_platoon_leader = 50.0  # 车队领队的ignore_vehicles
        self.ignore_vehicles_platoon_follower = 90.0  # 车队跟随者的ignore_vehicles
        
        # 动态适应参数
        self.performance_window = 100
        self.performance_history = deque(maxlen=self.performance_window)
        
        # 基础控制参数
        self.speed_diff_base = -50.0
        self.follow_distance_base = 1.5
        
        # 性能跟踪
        self.bid_history = {}
        self.success_history = deque(maxlen=200)
        self.episode_bids = []
        self.episode_rewards = []
        
        print("🎯 扩展可训练出价策略初始化 - 包含ignore_vehicles控制")

    def reset_episode(self):
        """重置回合状态"""
        self.episode_bids = []
        self.episode_rewards = []
        self.bid_history.clear()
        print("🔄 策略状态已重置")

    def update_bid_scale(self, bid_scale: float):
        """Update only bid scale (backward compatibility)"""
        self.bid_scale = np.clip(bid_scale, 0.1, 5.0)

    def update_all_bid_params(self, bid_scale: float = None, eta_weight: float = None,
                             speed_weight: float = None, congestion_sensitivity: float = None,
                             platoon_bonus: float = None, junction_penalty: float = None,
                             fairness_factor: float = None, urgency_threshold: float = None,
                             proximity_bonus_weight: float = None):
        """更新所有出价相关参数"""
        if bid_scale is not None:
            self.bid_scale = np.clip(bid_scale, 0.1, 5.0)
        if eta_weight is not None:
            self.eta_weight = np.clip(eta_weight, 0.5, 3.0)
        if speed_weight is not None:
            self.speed_weight = np.clip(speed_weight, 0.0, 1.0)
        if congestion_sensitivity is not None:
            self.congestion_sensitivity = np.clip(congestion_sensitivity, 0.0, 1.0)
        if platoon_bonus is not None:
            self.platoon_bonus = np.clip(platoon_bonus, 0.0, 2.0)
        if junction_penalty is not None:
            self.junction_penalty = np.clip(junction_penalty, 0.0, 1.0)
        if fairness_factor is not None:
            self.fairness_factor = np.clip(fairness_factor, 0.0, 0.5)
        if urgency_threshold is not None:
            self.urgency_threshold = np.clip(urgency_threshold, 1.0, 10.0)
        if proximity_bonus_weight is not None:
            self.proximity_bonus_weight = np.clip(proximity_bonus_weight, 0.0, 3.0)

    def update_control_params(self, speed_diff_modifier: float = None, 
                            follow_distance_modifier: float = None):
        """更新控制参数修正值"""
        if speed_diff_modifier is not None:
            self.speed_diff_modifier = np.clip(speed_diff_modifier, -30.0, 30.0)
        if follow_distance_modifier is not None:
            self.follow_distance_modifier = np.clip(follow_distance_modifier, -2.0, 3.0)

    def update_ignore_vehicles_params(self, ignore_vehicles_go: float = None,
                                    ignore_vehicles_wait: float = None,
                                    ignore_vehicles_platoon_leader: float = None,
                                    ignore_vehicles_platoon_follower: float = None):
        """更新ignore_vehicles相关参数"""
        if ignore_vehicles_go is not None:
            self.ignore_vehicles_go = np.clip(ignore_vehicles_go, 0.0, 100.0)
        if ignore_vehicles_wait is not None:
            self.ignore_vehicles_wait = 0.0  # Always fixed at 0 (not trainable)
        if ignore_vehicles_platoon_leader is not None:
            self.ignore_vehicles_platoon_leader = np.clip(ignore_vehicles_platoon_leader, 0.0, 80.0)
        if ignore_vehicles_platoon_follower is not None:
            self.ignore_vehicles_platoon_follower = np.clip(ignore_vehicles_platoon_follower, 50.0, 100.0)

    def calculate_bid(self, vehicle_state: Dict, is_platoon_leader: bool = False, 
                     platoon_size: int = 1, context: Dict = None) -> float:
        """计算增强的训练驱动出价"""
        try:
            # 基础出价组件
            base_bid = 10.0
            
            # 1. ETA因子 (可训练权重)
            eta = vehicle_state.get('eta_to_intersection', 10.0)
            eta_factor = self._calculate_urgency_factor(eta) * self.eta_weight
            
            # 2. 速度因子 (可训练权重)
            speed = self._extract_speed(vehicle_state.get('velocity', 0))
            speed_factor = self._calculate_speed_factor(speed) * self.speed_weight
            
            # 3. 车队加成
            platoon_factor = 0.0
            if is_platoon_leader and platoon_size > 1:
                platoon_factor = self.platoon_bonus * np.log(platoon_size)
            
            # 4. 路口位置惩罚
            junction_factor = 0.0
            if vehicle_state.get('is_junction', False):
                junction_factor = -self.junction_penalty
            
            # 5. 上下文调整 (拥堵响应)
            context_adjustment = self._apply_context_adjustments(vehicle_state, context or {})
            
            # 6. 公平性调整
            fairness_adjustment = self._calculate_fairness_adjustment(vehicle_state, context or {})
            
            # 7. 邻近性奖励
            proximity_bonus = self._calculate_proximity_bonus(vehicle_state)
            
            # 综合出价计算
            raw_bid = base_bid + eta_factor + speed_factor + platoon_factor + junction_factor + \
                     context_adjustment + fairness_adjustment + proximity_bonus
            
            # 应用可训练的缩放因子
            final_bid = raw_bid * self.bid_scale
            
            # 确保出价在合理范围内
            final_bid = np.clip(final_bid, 1.0, 200.0)
            
            # 记录出价用于分析
            vehicle_id = vehicle_state.get('id', 'unknown')
            self._track_bid(vehicle_id, final_bid, context or {})
            
            return float(final_bid)
            
        except Exception as e:
            print(f"⚠️ 出价计算错误: {e}")
            return 20.0  # 返回默认出价

    def _calculate_urgency_factor(self, eta: float) -> float:
        """计算紧急程度因子"""
        if eta <= 0:
            return 5.0  # 最高紧急度
        elif eta <= self.urgency_threshold:
            return 3.0 * (self.urgency_threshold - eta) / self.urgency_threshold
        else:
            return max(0.1, 1.0 / (1.0 + 0.1 * (eta - self.urgency_threshold)))

    def _extract_speed(self, velocity) -> float:
        """提取速度标量"""
        if hasattr(velocity, 'length'):
            return velocity.length()
        elif isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            return np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
        elif isinstance(velocity, (int, float)):
            return abs(velocity)
        return 0.0

    def _calculate_speed_factor(self, speed: float) -> float:
        """计算速度因子"""
        if speed < 2.0:  # 低速惩罚
            return -2.0
        elif speed > 12.0:  # 高速小幅奖励
            return 1.0
        else:
            return (speed - 2.0) / 10.0  # 线性奖励

    def _apply_context_adjustments(self, vehicle_state: Dict, context: Dict) -> float:
        """应用拥堵和上下文调整"""
        adjustment = 0.0
        
        # 拥堵响应
        congestion_level = context.get('congestion_level', 0.0)
        if congestion_level > 0.5:
            # 高拥堵时增加出价
            adjustment += self.congestion_sensitivity * congestion_level * 5.0
        
        # 路口车辆密度调整
        junction_vehicles = context.get('junction_vehicles', 0)
        if junction_vehicles > 10:
            adjustment += 2.0  # 路口繁忙时增加出价
        
        return adjustment

    def _calculate_fairness_adjustment(self, vehicle_state: Dict, context: Dict) -> float:
        """计算公平性调整"""
        vehicle_id = vehicle_state.get('id', 'unknown')
        
        # 检查该车辆的历史等待时间
        if vehicle_id in self.bid_history:
            wait_count = self.bid_history[vehicle_id].get('wait_count', 0)
            if wait_count > 5:  # 等待过久
                return self.fairness_factor * wait_count * 2.0
        
        return 0.0

    def _calculate_proximity_bonus(self, vehicle_state: Dict) -> float:
        """计算接近路口的奖励"""
        # Handle both 'position' and 'location' keys, and both dict/tuple formats
        position = vehicle_state.get('position') or vehicle_state.get('location', [0, 0, 0])
        
        if isinstance(position, dict):
            pos_x = position.get('x', 0.0)
            pos_y = position.get('y', 0.0)
        elif isinstance(position, (list, tuple)) and len(position) >= 2:
            pos_x = float(position[0])
            pos_y = float(position[1])
        else:
            pos_x, pos_y = 0.0, 0.0
        
        center = [-188.9, -89.7, 0.0]
        
        distance = np.sqrt((pos_x - center[0])**2 + (pos_y - center[1])**2)
        
        if distance < 50.0:  # 50米内
            return max(0.0, (50.0 - distance) / 50.0 * 3.0)
        
        return 0.0

    def _track_bid(self, vehicle_id: str, bid_value: float, context: Dict):
        """跟踪出价历史"""
        if vehicle_id not in self.bid_history:
            self.bid_history[vehicle_id] = {
                'bids': [],
                'outcomes': [],
                'wait_count': 0,
                'first_seen': time.time()
            }
        
        self.bid_history[vehicle_id]['bids'].append(bid_value)
        self.episode_bids.append({
            'vehicle_id': vehicle_id,
            'bid': bid_value,
            'timestamp': time.time(),
            'context': context.copy()
        })

    def get_enhanced_control_params(self, action: str, is_platoon_member: bool = False, 
                                  is_leader: bool = False, vehicle_state: Dict = None) -> Dict[str, float]:
        """获取增强的控制参数，包含可训练的ignore_vehicles"""
        # 基础参数
        speed_diff = self.speed_diff_base + self.speed_diff_modifier
        follow_distance = self.follow_distance_base + self.follow_distance_modifier
        
        # 确定ignore_vehicles参数
        if is_platoon_member:
            if is_leader:
                ignore_vehicles = self.ignore_vehicles_platoon_leader
            else:
                ignore_vehicles = self.ignore_vehicles_platoon_follower
        else:
            if action == 'go':
                ignore_vehicles = self.ignore_vehicles_go
            else:  # wait
                ignore_vehicles = self.ignore_vehicles_wait
        
        # 根据动作调整基础参数
        if action == 'go':
            speed_diff = max(speed_diff, -30.0)  # 允许更积极的速度
            follow_distance = max(0.5, follow_distance - 0.2)
        elif action == 'wait':
            # CRITICAL FIX: Make waiting vehicles strictly stop
            speed_diff = -100.0  # 强制停止 (much more strict than -70.0)
            follow_distance = follow_distance + 1.0  # 增加跟车距离确保安全
            # Force ignore_vehicles to 0 for waiting vehicles
            ignore_vehicles = 0.0
        
        # 车队特殊调整
        if is_platoon_member:
            follow_distance *= 0.8  # 车队内更紧密
            if is_leader:
                speed_diff += 5.0  # 领队稍微积极
        
        return {
            'speed_diff': float(speed_diff),           
            'follow_distance': float(follow_distance), 
            'ignore_lights': 100.0,                   
            'ignore_signs': 100.0,                    
            'ignore_vehicles': float(ignore_vehicles)  
        }

    def adapt_performance(self, performance_metrics: Dict):
        """根据性能指标调整策略"""
        self.performance_history.append(performance_metrics)
        
        if len(self.performance_history) >= 50:  # 足够的历史数据
            recent_performance = list(self.performance_history)[-20:]
            avg_reward = np.mean([p.get('reward', 0) for p in recent_performance])
            
            # 简单的自适应调整
            if avg_reward < -10:  # 性能不佳
                self.bid_scale *= (1.0 - self.adaptation_rate)
                self.congestion_sensitivity *= (1.0 + self.adaptation_rate)
            elif avg_reward > 20:  # 性能良好
                self.bid_scale *= (1.0 + self.adaptation_rate * 0.5)
            
            # 确保参数在合理范围内
            self.bid_scale = np.clip(self.bid_scale, 0.5, 3.0)
            self.congestion_sensitivity = np.clip(self.congestion_sensitivity, 0.1, 0.8)

    def get_policy_stats(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        stats = {
            'current_bid_scale': self.bid_scale,
            'eta_weight': self.eta_weight,
            'speed_weight': self.speed_weight,
            'congestion_sensitivity': self.congestion_sensitivity,
            'total_bids_this_episode': len(self.episode_bids),
            'unique_vehicles_bid': len(set(b['vehicle_id'] for b in self.episode_bids)),
            'avg_bid_value': np.mean([b['bid'] for b in self.episode_bids]) if self.episode_bids else 0.0,
            'performance_history_length': len(self.performance_history)
        }
        
        # 计算成功率（如果有历史数据）
        if self.success_history:
            stats['success_rate'] = np.mean(self.success_history)
        else:
            stats['success_rate'] = 0.0
        
        return stats

    def get_current_bid_scale(self) -> float:
        """获取当前出价缩放因子"""
        return self.bid_scale

    def get_all_trainable_params(self) -> Dict[str, float]:
        """获取所有可训练参数"""
        return {
            # 出价策略参数
            'bid_scale': self.bid_scale,
            'eta_weight': self.eta_weight,
            'speed_weight': self.speed_weight,
            'congestion_sensitivity': self.congestion_sensitivity,
            'platoon_bonus': self.platoon_bonus,
            'junction_penalty': self.junction_penalty,
            'fairness_factor': self.fairness_factor,
            'urgency_threshold': self.urgency_threshold,
            'proximity_bonus_weight': self.proximity_bonus_weight,
            
            # 控制参数
            'speed_diff_modifier': self.speed_diff_modifier,
            'follow_distance_modifier': self.follow_distance_modifier,
            
            # ignore_vehicles参数
            'ignore_vehicles_go': self.ignore_vehicles_go,
            'ignore_vehicles_wait': self.ignore_vehicles_wait,
            'ignore_vehicles_platoon_leader': self.ignore_vehicles_platoon_leader,
            'ignore_vehicles_platoon_follower': self.ignore_vehicles_platoon_follower
        }
