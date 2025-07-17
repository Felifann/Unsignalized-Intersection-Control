import itertools
from env.simulation_config import SimulationConfig

class NashSolver:
    def __init__(self, agents):
        self.agents = agents  # 发生冲突的agent列表

    def resolve_conflict(self):
        # Step 1: 为每个 agent 定义可选策略（例如：wait / go）
        strategy_space = self._define_strategy_space()

        # Step 2: 构建 payoff matrix（收益矩阵）
        payoff_matrix = self._build_payoff_matrix(strategy_space)

        # Step 3: 在 payoff matrix 中寻找纳什均衡解（简化版）
        nash_equilibrium = self._compute_nash_equilibrium(payoff_matrix, strategy_space)

        # Step 4: 将解映射回每个 agent 的动作
        resolution = self._map_strategy_to_agents(nash_equilibrium, strategy_space)
        return resolution

    def _define_strategy_space(self):
        # 每个 agent 可以选择 'go' 或 'wait'
        return {agent.get_id(): ['go', 'wait'] for agent in self.agents}

    def _build_payoff_matrix(self, strategy_space):
        # 对每种策略组合，计算所有 agent 的收益
        payoff_matrix = {}
        all_combinations = self._enumerate_strategies(strategy_space)

        for combo in all_combinations:
            payoffs = {}
            for agent in self.agents:
                action = combo[agent.get_id()]
                payoffs[agent.get_id()] = self._evaluate_payoff(agent, action, combo)
            
            # 使用字典的frozenset作为key，避免tuple key问题
            combo_key = frozenset(combo.items())
            payoff_matrix[combo_key] = payoffs

        return payoff_matrix

    def _enumerate_strategies(self, strategy_space):
        """枚举所有策略组合"""
        agent_ids = list(strategy_space.keys())
        strategy_lists = [strategy_space[agent_id] for agent_id in agent_ids]
        
        combinations = []
        for combo_tuple in itertools.product(*strategy_lists):
            combo = dict(zip(agent_ids, combo_tuple))
            combinations.append(combo)
        
        return combinations

    def _evaluate_payoff(self, agent, action, combo):
        """
        改进的收益评估函数 - 修复逻辑错误
        """
        base_payoff = 0
        
        # 基础行动收益
        if action == 'go':
            base_payoff = 25  # 通行基础奖励
            
            # 检查是否有路径冲突
            if self._has_path_conflict(agent, combo):
                # 使用更细致的冲突惩罚
                conflict_penalty = self._calculate_conflict_penalty(agent, combo)
                base_payoff -= conflict_penalty
                
        else:  # action == 'wait'
            base_payoff = -5  # 等待成本
            
            # 等待时间惩罚
            wait_time = agent.get_wait_time()
            if wait_time > 3.0:
                base_payoff -= wait_time * 1.5  # 等待越久惩罚越重
        
        # 添加位置和紧急度奖励
        position_bonus = self._calculate_position_bonus(agent, action)
        urgency_bonus = self._calculate_urgency_bonus(agent, action)
        
        return base_payoff + position_bonus + urgency_bonus

    def _has_path_conflict(self, agent, combo):
        """修复路径冲突检测逻辑"""
        agent_path = self._get_agent_path(agent)
        if not agent_path:
            # 如果无法获取路径，保守地假设有冲突
            return True
        
        # 检查与其他选择go的agent是否有路径冲突
        for other_agent in self.agents:
            if other_agent.get_id() == agent.get_id():
                continue
            
            other_action = combo[other_agent.get_id()]
            if other_action == 'go':
                other_path = self._get_agent_path(other_agent)
                if other_path and self._paths_conflict(agent_path, other_path):
                    return True
        
        return False

    def _calculate_conflict_penalty(self, agent, combo):
        """计算冲突惩罚 - 基于路径类型"""
        agent_path = self._get_agent_path(agent)
        if not agent_path:
            return 20  # 默认冲突惩罚
        
        penalty = 0
        for other_agent in self.agents:
            if other_agent.get_id() == agent.get_id():
                continue
            
            other_action = combo[other_agent.get_id()]
            if other_action == 'go':
                other_path = self._get_agent_path(other_agent)
                if other_path and self._paths_conflict(agent_path, other_path):
                    # 基于冲突类型计算不同的惩罚
                    conflict_severity = self._assess_conflict_severity(agent_path, other_path)
                    penalty += conflict_severity
        
        return penalty

    def _assess_conflict_severity(self, path1, path2):
        """评估冲突严重程度"""
        try:
            dir1, turn1 = path1.split('_')
            dir2, turn2 = path2.split('_')
        except ValueError:
            return 30  # 默认冲突严重度
        
        # 对向冲突（较轻）
        if self._are_opposite_directions(dir1, dir2):
            if turn1 == 'L' or turn2 == 'L':
                return 40  # 涉及左转的对向冲突
            else:
                return 15  # 其他对向冲突
        
        # 相邻方向冲突（较重）
        elif self._are_adjacent_directions(dir1, dir2):
            if turn1 == 'L' or turn2 == 'L':
                return 50  # 涉及左转的相邻冲突
            elif turn1 == 'S' and turn2 == 'S':
                return 45  # 直行交叉冲突
            else:
                return 25  # 其他相邻冲突
        
        return 30  # 默认冲突严重度

    def _calculate_position_bonus(self, agent, action):
        """修复位置奖励计算 - 使用正方形检测"""
        if action != 'go':
            return 0
        
        # 已在路口内的agent有更高优先级
        if self._is_agent_in_junction(agent):
            return 15
        
        # 在正方形区域内的bonus
        location = self._get_agent_location(agent)
        if SimulationConfig.is_in_intersection_area(location):
            # 距离中心越近奖励越高
            distance = SimulationConfig.distance_to_intersection_center(location)
            half_size = SimulationConfig.INTERSECTION_HALF_SIZE
            normalized_distance = distance / half_size  # 归一化到0-1
            return max(0, 10 * (1 - normalized_distance))  # 距离越近分数越高
        
        return 0

    def _calculate_urgency_bonus(self, agent, action):
        """修复紧急度奖励计算"""
        if action != 'go':
            return 0
        
        # 等待时间越长，通行奖励越高
        wait_time = agent.get_wait_time()
        if wait_time > 8:
            return 15
        elif wait_time > 5:
            return 10
        elif wait_time > 2:
            return 5
        else:
            return 0

    def _get_agent_path(self, agent):
        """获取agent的路径标识 - 修复数据访问逻辑"""
        try:
            # 直接获取目标转向方向（已由导航系统提供）
            goal_direction = self._get_agent_direction(agent)
            turn_code = self._convert_direction_to_code(goal_direction)
            
            # 获取进入方向
            entry_direction = self._infer_entry_direction(agent)
            
            if entry_direction and turn_code:
                return f"{entry_direction}_{turn_code}"
            else:
                return None
        except Exception as e:
            print(f"[Warning] 获取agent {agent.get_id()} 路径失败: {e}")
            return None

    def _convert_direction_to_code(self, direction):
        """将方向转换为代码"""
        direction_map = {
            'left': 'L',
            'straight': 'S', 
            'right': 'R'
        }
        return direction_map.get(direction)

    def _infer_entry_direction(self, agent):
        """从agent位置推断进入路口的方向 - 修复数据访问"""
        try:
            # 获取agent位置 - 统一数据访问方式
            location = self._get_agent_location(agent)
            if not location:
                return None
            
            # 路口中心
            intersection_center = (-188.9, -89.7, 0.0)
            dx = location[0] - intersection_center[0]
            dy = location[1] - intersection_center[1]
            
            # 基于相对位置推断进入方向
            if abs(dx) > abs(dy):
                if dx > 0:
                    return 'W'  # 从西侧进入（向东行驶）
                else:
                    return 'E'  # 从东侧进入（向西行驶）
            else:
                if dy > 0:
                    return 'S'  # 从南侧进入（向北行驶）
                else:
                    return 'N'  # 从北侧进入（向南行驶）
        except Exception as e:
            print(f"[Warning] 推断agent {agent.get_id()} 进入方向失败: {e}")
            return None

    def _get_agent_location(self, agent):
        """统一的agent位置获取方法"""
        try:
            if hasattr(agent, 'data'):
                if agent.data.get('type') == 'platoon':
                    if 'vehicles' in agent.data and agent.data['vehicles']:
                        return agent.data['vehicles'][0].get('location', (0, 0, 0))
                    else:
                        return agent.data.get('leader_location', agent.data.get('location', (0, 0, 0)))
                else:
                    return agent.data.get('location', (0, 0, 0))
            else:
                return (0, 0, 0)
        except Exception:
            return (0, 0, 0)

    def _get_agent_distance_to_intersection(self, agent):
        """计算agent到路口的距离"""
        location = self._get_agent_location(agent)
        return SimulationConfig.distance_to_intersection_center(location)

    def _paths_conflict(self, path1, path2):
        """检查两条路径是否冲突 - 使用完整冲突矩阵逻辑"""
        if path1 == path2:
            return False
        
        try:
            dir1, turn1 = path1.split('_')
            dir2, turn2 = path2.split('_')
        except ValueError:
            return True  # 路径格式错误时保守地认为冲突
        
        # 对向车道的冲突规则
        if self._are_opposite_directions(dir1, dir2):
            return self._check_opposite_conflict(turn1, turn2)
        
        # 相邻车道的冲突规则
        elif self._are_adjacent_directions(dir1, dir2):
            return self._check_adjacent_conflict(dir1, turn1, dir2, turn2)
        
        return False

    def _are_opposite_directions(self, dir1, dir2):
        """判断是否为对向车道"""
        opposite_pairs = [('N', 'S'), ('S', 'N'), ('E', 'W'), ('W', 'E')]
        return (dir1, dir2) in opposite_pairs

    def _are_adjacent_directions(self, dir1, dir2):
        """判断是否为相邻车道"""
        adjacent_pairs = [
            ('N', 'E'), ('E', 'S'), ('S', 'W'), ('W', 'N'),  # 顺时针相邻
            ('N', 'W'), ('W', 'S'), ('S', 'E'), ('E', 'N')   # 逆时针相邻
        ]
        return (dir1, dir2) in adjacent_pairs

    def _check_opposite_conflict(self, turn1, turn2):
        """检查对向车道的冲突"""
        if turn1 == 'S' and turn2 == 'S':
            return False  # 对向直行不冲突
        if turn1 == 'R' and turn2 == 'R':
            return False  # 对向右转不冲突
        if (turn1 == 'S' and turn2 == 'R') or (turn1 == 'R' and turn2 == 'S'):
            return False  # 直行与右转不冲突
        if turn1 == 'L' or turn2 == 'L':
            return True   # 包含左转的情况都冲突
        return False

    def _check_adjacent_conflict(self, dir1, turn1, dir2, turn2):
        """检查相邻车道的冲突"""
        clockwise_pairs = [('N', 'E'), ('E', 'S'), ('S', 'W'), ('W', 'N')]
        is_clockwise = (dir1, dir2) in clockwise_pairs
        
        if is_clockwise:
            return self._check_clockwise_conflict(turn1, turn2)
        else:
            return self._check_clockwise_conflict(turn2, turn1)

    def _check_clockwise_conflict(self, turn_left, turn_right):
        """检查顺时针相邻车道的冲突"""
        if turn_left == 'L':
            return True  # 左侧左转与右侧任何方向冲突
        if turn_left == 'S' and turn_right == 'L':
            return True  # 左侧直行与右侧左转冲突
        if turn_left == 'S' and turn_right == 'S':
            return True  # 左侧直行与右侧直行冲突
        if turn_left == 'R' and turn_right == 'L':
            return True  # 左侧右转与右侧左转冲突
        return False

    def _is_agent_in_junction(self, agent):
        """判断agent是否在路口内 - 修复数据访问"""
        try:
            if hasattr(agent, 'data'):
                if agent.data.get('type') == 'platoon':
                    return agent.data.get('at_junction', False)
                else:
                    return agent.data.get('at_junction', False)
            return False
        except Exception:
            return False

    def _get_agent_direction(self, agent):
        """获取agent的行驶方向（来自导航系统）- 修复数据访问"""
        try:
            if hasattr(agent, 'goal_direction'):
                return agent.goal_direction
            elif hasattr(agent, 'data') and 'goal_direction' in agent.data:
                return agent.data['goal_direction']
            else:
                return 'straight'  # 默认直行
        except Exception:
            return 'straight'

    def _compute_nash_equilibrium(self, payoff_matrix, strategy_space):
        """计算纳什均衡 - 寻找所有纯策略纳什解"""
        nash_solutions = []
        
        for combo_key, payoffs in payoff_matrix.items():
            combo = dict(combo_key)
            if self._is_nash(combo, payoff_matrix):
                nash_solutions.append(combo)
        
        if nash_solutions:
            # 选择社会福利最大的解（所有agent收益之和最大）
            best_solution = max(nash_solutions, 
                              key=lambda combo: sum(payoff_matrix[frozenset(combo.items())].values()))
            return best_solution
        
        return None  # 没有找到纯策略纳什均衡

    def _is_nash(self, combo, payoff_matrix):
        """检查是否每个 agent 都无意单方面改变策略"""
        if isinstance(combo, frozenset):
            combo = dict(combo)
        
        for agent_id in combo:
            original_action = combo[agent_id]
            for alt_action in ['go', 'wait']:
                if alt_action == original_action:
                    continue
                
                alt_combo = combo.copy()
                alt_combo[agent_id] = alt_action
                
                alt_key = frozenset(alt_combo.items())
                orig_key = frozenset(combo.items())
                
                if (alt_key in payoff_matrix and orig_key in payoff_matrix and
                    payoff_matrix[alt_key][agent_id] > payoff_matrix[orig_key][agent_id]):
                    return False
        return True

    def _map_strategy_to_agents(self, nash_equilibrium, strategy_space):
        """将纳什均衡策略映射回agent动作"""
        if nash_equilibrium is None:
            print("⚠️ 未找到纯策略纳什均衡，使用智能fallback策略")
            return self._intelligent_fallback_strategy()
        
        resolution = {}
        for agent_id, action in nash_equilibrium.items():
            resolution[agent_id] = action
        
        return resolution

    def _intelligent_fallback_strategy(self):
        """智能备用策略 - 基于优先级和路径冲突"""
        if not self.agents:
            return {}
        
        # 按优先级排序：路口内 > 等待时间长 > 距离近
        sorted_agents = sorted(self.agents, key=self._agent_priority, reverse=True)
        
        resolution = {}
        go_agents = []
        
        for agent in sorted_agents:
            agent_id = agent.get_id()
            agent_path = self._get_agent_path(agent)
            
            # 检查与已决定通行的agents是否冲突
            has_conflict = False
            if agent_path:  # 只有能获取路径的agent才检查冲突
                for go_agent in go_agents:
                    go_path = self._get_agent_path(go_agent)
                    if go_path and self._paths_conflict(agent_path, go_path):
                        has_conflict = True
                        break
            
            if not has_conflict:
                resolution[agent_id] = 'go'
                go_agents.append(agent)
            else:
                resolution[agent_id] = 'wait'
        
        return resolution

    def _agent_priority(self, agent):
        """计算agent优先级分数"""
        score = 0
        
        # 路口内优先
        if self._is_agent_in_junction(agent):
            score += 100
        
        # 等待时间长优先
        wait_time = agent.get_wait_time()
        score += wait_time * 3
        
        # 距离近优先
        distance = self._get_agent_distance_to_intersection(agent)
        score += max(0, 30 - distance)
        
        return score

# Agent包装器保持不变，但增加新方法
class AgentWrapper:
    """Agent包装器，提供统一接口"""
    def __init__(self, agent_data):
        self.data = agent_data
    
    def get_id(self):
        return self.data.get('id', 'unknown')
    
    def get_wait_time(self):
        return self.data.get('wait_time', 0.0)
    
    def distance_to_intersection(self):
        location = self.data.get('location', (0, 0, 0))
        return SimulationConfig.distance_to_intersection_center(location)
    
    @property
    def goal_direction(self):
        return self.data.get('goal_direction', 'straight')
    @property
    def goal_direction(self):
        return self.data.get('goal_direction', 'straight')
