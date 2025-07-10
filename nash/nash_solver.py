import itertools

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
        # 收益函数可基于时间成本、碰撞风险、交通效率等
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
        # 简化收益评估规则：
        # go: 快但可能冲突
        # wait: 安全但耗时
        if action == 'go':
            if self._has_conflict(agent, combo):
                return -100  # 冲突惩罚
            return 10  # 安全通行
        else:
            return -1  # 等待惩罚

    def _has_conflict(self, agent, combo):
        """检查该agent的go动作是否与其他agent冲突"""
        agent_direction = self._get_agent_direction(agent)
        
        # 检查与其他选择go的agent是否冲突
        for other_agent in self.agents:
            if other_agent.get_id() == agent.get_id():
                continue
            
            other_action = combo[other_agent.get_id()]
            if other_action == 'go':
                other_direction = self._get_agent_direction(other_agent)
                if self._directions_conflict(agent_direction, other_direction):
                    return True
        
        return False

    def _get_agent_direction(self, agent):
        """获取agent的行驶方向"""
        if hasattr(agent, 'goal_direction'):
            return agent.goal_direction
        elif isinstance(agent, dict) and 'goal_direction' in agent:
            return agent['goal_direction']
        else:
            # 从agent数据中推断方向
            return 'straight'  # 默认直行

    def _directions_conflict(self, dir1, dir2):
        """判断两个方向是否冲突"""
        # 定义冲突规则
        conflict_matrix = {
            ('left', 'straight'): True,
            ('left', 'right'): True,
            ('straight', 'left'): True,
            ('straight', 'right'): False,  # 直行与右转冲突较小
            ('right', 'left'): True,
            ('right', 'straight'): False,
        }
        
        # 同方向不冲突
        if dir1 == dir2:
            return False
        
        return conflict_matrix.get((dir1, dir2), True)

    def _compute_nash_equilibrium(self, payoff_matrix, strategy_space):
        # 可使用简化的"找所有纯策略纳什解"方式：即找出所有稳定组合
        for combo_key, payoffs in payoff_matrix.items():
            # 从frozenset key恢复原始combo字典
            combo = dict(combo_key)
            if self._is_nash(combo, payoff_matrix):
                return combo  # 返回一个纯策略纳什均衡
        return None  # 找不到时可 fallback（例如随机决策）

    def _is_nash(self, combo, payoff_matrix):
        """检查是否每个 agent 都无意单方面改变策略"""
        # 确保combo是字典类型
        if isinstance(combo, frozenset):
            combo = dict(combo)
        
        for agent_id in combo:
            original_action = combo[agent_id]
            for alt_action in ['go', 'wait']:
                if alt_action == original_action:
                    continue
                
                # 创建替代策略组合
                alt_combo = combo.copy()
                alt_combo[agent_id] = alt_action
                
                # 使用frozenset作为key
                alt_key = frozenset(alt_combo.items())
                orig_key = frozenset(combo.items())
                
                # 检查是否存在更优策略
                if (alt_key in payoff_matrix and orig_key in payoff_matrix and
                    payoff_matrix[alt_key][agent_id] > payoff_matrix[orig_key][agent_id]):
                    return False  # 存在更优动作，非纳什
        return True

    def _map_strategy_to_agents(self, nash_equilibrium, strategy_space):
        """将纳什均衡策略映射回agent动作"""
        if nash_equilibrium is None:
            # 找不到纳什均衡时的fallback策略
            print("⚠️ 未找到纳什均衡，使用fallback策略")
            return self._fallback_strategy()
        
        resolution = {}
        for agent_id, action in nash_equilibrium.items():
            resolution[agent_id] = action
        
        return resolution

    def _fallback_strategy(self):
        """当找不到纳什均衡时的备用策略"""
        # 简单策略：让优先级最高的agent通行，其他等待
        if not self.agents:
            return {}
        
        # 按某种优先级排序（这里简化为按ID排序）
        sorted_agents = sorted(self.agents, key=lambda a: a.get_id())
        
        resolution = {}
        resolution[sorted_agents[0].get_id()] = 'go'  # 第一个agent通行
        
        for agent in sorted_agents[1:]:
            resolution[agent.get_id()] = 'wait'  # 其他agent等待
        
        return resolution

# 为agent对象添加必要的接口
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
        # 使用固定的交叉口坐标
        intersection_center = (-188.9, -89.7, 0.0)
        dx = location[0] - intersection_center[0]
        dy = location[1] - intersection_center[1]
        return (dx*dx + dy*dy)**0.5
    
    @property
    def goal_direction(self):
        return self.data.get('goal_direction', 'straight')
