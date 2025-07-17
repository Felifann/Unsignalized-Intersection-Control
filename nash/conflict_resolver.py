import time
from nash.nash_solver import NashSolver, AgentWrapper
from env.simulation_config import SimulationConfig

class ConflictResolver:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0)):
        self.intersection_center = intersection_center
        self.deadlock_threshold = 10.0
        self.proximity_threshold = SimulationConfig.INTERSECTION_HALF_SIZE * 0.8
        self.agent_wait_times = {}
        self.last_positions = {}
        self.last_speed_check = {}
        
        # 新增：控制动作缓存
        self.agent_control_actions = {}  # {agent_id: {'action': 'WAIT'|'GO', 'timestamp': time}}
        self.action_timeout = 5.0  # 动作超时时间（秒）
        
        # 完整的四方向冲突矩阵
        self.conflict_matrix = self._build_complete_conflict_matrix()
        
    def check_and_resolve(self, agents):
        """检查冲突并解决死锁 - 返回带控制标志的解决方案"""
        current_time = time.time()
        
        # Step 1: 清理过期的控制动作
        self._cleanup_expired_actions(current_time)
        
        # Step 2: 更新agent等待时间
        self._update_wait_times(agents)
        
        # Step 3: 扩展冲突检测到所有agents（不只是前3名）
        conflict_groups = self._detect_all_conflict_groups(agents)
        
        if not conflict_groups:
            # 无冲突，所有agent可以通行
            resolution = {agent['id']: {'action': 'GO', 'reason': 'no_conflict'} for agent in agents}
            if agents:
                print(f"✅ 无冲突检测到，{len(agents)}个agents均可通行")
            return resolution
        
        # Step 4: 对每个冲突组应用Nash均衡求解
        full_resolution = {}
        
        for group_id, conflict_group in enumerate(conflict_groups):
            print(f"🚨 检测到冲突组 {group_id+1}，涉及{len(conflict_group)}个agents")
            
            # 调用Nash求解器
            try:
                wrapped_agents = [AgentWrapper(agent) for agent in conflict_group]
                solver = NashSolver(wrapped_agents)
                group_resolution = solver.resolve_conflict()
                
                if group_resolution and isinstance(group_resolution, dict):
                    # 转换为带控制标志的格式
                    for agent in conflict_group:
                        agent_id = agent['id']
                        nash_action = group_resolution.get(agent_id, 'wait')
                        
                        if nash_action == 'go':
                            control_action = {
                                'action': 'GO',
                                'reason': 'nash_winner',
                                'group_id': group_id,
                                'timestamp': current_time
                            }
                        else:
                            control_action = {
                                'action': 'WAIT',
                                'reason': 'nash_loser',
                                'group_id': group_id,
                                'timestamp': current_time,
                                'wait_duration': agent.get('wait_time', 0.0)
                            }
                        
                        full_resolution[agent_id] = control_action
                        # 更新缓存
                        self.agent_control_actions[agent_id] = control_action
                else:
                    # Nash求解失败，使用fallback策略
                    fallback = self._fallback_resolution_with_flags(conflict_group, group_id, current_time)
                    full_resolution.update(fallback)
                    
            except Exception as e:
                print(f"❌ Nash求解器执行失败: {e}")
                fallback = self._fallback_resolution_with_flags(conflict_group, group_id, current_time)
                full_resolution.update(fallback)
        
        # Step 5: 处理非冲突agents
        conflict_agent_ids = set(full_resolution.keys())
        for agent in agents:
            if agent['id'] not in conflict_agent_ids:
                full_resolution[agent['id']] = {
                    'action': 'GO',
                    'reason': 'no_conflict',
                    'timestamp': current_time
                }
        
        # Step 6: 打印解决方案
        self._print_control_resolution(full_resolution)
        
        return full_resolution

    def _detect_all_conflict_groups(self, agents):
        """检测所有冲突组（不限于前3名）"""
        if len(agents) < 2:
            return []
        
        # 构建冲突图
        conflict_pairs = []
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                if self._have_path_conflict(agent1, agent2) and self._agents_close_enough(agent1, agent2):
                    conflict_pairs.append((i, j))
        
        if not conflict_pairs:
            return []
        
        # 使用图算法找到所有连通的冲突组
        conflict_groups = self._find_connected_groups(agents, conflict_pairs)
        
        return conflict_groups

    def _agents_close_enough(self, agent1, agent2):
        """判断两个agents是否足够接近以产生实际冲突"""
        dist1 = self._distance_to_intersection(agent1)
        dist2 = self._distance_to_intersection(agent2)
        
        # 只有当两个agents都在30米范围内时才考虑冲突
        return dist1 <= 30.0 and dist2 <= 30.0

    def _find_connected_groups(self, agents, conflict_pairs):
        """使用并查集算法找到所有连通的冲突组"""
        n = len(agents)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 合并冲突的agents
        for i, j in conflict_pairs:
            union(i, j)
        
        # 分组
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(agents[i])
        
        # 只返回包含多个agents的组
        return [group for group in groups.values() if len(group) > 1]

    def _fallback_resolution_with_flags(self, conflict_agents, group_id, current_time):
        """带控制标志的备用解决策略"""
        if not conflict_agents:
            return {}
        
        # 按优先级排序
        sorted_agents = sorted(conflict_agents, key=self._agent_priority, reverse=True)
        
        resolution = {}
        allowed_paths = set()
        
        for rank, agent in enumerate(sorted_agents):
            agent_id = agent['id']
            agent_path = self._get_agent_path(agent)
            
            # 检查是否与已允许的路径冲突
            has_conflict = False
            if agent_path:
                for allowed_path in allowed_paths:
                    if self.conflict_matrix.get(agent_path, {}).get(allowed_path, False):
                        has_conflict = True
                        break
            
            if not has_conflict and rank == 0:  # 最高优先级且无冲突
                control_action = {
                    'action': 'GO',
                    'reason': 'fallback_winner',
                    'group_id': group_id,
                    'priority_rank': rank + 1,
                    'timestamp': current_time
                }
                if agent_path:
                    allowed_paths.add(agent_path)
            else:
                control_action = {
                    'action': 'WAIT',
                    'reason': 'fallback_conflict',
                    'group_id': group_id,
                    'priority_rank': rank + 1,
                    'timestamp': current_time,
                    'wait_duration': agent.get('wait_time', 0.0)
                }
            
            resolution[agent_id] = control_action
            # 更新缓存
            self.agent_control_actions[agent_id] = control_action
        
        return resolution

    def _cleanup_expired_actions(self, current_time):
        """清理过期的控制动作"""
        expired_agents = []
        for agent_id, action_data in self.agent_control_actions.items():
            if current_time - action_data['timestamp'] > self.action_timeout:
                expired_agents.append(agent_id)
        
        for agent_id in expired_agents:
            del self.agent_control_actions[agent_id]

    def _print_control_resolution(self, resolution):
        """打印带控制标志的解决方案"""
        if not resolution:
            return
        
        go_agents = []
        wait_agents = []
        
        for agent_id, action_data in resolution.items():
            if action_data['action'] == 'GO':
                go_agents.append((agent_id, action_data))
            else:
                wait_agents.append((agent_id, action_data))
        
        print(f"🎮 冲突解决方案:")
        print(f"   🟢 允许通行: {len(go_agents)}个 | 🔴 强制等待: {len(wait_agents)}个")
        
        if go_agents:
            print("   🟢 通行agents:")
            for agent_id, action_data in go_agents:
                reason = action_data.get('reason', 'unknown')
                group_id = action_data.get('group_id', 'N/A')
                print(f"      ✅ Agent {agent_id} - {reason} (组{group_id})")
        
        if wait_agents:
            print("   🔴 等待agents:")
            for agent_id, action_data in wait_agents:
                reason = action_data.get('reason', 'unknown')
                group_id = action_data.get('group_id', 'N/A')
                wait_duration = action_data.get('wait_duration', 0.0)
                print(f"      ⏸️ Agent {agent_id} - {reason} (组{group_id}) 已等待:{wait_duration:.1f}s")

    def get_current_control_actions(self):
        """获取当前所有agent的控制动作"""
        return self.agent_control_actions.copy()

    def force_agent_action(self, agent_id, action, reason="manual_override"):
        """手动强制设置agent的控制动作"""
        current_time = time.time()
        self.agent_control_actions[agent_id] = {
            'action': action,
            'reason': reason,
            'timestamp': current_time
        }

    # ... 其他现有方法保持不变 ...
    def _build_complete_conflict_matrix(self):
        """构建完整的四方向路口冲突矩阵"""
        paths = [
            'N_L', 'N_S', 'N_R',  # 北向：左转、直行、右转
            'S_L', 'S_S', 'S_R',  # 南向：左转、直行、右转
            'E_L', 'E_S', 'E_R',  # 东向：左转、直行、右转
            'W_L', 'W_S', 'W_R'   # 西向：左转、直行、右转
        ]
        
        conflict_matrix = {}
        for path1 in paths:
            conflict_matrix[path1] = {}
            for path2 in paths:
                conflict_matrix[path1][path2] = self._check_path_conflict(path1, path2)
        
        return conflict_matrix
    
    def _check_path_conflict(self, path1, path2):
        """检查两条路径是否冲突"""
        if path1 == path2:
            return False
        
        dir1, turn1 = path1.split('_')
        dir2, turn2 = path2.split('_')
        
        if self._are_opposite_directions(dir1, dir2):
            return self._check_opposite_conflict(turn1, turn2)
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
            ('N', 'E'), ('E', 'S'), ('S', 'W'), ('W', 'N'),
            ('N', 'W'), ('W', 'S'), ('S', 'E'), ('E', 'N')
        ]
        return (dir1, dir2) in adjacent_pairs
    
    def _check_opposite_conflict(self, turn1, turn2):
        """检查对向车道的冲突"""
        if turn1 == 'S' and turn2 == 'S':
            return False
        if turn1 == 'R' and turn2 == 'R':
            return False
        if (turn1 == 'S' and turn2 == 'R') or (turn1 == 'R' and turn2 == 'S'):
            return False
        if turn1 == 'L' or turn2 == 'L':
            return True
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
            return True
        if turn_left == 'S' and turn_right == 'L':
            return True
        if turn_left == 'S' and turn_right == 'S':
            return True
        if turn_left == 'R' and turn_right == 'L':
            return True
        return False

    def _have_path_conflict(self, agent1, agent2):
        """使用完整冲突矩阵判断两个agent是否存在路径冲突"""
        path1 = self._get_agent_path(agent1)
        path2 = self._get_agent_path(agent2)
        
        if not path1 or not path2:
            return False
        
        dist1 = self._distance_to_intersection(agent1)
        dist2 = self._distance_to_intersection(agent2)
        
        if dist1 > 15.0 or dist2 > 15.0:
            return False
        
        return self.conflict_matrix.get(path1, {}).get(path2, False)
    
    def _get_agent_path(self, agent):
        """获取agent的完整路径标识"""
        goal_direction = agent.get('goal_direction', 'straight')
        turn_code = self._convert_direction_to_code(goal_direction)
        entry_direction = self._infer_entry_direction(agent)
        
        if entry_direction and turn_code:
            return f"{entry_direction}_{turn_code}"
        else:
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
        """从agent位置推断进入路口的方向"""
        if agent.get('type') == 'platoon':
            if 'vehicles' in agent and agent['vehicles']:
                location = agent['vehicles'][0].get('location', (0, 0, 0))
            else:
                location = agent.get('leader_location', (0, 0, 0))
        else:
            location = agent.get('location', (0, 0, 0))
        
        dx = location[0] - self.intersection_center[0]
        dy = location[1] - self.intersection_center[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return 'W'
            else:
                return 'E'
        else:
            if dy > 0:
                return 'S'
            else:
                return 'N'

    def _update_wait_times(self, agents):
        """更新agent等待时间"""
        current_time = time.time()
        
        for agent in agents:
            agent_id = agent['id']
            location = agent.get('location', (0, 0, 0))
            
            if agent.get('type') == 'platoon':
                current_speed = self._get_vehicle_speed(agent['vehicles'][0]) if agent.get('vehicles') else 0.0
            else:
                current_speed = self._get_vehicle_speed(agent.get('data', agent))
            
            if agent_id in self.last_positions:
                last_location = self.last_positions[agent_id]['location']
                last_time = self.last_positions[agent_id]['time']
                
                distance_moved = ((location[0] - last_location[0])**2 + 
                                (location[1] - last_location[1])**2)**0.5
                time_diff = current_time - last_time
                
                if current_speed < 1.0 and distance_moved < 1.5 and time_diff > 0:
                    if agent_id not in self.agent_wait_times:
                        self.agent_wait_times[agent_id] = 0.0
                    self.agent_wait_times[agent_id] += time_diff
                else:
                    self.agent_wait_times[agent_id] = 0.0
            else:
                self.agent_wait_times[agent_id] = 0.0
            
            self.last_positions[agent_id] = {
                'location': location,
                'time': current_time
            }
            
            agent['wait_time'] = self.agent_wait_times[agent_id]

    def _get_vehicle_speed(self, vehicle_data):
        """获取车辆速度"""
        velocity = vehicle_data.get('velocity', (0, 0, 0))
        return (velocity[0]**2 + velocity[1]**2)**0.5

    def _distance_to_intersection(self, agent):
        """计算agent到交叉口的距离"""
        if agent.get('type') == 'platoon':
            if 'vehicles' in agent and agent['vehicles']:
                location = agent['vehicles'][0].get('location', (0, 0, 0))
            else:
                location = agent.get('leader_location', (0, 0, 0))
        else:
            location = agent.get('location', (0, 0, 0))
        
        return SimulationConfig.distance_to_intersection_center(location)

    def _agent_priority(self, agent):
        """计算agent优先级分数"""
        score = 0
        
        if agent.get('at_junction', False):
            score += 100
        
        wait_time = agent.get('wait_time', 0.0)
        score += wait_time * 3
        
        distance = self._distance_to_intersection(agent)
        score += max(0, 30 - distance)
        
        return score

    def cleanup_old_agents(self, current_agent_ids):
        """清理已不在当前agent列表中的旧数据"""
        try:
            # 清理等待时间记录
            old_wait_agents = set(self.agent_wait_times.keys()) - set(current_agent_ids)
            for agent_id in old_wait_agents:
                del self.agent_wait_times[agent_id]
            
            # 清理位置记录
            old_position_agents = set(self.last_positions.keys()) - set(current_agent_ids)
            for agent_id in old_position_agents:
                del self.last_positions[agent_id]
            
            # 清理速度检查记录
            old_speed_agents = set(self.last_speed_check.keys()) - set(current_agent_ids)
            for agent_id in old_speed_agents:
                del self.last_speed_check[agent_id]
            
            # 清理控制动作记录
            old_control_agents = set(self.agent_control_actions.keys()) - set(current_agent_ids)
            for agent_id in old_control_agents:
                del self.agent_control_actions[agent_id]
            
            # 如果清理了一些数据，记录日志
            total_cleaned = len(old_wait_agents) + len(old_position_agents) + len(old_speed_agents) + len(old_control_agents)
            if total_cleaned > 0:
                print(f"🧹 冲突解决器清理旧数据：{total_cleaned}条记录")
                
        except Exception as e:
            print(f"[Warning] 清理旧agent数据失败: {e}")

    def get_conflict_stats(self):
        """获取冲突统计信息"""
        current_time = time.time()
        
        # 统计当前等待的agents
        waiting_agents = 0
        for agent_id, wait_time in self.agent_wait_times.items():
            if wait_time > 1.0:  # 等待超过1秒的
                waiting_agents += 1
        
        # 统计死锁的agents
        deadlocked_agents = 0
        for agent_id, wait_time in self.agent_wait_times.items():
            if wait_time > self.deadlock_threshold:
                deadlocked_agents += 1
        
        # 统计控制动作
        controlled_agents = len(self.agent_control_actions)
        wait_controlled = sum(1 for action in self.agent_control_actions.values() 
                             if action.get('action') == 'WAIT')
        go_controlled = controlled_agents - wait_controlled
        
        return {
            'waiting_agents': waiting_agents,
            'deadlocked_agents': deadlocked_agents,
            'deadlock_threshold': self.deadlock_threshold,
            'controlled_agents': controlled_agents,
            'wait_controlled': wait_controlled,
            'go_controlled': go_controlled,
            'total_tracked_agents': len(self.agent_wait_times)
        }

