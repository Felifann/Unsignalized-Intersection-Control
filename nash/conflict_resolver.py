import time
from nash.nash_solver import NashSolver, AgentWrapper

class ConflictResolver:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0)):
        self.intersection_center = intersection_center
        self.deadlock_threshold = 10.0
        self.proximity_threshold = 8.0
        self.agent_wait_times = {}
        self.last_positions = {}
        self.last_speed_check = {}
        
        # 新增：完整的四方向冲突矩阵
        self.conflict_matrix = self._build_complete_conflict_matrix()
        
    def _build_complete_conflict_matrix(self):
        """
        构建完整的四方向路口冲突矩阵
        假设四个进入方向为：North, South, East, West
        每个方向可以左转(L)、直行(S)、右转(R)
        """
        # 定义所有可能的路径
        paths = [
            'N_L', 'N_S', 'N_R',  # 北向：左转、直行、右转
            'S_L', 'S_S', 'S_R',  # 南向：左转、直行、右转
            'E_L', 'E_S', 'E_R',  # 东向：左转、直行、右转
            'W_L', 'W_S', 'W_R'   # 西向：左转、直行、右转
        ]
        
        # 构建冲突矩阵 - True表示冲突，False表示不冲突
        conflict_matrix = {}
        
        for path1 in paths:
            conflict_matrix[path1] = {}
            for path2 in paths:
                conflict_matrix[path1][path2] = self._check_path_conflict(path1, path2)
        
        return conflict_matrix
    
    def _check_path_conflict(self, path1, path2):
        """
        检查两条路径是否冲突
        路径格式：方向_转向 (如 'N_L' 表示北向左转)
        """
        if path1 == path2:
            return False  # 相同路径不冲突
        
        # 解析路径
        dir1, turn1 = path1.split('_')
        dir2, turn2 = path2.split('_')
        
        # 对向车道的冲突规则
        if self._are_opposite_directions(dir1, dir2):
            return self._check_opposite_conflict(turn1, turn2)
        
        # 相邻车道的冲突规则
        elif self._are_adjacent_directions(dir1, dir2):
            return self._check_adjacent_conflict(dir1, turn1, dir2, turn2)
        
        # 同向车道（理论上不应该发生，除非多车道）
        else:
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
        # 对向直行不冲突
        if turn1 == 'S' and turn2 == 'S':
            return False
        
        # 对向右转不冲突（各自转向右侧）
        if turn1 == 'R' and turn2 == 'R':
            return False
        
        # 一个直行一个右转，不冲突
        if (turn1 == 'S' and turn2 == 'R') or (turn1 == 'R' and turn2 == 'S'):
            return False
        
        # 包含左转的情况都冲突
        if turn1 == 'L' or turn2 == 'L':
            return True
        
        return False
    
    def _check_adjacent_conflict(self, dir1, turn1, dir2, turn2):
        """检查相邻车道的冲突"""
        # 获取相对位置关系
        clockwise_pairs = [('N', 'E'), ('E', 'S'), ('S', 'W'), ('W', 'N')]
        is_clockwise = (dir1, dir2) in clockwise_pairs
        
        if is_clockwise:
            # dir1在dir2的逆时针方向
            return self._check_clockwise_conflict(turn1, turn2)
        else:
            # dir1在dir2的顺时针方向
            return self._check_clockwise_conflict(turn2, turn1)
    
    def _check_clockwise_conflict(self, turn_left, turn_right):
        """
        检查顺时针相邻车道的冲突
        turn_left: 左侧车道的转向
        turn_right: 右侧车道的转向
        """
        # 左侧左转 vs 右侧任何方向 = 冲突
        if turn_left == 'L':
            return True
        
        # 左侧直行 vs 右侧左转 = 冲突
        if turn_left == 'S' and turn_right == 'L':
            return True
        
        # 左侧直行 vs 右侧直行 = 冲突（交叉路径）
        if turn_left == 'S' and turn_right == 'S':
            return True
        
        # 左侧右转 vs 右侧左转 = 冲突
        if turn_left == 'R' and turn_right == 'L':
            return True
        
        # 其他情况不冲突
        return False

    def check_and_resolve(self, agents):
        """检查冲突并解决死锁 - 改进版"""
        # Step 1: 更新agent等待时间
        self._update_wait_times(agents)
        
        # Step 2: 检查是否需要严格顺序控制
        sequential_resolution = self._check_sequential_passing_needed(agents)
        if sequential_resolution:
            print(f"🚦 检测到路径冲突，启用严格顺序通行")
            self._print_resolution(agents, sequential_resolution)
            return sequential_resolution
        
        # Step 3: 检测真正的死锁冲突
        conflict_agents = self._detect_real_deadlock(agents)
        
        if conflict_agents:
            print(f"🚨 检测到真正死锁，涉及{len(conflict_agents)}个agents")
            
            # Step 4: 调用 NashSolver - 确保正确传递数据
            try:
                wrapped_agents = [AgentWrapper(agent) for agent in conflict_agents]
                solver = NashSolver(wrapped_agents)
                resolution = solver.resolve_conflict()
                
                # 验证resolution格式
                if resolution and isinstance(resolution, dict):
                    # 确保所有agent_id都在resolution中
                    for agent in conflict_agents:
                        if agent['id'] not in resolution:
                            resolution[agent['id']] = 'wait'  # 默认等待
                    
                    print(f"✅ Nash求解器返回有效解决方案")
                    self._print_resolution(conflict_agents, resolution)
                    return resolution
                else:
                    print(f"❌ Nash求解器返回无效解决方案: {resolution}")
                    # 使用fallback策略
                    fallback_resolution = self._fallback_resolution(conflict_agents)
                    self._print_resolution(conflict_agents, fallback_resolution)
                    return fallback_resolution
                    
            except Exception as e:
                print(f"❌ Nash求解器执行失败: {e}")
                # 使用fallback策略
                fallback_resolution = self._fallback_resolution(conflict_agents)
                self._print_resolution(conflict_agents, fallback_resolution)
                return fallback_resolution
        else:
            # 无冲突，所有agent可以通行
            resolution = {agent['id']: 'go' for agent in agents}
            if agents:  # 只在有agents时才打印
                print(f"✅ 无冲突检测到，{len(agents)}个agents均可通行")
            return resolution

    def _fallback_resolution(self, conflict_agents):
        """备用解决策略：基于距离和等待时间的简单优先级"""
        if not conflict_agents:
            return {}
        
        # 按综合优先级排序
        def priority_score(agent):
            distance = self._distance_to_intersection(agent)
            wait_time = agent.get('wait_time', 0.0)
            in_junction = agent.get('at_junction', False)
            
            # 路口内优先，然后是等待时间长的，最后是距离近的
            score = 0
            if in_junction:
                score += 100
            score += wait_time * 5  # 等待时间权重
            score += max(0, 20 - distance)  # 距离权重（距离越近分数越高）
            
            return score
        
        sorted_agents = sorted(conflict_agents, key=priority_score, reverse=True)
        
        resolution = {}
        allowed_paths = set()
        
        for agent in sorted_agents:
            agent_id = agent['id']
            agent_path = self._get_agent_path(agent)
            
            # 检查是否与已允许的路径冲突
            has_conflict = False
            if agent_path:
                for allowed_path in allowed_paths:
                    if self.conflict_matrix.get(agent_path, {}).get(allowed_path, False):
                        has_conflict = True
                        break
            
            if not has_conflict:
                resolution[agent_id] = 'go'
                if agent_path:
                    allowed_paths.add(agent_path)
            else:
                resolution[agent_id] = 'wait'
        
        return resolution

    def _check_sequential_passing_needed(self, agents):
        """检查是否需要严格顺序通行"""
        if len(agents) < 2:
            return None
            
        # 找出所有路径冲突的agent对
        conflicting_pairs = []
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                if self._have_path_conflict(agent1, agent2):
                    conflicting_pairs.append((agent1, agent2))
        
        if not conflicting_pairs:
            return None  # 没有路径冲突
        
        # 按距离路口远近排序，最近的优先通行
        sorted_agents = sorted(agents, key=lambda a: self._distance_to_intersection(a))
        
        # 生成严格顺序：只允许第一个通行，其他等待
        resolution = {}
        for i, agent in enumerate(sorted_agents):
            if i == 0:
                resolution[agent['id']] = 'go'  # 只有最近的agent通行
            else:
                resolution[agent['id']] = 'wait'  # 其他都等待
                
        return resolution

    def _have_path_conflict(self, agent1, agent2):
        """使用完整冲突矩阵判断两个agent是否存在路径冲突"""
        # 获取agent的进入方向和目标方向
        path1 = self._get_agent_path(agent1)
        path2 = self._get_agent_path(agent2)
        
        if not path1 or not path2:
            return False  # 无法确定路径的agent不参与冲突检测
        
        # 检查是否都在路口附近
        dist1 = self._distance_to_intersection(agent1)
        dist2 = self._distance_to_intersection(agent2)
        
        if dist1 > 15.0 or dist2 > 15.0:
            return False  # 距离太远，暂时不冲突
        
        # 使用冲突矩阵判断
        return self.conflict_matrix.get(path1, {}).get(path2, False)
    
    def _get_agent_path(self, agent):
        """
        获取agent的完整路径标识 (进入方向_转向方向)
        返回格式如：'N_L', 'S_S', 'E_R' 等
        """
        # 获取目标转向方向
        goal_direction = agent.get('goal_direction', 'straight')
        turn_code = self._convert_direction_to_code(goal_direction)
        
        # 获取进入方向（需要从车辆位置推断）
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
        """
        从agent位置推断进入路口的方向
        这需要根据实际路口布局来实现
        """
        # 获取agent位置
        if agent.get('type') == 'platoon':
            if 'vehicles' in agent and agent['vehicles']:
                location = agent['vehicles'][0].get('location', (0, 0, 0))
            else:
                location = agent.get('leader_location', (0, 0, 0))
        else:
            location = agent.get('location', (0, 0, 0))
        
        # 相对于路口中心的位置
        dx = location[0] - self.intersection_center[0]
        dy = location[1] - self.intersection_center[1]
        
        # 简化版本：基于相对位置推断进入方向
        # 这里需要根据具体的路口布局调整
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

    def _detect_real_deadlock(self, agents):
        """检测真正的死锁冲突 - 更严格的条件"""
        if len(agents) < 2:
            return []
        
        deadlocked_agents = []
        
        for agent in agents:
            agent_id = agent['id']
            wait_time = self.agent_wait_times.get(agent_id, 0.0)
            distance_to_intersection = self._distance_to_intersection(agent)
            
            # 更严格的死锁条件：
            # 1. 等待时间很长（>10秒）
            # 2. 距离路口很近（<8米）
            # 3. 速度很低（几乎静止）
            # 4. 前方确实有阻塞
            if (wait_time > self.deadlock_threshold and 
                distance_to_intersection < self.proximity_threshold and
                self._is_actually_stuck(agent) and
                self._is_truly_blocked(agent, agents)):
                deadlocked_agents.append(agent)
        
        # 至少需要2个agent才算死锁
        if len(deadlocked_agents) < 2:
            return []
        
        return self._filter_conflicting_agents(deadlocked_agents)
    
    def _is_truly_blocked(self, agent, all_agents):
        """检查agent是否真的被其他agent阻塞"""
        agent_location = self._get_agent_location(agent)
        agent_direction = agent.get('goal_direction', 'straight')
        
        # 检查前方是否有其他车辆
        for other_agent in all_agents:
            if other_agent['id'] == agent['id']:
                continue
            
            other_location = self._get_agent_location(other_agent)
            
            # 计算距离
            distance = self._calculate_distance(agent_location, other_location)
            
            # 如果前方有车辆且距离很近，认为被阻塞
            if distance < 10.0:  # 10米内有其他车辆
                # 检查是否在同一路径上或冲突路径上
                if self._agents_on_conflicting_paths(agent, other_agent):
                    return True
        
        return False

    def _get_agent_location(self, agent):
        """获取agent的位置"""
        if agent.get('type') == 'platoon':
            if 'vehicles' in agent and agent['vehicles']:
                return agent['vehicles'][0].get('location', (0, 0, 0))
            else:
                return agent.get('leader_location', (0, 0, 0))
        else:
            return agent.get('location', (0, 0, 0))

    def _calculate_distance(self, location1, location2):
        """计算两个位置之间的距离"""
        dx = location1[0] - location2[0]
        dy = location1[1] - location2[1]
        return (dx*dx + dy*dy)**0.5

    def _agents_on_conflicting_paths(self, agent1, agent2):
        """检查两个agent是否在冲突的路径上"""
        path1 = self._get_agent_path(agent1)
        path2 = self._get_agent_path(agent2)
        
        if not path1 or not path2:
            return False
        
        return self.conflict_matrix.get(path1, {}).get(path2, False)
    def _is_actually_stuck(self, agent):
        """检查agent是否真的卡住了（速度和位置都没变化）"""
        agent_id = agent['id']
        current_location = agent.get('location', (0, 0, 0))
        
        # 检查速度
        if agent.get('type') == 'platoon':
            if 'vehicles' in agent and agent['vehicles']:
                current_speed = self._get_vehicle_speed(agent['vehicles'][0])
            else:
                current_speed = 0.0
        else:
            current_speed = self._get_vehicle_speed(agent.get('data', agent))
        
        # 如果速度太低且位置变化很小，认为是卡住了
        if current_speed < 0.5:  # 速度小于0.5m/s
            if agent_id in self.last_positions:
                last_location = self.last_positions[agent_id]['location']
                distance_moved = ((current_location[0] - last_location[0])**2 + 
                                (current_location[1] - last_location[1])**2)**0.5
                return distance_moved < 1.0  # 1米内视为卡住
        
        return False

    def _get_vehicle_speed(self, vehicle_data):
        """获取车辆速度"""
        velocity = vehicle_data.get('velocity', (0, 0, 0))
        return (velocity[0]**2 + velocity[1]**2)**0.5

    def _update_wait_times(self, agents):
        """更新agent等待时间 - 改进版"""
        current_time = time.time()
        
        for agent in agents:
            agent_id = agent['id']
            location = agent.get('location', (0, 0, 0))
            
            # 获取当前速度
            if agent.get('type') == 'platoon':
                current_speed = self._get_vehicle_speed(agent['vehicles'][0]) if agent.get('vehicles') else 0.0
            else:
                current_speed = self._get_vehicle_speed(agent.get('data', agent))
            
            # 检查是否在移动
            if agent_id in self.last_positions:
                last_location = self.last_positions[agent_id]['location']
                last_time = self.last_positions[agent_id]['time']
                
                distance_moved = ((location[0] - last_location[0])**2 + 
                                (location[1] - last_location[1])**2)**0.5
                time_diff = current_time - last_time
                
                # 更严格的停滞判断：速度<1m/s 且 移动距离<1.5m
                if current_speed < 1.0 and distance_moved < 1.5 and time_diff > 0:
                    if agent_id not in self.agent_wait_times:
                        self.agent_wait_times[agent_id] = 0.0
                    self.agent_wait_times[agent_id] += time_diff
                else:
                    # 车辆在正常移动，重置等待时间
                    self.agent_wait_times[agent_id] = 0.0
            else:
                self.agent_wait_times[agent_id] = 0.0
            
            # 更新位置记录
            self.last_positions[agent_id] = {
                'location': location,
                'time': current_time
            }
            
            # 将等待时间添加到agent数据中
            agent['wait_time'] = self.agent_wait_times[agent_id]

    def _filter_conflicting_agents(self, potential_conflicts):
        """过滤出真正相互冲突的agents"""
        if len(potential_conflicts) < 2:
            return []
        
        conflicting_agents = []
        
        # 检查方向冲突
        for i, agent1 in enumerate(potential_conflicts):
            has_conflict = False
            for j, agent2 in enumerate(potential_conflicts):
                if i == j:
                    continue
                
                dir1 = agent1.get('goal_direction', 'straight')
                dir2 = agent2.get('goal_direction', 'straight')
                
                # 如果方向冲突，加入冲突列表
                if self._directions_conflict(dir1, dir2):
                    has_conflict = True
                    break
            
            if has_conflict and agent1 not in conflicting_agents:
                conflicting_agents.append(agent1)
        
        return conflicting_agents

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

    def _distance_to_intersection(self, agent):
        """计算agent到交叉口的距离"""
        if agent.get('type') == 'platoon':
            # 车队使用队长位置
            if 'vehicles' in agent and agent['vehicles']:
                location = agent['vehicles'][0].get('location', (0, 0, 0))
            else:
                location = agent.get('leader_location', (0, 0, 0))
        else:
            # 单车
            location = agent.get('location', (0, 0, 0))
        
        dx = location[0] - self.intersection_center[0]
        dy = location[1] - self.intersection_center[1]
        return (dx*dx + dy*dy)**0.5

    def _print_resolution(self, conflict_agents, resolution):
        """打印冲突解决方案 - 修复版"""
        if not resolution:
            print("⚠️ 未获得有效的纳什均衡解决方案")
            return
        
        print(f"🎯 纳什均衡冲突解决方案 (涉及 {len(conflict_agents)} 个agents):")
        print(f"   解决方案类型: {'纳什均衡解' if resolution else '备用策略'}")
        
        # 统计决策结果
        go_count = sum(1 for action in resolution.values() if action == 'go')
        wait_count = sum(1 for action in resolution.values() if action == 'wait')
        
        print(f"   决策分布: 通行({go_count}) / 等待({wait_count})")
        print("   " + "="*50)
        
        # 按决策结果分组显示
        go_agents = []
        wait_agents = []
        
        for agent in conflict_agents:
            agent_id = agent['id']
            action = resolution.get(agent_id, 'wait')  # 默认等待
            
            if action == 'go':
                go_agents.append(agent)
            else:
                wait_agents.append(agent)
        
        # 显示通行的agents
        if go_agents:
            print("   🟢 允许通行:")
            for agent in go_agents:
                agent_id = agent['id']
                agent_type = agent.get('type', 'vehicle')
                direction = agent.get('goal_direction', 'unknown')
                wait_time = agent.get('wait_time', 0.0)
                distance = self._distance_to_intersection(agent)
                path = self._get_agent_path(agent)
                
                type_emoji = "🚛" if agent_type == 'platoon' else "🚗"
                
                print(f"      {type_emoji} {agent_id} ({direction}) "
                      f"路径:{path} 距离:{distance:.1f}m 等待:{wait_time:.1f}s")
        
        # 显示等待的agents
        if wait_agents:
            print("   🔴 要求等待:")
            for agent in wait_agents:
                agent_id = agent['id']
                agent_type = agent.get('type', 'vehicle')
                direction = agent.get('goal_direction', 'unknown')
                wait_time = agent.get('wait_time', 0.0)
                distance = self._distance_to_intersection(agent)
                path = self._get_agent_path(agent)
                
                type_emoji = "🚛" if agent_type == 'platoon' else "🚗"
                
                print(f"      {type_emoji} {agent_id} ({direction}) "
                      f"路径:{path} 距离:{distance:.1f}m 等待:{wait_time:.1f}s")
        
        # 显示路径冲突分析
        print("   📊 路径冲突分析:")
        go_paths = [self._get_agent_path(agent) for agent in go_agents if self._get_agent_path(agent)]
        
        if len(go_paths) > 1:
            conflicts_found = []
            for i, path1 in enumerate(go_paths):
                for j, path2 in enumerate(go_paths[i+1:], i+1):
                    if self.conflict_matrix.get(path1, {}).get(path2, False):
                        conflicts_found.append((path1, path2))
            
            if conflicts_found:
                print(f"      ⚠️ 警告: 通行路径仍有冲突 {conflicts_found}")
            else:
                print(f"      ✅ 通行路径无冲突")
        else:
            print(f"      ✅ 单一通行路径，无冲突")
        
        print("   " + "="*50)

    def get_conflict_stats(self):
        """获取冲突统计信息"""
        total_agents = len(self.agent_wait_times)
        waiting_agents = len([t for t in self.agent_wait_times.values() if t > 1.0])
        deadlocked_agents = len([t for t in self.agent_wait_times.values() if t > self.deadlock_threshold])
        
        return {
            'total_tracked_agents': total_agents,
            'waiting_agents': waiting_agents,
            'deadlocked_agents': deadlocked_agents,
            'deadlock_threshold': self.deadlock_threshold
        }

    def reset_agent_state(self, agent_id):
        """重置特定agent的状态"""
        self.agent_wait_times.pop(agent_id, None)
        self.last_positions.pop(agent_id, None)

    def cleanup_old_agents(self, current_agent_ids):
        """清理已经离开的agents"""
        # 清理不再存在的agent记录
        old_agents = set(self.agent_wait_times.keys()) - set(current_agent_ids)
        for agent_id in old_agents:
            self.reset_agent_state(agent_id)

    def print_conflict_matrix(self):
        """打印完整的冲突矩阵（调试用）"""
        print("\n🚦 完整路口冲突矩阵:")
        print("   ", end="")
        paths = ['N_L', 'N_S', 'N_R', 'S_L', 'S_S', 'S_R', 'E_L', 'E_S', 'E_R', 'W_L', 'W_S', 'W_R']
        for path in paths:
            print(f"{path:>4}", end="")
        print()
        
        for path1 in paths:
            print(f"{path1:>3}:", end="")
            for path2 in paths:
                conflict = self.conflict_matrix[path1][path2]
                symbol = " ✗ " if conflict else " ○ "
                print(symbol, end="")
            print()
        
        print("✗ = 冲突, ○ = 不冲突")
