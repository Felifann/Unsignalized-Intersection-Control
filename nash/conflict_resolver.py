import time
from nash.nash_solver import NashSolver, AgentWrapper

class ConflictResolver:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0)):
        self.intersection_center = intersection_center
        self.deadlock_threshold = 10.0  # 增加到10秒，减少误报
        self.proximity_threshold = 8.0   # 减少到8米，更精确的冲突检测
        self.agent_wait_times = {}
        self.last_positions = {}
        self.last_speed_check = {}  # 新增：记录速度检查
        
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
            
            # Step 4: 调用 NashSolver
            wrapped_agents = [AgentWrapper(agent) for agent in conflict_agents]
            solver = NashSolver(wrapped_agents)
            resolution = solver.resolve_conflict()
            
            self._print_resolution(conflict_agents, resolution)
            return resolution
        else:
            # 无冲突，所有agent可以通行
            return {agent['id']: 'go' for agent in agents}

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
        """判断两个agent是否存在路径冲突"""
        dir1 = agent1.get('goal_direction', 'straight')
        dir2 = agent2.get('goal_direction', 'straight')
        
        # 检查是否都在路口附近（距离<15米）
        dist1 = self._distance_to_intersection(agent1)
        dist2 = self._distance_to_intersection(agent2)
        
        if dist1 > 15.0 or dist2 > 15.0:
            return False  # 距离太远，暂时不冲突
            
        # 使用更精确的冲突规则
        return self._directions_conflict_strict(dir1, dir2)

    def _directions_conflict_strict(self, dir1, dir2):
        """严格的方向冲突判断"""
        # 同方向不冲突
        if dir1 == dir2:
            return False
            
        # 严格冲突规则：左转与所有方向冲突，直行与左转冲突
        conflict_rules = {
            ('left', 'straight'): True,
            ('left', 'right'): True,
            ('straight', 'left'): True,
            ('straight', 'right'): False,  # 直行与右转可以并行
            ('right', 'left'): True,
            ('right', 'straight'): False,  # 右转与直行可以并行
        }
        
        return conflict_rules.get((dir1, dir2), False)

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
        """打印冲突解决方案"""
        print(f"🎯 纳什均衡冲突解决方案:")
        for agent in conflict_agents:
            agent_id = agent['id']
            action = resolution.get(agent_id, 'wait')
            agent_type = agent.get('type', 'vehicle')
            direction = agent.get('goal_direction', 'unknown')
            wait_time = agent.get('wait_time', 0.0)
            
            action_emoji = "🟢" if action == 'go' else "🔴"
            type_emoji = "🚛" if agent_type == 'platoon' else "🚗"
            
            print(f"   {action_emoji} {type_emoji} {agent_id} ({direction}) "
                  f"-> {action.upper()} (等待:{wait_time:.1f}s)")

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
