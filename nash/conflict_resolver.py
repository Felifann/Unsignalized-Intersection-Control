import time
from nash.nash_solver import NashSolver, AgentWrapper

class ConflictResolver:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0)):
        self.intersection_center = intersection_center
        self.deadlock_threshold = 5.0  # 增加死锁检测阈值到5秒
        self.proximity_threshold = 12.0  # 增加冲突检测距离阈值
        self.agent_wait_times = {}  # 记录agent等待时间
        self.last_positions = {}  # 记录上次位置用于检测停滞
        
    def check_and_resolve(self, agents):
        """检查冲突并解决死锁"""
        # Step 1: 更新agent等待时间
        self._update_wait_times(agents)
        
        # Step 2: 检测是否存在阻塞型冲突
        conflict_agents = self._detect_conflicts(agents)
        
        if conflict_agents:
            print(f"🚨 检测到死锁冲突，涉及{len(conflict_agents)}个agents")
            
            # Step 3: 若存在，调用 NashSolver
            wrapped_agents = [AgentWrapper(agent) for agent in conflict_agents]
            solver = NashSolver(wrapped_agents)
            resolution = solver.resolve_conflict()
            
            # 输出解决方案
            self._print_resolution(conflict_agents, resolution)
            return resolution
        else:
            # 无冲突，维持原排序
            return {agent['id']: 'go' for agent in agents}

    def _update_wait_times(self, agents):
        """更新agent等待时间"""
        current_time = time.time()
        
        for agent in agents:
            agent_id = agent['id']
            location = agent.get('location', (0, 0, 0))
            
            # 检查agent是否在移动
            if agent_id in self.last_positions:
                last_location = self.last_positions[agent_id]['location']
                last_time = self.last_positions[agent_id]['time']
                
                # 计算移动距离
                distance_moved = ((location[0] - last_location[0])**2 + 
                                (location[1] - last_location[1])**2)**0.5
                
                time_diff = current_time - last_time
                
                # 如果移动距离很小且时间间隔合理，增加等待时间
                if distance_moved < 2.0 and time_diff > 0:  # 2米内视为停滞
                    if agent_id not in self.agent_wait_times:
                        self.agent_wait_times[agent_id] = 0.0
                    self.agent_wait_times[agent_id] += time_diff
                else:
                    # 重置等待时间
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

    def _detect_conflicts(self, agents):
        """检测死锁冲突 - 更严格的条件"""
        if len(agents) < 2:
            return []
        
        conflict_agents = []
        
        for agent in agents:
            agent_id = agent['id']
            wait_time = self.agent_wait_times.get(agent_id, 0.0)
            distance_to_intersection = self._distance_to_intersection(agent)
            
            # 更严格的死锁条件：
            # 1. 等待时间超过阈值
            # 2. 距离交叉口很近
            # 3. 在路口内或即将进入路口
            # 4. 确实存在其他阻塞车辆
            if (wait_time > self.deadlock_threshold and 
                distance_to_intersection < self.proximity_threshold and
                self._is_truly_blocked(agent, agents)):
                conflict_agents.append(agent)
        
        # 如果只有一个agent停滞，不算死锁
        if len(conflict_agents) <= 1:
            return []
        
        # 进一步检查这些agent是否真的相互冲突
        return self._filter_conflicting_agents(conflict_agents)

    def _is_truly_blocked(self, agent, all_agents):
        """检查agent是否真的被阻塞"""
        agent_location = agent.get('location', (0, 0, 0))
        agent_direction = agent.get('goal_direction', 'straight')
        
        # 检查前方是否有其他停滞的车辆
        for other_agent in all_agents:
            if other_agent['id'] == agent['id']:
                continue
            
            other_location = other_agent.get('location', (0, 0, 0))
            other_wait_time = self.agent_wait_times.get(other_agent['id'], 0.0)
            
            # 计算两车距离
            distance = ((agent_location[0] - other_location[0])**2 + 
                       (agent_location[1] - other_location[1])**2)**0.5
            
            # 如果前方有其他等待的车辆，且距离很近
            if distance < 10.0 and other_wait_time > 2.0:
                return True
        
        return False

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
