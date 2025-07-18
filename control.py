import carla
import math
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from env.simulation_config import SimulationConfig

class TrafficController:
    """
    基于拍卖结果的统一交通控制器
    核心思想：所有控制都基于拍卖获胜者的优先级排序
    """
    
    def __init__(self, carla_wrapper, state_extractor):
        self.carla = carla_wrapper
        self.state_extractor = state_extractor
        self.world = carla_wrapper.world
        self.traffic_manager = carla_wrapper.client.get_trafficmanager()
        
        # 添加交叉口中心和检测区域配置
        self.intersection_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        self.intersection_half_size = SimulationConfig.INTERSECTION_HALF_SIZE
        
        # 初始化ConflictResolver
        from nash.conflict_resolver import ConflictResolver
        self.conflict_resolver = ConflictResolver(self.intersection_center)

        # 控制状态记录
        self.controlled_vehicles = {}  # {vehicle_id: control_info}
        self.current_controlled_vehicles = set()
        
        # 控制参数
        self.default_speed_diff = -40.0  # 默认速度差异
        self.default_follow_distance = 1.5  # 默认跟车距离
        
        # 新增：安全和撞车恢复机制
        self.crashed_vehicles = set()  # 记录撞车车辆
        self.stuck_vehicles = {}  # 记录卡住的车辆 {vehicle_id: stuck_time}
        self.emergency_recovery_vehicles = set()  # 紧急恢复中的车辆
        self.last_positions = {}  # 记录车辆上次位置
        self.collision_check_enabled = True  # 碰撞检测开关
        
        # 新增：防止重复控制的状态记录
        self.intersection_pass_vehicles = set()  # 正在强制通过路口的车辆
        self.last_control_log_time = {}  # 记录上次日志输出时间
        self.control_log_interval = 5.0  # 日志输出间隔（秒）
    
        # 新增：车队管理器引用（将在主程序中设置）
        self.platoon_manager = None
    
        # 新增：路口容量限制
        self.max_concurrent_agents = 4  # 最多同时通过4个agent
    
        print("🎮 基于拍卖的交通控制器初始化完成 - 集成安全控制和冲突解决")
    
    def set_platoon_manager(self, platoon_manager):
        """设置车队管理器引用"""
        self.platoon_manager = platoon_manager
    
    def update_control(self, platoon_manager, auction_engine=None):
        """
        主控制更新函数 - 增加安全检查和撞车恢复
        """
        # 1. 检测和处理撞车车辆
        self._detect_and_handle_crashes()
        
        # 2. 检测和处理卡住的车辆
        self._detect_and_handle_stuck_vehicles()
        
        # 3. 确保路口内车辆完成通过
        # self._ensure_intersection_vehicles_complete()
        
        # 4. 获取拍卖优先级排序
        auction_priority = []
        if auction_engine:
            auction_priority = auction_engine._get_current_priority_order()
        
        # 5. 基于拍卖结果应用控制（使用安全参数）
        current_controlled = set()
        if auction_priority:
            current_controlled = self._apply_auction_based_control(auction_priority)
        
        # 6. 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 7. 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _detect_and_handle_crashes(self):
        """检测撞车并进行紧急处理"""
        if not self.collision_check_enabled:
            return
            
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in all_vehicles:
            if not vehicle.is_alive:
                continue
                
            vehicle_id = vehicle.id
            
            # 检查是否发生碰撞
            if self._is_vehicle_crashed(vehicle):
                if vehicle_id not in self.crashed_vehicles:
                    self.crashed_vehicles.add(vehicle_id)
                    print(f"🚨 检测到车辆 {vehicle_id} 发生碰撞，启动紧急恢复")
                
                # 应用紧急恢复控制
                self._apply_emergency_recovery(vehicle)
                self.emergency_recovery_vehicles.add(vehicle_id)
            else:
                # 如果车辆已经恢复，从撞车列表中移除，并重置碰撞状态
                if vehicle_id in self.crashed_vehicles:
                    self.crashed_vehicles.discard(vehicle_id)
                    self.emergency_recovery_vehicles.discard(vehicle_id)
                    print(f"✅ 车辆 {vehicle_id} 已恢复正常")
                    # 重置碰撞状态
                    if hasattr(self.carla, 'traffic_generator'):
                        self.carla.traffic_generator.reset_collision_status(vehicle_id)

    def _is_vehicle_crashed(self, vehicle):
        """判断车辆是否发生碰撞（优先使用CollisionSensor）"""
        try:
            # 优先使用 traffic_generator 的碰撞状态
            if hasattr(self.carla, 'traffic_generator'):
                tg = self.carla.traffic_generator
                if tg.get_collision_status(vehicle.id):
                    print(f"[Collision] 车辆 {vehicle.id} 发生碰撞")
                    return True
            return False
        except Exception as e:
            print(f"[Warning] 碰撞检测失败 {vehicle.id}: {e}")
            return False

    def _apply_emergency_recovery(self, vehicle):
        """应用紧急恢复控制"""
        try:
            # 紧急恢复参数：温和控制，避免进一步碰撞
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 20.0)  # 降低速度
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)  # 增大跟车距离
            self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)  # 严格遵守信号
            self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)  # 严格避让其他车辆

        except Exception as e:
            print(f"[Warning] 紧急恢复控制失败 {vehicle.id}: {e}")

    def _detect_and_handle_stuck_vehicles(self):
        """检测和处理卡住的车辆"""
        current_time = time.time()
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in all_vehicles:
            if not vehicle.is_alive:
                continue
                
            vehicle_id = vehicle.id
            location = vehicle.get_transform().location
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            # 检查车辆是否移动
            if vehicle_id in self.last_positions:
                last_location = self.last_positions[vehicle_id]['location']
                last_time = self.last_positions[vehicle_id]['time']
                
                distance_moved = location.distance(last_location)
                time_diff = current_time - last_time
                
                # 如果车辆长时间不移动，认为卡住了
                if speed < 0.5 and distance_moved < 1.0 and time_diff > 5.0:
                    if vehicle_id not in self.stuck_vehicles:
                        self.stuck_vehicles[vehicle_id] = current_time
                        print(f"🚧 检测到车辆 {vehicle_id} 卡住，启动疏导")
                    
                    # 应用疏导控制
                    self._apply_unstuck_control(vehicle)
                else:
                    # 车辆正常移动，从卡住列表中移除
                    if vehicle_id in self.stuck_vehicles:
                        del self.stuck_vehicles[vehicle_id]
            
            # 更新位置记录
            self.last_positions[vehicle_id] = {
                'location': location,
                'time': current_time
            }

    def _apply_unstuck_control(self, vehicle):
        """应用疏导控制帮助车辆脱困"""
        try:
            # 疏导参数：略微激进以帮助脱困
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -30.0)
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)
            self.traffic_manager.ignore_lights_percentage(vehicle, 60.0)
            self.traffic_manager.ignore_signs_percentage(vehicle, 50.0)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 30.0)  # 适度忽略其他车辆

        except Exception as e:
            print(f"[Warning] 疏导控制失败 {vehicle.id}: {e}")

    def _apply_auction_based_control(self, auction_priority):
        """基于拍卖结果应用统一控制 - 优化为竞价排队机制"""
        controlled_vehicles = set()
        
        if not auction_priority:
            return controlled_vehicles
        
        print(f"🎯 基于竞价排序应用控制，共{len(auction_priority)}个参与agents")
        
        # 1. 分析冲突路径的agent组合
        conflict_groups = self._identify_conflict_groups(auction_priority)
        
        # 2. 为每个agent分配控制状态
        agent_control_status = self._determine_agent_control_status(auction_priority, conflict_groups)
        
        # 3. 应用控制参数
        for winner_data in auction_priority:
            agent = winner_data['agent']
            bid_value = winner_data['bid_value']
            rank = winner_data['rank']
            
            # 安全检查：跳过有问题的车辆
            if self._agent_has_problematic_vehicles(agent):
                print(f"⚠️ Agent {agent['id']} 包含问题车辆，跳过控制")
                continue
            
            # 获取该agent的控制状态
            control_status = agent_control_status.get(agent['id'], 'wait')
            
            try:
                if agent['type'] == 'vehicle':
                    vehicle_id = agent['id']
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_status):
                        controlled_vehicles.add(vehicle_id)
                        status_emoji = "🟢" if control_status == 'go' else "🔴"
                        print(f"   #{rank}: {status_emoji}🚗单车{vehicle_id} (出价:{bid_value:.1f}) - {control_status}")
                
                elif agent['type'] == 'platoon':
                    platoon_vehicles = agent['vehicles']
                    direction = agent['goal_direction']
                    controlled_in_platoon = self._apply_platoon_agent_control(
                        platoon_vehicles, rank, bid_value, direction, control_status
                    )
                    controlled_vehicles.update(controlled_in_platoon)
                    
                    status_emoji = "🟢" if control_status == 'go' else "🔴"
                    print(f"   #{rank}: {status_emoji}🚛车队{agent['id']} "
                          f"({len(platoon_vehicles)}车-{direction}) (出价:{bid_value:.1f}) - {control_status}")
            
            except Exception as e:
                print(f"[Warning] agent {agent['id']} 控制应用失败: {e}")
        
        return controlled_vehicles

    def _identify_conflict_groups(self, auction_priority):
        """识别冲突路径的agent组合"""
        conflict_groups = []
        agents = [w['agent'] for w in auction_priority]
        
        # 找出所有有冲突的agent对
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                dir1 = agent1.get('goal_direction', 'straight')
                dir2 = agent2.get('goal_direction', 'straight')
                
                if self._directions_have_conflict(dir1, dir2):
                    # 找到冲突对，检查是否已在某个冲突组中
                    group_found = False
                    for group in conflict_groups:
                        if agent1['id'] in [a['id'] for a in group] or agent2['id'] in [a['id'] for a in group]:
                            # 加入现有组
                            if agent1 not in group:
                                group.append(agent1)
                            if agent2 not in group:
                                group.append(agent2)
                            group_found = True
                            break
                    
                    if not group_found:
                        # 创建新的冲突组
                        conflict_groups.append([agent1, agent2])
        
        return conflict_groups

    def _determine_agent_control_status(self, auction_priority, conflict_groups):
        """确定agent控制状态 - 增加路口容量限制"""
        agent_control_status = {}
        bid_rank_map = {w['agent']['id']: w for w in auction_priority}

        # 统计当前路口内的agent
        current_agents_in_intersection = 0
        
        for winner_data in auction_priority:
            agent = winner_data['agent']
            if self._is_agent_in_intersection(agent):
                current_agents_in_intersection += 1

        print(f"🏢 路口当前状态: {current_agents_in_intersection}个agent")
        
        # 默认所有agent都等待
        for winner_data in auction_priority:
            agent_control_status[winner_data['agent']['id']] = 'wait'

        # 优先处理受保护的agent（已在路口内）
        protected_agents = []
        for winner_data in auction_priority:
            if winner_data.get('protected', False):
                protected_agents.append(winner_data)
                agent_control_status[winner_data['agent']['id']] = 'go'
                print(f"🛡️ 受保护agent {winner_data['agent']['id']} 继续通行")

        # 如果路口容量已满，不允许新agent进入
        if current_agents_in_intersection >= self.max_concurrent_agents:
            print(f"🚫 路口容量已满 ({current_agents_in_intersection}/{self.max_concurrent_agents})，新agent等待")
            return agent_control_status

        # 按优先级允许新agent进入，但不超过容量限制
        agents_allowed = 0
        vehicles_allowed = 0
        
        for winner_data in auction_priority:
            agent = winner_data['agent']
            agent_id = agent['id']
            
            # 跳过已经在路口的agent
            if winner_data.get('protected', False):
                continue
                
            # 检查是否有容量
            agent_vehicle_count = len(agent['vehicles']) if agent['type'] == 'platoon' else 1
            
            if (agents_allowed < self.max_concurrent_agents):
                
                # 检查冲突
                has_conflict = False
                for conflict_group in conflict_groups:
                    if agent in conflict_group:
                        # 检查冲突组内是否有其他agent已经获得go状态
                        for other_agent in conflict_group:
                            if (other_agent['id'] != agent_id and 
                                agent_control_status.get(other_agent['id']) == 'go'):
                                has_conflict = True
                                break
                        if has_conflict:
                            break
                
                if not has_conflict:
                    agent_control_status[agent_id] = 'go'
                    agents_allowed += 1
                    vehicles_allowed += agent_vehicle_count
                    print(f"✅ 允许agent {agent_id} 进入路口 ({agents_allowed}/{self.max_concurrent_agents})")
                else:
                    print(f"🚦 Agent {agent_id} 因冲突等待")
            else:
                print(f"🚫 Agent {agent_id} 因容量限制等待")
                break  # 容量已满，后续agent都等待

        return agent_control_status

    def _is_agent_in_intersection(self, agent):
        """检查agent是否在路口内"""
        if agent['type'] == 'vehicle':
            return agent['data'].get('is_junction', False)
        elif agent['type'] == 'platoon':
            # 车队中任何一辆车在路口内就认为整个车队在路口内
            return any(v.get('is_junction', False) for v in agent['vehicles'])
        return False
    
    def _check_if_someone_in_group_passing(self, group_with_bids):
        """检查冲突组内是否有agent正在通过路口"""
        for item in group_with_bids:
            if self._is_agent_passing_intersection(item['agent']):
                return True
        return False

    def _is_agent_passing_intersection(self, agent):
        """检查agent是否正在通过路口"""
        if agent['type'] == 'vehicle':
            vehicle_id = agent['id']
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    location = carla_vehicle.get_location()
                    return self._is_vehicle_in_intersection(location)
            except:
                pass
            return False
        
        elif agent['type'] == 'platoon':
            # 检查车队是否有车辆在路口内
            for vehicle_state in agent['vehicles']:
                vehicle_id = vehicle_state['id']
                try:
                    carla_vehicle = self.world.get_actor(vehicle_id)
                    if carla_vehicle and carla_vehicle.is_alive:
                        location = carla_vehicle.get_location()
                        if self._is_vehicle_in_intersection(location):
                            return True
                except:
                    pass
        return False
    
    def _agent_has_problematic_vehicles(self, agent):
        """检查agent是否包含有问题的车辆"""
        if agent['type'] == 'vehicle':
            vehicle_id = agent['id']
            return (vehicle_id in self.crashed_vehicles or 
                   vehicle_id in self.stuck_vehicles or
                   vehicle_id in self.emergency_recovery_vehicles)
        elif agent['type'] == 'platoon':
            for vehicle in agent['vehicles']:
                vehicle_id = vehicle['id']
                if (vehicle_id in self.crashed_vehicles or 
                   vehicle_id in self.stuck_vehicles or
                   vehicle_id in self.emergency_recovery_vehicles):
                    return True
        return False

    def _identify_conflicting_agents(self, auction_priority):
        """识别有路径冲突的agents"""
        conflicting_ids = set()
        
        for i, winner1 in enumerate(auction_priority):
            for j, winner2 in enumerate(auction_priority[i+1:], i+1):
                agent1 = winner1['agent']
                agent2 = winner2['agent']
                
                dir1 = agent1.get('goal_direction', 'straight')
                dir2 = agent2.get('goal_direction', 'straight')
                
                if self._directions_have_conflict(dir1, dir2):
                    conflicting_ids.add(agent1['id'])
                    conflicting_ids.add(agent2['id'])
        
        return conflicting_ids

    def _directions_have_conflict(self, dir1, dir2):
        """判断两个方向是否冲突 - 使用ConflictResolver的完整冲突矩阵"""
        if dir1 == dir2:
            return False
        
        # 转换目标方向到路径代码
        turn1 = self._convert_direction_to_code(dir1)
        turn2 = self._convert_direction_to_code(dir2)
        
        if not turn1 or not turn2:
            return False
        
        # 对于不知道具体进入方向的情况，检查是否存在任何可能的冲突组合
        # 如果两个目标方向在任何进入方向组合下都会产生冲突，则认为冲突
        entry_directions = ['N', 'S', 'E', 'W']
        conflict_found = False
        
        # 检查所有可能的进入方向组合
        for entry1 in entry_directions:
            for entry2 in entry_directions:
                # 跳过相同进入方向（同一车道不会冲突）
                if entry1 == entry2:
                    continue
                    
                path1 = f"{entry1}_{turn1}"
                path2 = f"{entry2}_{turn2}"
                
                # 使用ConflictResolver的冲突矩阵检查
                if (path1 in self.conflict_resolver.conflict_matrix and 
                    path2 in self.conflict_resolver.conflict_matrix[path1] and
                    self.conflict_resolver.conflict_matrix[path1][path2]):
                    conflict_found = True
                    break
            
            if conflict_found:
                break
        
        return conflict_found

    def _convert_direction_to_code(self, direction):
        """将方向转换为代码"""
        direction_map = {
            'left': 'L',
            'straight': 'S', 
            'right': 'R'
        }
        return direction_map.get(direction)

    def _apply_single_vehicle_control(self, vehicle_id, rank, bid_value, control_modifier='normal'):
        """为单车agent应用控制 - 增加控制修饰符"""
        try:
            carla_vehicle = self.world.get_actor(vehicle_id)
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False

            # 根据排名和修饰符调整控制强度
            control_params = self._get_control_params_by_rank(rank, control_modifier)

            # 应用控制参数
            self.traffic_manager.vehicle_percentage_speed_difference(
                carla_vehicle, control_params['speed_diff']
            )
            self.traffic_manager.distance_to_leading_vehicle(
                carla_vehicle, control_params['follow_distance']
            )
            self.traffic_manager.ignore_lights_percentage(
                carla_vehicle, control_params['ignore_lights']
            )
            self.traffic_manager.ignore_signs_percentage(
                carla_vehicle, control_params['ignore_signs']
            )
            self.traffic_manager.ignore_vehicles_percentage(
                carla_vehicle, control_params['ignore_vehicles']
            )

            # 记录控制状态
            self.controlled_vehicles[vehicle_id] = {
                'type': 'single_vehicle',
                'rank': rank,
                'bid_value': bid_value,
                'control_params': control_params,
                'control_modifier': control_modifier
            }

            return True

        except Exception as e:
            print(f"[Warning] 单车控制失败 {vehicle_id}: {e}")
            return False

    def _apply_platoon_agent_control(self, platoon_vehicles, rank, bid_value, direction, control_modifier='normal'):
        """为车队agent应用控制 - 增加控制修饰符参数"""
        controlled_vehicles = set()

        try:
            for i, vehicle_state in enumerate(platoon_vehicles):
                vehicle_id = vehicle_state['id']
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    continue

                # 车队内角色：队长 vs 跟随者
                if i == 0:  # 队长
                    control_params = self._get_platoon_leader_params(rank, control_modifier)
                    role = 'platoon_leader'
                else:  # 跟随者
                    control_params = self._get_platoon_follower_params(rank, control_modifier)
                    role = 'platoon_follower'

                # 应用控制参数
                self.traffic_manager.vehicle_percentage_speed_difference(
                    carla_vehicle, control_params['speed_diff']
                )
                self.traffic_manager.distance_to_leading_vehicle(
                    carla_vehicle, control_params['follow_distance']
                )
                self.traffic_manager.ignore_lights_percentage(
                    carla_vehicle, control_params['ignore_lights']
                )
                self.traffic_manager.ignore_signs_percentage(
                    carla_vehicle, control_params['ignore_signs']
                )
                self.traffic_manager.ignore_vehicles_percentage(
                    carla_vehicle, control_params['ignore_vehicles']
                )

                # 记录控制状态
                self.controlled_vehicles[vehicle_id] = {
                    'type': role,
                    'rank': rank,
                    'bid_value': bid_value,
                    'direction': direction,
                    'control_params': control_params,
                    'control_modifier': control_modifier  # 添加这一行
                }

                controlled_vehicles.add(vehicle_id)

        except Exception as e:
            print(f"[Warning] 车队控制失败: {e}")

        return controlled_vehicles

    def _get_control_params_by_rank(self, rank, control_modifier='normal'):
        """根据拍卖排名和修饰符获取控制参数 - 更安全的参数"""
        # 基础参数（更保守）
        if rank == 1:
            base_params = {
                'speed_diff': -70.0,    # 从-60.0增加到-70.0，让第一名更激进
                'follow_distance': 1.2,  # 从1.5减少到1.2，更紧密跟随
                'ignore_lights': 90.0,   # 从85.0增加到90.0
                'ignore_signs': 80.0,    # 从75.0增加到80.0
                'ignore_vehicles': 50.0  # 从40.0增加到50.0
            }
        elif rank <= 2:
            base_params = {
                'speed_diff': -55.0,    # 从-45.0增加到-55.0
                'follow_distance': 1.8,  # 从2.0减少到1.8
                'ignore_lights': 75.0,   # 从70.0增加到75.0
                'ignore_signs': 65.0,    # 从60.0增加到65.0
                'ignore_vehicles': 35.0  # 从25.0增加到35.0
            }
        elif rank <= 3:
            base_params = {
                'speed_diff': -40.0,    # 从-30.0增加到-40.0
                'follow_distance': 2.2,  # 从2.5减少到2.2
                'ignore_lights': 60.0,   # 从50.0增加到60.0
                'ignore_signs': 50.0,    # 从40.0增加到50.0
                'ignore_vehicles': 25.0  # 从15.0增加到25.0
            }
        else:
            base_params = {
                'speed_diff': -20.0,    # 从-10.0增加到-20.0
                'follow_distance': 2.8,  # 从3.0减少到2.8
                'ignore_lights': 10.0,   # 从5.0增加到10.0
                'ignore_signs': 10.0,    # 从5.0增加到10.0
                'ignore_vehicles': 5.0   # 从0.0增加到5.0
            }
        
        # 根据修饰符调整参数
        if control_modifier == 'wait':
            # 强制等待的车辆使用非常保守的参数
            return {
                'speed_diff': 10.0,      # 减速
                'follow_distance': 4.0,  # 大跟车距离
                'ignore_lights': 0.0,    # 完全遵守信号
                'ignore_signs': 0.0,
                'ignore_vehicles': 0.0
            }
        elif control_modifier == 'cautious':
            # 有冲突风险的车辆使用谨慎参数
            base_params['speed_diff'] += 15.0  # 进一步减速
            base_params['follow_distance'] += 1.0
            base_params['ignore_lights'] = min(30.0, base_params['ignore_lights'])
            base_params['ignore_vehicles'] = min(10.0, base_params['ignore_vehicles'])
        
        return base_params

    def _get_platoon_leader_params(self, rank, control_modifier='normal'):
        """获取车队队长的控制参数 - 增强版"""
        base_params = self._get_control_params_by_rank(rank, control_modifier)
        
        # 🔥 车队队长获得更激进的参数确保带领整个车队通过
        if control_modifier != 'wait':
            base_params['speed_diff'] -= 15.0  # 更激进的速度
            base_params['ignore_vehicles'] = min(100.0, base_params['ignore_vehicles'] + 20.0)
            base_params['ignore_lights'] = min(100.0, base_params['ignore_lights'] + 15.0)
            base_params['follow_distance'] = max(0.8, base_params['follow_distance'] * 0.7)  # 更紧密
    
        return base_params

    def _get_platoon_follower_params(self, rank, control_modifier='normal'):
        """获取车队跟随者的控制参数 - 完全跟随队长"""
        base_params = self._get_control_params_by_rank(rank, control_modifier)
        
        # 🔥 车队跟随者完全跟随队长，保持固定跟随距离
        if control_modifier != 'wait':
            base_params['follow_distance'] = 1.0  # 跟随距离固定为1米
            base_params['ignore_lights'] = min(100.0, 100)
            base_params['ignore_signs'] = min(100.0, 100)
            base_params['ignore_vehicles'] = min(100.0, 100)
            base_params['speed_diff'] = 0.0  # 跟随者速度与队长保持一致
    
        return base_params
    
        return base_params
    
    def _restore_uncontrolled_vehicles(self, current_controlled_vehicles):
        """恢复不再被控制的车辆的默认行为"""
        vehicles_to_restore = self.current_controlled_vehicles - current_controlled_vehicles
        
        for vehicle_id in vehicles_to_restore:
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    self._restore_default_behavior(carla_vehicle)
                
                # 清除控制记录
                self.controlled_vehicles.pop(vehicle_id, None)
                
            except Exception as e:
                print(f"[Warning] 恢复车辆 {vehicle_id} 默认行为失败: {e}")
    
    def _restore_default_behavior(self, vehicle):
        """恢复车辆默认行为 - 防重复版 + 车队协调通过"""
        vehicle_id = vehicle.id
        current_time = time.time()
        
        try:
            # 检查车辆是否在路口内
            vehicle_location = vehicle.get_location()
            if self._is_vehicle_in_intersection(vehicle_location):
                # 检查是否已经在强制通过状态
                if vehicle_id not in self.intersection_pass_vehicles:
                    # 🔥 新增：检查是否为车队成员，确保车队协调通过
                    if self._is_vehicle_in_platoon(vehicle_id):
                        self._log_intersection_pass(vehicle_id, current_time, "车队成员在路口内，使用车队强制通过参数")
                        self._apply_platoon_intersection_pass_params(vehicle)
                    else:
                        self._log_intersection_pass(vehicle_id, current_time, "在路口内，使用强制通过参数")
                        self._apply_intersection_pass_params(vehicle)
                    self.intersection_pass_vehicles.add(vehicle_id)
            else:
                # 非路口内车辆恢复正常默认行为
                if vehicle_id in self.intersection_pass_vehicles:
                    self.intersection_pass_vehicles.discard(vehicle_id)
                    self._log_intersection_pass(vehicle_id, current_time, "离开路口，恢复默认行为")
        
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
                self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
                self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)

        except Exception as e:
            self._log_intersection_pass(vehicle_id, current_time, f"恢复行为失败: {e}")
            # 失败时使用默认恢复
            try:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
                self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
                self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
            except:
                pass

    def _is_vehicle_in_intersection(self, vehicle_location):
        """检查车辆是否在路口内（使用正方形区域）"""
        try:
            # 使用配置文件中的正方形检测方法
            return SimulationConfig.is_in_intersection_area(vehicle_location)
        except Exception as e:
            print(f"[Warning] 检查车辆是否在路口失败: {e}")
            return False

    # def _calculate_distance_to_intersection(self, location):
    #     """计算位置到路口中心的距离（保持兼容性）"""
    #     return SimulationConfig.distance_to_intersection_center(location)

    def emergency_reset_all_controls(self):
        """紧急重置所有控制"""
        try:
            print("🚨 执行紧急重置所有车辆控制...")
            
            # 重置所有受控车辆
            for vehicle_id in list(self.controlled_vehicles.keys()):
                try:
                    carla_vehicle = self.world.get_actor(vehicle_id)
                    if carla_vehicle and carla_vehicle.is_alive:
                        self._restore_default_behavior(carla_vehicle)
                except:
                    pass
            
            # 清空所有状态
            self.controlled_vehicles.clear()
            self.current_controlled_vehicles.clear()
            self.crashed_vehicles.clear()
            self.stuck_vehicles.clear()
            self.emergency_recovery_vehicles.clear()
            self.intersection_pass_vehicles.clear()
            self.last_positions.clear()
            self.last_control_log_time.clear()
            
            print("✅ 紧急重置完成")
            
        except Exception as e:
            print(f"[Error] 紧急重置失败: {e}")

    def get_safety_stats(self):
        """获取安全控制统计信息"""
        return {
            'controlled_vehicles': len(self.controlled_vehicles),
            'crashed_vehicles': len(self.crashed_vehicles),
            'stuck_vehicles': len(self.stuck_vehicles),
            'emergency_recovery_vehicles': len(self.emergency_recovery_vehicles),
            'intersection_pass_vehicles': len(self.intersection_pass_vehicles)
        }
    
    # def _ensure_intersection_vehicles_complete(self):
    #     """确保路口内的受控车辆完成通过 - 增强车队协调"""
    #     current_time = time.time()
        
    #     for vehicle_id, control_info in self.controlled_vehicles.items():
    #         try:
    #             carla_vehicle = self.world.get_actor(vehicle_id)
    #             if not carla_vehicle or not carla_vehicle.is_alive:
    #                 continue
                
    #             vehicle_location = carla_vehicle.get_location()
    #             if self._is_vehicle_in_intersection(vehicle_location):
    #                 # 🔥 增强：车队成员在路口内使用车队专用参数
    #                 if self._is_vehicle_in_platoon(vehicle_id):
    #                     self._apply_platoon_intersection_pass_params(carla_vehicle)
    #                     # 记录车队强制通过状态
    #                     platoon_info = self._get_vehicle_platoon_info(vehicle_id)
    #                     if platoon_info:
    #                         self._log_intersection_pass(vehicle_id, current_time, 
    #                             f"车队{platoon_info['platoon_id']}成员强制通过路口 (位置:{platoon_info['position_in_platoon']})")
    #                     else:
    #                         self._log_intersection_pass(vehicle_id, current_time, "车队成员强制通过路口")
    #                 else:
    #                     self._apply_intersection_pass_params(carla_vehicle)
    #                     self._log_intersection_pass(vehicle_id, current_time, "单车强制通过路口")
    #             else:
    #                 # 车辆已离开路口，恢复默认行为
    #                 self._restore_default_behavior(carla_vehicle)
                        
    #         except Exception as e:
    #             print(f"[Warning] 检查路口内车辆失败: {e}")

    def _is_vehicle_in_platoon(self, vehicle_id):
        """检查车辆是否属于某个车队 - 增强错误处理"""
        try:
            if hasattr(self, 'platoon_manager') and self.platoon_manager:
                # 遍历所有车队检查车辆是否在其中
                all_platoons = self.platoon_manager.get_all_platoons()
                for platoon in all_platoons:
                    if hasattr(platoon, 'vehicles') and platoon.vehicles:
                        platoon_vehicle_ids = [v['id'] for v in platoon.vehicles]
                        if vehicle_id in platoon_vehicle_ids:
                            return True
            return False
        except Exception as e:
            print(f"[Warning] 检查车辆{vehicle_id}是否在车队失败: {e}")
            return False

    def _get_vehicle_platoon_info(self, vehicle_id):
        """获取车辆所在车队的信息"""
        if hasattr(self, 'platoon_manager') and self.platoon_manager:
            for platoon in self.platoon_manager.get_all_platoons():
                platoon_vehicle_ids = [v['id'] for v in platoon.vehicles]
                if vehicle_id in platoon_vehicle_ids:
                    # 修正：使用正确的车队ID属性名
                    platoon_id = getattr(platoon, 'platoon_id', getattr(platoon, 'id', f'platoon_{hash(platoon)}'))
                    return {
                        'platoon_id': platoon_id,
                        'platoon_size': len(platoon.vehicles),
                        'is_leader': platoon.vehicles[0]['id'] == vehicle_id,
                        'position_in_platoon': platoon_vehicle_ids.index(vehicle_id)
                    }
        return None

    # def _apply_platoon_intersection_pass_params(self, carla_vehicle):
    #     """为路口内车队车辆应用更激进的强制通过参数（所有成员与队长完全一致）"""
    #     try:
    #         vehicle_id = carla_vehicle.id
    #         platoon_info = self._get_vehicle_platoon_info(vehicle_id)
            
    #         if platoon_info:
    #             # 所有成员都采用队长参数，确保同步
    #             self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -95.0)
    #             self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 0.8)
    #             self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
    #             self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
    #             self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 90.0)
    #         else:
    #             # 如果获取车队信息失败，使用默认车队参数
    #             self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -90.0)
    #             self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 0.2)
    #             self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
    #             self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
    #             self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 95.0)
                
    #     except Exception as e:
    #         print(f"[Warning] 应用车队路口强制通过参数失败 {carla_vehicle.id}: {e}")

    def _apply_intersection_pass_params(self, carla_vehicle):
        """为路口内单车应用强制通过参数"""
        try:
            # 单车在路口内的强制通过参数
            self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -70.0)
            self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 1.0)
            self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
            self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
            self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 80.0)
        except Exception as e:
            print(f"[Warning] 应用单车路口强制通过参数失败 {carla_vehicle.id}: {e}")

    def _log_intersection_pass(self, vehicle_id, current_time, message):
        """记录路口通过日志 - 避免重复输出"""
        # 限制日志输出频率
        if vehicle_id not in self.last_control_log_time:
            self.last_control_log_time[vehicle_id] = 0
        
        if current_time - self.last_control_log_time[vehicle_id] >= self.control_log_interval:
            print(f"🚧 [路口控制] 车辆{vehicle_id}: {message}")
            self.last_control_log_time[vehicle_id] = current_time