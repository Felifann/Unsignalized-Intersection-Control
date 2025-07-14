import carla
import math
import time

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
        
        # 添加交叉口中心和半径配置
        from env.simulation_config import SimulationConfig
        self.intersection_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        self.intersection_radius = SimulationConfig.INTERSECTION_RADIUS
        
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
    
        print("🎮 基于拍卖的交通控制器初始化完成 - 集成安全控制")
    
    def update_control(self, platoon_manager, auction_engine=None):
        """
        主控制更新函数 - 增加安全检查和撞车恢复
        """
        # 1. 检测和处理撞车车辆
        self._detect_and_handle_crashes()
        
        # 2. 检测和处理卡住的车辆
        self._detect_and_handle_stuck_vehicles()
        
        # 3. 确保路口内车辆完成通过
        self._ensure_intersection_vehicles_complete()
        
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
                # 如果车辆已经恢复，从撞车列表中移除
                if vehicle_id in self.crashed_vehicles:
                    self.crashed_vehicles.discard(vehicle_id)
                    self.emergency_recovery_vehicles.discard(vehicle_id)
                    print(f"✅ 车辆 {vehicle_id} 已恢复正常")

    def _is_vehicle_crashed(self, vehicle):
        """判断车辆是否撞车"""
        try:
            # 检查1：车辆速度是否异常低且有碰撞历史
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            # 检查2：车辆是否卡在不合理的位置
            transform = vehicle.get_transform()
            location = transform.location
            
            # 获取车辆的碰撞边界框
            bounding_box = vehicle.bounding_box
            
            # 检查是否与其他车辆重叠
            for other_vehicle in self.world.get_actors().filter('vehicle.*'):
                if other_vehicle.id == vehicle.id or not other_vehicle.is_alive:
                    continue
                    
                other_location = other_vehicle.get_transform().location
                distance = location.distance(other_location)
                
                # 如果两车距离过近且速度都很低，可能发生碰撞
                if distance < 3.0:  # 小于3米
                    other_velocity = other_vehicle.get_velocity()
                    other_speed = math.sqrt(other_velocity.x**2 + other_velocity.y**2)
                    
                    if speed < 1.0 and other_speed < 1.0:  # 两车都几乎静止
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
            
            # 移除set_path调用，使用其他方法帮助车辆恢复
            # 设置更保守的行为参数
            self.traffic_manager.set_desired_speed(vehicle, 10.0)  # 设置较低的目标速度
            self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)  # 全局增大跟车距离
            
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
        """基于拍卖结果应用统一控制 - 增加安全检查"""
        controlled_vehicles = set()
        
        if not auction_priority:
            return controlled_vehicles
        
        print(f"🎯 基于拍卖结果应用控制，共{len(auction_priority)}个获胜agents")
        
        # 分析是否有冲突路径的agents
        conflicting_agents = self._identify_conflicting_agents(auction_priority)
        
        for winner_data in auction_priority:
            agent = winner_data['agent']
            bid_value = winner_data['bid_value']
            rank = winner_data['rank']
            conflict_action = winner_data.get('conflict_action', 'go')
            
            # 安全检查：如果涉及撞车或卡住的车辆，跳过控制
            if self._agent_has_problematic_vehicles(agent):
                print(f"⚠️ Agent {agent['id']} 包含问题车辆，跳过控制")
                continue
            
            # 如果被冲突解决器要求等待，使用更保守的控制参数
            if conflict_action == 'wait':
                control_modifier = 'wait'
            elif agent['id'] in conflicting_agents and rank > 1:
                control_modifier = 'cautious'  # 冲突路径的非第一名使用谨慎参数
            else:
                control_modifier = 'normal'
            
            try:
                if agent['type'] == 'vehicle':
                    vehicle_id = agent['id']
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value, control_modifier):
                        controlled_vehicles.add(vehicle_id)
                        action_emoji = "🟢" if conflict_action == 'go' else "🔴"
                        print(f"   🏆 #{rank}: {action_emoji}🚗单车{vehicle_id} (出价:{bid_value:.1f})")
                    
                elif agent['type'] == 'platoon':
                    platoon_vehicles = agent['vehicles']
                    direction = agent['goal_direction']
                    controlled_in_platoon = self._apply_platoon_agent_control(
                        platoon_vehicles, rank, bid_value, direction, control_modifier
                    )
                    controlled_vehicles.update(controlled_in_platoon)
                    
                    action_emoji = "🟢" if conflict_action == 'go' else "🔴"
                    print(f"   🏆 #{rank}: {action_emoji}🚛车队{agent['id']} "
                          f"({len(platoon_vehicles)}车-{direction}) (出价:{bid_value:.1f})")
                
            except Exception as e:
                print(f"[Warning] agent {agent['id']} 控制应用失败: {e}")
        
        return controlled_vehicles

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
        """判断两个方向是否冲突"""
        if dir1 == dir2:
            return False
            
        conflict_rules = {
            ('left', 'straight'): True,
            ('left', 'right'): True,
            ('straight', 'left'): True,
            ('straight', 'right'): False,
            ('right', 'left'): True,
            ('right', 'straight'): False,
        }
        
        return conflict_rules.get((dir1, dir2), False)

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
        """获取车队队长的控制参数"""
        base_params = self._get_control_params_by_rank(rank, control_modifier)
        
        # 只有在非等待模式下才应用队长的激进参数
        if control_modifier != 'wait':
            base_params['speed_diff'] -= 5.0
            base_params['ignore_vehicles'] = min(100.0, base_params['ignore_vehicles'] + 10.0)
        
        return base_params

    def _get_platoon_follower_params(self, rank, control_modifier='normal'):
        """获取车队跟随者的控制参数 - 更安全的跟车"""
        base_params = self._get_control_params_by_rank(rank, control_modifier)
        
        # 只有在非等待模式下才应用跟随者的紧密跟随参数
        if control_modifier != 'wait':
            base_params['follow_distance'] = max(1.5, base_params['follow_distance'] * 0.8)  # 不能太小
            base_params['ignore_lights'] = min(100.0, base_params['ignore_lights'] + 10.0)
            base_params['ignore_signs'] = min(100.0, base_params['ignore_signs'] + 10.0)
            base_params['ignore_vehicles'] = min(30.0, base_params['ignore_vehicles'] + 5.0)  # 限制最大值
        
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
        """恢复车辆默认行为 - 防重复版"""
        vehicle_id = vehicle.id
        current_time = time.time()
        
        try:
            # 检查车辆是否在路口内
            vehicle_location = vehicle.get_location()
            if self._is_vehicle_in_intersection(vehicle_location):
                # 检查是否已经在强制通过状态
                if vehicle_id not in self.intersection_pass_vehicles:
                    # 路口内车辆使用强制通过参数
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

    def _log_intersection_pass(self, vehicle_id, current_time, message):
        """有限制的路口通过日志输出"""
        # 检查是否需要输出日志（限制频率）
        if vehicle_id not in self.last_control_log_time:
            self.last_control_log_time[vehicle_id] = 0
    
        if current_time - self.last_control_log_time[vehicle_id] >= self.control_log_interval:
            print(f"🚧 车辆 {vehicle_id} {message}")
            self.last_control_log_time[vehicle_id] = current_time

    def _ensure_intersection_vehicles_complete(self):
        """确保路口内的受控车辆完成通过"""
        current_time = time.time()
        
        for vehicle_id, control_info in self.controlled_vehicles.items():
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    continue
                
                vehicle_location = carla_vehicle.get_location()
                if self._is_vehicle_in_intersection(vehicle_location):
                    # 应用强制通过参数
                    self._apply_intersection_pass_params(carla_vehicle)
                else:
                    # 车辆已离开路口，恢复默认行为
                    self._restore_default_behavior(carla_vehicle)
                        
            except Exception as e:
                print(f"[Warning] 检查路口内车辆失败: {e}")

    def _is_vehicle_in_intersection(self, location):
        """判断车辆是否在路口区域"""
        dx = location.x - self.intersection_center[0]
        dy = location.y - self.intersection_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.intersection_radius

    def _apply_intersection_pass_params(self, carla_vehicle):
        """为路口内车辆应用强制通过参数"""
        try:
            self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -80.0)
            self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 0.3)
            self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
            self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
            self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 90.0)
        except Exception as e:
            print(f"[Warning] 应用路口强制通过参数失败 {carla_vehicle.id}: {e}")

    def emergency_reset_all_controls(self):
        """紧急重置所有车辆控制 - 增强版"""
        print("🚨 紧急重置所有车辆控制")
        
        # 1. 重置所有受控车辆
        for vehicle_id in list(self.controlled_vehicles.keys()):
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    self._restore_default_behavior(carla_vehicle)
            except:
                continue
        
        # 2. 重置所有撞车和卡住的车辆
        all_vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in all_vehicles:
            if vehicle.is_alive:
                try:
                    self._restore_default_behavior(vehicle)
                except:
                    continue
        
        # 3. 清空所有状态记录
        self.controlled_vehicles.clear()
        self.current_controlled_vehicles.clear()
        self.crashed_vehicles.clear()
        self.stuck_vehicles.clear()
        self.emergency_recovery_vehicles.clear()
        self.last_positions.clear()
        
        # 4. 清空新增的状态记录
        self.intersection_pass_vehicles.clear()
        self.last_control_log_time.clear()
        
        print("✅ 所有车辆已恢复默认行为，状态已重置")

    def get_safety_stats(self):
        """获取安全统计信息 - 增强版"""
        return {
            'crashed_vehicles': len(self.crashed_vehicles),
            'stuck_vehicles': len(self.stuck_vehicles),
            'emergency_recovery_vehicles': len(self.emergency_recovery_vehicles),
            'controlled_vehicles': len(self.current_controlled_vehicles),
            'intersection_pass_vehicles': len(self.intersection_pass_vehicles)  # 新增
        }