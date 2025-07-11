import carla
import math

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
        
        # 控制状态记录
        self.controlled_vehicles = {}  # {vehicle_id: control_info}
        self.current_controlled_vehicles = set()
        
        # 控制参数
        self.default_speed_diff = -40.0  # 默认速度差异
        self.default_follow_distance = 1.5  # 默认跟车距离
        
        print("🎮 基于拍卖的交通控制器初始化完成")
    
    def update_control(self, platoon_manager, auction_engine=None):
        """
        主控制更新函数 - 完全基于拍卖结果，增加路口内车辆检查
        Args:
            platoon_manager: 车队管理器实例（用于验证）
            auction_engine: 拍卖引擎实例
        """
        # 首先检查并处理路口内的已控制车辆
        self._ensure_intersection_vehicles_complete()
        
        # 获取拍卖优先级排序
        auction_priority = []
        if auction_engine:
            auction_priority = auction_engine._get_current_priority_order()
        
        # 基于拍卖结果应用控制
        current_controlled = set()
        if auction_priority:
            current_controlled = self._apply_auction_based_control(auction_priority)
        
        # 恢复不再被控制的车辆
        self._restore_uncontrolled_vehicles(current_controlled)
        
        # 更新当前控制状态
        self.current_controlled_vehicles = current_controlled

    def _apply_auction_based_control(self, auction_priority):
        """基于拍卖结果应用统一控制 - 改进冲突控制"""
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
        """根据拍卖排名和修饰符获取控制参数"""
        # 基础参数
        if rank == 1:
            base_params = {
                'speed_diff': -80.0,
                'follow_distance': 0.5,
                'ignore_lights': 98.0,
                'ignore_signs': 90.0,
                'ignore_vehicles': 70.0
            }
        elif rank <= 2:
            base_params = {
                'speed_diff': -60.0,
                'follow_distance': 0.8,
                'ignore_lights': 85.0,
                'ignore_signs': 75.0,
                'ignore_vehicles': 50.0
            }
        elif rank <= 3:
            base_params = {
                'speed_diff': -45.0,
                'follow_distance': 1.0,
                'ignore_lights': 70.0,
                'ignore_signs': 60.0,
                'ignore_vehicles': 35.0
            }
        else:
            base_params = {
                'speed_diff': -20.0,
                'follow_distance': 2.0,
                'ignore_lights': 10.0,
                'ignore_signs': 10.0,
                'ignore_vehicles': 5.0
            }
        
        # 根据修饰符调整参数
        if control_modifier == 'wait':
            # 强制等待的车辆使用非常保守的参数
            return {
                'speed_diff': 0.0,      # 正常速度
                'follow_distance': 3.0,  # 大跟车距离
                'ignore_lights': 0.0,    # 完全遵守信号
                'ignore_signs': 0.0,
                'ignore_vehicles': 0.0
            }
        elif control_modifier == 'cautious':
            # 有冲突风险的车辆使用谨慎参数
            base_params['speed_diff'] += 20.0  # 减速
            base_params['follow_distance'] += 0.5
            base_params['ignore_lights'] = min(50.0, base_params['ignore_lights'])
            base_params['ignore_vehicles'] = min(30.0, base_params['ignore_vehicles'])
        
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
        """获取车队跟随者的控制参数"""
        base_params = self._get_control_params_by_rank(rank, control_modifier)
        
        # 只有在非等待模式下才应用跟随者的紧密跟随参数
        if control_modifier != 'wait':
            base_params['follow_distance'] *= 0.7
            base_params['ignore_lights'] = min(100.0, base_params['ignore_lights'] + 15.0)
            base_params['ignore_signs'] = min(100.0, base_params['ignore_signs'] + 15.0)
            base_params['ignore_vehicles'] = min(100.0, base_params['ignore_vehicles'] + 15.0)
        
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
        """恢复车辆默认行为 - 增强版：确保路口内车辆完成通过"""
        try:
            # 检查车辆是否在路口内
            vehicle_location = vehicle.get_location()
            if self._is_vehicle_in_intersection(vehicle_location):
                # 路口内车辆使用强制通过参数
                print(f"🚧 车辆 {vehicle.id} 在路口内，使用强制通过参数")
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -60.0)  # 加速通过
                self.traffic_manager.distance_to_leading_vehicle(vehicle, 0.5)  # 紧密跟随
                self.traffic_manager.ignore_lights_percentage(vehicle, 100.0)  # 忽略信号灯
                self.traffic_manager.ignore_signs_percentage(vehicle, 100.0)  # 忽略标志
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 80.0)  # 部分忽略其他车辆
            else:
                # 非路口内车辆恢复正常默认行为
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
                self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
                self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
        except Exception as e:
            print(f"[Warning] 恢复车辆 {vehicle.id} 行为失败: {e}")
            # 失败时使用默认恢复
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
            self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
            self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
            self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)

    def _is_vehicle_in_intersection(self, vehicle_location):
        """检查车辆是否在路口内部"""
        try:
            # 获取路口中心位置（假设在原点附近）
            intersection_center = carla.Location(x=0.0, y=0.0, z=0.0)
            
            # 计算车辆到路口中心的距离
            distance_to_center = math.sqrt(
                (vehicle_location.x - intersection_center.x) ** 2 + 
                (vehicle_location.y - intersection_center.y) ** 2
            )
            
            # 路口半径（可根据实际路口大小调整）
            intersection_radius = 30.0  # 米
            
            return distance_to_center <= intersection_radius
            
        except Exception as e:
            print(f"[Warning] 检查路口位置失败: {e}")
            return False

    def _ensure_intersection_vehicles_complete(self):
        """确保路口内的受控车辆完成通过"""
        for vehicle_id, control_info in self.controlled_vehicles.items():
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    continue
                
                vehicle_location = carla_vehicle.get_location()
                if self._is_vehicle_in_intersection(vehicle_location):
                    # 路口内车辆强制使用通过参数
                    control_modifier = control_info.get('control_modifier', 'normal')
                    if control_modifier == 'wait':
                        # 即使是等待状态的车辆，在路口内也要强制通过
                        print(f"🚧 强制路口内等待车辆 {vehicle_id} 完成通过")
                        self._apply_intersection_pass_params(carla_vehicle)
                        
            except Exception as e:
                print(f"[Warning] 检查路口内车辆 {vehicle_id} 失败: {e}")

    def _apply_intersection_pass_params(self, carla_vehicle):
        """为路口内车辆应用强制通过参数"""
        self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -60.0)
        self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 0.5)
        self.traffic_manager.ignore_lights_percentage(carla_vehicle, 100.0)
        self.traffic_manager.ignore_signs_percentage(carla_vehicle, 100.0)
        self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 80.0)

    def get_control_stats(self):
        """获取控制统计信息"""
        total_controlled = len(self.current_controlled_vehicles)
        
        # 统计不同类型的控制
        single_vehicle_count = len([v for v in self.controlled_vehicles.values() 
                                   if v['type'] == 'single_vehicle'])
        platoon_leader_count = len([v for v in self.controlled_vehicles.values() 
                                   if v['type'] == 'platoon_leader'])
        platoon_follower_count = len([v for v in self.controlled_vehicles.values() 
                                     if v['type'] == 'platoon_follower'])
        
        return {
            'total_controlled_vehicles': total_controlled,
            'single_vehicle_controlled': single_vehicle_count,
            'platoon_leader_controlled': platoon_leader_count,
            'platoon_follower_controlled': platoon_follower_count,
            'total_platoon_controlled': platoon_leader_count + platoon_follower_count
        }
    
    def print_control_status(self):
        """打印控制状态"""
        stats = self.get_control_stats()
        
        if stats['total_controlled_vehicles'] > 0:
            print(f"🎮 路口控制状态: 总控制{stats['total_controlled_vehicles']}辆 | "
                  f"单车{stats['single_vehicle_controlled']}辆 | "
                  f"车队{stats['total_platoon_controlled']}辆 "
                  f"(队长{stats['platoon_leader_controlled']}+跟随{stats['platoon_follower_controlled']}) | "
                  f"优先通行 vs 让行控制")

    def emergency_reset_all_controls(self):
        """紧急重置所有控制"""
        print("🚨 紧急重置所有车辆控制")
        
        for vehicle_id in list(self.controlled_vehicles.keys()):
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    self._restore_default_behavior(carla_vehicle)
            except:
                continue
        
        self.controlled_vehicles.clear()
        self.current_controlled_vehicles.clear()
        
        print("✅ 所有车辆已恢复默认行为")