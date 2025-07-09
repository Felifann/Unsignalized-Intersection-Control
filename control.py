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
        主控制更新函数 - 完全基于拍卖结果
        Args:
            platoon_manager: 车队管理器实例（用于验证）
            auction_engine: 拍卖引擎实例
        """
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
        """基于拍卖结果应用统一控制"""
        controlled_vehicles = set()
        
        if not auction_priority:
            return controlled_vehicles
        
        print(f"🎯 基于拍卖结果应用控制，共{len(auction_priority)}个获胜agents")
        
        for winner_data in auction_priority:
            agent = winner_data['agent']
            bid_value = winner_data['bid_value']
            rank = winner_data['rank']
            
            try:
                if agent['type'] == 'vehicle':
                    # 单车agent控制
                    vehicle_id = agent['id']
                    if self._apply_single_vehicle_control(vehicle_id, rank, bid_value):
                        controlled_vehicles.add(vehicle_id)
                        print(f"   🏆 #{rank}: 🚗单车{vehicle_id} (出价:{bid_value:.1f})")
                    
                elif agent['type'] == 'platoon':
                    # 车队agent控制
                    platoon_vehicles = agent['vehicles']
                    direction = agent['goal_direction']
                    controlled_in_platoon = self._apply_platoon_agent_control(
                        platoon_vehicles, rank, bid_value, direction
                    )
                    controlled_vehicles.update(controlled_in_platoon)
                    
                    print(f"   🏆 #{rank}: 🚛车队{agent['id']} "
                          f"({len(platoon_vehicles)}车-{direction}) (出价:{bid_value:.1f})")
                
            except Exception as e:
                print(f"[Warning] agent {agent['id']} 控制应用失败: {e}")
        
        return controlled_vehicles

    def _apply_single_vehicle_control(self, vehicle_id, rank, bid_value):
        """为单车agent应用控制"""
        try:
            carla_vehicle = self.world.get_actor(vehicle_id)
            if not carla_vehicle or not carla_vehicle.is_alive:
                return False
            
            # 根据排名调整控制强度
            control_params = self._get_control_params_by_rank(rank)
            
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
                'control_params': control_params
            }
            
            return True
            
        except Exception as e:
            print(f"[Warning] 单车控制失败 {vehicle_id}: {e}")
            return False

    def _apply_platoon_agent_control(self, platoon_vehicles, rank, bid_value, direction):
        """为车队agent应用控制"""
        controlled_vehicles = set()
        
        try:
            for i, vehicle_state in enumerate(platoon_vehicles):
                vehicle_id = vehicle_state['id']
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    continue
                
                # 车队内角色：队长 vs 跟随者
                if i == 0:  # 队长
                    control_params = self._get_platoon_leader_params(rank)
                    role = 'platoon_leader'
                else:  # 跟随者
                    control_params = self._get_platoon_follower_params(rank)
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
                    'control_params': control_params
                }
                
                controlled_vehicles.add(vehicle_id)
                
        except Exception as e:
            print(f"[Warning] 车队控制失败: {e}")
        
        return controlled_vehicles

    def _get_control_params_by_rank(self, rank):
        """根据拍卖排名获取控制参数"""
        if rank == 1:  # 第一名：最激进
            return {
                'speed_diff': -70.0,
                'follow_distance': 0.8,
                'ignore_lights': 95.0,
                'ignore_signs': 85.0,
                'ignore_vehicles': 60.0
            }
        elif rank <= 3:  # 前三名：较激进
            return {
                'speed_diff': -55.0,
                'follow_distance': 1.0,
                'ignore_lights': 80.0,
                'ignore_signs': 70.0,
                'ignore_vehicles': 45.0
            }
        elif rank <= 5:  # 前五名：中等
            return {
                'speed_diff': -45.0,
                'follow_distance': 1.2,
                'ignore_lights': 60.0,
                'ignore_signs': 50.0,
                'ignore_vehicles': 30.0
            }
        else:  # 其他：温和
            return {
                'speed_diff': -35.0,
                'follow_distance': 1.5,
                'ignore_lights': 40.0,
                'ignore_signs': 30.0,
                'ignore_vehicles': 20.0
            }

    def _get_platoon_leader_params(self, rank):
        """获取车队队长的控制参数"""
        base_params = self._get_control_params_by_rank(rank)
        # 队长稍微激进一些
        base_params['speed_diff'] -= 5.0
        base_params['ignore_vehicles'] += 10.0
        return base_params

    def _get_platoon_follower_params(self, rank):
        """获取车队跟随者的控制参数"""
        base_params = self._get_control_params_by_rank(rank)
        # 跟随者更紧密跟随
        base_params['follow_distance'] *= 0.7
        base_params['ignore_lights'] = min(100.0, base_params['ignore_lights'] + 15.0)
        base_params['ignore_signs'] = min(100.0, base_params['ignore_signs'] + 15.0)
        base_params['ignore_vehicles'] += 15.0
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
        """恢复车辆默认行为"""
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
        self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
        self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
        self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
    
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
            print(f"🎮 拍卖控制状态: 总控制{stats['total_controlled_vehicles']}辆 | "
                  f"单车{stats['single_vehicle_controlled']}辆 | "
                  f"车队{stats['total_platoon_controlled']}辆 "
                  f"(队长{stats['platoon_leader_controlled']}+跟随{stats['platoon_follower_controlled']})")
    
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