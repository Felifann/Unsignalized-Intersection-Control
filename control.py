import carla
import math

class TrafficController:
    """
    集成多种控制逻辑的交通控制器
    目前支持：车队协调控制
    未来可扩展：拍卖系统、纳什均衡等
    """
    
    def __init__(self, carla_wrapper, state_extractor):
        self.carla = carla_wrapper
        self.state_extractor = state_extractor
        self.world = carla_wrapper.world
        self.traffic_manager = carla_wrapper.client.get_trafficmanager()
        
        # 控制状态记录
        self.controlled_vehicles = {}  # {vehicle_id: control_type}
        self.platoon_controlled_vehicles = set()  # 当前被车队控制的车辆ID
        
        # 控制参数
        self.default_speed_diff = -40.0  # 默认速度差异
        self.default_follow_distance = 1.5  # 默认跟车距离
        
        print("🎮 交通控制器初始化完成")
    
    def update_control(self, platoon_manager):
        """
        主控制更新函数
        Args:
            platoon_manager: 车队管理器实例
        """
        # 1. 获取当前车队信息
        current_platoons = platoon_manager.get_all_platoons()
        
        # 2. 车队协调控制
        self._apply_platoon_control(current_platoons)
        
        # 3. 恢复非车队车辆的默认行为
        self._restore_non_platoon_vehicles(current_platoons)
        
        # 4. 未来可在此添加其他控制逻辑
        # self._apply_auction_control()
        # self._apply_nash_control()
    
    def _apply_platoon_control(self, platoons):
        """应用车队协调控制"""
        current_platoon_vehicles = set()
        
        for platoon in platoons:
            if self._should_activate_platoon_control(platoon):
                self._execute_platoon_coordination(platoon)
                
                # 记录被控制的车辆
                for vehicle_state in platoon.vehicles:
                    current_platoon_vehicles.add(vehicle_state['id'])
        
        # 更新车队控制车辆列表
        self.platoon_controlled_vehicles = current_platoon_vehicles
    
    def _should_activate_platoon_control(self, platoon):
        """判断是否应该激活车队控制 - 仅针对目标无信号灯路口"""
        if platoon.get_size() < 2:
            return False
        
        leader = platoon.get_leader()
        if not leader:
            return False
        
        # 🔧 验证是否在目标无信号灯路口范围内
        if not self._is_in_target_intersection(leader):
            return False
        
        # 🔧 验证路口是否为无信号灯路口
        if not self._is_unsignalized_intersection(leader):
            return False
        
        # 计算队长到交叉口的距离
        leader_location = leader['location']
        intersection_center = (-188.9, -89.7, 0.0)  # 从配置获取
        dist_to_center = math.sqrt(
            (leader_location[0] - intersection_center[0])**2 + 
            (leader_location[1] - intersection_center[1])**2
        )
        
        # 🔧 放宽激活距离，提前开始协调控制
        is_approaching = dist_to_center < 40  # 从20米放宽到40米
        is_in_junction = leader['is_junction']
        
        # 检查车队相邻性
        is_adjacent = self._verify_platoon_adjacency_relaxed(platoon)
        
        # 🔧 增加调试信息
        if platoon.get_size() >= 2:
            print(f"🔍 目标路口车队检查 [队长:{leader['id']}]: "
                  f"距离{dist_to_center:.1f}m, 接近中:{is_approaching}, "
                  f"在路口:{is_in_junction}, 相邻:{is_adjacent}")
        
        return (is_approaching or is_in_junction) and is_adjacent

    def _is_in_target_intersection(self, vehicle_state):
        """验证车辆是否在目标交叉口范围内"""
        from env.simulation_config import SimulationConfig
        
        vehicle_location = vehicle_state['location']
        target_center = SimulationConfig.TARGET_INTERSECTION_CENTER
        target_radius = SimulationConfig.INTERSECTION_RADIUS
        
        distance = math.sqrt(
            (vehicle_location[0] - target_center[0])**2 + 
            (vehicle_location[1] - target_center[1])**2
        )
        
        is_in_target = distance <= target_radius
        
        if not is_in_target:
            print(f"🚫 车辆{vehicle_state['id']}不在目标路口范围内 (距离{distance:.1f}m > {target_radius}m)")
        
        return is_in_target

    def _is_unsignalized_intersection(self, vehicle_state):
        """验证是否为无信号灯路口"""
        try:
            # 获取车辆当前位置的waypoint
            vehicle_location = carla.Location(
                x=vehicle_state['location'][0],
                y=vehicle_state['location'][1], 
                z=vehicle_state['location'][2]
            )
            
            world_map = self.world.get_map()
            waypoint = world_map.get_waypoint(vehicle_location)
            
            if waypoint and waypoint.is_junction:
                # 检查路口是否有交通信号灯
                traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
                
                for traffic_light in traffic_lights:
                    light_location = traffic_light.get_location()
                    distance_to_light = math.sqrt(
                        (vehicle_location.x - light_location.x)**2 + 
                        (vehicle_location.y - light_location.y)**2
                    )
                    
                    # 如果50米内有交通信号灯，则认为是有信号灯路口
                    if distance_to_light < 50:
                        print(f"🚦 车辆{vehicle_state['id']}在有信号灯路口，跳过控制")
                        return False
                
                print(f"✅ 车辆{vehicle_state['id']}在无信号灯路口，可以控制")
                return True
            else:
                # 不在路口或接近路口的车辆也可以控制
                return True
                
        except Exception as e:
            print(f"[Warning] 检查路口信号灯状态失败: {e}")
            return True  # 发生错误时默认允许控制

    def _verify_platoon_adjacency_relaxed(self, platoon):
        """放宽的相邻性验证"""
        vehicles = platoon.vehicles
        if len(vehicles) < 2:
            return True
        
        max_distance = 20.0  # 从15米放宽到20米
        for i in range(len(vehicles) - 1):
            x1, y1, _ = vehicles[i]['location']
            x2, y2, _ = vehicles[i+1]['location']
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance > max_distance:
                print(f"⚠️ 车队相邻性检查失败: 车辆{vehicles[i]['id']}到{vehicles[i+1]['id']}距离{distance:.1f}m > {max_distance}m")
                return False
        
        return True
    
    def _execute_platoon_coordination(self, platoon):
        """执行车队协调控制 - 仅针对目标无信号灯路口"""
        direction = platoon.get_goal_direction()
        platoon_size = platoon.get_size()
        
        # 🔧 明确标识这是针对无信号灯路口的控制
        print(f"🚦 [无信号灯路口] 激活车队协调控制: {platoon_size}车编队 {direction}向通过")
        
        for i, vehicle_state in enumerate(platoon.vehicles):
            vehicle_id = vehicle_state['id']
            
            # 🔧 再次验证车辆是否在目标路口
            if not self._is_in_target_intersection(vehicle_state):
                print(f"⚠️ 跳过车辆{vehicle_id}: 不在目标无信号灯路口范围内")
                continue
            
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if not carla_vehicle or not carla_vehicle.is_alive:
                    continue
                
                if i == 0:  # 队长
                    self._apply_leader_control(carla_vehicle, direction)
                    print(f"   🔰 [无信号灯路口] 队长 {vehicle_id}: 引导通过路口")
                else:  # 跟随者
                    self._apply_follower_control(carla_vehicle, direction)
                    print(f"   🚗 [无信号灯路口] 成员{i} {vehicle_id}: 紧跟队长通过")
                
                # 记录控制状态
                self.controlled_vehicles[vehicle_id] = 'platoon_unsignalized'
                
            except Exception as e:
                print(f"[Warning] 控制车辆 {vehicle_id} 失败: {e}")
    
    def _apply_leader_control(self, vehicle, direction):
        """为队长应用更激进的控制策略"""
        # 🔧 队长更加激进地通过路口
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -60.0)  # 提速60%
        self.traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)  # 缩短跟车距离
        
        # 🔧 队长几乎忽略所有交通规则
        self.traffic_manager.ignore_lights_percentage(vehicle, 90.0)  # 90%忽略红绿灯
        self.traffic_manager.ignore_signs_percentage(vehicle, 80.0)   # 80%忽略交通标志
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 50.0)  # 50%忽略其他车辆
        
        print(f"🔰 队长 {vehicle.id} 激活激进通行模式 ({direction}向)")

    def _apply_follower_control(self, vehicle, direction):
        """为跟随者应用紧跟控制策略"""
        # 🔧 跟随者紧密跟随，忽略几乎所有规则
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -50.0)  # 提速50%
        self.traffic_manager.distance_to_leading_vehicle(vehicle, 0.5)  # 极短跟车距离
        
        # 🔧 跟随者完全忽略交通规则
        self.traffic_manager.ignore_lights_percentage(vehicle, 100.0)  # 完全忽略红绿灯
        self.traffic_manager.ignore_signs_percentage(vehicle, 100.0)   # 完全忽略交通标志
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 70.0)  # 70%忽略其他车辆
        
        print(f"🚗 跟随者 {vehicle.id} 激活紧跟模式")
    
    def _restore_non_platoon_vehicles(self, current_platoons):
        """恢复非车队车辆的默认行为"""
        # 获取当前车队中的所有车辆ID
        current_platoon_vehicle_ids = set()
        for platoon in current_platoons:
            for vehicle_state in platoon.vehicles:
                current_platoon_vehicle_ids.add(vehicle_state['id'])
        
        # 找出之前被控制但现在不在车队中的车辆
        vehicles_to_restore = self.platoon_controlled_vehicles - current_platoon_vehicle_ids
        
        for vehicle_id in vehicles_to_restore:
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    self._restore_default_behavior(carla_vehicle)
                    print(f"🔄 恢复车辆 {vehicle_id} 默认行为")
                
                # 从控制记录中移除
                self.controlled_vehicles.pop(vehicle_id, None)
                
            except Exception as e:
                print(f"[Warning] 恢复车辆 {vehicle_id} 默认行为失败: {e}")
    
    def _restore_default_behavior(self, vehicle):
        """恢复车辆默认行为"""
        # 恢复默认交通管理参数
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, self.default_speed_diff)
        self.traffic_manager.distance_to_leading_vehicle(vehicle, self.default_follow_distance)
        
        # 恢复交通规则遵守
        self.traffic_manager.ignore_lights_percentage(vehicle, 0.0)
        self.traffic_manager.ignore_signs_percentage(vehicle, 0.0)
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
    
    def get_control_stats(self):
        """获取控制统计信息"""
        total_controlled = len(self.controlled_vehicles)
        platoon_controlled = len(self.platoon_controlled_vehicles)
        
        return {
            'total_controlled_vehicles': total_controlled,
            'platoon_controlled_vehicles': platoon_controlled,
            'control_types': {
                'platoon': platoon_controlled,
                'auction': 0,  # 未来实现
                'nash': 0      # 未来实现
            }
        }
    
    def print_control_status(self):
        """打印控制状态（调试用）"""
        stats = self.get_control_stats()
        
        if stats['total_controlled_vehicles'] > 0:
            print(f"🎮 交通控制状态: 总控制{stats['total_controlled_vehicles']}辆 | "
                  f"车队控制{stats['platoon_controlled_vehicles']}辆")
    
    def print_detailed_control_status(self):
        """打印详细控制状态"""
        stats = self.get_control_stats()
        
        print(f"   总控制车辆: {stats['total_controlled_vehicles']}")
        print(f"   车队控制车辆: {stats['platoon_controlled_vehicles']}")
        
        if self.controlled_vehicles:
            print("   当前控制车辆详情:")
            for vehicle_id, control_type in self.controlled_vehicles.items():
                print(f"     - 车辆{vehicle_id}: {control_type}控制")
        else:
            print("   ⚠️ 当前无车辆被控制 - 可能车队控制未激活")
    
    # 未来扩展方法
    def _apply_auction_control(self):
        """应用拍卖系统控制（未来实现）"""
        pass
    
    def _apply_nash_control(self):
        """应用纳什均衡控制（未来实现）"""
        pass
    
    def emergency_reset_all_controls(self):
        """紧急重置所有控制（安全功能）"""
        print("🚨 紧急重置所有车辆控制")
        
        for vehicle_id in list(self.controlled_vehicles.keys()):
            try:
                carla_vehicle = self.world.get_actor(vehicle_id)
                if carla_vehicle and carla_vehicle.is_alive:
                    self._restore_default_behavior(carla_vehicle)
            except:
                continue
        
        # 清空所有控制记录
        self.controlled_vehicles.clear()
        self.platoon_controlled_vehicles.clear()
        
        print("✅ 所有车辆已恢复默认行为")