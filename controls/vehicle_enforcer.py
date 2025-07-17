import carla
import time
from typing import Dict, Set
import math

class VehicleControlEnforcer:
    """
    车辆控制强制器 - 基于冲突解决结果强制车辆行为
    """
    
    def __init__(self, carla_wrapper, state_extractor):
        self.carla = carla_wrapper
        self.state_extractor = state_extractor
        self.world = carla_wrapper.world
        self.traffic_manager = carla_wrapper.client.get_trafficmanager()
        
        # 控制状态记录
        self.enforced_vehicles = {}  # {vehicle_id: enforcement_data}
        self.last_control_log = {}   # 防止重复日志
        self.control_log_interval = 3.0
        
        # 强制控制参数
        self.wait_params = {
            'speed_diff': 95.0,      # 强制减速95%
            'follow_distance': 8.0,   # 保持大距离
            'ignore_lights': 0.0,     # 严格遵守信号
            'ignore_signs': 0.0,
            'ignore_vehicles': 0.0,   # 严格避让其他车辆
            'target_speed': 0.0       # 目标速度为0
        }
        
        self.go_params = {
            'speed_diff': -70.0,      # 允许加速
            'follow_distance': 1.5,   # 正常跟车距离
            'ignore_lights': 90.0,    # 可以闯红灯通过路口
            'ignore_signs': 80.0,
            'ignore_vehicles': 40.0,  # 适度忽略其他车辆
        }
        
        print("🎮 车辆控制强制器初始化完成")

    def enforce_control_actions(self, control_actions):
        """
        根据冲突解决结果强制执行车辆控制
        control_actions: {agent_id: {'action': 'GO'|'WAIT', 'reason': str, ...}}
        """
        if not control_actions:
            return
        
        current_time = time.time()
        enforced_count = {'GO': 0, 'WAIT': 0}
        
        for agent_id, action_data in control_actions.items():
            action = action_data.get('action', 'WAIT')
            reason = action_data.get('reason', 'unknown')
            
            # 尝试强制执行控制
            success = self._enforce_single_agent(agent_id, action, reason, current_time)
            
            if success:
                enforced_count[action] += 1
                
                # 记录强制控制状态
                self.enforced_vehicles[agent_id] = {
                    'action': action,
                    'reason': reason,
                    'timestamp': current_time,
                    'group_id': action_data.get('group_id', 'unknown')
                }
        
        # 打印强制执行统计
        if enforced_count['GO'] > 0 or enforced_count['WAIT'] > 0:
            print(f"🎮 控制强制执行: 🟢{enforced_count['GO']}通行 | 🔴{enforced_count['WAIT']}等待")

    def _enforce_single_agent(self, agent_id, action, reason, current_time):
        """强制执行单个agent的控制"""
        try:
            # 对于车队，强制控制所有成员车辆
            if str(agent_id).startswith('platoon_'):
                return self._enforce_platoon_control(agent_id, action, reason, current_time)
            else:
                return self._enforce_vehicle_control(agent_id, action, reason, current_time)
                
        except Exception as e:
            print(f"[Warning] 强制控制 {agent_id} 失败: {e}")
            return False

    def _enforce_vehicle_control(self, vehicle_id, action, reason, current_time):
        """强制控制单个车辆"""
        carla_vehicle = self.world.get_actor(vehicle_id)
        if not carla_vehicle or not carla_vehicle.is_alive:
            return False
        
        try:
            if action == 'WAIT':
                # 强制停止
                self._apply_wait_control(carla_vehicle)
                self._log_control_action(vehicle_id, current_time, f"🔴强制停止: {reason}")
                
                # 可选：显示调试文本
                self._show_debug_text(carla_vehicle, "WAIT", carla.Color(255, 0, 0))
                
            elif action == 'GO':
                # 恢复通行
                self._apply_go_control(carla_vehicle)
                self._log_control_action(vehicle_id, current_time, f"🟢允许通行: {reason}")
                
                # 可选：显示调试文本
                self._show_debug_text(carla_vehicle, "GO", carla.Color(0, 255, 0))
            
            return True
            
        except Exception as e:
            print(f"[Warning] 应用控制到车辆 {vehicle_id} 失败: {e}")
            return False

    def _enforce_platoon_control(self, platoon_id, action, reason, current_time):
        """强制控制车队所有成员"""
        # 从车队ID中提取领导车辆ID
        try:
            leader_id = int(platoon_id.replace('platoon_', ''))
        except:
            return False
        
        # 获取车队成员（通过platoon_manager）
        if hasattr(self, 'platoon_manager') and self.platoon_manager:
            platoon_vehicles = self._get_platoon_vehicles(leader_id)
        else:
            # 如果无法获取车队信息，只控制领导车辆
            platoon_vehicles = [leader_id]
        
        success_count = 0
        for vehicle_id in platoon_vehicles:
            if self._enforce_vehicle_control(vehicle_id, action, reason, current_time):
                success_count += 1
        
        if success_count > 0:
            self._log_control_action(platoon_id, current_time, 
                                   f"🚛车队{action}: {success_count}辆车 - {reason}")
        
        return success_count > 0

    def _apply_wait_control(self, carla_vehicle):
        """应用强制等待控制参数"""
        params = self.wait_params
        
        # 应用Traffic Manager参数
        self.traffic_manager.vehicle_percentage_speed_difference(
            carla_vehicle, params['speed_diff']
        )
        self.traffic_manager.distance_to_leading_vehicle(
            carla_vehicle, params['follow_distance']
        )
        self.traffic_manager.ignore_lights_percentage(
            carla_vehicle, params['ignore_lights']
        )
        self.traffic_manager.ignore_signs_percentage(
            carla_vehicle, params['ignore_signs']
        )
        self.traffic_manager.ignore_vehicles_percentage(
            carla_vehicle, params['ignore_vehicles']
        )
        
        # 额外的强制停止措施：直接设置车辆控制
        try:
            # 可以添加更直接的控制，如设置刹车
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0  # 全力刹车
            control.steer = 0.0
            # carla_vehicle.apply_control(control)  # 谨慎使用，可能与TrafficManager冲突
        except:
            pass  # 如果直接控制失败，依赖TrafficManager参数

    def _apply_go_control(self, carla_vehicle):
        """应用允许通行控制参数"""
        params = self.go_params
        
        self.traffic_manager.vehicle_percentage_speed_difference(
            carla_vehicle, params['speed_diff']
        )
        self.traffic_manager.distance_to_leading_vehicle(
            carla_vehicle, params['follow_distance']
        )
        self.traffic_manager.ignore_lights_percentage(
            carla_vehicle, params['ignore_lights']
        )
        self.traffic_manager.ignore_signs_percentage(
            carla_vehicle, params['ignore_signs']
        )
        self.traffic_manager.ignore_vehicles_percentage(
            carla_vehicle, params['ignore_vehicles']
        )

    def _show_debug_text(self, carla_vehicle, status, color):
        """在车辆上方显示调试文本"""
        try:
            location = carla_vehicle.get_transform().location
            debug_location = carla.Location(
                location.x, location.y, location.z + 3.0
            )
            
            # 显示状态文本
            self.world.debug.draw_string(
                debug_location,
                status,
                draw_shadow=True,
                color=color,
                life_time=1.0,  # 短暂显示
                persistent_lines=False
            )
        except Exception as e:
            pass  # 调试文本失败不影响主要功能

    def _log_control_action(self, agent_id, current_time, message):
        """记录控制动作（避免重复日志）"""
        if agent_id not in self.last_control_log:
            self.last_control_log[agent_id] = 0
        
        if current_time - self.last_control_log[agent_id] >= self.control_log_interval:
            print(f"🎮 [控制强制] {message}")
            self.last_control_log[agent_id] = current_time

    def _get_platoon_vehicles(self, leader_id):
        """获取车队的所有车辆ID"""
        # 这需要与platoon_manager集成
        # 这里提供一个简化版本
        if hasattr(self, 'platoon_manager') and self.platoon_manager:
            for platoon in self.platoon_manager.get_all_platoons():
                if platoon.vehicles and platoon.vehicles[0]['id'] == leader_id:
                    return [v['id'] for v in platoon.vehicles]
        
        return [leader_id]  # 如果找不到车队，只返回领导车辆

    def set_platoon_manager(self, platoon_manager):
        """设置车队管理器引用"""
        self.platoon_manager = platoon_manager

    def cleanup_expired_controls(self, max_age=10.0):
        """清理过期的控制记录"""
        current_time = time.time()
        expired_vehicles = []
        
        for vehicle_id, control_data in self.enforced_vehicles.items():
            if current_time - control_data['timestamp'] > max_age:
                expired_vehicles.append(vehicle_id)
        
        for vehicle_id in expired_vehicles:
            del self.enforced_vehicles[vehicle_id]

    def get_enforcement_stats(self):
        """获取强制执行统计信息"""
        wait_count = sum(1 for data in self.enforced_vehicles.values() if data['action'] == 'WAIT')
        go_count = sum(1 for data in self.enforced_vehicles.values() if data['action'] == 'GO')
        
        return {
            'total_enforced': len(self.enforced_vehicles),
            'waiting_vehicles': wait_count,
            'go_vehicles': go_count
        }

    def emergency_release_all(self):
        """紧急释放所有控制（恢复默认行为）"""
        try:
            print("🚨 紧急释放所有强制控制...")
            
            for vehicle_id in list(self.enforced_vehicles.keys()):
                try:
                    if str(vehicle_id).startswith('platoon_'):
                        # 处理车队
                        platoon_vehicles = self._get_platoon_vehicles(
                            int(vehicle_id.replace('platoon_', ''))
                        )
                        for v_id in platoon_vehicles:
                            carla_vehicle = self.world.get_actor(v_id)
                            if carla_vehicle and carla_vehicle.is_alive:
                                self._restore_default_behavior(carla_vehicle)
                    else:
                        # 处理单个车辆
                        carla_vehicle = self.world.get_actor(vehicle_id)
                        if carla_vehicle and carla_vehicle.is_alive:
                            self._restore_default_behavior(carla_vehicle)
                except Exception as e:
                    print(f"[Warning] 释放控制失败 {vehicle_id}: {e}")
            
            # 清空控制记录
            self.enforced_vehicles.clear()
            self.last_control_log.clear()
            
            print("✅ 所有强制控制已释放")
            
        except Exception as e:
            print(f"[Error] 紧急释放失败: {e}")

    def _restore_default_behavior(self, carla_vehicle):
        """恢复车辆默认行为"""
        try:
            self.traffic_manager.vehicle_percentage_speed_difference(carla_vehicle, -40.0)
            self.traffic_manager.distance_to_leading_vehicle(carla_vehicle, 1.5)
            self.traffic_manager.ignore_lights_percentage(carla_vehicle, 0.0)
            self.traffic_manager.ignore_signs_percentage(carla_vehicle, 0.0)
            self.traffic_manager.ignore_vehicles_percentage(carla_vehicle, 0.0)
        except Exception as e:
            print(f"[Warning] 恢复默认行为失败: {e}")