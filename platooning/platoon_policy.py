import carla
import math

class PlatoonPolicy:
    def __init__(self, carla_wrapper):
        self.carla = carla_wrapper
        self.target_speed = 30.0  # km/h
        self.following_distance = 5.0  # 跟车距离(米)
        self.control_gains = {'kp': 0.5, 'ki': 0.1, 'kd': 0.2}  # PID参数
        self.vehicle_errors = {}  # 存储每辆车的误差历史

    def control_platoon(self, platoon):
        """
        控制一个编队内车辆的运动：
        - 第一个车按固定速度前进
        - 后车跟随前车，保持安全距离
        """
        if not platoon or len(platoon) < 2:
            return
        
        # Step 1: 控制队首车辆
        leader = platoon[0]
        self._control_leader(leader)
        
        # Step 2: 控制跟随车辆
        for i in range(1, len(platoon)):
            follower = platoon[i]
            leader_vehicle = platoon[i-1]
            self._control_follower(follower, leader_vehicle)

    def _control_leader(self, leader_state):
        """控制队首车辆保持目标速度"""
        try:
            vehicle_actor = self._get_vehicle_actor(leader_state['id'])
            if vehicle_actor is None:
                return
                
            # 获取当前速度
            velocity = vehicle_actor.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6  # 转换为km/h
            
            # 简单的速度控制
            if current_speed < self.target_speed * 0.9:
                throttle = 0.6
                brake = 0.0
            elif current_speed > self.target_speed * 1.1:
                throttle = 0.0
                brake = 0.3
            else:
                throttle = 0.3
                brake = 0.0
            
            # 应用控制
            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=0.0
            )
            vehicle_actor.apply_control(control)
            
        except Exception as e:
            print(f"Leader control error: {e}")

    def _control_follower(self, follower_state, leader_state):
        """控制跟随车辆保持与前车的距离"""
        try:
            follower_actor = self._get_vehicle_actor(follower_state['id'])
            if follower_actor is None:
                return
            
            # 计算与前车的距离
            follower_pos = follower_state['location']
            leader_pos = leader_state['location']
            distance = math.sqrt(
                (follower_pos[0] - leader_pos[0])**2 + 
                (follower_pos[1] - leader_pos[1])**2
            )
            
            # PID控制距离
            error = distance - self.following_distance
            vehicle_id = follower_state['id']
            
            # 初始化误差历史
            if vehicle_id not in self.vehicle_errors:
                self.vehicle_errors[vehicle_id] = {'prev_error': 0, 'integral': 0}
            
            # PID计算
            kp, ki, kd = self.control_gains['kp'], self.control_gains['ki'], self.control_gains['kd']
            integral = self.vehicle_errors[vehicle_id]['integral'] + error
            derivative = error - self.vehicle_errors[vehicle_id]['prev_error']
            
            control_output = kp * error + ki * integral + kd * derivative
            
            # 更新误差历史
            self.vehicle_errors[vehicle_id]['prev_error'] = error
            self.vehicle_errors[vehicle_id]['integral'] = integral
            
            # 转换为车辆控制命令
            if control_output > 0:  # 距离太远，加速
                throttle = min(0.8, abs(control_output) * 0.1)
                brake = 0.0
            else:  # 距离太近，减速
                throttle = 0.0
                brake = min(0.8, abs(control_output) * 0.1)
            
            # 应用控制
            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=0.0
            )
            follower_actor.apply_control(control)
            
        except Exception as e:
            print(f"Follower control error: {e}")

    def _get_vehicle_actor(self, vehicle_id):
        """根据车辆ID获取CARLA车辆对象"""
        try:
            actors = self.carla.world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.id == vehicle_id:
                    return actor
            return None
        except:
            return None

    def control_all_platoons(self, platoons):
        """控制所有车队"""
        for platoon in platoons:
            self.control_platoon(platoon)
