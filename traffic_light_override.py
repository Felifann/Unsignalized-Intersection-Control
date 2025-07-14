import carla
import argparse
import sys

def connect_to_carla(host='localhost', port=2000, timeout=10.0):
    """连接到CARLA服务器"""
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        tm = client.get_trafficmanager()
        print(f"✅ 已连接到CARLA服务器 {host}:{port}")
        return client, world, tm
    except Exception as e:
        print(f"❌ 连接CARLA失败: {e}")
        sys.exit(1)

def force_vehicles_run_lights(world, tm):
    """方法A: 强制所有车辆闯红灯"""
    vehicles = world.get_actors().filter('vehicle.*')
    affected_count = 0
    
    for vehicle in vehicles:
        if vehicle.is_alive:
            try:
                tm.set_percentage_running_light(vehicle, 100.0)
                affected_count += 1
            except Exception as e:
                print(f"[Warning] 设置车辆 {vehicle.id} 闯红灯失败: {e}")
    
    print(f"🚦 已设置 {affected_count} 辆车辆强制闯红灯")
    return affected_count

def freeze_lights_green(world):
    """方法B: 冻结所有信号灯为绿灯"""
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    affected_count = 0
    
    for tl in traffic_lights:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
            affected_count += 1
        except Exception as e:
            print(f"[Warning] 冻结信号灯 {tl.id} 失败: {e}")
    
    print(f"🟢 已冻结 {affected_count} 个信号灯为绿灯状态")
    return affected_count

def restore_normal_behavior(world, tm):
    """恢复正常交通行为"""
    # 恢复车辆正常行为
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        if vehicle.is_alive:
            try:
                tm.set_percentage_running_light(vehicle, 0.0)
            except:
                pass
    
    # 解冻信号灯
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    for tl in traffic_lights:
        try:
            tl.freeze(False)
        except:
            pass
    
    print("🔄 已恢复正常交通行为")

def main():
    parser = argparse.ArgumentParser(description='无信号交叉口上游交通优化工具')
    parser.add_argument('--method', choices=['runlight', 'greenthrough'], 
                       required=True, help='优化方法: runlight=强制闯红灯, greenthrough=冻结绿灯')
    parser.add_argument('--host', default='localhost', help='CARLA主机地址')
    parser.add_argument('--port', type=int, default=2000, help='CARLA端口')
    parser.add_argument('--restore', action='store_true', help='恢复正常交通行为')
    
    args = parser.parse_args()
    
    # 连接CARLA
    client, world, tm = connect_to_carla(args.host, args.port)
    
    if args.restore:
        restore_normal_behavior(world, tm)
        return
    
    print(f"🎯 目标: 最大化无信号交叉口连续交通流")
    print(f"📍 地图: {world.get_map().name}")
    
    # 执行选定的方法
    if args.method == 'runlight':
        print("🚨 方法A: 强制所有车辆闯红灯")
        affected = force_vehicles_run_lights(world, tm)
        
        # 监控新生成的车辆
        print("🔄 持续监控新车辆...")
        try:
            while True:
                world.tick()
                new_vehicles = [v for v in world.get_actors().filter('vehicle.*') 
                              if v.is_alive]
                for vehicle in new_vehicles:
                    tm.set_percentage_running_light(vehicle, 100.0)
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断，恢复正常行为")
            restore_normal_behavior(world, tm)
            
    elif args.method == 'greenthrough':
        print("🟢 方法B: 冻结所有信号灯为绿灯")
        affected = freeze_lights_green(world)
        print(f"✅ 优化完成，{affected} 个信号灯已永久设为绿灯")
        print("💡 使用 --restore 参数恢复正常行为")

if __name__ == '__main__':
    main()