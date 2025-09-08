import carla
import argparse
import sys

def connect_to_carla(host='localhost', port=2000, timeout=10.0):
    """Connect to CARLA server"""
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        tm = client.get_trafficmanager()
        print(f"✅ Connected to CARLA server {host}:{port}")
        return client, world, tm
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        sys.exit(1)

def force_vehicles_run_lights(world, tm):
    """Method A: Force all vehicles to run red lights"""
    vehicles = world.get_actors().filter('vehicle.*')
    affected_count = 0
    
    for vehicle in vehicles:
        if vehicle.is_alive:
            try:
                tm.set_percentage_running_light(vehicle, 100.0)
                affected_count += 1
            except Exception as e:
                print(f"[Warning] Failed to set vehicle {vehicle.id} to run red lights: {e}")
    
    print(f"🚦 Set {affected_count} vehicles to force run red lights")
    return affected_count

def freeze_lights_green(world):
    """Method B: Freeze all traffic lights to green"""
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    affected_count = 0
    
    for tl in traffic_lights:
        try:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
            affected_count += 1
        except Exception as e:
            print(f"[Warning] Failed to freeze traffic light {tl.id}: {e}")
    
    print(f"🟢 Froze {affected_count} traffic lights to green state")
    return affected_count

def restore_normal_behavior(world, tm):
    """Restore normal traffic behavior"""
    # Restore normal vehicle behavior
    vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in vehicles:
        if vehicle.is_alive:
            try:
                tm.set_percentage_running_light(vehicle, 0.0)
            except:
                pass
    
    # Unfreeze traffic lights
    traffic_lights = world.get_actors().filter('traffic.traffic_light*')
    for tl in traffic_lights:
        try:
            tl.freeze(False)
        except:
            pass
    
    print("🔄 Restored normal traffic behavior")

def main():
    parser = argparse.ArgumentParser(description='Unsignalized intersection upstream traffic optimization tool')
    parser.add_argument('--method', choices=['runlight', 'greenthrough'], 
                       required=True, help='Optimization method: runlight=force run red lights, greenthrough=freeze green lights')
    parser.add_argument('--host', default='localhost', help='CARLA host address')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--restore', action='store_true', help='Restore normal traffic behavior')
    
    args = parser.parse_args()
    
    # Connect to CARLA
    client, world, tm = connect_to_carla(args.host, args.port)
    
    if args.restore:
        restore_normal_behavior(world, tm)
        return
    
    print(f"🎯 Target: Maximize continuous traffic flow at unsignalized intersection")
    print(f"📍 Map: {world.get_map().name}")
    
    # Execute selected method
    if args.method == 'runlight':
        print("🚨 Method A: Force all vehicles to run red lights")
        affected = force_vehicles_run_lights(world, tm)
        
        # Monitor newly spawned vehicles
        print("🔄 Continuously monitoring new vehicles...")
        try:
            while True:
                world.tick()
                new_vehicles = [v for v in world.get_actors().filter('vehicle.*') 
                              if v.is_alive]
                for vehicle in new_vehicles:
                    tm.set_percentage_running_light(vehicle, 100.0)
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted, restoring normal behavior")
            restore_normal_behavior(world, tm)
            
    elif args.method == 'greenthrough':
        print("🟢 Method B: Freeze all traffic lights to green")
        affected = freeze_lights_green(world)
        print(f"✅ Optimization completed, {affected} traffic lights permanently set to green")
        print("💡 Use --restore parameter to restore normal behavior")

if __name__ == '__main__':
    main()