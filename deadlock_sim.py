#!/usr/bin/env python3
"""
Deadlock Nash Controller Test Script for CARLA

This script sets up a deterministic CARLA simulation that guarantees a 3-4 vehicle 
deadlock at an intersection, then uses the DeadlockNashController to resolve it.

Prerequisites:
1. CARLA server running (e.g., ./CarlaUE4.sh)
2. DeadlockNashController available at nash/deadlock_nash_solver.py
3. Python packages: carla, numpy, json

Usage:
    python deadlock_sim.py

The script will:
- Spawn vehicles at intersection approaches
- Force them into deadlock at intersection center
- Use DeadlockNashController to resolve the deadlock
- Log results to deadlock_nash_test_log.json
"""

import sys
import os
import time
import math
import json
import numpy as np
import random
import glob

# Add path for our controller
base_dir = os.path.dirname(os.path.abspath(__file__))
egg_path = []

if sys.platform.startswith('linux'):
    egg_path = glob.glob(os.path.join(base_dir, "carla_l", "carla-*linux-x86_64.egg"))

if egg_path:
    sys.path.insert(0, egg_path[0])
else:
    raise RuntimeError(
        "CARLA egg not found.\n"
    )

try:
    import carla
except ImportError:
    raise RuntimeError("CARLA Python API not found. Make sure CARLA is installed and accessible.")

from nash.deadlock_nash_solver import DeadlockNashController, SimpleAgent

# Configuration
TOWN = 'Town05'
N_PLAYERS = 3
SPAWN_DISTANCE = 30.0  # Reduced distance
SIM_STEP = 0.05  # Simulation timestep
DEADLOCK_WINDOW = 0.1  # More frequent checks
INITIAL_SPEED = 0.3  # m/s - Reduced from 1.0 for slower approach
INTERSECTION_CENTER = carla.Location(x=-188.9327573776245, y=-89.66813325881958, z=75.02772521972656)  # Town05 intersection
INTERSECTION_BBOX_SIZE = 6.0  # Smaller bounding box to ensure vehicles enter
MAX_SIM_TIME = 400.0  # Maximum simulation time
DEADLOCK_TIMEOUT = 300.0  # Time to wait for deadlock resolution
DEADLOCK_FORCE_TIME = 3.0  # Time to wait before forcing deadlock - Increased for slower approach

class DeadlockSimulation:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicles = []
        self.spawn_points = []
        self.intersection_center = INTERSECTION_CENTER
        self.deadlock_controller = None
        self.log_events = []
        self.deadlock_detected_time = None
        self.deadlock_resolved = False
        self.simulation_start_time = None
        self.vehicles_released = False
        self.last_deadlock_check = 0.0
        # Add tracking for current Nash actions
        self.current_nash_actions = {}
        self.go_vehicles_left_region = False
        self.autopilot_states = {}
        
    def setup_carla(self):
        """Initialize CARLA client and world"""
        print("Connecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Load town
        self.world = self.client.load_world(TOWN)
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SIM_STEP
        self.world.apply_settings(settings)
        
        print(f"Loaded {TOWN} in synchronous mode")
        
    def find_intersection_spawn_points(self):
        """Find suitable spawn points around an intersection"""
        # For Town03, use a known intersecti
        if TOWN == 'Town05':
            self.intersection_center = carla.Location(x=-188.9, y=-89.7, z=0.3)
            spawn_transforms = [
                # carla.Transform(carla.Location(x=-188.9, y=-89.7 - SPAWN_DISTANCE, z=0.3), 
                #                carla.Rotation(yaw=90.0)),   # South approach
                carla.Transform(carla.Location(x=-188.9 + SPAWN_DISTANCE, y=-89.7, z=0.3), 
                               carla.Rotation(yaw=180.0)),  # East approach
                carla.Transform(carla.Location(x=-188.9, y=-89.7 + SPAWN_DISTANCE, z=0.3), 
                               carla.Rotation(yaw=270.0)),  # North approach
                carla.Transform(carla.Location(x=-188.9 - SPAWN_DISTANCE, y=-89.7, z=0.3), 
                               carla.Rotation(yaw=0.0)),    # West approach
            ]
        else:
            # Fallback: use first 4 spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            self.intersection_center = spawn_points[0].location
            spawn_transforms = spawn_points[:3]
            
        self.spawn_points = spawn_transforms[:N_PLAYERS]
        print(f"Using intersection center: {self.intersection_center}")
        
    def spawn_vehicles(self):
        """Spawn vehicles at the chosen spawn points"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        print(f"Spawning {N_PLAYERS} vehicles...")
        for i, spawn_point in enumerate(self.spawn_points):
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                self.vehicles.append(vehicle)
                print(f"Spawned vehicle {vehicle.id} at {spawn_point.location}")
                
                # Set initial control to keep vehicle stationary
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                vehicle.apply_control(control)
                
                # Log spawn event
                self.log_events.append({
                    'time': 0.0,
                    'event': 'spawn',
                    'vehicle_id': vehicle.id,
                    'location': [spawn_point.location.x, spawn_point.location.y, spawn_point.location.z]
                })
            else:
                print(f"Failed to spawn vehicle at {spawn_point.location}")
                
        if len(self.vehicles) < 2:
            raise RuntimeError("Need at least 2 vehicles for deadlock test")
            
    def setup_deadlock_controller(self):
        """Initialize the DeadlockNashController"""
        # Define intersection bounding box
        bbox_min = [
            self.intersection_center.x - INTERSECTION_BBOX_SIZE,
            self.intersection_center.y - INTERSECTION_BBOX_SIZE
        ]
        bbox_max = [
            self.intersection_center.x + INTERSECTION_BBOX_SIZE,
            self.intersection_center.y + INTERSECTION_BBOX_SIZE
        ]
        
        self.deadlock_controller = DeadlockNashController(
            intersection_polygon=(bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1])
        )
        print(f"Initialized DeadlockNashController with bbox: {bbox_min} to {bbox_max}")
        
    def create_simple_agent(self, vehicle, wait_time: float) -> SimpleAgent:
        """Convert CARLA vehicle to SimpleAgent for controller"""
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Create intended path (straight line across intersection)
        start_loc = transform.location
        forward = transform.get_forward_vector()
        end_loc = carla.Location(
            x=start_loc.x + forward.x * 60.0,
            y=start_loc.y + forward.y * 60.0,
            z=start_loc.z
        )
        
        intended_path = [
            [start_loc.x, start_loc.y],
            [end_loc.x, end_loc.y]
        ]
        
        # Generate synthetic bid based on distance to intersection
        distance_to_intersection = start_loc.distance(self.intersection_center)
        bid = max(10.0, 100.0 - distance_to_intersection)
        
        return SimpleAgent(
            id=str(vehicle.id),
            position=(start_loc.x, start_loc.y),
            speed=speed,
            heading=math.radians(transform.rotation.yaw),
            intended_path=intended_path,
            bid=bid,
            wait_time=wait_time
        )
        
    def release_vehicles(self):
        """Release vehicles to drive towards intersection slowly"""
        print("🚀 Releasing vehicles towards intersection...")
        for vehicle in self.vehicles:
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.brake = 0.0
            vehicle.apply_control(control)
        
        self.vehicles_released = True
        self.log_events.append({
            'time': time.time() - self.simulation_start_time,
            'event': 'vehicles_released',
            'vehicle_count': len(self.vehicles)
        })
        
    def force_deadlock_in_intersection(self, current_time: float):
        """Force vehicles to stop in intersection with proper spacing to avoid crashes"""
        vehicles_in_intersection = []
        
        for vehicle in self.vehicles:
            location = vehicle.get_location()
            distance_to_center = location.distance(self.intersection_center)
            
            print(f"Vehicle {vehicle.id}: distance to center = {distance_to_center:.2f}m")
            
            if distance_to_center < INTERSECTION_BBOX_SIZE * 1.2:  # Slightly larger area
                vehicles_in_intersection.append(vehicle)
    
        # If vehicles are near/in intersection, force them to stop
        if len(vehicles_in_intersection) >= 1:  # Start with just 1 vehicle
            print(f"🔒 Forcing deadlock: {len(vehicles_in_intersection)} vehicles near intersection")
            for vehicle in vehicles_in_intersection:
                location = vehicle.get_location()
                distance_to_center = location.distance(self.intersection_center)
                
                control = carla.VehicleControl()
                
                if distance_to_center < INTERSECTION_BBOX_SIZE * 0.8:
                    # Inside intersection - full stop
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                    print(f"  - Vehicle {vehicle.id}: FULL STOP (in intersection)")
                else:
                    # Approaching - slow down significantly
                    control.throttle = 0.05  # Very slow
                    control.brake = 0.3
                    control.steer = 0.0
                    print(f"  - Vehicle {vehicle.id}: SLOW DOWN (approaching)")
                
                vehicle.apply_control(control)
        
            self.log_events.append({
                'time': current_time,
                'event': 'deadlock_forced',
                'vehicles_in_intersection': [v.id for v in vehicles_in_intersection]
            })
            
    def check_for_deadlock(self, current_time: float) -> bool:
        """Check for deadlock and attempt resolution"""
        # Create agents from current vehicle states
        agents = []
        vehicles_near_intersection = []
        
        for vehicle in self.vehicles:
            location = vehicle.get_location()
            distance_to_center = location.distance(self.intersection_center)
            
            print(f"Vehicle {vehicle.id}: distance to center = {distance_to_center:.2f}m, threshold = {INTERSECTION_BBOX_SIZE * 1.5:.2f}m")
            
            # Only consider vehicles near/in intersection
            if distance_to_center < INTERSECTION_BBOX_SIZE * 1.5:
                vehicles_near_intersection.append(vehicle)
                wait_time = max(0.1, current_time - DEADLOCK_FORCE_TIME)
                agent = self.create_simple_agent(vehicle, wait_time)
                agents.append(agent)
                print(f"  -> Added as agent: pos=({agent.position[0]:.2f}, {agent.position[1]:.2f}), speed={agent.speed:.2f}")
    
        print(f"Found {len(agents)} agents near intersection (need >= 2 for deadlock)")
    
        if len(agents) < 2:
            return False
        
        # Debug: Check if agents are actually in intersection
        agents_in_intersection = [a for a in agents if self.deadlock_controller._in_intersection(a.position)]
        print(f"Agents actually in intersection: {len(agents_in_intersection)}")
        for agent in agents_in_intersection:
            print(f"  - Agent {agent.id}: pos=({agent.position[0]:.2f}, {agent.position[1]:.2f})")
            
        # Check for deadlock using Nash controller
        actions = self.deadlock_controller.handle_deadlock(agents, current_time)
        
        if actions:
            if self.deadlock_detected_time is None:
                self.deadlock_detected_time = current_time
                print(f"🚨 Deadlock detected at time {current_time:.1f}s")
                
                self.log_events.append({
                    'time': current_time,
                    'event': 'deadlock_detected',
                    'agent_count': len(agents),
                    'actions': actions
                })
        
            # Apply actions to vehicles
            self.apply_nash_actions(actions, current_time)
            return True
        else:
            print("No deadlock detected by Nash controller")
            
        return False
        
    def apply_nash_actions(self, actions: dict, current_time: float):
        """Apply Nash controller actions to vehicles"""
        print(f"🎯 Applying Nash actions: {actions}")
        
        # Store current actions for continuous monitoring
        self.current_nash_actions = actions.copy()
        self.go_vehicles_left_region = False  # Reset flag
        
        for vehicle in self.vehicles:
            vehicle_id = str(vehicle.id)
            if vehicle_id in actions:
                action = actions[vehicle_id]
                if action == 'go':
                    vehicle.set_autopilot(True)
                    self.autopilot_states[vehicle_id] = True
                    print(f"   ✅ Vehicle {vehicle_id}: GO (autopilot enabled)")
                elif action == 'wait':
                    vehicle.set_autopilot(False)
                    self.autopilot_states[vehicle_id] = False
                    control = carla.VehicleControl()
                    control.throttle = 0.5
                    control.brake = 0.0
                    control.reverse = True
                    vehicle.apply_control(control)
                    print(f"   🔙 Vehicle {vehicle_id}: WAIT (reversing)")

        # Immediately check if any go vehicle is already outside (edge case)
        self.monitor_and_update_wait_vehicles(current_time)

        self.log_events.append({
            'time': current_time,
            'event': 'nash_actions_applied',
            'actions': actions
        })

    def monitor_and_update_wait_vehicles(self, current_time: float):
        """Monitor 'go' vehicles and switch 'wait' vehicles to 'go' when ANY 'go' vehicle leaves region"""
        if not self.current_nash_actions:
            return

        # Find ALL 'go' vehicles (both in and out of region)
        go_vehicle_ids = [vid for vid, act in self.current_nash_actions.items() if act == 'go']
        if not go_vehicle_ids:
            return

        # Find 'go' vehicles still in the region
        go_vehicles_in_region = []
        go_vehicles_out_of_region = []

        for vehicle in self.vehicles:
            vehicle_id = str(vehicle.id)
            if vehicle_id in go_vehicle_ids:
                location = vehicle.get_location()
                distance_to_center = location.distance(self.intersection_center)
                if distance_to_center < INTERSECTION_BBOX_SIZE * 1.5:  # Use same threshold as detection
                    go_vehicles_in_region.append(vehicle_id)
                else:
                    go_vehicles_out_of_region.append(vehicle_id)

        # Debug information
        print(f"DEBUG: Go vehicles - Total: {len(go_vehicle_ids)}, In region: {len(go_vehicles_in_region)}, Out of region: {len(go_vehicles_out_of_region)}")

        # If ANY 'go' vehicle has left the region, switch ALL 'wait' vehicles to 'go'
        any_go_vehicle_left = len(go_vehicles_out_of_region) > 0
        
        if any_go_vehicle_left and not self.go_vehicles_left_region:
            print(f"🚦 Triggering wait->go switch: {go_vehicles_out_of_region} vehicles left region")
            
            # Find ALL current 'wait' vehicles and switch them
            wait_vehicles_switched = []
            for vehicle in self.vehicles:
                vehicle_id = str(vehicle.id)
                if vehicle_id in self.current_nash_actions and self.current_nash_actions[vehicle_id] == 'wait':
                    # Update action to 'go' in our tracking
                    self.current_nash_actions[vehicle_id] = 'go'
                    
                    # Apply 'go' action immediately - FORCE autopilot
                    try:
                        vehicle.set_autopilot(True)
                        self.autopilot_states[vehicle_id] = True
                        wait_vehicles_switched.append(vehicle_id)
                        print(f"   🟢 Vehicle {vehicle_id}: SWITCHED to GO (autopilot enabled)")
                    except Exception as e:
                        print(f"   ❌ Vehicle {vehicle_id}: FAILED to switch - {e}")

            # Additional safety check - force autopilot on all non-go vehicles
            for vehicle in self.vehicles:
                vehicle_id = str(vehicle.id)
                if vehicle_id not in go_vehicle_ids and vehicle_id not in go_vehicles_out_of_region:
                    # This should be a wait vehicle - ensure it has autopilot
                    if not self.autopilot_states.get(vehicle_id, False):
                        try:
                            vehicle.set_autopilot(True)
                            self.autopilot_states[vehicle_id] = True
                            print(f"   🔧 Vehicle {vehicle_id}: FORCE autopilot (safety check)")
                        except Exception as e:
                            print(f"   ❌ Vehicle {vehicle_id}: FAILED force autopilot - {e}")

            if wait_vehicles_switched:
                self.go_vehicles_left_region = True
                self.log_events.append({
                    'time': current_time,
                    'event': 'wait_vehicles_switched_to_go',
                    'reason': 'first_go_vehicle_left_region',
                    'switched_vehicles': wait_vehicles_switched,
                    'go_vehicles_out_of_region': go_vehicles_out_of_region,
                    'go_vehicles_remaining_in_region': go_vehicles_in_region
                })
                print(f"🚦 ALL wait vehicles switched to GO - {len(wait_vehicles_switched)} switched, {len(go_vehicles_in_region)} go vehicles still in region")

    def check_deadlock_resolution(self, current_time: float) -> bool:
        """Check if deadlock has been resolved (vehicles clearing intersection)"""
        vehicles_cleared = 0
        
        for vehicle in self.vehicles:
            location = vehicle.get_location()
            distance_to_center = location.distance(self.intersection_center)
            
            if distance_to_center > INTERSECTION_BBOX_SIZE * 1.2:
                vehicles_cleared += 1
        
        # Consider resolved if at least one vehicle has cleared
        if vehicles_cleared > 0 and not self.deadlock_resolved:
            self.deadlock_resolved = True
            resolution_time = current_time - (self.deadlock_detected_time or current_time)
            
            print(f"✅ Deadlock resolved! {vehicles_cleared} vehicles cleared intersection")
            print(f"   Resolution time: {resolution_time:.1f}s")
            
            self.log_events.append({
                'time': current_time,
                'event': 'deadlock_resolved',
                'vehicles_cleared': vehicles_cleared,
                'resolution_time': resolution_time
            })
            
            return True
            
        return False
        
    def cleanup(self):
        """Clean up spawned vehicles and restore settings"""
        print("🧹 Cleaning up simulation...")
        
        # Destroy vehicles
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        # Restore synchronous mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("Cleanup complete")
        
    def save_log(self):
        """Save simulation log to JSON file"""
        log_data = {
            'config': {
                'town': TOWN,
                'n_players': N_PLAYERS,
                'spawn_distance': SPAWN_DISTANCE,
                'intersection_center': [self.intersection_center.x, self.intersection_center.y, self.intersection_center.z]
            },
            'results': {
                'deadlock_detected': self.deadlock_detected_time is not None,
                'deadlock_detected_time': self.deadlock_detected_time,
                'deadlock_resolved': self.deadlock_resolved,
                'total_simulation_time': time.time() - self.simulation_start_time if self.simulation_start_time else 0
            },
            'events': self.log_events
        }

        log_folder = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, 'deadlock_nash_test_log.json')

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"📄 Log saved to {log_path}")
        
    def set_spectator_view(self):
        """Set the spectator camera to top-down view over intersection"""
        spectator = self.world.get_spectator()
        # Top-down view: z=75, pitch=-90, yaw=0
        location = carla.Location(
            x=self.intersection_center.x,
            y=self.intersection_center.y,
            z=75.0
        )
        rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        spectator.set_transform(carla.Transform(location, rotation))
        print(f"Spectator camera set to intersection top-down view at {location}")

    def run_simulation(self):
        """Main simulation loop"""
        print("🎬 Starting deadlock simulation...")
        self.simulation_start_time = time.time()
        
        try:
            # Setup phase
            self.setup_carla()
            self.find_intersection_spawn_points()
            self.spawn_vehicles()
            self.set_spectator_view()
            self.setup_deadlock_controller()
            
            # Initial world tick
            self.world.tick()
            
            # Simulation loop
            sim_time = 0.0
            deadlock_check_timer = 0.0
            
            while sim_time < MAX_SIM_TIME:
                self.world.tick()
                sim_time += SIM_STEP
                deadlock_check_timer += SIM_STEP
                
                # Release vehicles after initial delay
                if not self.vehicles_released and sim_time >= 2.0:
                    self.release_vehicles()
                
                # Force deadlock after vehicles have had time to reach intersection
                if self.vehicles_released and sim_time >= DEADLOCK_FORCE_TIME and sim_time < DEADLOCK_FORCE_TIME + 2.0:
                    self.force_deadlock_in_intersection(sim_time)
                
                # Monitor and update wait vehicles continuously
                self.monitor_and_update_wait_vehicles(sim_time)
                
                # Check for deadlock periodically
                if deadlock_check_timer >= DEADLOCK_WINDOW:
                    deadlock_check_timer = 0.0
                    
                    self.check_for_deadlock(sim_time)
                    self.check_deadlock_resolution(sim_time)
                
                # Timeout check for deadlock resolution
                if (self.deadlock_detected_time is not None and 
                    sim_time - self.deadlock_detected_time > DEADLOCK_TIMEOUT):
                    print("⏰ Deadlock resolution timeout")
                    break
            
            # Final assessment
            self.assess_results()
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            raise
        finally:
            self.cleanup()
            self.save_log()
            
    def assess_results(self):
        """Assess simulation results and print summary"""
        print("\n" + "="*60)
        print("📊 DEADLOCK NASH CONTROLLER TEST RESULTS")
        print("="*60)
        
        success = True
        
        # Test 1: Deadlock detection
        if self.deadlock_detected_time is not None:
            print(f"✅ Test 1 PASSED: Deadlock detected at {self.deadlock_detected_time:.1f}s")
        else:
            print("❌ Test 1 FAILED: No deadlock detected")
            success = False
        
        # Test 2: Nash controller response
        nash_events = [e for e in self.log_events if e['event'] == 'nash_actions_applied']
        if nash_events:
            actions = nash_events[0]['actions']
            go_count = sum(1 for action in actions.values() if action == 'go')
            print(f"✅ Test 2 PASSED: Nash controller returned actions ({go_count} go, {len(actions)-go_count} wait)")
        else:
            print("❌ Test 2 FAILED: Nash controller did not return actions")
            success = False
        
        # Test 3: Deadlock resolution
        if self.deadlock_resolved:
            resolution_events = [e for e in self.log_events if e['event'] == 'deadlock_resolved']
            if resolution_events:
                resolution_time = resolution_events[0]['resolution_time']
                print(f"✅ Test 3 PASSED: Deadlock resolved in {resolution_time:.1f}s")
            else:
                print("✅ Test 3 PASSED: Deadlock resolved")
        else:
            print("❌ Test 3 FAILED: Deadlock not resolved")
            success = False
        
        # Overall result
        if success:
            print("\n🎉 ALL TESTS PASSED - Nash controller successfully resolved deadlock!")
        else:
            print("\n❌ SOME TESTS FAILED - Check implementation")
        
        print("="*60)

def main():
    """Main entry point"""
    print("🚗 CARLA Deadlock Nash Controller Test")
    print("Make sure CARLA server is running!")
    
    # Set random seed for deterministic behavior
    random.seed(42)
    np.random.seed(42)
    
    # Run simulation
    sim = DeadlockSimulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()