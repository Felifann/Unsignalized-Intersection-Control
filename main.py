import sys
import os
import glob

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

# Import unified configuration
from config.unified_config import UnifiedConfig, get_config, print_config_summary

# ===== Environment related modules =====
from env.scenario_manager import ScenarioManager
from env.state_extractor import StateExtractor
# from env.simulation_config import SimulationConfig  # Removed: using unified config instead

# ===== Platoon management modules =====
from platooning.platoon_manager import PlatoonManager

# ===== Auction system modules =====
from auction.auction_engine import DecentralizedAuctionEngine

# ===== Traffic control modules =====
from control import TrafficController

# ===== Nash deadlock solver =====
from nash.deadlock_nash_solver import DeadlockNashSolver

# Initialize unified configuration
unified_config = get_config()
print_config_summary(unified_config)

# Initialize environment modules
scenario = ScenarioManager()
state_extractor = StateExtractor(scenario.carla)

# Initialize platoon management - pass state_extractor for navigation
platoon_manager = PlatoonManager(state_extractor)

# ===== INDEPENDENT DRL Configuration Management =====
# Fixed parameters for main.py - independent from unified config and DRL training
class DRLConfig:
    """Independent DRL parameter interface for main.py - separate from unified config"""
    
    # Fixed configuration parameters - all values are now fixed
    CONFLICT_TIME_WINDOW = 2.5  # Fixed: conflict detection time window
    MAX_GO_AGENTS = None        # Fixed: unlimited agents can go
    MIN_SAFE_DISTANCE = 3.0     # Fixed: minimum safe distance
    DEADLOCK_SPEED_THRESHOLD = 0.2  # Fixed: deadlock speed threshold
    AUCTION_INTERVAL = 4.0      # Fixed: auction cycle interval
    BIDDING_DURATION = 2.0      # Fixed: bidding phase duration
    LOGIC_UPDATE_INTERVAL = 1.0 # Fixed: decision update interval
    DEADLOCK_CHECK_INTERVAL = 8.0  # Fixed: deadlock check interval
    
    # FIXED DRL Action Space Parameters - Exact values for optimal performance
    URGENCY_POSITION_RATIO_EXACT = 1.89    # Fixed: urgency vs position advantage balance
    SPEED_DIFF_MODIFIER_EXACT = 8           # Fixed: speed control adjustment
    MAX_PARTICIPANTS_EXACT = 4              # Fixed: auction participants count
    IGNORE_VEHICLES_GO_EXACT = 49           # Fixed: GO state vehicle ignore percentage
    
    @classmethod
    def get_fixed_action_space_params(cls):
        """Get the fixed action space parameters for DRL training"""
        return {
            'urgency_position_ratio': cls.URGENCY_POSITION_RATIO_EXACT,
            'speed_diff_modifier': cls.SPEED_DIFF_MODIFIER_EXACT,
            'max_participants_per_auction': cls.MAX_PARTICIPANTS_EXACT,
            'ignore_vehicles_go': cls.IGNORE_VEHICLES_GO_EXACT
        }
    
    @classmethod
    def update_from_drl_params(cls, **kwargs):
        """Update system configuration - independent from DRL training"""
        # Update unified config for system parameters (not DRL-specific)
        unified_config.update_from_drl_params(**kwargs)
        
        # Update all system components with new config
        update_system_configuration()
        
        print(f"🤖 System configuration updated (independent from DRL training):")
        print(f"   Conflict window: {unified_config.conflict.conflict_time_window}s")
        print(f"   Max go agents: {'unlimited' if unified_config.mwis.max_go_agents is None else unified_config.mwis.max_go_agents}")
        
        # Print fixed action space parameters (independent from DRL training)
        fixed_params = cls.get_fixed_action_space_params()
        print(f"🎯 FIXED Action Space Parameters (independent from DRL):")
        print(f"   Urgency Position Ratio: {fixed_params['urgency_position_ratio']}")
        print(f"   Speed Diff Modifier: {fixed_params['speed_diff_modifier']}")
        print(f"   Max Participants: {fixed_params['max_participants_per_auction']}")
        print(f"   Ignore Vehicles GO: {fixed_params['ignore_vehicles_go']}%")

# Initialize decentralized auction engine - pass state_extractor
auction_engine = DecentralizedAuctionEngine(
    state_extractor=state_extractor, 
    max_go_agents=unified_config.mwis.max_go_agents
)

# Initialize Nash deadlock solver with unified config
nash_solver = DeadlockNashSolver(
    unified_config=unified_config,
    intersection_center=unified_config.system.intersection_center,
    max_go_agents=unified_config.mwis.max_go_agents
)

# Add dynamic configuration updates before main loop starts
def update_system_configuration():
    """Update all system components with current unified configuration"""
    # Update Nash solver with new config
    nash_solver.update_config_params(
        conflict_time_window=unified_config.conflict.conflict_time_window,
        max_go_agents=unified_config.mwis.max_go_agents,
        min_safe_distance=unified_config.conflict.min_safe_distance,
        deadlock_speed_threshold=unified_config.deadlock.deadlock_speed_threshold
    )
    
    # Apply fixed DRL action space parameters
    apply_fixed_drl_parameters()
    
    print(f"🔄 System configuration updated via UNIFIED CONFIG")

def apply_fixed_drl_parameters():
    """Apply fixed action space parameters to the system (independent from DRL training)"""
    fixed_params = DRLConfig.get_fixed_action_space_params()
    
    print(f"🎯 Applying FIXED parameters to system components (independent from DRL training):")
    print(f"   Urgency Position Ratio: {fixed_params['urgency_position_ratio']}")
    print(f"   Speed Diff Modifier: {fixed_params['speed_diff_modifier']}")
    print(f"   Max Participants: {fixed_params['max_participants_per_auction']}")
    print(f"   Ignore Vehicles GO: {fixed_params['ignore_vehicles_go']}%")
    
    # Update auction engine with fixed parameters
    if hasattr(auction_engine, 'update_max_participants_per_auction'):
        print(f"🔄 Updating auction engine max participants...")
        auction_engine.update_max_participants_per_auction(fixed_params['max_participants_per_auction'])
        print(f"✅ Auction engine updated: max_participants_per_auction = {fixed_params['max_participants_per_auction']}")
    else:
        print(f"⚠️ Auction engine does not have update_max_participants_per_auction method")
    
    # Update bid policy with fixed parameters (if accessible)
    if hasattr(auction_engine, 'bid_policy') and auction_engine.bid_policy:
        bid_policy = auction_engine.bid_policy
        if hasattr(bid_policy, 'update_parameters'):
            print(f"🔄 Updating bid policy parameters...")
            bid_policy.update_parameters(
                urgency_position_ratio=fixed_params['urgency_position_ratio'],
                speed_diff_modifier=fixed_params['speed_diff_modifier'],
                max_participants_per_auction=fixed_params['max_participants_per_auction'],
                ignore_vehicles_go=fixed_params['ignore_vehicles_go']
            )
        else:
            print(f"⚠️ Bid policy does not have update_parameters method")

    
    # Also update traffic controller's bid policy if it has one
    if hasattr(traffic_controller, 'bid_policy') and traffic_controller.bid_policy:
        if hasattr(traffic_controller.bid_policy, 'update_parameters'):
            print(f"🔄 Updating traffic controller bid policy parameters...")
            traffic_controller.bid_policy.update_parameters(
                urgency_position_ratio=fixed_params['urgency_position_ratio'],
                speed_diff_modifier=fixed_params['speed_diff_modifier'],
                max_participants_per_auction=fixed_params['max_participants_per_auction'],
                ignore_vehicles_go=fixed_params['ignore_vehicles_go']
            )
        else:
            print(f"⚠️ Traffic controller bid policy does not have update_parameters method")
    else:
        print(f"ℹ️ Traffic controller does not have bid_policy (this is normal)")
    
    print(f"✅ FIXED parameters applied to all system components (independent from DRL training)")


# Initialize traffic controller
traffic_controller = TrafficController(scenario.carla, state_extractor, max_go_agents=unified_config.mwis.max_go_agents)

# REACTIVATED: Set platoon manager reference
traffic_controller.set_platoon_manager(platoon_manager)

# Connect Nash solver to auction engine
auction_engine.set_nash_controller(nash_solver)

# Apply fixed DRL parameters during initialization
apply_fixed_drl_parameters()

# Display map information
spawn_points = scenario.carla.world.get_map().get_spawn_points()
print(f"=== Unsignalized Intersection Simulation (Integrated Auction System) ===")

# Generate traffic flow
scenario.reset_scenario()
scenario.start_time_counters()  # <-- start real/sim timers immediately after reset
scenario.show_intersection_area()      # Show larger general intersection area
scenario.show_intersection_area1()     # Show smaller core deadlock detection area

print("🔍 Deadlock detection area: Using small core area (blue border)")
print("🚦 General auction area: Using large detection area (green border)")

# Add before simulation starts
from traffic_light_override import force_vehicles_run_lights, freeze_lights_green

# Choose one method
# force_vehicles_run_lights(scenario.carla.world, scenario.carla.traffic_manager)
# or
# freeze_lights_green(scenario.carla.world)

# Main simulation loop
try:
    step = 0
    # Derive logic update interval (in steps) from unified seconds-based config
    try:
        logic_seconds = getattr(unified_config.system, 'logic_update_interval_seconds', 0.5)
        fixed_delta = max(1e-6, float(unified_config.system.fixed_delta_seconds))
        unified_update_interval = max(1, int(round(float(logic_seconds) / fixed_delta)))
    except Exception:
        unified_update_interval = 10
    unified_print_interval = 50  # Fixed: print interval every 50 steps
    
    while True:
        scenario.carla.world.tick()
        vehicle_states = state_extractor.get_vehicle_states()
        
        if step % unified_update_interval == 0:
            try:
                # Optional: Check for configuration updates every few cycles
                if step % (unified_update_interval * 10) == 0:  # Every 100 steps
                    update_system_configuration()
                
                # 1. Update platoon grouping
                platoon_manager.update()
                
                # 2. Update auction system
                auction_winners = auction_engine.update(vehicle_states, platoon_manager)

                # 3. Update traffic control - Pass winners directly
                traffic_controller.update_control(platoon_manager, auction_engine, auction_winners)
                
            except Exception as e:
                if "deadlock" in str(e).lower():
                    print(f"\n🚨 Deadlock detected: {e}")
                    print("🛑 Stopping simulation due to deadlock...")
                    break
                else:
                    print(f"⚠️  Error in simulation update: {e}")
                    # Continue simulation for other errors
        
        # Unified print frequency: all status information output simultaneously
        if step % unified_print_interval == 0:
            # Clear screen (optional, for clearer output)
            os.system('clear')  # Linux: use 'clear' to clear the terminal
            
            print(f"\n{'='*80}")
            print(f"[Step {step}] Unsignalized Intersection Simulation Status Report")
            print(f"{'='*80}")
            
            # Basic simulation information
            actual_fps = 1 / unified_config.system.fixed_delta_seconds
            vehicles_in_radius = vehicle_states
            vehicles_in_junction = [v for v in vehicle_states if v['is_junction']]
            
            print(f"📊 Basic Info: FPS:{actual_fps:.1f}, Total Vehicles:{len(vehicles_in_radius)}, In Intersection:{len(vehicles_in_junction)}")
            fixed_params = DRLConfig.get_fixed_action_space_params()
            print(f"🎮 System Config: NO GO LIMIT, CONFLICT_WINDOW={DRLConfig.CONFLICT_TIME_WINDOW}s")
            print(f"🎯 FIXED DRL Params: URGENCY={fixed_params['urgency_position_ratio']}, SPEED={fixed_params['speed_diff_modifier']}, MAX_PART={fixed_params['max_participants_per_auction']}, IGNORE={fixed_params['ignore_vehicles_go']}%")
            
            # 1. Platoon management status
            # platoon_manager.print_platoon_info()
            
            # ENHANCED: Show detailed platoon coordination status
            platoons = platoon_manager.get_all_platoons()
            if platoons:
                print(f"\n🔍 Platoon Coordination Status:")
                for platoon in platoons[:4]:  # Show top 3 platoons
                    leader_id = platoon.get_leader_id()
                    follower_ids = platoon.get_follower_ids()
                    
                    # Check if platoon vehicles are under control
                    controlled_count = 0
                    total_vehicles = platoon.get_size()
                    
                    control_stats = traffic_controller.get_control_stats()
                    controlled_vehicle_ids = set(control_stats.get('active_controls', []))
                    
                    platoon_vehicle_ids = platoon.get_vehicle_ids()
                    for vid in platoon_vehicle_ids:
                        if vid in controlled_vehicle_ids:
                            controlled_count += 1
                    
                    coordination_status = "🟢" if controlled_count == total_vehicles else "🟡" if controlled_count > 0 else "🔴"
                    
                    print(f"   {coordination_status} {platoon.platoon_id}: "
                          f"{controlled_count}/{total_vehicles} controlled "
                          f"(L:{leader_id}, F:{len(follower_ids)})")

            # 2. Auction system status - ENHANCED WITH CONFLICT INFO
            print(f"\n🎯 Auction System Status:")
            
            # Display current priority ranking (top 5)
            priority_order = auction_engine.get_current_priority_order()
            if priority_order:
                go_count = sum(1 for w in priority_order if w.conflict_action == 'go')
                wait_count = sum(1 for w in priority_order if w.conflict_action == 'wait')
                print(f"   📋 Current Decision: {go_count} GO, {wait_count} WAIT (no limit)")
                print(f"   🏆 Current Traffic Priority (Top 5):")
                for winner in priority_order[:5]:
                    participant = winner.participant
                    bid_value = winner.bid.value
                    rank = winner.rank
                    conflict_action = winner.conflict_action
                    action_emoji = "🟢" if conflict_action == 'go' else "🔴"
                    
                    # ENHANCED: Show both vehicle and platoon info
                    if participant.type == 'vehicle':
                        print(f"      #{rank}: {action_emoji}🚗Vehicle{participant.id} "
                              f"Bid:{bid_value:.1f}")
                    elif participant.type == 'platoon':
                        vehicle_count = len(participant.vehicles)
                        print(f"      #{rank}: {action_emoji}🚛Platoon{participant.id} "
                              f"({vehicle_count} vehicles) Bid:{bid_value:.1f}")
            
            # 3. Controller status - ENHANCED WITH EXIT TRACKING
            control_stats = traffic_controller.get_control_stats()
            if control_stats['total_controlled'] > 0:
                platoon_info = f"Platoon members:{control_stats['platoon_members']}, Leaders:{control_stats['platoon_leaders']}" if control_stats['platoon_members'] > 0 else ""
                print(f"🎮 Controller Status: Currently controlling:{control_stats['total_controlled']} | "
                      f"Waiting:{control_stats['waiting_vehicles']} | "
                      f"Going:{control_stats['go_vehicles']} | {platoon_info}")
                print(f"   📊 Statistics: Total controlled vehicles:{control_stats['total_vehicles_ever_controlled']} | "
                      f"Exited intersection:{control_stats['vehicles_exited_intersection']}")
            
            # 4. Auction system statistics - ENHANCED
            auction_stats = auction_engine.get_auction_stats()
            if auction_stats['current_agents'] > 0:
                print(f"🎯 Auction Statistics: Participants:{auction_stats['current_agents']} "
                      f"(Platoons:{auction_stats['platoon_agents']}, Single vehicles:{auction_stats['vehicle_agents']})")
                print(f"   Status: {auction_stats['auction_status']}, "
                      f"GO decisions: {auction_stats['current_go_count']} (no limit)")

        # Update vehicle ID label display (maintain original frequency)
        scenario.update_vehicle_labels()
        
        step += 1

except KeyboardInterrupt:
    print("\nSimulation manually terminated.")
except Exception as e:
    if "deadlock" in str(e).lower():
        print(f"\n🚨 Simulation terminated due to deadlock: {e}")
    else:
        print(f"\n❌ Simulation unexpectedly terminated: {e}")
finally:
    # Stop timers and print elapsed times before exiting
    try:
        scenario.stop_time_counters()
        real_elapsed = scenario.get_real_elapsed()
        sim_elapsed = scenario.get_sim_elapsed()
        print("\n⏱ Simulation Time Statistics:")
        print(f"   • Real time elapsed (wall-clock): {scenario.format_elapsed(real_elapsed)} ({real_elapsed:.2f}s)")
        print(f"   • Simulation world time    : {scenario.format_elapsed(sim_elapsed)} "
              f"({sim_elapsed:.2f}s)" if sim_elapsed is not None else "   • Simulation world time    : N/A")
    except Exception as e:
        print(f"⚠️ Unable to get time statistics: {e}")

    # Print traffic control statistics
    try:
        control_final_stats = traffic_controller.get_final_statistics()
        print("\n🎮 Traffic Control Statistics:")
        print(f"   • Total controlled vehicles: {control_final_stats['total_vehicles_controlled']}")
        print(f"   • Successfully exited intersection: {control_final_stats['vehicles_exited_intersection']}")
        print(f"   • Still under control: {control_final_stats['vehicles_still_controlled']}")
        print(f"   • Control history records: {control_final_stats['control_history_count']}")
        
        # New: Print enhanced acceleration statistics
        avg_pos_accel = control_final_stats['average_positive_acceleration']
        avg_neg_accel = control_final_stats['average_negative_acceleration']
        avg_abs_accel = control_final_stats['average_absolute_acceleration']
        
        # NEW: Print separate absolute averages for positive/negative accelerations
        avg_abs_pos_accel = control_final_stats.get('average_absolute_positive_acceleration', 0.0)
        avg_abs_neg_accel = control_final_stats.get('average_absolute_negative_acceleration', 0.0)
        
        pos_samples = control_final_stats['positive_acceleration_samples']
        neg_samples = control_final_stats['negative_acceleration_samples']
        abs_samples = control_final_stats['absolute_acceleration_samples']
        
        pos_vehicles = control_final_stats['positive_acceleration_vehicles']
        neg_vehicles = control_final_stats['negative_acceleration_vehicles']
        abs_vehicles = control_final_stats['absolute_acceleration_vehicles']
        
        print(f"   • Average positive acceleration: {avg_pos_accel:.3f} m/s² (absolute: {avg_abs_pos_accel:.3f} m/s²) ({pos_samples} samples, {pos_vehicles} vehicles)")
        print(f"   • Average negative acceleration: {avg_neg_accel:.3f} m/s² (absolute: {avg_abs_neg_accel:.3f} m/s²) ({neg_samples} samples, {neg_vehicles} vehicles)")
        print(f"   • Average absolute acceleration: {avg_abs_accel:.3f} m/s² ({abs_samples} samples, {abs_vehicles} vehicles)")
    
        # Print throughput per unit time
        throughput = control_final_stats['vehicles_exited_intersection'] / sim_elapsed * 3600 if sim_elapsed > 0 else 0
        print(f"   • Throughput per unit time: {throughput:.1f} vehicles/h")
        
    except Exception as e:
        print(f"⚠️ Unable to get control statistics: {e}")

    # Print collision report (only printed at simulation end)
    try:
        scenario.traffic_generator.print_collision_report()
    except Exception as e:
        print(f"⚠️ Unable to get collision statistics: {e}")

    print("\n🏁 Simulation ended")


