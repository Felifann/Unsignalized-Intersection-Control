import random
import carla
import time
from .simulation_config import SimulationConfig

class TrafficGenerator:
    def __init__(self, carla_wrapper, max_vehicles=None):
        self.carla = carla_wrapper
        self.max_vehicles = max_vehicles or SimulationConfig.MAX_VEHICLES
        self.vehicle_labels = {}
        self.collision_sensors = {}  # 新增：存储每辆车的碰撞传感器
        
        # Collision tracking
        self.collision_incidents = []  # List of collision incidents with details
        self.collision_count = 0  # Total collision count
        self.collision_status = {}  # Per-vehicle collision status

        # New: dedupe structures to avoid duplicate sensor reports
        self._recent_collision_keys = {}  # key -> timestamp (cleanup old keys periodically)
        self._dedupe_window = 0.5  # seconds: treat events within this window as same immediate incident
        self._per_vehicle_time_window = 1.0  # seconds: when summarizing, count at most 1 incident per car per this window

        # New clustering parameters to collapse repeated incidents over short time/space
        self._cluster_time_window = 2.0        # seconds: incidents within this are candidates for merging
        self._cluster_distance_threshold = 2.0 # meters: incidents within this distance are considered same event

    def _create_vehicle_label(self, vehicle):
        """为车辆创建ID标签"""
        try:
            # 获取车辆位置并在上方创建文字标签
            vehicle_location = vehicle.get_location()
            label_location = carla.Location(
                vehicle_location.x, 
                vehicle_location.y, 
                vehicle_location.z + 3.0  # 在车辆上方3米
            )
            
            # 创建文字标签 - 使用debug功能显示ID
            self.carla.world.debug.draw_string(
                label_location,
                str(vehicle.id),
                draw_shadow=False,
                color=carla.Color(255, 255, 255),  # 白色文字
                life_time=0.1,  # 短暂显示，需要持续更新
                persistent_lines=False
            )
            
            return True
        except:
            return False

    def generate_traffic(self):
        spawn_points = self.carla.world.get_map().get_spawn_points()
        num_vehicles = min(self.max_vehicles, len(spawn_points))
        random.shuffle(spawn_points)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        
        # 全局设置
        traffic_manager.global_percentage_speed_difference(-50.0)  # 全局提速50%
        traffic_manager.set_global_distance_to_leading_vehicle(1.5)  # 跟车距离1.5米

        self.vehicles = []
        for i in range(num_vehicles):
            transform = spawn_points[i]
            vehicle = self.carla.spawn_vehicle(transform=transform)
            if vehicle is not None:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                
                # 设置每辆车的 ignore_vehicles_percentage
                traffic_manager.ignore_vehicles_percentage(vehicle, 10.0)

                self.vehicles.append(vehicle)

                # 新增：为每辆车添加碰撞传感器
                collision_sensor = self.carla.world.spawn_actor(
                    self.carla.blueprint_library.find('sensor.other.collision'),
                    carla.Transform(),
                    attach_to=vehicle
                )
                self.collision_sensors[vehicle.id] = collision_sensor
                collision_sensor.listen(lambda event, vid=vehicle.id: self._on_collision(event, vid))

    def _on_collision(self, event, vehicle_id):
        """Enhanced collision event callback with detailed incident logging.
           Only record vehicle-vehicle collisions and debounce duplicate events."""
        try:
            current_time = time.time()

            # Identify other actor and ensure it's a vehicle
            other_actor = getattr(event, 'other_actor', None)
            if other_actor is None:
                return  # nothing to log

            other_type = getattr(other_actor, 'type_id', '')
            if 'vehicle' not in other_type:
                return  # only consider vehicle-vehicle collisions

            # Normalize ids
            try:
                a_id = int(vehicle_id)
            except:
                a_id = int(str(vehicle_id))
            try:
                b_id = int(other_actor.id)
            except:
                b_id = int(str(other_actor.id))

            # Build dedupe key (unordered pair + time-slot)
            pair = tuple(sorted((a_id, b_id)))
            time_slot = int(current_time / self._dedupe_window)
            key = (pair[0], pair[1], time_slot)

            # Cleanup old keys
            cutoff = current_time - 10.0
            keys_to_remove = [k for k, t in self._recent_collision_keys.items() if t < cutoff]
            for k in keys_to_remove:
                self._recent_collision_keys.pop(k, None)

            # If recently recorded the same pair within dedupe window, ignore
            if key in self._recent_collision_keys:
                return
            self._recent_collision_keys[key] = current_time

            # Extract collision details (defensive access)
            try:
                collision_location = event.transform.location
                loc_x, loc_y, loc_z = collision_location.x, collision_location.y, collision_location.z
            except:
                loc_x = loc_y = loc_z = 0.0

            try:
                collision_impulse = event.normal_impulse
                impulse_magnitude = (collision_impulse.x**2 + collision_impulse.y**2 + collision_impulse.z**2)**0.5
            except:
                impulse_magnitude = 0.0

            incident = {
                'incident_id': len(self.collision_incidents) + 1,
                'timestamp': current_time,
                'vehicles': [a_id, b_id],   # record both involved vehicles
                'vehicle_id': a_id,         # keep primary id for compatibility
                'other_actor_id': b_id,
                'collision_type': 'vehicle-vehicle',
                'location': {'x': loc_x, 'y': loc_y, 'z': loc_z},
                'impulse_magnitude': impulse_magnitude,
                'severity': self._classify_collision_severity(impulse_magnitude)
            }

            # Add to incident log
            self.collision_incidents.append(incident)
            self.collision_count += 1

            # Update per-vehicle status
            self.collision_status[a_id] = True
            self.collision_status[b_id] = True

            # NOTE: immediate printing suppressed (per your preference)
        except Exception:
            # be defensive: don't crash the simulation for logging errors
            return

    def _classify_collision_severity(self, impulse_magnitude):
        """Classify collision severity based on impulse magnitude"""
        if impulse_magnitude < 100:
            return "minor"
        elif impulse_magnitude < 500:
            return "moderate"
        elif impulse_magnitude < 1000:
            return "severe"
        else:
            return "critical"

    def get_collision_statistics(self):
        """Get comprehensive collision statistics with clustering/dedupe:
           - Only vehicle-vehicle collisions considered
           - Incidents are clustered per vehicle-pair if they are close in time and space
           - Each vehicle contributes at most 1 counted incident per _per_vehicle_time_window
        """
        # Filter to vehicle-vehicle and occurrences inside the detection area
        center = SimulationConfig.TARGET_INTERSECTION_CENTER
        half_size = SimulationConfig.INTERSECTION_HALF_SIZE

        def in_area(loc):
            dx = loc['x'] - center[0]
            dy = loc['y'] - center[1]
            return (abs(dx) <= half_size) and (abs(dy) <= half_size)

        # Only keep vehicle-vehicle incidents within detection square
        incidents = [
            inc for inc in self.collision_incidents
            if inc.get('collision_type') == 'vehicle-vehicle' and in_area(inc['location'])
        ]

        # Group incidents by unordered vehicle pair
        pairs = {}
        for inc in incidents:
            a, b = inc.get('vehicles', [inc.get('vehicle_id'), inc.get('other_actor_id')])
            pair = tuple(sorted((int(a), int(b))))
            pairs.setdefault(pair, []).append(inc)

        # For each pair cluster incidents by time + space
        representative_incidents = []
        for pair, inc_list in pairs.items():
            # sort by timestamp
            inc_list.sort(key=lambda x: x['timestamp'])
            cluster = []
            last_ts = None
            last_loc = None

            for inc in inc_list:
                ts = inc['timestamp']
                loc = (inc['location']['x'], inc['location']['y'])
                if not cluster:
                    cluster = [inc]
                    last_ts = ts
                    last_loc = loc
                    continue

                # If within cluster time window AND close spatially, merge into same cluster
                time_close = (ts - last_ts) <= self._cluster_time_window
                dist_sq = (loc[0] - last_loc[0])**2 + (loc[1] - last_loc[1])**2
                spatial_close = dist_sq <= (self._cluster_distance_threshold ** 2)

                if time_close and spatial_close:
                    cluster.append(inc)
                    # update last_ts and last_loc to newest to allow incremental merging
                    last_ts = ts
                    last_loc = loc
                else:
                    # finalize current cluster -> pick representative incident
                    rep = self._choose_representative_incident(cluster)
                    representative_incidents.append(rep)
                    # start new cluster
                    cluster = [inc]
                    last_ts = ts
                    last_loc = loc

            if cluster:
                rep = self._choose_representative_incident(cluster)
                representative_incidents.append(rep)

        # Now enforce at-most-one-incident-per-vehicle-per-time-window globally
        representative_incidents.sort(key=lambda x: x['timestamp'])
        counted_vehicle_times = {}  # vehicle_id -> last_counted_timestamp
        unique_incidents = []
        type_counts = {}
        severity_counts = {}
        unique_count = 0

        for inc in representative_incidents:
            ts = inc['timestamp']
            a, b = inc.get('vehicles', [inc.get('vehicle_id'), inc.get('other_actor_id')])
            a = int(a); b = int(b)

            # If either vehicle counted within per_vehicle_time_window, skip
            a_last = counted_vehicle_times.get(a, -1e9)
            b_last = counted_vehicle_times.get(b, -1e9)
            if (ts - a_last) < self._per_vehicle_time_window or (ts - b_last) < self._per_vehicle_time_window:
                continue

            # Accept this incident
            unique_count += 1
            counted_vehicle_times[a] = ts
            counted_vehicle_times[b] = ts

            ctype = inc.get('collision_type', 'vehicle-vehicle')
            sev = inc.get('severity', 'unknown')
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            unique_incidents.append(inc)

        return {
            'total_collisions': unique_count,
            'collision_types': type_counts,
            'severity_breakdown': severity_counts,
            'collision_rate': unique_count,
            'incidents': unique_incidents
        }

    def _choose_representative_incident(self, cluster):
        """Choose a representative incident from a cluster.
           Strategy: prefer highest severity, then highest impulse, then earliest timestamp.
        """
        if not cluster:
            return None
        # Map severity to rank
        sev_rank = {'critical': 4, 'severe': 3, 'moderate': 2, 'minor': 1}
        best = cluster[0]
        best_score = (sev_rank.get(best.get('severity', 'minor'), 1), best.get('impulse_magnitude', 0.0), -best.get('timestamp', 0))
        for inc in cluster[1:]:
            score = (sev_rank.get(inc.get('severity', 'minor'), 1), inc.get('impulse_magnitude', 0.0), -inc.get('timestamp', 0))
            if score > best_score:
                best = inc
                best_score = score
        return best

    def print_collision_report(self):
        """Print detailed collision report using deduplicated vehicle-vehicle incidents"""
        stats = self.get_collision_statistics()

        print(f"\n{'='*60}")
        print(f"🚨 COLLISION INCIDENT REPORT (vehicle-vehicle only, deduplicated)")
        print(f"{'='*60}")
        print(f"📊 Total Collisions (deduped): {stats['total_collisions']}")

        if stats['total_collisions'] > 0:
            print(f"\n🏷️  Collision Types:")
            for collision_type, count in stats['collision_types'].items():
                print(f"   • {collision_type}: {count}")

            print(f"\n⚠️  Severity Breakdown:")
            for severity, count in stats['severity_breakdown'].items():
                print(f"   • {severity}: {count}")

            print(f"\n📋 Recent Incidents (last 5 deduped):")
            recent_incidents = stats['incidents'][-5:]
            for incident in recent_incidents:
                vid = incident['vehicle_id']
                other = incident['other_actor_id']
                sev = incident['severity']
                x = incident['location']['x']
                y = incident['location']['y']
                print(f"   #{incident['incident_id']}: Vehicle {vid} vs {other} ({incident['collision_type']}) - {sev} at ({x:.1f}, {y:.1f})")
        else:
            print("✅ No vehicle-vehicle collisions detected in intersection area")

    def reset_collision_status(self, vehicle_id):
        """重置车辆碰撞状态（可在恢复后调用）"""
        if hasattr(self, 'collision_status'):
            self.collision_status[vehicle_id] = False

    def get_collision_status(self, vehicle_id):
        """获取车辆是否发生碰撞"""
        if hasattr(self, 'collision_status'):
            return self.collision_status.get(vehicle_id, False)
        return False

    def update_vehicle_labels(self):
        """更新所有车辆的ID标签位置"""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                self._create_vehicle_label(vehicle)