import time
import math

from env.simulation_config import SimulationConfig
from .bid_policy import AgentBidPolicy
from nash.conflict_resolver import ConflictResolver

class DecentralizedAuctionEngine:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0), communication_range=50.0, state_extractor=None):
        self.intersection_center = intersection_center
        self.communication_range = communication_range
        self.state_extractor = state_extractor  # 添加state_extractor参数
        
        # 分布式拍卖状态
        self.active_auctions = {}  # {auction_id: auction_data}
        self.agent_bids = {}       # {agent_id: bid_data}
        self.auction_results = {}  # {auction_id: winner_list}
        
        # 通信模拟
        self.message_queue = []    # 模拟车车通信消息队列
        self.last_auction_time = 0
        self.auction_interval = 2.0  # 每2秒举行一次拍卖
        
        # 路口区域定义
        self.intersection_radius = 15.0  # 路口区域半径
        
        # 纳什均衡冲突解决器
        self.conflict_resolver = ConflictResolver(intersection_center)
        
        # 新增：车辆控制强制器
        self.vehicle_enforcer = None  # 将在主程序中设置
        
        print("🎯 分布式拍卖引擎初始化完成 - 集成Nash均衡冲突解决和控制强制")

    def set_vehicle_enforcer(self, vehicle_enforcer):
        """设置车辆控制强制器"""
        self.vehicle_enforcer = vehicle_enforcer

    def update(self, vehicle_states, platoon_manager):
        """
        主更新函数：管理分布式拍卖过程并强制执行控制
        """
        current_time = time.time()
        
        # 1. 识别路口处的agents
        junction_agents = self._identify_junction_agents(vehicle_states, platoon_manager)
        
        # 2. 清理旧的agent状态
        current_agent_ids = [agent['id'] for agent in junction_agents]
        self.conflict_resolver.cleanup_old_agents(current_agent_ids)
        
        # 3. 定期启动新拍卖
        if current_time - self.last_auction_time >= self.auction_interval:
            if junction_agents:
                auction_id = self._start_new_auction(junction_agents, current_time)
                self.last_auction_time = current_time
        
        # 4. 处理正在进行的拍卖
        self._process_active_auctions(current_time)
        
        # 5. 获取当前优先级排序
        priority_order = self._get_current_priority_order()
        
        # 6. 应用纳什均衡冲突解决（扩展到所有agents）
        control_actions = {}
        if priority_order:
            # 获取所有获胜者agents（不限于前3名）
            all_winner_agents = [winner['agent'] for winner in priority_order]
            
            # 检查并解决冲突
            control_actions = self.conflict_resolver.check_and_resolve(all_winner_agents)
            
            # 更新优先级排序
            priority_order = self._apply_conflict_resolution(priority_order, control_actions)
        
        # 7. 🔥 新增：强制执行控制动作
        if self.vehicle_enforcer and control_actions:
            self.vehicle_enforcer.enforce_control_actions(control_actions)
        
        # 8. 模拟车车通信
        self._simulate_v2v_communication()
        
        return priority_order

    def _apply_conflict_resolution(self, priority_order, conflict_resolution):
        """应用冲突解决结果到优先级排序"""
        if not conflict_resolution:
            return priority_order
        
        # 创建新的优先级列表
        resolved_priority = []
        waiting_agents = []
        
        for winner in priority_order:
            agent_id = winner['agent']['id']
            action = conflict_resolution.get(agent_id, 'go')
            
            if action == 'go':
                # 保持原排名
                resolved_priority.append(winner)
            else:
                # 移到队列末尾
                winner_copy = winner.copy()
                winner_copy['conflict_action'] = 'wait'
                waiting_agents.append(winner_copy)
        
        # 等待的agents排在后面
        resolved_priority.extend(waiting_agents)
        
        # 重新分配排名
        for i, winner in enumerate(resolved_priority):
            winner['rank'] = i + 1
        
        if waiting_agents:
            print(f"🎮 冲突解决：{len(waiting_agents)}个agents被要求等待")
        
        return resolved_priority

    def _identify_junction_agents(self, vehicle_states, platoon_manager):
        """
        识别路口处的agents：
        只要车队队长在路口区域就将该车队加入agents
        """
        agents = []
        
        # 获取路口区域内及接近路口的车辆
        junction_vehicles = self._get_junction_area_vehicles(vehicle_states)
        
        if not junction_vehicles:
            return agents
        
        print(f"🏢 路口区域发现 {len(junction_vehicles)} 辆车")
        
        # 1. 添加路口处的platoons作为agents
        platoon_vehicle_ids = set()
        
        for platoon in platoon_manager.get_all_platoons():
            leader = platoon.get_leader()
            if leader and leader.get('is_junction', False):
                # 只要队长在路口区域就加入
                platoon_agent = {
                    'type': 'platoon',
                    'id': f"platoon_{leader['id']}",
                    'vehicles': platoon.vehicles,
                    'goal_direction': platoon.get_goal_direction(),
                    'leader_location': leader['location'],
                    'location': leader['location'],
                    'size': platoon.get_size(),
                    'at_junction': any(v['is_junction'] for v in platoon.vehicles)
                }
                agents.append(platoon_agent)
                # 记录platoon中的所有车辆ID
                for vehicle in platoon.vehicles:
                    platoon_vehicle_ids.add(vehicle['id'])
        
        # 2. 添加路口处的单个车辆作为agents
        for vehicle in junction_vehicles:
            if vehicle['id'] not in platoon_vehicle_ids:
                if self._vehicle_has_destination(vehicle):
                    vehicle_agent = {
                        'type': 'vehicle',
                        'id': vehicle['id'],
                        'data': vehicle,
                        'location': vehicle['location'],
                        'goal_direction': self._infer_vehicle_direction(vehicle),
                        'at_junction': vehicle['is_junction']
                    }
                    agents.append(vehicle_agent)
        
        return agents

    def _infer_vehicle_direction(self, vehicle):
        """使用导航系统获取车辆行驶方向"""
        if not vehicle.get('destination'):
            print(f"[Warning] 车辆 {vehicle['id']} 没有目的地，使用默认方向")
            return 'straight'  # 返回默认方向而不是None

        if not self.state_extractor:
            print(f"[Warning] StateExtractor未初始化，车辆 {vehicle['id']} 使用默认方向")
            return 'straight'  # 返回默认方向而不是None

        vehicle_location = vehicle['location']
        try:
            # 转换为carla.Location对象
            import carla
            carla_location = carla.Location(
                x=vehicle_location[0],
                y=vehicle_location[1], 
                z=vehicle_location[2]
            )
            direction = self.state_extractor.get_route_direction(carla_location, vehicle['destination'])
            return direction if direction else 'straight'  # 确保总是返回有效方向
        except Exception as e:
            print(f"[Warning] 车辆 {vehicle['id']} 导航方向获取失败：{e}，使用默认方向")
            return 'straight'  # 返回默认方向

    def _get_junction_area_vehicles(self, vehicle_states):
        """只获取已在路口内的车辆"""
        junction_vehicles = []
        for vehicle in vehicle_states:
            if vehicle['is_junction']:
                junction_vehicles.append(vehicle)
        return junction_vehicles

    # def _is_heading_to_intersection(self, vehicle):
    #     """判断车辆是否朝向路口行驶"""
    #     # 简化版本：基于车辆有目的地且在检测区域内
    #     return (vehicle.get('destination') is not None and 
    #             SimulationConfig.is_in_intersection_area(vehicle['location']))

    def _is_at_junction_area(self, vehicle):
        """判断车辆是否在路口区域（使用正方形检测）"""
        return (vehicle['is_junction'] or 
                SimulationConfig.is_in_intersection_area(vehicle['location']))

    def _vehicle_has_destination(self, vehicle):
        """检查车辆是否有明确的目的地"""
        return vehicle.get('destination') is not None

    def _start_new_auction(self, agents, start_time):
        """启动新的分布式拍卖"""
        auction_id = f"junction_auction_{int(start_time)}"
        
        auction_data = {
            'id': auction_id,
            'start_time': start_time,
            'participants': agents,
            'bids': {},
            'status': 'bidding',
            'deadline': start_time + 1.0,  # 1秒竞价时间
            'winner_list': []
        }
        
        self.active_auctions[auction_id] = auction_data
        
        # 广播拍卖开始消息
        self._broadcast_auction_start(auction_id, agents)
        
        return auction_id

    def _process_active_auctions(self, current_time):
        """处理正在进行的拍卖"""
        completed_auctions = []
        
        for auction_id, auction_data in self.active_auctions.items():
            if auction_data['status'] == 'bidding':
                # 收集竞价
                self._collect_bids_for_auction(auction_id, auction_data)
                
                # 检查是否到达截止时间
                if current_time >= auction_data['deadline']:
                    auction_data['status'] = 'evaluating'
                    
            elif auction_data['status'] == 'evaluating':
                # 评估竞价并确定获胜者
                winners = self._evaluate_auction(auction_id, auction_data)
                auction_data['winner_list'] = winners
                auction_data['status'] = 'completed'
                
                # 广播拍卖结果
                self._broadcast_auction_results(auction_id, winners)
                
                # 打印详细的获胜者信息
                self._print_auction_winners(auction_id, winners)
                
            elif auction_data['status'] == 'completed':
                # 标记为可删除
                completed_auctions.append(auction_id)
        
        # 清理已完成的拍卖
        for auction_id in completed_auctions:
            self.auction_results[auction_id] = self.active_auctions[auction_id]['winner_list']
            del self.active_auctions[auction_id]

    def _print_auction_winners(self, auction_id, winners):
        """打印拍卖获胜者详细信息"""
        if not winners:
            return
        
        print(f"🏆 路口竞价 {auction_id} 完成，通行优先级:")
        for i, winner in enumerate(winners[:5]):  # 只显示前5名
            agent = winner['agent']
            bid_value = winner['bid_value']
            rank = winner['rank']
            at_junction = agent.get('at_junction', False)
            status_emoji = "🏢" if at_junction else "🚦"
            
            if agent['type'] == 'platoon':
                print(f"   #{rank}: {status_emoji}🚛 车队{agent['id']} "
                      f"({agent['size']}车-{agent['goal_direction']}) "
                      f"出价:{bid_value:.1f}")
            else:
                print(f"   #{rank}: {status_emoji}🚗 单车{agent['id']} "
                      f"出价:{bid_value:.1f}")

    def _collect_bids_for_auction(self, auction_id, auction_data):
        """为特定拍卖收集竞价"""
        for agent in auction_data['participants']:
            agent_id = agent['id']
            
            # 检查是否已经出价
            if agent_id not in auction_data['bids']:
                # 创建竞价策略并计算出价，传入state_extractor
                bid_policy = AgentBidPolicy(agent, self.intersection_center, self.state_extractor)
                bid_value = bid_policy.compute_bid()
                
                auction_data['bids'][agent_id] = {
                    'agent': agent,
                    'bid_value': bid_value,
                    'timestamp': time.time()
                }

    def _evaluate_auction(self, auction_id, auction_data):
        """评估拍卖并确定获胜者优先级"""
        bids = auction_data['bids']
        
        if not bids:
            return []
        
        # 按出价从高到低排序
        sorted_bidders = sorted(
            bids.items(),
            key=lambda x: x[1]['bid_value'],
            reverse=True
        )
        
        # 构建获胜者列表
        winners = []
        for bidder_id, bid_data in sorted_bidders:
            winner_entry = {
                'id': bidder_id,
                'agent': bid_data['agent'],
                'bid_value': bid_data['bid_value'],
                'rank': len(winners) + 1
            }
            winners.append(winner_entry)
        
        return winners

    def _broadcast_auction_start(self, auction_id, agents):
        """广播拍卖开始消息"""
        message = {
            'type': 'auction_start',
            'auction_id': auction_id,
            'timestamp': time.time(),
            'participants': [a['id'] for a in agents]
        }
        self.message_queue.append(message)

    def _broadcast_auction_results(self, auction_id, winners):
        """广播拍卖结果"""
        message = {
            'type': 'auction_results',
            'auction_id': auction_id,
            'timestamp': time.time(),
            'winners': [(w['id'], w['bid_value'], w['rank']) for w in winners[:5]]
        }
        self.message_queue.append(message)

    def _simulate_v2v_communication(self):
        """模拟车车通信"""
        # 简化版本：直接处理消息队列
        processed_messages = []
        
        for message in self.message_queue:
            # 模拟通信延迟和丢包
            if time.time() - message['timestamp'] < 0.5:  # 0.5秒内有效
                processed_messages.append(message)
        
        # 清理过期消息
        self.message_queue = processed_messages

    def _get_current_priority_order(self):
        """获取当前优先级排序"""
        if not self.auction_results:
            return []
        
        # 合并所有拍卖结果，按照最近的拍卖为准
        latest_auction = max(self.auction_results.keys())
        return self.auction_results[latest_auction]

    def _distance_to_intersection(self, vehicle_or_location):
        """计算到交叉口的距离"""
        if isinstance(vehicle_or_location, dict):
            if 'location' in vehicle_or_location:
                location = vehicle_or_location['location']
            elif 'leader_location' in vehicle_or_location:
                location = vehicle_or_location['leader_location']
            else:
                return float('inf')
        else:
            location = vehicle_or_location
        
        dx = location[0] - self.intersection_center[0]
        dy = location[1] - self.intersection_center[1]
        return math.sqrt(dx*dx + dy*dy)

    def get_auction_stats(self):
        """获取拍卖统计信息"""
        active_count = len(self.active_auctions)
        total_participants = sum(len(auction['participants']) for auction in self.active_auctions.values())
        
        # 统计参与者类型和位置
        platoon_count = 0
        vehicle_count = 0
        in_junction_count = 0
        approaching_count = 0
        
        for auction in self.active_auctions.values():
            for participant in auction['participants']:
                if participant['type'] == 'platoon':
                    platoon_count += 1
                else:
                    vehicle_count += 1
                
                if participant.get('at_junction', False):
                    in_junction_count += 1
                else:
                    approaching_count += 1
        
        return {
            'active_auctions': active_count,
            'total_participants': total_participants,
            'platoon_participants': platoon_count,
            'vehicle_participants': vehicle_count,
            'in_junction_participants': in_junction_count,
            'approaching_participants': approaching_count,
            'completed_auctions': len(self.auction_results),
            'message_queue_size': len(self.message_queue)
        }

    def print_auction_status(self):
        """打印拍卖状态（包含冲突信息）"""
        stats = self.get_auction_stats()
        conflict_stats = self.conflict_resolver.get_conflict_stats()
        
        if stats['active_auctions'] > 0 or stats['completed_auctions'] > 0:
            print(f"🎯 路口竞价状态: {stats['active_auctions']}进行中 | "
                  f"{stats['completed_auctions']}已完成 | "
                  f"参与者: {stats['platoon_participants']}车队+{stats['vehicle_participants']}单车 | "
                  f"路口内:{stats['in_junction_participants']} 接近:{stats['approaching_participants']}")
            
            # 打印冲突状态
            if conflict_stats['deadlocked_agents'] > 0:
                print(f"🚨 冲突状态: {conflict_stats['deadlocked_agents']}死锁/{conflict_stats['waiting_agents']}等待 "
                      f"(阈值:{conflict_stats['deadlock_threshold']}s)")
