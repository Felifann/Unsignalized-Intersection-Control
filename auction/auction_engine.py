import time
import math
from .bid_policy import AgentBidPolicy

class DecentralizedAuctionEngine:
    def __init__(self, intersection_center=(-188.9, -89.7, 0.0), communication_range=50.0):
        self.intersection_center = intersection_center
        self.communication_range = communication_range
        
        # 分布式拍卖状态
        self.active_auctions = {}  # {auction_id: auction_data}
        self.agent_bids = {}       # {agent_id: bid_data}
        self.auction_results = {}  # {auction_id: winner_list}
        
        # 通信模拟
        self.message_queue = []    # 模拟车车通信消息队列
        self.last_auction_time = 0
        self.auction_interval = 2.0  # 每2秒举行一次拍卖
        
        print("🎯 分布式拍卖引擎初始化完成")

    def update(self, vehicle_states, platoon_manager):
        """
        主更新函数：管理分布式拍卖过程
        按照中心思想：先分platoon，然后所有agent（platoon+剩余单车）参与竞价
        """
        current_time = time.time()
        
        # 1. 按照中心思想识别agents：platoons + 剩余单车
        auction_agents = self._identify_agents_by_central_idea(vehicle_states, platoon_manager)
        
        # 2. 定期启动新拍卖
        if current_time - self.last_auction_time >= self.auction_interval:
            if auction_agents:
                auction_id = self._start_new_auction(auction_agents, current_time)
                self.last_auction_time = current_time
                print(f"🎯 启动新拍卖 {auction_id}，参与agents: {len(auction_agents)}个 "
                      f"(车队:{len([a for a in auction_agents if a['type']=='platoon'])}个, "
                      f"单车:{len([a for a in auction_agents if a['type']=='vehicle'])}个)")
        
        # 3. 处理正在进行的拍卖
        self._process_active_auctions(current_time)
        
        # 4. 模拟车车通信
        self._simulate_v2v_communication()
        
        # 5. 返回当前优先级排序
        return self._get_current_priority_order()

    def _identify_agents_by_central_idea(self, vehicle_states, platoon_manager):
        """
        按照中心思想识别agents：
        1. 先获取已经分好的platoons
        2. 然后获取剩余的单个车辆
        3. 只考虑交叉口附近的agents
        """
        agents = []
        
        # 获取交叉口附近的车辆
        intersection_vehicles = [v for v in vehicle_states 
                               if self._distance_to_intersection(v) <= 30.0]
        
        # 1. 添加所有platoons作为agents（无论是否在交叉口附近，由platoon的leader位置决定）
        platoon_vehicle_ids = set()  # 记录已编队车辆ID
        
        for platoon in platoon_manager.get_all_platoons():
            leader = platoon.get_leader()
            if leader and self._distance_to_intersection(leader) <= 30.0:
                # 创建platoon agent
                platoon_agent = {
                    'type': 'platoon',
                    'id': f"platoon_{leader['id']}",
                    'vehicles': platoon.vehicles,
                    'goal_direction': platoon.get_goal_direction(),
                    'leader_location': leader['location'],
                    'size': platoon.get_size()
                }
                agents.append(platoon_agent)
                
                # 记录platoon中的所有车辆ID
                for vehicle in platoon.vehicles:
                    platoon_vehicle_ids.add(vehicle['id'])
        
        # 2. 添加剩余的单个车辆作为agents（不在任何platoon中的车辆）
        for vehicle in intersection_vehicles:
            if vehicle['id'] not in platoon_vehicle_ids:
                # 只有有明确目的地的单车才参与竞价
                if self._vehicle_has_destination(vehicle):
                    vehicle_agent = {
                        'type': 'vehicle',
                        'id': vehicle['id'],
                        'data': vehicle,
                        'location': vehicle['location']
                    }
                    agents.append(vehicle_agent)
        
        return agents

    def _vehicle_has_destination(self, vehicle):
        """检查车辆是否有明确的目的地（用于判断是否参与竞价）"""
        # 简化版本：检查是否有destination信息
        return vehicle.get('destination') is not None

    def _start_new_auction(self, agents, start_time):
        """启动新的分布式拍卖"""
        auction_id = f"auction_{int(start_time)}"
        
        auction_data = {
            'id': auction_id,
            'start_time': start_time,
            'participants': agents,  # 现在是真正的agents（platoons + 剩余单车）
            'bids': {},
            'status': 'bidding',  # bidding -> evaluating -> completed
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
        
        print(f"🏆 拍卖 {auction_id} 完成，获胜者排序:")
        for i, winner in enumerate(winners[:5]):  # 只显示前5名
            agent = winner['agent']
            bid_value = winner['bid_value']
            rank = winner['rank']
            
            if agent['type'] == 'platoon':
                print(f"   #{rank}: 🚛 车队{agent['id']} "
                      f"({agent['size']}车-{agent['goal_direction']}) "
                      f"出价:{bid_value:.1f}")
            else:
                print(f"   #{rank}: 🚗 单车{agent['id']} "
                      f"出价:{bid_value:.1f}")

    def _collect_bids_for_auction(self, auction_id, auction_data):
        """为特定拍卖收集竞价"""
        for agent in auction_data['participants']:
            agent_id = agent['id']
            
            # 检查是否已经出价
            if agent_id not in auction_data['bids']:
                # 创建竞价策略并计算出价
                bid_policy = AgentBidPolicy(agent, self.intersection_center)
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
        
        # 统计参与者类型
        platoon_count = 0
        vehicle_count = 0
        for auction in self.active_auctions.values():
            for participant in auction['participants']:
                if participant['type'] == 'platoon':
                    platoon_count += 1
                else:
                    vehicle_count += 1
        
        return {
            'active_auctions': active_count,
            'total_participants': total_participants,
            'platoon_participants': platoon_count,
            'vehicle_participants': vehicle_count,
            'completed_auctions': len(self.auction_results),
            'message_queue_size': len(self.message_queue)
        }

    def print_auction_status(self):
        """打印拍卖状态"""
        stats = self.get_auction_stats()
        
        if stats['active_auctions'] > 0 or stats['completed_auctions'] > 0:
            print(f"🎯 拍卖状态: {stats['active_auctions']}进行中 | "
                  f"{stats['completed_auctions']}已完成 | "
                  f"参与者: {stats['platoon_participants']}车队+{stats['vehicle_participants']}单车")
