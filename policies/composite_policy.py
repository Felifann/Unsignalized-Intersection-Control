class CompositePolicy:
    def __init__(self, platoon_mgr, auction, nash, rl):
        self.platoon_mgr = platoon_mgr
        self.auction = auction
        self.nash = nash
        self.rl = rl

    def get_actions(self, state, order):
        """统一封装所有决策来源"""
        actions = {}
        for vehicle in state.vehicles:
            obs = self.rl.encode_observation(vehicle, state.neighborhood(vehicle))
            actions[vehicle.id] = self.rl.get_action(obs)
        return actions
