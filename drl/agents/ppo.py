class PPOAgent:
    def get_action(self, state):
        """根据邻域信息和局部策略决定动作"""
        obs = self.encode_observation(state)
        action = self.policy(obs)
        return action
