class NashSolver:
    def resolve_conflict(self, bids):
        """当出现相同竞标时，用纳什均衡处理"""
        # 构建博弈矩阵
        # 返回均衡通行顺序
        return nash_equilibrium_order(bids)
