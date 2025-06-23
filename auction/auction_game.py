class AuctionEngine:
    def collect_bids(self, state):
        """让每辆车根据状态出价"""
        bids = {}
        for car in state.vehicles:
            bids[car.id] = self.calculate_bid(car)
        return bids
