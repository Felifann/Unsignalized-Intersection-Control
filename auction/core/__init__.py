"""
Core auction module for unsignalized intersection control.

This module provides:
- Clean auction engine implementation
- Modular bidding strategies
- Extensible tie-breaking mechanisms
- Integration points for RL and Nash Equilibrium systems
"""

from .auction_engine_v2 import AuctionEngine
from .bid_strategies import *
from .auction_result import AuctionResult

__all__ = [
    'AuctionEngine',
    'BiddingStrategy',
    'DefaultBiddingStrategy', 
    'PlatoonBiddingStrategy',
    'AuctionResult'
]
