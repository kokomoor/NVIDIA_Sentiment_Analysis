"""Investor-branch adapters (§6.3)."""

from .analyst_signals import AnalystSignalsAdapter
from .broad_market import BroadMarketAdapter
from .options_flow import OptionsFlowAdapter
from .short_interest import ShortInterestAdapter
from .social import SocialAdapter

__all__ = [
    "AnalystSignalsAdapter",
    "BroadMarketAdapter",
    "OptionsFlowAdapter",
    "ShortInterestAdapter",
    "SocialAdapter",
]
