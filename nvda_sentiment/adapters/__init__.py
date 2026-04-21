from .investor.broad_market import BroadMarketAdapter
from .nvidia_ir import NvidiaIRAdapter
from .sec_api import SECAdapter

# Deprecated alias (§6.2.1). Remove in next release.
MarketContextAdapter = BroadMarketAdapter

__all__ = [
    "SECAdapter",
    "NvidiaIRAdapter",
    "BroadMarketAdapter",
    "MarketContextAdapter",
]
