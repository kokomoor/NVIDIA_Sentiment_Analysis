"""Investor-branch aggregator (§6.4).

Coordinates the five investor sub-adapters, applies pro-rata
redistribution across the *usable* ones, and returns an
:class:`InvestorBranchOutput` ready for the combiner.
"""

from __future__ import annotations

from typing import List, Optional

from ..adapters.investor.analyst_signals import AnalystSignalsAdapter
from ..adapters.investor.broad_market import BroadMarketAdapter
from ..adapters.investor.options_flow import OptionsFlowAdapter
from ..adapters.investor.short_interest import ShortInterestAdapter
from ..adapters.investor.social import SocialAdapter
from ..config import INVESTOR_COMPONENT_WEIGHTS, Settings
from ..schemas import InvestorBranchOutput, InvestorSubScore
from ..utils.logging import get_logger


logger = get_logger(__name__)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class InvestorBranch:
    """Coordinate the five investor sub-adapters."""

    def __init__(
        self,
        settings: Settings,
        *,
        options: Optional[OptionsFlowAdapter] = None,
        shorts: Optional[ShortInterestAdapter] = None,
        analyst: Optional[AnalystSignalsAdapter] = None,
        social: Optional[SocialAdapter] = None,
        broad_market: Optional[BroadMarketAdapter] = None,
    ) -> None:
        self.settings = settings
        self.options = options or OptionsFlowAdapter(settings)
        self.shorts = shorts or ShortInterestAdapter(settings)
        self.analyst = analyst or AnalystSignalsAdapter(settings)
        self.social = social or SocialAdapter(settings)
        self.broad_market = broad_market or BroadMarketAdapter(settings)

    def set_use_cache(self, use_cache: bool) -> None:
        """Propagate the per-run `use_cache` flag to every sub-adapter."""
        for adapter in (
            self.options,
            self.shorts,
            self.analyst,
            self.social,
            self.broad_market,
        ):
            if hasattr(adapter, "use_cache"):
                adapter.use_cache = use_cache

    def run(self, *, include_broad_market: bool) -> InvestorBranchOutput:
        subs: List[InvestorSubScore] = []
        for name, adapter in (
            ("options_flow", self.options),
            ("short_interest", self.shorts),
            ("analyst_signal", self.analyst),
            ("social", self.social),
        ):
            subs.append(self._safe_signal(name, adapter))

        if include_broad_market:
            subs.append(self._safe_signal("broad_market", self.broad_market))

        investor_component = self._weighted_mean(subs)
        investor_score = 50.0 + 50.0 * _clip(investor_component)
        ok_any = any(s.ok for s in subs)

        return InvestorBranchOutput(
            sub_scores=subs,
            investor_component=investor_component,
            investor_score=investor_score,
            ok=ok_any,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_signal(name: str, adapter) -> InvestorSubScore:
        try:
            sub = adapter.get_signal()
        except Exception as exc:
            logger.warning("%s adapter failed: %s", name, exc)
            return InvestorSubScore(name=name, score=0.0, ok=False, detail={})
        if not isinstance(sub, InvestorSubScore):
            logger.warning("%s adapter returned unexpected type %r", name, type(sub))
            return InvestorSubScore(name=name, score=0.0, ok=False, detail={})
        return sub

    @staticmethod
    def _weighted_mean(subs: List[InvestorSubScore]) -> float:
        """Weighted mean across `ok` sub-scores; missing weight redistributed pro-rata.

        Sub-sources that are not ok have their weight reallocated to the
        remaining ok sub-sources in proportion to their own weights. This
        prevents a single failure from anchoring the component near zero.
        """
        w = INVESTOR_COMPONENT_WEIGHTS
        usable = [s for s in subs if s.ok and s.name in w]
        if not usable:
            return 0.0
        total_w = sum(w[s.name] for s in usable)
        if total_w <= 0.0:
            return 0.0
        weighted = sum(w[s.name] * float(s.score) for s in usable)
        return _clip(weighted / total_w)
