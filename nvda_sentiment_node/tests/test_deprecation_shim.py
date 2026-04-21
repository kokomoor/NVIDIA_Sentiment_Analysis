"""§6.6.1 — backwards-compat shim for the old `market_context=` kwarg."""

from __future__ import annotations

import warnings

import pytest

from nvda_sentiment.config import Settings
from nvda_sentiment.features.investor_branch import InvestorBranch
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.schemas import InvestorSubScore


class _StubBroadMarket:
    """Stand-in for the historical BroadMarketAdapter call signature."""

    def __init__(self):
        self.use_cache = True

    def get_signal(self) -> InvestorSubScore:
        return InvestorSubScore(
            name="broad_market", score=0.2, ok=True, detail={"stub": True}
        )


class _StubSEC:
    def get_relevant_filings(self, n):
        return []

    def fetch_filing_html(self, doc):  # pragma: no cover
        raise RuntimeError("not called")


class _StubIR:
    def get_quarterly_results_documents(self, n):
        return []

    def fetch_document_html(self, doc):  # pragma: no cover
        raise RuntimeError("not called")


def test_market_context_kwarg_emits_deprecation_warning(tmp_path):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        node = NVDASentimentNode(
            settings=Settings(cache_dir=tmp_path),
            sec_adapter=_StubSEC(),
            ir_adapter=_StubIR(),
            market_context=_StubBroadMarket(),
        )
    dep = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert dep, "Expected a DeprecationWarning when passing market_context="
    assert "investor_branch" in str(dep[0].message)

    # The shim must wire market_context as the broad_market sub-adapter.
    assert isinstance(node.investor_branch, InvestorBranch)
    assert isinstance(node.investor_branch.broad_market, _StubBroadMarket)


def test_market_context_and_investor_branch_together_raises(tmp_path):
    with pytest.raises(ValueError):
        NVDASentimentNode(
            settings=Settings(cache_dir=tmp_path),
            sec_adapter=_StubSEC(),
            ir_adapter=_StubIR(),
            investor_branch=InvestorBranch(Settings(cache_dir=tmp_path)),
            market_context=_StubBroadMarket(),
        )
