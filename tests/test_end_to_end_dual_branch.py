"""§9.2 — full dual-branch integration with stubbed adapters (no network)."""

from __future__ import annotations

from datetime import date
from typing import List

from nvda_sentiment.config import Settings
from nvda_sentiment.features.combiner import combine
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.schemas import (
    InvestorBranchOutput,
    InvestorSubScore,
    SentimentRequest,
    SourceDocument,
)


_POS_MDA = """Item 2. Management's Discussion and Analysis
Record revenue growth and strong momentum. Successful expansion drove improved operating income.
We expect continued robust demand across our data center business.

Item 3. Quantitative and Qualitative Disclosures
unrelated content.
"""

_NEG_MDA = """Item 2. Management's Discussion and Analysis
Revenue declined materially. We face ongoing pressure and elevated uncertainty.
Outlook is cautious given regulatory risk.

Item 3. Quantitative and Qualitative Disclosures
unrelated content.
"""


class FakeSEC:
    def __init__(self, docs: List[SourceDocument], html_map: dict[str, str]):
        self._docs = docs
        self._html_map = html_map

    def get_relevant_filings(self, lookback_quarters: int) -> List[SourceDocument]:
        return list(self._docs)

    def fetch_filing_html(self, doc: SourceDocument) -> str:
        return self._html_map[doc.url]


class FakeIR:
    def get_quarterly_results_documents(self, lookback_quarters: int):
        return []

    def fetch_document_html(self, doc):  # pragma: no cover
        raise RuntimeError("should not be called")


class FakeInvestorBranch:
    """Deterministic stub exercising the investor-branch output contract."""

    def __init__(self, *, component: float, ok: bool = True, subs=None):
        self._component = component
        self._ok = ok
        self._subs = subs if subs is not None else [
            InvestorSubScore(name="options_flow", score=component, ok=True, detail={}),
            InvestorSubScore(name="short_interest", score=component, ok=True, detail={}),
            InvestorSubScore(name="analyst_signal", score=component, ok=True, detail={}),
            InvestorSubScore(name="social", score=component, ok=True, detail={}),
            InvestorSubScore(name="broad_market", score=component, ok=True, detail={}),
        ]

    def set_use_cache(self, use_cache):
        return None

    def run(self, *, include_broad_market):
        subs = [s for s in self._subs if include_broad_market or s.name != "broad_market"]
        return InvestorBranchOutput(
            sub_scores=subs,
            investor_component=self._component,
            investor_score=50.0 + 50.0 * max(-1.0, min(1.0, self._component)),
            ok=self._ok,
        )


def _docs_pair(pos_first: bool):
    curr_text = _POS_MDA if pos_first else _NEG_MDA
    prior_text = _NEG_MDA if pos_first else _POS_MDA
    docs = [
        SourceDocument(source_type="10-Q", title="curr", url="http://sec/curr", filed_at="2026-02-15"),
        SourceDocument(source_type="10-Q", title="prior", url="http://sec/prior", filed_at="2025-02-15"),
    ]
    html_map = {
        "http://sec/curr": f"<html><body><pre>{curr_text}</pre></body></html>",
        "http://sec/prior": f"<html><body><pre>{prior_text}</pre></body></html>",
    }
    return docs, html_map


def _build(tmp_path, pos_first: bool, *, investor_component: float, investor_ok: bool = True):
    docs, html_map = _docs_pair(pos_first)
    settings = Settings(cache_dir=tmp_path)
    return NVDASentimentNode(
        settings=settings,
        sec_adapter=FakeSEC(docs, html_map),
        ir_adapter=FakeIR(),
        investor_branch=FakeInvestorBranch(component=investor_component, ok=investor_ok),
    )


def test_response_carries_all_new_fields_and_alias(tmp_path) -> None:
    node = _build(tmp_path, pos_first=True, investor_component=0.2)
    resp = node.run(SentimentRequest(persist_history=False))
    # Alias invariant — exact equality after identical rounding.
    assert resp.market_sentiment_score == resp.combined_score
    # All new fields populated.
    assert 0.0 <= resp.leadership_score <= 100.0
    assert 0.0 <= resp.investor_score <= 100.0
    assert -100.0 <= resp.divergence <= 100.0
    # Components carry new keys (sub-components + aggregates), not the old `investor_context`.
    for key in ("options_flow", "short_interest", "analyst_signal", "social",
                "broad_market", "leadership_component", "investor_component"):
        assert key in resp.components
    assert "investor_context" not in resp.components


def test_mgmt_bullish_mkt_bearish_combined_below_leadership(tmp_path) -> None:
    # Positive MDA → positive leadership; inject strongly bearish investor component.
    node = _build(tmp_path, pos_first=True, investor_component=-0.8)
    resp = node.run(SentimentRequest(persist_history=False))
    assert resp.combined_score < resp.leadership_score
    # Divergence sign should match (leadership positive, investor negative).
    assert resp.divergence > 0


def test_both_bullish_roughly_weighted_blend(tmp_path) -> None:
    node = _build(tmp_path, pos_first=True, investor_component=0.3)
    resp = node.run(SentimentRequest(persist_history=False))
    # In agreement quadrant: adjustment=0, so combined is a strict base blend.
    # combined_score must sit between leadership_score and investor_score.
    lo = min(resp.leadership_score, resp.investor_score)
    hi = max(resp.leadership_score, resp.investor_score)
    assert lo - 0.05 <= resp.combined_score <= hi + 0.05


def test_no_investor_branch_combined_equals_leadership(tmp_path) -> None:
    node = _build(tmp_path, pos_first=True, investor_component=-0.9, investor_ok=True)
    resp = node.run(
        SentimentRequest(
            include_investor_branch=False,
            persist_history=False,
        )
    )
    assert resp.combined_score == resp.leadership_score
    assert resp.market_sentiment_score == resp.leadership_score


def test_investor_branch_failed_falls_back_to_leadership(tmp_path) -> None:
    node = _build(tmp_path, pos_first=True, investor_component=-0.9, investor_ok=False)
    resp = node.run(SentimentRequest(persist_history=False))
    assert resp.combined_score == resp.leadership_score
    # Warning about investor-branch fallback should be present.
    assert any("Investor branch unavailable" in w for w in resp.metadata["warnings"])


def test_combined_score_matches_combiner_formula_within_rounding(tmp_path) -> None:
    # Fakes: component=+0.5, ok=True; positive MDA drives leadership positive.
    # Recompute combined via combine() from the 3dp-exposed components and assert
    # equality to within one 0.1 rounding unit (the tightest assertion the public
    # component contract supports — §10).
    node = _build(tmp_path, pos_first=True, investor_component=0.5)
    resp = node.run(SentimentRequest(persist_history=False))
    L = float(resp.components["leadership_component"])
    I = float(resp.components["investor_component"])
    out = combine(
        leadership_component=L,
        investor_component=I,
        alpha=0.55,
        lambda_neg=0.25,
        lambda_pos=0.10,
        investor_branch_ok=True,
    )
    expected = round(out.combined_score, 1)
    assert abs(resp.combined_score - expected) <= 0.1
