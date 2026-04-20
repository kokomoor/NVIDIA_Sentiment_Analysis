"""§41.6 — end-to-end with fully mocked adapters (no network)."""

from datetime import date
from typing import List

from nvda_sentiment.config import Settings
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.schemas import SentimentRequest, SourceDocument


_POSITIVE_MDA = """Item 2. Management's Discussion and Analysis
Record revenue growth and strong momentum across the business. Successful expansion
of our platform drove improved operating income. We expect continued robust demand
for our data center products.

Our outlook remains constructive. We believe growth will accelerate in coming
quarters. Leadership in AI infrastructure is driving opportunity across segments.

Item 3. Quantitative and Qualitative Disclosures
unrelated content.
"""

_NEGATIVE_MDA = """Item 2. Management's Discussion and Analysis
Revenue declined materially this quarter, with weakness across gaming and automotive.
We face ongoing pressure from headwinds and elevated uncertainty. Our outlook for
the next quarter is cautious given regulatory uncertainty and litigation exposure.

We may experience further disruption and declining margins as challenges persist.

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
    def __init__(self, docs: List[SourceDocument], html_map: dict[str, str]):
        self._docs = docs
        self._html_map = html_map

    def get_quarterly_results_documents(self, lookback_quarters: int) -> List[SourceDocument]:
        return list(self._docs)

    def fetch_document_html(self, doc: SourceDocument) -> str:
        return self._html_map[doc.url]


class FakeMarket:
    def __init__(self, score=0.1, ok=True):
        self._score = score
        self._ok = ok

    def get_combined_score(self):
        return (self._score, self._ok)


def _positive_node(tmp_path):
    docs = [
        SourceDocument(source_type="10-Q", title="10-Q Feb 2026", url="http://sec/q1-curr", filed_at="2026-02-15"),
        SourceDocument(source_type="10-Q", title="10-Q Feb 2025", url="http://sec/q1-prior", filed_at="2025-02-15"),
    ]
    html_map = {
        "http://sec/q1-curr": f"<html><body><pre>{_POSITIVE_MDA}</pre></body></html>",
        "http://sec/q1-prior": f"<html><body><pre>{_NEGATIVE_MDA}</pre></body></html>",
    }
    settings = Settings(cache_dir=tmp_path)
    return NVDASentimentNode(
        settings=settings,
        sec_adapter=FakeSEC(docs, html_map),
        ir_adapter=FakeIR([], {}),
        market_context=FakeMarket(score=0.2, ok=True),
    )


def _negative_node(tmp_path):
    docs = [
        SourceDocument(source_type="10-Q", title="10-Q Feb 2026", url="http://sec/q1-curr", filed_at="2026-02-15"),
        SourceDocument(source_type="10-Q", title="10-Q Feb 2025", url="http://sec/q1-prior", filed_at="2025-02-15"),
    ]
    html_map = {
        "http://sec/q1-curr": f"<html><body><pre>{_NEGATIVE_MDA}</pre></body></html>",
        "http://sec/q1-prior": f"<html><body><pre>{_POSITIVE_MDA}</pre></body></html>",
    }
    settings = Settings(cache_dir=tmp_path)
    return NVDASentimentNode(
        settings=settings,
        sec_adapter=FakeSEC(docs, html_map),
        ir_adapter=FakeIR([], {}),
        market_context=FakeMarket(score=-0.2, ok=True),
    )


def test_positive_input_produces_bullish_leaning_score(tmp_path):
    node = _positive_node(tmp_path)
    resp = node.run(SentimentRequest(as_of_date=date(2026, 4, 20)))
    assert 0 <= resp.market_sentiment_score <= 100
    assert 0.10 <= resp.confidence <= 0.95
    assert resp.market_sentiment_score > 50
    assert len(resp.signals) >= 4


def test_negative_input_produces_bearish_leaning_score(tmp_path):
    node = _negative_node(tmp_path)
    resp = node.run(SentimentRequest(as_of_date=date(2026, 4, 20)))
    assert resp.market_sentiment_score < 50


def test_no_sources_returns_neutral_floor(tmp_path):
    settings = Settings(cache_dir=tmp_path)
    node = NVDASentimentNode(
        settings=settings,
        sec_adapter=FakeSEC([], {}),
        ir_adapter=FakeIR([], {}),
        market_context=FakeMarket(score=0.0, ok=False),
    )
    resp = node.run(SentimentRequest(as_of_date=date(2026, 4, 20)))
    assert resp.market_sentiment_score == 50.0
    assert resp.label == "neutral"
    assert resp.confidence == 0.10
    assert "No scorable documents available" in resp.metadata["warnings"]
