"""§41.7 — determinism: two identical runs must produce identical output.

Drops ``generated_at`` from comparison (it is a wall-clock timestamp).
"""

from datetime import date
from typing import List

from nvda_sentiment.config import Settings
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.schemas import (
    InvestorBranchOutput,
    InvestorSubScore,
    SentimentRequest,
    SourceDocument,
)


_TEXT_A = """Item 2. Management's Discussion and Analysis
Record revenue growth and strong momentum. Successful expansion of our platform
drove improved operating income. We expect continued robust demand.

Item 3. Quantitative and Qualitative Disclosures
unrelated.
"""

_TEXT_B = """Item 2. Management's Discussion and Analysis
Revenue was modestly lower. We face pressure but operations remain stable.
Outlook is balanced heading into the next quarter.

Item 3. Quantitative and Qualitative Disclosures
unrelated.
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

    def fetch_document_html(self, doc):
        raise RuntimeError("should not be called")


class FakeInvestorBranch:
    """Deterministic stub — returns one always-ok broad_market sub-score."""

    def set_use_cache(self, use_cache):
        return None

    def run(self, *, include_broad_market):
        subs = []
        if include_broad_market:
            subs.append(
                InvestorSubScore(
                    name="broad_market", score=0.1, ok=True, detail={"stub": True}
                )
            )
        comp = subs[0].score if subs else 0.0
        return InvestorBranchOutput(
            sub_scores=subs,
            investor_component=comp,
            investor_score=50.0 + 50.0 * comp,
            ok=bool(subs),
        )


def _make_node(tmp_path):
    docs = [
        SourceDocument(source_type="10-Q", title="curr", url="u1", filed_at="2026-02-15"),
        SourceDocument(source_type="10-Q", title="prior", url="u2", filed_at="2025-02-15"),
    ]
    html_map = {
        "u1": f"<html><body><pre>{_TEXT_A}</pre></body></html>",
        "u2": f"<html><body><pre>{_TEXT_B}</pre></body></html>",
    }
    settings = Settings(cache_dir=tmp_path)
    return NVDASentimentNode(
        settings=settings,
        sec_adapter=FakeSEC(docs, html_map),
        ir_adapter=FakeIR(),
        investor_branch=FakeInvestorBranch(),
    )


def test_two_runs_are_identical(tmp_path):
    node_a = _make_node(tmp_path / "a")
    node_b = _make_node(tmp_path / "b")
    req = SentimentRequest(as_of_date=date(2026, 4, 20))
    resp_a = node_a.run(req).model_dump()
    resp_b = node_b.run(req).model_dump()
    resp_a["metadata"] = {k: v for k, v in resp_a["metadata"].items() if k != "generated_at"}
    resp_b["metadata"] = {k: v for k, v in resp_b["metadata"].items() if k != "generated_at"}
    assert resp_a == resp_b
