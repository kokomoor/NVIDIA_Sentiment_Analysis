"""Public and internal pydantic schemas (§§4, 11)."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Public request / response
# --------------------------------------------------------------------------- #


class SentimentRequest(BaseModel):
    """Request schema. ``as_of_date`` is intentionally absent — the pipeline
    can only produce same-day sentiment (investor-branch adapters fetch live
    data with no historical snapshots), and asymmetric as-of application
    across branches would be incoherent. The response's ``as_of_date`` is
    always stamped as ``date.today()`` by the node.
    """

    ticker: str = Field(default="NVDA")
    lookback_quarters: int = Field(default=4, ge=1, le=8)
    include_market_context: bool = True      # gates the BroadMarketAdapter only
    include_investor_branch: bool = True     # master toggle for investor branch
    use_cache: bool = True
    # When True, node.run() also appends/upserts the response into the
    # history SQLite DB (see nvda_sentiment.persistence).
    persist_history: bool = True


class SentimentResponse(BaseModel):
    ticker: str
    as_of_date: date
    # Backwards-compat alias (== combined_score); do not drop.
    market_sentiment_score: float
    leadership_score: float
    investor_score: float
    combined_score: float
    divergence: float
    label: str
    confidence: float
    components: Dict[str, float]
    signals: List[str]
    source_coverage: Dict[str, int]
    metadata: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Internal document models
# --------------------------------------------------------------------------- #

SourceType = Literal[
    "10-K",
    "10-Q",
    "8-K",
    "press_release",
    "transcript",
    "cfo_commentary",
]


class SourceDocument(BaseModel):
    source_type: SourceType
    title: str
    url: str
    filed_at: str
    fiscal_period: Optional[str] = None
    raw_html: Optional[str] = None
    clean_text: Optional[str] = None


class SectionScore(BaseModel):
    section_name: str
    finbert_score: float
    lexicon_score: float
    uncertainty_penalty: float
    final_score: float
    sentence_count: int


class DocumentScore(BaseModel):
    source_type: str
    filed_at: str
    title: str
    section_scores: List[SectionScore]
    final_score: float


# --------------------------------------------------------------------------- #
# Investor-branch internal models (§4.3)
# --------------------------------------------------------------------------- #


InvestorSubName = Literal[
    "options_flow",
    "short_interest",
    "analyst_signal",
    "social",
    "broad_market",
]


class InvestorSubScore(BaseModel):
    name: InvestorSubName
    score: float            # [-1, +1]
    ok: bool                # whether the source produced usable data
    detail: Dict[str, Any]  # raw numbers for signals/debug (not public)


class InvestorBranchOutput(BaseModel):
    sub_scores: List[InvestorSubScore]
    investor_component: float   # weighted combination, [-1, +1]
    investor_score: float       # 0-100
    ok: bool                    # true if at least one sub-source returned ok
