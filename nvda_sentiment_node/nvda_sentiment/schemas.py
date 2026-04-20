"""Public and internal pydantic schemas (§11)."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Public request / response
# --------------------------------------------------------------------------- #


class SentimentRequest(BaseModel):
    ticker: str = Field(default="NVDA")
    as_of_date: date
    lookback_quarters: int = Field(default=4, ge=1, le=8)
    include_market_context: bool = True
    use_cache: bool = True


class SentimentResponse(BaseModel):
    ticker: str
    as_of_date: date
    market_sentiment_score: float
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
