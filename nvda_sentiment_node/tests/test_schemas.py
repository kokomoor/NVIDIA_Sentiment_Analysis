"""§41.1 — schema validation and defaults."""

from datetime import date

import pytest
from pydantic import ValidationError

from nvda_sentiment.schemas import SentimentRequest, SentimentResponse


def test_request_defaults():
    req = SentimentRequest(as_of_date=date(2026, 4, 20))
    assert req.ticker == "NVDA"
    assert req.lookback_quarters == 4
    assert req.include_market_context is True
    assert req.include_investor_branch is True
    assert req.use_cache is True


def test_request_invalid_lookback_low():
    with pytest.raises(ValidationError):
        SentimentRequest(as_of_date=date(2026, 4, 20), lookback_quarters=0)


def test_request_invalid_lookback_high():
    with pytest.raises(ValidationError):
        SentimentRequest(as_of_date=date(2026, 4, 20), lookback_quarters=9)


def test_response_roundtrip():
    resp = SentimentResponse(
        ticker="NVDA",
        as_of_date=date(2026, 4, 20),
        market_sentiment_score=61.4,
        leadership_score=61.4,
        investor_score=55.0,
        combined_score=61.4,
        divergence=6.4,
        label="mildly bullish",
        confidence=0.8,
        components={"filing_tone": 0.3, "filing_delta": 0.1, "guidance_tone": 0.2},
        signals=["a"],
        source_coverage={"10k_count": 1},
        metadata={"node_version": "0.2.0", "warnings": []},
    )
    dumped = resp.model_dump()
    assert dumped["label"] == "mildly bullish"
    # §4.2: market_sentiment_score must equal combined_score exactly.
    assert dumped["market_sentiment_score"] == dumped["combined_score"]
