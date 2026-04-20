"""§41.4 — FinBERT scorer fallback + (optional) model path."""

import pytest

from nvda_sentiment.config import Settings
from nvda_sentiment.scorers.finbert_scorer import FINBERT_AVAILABLE, FinBERTScorer


def test_fallback_when_unavailable(monkeypatch):
    # Force the unavailable path regardless of whether torch/transformers installed.
    monkeypatch.setattr(
        "nvda_sentiment.scorers.finbert_scorer.FINBERT_AVAILABLE", False
    )
    scorer = FinBERTScorer(Settings(cache_dir="/tmp/nvda-test-cache"))
    assert scorer.available is False
    score, count = scorer.score_text("Revenue grew strongly. We are pleased.")
    assert score == 0.0
    assert count == 0


def test_empty_text_returns_zero():
    scorer = FinBERTScorer(Settings(cache_dir="/tmp/nvda-test-cache"))
    # Regardless of availability, empty text is a no-op
    s, n = scorer.score_text("")
    assert s == 0.0 and n == 0


@pytest.mark.skipif(not FINBERT_AVAILABLE, reason="FinBERT extras not installed")
def test_finbert_bounded_score_if_available():  # pragma: no cover - exercised only with [finbert] extras
    scorer = FinBERTScorer(Settings(cache_dir="/tmp/nvda-test-cache"))
    if not scorer.available:
        pytest.skip("FinBERT model failed to load")
    s, n = scorer.score_text("Record revenue and strong gross margin expansion this quarter.")
    assert -1.0 <= s <= 1.0
    assert n >= 1
