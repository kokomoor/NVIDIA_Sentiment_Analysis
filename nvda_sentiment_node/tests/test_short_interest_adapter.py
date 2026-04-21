"""§41 — short-interest adapter (bucket tables + monkeypatched yfinance)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nvda_sentiment.adapters.investor import ShortInterestAdapter
from nvda_sentiment.adapters.investor.short_interest import _bucket_pct, _bucket_ratio
from nvda_sentiment.config import Settings


# --------------------------------------------------------------------------- #
# Bucket functions (§3.2)
# --------------------------------------------------------------------------- #


def test_bucket_pct_thresholds():
    assert _bucket_pct(0.005) == 0.20
    assert _bucket_pct(0.01) == 0.20
    assert _bucket_pct(0.02) == 0.10
    assert _bucket_pct(0.04) == 0.00
    assert _bucket_pct(0.06) == -0.15
    assert _bucket_pct(0.07) == -0.15
    assert _bucket_pct(0.10) == -0.30


def test_bucket_ratio_thresholds():
    assert _bucket_ratio(0.5) == 0.20
    assert _bucket_ratio(1.0) == 0.20
    assert _bucket_ratio(2.0) == 0.00
    assert _bucket_ratio(3.0) == -0.10
    assert _bucket_ratio(3.5) == -0.10
    assert _bucket_ratio(5.0) == -0.25


# --------------------------------------------------------------------------- #
# Monkeypatched yfinance paths
# --------------------------------------------------------------------------- #


def test_short_interest_adapter_low_short_bullish(tmp_path, monkeypatch):
    # shortPct 1% → +0.20; shortRatio 1.0 → +0.20; combined 0.60*0.20 + 0.40*0.20 = 0.20
    class _YF:
        @staticmethod
        def Ticker(*a, **k):  # noqa: N802
            return SimpleNamespace(info={"shortPercentOfFloat": 0.01, "shortRatio": 1.0})

    import nvda_sentiment.adapters.investor.short_interest as mod
    monkeypatch.setattr(mod, "yf", _YF)
    adapter = ShortInterestAdapter(Settings(cache_dir=tmp_path / ".c"))
    sub = adapter.get_signal()
    assert sub.ok is True
    assert sub.score == pytest.approx(0.20)


def test_short_interest_adapter_missing_both_fields(tmp_path, monkeypatch):
    class _YF:
        @staticmethod
        def Ticker(*a, **k):  # noqa: N802
            return SimpleNamespace(info={})

    import nvda_sentiment.adapters.investor.short_interest as mod
    monkeypatch.setattr(mod, "yf", _YF)
    adapter = ShortInterestAdapter(Settings(cache_dir=tmp_path / ".c"))
    sub = adapter.get_signal()
    assert sub.ok is False
