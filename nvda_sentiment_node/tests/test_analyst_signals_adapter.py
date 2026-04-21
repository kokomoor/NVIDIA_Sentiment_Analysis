"""§41 — analyst-signals adapter (monkeypatched yfinance)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nvda_sentiment.adapters.investor import AnalystSignalsAdapter
from nvda_sentiment.config import Settings


def test_analyst_adapter_bullish_upside_and_buy_rec(tmp_path, monkeypatch):
    # upside = 0.20 → upside_score = 0.50
    # dispersion = 20/120 = 0.1667 → damper = 0.8333
    # rec 2.0 → rec_score = 0.50
    # combined = 0.5*(0.50*0.8333) + 0.5*0.50 = 0.45833
    info = {
        "currentPrice": 100.0,
        "targetMeanPrice": 120.0,
        "targetHighPrice": 130.0,
        "targetLowPrice": 110.0,
        "recommendationMean": 2.0,
        "numberOfAnalystOpinions": 25,
    }

    class _YF:
        @staticmethod
        def Ticker(*a, **k):  # noqa: N802
            return SimpleNamespace(info=info)

    import nvda_sentiment.adapters.investor.analyst_signals as mod
    monkeypatch.setattr(mod, "yf", _YF)
    adapter = AnalystSignalsAdapter(Settings(cache_dir=tmp_path / ".c"))
    sub = adapter.get_signal()
    assert sub.ok is True
    assert sub.score == pytest.approx(0.5 * (0.5 * (1.0 - (20 / 120))) + 0.5 * 0.5)


def test_analyst_adapter_few_analysts_not_ok(tmp_path, monkeypatch):
    info = {
        "currentPrice": 100.0,
        "targetMeanPrice": 120.0,
        "targetHighPrice": 130.0,
        "targetLowPrice": 110.0,
        "recommendationMean": 2.0,
        "numberOfAnalystOpinions": 2,
    }

    class _YF:
        @staticmethod
        def Ticker(*a, **k):  # noqa: N802
            return SimpleNamespace(info=info)

    import nvda_sentiment.adapters.investor.analyst_signals as mod
    monkeypatch.setattr(mod, "yf", _YF)
    adapter = AnalystSignalsAdapter(Settings(cache_dir=tmp_path / ".c"))
    sub = adapter.get_signal()
    assert sub.ok is False
