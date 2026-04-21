"""§41 — options-flow adapter (bucket tables + monkeypatched yfinance)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from nvda_sentiment.adapters.investor import OptionsFlowAdapter
from nvda_sentiment.adapters.investor.options_flow import (
    _bucket_iv,
    _bucket_pcr,
    _clip,
    _days_out,
)
from nvda_sentiment.config import Settings
from nvda_sentiment.schemas import InvestorSubScore


# --------------------------------------------------------------------------- #
# Bucket functions (§3.1)
# --------------------------------------------------------------------------- #


def test_bucket_pcr_thresholds():
    assert _bucket_pcr(0.40) == 0.50
    assert _bucket_pcr(0.60) == 0.50
    assert _bucket_pcr(0.70) == 0.25
    assert _bucket_pcr(0.85) == 0.25
    assert _bucket_pcr(1.00) == 0.00
    assert _bucket_pcr(1.20) == -0.25
    assert _bucket_pcr(1.50) == -0.25
    assert _bucket_pcr(2.00) == -0.50


def test_bucket_iv_thresholds():
    assert _bucket_iv(-2.0) == 0.20
    assert _bucket_iv(-1.0) == 0.20
    assert _bucket_iv(0.0) == 0.00
    assert _bucket_iv(0.5) == 0.00
    assert _bucket_iv(1.0) == -0.15
    assert _bucket_iv(1.5) == -0.15
    assert _bucket_iv(2.0) == -0.30


def test_days_out():
    today = date(2026, 1, 1)
    assert _days_out("2026-01-08", today) == 7


def test_clip_in_options_flow():
    assert _clip(2.0) == 1.0
    assert _clip(-2.0) == -1.0


# --------------------------------------------------------------------------- #
# Monkeypatched yfinance paths
# --------------------------------------------------------------------------- #


class _FakeChain:
    """Minimal duck-typed stand-in for yfinance's option_chain namedtuple."""

    def __init__(self, calls_rows: List[Dict[str, Any]], puts_rows: List[Dict[str, Any]]):
        self.calls = _FakeDF(calls_rows)
        self.puts = _FakeDF(puts_rows)


class _FakeDF:
    """Tiny DataFrame-ish that supports `df[column]` → list."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __getitem__(self, col: str) -> List[Any]:
        return [r.get(col) for r in self._rows]


class _FakeTicker:
    def __init__(
        self,
        *,
        options: List[str],
        chains: Dict[str, _FakeChain],
        spot: float,
        info: Dict[str, Any] | None = None,
    ):
        self.options = tuple(options)
        self._chains = chains
        self.fast_info = SimpleNamespace(last_price=spot)
        self.info = info or {}

    def option_chain(self, expiry: str) -> _FakeChain:
        return self._chains[expiry]


def _install_fake_yf(monkeypatch, fake_ticker: _FakeTicker):
    import nvda_sentiment.adapters.investor.options_flow as mod

    class _FakeYF:
        @staticmethod
        def Ticker(ticker: str, session=None):  # noqa: N802
            return fake_ticker

    monkeypatch.setattr(mod, "yf", _FakeYF)


def test_options_flow_adapter_call_heavy_bullish(tmp_path, monkeypatch):
    # PCR = 50/200 = 0.25 → bucket +0.50; IV z ≈ 0 → bucket 0.00
    # combined = 0.70*0.50 + 0.30*0.00 = +0.35
    calls = [
        {"strike": 90, "volume": 100, "impliedVolatility": 0.30},
        {"strike": 100, "volume": 100, "impliedVolatility": 0.30},
    ]
    puts = [
        {"strike": 90, "volume": 25, "impliedVolatility": 0.30},
        {"strike": 100, "volume": 25, "impliedVolatility": 0.30},
    ]
    today = date.today()
    exp1 = date.fromordinal(today.toordinal() + 14).isoformat()
    exp2 = date.fromordinal(today.toordinal() + 30).isoformat()
    chains = {
        exp1: _FakeChain(calls, puts),
        exp2: _FakeChain(calls, puts),
    }
    fake = _FakeTicker(options=[exp1, exp2], chains=chains, spot=100.0)

    monkeypatch.chdir(tmp_path)
    adapter = OptionsFlowAdapter(Settings(cache_dir=tmp_path / ".c"))
    _install_fake_yf(monkeypatch, fake)

    sub = adapter.get_signal()
    assert isinstance(sub, InvestorSubScore)
    assert sub.ok is True
    assert sub.name == "options_flow"
    assert sub.score == pytest.approx(0.35)
    assert sub.detail["pcr"] == pytest.approx(0.25)


def test_options_flow_adapter_no_expiries_returns_not_ok(tmp_path, monkeypatch):
    fake = _FakeTicker(options=[], chains={}, spot=100.0)
    adapter = OptionsFlowAdapter(Settings(cache_dir=tmp_path / ".c"))
    _install_fake_yf(monkeypatch, fake)
    sub = adapter.get_signal()
    assert sub.ok is False
    assert sub.score == 0.0


def test_options_flow_adapter_raises_to_not_ok(tmp_path, monkeypatch):
    class _BoomYF:
        @staticmethod
        def Ticker(*a, **k):  # noqa: N802
            raise RuntimeError("network down")

    import nvda_sentiment.adapters.investor.options_flow as mod
    monkeypatch.setattr(mod, "yf", _BoomYF)
    adapter = OptionsFlowAdapter(Settings(cache_dir=tmp_path / ".c"))
    sub = adapter.get_signal()
    assert sub.ok is False
    assert sub.score == 0.0
