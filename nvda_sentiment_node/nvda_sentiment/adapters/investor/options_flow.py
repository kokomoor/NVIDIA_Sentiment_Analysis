"""Put/call ratio + ATM IV z-score via yfinance (§3.1, §6.3.1)."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Tuple

import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - yfinance is a declared dependency
    yf = None  # type: ignore

from ...config import NVIDIA_TICKER, Settings
from ...schemas import InvestorSubScore
from ...utils.cache import SimpleCache
from ...utils.logging import get_logger


logger = get_logger(__name__)
_TTL_SECONDS = 4 * 60 * 60


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _days_out(iso_date: str, today: date) -> int:
    return (datetime.strptime(iso_date, "%Y-%m-%d").date() - today).days


def _bucket_pcr(pcr: float) -> float:
    if pcr <= 0.60:
        return 0.50
    if pcr <= 0.85:
        return 0.25
    if pcr <= 1.15:
        return 0.00
    if pcr <= 1.50:
        return -0.25
    return -0.50


def _bucket_iv(iv_z: float) -> float:
    if iv_z <= -1.0:
        return 0.20
    if iv_z <= 0.5:
        return 0.00
    if iv_z <= 1.5:
        return -0.15
    return -0.30


class OptionsFlowAdapter:
    """Compute the options-flow investor sub-signal."""

    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
        ticker: str = NVIDIA_TICKER,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "options_flow")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.ticker = ticker
        self.use_cache = True

    # --- public -------------------------------------------------------------

    def get_signal(self) -> InvestorSubScore:
        today = date.today()
        cache_key = f"opts:{self.ticker}:{today.isoformat()}"
        if self.use_cache:
            cached = self.cache.get_fresh(cache_key, _TTL_SECONDS)
            if cached is not None:
                try:
                    return InvestorSubScore.model_validate_json(cached)
                except Exception:
                    pass  # fall through and recompute

        try:
            sub = self._compute(today)
        except Exception as exc:
            logger.warning("options_flow adapter failed: %s", exc)
            return InvestorSubScore(
                name="options_flow", score=0.0, ok=False, detail={}
            )

        if sub.ok:
            try:
                self.cache.set(cache_key, sub.model_dump_json())
            except Exception as exc:
                logger.warning("options_flow cache write failed: %s", exc)
        return sub

    # --- internal -----------------------------------------------------------

    def _compute(self, today: date) -> InvestorSubScore:
        if yf is None:
            raise RuntimeError("yfinance is not installed")

        ticker_obj = yf.Ticker(self.ticker, session=self.session)
        expiries: Tuple[str, ...] = tuple(ticker_obj.options or ())
        if not expiries:
            return InvestorSubScore(
                name="options_flow", score=0.0, ok=False, detail={"reason": "no expiries"}
            )

        relevant = [e for e in expiries if _days_out(e, today) >= 7][:3]
        if not relevant:
            return InvestorSubScore(
                name="options_flow", score=0.0, ok=False, detail={"reason": "no expiries >= 7d"}
            )

        pcr = self._put_call_ratio(ticker_obj, relevant)
        iv_z = self._atm_iv_zscore(ticker_obj, expiries, today)

        pcr_score = _bucket_pcr(pcr)
        iv_score = _bucket_iv(iv_z)
        combined = 0.70 * pcr_score + 0.30 * iv_score

        return InvestorSubScore(
            name="options_flow",
            score=_clip(combined, -1.0, 1.0),
            ok=True,
            detail={
                "pcr": round(pcr, 4),
                "iv_z": round(iv_z, 4),
                "expiries_used": list(relevant),
            },
        )

    @staticmethod
    def _put_call_ratio(ticker_obj, expiries: List[str]) -> float:
        total_put_vol = 0.0
        total_call_vol = 0.0
        for exp in expiries:
            chain = ticker_obj.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            total_call_vol += float(_safe_sum(calls["volume"]))
            total_put_vol += float(_safe_sum(puts["volume"]))
        return total_put_vol / max(total_call_vol, 1.0)

    @staticmethod
    def _atm_iv_zscore(ticker_obj, expiries: Tuple[str, ...], today: date) -> float:
        # Nearest expiry ≥ 21 days out.
        chosen = next((e for e in expiries if _days_out(e, today) >= 21), None)
        if chosen is None:
            return 0.0
        chain = ticker_obj.option_chain(chosen)
        calls = chain.calls
        puts = chain.puts
        # Current spot via fast_info (deterministic): fall back to last lastPrice.
        spot = _safe_spot(ticker_obj)
        if spot is None:
            return 0.0
        # Nearest-to-spot call and put.
        call_iv = _nearest_iv(calls, spot)
        put_iv = _nearest_iv(puts, spot)
        if call_iv is None and put_iv is None:
            return 0.0
        atm_iv = (
            (call_iv + put_iv) / 2.0 if call_iv is not None and put_iv is not None
            else (call_iv if call_iv is not None else put_iv)
        )
        # Z-score across the concat IV distribution of this chain.
        all_ivs = [
            float(v) for v in list(calls["impliedVolatility"]) + list(puts["impliedVolatility"])
            if v is not None and not _is_nan(v)
        ]
        if len(all_ivs) < 2:
            return 0.0
        mean = sum(all_ivs) / len(all_ivs)
        var = sum((v - mean) ** 2 for v in all_ivs) / len(all_ivs)
        std = var ** 0.5
        if std <= 0:
            return 0.0
        return (atm_iv - mean) / std


def _safe_sum(series) -> float:
    total = 0.0
    for v in series:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if _is_nan(fv):
            continue
        total += fv
    return total


def _safe_spot(ticker_obj) -> float | None:
    # Prefer fast_info.last_price; fall back to info['currentPrice'].
    try:
        fi = ticker_obj.fast_info
        price = getattr(fi, "last_price", None)
        if price is None and hasattr(fi, "__getitem__"):
            try:
                price = fi["last_price"]
            except Exception:
                price = None
        if price is not None:
            return float(price)
    except Exception:
        pass
    try:
        info = ticker_obj.info
        price = info.get("currentPrice") if isinstance(info, dict) else None
        if price is not None:
            return float(price)
    except Exception:
        pass
    return None


def _nearest_iv(df, spot: float) -> float | None:
    best_iv: float | None = None
    best_distance = float("inf")
    strikes = list(df["strike"])
    ivs = list(df["impliedVolatility"])
    for strike, iv in zip(strikes, ivs):
        if strike is None or iv is None:
            continue
        try:
            s = float(strike)
            i = float(iv)
        except (TypeError, ValueError):
            continue
        if _is_nan(s) or _is_nan(i):
            continue
        d = abs(s - spot)
        if d < best_distance:
            best_distance = d
            best_iv = i
    return best_iv


def _is_nan(value: float) -> bool:
    return value != value  # NaN self-inequality
