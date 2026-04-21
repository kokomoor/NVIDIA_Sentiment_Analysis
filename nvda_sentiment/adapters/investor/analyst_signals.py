"""Analyst targets + recommendation signal via yfinance (§3.3, §6.3.3)."""

from __future__ import annotations

from datetime import date

import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

from ...config import NVIDIA_TICKER, Settings
from ...schemas import InvestorSubScore
from ...utils.cache import SimpleCache
from ...utils.logging import get_logger


logger = get_logger(__name__)
_TTL_SECONDS = 12 * 60 * 60


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class AnalystSignalsAdapter:
    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
        ticker: str = NVIDIA_TICKER,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "analyst_signals")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.ticker = ticker
        self.use_cache = True

    def get_signal(self) -> InvestorSubScore:
        cache_key = f"analyst:{self.ticker}:{date.today().isoformat()}"
        if self.use_cache:
            cached = self.cache.get_fresh(cache_key, _TTL_SECONDS)
            if cached is not None:
                try:
                    return InvestorSubScore.model_validate_json(cached)
                except Exception:
                    pass

        try:
            sub = self._compute()
        except Exception as exc:
            logger.warning("analyst_signals adapter failed: %s", exc)
            return InvestorSubScore(
                name="analyst_signal", score=0.0, ok=False, detail={}
            )

        if sub.ok:
            try:
                self.cache.set(cache_key, sub.model_dump_json())
            except Exception as exc:
                logger.warning("analyst_signals cache write failed: %s", exc)
        return sub

    def _compute(self) -> InvestorSubScore:
        if yf is None:
            raise RuntimeError("yfinance is not installed")
        info = yf.Ticker(self.ticker, session=self.session).info or {}
        cur = info.get("currentPrice")
        tmean = info.get("targetMeanPrice")
        thigh = info.get("targetHighPrice")
        tlow = info.get("targetLowPrice")
        rec = info.get("recommendationMean")
        n = info.get("numberOfAnalystOpinions") or 0

        # §6.3.3 guard: need ≥3 analysts and all relevant fields present.
        if n < 3 or any(x is None for x in (cur, tmean, thigh, tlow, rec)):
            return InvestorSubScore(
                name="analyst_signal",
                score=0.0,
                ok=False,
                detail={"numberOfAnalystOpinions": n},
            )

        cur_f = float(cur)
        tmean_f = float(tmean)
        thigh_f = float(thigh)
        tlow_f = float(tlow)
        rec_f = float(rec)

        if cur_f <= 0 or tmean_f <= 0:
            return InvestorSubScore(
                name="analyst_signal",
                score=0.0,
                ok=False,
                detail={"reason": "non-positive price/target"},
            )

        upside = (tmean_f - cur_f) / cur_f
        disp = (thigh_f - tlow_f) / tmean_f
        upside_score = _clip(upside / 0.40, -1.0, 1.0)
        damper = max(0.0, 1.0 - disp)
        rec_score = _clip((3.0 - rec_f) / 2.0, -1.0, 1.0)

        combined = 0.50 * (upside_score * damper) + 0.50 * rec_score

        return InvestorSubScore(
            name="analyst_signal",
            score=_clip(combined, -1.0, 1.0),
            ok=True,
            detail={
                "upside": round(upside, 4),
                "dispersion": round(disp, 4),
                "recMean": rec_f,
                "n": n,
            },
        )
