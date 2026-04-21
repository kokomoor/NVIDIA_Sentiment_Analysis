"""Short interest (% of float, days-to-cover) via yfinance (§3.2, §6.3.2)."""

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
_TTL_SECONDS = 24 * 60 * 60


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _bucket_pct(pct: float) -> float:
    if pct <= 0.01:
        return 0.20
    if pct <= 0.02:
        return 0.10
    if pct <= 0.04:
        return 0.00
    if pct <= 0.07:
        return -0.15
    return -0.30


def _bucket_ratio(ratio: float) -> float:
    if ratio <= 1.0:
        return 0.20
    if ratio <= 2.0:
        return 0.00
    if ratio <= 3.5:
        return -0.10
    return -0.25


class ShortInterestAdapter:
    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
        ticker: str = NVIDIA_TICKER,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "short_interest")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.ticker = ticker
        self.use_cache = True

    def get_signal(self) -> InvestorSubScore:
        cache_key = f"short:{self.ticker}:{date.today().isoformat()}"
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
            logger.warning("short_interest adapter failed: %s", exc)
            return InvestorSubScore(
                name="short_interest", score=0.0, ok=False, detail={}
            )

        if sub.ok:
            try:
                self.cache.set(cache_key, sub.model_dump_json())
            except Exception as exc:
                logger.warning("short_interest cache write failed: %s", exc)
        return sub

    def _compute(self) -> InvestorSubScore:
        if yf is None:
            raise RuntimeError("yfinance is not installed")
        info = yf.Ticker(self.ticker, session=self.session).info or {}
        short_pct = info.get("shortPercentOfFloat")
        short_rat = info.get("shortRatio")

        if short_pct is None and short_rat is None:
            return InvestorSubScore(
                name="short_interest", score=0.0, ok=False, detail={}
            )

        pct_score = _bucket_pct(float(short_pct)) if short_pct is not None else 0.0
        rat_score = _bucket_ratio(float(short_rat)) if short_rat is not None else 0.0

        if short_pct is not None and short_rat is not None:
            combined = 0.60 * pct_score + 0.40 * rat_score
        elif short_pct is not None:
            combined = pct_score
        else:
            combined = rat_score

        return InvestorSubScore(
            name="short_interest",
            score=_clip(combined, -1.0, 1.0),
            ok=True,
            detail={
                "shortPercentOfFloat": short_pct,
                "shortRatio": short_rat,
            },
        )
