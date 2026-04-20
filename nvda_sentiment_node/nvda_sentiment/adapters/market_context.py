"""AAII + VIX small context layer (§15)."""

from __future__ import annotations

import csv
import io
import re
from typing import Tuple

import requests
from bs4 import BeautifulSoup

from ..config import AAII_URL, FRED_VIX_CSV_URL, Settings
from ..utils.cache import SimpleCache
from ..utils.logging import get_logger


logger = get_logger(__name__)
_CONTEXT_TTL_SECONDS = 24 * 60 * 60


def _clip(v: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


class MarketContextAdapter:
    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "context")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.use_cache = True

    def _get(self, url: str) -> str:
        if self.use_cache:
            cached = self.cache.get_fresh(url, _CONTEXT_TTL_SECONDS)
            if cached is not None:
                return cached
        response = self.session.get(
            url,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        body = response.text
        self.cache.set(url, body)
        return body

    # --- AAII ---------------------------------------------------------------

    def get_aaii_score(self) -> Tuple[float, bool]:
        """Return ``(score, ok)``.  Degrades gracefully on failure (§15.2)."""
        try:
            html = self._get(AAII_URL)
        except Exception as exc:
            logger.warning("AAII fetch failed: %s", exc)
            return (0.0, False)

        try:
            bullish, neutral, bearish = self._parse_aaii(html)
        except Exception as exc:
            logger.warning("AAII parse failed: %s", exc)
            return (0.0, False)

        aaii_raw = bullish - bearish
        return (_clip(aaii_raw / 100.0, -1.0, 1.0), True)

    def _parse_aaii(self, html: str) -> Tuple[float, float, float]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        # Very heuristic: look for "Bullish NN.N%" style patterns
        def _match(label: str) -> float:
            m = re.search(
                rf"{label}[^0-9%]{{0,40}}(\d{{1,3}}(?:\.\d+)?)\s*%", text, re.IGNORECASE
            )
            if not m:
                raise ValueError(f"{label} not found")
            return float(m.group(1))
        bullish = _match("Bullish")
        neutral = _match("Neutral")
        bearish = _match("Bearish")
        return (bullish, neutral, bearish)

    # --- VIX ----------------------------------------------------------------

    def get_vix_score(self) -> Tuple[float, bool]:
        """Return ``(score, ok)``.  Uses FRED CSV endpoint (§15.3)."""
        try:
            csv_text = self._get(FRED_VIX_CSV_URL)
        except Exception as exc:
            logger.warning("VIX (FRED) fetch failed: %s", exc)
            return (0.0, False)

        try:
            vix_close = self._parse_vix_csv(csv_text)
        except Exception as exc:
            logger.warning("VIX parse failed: %s", exc)
            return (0.0, False)

        if vix_close <= 15:
            score = 0.50
        elif vix_close <= 20:
            score = 0.25
        elif vix_close <= 25:
            score = 0.00
        elif vix_close <= 30:
            score = -0.25
        else:
            score = -0.50
        return (_clip(score, -1.0, 1.0), True)

    def _parse_vix_csv(self, text: str) -> float:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise ValueError("empty CSV")
        # Identify the VIXCLS value column
        header = rows[0]
        try:
            value_idx = next(
                i for i, c in enumerate(header)
                if c and c.strip().lower() in ("vixcls", "value")
            )
        except StopIteration:
            value_idx = 1

        last_valid: float | None = None
        for row in rows[1:]:
            if len(row) <= value_idx:
                continue
            raw = (row[value_idx] or "").strip()
            if not raw or raw == ".":
                continue
            try:
                last_valid = float(raw)
            except ValueError:
                continue
        if last_valid is None:
            raise ValueError("no numeric VIX values found")
        return last_valid

    # --- Combined -----------------------------------------------------------

    def get_combined_score(self) -> Tuple[float, bool]:
        """Return ``(combined_score, any_source_ok)``."""
        aaii, aaii_ok = self.get_aaii_score()
        vix, vix_ok = self.get_vix_score()
        combined = 0.6 * aaii + 0.4 * vix
        return (_clip(combined, -1.0, 1.0), (aaii_ok or vix_ok))
