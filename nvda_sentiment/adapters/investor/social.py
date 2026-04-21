"""StockTwits + Reddit social sentiment (§3.4, §6.3.4)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests

from ...config import NVIDIA_TICKER, Settings
from ...schemas import InvestorSubScore
from ...scorers.finbert_scorer import FinBERTScorer
from ...scorers.lexicon import LexiconScorer
from ...utils.cache import SimpleCache
from ...utils.logging import get_logger


logger = get_logger(__name__)
_TTL_SECONDS = 60 * 60   # 1 hour (§3.4)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _today_hour_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")


class SocialAdapter:
    STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{t}.json"
    REDDIT_URLS = (
        "https://www.reddit.com/r/wallstreetbets/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
        "https://www.reddit.com/r/stocks/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
        "https://www.reddit.com/r/investing/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
    )

    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
        ticker: str = NVIDIA_TICKER,
        finbert: FinBERTScorer | None = None,
        lexicon: LexiconScorer | None = None,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "social")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.sec_user_agent})
        self.ticker = ticker
        self.finbert = finbert if finbert is not None else FinBERTScorer(settings)
        self.lexicon = lexicon if lexicon is not None else LexiconScorer()
        self.use_cache = True

    # --- public -------------------------------------------------------------

    def get_signal(self) -> InvestorSubScore:
        st_score, st_ok = self._stocktwits()
        rd_score, rd_ok = self._reddit()

        if not (st_ok or rd_ok):
            return InvestorSubScore(
                name="social", score=0.0, ok=False, detail={}
            )

        if st_ok and rd_ok:
            score = 0.60 * st_score + 0.40 * rd_score
        elif st_ok:
            score = st_score
        else:
            score = rd_score

        return InvestorSubScore(
            name="social",
            score=_clip(score, -1.0, 1.0),
            ok=True,
            detail={
                "stocktwits": round(st_score, 4) if st_ok else None,
                "reddit": round(rd_score, 4) if rd_ok else None,
            },
        )

    # --- StockTwits ---------------------------------------------------------

    def _stocktwits(self) -> Tuple[float, bool]:
        url = self.STOCKTWITS_URL.format(t=self.ticker)
        text = self._fetch(url, f"st:{self.ticker}:{_today_hour_iso()}")
        if text is None:
            return (0.0, False)
        try:
            data = json.loads(text)
        except Exception as exc:
            logger.warning("stocktwits parse failed: %s", exc)
            return (0.0, False)

        bullish = 0
        bearish = 0
        for msg in data.get("messages", []) or []:
            entities = (msg or {}).get("entities") or {}
            sentiment = (entities.get("sentiment") or {}) if isinstance(entities, dict) else {}
            basic = (sentiment or {}).get("basic") if isinstance(sentiment, dict) else None
            if basic == "Bullish":
                bullish += 1
            elif basic == "Bearish":
                bearish += 1

        tagged = bullish + bearish
        if tagged < 5:
            return (0.0, False)
        return ((bullish - bearish) / tagged, True)

    # --- Reddit -------------------------------------------------------------

    def _reddit(self) -> Tuple[float, bool]:
        # Deterministic iteration: sort posts by id after dedupe.
        posts_by_id: Dict[str, Dict] = {}
        any_fetched = False
        for url_tmpl in self.REDDIT_URLS:
            url = url_tmpl.format(t=self.ticker)
            text = self._fetch(url, f"rd:{self.ticker}:{url_tmpl.split('/r/')[1].split('/')[0]}:{_today_hour_iso()}")
            if text is None:
                continue
            any_fetched = True
            try:
                data = json.loads(text)
            except Exception as exc:
                logger.warning("reddit parse failed (%s): %s", url, exc)
                continue
            children = ((data or {}).get("data") or {}).get("children") or []
            for c in children:
                d = (c or {}).get("data") or {}
                post_id = d.get("id")
                if not post_id or post_id in posts_by_id:
                    continue
                posts_by_id[post_id] = d

        if not any_fetched:
            return (0.0, False)

        # §3.4 — need ≥3 posts to mark ok.
        posts: List[Dict] = [posts_by_id[pid] for pid in sorted(posts_by_id.keys())]
        if len(posts) < 3:
            return (0.0, False)

        weighted_sum = 0.0
        total_weight = 0.0
        for post in posts:
            title = post.get("title") or ""
            selftext = post.get("selftext") or ""
            text = f"{title} {selftext}".strip()
            if not text:
                continue

            finbert_mean, _ = self.finbert.score_text(text)
            lex = self.lexicon.score_text(text)
            post_score = 0.5 * float(finbert_mean) + 0.5 * float(lex["lexicon_score"])

            raw_weight = post.get("score")
            try:
                up = max(int(raw_weight or 0), 0)
            except (TypeError, ValueError):
                up = 0
            weight = math.log1p(up)
            weighted_sum += post_score * weight
            total_weight += weight

        denominator = max(total_weight, 1.0)
        reddit_score = weighted_sum / denominator
        return (_clip(reddit_score, -1.0, 1.0), True)

    # --- fetch helper -------------------------------------------------------

    def _fetch(self, url: str, cache_key: str) -> str | None:
        if self.use_cache:
            cached = self.cache.get_fresh(cache_key, _TTL_SECONDS)
            if cached is not None:
                return cached
        try:
            response = self.session.get(
                url, timeout=self.settings.request_timeout_seconds
            )
            response.raise_for_status()
            body = response.text
        except Exception as exc:
            logger.warning("social fetch failed for %s: %s", url, exc)
            return None
        try:
            self.cache.set(cache_key, body)
        except Exception as exc:
            logger.warning("social cache write failed: %s", exc)
        return body
