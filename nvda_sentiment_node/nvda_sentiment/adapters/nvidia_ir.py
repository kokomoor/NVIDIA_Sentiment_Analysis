"""NVIDIA Investor Relations adapter (§14)."""

from __future__ import annotations

import re
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..config import NVIDIA_QUARTERLY_RESULTS_URL, Settings
from ..schemas import SourceDocument, SourceType
from ..utils.cache import SimpleCache
from ..utils.logging import get_logger


logger = get_logger(__name__)
_IR_TTL_SECONDS = 24 * 60 * 60


_QUARTER_RE = re.compile(r"(fourth|third|second|first)\s+quarter.*?(fiscal\s+\d{4}|fy\s*\d{2,4})", re.IGNORECASE)


def _detect_source_type(link_text: str) -> SourceType | None:
    low = (link_text or "").lower()
    if "transcript" in low:
        return "transcript"
    if "cfo commentary" in low or "cfo-commentary" in low:
        return "cfo_commentary"
    if "press release" in low or "financial results" in low:
        return "press_release"
    return None


class NvidiaIRAdapter:
    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "ir")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": settings.sec_user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        self.use_cache = True

    def _get(self, url: str, *, permanent: bool = False) -> str:
        if self.use_cache:
            if permanent:
                cached = self.cache.get_permanent(url)
                if cached is not None:
                    return cached
            else:
                cached = self.cache.get_fresh(url, _IR_TTL_SECONDS)
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

    def fetch_document_html(self, doc: SourceDocument) -> str:
        # IR transcripts/press releases are effectively immutable once
        # published; treat them as permanent-cache for the MVP.
        return self._get(doc.url, permanent=True)

    def get_quarterly_results_documents(self, lookback_quarters: int) -> List[SourceDocument]:
        try:
            html = self._get(NVIDIA_QUARTERLY_RESULTS_URL)
        except Exception as exc:
            logger.warning("NVIDIA IR page fetch failed: %s", exc)
            return []
        soup = BeautifulSoup(html, "html.parser")
        base_url = NVIDIA_QUARTERLY_RESULTS_URL

        docs: List[SourceDocument] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a"):
            text = (anchor.get_text() or "").strip()
            href = anchor.get("href")
            if not text or not href:
                continue
            source_type = _detect_source_type(text)
            if source_type is None:
                continue
            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            docs.append(
                SourceDocument(
                    source_type=source_type,
                    title=text[:200],
                    url=absolute,
                    filed_at="",  # not reliably on the page; left blank
                )
            )

        # Keep roughly lookback_quarters * 3 docs (one of each type per qtr)
        # but we cannot reliably order them without dates, so return as-is.
        return docs[: max(3, lookback_quarters * 3)]
