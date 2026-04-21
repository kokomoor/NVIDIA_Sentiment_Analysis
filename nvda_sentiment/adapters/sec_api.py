"""SEC EDGAR adapter (§13).

NOTE ON PAGINATION (§13.8): The base submissions JSON contains only the most
recent ~1000 filings.  ``lookback_quarters`` is capped at 8, which fits easily
inside the base file for NVIDIA.  This adapter intentionally does NOT follow
``filings.files`` pagination.  If future work requires deeper lookback, start
here.
"""

from __future__ import annotations

import json
from typing import Dict, List

import requests

from ..config import (
    EIGHT_K_EARNINGS_KEYWORDS,
    NVIDIA_CIK,
    NVIDIA_CIK_PADDED,
    SEC_ARCHIVES_BASE,
    SEC_SUBMISSIONS_URL,
    Settings,
)
from ..schemas import SourceDocument, SourceType
from ..utils.cache import SimpleCache
from ..utils.logging import get_logger
from ..utils.rate_limiter import RateLimiter


logger = get_logger(__name__)

# TTLs
_SUBMISSIONS_TTL_SECONDS = 24 * 60 * 60  # 24h


def _form_to_source_type(form: str) -> SourceType | None:
    f = (form or "").upper()
    if f == "10-K":
        return "10-K"
    if f == "10-Q":
        return "10-Q"
    if f == "8-K":
        return "8-K"
    return None


def _is_earnings_8k(description: str, primary_doc: str) -> bool:
    blob = f"{description or ''} {primary_doc or ''}".lower()
    return any(kw in blob for kw in EIGHT_K_EARNINGS_KEYWORDS)


class SECAdapter:
    def __init__(
        self,
        settings: Settings,
        cache: SimpleCache | None = None,
        session: requests.Session | None = None,
    ):
        self.settings = settings
        self.cache = cache or SimpleCache(settings.cache_dir / "sec")
        self.session = session or requests.Session()
        # NOTE: Do NOT pin a Host header on the session — submissions JSON
        # lives on data.sec.gov but filing HTML lives on www.sec.gov/Archives.
        # requests derives Host from the URL automatically.
        self.session.headers.update(
            {
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.rate_limiter = RateLimiter(settings.sec_max_requests_per_second)
        # Toggled per-request by NVDASentimentNode.run() to honor
        # SentimentRequest.use_cache (§3.1).  When False, skip cache reads but
        # still write, so a forced-refresh run repopulates the cache.
        self.use_cache = True

    # --- HTTP ---------------------------------------------------------------

    def _get(self, url: str, *, ttl_seconds: int | None, permanent: bool = False) -> str:
        """Fetch ``url`` with caching + rate limiting."""
        if self.use_cache:
            if permanent:
                cached = self.cache.get_permanent(url)
                if cached is not None:
                    return cached
            elif ttl_seconds is not None:
                cached = self.cache.get_fresh(url, ttl_seconds)
                if cached is not None:
                    return cached

        self.rate_limiter.wait()
        # Different hosts need different Host header; requests will set it.
        headers = {"User-Agent": self.settings.sec_user_agent}
        response = self.session.get(
            url,
            headers=headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        body = response.text
        self.cache.set(url, body)
        return body

    # --- Submissions --------------------------------------------------------

    def get_recent_filings(self) -> List[Dict]:
        raw = self._get(SEC_SUBMISSIONS_URL, ttl_seconds=_SUBMISSIONS_TTL_SECONDS)
        payload = json.loads(raw)
        recent = payload.get("filings", {}).get("recent", {})
        if not recent:
            return []

        keys = (
            "accessionNumber",
            "filingDate",
            "form",
            "primaryDocument",
            "primaryDocDescription",
        )
        rows: List[Dict] = []
        size = len(recent.get("accessionNumber", []))
        for i in range(size):
            rows.append({k: recent.get(k, [None] * size)[i] for k in keys})
        return rows

    # --- Filing selection ---------------------------------------------------

    def _build_document_url(self, accession_no: str, primary_doc: str) -> str:
        acc_nodash = (accession_no or "").replace("-", "")
        cik_no_zero = NVIDIA_CIK_PADDED.lstrip("0") or NVIDIA_CIK
        return f"{SEC_ARCHIVES_BASE}/{cik_no_zero}/{acc_nodash}/{primary_doc}"

    def get_relevant_filings(self, lookback_quarters: int) -> List[SourceDocument]:
        rows = self.get_recent_filings()
        tenk: List[Dict] = []
        tenq: List[Dict] = []
        eightk: List[Dict] = []
        for row in rows:
            form = row.get("form") or ""
            if form == "10-K":
                tenk.append(row)
            elif form == "10-Q":
                tenq.append(row)
            elif form == "8-K":
                if _is_earnings_8k(row.get("primaryDocDescription") or "", row.get("primaryDocument") or ""):
                    eightk.append(row)

        # Sort by filingDate DESC (strings in ISO format compare correctly)
        tenk.sort(key=lambda r: r.get("filingDate") or "", reverse=True)
        tenq.sort(key=lambda r: r.get("filingDate") or "", reverse=True)
        eightk.sort(key=lambda r: r.get("filingDate") or "", reverse=True)

        # §13.5: latest 1 x 10-K, N x 10-Q, earnings 8-Ks in the same window
        selected: List[Dict] = []
        if tenk:
            selected.append(tenk[0])
        selected.extend(tenq[: max(1, lookback_quarters)])
        selected.extend(eightk[: max(1, lookback_quarters)])

        docs: List[SourceDocument] = []
        for row in selected:
            source_type = _form_to_source_type(row.get("form") or "")
            if source_type is None:
                continue
            accession = row.get("accessionNumber")
            primary = row.get("primaryDocument")
            if not accession or not primary:
                continue
            url = self._build_document_url(accession, primary)
            title = row.get("primaryDocDescription") or f"{source_type} {row.get('filingDate', '')}"
            docs.append(
                SourceDocument(
                    source_type=source_type,
                    title=str(title),
                    url=url,
                    filed_at=str(row.get("filingDate") or ""),
                )
            )
        return docs

    # --- Fetch filing HTML --------------------------------------------------

    def fetch_filing_html(self, doc: SourceDocument) -> str:
        # SEC filings are immutable once filed (§30.3): permanent cache tier.
        return self._get(doc.url, ttl_seconds=None, permanent=True)
