"""HTML -> clean text (§16)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from ..utils.text import normalize_unicode_punctuation, normalize_whitespace


def html_to_clean_text(html: str) -> str:
    """Convert HTML into normalized plain text.

    Not perfect — just clean enough for heading-based section extraction.
    """
    if not html:
        return ""

    # lxml if available; fall back to html.parser if lxml not installed at runtime
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - lxml fallback
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = normalize_unicode_punctuation(text)
    text = normalize_whitespace(text)
    return text
