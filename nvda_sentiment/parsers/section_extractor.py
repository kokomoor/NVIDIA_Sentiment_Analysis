"""Heuristic section extraction (§17).

A crude but reliable heading-based splitter. By design we do not try to be
perfect — we only need text segments clean enough for sentence scoring.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..schemas import SourceDocument


# --------------------------------------------------------------------------- #
# Heading patterns
# --------------------------------------------------------------------------- #

_MDA_PATTERNS = [
    r"management'?s discussion and analysis",
    r"management discussion and analysis",
    r"md&a",
    r"item\s+[27]\.?\s+management'?s discussion and analysis",
]

_MDA_STOP = [
    r"quantitative and qualitative disclosures",
    r"controls and procedures",
    r"risk factors",
    r"financial statements",
    r"legal proceedings",
    r"item\s+[3-9]\.?",
    r"item\s+1[0-9]\.?",
]

_RISK_PATTERNS = [
    r"item\s+1a\.?\s+risk factors",
    r"risk factors",
]

_RISK_STOP = [
    r"unresolved staff comments",
    r"properties",
    r"legal proceedings",
    r"mine safety disclosures",
    r"management'?s discussion and analysis",
    r"item\s+[1-9]b\.?",
    r"item\s+[2-9]\.?",
]

_GUIDANCE_KEYWORDS = [
    "outlook",
    "guidance",
    "we expect",
    "we believe",
    "looking ahead",
    "for the next quarter",
    "for fiscal",
    "our outlook",
    "revenue outlook",
    "expected",
]

_QA_MARKERS = [
    "question-and-answer",
    "question and answer",
    "q&a",
    "q & a",
    "questions and answers",
    "begin the question",
]

# Used as a secondary marker once we think Q&A has started
_QA_SPEAKER_MARKERS = ["operator", "analyst"]


def _find_first_match(text: str, patterns: List[str]) -> Optional[int]:
    best = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m is None:
            continue
        if best is None or m.start() < best:
            best = m.start()
    return best


def _find_first_match_after(text: str, patterns: List[str], start: int) -> Optional[int]:
    best = None
    for pat in patterns:
        m = re.search(pat, text[start:], re.IGNORECASE)
        if m is None:
            continue
        pos = start + m.start()
        if best is None or pos < best:
            best = pos
    return best


def _extract_between(text: str, start_patterns: List[str], stop_patterns: List[str]) -> str:
    start = _find_first_match(text, start_patterns)
    if start is None:
        return ""
    # advance past the heading itself
    # find end of line after the heading
    newline = text.find("\n", start)
    body_start = newline + 1 if newline != -1 else start
    stop = _find_first_match_after(text, stop_patterns, body_start)
    segment = text[body_start:stop] if stop is not None else text[body_start:]
    return segment.strip()


def _extract_guidance(text: str) -> str:
    """Collect paragraphs that contain guidance-flavored keywords."""
    paragraphs = re.split(r"\n{2,}", text)
    hits: List[str] = []
    for para in paragraphs:
        low = para.lower()
        if any(kw in low for kw in _GUIDANCE_KEYWORDS):
            cleaned = para.strip()
            if len(cleaned) >= 40:
                hits.append(cleaned)
    return "\n\n".join(hits).strip()


def _transcript_split(text: str) -> Dict[str, str]:
    """Split transcript into prepared_remarks and qa (§17.5)."""
    low = text.lower()
    split_pos = None
    for marker in _QA_MARKERS:
        idx = low.find(marker)
        if idx != -1 and (split_pos is None or idx < split_pos):
            split_pos = idx

    if split_pos is None:
        # fall back: find the first "operator" line after a reasonable offset
        mid = len(text) // 3
        for marker in _QA_SPEAKER_MARKERS:
            idx = low.find(marker, mid)
            if idx != -1 and (split_pos is None or idx < split_pos):
                split_pos = idx

    result: Dict[str, str] = {}
    if split_pos is None:
        prepared = text.strip()
        qa = ""
    else:
        prepared = text[:split_pos].strip()
        qa = text[split_pos:].strip()
    if prepared:
        result["prepared_remarks"] = prepared
    if qa:
        result["qa"] = qa
    return result


def _extract_press_release(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return result

    # headline + summary: first ~3 substantive paragraphs
    summary_chunks: List[str] = []
    for p in paragraphs:
        if len(summary_chunks) >= 3:
            break
        if len(p) < 40:
            continue
        summary_chunks.append(p)
    if summary_chunks:
        result["headline_and_summary"] = "\n\n".join(summary_chunks)

    # financial highlights: paragraphs mentioning revenue / gaap / margin
    fin_kws = ("revenue", "gaap", "non-gaap", "gross margin", "operating income",
               "net income", "earnings per share", "eps", "cash flow")
    fin_hits = [p for p in paragraphs if any(k in p.lower() for k in fin_kws)]
    if fin_hits:
        result["financial_highlights"] = "\n\n".join(fin_hits[:8])

    guidance = _extract_guidance(text)
    if guidance:
        result["outlook_guidance"] = guidance
    return result


def _extract_eight_k(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    # earnings_summary = first several substantive paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    summary: List[str] = []
    for p in paragraphs:
        if len(summary) >= 5:
            break
        if len(p) >= 40:
            summary.append(p)
    if summary:
        result["earnings_summary"] = "\n\n".join(summary)

    guidance = _extract_guidance(text)
    if guidance:
        result["outlook_guidance"] = guidance
    return result


def _extract_filing(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    mda = _extract_between(text, _MDA_PATTERNS, _MDA_STOP)
    if mda:
        result["mda"] = mda
    risk = _extract_between(text, _RISK_PATTERNS, _RISK_STOP)
    if risk:
        result["risk_factors"] = risk
    guidance = _extract_guidance(text)
    if guidance:
        result["outlook_guidance"] = guidance
    return result


class SectionExtractor:
    """Dispatch extraction based on document type (§17.7)."""

    def extract_sections(self, doc: SourceDocument) -> Dict[str, str]:
        text = doc.clean_text or ""
        if not text:
            return {}

        if doc.source_type in ("10-K", "10-Q"):
            return _extract_filing(text)
        if doc.source_type == "8-K":
            return _extract_eight_k(text)
        if doc.source_type == "press_release":
            return _extract_press_release(text)
        if doc.source_type == "transcript":
            return _transcript_split(text)
        if doc.source_type == "cfo_commentary":
            body = text.strip()
            return {"cfo_commentary_body": body} if body else {}
        return {}
