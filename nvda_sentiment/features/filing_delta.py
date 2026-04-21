"""Filing delta — tone-delta + risk-delta (§24).

Implementation note: the spec's own ``nvda_fiscal_bucket`` pseudocode is
mildly buggy (it over-advances the fiscal year in Nov/Dec and in January).
This corrected version follows NVIDIA's actual convention: ``FYnn`` ends in
late January of calendar year ``nn`` (e.g. FY26 ends Jan 2026).

Sanity:
- Feb 2025 -> FY26Q1     (FY26 ends Jan 2026)
- Feb 2026 -> FY27Q1
- Nov 2026 -> FY27Q4     (NVIDIA's fiscal Q4 ends in Jan 2027)
- Jan 2027 -> FY27Q4
- Jan 2026 -> FY26Q4
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from ..schemas import DocumentScore
from ..utils.dates import parse_date


def nvda_fiscal_bucket(filing_date: date) -> str:
    """Return a label like ``FY27Q1`` for a given filing date."""
    m, y = filing_date.month, filing_date.year
    if m in (2, 3, 4):
        return f"FY{(y + 1) % 100:02d}Q1"
    if m in (5, 6, 7):
        return f"FY{(y + 1) % 100:02d}Q2"
    if m in (8, 9, 10):
        return f"FY{(y + 1) % 100:02d}Q3"
    if m in (11, 12):
        return f"FY{(y + 1) % 100:02d}Q4"
    # January: same FY label as its calendar year (FY ends in January)
    return f"FY{y % 100:02d}Q4"


def _bucket_doc(doc: DocumentScore) -> str | None:
    parsed = parse_date(doc.filed_at)
    if parsed is None:
        return None
    return nvda_fiscal_bucket(parsed)


_FY_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _bucket_sort_key(bucket: str) -> Tuple[int, int]:
    # Format is FYnnQx where nn is last two digits of year
    try:
        year_part = int(bucket[2:4])
        q_part = _FY_ORDER[bucket[4:]]
        return (year_part, q_part)
    except (ValueError, KeyError, IndexError):
        return (-1, -1)


def _prior_yoy_bucket(bucket: str) -> str | None:
    try:
        yy = int(bucket[2:4])
        q = bucket[4:]
        prior_yy = (yy - 1) % 100
        return f"FY{prior_yy:02d}{q}"
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Weighted-mean of document scores in a bucket (§24.3 refers to §26)
# --------------------------------------------------------------------------- #

def _weighted_bucket_score(docs: List[DocumentScore]) -> float:
    from ..config import DOCUMENT_TYPE_WEIGHTS
    from ..scorers.composite import clip, renormalize_weights

    if not docs:
        return 0.0
    type_scores: Dict[str, List[float]] = defaultdict(list)
    for d in docs:
        type_scores[d.source_type].append(d.final_score)
    type_means = {t: sum(v) / len(v) for t, v in type_scores.items()}
    weights = renormalize_weights(DOCUMENT_TYPE_WEIGHTS, set(type_means.keys()))
    if not weights:
        return 0.0
    return clip(sum(w * type_means[t] for t, w in weights.items()), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Tone delta (§24.3)
# --------------------------------------------------------------------------- #

def compute_tone_delta(
    document_scores: List[DocumentScore],
    warnings: list[str] | None = None,
) -> Tuple[float, bool]:
    """Return ``(tone_delta, yoy_matched)``.

    Prefers year-over-year comparison for the latest bucket (§24.6); falls
    back to the immediately prior bucket; returns ``(0.0, False)`` when no
    prior comparable data exists.
    """
    buckets: Dict[str, List[DocumentScore]] = defaultdict(list)
    for ds in document_scores:
        b = _bucket_doc(ds)
        if b is not None:
            buckets[b].append(ds)

    if len(buckets) < 2:
        if warnings is not None:
            warnings.append("No prior comparable period found for filing delta")
        return (0.0, False)

    ordered = sorted(buckets.keys(), key=_bucket_sort_key)
    current = ordered[-1]
    current_score = _weighted_bucket_score(buckets[current])

    # YoY preference
    yoy = _prior_yoy_bucket(current)
    if yoy is not None and yoy in buckets:
        prior_score = _weighted_bucket_score(buckets[yoy])
        return (current_score - prior_score, True)

    # Sequential fallback
    prior = ordered[-2]
    prior_score = _weighted_bucket_score(buckets[prior])
    return (current_score - prior_score, False)


# --------------------------------------------------------------------------- #
# Risk delta (§24.4)
# --------------------------------------------------------------------------- #

def _risk_score_for_bucket(docs: List[DocumentScore]) -> Tuple[float, int]:
    values: List[float] = []
    for d in docs:
        if d.source_type not in ("10-K", "10-Q"):
            continue
        for s in d.section_scores:
            if s.section_name == "risk_factors":
                values.append(s.final_score)
    if not values:
        return (0.0, 0)
    return (sum(values) / len(values), len(values))


def compute_risk_delta(
    document_scores: List[DocumentScore],
    warnings: list[str] | None = None,
) -> float:
    buckets: Dict[str, List[DocumentScore]] = defaultdict(list)
    for ds in document_scores:
        b = _bucket_doc(ds)
        if b is not None:
            buckets[b].append(ds)

    if len(buckets) < 2:
        return 0.0

    ordered = sorted(buckets.keys(), key=_bucket_sort_key)
    current = ordered[-1]
    current_score, current_n = _risk_score_for_bucket(buckets[current])

    yoy = _prior_yoy_bucket(current)
    prior_docs = None
    if yoy is not None and yoy in buckets:
        prior_docs = buckets[yoy]
    if prior_docs is None:
        prior_docs = buckets[ordered[-2]]
    prior_score, prior_n = _risk_score_for_bucket(prior_docs)

    if current_n == 0 or prior_n == 0:
        return 0.0
    return current_score - prior_score


# --------------------------------------------------------------------------- #
# Combined (§24.5)
# --------------------------------------------------------------------------- #

def compute_filing_delta(
    document_scores: List[DocumentScore],
    warnings: list[str] | None = None,
) -> Tuple[float, float, float, bool]:
    """Return ``(filing_delta, tone_delta, risk_delta, yoy_matched)``."""
    from ..scorers.composite import clip

    tone_delta, yoy = compute_tone_delta(document_scores, warnings)
    risk_delta = compute_risk_delta(document_scores, warnings)
    combined = clip(0.70 * tone_delta + 0.30 * risk_delta, -1.0, 1.0)
    return combined, tone_delta, risk_delta, yoy
