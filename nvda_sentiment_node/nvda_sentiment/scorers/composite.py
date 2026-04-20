"""Document- and node-level aggregation (§§22-23, 25-27)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import (
    CFO_COMMENTARY_SECTION_WEIGHTS,
    DOCUMENT_TYPE_WEIGHTS,
    EIGHT_K_SECTION_WEIGHTS,
    FILING_SECTION_WEIGHTS,
    PRESS_RELEASE_SECTION_WEIGHTS,
    TRANSCRIPT_SECTION_WEIGHTS,
)
from ..schemas import DocumentScore, SectionScore, SourceDocument


def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def renormalize_weights(weights: Dict[str, float], available_keys: set[str]) -> Dict[str, float]:
    filtered = {k: v for k, v in weights.items() if k in available_keys}
    total = sum(filtered.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in filtered.items()}


# --------------------------------------------------------------------------- #
# Section weight lookup
# --------------------------------------------------------------------------- #

def _section_weights_for(source_type: str) -> Dict[str, float]:
    if source_type in ("10-K", "10-Q"):
        return FILING_SECTION_WEIGHTS
    if source_type == "8-K":
        return EIGHT_K_SECTION_WEIGHTS
    if source_type == "press_release":
        return PRESS_RELEASE_SECTION_WEIGHTS
    if source_type == "transcript":
        return TRANSCRIPT_SECTION_WEIGHTS
    if source_type == "cfo_commentary":
        return CFO_COMMENTARY_SECTION_WEIGHTS
    return {}


# --------------------------------------------------------------------------- #
# Document scoring (§23)
# --------------------------------------------------------------------------- #

def score_document(
    doc: SourceDocument,
    section_scores: List[SectionScore],
) -> DocumentScore | None:
    """Aggregate section scores using the document-type weights.

    Returns ``None`` when no section scored or weights resolve to zero.
    """
    if not section_scores:
        return None

    # Filter out risk_factors from absolute filing tone (§22.1).
    scored_by_name = {s.section_name: s for s in section_scores}
    weights = _section_weights_for(doc.source_type)
    renorm = renormalize_weights(weights, set(scored_by_name) & set(weights))
    if not renorm:
        return None

    final = 0.0
    for name, w in renorm.items():
        final += w * scored_by_name[name].final_score
    final = clip(final, -1.0, 1.0)

    return DocumentScore(
        source_type=doc.source_type,
        filed_at=doc.filed_at,
        title=doc.title,
        section_scores=section_scores,
        final_score=final,
    )


# --------------------------------------------------------------------------- #
# Filing tone (§26)
# --------------------------------------------------------------------------- #

def compute_filing_tone(document_scores: List[DocumentScore]) -> float:
    if not document_scores:
        return 0.0
    # Group-weighted mean using DOCUMENT_TYPE_WEIGHTS; if a type is absent we
    # renormalize across the remaining types (§26.3).
    scores_by_type: Dict[str, List[float]] = {}
    for ds in document_scores:
        scores_by_type.setdefault(ds.source_type, []).append(ds.final_score)

    type_means = {t: sum(v) / len(v) for t, v in scores_by_type.items()}
    renorm = renormalize_weights(DOCUMENT_TYPE_WEIGHTS, set(type_means.keys()))
    if not renorm:
        return 0.0
    tone = sum(renorm[t] * type_means[t] for t in renorm)
    return clip(tone, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Guidance tone (§25)
# --------------------------------------------------------------------------- #

def compute_guidance_tone(
    document_scores: List[DocumentScore],
    warnings: list[str] | None = None,
) -> Tuple[float, int]:
    """Return ``(score, source_count)`` averaged over outlook_guidance sections."""
    relevant: List[float] = []
    for ds in document_scores:
        for s in ds.section_scores:
            if s.section_name == "outlook_guidance":
                relevant.append(s.final_score)

    if not relevant:
        if warnings is not None:
            warnings.append("No guidance language found in current lookback window")
        return (0.0, 0)
    mean = sum(relevant) / len(relevant)
    return (clip(mean, -1.0, 1.0), len(relevant))


# --------------------------------------------------------------------------- #
# Composite (§27)
# --------------------------------------------------------------------------- #

def compute_final_score(
    filing_tone: float,
    filing_delta: float,
    guidance_tone: float,
    investor_context: float,
) -> Tuple[float, float, str]:
    raw = (
        0.50 * filing_tone
        + 0.25 * filing_delta
        + 0.15 * guidance_tone
        + 0.10 * investor_context
    )
    raw = clip(raw, -1.0, 1.0)
    score_0_100 = 50.0 + 50.0 * raw
    score_0_100 = max(0.0, min(100.0, score_0_100))
    return raw, score_0_100, map_score_to_label(score_0_100)


def map_score_to_label(score_0_100: float) -> str:
    if score_0_100 < 35:
        return "bearish"
    if score_0_100 < 45:
        return "mildly bearish"
    if score_0_100 < 55:
        return "neutral"
    if score_0_100 < 65:
        return "mildly bullish"
    return "bullish"
