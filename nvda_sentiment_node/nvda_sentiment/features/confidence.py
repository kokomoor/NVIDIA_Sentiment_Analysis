"""Confidence score (§28)."""

from __future__ import annotations

from typing import List

from ..schemas import DocumentScore, SourceDocument


def compute_confidence(
    *,
    fetched_docs: List[SourceDocument],
    document_scores: List[DocumentScore],
    warnings: List[str],
    fetch_failures: int,
    extraction_attempts: int,
    extraction_successes: int,
    finbert_available: bool,
    include_market_context: bool,
    investor_context_available: bool,
    guidance_tone_source_count: int,
    filing_delta_computed: bool,
    yoy_matched: bool,
) -> float:
    # §28.5 — no scorable documents short-circuit
    if len(document_scores) == 0:
        return 0.10

    confidence = 0.55
    additions = 0.0

    def _add(cond: bool, amount: float) -> float:
        return amount if cond else 0.0

    # Latest 10-K available
    has_10k = any(d.source_type == "10-K" for d in fetched_docs)
    additions += _add(has_10k, 0.05)

    # At least 2 recent 10-Qs
    ten_q_count = sum(1 for d in fetched_docs if d.source_type == "10-Q")
    additions += _add(ten_q_count >= 2, 0.05)

    # Latest transcript
    has_transcript = any(d.source_type == "transcript" for d in fetched_docs)
    additions += _add(has_transcript, 0.05)

    # Latest CFO commentary
    has_cfo = any(d.source_type == "cfo_commentary" for d in fetched_docs)
    additions += _add(has_cfo, 0.05)

    # Guidance computed from at least 2 sources
    additions += _add(guidance_tone_source_count >= 2, 0.05)

    # FinBERT ran successfully
    additions += _add(finbert_available, 0.05)

    # Filing delta against YoY-matched comparison
    additions += _add(yoy_matched, 0.05)

    # Investor context retrieved
    additions += _add(investor_context_available, 0.05)

    # Cap additions at 0.40 (§28.2)
    additions = min(additions, 0.40)
    confidence += additions

    # Penalties (§28.3)
    if not filing_delta_computed:
        confidence -= 0.05
    if include_market_context and not investor_context_available:
        confidence -= 0.05

    total_fetch_attempts = len(fetched_docs) + fetch_failures
    if total_fetch_attempts > 0 and (fetch_failures / total_fetch_attempts) > 0.25:
        confidence -= 0.05

    if extraction_attempts > 0:
        success_rate = extraction_successes / extraction_attempts
        if success_rate < 0.60:
            confidence -= 0.05

    # §28.4 — final clip
    return max(0.10, min(0.95, confidence))
