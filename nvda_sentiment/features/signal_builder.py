"""Human-readable signal builder (§29)."""

from __future__ import annotations

from typing import Dict, List

from ..schemas import DocumentScore, SectionScore


def build_signals(
    *,
    filing_tone: float,
    filing_delta: float,
    tone_delta: float,
    risk_delta: float,
    guidance_tone: float,
    document_scores: List[DocumentScore],
    section_scores_by_type: Dict[str, List[SectionScore]],
    investor_component: float = 0.0,
    divergence: float = 0.0,
    include_investor_branch: bool = False,
) -> List[str]:
    signals: List[str] = []

    # Tier 1 — Overall tone (always emit one)
    if filing_tone >= 0.20:
        signals.append("Management tone is positive across recent official materials")
    elif filing_tone <= -0.20:
        signals.append("Management tone is negative across recent official materials")
    else:
        signals.append("Management tone is broadly balanced across recent official materials")

    # Tier 2 — Tone trajectory (always emit one)
    if tone_delta >= 0.10:
        signals.append("Tone improved versus the prior comparable period")
    elif tone_delta <= -0.10:
        signals.append("Tone weakened versus the prior comparable period")
    else:
        signals.append("Tone changed only modestly versus the prior comparable period")

    # Tier 3 — Risk language (always emit one)
    if risk_delta >= 0.05:
        signals.append("Risk-factor language softened versus prior filing")
    elif risk_delta <= -0.05:
        signals.append("Risk-factor language intensified versus prior filing")
    else:
        signals.append("Risk-factor language was substantively stable versus prior filing")

    # Tier 4 — Guidance (always emit one)
    if guidance_tone >= 0.15:
        signals.append("Forward-looking guidance language is constructive")
    elif guidance_tone <= -0.15:
        signals.append("Forward-looking guidance language is cautious")
    else:
        signals.append("Forward-looking guidance language is mixed")

    # Tier 5 — Q&A vs prepared-remarks divergence
    prepared_scores = [s.final_score for s in section_scores_by_type.get("prepared_remarks", [])]
    qa_scores = [s.final_score for s in section_scores_by_type.get("qa", [])]
    if prepared_scores and qa_scores:
        prepared_mean = sum(prepared_scores) / len(prepared_scores)
        qa_mean = sum(qa_scores) / len(qa_scores)
        qa_gap = prepared_mean - qa_mean
        if qa_gap >= 0.15:
            signals.append("Analyst Q&A tone ran meaningfully below prepared remarks — a traditional caution signal")
        elif qa_gap <= -0.15:
            signals.append("Analyst Q&A tone ran above prepared remarks — management handled pushback well")
        # else: skip (§29.2 tier 5)

    # Tier 6 — Investor positioning (§6.5.3)
    if include_investor_branch:
        if investor_component >= 0.10:
            signals.append("Market positioning is bullish")
        elif investor_component <= -0.10:
            signals.append("Market positioning is bearish")
        else:
            signals.append("Market positioning is neutral")

    # Tier 7 — Divergence (§6.5.3)
    if include_investor_branch:
        abs_div = abs(divergence)
        if abs_div < 10:
            signals.append("Leadership and market signals are aligned")
        elif divergence > 20:
            signals.append("Expectations gap: management bullish while market skeptical")
        elif divergence < -20:
            signals.append("Expectations gap: management cautious while market optimistic")

    # §6.5.3 cap at 8 (was 6)
    return signals[:8]
