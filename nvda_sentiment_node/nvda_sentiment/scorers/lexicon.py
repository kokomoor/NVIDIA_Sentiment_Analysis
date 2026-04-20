"""Small deterministic finance lexicon scorer (§20)."""

from __future__ import annotations

from typing import Dict

from ..utils.text import tokenize_words


POSITIVE_TERMS = {
    "growth", "strong", "record", "improved", "increase", "increased",
    "accelerate", "accelerated", "opportunity", "leadership", "momentum",
    "healthy", "robust", "benefit", "benefits", "confidence", "efficient",
    "innovation", "successful", "success", "expansion", "expanded",
}

NEGATIVE_TERMS = {
    "decline", "declined", "decrease", "decreased", "weakness", "softness",
    "risk", "adverse", "pressure", "headwind", "uncertain", "uncertainty",
    "challenge", "challenging", "constraint", "constraints", "litigation",
    "disruption", "loss", "losses", "slowdown", "volatile",
}

UNCERTAINTY_TERMS = {
    "may", "might", "could", "uncertain", "uncertainty", "depends",
    "potential", "possibly", "assume", "assumption",
}

LITIGIOUS_TERMS = {
    "litigation", "claim", "claims", "proceeding", "proceedings",
    "investigation", "investigations", "regulatory", "compliance",
}


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class LexiconScorer:
    """Return ``lexicon_score`` plus individual component rates (§20.5)."""

    def score_text(self, text: str) -> Dict[str, float]:
        tokens = tokenize_words(text or "")
        token_count = len(tokens)
        if token_count == 0:
            return {
                "lexicon_score": 0.0,
                "positive_rate": 0.0,
                "negative_rate": 0.0,
                "uncertainty_rate": 0.0,
                "litigious_rate": 0.0,
                "token_count": 0,
            }

        positive_count = sum(1 for t in tokens if t in POSITIVE_TERMS)
        negative_count = sum(1 for t in tokens if t in NEGATIVE_TERMS)
        uncertainty_count = sum(1 for t in tokens if t in UNCERTAINTY_TERMS)
        litigious_count = sum(1 for t in tokens if t in LITIGIOUS_TERMS)

        positive_rate = positive_count / token_count
        negative_rate = negative_count / token_count
        uncertainty_rate = uncertainty_count / token_count
        litigious_rate = litigious_count / token_count

        raw = (
            6.0 * (positive_rate - negative_rate)
            - 3.0 * uncertainty_rate
            - 2.0 * litigious_rate
        )
        score = _clip(raw, -1.0, 1.0)

        return {
            "lexicon_score": score,
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
            "uncertainty_rate": uncertainty_rate,
            "litigious_rate": litigious_rate,
            "token_count": float(token_count),
        }
