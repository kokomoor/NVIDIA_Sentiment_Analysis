"""Combine FinBERT + lexicon into a section score (§21)."""

from __future__ import annotations

from ..schemas import SectionScore
from .finbert_scorer import FinBERTScorer
from .lexicon import LexiconScorer


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class SectionScorer:
    """Produce a ``SectionScore`` for a given section of text."""

    def __init__(self, settings, finbert: FinBERTScorer | None = None, lexicon: LexiconScorer | None = None):
        self.settings = settings
        self.finbert = finbert or FinBERTScorer(settings)
        self.lexicon = lexicon or LexiconScorer()

    @property
    def finbert_available(self) -> bool:
        return bool(self.finbert and self.finbert.available)

    def score_section(self, section_name: str, text: str) -> SectionScore:
        finbert_score, sentence_count = self.finbert.score_text(text)
        lex = self.lexicon.score_text(text)
        lexicon_score = lex["lexicon_score"]
        uncertainty_rate = lex["uncertainty_rate"]

        if self.finbert_available:
            base = 0.75 * finbert_score + 0.25 * lexicon_score
        else:
            # Fallback: lexicon alone (§36.3). Its small native magnitude is
            # promoted to full weight by dropping the 0.75/0.25 blend.
            base = lexicon_score

        penalty = 0.10 * uncertainty_rate
        adjusted = _clip(base - penalty, -1.0, 1.0)

        return SectionScore(
            section_name=section_name,
            finbert_score=finbert_score,
            lexicon_score=lexicon_score,
            uncertainty_penalty=penalty,
            final_score=adjusted,
            sentence_count=sentence_count,
        )
