"""Lexicon scorer (§20)."""

from nvda_sentiment.scorers.lexicon import LexiconScorer


def test_positive_beats_negative():
    pos = LexiconScorer().score_text(
        "Record strong growth. Momentum and expansion across the business."
    )
    neg = LexiconScorer().score_text(
        "Weakness and decline. Pressure and headwinds on the business with losses and disruption."
    )
    assert pos["lexicon_score"] > 0
    assert neg["lexicon_score"] < 0


def test_empty_returns_zero():
    out = LexiconScorer().score_text("")
    assert out["lexicon_score"] == 0.0
    assert out["token_count"] == 0


def test_uncertainty_contributes_penalty():
    # Same positive words, one with uncertainty markers
    base = LexiconScorer().score_text("Growth and momentum are strong.")
    uncertain = LexiconScorer().score_text(
        "Growth and momentum may potentially be strong."
    )
    assert uncertain["uncertainty_rate"] > 0
    # score with uncertainty should be <= base score
    assert uncertain["lexicon_score"] <= base["lexicon_score"]
