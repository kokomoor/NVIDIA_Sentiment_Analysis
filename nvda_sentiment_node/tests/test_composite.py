"""§41.5 — composite score transform and label mapping."""

import pytest

from nvda_sentiment.scorers.composite import (
    clip,
    compute_filing_tone,
    compute_final_score,
    map_score_to_label,
    renormalize_weights,
)
from nvda_sentiment.schemas import DocumentScore, SectionScore


def test_clip_bounds():
    assert clip(2.0) == 1.0
    assert clip(-2.0) == -1.0
    assert clip(0.3) == 0.3


def test_renormalize_weights_drops_missing():
    w = {"a": 0.3, "b": 0.7}
    out = renormalize_weights(w, {"a"})
    assert out == {"a": 1.0}


def test_final_score_transform():
    raw, score, label = compute_final_score(0.32, 0.10, 0.25, 0.05)
    # 0.50*0.32 + 0.25*0.10 + 0.15*0.25 + 0.10*0.05 = 0.2275
    assert raw == pytest.approx(0.2275)
    assert score == pytest.approx(61.375)
    assert label == "mildly bullish"


def test_final_score_clipping():
    # Any input > 1 in raw should not happen, but map must stay in [0,100]
    _, score, _ = compute_final_score(1.0, 1.0, 1.0, 1.0)
    assert score == 100.0
    _, score, _ = compute_final_score(-1.0, -1.0, -1.0, -1.0)
    assert score == 0.0


def test_label_mapping_boundaries():
    assert map_score_to_label(0) == "bearish"
    assert map_score_to_label(34.9) == "bearish"
    assert map_score_to_label(35) == "mildly bearish"
    assert map_score_to_label(44.99) == "mildly bearish"
    assert map_score_to_label(45) == "neutral"
    assert map_score_to_label(54.99) == "neutral"
    assert map_score_to_label(55) == "mildly bullish"
    assert map_score_to_label(64.99) == "mildly bullish"
    assert map_score_to_label(65) == "bullish"
    assert map_score_to_label(100) == "bullish"


def _ds(source_type, score, filed="2026-02-01"):
    return DocumentScore(
        source_type=source_type,
        filed_at=filed,
        title=source_type,
        section_scores=[],
        final_score=score,
    )


def test_filing_tone_weighted_mean():
    scores = [_ds("10-K", 0.5), _ds("10-Q", 0.3), _ds("transcript", 0.1)]
    tone = compute_filing_tone(scores)
    # Filtered weights renormalized: 10-K 0.3, 10-Q 0.3, transcript 0.1 -> /0.7
    expected = (0.30 * 0.5 + 0.30 * 0.3 + 0.10 * 0.1) / 0.70
    assert tone == pytest.approx(expected)
