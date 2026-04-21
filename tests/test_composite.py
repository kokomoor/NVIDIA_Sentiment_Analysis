"""§41.5 — composite score transform and label mapping."""

import pytest

from nvda_sentiment.config import LEADERSHIP_COMPONENT_WEIGHTS
from nvda_sentiment.scorers.composite import (
    clip,
    compute_filing_tone,
    compute_leadership_component,
    map_score_to_label,
    renormalize_weights,
    score_to_0_100_and_label,
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


def test_leadership_component_uses_renormalized_weights():
    # Inputs chosen to verify the renormalized weights (0.50/0.90 etc.)
    comp = compute_leadership_component(0.32, 0.10, 0.25)
    w = LEADERSHIP_COMPONENT_WEIGHTS
    expected = w["filing_tone"] * 0.32 + w["filing_delta"] * 0.10 + w["guidance_tone"] * 0.25
    assert comp == pytest.approx(expected)


def test_leadership_weights_sum_to_one():
    assert sum(LEADERSHIP_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)


def test_score_to_0_100_monotonic_and_clipped():
    score_lo, _ = score_to_0_100_and_label(-2.0)   # outside -1
    score_hi, _ = score_to_0_100_and_label(2.0)    # outside +1
    assert score_lo == 0.0
    assert score_hi == 100.0
    mid_score, mid_label = score_to_0_100_and_label(0.0)
    assert mid_score == 50.0
    assert mid_label == "neutral"


def test_leadership_component_clipped_even_with_large_input():
    # Saturate every sub-input; result still in [-1, 1].
    comp = compute_leadership_component(1.0, 1.0, 1.0)
    assert comp == pytest.approx(1.0)
    comp = compute_leadership_component(-1.0, -1.0, -1.0)
    assert comp == pytest.approx(-1.0)


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
