"""§9.1 — divergence combiner unit tests.

Covers every quadrant of (sign L, sign I), the boundary L·I = 0,
clipping, and the investor_branch_ok=False fallback.
"""

from __future__ import annotations

import math

import pytest

from nvda_sentiment.features.combiner import CombinerOutput, combine


ALPHA = 0.55
LAMBDA_NEG = 0.25
LAMBDA_POS = 0.10


def _run(L: float, I: float, *, ok: bool = True, lambda_pos: float = LAMBDA_POS) -> CombinerOutput:
    return combine(
        leadership_component=L,
        investor_component=I,
        alpha=ALPHA,
        lambda_neg=LAMBDA_NEG,
        lambda_pos=lambda_pos,
        investor_branch_ok=ok,
    )


def test_both_positive_no_adjustment() -> None:
    out = _run(0.5, 0.3)
    # base = 0.55*0.5 + 0.45*0.3 = 0.41
    assert math.isclose(out.adjustment, 0.0, abs_tol=1e-9)
    assert math.isclose(out.combined_score, 50.0 + 50.0 * 0.41, abs_tol=1e-6)
    assert out.agreement is True


def test_both_negative_no_adjustment() -> None:
    out = _run(-0.6, -0.4)
    # base = 0.55*(-0.6) + 0.45*(-0.4) = -0.51
    assert math.isclose(out.adjustment, 0.0, abs_tol=1e-9)
    assert math.isclose(out.combined_score, 50.0 + 50.0 * -0.51, abs_tol=1e-6)
    assert out.agreement is True


def test_leadership_bull_investor_bear_applies_negative_penalty() -> None:
    # L=+0.5, I=-0.5 → base = 0.55*0.5 + 0.45*(-0.5) = 0.05
    # adjustment = -0.25 * |0.5 - (-0.5)| = -0.25
    # combined = clip(0.05 + -0.25) = -0.20 → score = 40.0
    out = _run(0.5, -0.5)
    assert out.agreement is False
    assert math.isclose(out.adjustment, -0.25, abs_tol=1e-9)
    assert math.isclose(out.combined_score, 40.0, abs_tol=1e-6)


def test_leadership_bear_investor_bull_applies_contrarian_premium() -> None:
    # L=-0.5, I=+0.5 → base = 0.55*(-0.5) + 0.45*0.5 = -0.05
    # adjustment = +0.10 * |L - I| = +0.10
    # combined = clip(-0.05 + +0.10) = +0.05 → score = 52.5
    out = _run(-0.5, 0.5)
    assert out.agreement is False
    assert math.isclose(out.adjustment, 0.10, abs_tol=1e-9)
    assert math.isclose(out.combined_score, 52.5, abs_tol=1e-6)


def test_boundary_L_times_I_equals_zero_is_agreement() -> None:
    out = _run(0.0, 0.5)
    # L*I == 0 → signs_agree=True, adjustment=0
    assert out.agreement is True
    assert math.isclose(out.adjustment, 0.0, abs_tol=1e-9)
    # base = 0.55*0 + 0.45*0.5 = 0.225 → 61.25
    assert math.isclose(out.combined_score, 50.0 + 50.0 * 0.225, abs_tol=1e-6)


def test_upper_clip_both_plus_one() -> None:
    out = _run(1.0, 1.0)
    assert math.isclose(out.combined_score, 100.0, abs_tol=1e-9)
    assert out.agreement is True


def test_lower_clip_both_minus_one() -> None:
    out = _run(-1.0, -1.0)
    assert math.isclose(out.combined_score, 0.0, abs_tol=1e-9)


def test_contrarian_premium_large_does_not_overflow_clip() -> None:
    # L=-1, I=+1, λ_pos=+0.30 → base = -0.10, adj = 0.30*2 = 0.60, combined = clip(0.50) → 75
    out = _run(-1.0, 1.0, lambda_pos=0.30)
    assert math.isclose(out.combined_score, 75.0, abs_tol=1e-6)
    assert math.isclose(out.adjustment, 0.60, abs_tol=1e-9)


def test_investor_branch_not_ok_falls_back_to_leadership() -> None:
    out = _run(0.4, -0.9, ok=False)
    # Leadership-only fallback: combined_score == leadership_score
    assert math.isclose(out.combined_score, out.leadership_score, abs_tol=1e-9)
    assert math.isclose(out.combined_score, 50.0 + 50.0 * 0.4, abs_tol=1e-6)
    assert out.agreement is True
    assert math.isclose(out.adjustment, 0.0, abs_tol=1e-9)


def test_divergence_equals_leadership_minus_investor_score() -> None:
    out = _run(0.5, -0.5)
    # leadership_score=75, investor_score=25 → divergence=50
    assert math.isclose(out.divergence, 50.0, abs_tol=1e-9)


@pytest.mark.parametrize("L,I", [(2.0, -2.0), (-5.0, 5.0)])
def test_inputs_are_clipped_to_unit_interval(L: float, I: float) -> None:
    out_wild = _run(L, I)
    out_clipped = _run(max(-1.0, min(1.0, L)), max(-1.0, min(1.0, I)))
    assert math.isclose(out_wild.combined_score, out_clipped.combined_score, abs_tol=1e-9)
    assert math.isclose(out_wild.divergence, out_clipped.divergence, abs_tol=1e-9)
