"""Divergence-aware combiner for leadership and investor branches (§6.5.1)."""

from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class CombinerOutput:
    leadership_score: float   # 0-100
    investor_score: float     # 0-100
    combined_score: float     # 0-100
    divergence: float         # [-100, +100]  (leadership_score - investor_score)
    agreement: bool           # signs of leadership & investor components agree
    adjustment: float         # divergence adjustment applied to base blend (component units)


def combine(
    *,
    leadership_component: float,
    investor_component: float,
    alpha: float,
    lambda_neg: float,
    lambda_pos: float,
    investor_branch_ok: bool,
) -> CombinerOutput:
    """Blend leadership + investor components with asymmetric divergence adjustment.

    Positive divergence (leadership > investor) incurs a *bearish* penalty
    weighted by ``lambda_neg`` (insider optimism without market confirmation).
    Negative divergence (investor > leadership) earns a modest *contrarian*
    premium weighted by ``lambda_pos``.

    When ``investor_branch_ok`` is False, the combiner degrades to a
    leadership-only fallback (combined := leadership score).
    """
    L = _clip(leadership_component)
    I = _clip(investor_component)
    leadership_score = 50.0 + 50.0 * L
    investor_score = 50.0 + 50.0 * I

    if not investor_branch_ok:
        return CombinerOutput(
            leadership_score=leadership_score,
            investor_score=investor_score,
            combined_score=leadership_score,
            divergence=leadership_score - investor_score,
            agreement=True,
            adjustment=0.0,
        )

    base = alpha * L + (1.0 - alpha) * I
    signs_agree = (L * I) >= 0.0
    if signs_agree:
        adjustment = 0.0
    else:
        d = abs(L - I)
        if L > I:
            adjustment = -lambda_neg * d
        else:
            adjustment = +lambda_pos * d

    combined_raw = _clip(base + adjustment)
    combined_score = 50.0 + 50.0 * combined_raw

    return CombinerOutput(
        leadership_score=leadership_score,
        investor_score=investor_score,
        combined_score=combined_score,
        divergence=leadership_score - investor_score,
        agreement=signs_agree,
        adjustment=adjustment,
    )
