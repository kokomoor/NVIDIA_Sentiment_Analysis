"""§24 — filing delta + fiscal bucket correctness (fixes advisor-flagged bug)."""

from datetime import date

import pytest

from nvda_sentiment.features.filing_delta import (
    compute_filing_delta,
    compute_tone_delta,
    nvda_fiscal_bucket,
)
from nvda_sentiment.schemas import DocumentScore, SectionScore


def test_fiscal_bucket_feb_2026_is_fy27q1():
    assert nvda_fiscal_bucket(date(2026, 2, 15)) == "FY27Q1"


def test_fiscal_bucket_feb_2025_is_fy26q1():
    assert nvda_fiscal_bucket(date(2025, 2, 15)) == "FY26Q1"


def test_fiscal_bucket_nov_2026_is_fy27q4():
    # NVIDIA fiscal Q4 spans Nov-Jan, fiscal year ends Jan 2027 -> FY27
    assert nvda_fiscal_bucket(date(2026, 11, 15)) == "FY27Q4"


def test_fiscal_bucket_jan_2026_is_fy26q4():
    # FY26 ends in Jan 2026
    assert nvda_fiscal_bucket(date(2026, 1, 20)) == "FY26Q4"


def test_fiscal_bucket_jan_2027_is_fy27q4():
    assert nvda_fiscal_bucket(date(2027, 1, 20)) == "FY27Q4"


def test_fiscal_bucket_q2_q3():
    assert nvda_fiscal_bucket(date(2026, 5, 1)) == "FY27Q2"
    assert nvda_fiscal_bucket(date(2026, 7, 31)) == "FY27Q2"
    assert nvda_fiscal_bucket(date(2026, 8, 1)) == "FY27Q3"
    assert nvda_fiscal_bucket(date(2026, 10, 31)) == "FY27Q3"


def _ds(source_type, score, filed, risk_score=None):
    sections = []
    if risk_score is not None:
        sections.append(
            SectionScore(
                section_name="risk_factors",
                finbert_score=0.0,
                lexicon_score=0.0,
                uncertainty_penalty=0.0,
                final_score=risk_score,
                sentence_count=5,
            )
        )
    return DocumentScore(
        source_type=source_type,
        filed_at=filed,
        title=source_type,
        section_scores=sections,
        final_score=score,
    )


def test_tone_delta_yoy_preferred():
    # Current FY27Q1 (Feb 2026), prior-year FY26Q1 (Feb 2025), and intermediate
    docs = [
        _ds("10-Q", 0.1, "2025-02-15"),   # FY26Q1
        _ds("10-Q", 0.2, "2025-05-15"),   # FY26Q2
        _ds("10-Q", 0.3, "2026-02-15"),   # FY27Q1 current
    ]
    delta, yoy = compute_tone_delta(docs)
    # YoY prefers FY26Q1 as prior (same quarter prior year)
    assert yoy is True
    assert delta == pytest.approx(0.3 - 0.1)


def test_tone_delta_sequential_fallback():
    # No YoY available; must fall back to previous bucket
    docs = [
        _ds("10-Q", 0.2, "2026-05-15"),   # FY27Q2
        _ds("10-Q", 0.5, "2026-08-15"),   # FY27Q3 current
    ]
    delta, yoy = compute_tone_delta(docs)
    assert yoy is False
    assert delta == pytest.approx(0.3)


def test_filing_delta_combines_tone_and_risk():
    # Risk got worse YoY: current risk = -0.4, prior risk = -0.2 => risk_delta = -0.2
    docs = [
        _ds("10-Q", 0.0, "2025-02-15", risk_score=-0.2),
        _ds("10-Q", 0.2, "2026-02-15", risk_score=-0.4),
    ]
    combined, tone_delta, risk_delta, yoy = compute_filing_delta(docs)
    assert yoy is True
    assert tone_delta == pytest.approx(0.2)
    assert risk_delta == pytest.approx(-0.2)
    # 0.70 * 0.2 + 0.30 * -0.2 = 0.14 - 0.06 = 0.08
    assert combined == pytest.approx(0.08)


def test_filing_delta_no_prior_returns_zero():
    docs = [_ds("10-Q", 0.5, "2026-02-15")]
    combined, tone_delta, risk_delta, yoy = compute_filing_delta(docs)
    assert combined == 0.0 and tone_delta == 0.0 and risk_delta == 0.0
    assert yoy is False
