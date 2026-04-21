"""§28 — confidence arithmetic."""

import pytest

from nvda_sentiment.features.confidence import compute_confidence
from nvda_sentiment.schemas import DocumentScore, SourceDocument


def _doc(t):
    return SourceDocument(source_type=t, title=t, url="u", filed_at="2026-02-01")


def _ds(t):
    return DocumentScore(source_type=t, filed_at="2026-02-01", title=t, section_scores=[], final_score=0.1)


def test_empty_documents_short_circuits_to_floor():
    c = compute_confidence(
        fetched_docs=[],
        document_scores=[],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=0,
        extraction_successes=0,
        finbert_available=True,
        include_market_context=True,
        investor_context_available=True,
        guidance_tone_source_count=0,
        filing_delta_computed=False,
        yoy_matched=False,
    )
    assert c == 0.10


def test_ceiling_at_0p95_with_full_bonuses():
    fetched = [_doc("10-K"), _doc("10-Q"), _doc("10-Q"), _doc("transcript"), _doc("cfo_commentary")]
    scored = [_ds("10-K"), _ds("10-Q"), _ds("10-Q"), _ds("transcript"), _ds("cfo_commentary")]
    c = compute_confidence(
        fetched_docs=fetched,
        document_scores=scored,
        warnings=[],
        fetch_failures=0,
        extraction_attempts=5,
        extraction_successes=5,
        finbert_available=True,
        include_market_context=True,
        investor_context_available=True,
        guidance_tone_source_count=3,
        filing_delta_computed=True,
        yoy_matched=True,
    )
    # 0.55 + 0.40 (capped) = 0.95
    assert c == 0.95


def test_penalties_apply_below_base():
    # single doc, nothing going right: base 0.55, 0 additions, 4x -0.05 penalties
    c = compute_confidence(
        fetched_docs=[_doc("press_release")],
        document_scores=[_ds("press_release")],
        warnings=[],
        fetch_failures=10,
        extraction_attempts=1,
        extraction_successes=0,
        finbert_available=False,
        include_market_context=True,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=False,
        yoy_matched=False,
    )
    # 0.55 - 0.05 (no delta) - 0.05 (no ctx) - 0.05 (fetch fail) - 0.05 (extract fail) = 0.35
    assert c == pytest.approx(0.35)


def test_investor_branch_bonus_caps_at_0p05_for_three_plus_subsources():
    # 3+ subsources ok → +0.05 investor bonus; without broad_market detection the
    # ctx-penalty must NOT apply because we pass investor_context_available=True.
    fetched = [_doc("10-K"), _doc("10-Q"), _doc("10-Q"), _doc("transcript"), _doc("cfo_commentary")]
    scored = [_ds("10-K"), _ds("10-Q"), _ds("10-Q"), _ds("transcript"), _ds("cfo_commentary")]
    c = compute_confidence(
        fetched_docs=fetched,
        document_scores=scored,
        warnings=[],
        fetch_failures=0,
        extraction_attempts=5,
        extraction_successes=5,
        finbert_available=True,
        include_market_context=True,
        investor_context_available=True,
        guidance_tone_source_count=3,
        filing_delta_computed=True,
        yoy_matched=True,
        include_investor_branch=True,
        investor_branch_ok=True,
        investor_subsources_ok=4,
    )
    # All 8 base bonuses sum to 0.40, +0.05 investor → capped at 0.40 → 0.95 ceiling.
    assert c == 0.95


def test_investor_branch_zero_subsources_no_bonus_no_penalty_when_not_requested():
    # include_investor_branch=False → no bonus, no penalty.
    c_baseline = compute_confidence(
        fetched_docs=[_doc("10-Q")],
        document_scores=[_ds("10-Q")],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=1,
        extraction_successes=1,
        finbert_available=True,
        include_market_context=False,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=True,
        yoy_matched=False,
        include_investor_branch=False,
        investor_branch_ok=False,
        investor_subsources_ok=0,
    )
    c_with_bonus = compute_confidence(
        fetched_docs=[_doc("10-Q")],
        document_scores=[_ds("10-Q")],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=1,
        extraction_successes=1,
        finbert_available=True,
        include_market_context=False,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=True,
        yoy_matched=False,
        include_investor_branch=True,
        investor_branch_ok=True,
        investor_subsources_ok=3,
    )
    # 3+ ok subs should add +0.05 above baseline.
    assert c_with_bonus - c_baseline == pytest.approx(0.05)


def test_investor_branch_requested_but_failed_applies_penalty():
    c_ok = compute_confidence(
        fetched_docs=[_doc("10-Q")],
        document_scores=[_ds("10-Q")],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=1,
        extraction_successes=1,
        finbert_available=True,
        include_market_context=False,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=True,
        yoy_matched=False,
        include_investor_branch=True,
        investor_branch_ok=True,
        investor_subsources_ok=0,
    )
    c_failed = compute_confidence(
        fetched_docs=[_doc("10-Q")],
        document_scores=[_ds("10-Q")],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=1,
        extraction_successes=1,
        finbert_available=True,
        include_market_context=False,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=True,
        yoy_matched=False,
        include_investor_branch=True,
        investor_branch_ok=False,
        investor_subsources_ok=0,
    )
    # Same params, but ok=False → -0.05 penalty.
    assert c_failed - c_ok == pytest.approx(-0.05)


def test_floor_clamp_when_deeply_penalized():
    # Confidence cannot go below 0.10 (§28.4)
    # Start at base, add no bonuses, and we only have 4 penalties defined,
    # so we can't actually drive below 0.10 with a non-empty doc set - this
    # just confirms the clamp is wired in with empty docs.
    c = compute_confidence(
        fetched_docs=[],
        document_scores=[],
        warnings=[],
        fetch_failures=0,
        extraction_attempts=0,
        extraction_successes=0,
        finbert_available=False,
        include_market_context=False,
        investor_context_available=False,
        guidance_tone_source_count=0,
        filing_delta_computed=False,
        yoy_matched=False,
    )
    assert c == 0.10
