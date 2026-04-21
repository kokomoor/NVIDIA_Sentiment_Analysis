"""§29, §6.5.3 — signal builder (dual-branch tiers)."""

from nvda_sentiment.features.signal_builder import build_signals
from nvda_sentiment.schemas import SectionScore


def _ss(name, score):
    return SectionScore(
        section_name=name,
        finbert_score=0.0,
        lexicon_score=0.0,
        uncertainty_penalty=0.0,
        final_score=score,
        sentence_count=5,
    )


def test_positive_everything_leadership_only():
    signals = build_signals(
        filing_tone=0.3,
        filing_delta=0.15,
        tone_delta=0.15,
        risk_delta=0.10,
        guidance_tone=0.20,
        document_scores=[],
        section_scores_by_type={
            "prepared_remarks": [_ss("prepared_remarks", 0.2)],
            "qa": [_ss("qa", 0.4)],  # qa higher -> handled pushback well
        },
    )
    joined = " | ".join(signals)
    assert "positive across recent official materials" in joined
    assert "Tone improved" in joined
    assert "softened" in joined
    assert "constructive" in joined
    assert "handled pushback well" in joined
    # Leadership-only mode: no market-positioning / divergence signals.
    assert not any("Market positioning" in s for s in signals)
    assert not any("Expectations gap" in s for s in signals)
    assert not any("aligned" in s for s in signals)
    assert len(signals) <= 8


def test_negative_everything_leadership_only():
    signals = build_signals(
        filing_tone=-0.3,
        filing_delta=-0.2,
        tone_delta=-0.2,
        risk_delta=-0.1,
        guidance_tone=-0.2,
        document_scores=[],
        section_scores_by_type={},
    )
    joined = " | ".join(signals)
    assert "negative across recent official materials" in joined
    assert "weakened" in joined
    assert "intensified" in joined
    assert "cautious" in joined


def test_qa_gap_not_emitted_when_small():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0,
        document_scores=[],
        section_scores_by_type={
            "prepared_remarks": [_ss("prepared_remarks", 0.1)],
            "qa": [_ss("qa", 0.05)],  # gap 0.05 < 0.15
        },
    )
    assert not any("pushback" in s or "traditional caution" in s for s in signals)


def test_investor_tier_emitted_when_branch_included():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0,
        document_scores=[],
        section_scores_by_type={},
        investor_component=0.25,
        divergence=5.0,
        include_investor_branch=True,
    )
    joined = " | ".join(signals)
    assert "Market positioning is bullish" in joined
    # |divergence| < 10 → aligned signal emitted.
    assert "aligned" in joined


def test_divergence_signal_bullish_management_skeptical_market():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0,
        document_scores=[],
        section_scores_by_type={},
        investor_component=-0.3,
        divergence=30.0,  # leadership > investor
        include_investor_branch=True,
    )
    joined = " | ".join(signals)
    assert "management bullish while market skeptical" in joined


def test_divergence_signal_cautious_management_optimistic_market():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0,
        document_scores=[],
        section_scores_by_type={},
        investor_component=0.4,
        divergence=-30.0,
        include_investor_branch=True,
    )
    joined = " | ".join(signals)
    assert "management cautious while market optimistic" in joined


def test_no_investor_branch_suppresses_investor_tiers():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0,
        document_scores=[],
        section_scores_by_type={},
        investor_component=0.5,
        divergence=25.0,
        include_investor_branch=False,
    )
    assert not any("Market positioning" in s for s in signals)
    assert not any("Expectations gap" in s for s in signals)
