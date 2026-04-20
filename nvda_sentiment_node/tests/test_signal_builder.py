"""§29 — signal builder."""

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


def test_positive_everything():
    signals = build_signals(
        filing_tone=0.3,
        filing_delta=0.15,
        tone_delta=0.15,
        risk_delta=0.10,
        guidance_tone=0.20,
        investor_context=0.15,
        document_scores=[],
        section_scores_by_type={
            "prepared_remarks": [_ss("prepared_remarks", 0.2)],
            "qa": [_ss("qa", 0.4)],  # qa higher → handled pushback well
        },
    )
    joined = " | ".join(signals)
    assert "positive across recent official materials" in joined
    assert "Tone improved" in joined
    assert "softened" in joined
    assert "constructive" in joined
    assert "handled pushback well" in joined
    assert "supportive" in joined
    assert len(signals) <= 6


def test_negative_everything():
    signals = build_signals(
        filing_tone=-0.3,
        filing_delta=-0.2,
        tone_delta=-0.2,
        risk_delta=-0.1,
        guidance_tone=-0.2,
        investor_context=-0.2,
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
        guidance_tone=0.0, investor_context=0.0,
        document_scores=[],
        section_scores_by_type={
            "prepared_remarks": [_ss("prepared_remarks", 0.1)],
            "qa": [_ss("qa", 0.05)],  # gap 0.05 < 0.15
        },
    )
    assert not any("pushback" in s or "traditional caution" in s for s in signals)


def test_no_market_context_suppresses_tier_6():
    signals = build_signals(
        filing_tone=0.0, filing_delta=0.0, tone_delta=0.0, risk_delta=0.0,
        guidance_tone=0.0, investor_context=0.0,
        document_scores=[],
        section_scores_by_type={},
        include_market_context=False,
    )
    assert not any("investor risk appetite" in s for s in signals)
