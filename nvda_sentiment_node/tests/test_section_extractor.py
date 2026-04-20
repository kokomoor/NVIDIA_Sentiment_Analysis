"""§41.3 — section extraction behavior."""

from nvda_sentiment.parsers.section_extractor import SectionExtractor
from nvda_sentiment.schemas import SourceDocument


def _make_doc(source_type, text):
    return SourceDocument(
        source_type=source_type,
        title="t",
        url="http://x",
        filed_at="2026-01-01",
        clean_text=text,
    )


def test_mda_extraction_and_guidance():
    text = """Item 2. Management's Discussion and Analysis
Revenue grew 15% on strong data center demand. Our outlook for the next quarter is constructive.

Our operating margin expanded materially. We expect continued momentum across the data center business.

Item 3. Quantitative and Qualitative Disclosures
Should not be included in MDA output.
"""
    ext = SectionExtractor()
    out = ext.extract_sections(_make_doc("10-Q", text))
    assert "mda" in out
    assert "data center" in out["mda"].lower()
    assert "should not be included" not in out["mda"].lower()
    # guidance paragraph should be picked up (contains "outlook" / "we expect")
    assert "outlook_guidance" in out
    assert "outlook" in out["outlook_guidance"].lower() or "we expect" in out["outlook_guidance"].lower()


def test_risk_factors_extraction_with_stop():
    text = """Item 1A. Risk Factors
Our business faces several risks, including regulatory challenges and supply constraints.
We may experience significant volatility in any given quarter.

Item 1B. Unresolved Staff Comments
None.
"""
    ext = SectionExtractor()
    out = ext.extract_sections(_make_doc("10-K", text))
    assert "risk_factors" in out
    assert "regulatory challenges" in out["risk_factors"].lower()
    assert "unresolved staff comments" not in out["risk_factors"].lower()


def test_transcript_split():
    text = """Prepared remarks: revenue grew strongly and we are encouraged by demand.
Our margin trajectory is favorable.

Question-and-Answer
Analyst: Can you comment on data center growth?
CEO: Growth remains robust across the portfolio.
"""
    ext = SectionExtractor()
    out = ext.extract_sections(_make_doc("transcript", text))
    assert "prepared_remarks" in out
    assert "qa" in out
    assert "prepared remarks" in out["prepared_remarks"].lower()
    assert "analyst" in out["qa"].lower()


def test_guidance_paragraph_detection_in_press_release():
    text = """NVIDIA reported record revenue in the quarter with strong data center momentum.

Revenue was $30 billion, up 80% year over year, driven by GAAP and non-GAAP improvements.

Outlook: For the next quarter, we expect revenue of $32 billion, plus or minus 2 percent.
"""
    ext = SectionExtractor()
    out = ext.extract_sections(_make_doc("press_release", text))
    assert "headline_and_summary" in out
    assert "financial_highlights" in out
    assert "outlook_guidance" in out


def test_cfo_commentary_returns_body():
    text = "CFO commentary body content with financial detail and guidance context."
    ext = SectionExtractor()
    out = ext.extract_sections(_make_doc("cfo_commentary", text))
    assert out.get("cfo_commentary_body") == text
