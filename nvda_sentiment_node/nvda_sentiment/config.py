"""Configuration and constants for the NVDA sentiment node.

All fixed weights, URLs, and tunable defaults live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Source URLs (§55)
# --------------------------------------------------------------------------- #

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0001045810.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
NVIDIA_IR_HOME = "https://investor.nvidia.com/home/default.aspx"
NVIDIA_QUARTERLY_RESULTS_URL = (
    "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"
)
NVIDIA_FINANCIAL_REPORTS_URL = (
    "https://investor.nvidia.com/financial-info/financial-reports/"
)
AAII_URL = "https://www.aaii.com/sentimentsurvey"
FRED_VIX_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


# --------------------------------------------------------------------------- #
# NVIDIA identifiers (§13.2)
# --------------------------------------------------------------------------- #

NVIDIA_TICKER = "NVDA"
NVIDIA_COMPANY_NAME = "NVIDIA CORP"
NVIDIA_CIK = "1045810"
NVIDIA_CIK_PADDED = "0001045810"


# --------------------------------------------------------------------------- #
# Weights (§22, §26, §27)
# --------------------------------------------------------------------------- #

FILING_SECTION_WEIGHTS = {
    "mda": 0.70,
    "outlook_guidance": 0.30,
    # risk_factors intentionally excluded (§22.1, §19.8)
}

EIGHT_K_SECTION_WEIGHTS = {
    "earnings_summary": 0.70,
    "outlook_guidance": 0.30,
}

PRESS_RELEASE_SECTION_WEIGHTS = {
    "headline_and_summary": 0.40,
    "financial_highlights": 0.35,
    "outlook_guidance": 0.25,
}

TRANSCRIPT_SECTION_WEIGHTS = {
    "prepared_remarks": 0.70,
    "qa": 0.30,
}

CFO_COMMENTARY_SECTION_WEIGHTS = {
    "cfo_commentary_body": 1.00,
}

DOCUMENT_TYPE_WEIGHTS = {
    "10-K": 0.30,
    "10-Q": 0.30,
    "8-K": 0.10,
    "press_release": 0.10,
    "transcript": 0.10,
    "cfo_commentary": 0.10,
}

FINAL_COMPONENT_WEIGHTS = {
    "filing_tone": 0.50,
    "filing_delta": 0.25,
    "guidance_tone": 0.15,
    "investor_context": 0.10,
}

# Leadership branch weights (renormalized from old 0.50/0.25/0.15 = 0.90).
LEADERSHIP_COMPONENT_WEIGHTS = {
    "filing_tone":   0.50 / 0.90,    # 0.5556
    "filing_delta":  0.25 / 0.90,    # 0.2778
    "guidance_tone": 0.15 / 0.90,    # 0.1667
}
assert abs(sum(LEADERSHIP_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-6

# Investor branch weights (§7, Appendix C).
INVESTOR_COMPONENT_WEIGHTS = {
    "options_flow":   0.30,
    "short_interest": 0.15,
    "analyst_signal": 0.25,
    "social":         0.15,
    "broad_market":   0.15,
}
assert abs(sum(INVESTOR_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-6

# Combiner weights (§7.2).
COMBINER_DEFAULTS = {
    "alpha":          0.55,    # leadership weight in base blend
    "lambda_neg":     0.25,    # bearish-divergence penalty magnitude (L>I)
    "lambda_pos":     0.10,    # contrarian premium magnitude (I>L)
    "agreement_zero": True,    # when signs agree, divergence adjustment = 0
}


# --------------------------------------------------------------------------- #
# 8-K filter keywords (§13.6)
# --------------------------------------------------------------------------- #

EIGHT_K_EARNINGS_KEYWORDS = (
    "results of operations",
    "earnings",
    "financial results",
    "quarterly results",
    "exhibit 99.1",
)


@dataclass
class Settings:
    """Runtime settings — one source of truth for tunable values."""

    # MUST follow SEC-required format: "Name ContactEmail"
    sec_user_agent: str = "NVDA Sentiment Node contact@example.com"
    sec_max_requests_per_second: float = 8.0  # SEC hard limit is 10/sec

    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    request_timeout_seconds: int = 30
    default_lookback_quarters: int = 4

    finbert_model_name: str = "ProsusAI/finbert"
    finbert_model_revision: str = "main"

    filing_tone_weight: float = FINAL_COMPONENT_WEIGHTS["filing_tone"]
    filing_delta_weight: float = FINAL_COMPONENT_WEIGHTS["filing_delta"]
    guidance_tone_weight: float = FINAL_COMPONENT_WEIGHTS["guidance_tone"]
    investor_context_weight: float = FINAL_COMPONENT_WEIGHTS["investor_context"]

    # Combiner overrides (§6.1.3, §7.3).
    combiner_alpha:      float = COMBINER_DEFAULTS["alpha"]
    combiner_lambda_neg: float = COMBINER_DEFAULTS["lambda_neg"]
    combiner_lambda_pos: float = COMBINER_DEFAULTS["lambda_pos"]

    node_version: str = "0.2.0"

    def __post_init__(self) -> None:
        # Ensure cache dir is a Path and exists (§38.2)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
