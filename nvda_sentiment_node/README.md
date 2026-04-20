# NVIDIA Sentiment Node MVP

Standalone Python component that returns a single **NVIDIA market sentiment
score** (0-100) plus a breakdown of the underlying components.

## What it does

- Ingests official NVIDIA and SEC source material (10-K, 10-Q, earnings 8-K,
  press releases, webcast transcripts, CFO commentary).
- Extracts the most relevant sections (MD&A, risk factors, guidance,
  prepared remarks, Q&A, etc.).
- Scores tone using a small finance lexicon and, optionally, FinBERT.
- Compares current tone with prior comparable period (NVIDIA fiscal
  calendar).
- Overlays a small investor context layer (AAII + VIX).
- Emits a single interpretable score, English signals, and confidence.

## Install

```bash
# Core install (lexicon-only scoring, no model weights)
pip install -e .

# FinBERT install (adds transformers + torch, ~1GB download)
pip install -e ".[finbert]"
```

Python 3.11+ required.

## CLI

```bash
nvda-sentiment run --as-of-date 2026-04-20 --lookback-quarters 4
```

Or from source:

```bash
python -m nvda_sentiment.node run --as-of-date 2026-04-20
```

## Python usage

```python
from datetime import date
from nvda_sentiment import NVDASentimentNode, SentimentRequest

node = NVDASentimentNode()
response = node.run(SentimentRequest(
    ticker="NVDA",
    as_of_date=date(2026, 4, 20),
    lookback_quarters=4,
    include_market_context=True,
    use_cache=True,
))
print(response.model_dump_json(indent=2))
```

## Output example

```json
{
  "ticker": "NVDA",
  "as_of_date": "2026-04-20",
  "market_sentiment_score": 63.7,
  "label": "mildly bullish",
  "confidence": 0.80,
  "components": {
    "filing_tone": 0.29,
    "filing_delta": 0.08,
    "guidance_tone": 0.19,
    "investor_context": 0.06
  },
  "signals": [
    "Management tone is positive across recent official materials",
    "Tone improved versus the prior comparable period",
    "Risk-factor language was substantively stable versus prior filing",
    "Forward-looking guidance language is constructive",
    "Broader investor risk appetite is neutral"
  ],
  "source_coverage": { "10k_count": 1, "10q_count": 3, "8k_count": 1 },
  "metadata": { "node_version": "0.1.0", "warnings": [] }
}
```

## Known limitations

- Section extraction is heuristic, not perfect.
- Fiscal-quarter bucketing uses NVIDIA's hardcoded calendar.
- Lexicon is intentionally small; FinBERT is the primary semantic load.
- Investor context is broad market, not NVDA-specific.
- SEC submissions JSON pagination is not followed (lookback effectively
  capped at 8 quarters - plenty of NVIDIA filings fit in the base file).
- AAII scraping is brittle - node tolerates its absence cleanly.

## SEC user agent

Edit `Settings.sec_user_agent` in `nvda_sentiment/config.py` (or pass a
custom `Settings` to the node) to include **your** real contact email.
SEC blocks requests without one.
