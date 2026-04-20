# NVIDIA Sentiment Node

A standalone Python component that turns NVIDIA's official disclosures into a
single, interpretable **market sentiment score from 0–100**, with a structured
breakdown of *why* it landed there.

The node is designed to drop into a larger research or trading pipeline as a
pure function: given a date, it returns a schema-validated JSON response. No
state outside the filesystem cache, no side effects beyond HTTP fetches.

---

## Part 1 — Qualitative Overview

### What it is

`NVDASentimentNode` consumes official NVIDIA and SEC source material and
produces a sentiment reading of the form:

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
    "Forward-looking guidance language is constructive",
    "Broader investor risk appetite is neutral"
  ],
  "source_coverage": {"10k_count": 1, "10q_count": 3, "8k_count": 1, ...},
  "metadata": {"node_version": "0.1.0", "warnings": []}
}
```

The score is a weighted blend of four orthogonal components, plus a confidence
figure so downstream consumers can weight or discard the reading.

### Why it exists

Most "sentiment" tools for public equities start with social media and news
headlines — noisy, easily gamed, and semantically distant from the mechanical
drivers of the stock. This node deliberately inverts that premise: it reads
**only primary-source corporate disclosures** (the same documents an analyst
would read) and scores the *language management themselves chose* to use.

That makes the output interpretable (every signal ties back to a specific
filing section), auditable (all sources are permanent URLs), and stable
(the same inputs always produce the same output — see §Determinism below).

### What it reads

| Source | Where it comes from | What gets scored |
|---|---|---|
| 10-K annual filing | SEC EDGAR | MD&A, outlook/guidance |
| 10-Q quarterly filing | SEC EDGAR | MD&A, outlook/guidance |
| Earnings 8-K | SEC EDGAR (filtered by keyword) | earnings summary, outlook |
| Press release | NVIDIA IR site | headline, financial highlights, outlook |
| Earnings-call transcript | NVIDIA IR site | prepared remarks, Q&A |
| CFO commentary | NVIDIA IR site | full body |
| AAII sentiment survey | aaii.com | investor-context overlay |
| VIX daily close | FRED (St. Louis Fed) | investor-context overlay |

Risk-factors language is extracted but kept **out of absolute tone** — it only
contributes as a *delta* versus the prior comparable filing, because reading
the risk-factors section in isolation would bias every score negative.

### What it outputs

Four numeric components in `[-1, 1]`:

- **filing_tone (50%)** — document-weighted absolute tone of all current filings.
- **filing_delta (25%)** — change versus the prior comparable period
  (same fiscal quarter last year), 70% tone delta + 30% risk-language delta.
- **guidance_tone (15%)** — isolated tone of outlook/guidance sections only,
  since forward-looking language moves the stock most.
- **investor_context (10%)** — small AAII+VIX overlay; gracefully zero if unavailable.

These collapse to a single 0–100 score, a label (`bearish` → `strongly bullish`),
and a `confidence` in `[0.10, 0.95]` driven by source coverage, extraction
success rate, and whether the YoY comparison actually matched.

### What it is *not*

- Not a price predictor. It scores language, not returns.
- Not a news aggregator. Social/news feeds are out of scope by design.
- Not a multi-ticker tool. NVIDIA's fiscal calendar is hardcoded; extending
  to other tickers means replacing the fiscal-bucket logic (see §Extending).

---

## Part 2 — Using the Node

### Install

Python 3.11+ required.

```bash
cd nvda_sentiment_node
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional FinBERT extra (adds `transformers` + `torch`, ~1 GB download). Without
it, the node falls back to lexicon-only scoring and surfaces a warning:

```bash
pip install -e '.[finbert]'
```

**SEC requires a real contact email** in the User-Agent header or it will
throttle/block you. Edit `Settings.sec_user_agent` in
`nvda_sentiment/config.py:104` or construct a custom `Settings` and pass it in.

### Command line

```bash
nvda-sentiment --as-of-date 2026-04-20 --lookback-quarters 4
```

Flags:

- `--as-of-date YYYY-MM-DD` — defaults to today.
- `--lookback-quarters N` — 1 to 8, default 4.
- `--no-include-market-context` — skip the AAII/VIX overlay.
- `--no-use-cache` — bypass cache reads (cache is still *written*).

Output is JSON on stdout. Pipe to `jq` or redirect to a file.

### Python API

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

`response` is a pydantic `SentimentResponse`; treat it as a validated object
or dump to dict/JSON.

### Configuration

All tunables live in `nvda_sentiment/config.py`. Construct your own `Settings`
and pass it to the node to override any of them:

```python
from pathlib import Path
from nvda_sentiment import NVDASentimentNode
from nvda_sentiment.config import Settings

settings = Settings(
    sec_user_agent="Your Name you@company.com",
    cache_dir=Path("/var/cache/nvda_sentiment"),
    request_timeout_seconds=60,
)
node = NVDASentimentNode(settings)
```

---

## Part 3 — Architectural Breakdown

### Package layout

```
nvda_sentiment/
├── node.py                  # NVDASentimentNode orchestration + Typer CLI
├── schemas.py               # SentimentRequest / SentimentResponse / internals
├── config.py                # Settings, URLs, weights, keywords
├── adapters/                # External-I/O boundary
│   ├── sec_api.py           #   SEC EDGAR filings
│   ├── nvidia_ir.py         #   NVIDIA investor.nvidia.com
│   └── market_context.py    #   AAII survey + FRED VIX
├── parsers/
│   ├── html_to_text.py      # strip-HTML, whitespace normalization
│   └── section_extractor.py # heading-regex section carving
├── scorers/
│   ├── lexicon.py           # small finance lexicon (pos/neg/uncertainty/litigious)
│   ├── finbert_scorer.py    # optional ProsusAI/finbert inference
│   ├── section_scorer.py    # blends finbert + lexicon per section
│   └── composite.py         # document → filing_tone / guidance_tone / final score
├── features/
│   ├── filing_delta.py      # YoY comparison, fiscal-bucket math, risk-delta
│   ├── confidence.py        # §28 confidence arithmetic
│   └── signal_builder.py    # English rationale strings
└── utils/
    ├── cache.py             # filesystem cache with permanent tier
    ├── rate_limiter.py      # token-bucket throttler for SEC 10 req/s
    ├── text.py, dates.py, logging.py
```

### Data flow

```
          SentimentRequest(as_of_date, lookback_quarters, …)
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 1. Adapters fetch metadata and raw HTML           │
   │    SECAdapter.get_relevant_filings()              │
   │    NvidiaIRAdapter.get_quarterly_results_docs()   │
   │    MarketContextAdapter.get_combined_score()      │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 2. Dedupe (by URL and by (title, filed_at))      │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 3. Parse: html_to_clean_text → SectionExtractor  │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 4. Score each section: FinBERT probs + lexicon    │
   │    → SectionScore → DocumentScore                 │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 5. Feature assembly                               │
   │    filing_tone     (all docs, excl. risk)         │
   │    filing_delta    (curr vs YoY, 0.7·tone + 0.3·risk) │
   │    guidance_tone   (outlook sections only)        │
   │    investor_context (AAII + VIX, optional)        │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 6. Composite → 0–100 + label + confidence + signals │
   └──────────────────────────────────────────────────┘
                          │
                          ▼
                 SentimentResponse (JSON)
```

### Dependency injection

`NVDASentimentNode.__init__` accepts every adapter and scorer as an optional
keyword argument. This is the seam that makes the test suite hermetic:
`tests/test_end_to_end.py` and `tests/test_determinism.py` pass `FakeSEC`,
`FakeIR`, and `FakeMarket` objects, so the full pipeline runs with zero
network I/O.

```python
NVDASentimentNode(
    settings=settings,
    sec_adapter=FakeSEC(docs, html_map),
    ir_adapter=FakeIR([], {}),
    market_context=FakeMarket(score=0.2, ok=True),
)
```

The only contract a fake must satisfy is the method signatures the node calls
(`get_relevant_filings`, `fetch_filing_html`, `get_quarterly_results_documents`,
`fetch_document_html`, `get_combined_score`). Pydantic validation on
`SourceDocument` enforces the rest.

### Graceful degradation

Every external dependency has a fallback path:

| Failure | Behavior |
|---|---|
| SEC adapter raises | Warning appended, `sec_docs=[]`, pipeline continues |
| IR adapter raises | Warning appended, `ir_docs=[]`, pipeline continues |
| AAII 403 / unreachable | `investor_context=0`, flagged as unavailable |
| VIX CSV unparseable | `investor_context=0`, flagged as unavailable |
| FinBERT import fails | Module-level `FINBERT_AVAILABLE=False`, section scoring uses lexicon alone, warning surfaced |
| Individual fetch fails | `fetch_failures` incremented (feeds confidence penalty), doc dropped |
| All documents fail | `_empty_response` returns neutral-50.0 / confidence-0.10 with warnings |

---

## Part 4 — Technical Deep-Dive

### Scoring pipeline

**1. Section scoring.** For each extracted section:

```
section_score = 0.5·finbert_score + 0.5·lexicon_score − uncertainty_penalty
```

- `finbert_score = P(positive) − P(negative)` averaged across sentences (or 0 if FinBERT is absent).
- `lexicon_score` uses the Loughran-McDonald-style word set in `scorers/lexicon.py` (normalized positive-vs-negative hit count).
- `uncertainty_penalty` is a small deduction when uncertainty/litigious vocabulary dominates.

When FinBERT is unavailable, the blend collapses to pure lexicon — explicitly
handled in `section_scorer.py` so section scores remain in the same range.

**2. Document scoring.** Sections aggregate via per-document-type weights
(`config.py:69`). Example: a 10-Q is 70% MD&A + 30% outlook; a transcript is
70% prepared remarks + 30% Q&A.

**3. `filing_tone`.** Document scores aggregated by `DOCUMENT_TYPE_WEIGHTS`
(10-K=0.30, 10-Q=0.30, press_release=0.10, transcript=0.10, etc.), with
**risk_factors sections excluded** — they would anchor every reading
negative (§22.1).

**4. `filing_delta`.** For each document type with a prior-year match,
compute `(curr_tone − prior_tone)`; average across matches; combine with a
separately-computed `risk_delta`:

```
filing_delta = 0.70·tone_delta + 0.30·risk_delta
```

If no YoY match exists, `yoy_matched=False` and this component is 0 (and
confidence takes a hit).

**5. `guidance_tone`.** Averaged scores of outlook/guidance sections **only**.
Recomputed independently because forward-looking language is the most
price-relevant.

**6. `investor_context`.** `0.6·AAII + 0.4·VIX`, both clipped to `[-1, 1]`.
VIX is bucketed: ≤15 → +0.50, ≤20 → +0.25, ≤25 → 0, ≤30 → −0.25, else −0.50.

**7. Final score.**

```
raw = 0.50·filing_tone + 0.25·filing_delta + 0.15·guidance_tone + 0.10·investor_context
score_0_100 = 50 + 50·clip(raw, -1, 1)
```

Label thresholds: <35 bearish, <45 mildly bearish, <55 neutral,
<65 mildly bullish, <80 bullish, else strongly bullish.

### NVIDIA fiscal calendar

NVIDIA's fiscal year ends in **late January**, so calendar-month → fiscal-quarter
mapping is non-obvious. The logic lives in `features/filing_delta.py`:

| Filing month | Fiscal quarter (NVIDIA) |
|---|---|
| Feb–Apr | Q1 of next-year's FY label (e.g. Feb 2026 → FY27Q1) |
| May–Jul | Q2 |
| Aug–Oct | Q3 |
| Nov–Dec | Q4 of next-year's FY label |
| Jan | Q4 of **same** calendar year's FY label (fiscal year end) |

Test coverage in `tests/test_filing_delta.py` nails the corner cases
(Jan 2026 → FY26Q4, Jan 2027 → FY27Q4, Nov 2026 → FY27Q4). The spec's
original pseudocode was subtly wrong here; the corrected implementation is
authoritative.

### Confidence

`features/confidence.py` — base `0.55`, bonuses capped at `+0.40`, penalties
unbounded, clamped to `[0.10, 0.95]`. An empty-documents run short-circuits
to `0.10`.

Bonuses: multiple doc types, high extraction success rate, FinBERT available,
investor context available, guidance has multiple sources, filing-delta computed,
YoY match found. Penalties: no delta, no investor context, fetch failures >0,
extraction success rate <100%.

### Caching

Two-tier filesystem cache (`utils/cache.py`):

- **TTL tier** (default 24h): submissions JSON, IR index pages, AAII, VIX.
- **Permanent tier**: individual SEC filing HTML (immutable once filed) and
  IR press releases / transcripts.

Cache keys are SHA-256 of the URL. `SentimentRequest.use_cache=False`
bypasses reads across all adapters but still writes (so a forced refresh
repopulates rather than skipping).

### Rate limiting

SEC EDGAR enforces 10 requests/second. `utils/rate_limiter.py` is a simple
sleep-based throttle, initialized at 8 req/sec (headroom under the cap).
Applied only on cache *misses*.

### Determinism

Same inputs → byte-identical output (modulo the `generated_at` wall clock
timestamp). Verified by `tests/test_determinism.py`, which runs the full
pipeline twice and asserts `resp_a == resp_b`. Guarantees:

- FinBERT runs in `eval()` mode under `torch.no_grad()` with
  `torch.use_deterministic_algorithms(True)` and `TOKENIZERS_PARALLELISM=false`.
- No iteration over sets; doc ordering is by filing date (a total order).
- No randomness anywhere in the scoring stack.

---

## Part 5 — Development

### Running the test suite

```bash
source .venv/bin/activate
python -m pytest -q
```

47 tests + 1 skipped (FinBERT, runs only if `transformers`+`torch` installed).
All tests are offline — no network access required.

### Adding a new source

1. Create `adapters/your_source.py` with two methods:
   `get_<something>_documents(lookback_quarters) -> List[SourceDocument]`
   and `fetch_document_html(doc) -> str`.
2. Add a new `SourceType` literal to `schemas.py:40`.
3. Add section weights in `config.py` (see `PRESS_RELEASE_SECTION_WEIGHTS`
   as a template).
4. Add the new type to `DOCUMENT_TYPE_WEIGHTS` (all weights must still sum to 1.0).
5. Wire it into `NVDASentimentNode.__init__` and `node.run()` step 1.
6. Add a section-detection branch to `parsers/section_extractor.py`.
7. Update `_build_source_coverage` in `node.py`.

### Replacing the scorer

`SectionScorer` is swappable via the `section_scorer` kwarg on
`NVDASentimentNode`. A conforming scorer needs one method:

```python
def score_section(self, name: str, text: str) -> SectionScore: ...
```

and one property `finbert_available: bool` (used only for confidence accounting).

### Extending beyond NVIDIA

The node is single-company by design (spec §47.3). To generalize:

1. Parameterize `NVIDIA_CIK` / `NVIDIA_CIK_PADDED` in `config.py`.
2. Replace `nvda_fiscal_bucket()` in `features/filing_delta.py` with a
   per-company fiscal calendar (or a general calendar-quarter fallback).
3. The NVIDIA IR adapter is page-structure-specific; most issuers will need
   a replacement (or skip IR and rely on 8-K/10-Q alone).

### Known limitations

- Section extraction is heuristic regex on headings — it handles standard SEC
  filing structure well but can mis-segment unusual layouts.
- SEC submissions JSON pagination is not followed; effective lookback is capped
  at ~1000 most-recent filings, which comfortably covers 8 quarters for NVIDIA.
- AAII scraping is brittle (site occasionally returns 403); the node tolerates
  its absence but loses a small bit of signal when it's down.
- Lexicon is intentionally small; FinBERT carries most of the semantic load
  when installed.
- Fiscal calendar is hardcoded to NVIDIA.

### Where the spec lives

The full design spec is `nvda_sentiment_final.md` in the repo root. Section
numbers referenced in code comments (e.g. `§22.1`, `§28.4`) point back to it.
