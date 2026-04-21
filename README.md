# NVIDIA Sentiment Node

A standalone, deterministic Python component that converts NVIDIA's official
disclosures *and* live investor-positioning data into a single, interpretable
**market-sentiment score from 0–100**, with a full audit trail of the signals
that drove it. Designed as a drop-in node for a larger research or trading
pipeline: given a request, it returns a schema-validated JSON response. No
hidden state, no external side effects beyond HTTP fetches and a local
SQLite log.

---

## 1. Executive Summary

### 1.1 The thesis

Most public-equity "sentiment" products start and end with social-media
chatter or news headlines — a noisy surface layer that is easily gamed and
semantically distant from the mechanical drivers of the stock. This node
inverts that premise. It scores sentiment along **two orthogonal axes at once**,
then uses the disagreement between them as additional signal:

- **Leadership** — what the company itself is saying. Primary-source corporate
  disclosures (10-K, 10-Q, 8-K filings from SEC EDGAR; press releases,
  earnings-call transcripts, and CFO commentary from NVIDIA's IR site) are
  scored with a blend of FinBERT (a finance-domain transformer) and a
  hand-curated Loughran-McDonald-style lexicon.
- **Investor** — what the market is doing. Live options-chain positioning
  (put/call ratio, implied volatility), short interest and days-to-cover,
  analyst price-target upside and recommendation distribution, 7-day social
  chatter across Reddit (WSB, stocks, investing) and StockTwits (again
  FinBERT-scored), plus a broad-market VIX overlay.

A divergence-aware combiner then blends the two with `α·L + (1-α)·I` and
applies an asymmetric adjustment: **bearish divergence** (management bullish,
market bearish — often the prelude to a disappointment) gets penalized by
`λ_neg`; **contrarian bullish divergence** (management bearish, market bullish
— often capitulation bottoming) earns a small premium `λ_pos`. Signs that
agree pass through cleanly.

The resulting score is interpretable (every signal ties back to a specific
filing section or adapter), auditable (all sources are permanent URLs or
named APIs), and stable (same inputs produce byte-identical outputs —
see §6.3 Determinism).

### 1.2 Core feature set

| Feature | Summary |
|---|---|
| Dual-branch analysis | Independent leadership + investor pipelines, combined with divergence-aware math |
| FinBERT + Lexicon blend | 50/50 average per section; lexicon-only fallback if torch is unavailable |
| Five investor adapters | Options flow, short interest, analyst signals, social, broad market (VIX) |
| Divergence combiner | α=0.55 base blend; λ_neg=0.25 bearish penalty; λ_pos=0.10 contrarian premium |
| Graceful degradation | Every external source has a neutral fallback; pipeline never crashes |
| Deterministic output | Same inputs → identical JSON (modulo wall-clock timestamp), verified in test suite |
| Two-tier filesystem cache | TTL tier (~24h) for volatile data, permanent tier for immutable SEC filings |
| SEC-compliant rate limiting | 8 req/s throttle (headroom under the 10/s hard limit), real-email User-Agent |
| SQLite longitudinal log | Daily snapshots upserted on `(ticker, as_of_date)`; cross-day accumulation |
| Grid-sweep log | 45 `(α, λ_neg, λ_pos)` rows per sweep, persisted for multi-day regression |
| Tier-2 overlay tuning | `--write-best-alpha` writes a file-backed α override the node auto-applies |
| Confidence arithmetic | Coverage + extraction + degradation penalties → `[0.10, 0.95]` confidence figure |
| Dependency-injected seams | Every adapter and scorer is replaceable; test suite runs fully offline |
| Calibration harness | `tools/calibrate_weights.py` sweeps (α, λ_neg, λ_pos) against EPS-revision direction |
| Typer CLI | `nvda-sentiment run` for same-day scoring; `nvda-sentiment history` for the log dump |

### 1.3 What you get back

```json
{
  "ticker": "NVDA",
  "as_of_date": "2026-04-21",
  "market_sentiment_score": 63.7,
  "combined_score": 63.7,
  "leadership_score": 69.5,
  "investor_score": 54.0,
  "divergence": 15.5,
  "label": "mildly bullish",
  "confidence": 0.80,
  "components": {
    "filing_tone": 0.290, "filing_delta": 0.080, "guidance_tone": 0.190,
    "options_flow": 0.120, "short_interest": -0.050, "analyst_signal": 0.310,
    "social": 0.040, "broad_market": 0.000,
    "leadership_component": 0.390, "investor_component": 0.080
  },
  "signals": [
    "Management tone is positive across recent official materials",
    "Forward-looking guidance language is constructive",
    "Options market is moderately bullish (put/call < 0.9)",
    "Analyst targets imply mid-teens upside with a buy-skewed distribution",
    "Leadership runs ahead of the market — moderate bearish-divergence penalty applied"
  ],
  "source_coverage": {"10k_count": 1, "10q_count": 3, "8k_count": 1,
                      "earnings_release_count": 4, "transcript_count": 3,
                      "cfo_commentary_count": 1},
  "metadata": {"node_version": "0.2.0", "warnings": [],
               "generated_at": "2026-04-21T15:03:11Z"}
}
```

### 1.4 Same-day-only constraint

The pipeline always stamps the response with `date.today()` — there is no
`as_of_date` in the request. Investor-branch adapters fetch **live** data:
yfinance option chains, current short interest, today's analyst snapshot,
the last 7 days of social posts, the latest VIX print. None of those free
APIs expose point-in-time historical snapshots, so applying an arbitrary
lookback date asymmetrically (leadership only) would produce an incoherent
comparison (filings from 2023 blended with today's market state).

Both branches are therefore held to the same constraint: the node answers
"what is sentiment right now," not "what was sentiment on date X." For
longitudinal analysis the node ships with a SQLite log (§4.5) that
accumulates one row per `(ticker, as_of_date)` across daily runs. Paid
point-in-time vendors (FactSet, Refinitiv, S&P/I/B/E/S) would unlock a true
backtest mode — see §7.3.

---

## 2. System Architecture

### 2.1 Dual-branch topology

```
                   ┌─────────────────────────────────────────┐
                   │          SentimentRequest               │
                   │ (ticker, lookback_quarters, flags…)     │
                   └────────────────┬────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 ┌────────────────────┐                             ┌────────────────────┐
 │ LEADERSHIP BRANCH  │                             │ INVESTOR BRANCH    │
 │  SECAdapter        │                             │  OptionsFlow       │
 │  NvidiaIRAdapter   │                             │  ShortInterest     │
 │                    │                             │  AnalystSignals    │
 │  → dedupe, parse   │                             │  Social            │
 │  → FinBERT+Lexicon │                             │  BroadMarket(VIX)  │
 │  → section scores  │                             │                    │
 │  → document scores │                             │  → 5 sub-scores    │
 │  → filing_tone     │                             │  → weighted mean   │
 │  → filing_delta    │                             │    w/ pro-rata     │
 │  → guidance_tone   │                             │    redistribution  │
 │                    │                             │                    │
 │ leadership_component│                             │ investor_component │
 └──────────┬─────────┘                             └─────────┬──────────┘
            │                                                 │
            └───────────────────────┬─────────────────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │  DIVERGENCE-AWARE COMBINER  │
                     │   base = α·L + (1-α)·I       │
                     │   signs agree → pass through │
                     │   L>I  → adjust −λ_neg·|L-I| │
                     │   I>L  → adjust +λ_pos·|L-I| │
                     │   clip to [-1, +1], ×50 + 50 │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │  combined_score  0..100     │
                     │  + label + confidence       │
                     │  + signals + source_coverage│
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │ SQLite upsert (optional)    │
                     │ data/history.sqlite3        │
                     │ PK=(ticker, as_of_date)     │
                     └──────────────┬──────────────┘
                                    ▼
                         SentimentResponse (JSON)
```

### 2.2 Package layout

```
NVIDIA_Sentiment_Analysis/
├── pyproject.toml                     # single source of truth for deps
├── .env.example                       # SEC User-Agent + cache dir env vars
├── README.md                          # this file
├── SA_NewArch.md                      # original architecture spec
├── data/
│   └── history.sqlite3                # accumulating daily-snapshot log (gitignored)
├── nvda_sentiment/
│   ├── node.py                        # NVDASentimentNode + Typer CLI
│   ├── schemas.py                     # Pydantic models (Request/Response/internals)
│   ├── config.py                      # Settings, weights, thresholds, keywords
│   ├── persistence.py                 # SQLite logger (upsert, load_history)
│   ├── adapters/                      # External-I/O boundary
│   │   ├── sec_api.py                 #   SEC EDGAR submissions + filings
│   │   ├── nvidia_ir.py               #   NVIDIA IR site scraper
│   │   └── investor/
│   │       ├── options_flow.py        #   yfinance option chains → put/call, IV
│   │       ├── short_interest.py      #   yfinance short % float, days-to-cover
│   │       ├── analyst_signals.py     #   yfinance targets + recommendations
│   │       ├── social.py              #   Reddit + StockTwits (FinBERT-scored)
│   │       └── broad_market.py        #   VIX overlay
│   ├── parsers/
│   │   ├── html_to_text.py            # strip-HTML + whitespace normalization
│   │   └── section_extractor.py       # heading-regex section carving
│   ├── scorers/
│   │   ├── lexicon.py                 # pos/neg/uncertainty/litigious word buckets
│   │   ├── finbert_scorer.py          # ProsusAI/finbert, torch.no_grad + deterministic
│   │   ├── section_scorer.py          # 0.5·finbert + 0.5·lexicon − uncertainty
│   │   └── composite.py               # section→doc→filing_tone / guidance_tone
│   ├── features/
│   │   ├── filing_delta.py            # YoY comparison w/ NVIDIA fiscal calendar
│   │   ├── investor_branch.py         # 5-adapter aggregator + pro-rata redistribution
│   │   ├── combiner.py                # divergence-aware L+I combiner
│   │   ├── confidence.py              # bonus/penalty arithmetic → [0.10, 0.95]
│   │   └── signal_builder.py          # English rationale strings
│   └── utils/
│       ├── cache.py                   # two-tier filesystem cache
│       ├── rate_limiter.py            # SEC token-bucket throttler
│       └── text.py, dates.py, logging.py
├── tools/
│   └── calibrate_weights.py           # α / λ_neg / λ_pos sweep harness
└── tests/                             # 105 offline tests
    ├── test_end_to_end_dual_branch.py
    ├── test_persistence.py
    ├── test_options_flow_adapter.py
    ├── test_short_interest_adapter.py
    ├── test_analyst_signals_adapter.py
    ├── test_social_adapter.py
    ├── test_combiner.py
    ├── test_investor_branch.py
    ├── test_determinism.py
    └── … (see §6)
```

### 2.3 Dependency injection

`NVDASentimentNode.__init__` accepts every adapter and scorer as an optional
keyword argument:

```python
NVDASentimentNode(
    settings=settings,
    sec_adapter=FakeSEC(docs, html_map),
    ir_adapter=FakeIR([], {}),
    investor_branch=InvestorBranch(
        settings,
        options=FakeOptions(...),
        shorts=FakeShorts(...),
        analyst=FakeAnalyst(...),
        social=FakeSocial(...),
        broad_market=FakeMarket(...),
    ),
    section_extractor=SectionExtractor(),
    section_scorer=SectionScorer(settings, finbert=..., lexicon=...),
)
```

This is the seam that makes the test suite hermetic: `test_end_to_end_dual_branch.py`,
`test_determinism.py`, and the per-adapter tests all inject fakes, so the full
pipeline runs with zero network I/O. A backward-compatible `market_context=` kwarg
is still accepted (with a `DeprecationWarning`) and internally upgraded to an
`InvestorBranch(..., broad_market=market_context)`.

### 2.4 Graceful degradation matrix

| Failure mode | Behavior |
|---|---|
| SEC adapter raises | Warning appended, `sec_docs=[]`, pipeline continues |
| IR adapter raises | Warning appended, `ir_docs=[]`, pipeline continues |
| Individual SEC/IR fetch fails | `fetch_failures` incremented (confidence penalty), doc dropped |
| No documents at all | `_empty_response` returns neutral 50.0 / confidence 0.10 with warnings |
| Any investor sub-adapter raises | That sub returns `ok=False, score=0.0`; its weight reallocates pro-rata across the remaining usable sub-scores |
| All investor sub-adapters fail | `investor_branch_ok=False` → combiner degrades to leadership-only fallback (`combined := leadership_score`) |
| FinBERT/torch unavailable at import | `FINBERT_AVAILABLE=False`, section scoring uses lexicon alone, warning surfaced |
| yfinance unavailable at import | Four investor sub-adapters hard-fail cleanly and return neutral sub-scores |
| AAII 403 / FRED unreachable | `broad_market` sub returns `ok=False`; combined sub-scores continue |
| Cache directory read-only | Cache writes fail silently; reads still attempt direct fetch |

The invariant: **no runtime error in any external dependency can crash the node.**
The worst case is a response with `confidence=0.10`, `label="neutral"`, and a
populated `warnings` array.

---

## 3. Data Sources & Lookback Windows

### 3.1 Leadership side (rolling ~1 year of filings)

| Source | Provider | Lookback | Sections scored |
|---|---|---|---|
| 10-K annual filing | SEC EDGAR | 1 filing (current-year annual) | MD&A, outlook/guidance |
| 10-Q quarterly filing | SEC EDGAR | Last 4 quarters (configurable 1–8) | MD&A, outlook/guidance |
| 8-K material events | SEC EDGAR, keyword-filtered | Last 4 (configurable 1–8) | full body of earnings-related 8-Ks |
| Press release | NVIDIA IR site | ≈ `lookback_quarters × 3` docs | headline, financial highlights, outlook |
| Earnings-call transcript | NVIDIA IR site | Same window | prepared remarks (0.7), Q&A (0.3) |
| CFO commentary | NVIDIA IR site | Same window | full body |

Risk-factors language is extracted but **kept out of absolute tone** — it only
contributes as a *delta* versus the prior comparable filing, because reading
risk-factors in isolation would bias every score negative.

**Hard ceiling:** the SEC submissions JSON returns the most-recent ~1000
filings without pagination, which comfortably covers the 8-quarter max. For
deeper lookback, `filings.files` pagination would need to be followed.

### 3.2 Investor side (today + last 7 days)

| Sub-adapter | Provider | Temporal window | Signal extracted |
|---|---|---|---|
| `options_flow` | yfinance | **Today's chain**; put/call from nearest expiry ≥ 21 days out; IV averaged from next 3 expiries ≥ 7 days out | P/C ratio bucketed, IV bucketed |
| `short_interest` | yfinance | Latest reported figures (FINRA updates bi-monthly) | short % of float, days-to-cover |
| `analyst_signals` | yfinance | Current snapshot | price-target upside, buy/hold/sell distribution, analyst count |
| `social` | Reddit JSON API + StockTwits | **Last 7 days** of Reddit (WSB, stocks, investing — up to 50 posts each, `sort=new&t=week`); current StockTwits stream | FinBERT-scored, clipped and weighted |
| `broad_market` | FRED VIX CSV (+ optional AAII) | Latest VIX close | bucketed: ≤15→+0.50, ≤20→+0.25, ≤25→0, ≤30→−0.25, else −0.50 |

The investor branch is therefore a **right-now positioning snapshot** plus a
7-day chatter window, not a history. It captures what's happening in the
market today, while the leadership branch captures the cumulative tone of
what NVIDIA has communicated over the last ~12 months.

### 3.3 Cache tiers

| Tier | TTL | Contents |
|---|---|---|
| Permanent | never expires | Individual SEC filing HTML, IR press releases, transcripts, CFO commentary (all immutable once published) |
| TTL (default 24h) | time-based | SEC submissions JSON, IR index pages, AAII, VIX, yfinance snapshots, Reddit/StockTwits responses |

Cache keys are SHA-256 of the URL. `SentimentRequest.use_cache=False` bypasses
**reads** across all adapters but still writes — a forced refresh repopulates
the cache rather than skipping it.

### 3.4 SEC compliance

SEC EDGAR enforces 10 requests/second and requires a real contact email in the
User-Agent header. The node is configured at 8 req/s (headroom under the cap)
via `utils/rate_limiter.py`, applied only on cache misses. The default
User-Agent is `NVDA Sentiment Node kokomoor@mit.edu` — override via
`Settings.sec_user_agent` or the `NVDA_SENTIMENT_SEC_USER_AGENT` env var.

---

## 4. Scoring Methodology

### 4.1 Section scoring

For each section extracted from a document:

```
section_score = 0.5 · finbert_score + 0.5 · lexicon_score − uncertainty_penalty
```

- `finbert_score = P(positive) − P(negative)` averaged across sentences (or
  `0` if FinBERT is unavailable — the blend then collapses to pure lexicon).
- `lexicon_score` uses a Loughran-McDonald-style word set (positive, negative,
  uncertainty, litigious buckets), normalized as positive-vs-negative hit
  count per 100 words.
- `uncertainty_penalty` is a small deduction when uncertainty/litigious
  vocabulary dominates (lots of *"may"*, *"could"*, *"subject to"* signals
  management hedging).

### 4.2 Document aggregation

Sections aggregate to a single document score via per-document-type weights
(`config.py:DOCUMENT_TYPE_WEIGHTS`):

| Doc type | Section weights | Doc-type weight in filing_tone |
|---|---|---|
| 10-K | MD&A 0.70 + outlook 0.30 | 0.30 |
| 10-Q | MD&A 0.70 + outlook 0.30 | 0.30 |
| 8-K | earnings_summary 0.60 + outlook 0.40 | 0.10 |
| press_release | headline 0.40 + financials 0.35 + outlook 0.25 | 0.10 |
| transcript | prepared_remarks 0.70 + Q&A 0.30 | 0.10 |
| cfo_commentary | full body 1.00 | 0.10 |

Risk-factors sections are **excluded** at this step to prevent them from
anchoring every reading negative.

### 4.3 Leadership component

Three independently-computed views of the same documents:

- **`filing_tone`** — weighted mean of document scores (weights above).
- **`filing_delta`** — current vs prior-year-comparable: `0.70·tone_delta + 0.30·risk_delta`.
  If no YoY match is found, `yoy_matched=False` and the component is 0 (with
  a confidence penalty). NVIDIA's fiscal calendar (FY ends late January) is
  handled in `features/filing_delta.py:nvda_fiscal_bucket()`.
- **`guidance_tone`** — averaged score of outlook/guidance sections **only**,
  recomputed independently since forward-looking language is the most
  price-relevant.

These collapse via `LEADERSHIP_COMPONENT_WEIGHTS` (renormalized from the
original 0.50/0.25/0.15) to:

```
leadership_component = 0.5556·filing_tone + 0.2778·filing_delta + 0.1667·guidance_tone
leadership_score     = 50 + 50 · clip(leadership_component, -1, +1)
```

### 4.4 Investor component

Each of the five investor sub-adapters returns an `InvestorSubScore` in `[-1, +1]`
with an `ok` flag. The branch aggregator computes a **pro-rata-redistributed
weighted mean** across the `ok` sub-scores:

| Sub-score | Default weight |
|---|---|
| `options_flow` | 0.30 |
| `analyst_signal` | 0.25 |
| `short_interest` | 0.15 |
| `social` | 0.15 |
| `broad_market` | 0.15 |

When a sub-score has `ok=False`, its weight is redistributed across the
remaining usable sub-scores in proportion to their own weights — a single
failure can't anchor the branch near zero. Then:

```
investor_component = Σ(w_i · score_i) / Σ(w_i)    over ok sub-scores
investor_score     = 50 + 50 · clip(investor_component, -1, +1)
```

### 4.5 Divergence-aware combiner

The combiner (`features/combiner.py`) is where the two branches meet. Let
`L = leadership_component`, `I = investor_component`, both in `[-1, +1]`:

```
base = α · L + (1 - α) · I                               # α = 0.55

if sign(L) == sign(I):
    adjustment = 0                                        # branches agree
elif L > I:                                               # leadership bullish, market bearish
    adjustment = −λ_neg · |L − I|                        # λ_neg = 0.25 (penalty)
else:                                                     # market bullish, leadership bearish
    adjustment = +λ_pos · |L − I|                        # λ_pos = 0.10 (contrarian premium)

combined_raw   = clip(base + adjustment, -1, +1)
combined_score = 50 + 50 · combined_raw
divergence     = leadership_score − investor_score
```

**Why asymmetric?** Insider optimism without market confirmation historically
correlates with disappointment (earnings misses, guidance cuts), so the
penalty is larger. Market optimism without insider confirmation is more often
capitulation bottoming or early re-rating, so the premium is smaller and
less certain.

**Leadership-only fallback:** if `investor_branch_ok=False` (all five
sub-scores failed), the combiner returns `combined := leadership_score`
with zero adjustment.

### 4.6 Label thresholds

```
<35 → bearish        <45 → mildly bearish    <55 → neutral
<65 → mildly bullish <80 → bullish           else → strongly bullish
```

### 4.7 Confidence arithmetic

Base `0.55`, capped bonuses up to `+0.40`, unbounded penalties, clamped to
`[0.10, 0.95]`. An empty-documents run short-circuits to `0.10`.

**Bonuses:** multiple doc types present, high extraction success rate, FinBERT
available, investor context available, guidance has multiple sources,
filing-delta computed, YoY match found, investor-branch ok, multiple investor
sub-scores ok.

**Penalties:** no delta, no investor context, fetch failures > 0, extraction
success rate < 100%, investor-branch unavailable.

---

## 5. Persistence & Longitudinality

### 5.1 SQLite schema

`nvda_sentiment/persistence.py` maintains a single table at
`data/history.sqlite3` (path override via `Settings.history_db_path`):

```sql
CREATE TABLE IF NOT EXISTS scores (
  ticker TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  combined_score REAL, leadership_score REAL, investor_score REAL,
  divergence REAL, confidence REAL, label TEXT,
  leadership_component REAL, investor_component REAL,
  components_json TEXT, signals_json TEXT, warnings_json TEXT,
  node_version TEXT, generated_at TEXT,
  PRIMARY KEY (ticker, as_of_date)
);
```

### 5.2 Upsert semantics

`record_response()` uses `INSERT OR REPLACE`:

- **Same-day rerun** — the previous row for `(ticker, today)` is overwritten.
  Idempotent: run the node 20 times today, get one row.
- **Cross-day run** — a new row is inserted. The table accumulates one row
  per ticker per day indefinitely.

This gives you **forward longitudinality without duplicating a single day's
noise.** After ~60 daily runs you have a meaningful time series to backtest
against realized returns or analyst revisions.

### 5.3 History CLI

```bash
nvda-sentiment history --ticker NVDA
```

Prints a compact per-row table (date, combined, leadership, investor,
divergence, confidence, label) in chronological order. Full JSON for any row
can be reconstructed from the `components_json` / `signals_json` / `warnings_json`
columns via `load_history()`.

---

## 6. Using the Node

### 6.1 Install

Python 3.11+ required. Run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pyproject.toml` is the single source of truth for dependencies. `torch` and
`transformers` are pulled in as hard deps (~1 GB download) so FinBERT is
always available. If those imports fail at runtime for any reason, the node
gracefully falls back to lexicon-only scoring and surfaces a warning.

Set a real contact email in the SEC User-Agent before first run:

```bash
cp .env.example .env
# edit NVDA_SENTIMENT_SEC_USER_AGENT=YourName your.email@example.com
```

Or override `Settings.sec_user_agent` directly in Python.

### 6.2 Command line

```bash
nvda-sentiment run --lookback-quarters 4 --show-sub-components
nvda-sentiment run --no-investor-branch                    # leadership only
nvda-sentiment run --no-cache --no-persist-history         # one-shot, no side effects
nvda-sentiment history --ticker NVDA                       # print the log
```

**`run` flags:**

| Flag | Default | Effect |
|---|---|---|
| `--lookback-quarters N` | 4 | Quarters of filings to pull (1–8) |
| `--include-investor-branch / --no-investor-branch` | on | Drop the investor branch entirely |
| `--include-market-context / --no-market-context` | on | Drop just the VIX sub-adapter |
| `--show-sub-components` | off | Also echo the 5 investor sub-score values |
| `--use-cache / --no-cache` | on | Bypass cache reads (writes still happen) |
| `--persist-history / --no-persist-history` | on | Upsert into the SQLite log |

Output is JSON on stdout — pipe to `jq` or redirect to a file.

### 6.3 Python API

```python
from nvda_sentiment import NVDASentimentNode, SentimentRequest

node = NVDASentimentNode()
response = node.run(SentimentRequest(
    ticker="NVDA",
    lookback_quarters=4,
    include_market_context=True,
    include_investor_branch=True,
    use_cache=True,
    persist_history=True,
))
print(response.model_dump_json(indent=2))
```

`response` is a Pydantic `SentimentResponse` — treat it as a validated object
or dump to dict/JSON.

### 6.4 Settings

All tunables live in `nvda_sentiment/config.py`. Construct your own `Settings`
and pass it to the node to override any of them:

```python
from pathlib import Path
from nvda_sentiment import NVDASentimentNode
from nvda_sentiment.config import Settings

settings = Settings(
    sec_user_agent="Your Name you@company.com",
    cache_dir=Path("/var/cache/nvda_sentiment"),
    history_db_path=Path("/var/lib/nvda_sentiment/history.sqlite3"),
    request_timeout_seconds=60,
    combiner_alpha=0.60,          # slightly more leadership weight
    combiner_lambda_neg=0.30,     # stronger bearish-divergence penalty
    combiner_lambda_pos=0.08,     # lighter contrarian premium
)
node = NVDASentimentNode(settings)
```

---

## 7. Development

### 7.1 Running the test suite

```bash
source .venv/bin/activate
python -m pytest -q
```

**105 tests passing, all offline** — per-adapter unit tests (options, short,
analyst, social), end-to-end dual-branch integration, combiner formula
tolerance, persistence upsert/filter, determinism, confidence arithmetic,
section extraction, NVIDIA fiscal-calendar edge cases, signal-building, text
utilities. Hermetic: every test injects fakes and avoids network I/O.

### 7.2 Determinism guarantees

Same inputs → byte-identical output (modulo the `generated_at` wall-clock
timestamp). Verified by `tests/test_determinism.py`, which runs the full
pipeline twice and asserts `resp_a == resp_b`. The guarantees:

- FinBERT runs in `eval()` mode under `torch.no_grad()` with
  `torch.use_deterministic_algorithms(True)` and `TOKENIZERS_PARALLELISM=false`.
- No iteration over sets; doc ordering is by filing date (a total order).
- No randomness anywhere in the scoring stack.

### 7.3 Calibration, tuning & longitudinal accuracy

The combiner exposes three knobs via `Settings` / `COMBINER_DEFAULTS`:

| Knob | Default | Meaning | Single-day-tunable? |
|---|---|---|---|
| `combiner_alpha` (α) | 0.55 | Leadership weight in the base blend (market gets 1−α) | **Yes** — α is applied on every run, so today's grid genuinely ranks it |
| `combiner_lambda_neg` (λ_neg) | 0.25 | Penalty when management is bullish but the market isn't | **No** — only activates when `L > I`; silent on days of reverse divergence |
| `combiner_lambda_pos` (λ_pos) | 0.10 | Bonus when the market is bullish but management isn't | **No** — only activates when `I > L`; silent on days of reverse divergence |

On any given day, at most **one** of the two λ adjustments fires (or neither,
if branches agree). A single-day grid therefore provides zero information
about the inactive λ, and the "winner" within the active λ's range is
usually just pinned to the grid edge rather than an interior optimum. Only
α fires every run regardless of divergence direction — which is why
**Tier-2 tuning updates α only** and leaves λ values at their principled
defaults until a multi-day fit justifies otherwise.

#### The calibration harness

`tools/calibrate_weights.py` runs the pipeline once, computes a target
(weighted analyst EPS-revision direction over the last 30 days, in `[-1, +1]`),
and sweeps a 5×3×3 grid of `(α, λ_neg, λ_pos)` values against today's
`leadership_component` and `investor_component`:

```bash
python -m tools.calibrate_weights --ticker NVDA                         # one-shot report
python -m tools.calibrate_weights --ticker NVDA --grid                  # sweep + upsert all 45 rows
python -m tools.calibrate_weights --ticker NVDA --grid --no-persist-grid    # sweep only, no DB write
python -m tools.calibrate_weights --ticker NVDA --write-best-alpha      # Tier-2: persist today's best α
python -m tools.calibrate_weights --ticker NVDA --clear-tuned-weights   # revert to config defaults
python -m tools.calibrate_weights --ticker NVDA --show-history          # dump daily-snapshot log
python -m tools.calibrate_weights --ticker NVDA --show-grid-history     # dump accumulated grid sweeps
```

All 45 grid rows are persisted into the `grid_sweeps` SQLite table,
alongside the `scores` table in `data/history.sqlite3`, keyed on
`(ticker, as_of_date, α, λ_neg, λ_pos)`. Same-day reruns upsert; each new
day appends a fresh 45. Over weeks and months this accumulates a matrix
suitable for real regression against realized forward returns or
analyst-revision direction.

#### Tier-2 tuning — file-backed overlay

`--write-best-alpha` extracts today's top-row α (sorted by sign-match first,
then by distance to target) and writes it to the **tuned-weights overlay**
at `data/tuned_weights.json`. From that point forward, any
`NVDASentimentNode()` instantiated **without an explicit `Settings=`** reads
the overlay and applies it to its combiner. Passing an explicit
`Settings(...)` to the constructor bypasses the overlay entirely — useful
for reproducibility in tests and library integrations.

```json
{
  "alpha": 0.40,
  "updated_at": "2026-04-21T15:03:11Z",
  "basis": {
    "ticker": "NVDA",
    "as_of_date": "2026-04-21",
    "target": 0.805,
    "leadership_component": -0.039,
    "investor_component": 0.202,
    "dist": 0.663,
    "sign_match": true,
    "tier": 2,
    "note": "Only alpha is persisted from single-day data. ..."
  }
}
```

The overlay records the *basis* of the tuning (target, raw components,
distance) so a future reader can judge whether the value is stale or
whether today's regime still resembles the day it was written.

#### Longitudinal refinement — why this model gets better over time

Three feedback loops ship out of the box:

1. **Daily snapshots** (`scores` table) — one row per `(ticker, date)`,
   upserted on same-day reruns, accumulated across days. After ~30 days you
   can visually check which labels preceded meaningful price moves.
2. **Daily grid sweeps** (`grid_sweeps` table) — 45 rows per `--grid` run.
   After ~20 days, the same winning (α, λ_neg, λ_pos) tuple repeatedly
   topping the dist-sorted leaderboard is a *real* signal that the defaults
   should be revised. A single day's winner is not.
3. **Overlay revisions** — `--write-best-alpha` can be re-run periodically as
   the SQLite matrix grows. The overlay is idempotent (writes overwrite,
   don't append), so the latest tune is always the active one. To revert:
   `--clear-tuned-weights` or delete `data/tuned_weights.json`.

This is what's meant by "the model gets better over time without code
changes": as the SQLite log fills up, both the target (EPS revisions) and
the raw branches can be compared historically, and the knobs can be
re-tuned from a progressively larger evidence base.

#### Limitation — data access

**The free-data path is inherently same-day.** The investor-branch adapters
consume live APIs (yfinance, Reddit, StockTwits, FRED) that do not expose
point-in-time historical snapshots of option chains, short interest,
analyst targets, or intraday sentiment. The leadership branch *does* have
historical filings available from SEC EDGAR, but honoring an `as_of_date`
there without matching historical investor data would produce an incoherent
blend (past filings mixed with today's market state).

The architecture is fully **time-horizon agnostic** — nothing in the
combiner, scorer, or persistence layer assumes "today." The binding
constraint is the absence of historical feeds for the investor branch.
With access to a paid point-in-time data vendor (FactSet, Refinitiv,
S&P/I/B/E/S archives, Bloomberg, etc.):

- An `--as-of-date` parameter could trivially be added to the request schema
  and propagated to every adapter.
- The calibration harness could run against hundreds of historical days in a
  single pass, regressing combined scores against *forward* NVDA returns
  (rather than a same-day proxy like EPS revisions).
- `α`, `λ_neg`, **and** `λ_pos` could all be fit simultaneously via proper
  cross-validation, grid search, or gradient methods over a real label set.
- Varied time horizons (1-day forward, 5-day forward, 20-day forward) would
  each yield a separate calibrated tuple — matching the combiner to the
  holding period of the downstream strategy.

This is not a software limitation of the node; it is a **data-access
limitation** of the free APIs the MVP was built against. The extension is
trivial in engineering terms and would substantially tighten the
statistical calibration.

### 7.4 Adding a new leadership source

1. Create `adapters/your_source.py` with
   `get_<something>_documents(lookback_quarters) -> List[SourceDocument]`
   and `fetch_document_html(doc) -> str`.
2. Add a new `SourceType` literal to `schemas.py`.
3. Add section weights in `config.py` (see `PRESS_RELEASE_SECTION_WEIGHTS` as a template).
4. Add the new type to `DOCUMENT_TYPE_WEIGHTS` (weights must still sum to 1.0).
5. Wire it into `NVDASentimentNode.__init__` and step 1 of `node.run()`.
6. Add a section-detection branch to `parsers/section_extractor.py`.
7. Update `_build_source_coverage` in `node.py`.

### 7.5 Adding a new investor sub-adapter

1. Create `adapters/investor/your_signal.py` implementing
   `get_signal() -> InvestorSubScore`.
2. Add the name to `INVESTOR_COMPONENT_WEIGHTS` in `config.py` (all weights
   must sum to 1.0).
3. Wire it into `InvestorBranch.__init__` and `.run()` in
   `features/investor_branch.py`.
4. Register in `_get_sub()` / `components` dict in `node.py`.
5. Add a per-adapter test file under `tests/`.

### 7.6 Replacing the scorer

`SectionScorer` is swappable via the `section_scorer=` kwarg on
`NVDASentimentNode`. A conforming scorer needs one method
`score_section(name: str, text: str) -> SectionScore` and one property
`finbert_available: bool` (used only for confidence accounting).

### 7.7 Extending beyond NVIDIA

The node is single-company by design. To generalize:

1. Parameterize `NVIDIA_CIK` / `NVIDIA_CIK_PADDED` in `config.py`.
2. Replace `nvda_fiscal_bucket()` in `features/filing_delta.py` with a
   per-company fiscal calendar (or a general calendar-quarter fallback).
3. The NVIDIA IR adapter is page-structure-specific; most issuers will need
   a replacement (or skip IR and rely on 8-K / 10-Q alone).

### 7.8 Known limitations

- Section extraction is heuristic regex on headings — standard SEC layouts
  parse cleanly but unusual formatting can mis-segment.
- SEC submissions JSON pagination is not followed; effective lookback is
  capped at ~1000 most-recent filings (comfortably covers 8 quarters).
- Reddit's public JSON endpoints can rate-limit without notice; the social
  adapter tolerates empty results and flags `ok=False`.
- Lexicon is intentionally small; FinBERT carries most of the semantic load
  when installed.
- Fiscal calendar is hardcoded to NVIDIA.
- Same-day-only constraint (§1.4) means backtesting requires either the SQLite
  accumulation path or paid historical data.

---

## 8. Reference

### 8.1 NVIDIA fiscal calendar

NVIDIA's fiscal year ends in **late January**, so calendar-month →
fiscal-quarter mapping is non-obvious. The logic lives in
`features/filing_delta.py:nvda_fiscal_bucket()`:

| Filing month | Fiscal quarter |
|---|---|
| Feb–Apr | Q1 of next-year's FY label (e.g. Feb 2026 → FY27Q1) |
| May–Jul | Q2 |
| Aug–Oct | Q3 |
| Nov–Dec | Q4 of next-year's FY label |
| Jan | Q4 of **same** calendar year's FY label (fiscal year end) |

Test coverage in `tests/test_filing_delta.py` nails the corner cases
(Jan 2026 → FY26Q4, Jan 2027 → FY27Q4, Nov 2026 → FY27Q4).

### 8.2 Response schema (abridged)

```python
class SentimentResponse(BaseModel):
    ticker: str
    as_of_date: date
    market_sentiment_score: float         # 0..100 (alias of combined_score)
    combined_score: float                 # 0..100
    leadership_score: float               # 0..100
    investor_score: float                 # 0..100
    divergence: float                     # leadership_score − investor_score
    label: Literal["bearish", "mildly bearish", "neutral",
                   "mildly bullish", "bullish", "strongly bullish"]
    confidence: float                     # [0.10, 0.95]
    components: Dict[str, float]          # all 10 numeric inputs, 3dp
    signals: List[str]                    # plain-English rationale
    source_coverage: Dict[str, int]       # per-source-type doc counts
    metadata: Dict[str, Any]              # node_version, warnings[], generated_at
```

### 8.3 Where the original spec lives

`SA_NewArch.md` in the repo root. Section numbers referenced in code
(e.g. `§6.5.1`, `§22.1`) point back to that document — the code is
authoritative where it diverges.
