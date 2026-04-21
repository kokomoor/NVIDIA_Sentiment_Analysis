# SA_NewArch.md — Dual-Branch Sentiment Architecture with Divergence Combiner

**Status:** Design + implementation blueprint. Due same-day. No multi-phase rollout.
**Reader:** An engineer (human or LLM) armed with this document, the existing
codebase, and nothing else must be able to execute this plan end-to-end.
**Audit target:** Codex / GPT-pro-thinking must find no inconsistency, no missing
step, no hallucinated API, no broken weight sum.

---

## 0. Executive Summary

### The change in one paragraph

The node currently emits one score that blends management-tone signals (SEC
filings, NVIDIA IR) with broad-market investor signals (AAII + VIX) bolted onto
the tail of the composite formula. That confounds two distinct things. This
redesign separates them into two parallel branches — a **leadership branch**
(what management says) and an **investor branch** (how the market responds
and positions around NVDA) — then combines them through a divergence-aware,
tunable combiner. The investor branch gets real NVDA-specific inputs
(options flow, short interest, analyst dispersion, social sentiment) in
addition to the retained AAII+VIX overlay.

### Scores emitted after the change

| Field | Range | Meaning |
|---|---|---|
| `leadership_score` | 0–100 | Sentiment of NVIDIA's own disclosures (no market overlay) |
| `investor_score` | 0–100 | Sentiment of the market's NVDA-specific positioning and talk |
| `divergence` | −100…+100 | `leadership_score − investor_score` |
| `combined_score` | 0–100 | Tunable combiner output; the headline number |
| `market_sentiment_score` | 0–100 | **Alias of `combined_score`** (backwards compatibility only) |

### What stays exactly the same

- The existing leadership scoring pipeline (SEC adapter, IR adapter, parsers,
  FinBERT+lexicon scorers, `filing_tone` / `filing_delta` / `guidance_tone`,
  confidence, signals, determinism guarantees).
- The response is still a single pydantic `SentimentResponse` JSON blob.
- Offline-testable architecture via adapter dependency injection.
- SEC rate limiting, caching tiers, graceful degradation on any source failure.

### What is strictly new

- `nvda_sentiment/adapters/investor/` subpackage with five adapters.
- `nvda_sentiment/features/investor_branch.py` aggregator.
- `nvda_sentiment/features/combiner.py` divergence-aware combiner.
- `tools/calibrate_weights.py` calibration harness with EPS-revisions target.
- New pydantic fields on the response.
- New config weights with defaults derived in §7.2 and Appendix C.
- New tests: one per adapter + integration tests for the combiner and the
  two-branch pipeline.

---

## 1. Goals, Non-Goals, Constraints

### Goals

1. Cleanly isolate management-tone signal from market-response signal.
2. Introduce an NVDA-specific investor-sentiment branch with four independent
   inputs plus a broad-market overlay.
3. Emit a divergence figure so consumers can reason about expectations gap.
4. Provide a tunable combiner with defaults that are theoretically motivated
   and empirically sanity-checked.
5. Maintain full offline testability (no test requires network).
6. Maintain determinism (`test_determinism.py` must still pass).
7. Keep the public response schema backwards-compatible under the
   `market_sentiment_score` alias.

### Non-goals

1. Multi-ticker generalization. NVDA-only as before.
2. Paid data sources. Everything free, no signups requiring review/wait.
3. Rigorous ML calibration. We use a harness to **sanity check** starting
   weights, not to fit a model on 40 earnings events.
4. Real-time streaming. Request/response semantics preserved.
5. A full re-architecture of the leadership branch. Keep its internals
   intact; only shear off the AAII+VIX tail.

### Hard constraints

- Python 3.11+.
- Zero API keys required. Zero account signups with approval windows.
- Every external call mockable. Every adapter must accept a
  `session: requests.Session | None = None` (or equivalent injection point)
  so tests can replace it.
- All new tests must run under ~3 seconds in aggregate, offline.
- No breaking change to `SentimentRequest` (only additive fields allowed).
- `combined_score` must equal prior behavior's `market_sentiment_score`
  within rounding if the combiner is set to identity-like weights that mimic
  the old formula (documented in §7.2 fallback path).

---

## 2. Architecture

### 2.1 Before (current state)

```
                 ┌─────────────────────────┐
                 │   SentimentRequest       │
                 └────────────┬────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
 SEC adapter           NVIDIA IR adapter     MarketContext adapter
 (10-K/10-Q/8-K)       (press, transcripts,   (AAII + VIX)
                        CFO commentary)
      │                       │                       │
      └───────────┬───────────┘                       │
                  ▼                                   │
          extract → score                             │
          per-section → per-doc                       │
                  │                                   │
                  ▼                                   │
       filing_tone / filing_delta /                   │
       guidance_tone                                  │
                  │                                   │
                  └──────────────┬────────────────────┘
                                 ▼
              0.50·filing_tone + 0.25·filing_delta +
              0.15·guidance_tone + 0.10·investor_context
                                 ▼
                       market_sentiment_score
```

**Problem:** `investor_context` (broad-market AAII+VIX) is 10% of the number
labeled as NVIDIA sentiment. It's neither NVDA-specific nor management-specific.
It confounds the signal.

### 2.2 After (dual-branch)

```
                 ┌─────────────────────────┐
                 │   SentimentRequest       │
                 └────────────┬────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      │                                               │
      ▼                                               ▼
┌──────────────────────┐                ┌───────────────────────────┐
│  LEADERSHIP BRANCH    │                │     INVESTOR BRANCH        │
│                      │                │                           │
│  SEC adapter         │                │  OptionsFlow adapter       │
│  NVIDIA IR adapter   │                │  ShortInterest adapter     │
│                      │                │  AnalystSignals adapter    │
│  filing_tone  (0.55) │                │  SocialSentiment adapter   │
│  filing_delta (0.28) │                │  BroadMarket adapter       │
│  guidance_tone(0.17) │                │   (AAII + VIX, retained)   │
│                      │                │                           │
│  (weights renormalized│                │  options_flow    (0.30)    │
│   from 0.50/0.25/0.15)│                │  short_interest  (0.15)    │
│                      │                │  analyst_signal  (0.25)    │
│                      │                │  social          (0.15)    │
│                      │                │  broad_market    (0.15)    │
└───────────┬──────────┘                └─────────────┬─────────────┘
            │                                         │
            ▼                                         ▼
   leadership_score (0-100)                investor_score (0-100)
            │                                         │
            └──────────────────┬──────────────────────┘
                               ▼
                   divergence = L_score - I_score
                               ▼
                 ┌───────────────────────────┐
                 │    DIVERGENCE COMBINER    │
                 │  (sign-aware asymmetric)  │
                 └─────────────┬─────────────┘
                               ▼
                       combined_score (0-100)
                  + label, confidence, signals
                               ▼
                     SentimentResponse
```

### 2.3 Data-flow invariants

- Both branches run regardless of each other's success. If the investor
  branch fully fails, `investor_score` is `50.0` (neutral) with
  `investor_context_available=False`; the combiner falls back to
  `combined_score = leadership_score`.
- Conversely if the leadership branch returns empty (no scorable
  filings) the existing `_empty_response` path still fires — but the response
  now carries the investor branch's score as an additional observation.
- Caching and rate limiting are per-adapter as today.
- Every adapter exposes `use_cache: bool` attribute (pattern established in
  the current codebase); `node.run()` sets it from `request.use_cache`.

---

## 3. Data Sources — Exact API Surfaces

All four new sources verified live on 2026-04-20 against NVDA. Exact call
patterns and parsing rules below; no guessing at runtime.

### 3.1 Options flow (yfinance)

**Library:** `yfinance>=0.2.40` (tested on 1.3.0).

**Endpoint:**
```python
import yfinance as yf
t = yf.Ticker("NVDA")
expiries: tuple[str, ...] = t.options             # ("YYYY-MM-DD", ...)
chain = t.option_chain(expiry)                    # namedtuple(.calls, .puts)
```

**Returned fields (DataFrame columns):** `contractSymbol`, `lastTradeDate`,
`strike`, `lastPrice`, `bid`, `ask`, `change`, `percentChange`, `volume`,
`openInterest`, `impliedVolatility`, `inTheMoney`, `contractSize`, `currency`.

**Two extracted metrics:**

1. **Put/call volume ratio** — aggregate over the **nearest 3 expiries that
   are ≥ 7 days out** (avoids day-of-expiry noise). Formula:
   ```
   pcr = sum(puts.volume) / max(sum(calls.volume), 1)
   ```
   Then bucketize to a score in `[-1, +1]`:
   | `pcr` | score |
   |---|---|
   | ≤ 0.60 | +0.50 (bullish — call-heavy) |
   | ≤ 0.85 | +0.25 |
   | ≤ 1.15 | 0.00 (balanced) |
   | ≤ 1.50 | −0.25 |
   | > 1.50 | −0.50 (bearish — put-heavy) |

2. **ATM implied-volatility z-score (IV rank proxy)** — pull IV for the
   call and put nearest the current spot on the nearest expiry ≥ 21 days.
   ATM IV = mean(call.iv, put.iv). Compute rolling z-score **across the
   current chain's IV distribution** (not historical — we don't have free
   historical IV). The proxy: IV z-score below −1 indicates complacency
   (mildly bullish), above +1 indicates fear (mildly bearish, but can also
   be bullish pre-catalyst). We use an asymmetric mapping:
   | ATM IV z | score |
   |---|---|
   | ≤ −1.0 | +0.20 |
   | ≤ +0.5 | 0.00 |
   | ≤ +1.5 | −0.15 |
   | > +1.5 | −0.30 |

   **Caveat captured as a warning when emitted:** IV alone is directionally
   ambiguous. The bucketing is conservative.

**Combined options_flow sub-score:** `0.70·pcr_score + 0.30·iv_score`,
clipped to `[-1, +1]`.

### 3.2 Short interest (yfinance)

**Endpoint:** `yf.Ticker("NVDA").info` — returns `dict`. Relevant fields:
- `shortRatio` (days-to-cover; verified 1.54 for NVDA)
- `shortPercentOfFloat` (verified 0.0121 = 1.21%)

**Score:**

| `shortPercentOfFloat` | score contribution |
|---|---|
| ≤ 0.01 (1%) | +0.20 |
| ≤ 0.02 | +0.10 |
| ≤ 0.04 | 0.00 |
| ≤ 0.07 | −0.15 |
| > 0.07 | −0.30 |

| `shortRatio` (days-to-cover) | score contribution |
|---|---|
| ≤ 1.0 | +0.20 |
| ≤ 2.0 | 0.00 |
| ≤ 3.5 | −0.10 |
| > 3.5 | −0.25 |

**Combined short_interest sub-score:** `0.60·short_pct + 0.40·short_ratio`.

### 3.3 Analyst signals (yfinance)

**Endpoint:** `yf.Ticker("NVDA").info` — relevant fields: `targetMeanPrice`,
`targetMedianPrice`, `targetHighPrice`, `targetLowPrice`,
`numberOfAnalystOpinions`, `recommendationMean`, `currentPrice`.

**Three extracted metrics:**

1. **Target mean vs current price (upside):**
   ```
   upside = (targetMeanPrice - currentPrice) / currentPrice
   ```
   Map to `[-1, +1]` linearly with saturation at ±0.40:
   ```
   upside_score = clip(upside / 0.40, -1, +1)
   ```

2. **Target dispersion (analyst disagreement):**
   ```
   dispersion = (targetHighPrice - targetLowPrice) / targetMeanPrice
   ```
   High dispersion is *ambiguous* — can precede either breakout or breakdown.
   We treat it as a **signal damper**, not a directional score. Map:
   ```
   dispersion_damper = max(0.0, 1.0 - dispersion)   # 1.0 at zero dispersion,
                                                    # 0.0 at dispersion ≥ 1.0
   ```
   Applied multiplicatively to `upside_score`.

3. **Recommendation mean.** yfinance convention: `1.0 = Strong Buy`,
   `5.0 = Strong Sell`. Map with linear interpolation:
   ```
   rec_score = clip((3.0 - recommendationMean) / 2.0, -1, +1)
   ```
   Strong Buy → +1.0, Hold → 0.0, Strong Sell → −1.0.

**Combined analyst_signal sub-score:**
```
analyst_signal = 0.50·(upside_score·dispersion_damper) + 0.50·rec_score
```

Clip to `[-1, +1]`.

### 3.4 Social sentiment (StockTwits + Reddit)

**StockTwits endpoint (no auth, public):**
```
GET https://api.stocktwits.com/api/2/streams/symbol/NVDA.json
User-Agent: nvda-sentiment-node/0.2 (your@email.com)
```

Response `messages[].entities.sentiment.basic ∈ {"Bullish", "Bearish", null}`.
~30 messages per call; sufficient for intraday read. Score:
```
bullish = count(sentiment.basic == "Bullish")
bearish = count(sentiment.basic == "Bearish")
tagged  = bullish + bearish
if tagged < 5:
    stocktwits_score = 0.0        # insufficient signal
    stocktwits_ok = False
else:
    stocktwits_score = (bullish - bearish) / tagged
    stocktwits_ok = True
```

**Reddit endpoint (no auth, public JSON):**
```
GET https://www.reddit.com/r/wallstreetbets/search.json
    ?q=NVDA&restrict_sr=on&sort=new&t=week&limit=50
User-Agent: nvda-sentiment-node/0.2 (your@email.com)
```

Also pull `r/stocks` and `r/investing` with the same pattern. Dedupe by
post `id`. For each post, score its `title + selftext` with the **existing
FinBERT+lexicon scorer** (reuse `LexiconScorer` and `FinBERTScorer`). Weight
each post by `log1p(score)` where `score` is Reddit upvotes (caps the
influence of megathread outliers).

```
post_score_i = 0.5·finbert(title+body) + 0.5·lexicon(title+body)
weight_i     = log1p(max(post.score, 0))
reddit_score = sum(post_score_i * weight_i) / max(sum(weight_i), 1.0)
reddit_ok    = (len(posts) >= 3)
```

Clip `reddit_score` to `[-1, +1]`.

**Combined social sub-score:**
```
if stocktwits_ok and reddit_ok:
    social_score = 0.60·stocktwits_score + 0.40·reddit_score
elif stocktwits_ok:
    social_score = stocktwits_score
elif reddit_ok:
    social_score = reddit_score
else:
    social_score = 0.0; social_ok = False
```

**Caching:** TTL 1 hour (social is fast-moving; 24h is too stale). Cache file
per URL as with other adapters.

### 3.5 Broad market (AAII + VIX) — retained, relocated

No behavior change. Move `nvda_sentiment/adapters/market_context.py` into
`nvda_sentiment/adapters/investor/broad_market.py`. Rename class to
`BroadMarketAdapter`. Existing tests continue to pass against the renamed
symbol.

Score: `0.6·AAII + 0.4·VIX` clipped to `[-1, +1]` (unchanged).

### 3.6 EPS revisions (calibration harness only)

**Endpoint:**
```python
t = yf.Ticker("NVDA")
eps_trend = t.eps_trend            # DataFrame: current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo
eps_rev   = t.eps_revisions        # DataFrame: upLast7days, upLast30days, downLast30days, downLast7Days
```

**Calibration target (used only offline, never in the hot path):**
```
rev_score_period = (upLast30days - downLast30days) / max(upLast30days + downLast30days, 1)
rev_score = weighted_mean_across_periods(rev_score_period, weights=[0.4, 0.3, 0.2, 0.1])
           # periods ordered [0q, +1q, 0y, +1y]
```

`rev_score ∈ [-1, +1]`. This is the "truth label" the harness tries to
predict using the two branches' outputs.

**Verified live values (2026-04-20 NVDA):**
- 0q: 31 up, 1 down → +0.9375
- +1q: 26 up, 5 down → +0.677
- 0y: 38 up, 5 down → +0.767
- +1y: 26 up, 4 down → +0.733

Weighted: `0.4·0.9375 + 0.3·0.677 + 0.2·0.767 + 0.1·0.733 = +0.806`.

---

## 4. Schema Changes

### 4.1 `SentimentRequest` — additive only

```python
class SentimentRequest(BaseModel):
    ticker: str = "NVDA"
    as_of_date: date
    lookback_quarters: int = Field(default=4, ge=1, le=8)
    include_market_context: bool = True      # existing; now gates the BroadMarketAdapter only
    include_investor_branch: bool = True     # NEW: master toggle for investor branch
    use_cache: bool = True
```

### 4.2 `SentimentResponse` — additive + one rename

```python
class SentimentResponse(BaseModel):
    ticker: str
    as_of_date: date
    # Backwards-compat alias (== combined_score); DO NOT drop.
    market_sentiment_score: float
    # NEW
    leadership_score: float
    investor_score: float
    combined_score: float
    divergence: float
    label: str
    confidence: float
    components: Dict[str, float]      # enriched keys — see below
    signals: List[str]
    source_coverage: Dict[str, int]
    metadata: Dict[str, Any]
```

**`components` keys after the change:**

```
# Leadership sub-components (existing)
filing_tone
filing_delta
guidance_tone
# Investor sub-components (new)
options_flow
short_interest
analyst_signal
social
broad_market                      # AAII+VIX, formerly "investor_context"
# Aggregates (new)
leadership_component              # ∈ [-1, +1], the branch's raw signed score
investor_component                # ∈ [-1, +1], the branch's raw signed score
```

All numeric, all in `[-1, +1]`.

### 4.3 Internal pydantic models (new)

```python
class InvestorSubScore(BaseModel):
    name: Literal["options_flow","short_interest","analyst_signal","social","broad_market"]
    score: float            # [-1, +1]
    ok: bool                # whether the source produced usable data
    detail: Dict[str, Any]  # raw numbers for signals/debug (not in public schema)

class InvestorBranchOutput(BaseModel):
    sub_scores: List[InvestorSubScore]
    investor_component: float   # weighted combination, [-1, +1]
    investor_score: float       # 0-100
    ok: bool                    # true if at least one sub-source returned ok
```

---

## 5. Module Layout After Refactor

```
nvda_sentiment/
├── node.py                          # orchestrator (modified §6.6)
├── schemas.py                       # enriched response model (§4)
├── config.py                        # new weights block (§7.2)
├── adapters/
│   ├── sec_api.py                   # unchanged
│   ├── nvidia_ir.py                 # unchanged
│   └── investor/                    # NEW subpackage
│       ├── __init__.py
│       ├── options_flow.py          # §3.1
│       ├── short_interest.py        # §3.2
│       ├── analyst_signals.py       # §3.3
│       ├── social.py                # §3.4
│       └── broad_market.py          # renamed from adapters/market_context.py (§3.5)
├── features/
│   ├── filing_delta.py              # unchanged
│   ├── confidence.py                # extended (§6.5)
│   ├── signal_builder.py            # extended with divergence signals (§6.5)
│   ├── investor_branch.py           # NEW aggregator (§6.4)
│   └── combiner.py                  # NEW divergence combiner (§6.5)
├── scorers/                         # unchanged (reused for social)
├── parsers/                         # unchanged
└── utils/                           # unchanged
tools/
└── calibrate_weights.py             # NEW calibration harness (§8)
tests/
├── (all existing tests)             # green after refactor
├── test_options_flow_adapter.py     # NEW
├── test_short_interest_adapter.py   # NEW
├── test_analyst_signals_adapter.py  # NEW
├── test_social_adapter.py           # NEW
├── test_investor_branch.py          # NEW aggregator tests
├── test_combiner.py                 # NEW combiner unit tests
└── test_end_to_end_dual_branch.py   # NEW full pipeline integration
```

---

## 6. Implementation Steps — File-by-File

Ordering is load-bearing. Do not reorder. Each step must leave the test
suite green.

### 6.1 Preflight

**Step 6.1.1 — Add dependencies.** Edit `pyproject.toml` `[project]
dependencies`:

```toml
dependencies = [
  "pydantic>=2",
  "requests>=2.31",
  "beautifulsoup4>=4.12",
  "lxml>=5",
  "pandas>=2",
  "numpy>=1.26",
  "python-dateutil>=2.8",
  "typer>=0.9",
  "yfinance>=0.2.40",       # NEW
]
```

Optional FinBERT extra unchanged. Run `pip install -e .` to install yfinance.

**Step 6.1.2 — Extend `config.py`.** Append after the existing
`FINAL_COMPONENT_WEIGHTS` block:

```python
# Leadership branch weights (renormalized from old 0.50/0.25/0.15 = 0.90).
LEADERSHIP_COMPONENT_WEIGHTS = {
    "filing_tone":   0.556,    # 0.50 / 0.90
    "filing_delta":  0.278,    # 0.25 / 0.90
    "guidance_tone": 0.167,    # 0.15 / 0.90
}   # must sum to 1.0 ± 1e-6

# Investor branch weights.
INVESTOR_COMPONENT_WEIGHTS = {
    "options_flow":   0.30,
    "short_interest": 0.15,
    "analyst_signal": 0.25,
    "social":         0.15,
    "broad_market":   0.15,
}   # must sum to 1.0 ± 1e-6

# Combiner weights (see §7.2).
COMBINER_DEFAULTS = {
    "alpha":          0.55,    # leadership weight in base blend
    "lambda_neg":     0.25,    # bearish-divergence penalty magnitude (L>I)
    "lambda_pos":     0.10,    # contrarian premium magnitude (I>L)
    "agreement_zero": True,    # when signs agree, divergence adjustment = 0
}
```

Add a `__post_init__` assertion on `Settings` (or a module-level test) that
the two weight dicts sum to 1.0 within tolerance.

**Step 6.1.3 — Extend `Settings` dataclass** with combiner overrides:

```python
combiner_alpha:      float = COMBINER_DEFAULTS["alpha"]
combiner_lambda_neg: float = COMBINER_DEFAULTS["lambda_neg"]
combiner_lambda_pos: float = COMBINER_DEFAULTS["lambda_pos"]
```

### 6.2 Leadership Branch Refactor

**Step 6.2.1 — Move `adapters/market_context.py` → `adapters/investor/broad_market.py`.**
`git mv` the file. Rename `MarketContextAdapter` → `BroadMarketAdapter`
(keep a deprecation alias `MarketContextAdapter = BroadMarketAdapter` in
`adapters/__init__.py` so old imports still work during the transition).

**Step 6.2.2 — Remove `investor_context` from leadership composite.**

In `nvda_sentiment/scorers/composite.py`, locate `compute_final_score`. Its
current signature:

```python
def compute_final_score(filing_tone, filing_delta, guidance_tone, investor_context)
    -> tuple[float, float, str]:
    # 0.50·filing_tone + 0.25·filing_delta + 0.15·guidance_tone + 0.10·investor_context
```

Split into two pure functions:

```python
def compute_leadership_component(
    filing_tone: float, filing_delta: float, guidance_tone: float,
) -> float:
    w = LEADERSHIP_COMPONENT_WEIGHTS
    return (w["filing_tone"]   * filing_tone
          + w["filing_delta"]  * filing_delta
          + w["guidance_tone"] * guidance_tone)

def score_to_0_100_and_label(component: float) -> tuple[float, str]:
    score = 50.0 + 50.0 * max(-1.0, min(1.0, component))
    return score, _label_for(score)
```

Delete the old `compute_final_score` (or keep it as a thin wrapper that
delegates to the two new functions for one release).

**Step 6.2.3 — Update `node.run()` leadership path.** Replace the call site
of `compute_final_score(...)` with the two new calls; capture
`leadership_component` and `leadership_score` locally. Do not yet wire
combiner — next phase.

**Step 6.2.4 — Update signals.** `features/signal_builder.py` already
produces six tiers; tier-6 is the broad-market tier. Move its logic into
`investor_branch.py` (§6.4) and **delete** it from `signal_builder.py`.
Leadership signals are now purely about filings.

**Step 6.2.5 — Run existing test suite.**

```bash
source .venv/bin/activate && python -m pytest -q
```

Tests that will require tiny edits:
- `test_end_to_end.py`: `FakeMarket` is no longer passed into `NVDASentimentNode`
  as `market_context=...`; instead it'll feed the investor branch. **For
  this step**, just stub `FakeMarket` to return `(0.0, False)` and assert
  the leadership score alone. Full re-point happens in §6.6.
- `test_confidence.py`: uses `include_market_context` and
  `investor_context_available`. Keep signatures intact — confidence arithmetic
  will absorb the investor branch in §6.5 with additional kwargs (defaulted).
- `test_signal_builder.py`: remove the "tier 6" assertions (deleted).

All green before proceeding.

### 6.3 Investor Branch — New Adapters

Implement one file at a time. Each adapter MUST:

- Accept `settings: Settings`, `cache: SimpleCache | None = None`,
  `session: requests.Session | None = None`.
- Expose `use_cache: bool` attribute (default `True`).
- Catch any network/parse exception and return a `(score, ok)` tuple with
  `score=0.0, ok=False` on any failure. Logged as a warning.
- Emit a `detail: dict` of raw numbers alongside the score for signal
  construction and debugging. Details are **not** published in the public
  `components` dict; they go into the per-sub-score `InvestorSubScore.detail`
  used by the signal builder.

**Step 6.3.1 — `adapters/investor/options_flow.py`.** Key methods:

```python
class OptionsFlowAdapter:
    def get_signal(self) -> InvestorSubScore:
        expiries = self._get_expiries()
        relevant = [e for e in expiries if _days_out(e) >= 7][:3]
        pcr = self._put_call_ratio(relevant)
        iv_z = self._atm_iv_zscore()
        pcr_score = _bucket_pcr(pcr)
        iv_score  = _bucket_iv(iv_z)
        combined  = 0.70*pcr_score + 0.30*iv_score
        return InvestorSubScore(
            name="options_flow",
            score=clip(combined, -1, 1),
            ok=True,
            detail={"pcr": pcr, "iv_z": iv_z, "expiries_used": relevant},
        )
```

Wrap all `yfinance` calls in try/except. yfinance internally retries; set a
request timeout via `yf.Ticker(..., session=self.session)` + a session-level
timeout via custom adapter or rely on the default (acceptable for our scope).

**Cache key:** `f"opts:{ticker}:{today_iso}"`. TTL 4 hours.

**Step 6.3.2 — `adapters/investor/short_interest.py`.**

```python
class ShortInterestAdapter:
    def get_signal(self) -> InvestorSubScore:
        info = yf.Ticker(self.ticker, session=self.session).info
        short_pct  = info.get("shortPercentOfFloat")
        short_rat  = info.get("shortRatio")
        if short_pct is None and short_rat is None:
            return InvestorSubScore(name="short_interest", score=0.0, ok=False, detail={})
        pct_score = _bucket_pct(short_pct) if short_pct is not None else 0.0
        rat_score = _bucket_ratio(short_rat) if short_rat is not None else 0.0
        # if one is missing, use the other alone
        if short_pct is not None and short_rat is not None:
            combined = 0.6*pct_score + 0.4*rat_score
        else:
            combined = pct_score if short_pct is not None else rat_score
        return InvestorSubScore(
            name="short_interest",
            score=clip(combined, -1, 1),
            ok=True,
            detail={"shortPercentOfFloat": short_pct, "shortRatio": short_rat},
        )
```

**Cache key:** `f"short:{ticker}:{today_iso}"`. TTL 24 hours (short interest
updates slowly).

**Step 6.3.3 — `adapters/investor/analyst_signals.py`.**

```python
class AnalystSignalsAdapter:
    def get_signal(self) -> InvestorSubScore:
        info = yf.Ticker(self.ticker, session=self.session).info
        cur = info.get("currentPrice")
        tmean = info.get("targetMeanPrice")
        thigh = info.get("targetHighPrice")
        tlow  = info.get("targetLowPrice")
        rec   = info.get("recommendationMean")
        n     = info.get("numberOfAnalystOpinions") or 0
        if n < 3 or any(x is None for x in (cur, tmean, thigh, tlow, rec)):
            return InvestorSubScore(name="analyst_signal", score=0.0, ok=False,
                                    detail={"numberOfAnalystOpinions": n})
        upside = (tmean - cur) / cur
        disp   = (thigh - tlow) / tmean
        upside_score = clip(upside / 0.40, -1, 1)
        damper = max(0.0, 1.0 - disp)
        rec_score = clip((3.0 - rec) / 2.0, -1, 1)
        combined = 0.5*(upside_score * damper) + 0.5*rec_score
        return InvestorSubScore(
            name="analyst_signal",
            score=clip(combined, -1, 1),
            ok=True,
            detail={"upside": upside, "dispersion": disp, "recMean": rec, "n": n},
        )
```

**Cache key:** `f"analyst:{ticker}:{today_iso}"`. TTL 12 hours.

**Step 6.3.4 — `adapters/investor/social.py`.**

Dependencies reused: `scorers.lexicon.LexiconScorer`, `scorers.finbert_scorer.FinBERTScorer`.
Inject both via constructor so tests can mock.

```python
class SocialAdapter:
    STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{t}.json"
    REDDIT_URLS = [
        "https://www.reddit.com/r/wallstreetbets/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
        "https://www.reddit.com/r/stocks/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
        "https://www.reddit.com/r/investing/search.json?q={t}&restrict_sr=on&sort=new&t=week&limit=50",
    ]

    def __init__(self, settings, cache=None, session=None, finbert=None, lexicon=None):
        ...
        self.finbert = finbert or FinBERTScorer(settings)
        self.lexicon = lexicon or LexiconScorer()

    def get_signal(self) -> InvestorSubScore:
        st_score, st_ok = self._stocktwits()
        rd_score, rd_ok = self._reddit()
        if not (st_ok or rd_ok):
            return InvestorSubScore(name="social", score=0.0, ok=False, detail={})
        if st_ok and rd_ok:
            score = 0.60*st_score + 0.40*rd_score
        else:
            score = st_score if st_ok else rd_score
        return InvestorSubScore(
            name="social", score=clip(score, -1, 1), ok=True,
            detail={"stocktwits": st_score if st_ok else None,
                    "reddit":     rd_score if rd_ok else None},
        )
```

Reddit parse: iterate `data.data.children[].data`, extract `title`,
`selftext`, `score`, `id`. Dedupe by `id` across subreddits. For each post,
score `title + " " + selftext` via `finbert.score_text(text)` and
`lexicon.score_text(text)`; average; weight by `log1p(max(post["score"], 0))`.

**StockTwits parse:** iterate `messages[].entities.sentiment.basic`; count
Bullish/Bearish; if `tagged < 5`, return `(0.0, False)`.

**Cache key:** `f"st:{ticker}:{today_iso_hour}"`, `f"rd:{ticker}:{today_iso_hour}"`.
TTL 1 hour.

**Rate limiting:** StockTwits publicly allows ~200 req/hour; Reddit JSON
~60 req/min with User-Agent. We make 4 calls per node run, well under any
limit. No rate-limiter needed.

**User-Agent requirement (Reddit):** Reddit returns 429/403 without a
non-default User-Agent. Pass `User-Agent: nvda-sentiment-node/0.2 <email>`
from `Settings.sec_user_agent` (reuse existing field).

**Step 6.3.5 — `adapters/investor/broad_market.py` (renamed).**

Rename the class, update imports. Wrap the existing
`get_combined_score() -> tuple[float, bool]` to also expose `get_signal()
-> InvestorSubScore`:

```python
def get_signal(self) -> InvestorSubScore:
    score, ok = self.get_combined_score()
    aaii_raw, aaii_ok = self.get_aaii_score()
    vix_raw,  vix_ok  = self.get_vix_score()
    return InvestorSubScore(
        name="broad_market", score=score, ok=ok,
        detail={"aaii": aaii_raw if aaii_ok else None,
                "vix":  vix_raw  if vix_ok  else None},
    )
```

### 6.4 Investor Branch Aggregator

**Step 6.4.1 — `features/investor_branch.py`.**

```python
from ..adapters.investor.options_flow    import OptionsFlowAdapter
from ..adapters.investor.short_interest  import ShortInterestAdapter
from ..adapters.investor.analyst_signals import AnalystSignalsAdapter
from ..adapters.investor.social          import SocialAdapter
from ..adapters.investor.broad_market    import BroadMarketAdapter
from ..config import INVESTOR_COMPONENT_WEIGHTS
from ..schemas import InvestorSubScore, InvestorBranchOutput


class InvestorBranch:
    """Coordinates the five investor sub-adapters."""

    def __init__(self, settings, *, options=None, shorts=None, analyst=None,
                 social=None, broad_market=None):
        self.settings = settings
        self.options      = options      or OptionsFlowAdapter(settings)
        self.shorts       = shorts       or ShortInterestAdapter(settings)
        self.analyst      = analyst      or AnalystSignalsAdapter(settings)
        self.social       = social       or SocialAdapter(settings)
        self.broad_market = broad_market or BroadMarketAdapter(settings)

    def run(self, *, include_broad_market: bool) -> InvestorBranchOutput:
        subs: list[InvestorSubScore] = []
        for name, adapter in (
            ("options_flow",   self.options),
            ("short_interest", self.shorts),
            ("analyst_signal", self.analyst),
            ("social",         self.social),
        ):
            try:
                subs.append(adapter.get_signal())
            except Exception as exc:
                logger.warning("%s adapter failed: %s", name, exc)
                subs.append(InvestorSubScore(name=name, score=0.0, ok=False, detail={}))

        if include_broad_market:
            try:
                subs.append(self.broad_market.get_signal())
            except Exception as exc:
                logger.warning("broad_market adapter failed: %s", exc)
                subs.append(InvestorSubScore(name="broad_market", score=0.0, ok=False, detail={}))

        investor_component = self._weighted_mean(subs)
        investor_score     = 50.0 + 50.0 * max(-1.0, min(1.0, investor_component))
        ok_any             = any(s.ok for s in subs)
        return InvestorBranchOutput(
            sub_scores=subs,
            investor_component=investor_component,
            investor_score=investor_score,
            ok=ok_any,
        )

    @staticmethod
    def _weighted_mean(subs):
        w = INVESTOR_COMPONENT_WEIGHTS
        # When a sub-source is not ok, redistribute its weight pro-rata to the
        # ones that are ok. This prevents a single failure from anchoring the
        # component near zero.
        usable = [s for s in subs if s.ok]
        if not usable:
            return 0.0
        total_w = sum(w[s.name] for s in usable)
        return sum(w[s.name] * s.score for s in usable) / total_w
```

Use attribute propagation of `use_cache`:

```python
for a in (self.options, self.shorts, self.analyst, self.social, self.broad_market):
    if hasattr(a, "use_cache"):
        a.use_cache = self.settings_use_cache   # set by node.run()
```

### 6.5 Divergence + Combiner

**Step 6.5.1 — `features/combiner.py`.**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CombinerOutput:
    leadership_score:   float     # 0-100
    investor_score:     float     # 0-100
    combined_score:     float     # 0-100
    divergence:         float     # -100 .. +100
    agreement:          bool      # signs of leadership & investor components agree
    adjustment:         float     # divergence adjustment applied to base blend (signed, in component units)

def combine(
    *,
    leadership_component: float,       # [-1, +1]
    investor_component:   float,       # [-1, +1]
    alpha:        float,               # leadership weight in base blend
    lambda_neg:   float,               # bearish-divergence penalty (L>I)
    lambda_pos:   float,               # contrarian premium (I>L)
    investor_branch_ok:   bool,        # if False → fall back to leadership-only
) -> CombinerOutput:
    L = max(-1.0, min(1.0, leadership_component))
    I = max(-1.0, min(1.0, investor_component))
    leadership_score = 50.0 + 50.0 * L
    investor_score   = 50.0 + 50.0 * I

    if not investor_branch_ok:
        # degrade to leadership-only
        return CombinerOutput(
            leadership_score=leadership_score,
            investor_score=investor_score,       # will be 50.0 when component=0
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
            # leadership bullish, investor bearish  → bearish penalty on combined
            adjustment = -lambda_neg * d
        else:
            # leadership bearish, investor bullish → modest contrarian premium
            adjustment = +lambda_pos * d
    combined_raw = max(-1.0, min(1.0, base + adjustment))
    combined_score = 50.0 + 50.0 * combined_raw
    return CombinerOutput(
        leadership_score=leadership_score,
        investor_score=investor_score,
        combined_score=combined_score,
        divergence=leadership_score - investor_score,
        agreement=signs_agree,
        adjustment=adjustment,
    )
```

Unit tests (§9.1) cover every quadrant, the boundary `L·I = 0`, clipping,
and the `investor_branch_ok=False` fallback.

**Step 6.5.2 — Extend `features/confidence.py`.** Add kwargs (all defaulted)
reflecting investor-branch success without changing existing call sites
unless necessary:

```python
def compute_confidence(
    ...existing args...,
    investor_branch_ok: bool = False,            # NEW
    investor_subsources_ok: int = 0,             # NEW (count of 5)
):
    # Existing penalties/bonuses unchanged.
    # NEW bonus: +0.05 if ≥ 3 investor subsources ok, +0.02 if ≥ 1.
    # NEW penalty: -0.05 if investor branch requested but fully failed.
```

Keep the total bonus cap at 0.40 (preserves §28.2 invariant).

**Step 6.5.3 — Extend `features/signal_builder.py`.** Add a new tier:

- "Market positioning is bullish/bearish" (investor component sign).
- "Expectations gap: management bullish while market skeptical"
  (divergence sign + magnitude threshold |divergence| > 20).
- "Expectations gap: management cautious while market optimistic" (inverse).
- "Leadership and market signals are aligned" (agreement, |divergence| < 10).

Cap total signals at 8. Ordering: leadership tiers first, investor tier
next, divergence tier last.

### 6.6 Node Orchestration Update

**Step 6.6.1 — Modify `NVDASentimentNode.__init__`.**

```python
def __init__(
    self,
    settings: Settings | None = None,
    *,
    sec_adapter=None,
    ir_adapter=None,
    investor_branch: InvestorBranch | None = None,   # NEW (replaces market_context kwarg)
    section_extractor=None,
    section_scorer=None,
):
    ...
    self.investor_branch = investor_branch or InvestorBranch(self.settings)
```

**Backwards-compat shim:** accept a `market_context=...` kwarg; if supplied,
wire it into a default `InvestorBranch(..., broad_market=market_context)`.
Emit a `DeprecationWarning`. Remove the shim in the next release.

**Step 6.6.2 — Modify `NVDASentimentNode.run`.**

```python
def run(self, request: SentimentRequest) -> SentimentResponse:
    # existing leadership pipeline up to `filing_tone`, `filing_delta`, `guidance_tone`
    leadership_component = compute_leadership_component(filing_tone, filing_delta, guidance_tone)

    # investor branch
    if request.include_investor_branch:
        for attr in ("options","shorts","analyst","social","broad_market"):
            a = getattr(self.investor_branch, attr)
            if hasattr(a, "use_cache"):
                a.use_cache = request.use_cache
        ib = self.investor_branch.run(include_broad_market=request.include_market_context)
    else:
        ib = InvestorBranchOutput(sub_scores=[], investor_component=0.0, investor_score=50.0, ok=False)

    # combine
    cmb = combine(
        leadership_component=leadership_component,
        investor_component=ib.investor_component,
        alpha=self.settings.combiner_alpha,
        lambda_neg=self.settings.combiner_lambda_neg,
        lambda_pos=self.settings.combiner_lambda_pos,
        investor_branch_ok=ib.ok,
    )

    # confidence (now takes investor-branch inputs)
    confidence = compute_confidence(
        ...,
        investor_branch_ok=ib.ok,
        investor_subsources_ok=sum(1 for s in ib.sub_scores if s.ok),
    )

    # signals (extended)
    signals = build_signals(..., investor_component=ib.investor_component, divergence=cmb.divergence)

    components = {
        "filing_tone":         round(filing_tone, 3),
        "filing_delta":        round(filing_delta, 3),
        "guidance_tone":       round(guidance_tone, 3),
        "options_flow":        _get_sub(ib, "options_flow"),
        "short_interest":      _get_sub(ib, "short_interest"),
        "analyst_signal":      _get_sub(ib, "analyst_signal"),
        "social":              _get_sub(ib, "social"),
        "broad_market":        _get_sub(ib, "broad_market"),
        "leadership_component":round(leadership_component, 3),
        "investor_component":  round(ib.investor_component, 3),
    }

    return SentimentResponse(
        ticker=request.ticker,
        as_of_date=request.as_of_date,
        market_sentiment_score=round(cmb.combined_score, 1),   # alias
        leadership_score=round(cmb.leadership_score, 1),
        investor_score=round(cmb.investor_score, 1),
        combined_score=round(cmb.combined_score, 1),
        divergence=round(cmb.divergence, 1),
        label=_label_for(cmb.combined_score),
        confidence=round(confidence, 2),
        components=components,
        signals=signals,
        source_coverage=coverage,
        metadata={...},
    )
```

Helper: `_get_sub(ib, name) -> float` returns the sub-score value or 0.0 if
absent.

### 6.7 CLI

**Step 6.7.1 — Add two flags to `node.py` Typer command:**

```python
include_investor_branch: bool = typer.Option(True, "--include-investor-branch/--no-investor-branch"),
show_sub_components:     bool = typer.Option(False, "--show-sub-components"),
```

With `--show-sub-components`, additionally print a table of the five
investor sub-scores and their `ok` flags before the JSON body. For debugging
only.

---

## 7. Tunable Combiner

### 7.1 Functional form (final)

Given leadership component `L ∈ [-1, +1]` and investor component
`I ∈ [-1, +1]`:

```
base        = α·L + (1 − α)·I
agreement   = (L · I ≥ 0)
if agreement:
    adjustment = 0
elif L > I:        # leadership more bullish than investors
    adjustment = −λ_neg · |L − I|
else:              # investors more bullish than leadership
    adjustment = +λ_pos · |L − I|

combined  = clip(base + adjustment, −1, +1)
score_100 = 50 + 50·combined
```

**Properties (provable):**

1. Idempotent when `L = I`: `adjustment = 0`, `combined = L = I`.
2. Monotonic in `L` and `I` within their own sign quadrants.
3. When signs agree, the combiner is a pure convex blend (no adjustment).
4. The asymmetry `λ_neg > λ_pos` encodes the empirical prior: the
   management-optimistic / market-skeptical quadrant is historically
   better documented as a bearish precursor than the inverse (see Appendix B).
5. Bounded: `|combined| ≤ 1` enforced by the final clip.

### 7.2 Default weights

| Symbol | Default | Rationale |
|---|---|---|
| `α` | 0.55 | Management-tone is the richer, more deliberate signal — primary-source text by insiders. Investor signal is noisier. We give leadership a slight edge. |
| `λ_neg` | 0.25 | Strong penalty when mgmt bullish outruns market. Calibrated so that `L = +0.5, I = −0.5` produces `adjustment = −0.25` → combined `≈ −0.25·100` shift from base 50 to ~25, a decisive bearish read. |
| `λ_pos` | 0.10 | Weak premium when market bullish outruns cautious mgmt. Contrarian signal is less well-validated in the literature; we don't over-reward it. Symmetric example: `L = −0.5, I = +0.5` gives `adjustment = +0.10` → modest upward nudge. |

Equivalence check against prior behavior: if the investor branch is disabled
(`include_investor_branch=False`), `combined_score = leadership_score`.
Previous "market_sentiment_score" behavior is recovered *up to the 10%
broad-market contribution* by disabling the investor branch entirely. This
is the acceptance criterion for "does not silently break consumers that
upgrade without reconfiguring."

### 7.3 Tunability

All three weights are `Settings` fields with defaults from
`COMBINER_DEFAULTS`. Users can override:

```python
settings = Settings(combiner_alpha=0.65, combiner_lambda_neg=0.30, combiner_lambda_pos=0.05)
```

CLI passthrough is **not** added in this pass (avoid flag proliferation);
users who want to tune override the default `Settings` in Python code, or
edit `config.py` if they're pinning the deployment.

---

## 8. Calibration Harness

### 8.1 Target

For a given snapshot, compute the **analyst EPS-revision direction over the
past 30 days** as `rev_score ∈ [-1, +1]` (definition in §3.6). This is the
external signal we treat as "truth" for weight sanity-checking.

**Honest disclaimer preserved in the harness output:** with one snapshot
per ticker per day and NVDA's single ticker, we have **O(1) data points per
run**. This is not a statistical calibration. It is a **directional
sanity check**: does the combiner's output agree in sign and rough magnitude
with where analysts have been revising estimates?

### 8.2 Script — `tools/calibrate_weights.py`

```python
#!/usr/bin/env python3
"""Calibration harness — produces a same-day sanity report, not a fit.

Usage:
    python -m tools.calibrate_weights --ticker NVDA
    python -m tools.calibrate_weights --ticker NVDA --grid
"""

import argparse, json
from dataclasses import asdict
from datetime import date
import yfinance as yf

from nvda_sentiment.config import Settings, COMBINER_DEFAULTS
from nvda_sentiment.node   import NVDASentimentNode
from nvda_sentiment.schemas import SentimentRequest
from nvda_sentiment.features.combiner import combine


def eps_revision_target(ticker: str) -> float:
    t = yf.Ticker(ticker)
    rev = t.eps_revisions        # DataFrame indexed by period
    if rev is None or rev.empty:
        return 0.0
    weights = {"0q": 0.4, "+1q": 0.3, "0y": 0.2, "+1y": 0.1}
    total, wsum = 0.0, 0.0
    for period, w in weights.items():
        if period not in rev.index: continue
        up = float(rev.loc[period, "upLast30days"])
        dn = float(rev.loc[period, "downLast30days"])
        if up + dn <= 0: continue
        total += w * (up - dn) / (up + dn)
        wsum += w
    return total / wsum if wsum > 0 else 0.0


def grid_search(settings, L, I, target) -> list[dict]:
    """Report disagreement between combiner output and target across a weight grid."""
    rows = []
    for alpha in (0.40, 0.50, 0.55, 0.60, 0.70):
        for lneg in (0.15, 0.25, 0.35):
            for lpos in (0.05, 0.10, 0.15):
                out = combine(leadership_component=L, investor_component=I,
                              alpha=alpha, lambda_neg=lneg, lambda_pos=lpos,
                              investor_branch_ok=True)
                combined_component = (out.combined_score - 50.0) / 50.0
                # "agreement" with target: product of signs, plus magnitude closeness
                sign_match = (combined_component * target) >= 0
                dist = abs(combined_component - target)
                rows.append({"alpha": alpha, "lneg": lneg, "lpos": lpos,
                             "combined": round(combined_component, 3),
                             "target": round(target, 3),
                             "sign_match": sign_match,
                             "dist": round(dist, 3)})
    rows.sort(key=lambda r: (not r["sign_match"], r["dist"]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--grid", action="store_true")
    args = p.parse_args()

    target = eps_revision_target(args.ticker)
    print(f"[target] weighted EPS-revision direction: {target:+.3f}")

    settings = Settings()
    node = NVDASentimentNode(settings)
    resp = node.run(SentimentRequest(ticker=args.ticker, as_of_date=date.today()))

    L = resp.components["leadership_component"]
    I = resp.components["investor_component"]
    print(f"[node]   leadership_component={L:+.3f}  investor_component={I:+.3f}")
    print(f"[node]   combined_score={resp.combined_score}  divergence={resp.divergence}")

    if args.grid:
        rows = grid_search(settings, L, I, target)
        print("\n[grid] top 10 (sign_match first, then closest to target):")
        print(f"{'alpha':>6} {'lneg':>6} {'lpos':>6} {'combined':>10} {'target':>8} {'match':>6} {'dist':>6}")
        for r in rows[:10]:
            print(f"{r['alpha']:>6} {r['lneg']:>6} {r['lpos']:>6} {r['combined']:>10} "
                  f"{r['target']:>8} {str(r['sign_match']):>6} {r['dist']:>6}")

    report = {
        "ticker": args.ticker,
        "target": target,
        "leadership_component": L,
        "investor_component": I,
        "combined_score": resp.combined_score,
        "current_defaults": COMBINER_DEFAULTS,
    }
    print("\n[report]")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
```

### 8.3 How to interpret the output

- If `sign(combined_component) == sign(target)` with the current defaults,
  the combiner is directionally consistent with consensus revision flow.
  Ship as-is.
- If sign disagrees, consult the `--grid` output for weights that would
  flip the sign. If a nearby weight setting does, consider whether the
  current defaults are wrong *or* whether our sentiment reading is
  genuinely out of step with analysts (which is informative — divergence
  is the whole point).
- Do **not** auto-commit grid-search winners to `config.py`. Humans decide.

### 8.4 Limits of the calibration

- **One data point per run.** This is a snapshot sanity check, not a fit.
- **Analyst revisions lag information.** They confirm direction but are
  often already priced in. A divergence between our score and revisions
  may reflect our signal leading, not being wrong.
- **NVDA-only.** Cross-sectional calibration across tickers is out of scope
  today but is the correct follow-up: run the harness against S&P 100
  mega-caps to derive weights on a richer panel.
- **No time-series.** We cannot measure whether yesterday's score predicted
  today's returns; that would require the separate work of building a
  per-day archive of the node's outputs.

---

## 9. Testing Plan

### 9.1 Unit tests (new)

**`test_options_flow_adapter.py`** — mocks `yf.Ticker` with a fixture that
returns canned `options` tuple and `option_chain(expiry)` results. Asserts:
- PCR score bucketing at each bucket boundary (5 cases).
- IV z-score bucketing (4 cases).
- Combined score = 0.70·pcr + 0.30·iv (2 canned inputs).
- Network failure → `(0.0, ok=False)` returned, warning logged.

**`test_short_interest_adapter.py`** — mocks `yf.Ticker().info` dict.
- Both fields present → weighted combination.
- Only `shortPercentOfFloat` present → pct-only.
- Only `shortRatio` present → ratio-only.
- Neither → `(0.0, ok=False)`.

**`test_analyst_signals_adapter.py`**.
- Full info dict → expected combined score (3 canned inputs spanning bull/neutral/bear).
- `numberOfAnalystOpinions < 3` → `(0.0, ok=False)`.
- Missing `currentPrice` → `(0.0, ok=False)`.
- `targetHighPrice == targetLowPrice` → `damper = 1.0` (no damping).

**`test_social_adapter.py`** — mocks the HTTP layer, not yfinance.
- StockTwits response with 20 Bullish / 5 Bearish → `+0.60`.
- StockTwits response with `tagged < 5` → not ok.
- Reddit response with 3 positive posts → lexicon+finbert blended score > 0.
- Both sources fail → combined `(0.0, ok=False)`.
- Only one source ok → uses that one.

**`test_investor_branch.py`**.
- All 5 sub-sources ok → weighted mean equals hand-computed value.
- 2 fail, 3 ok → weights renormalized over the 3 ok.
- All fail → `investor_component = 0.0`, `ok=False`.
- `include_broad_market=False` → broad_market excluded, weights renormalized over 4.

**`test_combiner.py`** — every quadrant of (sign L, sign I):
- `L=+0.5, I=+0.3` (both positive, agree) → base=0.5·0.5 + 0.5·0.3 (with α=0.55: 0.55·0.5+0.45·0.3 = 0.41), adjustment=0.
- `L=+0.5, I=−0.5` (mgmt bull, mkt bear) → adjustment = −λ_neg · 1.0 = −0.25. base = 0.55·0.5 + 0.45·(−0.5) = 0.05. combined = −0.20 → score=40.
- `L=−0.5, I=+0.5` (mgmt bear, mkt bull) → adjustment = +λ_pos·1.0 = +0.10. base = 0.55·(−0.5)+0.45·0.5 = −0.05. combined = +0.05 → score=52.5.
- Boundary `L=0, I=+0.5` → `L·I = 0`, treated as agreement.
- Clipping: `L=+1, I=+1` → base=+1, adj=0, combined clipped to +1 → score=100.
- Clipping: `L=−1, I=+1` (and λ_pos=+0.30) → combined = 0.55·(−1)+0.45·1+0.60=0.50 → score=75.
- `investor_branch_ok=False` → combined = leadership_score exactly.

**`test_confidence.py`** — add cases: investor branch ok with 3 subsources ok,
with 0 subsources ok; verify cap 0.40 bonus preserved.

### 9.2 Integration tests

**`test_end_to_end_dual_branch.py`** — extend `test_end_to_end.py` pattern.
Fakes for all 5 investor adapters plus SEC/IR. Run full `node.run()`. Assert:
- Response has all new fields populated.
- `market_sentiment_score == combined_score` exactly (alias invariant).
- With mgmt-bullish filings + mgmt-bearish investor fakes → `combined_score < leadership_score`.
- With both bullish → `combined_score` ≈ weighted blend (within rounding).
- `--no-investor-branch` → `combined_score == leadership_score`.

**`test_determinism.py`** — extend with investor-branch fakes; re-assert
byte-identical across two runs.

### 9.3 Determinism

- All new adapters: no sets, no dict-ordering-dependent iteration, no RNG.
- `SocialAdapter`: when iterating Reddit posts, sort by `id` before scoring
  (dedupe is already set-based, but aggregation weights must be applied in
  deterministic order).
- FinBERT already deterministic per existing config.

### 9.4 Live smoke

After all tests green:

```bash
nvda-sentiment --as-of-date 2026-04-21 --lookback-quarters 2 --show-sub-components
```

**Success criteria:** CLI prints (a) sub-component table with at least 3
of 5 investor sub-sources `ok=True`, (b) a valid JSON blob containing all
new fields, (c) no unhandled exceptions.

Then the calibration harness:

```bash
python -m tools.calibrate_weights --ticker NVDA --grid
```

Print inspection: weighted EPS-revision target ~+0.77 for NVDA today;
combiner's `combined_component` should be in the `[+0.3, +0.9]` range for
consistency. If not, investigate (don't auto-tune).

---

## 10. Verification Checklist

Every item must be checked before marking done.

- [ ] `pyproject.toml` adds `yfinance>=0.2.40`.
- [ ] `pip install -e .` completes with no errors.
- [ ] `LEADERSHIP_COMPONENT_WEIGHTS` and `INVESTOR_COMPONENT_WEIGHTS` each sum to 1.0 ± 1e-6 (asserted in `config.py` or a test).
- [ ] `adapters/investor/` subpackage exists with all 5 files.
- [ ] `adapters/market_context.py` removed; import alias in `adapters/__init__.py` preserves backwards compatibility.
- [ ] `features/investor_branch.py` and `features/combiner.py` exist and are imported by `node.py`.
- [ ] `schemas.py` includes `leadership_score`, `investor_score`, `combined_score`, `divergence` fields; `market_sentiment_score` retained as alias.
- [ ] `SentimentRequest.include_investor_branch` defaults to `True`.
- [ ] All existing tests (47 + 1 skipped) continue to pass after refactor.
- [ ] All new tests pass (at least: 5 adapter files × ~4 cases, combiner × ~7 cases, investor_branch × ~4 cases, integration × ~4 cases).
- [ ] `test_determinism.py` passes with investor branch enabled.
- [ ] CLI `--show-sub-components` prints the table.
- [ ] `python -m tools.calibrate_weights --ticker NVDA --grid` prints target, node output, and grid — no exceptions.
- [ ] Live `nvda-sentiment` run against NVDA produces all new response fields populated; at least 3 of 5 investor sub-sources return `ok=True`.
- [ ] `combined_score` equals `leadership_score` exactly when `--no-investor-branch` is passed.
- [ ] `combined_score` equals `round(50 + 50·combined_raw, 1)` per the combiner formula — verified in at least one integration test.
- [ ] No test requires network.

---

## 11. Known Limitations (documented, not blockers)

1. **yfinance is an unofficial scrape** of Yahoo Finance. It can break on
   site changes. Pin version in `pyproject.toml`; mock in tests; treat
   adapter failures as expected degradation.
2. **Reddit JSON** occasionally returns 429 without warning. Graceful
   `ok=False` handles this; social score falls back to StockTwits alone.
3. **StockTwits user-tagged sentiment** is self-reported by posters, not
   machine-inferred. It encodes bias of that platform's user base (retail,
   typically bullish). Acceptable signal for divergence purposes.
4. **No time-series.** Today's build produces today's snapshot. The
   harness cannot validate predictive power, only cross-sectional
   consistency. Follow-up: persist daily outputs to enable backtesting.
5. **NVDA-only.** Fiscal calendar, ticker string, and CIK are hardcoded.
   Generalizing is a separate project.
6. **Combiner weights are theoretically motivated, not empirically fit.**
   See §8.4. This is a deliberate choice given the same-day constraint.
7. **Options data is delayed ~15 minutes** on Yahoo. Short-interest fields
   lag actual settlement by ~1–2 trading days. Accept and document.

---

## Appendix A — API Reference (curated, verified)

### yfinance surfaces used

```python
import yfinance as yf
t = yf.Ticker("NVDA", session=session)

# Options
t.options                      # tuple[str, ...] of expiry dates "YYYY-MM-DD"
t.option_chain(expiry).calls   # pd.DataFrame
t.option_chain(expiry).puts    # pd.DataFrame
# Columns: contractSymbol, lastTradeDate, strike, lastPrice, bid, ask,
#          change, percentChange, volume, openInterest, impliedVolatility,
#          inTheMoney, contractSize, currency

# Info (dict with many keys; we use these)
info = t.info
info["currentPrice"]                  # float
info["shortRatio"]                    # float (days to cover)
info["shortPercentOfFloat"]           # float (fraction, e.g. 0.0121)
info["targetMeanPrice"]               # float
info["targetMedianPrice"]             # float
info["targetHighPrice"]               # float
info["targetLowPrice"]                # float
info["numberOfAnalystOpinions"]       # int
info["recommendationMean"]            # float in [1, 5]

# EPS trend / revisions (harness only)
t.eps_trend                  # pd.DataFrame, index = {"0q","+1q","0y","+1y"}
                             # cols = ["current","7daysAgo","30daysAgo","60daysAgo","90daysAgo","currency"]
t.eps_revisions              # pd.DataFrame, index = {"0q","+1q","0y","+1y"}
                             # cols = ["upLast7days","upLast30days","downLast30days","downLast7Days","currency"]
```

### StockTwits public streams

```
GET https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
Headers: User-Agent required
Response: { "cursor": {...}, "messages": [
    { "id": int,
      "body": str,
      "created_at": "YYYY-MM-DDTHH:MM:SSZ",
      "user": {...},
      "entities": { "sentiment": {"basic": "Bullish"|"Bearish"|null}, ... },
      ...
    }, ... ],
    "response": {"status": 200}
}
Typical count: 30 messages.
Rate limit: ~200 requests/hour per IP (undocumented but observed).
```

### Reddit public JSON

```
GET https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=on&sort=new&t=week&limit={N}
Headers: User-Agent required (default UA will be throttled/blocked)
Response: { "kind": "Listing", "data": { "children": [
    { "kind": "t3", "data": {
        "id": str, "title": str, "selftext": str,
        "score": int, "num_comments": int, "created_utc": float,
        ...
    } }, ... ], "after": str|null, "before": str|null
}}
Subreddits used: wallstreetbets, stocks, investing.
```

### FRED VIX CSV (already in use, unchanged)

```
GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
Response: CSV with header including "VIXCLS" or "value" column.
```

### AAII (already in use, unchanged)

```
GET https://www.aaii.com/sentimentsurvey
HTML scrape for "Bullish NN.N%", "Neutral NN.N%", "Bearish NN.N%".
Occasionally 403s; node tolerates.
```

### SEC EDGAR (already in use, unchanged)

Already documented in existing code and README.

---

## Appendix B — Literature References

- **Price, Doran, Peterson, Bliss (2012)** — *Earnings Conference Calls and Stock Returns: The Incremental Informativeness of Textual Tone.* Journal of Banking & Finance. Finds that earnings-call language tone predicts post-announcement returns and trading volume. Motivates the leadership branch.
- **Larcker & Zakolyukina (2012)** — *Detecting Deceptive Discussions in Conference Calls.* Journal of Accounting Research. Management language patterns correlate with subsequent restatements. Motivates the "mgmt bullish + market skeptical → bearish" asymmetry (`λ_neg > λ_pos`).
- **Tetlock (2007)** — *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.* Journal of Finance. High pessimism in media predicts downward pressure and reversal. Motivates the investor branch's social component.
- **Loughran & McDonald (2011)** — *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance. The lexicon already in use is derived from this work; references are for the benefit of new readers.
- **Stickel (1991)** — *Common Stock Returns Surrounding Earnings Forecast Revisions.* Accounting Review. Classic result: analyst revisions predict short-horizon returns. Motivates EPS revisions as the calibration target.
- **Frazzini & Lamont (2008)** — *Dumb Money: Mutual Fund Flows and the Cross-Section of Stock Returns.* Documents retail-sentiment reversal. Caution: do not over-weight retail-heavy social sentiment as a directional signal on long horizons.
- **Da, Engelberg, Gao (2011)** — *In Search of Attention.* Journal of Finance. Retail attention proxies (search volume, posts) predict short-term price pressure. Motivates weighting Reddit posts by log-score.

---

## Appendix C — Weight Derivation Notes

### Why α = 0.55 (leadership weight)

The leadership branch processes primary-source, legally-filed text produced
by insiders with a duty to be accurate. The investor branch aggregates
derived and crowd signals. Information-theoretically, leadership is the
richer per-document signal. Empirically (Price et al. 2012, Larcker &
Zakolyukina 2012), conference-call tone has a documented post-announcement
return signature; retail social sentiment has documented short-term
reversal (Frazzini & Lamont 2008). A 55/45 split gives leadership the edge
without deprecating the investor signal. Tunable.

### Why λ_neg = 0.25 ≈ 2.5·λ_pos

The asymmetry encodes an *empirically grounded prior*: when management
language outruns market reaction, literature on deceptive disclosure
(Larcker & Zakolyukina 2012) and earnings-management detection gives the
bearish-follow-through interpretation substantial weight. The reverse case
— management cautious while market optimistic — has mixed interpretations:
sometimes contrarian bullish (sandbagged guidance), sometimes late-cycle
retail melt-up (Frazzini & Lamont 2008). The literature does not support
betting equally on either side; the prior conservatively under-weights the
contrarian case.

### Sub-component weights within the investor branch

| Sub | Weight | Rationale |
|---|---|---|
| options_flow | 0.30 | Options are the most information-rich investor signal: informed traders prefer options for leverage; directional flow contains real-money conviction. |
| analyst_signal | 0.25 | Sell-side coverage is dense for NVDA (56 analysts) — signal is robust. Recommendation mean + price target direction are independent inputs. |
| short_interest | 0.15 | Slow-moving but reliable. NVDA short interest is low (1.21%); the signal is marginally informative near the boundary, more so during spikes. |
| social | 0.15 | Noisy, high-frequency, retail-biased. Useful as divergence input but not as the main driver. |
| broad_market | 0.15 | AAII+VIX are macro proxies, not NVDA-specific. Kept for tail-risk context; not the primary signal. |

Weights are theoretically motivated. The calibration harness (§8) validates
that, under these weights, the combiner's output is directionally consistent
with consensus EPS-revision direction. If it is not, weights are the first
place to look, but human judgment decides any change.

---

## Appendix D — Acceptance Definition

This redesign is complete and correct when every box in §10 is checked
AND:

1. The full test suite (existing + new) is green in under 5 seconds.
2. The live CLI produces a schema-valid response with ≥ 3 of 5 investor
   sub-sources reporting `ok=True` against live NVDA data.
3. The calibration harness runs without exception and prints both the
   target and the node's output.
4. No new lint/type warnings are introduced.
5. The response's `market_sentiment_score` alias exactly equals
   `combined_score` in every integration test.

When all five conditions hold, the node is ready to ship.
