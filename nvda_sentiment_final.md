# NVIDIA Sentiment Node MVP — Full Implementation Playbook

> **Document version: 0.2** (supersedes 0.1). Key changes from 0.1:
> - FinBERT (transformers + torch) is now an **optional** install extra, not a required dep (§9)
> - Risk factor sections are excluded from absolute `filing_tone` and routed into `filing_delta` as a dedicated risk-delta signal (§22.1, §24)
> - Fiscal quarter bucketing uses **NVIDIA's hardcoded fiscal calendar**, not calendar quarters (§24.2)
> - VIX pulls from FRED's CSV endpoint (stable) instead of HTML scraping (§15.3)
> - SEC filing HTML cache TTL is **permanent** (filings are immutable) (§30.3)
> - Confidence arithmetic reworked so ceiling is actually reachable and floor is consistent with §36.5 fallback (§28)
> - Signal builder upgraded to pull from document-level and section-level scores, including a Q&A-vs-prepared-remarks divergence signal (§29)
> - Determinism requirements made explicit for FinBERT (§19.4), with a dedicated determinism test (§41.7)
> - `filing_parser.py` and `ir_parser.py` removed as unneeded indirection (§38.8)
> - SEC rate limiting and user-agent format requirements made explicit (§10, §13.9)
>
> If an implementer or LLM encounters a conflict between this version and any prior version, **this version wins**.

## 0. Document Purpose

This document is the **single source of truth** for building a **standalone NVIDIA sentiment node MVP**.

A capable engineer or coding agent should be able to implement the node **from scratch** using only this document, without additional research.

This document is optimized for:
- **implementation speed**
- **clarity**
- **low operational complexity**
- **robustness**
- **LLM execution efficiency**

This is **not** a research system, agentic crawler, or general-purpose sentiment platform.  
This is a **small, deterministic, production-friendly component** that estimates **NVDA market sentiment** from:
1. official NVIDIA filings and investor communications
2. a small investor-risk context layer

---

# 1. Product Definition

## 1.1 What this node is

A standalone Python component that returns a **single NVIDIA market sentiment score** on a 0–100 scale, plus a breakdown of the underlying components.

It should be:
- easy to run locally
- easy to integrate into a larger valuation system
- deterministic
- understandable
- testable
- based on stable, reputable data sources

## 1.2 What this node is NOT

Do **not** build any of the following into the MVP:
- social media scraping
- news scraping
- Reddit/X/Twitter ingestion
- browser automation
- vector databases
- retrieval orchestration frameworks
- multi-agent systems
- distributed pipelines
- streaming infrastructure
- generic sentiment platform abstractions
- options-flow sentiment
- analyst feed integrations
- paid data vendor dependencies

If a design decision makes the system materially more complicated, it is probably wrong for this MVP.

---

# 2. Core Philosophy

## 2.1 Design principle

The node should estimate **market sentiment about NVIDIA**, not generic text sentiment and not pure consumer sentiment.

For NVIDIA, the most useful stable signals are:
- management tone
- tone changes across filings
- guidance tone
- broad investor risk appetite

## 2.2 MVP success criteria

The MVP is successful if it:
1. ingests official NVIDIA and SEC source material reliably
2. extracts the most relevant sections
3. scores tone consistently
4. compares current tone with prior tone
5. emits a single interpretable score
6. can be called from another system with simple I/O

## 2.3 Non-goals

The node does **not** need to:
- predict price directly
- replace a full equity research workflow
- infer exact consumer product satisfaction
- parse every SEC filing type under the sun
- be state of the art

The goal is **impressive, robust, simple, and useful**.

---

# 3. Final Output Contract

The node must expose one public method:

```python
result = NVDASentimentNode().run(request)
```

## 3.1 Input schema

```json
{
  "ticker": "NVDA",
  "as_of_date": "2026-04-20",
  "lookback_quarters": 4,
  "include_market_context": true,
  "use_cache": true
}
```

## 3.2 Output schema

```json
{
  "ticker": "NVDA",
  "as_of_date": "2026-04-20",
  "market_sentiment_score": 68.4,
  "label": "mildly bullish",
  "confidence": 0.80,
  "components": {
    "filing_tone": 0.31,
    "filing_delta": 0.12,
    "guidance_tone": 0.28,
    "investor_context": 0.09
  },
  "signals": [
    "Management tone is positive across recent official materials",
    "Tone improved versus the prior comparable period",
    "Risk-factor language was substantively stable versus prior filing",
    "Forward-looking guidance language is constructive",
    "Analyst Q&A tone ran above prepared remarks — management handled pushback well",
    "Broader investor risk appetite is supportive"
  ],
  "source_coverage": {
    "10k_count": 1,
    "10q_count": 3,
    "8k_count": 2,
    "earnings_release_count": 4,
    "transcript_count": 4,
    "cfo_commentary_count": 4
  },
  "metadata": {
    "generated_at": "2026-04-20T12:34:56Z",
    "node_version": "0.1.0",
    "warnings": []
  }
}
```

---

# 4. Recommended Technology Stack

Keep the stack small.

## 4.1 Language

- Python 3.11

## 4.2 Required libraries

Use these libraries unless there is a strong reason not to:

**Required (core install):**
- `pydantic` — schemas and validation
- `requests` — HTTP client
- `beautifulsoup4` — HTML parsing
- `lxml` — HTML/XML parsing backend
- `pandas` — tabular aggregation
- `numpy` — math utilities
- `python-dateutil` — date parsing
- `typer` — optional CLI
- `pytest` — tests

**Optional (FinBERT extra):**
- `transformers` — FinBERT inference
- `torch` — model runtime

The node MUST run without the FinBERT extras installed, falling back to lexicon-only scoring. See §9.1 and §36.3.

## 4.3 Do not use

Do not use the following in the MVP:
- Airflow
- Prefect
- LangChain
- LlamaIndex
- Celery
- Kafka
- vector stores
- database servers
- Selenium/Playwright

If you need persistence, use the filesystem.

---

# 5. Data Sources

Use only stable, official, or highly reputable sources.

## 5.1 Tier 1 sources — required

These are the core sources for the MVP.

### A. SEC EDGAR
Use SEC data for:
- 10-K
- 10-Q
- 8-K
- company submissions JSON
- filing URLs

### B. NVIDIA Investor Relations
Use NVIDIA IR for:
- quarterly earnings press releases
- webcast transcripts
- CFO commentary
- quarterly results pages
- form 10-Q / 10-K links if useful

## 5.2 Tier 2 sources — optional but recommended

These provide a small investor sentiment context layer.

### C. AAII Investor Sentiment Survey
Use only the latest values:
- bullish %
- neutral %
- bearish %

### D. VIX from FRED
Use only the latest daily close or a short recent average.

## 5.3 Do not add in MVP

Do not add:
- analyst estimates feeds
- Google Trends
- app review sentiment
- product review scraping
- earnings whisper sites
- blog/news aggregators

---

# 6. High-Level Architecture

The system has 6 layers:

1. **input validation**
2. **data fetching**
3. **document parsing**
4. **section extraction**
5. **scoring**
6. **aggregation and output formatting**

## 6.1 Architecture diagram

```text
Request
  -> Validator
  -> Fetch official sources
  -> Parse documents into plain text
  -> Extract relevant sections
  -> Score sections
  -> Compare with prior periods
  -> Add investor context
  -> Aggregate into final score
  -> Return response
```

---

# 7. Repository Structure

Implement this exact or near-exact structure.

```text
nvda_sentiment_node/
  README.md
  pyproject.toml
  .env.example

  nvda_sentiment/
    __init__.py
    config.py
    schemas.py
    node.py

    adapters/
      __init__.py
      sec_api.py
      nvidia_ir.py
      market_context.py

    parsers/
      __init__.py
      html_to_text.py
      section_extractor.py

    scorers/
      __init__.py
      finbert_scorer.py
      lexicon.py
      section_scorer.py
      composite.py

    features/
      __init__.py
      filing_delta.py
      confidence.py
      signal_builder.py

    utils/
      __init__.py
      cache.py
      dates.py
      text.py
      logging.py
      rate_limiter.py

  tests/
    test_schemas.py
    test_text_utils.py
    test_section_extractor.py
    test_finbert_scorer.py
    test_composite.py
    test_determinism.py
    test_end_to_end.py
```

---

# 8. Exact Build Order

Implement in this order. Do not jump around.

## Step 1 — Initialize package
Create the repository structure and dependencies.

## Step 2 — Define schemas
Implement request/response models and internal document models.

## Step 3 — Implement SEC adapter
Fetch company submissions and relevant filing URLs.

## Step 4 — Implement NVIDIA IR adapter
Fetch quarterly results assets and transcript/CFO commentary URLs.

## Step 5 — Implement text extraction
Convert HTML documents into clean plain text.

## Step 6 — Implement section extraction
Extract:
- MD&A
- outlook/guidance language
- risk factors
- prepared remarks
- Q&A
- press release summary bullets
- CFO commentary body

## Step 7 — Implement scoring
Build:
- FinBERT scorer
- lexicon scorer
- section scorer
- document scorer

## Step 8 — Implement filing delta
Compare current quarter tone to prior comparable quarter or prior filing.

## Step 9 — Implement investor context
Add AAII + VIX small overlay.

## Step 10 — Implement composite score
Combine all sub-scores into one final output.

## Step 11 — Implement confidence
Compute confidence based on source availability and score agreement.

## Step 12 — Implement CLI / direct runner
Allow easy local execution for testing.

## Step 13 — Write tests
Focus on deterministic behavior and section extraction correctness.

---

# 9. Exact Package Setup

## 9.1 `pyproject.toml`

Use this as a starting point.

```toml
[project]
name = "nvda-sentiment-node"
version = "0.1.0"
description = "Standalone NVIDIA sentiment node MVP"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.6,<3",
  "requests>=2.31,<3",
  "beautifulsoup4>=4.12,<5",
  "lxml>=5.1,<6",
  "pandas>=2.2,<3",
  "numpy>=1.26,<3",
  "python-dateutil>=2.9,<3",
  "typer>=0.12,<1",
  "pytest>=8.0,<9"
]

[project.optional-dependencies]
finbert = [
  "transformers>=4.40,<5",
  "torch>=2.2,<3"
]

[project.scripts]
nvda-sentiment = "nvda_sentiment.node:cli"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

## 9.2 Rationale for optional FinBERT

Torch + transformers + FinBERT weights add roughly 1 GB to the install footprint. The node must run cleanly with **lexicon-only scoring** (see §36.3), so these dependencies are opt-in.

- Default install (`pip install nvda-sentiment-node`) — lexicon-only, installs in seconds.
- FinBERT install (`pip install nvda-sentiment-node[finbert]`) — full model-backed scoring.

The node detects at import time whether `transformers` and `torch` are available and routes through the appropriate scorer. See §19.8 and §36.3.

---

# 10. Configuration Rules

## 10.1 `config.py`

Create a config object with:
- user agent string for SEC requests
- cache directory
- default lookback quarters
- FinBERT model name
- score weights
- timeout settings

## 10.2 Example config object

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # MUST follow SEC-required format: "Name ContactEmail"
    # Example: "NVDA Sentiment Node contact@example.com"
    # A real reachable email is required by SEC policy.
    sec_user_agent: str = "NVDA Sentiment Node contact@example.com"
    sec_max_requests_per_second: float = 8.0  # SEC limit is 10/sec; stay below
    cache_dir: Path = Path(".cache")
    request_timeout_seconds: int = 30
    default_lookback_quarters: int = 4
    finbert_model_name: str = "ProsusAI/finbert"
    finbert_model_revision: str = "main"  # Pin a commit SHA for true determinism

    filing_tone_weight: float = 0.50
    filing_delta_weight: float = 0.25
    guidance_tone_weight: float = 0.15
    investor_context_weight: float = 0.10
```

## 10.3 Important rules for SEC requests

1. The `User-Agent` header MUST contain a real contact email. The SEC rejects requests without one and may block the IP on repeated violations.
2. Requests to `data.sec.gov` and `www.sec.gov` must stay under **10 requests per second** across the whole client. The node enforces `sec_max_requests_per_second` as a simple token-bucket or sleep-based limiter in the SEC adapter.
3. All SEC responses should be cached (see §30).

---

# 11. Schemas

Implement these schemas in `schemas.py`.

## 11.1 Public schemas

### Request schema

```python
from pydantic import BaseModel, Field
from datetime import date

class SentimentRequest(BaseModel):
    ticker: str = Field(default="NVDA")
    as_of_date: date
    lookback_quarters: int = Field(default=4, ge=1, le=8)
    include_market_context: bool = True
    use_cache: bool = True
```

### Response schema

```python
from typing import Dict, List

class SentimentResponse(BaseModel):
    ticker: str
    as_of_date: date
    market_sentiment_score: float
    label: str
    confidence: float
    components: Dict[str, float]
    signals: List[str]
    source_coverage: Dict[str, int]
    metadata: Dict[str, object]
```

## 11.2 Internal document schemas

```python
from typing import Literal, Optional
from pydantic import BaseModel

class SourceDocument(BaseModel):
    source_type: Literal["10-K", "10-Q", "8-K", "press_release", "transcript", "cfo_commentary"]
    title: str
    url: str
    filed_at: str
    fiscal_period: Optional[str] = None
    raw_html: Optional[str] = None
    clean_text: Optional[str] = None

class SectionScore(BaseModel):
    section_name: str
    finbert_score: float
    lexicon_score: float
    uncertainty_penalty: float
    final_score: float
    sentence_count: int

class DocumentScore(BaseModel):
    source_type: str
    filed_at: str
    title: str
    section_scores: list[SectionScore]
    final_score: float
```

---

# 12. Source Fetching — Detailed Instructions

This section tells the implementer exactly how to fetch the required data.

---

# 13. SEC Adapter Specification

Create `adapters/sec_api.py`.

## 13.1 Responsibilities

This module must:
1. map `NVDA` to the correct SEC company identifier
2. fetch recent submissions
3. identify the latest relevant 10-K, 10-Q, and 8-K filings
4. construct URLs for the filing index page and main filing document
5. return normalized `SourceDocument` objects

## 13.2 Simplifying assumption

Hardcode NVIDIA metadata in the MVP instead of building a general ticker-to-CIK service.

Use:
- ticker: `NVDA`
- company name: `NVIDIA CORP`
- CIK: `1045810`

Hardcoding this is acceptable because this is a single-company node.

## 13.3 SEC endpoints to use

Use these endpoints:

### Company submissions JSON
```text
https://data.sec.gov/submissions/CIK0001045810.json
```

This is the primary source for recent filing metadata.

### Filing index page format
```text
https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_no_dashes}/{index}.html
```

### Filing main document URL pattern
Use the metadata in submissions JSON to locate:
- accession number
- primary document filename

Then construct:
```text
https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_no_dashes}/{primary_document}
```

## 13.4 What to pull

From the submissions JSON, use the recent filings arrays:
- accessionNumber
- filingDate
- form
- primaryDocument
- primaryDocDescription

Normalize them into records.

## 13.5 Forms to fetch

Required:
- latest 1 x 10-K
- latest N x 10-Q, where N = min(lookback_quarters, available quarters)
- latest earnings-related 8-Ks within the same lookback window

## 13.6 8-K filtering rule

Do not ingest every 8-K.

Only include 8-Ks that are likely earnings-related.

Use this filter:
- include if `primaryDocDescription` or linked text contains one of:
  - `results of operations`
  - `earnings`
  - `financial results`
  - `quarterly results`
  - `exhibit 99.1`
- otherwise skip

## 13.7 Required SEC adapter methods

Implement these methods:

```python
class SECAdapter:
    def get_recent_filings(self) -> list[dict]:
        ...

    def get_relevant_filings(self, lookback_quarters: int) -> list[SourceDocument]:
        ...

    def fetch_filing_html(self, doc: SourceDocument) -> str:
        ...
```

## 13.8 Submissions JSON pagination — important

The base submissions JSON at `https://data.sec.gov/submissions/CIK0001045810.json` only contains the **most recent ~1000 filings**. Older filings live in additional files referenced under the `filings.files` array.

**For the MVP, this is acceptable:**
- `lookback_quarters` is capped at 8 (§11.1).
- The base file contains far more than 8 quarters of NVIDIA filings.
- The MVP **must not** attempt to follow the `filings.files` pagination.

If a future use case requires deeper lookback, this is where the adapter will need to grow. Document this limitation as a code comment in `sec_api.py`.

## 13.9 Rate limiting

SEC endpoints enforce a 10 requests/second limit. The adapter must throttle to `settings.sec_max_requests_per_second` (default 8) across all SEC calls within a single node run. A simple sleep-based limiter is sufficient:

```python
import time

class RateLimiter:
    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()
```

## 13.10 Error handling

If a filing fetch fails:
- log a warning
- skip the document
- continue processing

The node must degrade gracefully.

---

# 14. NVIDIA IR Adapter Specification

Create `adapters/nvidia_ir.py`.

## 14.1 Responsibilities

This module must:
1. fetch the quarterly results page
2. find links for:
   - press release
   - webcast transcript
   - CFO commentary
3. retrieve the relevant documents for the requested lookback period
4. normalize them into `SourceDocument`

## 14.2 Starting page

Use the NVIDIA quarterly results page.

## 14.3 What to ingest

For each quarter in scope, ingest if available:
- press release
- webcast transcript
- CFO commentary

These are high-value sources and are usually cleaner than raw filings.

## 14.4 Required adapter methods

```python
class NvidiaIRAdapter:
    def get_quarterly_results_documents(self, lookback_quarters: int) -> list[SourceDocument]:
        ...

    def fetch_document_html(self, doc: SourceDocument) -> str:
        ...
```

## 14.5 Parsing rules

When parsing the quarterly results page:
- extract all link text + href pairs
- group links by quarter
- identify transcript links by text containing `transcript`
- identify CFO commentary links by text containing `cfo commentary`
- identify press release links by text containing `press release`

## 14.6 Fallback behavior

If transcript or CFO commentary is missing for a quarter:
- keep going
- do not fail the run
- confidence should drop slightly

---

# 15. Market Context Adapter Specification

Create `adapters/market_context.py`.

## 15.1 Responsibilities

This module must fetch two small context signals:
1. AAII sentiment
2. VIX

## 15.2 AAII logic

The AAII sentiment survey page does not expose a clean API. The MVP must:

1. Fetch `https://www.aaii.com/sentimentsurvey` (or the current public URL).
2. Parse three percentages: bullish, neutral, bearish.
3. If parsing fails (site redesign, rate limit, etc.), **gracefully degrade**:
   - return `aaii_score = 0.0`
   - add warning: `"AAII sentiment unavailable; contribution set to 0"`
   - the node must NOT crash

Extract latest values for:
- bullish
- neutral
- bearish

Convert to a single score:

```text
aaii_raw = bullish_pct - bearish_pct
aaii_score = clip(aaii_raw / 100, -1.0, 1.0)
```

Examples:
- bullish 40, bearish 20 => +0.20
- bullish 20, bearish 45 => -0.25

Because AAII parsing is inherently brittle, the contribution weight is small (see §15.4) and the node is designed to tolerate its absence cleanly.

## 15.3 VIX logic

Fetch the latest VIX close via FRED's CSV endpoint. This is stable and does not require HTML scraping.

### Endpoint
```text
https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
```

### Parsing rules
- CSV with columns `DATE,VIXCLS` (or similar — handle either).
- Read the last row with a non-empty, numeric VIXCLS value (FRED sometimes includes trailing rows with `.` placeholders for non-trading days).
- Use that value as the latest VIX close.

Convert to a normalized sentiment contribution using a simple piecewise rule:

```text
if VIX <= 15: vix_score = +0.50
elif VIX <= 20: vix_score = +0.25
elif VIX <= 25: vix_score = 0.00
elif VIX <= 30: vix_score = -0.25
else: vix_score = -0.50
```

This is intentionally simple. The expanded range (±0.5 instead of ±0.2) keeps VIX on a comparable scale to AAII in the combined score.

## 15.4 Combined investor context score

```text
investor_context_score = 0.6 * aaii_score + 0.4 * vix_score
clip to [-1.0, 1.0]
```

Both `aaii_score` and `vix_score` are now on roughly matching scales (`[-1, +1]` bounded but typically `±0.5`), so neither structurally dominates.

## 15.5 Required methods

```python
class MarketContextAdapter:
    def get_aaii_score(self) -> float:
        ...

    def get_vix_score(self) -> float:
        ...

    def get_combined_score(self) -> float:
        ...
```

## 15.6 Important rule

This layer must remain small.  
It is a context overlay, not the main event.

---

# 16. HTML/Text Parsing

Create:
- `parsers/html_to_text.py`
- `parsers/filing_parser.py`
- `parsers/ir_parser.py`

## 16.1 Goal

Convert raw HTML into clean, predictable text suitable for section extraction and sentence scoring.

## 16.2 Basic HTML cleanup process

For every document:
1. parse HTML with BeautifulSoup
2. remove `script`, `style`, `noscript`
3. extract visible text
4. collapse repeated whitespace
5. normalize smart quotes and dashes
6. preserve headings if possible
7. return a plain text string

## 16.3 Required helper function

```python
def html_to_clean_text(html: str) -> str:
    ...
```

## 16.4 Text cleanup rules

Implement these exact cleanup transformations:
- convert `\r\n` to `\n`
- replace tabs with spaces
- collapse 3+ newlines into 2 newlines
- collapse repeated spaces
- strip non-printable characters except newline
- trim leading/trailing whitespace

## 16.5 Filing-specific cleanup

SEC filings can contain:
- HTML tables
- duplicated headers
- navigation text
- XBRL residue

That is acceptable.  
You do not need perfect cleanup.  
You only need the text clean enough for section extraction.

---

# 17. Section Extraction Strategy

Create `parsers/section_extractor.py`.

This is one of the most important modules.

## 17.1 Core idea

Do not score entire documents blindly.  
Extract the parts that matter.

## 17.2 Required target sections

Implement extraction for the following section types:

### For 10-K and 10-Q
- `mda` — management discussion and analysis
- `risk_factors`
- `outlook_guidance` — only if identifiable
- `business_highlights` — optional small snippets from summary areas

### For 8-K
- `earnings_summary`
- `outlook_guidance`

### For press release
- `headline_and_summary`
- `financial_highlights`
- `outlook_guidance`

### For transcript
- `prepared_remarks`
- `qa`

### For CFO commentary
- `cfo_commentary_body`

## 17.3 Use simple heading-based extraction

Use regex or string matching on normalized text.

Do not over-engineer this.

## 17.4 Filing heading patterns

Use heading patterns like these:

### MD&A
Match nearest text after headings containing:
- `management's discussion and analysis`
- `management discussion and analysis`
- `md&a`

Stop at the next major heading likely matching:
- `quantitative and qualitative disclosures`
- `controls and procedures`
- `risk factors`
- `financial statements`
- `legal proceedings`

### Risk factors
Match text after:
- `risk factors`

Stop at next likely major heading:
- `unresolved staff comments`
- `properties`
- `legal proceedings`
- `mine safety disclosures`
- `management's discussion and analysis`

### Outlook/guidance
Search for paragraphs containing:
- `outlook`
- `guidance`
- `we expect`
- `we believe`
- `looking ahead`
- `for the next quarter`
- `for fiscal`
- `our outlook`

Return the concatenation of matching paragraphs.

## 17.5 Transcript extraction rules

Use the transcript text and split into:
- prepared remarks: everything before analyst Q&A starts
- Q&A: everything from first clear Q&A marker onward

Match markers like:
- `question-and-answer`
- `q&a`
- `operator`
- `analyst`

## 17.6 Press release extraction rules

Look for sections/paragraphs containing:
- opening performance summary
- guidance paragraphs
- phrases like `outlook`, `expected`, `revenue outlook`

## 17.7 Required extractor interface

```python
class SectionExtractor:
    def extract_sections(self, doc: SourceDocument) -> dict[str, str]:
        ...
```

Return only non-empty sections.

## 17.8 Quality rule

A crude but reliable section extraction is preferable to a clever fragile one.

---

# 18. Sentence Segmentation

Keep sentence splitting simple.

## 18.1 Rule

Create a helper function that splits text on:
- `.`
- `?`
- `!`
- newline boundaries when paragraphs are long

Do not introduce a heavy NLP dependency just for sentence splitting.

## 18.2 Minimum quality rule

Ignore sentences:
- shorter than 20 characters
- with fewer than 4 words

This reduces noise.

---

# 19. FinBERT Scorer

Create `scorers/finbert_scorer.py`.

## 19.1 Model choice

Use:
- `ProsusAI/finbert`

Pin a specific model revision in config (`settings.finbert_model_revision`). Using `"main"` is acceptable for development but for a deterministic production run, pin a commit SHA.

## 19.2 Why

This model is strong enough for the MVP and designed for finance text.

## 19.3 Optional import — CRITICAL

FinBERT (transformers + torch) is an **optional** dependency (§9.1). The scorer MUST handle missing imports:

```python
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

class FinBERTScorer:
    def __init__(self, settings):
        self.available = FINBERT_AVAILABLE
        if not self.available:
            return
        # ... load model and tokenizer
```

If `self.available` is False, `score_text` returns `(0.0, 0)` and the node proceeds lexicon-only. See §36.3.

## 19.4 Determinism

For deterministic FinBERT output across runs:
- pin model revision (commit SHA, not `"main"`)
- call `torch.use_deterministic_algorithms(True)` at init
- set `model.eval()` and wrap inference in `with torch.no_grad():`
- disable tokenizer parallelism: `os.environ["TOKENIZERS_PARALLELISM"] = "false"`

## 19.5 Input strategy

Do not feed entire documents to the model. FinBERT is BERT-base with a 512-token limit, and MD&A sections routinely exceed this.

Instead:
1. split section text into sentences
2. score each sentence (truncating to 512 tokens as a safety net)
3. average sentence-level results into a section score

## 19.6 Sentence score formula

Assume FinBERT returns probabilities for:
- positive
- negative
- neutral

Convert to a scalar score:

```text
sentence_finbert_score = positive_probability - negative_probability
```

This naturally falls roughly in `[-1, 1]`.

## 19.7 Section-level FinBERT score

```text
section_finbert_score = mean(sentence_finbert_score over all valid sentences)
```

## 19.8 Known limitation — Risk Factors sections

FinBERT will score 10-K and 10-Q **risk factor** sections as strongly negative regardless of company health. Risk factors are structurally a list of things that could go wrong — that is their regulatory purpose. Absolute FinBERT scores on risk factors are therefore uninformative.

The system handles this in two places:
1. Risk factors are NOT used in the absolute `filing_tone` calculation — see §22.1 (revised weights).
2. Risk factors ARE used in `filing_delta` via a dedicated risk-delta signal — see §24.

Do not remove this design. Risk factor tone levels are misleading; risk factor tone *deltas* are valuable.

## 19.9 Required class

```python
class FinBERTScorer:
    def score_text(self, text: str) -> tuple[float, int]:
        # Returns (mean_score, sentence_count)
        # Returns (0.0, 0) if FinBERT unavailable or text has no valid sentences
        ...
```

## 19.10 Batch inference rule

Score sentences in batches for efficiency.
Batch size 8 or 16 is enough.

## 19.11 Fallback rule

If model load or inference fails:
- log warning
- return `(0.0, 0)`
- set the warning flag for metadata
- the node continues with lexicon-only scoring

The node must still run.

---

# 20. Lexicon Scorer

Create `scorers/lexicon.py`.

## 20.1 Purpose

Add deterministic, interpretable finance-aware language features.

## 20.2 MVP implementation choice

Implement a **small local lexicon**, not the full research package.

Use a compact finance lexicon with these categories:
- positive
- negative
- uncertainty
- litigious

## 20.3 Lexicon file approach

Hardcode small word sets directly in Python.

### Positive terms
```python
POSITIVE_TERMS = {
  "growth", "strong", "record", "improved", "increase", "increased",
  "accelerate", "accelerated", "opportunity", "leadership", "momentum",
  "healthy", "robust", "benefit", "benefits", "confidence", "efficient",
  "innovation", "successful", "success", "expansion", "expanded"
}
```

### Negative terms
```python
NEGATIVE_TERMS = {
  "decline", "declined", "decrease", "decreased", "weakness", "softness",
  "risk", "adverse", "pressure", "headwind", "uncertain", "uncertainty",
  "challenge", "challenging", "constraint", "constraints", "litigation",
  "disruption", "loss", "losses", "slowdown", "volatile"
}
```

### Uncertainty terms
```python
UNCERTAINTY_TERMS = {
  "may", "might", "could", "uncertain", "uncertainty", "depends",
  "subject to", "potential", "possibly", "assume", "assumption"
}
```

### Litigious terms
```python
LITIGIOUS_TERMS = {
  "litigation", "claim", "claims", "proceeding", "proceedings",
  "investigation", "investigations", "regulatory", "compliance"
}
```

These lists are intentionally small.  
The purpose is stability and interpretability, not lexical perfection.

## 20.4 Lexicon scoring formula

Tokenize text to lowercase words.

Compute:
```text
positive_rate = positive_count / token_count
negative_rate = negative_count / token_count
uncertainty_rate = uncertainty_count / token_count
litigious_rate = litigious_count / token_count
```

Then:

```text
lexicon_score =
    6.0 * (positive_rate - negative_rate)
  - 3.0 * uncertainty_rate
  - 2.0 * litigious_rate
```

Clip to `[-1.0, 1.0]`.

**Scale note:** In typical 10-K / 10-Q prose, positive and negative word rates run 0.5–2%. The resulting `lexicon_score` usually falls in `[-0.1, +0.1]` — much smaller than FinBERT's typical `[-0.5, +0.5]` range. This is intentional. The lexicon's purpose in the MVP is:
1. **Interpretability** — token-level contributions are auditable.
2. **Fallback signal** — if FinBERT is unavailable, the lexicon score drives scoring alone (§36.3) and its smaller magnitude gets promoted to full weight.
3. **Uncertainty penalty surface** — the uncertainty and litigious rates feed the section scorer directly (§21).

Do not inflate these multipliers to match FinBERT magnitudes — the small scale is a feature, not a bug.

## 20.5 Required class

```python
class LexiconScorer:
    def score_text(self, text: str) -> dict[str, float]:
        # Returns a dict with lexicon_score and component rates
        ...
```

---

# 21. Section Scoring

Create `scorers/section_scorer.py`.

## 21.1 Purpose

Combine model-based and deterministic sentiment into one section score.

## 21.2 Formula

For each extracted section:

```text
section_score =
    0.75 * finbert_score
  + 0.25 * lexicon_score
```

Then apply a small uncertainty penalty:

```text
section_score_adjusted = section_score - 0.10 * uncertainty_rate
```

Clip to `[-1.0, 1.0]`.

## 21.3 Why this formula

- FinBERT carries most of the semantic load
- lexicon adds interpretability
- uncertainty slightly dampens over-optimistic language

## 21.4 Required class

```python
class SectionScorer:
    def score_section(self, section_name: str, text: str) -> SectionScore:
        ...
```

---

# 22. Section Weights by Document Type

These weights are critical and should be implemented exactly.

## 22.1 10-K / 10-Q section weights

Risk factors sections are structurally negative by regulatory design (§19.8). Their absolute tone is uninformative. They are therefore **excluded from the absolute filing tone** and fed separately into `filing_delta` (§24).

```python
FILING_SECTION_WEIGHTS = {
    "mda": 0.70,
    "outlook_guidance": 0.30
    # risk_factors intentionally excluded here
    # See §24 for risk_factors delta handling
}
```

If `outlook_guidance` is absent in a given 10-Q, MD&A picks up the full 1.0 via renormalization (§22.6).

## 22.2 8-K section weights

```python
EIGHT_K_SECTION_WEIGHTS = {
    "earnings_summary": 0.70,
    "outlook_guidance": 0.30
}
```

## 22.3 Press release section weights

```python
PRESS_RELEASE_SECTION_WEIGHTS = {
    "headline_and_summary": 0.40,
    "financial_highlights": 0.35,
    "outlook_guidance": 0.25
}
```

## 22.4 Transcript section weights

```python
TRANSCRIPT_SECTION_WEIGHTS = {
    "prepared_remarks": 0.70,
    "qa": 0.30
}
```

## 22.5 CFO commentary section weights

```python
CFO_COMMENTARY_SECTION_WEIGHTS = {
    "cfo_commentary_body": 1.00
}
```

## 22.6 Important rule

If some sections are missing:
- renormalize the remaining weights
- do not force missing sections to zero

---

# 23. Document Scoring

Create logic that maps multiple section scores to one document score.

## 23.1 Formula

For each document:
1. extract sections
2. score each section
3. take a weighted mean using the document-type section weights
4. clip final document score to `[-1, 1]`

## 23.2 Required function

```python
def score_document(doc: SourceDocument) -> DocumentScore:
    ...
```

---

# 24. Filing Delta Logic

Create `features/filing_delta.py`.

This is one of the highest value features.

## 24.1 Purpose

Raw tone alone is not enough.
We want to know whether tone is **improving or deteriorating**, and whether **risk language has shifted**.

Filing delta has two sub-components:
- **Tone delta** — change in overall tone across comparable materials
- **Risk delta** — change in risk factor tone specifically (see §19.8)

## 24.2 NVIDIA fiscal calendar — hardcode it

NVIDIA's fiscal year ends in late January. NVIDIA-specific fiscal quarters are:

```python
# NVIDIA fiscal quarter boundaries (approximate, aligned to their reporting cadence).
# NVIDIA FY ends on the last Sunday of January.
NVIDIA_FISCAL_QUARTER_ANCHORS = {
    # Q1 covers roughly Feb - Apr
    # Q2 covers roughly May - Jul
    # Q3 covers roughly Aug - Oct
    # Q4 covers roughly Nov - Jan
}

def nvda_fiscal_bucket(filing_date: date) -> str:
    """Return 'FY{YY}Q{N}' for a NVIDIA filing date."""
    month = filing_date.month
    year = filing_date.year
    if month in (2, 3, 4):
        return f"FY{year % 100 + 1:02d}Q1"  # e.g. Feb 2026 -> FY27Q1
    if month in (5, 6, 7):
        return f"FY{year % 100 + 1:02d}Q2"
    if month in (8, 9, 10):
        return f"FY{year % 100 + 1:02d}Q3"
    if month in (11, 12):
        return f"FY{(year + 1) % 100 + 1:02d}Q4"
    # January is NVIDIA's fiscal Q4 of the prior FY label
    return f"FY{year % 100 + 1:02d}Q4"
```

The key invariant: a 10-K filed in late February/March aligns with NVIDIA's FY-end reporting (their fiscal Q4 / full year). A 10-Q filed in late May is their fiscal Q1. Using this bucketing prevents cross-fiscal-year noise in the delta.

**Do not fall back to calendar quarters.** This node is NVIDIA-specific (§47.3); hardcoding the fiscal calendar is appropriate.

## 24.3 Tone delta

Compare:
- latest 10-Q with previous 10-Q (same fiscal quarter position, prior FY, OR immediately prior fiscal quarter — see §24.5)
- latest 10-K with prior 10-K if available, otherwise ignore
- latest earnings materials with previous quarter earnings materials

Compute:

```text
tone_delta = current_period_score - prior_period_score
```

Where `current_period_score` and `prior_period_score` are weighted means of document scores within each fiscal-quarter bucket, using DOCUMENT_TYPE_WEIGHTS (§26).

## 24.4 Risk-factor delta — important

Score 10-K and 10-Q risk factor sections **separately** and compute:

```text
risk_delta = current_risk_score - prior_risk_score
```

Interpret carefully: a **more negative** risk_delta (risk language got worse) is a **negative sentiment signal**; a **more positive** risk_delta (risk language softened, or fewer new risks added) is a **positive sentiment signal**.

Use `risk_delta` directly — its sign already aligns with the sentiment direction.

## 24.5 Combined filing_delta

```text
filing_delta = 0.70 * tone_delta + 0.30 * risk_delta
```

Clip `filing_delta` to `[-1.0, 1.0]`.

## 24.6 Comparison strategy

Prefer year-over-year comparison (same fiscal quarter, prior FY) when available — this controls for NVIDIA's seasonal cadence. Fall back to sequential-quarter comparison when YoY data is not in the lookback window.

## 24.7 If insufficient prior data

If no prior comparable period exists:
- set `tone_delta = 0.0`
- set `risk_delta = 0.0`
- set `filing_delta = 0.0`
- add warning: `"No prior comparable period found for filing delta"`

---

# 25. Guidance Tone Feature

Guidance deserves its own component because it often matters more than general tone.

## 25.1 Rule

Collect all `outlook_guidance` sections across:
- latest 10-Q or 10-K
- latest earnings press release
- latest transcript
- latest CFO commentary

Score them and average them:

```text
guidance_tone = mean(all guidance-related section scores)
```

If no guidance sections exist:
- set to 0.0
- add warning

---

# 26. Filing Tone Component

This is the broad issuer tone score.

## 26.1 Rule

Compute the weighted average of all in-scope official document scores from the current lookback period.

Recommended document-type weights:

```python
DOCUMENT_TYPE_WEIGHTS = {
    "10-K": 0.30,
    "10-Q": 0.30,
    "8-K": 0.10,
    "press_release": 0.10,
    "transcript": 0.10,
    "cfo_commentary": 0.10
}
```

## 26.2 Important note

These weights are for the **current-period tone component**, not for per-document section scoring.

## 26.3 Renormalization

If some document types are absent, renormalize the remaining weights.

---

# 27. Composite Final Score

Create `scorers/composite.py`.

## 27.1 Component weights

Use these exact weights in the MVP:

```python
FINAL_COMPONENT_WEIGHTS = {
    "filing_tone": 0.50,
    "filing_delta": 0.25,
    "guidance_tone": 0.15,
    "investor_context": 0.10
}
```

## 27.2 Final raw score

All components must be in `[-1, 1]`.

```text
final_raw =
    0.50 * filing_tone
  + 0.25 * filing_delta
  + 0.15 * guidance_tone
  + 0.10 * investor_context
```

Clip to `[-1, 1]`.

## 27.3 Map to 0–100

Use this simple linear transform:

```text
final_score_0_100 = 50 + 50 * final_raw
```

Then clip to `[0, 100]`.

Examples:
- `-1.0 => 0`
- `0.0 => 50`
- `+1.0 => 100`

## 27.4 Label mapping

Use this exact mapping:

```text
0   to <35  => bearish
35  to <45  => mildly bearish
45  to <55  => neutral
55  to <65  => mildly bullish
65  to 100  => bullish
```

---

# 28. Confidence Score

Create `features/confidence.py`.

Confidence must be simple and interpretable.

## 28.1 Base confidence

Start with:
```text
confidence = 0.55
```

## 28.2 Additions

Add each if the condition is met (max total additions = 0.40, bringing max pre-penalty to 0.95):
- `+0.05` if latest 10-K available
- `+0.05` if at least 2 recent 10-Qs available
- `+0.05` if latest transcript available
- `+0.05` if latest CFO commentary available
- `+0.05` if guidance_tone was computed from at least 2 sources
- `+0.05` if FinBERT ran successfully
- `+0.05` if filing_delta was computed against YoY-matched comparison (§24.6)
- `+0.05` if investor context was retrieved successfully

## 28.3 Penalties

Subtract:
- `-0.05` if filing delta unavailable
- `-0.05` if investor context unavailable but requested
- `-0.05` if more than 25% of document fetches failed
- `-0.05` if section extraction success rate is below 60%

## 28.4 Clip

Clip confidence to:
```text
[0.10, 0.95]
```

- Never emit 1.0 in the MVP (calibration is not validated).
- The floor `0.10` matches the "no scorable documents" fallback (§36.5) so the confidence surface is consistent.

## 28.5 Empty-data short circuit

If `len(document_scores) == 0`, skip the arithmetic entirely and return `0.10`. This keeps §36.5 self-consistent.

---

# 29. Human-Readable Signals

Create `features/signal_builder.py`.

Generate 4–6 simple English bullet signals from the results. Signals are the most user-visible part of the output — they must be concrete and evidence-backed, not generic threshold announcements.

## 29.1 Signal inputs

The signal builder takes the full scoring state, not just the four final components:

```python
def build_signals(
    filing_tone: float,
    filing_delta: float,
    tone_delta: float,          # sub-component of filing_delta, §24.3
    risk_delta: float,          # sub-component of filing_delta, §24.4
    guidance_tone: float,
    investor_context: float,
    document_scores: list[DocumentScore],
    section_scores_by_type: dict[str, list[SectionScore]],
) -> list[str]:
    ...
```

## 29.2 Required signals

Emit these, in roughly this order:

### Tier 1 — Overall tone (always emit one)
- if `filing_tone >= 0.20`: `"Management tone is positive across recent official materials"`
- if `filing_tone <= -0.20`: `"Management tone is negative across recent official materials"`
- else: `"Management tone is broadly balanced across recent official materials"`

### Tier 2 — Tone trajectory (always emit one)
- if `tone_delta >= 0.10`: `"Tone improved versus the prior comparable period"`
- if `tone_delta <= -0.10`: `"Tone weakened versus the prior comparable period"`
- else: `"Tone changed only modestly versus the prior comparable period"`

### Tier 3 — Risk language (always emit one)
- if `risk_delta >= 0.05`: `"Risk-factor language softened versus prior filing"`
- if `risk_delta <= -0.05`: `"Risk-factor language intensified versus prior filing"`
- else: `"Risk-factor language was substantively stable versus prior filing"`

### Tier 4 — Guidance (always emit one)
- if `guidance_tone >= 0.15`: `"Forward-looking guidance language is constructive"`
- if `guidance_tone <= -0.15`: `"Forward-looking guidance language is cautious"`
- else: `"Forward-looking guidance language is mixed"`

### Tier 5 — Q&A vs prepared-remarks divergence (emit if transcripts available)
Compute `qa_gap = mean(prepared_remarks_scores) - mean(qa_scores)` across in-window transcripts.
- if `qa_gap >= 0.15`: `"Analyst Q&A tone ran meaningfully below prepared remarks — a traditional caution signal"`
- if `qa_gap <= -0.15`: `"Analyst Q&A tone ran above prepared remarks — management handled pushback well"`
- else: skip (do not emit a null signal)

### Tier 6 — Investor context (emit if included)
- if `investor_context >= 0.10`: `"Broader investor risk appetite is supportive"`
- if `investor_context <= -0.10`: `"Broader investor risk appetite is cautious"`
- else: `"Broader investor risk appetite is neutral"`

## 29.3 Signal cap

Emit at most 6 signals. If more tiers qualify, prioritize tiers 1–4 first, then 5, then 6.

## 29.4 Signal writing rules

- Keep each signal under 18 words.
- Do not include numeric scores in signal text — signals are the qualitative layer; numerics live in `components`.
- Always use the same phrasing for the same condition across runs (deterministic).

---

# 30. Caching

Create `utils/cache.py`.

## 30.1 Why

External sources are stable enough to cache.  
Caching reduces latency and rate-limit issues.

## 30.2 Approach

Use a filesystem cache:
- one file per URL
- hashed filename
- JSON or raw HTML content

## 30.3 TTLs

Use these TTLs:
- SEC submissions JSON: 24 hours (new filings appear over time)
- **SEC filing HTML: permanent** (once filed under an accession number, SEC filings are immutable — amendments use new accession numbers)
- NVIDIA IR HTML pages: 24 hours
- AAII page: 24 hours
- VIX data (FRED CSV): 24 hours

## 30.3.1 Permanent cache implementation

For the "permanent" tier, the cache check is simply file existence — no staleness check is needed. This also naturally rate-limits SEC requests: a second run against the same filing set makes zero SEC calls.

```python
def get_permanent(self, key: str) -> str | None:
    path = self._path_for(key)
    return path.read_text() if path.exists() else None
```

## 30.4 Required methods

```python
class SimpleCache:
    def get(self, key: str):
        ...
    def set(self, key: str, value: str):
        ...
    def is_fresh(self, key: str, ttl_seconds: int) -> bool:
        ...
```

---

# 31. Logging

Create `utils/logging.py`.

## 31.1 Rule

Use standard Python logging only.

## 31.2 Log levels
- INFO for high-level pipeline stages
- WARNING for fetch/parse/model failures
- DEBUG optional for local development

Do not over-log.

---

# 32. Utility Functions

Create:
- `utils/text.py`
- `utils/dates.py`

## 32.1 `utils/text.py`
Implement:
- `normalize_whitespace(text)`
- `normalize_unicode_punctuation(text)`
- `simple_sentence_split(text)`
- `tokenize_words(text)`

## 32.2 `utils/dates.py`
Implement:
- `parse_date(value)`
- `calendar_quarter_bucket(date_obj)` -> `"YYYY-QN"`
- `sort_docs_by_date(docs)`

---

# 33. Exact Orchestration Logic

Create `node.py`.

## 33.1 Public class

```python
class NVDASentimentNode:
    def __init__(self, settings: Settings | None = None):
        ...
    def run(self, request: SentimentRequest) -> SentimentResponse:
        ...
```

## 33.2 Exact pipeline sequence

Implement the `run()` method with this order:

1. validate request
2. fetch SEC filings
3. fetch NVIDIA IR documents
4. merge and deduplicate documents by URL
5. fetch raw HTML for each document
6. convert HTML to clean text
7. extract sections
8. score each section
9. score each document
10. compute filing tone
11. compute filing delta
12. compute guidance tone
13. optionally fetch investor context
14. compute final composite score
15. compute confidence
16. build human-readable signals
17. build source coverage
18. return response

---

# 34. Deduplication Rules

Documents may overlap between SEC and NVIDIA IR references.

## 34.1 Deduplicate by URL first

If URLs match, keep one.

## 34.2 Deduplicate by title + filed date second

If title and filed date are highly similar, keep one.

## 34.3 Priority order

If duplicate candidates exist, prefer:
1. NVIDIA IR transcript / commentary pages for transcript-like materials
2. SEC pages for official filings

---

# 35. Document Type Inference Rules

Implement a simple classifier based on source metadata or title text.

## 35.1 Rules

If title or source says:
- contains `10-k` => `10-K`
- contains `10-q` => `10-Q`
- contains `8-k` => `8-K`
- contains `transcript` => `transcript`
- contains `cfo commentary` => `cfo_commentary`
- contains `press release` or `financial results` => `press_release`

---

# 36. Missing Data Strategy

The node must never crash just because some sources are missing.

## 36.1 If SEC data unavailable
Continue with NVIDIA IR materials only.

## 36.2 If NVIDIA IR data unavailable
Continue with SEC materials only.

## 36.3 If FinBERT unavailable

"Unavailable" covers two cases:
1. **Not installed** — `transformers` / `torch` not importable (the default install path, §9.2)
2. **Runtime failure** — model download, tokenization, or inference raised an exception

In both cases:
- the section scorer runs with `finbert_score = 0.0`
- the lexicon score drives the section score alone (`section_score = lexicon_score`, clipped)
- add warning: `"FinBERT unavailable; lexicon-only scoring"`
- `confidence` does not receive the FinBERT-availability bonus (§28.2)

The node must still produce a complete, schema-valid response.

## 36.4 If market context unavailable
Set investor context to 0 and add warning.

## 36.5 If no scorable documents exist
Return:
- score = 50
- label = neutral
- confidence = 0.10
- warning: `"No scorable documents available"`

This is the confidence floor (§28.4). Do not go below 0.10 in any path.

---

# 37. End-to-End Scoring Example

This is an illustrative example only.

## 37.1 Example component values

```text
filing_tone = 0.32
filing_delta = 0.10
guidance_tone = 0.25
investor_context = 0.05
```

## 37.2 Final raw score

```text
final_raw =
  0.50*0.32 + 0.25*0.10 + 0.15*0.25 + 0.10*0.05
= 0.16 + 0.025 + 0.0375 + 0.005
= 0.2275
```

## 37.3 Final mapped score

```text
score = 50 + 50*0.2275 = 61.375
```

Rounded:
```text
61.4 => mildly bullish
```

---

# 38. Minimal Implementation Details Per File

This section exists so an LLM or engineer can implement file by file without ambiguity.

---

## 38.1 `nvda_sentiment/__init__.py`

Export:
```python
from .node import NVDASentimentNode
from .schemas import SentimentRequest, SentimentResponse
```

---

## 38.2 `nvda_sentiment/config.py`

Contents:
- `Settings` dataclass
- constants for weights and URLs
- helper to create cache dir on init

---

## 38.3 `nvda_sentiment/schemas.py`

Contents:
- `SentimentRequest`
- `SentimentResponse`
- `SourceDocument`
- `SectionScore`
- `DocumentScore`

No business logic.

---

## 38.4 `nvda_sentiment/adapters/sec_api.py`

Contents:
- `SECAdapter`
- methods for:
  - get submissions JSON
  - normalize filing rows
  - filter forms
  - build filing document URL
  - fetch filing HTML

Implementation notes:
- use `requests.get`
- set user-agent
- timeout from settings
- cache responses if enabled

---

## 38.5 `nvda_sentiment/adapters/nvidia_ir.py`

Contents:
- `NvidiaIRAdapter`
- methods for:
  - fetch quarterly results page
  - extract links
  - group by quarter
  - identify press release / transcript / commentary docs
  - fetch document HTML

Implementation notes:
- use BeautifulSoup
- preserve absolute URLs

---

## 38.6 `nvda_sentiment/adapters/market_context.py`

Contents:
- `MarketContextAdapter`
- methods for:
  - fetch AAII page and parse three percentages
  - fetch VIX latest value
  - compute normalized combined score

Implementation notes:
- keep parsers simple
- if parsing fails, return 0 with warning

---

## 38.7 `nvda_sentiment/parsers/html_to_text.py`

Contents:
- `html_to_clean_text(html: str) -> str`

Implementation notes:
- remove script/style/noscript
- get text with separator newline
- cleanup whitespace

---

## 38.8 `nvda_sentiment/parsers/filing_parser.py` and `ir_parser.py`

**Do NOT create these files.** Earlier drafts proposed thin wrappers around `html_to_clean_text`, but they add no logic and harm the readability criterion in §49.6. Call `html_to_clean_text` directly from `node.py`.

The `parsers/` directory contains only:
- `html_to_text.py`
- `section_extractor.py`

---

## 38.10 `nvda_sentiment/parsers/section_extractor.py`

Contents:
- `SectionExtractor`
- section heading regexes
- extraction helpers
- section dispatch by document type

This file is important and should be explicit.

---

## 38.11 `nvda_sentiment/scorers/finbert_scorer.py`

Contents:
- model loading
- tokenizer loading
- batch scoring
- sentence scoring

Implementation notes:
- load once in class init
- reuse model object

---

## 38.12 `nvda_sentiment/scorers/lexicon.py`

Contents:
- small lexicon sets
- tokenization
- rate calculations
- final lexicon score

---

## 38.13 `nvda_sentiment/scorers/section_scorer.py`

Contents:
- combine FinBERT and lexicon
- return `SectionScore`

---

## 38.14 `nvda_sentiment/scorers/composite.py`

Contents:
- document-type aggregation
- final component aggregation
- label mapping
- clip helper

---

## 38.15 `nvda_sentiment/features/filing_delta.py`

Contents:
- group docs by quarter bucket
- compute current vs prior score delta

---

## 38.16 `nvda_sentiment/features/confidence.py`

Contents:
- `compute_confidence(...)`

---

## 38.17 `nvda_sentiment/features/signal_builder.py`

Contents:
- threshold-based English signal builder

---

## 38.18 `nvda_sentiment/utils/cache.py`

Contents:
- lightweight hash-based filesystem cache

---

## 38.19 `nvda_sentiment/utils/text.py`

Contents:
- normalization helpers
- sentence splitter
- tokenizer

---

## 38.20 `nvda_sentiment/utils/dates.py`

Contents:
- date parser
- quarter bucket function

---

## 38.21 `nvda_sentiment/node.py`

Contents:
- orchestration class
- CLI entrypoint

---

# 39. Exact Pseudocode for `run()`

Use this pseudocode very closely.

```python
def run(self, request):
    warnings = []

    # 1. Fetch source metadata
    sec_docs = self.sec_adapter.get_relevant_filings(request.lookback_quarters)
    ir_docs = self.ir_adapter.get_quarterly_results_documents(request.lookback_quarters)

    # 2. Merge documents
    docs = deduplicate(sec_docs + ir_docs)

    # 3. Fetch raw content
    fetched_docs = []
    fetch_failures = 0
    for doc in docs:
        try:
            html = fetch_html_by_source(doc)
            doc.raw_html = html
            doc.clean_text = html_to_clean_text(html)
            fetched_docs.append(doc)
        except Exception:
            fetch_failures += 1
            warnings.append(f"Failed to fetch document: {doc.title}")

    # 4. Extract and score documents
    document_scores = []
    section_scores_by_type = defaultdict(list)  # keyed by section_name for signal builder
    extraction_successes = 0
    extraction_attempts = 0

    for doc in fetched_docs:
        extraction_attempts += 1
        sections = self.section_extractor.extract_sections(doc)
        if sections:
            extraction_successes += 1
        doc_score = self.document_scorer.score_document(doc, sections)
        if doc_score:
            document_scores.append(doc_score)
            for section_score in doc_score.section_scores:
                section_scores_by_type[section_score.section_name].append(section_score)

    # 5. Compute tone components
    filing_tone = compute_filing_tone(document_scores)

    # 5a. Compute delta sub-components (NVIDIA fiscal bucketing, §24)
    tone_delta = compute_tone_delta(document_scores, warnings)
    risk_delta = compute_risk_delta(document_scores, warnings)  # uses risk_factors sections
    filing_delta = clip(0.70 * tone_delta + 0.30 * risk_delta, -1.0, 1.0)

    guidance_tone = compute_guidance_tone(document_scores, warnings)

    investor_context = 0.0
    investor_context_available = False
    if request.include_market_context:
        try:
            investor_context = self.market_context.get_combined_score()
            investor_context_available = True
        except Exception:
            warnings.append("Investor context unavailable; contribution set to 0")

    # 6. Composite score
    final_raw, final_score, label = compute_final_score(
        filing_tone, filing_delta, guidance_tone, investor_context
    )

    # 7. Confidence
    confidence = compute_confidence(
        fetched_docs=fetched_docs,
        document_scores=document_scores,
        warnings=warnings,
        fetch_failures=fetch_failures,
        extraction_attempts=extraction_attempts,
        extraction_successes=extraction_successes,
        finbert_available=self.section_scorer.finbert_available,
        include_market_context=request.include_market_context,
        investor_context_available=investor_context_available,
        guidance_tone=guidance_tone,
        filing_delta=filing_delta,
        yoy_matched=delta_used_yoy_comparison,  # from compute_tone_delta
    )

    # 8. Signals and metadata
    signals = build_signals(
        filing_tone=filing_tone,
        filing_delta=filing_delta,
        tone_delta=tone_delta,
        risk_delta=risk_delta,
        guidance_tone=guidance_tone,
        investor_context=investor_context,
        document_scores=document_scores,
        section_scores_by_type=section_scores_by_type,
    )
    coverage = build_source_coverage(fetched_docs)

    return SentimentResponse(...)
```

---

# 40. Scoring Guardrails

These rules prevent weird behavior.

## 40.1 Clip everything
Every intermediate score should be clipped to `[-1, 1]`.

## 40.2 Neutral fallback
If a calculation cannot be performed reliably, use `0.0`, not a guess.

## 40.3 Don’t double-count
If the same content appears in multiple places, deduplicate if obvious.

## 40.4 Avoid overreacting to risk-factor sections
Risk factors should matter, but not dominate.

## 40.5 Missing data should reduce confidence, not explode the score

---

# 41. Testing Plan

Write tests as you implement.

## 41.1 `test_schemas.py`
Test:
- request validation
- default values
- invalid lookback ranges

## 41.2 `test_text_utils.py`
Test:
- whitespace normalization
- sentence splitting
- tokenization

## 41.3 `test_section_extractor.py`
Test with small synthetic text blocks:
- MD&A extraction
- risk factors extraction
- transcript prepared remarks / Q&A split
- guidance paragraph detection

## 41.4 `test_finbert_scorer.py`
Test:
- model loads
- returns bounded numeric score
- fallback returns 0 on failure

## 41.5 `test_composite.py`
Test:
- final score transform
- label mapping
- clipping behavior

## 41.6 `test_end_to_end.py`
Create a small mocked pipeline:
- two fake documents with clear positive tone
- two fake documents with clear negative tone
- verify output directionally makes sense

## 41.7 `test_determinism.py` — important

The node's core claim is determinism. Verify it explicitly:
- run the full pipeline twice with identical mocked inputs
- assert bit-identical `SentimentResponse` output (modulo `generated_at` timestamp)
- if FinBERT is installed, run the FinBERT path and assert score equality to 6 decimal places across two runs

This test is cheap to write and catches regressions that silently break determinism (e.g., dict ordering, unseeded sampling in future model swaps).

---

# 42. Manual Validation Checklist

Before declaring the MVP done, manually verify these items.

## 42.1 Data retrieval
- latest SEC submissions fetch successfully
- at least one 10-K or 10-Q retrieved
- NVIDIA IR page parsed successfully
- at least one press release or transcript found

## 42.2 Parsing
- HTML converts to readable text
- sections are extracted with non-empty content
- transcripts split into prepared remarks and Q&A

## 42.3 Scoring
- obviously positive text produces positive scores
- obviously negative text produces negative scores
- mixed text lands near neutral

## 42.4 Output
- response schema validates
- score always in `[0, 100]`
- confidence always in `[0.10, 0.95]`
- signals are understandable English

---

# 43. CLI Specification

In `node.py`, add a CLI with Typer.

## 43.1 Example command

```bash
nvda-sentiment --as-of-date 2026-04-20 --lookback-quarters 4 --include-market-context true
```

## 43.2 CLI behavior
- build a request object
- run the node
- print pretty JSON to stdout

## 43.3 Helpful local dev behavior
Also add a `main()` block so developers can run:

```bash
python -m nvda_sentiment.node
```

---

# 44. Example Minimal README Content

The repository README should contain:
1. what the node does
2. install instructions
3. CLI example
4. Python usage example
5. output example
6. known limitations

---

# 45. Known Limitations

Document these honestly.

## 45.1 Expected limitations
- section extraction is heuristic, not perfect
- fiscal quarter bucketing uses hardcoded NVIDIA calendar; if NVIDIA changes its fiscal year, update §24.2
- lexicon is intentionally small and contributes small magnitudes (§20.4)
- investor context is broad market context, not NVDA-specific
- no news/social sentiment is included
- official-source availability may vary by quarter
- SEC submissions JSON pagination is not followed, so effective lookback is capped by the base file (§13.8)
- risk factor absolute tone is excluded from filing_tone (§22.1); only risk deltas are used
- AAII sentiment scraping is brittle by design — node tolerates its absence

These are acceptable for the MVP.

---

# 46. Recommended First Implementation Pass

To get the MVP up fast, do exactly this:

## Pass 1 — Working prototype (lexicon-only)
Implement:
- schemas
- SEC adapter (with rate limiter and permanent HTML cache)
- NVIDIA IR adapter
- HTML-to-text
- section extractor (MD&A, risk_factors, outlook_guidance, transcript split)
- lexicon scorer only
- filing_tone (MD&A + guidance)
- filing_delta with NVIDIA fiscal bucketing (tone_delta + risk_delta)
- composite score
- CLI

At this point the node should run end-to-end with no ML dependencies and return a valid response. Confidence will be lower (no FinBERT bonus) but the pipeline works.

## Pass 2 — Model + context overlay
Add:
- FinBERT scorer behind optional import (install with `[finbert]` extra)
- FinBERT determinism pinning
- confidence score
- AAII + VIX investor context (VIX via FRED CSV)
- upgraded signal builder with Q&A-vs-prepared-remarks divergence

## Pass 3 — Polish
Polish:
- caching refinement
- better dedupe
- better section regexes
- end-to-end tests
- determinism test

This phased approach is strongly recommended.

---

# 47. MVP “Do Not Overbuild” Rules

These rules are mandatory.

## 47.1 No database
Use in-memory objects + local cache files only.

## 47.2 No asynchronous architecture
Synchronous requests are fine.

## 47.3 No generalization beyond NVIDIA
This is an NVDA node. Hardcode what is practical.

## 47.4 No fancy NLP extras
No topic models, embeddings, summarization, or custom classifiers in MVP.

## 47.5 No calibration project
Use the prescribed weights. Tune later only if needed.

---

# 48. Implementation Checklist

Use this list to track completion.

## Foundation
- [ ] create repo and package layout (without `filing_parser.py` / `ir_parser.py`)
- [ ] add `pyproject.toml` with FinBERT as an **optional** extra
- [ ] add settings/config with SEC-compliant User-Agent and rate limiter
- [ ] add schemas

## Source adapters
- [ ] implement SEC submissions fetch (respecting 10 req/sec limit)
- [ ] filter and normalize 10-K / 10-Q / 8-K
- [ ] document the submissions-JSON pagination limitation in code
- [ ] implement NVIDIA IR page fetch
- [ ] extract press release / transcript / CFO commentary links
- [ ] implement AAII fetch (with graceful failure)
- [ ] implement VIX fetch via FRED CSV endpoint

## Parsing
- [ ] HTML to text
- [ ] filing section extraction (MD&A, risk_factors, outlook_guidance)
- [ ] transcript split (prepared_remarks vs qa)
- [ ] guidance paragraph detection

## Scoring
- [ ] lexicon scorer
- [ ] FinBERT scorer with optional-import guard
- [ ] FinBERT determinism pinning (revision, deterministic algos, no_grad)
- [ ] section scoring
- [ ] document scoring
- [ ] filing tone (MD&A + guidance only; risk_factors excluded)
- [ ] filing delta with NVIDIA fiscal quarter bucketing
- [ ] risk delta (dedicated sub-component of filing_delta)
- [ ] guidance tone
- [ ] investor context
- [ ] final composite

## Output quality
- [ ] confidence score (base 0.55, max 0.95, floor 0.10)
- [ ] human-readable signals (including Q&A-vs-prepared-remarks divergence)
- [ ] source coverage
- [ ] warnings metadata

## Reliability
- [ ] cache (permanent TTL for SEC filing HTML)
- [ ] logging
- [ ] graceful fallbacks (no-SEC, no-IR, no-FinBERT, no-AAII, no-VIX)
- [ ] unit tests
- [ ] determinism test (`test_determinism.py`)

## Usability
- [ ] CLI
- [ ] README (documenting both core and `[finbert]` install paths)
- [ ] example run output

---

# 49. Recommended Acceptance Criteria

The MVP is ready when all of the following are true:

1. running the node returns a valid response object
2. the response contains a score, label, confidence, components, signals, and coverage
3. the node can run without manual intervention
4. missing sources do not crash the pipeline
5. the output directionally changes when input tone changes
6. code is understandable by another engineer in under 30 minutes

---

# 50. Final Instruction to the Implementer

Build this node as a **small, serious, deterministic financial sentiment component**.

Bias toward:
- simple code
- explicit rules
- graceful degradation
- interpretable scores
- fast implementation

Avoid:
- abstraction for its own sake
- infrastructure for its own sake
- complexity for its own sake

If you are uncertain between a clever design and a simple design, choose the simple design.

That is the correct MVP choice.

---

# 51. Appendix — Suggested Helper Snippets

## 51.1 Clip helper

```python
def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
```

## 51.2 Weight renormalization helper

```python
def renormalize_weights(weights: dict[str, float], available_keys: set[str]) -> dict[str, float]:
    filtered = {k: v for k, v in weights.items() if k in available_keys}
    total = sum(filtered.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in filtered.items()}
```

## 51.3 Word tokenizer

```python
import re

def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']*", text.lower())
```

## 51.4 Simple sentence splitter

```python
import re

def simple_sentence_split(text: str) -> list[str]:
    chunks = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    results = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue
        if len(chunk.split()) < 4:
            continue
        results.append(chunk)
    return results
```

## 51.5 Label mapper

```python
def map_score_to_label(score_0_100: float) -> str:
    if score_0_100 < 35:
        return "bearish"
    if score_0_100 < 45:
        return "mildly bearish"
    if score_0_100 < 55:
        return "neutral"
    if score_0_100 < 65:
        return "mildly bullish"
    return "bullish"
```

---

# 52. Appendix — Minimal Example Python Usage

```python
from datetime import date
from nvda_sentiment import NVDASentimentNode, SentimentRequest

node = NVDASentimentNode()

request = SentimentRequest(
    ticker="NVDA",
    as_of_date=date(2026, 4, 20),
    lookback_quarters=4,
    include_market_context=True,
    use_cache=True,
)

response = node.run(request)
print(response.model_dump_json(indent=2))
```

---

# 53. Appendix — Minimal Example of Expected Output

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
  "source_coverage": {
    "10k_count": 1,
    "10q_count": 3,
    "8k_count": 1,
    "earnings_release_count": 4,
    "transcript_count": 3,
    "cfo_commentary_count": 4
  },
  "metadata": {
    "generated_at": "2026-04-20T12:34:56Z",
    "node_version": "0.1.0",
    "warnings": []
  }
}
```

---

# 54. Final Summary

This MVP should be built as:
- a **single-company**
- **single-node**
- **official-source-first**
- **deterministic**
- **interpretable**
- **lightweight** sentiment engine

The essential ingredients are:
- SEC filings
- NVIDIA IR earnings materials
- section-aware parsing
- FinBERT + tiny finance lexicon
- filing-delta logic
- a very small investor-context overlay

That is enough to produce an implementation that is:
- credible
- impressive
- simple
- fast to build
- useful inside a larger NVDA valuation system


---

# 55. Appendix — Official URLs to Hardcode or Reference

Use these URLs directly in the implementation.

## 55.1 SEC
- Company submissions JSON: `https://data.sec.gov/submissions/CIK0001045810.json`
- SEC API documentation: `https://www.sec.gov/edgar/sec-api-documentation`
- SEC data API root: `https://data.sec.gov/`

## 55.2 NVIDIA Investor Relations
- NVIDIA IR home: `https://investor.nvidia.com/home/default.aspx`
- NVIDIA quarterly results: `https://investor.nvidia.com/financial-info/quarterly-results/default.aspx`
- NVIDIA financial reports: `https://investor.nvidia.com/financial-info/financial-reports/`

## 55.3 Investor Context
- AAII Sentiment Survey: `https://www.aaii.com/sentimentsurvey`
- FRED VIX series page: `https://fred.stlouisfed.org/series/VIXCLS`
- FRED VIX table data: `https://fred.stlouisfed.org/data/VIXCLS`

## 55.4 Request etiquette
- Always send a descriptive user-agent to SEC endpoints.
- Respect timeouts and cache official pages.
- Do not scrape unnecessary pages beyond the sources listed above.
