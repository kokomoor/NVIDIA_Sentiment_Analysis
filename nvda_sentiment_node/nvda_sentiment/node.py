"""NVDASentimentNode orchestration + Typer CLI (§§33, 39, 43)."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Dict, List, Tuple

import typer

from .adapters.market_context import MarketContextAdapter
from .adapters.nvidia_ir import NvidiaIRAdapter
from .adapters.sec_api import SECAdapter
from .config import Settings
from .features.confidence import compute_confidence
from .features.filing_delta import compute_filing_delta
from .features.signal_builder import build_signals
from .parsers.html_to_text import html_to_clean_text
from .parsers.section_extractor import SectionExtractor
from .schemas import (
    DocumentScore,
    SectionScore,
    SentimentRequest,
    SentimentResponse,
    SourceDocument,
)
from .scorers.composite import (
    compute_filing_tone,
    compute_final_score,
    compute_guidance_tone,
    score_document,
)
from .scorers.finbert_scorer import FinBERTScorer
from .scorers.lexicon import LexiconScorer
from .scorers.section_scorer import SectionScorer
from .utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Deduplication (§34)
# --------------------------------------------------------------------------- #

def _deduplicate(docs: List[SourceDocument]) -> List[SourceDocument]:
    seen_url: set[str] = set()
    seen_title_date: set[Tuple[str, str]] = set()
    result: List[SourceDocument] = []
    # Priority: IR transcripts/commentary first for transcript-like sources,
    # SEC first for official filings (§34.3). We approximate by keeping insertion
    # order deterministic and preferring the earlier-added item.
    for doc in docs:
        if doc.url and doc.url in seen_url:
            continue
        title_key = ((doc.title or "").lower().strip(), doc.filed_at or "")
        if title_key in seen_title_date:
            continue
        seen_url.add(doc.url)
        seen_title_date.add(title_key)
        result.append(doc)
    return result


# --------------------------------------------------------------------------- #
# Source coverage (§3.2)
# --------------------------------------------------------------------------- #

def _build_source_coverage(docs: List[SourceDocument]) -> Dict[str, int]:
    counts = {
        "10k_count": 0,
        "10q_count": 0,
        "8k_count": 0,
        "earnings_release_count": 0,
        "transcript_count": 0,
        "cfo_commentary_count": 0,
    }
    for d in docs:
        if d.source_type == "10-K":
            counts["10k_count"] += 1
        elif d.source_type == "10-Q":
            counts["10q_count"] += 1
        elif d.source_type == "8-K":
            counts["8k_count"] += 1
        elif d.source_type == "press_release":
            counts["earnings_release_count"] += 1
        elif d.source_type == "transcript":
            counts["transcript_count"] += 1
        elif d.source_type == "cfo_commentary":
            counts["cfo_commentary_count"] += 1
    return counts


# --------------------------------------------------------------------------- #
# The node
# --------------------------------------------------------------------------- #


class NVDASentimentNode:
    """Public entry point — see §3 for input/output contract."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sec_adapter: SECAdapter | None = None,
        ir_adapter: NvidiaIRAdapter | None = None,
        market_context: MarketContextAdapter | None = None,
        section_extractor: SectionExtractor | None = None,
        section_scorer: SectionScorer | None = None,
    ):
        configure_logging()
        self.settings = settings or Settings()
        self.sec_adapter = sec_adapter or SECAdapter(self.settings)
        self.ir_adapter = ir_adapter or NvidiaIRAdapter(self.settings)
        self.market_context = market_context or MarketContextAdapter(self.settings)
        self.section_extractor = section_extractor or SectionExtractor()
        if section_scorer is None:
            finbert = FinBERTScorer(self.settings)
            lexicon = LexiconScorer()
            section_scorer = SectionScorer(self.settings, finbert=finbert, lexicon=lexicon)
        self.section_scorer = section_scorer

    # --- helpers ------------------------------------------------------------

    def _fetch_html(self, doc: SourceDocument) -> str:
        if doc.source_type in ("10-K", "10-Q", "8-K"):
            return self.sec_adapter.fetch_filing_html(doc)
        return self.ir_adapter.fetch_document_html(doc)

    def _score_document(
        self, doc: SourceDocument
    ) -> Tuple[DocumentScore | None, List[SectionScore], Dict[str, str]]:
        sections = self.section_extractor.extract_sections(doc)
        section_scores: List[SectionScore] = []
        for name, text in sections.items():
            section_scores.append(self.section_scorer.score_section(name, text))
        doc_score = score_document(doc, section_scores)
        return doc_score, section_scores, sections

    # --- main entry ---------------------------------------------------------

    def run(self, request: SentimentRequest) -> SentimentResponse:
        warnings: List[str] = []

        # Honor request.use_cache (§3.1) — adapters support it via attribute.
        for adapter in (self.sec_adapter, self.ir_adapter, self.market_context):
            if hasattr(adapter, "use_cache"):
                adapter.use_cache = request.use_cache

        # 1. Fetch source metadata
        try:
            sec_docs = self.sec_adapter.get_relevant_filings(request.lookback_quarters)
        except Exception as exc:
            logger.warning("SEC adapter failed: %s", exc)
            warnings.append("SEC data unavailable")
            sec_docs = []

        try:
            ir_docs = self.ir_adapter.get_quarterly_results_documents(request.lookback_quarters)
        except Exception as exc:
            logger.warning("NVIDIA IR adapter failed: %s", exc)
            warnings.append("NVIDIA IR data unavailable")
            ir_docs = []

        # 2. Merge + dedupe
        docs = _deduplicate(list(sec_docs) + list(ir_docs))

        # 3. Fetch raw content
        fetched_docs: List[SourceDocument] = []
        fetch_failures = 0
        for doc in docs:
            try:
                html = self._fetch_html(doc)
                doc.raw_html = html
                doc.clean_text = html_to_clean_text(html)
                fetched_docs.append(doc)
            except Exception as exc:
                fetch_failures += 1
                logger.warning("Failed to fetch %s: %s", doc.url, exc)
                warnings.append(f"Failed to fetch document: {doc.title}")

        # 4. Extract & score
        document_scores: List[DocumentScore] = []
        section_scores_by_type: Dict[str, List[SectionScore]] = defaultdict(list)
        extraction_attempts = 0
        extraction_successes = 0

        for doc in fetched_docs:
            extraction_attempts += 1
            doc_score, section_scores, sections = self._score_document(doc)
            if sections:
                extraction_successes += 1
            for s in section_scores:
                section_scores_by_type[s.section_name].append(s)
            if doc_score is not None:
                document_scores.append(doc_score)

        # 5. Short-circuit if truly nothing scored (§36.5)
        if not document_scores:
            warnings.append("No scorable documents available")
            return self._empty_response(request, warnings)

        # 6. Tone components
        filing_tone = compute_filing_tone(document_scores)

        filing_delta, tone_delta, risk_delta, yoy_matched = compute_filing_delta(
            document_scores, warnings
        )

        guidance_tone, guidance_sources = compute_guidance_tone(document_scores, warnings)

        # 7. Investor context
        investor_context = 0.0
        investor_context_available = False
        if request.include_market_context:
            try:
                investor_context, investor_context_available = self.market_context.get_combined_score()
                if not investor_context_available:
                    warnings.append("Investor context unavailable; contribution set to 0")
            except Exception as exc:
                logger.warning("market context failed: %s", exc)
                warnings.append("Investor context unavailable; contribution set to 0")
                investor_context = 0.0

        # 8. Composite score
        _, final_score_0_100, label = compute_final_score(
            filing_tone, filing_delta, guidance_tone, investor_context
        )

        # 9. Confidence
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
            guidance_tone_source_count=guidance_sources,
            filing_delta_computed=(tone_delta != 0.0 or risk_delta != 0.0),
            yoy_matched=yoy_matched,
        )

        # Warn if FinBERT unavailable (§36.3)
        if not self.section_scorer.finbert_available:
            warnings.append("FinBERT unavailable; lexicon-only scoring")

        # 10. Signals
        signals = build_signals(
            filing_tone=filing_tone,
            filing_delta=filing_delta,
            tone_delta=tone_delta,
            risk_delta=risk_delta,
            guidance_tone=guidance_tone,
            investor_context=investor_context,
            document_scores=document_scores,
            section_scores_by_type=section_scores_by_type,
            include_market_context=request.include_market_context,
        )

        coverage = _build_source_coverage(fetched_docs)

        return SentimentResponse(
            ticker=request.ticker,
            as_of_date=request.as_of_date,
            market_sentiment_score=round(final_score_0_100, 1),
            label=label,
            confidence=round(confidence, 2),
            components={
                "filing_tone": round(filing_tone, 3),
                "filing_delta": round(filing_delta, 3),
                "guidance_tone": round(guidance_tone, 3),
                "investor_context": round(investor_context, 3),
            },
            signals=signals,
            source_coverage=coverage,
            metadata={
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "node_version": self.settings.node_version,
                "warnings": warnings,
            },
        )

    def _empty_response(
        self, request: SentimentRequest, warnings: List[str]
    ) -> SentimentResponse:
        return SentimentResponse(
            ticker=request.ticker,
            as_of_date=request.as_of_date,
            market_sentiment_score=50.0,
            label="neutral",
            confidence=0.10,
            components={
                "filing_tone": 0.0,
                "filing_delta": 0.0,
                "guidance_tone": 0.0,
                "investor_context": 0.0,
            },
            signals=[
                "Management tone is broadly balanced across recent official materials",
                "Tone changed only modestly versus the prior comparable period",
                "Risk-factor language was substantively stable versus prior filing",
                "Forward-looking guidance language is mixed",
            ],
            source_coverage=_build_source_coverage([]),
            metadata={
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "node_version": self.settings.node_version,
                "warnings": warnings,
            },
        )


# --------------------------------------------------------------------------- #
# Typer CLI (§43)
# --------------------------------------------------------------------------- #

cli = typer.Typer(add_completion=False, help="NVIDIA Sentiment Node MVP")


@cli.command("run")
def run_command(
    as_of_date: str = typer.Option(
        datetime.now(timezone.utc).date().isoformat(),
        "--as-of-date",
        help="ISO date YYYY-MM-DD",
    ),
    lookback_quarters: int = typer.Option(4, "--lookback-quarters", min=1, max=8),
    include_market_context: bool = typer.Option(True, "--include-market-context"),
    use_cache: bool = typer.Option(True, "--use-cache"),
) -> None:
    """Run the sentiment node and print the JSON response."""
    request = SentimentRequest(
        ticker="NVDA",
        as_of_date=date.fromisoformat(as_of_date),
        lookback_quarters=lookback_quarters,
        include_market_context=include_market_context,
        use_cache=use_cache,
    )
    node = NVDASentimentNode()
    response = node.run(request)
    print(response.model_dump_json(indent=2))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
