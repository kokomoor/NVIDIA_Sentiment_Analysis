"""NVDASentimentNode orchestration + Typer CLI (§§33, 39, 43, 6.6, 6.7)."""

from __future__ import annotations

import warnings as _warnings
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Dict, List, Tuple

import typer

from .adapters.nvidia_ir import NvidiaIRAdapter
from .adapters.sec_api import SECAdapter
from .config import Settings
from .features.combiner import CombinerOutput, combine
from .features.confidence import compute_confidence
from .features.filing_delta import compute_filing_delta
from .features.investor_branch import InvestorBranch
from .features.signal_builder import build_signals
from .parsers.html_to_text import html_to_clean_text
from .parsers.section_extractor import SectionExtractor
from .persistence import record_response
from .tuned_weights import apply_overlay
from .schemas import (
    DocumentScore,
    InvestorBranchOutput,
    InvestorSubScore,
    SectionScore,
    SentimentRequest,
    SentimentResponse,
    SourceDocument,
)
from .scorers.composite import (
    compute_filing_tone,
    compute_guidance_tone,
    compute_leadership_component,
    map_score_to_label,
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


def _get_sub(ib: InvestorBranchOutput, name: str) -> float:
    for sub in ib.sub_scores:
        if sub.name == name:
            return round(float(sub.score), 3)
    return 0.0


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
        investor_branch: InvestorBranch | None = None,
        section_extractor: SectionExtractor | None = None,
        section_scorer: SectionScorer | None = None,
        market_context=None,  # deprecated — see §6.6.1
    ):
        configure_logging()
        if settings is None:
            self.settings = Settings()
            # Tier-2 tuning: auto-apply file-backed overlay when the caller
            # hasn't explicitly provided its own Settings (§7.3). Explicit
            # Settings() objects are left untouched so tests and library
            # callers stay deterministic.
            applied = apply_overlay(self.settings, self.settings.tuned_weights_path)
            if applied:
                logger.info("tuned_weights overlay applied: %s", applied)
        else:
            self.settings = settings
        self.sec_adapter = sec_adapter or SECAdapter(self.settings)
        self.ir_adapter = ir_adapter or NvidiaIRAdapter(self.settings)

        if investor_branch is not None and market_context is not None:
            raise ValueError(
                "Pass either `investor_branch=` (preferred) or the deprecated "
                "`market_context=`, not both."
            )
        if market_context is not None:
            _warnings.warn(
                "`market_context=` is deprecated; pass `investor_branch=InvestorBranch(...)` "
                "with `broad_market=...` instead. This shim will be removed in the next release.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.investor_branch = InvestorBranch(
                self.settings, broad_market=market_context
            )
        else:
            self.investor_branch = investor_branch or InvestorBranch(self.settings)

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

        # Honor request.use_cache on document adapters (§3.1).
        for adapter in (self.sec_adapter, self.ir_adapter):
            if hasattr(adapter, "use_cache"):
                adapter.use_cache = request.use_cache
        # Propagate use_cache to every investor sub-adapter too (§6.4).
        self.investor_branch.set_use_cache(request.use_cache)

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

        # 7. Leadership component
        leadership_component = compute_leadership_component(
            filing_tone, filing_delta, guidance_tone
        )

        # 8. Investor branch (§6.6)
        if request.include_investor_branch:
            ib = self.investor_branch.run(
                include_broad_market=request.include_market_context
            )
            if not ib.ok:
                warnings.append("Investor branch unavailable; leadership-only fallback")
        else:
            ib = InvestorBranchOutput(
                sub_scores=[],
                investor_component=0.0,
                investor_score=50.0,
                ok=False,
            )

        # 9. Combine leadership + investor with divergence adjustment.
        cmb: CombinerOutput = combine(
            leadership_component=leadership_component,
            investor_component=ib.investor_component,
            alpha=self.settings.combiner_alpha,
            lambda_neg=self.settings.combiner_lambda_neg,
            lambda_pos=self.settings.combiner_lambda_pos,
            investor_branch_ok=ib.ok,
        )

        # 10. Confidence
        investor_subsources_ok = sum(1 for s in ib.sub_scores if s.ok)
        broad_ok = any(s.name == "broad_market" and s.ok for s in ib.sub_scores)
        confidence = compute_confidence(
            fetched_docs=fetched_docs,
            document_scores=document_scores,
            warnings=warnings,
            fetch_failures=fetch_failures,
            extraction_attempts=extraction_attempts,
            extraction_successes=extraction_successes,
            finbert_available=self.section_scorer.finbert_available,
            include_market_context=request.include_market_context,
            investor_context_available=broad_ok,
            guidance_tone_source_count=guidance_sources,
            filing_delta_computed=(tone_delta != 0.0 or risk_delta != 0.0),
            yoy_matched=yoy_matched,
            include_investor_branch=request.include_investor_branch,
            investor_branch_ok=ib.ok,
            investor_subsources_ok=investor_subsources_ok,
        )

        if not self.section_scorer.finbert_available:
            warnings.append("FinBERT unavailable; lexicon-only scoring")

        # 11. Signals (tiers 6 + 7 live now that the combiner output is available)
        signals = build_signals(
            filing_tone=filing_tone,
            filing_delta=filing_delta,
            tone_delta=tone_delta,
            risk_delta=risk_delta,
            guidance_tone=guidance_tone,
            document_scores=document_scores,
            section_scores_by_type=section_scores_by_type,
            investor_component=ib.investor_component,
            divergence=cmb.divergence,
            include_investor_branch=request.include_investor_branch and ib.ok,
        )

        coverage = _build_source_coverage(fetched_docs)

        combined_rounded = round(cmb.combined_score, 1)
        leadership_rounded = round(cmb.leadership_score, 1)
        investor_rounded = round(cmb.investor_score, 1)
        divergence_rounded = round(cmb.divergence, 1)
        label = map_score_to_label(combined_rounded)

        components = {
            "filing_tone": round(filing_tone, 3),
            "filing_delta": round(filing_delta, 3),
            "guidance_tone": round(guidance_tone, 3),
            "options_flow": _get_sub(ib, "options_flow"),
            "short_interest": _get_sub(ib, "short_interest"),
            "analyst_signal": _get_sub(ib, "analyst_signal"),
            "social": _get_sub(ib, "social"),
            "broad_market": _get_sub(ib, "broad_market"),
            "leadership_component": round(leadership_component, 3),
            "investor_component": round(ib.investor_component, 3),
        }

        response = SentimentResponse(
            ticker=request.ticker,
            as_of_date=date.today(),
            market_sentiment_score=combined_rounded,
            leadership_score=leadership_rounded,
            investor_score=investor_rounded,
            combined_score=combined_rounded,
            divergence=divergence_rounded,
            label=label,
            confidence=round(confidence, 2),
            components=components,
            signals=signals,
            source_coverage=coverage,
            metadata={
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "node_version": self.settings.node_version,
                "warnings": warnings,
            },
        )

        if request.persist_history:
            try:
                record_response(self.settings.history_db_path, response)
            except Exception as exc:  # persistence is best-effort, never fatal
                logger.warning("history persistence failed: %s", exc)

        return response

    def _empty_response(
        self, request: SentimentRequest, warnings: List[str]
    ) -> SentimentResponse:
        resp = SentimentResponse(
            ticker=request.ticker,
            as_of_date=date.today(),
            market_sentiment_score=50.0,
            leadership_score=50.0,
            investor_score=50.0,
            combined_score=50.0,
            divergence=0.0,
            label="neutral",
            confidence=0.10,
            components={
                "filing_tone": 0.0,
                "filing_delta": 0.0,
                "guidance_tone": 0.0,
                "options_flow": 0.0,
                "short_interest": 0.0,
                "analyst_signal": 0.0,
                "social": 0.0,
                "broad_market": 0.0,
                "leadership_component": 0.0,
                "investor_component": 0.0,
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
        if request.persist_history:
            try:
                record_response(self.settings.history_db_path, resp)
            except Exception as exc:
                logger.warning("history persistence failed: %s", exc)
        return resp


# --------------------------------------------------------------------------- #
# Typer CLI (§43, §6.7)
# --------------------------------------------------------------------------- #

cli = typer.Typer(add_completion=False, help="NVIDIA Sentiment Node MVP")


@cli.command("run")
def run_command(
    lookback_quarters: int = typer.Option(4, "--lookback-quarters", min=1, max=8),
    include_market_context: bool = typer.Option(True, "--include-market-context/--no-market-context"),
    include_investor_branch: bool = typer.Option(True, "--include-investor-branch/--no-investor-branch"),
    show_sub_components: bool = typer.Option(False, "--show-sub-components"),
    use_cache: bool = typer.Option(True, "--use-cache/--no-cache"),
    persist_history: bool = typer.Option(True, "--persist-history/--no-persist-history"),
) -> None:
    """Run the sentiment node for today and print the JSON response.

    The pipeline always produces a same-day snapshot (date.today()). If
    ``--persist-history`` is on (the default), the snapshot is upserted into
    the SQLite log at ``settings.history_db_path`` for later longitudinal
    analysis.
    """
    request = SentimentRequest(
        ticker="NVDA",
        lookback_quarters=lookback_quarters,
        include_market_context=include_market_context,
        include_investor_branch=include_investor_branch,
        use_cache=use_cache,
        persist_history=persist_history,
    )
    node = NVDASentimentNode()
    response = node.run(request)

    if show_sub_components:
        sub_keys = ("options_flow", "short_interest", "analyst_signal", "social", "broad_market")
        typer.echo("Investor sub-components:")
        for key in sub_keys:
            val = response.components.get(key, 0.0)
            typer.echo(f"  {key:16s}  {val:+.3f}")
        typer.echo("")

    typer.echo(response.model_dump_json(indent=2))


@cli.command("history")
def history_command(
    ticker: str = typer.Option("NVDA", "--ticker"),
) -> None:
    """Print every recorded snapshot for a ticker as a compact table."""
    from .persistence import load_history

    settings = Settings()
    rows = load_history(settings.history_db_path, ticker)
    if not rows:
        typer.echo(f"(no history yet at {settings.history_db_path})")
        return
    typer.echo(f"{'date':>10}  {'combined':>8}  {'lead':>5}  {'inv':>5}  {'diverg':>6}  {'conf':>4}  label")
    for r in rows:
        typer.echo(
            f"{r['as_of_date']:>10}  {r['combined_score']:>8.1f}  "
            f"{r['leadership_score']:>5.1f}  {r['investor_score']:>5.1f}  "
            f"{r['divergence']:>+6.1f}  {r['confidence']:>4.2f}  {r['label']}"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
