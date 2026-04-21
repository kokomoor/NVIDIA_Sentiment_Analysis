"""Tier-2 tuned-weights overlay — file I/O + node auto-apply semantics."""

from __future__ import annotations

import json

from nvda_sentiment.config import COMBINER_DEFAULTS, Settings
from nvda_sentiment.tuned_weights import (
    apply_overlay,
    clear_overlay,
    load_overlay,
    write_tuned_alpha,
)


def _basis():
    return {
        "ticker": "NVDA",
        "as_of_date": "2026-04-21",
        "target": 0.8,
        "leadership_component": -0.04,
        "investor_component": 0.20,
        "dist": 0.66,
        "sign_match": True,
    }


def test_write_then_load_roundtrips(tmp_path):
    path = tmp_path / "tw.json"
    write_tuned_alpha(path, alpha=0.40, basis=_basis())
    data = load_overlay(path)
    assert data is not None
    assert data["alpha"] == 0.40
    assert "updated_at" in data
    assert data["basis"]["ticker"] == "NVDA"
    # Defaults not touched — only alpha persisted.
    assert "lambda_neg" not in data
    assert "lambda_pos" not in data


def test_load_missing_file_returns_none(tmp_path):
    assert load_overlay(tmp_path / "nope.json") is None


def test_load_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "tw.json"
    path.write_text("not-json{{")
    assert load_overlay(path) is None


def test_apply_overlay_mutates_settings_only_for_known_keys(tmp_path):
    path = tmp_path / "tw.json"
    write_tuned_alpha(path, alpha=0.42, basis=_basis())
    s = Settings(history_db_path=tmp_path / "h.sqlite3", cache_dir=tmp_path / ".cache")
    applied = apply_overlay(s, path)
    assert applied == {"alpha": 0.42}
    assert s.combiner_alpha == 0.42
    # λ knobs untouched
    assert s.combiner_lambda_neg == COMBINER_DEFAULTS["lambda_neg"]
    assert s.combiner_lambda_pos == COMBINER_DEFAULTS["lambda_pos"]


def test_apply_overlay_supports_all_three_knobs(tmp_path):
    path = tmp_path / "tw.json"
    path.write_text(json.dumps({"alpha": 0.45, "lambda_neg": 0.30, "lambda_pos": 0.12}))
    s = Settings(history_db_path=tmp_path / "h.sqlite3", cache_dir=tmp_path / ".cache")
    applied = apply_overlay(s, path)
    assert applied == {"alpha": 0.45, "lambda_neg": 0.30, "lambda_pos": 0.12}
    assert s.combiner_alpha == 0.45
    assert s.combiner_lambda_neg == 0.30
    assert s.combiner_lambda_pos == 0.12


def test_apply_overlay_noop_when_file_missing(tmp_path):
    s = Settings(history_db_path=tmp_path / "h.sqlite3", cache_dir=tmp_path / ".cache")
    applied = apply_overlay(s, tmp_path / "nope.json")
    assert applied == {}
    assert s.combiner_alpha == COMBINER_DEFAULTS["alpha"]


def test_write_tuned_alpha_preserves_prior_lambdas(tmp_path):
    path = tmp_path / "tw.json"
    # Seed with a full overlay (e.g. from a prior multi-day fit).
    path.write_text(json.dumps({"alpha": 0.50, "lambda_neg": 0.30, "lambda_pos": 0.12}))
    # Tier-2 tune updates only alpha.
    write_tuned_alpha(path, alpha=0.45, basis=_basis())
    data = load_overlay(path)
    assert data["alpha"] == 0.45
    assert data["lambda_neg"] == 0.30   # preserved
    assert data["lambda_pos"] == 0.12   # preserved


def test_clear_overlay(tmp_path):
    path = tmp_path / "tw.json"
    write_tuned_alpha(path, alpha=0.40, basis=_basis())
    assert path.exists()
    assert clear_overlay(path) is True
    assert not path.exists()
    assert clear_overlay(path) is False  # no-op second time


def test_explicit_settings_bypasses_node_overlay(tmp_path, monkeypatch):
    """An explicit Settings() passed to the node is *never* mutated by the overlay."""
    from nvda_sentiment.node import NVDASentimentNode

    # Seed an overlay at the default path — but settings we pass points elsewhere.
    overlay_path = tmp_path / "tw.json"
    write_tuned_alpha(overlay_path, alpha=0.99, basis=_basis())
    s = Settings(
        history_db_path=tmp_path / "h.sqlite3",
        cache_dir=tmp_path / ".cache",
        tuned_weights_path=overlay_path,
    )
    # Pass settings explicitly — overlay is NOT auto-applied.
    # We don't need to actually run the node to check this; construction is enough.
    node = NVDASentimentNode(
        settings=s,
        sec_adapter=_FakeSEC(),
        ir_adapter=_FakeIR(),
        investor_branch=_FakeBranch(),
    )
    assert node.settings.combiner_alpha == COMBINER_DEFAULTS["alpha"]
    assert node.settings is s


def test_default_settings_applies_node_overlay(tmp_path, monkeypatch):
    """When Settings() is constructed inside the node, the overlay IS applied."""
    from nvda_sentiment.node import NVDASentimentNode

    monkeypatch.chdir(tmp_path)
    overlay_path = tmp_path / "data" / "tuned_weights.json"
    write_tuned_alpha(overlay_path, alpha=0.42, basis=_basis())

    node = NVDASentimentNode(
        sec_adapter=_FakeSEC(),
        ir_adapter=_FakeIR(),
        investor_branch=_FakeBranch(),
    )
    assert node.settings.combiner_alpha == 0.42
    # λ knobs still default — Tier-2 never writes them.
    assert node.settings.combiner_lambda_neg == COMBINER_DEFAULTS["lambda_neg"]
    assert node.settings.combiner_lambda_pos == COMBINER_DEFAULTS["lambda_pos"]


# ---------- minimal fakes for the node constructor ---------- #


class _FakeSEC:
    def get_relevant_filings(self, *a, **kw): return []
    def fetch_filing_html(self, *a, **kw): return ""


class _FakeIR:
    def get_quarterly_results_documents(self, *a, **kw): return []
    def fetch_document_html(self, *a, **kw): return ""


class _FakeBranch:
    def set_use_cache(self, *a, **kw): pass
    def run(self, *a, **kw):
        from nvda_sentiment.schemas import InvestorBranchOutput
        return InvestorBranchOutput(sub_scores=[], investor_component=0.0,
                                    investor_score=50.0, ok=False)
