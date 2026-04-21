"""SQLite history log — record + dedup + load."""

from __future__ import annotations

import sqlite3
from datetime import date

from nvda_sentiment.persistence import (
    load_grid_sweeps,
    load_history,
    record_grid_sweep,
    record_response,
)
from nvda_sentiment.schemas import SentimentResponse


def _grid_rows():
    return [
        {"alpha": 0.40, "lneg": 0.15, "lpos": 0.05, "combined": 0.10,
         "target": 0.20, "sign_match": True, "dist": 0.10},
        {"alpha": 0.55, "lneg": 0.25, "lpos": 0.10, "combined": 0.18,
         "target": 0.20, "sign_match": True, "dist": 0.02},
        {"alpha": 0.70, "lneg": 0.35, "lpos": 0.15, "combined": -0.05,
         "target": 0.20, "sign_match": False, "dist": 0.25},
    ]


def test_grid_sweep_record_and_load(tmp_path):
    db = tmp_path / "h.sqlite3"
    n = record_grid_sweep(
        db,
        ticker="NVDA",
        as_of_date="2026-04-21",
        leadership_component=0.30,
        investor_component=0.10,
        target=0.20,
        generated_at="2026-04-21T10:00:00Z",
        rows=_grid_rows(),
    )
    assert n == 3
    got = load_grid_sweeps(db, "NVDA")
    assert len(got) == 3
    # sign_match rows come first, then sorted by dist ascending
    assert bool(got[0]["sign_match"]) is True
    assert got[0]["dist"] <= got[1]["dist"]
    assert bool(got[-1]["sign_match"]) is False


def test_grid_sweep_same_day_upsert(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_grid_sweep(db, ticker="NVDA", as_of_date="2026-04-21",
                      leadership_component=0.3, investor_component=0.1,
                      target=0.2, generated_at="t1", rows=_grid_rows())
    # Rerun same day with a new target; same (ticker, date, α, λneg, λpos) rows overwrite.
    updated = [dict(r, target=0.40, dist=0.30, sign_match=True) for r in _grid_rows()]
    record_grid_sweep(db, ticker="NVDA", as_of_date="2026-04-21",
                      leadership_component=0.3, investor_component=0.1,
                      target=0.4, generated_at="t2", rows=updated)
    got = load_grid_sweeps(db, "NVDA", as_of_date="2026-04-21")
    assert len(got) == 3  # still only 3 rows — upsert, not append
    assert all(r["target"] == 0.40 for r in got)
    assert all(r["generated_at"] == "t2" for r in got)


def test_grid_sweep_cross_day_accumulates(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_grid_sweep(db, ticker="NVDA", as_of_date="2026-04-20",
                      leadership_component=0.3, investor_component=0.1,
                      target=0.2, generated_at="t1", rows=_grid_rows())
    record_grid_sweep(db, ticker="NVDA", as_of_date="2026-04-21",
                      leadership_component=0.3, investor_component=0.1,
                      target=0.2, generated_at="t2", rows=_grid_rows())
    got = load_grid_sweeps(db, "NVDA")
    assert len(got) == 6
    dates = {r["as_of_date"] for r in got}
    assert dates == {"2026-04-20", "2026-04-21"}


def test_grid_sweep_missing_db_returns_empty(tmp_path):
    assert load_grid_sweeps(tmp_path / "nope.sqlite3") == []


def test_grid_sweep_empty_rows_noop(tmp_path):
    db = tmp_path / "h.sqlite3"
    n = record_grid_sweep(db, ticker="NVDA", as_of_date="2026-04-21",
                          leadership_component=0.0, investor_component=0.0,
                          target=0.0, generated_at="t", rows=[])
    assert n == 0
    assert load_grid_sweeps(db, "NVDA") == []


def _resp(*, ticker="NVDA", as_of=date(2026, 4, 20), combined=61.4, label="mildly bullish"):
    return SentimentResponse(
        ticker=ticker,
        as_of_date=as_of,
        market_sentiment_score=combined,
        leadership_score=combined,
        investor_score=55.0,
        combined_score=combined,
        divergence=6.4,
        label=label,
        confidence=0.80,
        components={
            "filing_tone": 0.3,
            "filing_delta": 0.1,
            "guidance_tone": 0.2,
            "leadership_component": 0.3,
            "investor_component": 0.1,
        },
        signals=["a", "b"],
        source_coverage={"10k_count": 1},
        metadata={"node_version": "0.2.0", "warnings": [], "generated_at": "2026-04-20T10:00:00Z"},
    )


def test_record_and_load_single_row(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_response(db, _resp())
    rows = load_history(db, "NVDA")
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "NVDA"
    assert r["as_of_date"] == "2026-04-20"
    assert r["combined_score"] == 61.4
    assert r["leadership_component"] == 0.3


def test_upsert_dedups_same_day(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_response(db, _resp(combined=60.0, label="mildly bullish"))
    record_response(db, _resp(combined=62.5, label="mildly bullish"))
    rows = load_history(db, "NVDA")
    assert len(rows) == 1, "same-day reruns must upsert, not duplicate"
    assert rows[0]["combined_score"] == 62.5  # second write wins


def test_different_days_accumulate(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_response(db, _resp(as_of=date(2026, 4, 19), combined=55.0))
    record_response(db, _resp(as_of=date(2026, 4, 20), combined=61.4))
    record_response(db, _resp(as_of=date(2026, 4, 21), combined=58.2))
    rows = load_history(db, "NVDA")
    assert [r["as_of_date"] for r in rows] == ["2026-04-19", "2026-04-20", "2026-04-21"]


def test_load_missing_db_returns_empty(tmp_path):
    assert load_history(tmp_path / "does_not_exist.sqlite3", "NVDA") == []


def test_load_filters_by_ticker(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_response(db, _resp(ticker="NVDA"))
    record_response(db, _resp(ticker="AMD"))
    assert len(load_history(db, "NVDA")) == 1
    assert len(load_history(db, "AMD")) == 1
    assert len(load_history(db, None)) == 2


def test_components_and_signals_roundtrip_as_json(tmp_path):
    import json

    db = tmp_path / "h.sqlite3"
    record_response(db, _resp())
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM scores").fetchone())
    assert json.loads(row["components_json"])["filing_tone"] == 0.3
    assert json.loads(row["signals_json"]) == ["a", "b"]


def test_primary_key_is_ticker_plus_date(tmp_path):
    db = tmp_path / "h.sqlite3"
    record_response(db, _resp())
    with sqlite3.connect(str(db)) as conn:
        info = conn.execute("PRAGMA table_info(scores)").fetchall()
        pk_cols = [row[1] for row in info if row[5] > 0]
    assert set(pk_cols) == {"ticker", "as_of_date"}
