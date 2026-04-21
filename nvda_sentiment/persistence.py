"""Tiny SQLite logger for daily sentiment snapshots (§11.4 follow-up).

The pipeline can only produce *same-day* sentiment — the investor-branch
adapters fetch live data with no historical snapshots. This module lets the
node durably append today's output so an operator accumulates a time series
over days/weeks/months without paid historical data.

Schema: one row per (ticker, as_of_date). Same-day reruns upsert (later run
wins); cross-day runs accumulate. Nothing here depends on network I/O.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from .schemas import SentimentResponse


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
  ticker TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  combined_score REAL NOT NULL,
  leadership_score REAL NOT NULL,
  investor_score REAL NOT NULL,
  divergence REAL NOT NULL,
  confidence REAL NOT NULL,
  label TEXT NOT NULL,
  leadership_component REAL NOT NULL,
  investor_component REAL NOT NULL,
  components_json TEXT NOT NULL,
  signals_json TEXT NOT NULL,
  warnings_json TEXT NOT NULL,
  node_version TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS grid_sweeps (
  ticker TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  alpha REAL NOT NULL,
  lambda_neg REAL NOT NULL,
  lambda_pos REAL NOT NULL,
  leadership_component REAL NOT NULL,
  investor_component REAL NOT NULL,
  combined_component REAL NOT NULL,
  target REAL NOT NULL,
  sign_match INTEGER NOT NULL,
  dist REAL NOT NULL,
  generated_at TEXT NOT NULL,
  PRIMARY KEY (ticker, as_of_date, alpha, lambda_neg, lambda_pos)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


def record_response(db_path: Path, response: SentimentResponse) -> None:
    """Upsert one snapshot keyed by (ticker, as_of_date). Idempotent same-day."""
    row = {
        "ticker": response.ticker,
        "as_of_date": response.as_of_date.isoformat(),
        "combined_score": response.combined_score,
        "leadership_score": response.leadership_score,
        "investor_score": response.investor_score,
        "divergence": response.divergence,
        "confidence": response.confidence,
        "label": response.label,
        "leadership_component": float(response.components.get("leadership_component", 0.0)),
        "investor_component": float(response.components.get("investor_component", 0.0)),
        "components_json": json.dumps(response.components, sort_keys=True),
        "signals_json": json.dumps(response.signals),
        "warnings_json": json.dumps(response.metadata.get("warnings", [])),
        "node_version": str(response.metadata.get("node_version", "")),
        "generated_at": str(response.metadata.get("generated_at", "")),
    }
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scores
              (ticker, as_of_date, combined_score, leadership_score, investor_score,
               divergence, confidence, label, leadership_component, investor_component,
               components_json, signals_json, warnings_json, node_version, generated_at)
            VALUES
              (:ticker, :as_of_date, :combined_score, :leadership_score, :investor_score,
               :divergence, :confidence, :label, :leadership_component, :investor_component,
               :components_json, :signals_json, :warnings_json, :node_version, :generated_at)
            """,
            row,
        )
        conn.commit()


def record_grid_sweep(
    db_path: Path,
    *,
    ticker: str,
    as_of_date: str,
    leadership_component: float,
    investor_component: float,
    target: float,
    generated_at: str,
    rows: List[Dict[str, Any]],
) -> int:
    """Upsert every grid point for a sweep. Idempotent per (ticker, date, α, λneg, λpos).

    ``rows`` is the return value of ``grid_search`` in ``tools/calibrate_weights.py``
    (keys: alpha, lneg, lpos, combined, target, sign_match, dist).
    Returns the number of rows upserted.
    """
    if not rows:
        return 0
    payload = [
        {
            "ticker": ticker,
            "as_of_date": as_of_date,
            "alpha": float(r["alpha"]),
            "lambda_neg": float(r["lneg"]),
            "lambda_pos": float(r["lpos"]),
            "leadership_component": float(leadership_component),
            "investor_component": float(investor_component),
            "combined_component": float(r["combined"]),
            "target": float(target),
            "sign_match": 1 if r["sign_match"] else 0,
            "dist": float(r["dist"]),
            "generated_at": generated_at,
        }
        for r in rows
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO grid_sweeps
              (ticker, as_of_date, alpha, lambda_neg, lambda_pos,
               leadership_component, investor_component,
               combined_component, target, sign_match, dist, generated_at)
            VALUES
              (:ticker, :as_of_date, :alpha, :lambda_neg, :lambda_pos,
               :leadership_component, :investor_component,
               :combined_component, :target, :sign_match, :dist, :generated_at)
            """,
            payload,
        )
        conn.commit()
    return len(payload)


def load_grid_sweeps(
    db_path: Path,
    ticker: str | None = None,
    as_of_date: str | None = None,
) -> List[Dict[str, Any]]:
    """Return grid-sweep rows ordered by (ticker, date, dist). Empty list if DB absent."""
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        clauses: List[str] = []
        params: List[Any] = []
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if as_of_date is not None:
            clauses.append("as_of_date = ?")
            params.append(as_of_date)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = conn.execute(
            f"SELECT * FROM grid_sweeps {where} "
            f"ORDER BY ticker, as_of_date, sign_match DESC, dist ASC",
            params,
        )
        return [dict(r) for r in cur.fetchall()]


def load_history(db_path: Path, ticker: str | None = None) -> List[Dict[str, Any]]:
    """Return rows ordered by (ticker, as_of_date). Empty list if DB absent."""
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if ticker is None:
            cur = conn.execute("SELECT * FROM scores ORDER BY ticker, as_of_date ASC")
        else:
            cur = conn.execute(
                "SELECT * FROM scores WHERE ticker = ? ORDER BY as_of_date ASC",
                (ticker,),
            )
        return [dict(r) for r in cur.fetchall()]
