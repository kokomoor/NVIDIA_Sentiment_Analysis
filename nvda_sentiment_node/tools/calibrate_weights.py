#!/usr/bin/env python3
"""Calibration harness — produces a same-day sanity report, not a fit (§8).

Usage:
    python -m tools.calibrate_weights --ticker NVDA
    python -m tools.calibrate_weights --ticker NVDA --grid

Honest disclaimer: with one snapshot per ticker per day, this is O(1) data
points per run — a directional sanity check, not a statistical calibration.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any, Dict, List

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover — runtime-only diagnostic tool
    yf = None  # type: ignore

from nvda_sentiment.config import COMBINER_DEFAULTS, Settings
from nvda_sentiment.features.combiner import combine
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.schemas import SentimentRequest


_EPS_PERIOD_WEIGHTS = {"0q": 0.4, "+1q": 0.3, "0y": 0.2, "+1y": 0.1}


def eps_revision_target(ticker: str) -> float:
    """Weighted analyst EPS-revision direction over the past 30 days, in [-1, +1].

    Returns 0.0 if yfinance is unavailable or the dataframe is empty.
    """
    if yf is None:
        return 0.0
    try:
        t = yf.Ticker(ticker)
        rev = getattr(t, "eps_revisions", None)
    except Exception:
        return 0.0
    if rev is None or getattr(rev, "empty", True):
        return 0.0

    total = 0.0
    wsum = 0.0
    for period, w in _EPS_PERIOD_WEIGHTS.items():
        if period not in rev.index:
            continue
        try:
            up = float(rev.loc[period, "upLast30days"])
            dn = float(rev.loc[period, "downLast30days"])
        except Exception:
            continue
        denom = up + dn
        if denom <= 0:
            continue
        total += w * (up - dn) / denom
        wsum += w

    return (total / wsum) if wsum > 0 else 0.0


def grid_search(L: float, I: float, target: float) -> List[Dict[str, Any]]:
    """Report combiner behavior across a fixed α/λ grid, sorted sign-first then by distance."""
    rows: List[Dict[str, Any]] = []
    for alpha in (0.40, 0.50, 0.55, 0.60, 0.70):
        for lneg in (0.15, 0.25, 0.35):
            for lpos in (0.05, 0.10, 0.15):
                out = combine(
                    leadership_component=L,
                    investor_component=I,
                    alpha=alpha,
                    lambda_neg=lneg,
                    lambda_pos=lpos,
                    investor_branch_ok=True,
                )
                combined_component = (out.combined_score - 50.0) / 50.0
                sign_match = (combined_component * target) >= 0
                dist = abs(combined_component - target)
                rows.append(
                    {
                        "alpha": alpha,
                        "lneg": lneg,
                        "lpos": lpos,
                        "combined": round(combined_component, 3),
                        "target": round(target, 3),
                        "sign_match": sign_match,
                        "dist": round(dist, 3),
                    }
                )
    rows.sort(key=lambda r: (not r["sign_match"], r["dist"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    target = eps_revision_target(args.ticker)
    print(f"[target] weighted EPS-revision direction: {target:+.3f}")

    settings = Settings()
    node = NVDASentimentNode(settings)
    resp = node.run(
        SentimentRequest(ticker=args.ticker, as_of_date=date.today())
    )

    L = float(resp.components.get("leadership_component", 0.0))
    I = float(resp.components.get("investor_component", 0.0))
    print(f"[node]   leadership_component={L:+.3f}  investor_component={I:+.3f}")
    print(f"[node]   combined_score={resp.combined_score}  divergence={resp.divergence}")

    if args.grid:
        rows = grid_search(L, I, target)
        print("\n[grid] top 10 (sign_match first, then closest to target):")
        print(f"{'alpha':>6} {'lneg':>6} {'lpos':>6} {'combined':>10} {'target':>8} {'match':>6} {'dist':>6}")
        for r in rows[:10]:
            print(
                f"{r['alpha']:>6} {r['lneg']:>6} {r['lpos']:>6} {r['combined']:>10} "
                f"{r['target']:>8} {str(r['sign_match']):>6} {r['dist']:>6}"
            )

    report = {
        "ticker": args.ticker,
        "target": round(target, 4),
        "leadership_component": round(L, 4),
        "investor_component": round(I, 4),
        "combined_score": resp.combined_score,
        "current_defaults": COMBINER_DEFAULTS,
    }
    print("\n[report]")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
