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
from typing import Any, Dict, List

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover — runtime-only diagnostic tool
    yf = None  # type: ignore

from nvda_sentiment.config import COMBINER_DEFAULTS, Settings
from nvda_sentiment.features.combiner import combine
from nvda_sentiment.node import NVDASentimentNode
from nvda_sentiment.persistence import load_grid_sweeps, load_history, record_grid_sweep
from nvda_sentiment.schemas import SentimentRequest
from nvda_sentiment.tuned_weights import (
    clear_overlay,
    load_overlay,
    write_tuned_alpha,
)


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
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Print every past snapshot logged for the ticker.",
    )
    parser.add_argument(
        "--persist-grid",
        dest="persist_grid",
        action="store_true",
        default=True,
        help="Upsert grid-sweep results into the SQLite log (default: on; implies --grid).",
    )
    parser.add_argument(
        "--no-persist-grid",
        dest="persist_grid",
        action="store_false",
        help="Do not write grid-sweep results to the SQLite log.",
    )
    parser.add_argument(
        "--show-grid-history",
        action="store_true",
        help="Print all accumulated grid-sweep rows for the ticker.",
    )
    parser.add_argument(
        "--write-best-alpha",
        action="store_true",
        help=(
            "Tier-2 tune: take today's best-row alpha from the grid and write it "
            "to the tuned-weights overlay (implies --grid). λ knobs are NOT "
            "updated — they require multi-day data (see README §7.3)."
        ),
    )
    parser.add_argument(
        "--clear-tuned-weights",
        action="store_true",
        help="Delete the tuned-weights overlay file and revert to config defaults.",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.clear_tuned_weights:
        removed = clear_overlay(settings.tuned_weights_path)
        if removed:
            print(f"[tune] cleared overlay at {settings.tuned_weights_path}")
        else:
            print(f"[tune] no overlay to clear at {settings.tuned_weights_path}")
        return

    overlay = load_overlay(settings.tuned_weights_path)
    if overlay:
        print(
            f"[tune] active overlay from {settings.tuned_weights_path}: "
            f"alpha={overlay.get('alpha', '—')}  "
            f"λ_neg={overlay.get('lambda_neg', '(default)')}  "
            f"λ_pos={overlay.get('lambda_pos', '(default)')}  "
            f"updated={overlay.get('updated_at', '?')}"
        )

    target = eps_revision_target(args.ticker)
    print(f"[target] weighted EPS-revision direction: {target:+.3f}")

    # Reuse the `settings` constructed above. Pass it explicitly so the node
    # does NOT apply the overlay a second time — calibration should reason
    # from the raw defaults unless the overlay is already visible above.
    node = NVDASentimentNode(settings)
    resp = node.run(SentimentRequest(ticker=args.ticker))

    L = float(resp.components.get("leadership_component", 0.0))
    I = float(resp.components.get("investor_component", 0.0))
    print(f"[node]   leadership_component={L:+.3f}  investor_component={I:+.3f}")
    print(f"[node]   combined_score={resp.combined_score}  divergence={resp.divergence}")

    run_grid = args.grid or args.write_best_alpha
    if run_grid:
        rows = grid_search(L, I, target)
        print("\n[grid] top 10 (sign_match first, then closest to target):")
        print(f"{'alpha':>6} {'lneg':>6} {'lpos':>6} {'combined':>10} {'target':>8} {'match':>6} {'dist':>6}")
        for r in rows[:10]:
            print(
                f"{r['alpha']:>6} {r['lneg']:>6} {r['lpos']:>6} {r['combined']:>10} "
                f"{r['target']:>8} {str(r['sign_match']):>6} {r['dist']:>6}"
            )

        if args.persist_grid:
            try:
                n = record_grid_sweep(
                    settings.history_db_path,
                    ticker=args.ticker,
                    as_of_date=resp.as_of_date.isoformat(),
                    leadership_component=L,
                    investor_component=I,
                    target=target,
                    generated_at=str(resp.metadata.get("generated_at", "")),
                    rows=rows,
                )
                print(f"\n[grid] persisted {n} rows to {settings.history_db_path}")
            except Exception as exc:
                print(f"\n[grid] persistence failed: {exc}")

        if args.write_best_alpha:
            best = rows[0]
            basis = {
                "ticker": args.ticker,
                "as_of_date": resp.as_of_date.isoformat(),
                "target": round(target, 4),
                "leadership_component": round(L, 4),
                "investor_component": round(I, 4),
                "dist": best["dist"],
                "sign_match": best["sign_match"],
                "tier": 2,
                "note": (
                    "Only alpha is persisted from single-day data. lambda_neg and "
                    "lambda_pos remain at their current Settings values — they are "
                    "not tunable from a single snapshot (see README §7.3)."
                ),
            }
            try:
                write_tuned_alpha(
                    settings.tuned_weights_path,
                    alpha=float(best["alpha"]),
                    basis=basis,
                )
                print(
                    f"\n[tune] wrote alpha={best['alpha']} to "
                    f"{settings.tuned_weights_path} "
                    f"(λ_neg/λ_pos left at current defaults). "
                    f"Dist={best['dist']:.3f}, sign_match={best['sign_match']}."
                )
                print(
                    "[tune] NOTE: single-day tuning is Tier-2. Re-run on future days "
                    "and compare; update only if the winning alpha is consistent."
                )
            except Exception as exc:
                print(f"\n[tune] failed to write overlay: {exc}")

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

    if args.show_history:
        rows = load_history(settings.history_db_path, args.ticker)
        if not rows:
            print(f"\n[history] (empty — DB at {settings.history_db_path} has no rows yet)")
        else:
            print(f"\n[history] {len(rows)} snapshot(s) logged at {settings.history_db_path}:")
            print(f"  {'date':>10}  {'combined':>8}  {'lead_c':>7}  {'inv_c':>7}  {'diverg':>6}  {'conf':>4}")
            for r in rows:
                print(
                    f"  {r['as_of_date']:>10}  {r['combined_score']:>8.1f}  "
                    f"{r['leadership_component']:>+7.3f}  {r['investor_component']:>+7.3f}  "
                    f"{r['divergence']:>+6.1f}  {r['confidence']:>4.2f}"
                )

    if args.show_grid_history:
        g_rows = load_grid_sweeps(settings.history_db_path, args.ticker)
        if not g_rows:
            print(f"\n[grid-history] (empty — no grid sweeps logged for {args.ticker})")
        else:
            print(f"\n[grid-history] {len(g_rows)} row(s) across all sweeps, sign-match first then by dist:")
            print(f"  {'date':>10}  {'alpha':>5}  {'lneg':>5}  {'lpos':>5}  "
                  f"{'combined':>8}  {'target':>7}  {'match':>5}  {'dist':>5}")
            for r in g_rows:
                print(
                    f"  {r['as_of_date']:>10}  {r['alpha']:>5.2f}  "
                    f"{r['lambda_neg']:>5.2f}  {r['lambda_pos']:>5.2f}  "
                    f"{r['combined_component']:>+8.3f}  {r['target']:>+7.3f}  "
                    f"{bool(r['sign_match']):>5}  {r['dist']:>5.3f}"
                )


if __name__ == "__main__":
    main()
