"""File-backed overlay for combiner knobs.

The combiner defaults in ``config.COMBINER_DEFAULTS`` are principled priors.
An operator can override them with values learned from the calibration grid
by writing a small JSON file (default ``data/tuned_weights.json``). When
``NVDASentimentNode`` instantiates a default ``Settings``, it applies the
overlay on top. Explicit ``Settings(...)`` instances are untouched.

The current tuner only updates ``alpha`` from one day's data — see §7.3 of the
README for why ``lambda_neg`` / ``lambda_pos`` are not single-day-tunable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def load_overlay(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed overlay dict or None if file is absent / unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def apply_overlay(settings, path: Path) -> Dict[str, Any]:
    """Mutate `settings` in place with any knobs from the overlay file.

    Returns a dict summarizing what was applied (empty if nothing).
    """
    overlay = load_overlay(path)
    if not overlay:
        return {}
    applied: Dict[str, Any] = {}
    if "alpha" in overlay:
        settings.combiner_alpha = float(overlay["alpha"])
        applied["alpha"] = settings.combiner_alpha
    if "lambda_neg" in overlay:
        settings.combiner_lambda_neg = float(overlay["lambda_neg"])
        applied["lambda_neg"] = settings.combiner_lambda_neg
    if "lambda_pos" in overlay:
        settings.combiner_lambda_pos = float(overlay["lambda_pos"])
        applied["lambda_pos"] = settings.combiner_lambda_pos
    return applied


def write_tuned_alpha(
    path: Path,
    *,
    alpha: float,
    basis: Dict[str, Any],
    lambda_neg: Optional[float] = None,
    lambda_pos: Optional[float] = None,
) -> None:
    """Persist a Tier-2 tuned alpha (and optionally λ values) to `path`.

    Tier 2 semantics: one-day data can legitimately inform `alpha` (the
    branch-weighting knob, which fires every run). `lambda_neg` / `lambda_pos`
    should be left at their current defaults unless a multi-day fit justifies
    overriding them. Callers typically pass only `alpha=`.

    `basis` records *why* this value was written (ticker, date, target, dist,
    components) so a future reader can evaluate whether the tune is stale.
    """
    existing = load_overlay(path) or {}
    payload = dict(existing)
    payload["alpha"] = float(alpha)
    if lambda_neg is not None:
        payload["lambda_neg"] = float(lambda_neg)
    if lambda_pos is not None:
        payload["lambda_pos"] = float(lambda_pos)
    payload["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["basis"] = basis
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def clear_overlay(path: Path) -> bool:
    """Delete the overlay file. Returns True if something was removed."""
    if not path.exists():
        return False
    path.unlink()
    return True
