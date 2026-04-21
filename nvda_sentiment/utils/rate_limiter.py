"""Simple sleep-based rate limiter (§13.9)."""

from __future__ import annotations

import time


class RateLimiter:
    """Enforce a minimum interval between successive calls."""

    def __init__(self, rps: float):
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._min_interval = 1.0 / rps
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()
