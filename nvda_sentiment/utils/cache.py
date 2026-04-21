"""Filesystem cache with optional TTL and a permanent tier (§30)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional


class SimpleCache:
    """One-file-per-key filesystem cache.

    - ``set(key, value)``: always writes.
    - ``get(key)``: returns value or None (no TTL check).
    - ``is_fresh(key, ttl_seconds)``: True iff file exists AND age <= ttl.
    - ``get_fresh(key, ttl_seconds)``: combines both.
    - ``get_permanent(key)``: file-existence check; used for immutable SEC
      filing HTML per §30.3.1.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.cache"

    def set(self, key: str, value: str) -> None:
        path = self._path_for(key)
        path.write_text(value, encoding="utf-8")

    def get(self, key: str) -> Optional[str]:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def is_fresh(self, key: str, ttl_seconds: int) -> bool:
        path = self._path_for(key)
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) <= ttl_seconds

    def get_fresh(self, key: str, ttl_seconds: int) -> Optional[str]:
        if self.is_fresh(key, ttl_seconds):
            return self.get(key)
        return None

    def get_permanent(self, key: str) -> Optional[str]:
        """Permanent-tier read — no staleness check (§30.3.1)."""
        path = self._path_for(key)
        return path.read_text(encoding="utf-8") if path.exists() else None
