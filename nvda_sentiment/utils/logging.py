"""Standard-logging wrapper (§31)."""

from __future__ import annotations

import logging
import os

_configured = False


def configure_logging(level: int | str | None = None) -> None:
    """Idempotently configure root logging. Safe to call from anywhere."""
    global _configured
    if _configured:
        return
    resolved_level = level or os.environ.get("NVDA_SENTIMENT_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
