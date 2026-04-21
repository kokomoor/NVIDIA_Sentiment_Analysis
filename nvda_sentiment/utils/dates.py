"""Date helpers (§32.2)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, List

from dateutil import parser as dateparser


def parse_date(value: str | date | datetime | None) -> date | None:
    """Parse a date from a variety of inputs. Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dateparser.parse(value).date()
    except (ValueError, TypeError, OverflowError):
        return None


def calendar_quarter_bucket(date_obj: date) -> str:
    """Return a calendar quarter label like "2026-Q1"."""
    quarter = (date_obj.month - 1) // 3 + 1
    return f"{date_obj.year}-Q{quarter}"


def sort_docs_by_date(docs: Iterable) -> List:
    """Sort a list of SourceDocument-like objects by filed_at, newest last."""
    def _key(doc):
        filed = getattr(doc, "filed_at", None) or ""
        parsed = parse_date(filed)
        return parsed or date.min
    return sorted(list(docs), key=_key)
