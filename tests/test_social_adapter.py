"""§41 — social adapter (stocktwits/reddit with fake HTTP sessions)."""

from __future__ import annotations

import pytest

from nvda_sentiment.adapters.investor import SocialAdapter
from nvda_sentiment.config import Settings


class _FakeLexicon:
    def __init__(self, score: float = 0.2):
        self._s = score

    def score_text(self, text: str):
        return {
            "lexicon_score": self._s,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "uncertainty_rate": 0.0,
            "litigious_rate": 0.0,
            "token_count": 1.0,
        }


class _FakeFinbert:
    def __init__(self, score: float = 0.4):
        self._s = score

    def score_text(self, text: str):
        return (self._s, 1)


class _FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self.text = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_social_adapter_stocktwits_only(tmp_path, monkeypatch):
    import json as _json

    st_body = _json.dumps({
        "messages": [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
        ]
    })

    class _FakeSession:
        headers: dict[str, str] = {}

        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            if "stocktwits" in url:
                return _FakeResponse(st_body)
            return _FakeResponse("{}", 404)

    adapter = SocialAdapter(
        Settings(cache_dir=tmp_path / ".c"),
        session=_FakeSession(),
        finbert=_FakeFinbert(0.0),
        lexicon=_FakeLexicon(0.0),
    )
    sub = adapter.get_signal()
    assert sub.ok is True
    # 5 tagged = 3 bull - 2 bear → (3-2)/5 = 0.2
    assert sub.score == pytest.approx(0.2)
    assert sub.detail["reddit"] is None


def test_social_adapter_all_sources_fail(tmp_path, monkeypatch):
    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            raise RuntimeError("boom")

    adapter = SocialAdapter(
        Settings(cache_dir=tmp_path / ".c"),
        session=_FakeSession(),
        finbert=_FakeFinbert(0.0),
        lexicon=_FakeLexicon(0.0),
    )
    sub = adapter.get_signal()
    assert sub.ok is False
    assert sub.score == 0.0


def test_social_adapter_reddit_too_few_posts(tmp_path, monkeypatch):
    import json as _json

    reddit_body = _json.dumps(
        {
            "data": {
                "children": [
                    {"data": {"id": "a", "title": "NVDA up", "selftext": "", "score": 10}},
                    {"data": {"id": "b", "title": "NVDA up", "selftext": "", "score": 5}},
                ]
            }
        }
    )

    class _FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            if "reddit" in url:
                return _FakeResponse(reddit_body)
            raise RuntimeError("no stocktwits")

    adapter = SocialAdapter(
        Settings(cache_dir=tmp_path / ".c"),
        session=_FakeSession(),
        finbert=_FakeFinbert(0.5),
        lexicon=_FakeLexicon(0.5),
    )
    sub = adapter.get_signal()
    assert sub.ok is False  # only 2 deduped posts, need >= 3
