"""§41.2 — text utilities."""

from nvda_sentiment.utils.text import (
    normalize_unicode_punctuation,
    normalize_whitespace,
    simple_sentence_split,
    tokenize_words,
)


def test_normalize_whitespace_basic():
    assert normalize_whitespace("a\r\n\r\nb") == "a\n\nb"
    assert normalize_whitespace("a\t\tb") == "a b"
    assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"
    assert normalize_whitespace("a    b") == "a b"


def test_normalize_whitespace_strip_nonprintable():
    out = normalize_whitespace("a\x00b")
    assert out == "ab"


def test_normalize_unicode_punctuation():
    s = "\u201chello\u201d \u2014 world \u2026 \xa0"
    out = normalize_unicode_punctuation(s)
    assert out == '"hello" - world ...  '


def test_simple_sentence_split_drops_short():
    text = "This is a real sentence with enough words. too short. Here is another valid sentence with words."
    parts = simple_sentence_split(text)
    assert len(parts) == 2
    assert all(len(p) >= 20 for p in parts)


def test_tokenize_words_lower():
    assert tokenize_words("Hello, WORLD 123 it's") == ["hello", "world", "it's"]
