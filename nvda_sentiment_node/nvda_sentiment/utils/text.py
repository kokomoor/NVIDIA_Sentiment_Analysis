"""Text normalization, sentence splitting, tokenization (§32.1)."""

from __future__ import annotations

import re
import unicodedata


_SMART_PUNCT = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\xa0": " ",  # non-breaking space
}


def normalize_unicode_punctuation(text: str) -> str:
    """Replace smart quotes, dashes, ellipsis, nbsp with ASCII equivalents."""
    if not text:
        return ""
    for bad, good in _SMART_PUNCT.items():
        text = text.replace(bad, good)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace per §16.4 rules."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    # strip non-printable chars except \n
    text = "".join(
        ch for ch in text
        if ch == "\n" or (unicodedata.category(ch)[0] != "C")
    )
    # collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)
    # trim trailing spaces on each line
    text = re.sub(r"[ ]+\n", "\n", text)
    return text.strip()


def simple_sentence_split(text: str) -> list[str]:
    """Split text into sentences, dropping tiny fragments (§18, §51.4)."""
    if not text:
        return []
    chunks = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    results: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:
            continue
        if len(chunk.split()) < 4:
            continue
        results.append(chunk)
    return results


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']*")


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokenizer (§51.3)."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())
