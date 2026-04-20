"""FinBERT scorer behind an optional import guard (§19).

If ``transformers``/``torch`` aren't installed (the default install path, §9.2),
the scorer becomes a no-op returning ``(0.0, 0)`` and the rest of the pipeline
proceeds with lexicon-only scoring (§36.3).
"""

from __future__ import annotations

import os
from typing import Tuple

from ..utils.logging import get_logger
from ..utils.text import simple_sentence_split


logger = get_logger(__name__)

try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    FINBERT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on default install path
    FINBERT_AVAILABLE = False


class FinBERTScorer:
    """Sentence-level FinBERT scorer with graceful absence."""

    def __init__(self, settings):
        self.settings = settings
        self.available = FINBERT_AVAILABLE
        self._model = None
        self._tokenizer = None
        self._label_positions: dict[str, int] | None = None

        if not self.available:
            logger.info("FinBERT unavailable — falling back to lexicon-only scoring")
            return

        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:
                # Some ops don't have deterministic impls; not fatal for MVP.
                pass

            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.finbert_model_name,
                revision=settings.finbert_model_revision,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.finbert_model_name,
                revision=settings.finbert_model_revision,
            )
            self._model.eval()

            # Map the model's own id2label onto positions for pos/neg/neutral.
            id2label = {
                int(k): v.lower() for k, v in self._model.config.id2label.items()
            }
            self._label_positions = {name: -1 for name in ("positive", "negative", "neutral")}
            for idx, name in id2label.items():
                if name in self._label_positions:
                    self._label_positions[name] = idx
            # If labels look different, assume [positive, negative, neutral] order
            if any(v == -1 for v in self._label_positions.values()):
                self._label_positions = {"positive": 0, "negative": 1, "neutral": 2}
        except Exception as exc:  # pragma: no cover - model load failures
            logger.warning("FinBERT load failed: %s — using lexicon-only", exc)
            self.available = False
            self._model = None
            self._tokenizer = None

    def score_text(self, text: str) -> Tuple[float, int]:
        """Return ``(mean_score, sentence_count)`` in ``[-1, 1]``.

        Returns ``(0.0, 0)`` if FinBERT is unavailable or text has no
        valid sentences.
        """
        if not self.available or not text:
            return (0.0, 0)

        sentences = simple_sentence_split(text)
        if not sentences:
            return (0.0, 0)

        try:
            import torch  # type: ignore (already imported if available)

            scores: list[float] = []
            batch_size = 16
            pos = self._label_positions["positive"]  # type: ignore[index]
            neg = self._label_positions["negative"]  # type: ignore[index]
            with torch.no_grad():
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i : i + batch_size]
                    enc = self._tokenizer(  # type: ignore[misc]
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    logits = self._model(**enc).logits  # type: ignore[misc]
                    probs = torch.softmax(logits, dim=-1)
                    positive_probs = probs[:, pos]
                    negative_probs = probs[:, neg]
                    batch_scores = (positive_probs - negative_probs).tolist()
                    scores.extend(batch_scores)

            if not scores:
                return (0.0, 0)
            mean = sum(scores) / len(scores)
            return (max(-1.0, min(1.0, mean)), len(scores))
        except Exception as exc:  # pragma: no cover
            logger.warning("FinBERT inference failed: %s — returning neutral", exc)
            return (0.0, 0)
