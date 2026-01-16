"""src/nlp/sentiment_finbert.py

FinBERT sentiment scoring helpers.

- Uses HuggingFace model: ProsusAI/finbert
- Designed for CPU (GitHub Actions) and GPU (Colab) usage.

Outputs per text:
- sentiment_label: {positive, negative, neutral}
- sentiment_score: float in [0, 1] (confidence for predicted label)

Note:
- We intentionally keep this module small + dependency-light.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinBertConfig:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 128
    device: int = -1  # -1 CPU, 0 GPU


def _normalize_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s


def _build_text(headline: object, summary: object | None) -> str:
    h = _normalize_text(headline)
    s = _normalize_text(summary)
    if s:
        return f"{h}. {s}".strip()
    return h


def score_finbert(
    headlines: Sequence[object],
    summaries: Sequence[object] | None = None,
    cfg: FinBertConfig | None = None,
) -> Tuple[List[str], np.ndarray]:
    """Score (headline, summary) pairs with FinBERT.

    Returns:
      labels: list[str]
      scores: np.ndarray float32

    Raises:
      ImportError if transformers/torch are missing.
    """

    cfg = cfg or FinBertConfig()

    try:
        from transformers import pipeline
    except Exception as e:
        raise ImportError(
            "transformers is required for FinBERT scoring. Install: pip install transformers"
        ) from e

    if summaries is None:
        summaries = [""] * len(headlines)

    if len(headlines) != len(summaries):
        raise ValueError("headlines and summaries must have same length")

    texts = [_build_text(h, s) for h, s in zip(headlines, summaries)]

    # Avoid pipeline crashing on empty strings
    safe_texts = [t if t else "." for t in texts]

    clf = pipeline(
        task="text-classification",
        model=cfg.model_name,
        tokenizer=cfg.model_name,
        device=cfg.device,
        truncation=True,
        max_length=cfg.max_length,
        # return_all_scores=False by default
    )

    labels: List[str] = []
    scores: List[float] = []

    bs = max(1, int(cfg.batch_size))
    for i in range(0, len(safe_texts), bs):
        batch = safe_texts[i : i + bs]
        preds = clf(batch)
        for p in preds:
            # HF returns {'label': 'positive', 'score': 0.xx}
            labels.append(str(p["label"]).lower())
            scores.append(float(p["score"]))

    return labels, np.asarray(scores, dtype=np.float32)
