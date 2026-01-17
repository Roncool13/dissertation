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
import os
import shutil
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinBertConfig:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_length: int = 128
    device: int = -1  # -1 CPU, 0 GPU


def _candidate_model_cache_dirs(model_name: str) -> List[str]:
    """Return candidate on-disk cache dirs for a HF model.

    HF cache layouts vary slightly by version/config. We try the common ones.
    """

    # Example model_name: "ProsusAI/finbert" -> "models--ProsusAI--finbert"
    parts = model_name.split("/")
    if len(parts) == 2:
        org, repo = parts
        cache_leaf = f"models--{org}--{repo}"
    else:
        cache_leaf = "models--" + "--".join(parts)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

    # Newer layout puts everything under HF_HOME/hub
    return [
        os.path.join(hf_home, "hub", cache_leaf),
        os.path.join(hf_home, cache_leaf),
    ]


def _load_finbert_pipeline(cfg: FinBertConfig):
    """Load FinBERT with a cache-safe fallback.

    Strategy:
      1) Try normal load (uses cache, downloads missing files if needed).
      2) If we hit a missing-file error (common in CI restored caches),
         delete ONLY the FinBERT cache entry and force re-download.
    """

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except Exception as e:
        raise ImportError(
            "transformers is required for FinBERT scoring. Install: pip install transformers"
        ) from e

    # Optional exception type (depends on huggingface_hub version)
    try:
        from huggingface_hub.utils import LocalEntryNotFoundError  # type: ignore
    except Exception:  # pragma: no cover
        LocalEntryNotFoundError = ()  # type: ignore

    def _create(force: bool):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            local_files_only=False,
            resume_download=True,
            force_download=force,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            local_files_only=False,
            resume_download=True,
            force_download=force,
        )
        return pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=cfg.device,
        )

    try:
        return _create(force=False)
    except (FileNotFoundError, OSError, LocalEntryNotFoundError) as e:
        # Cache entry exists but some blob/snapshot file is missing.
        # Nuke only this model cache and force a clean download.
        logger.warning(
            "FinBERT cache appears incomplete (missing file). Forcing re-download. Error: %s",
            e,
        )
        for d in _candidate_model_cache_dirs(cfg.model_name):
            shutil.rmtree(d, ignore_errors=True)
        return _create(force=True)


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

    if summaries is None:
        summaries = [""] * len(headlines)

    if len(headlines) != len(summaries):
        raise ValueError("headlines and summaries must have same length")

    texts = [_build_text(h, s) for h, s in zip(headlines, summaries)]

    # Avoid pipeline crashing on empty strings
    safe_texts = [t if t else "." for t in texts]

    clf = _load_finbert_pipeline(cfg)

    labels: List[str] = []
    scores: List[float] = []

    bs = max(1, int(cfg.batch_size))
    for i in range(0, len(safe_texts), bs):
        batch = safe_texts[i : i + bs]
        preds = clf(batch, truncation=True, max_length=cfg.max_length)
        for p in preds:
            # HF returns {'label': 'positive', 'score': 0.xx}
            labels.append(str(p["label"]).lower())
            scores.append(float(p["score"]))

    return labels, np.asarray(scores, dtype=np.float32)
