# src/nlp/relevance_scoring.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_FINANCE_TERMS = [
    "earnings", "results", "profit", "revenue", "guidance", "margin", "deal", "order",
    "contract", "client", "dividend", "buyback", "acquisition", "merger", "stake",
    "SEBI", "NSE", "BSE", "share", "stock", "quarter", "Q1", "Q2", "Q3", "Q4",
    "FY", "fiscal", "outlook", "target", "rating", "upgrade", "downgrade",
]


def _safe_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _build_doc_text(df: pd.DataFrame) -> pd.Series:
    headline = df.get("headline", pd.Series([""] * len(df))).map(_safe_text)
    summary = df.get("summary", pd.Series([""] * len(df))).map(_safe_text)
    # headline carries more signal → duplicate it slightly
    return (headline + " " + headline + " " + summary).str.strip()


def _build_query_text(symbol: str, company_name: str | None, extra_terms: Iterable[str] | None) -> str:
    parts = [symbol]
    if company_name:
        parts.append(company_name)
        # common aliasing improvements
        parts.append(company_name.replace("Limited", "").replace("Ltd", "").strip())
    parts.extend(DEFAULT_FINANCE_TERMS)
    if extra_terms:
        parts.extend(list(extra_terms))
    # keep non-empty unique-ish
    parts = [p for p in parts if p and isinstance(p, str)]
    return " ".join(parts)


@dataclass
class PercentileLabelConfig:
    high_pct: float = 0.90    # top 10%
    medium_pct: float = 0.70  # next 20%
    # rest → low


class RelevanceScorer:
    """
    TF-IDF + cosine similarity relevance scorer.

    Key: labels are assigned by percentiles so you don't get "all low"
    due to tiny cosine similarity values.
    """

    def __init__(
        self,
        label_cfg: PercentileLabelConfig | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        stop_words: str | None = "english",
    ):
        self.label_cfg = label_cfg or PercentileLabelConfig()
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True,
            min_df=1,
        )

    def score_news(
        self,
        df: pd.DataFrame,
        symbol: str,
        company_name: str | None = None,
        extra_terms: Iterable[str] | None = None,
        score_col: str = "relevance_score",
        label_col: str = "relevance_label",
    ) -> pd.DataFrame:
        """
        Adds relevance_score + relevance_label to cleaned news DF.

        Expected columns: headline, summary (summary optional but recommended)
        """
        if df is None or df.empty:
            logger.info("Relevance scoring: empty DF for %s", symbol)
            return df

        out = df.copy()

        doc_text = _build_doc_text(out)
        query_text = _build_query_text(symbol=symbol, company_name=company_name, extra_terms=extra_terms)

        # Fit on corpus + query so query tokens aren't completely OOV
        corpus = doc_text.tolist() + [query_text]
        X = self.vectorizer.fit_transform(corpus)

        doc_vecs = X[:-1]
        q_vec = X[-1]

        sims = cosine_similarity(doc_vecs, q_vec).reshape(-1)
        out[score_col] = sims.astype(float)

        # Percentile-based labels (per symbol batch)
        hi = float(np.quantile(sims, self.label_cfg.high_pct)) if len(sims) > 1 else float(sims[0])
        med = float(np.quantile(sims, self.label_cfg.medium_pct)) if len(sims) > 1 else float(sims[0])

        def _label(v: float) -> str:
            if v >= hi and hi > 0:
                return "high"
            if v >= med and med > 0:
                return "medium"
            return "low"

        out[label_col] = [ _label(v) for v in sims ]

        logger.info(
            "Relevance scoring done for %s: score[min=%.6f, median=%.6f, max=%.6f], "
            "labels=%s",
            symbol,
            float(np.min(sims)),
            float(np.median(sims)),
            float(np.max(sims)),
            dict(pd.Series(out[label_col]).value_counts()),
        )
        return out