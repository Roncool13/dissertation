# src/nlp/relevance_scoring.py
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from src.constants import SYMBOL_TO_DESCRIPTION


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


class RelevanceScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing SentenceTransformer model %s on device=%s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _build_text(headline: str, summary: Optional[str]) -> str:
        if summary and isinstance(summary, str) and summary.strip():
            return f"{headline}. {summary}"
        return headline

    def score(self, symbol: str, headlines: List[str], summaries: Optional[List[str]] = None) -> np.ndarray:
        if symbol not in SYMBOL_TO_DESCRIPTION:
            raise ValueError(f"No SYMBOL_TO_DESCRIPTION found for {symbol!r} in constants.py")

        if summaries is None:
            summaries = [None] * len(headlines)

        logger.debug("Scoring relevance for %s headlines for symbol %s", len(headlines), symbol)
        stock_text = SYMBOL_TO_DESCRIPTION[symbol]
        news_texts = [self._build_text(h, s) for h, s in zip(headlines, summaries)]

        logger.debug("Encoding stock description for %s", symbol)
        stock_emb = self.model.encode(stock_text, convert_to_tensor=True, normalize_embeddings=True)
        logger.debug("Encoding %s news items", len(news_texts))
        news_embs = self.model.encode(news_texts, convert_to_tensor=True, normalize_embeddings=True)

        cosine = util.cos_sim(news_embs, stock_emb).squeeze(dim=1)  # [-1,1]
        rel = (cosine + 1.0) / 2.0  # [0,1]
        logger.info("Computed relevance scores for %s items for symbol %s", len(rel), symbol)
        return rel.cpu().numpy()
    

# # src/nlp/relevance_scoring.py
# from __future__ import annotations

# from typing import List, Optional

# import numpy as np
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer, util

# from src.constants_news import SYMBOL_TO_DESCRIPTION


# DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# class RelevanceScorer:
#     """
#     Computes semantic relevance between stock description and news text.
#     """

#     def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = SentenceTransformer(model_name, device=self.device)

#     def _build_news_text(self, headline: str, summary: Optional[str]) -> str:
#         if summary and isinstance(summary, str):
#             return f"{headline}. {summary}"
#         return headline

#     def score(
#         self,
#         symbol: str,
#         headlines: List[str],
#         summaries: Optional[List[str]] = None,
#     ) -> np.ndarray:
#         """
#         Returns relevance scores in [0, 1] for each article.
#         """
#         if symbol not in SYMBOL_TO_DESCRIPTION:
#             raise ValueError(f"No semantic description found for symbol {symbol}")

#         stock_text = SYMBOL_TO_DESCRIPTION[symbol]

#         if summaries is None:
#             summaries = [None] * len(headlines)

#         news_texts = [
#             self._build_news_text(h, s)
#             for h, s in zip(headlines, summaries)
#         ]

#         # Encode
#         stock_emb = self.model.encode(
#             stock_text, convert_to_tensor=True, normalize_embeddings=True
#         )
#         news_embs = self.model.encode(
#             news_texts, convert_to_tensor=True, normalize_embeddings=True
#         )

#         # Cosine similarity → [-1, 1], convert to [0, 1]
#         cosine_scores = util.cos_sim(news_embs, stock_emb).squeeze(dim=1)
#         relevance_scores = (cosine_scores + 1.0) / 2.0

#         return relevance_scores.cpu().numpy()
