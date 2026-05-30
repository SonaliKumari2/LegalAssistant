"""
Second-stage retrieval: cross-encoder reranking (BGE-Reranker).

Problem with FAISS alone:
  bi-encoder compares question vector vs chunk vector separately.
  If the chunk shares keywords with the question but isn't the right clause, it still ranks high.

Fix:
  cross-encoder feeds [question + chunk] into the same transformer → much better relevance.
  Tradeoff: slow, so we only run it on ~10 candidates from FAISS, not the whole document.

We picked BGE over MiniLM because legal QA cares more about correctness than a few ms latency.
"""

import logging
from functools import lru_cache

from app.config import get_settings
from app.services.vector_store import StoredChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None  # lazy load — model is heavy, don't load until first question

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        question: str,
        candidates: list[tuple[StoredChunk, float]],
        top_k: int = 8,
    ) -> list[tuple[StoredChunk, float]]:
        if not candidates:
            return []
        try:
            self._load_model()
            # each pair is scored jointly — that's the whole point of cross-encoders
            pairs = [[question, c.chunk_text] for c, _ in candidates]
            scores = self._model.predict(pairs)
            ranked = sorted(
                zip([c for c, _ in candidates], scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
        except Exception as exc:
            # if model fails to load (GPU/mem), fall back to FAISS order rather than crash
            logger.warning("Reranking failed, using vector scores: %s", exc)
            return candidates[:top_k]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
