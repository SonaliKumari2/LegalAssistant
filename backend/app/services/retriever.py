"""
Hybrid retrieval: pull top small chunks AND top large chunks, then rerank.

Why two sizes?
  - small (200-400 tokens) → better precision for finding the exact clause
  - large (1000-1500 tokens) → enough surrounding context for Gemini to interpret it

FAISS alone is a bi-encoder (fast but keyword-noisy). We always rerank after this.
"""

import logging

from app.config import get_settings
from app.services.embedding_service import get_embedding_service
from app.services.reranker import get_reranker
from app.services.vector_store import FaissVectorStore, StoredChunk

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridRetriever:
    def __init__(self, vector_store: FaissVectorStore | None = None):
        self.vector_store = vector_store or FaissVectorStore()
        self.embeddings = get_embedding_service()
        self.reranker = get_reranker()

    def retrieve(
        self,
        document_id: int,
        question: str,
        small_k: int | None = None,
        large_k: int | None = None,
        rerank_top: int | None = None,
    ) -> list[tuple[StoredChunk, float]]:
        small_k = small_k or settings.retrieval_small_k
        large_k = large_k or settings.retrieval_large_k
        rerank_top = rerank_top or settings.rerank_top_k

        # embed the user question (retrieval_query task type in embedding_service)
        query_vec = self.embeddings.embed_query(question)

        # two separate FAISS searches filtered by chunk_size tag
        small = self.vector_store.search(document_id, query_vec, k=small_k, chunk_size="small")
        large = self.vector_store.search(document_id, query_vec, k=large_k, chunk_size="large")

        # merge & dedupe — same chunk might appear in both lists with different scores
        merged: dict[str, tuple[StoredChunk, float]] = {}
        for chunk, score in small + large:
            if chunk.chunk_id not in merged or score > merged[chunk.chunk_id][1]:
                merged[chunk.chunk_id] = (chunk, score)

        candidates = list(merged.values())
        candidates.sort(key=lambda x: x[1], reverse=True)

        # BGE cross-encoder re-scores (question, chunk) pairs — slower but much more accurate
        return self.reranker.rerank(question, candidates, top_k=rerank_top)
