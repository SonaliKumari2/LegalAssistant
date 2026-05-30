"""
FAISS = local vector database for fast similarity search.

Why FAISS over Pinecone/Chroma?
  - runs on our machine, no extra API, better privacy story for legal docs
  - free and fast enough for per-document indexes

Similarity: we L2-normalize vectors then use inner product → equivalent to cosine similarity.
Each uploaded contract gets its own index file pair (.faiss + .meta.pkl).
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class StoredChunk:
    """Everything we need at search time — text plus citation metadata."""

    chunk_id: str
    chunk_text: str
    page_number: int | None
    section_name: str | None
    document_id: int
    document_type: str | None
    chunk_size: str  # "small" or "large" — used by hybrid retrieval
    metadata: dict
    embedding_id: int


class FaissVectorStore:
    def __init__(self, index_dir: str | None = None):
        self.index_dir = Path(index_dir or settings.faiss_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, document_id: int) -> tuple[Path, Path]:
        base = self.index_dir / f"doc_{document_id}"
        return base.with_suffix(".faiss"), base.with_suffix(".meta.pkl")

    def add_document(
        self,
        document_id: int,
        embeddings: list[list[float]],
        chunks: list[StoredChunk],
    ) -> None:
        if not embeddings:
            raise ValueError("No embeddings to index")
        dim = len(embeddings[0])
        matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)  # so inner product ≈ cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        faiss_path, meta_path = self._paths(document_id)
        faiss.write_index(index, str(faiss_path))
        # pickle keeps chunk text + page numbers aligned with vector row indices
        with open(meta_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Indexed %d chunks for document %d", len(chunks), document_id)

    def search(
        self,
        document_id: int,
        query_embedding: list[float],
        k: int = 5,
        chunk_size: str | None = None,
    ) -> list[tuple[StoredChunk, float]]:
        faiss_path, meta_path = self._paths(document_id)
        if not faiss_path.exists():
            return []

        index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            chunks: list[StoredChunk] = pickle.load(f)

        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        # fetch extra candidates when filtering by small/large — then trim to k
        scores, indices = index.search(q, min(k * 3, len(chunks)))

        results: list[tuple[StoredChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = chunks[idx]
            if chunk_size and chunk.chunk_size != chunk_size:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results[:k]

    def delete_document(self, document_id: int) -> None:
        faiss_path, meta_path = self._paths(document_id)
        for p in (faiss_path, meta_path):
            if p.exists():
                p.unlink()

    def update_document(
        self,
        document_id: int,
        embeddings: list[list[float]],
        chunks: list[StoredChunk],
    ) -> None:
        self.delete_document(document_id)
        self.add_document(document_id, embeddings, chunks)
