"""
Turn text chunks into vectors via Google Embedding API.

Why Google (and not Legal-BERT)?
  - we need Hindi/English/Hinglish — domain-only models are usually English-centric
  - same vendor as Gemini keeps the stack simple

We cache hashes of text → vector so re-ingesting similar chunks doesn't burn API calls.
"""

import asyncio
import hashlib
import logging
from functools import lru_cache

import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BATCH_SIZE = 32
MAX_RETRIES = 3


class EmbeddingService:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = settings.embedding_model
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> list[float]:
        """For document chunks at ingest time."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        for attempt in range(MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self.model, content=text, task_type="retrieval_document"
                )
                vector = result["embedding"]
                self._cache[key] = vector
                return vector
            except Exception as exc:
                logger.warning("Embedding attempt %d failed: %s", attempt + 1, exc)
                if attempt == MAX_RETRIES - 1:
                    raise
        raise RuntimeError("Embedding failed")

    def embed_query(self, text: str) -> list[float]:
        """Separate task type — Google tunes the vector space slightly for queries vs docs."""
        key = self._cache_key("query:" + text)
        if key in self._cache:
            return self._cache[key]
        result = genai.embed_content(model=self.model, content=text, task_type="retrieval_query")
        vector = result["embedding"]
        self._cache[key] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_vectors = [self.embed_text(t) for t in batch]
            vectors.extend(batch_vectors)
        return vectors

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        # FastAPI is async — run the sync Google client in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
