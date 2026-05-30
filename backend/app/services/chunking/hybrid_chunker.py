"""
Chunking strategy for legal contracts.

We rejected fixed-size-only chunking (e.g. always 10k chars) because clauses vary in length
and fixed splits often cut mid-sentence.

Flow:
  1) Try structure-based splits at legal headings (Termination, Liability, …)
  2) If that fails, fall back to semantic splits (when meaning shifts)
  3) Always emit BOTH small and large chunk sets for hybrid retrieval later
"""

import logging

from app.models.document_object import DocumentObject
from app.services.chunking.semantic_chunker import semantic_chunk
from app.services.chunking.structure_chunker import TextChunk, structure_chunk

logger = logging.getLogger(__name__)

# token targets — validated with RAGAS on our question sets (see README)
SMALL_RANGE = (200, 400)
LARGE_RANGE = (1000, 1500)


class HybridChunker:
    def __init__(self, embed_fn=None):
        # embed_fn only needed when we fall back to semantic chunking
        self.embed_fn = embed_fn

    def chunk_document(
        self,
        doc: DocumentObject,
        document_type: str | None = None,
    ) -> tuple[list[TextChunk], dict]:
        all_chunks: list[TextChunk] = []

        # try splitting on section headers first — works well for most contracts
        small_struct = structure_chunk(doc, SMALL_RANGE, document_type)
        large_struct = structure_chunk(doc, LARGE_RANGE, document_type)
        used_structure = bool(small_struct or large_struct)

        # --- small chunks ---
        if small_struct:
            for c in small_struct:
                c.chunk_size = "small"
            all_chunks.extend(small_struct)
        else:
            # no clear headings → split when embedding similarity drops between sentences
            small_sem = semantic_chunk(doc, SMALL_RANGE, self.embed_fn)
            for c in small_sem:
                c.chunk_size = "small"
            all_chunks.extend(small_sem)

        # --- large chunks (same logic, bigger window) ---
        if large_struct:
            for c in large_struct:
                c.chunk_size = "large"
            all_chunks.extend(large_struct)
        else:
            large_sem = semantic_chunk(doc, LARGE_RANGE, self.embed_fn)
            for c in large_sem:
                c.chunk_size = "large"
            all_chunks.extend(large_sem)

        # returned meta ends up on Document.metadata_json["pipeline"] for demos
        meta = {
            "strategy": "structure+semantic" if used_structure else "semantic",
            "structure_used": used_structure,
            "document_type": document_type,
            "small_token_range": SMALL_RANGE,
            "large_token_range": LARGE_RANGE,
            "small_chunk_count": sum(1 for c in all_chunks if c.chunk_size == "small"),
            "large_chunk_count": sum(1 for c in all_chunks if c.chunk_size == "large"),
            "total_chunks": len(all_chunks),
        }
        logger.info("Hybrid chunking: %s", meta)
        return all_chunks, meta
