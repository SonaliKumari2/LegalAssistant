"""
Upload orchestration — ties the whole ingest pipeline together.

Interview tip: this is the "Phase 1" entry point. Everything after upload flows through here
once; questions later reuse the FAISS index without re-parsing the PDF.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db_models import Chunk, Document
from app.models.document_object import DocumentObject
from app.services.chunking.hybrid_chunker import HybridChunker
from app.services.document_classifier import DocumentClassifier
from app.services.embedding_service import get_embedding_service
from app.services.parsers import parse_document
from app.services.vector_store import FaissVectorStore, StoredChunk
from app.utils.pii_redaction import redact_pii

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentIngestionService:
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.embeddings = get_embedding_service()
        self.vector_store = FaissVectorStore()
        # semantic chunking needs embed_fn; we pass Google's batch embed here
        self.chunker = HybridChunker(embed_fn=lambda texts: self.embeddings.embed_batch(texts))

    async def ingest(
        self,
        db: AsyncSession,
        owner_id: int,
        file_path: str,
        filename: str,
        language: str = "en",
        forced_type: str | None = None,
    ) -> Document:
        # --- Step 1: Parse PDF/DOCX into structured DocumentObject ---
        doc_obj: DocumentObject = parse_document(file_path, filename)
        # mask emails, phones, PAN, Aadhaar before anything hits the DB or LLM
        doc_obj.text = redact_pii(doc_obj.text)

        # --- Step 2: Classify contract type (employment, NDA, lease, etc.) ---
        # uses representative text + optional mean embedding — not just first 5k chars
        classification = self.classifier.classify_document(
            doc_obj,
            embed_fn=lambda texts: self.embeddings.embed_batch(texts),
        )
        document_type = forced_type or classification.get("document_type")
        # if user already picked a type on upload, skip the manual flag
        manual = classification.get("manual_selection_required", False) and not forced_type

        # --- Step 3: Hybrid chunking (structure first, semantic fallback) ---
        # document_type tweaks which legal headings we look for when splitting
        text_chunks, chunk_meta = self.chunker.chunk_document(
            doc_obj,
            document_type=document_type or "General Legal Document",
        )

        # stash pipeline info on the document — handy for /pipeline API in demos
        pipeline_meta = {
            "classification": {
                "method": "representative_text + optional_mean_embedding + gemini",
                "confidence_gap": classification.get("confidence_gap"),
                "manual_selection_required": manual,
                "scores": classification.get("classification_scores"),
            },
            "chunking": chunk_meta,
            "embedding": {"provider": "google", "model": settings.embedding_model},
            "vector_store": {"engine": "faiss", "similarity": "cosine_via_l2_normalized_ip"},
            "retrieval": {
                "small_k": settings.retrieval_small_k,
                "large_k": settings.retrieval_large_k,
                "reranker": settings.reranker_model,
            },
        }

        # auto-delete files after TTL — data minimization for privacy story in interviews
        expires = datetime.utcnow() + timedelta(hours=settings.document_ttl_hours)
        record = Document(
            owner_id=owner_id,
            title=doc_obj.title or filename,
            filename=filename,
            file_path=file_path,
            document_type=document_type,
            classification_scores=classification.get("classification_scores"),
            manual_selection_required=manual,
            language=classification.get("detected_language", language),
            metadata_json={**(doc_obj.metadata or {}), "pipeline": pipeline_meta},
            page_count=doc_obj.page_count,
            expires_at=expires,
        )
        db.add(record)
        await db.flush()  # need record.id before we attach chunks / FAISS

        # --- Step 4: Embed every chunk, then index in FAISS ---
        texts = [c.text for c in text_chunks]
        vectors = await self.embeddings.embed_batch_async(texts)

        stored: list[StoredChunk] = []
        db_chunks: list[Chunk] = []
        for i, (tc, vec) in enumerate(zip(text_chunks, vectors)):
            # FAISS side keeps full text + page/section for retrieval & citations
            stored.append(
                StoredChunk(
                    chunk_id=tc.chunk_id,
                    chunk_text=tc.text,
                    page_number=tc.page_number,
                    section_name=tc.section_name,
                    document_id=record.id,
                    document_type=document_type,
                    chunk_size=tc.chunk_size,
                    metadata={"char_start": tc.char_start, "char_end": tc.char_end},
                    embedding_id=i,
                )
            )
            # SQLite stores a truncated copy for admin/debug; search uses FAISS
            db_chunks.append(
                Chunk(
                    document_id=record.id,
                    chunk_id=tc.chunk_id,
                    chunk_text=tc.text[:5000],
                    chunk_size=tc.chunk_size,
                    page_number=tc.page_number,
                    section_name=tc.section_name,
                    char_start=tc.char_start,
                    char_end=tc.char_end,
                    embedding_id=i,
                )
            )

        self.vector_store.add_document(record.id, vectors, stored)
        db.add_all(db_chunks)
        await db.flush()
        logger.info("Ingested document %s with %d chunks", record.id, len(db_chunks))
        return record
