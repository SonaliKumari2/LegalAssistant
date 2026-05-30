"""
Document CRUD + upload + pipeline metadata for demos.

Upload is the main entry: saves file to disk, then DocumentIngestionService does the AI pipeline.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.db_models import Document, User
from app.schemas.documents import DocumentDetail, DocumentOut, DocumentTypeUpdate, DocumentUploadResponse
from app.schemas.pipeline import PipelineInfo
from app.services.document_ingestion import DocumentIngestionService
from app.services.vector_store import FaissVectorStore

router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("en"),
    document_type: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(400, detail="Only PDF and DOCX supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # random name so uploads never collide between users
    safe_name = f"{user.id}_{uuid.uuid4().hex}{suffix}"
    dest = upload_dir / safe_name

    content = await file.read()
    dest.write_bytes(content)

    # heavy lifting: parse, classify, chunk, embed, FAISS — see document_ingestion.py
    service = DocumentIngestionService()
    doc = await service.ingest(db, user.id, str(dest), file.filename or safe_name, language, document_type)

    return DocumentUploadResponse(
        id=doc.id,
        title=doc.title,
        document_type=doc.document_type,
        manual_selection_required=doc.manual_selection_required,
        classification_scores=doc.classification_scores,
        page_count=doc.page_count,
        message="Document processed and indexed",
    )


@router.get("/", response_model=list[DocumentOut])
async def list_documents(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user.id)
    return doc


@router.patch("/{document_id}/type", response_model=DocumentOut)
async def set_document_type(
    document_id: int,
    body: DocumentTypeUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """User picked type manually after low classification confidence."""
    doc = await _get_owned_doc(db, document_id, user.id)
    doc.document_type = body.document_type
    doc.manual_selection_required = False
    return doc


@router.get("/{document_id}/pipeline", response_model=PipelineInfo)
async def get_document_pipeline(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # interview demo: prove which steps actually ran on this contract
    doc = await _get_owned_doc(db, document_id, user.id)
    meta = (doc.metadata_json or {}).get("pipeline", {})
    checklist = {
        "ingestion_pdf_docx": True,
        "classification_confidence_gap": doc.classification_scores is not None,
        "hybrid_chunking": "chunking" in meta,
        "google_embeddings": meta.get("embedding", {}).get("provider") == "google",
        "faiss_index": meta.get("vector_store", {}).get("engine") == "faiss",
        "hybrid_retrieval_5_plus_5": meta.get("retrieval", {}).get("small_k") == 5,
        "bge_reranker": "bge-reranker" in (meta.get("retrieval", {}).get("reranker") or ""),
    }
    return PipelineInfo(
        document_id=doc.id,
        document_type=doc.document_type,
        manual_selection_required=doc.manual_selection_required,
        classification_scores=doc.classification_scores,
        pipeline=meta,
        interview_checklist=checklist,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user.id)
    if doc.file_path and os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    FaissVectorStore().delete_document(document_id)
    await db.delete(doc)
    return {"message": "Document deleted"}


async def _get_owned_doc(db: AsyncSession, document_id: int, owner_id: int) -> Document:
    """Shared guard — users can't read each other's contracts."""
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, detail="Document not found")
    if doc.expires_at and doc.expires_at < datetime.utcnow():
        raise HTTPException(410, detail="Document expired")
    return doc
