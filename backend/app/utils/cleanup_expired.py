"""Delete expired uploads and FAISS indexes (data minimization / privacy)."""

import logging
import os
from datetime import datetime

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.db_models import Document
from app.services.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


async def cleanup_expired_documents() -> int:
    removed = 0
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.expires_at.isnot(None), Document.expires_at < datetime.utcnow())
        )
        docs = result.scalars().all()
        store = FaissVectorStore()
        for doc in docs:
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)
            store.delete_document(doc.id)
            await db.delete(doc)
            removed += 1
        await db.commit()
    if removed:
        logger.info("Cleaned up %d expired documents", removed)
    return removed
