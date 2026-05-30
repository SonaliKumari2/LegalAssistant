"""
Contract summary API — one-shot Gemini call with structured JSON output.

Note: summarization reads the parsed file directly (not FAISS).
For very long contracts we truncate in summarizer.py — Q&A still uses full RAG index.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.documents import _get_owned_doc
from app.database import get_db
from app.models.db_models import User
from app.schemas.summaries import SummaryRequest, SummaryResponse
from app.services.parsers import parse_document
from app.services.summarizer import ContractSummarizer

router = APIRouter()


@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(
    body: SummaryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, body.document_id, user.id)
    parsed = parse_document(doc.file_path, doc.filename)
    data = ContractSummarizer().summarize(parsed.text, body.language)
    return SummaryResponse(**data)
