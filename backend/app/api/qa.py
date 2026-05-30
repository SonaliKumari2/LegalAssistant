"""
Q&A API — thin layer over RAGPipeline.

Frontend ChatPage calls POST /api/qa/ask.
We also persist the exchange (Conversation + Citation rows) for history / audit.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.documents import _get_owned_doc
from app.database import get_db
from app.models.db_models import Citation, Conversation, User
from app.schemas.qa import AskQuestionRequest, AskQuestionResponse, CitationOut
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()


@router.post("/ask", response_model=AskQuestionResponse)
async def ask_question(
    body: AskQuestionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # make sure this user owns the document (authorization, not just authentication)
    doc = await _get_owned_doc(db, body.document_id, user.id)

    pipeline = RAGPipeline()
    result = pipeline.ask(
        document_id=doc.id,
        question=body.question,
        language=body.language,
        doc_language=doc.language or "en",
    )

    # save chat turn — useful if we add history UI later
    conv = Conversation(
        user_id=user.id,
        document_id=doc.id,
        question=body.question,
        answer=result["answer"],
        language=body.language,
        confidence=result["confidence"],
    )
    db.add(conv)
    await db.flush()

    for c in result["citations"]:
        db.add(
            Citation(
                conversation_id=conv.id,
                page=c.get("page"),
                section=c.get("section"),
                clause_number=c.get("clause_number"),
                char_offset=c.get("char_offset"),
                excerpt=c.get("excerpt"),
            )
        )

    return AskQuestionResponse(
        answer=result["answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
        confidence=result["confidence"],
        sources=result["sources"],
    )
