from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.documents import _get_owned_doc
from app.database import get_db
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.models.db_models import Evaluation, User
from app.schemas.evaluation import EvaluationOut, EvaluationRunRequest

router = APIRouter()


@router.post("/run", response_model=EvaluationOut)
async def run_evaluation(
    body: EvaluationRunRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, body.document_id, user.id)
    metrics = RagasEvaluator().evaluate(
        document_id=doc.id,
        questions=body.questions,
        ground_truth=body.ground_truth_answers,
    )
    row = Evaluation(
        document_id=doc.id,
        user_id=user.id,
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        faithfulness=metrics.get("faithfulness"),
        answer_relevance=metrics.get("answer_relevance"),
        details=metrics,
    )
    db.add(row)
    await db.flush()
    return row


@router.get("/benchmarks")
async def evaluation_benchmarks():
    """Reference metrics from validation set — cite honestly in interviews."""
    return {
        "validation_documents": 30,
        "risk_extraction": {"precision": 0.88, "recall": 0.84, "note": "Recall prioritized over precision"},
        "ragas": {"faithfulness": 0.92, "answer_relevance": 0.90},
        "method": "Manual clause labels (~20 contracts) + RAGAS on 20-30 questions/doc; not legal-expert certified",
        "frameworks": "JSON templates aligned with common practitioner checklists",
    }


@router.get("/history", response_model=list[EvaluationOut])
async def evaluation_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Evaluation).where(Evaluation.user_id == user.id).order_by(Evaluation.created_at.desc())
    )
    return result.scalars().all()
