"""
Risk analysis API — framework JSON + open-ended Gemini pass (risk_analyzer.py).

Runs on demand when user opens the Risks page (not during upload).
"""

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.documents import _get_owned_doc
from app.database import get_db
from app.models.db_models import RiskAnalysis, User
from app.schemas.risks import RiskAnalysisResponse, RiskItem
from app.services.parsers import parse_document
from app.services.risk_analyzer import RiskAnalyzer

router = APIRouter()


@router.post("/analyze/{document_id}", response_model=RiskAnalysisResponse)
async def analyze_risks(
    document_id: int,
    language: str = "en",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user.id)

    # can't pick the right JSON framework without a document type
    if doc.manual_selection_required and not doc.document_type:
        return RiskAnalysisResponse(
            document_id=doc.id, risks=[], high_count=0, medium_count=0, low_count=0
        )

    parsed = parse_document(doc.file_path, doc.filename)
    risks_raw = RiskAnalyzer().analyze(
        parsed.text, doc.document_type or "General Legal Document", language
    )

    # replace previous run — always show latest analysis
    await db.execute(delete(RiskAnalysis).where(RiskAnalysis.document_id == doc.id))
    items: list[RiskItem] = []
    for r in risks_raw:
        row = RiskAnalysis(
            document_id=doc.id,
            clause=r.get("clause", ""),
            explanation=r.get("explanation", ""),
            severity=r.get("severity", "Medium"),
            risk_type=r.get("risk_type"),
            page=r.get("page"),
            section=r.get("section"),
            framework_source=r.get("framework_source"),
        )
        db.add(row)
        items.append(RiskItem(**r))

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for i in items:
        counts[i.severity] = counts.get(i.severity, 0) + 1

    return RiskAnalysisResponse(
        document_id=doc.id,
        risks=items,
        high_count=counts["High"],
        medium_count=counts["Medium"],
        low_count=counts["Low"],
    )


@router.get("/{document_id}", response_model=RiskAnalysisResponse)
async def get_risks(
    document_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_doc(db, document_id, user.id)
    result = await db.execute(select(RiskAnalysis).where(RiskAnalysis.document_id == document_id))
    rows = result.scalars().all()
    items = [
        RiskItem(
            clause=r.clause,
            explanation=r.explanation,
            severity=r.severity,
            risk_type=r.risk_type,
            page=r.page,
            section=r.section,
            framework_source=r.framework_source,
        )
        for r in rows
    ]
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for i in items:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    return RiskAnalysisResponse(
        document_id=document_id,
        risks=items,
        high_count=counts["High"],
        medium_count=counts["Medium"],
        low_count=counts["Low"],
    )
