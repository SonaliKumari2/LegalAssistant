from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EvaluationRunRequest(BaseModel):
    document_id: int
    questions: list[str]
    ground_truth_answers: list[str] | None = None


class EvaluationOut(BaseModel):
    id: int
    document_id: int | None
    precision: float | None
    recall: float | None
    faithfulness: float | None
    answer_relevance: float | None
    details: dict[str, Any] | None
    created_at: datetime

    class Config:
        from_attributes = True
