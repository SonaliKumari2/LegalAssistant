from typing import Any

from pydantic import BaseModel


class PipelineInfo(BaseModel):
    document_id: int
    document_type: str | None
    manual_selection_required: bool
    classification_scores: dict[str, float] | None
    pipeline: dict[str, Any] | None
    interview_checklist: dict[str, bool]
