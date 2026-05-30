from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: int
    title: str
    document_type: str | None
    manual_selection_required: bool
    classification_scores: dict[str, float] | None
    page_count: int
    message: str


class DocumentOut(BaseModel):
    id: int
    title: str
    filename: str
    document_type: str | None
    page_count: int
    language: str
    created_at: datetime
    manual_selection_required: bool

    class Config:
        from_attributes = True


class DocumentDetail(DocumentOut):
    metadata_json: dict[str, Any] | None
    classification_scores: dict[str, float] | None


class DocumentTypeUpdate(BaseModel):
    document_type: str
