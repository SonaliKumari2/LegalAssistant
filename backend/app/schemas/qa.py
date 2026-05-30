from pydantic import BaseModel, Field


class AskQuestionRequest(BaseModel):
    document_id: int
    question: str
    language: str = Field(default="en", description="en | hi | hinglish")


class CitationOut(BaseModel):
    page: int | None = None
    section: str | None = None
    clause_number: str | None = None
    char_offset: int | None = None
    excerpt: str | None = None


class AskQuestionResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    sources: list[str]
