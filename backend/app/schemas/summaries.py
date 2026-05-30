from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    document_id: int
    language: str = Field(default="en", description="en | hi")


class SummaryResponse(BaseModel):
    executive_summary: str
    key_obligations: list[str]
    important_dates: list[str]
    payment_terms: str
    termination_conditions: str
    risks: list[str]
    legal_concerns: list[str]
