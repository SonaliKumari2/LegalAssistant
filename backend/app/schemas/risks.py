from pydantic import BaseModel


class RiskItem(BaseModel):
    clause: str
    explanation: str
    severity: str
    risk_type: str | None = None
    page: int | None = None
    section: str | None = None
    framework_source: str | None = None


class RiskAnalysisResponse(BaseModel):
    document_id: int
    risks: list[RiskItem]
    high_count: int
    medium_count: int
    low_count: int
