"""Public, validated result of a prescriptive request."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from alpha_engine.data.news_context import NewsContext
from alpha_engine.data.prescriptive_schemas import RecommendationRequestStatus


class RecommendationView(BaseModel):
    """Safe fields only; prompt, context and raw provider output stay private."""

    prediction_id: str
    candidate_signal: Literal["BUY", "SELL", "HOLD"]
    llm_signal: Literal["BUY", "SELL", "HOLD"] | None
    final_signal: Literal["BUY", "SELL", "HOLD"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    llm_model: str
    prompt_version: str
    summary: str | None = None
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    warning: str | None = None


class RecommendationResult(BaseModel):
    request_id: UUID
    prediction_id: str | None
    status: RecommendationRequestStatus
    requested_at: datetime
    generated_at: datetime | None
    finished_at: datetime
    valid_until: datetime | None
    timing_status: Literal["TIMELY", "LATE"] | None
    llm_latency_ms: float | None
    total_latency_ms: float
    cached: bool
    actionable: bool
    recommendation: RecommendationView | None = None
    guardrails: list[str] = Field(default_factory=list)
    news_context: NewsContext | None = None

    def current_validity(self, now: datetime) -> Literal["ACTIVE", "EXPIRED"] | None:
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if self.valid_until is None:
            return None
        return "ACTIVE" if now < self.valid_until else "EXPIRED"
