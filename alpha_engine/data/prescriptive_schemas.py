"""Domain records for on-demand prescriptive requests and their audit trail."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

RecommendationRequestStatus = Literal[
    "SUCCESS", "CACHED", "EXPIRED_BEFORE_REQUEST", "DEADLINE_TOO_CLOSE",
    "LATE_RESULT", "LLM_TIMEOUT", "PROVIDER_ERROR", "INVALID_RESPONSE",
    "GUARDRAIL_REJECTED", "PREDICTION_NOT_FOUND", "NO_ACTIVE_PREDICTION",
    "PREDICTION_INVALID", "PERSISTENCE_ERROR", "RATE_LIMITED",
]


class RecommendationRequestRecord(BaseModel):
    request_id: UUID
    prediction_id: str | None = None
    symbol: str
    interval: str
    provider: str
    llm_model: str
    prompt_version: str
    requested_at: datetime
    generated_at: datetime | None = None
    finished_at: datetime
    valid_until: datetime | None = None
    llm_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    status: RecommendationRequestStatus
    timing_status: Literal["TIMELY", "LATE"] | None = None
    cached: bool = False
    error_type: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    news_context_id: UUID | None = None
    news_context_used: bool = False
    news_context_fetched_at: datetime | None = None
    news_context_count: int | None = Field(default=None, ge=0)
    news_context_sentiment: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def validate_record(self) -> "RecommendationRequestRecord":
        for value in (
            self.requested_at, self.generated_at, self.finished_at,
            self.valid_until, self.created_at, self.news_context_fetched_at,
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError("Los timestamps prescriptivos deben incluir zona horaria")
        if self.finished_at < self.requested_at:
            raise ValueError("finished_at no puede preceder requested_at")
        if self.generated_at is not None and self.generated_at < self.requested_at:
            raise ValueError("generated_at no puede preceder requested_at")
        if self.timing_status is not None and (
            self.generated_at is None or self.valid_until is None
        ):
            raise ValueError("timing_status requiere generated_at y valid_until")
        return self
