# Frozen public DTO contract from 63983a1ebff5b7f8f095c472762cd081e36d0e5f.
# Source: alpha_engine/data/{news_context,recommendation_response}.py.
# No payloads, provider responses, credentials or production data.
"""Sanitized, optional news snapshot for one prescriptive request."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class NewsContext(BaseModel):
    provider: Literal["ALPHA_VANTAGE"] = "ALPHA_VANTAGE"
    role: Literal["CONTEXT_ONLY"] = "CONTEXT_ONLY"
    status: Literal["AVAILABLE", "NO_NEWS", "STALE", "UNAVAILABLE"]
    snapshot_id: UUID | None = None
    window_minutes: int = 60
    fetched_at: datetime | None = None
    latest_news_at: datetime | None = None
    news_count: int = Field(default=0, ge=0)
    btc_sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    average_relevance: float | None = Field(default=None, gt=0, le=1)
    age_minutes: float | None = Field(default=None, ge=0)
    source_count: int = Field(default=0, ge=0)
    reason: Literal["NOT_CONFIGURED", "STORAGE_UNAVAILABLE", "PROVIDER_UNAVAILABLE",
                    "RATE_LIMIT", "INVALID_RESPONSE", "TOO_OLD"] | None = None

    @property
    def available(self) -> bool:
        return self.status in {"AVAILABLE", "NO_NEWS"}

    def for_prompt(self) -> dict:
        """One compact snapshot; no provider raw response or article content."""
        return {
            "role": self.role,
            "provider": self.provider,
            "status": self.status,
            "snapshot_id": str(self.snapshot_id) if self.snapshot_id else None,
            "window_minutes": self.window_minutes,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "latest_news_at": self.latest_news_at.isoformat() if self.latest_news_at else None,
            "news_count": self.news_count,
            "btc_sentiment_score": self.btc_sentiment_score,
            "average_relevance": self.average_relevance,
            "age_minutes": self.age_minutes,
            "source_count": self.source_count,
            "limitation": (
                "News sentiment is contextual information only; it did not show robust "
                "incremental predictive value in the historical experiment. Absence of "
                "recent news is not confirmed neutral sentiment."
            ),
        }

"""Public, validated result of a prescriptive request."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


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
