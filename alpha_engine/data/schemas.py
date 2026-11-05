"""Esquemas del dominio sin dependencias del proveedor de persistencia."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Candle(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9]{3,20}$")
    interval: str = Field(pattern=r"^[1-9][0-9]*[mhd]$")
    open_time: datetime
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: float = Field(ge=0, allow_inf_nan=False)
    close_time: datetime

    @model_validator(mode="after")
    def validate_candle(self) -> "Candle":
        if self.open_time.tzinfo is None or self.close_time.tzinfo is None:
            raise ValueError("Las marcas de tiempo deben incluir zona horaria")
        if self.close_time <= self.open_time:
            raise ValueError("close_time debe ser posterior a open_time")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high es inferior a otro precio de la vela")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low es superior a otro precio de la vela")
        return self


class PredictionRecord(BaseModel):
    prediction_id: str = Field(min_length=1)
    symbol: str = Field(pattern=r"^[A-Z0-9]{3,20}$")
    timestamp: datetime
    interval: Literal["1m"] = "1m"
    horizon: Literal["1m"]
    prob_up: float = Field(ge=0, le=1, allow_inf_nan=False)
    prob_down: float = Field(ge=0, le=1, allow_inf_nan=False)
    predicted_class: Literal["UP", "DOWN"]
    model_version: str
    trained_at: datetime | None = None
    generated_at: datetime | None = None
    valid_until: datetime | None = None

    @property
    def source_candle_open_time(self) -> datetime:
        """Historical `timestamp` always means source candle open time."""
        return self.timestamp

    @property
    def timing_status(self) -> Literal["TIMELY", "LATE"] | None:
        if self.generated_at is None or self.valid_until is None:
            return None
        return "TIMELY" if self.generated_at < self.valid_until else "LATE"

    def current_validity(self, now: datetime) -> Literal["ACTIVE", "EXPIRED"] | None:
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if self.valid_until is None:
            return None
        return "ACTIVE" if now < self.valid_until else "EXPIRED"

    @model_validator(mode="after")
    def validate_prediction(self) -> "PredictionRecord":
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp debe incluir zona horaria")
        if self.trained_at is not None and self.trained_at.tzinfo is None:
            raise ValueError("trained_at debe incluir zona horaria")
        if (self.generated_at is None) != (self.valid_until is None):
            raise ValueError("generated_at y valid_until deben existir juntos")
        if self.generated_at is not None and self.generated_at.tzinfo is None:
            raise ValueError("generated_at debe incluir zona horaria")
        if self.valid_until is not None and self.valid_until.tzinfo is None:
            raise ValueError("valid_until debe incluir zona horaria")
        if abs(self.prob_up + self.prob_down - 1.0) > 1e-6:
            raise ValueError("Las probabilidades deben sumar 1")
        expected = "UP" if self.prob_up > self.prob_down else "DOWN"
        if self.predicted_class != expected:
            raise ValueError("predicted_class no coincide con las probabilidades")
        return self


class RecommendationRecord(BaseModel):
    prediction_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    timestamp: datetime
    generated_at: datetime
    requested_at: datetime | None = None
    valid_until: datetime | None = None
    llm_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    total_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    candidate_signal: Literal["BUY", "SELL", "HOLD"]
    llm_signal: Literal["BUY", "SELL", "HOLD"] | None = None
    final_signal: Literal["BUY", "SELL", "HOLD"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    context_json: str
    response_json: str | None = None
    guardrails_json: str = "[]"

    @model_validator(mode="after")
    def validate_times(self) -> "RecommendationRecord":
        if self.timestamp.tzinfo is None or self.generated_at.tzinfo is None:
            raise ValueError("Las marcas de tiempo deben incluir zona horaria")
        if self.requested_at is not None and self.requested_at.tzinfo is None:
            raise ValueError("requested_at debe incluir zona horaria")
        if self.valid_until is not None and self.valid_until.tzinfo is None:
            raise ValueError("valid_until debe incluir zona horaria")
        if self.requested_at is not None and self.generated_at < self.requested_at:
            raise ValueError("generated_at no puede preceder requested_at")
        return self

    @property
    def timing_status(self) -> Literal["TIMELY", "LATE"] | None:
        if self.valid_until is None:
            return None
        return "TIMELY" if self.generated_at < self.valid_until else "LATE"

    def current_validity(self, now: datetime) -> Literal["ACTIVE", "EXPIRED"] | None:
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if self.valid_until is None:
            return None
        return "ACTIVE" if now < self.valid_until else "EXPIRED"
