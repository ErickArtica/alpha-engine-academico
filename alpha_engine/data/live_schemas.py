"""Persisted technical runs and observed LIVE outcomes, separate from TEST."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

RunStatus = Literal[
    "SUCCESS", "NO_NEW_CANDLE", "PREDICTION_ALREADY_EXISTS", "INSUFFICIENT_HISTORY", "FAILED",
]


class PipelineRunRecord(BaseModel):
    run_id: UUID
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    market_data_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    prediction_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    evaluation_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    status: RunStatus
    symbol: str
    interval: str
    candle_open_time: datetime | None = None
    prediction_id: str | None = None
    new_candle: bool = False
    new_prediction: bool = False
    candles_ingested: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    evaluation_error_type: str | None = None
    evaluation_error_message: str | None = None
    created_at: datetime | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "PipelineRunRecord":
        dates = (self.started_at, self.finished_at, self.candle_open_time, self.created_at)
        if any(value is not None and value.tzinfo is None for value in dates):
            raise ValueError("Los timestamps de pipeline_run deben incluir zona horaria")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at debe ser posterior a started_at")
        if self.status == "FAILED" and not self.error_type:
            raise ValueError("FAILED requiere error_type")
        if self.status != "FAILED" and (self.error_type or self.error_message):
            raise ValueError("Solo FAILED puede contener error")
        return self


class LiveEvaluationRecord(BaseModel):
    evaluation_id: UUID
    prediction_id: str
    symbol: str
    evaluated_at: datetime
    source_close: float = Field(gt=0, allow_inf_nan=False)
    target_open_time: datetime
    target_close_time: datetime
    target_close: float = Field(gt=0, allow_inf_nan=False)
    observed_move: Literal["UP", "DOWN", "FLAT"]
    observed_class: Literal["UP", "DOWN"]
    correct: bool

    @model_validator(mode="after")
    def validate_record(self) -> "LiveEvaluationRecord":
        if any(value.tzinfo is None for value in (
            self.evaluated_at, self.target_open_time, self.target_close_time
        )):
            raise ValueError("Los timestamps de evaluacion deben incluir zona horaria")
        if self.target_close_time <= self.target_open_time:
            raise ValueError("La vela objetivo debe cerrarse despues de abrir")
        if self.evaluated_at <= self.target_close_time:
            raise ValueError("La evaluacion requiere una vela objetivo cerrada")
        expected_move = (
            "UP" if self.target_close > self.source_close else
            "DOWN" if self.target_close < self.source_close else "FLAT"
        )
        if self.observed_move != expected_move:
            raise ValueError("observed_move no coincide con los cierres")
        if self.observed_class != ("UP" if expected_move == "UP" else "DOWN"):
            raise ValueError("observed_class no coincide con el target binario")
        return self


class PendingLiveEvaluation(BaseModel):
    """One unevaluated prediction with its persisted source and closed H1 candle."""

    prediction_id: str
    symbol: str
    predicted_class: Literal["UP", "DOWN"]
    source_close: float = Field(gt=0, allow_inf_nan=False)
    target_open_time: datetime
    target_close_time: datetime
    target_close: float = Field(gt=0, allow_inf_nan=False)
