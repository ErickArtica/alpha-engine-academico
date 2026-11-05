"""Contratos que permiten cambiar InfluxDB por otra base de datos."""

from datetime import datetime
from typing import Protocol, Sequence

from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import (
    LiveEvaluationRecord, PendingLiveEvaluation, PipelineRunRecord, RunStatus,
)
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord, RecommendationRequestStatus


class MarketRepository(Protocol):
    def close(self) -> None:
        """Libera conexiones del adaptador."""

    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        """Persiste velas; repetir la misma serie y marca de tiempo es idempotente."""

    def get_candles(
        self, symbol: str, interval: str, start: datetime, stop: datetime
    ) -> list[Candle]:
        """Devuelve velas en [start, stop), ordenadas por open_time."""

    def get_latest_candle(self, symbol: str, interval: str) -> Candle | None:
        """Devuelve la última vela persistida."""


class PredictionRepository(Protocol):
    def close(self) -> None: ...

    def save(self, prediction: PredictionRecord) -> bool | None:
        """True if inserted, False if an identical ID existed; None if unknown."""

    def get_by_id(self, prediction_id: str) -> PredictionRecord | None: ...

    def get_latest(self, symbol: str) -> PredictionRecord | None: ...

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[PredictionRecord]: ...

    def list_recent_before(self, symbol: str, before: datetime, limit: int = 100) -> list[PredictionRecord]: ...

    def list_entry_signals(self, symbol: str, model_version: str,
                           minimum_up: float, limit: int = 100) -> list[PredictionRecord]: ...

    def list_evaluable_pending(
        self, symbol: str, now: datetime, limit: int = 100
    ) -> list[PendingLiveEvaluation]: ...


class PipelineRunRepository(Protocol):
    def close(self) -> None: ...

    def save(self, run: PipelineRunRecord) -> None: ...

    def get_latest(self, symbol: str, status: RunStatus | None = None) -> PipelineRunRecord | None: ...

    def list_recent(
        self, symbol: str, limit: int = 20, status: RunStatus | None = None
    ) -> list[PipelineRunRecord]: ...


class LiveEvaluationRepository(Protocol):
    def close(self) -> None: ...

    def save(self, evaluation: LiveEvaluationRecord) -> bool | None: ...

    def save_batch(self, evaluations: Sequence[LiveEvaluationRecord]) -> set[str]: ...

    def get_by_prediction(self, prediction_id: str) -> LiveEvaluationRecord | None: ...

    def get_latest(self, symbol: str) -> LiveEvaluationRecord | None: ...

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[LiveEvaluationRecord]: ...


class RecommendationRepository(Protocol):
    def close(self) -> None: ...

    def save(self, recommendation: RecommendationRecord) -> None: ...

    def save_if_absent(self, recommendation: RecommendationRecord) -> bool | None: ...

    def get_cached(
        self, prediction_id: str, prompt_version: str, llm_model: str
    ) -> RecommendationRecord | None: ...

    def list_recent(self, limit: int = 20) -> list[RecommendationRecord]: ...


class RecommendationRequestRepository(Protocol):
    def close(self) -> None: ...

    def save(self, request: RecommendationRequestRecord) -> None: ...

    def get_latest(
        self, symbol: str, status: RecommendationRequestStatus | None = None
    ) -> RecommendationRequestRecord | None: ...

    def list_recent(
        self, symbol: str, limit: int = 20, status: RecommendationRequestStatus | None = None
    ) -> list[RecommendationRequestRecord]: ...
