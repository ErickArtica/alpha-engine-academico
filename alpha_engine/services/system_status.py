"""Provider-neutral, factual LIVE status for later frontend use."""

from dataclasses import dataclass

from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.repositories.protocols import (
    LiveEvaluationRepository, MarketRepository, PipelineRunRepository, PredictionRepository,
)


@dataclass(frozen=True)
class SystemStatus:
    last_run: PipelineRunRecord | None
    last_successful_run: PipelineRunRecord | None
    last_candle: Candle | None
    last_prediction: PredictionRecord | None
    last_evaluation: LiveEvaluationRecord | None
    recent_runs: list[PipelineRunRecord]
    recent_errors: list[PipelineRunRecord]
    recent_evaluation_errors: list[PipelineRunRecord]


class SystemStatusService:
    def __init__(
        self, market: MarketRepository, predictions: PredictionRepository,
        runs: PipelineRunRepository, evaluations: LiveEvaluationRepository,
    ):
        self._market = market
        self._predictions = predictions
        self._runs = runs
        self._evaluations = evaluations

    def get(self, symbol: str, interval: str = "1m", limit: int = 20) -> SystemStatus:
        if not 1 <= limit <= 100:
            raise ValueError("limit debe estar entre 1 y 100")
        recent_runs = self._runs.list_recent(symbol, limit=limit)
        successful_statuses = (
            "SUCCESS", "NO_NEW_CANDLE", "PREDICTION_ALREADY_EXISTS", "INSUFFICIENT_HISTORY"
        )
        last_successful = next(
            (run for run in recent_runs if run.status in successful_statuses), None
        )
        if last_successful is None and recent_runs:
            # Preserve the historical last success when all recent runs failed.
            successful = [
                self._runs.get_latest(symbol, status=status) for status in successful_statuses
            ]
            last_successful = max(
                (run for run in successful if run is not None),
                key=lambda run: run.started_at, default=None,
            )
        return SystemStatus(
            last_run=recent_runs[0] if recent_runs else None,
            last_successful_run=last_successful,
            last_candle=self._market.get_latest_candle(symbol, interval),
            last_prediction=self._predictions.get_latest(symbol),
            last_evaluation=self._evaluations.get_latest(symbol),
            recent_runs=recent_runs,
            recent_errors=self._runs.list_recent(symbol, limit=limit, status="FAILED"),
            recent_evaluation_errors=[
                run for run in recent_runs if run.evaluation_error_type is not None
            ],
        )
