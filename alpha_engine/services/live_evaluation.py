"""Evaluate LIVE model predictions only after their next candle has closed."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from time import perf_counter
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from alpha_engine.data.live_schemas import LiveEvaluationRecord
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.repositories.protocols import (
    LiveEvaluationRepository, MarketRepository, PredictionRepository,
)

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class EvaluationBatchResult:
    created: int
    already_evaluated: int
    pending: int
    legacy_skipped: int = 0


@dataclass(frozen=True)
class LiveMetrics:
    evaluated: int
    correct: int
    incorrect: int
    accuracy: float | None
    timely: int
    late: int
    unknown_timing: int
    source_start: datetime | None = None
    source_end: datetime | None = None
    scored_predictions: int = 0
    missing_predictions: int = 0
    inconsistent_correct_rows: int = 0
    pred_down_actual_down: int = 0
    pred_down_actual_up: int = 0
    pred_up_actual_down: int = 0
    pred_up_actual_up: int = 0
    balanced_accuracy: float | None = None
    precision_up: float | None = None
    recall_up: float | None = None
    f1_up: float | None = None
    precision_down: float | None = None
    recall_down: float | None = None
    f1_down: float | None = None
    macro_f1: float | None = None
    roc_auc_up: float | None = None
    always_down_accuracy: float | None = None
    predicted_up_fraction: float | None = None
    predicted_down_fraction: float | None = None
    actual_up_fraction: float | None = None
    actual_down_fraction: float | None = None


def _class_scores(true_positive: int, predicted: int, actual: int) -> tuple[float | None, float | None, float | None]:
    precision = true_positive / predicted if predicted else None
    recall = true_positive / actual if actual else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _roc_auc_up(rows: list[tuple[float, bool]]) -> float | None:
    """Mann-Whitney AUC with average ranks for tied probabilities."""
    positive = sum(is_up for _, is_up in rows)
    negative = len(rows) - positive
    if not positive or not negative:
        return None
    ordered = sorted(rows, key=lambda item: item[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(is_up for _, is_up in ordered[start:end])
        start = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


class LiveEvaluationService:
    def __init__(
        self, market: MarketRepository, predictions: PredictionRepository,
        evaluations: LiveEvaluationRepository,
    ):
        self._market = market
        self._predictions = predictions
        self._evaluations = evaluations

    def _closed_pair(
        self, prediction: PredictionRecord, instant: datetime
    ) -> tuple[Candle, Candle] | None:
        target_open = prediction.timestamp + timedelta(minutes=1)
        candles = self._market.get_candles(
            prediction.symbol, prediction.interval, prediction.timestamp,
            target_open + timedelta(minutes=1),
        )
        by_open = {
            candle.open_time: candle for candle in candles
            if candle.symbol == prediction.symbol and candle.interval == "1m"
        }
        source = by_open.get(prediction.timestamp)
        target = by_open.get(target_open)
        if source is None or target is None or target.close_time >= instant:
            return None
        return source, target

    def readiness(
        self, prediction: PredictionRecord, now: datetime
    ) -> Literal["READY", "PENDING", "EVALUATED", "LEGACY"]:
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if prediction.generated_at is None:
            return "LEGACY"
        if self._evaluations.get_by_prediction(prediction.prediction_id) is not None:
            return "EVALUATED"
        return "READY" if self._closed_pair(prediction, now.astimezone(timezone.utc)) else "PENDING"

    def evaluate_pending(
        self, symbol: str, now: datetime, limit: int = 100
    ) -> EvaluationBatchResult:
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        instant = now.astimezone(timezone.utc)
        if not 1 <= limit <= 1000:
            raise ValueError("limit debe estar entre 1 y 1000")
        if (hasattr(self._predictions, "list_evaluable_pending")
                and hasattr(self._evaluations, "save_batch")):
            return self._evaluate_ready_batch(symbol, instant, limit)
        created = already = pending = legacy_skipped = offset = 0
        while created < limit:
            page = self._predictions.list_recent(symbol, limit=100, offset=offset)
            if not page:
                break
            offset += len(page)
            for prediction in page:
                if prediction.generated_at is None:
                    legacy_skipped += 1
                    # No historical migration: unknown-timing rows are not LIVE.
                    continue
                if self._evaluations.get_by_prediction(prediction.prediction_id) is not None:
                    already += 1
                    continue
                # The model predicts the binary close direction of the immediately
                # following 1m candle. No current quote or still-open candle is used.
                pair = self._closed_pair(prediction, instant)
                if pair is None:
                    pending += 1
                    continue
                source, target = pair
                move = "UP" if target.close > source.close else "DOWN" if target.close < source.close else "FLAT"
                observed_class = "UP" if move == "UP" else "DOWN"  # TRAIN/TEST tie rule.
                record = LiveEvaluationRecord(
                    evaluation_id=uuid5(NAMESPACE_URL, f"alpha-engine-live:{prediction.prediction_id}"),
                    prediction_id=prediction.prediction_id, symbol=symbol,
                    evaluated_at=instant, source_close=source.close,
                    target_open_time=target.open_time, target_close_time=target.close_time,
                    target_close=target.close, observed_move=move,
                    observed_class=observed_class,
                    correct=prediction.predicted_class == observed_class,
                )
                inserted = self._evaluations.save(record)
                if inserted is False:
                    already += 1
                else:
                    created += 1
                if created >= limit:
                    break
            if len(page) < 100:
                break
        return EvaluationBatchResult(created, already, pending, legacy_skipped)

    def _evaluate_ready_batch(
        self, symbol: str, instant: datetime, limit: int
    ) -> EvaluationBatchResult:
        started = perf_counter()
        reads = writes = 0
        candidates = []
        try:
            reads += 1
            candidates = self._predictions.list_evaluable_pending(symbol, instant, limit)
            rows = []
            for candidate in candidates:
                move = (
                    "UP" if candidate.target_close > candidate.source_close else
                    "DOWN" if candidate.target_close < candidate.source_close else "FLAT"
                )
                observed_class = "UP" if move == "UP" else "DOWN"
                rows.append(LiveEvaluationRecord(
                    evaluation_id=uuid5(NAMESPACE_URL, f"alpha-engine-live:{candidate.prediction_id}"),
                    prediction_id=candidate.prediction_id, symbol=candidate.symbol,
                    evaluated_at=instant, source_close=candidate.source_close,
                    target_open_time=candidate.target_open_time,
                    target_close_time=candidate.target_close_time,
                    target_close=candidate.target_close, observed_move=move,
                    observed_class=observed_class,
                    correct=candidate.predicted_class == observed_class,
                ))
            inserted = set()
            if rows:
                writes += (len(rows) + 499) // 500
                inserted = self._evaluations.save_batch(rows)
            return EvaluationBatchResult(len(inserted), len(rows) - len(inserted), 0)
        finally:
            logger.info(
                "LIVE_EVALUATION_TIMING pending=%d predictions_read=%d db_reads=%d "
                "db_writes=%d total_ms=%.1f",
                len(candidates), len(candidates), reads, writes,
                (perf_counter() - started) * 1000,
            )

    def metrics(self, symbol: str) -> LiveMetrics:
        """Aggregate all persisted LIVE evaluations, never TEST."""
        return self.metrics_for(self._predictions, self._evaluations, symbol)

    @staticmethod
    def metrics_for(
        predictions: PredictionRepository, evaluations_repository: LiveEvaluationRepository,
        symbol: str, model_version: str | None = None,
    ) -> LiveMetrics:
        """Read one paged evaluation/prediction cohort for all public statistics."""
        evaluations: list[LiveEvaluationRecord] = []
        offset = 0
        seen: set[str] = set()
        while True:
            page = evaluations_repository.list_recent(symbol, limit=500, offset=offset)
            for record in page:
                if record.prediction_id in seen:
                    continue
                seen.add(record.prediction_id)
                evaluations.append(record)
            if len(page) < 500:
                break
            offset += len(page)

        # Fetch prediction timing in pages. A per-evaluation lookup makes the
        # public dashboard issue one HTTP request for every LIVE evaluation.
        pending = set(seen)
        prediction_by_id: dict[str, PredictionRecord] = {}
        offset = 0
        while pending:
            page = predictions.list_recent(symbol, limit=500, offset=offset)
            for prediction in page:
                if prediction.prediction_id in pending:
                    prediction_by_id[prediction.prediction_id] = prediction
                    pending.remove(prediction.prediction_id)
            if len(page) < 500:
                break
            offset += len(page)

        if model_version is not None:
            evaluations = [row for row in evaluations
                           if (matched_prediction := prediction_by_id.get(row.prediction_id)) is not None
                           and matched_prediction.model_version == model_version]

        evaluated = len(evaluations)
        correct = sum(record.correct for record in evaluations)
        timely = sum(getattr(prediction_by_id.get(record.prediction_id), "timing_status", None) == "TIMELY" for record in evaluations)
        late = sum(getattr(prediction_by_id.get(record.prediction_id), "timing_status", None) == "LATE" for record in evaluations)
        unknown = evaluated - timely - late
        matched = [(prediction_by_id[row.prediction_id], row) for row in evaluations if row.prediction_id in prediction_by_id]
        confusion = {(predicted, actual): 0 for predicted in ("DOWN", "UP") for actual in ("DOWN", "UP")}
        for prediction, row in matched:
            confusion[(prediction.predicted_class, row.observed_class)] += 1
        dd, du, ud, uu = (confusion[("DOWN", "DOWN")], confusion[("DOWN", "UP")],
                          confusion[("UP", "DOWN")], confusion[("UP", "UP")])
        scored = len(matched)
        precision_up, recall_up, f1_up = _class_scores(uu, ud + uu, du + uu)
        precision_down, recall_down, f1_down = _class_scores(dd, dd + du, dd + ud)
        source_times = [row.target_open_time - timedelta(minutes=1) for row in evaluations]
        return LiveMetrics(
            evaluated=evaluated, correct=correct, incorrect=evaluated - correct,
            accuracy=correct / evaluated if evaluated else None,
            timely=timely, late=late, unknown_timing=unknown,
            source_start=min(source_times) if source_times else None,
            source_end=max(source_times) if source_times else None,
            scored_predictions=scored, missing_predictions=evaluated - scored,
            inconsistent_correct_rows=sum(row.correct != (prediction.predicted_class == row.observed_class) for prediction, row in matched),
            pred_down_actual_down=dd, pred_down_actual_up=du,
            pred_up_actual_down=ud, pred_up_actual_up=uu,
            balanced_accuracy=(recall_up + recall_down) / 2 if recall_up is not None and recall_down is not None else None,
            precision_up=precision_up, recall_up=recall_up, f1_up=f1_up,
            precision_down=precision_down, recall_down=recall_down, f1_down=f1_down,
            macro_f1=(f1_up + f1_down) / 2 if f1_up is not None and f1_down is not None else None,
            roc_auc_up=_roc_auc_up([(prediction.prob_up, row.observed_class == "UP") for prediction, row in matched]),
            always_down_accuracy=(dd + ud) / scored if scored else None,
            predicted_up_fraction=(ud + uu) / scored if scored else None,
            predicted_down_fraction=(dd + du) / scored if scored else None,
            actual_up_fraction=(du + uu) / scored if scored else None,
            actual_down_fraction=(dd + ud) / scored if scored else None,
        )
