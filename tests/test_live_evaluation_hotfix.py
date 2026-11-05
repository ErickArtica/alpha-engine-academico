"""Call-count and outcome regressions for the pending-only LIVE path."""

from datetime import datetime, timedelta, timezone

from alpha_engine.data.live_schemas import PendingLiveEvaluation
from alpha_engine.services.live_evaluation import LiveEvaluationService


SOURCE = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
TARGET = SOURCE + timedelta(minutes=1)
CLOSE = TARGET + timedelta(minutes=1) - timedelta(milliseconds=1)
NOW = CLOSE + timedelta(milliseconds=1)


def candidate(number: int, target_close: float = 101) -> PendingLiveEvaluation:
    return PendingLiveEvaluation(
        prediction_id=f"p-{number}", symbol="BTCUSDT", predicted_class="UP",
        source_close=100, target_open_time=TARGET, target_close_time=CLOSE,
        target_close=target_close,
    )


class PendingOnlyPredictions:
    def __init__(self, candidates, evaluated):
        self.candidates = candidates
        self.evaluated = evaluated
        self.calls = []

    def list_evaluable_pending(self, symbol, now, limit):
        self.calls.append((symbol, now, limit))
        return [item for item in self.candidates
                if item.prediction_id not in self.evaluated
                and item.target_close_time < now][:limit]

    def list_recent(self, *args, **kwargs):
        raise AssertionError("fast path must not page historical predictions")


class BatchEvaluations:
    def __init__(self, evaluated=()):
        self.records = {key: object() for key in evaluated}
        self.calls = []

    def save_batch(self, rows):
        self.calls.append(rows)
        inserted = set()
        for row in rows:
            if row.prediction_id not in self.records:
                self.records[row.prediction_id] = row
                inserted.add(row.prediction_id)
        return inserted

    def get_by_prediction(self, *args):
        raise AssertionError("fast path must not make per-prediction SELECTs")

    def save(self, *args):
        raise AssertionError("fast path must not write one row at a time")


def test_500_historical_499_evaluated_uses_one_read_one_batch_write(caplog):
    existing = {f"p-{i}" for i in range(499)}
    evaluations = BatchEvaluations(existing)
    predictions = PendingOnlyPredictions([candidate(i) for i in range(500)], evaluations.records)
    service = LiveEvaluationService(None, predictions, evaluations)

    with caplog.at_level("INFO"):
        first = service.evaluate_pending("BTCUSDT", NOW)
        second = service.evaluate_pending("BTCUSDT", NOW)

    assert first.created == 1 and second.created == 0
    assert len(predictions.calls) == 2
    assert len(evaluations.calls) == 1 and len(evaluations.calls[0]) == 1
    assert len(evaluations.records) == 500
    assert "LIVE_EVALUATION_TIMING pending=1 predictions_read=1 db_reads=1 db_writes=1" in caplog.text
    assert "LIVE_EVALUATION_TIMING pending=0 predictions_read=0 db_reads=1 db_writes=0" in caplog.text


def test_500_pending_are_written_in_one_batch_with_same_h1_outcome():
    evaluations = BatchEvaluations()
    predictions = PendingOnlyPredictions([candidate(i, 99 if i % 2 else 100)
                                          for i in range(500)], evaluations.records)
    service = LiveEvaluationService(None, predictions, evaluations)

    assert service.evaluate_pending("BTCUSDT", CLOSE).created == 0
    assert service.evaluate_pending("BTCUSDT", NOW, limit=500).created == 500
    assert service.evaluate_pending("BTCUSDT", NOW, limit=500).created == 0
    assert len(predictions.calls) == 3
    assert len(evaluations.calls) == 1 and len(evaluations.calls[0]) == 500
    assert len(evaluations.records) == 500
    flat = evaluations.records["p-0"]
    down = evaluations.records["p-1"]
    assert (flat.target_open_time, flat.target_close_time, flat.observed_move,
            flat.observed_class, flat.correct) == (TARGET, CLOSE, "FLAT", "DOWN", False)
    assert (down.observed_move, down.observed_class, down.correct) == ("DOWN", "DOWN", False)


def test_fixed_batch_limit_drains_backlog_without_skipping_old_rows():
    evaluations = BatchEvaluations()
    predictions = PendingOnlyPredictions([candidate(i) for i in range(500)], evaluations.records)
    service = LiveEvaluationService(None, predictions, evaluations)

    assert [service.evaluate_pending("BTCUSDT", NOW, limit=100).created
            for _ in range(6)] == [100, 100, 100, 100, 100, 0]
    assert len(evaluations.records) == 500
    assert len(evaluations.calls) == 5
