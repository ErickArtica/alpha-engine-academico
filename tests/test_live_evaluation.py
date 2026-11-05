from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.services.live_evaluation import LiveEvaluationService

ROOT = Path(__file__).resolve().parents[1]
OPEN = datetime(2026, 1, 1, 10, 4, tzinfo=timezone.utc)
SOURCE_CLOSE = OPEN + timedelta(minutes=1) - timedelta(milliseconds=1)
TARGET_OPEN = OPEN + timedelta(minutes=1)
TARGET_CLOSE = TARGET_OPEN + timedelta(minutes=1) - timedelta(milliseconds=1)


def candle(opened, close):
    return Candle(symbol="BTCUSDT", interval="1m", open_time=opened,
                  close_time=opened + timedelta(minutes=1) - timedelta(milliseconds=1),
                  open=100, high=max(100, close), low=min(100, close),
                  close=close, volume=1)


def prediction(direction="UP", generated_at=None):
    return PredictionRecord(
        prediction_id="p-" + direction, symbol="BTCUSDT", timestamp=OPEN,
        horizon="1m", prob_up=0.7 if direction == "UP" else 0.3,
        prob_down=0.3 if direction == "UP" else 0.7,
        predicted_class=direction, model_version="xgboost_v1",
        generated_at=generated_at or SOURCE_CLOSE + timedelta(seconds=5),
        valid_until=TARGET_CLOSE,
    )


class Market:
    def __init__(self, candles=()):
        self.candles = list(candles)
        self.calls = []

    def get_candles(self, symbol, interval, start, stop):
        self.calls.append((symbol, interval, start, stop))
        return [row for row in self.candles if row.symbol == symbol and row.interval == interval
                and start <= row.open_time < stop]


class Predictions:
    def __init__(self, records=()):
        self.records = {row.prediction_id: row for row in records}

    def list_recent(self, symbol, limit=100, offset=0):
        return list(self.records.values())[offset:offset + limit]

    def get_by_id(self, prediction_id):
        return self.records.get(prediction_id)


class Evaluations:
    def __init__(self):
        self.records = {}
        self.saves = 0

    def get_by_prediction(self, prediction_id):
        return self.records.get(prediction_id)

    def save(self, record):
        self.saves += 1
        if record.prediction_id in self.records:
            return False
        self.records[record.prediction_id] = record
        return True

    def list_recent(self, symbol, limit=100, offset=0):
        return list(self.records.values())[offset:offset + limit]


def service(direction="UP", target_close=101):
    market = Market([candle(OPEN, 100), candle(TARGET_OPEN, target_close)])
    predictions = Predictions([prediction(direction)])
    evaluations = Evaluations()
    return LiveEvaluationService(market, predictions, evaluations), market, evaluations


def test_target_candle_open_or_at_close_is_not_evaluable():
    evaluator, market, evaluations = service()
    for now in (TARGET_OPEN + timedelta(seconds=20), TARGET_CLOSE):
        result = evaluator.evaluate_pending("BTCUSDT", now)
        assert result.created == 0 and result.pending == 1
    assert evaluations.saves == 0
    assert market.calls[-1] == ("BTCUSDT", "1m", OPEN, OPEN + timedelta(minutes=2))


def test_readiness_reports_pending_ready_and_evaluated_without_current_quote():
    evaluator, _, _ = service()
    model = evaluator._predictions.records["p-UP"]
    assert evaluator.readiness(model, TARGET_CLOSE) == "PENDING"
    assert evaluator.readiness(model, TARGET_CLOSE + timedelta(milliseconds=1)) == "READY"
    evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(milliseconds=1))
    assert evaluator.readiness(model, TARGET_CLOSE + timedelta(seconds=1)) == "EVALUATED"


def test_closed_future_candle_is_evaluated_once_without_leakage():
    evaluator, market, evaluations = service()
    now = TARGET_CLOSE + timedelta(milliseconds=1)
    first = evaluator.evaluate_pending("BTCUSDT", now)
    second = evaluator.evaluate_pending("BTCUSDT", now)
    assert first.created == 1 and second.already_evaluated == 1
    assert evaluations.saves == 1
    row = evaluations.get_by_prediction("p-UP")
    assert row.target_open_time == TARGET_OPEN
    assert row.target_close_time == TARGET_CLOSE
    assert row.evaluated_at == now
    assert row.correct and row.observed_class == "UP" and row.observed_move == "UP"
    assert len(market.calls) == 1


@pytest.mark.parametrize("direction,target_price,observed,correct", [
    ("UP", 101, "UP", True), ("UP", 99, "DOWN", False),
    ("DOWN", 99, "DOWN", True), ("DOWN", 101, "UP", False),
    ("DOWN", 100, "FLAT", True), ("UP", 100, "FLAT", False),
])
def test_binary_target_matches_existing_history_audit(direction, target_price, observed, correct):
    evaluator, _, evaluations = service(direction, target_price)
    evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    row = evaluations.get_by_prediction("p-" + direction)
    assert row.observed_move == observed
    assert row.observed_class == ("UP" if observed == "UP" else "DOWN")
    assert row.correct is correct


def test_missing_future_or_source_data_stays_pending():
    evaluator, market, evaluations = service()
    market.candles.pop()
    result = evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    assert result.pending == 1 and evaluations.saves == 0
    market.candles = [candle(TARGET_OPEN, 101)]
    result = evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    assert result.pending == 1 and evaluations.saves == 0


def test_legacy_prediction_without_generation_time_is_not_called_live_evaluation():
    evaluator, _, evaluations = service()
    old = evaluator._predictions.records["p-UP"]
    evaluator._predictions.records["p-UP"] = PredictionRecord.model_validate({
        **old.model_dump(), "generated_at": None, "valid_until": None,
    })
    result = evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    assert result.legacy_skipped == 1 and result.created == 0
    assert evaluations.saves == 0


def test_pending_prediction_beyond_first_page_is_not_forgotten():
    class MostlyEvaluated(Evaluations):
        def get_by_prediction(self, prediction_id):
            return object() if prediction_id != "old-pending" else super().get_by_prediction(prediction_id)

    recent = [
        PredictionRecord.model_validate({
            **prediction().model_dump(), "prediction_id": f"recent-{i}",
            "timestamp": OPEN + timedelta(minutes=i + 1),
        })
        for i in range(100, 0, -1)
    ]
    old = PredictionRecord.model_validate({
        **prediction().model_dump(), "prediction_id": "old-pending",
    })
    predictions = Predictions([*recent, old])
    evaluations = MostlyEvaluated()
    market = Market([candle(OPEN, 100), candle(TARGET_OPEN, 101)])
    evaluator = LiveEvaluationService(market, predictions, evaluations)
    result = evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    assert result.already_evaluated == 100 and result.created == 1
    assert evaluations.get_by_prediction("old-pending") is not None


def test_live_metrics_have_no_fabricated_zero_percent():
    evaluator, _, _ = service()
    empty = evaluator.metrics("BTCUSDT")
    assert empty.evaluated == 0 and empty.accuracy is None
    evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    metrics = evaluator.metrics("BTCUSDT")
    assert (metrics.evaluated, metrics.correct, metrics.incorrect) == (1, 1, 0)
    assert metrics.accuracy == 1.0 and metrics.timely == 1 and metrics.late == 0


def test_live_metrics_include_rows_beyond_first_page_without_one_query_per_prediction(monkeypatch):
    evaluator, _, evaluations = service()
    evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    template = evaluations.get_by_prediction("p-UP")
    model = evaluator._predictions.records["p-UP"]
    for index in range(1, 501):
        prediction_id = f"p-{index}"
        evaluator._predictions.records[prediction_id] = model.model_copy(
            update={"prediction_id": prediction_id}
        )
        evaluations.records[prediction_id] = template.model_copy(update={
            "prediction_id": prediction_id, "evaluation_id": uuid4(),
            "evaluated_at": template.evaluated_at + timedelta(seconds=index),
        })
    def individual_lookup(_prediction_id):
        raise AssertionError("LIVE metrics must not look up every prediction separately")

    monkeypatch.setattr(evaluator._predictions, "get_by_id", individual_lookup)
    metrics = evaluator.metrics("BTCUSDT")
    assert metrics.evaluated == metrics.correct == metrics.timely == 501
    assert metrics.accuracy == 1.0


def test_live_evaluation_does_not_write_model_or_test_metadata():
    model = ROOT / "models" / "xgboost_v1.joblib"
    metadata = ROOT / "models" / "xgboost_v1_metadata.json"
    before = (model.read_bytes(), metadata.read_bytes())
    evaluator, _, _ = service()
    evaluator.evaluate_pending("BTCUSDT", TARGET_CLOSE + timedelta(seconds=1))
    assert (model.read_bytes(), metadata.read_bytes()) == before
