from datetime import datetime, timedelta, timezone
from math import sin
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.ml.trainer import save_training_outcome, train_from_candles
from alpha_engine.services.prediction import (
    ModelNotTrainedError,
    PredictionResult,
    PredictionService,
    PredictionUnavailableError,
    load_model_bundle,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candles(count=120):
    result = []
    for i in range(count):
        open_time = START + timedelta(minutes=i)
        price = 100 + 0.1 * i + sin(i / 4)
        result.append(Candle(
            symbol="BTCUSDT", interval="1m", open_time=open_time,
            close_time=open_time + timedelta(milliseconds=59_999),
            open=price, high=price + 1, low=price - 1,
            close=price, volume=10 + i % 3,
        ))
    return result


class FakeMarket:
    def __init__(self, records):
        self.records = records
        self.read_calls = 0

    def get_latest_candle(self, symbol, interval):
        return self.records[-1] if self.records else None

    def get_candles(self, symbol, interval, start, stop):
        self.read_calls += 1
        return [c for c in self.records if start <= c.open_time < stop]

    def close(self):
        pass


class FakePredictions:
    def __init__(self):
        self.records = {}
        self.saves = 0

    def get_by_id(self, prediction_id):
        return self.records.get(prediction_id)

    def get_latest(self, symbol):
        values = [r for r in self.records.values() if r.symbol == symbol]
        return max(values, key=lambda r: r.timestamp, default=None)

    def save(self, prediction):
        self.records[prediction.prediction_id] = prediction
        self.saves += 1

    def close(self):
        pass


class FakeEstimator:
    classes_ = np.array([0, 1])

    def __init__(self, values=(0.3, 0.7)):
        self.values = values
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += 1
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return np.array([self.values])


def bundle(estimator, trained_at="2026-01-01T00:00:00+00:00"):
    return {
        "estimator": estimator,
        "features": list(FEATURE_COLUMNS),
        "model_version": "xgboost_v1",
        "trained_at": trained_at,
        "symbol": "BTCUSDT",
        "interval": "1m",
        "horizon": "1m",
    }


def test_prediction_persists_and_reuses_same_model_and_candle(tmp_path):
    market = FakeMarket(candles())
    predictions = FakePredictions()
    estimator = FakeEstimator()
    service = PredictionService(
        market, predictions, tmp_path / "unused.joblib", loader=lambda path: bundle(estimator)
    )
    first = service.predict_latest("BTCUSDT")
    second = service.predict_latest("BTCUSDT")
    assert first == second
    assert first.timestamp == START + timedelta(minutes=119)
    assert first.prob_up == pytest.approx(0.7)
    assert first.prob_down == pytest.approx(0.3)
    assert first.predicted_class == "UP"
    assert predictions.saves == 1
    assert estimator.calls == 1
    assert market.read_calls == 1


def test_live_generation_time_is_injected_after_inference_and_expiry_uses_source_close(tmp_path):
    records = candles()
    generated = records[-1].close_time + timedelta(seconds=5)
    estimator = FakeEstimator()
    service = PredictionService(
        FakeMarket(records), FakePredictions(), tmp_path / "unused.joblib",
        loader=lambda path: bundle(estimator),
    )
    execution = service.predict_candle_execution(
        records[-1], "BTCUSDT", now=generated,
        generated_at_clock=lambda: generated,
    )
    prediction = execution.prediction
    assert execution.created and estimator.calls == 1
    assert prediction.timestamp == records[-1].open_time
    assert prediction.generated_at == generated
    assert prediction.valid_until == records[-1].close_time + timedelta(minutes=1)
    assert prediction.timing_status == "TIMELY"
    delayed = generated + timedelta(minutes=2)
    late = PredictionService(
        FakeMarket(records), FakePredictions(), tmp_path / "unused.joblib",
        loader=lambda path: bundle(FakeEstimator()),
    ).predict_candle_execution(
        records[-1], "BTCUSDT", now=delayed,
        generated_at_clock=lambda: delayed,
    ).prediction
    assert late.valid_until == prediction.valid_until
    assert late.timing_status == "LATE"


def test_retraining_changes_prediction_identity(tmp_path):
    market = FakeMarket(candles())
    predictions = FakePredictions()
    first = PredictionService(
        market, predictions, tmp_path / "model.joblib",
        loader=lambda path: bundle(FakeEstimator(), "2026-01-01T00:00:00+00:00"),
    ).predict_latest("BTCUSDT")
    second = PredictionService(
        market, predictions, tmp_path / "model.joblib",
        loader=lambda path: bundle(FakeEstimator(), "2026-01-02T00:00:00+00:00"),
    ).predict_latest("BTCUSDT")
    assert first.prediction_id != second.prediction_id
    assert predictions.saves == 2


def test_prediction_requires_history_and_matching_model(tmp_path):
    repo = FakePredictions()
    short = PredictionService(
        FakeMarket(candles(20)), repo, tmp_path / "model.joblib",
        loader=lambda path: bundle(FakeEstimator()),
    )
    with pytest.raises(PredictionUnavailableError, match="historia suficiente"):
        short.predict_latest("BTCUSDT")
    mismatched = PredictionService(
        FakeMarket(candles()), repo, tmp_path / "model.joblib",
        loader=lambda path: {**bundle(FakeEstimator()), "symbol": "ETHUSDT"},
    )
    with pytest.raises(PredictionUnavailableError, match="otra serie"):
        mismatched.predict_latest("BTCUSDT")
    assert repo.saves == 0


def test_prediction_rejects_current_open_candle(tmp_path):
    records = candles()
    service = PredictionService(
        FakeMarket(records), FakePredictions(), tmp_path / "model.joblib",
        loader=lambda path: bundle(FakeEstimator()),
    )
    now = records[-1].open_time + timedelta(seconds=30)
    with pytest.raises(PredictionUnavailableError, match="aún no está cerrada"):
        service.predict_latest("BTCUSDT", now=now)


def test_missing_model_and_invalid_probability_are_rejected(tmp_path):
    with pytest.raises(ModelNotTrainedError, match="pendiente"):
        load_model_bundle(tmp_path / "missing.joblib")
    service = PredictionService(
        FakeMarket(candles()), FakePredictions(), tmp_path / "model.joblib",
        loader=lambda path: bundle(FakeEstimator((float("nan"), 0.7))),
    )
    with pytest.raises(PredictionUnavailableError, match="probabilidades inválidas"):
        service.predict_latest("BTCUSDT")
    with pytest.raises(ValidationError, match="probabilidades"):
        PredictionResult(
            prediction_id="x", symbol="BTCUSDT", timestamp=START, horizon="1m",
            prob_up=0.8, prob_down=0.8, predicted_class="UP", model_version="v1",
        )


def test_real_xgboost_bundle_predicts_and_persists_with_fake_repositories(tmp_path):
    records = candles(300)
    outcome = train_from_candles(records, trained_at=START)
    paths = save_training_outcome(outcome, tmp_path)
    predictions = FakePredictions()
    service = PredictionService(FakeMarket(records), predictions, paths["model"])
    result = service.predict_latest("BTCUSDT")
    assert result.model_version == "xgboost_v1"
    assert result.prob_up + result.prob_down == pytest.approx(1.0)
    assert predictions.get_by_id(result.prediction_id) == result
