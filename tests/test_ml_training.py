from datetime import datetime, timedelta, timezone
from math import sin
import json

import joblib
import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.ml.baseline import MajorityBaseline
from alpha_engine.ml.evaluation import evaluate_binary
from alpha_engine.ml.trainer import save_training_outcome, train_from_candles


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sample_candles(count=300, flat=False):
    result = []
    for i in range(count):
        price = 100.0 if flat else 100 + 0.03 * i + 2 * sin(i / 3) + 0.4 * sin(i / 11)
        open_time = START + timedelta(minutes=i)
        result.append(
            Candle(
                symbol="BTCUSDT", interval="1m", open_time=open_time,
                close_time=open_time + timedelta(milliseconds=59_999),
                open=price, high=price + 1, low=price - 1,
                close=price, volume=10 + i % 5,
            )
        )
    return result


def test_majority_baseline_learns_train_class_and_tie_rule():
    assert MajorityBaseline.fit([1, 1, 0]).majority_class == 1
    assert MajorityBaseline.fit([1, 0]).majority_class == 0
    assert MajorityBaseline.fit([1, 1, 0]).predict(3).tolist() == [1, 1, 1]


def test_binary_metrics_have_fixed_confusion_order():
    metrics = evaluate_binary([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["confusion_labels"] == ["DOWN", "UP"]


def test_training_reports_only_validation_and_test_metrics(tmp_path):
    outcome = train_from_candles(sample_candles(), trained_at=START)
    metadata = outcome.metadata
    assert metadata["features"] == list(FEATURE_COLUMNS)
    assert metadata["trained_at"] == START.isoformat()
    assert set(metadata["evaluation"]) == {"validation", "test"}
    assert metadata["split"]["train"]["end"] < metadata["split"]["validation"]["start"]
    assert metadata["split"]["validation"]["end"] < metadata["split"]["test"]["start"]
    assert metadata["split"]["purge_rows_at_each_boundary"] == 1
    assert outcome.baseline["majority_class"] in (0, 1)
    assert outcome.baseline["test"]["sample_count"] == metadata["split"]["test"]["rows"]
    assert metadata["evaluation"]["test"]["sample_count"] == metadata["split"]["test"]["rows"]
    assert sum(metadata["split"]["train"]["class_distribution"].values()) == metadata["split"]["train"]["rows"]

    paths = save_training_outcome(outcome, tmp_path)
    assert all(path.exists() for path in paths.values())
    bundle = joblib.load(paths["model"])
    assert bundle["features"] == list(FEATURE_COLUMNS)
    assert bundle["model_version"] == "xgboost_v1"
    assert json.loads(paths["metadata"].read_text(encoding="utf-8"))["evaluation"]["test"] == metadata["evaluation"]["test"]
    assert json.loads(paths["baseline"].read_text(encoding="utf-8"))["name"] == "majority_class"


def test_training_rejects_one_class_data():
    with pytest.raises(ValueError, match="ambas clases"):
        train_from_candles(sample_candles(flat=True), trained_at=START)

