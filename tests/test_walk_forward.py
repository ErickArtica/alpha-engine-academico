from datetime import datetime, timedelta, timezone
from math import sin
import json

import numpy as np
import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.ml.walk_forward import (
    _select_threshold,
    evaluate_walk_forward,
    save_walk_forward_report,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sample_candles(count=700, future_shift=0.0):
    candles = []
    for i in range(count):
        price = 100 + i * 0.01 + 2 * sin(i / 5) + sin(i / 17)
        if i >= 450:
            price += future_shift
        open_time = START + timedelta(minutes=i)
        candles.append(Candle(
            symbol="BTCUSDT", interval="1m", open_time=open_time,
            close_time=open_time + timedelta(milliseconds=59_999),
            open=price, high=price + 1, low=price - 1,
            close=price, volume=10 + i % 5,
        ))
    return candles


def test_walk_forward_has_disjoint_tests_and_reports_calibration(tmp_path):
    report = evaluate_walk_forward(sample_candles(), windows=2)
    assert report["probabilities_calibrated"] is False
    assert len(report["input_sha256"]) == 64
    first, second = report["windows"]
    assert first["train"]["end"] < first["validation"]["start"]
    assert first["validation"]["end"] < first["test"]["start"]
    assert first["test"]["end"] < second["test"]["start"]
    assert first["test"]["end"] < second["validation"]["start"]
    assert first["test"]["rows"] + second["test"]["rows"] == report["summary"]["total_test_rows"]
    for window in report["windows"]:
        assert window["purge_rows_at_each_boundary"] == 1
        assert len(window["test_decisions"]) == window["test"]["rows"]
        assert {row["signal"] for row in window["test_decisions"]}.issubset({"BUY", "SELL", "HOLD"})
        assert sum(row["rows"] for row in window["model"]["probability"]["reliability"]) == window["test"]["rows"]
        assert sum(window["signal"]["counts"].values()) == window["test"]["rows"]
        assert 0 <= window["model"]["probability"]["brier_score"] <= 1
        assert window["model"]["probability"]["log_loss"] >= 0
    path = save_walk_forward_report(report, tmp_path / "walk_forward_v1.json")
    assert json.loads(path.read_text(encoding="utf-8"))["summary"] == report["summary"]


def test_future_test_prices_do_not_change_first_window_selection():
    original = evaluate_walk_forward(sample_candles(), windows=2)
    changed = evaluate_walk_forward(sample_candles(future_shift=50), windows=2)
    first = original["windows"][0]
    changed_first = changed["windows"][0]
    assert first["selection"] == changed_first["selection"]
    assert first["train"] == changed_first["train"]
    assert first["validation"] == changed_first["validation"]


def test_threshold_selection_can_choose_hold():
    threshold, candidates = _select_threshold(
        actual=np.array([0, 1, 0, 1]),
        prob_up=np.array([0.9, 0.1, 0.8, 0.2]),
    )
    assert threshold is None
    assert all(row["directional_edge_per_row"] <= 0 for row in candidates)


def test_walk_forward_rejects_too_few_candles():
    with pytest.raises(ValueError, match="suficientes"):
        evaluate_walk_forward(sample_candles(100), windows=4)
