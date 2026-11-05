"""Focused tests for the offline calibration study; no network or database."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_calibration import (
    apply_calibrator, calibration_error, class_metrics, confidence_gates,
    fit_calibrators, reliability_bins, temporal_calibration_split,
    threshold_table, wilson,
)


def test_reliability_bins_and_ece_are_weighted_by_observations():
    y = [0, 1, 1, 0]
    p = [0.10, 0.30, 0.70, 0.90]
    width = reliability_bins(y, p, count=2, strategy="width")
    assert [row["n"] for row in width] == [2, 2]
    assert width[0]["mean_predicted_up"] == pytest.approx(.20)
    assert width[0]["observed_up"] == .50
    assert calibration_error(width)["ece"] == pytest.approx(.30)
    quantile = reliability_bins(y, p, count=2, strategy="quantile")
    assert sum(row["n"] for row in quantile) == 4


def test_thresholds_use_strict_greater_than_and_gates_are_separate():
    y = [0, 1, 0, 1]
    p = [.50, .51, .44, .56]
    at_half = class_metrics(y, p, threshold=.50)
    assert at_half["predicted_up"] == 2
    assert at_half["accuracy"] == 1.0
    assert threshold_table(y, p)[6]["threshold"] == .50
    gate = confidence_gates(y, p, gates=(.56,))[0]
    assert gate["n"] == 2
    assert gate["coverage"] == .50
    assert gate["accuracy"] == 1.0
    assert gate["always_down_accuracy_selected"] == .50
    assert wilson(0, 0) is None
    assert wilson(6, 6)[0] < 1.0


def test_temporal_calibration_split_purges_boundary_label():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=i) for i in range(1500)]
    data = {"times": times, "y": np.arange(1500) % 2,
            "p": np.linspace(.45, .55, 1500), "momentum": np.zeros(1500)}
    fit, holdout = temporal_calibration_split(data)
    assert len(fit["y"]) == 999
    assert len(holdout["y"]) == 500
    assert fit["times"][-1] == times[998]
    assert holdout["times"][0] == times[1000]
    bad = dict(data, times=list(reversed(times)))
    with pytest.raises(ValueError, match="ordered"):
        temporal_calibration_split(bad)


def test_calibrators_fit_only_provided_block_and_do_not_touch_production_model():
    model_path = Path(__file__).resolve().parents[1] / "models/xgboost_v1.joblib"
    before = sha256(model_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(4)
    p = np.linspace(.30, .70, 1500)
    y = (rng.random(1500) < p).astype(int)
    platt, isotonic = fit_calibrators(y, p)
    sample = np.array([.40, .50, .60])
    for name in ("platt", "isotonic"):
        calibrated = apply_calibrator(name, sample, platt, isotonic)
        assert len(calibrated) == len(sample)
        assert np.all(np.diff(calibrated) >= 0)
        assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert sha256(model_path.read_bytes()).hexdigest() == before


def test_invalid_probabilities_and_too_small_calibration_are_rejected():
    with pytest.raises(ValueError):
        reliability_bins([0, 1], [.2, 1.2])
    with pytest.raises(ValueError, match="Insufficient"):
        fit_calibrators([0, 1], [.4, .6])
