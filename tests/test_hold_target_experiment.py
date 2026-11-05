"""Causal ternary target, signal accounting, and frozen production checks."""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from alpha_engine.features import FEATURE_COLUMNS
from scripts.analyze_hold_target import (LABELED, PRODUCT_SHA, ROOT, future_return_bps,
                                         load_frozen_cohort, ternary_labels)
from scripts.run_hold_target_experiment import (FEATURES, LOCK, PARAMS, baseline_metrics,
                                                freeze_candidates, model_inputs,
                                                multiclass_metrics)
from scripts.run_tfi_ablation import expanding_folds


def test_return_to_bps_and_explicit_ternary_boundaries():
    assert future_return_bps(np.array([100.0, 100.0]), np.array([100.01, 99.99])) == pytest.approx([1, -1])
    values = np.array([-1.01, -1.0, -0.7, 0.0, 0.3, 1.0, 1.01])
    assert ternary_labels(values, 1.0).tolist() == [0, 1, 1, 1, 1, 1, 2]
    assert ternary_labels(np.array([0.0]), 0).tolist() == [1]
    with pytest.raises(ValueError):
        ternary_labels(values, -1)


def test_feature_matrix_excludes_future_and_trade_flow():
    row = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
    row.update({"next_close": 999.0, "future_return_bps": 9999.0,
                "binary_target": 1, "trade_flow_imbalance": 0.8})
    frame = pd.DataFrame([row])
    result = model_inputs(frame)
    assert list(result.columns) == list(FEATURE_COLUMNS) == FEATURES
    assert "next_close" not in result and "trade_flow_imbalance" not in result
    assert PARAMS["objective"] == "multi:softprob" and PARAMS["num_class"] == 3
    assert "scale_pos_weight" not in PARAMS


def test_signal_coverage_false_actions_missed_moves_and_matrix_orientation():
    real = np.array([0, 1, 2, 2, 1, 0])
    predicted = np.array([0, 0, 1, 2, 2, 1])
    proba = np.eye(3)[predicted] * .9 + .1 / 3
    proba = proba / proba.sum(axis=1, keepdims=True)
    score = multiclass_metrics(real, proba)
    assert score["signal_coverage"] == pytest.approx(4 / 6)
    assert score["directional_accuracy"] == pytest.approx(.5)
    assert score["false_actionable_signals"] == 2
    assert score["false_actionable_rate_among_signals"] == pytest.approx(.5)
    assert score["missed_up"] == score["missed_down"] == 1
    assert score["missed_move_rate_among_true_moves"] == pytest.approx(.5)
    assert score["confusion_pred_rows_down_hold_up_real_columns_down_hold_up"] == [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    assert score["multiclass_brier_sum"] > 0
    assert score["multiclass_log_loss"] > 0


def test_baselines_do_not_reward_always_hold_as_balanced_accuracy():
    real = np.array([0, 1, 1, 1, 2])
    base = baseline_metrics(real, np.array([1, 1, 0, 2]))
    assert base["always_hold"]["accuracy"] == pytest.approx(.6)
    assert base["always_hold"]["balanced_accuracy"] == pytest.approx(1 / 3)
    assert base["always_hold"]["signal_coverage"] == 0


def test_walk_forward_purges_future_target_at_every_boundary():
    frame = pd.DataFrame({"future_return_bps": np.ones(1000)},
                         index=pd.date_range(datetime(2026, 8, 18, tzinfo=timezone.utc), periods=1000, freq="1min"))
    folds = list(expanding_folds(frame))
    assert len(folds) == 4
    for _, past, future in folds:
        assert future.index[0].value - past.index[-1].value == 120_000_000_000


def test_candidate_freeze_requires_development_and_max_two(tmp_path, monkeypatch):
    import scripts.run_hold_target_experiment as experiment
    development = tmp_path / "development.json"
    lock = tmp_path / "lock.json"
    development.write_text(json.dumps({"code_sha256": "fixed",
                                       "production_model_sha256": PRODUCT_SHA}))
    monkeypatch.setattr(experiment, "DEVELOPMENT", development)
    monkeypatch.setattr(experiment, "LOCK", lock)
    monkeypatch.setattr(experiment, "code_sha", lambda: "fixed")
    with pytest.raises(ValueError, match="one or two"):
        freeze_candidates([.5, 1, 2], "too many")
    assert not lock.exists()
    record = freeze_candidates([.5, 1], "TRAIN and VALIDATION only")
    assert record["test_metrics_seen_at_lock"] is False
    assert record["candidates_bps"] == [.5, 1]
    with pytest.raises(ValueError, match="already exists"):
        freeze_candidates([2], "would replace frozen lock")


def test_product_model_immutable_and_frozen_cohort_consistent():
    assert hashlib.sha256((ROOT / "models/xgboost_v1.joblib").read_bytes()).hexdigest() == PRODUCT_SHA
    if not LABELED.exists():
        pytest.skip("Private Binance research dataset is not present in this checkout")
    frame, split, _, _ = load_frozen_cohort()
    assert len(frame) == 43165
    assert (len(split.train), len(split.validation), len(split.test)) == (30214, 6473, 6476)
    assert (frame["next_close"] > frame["source_close"]).astype(int).equals(frame["binary_target"])
    assert int((split.test["source_close"] == split.test["next_close"]).sum()) == 145
