"""Causal horizons, common cohort, purge, lock, and product immutability."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from alpha_engine.features import FEATURE_COLUMNS
from scripts.analyze_hold_target import PRODUCT_SHA, ROOT
from scripts.analyze_horizons import (build_horizon_frames, common_frames, common_splits,
                                      future_target, split_horizon, target_concordance)
from scripts.run_horizon_experiment import (FEATURES, common_expanding_folds, compare_bootstrap,
                                            freeze_candidates, input_features)


def synthetic_index(count=1000):
    return pd.date_range("2026-08-18T00:00:00+00:00", periods=count, freq="1min")


def test_h1_h3_h5_use_exact_future_close_and_preserve_tie_down(monkeypatch):
    import scripts.analyze_horizons as horizons
    times = synthetic_index(10)
    closes = [100, 101, 99, 102, 100, 103, 99, 98, 104, 100]
    candle_map = dict(zip(times, closes))
    source = pd.DataFrame(index=times[:-1])
    source["source_close"] = closes[:-1]
    source["binary_target"] = [int(closes[i + 1] > closes[i]) for i in range(9)]
    for name in FEATURE_COLUMNS:
        source[name] = np.arange(9, dtype=float)
    monkeypatch.setattr(horizons, "load_frozen_cohort", lambda: (source, None,
        {"data_range": {"closed_candles": 10}}, {}))
    monkeypatch.setattr(horizons, "load_candle_closes", lambda: candle_map)
    frames, _ = build_horizon_frames()
    assert [len(frames[h]) for h in (1, 3, 5)] == [9, 7, 5]
    assert frames[1].iloc[0]["future_close"] == closes[1]
    assert frames[3].iloc[0]["future_close"] == closes[3]
    assert frames[5].iloc[0]["future_close"] == closes[5]
    assert frames[3].iloc[0]["target_candle_open_time"] == times[3]
    assert frames[5].iloc[0]["target_candle_open_time"] == times[5]
    for h in (1, 3, 5):
        assert frames[h].iloc[0]["source_candle_close_time"] < frames[h].iloc[0]["target_candle_open_time"]
        assert frames[h].iloc[0]["return_1m"] == source.iloc[0]["return_1m"]
    assert future_target(np.array([100, 100]), np.array([100, 101])).tolist() == [0, 1]


def test_horizon_aware_natural_and_common_purge():
    index = synthetic_index()
    base = pd.DataFrame({"target": np.arange(1000) % 2}, index=index)
    natural = {h: split_horizon(base, h) for h in (1, 3, 5)}
    assert [len(natural[h].train) for h in (1, 3, 5)] == [699, 697, 695]
    for h, parts in natural.items():
        assert parts.train.index[-1].value + h * 60_000_000_000 < parts.validation.index[0].value
        assert parts.validation.index[-1].value + h * 60_000_000_000 < parts.test.index[0].value
    common = common_frames({1: base, 3: base.iloc[:-2], 5: base.iloc[:-4]})
    assert all(len(frame) == 996 for frame in common.values())
    splits = common_splits(common)
    assert splits[1].train.index.equals(splits[3].train.index)
    assert splits[3].test.index.equals(splits[5].test.index)
    assert splits[1].train.index[-1].value + 5 * 60_000_000_000 < splits[1].validation.index[0].value


def test_common_walk_forward_boundaries_purge_five_minutes():
    frame = pd.DataFrame({"target": np.arange(1000) % 2}, index=synthetic_index())
    folds = list(common_expanding_folds(frame))
    assert len(folds) == 4
    for _, train, validation in folds:
        assert validation.index[0].value - train.index[-1].value == 6 * 60_000_000_000


def test_target_concordance_matrix_is_aligned():
    index = synthetic_index(4)
    frames = {1: pd.DataFrame({"target": [0, 1, 0, 1]}, index=index),
              3: pd.DataFrame({"target": [0, 0, 1, 1]}, index=index),
              5: pd.DataFrame({"target": [1, 0, 1, 1]}, index=index)}
    result = target_concordance(frames, index)
    assert result["H1_vs_H3"]["real_rows_down_up_future_columns_down_up"] == [[1, 1], [1, 1]]
    assert result["H1_vs_H5"]["same_fraction"] == .25


def test_input_features_exclude_all_future_columns_and_tfi():
    row = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
    row.update({"source_close": 100, "future_close": 999, "future_return_bps": 89900,
                "target": 1, "trade_flow_imbalance": .8})
    x = input_features(pd.DataFrame([row]))
    assert list(x.columns) == FEATURES == list(FEATURE_COLUMNS)
    assert "future_close" not in x and "trade_flow_imbalance" not in x


def test_paired_block_bootstrap_identical_horizons_gives_zero_difference():
    y = np.array([0, 1] * 120)
    p = np.array([.4, .6] * 120)
    result = compare_bootstrap(y, p, y, p, block=20, repetitions=50)
    for interval in result["difference"].values():
        assert interval["lower_95"] == interval["upper_95"] == 0


def test_candidate_lock_rejects_three_and_cannot_be_replaced(monkeypatch, tmp_path):
    import scripts.run_horizon_experiment as experiment
    dev = tmp_path / "development.json"
    description = tmp_path / "description.json"
    lock = tmp_path / "lock.json"
    dev.write_text(json.dumps({"code_sha256": "fixed"}))
    description.write_text("{}")
    monkeypatch.setattr(experiment, "DEVELOPMENT", dev)
    monkeypatch.setattr(experiment, "DEVELOPMENT_DESCRIPTION", description)
    monkeypatch.setattr(experiment, "LOCK", lock)
    monkeypatch.setattr(experiment, "code_sha", lambda: "fixed")
    with pytest.raises(ValueError, match="one or two"):
        freeze_candidates([1, 3, 5], "too many")
    result = freeze_candidates([1, 5], "TRAIN+VALIDATION evidence only")
    assert result["test_metrics_seen_at_lock"] is False
    assert result["candidates_minutes"] == [1, 5]
    with pytest.raises(ValueError, match="already exists"):
        freeze_candidates([3], "cannot replace")


def test_product_artifact_hash_and_real_h1_target_reproduction():
    assert hashlib.sha256((ROOT / "models/xgboost_v1.joblib").read_bytes()).hexdigest() == PRODUCT_SHA
    if not (ROOT / "data/experiments/tfi/experimental_labeled.jsonl").exists():
        pytest.skip("Private frozen Binance research dataset unavailable")
    frames, _ = build_horizon_frames()
    assert (len(frames[1]), len(frames[3]), len(frames[5])) == (43165, 43163, 43161)
