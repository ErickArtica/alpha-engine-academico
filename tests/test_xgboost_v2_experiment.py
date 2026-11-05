"""Small deterministic checks for the offline V1/V2 experiment."""

from datetime import datetime, timedelta, timezone
from math import sin

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from scripts.build_v2_dataset import V1, V2, _indicators, build_paired, temporal_splits, validate_canonical
from scripts.download_binance_btcusdt_1m import _expected_checksum, planned_archives
from scripts.analyze_v2_live_drift import replay_features
from scripts.train_compare_xgboost_v2 import ORIGINAL, ORIGINAL_SHA, evaluate, expanding_folds, sha256


def sample_frame(n=100):
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    price = np.array([100 + .05*i + sin(i/5) for i in range(n)])
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="min"),
        "open": price, "high": price + 1, "low": price - 1,
        "close": price, "volume": 10 + np.arange(n) % 7,
        "close_time": pd.date_range(start, periods=n, freq="min") + pd.Timedelta(minutes=1) - pd.Timedelta(microseconds=1),
    })


def test_original_formulas_match_production_and_relative_definitions():
    raw = sample_frame()
    candles = [Candle(symbol="BTCUSDT", interval="1m", open_time=r.timestamp.to_pydatetime(),
                      close_time=r.close_time.to_pydatetime(), open=r.open, high=r.high,
                      low=r.low, close=r.close, volume=r.volume)
               for r in raw.itertuples(index=False)]
    prod = build_feature_frame(candles)
    experimental = _indicators(raw)
    assert V1 == FEATURE_COLUMNS
    assert V2 == ("return_1m", "rsi_14", "ema_distance", "macd_relative",
                  "macd_signal_relative", "macd_histogram_relative", "rolling_volatility", "volume_change")
    for name in V1:
        np.testing.assert_allclose(experimental[name], prod[name], atol=1e-11, rtol=1e-11, equal_nan=True)
    close = experimental.close.to_numpy()
    np.testing.assert_allclose(experimental.ema_distance, (close-experimental.ema_20)/close, equal_nan=True)
    for absolute, relative in (("macd", "macd_relative"), ("macd_signal", "macd_signal_relative"),
                               ("macd_histogram", "macd_histogram_relative")):
        np.testing.assert_allclose(experimental[relative], experimental[absolute]/close, equal_nan=True)
    assert "close" not in V2 and "ema_20" not in V2


def test_features_are_causal_and_h1_ties_are_down():
    raw = sample_frame()
    original = _indicators(raw)
    changed = raw.copy()
    changed.loc[changed.index >= 70, "close"] += 1000
    changed.loc[changed.index >= 70, "open"] += 1000
    changed.loc[changed.index >= 70, "high"] += 1000
    changed.loc[changed.index >= 70, "low"] += 1000
    after = _indicators(changed)
    assert_frame_equal(original.loc[:69, list(dict.fromkeys((*V1, *V2)))],
                       after.loc[:69, list(dict.fromkeys((*V1, *V2)))])
    # Target at row 68 uses only close of row 69; row 69 target uses row 70.
    assert original.loc[68, "target"] == after.loc[68, "target"]
    flat = sample_frame()
    flat.loc[40, "close"] = flat.loc[39, "close"]
    flat.loc[40, "open"] = flat.loc[40, "close"]
    flat.loc[40, "high"] = flat.loc[40, "close"] + 1
    flat.loc[40, "low"] = flat.loc[40, "close"] - 1
    assert _indicators(flat).loc[39, "target"] == 0


def test_canonical_rejects_order_non_utc_and_nonfinite_values():
    raw = sample_frame(40)
    validate_canonical(raw)
    with pytest.raises(ValueError):
        validate_canonical(raw.iloc[::-1].copy())
    local = raw.copy()
    local["timestamp"] = local.timestamp.dt.tz_convert("America/Lima")
    with pytest.raises(ValueError, match="UTC"):
        validate_canonical(local)
    invalid = raw.copy()
    invalid.loc[8, "volume"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_canonical(invalid)


def test_paired_cohort_is_ordered_finite_and_excludes_future_columns():
    paired, audit = build_paired(sample_frame())
    assert audit["labeled_rows"] > 0
    assert paired.timestamp.is_monotonic_increasing
    assert (paired.target_open_time == paired.timestamp + pd.Timedelta(minutes=1)).all()
    assert np.isfinite(paired[list(dict.fromkeys((*V1, *V2)))].to_numpy()).all()
    assert not ({"target", "target_open_time", "close", "previous_close"} & set(V1 + V2))


def test_splits_and_walk_forward_purge_labels_before_validation():
    times = ["2026-07-31T23:58:00Z", "2026-07-31T23:59:00Z",
             "2026-08-01T00:00:00Z", "2026-08-31T23:59:00Z",
             "2026-09-01T00:00:00Z", "2026-09-15T23:59:00Z",
             "2026-09-16T00:00:00Z", "2026-09-17T00:00:00Z"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True), "target": [0,1,0,1,0,1,0,1]})
    frame["target_open_time"] = frame.timestamp + pd.Timedelta(minutes=1)
    parts, _ = temporal_splits(frame)
    assert [len(parts[name]) for name in ("TRAIN", "VALIDATION", "TEST", "FORWARD")] == [1,1,1,2]
    for left, right in (("TRAIN", "VALIDATION"), ("VALIDATION", "TEST"), ("TEST", "FORWARD")):
        assert parts[left].target_open_time.max() < parts[right].timestamp.min()
    develop = pd.DataFrame({"timestamp": pd.date_range("2026-03-01", periods=1200, freq="min", tz="UTC"),
                            "target_open_time": pd.date_range("2026-03-01", periods=1200, freq="min", tz="UTC") + pd.Timedelta(minutes=1)})
    folds = list(expanding_folds(develop))
    assert len(folds) == 4
    for _, train, valid in folds:
        assert train.target_open_time.max() < valid.timestamp.min()


def test_confusion_orientation_and_frozen_product_model():
    scored = evaluate([0,0,1,1], [.1,.6,.4,.9])
    assert scored["confusion_predicted_rows_actual_columns"] == {
        "DOWN": {"DOWN": 1, "UP": 1}, "UP": {"DOWN": 1, "UP": 1}}
    assert sha256(ORIGINAL) == ORIGINAL_SHA


def test_official_archive_plan_and_checksum_validation():
    planned = planned_archives()
    assert len(planned) == 27
    assert planned[0].period == "2026-03" and planned[-1].period == "2026-09-21"
    assert all("data.binance.vision/data/spot/" in item.url for item in planned)
    first = planned[0]
    checksum = "a"*64
    assert _expected_checksum(f"{checksum}  {first.filename}", first.filename) == checksum
    with pytest.raises(ValueError):
        _expected_checksum(f"{checksum}  other.zip", first.filename)


def test_live_replay_uses_only_complete_120_candle_windows():
    raw = sample_frame(122)
    candles = [Candle(symbol="BTCUSDT", interval="1m", open_time=r.timestamp.to_pydatetime(),
                      close_time=r.close_time.to_pydatetime(), open=r.open, high=r.high,
                      low=r.low, close=r.close, volume=r.volume)
               for r in raw.itertuples(index=False)]
    result, omitted = replay_features(candles)
    assert omitted == 0 and len(result) == 3
    assert result.timestamp.iloc[0] == candles[119].open_time
    expected = build_feature_frame(candles[:120]).iloc[-1]
    np.testing.assert_allclose(result.iloc[0][list(V1)].to_numpy(dtype=float),
                               expected[list(V1)].to_numpy(dtype=float))
    gap = candles[:120] + candles[121:]
    result, omitted = replay_features(gap)
    assert len(result) == 1 and omitted == 1
