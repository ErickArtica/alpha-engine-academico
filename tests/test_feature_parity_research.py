"""Research audit invariants without requiring market/network services."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from alpha_engine.data.schemas import Candle
from scripts.analyze_feature_parity import (
    METADATA, classify_difference, difference_summary, population_stability_index,
)

def make_candles(count):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [Candle(symbol="BTCUSDT", interval="1m", open_time=start + timedelta(minutes=i),
                   close_time=start + timedelta(minutes=i, seconds=59, milliseconds=999),
                   open=100+i/100, high=101+i/100, low=99+i/100,
                   close=100+i/100, volume=10+i/10) for i in range(count)]


def test_frozen_model_metadata_matches_live_feature_order_and_finite_dtype():
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["features"] == list(FEATURE_COLUMNS)
    frame = build_feature_frame(make_candles(120))
    input_frame = pd.DataFrame([frame.iloc[-1][list(FEATURE_COLUMNS)].to_dict()], columns=list(FEATURE_COLUMNS))
    assert list(input_frame.columns) == metadata["features"]
    assert input_frame.dtypes.map(lambda dtype: np.issubdtype(dtype, np.number)).all()
    assert np.isfinite(input_frame.to_numpy(dtype=float)).all()


def test_research_parity_classification_and_psi():
    equal = difference_summary([1, 2, 3] * 10, [1, 2, 3] * 10, 1)
    assert classify_difference(equal) == "PARITY_OK"
    minor = difference_summary([1] * 30, [1.001] * 30, 1)
    assert classify_difference(minor) == "MINOR_NUMERICAL_DIFFERENCE"
    material = difference_summary([1] * 30, [2] * 30, 1)
    assert classify_difference(material) == "MATERIAL_MISMATCH"
    assert population_stability_index(np.arange(100), np.arange(100)) == 0


def test_historical_full_history_vs_live_120_at_same_source_timestamp():
    candles = make_candles(300)
    source = 199
    full = build_feature_frame(candles[:source+1]).iloc[-1]
    live = build_feature_frame(candles[source-119:source+1]).iloc[-1]
    assert full.name == live.name == pd.Timestamp(candles[source].open_time)
    assert full["return_1m"] == live["return_1m"]
    assert full["volume_change"] == live["volume_change"]
    assert np.isfinite(live[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()
    # Restarting recursive EMA/MACD has a small, nonzero numerical effect.
    assert 0 < abs(full["macd_signal"] - live["macd_signal"]) < 0.001
    changed_target = make_candles(300)
    changed_target[source+1].close = 100_000
    assert build_feature_frame(changed_target[:source+1]).iloc[-1][list(FEATURE_COLUMNS)].equals(
        full[list(FEATURE_COLUMNS)])
