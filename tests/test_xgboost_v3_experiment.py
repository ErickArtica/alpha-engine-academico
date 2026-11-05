"""Causal price-action/market-flow feature and frozen-control checks."""

import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alpha_engine.data.schemas import Candle
from scripts.analyze_v3_live_drift import AVAILABLE, replay_ohlcv
from scripts.build_v2_dataset import V2, _indicators, temporal_splits
from scripts.build_v3_dataset import (
    V3, _zscore, build_v3_paired, load_extra_raw, make_v3_columns,
)
from scripts.download_binance_btcusdt_1m import Archive
from scripts.train_compare_xgboost_v2 import ORIGINAL, ORIGINAL_SHA, sha256


def candles(n=100):
    t = pd.date_range("2026-03-01", periods=n, freq="min", tz="UTC")
    close = np.array([100 + .1*i + np.sin(i/4) for i in range(n)])
    opening = close - .12
    return pd.DataFrame({"timestamp": t, "open": opening, "high": close + .8,
                         "low": opening - .6, "close": close,
                         "volume": 10 + np.arange(n)*.1,
                         "close_time": t + pd.Timedelta(seconds=59, microseconds=999999)})


def extras(raw):
    n = len(raw)
    return pd.DataFrame({"timestamp": raw.timestamp, "volume": raw.volume,
                         "quote_volume": raw.volume*raw.close,
                         "trade_count": 100 + np.arange(n),
                         "taker_buy_base": raw.volume*.4})


def test_v3_feature_order_and_exact_formulas():
    raw = candles()
    features = make_v3_columns(raw, extras(raw))
    assert len(V3) == 18 and len(set(V3)) == len(V3)
    assert set(V2).issubset(V3)
    assert not {"ema_20", "macd", "close", "target", "target_open_time", "taker_imbalance"} & set(V3)
    t = 40
    for lag in (3, 5, 15):
        assert features.loc[t, f"return_{lag}m"] == pytest.approx(raw.close.iloc[t]/raw.close.iloc[t-lag]-1)
    assert features.loc[t, "candle_range"] == pytest.approx((raw.high.iloc[t]-raw.low.iloc[t])/raw.close.iloc[t])
    assert features.loc[t, "body_return"] == pytest.approx((raw.close.iloc[t]-raw.open.iloc[t])/raw.open.iloc[t])
    assert features.loc[t, "upper_wick"] == pytest.approx((raw.high.iloc[t]-raw.close.iloc[t])/raw.close.iloc[t])
    assert features.loc[t, "lower_wick"] == pytest.approx((raw.open.iloc[t]-raw.low.iloc[t])/raw.close.iloc[t])
    assert features.loc[t, "volume_zscore"] == pytest.approx(
        (raw.volume.iloc[t]-raw.volume.iloc[t-19:t+1].mean())/raw.volume.iloc[t-19:t+1].std(ddof=0))
    counts = extras(raw).trade_count
    assert features.loc[t, "trades_zscore"] == pytest.approx(
        (counts.iloc[t]-counts.iloc[t-19:t+1].mean())/counts.iloc[t-19:t+1].std(ddof=0))
    assert features.loc[t, "taker_buy_ratio"] == pytest.approx(.4)
    assert np.isnan(features.loc[18, "volume_zscore"])


def test_causal_prefix_is_unchanged_by_future_candle_and_raw_activity():
    raw = candles()
    extra = extras(raw)
    original = make_v3_columns(raw, extra)
    changed = raw.copy()
    changed_extra = extra.copy()
    for name in ("open", "high", "low", "close"):
        changed.loc[70:, name] *= 1.2
    changed.loc[70:, "volume"] *= 3
    changed_extra.loc[70:, "volume"] *= 3
    changed_extra.loc[70:, "quote_volume"] *= 4
    changed_extra.loc[70:, "trade_count"] *= 3
    changed_extra.loc[70:, "taker_buy_base"] *= 3
    replay = make_v3_columns(changed, changed_extra)
    assert_frame_equal(original.iloc[:70], replay.iloc[:70], check_exact=True)


def test_zero_denominators_and_bad_raw_values_are_not_silently_filled():
    raw = candles()
    extra = extras(raw)
    raw.loc[40, "volume"] = 0
    extra.loc[40, "volume"] = 0
    extra.loc[40, "taker_buy_base"] = 0
    assert np.isnan(make_v3_columns(raw, extra).loc[40, "taker_buy_ratio"])
    values = pd.Series(np.ones(25))
    assert _zscore(values).iloc[-1] == 0
    invalid = extras(candles())
    invalid.loc[50, "taker_buy_base"] = np.inf
    with pytest.raises(ValueError, match="unexpected NaN/Inf"):
        source = candles()
        labeled = _indicators(source)
        v2 = labeled.loc[(labeled.target >= 0) & np.isfinite(labeled[list(V2)].to_numpy()).all(axis=1)].reset_index(drop=True)
        build_v3_paired(v2, source, invalid)


def test_v3_join_preserves_h1_target_cohort_order_and_temporal_splits():
    raw = candles()
    labeled = _indicators(raw)
    v2 = labeled.loc[(labeled.target >= 0) & np.isfinite(labeled[list(V2)].to_numpy()).all(axis=1)].reset_index(drop=True)
    v3, audit = build_v3_paired(v2, raw, extras(raw))
    assert audit["same_cohort_as_v2"] and audit["missing_v3_features"] == 0
    assert v3.timestamp.equals(v2.timestamp)
    assert v3.target.equals(v2.target)
    assert (v3.target_open_time == v3.timestamp + pd.Timedelta(minutes=1)).all()
    assert np.isfinite(v3[list(V3)].to_numpy()).all()
    assert sha256(ORIGINAL) == ORIGINAL_SHA
    boundaries = pd.to_datetime(["2026-07-31T23:58Z", "2026-07-31T23:59Z",
                                 "2026-08-01T00:00Z", "2026-08-31T23:59Z",
                                 "2026-09-01T00:00Z", "2026-09-15T23:59Z",
                                 "2026-09-16T00:00Z", "2026-09-17T00:00Z"], utc=True)
    split_source = pd.DataFrame({"timestamp": boundaries,
                                 "target_open_time": boundaries + pd.Timedelta(minutes=1),
                                 "target": [0, 1, 0, 1, 0, 1, 0, 1]})
    splits, _ = temporal_splits(split_source)
    for left, right in (("TRAIN", "VALIDATION"), ("VALIDATION", "TEST"), ("TEST", "FORWARD")):
        assert splits[left].target_open_time.max() < splits[right].timestamp.min()


def test_verified_zip_schema_and_taker_volume_validation(tmp_path, monkeypatch):
    import scripts.build_v3_dataset as builder
    filename = "BTCUSDT-1m-2026-03.zip"
    archive = Archive("monthly", "2026-03", filename, "https://data.binance.vision/test")
    monkeypatch.setattr(builder, "planned_archives", lambda: [archive])
    row = "1772323200000000,100,101,99,100,10,1772323259999999,1000,100,4,400,0\n"
    target = tmp_path / filename
    with zipfile.ZipFile(target, "w") as z:
        z.writestr(filename.replace(".zip", ".csv"), row)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"filename": filename, "granularity": "monthly",
                                                "period": "2026-03", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}), encoding="utf-8")
    frame, audit = load_extra_raw(manifest, tmp_path)
    assert audit["rows"] == 1 and audit["raw_field_count_by_file"][filename] == 12
    assert frame.taker_buy_base.iloc[0] == 4 and frame.trade_count.iloc[0] == 100
    with zipfile.ZipFile(target, "w") as z:
        z.writestr(filename.replace(".zip", ".csv"), row.replace(",4,400,0", ",14,400,0"))
    manifest.write_text(json.dumps({"files": [{"filename": filename, "granularity": "monthly",
                                                "period": "2026-03", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        load_extra_raw(manifest, tmp_path)


def test_live_ohlcv_replay_matches_causal_historical_features():
    raw = candles(121)
    items = [Candle(symbol="BTCUSDT", interval="1m", open_time=row.timestamp.to_pydatetime(),
                    close_time=row.close_time.to_pydatetime(), open=row.open, high=row.high,
                    low=row.low, close=row.close, volume=row.volume)
             for row in raw.itertuples(index=False)]
    live, omitted = replay_ohlcv(items)
    assert len(live) == 2 and omitted == 0
    added = make_v3_columns(raw, extras(raw))
    base = _indicators(raw)
    for name in AVAILABLE:
        expected = base.loc[119, name] if name in base else added.loc[119, name]
        assert live.loc[0, name] == pytest.approx(expected, abs=1e-11)
