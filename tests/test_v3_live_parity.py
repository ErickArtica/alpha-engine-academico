"""V3 historical Binance replay against the frozen twelve-month training cohort."""

import hashlib
import json
from datetime import timedelta

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest

from alpha_engine.config.settings import PROJECT_ROOT
from alpha_engine.data.binance import BinanceKlineClient
from alpha_engine.features.v3 import V3_FEATURE_COLUMNS, build_v3_feature_frame
from alpha_engine.services.v3_shadow import LOOKBACK, METADATA_PATH, MODEL_PATH, V3Shadow
from scripts.build_v3_dataset import make_v3_columns


COHORT = PROJECT_ROOT / "data/processed/btcusdt_v3_12m_paired.parquet"
TIMESTAMPS = (
    pd.Timestamp("2026-08-15T12:00:00Z"),
    pd.Timestamp("2026-09-10T12:00:00Z"),
    pd.Timestamp("2026-09-20T12:00:00Z"),
)


def _kline(row):
    opened = int(row.timestamp.timestamp() * 1000)
    return [opened, str(row.open), str(row.high), str(row.low), str(row.close),
            str(row.volume), opened + 59_999, "0", int(row.trade_count),
            str(row.taker_buy_base), "0", "0"]


def _cohort():
    if not COHORT.is_file() or not MODEL_PATH.is_file():
        pytest.skip("Frozen V3 cohort and candidate artifact required")
    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "close_time", "trade_count", "taker_buy_base", *V3_FEATURE_COLUMNS]
    return pd.read_parquet(COHORT, columns=list(dict.fromkeys(columns)))


def _replay(segment):
    rows = [_kline(row) for row in segment.itertuples(index=False)]
    client = httpx.Client(base_url="https://api.binance.com",
                          transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
    return BinanceKlineClient(client=client), client


def test_frozen_training_live_feature_and_prediction_parity(capsys):
    cohort = _cohort()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == metadata["model_sha256"]
    assert metadata["features"] == list(V3_FEATURE_COLUMNS)
    model = joblib.load(MODEL_PATH)["estimator"]
    max_error = {name: 0.0 for name in V3_FEATURE_COLUMNS}
    max_probability_error = 0.0
    class_matches = 0
    feature_ms = []
    prediction_ms = []
    for timestamp in TIMESTAMPS:
        segment = cohort.loc[(cohort.timestamp > timestamp - pd.Timedelta(minutes=LOOKBACK)) &
                             (cohort.timestamp <= timestamp)].reset_index(drop=True)
        assert len(segment) == LOOKBACK
        raw = segment[["timestamp", "open", "high", "low", "close", "volume", "close_time"]]
        extra = segment[["timestamp", "volume", "trade_count", "taker_buy_base"]].assign(quote_volume=0.0)
        offline = make_v3_columns(raw, extra).iloc[-1][list(V3_FEATURE_COLUMNS)]
        client, http_client = _replay(segment)
        try:
            live_raw = client.fetch_v3_page("BTCUSDT", "1m", timestamp.to_pydatetime() - timedelta(minutes=LOOKBACK-1),
                                            timestamp.to_pydatetime() + timedelta(minutes=1), limit=LOOKBACK)
            assert len(live_raw) == LOOKBACK
            live = build_v3_feature_frame(live_raw).iloc[-1][list(V3_FEATURE_COLUMNS)]
            assert np.isfinite(live.to_numpy(dtype=float)).all()
            for name in V3_FEATURE_COLUMNS:
                max_error[name] = max(max_error[name], abs(float(live[name]) - float(segment[name].iloc[-1])))
                assert abs(float(offline[name]) - float(segment[name].iloc[-1])) <= 1e-10
            offline_input = pd.DataFrame([segment.iloc[-1][list(V3_FEATURE_COLUMNS)].to_dict()], columns=list(V3_FEATURE_COLUMNS))
            offline_p = model.predict_proba(offline_input)[0]
            shadow = V3Shadow(client).predict("BTCUSDT", "1m", timestamp.to_pydatetime())
            feature_ms.append(shadow.feature_ms)
            prediction_ms.append(shadow.prediction_ms)
            live_p = np.array([shadow.prob_down, shadow.prob_up])
            max_probability_error = max(max_probability_error, float(np.max(np.abs(offline_p-live_p))))
            class_matches += int(("UP" if offline_p[1] > offline_p[0] else "DOWN") == shadow.predicted_class)
        finally:
            http_client.close()
    assert max(max_error.values()) <= 1e-10
    assert max_probability_error <= 1e-7
    assert class_matches == len(TIMESTAMPS)
    print("V3_PARITY=" + json.dumps({"feature_max_abs_error": max_error,
          "max_probability_error": max_probability_error, "class_matches": class_matches,
          "sample_count": len(TIMESTAMPS), "feature_ms": feature_ms,
          "prediction_ms": prediction_ms}))


def test_v3_shared_builder_is_causal_and_rejects_missing_fields():
    cohort = _cohort()
    timestamp = TIMESTAMPS[-1]
    segment = cohort.loc[(cohort.timestamp > timestamp - pd.Timedelta(minutes=LOOKBACK)) &
                         (cohort.timestamp <= timestamp)].reset_index(drop=True)
    raw = segment[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "taker_buy_base"]].copy()
    before = build_v3_feature_frame(raw)
    changed = raw.copy()
    changed.loc[450:, ["open", "high", "low", "close"]] *= 1.1
    changed.loc[450:, ["volume", "trade_count", "taker_buy_base"]] *= 2
    after = build_v3_feature_frame(changed)
    pd.testing.assert_frame_equal(before.iloc[:450], after.iloc[:450], check_exact=True)
    with pytest.raises(ValueError, match="requires"):
        build_v3_feature_frame(raw.drop(columns="taker_buy_base"))


def test_v3_live_reader_excludes_unclosed_kline():
    cohort = _cohort()
    segment = cohort.iloc[-2:].copy()
    rows = [_kline(row) for row in segment.itertuples(index=False)]
    rows[-1][6] = rows[-1][0] + 60_000
    client = httpx.Client(base_url="https://api.binance.com",
                          transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
    source = BinanceKlineClient(client=client)
    start = segment.timestamp.iloc[0].to_pydatetime()
    result = source.fetch_v3_page("BTCUSDT", "1m", start, start + timedelta(minutes=2))
    assert len(result) == 1 and result.timestamp.iloc[0] == pd.Timestamp(start)
    client.close()
