from datetime import datetime, timedelta, timezone
from math import sin

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candles(count: int, price_fn=None, volume_fn=None) -> list[Candle]:
    price_fn = price_fn or (lambda i: 100.0)
    volume_fn = volume_fn or (lambda i: 10.0)
    result = []
    for i in range(count):
        open_time = START + timedelta(minutes=i)
        price = price_fn(i)
        result.append(
            Candle(
                symbol="BTCUSDT", interval="1m", open_time=open_time,
                close_time=open_time + timedelta(milliseconds=59_999),
                open=price, high=price + 1, low=price - 1,
                close=price, volume=volume_fn(i),
            )
        )
    return result


def test_flat_market_indicators_have_expected_values():
    frame = build_feature_frame(candles(40))
    assert frame.index.tz is not None
    assert frame["macd_signal"].iloc[:33].isna().all()
    last = frame.iloc[-1]
    assert last["return_1m"] == 0
    assert last["rsi_14"] == 50
    assert last["ema_20"] == 100
    assert last["macd"] == 0
    assert last["macd_signal"] == 0
    assert last["macd_histogram"] == 0
    assert last["rolling_volatility"] == 0
    assert last["volume_change"] == 0
    assert np.isfinite(last[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()


def test_future_candles_do_not_change_past_features():
    base = candles(50, price_fn=lambda i: 100 + 0.1 * i + sin(i / 4))
    original = build_feature_frame(base)
    extreme_tail = candles(65, price_fn=lambda i: 100 + 0.1 * i + sin(i / 4) if i < 50 else 10_000.0)
    extended = build_feature_frame(extreme_tail)
    assert_frame_equal(original, extended.iloc[:50], check_exact=True)


def test_feature_frame_rejects_gap_and_duplicate():
    sequence = candles(40)
    with pytest.raises(ValueError, match="huecos"):
        build_feature_frame(sequence[:20] + sequence[21:])
    with pytest.raises(ValueError, match="duplicado"):
        build_feature_frame(sequence + [sequence[-1]])


def test_zero_previous_volume_does_not_create_infinity():
    sequence = candles(40, volume_fn=lambda i: 0 if i == 38 else 10)
    frame = build_feature_frame(sequence)
    assert pd.isna(frame["volume_change"].iloc[39])

