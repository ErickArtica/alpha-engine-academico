from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_frame,
    chronological_split,
    target_distribution,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_target_uses_next_close_and_drops_final_row():
    values = [100 + i for i in range(43)] + [142, 141]
    candles = []
    for i, price in enumerate(values):
        open_time = START + timedelta(minutes=i)
        candles.append(
            Candle(
                symbol="BTCUSDT", interval="1m", open_time=open_time,
                close_time=open_time + timedelta(milliseconds=59_999),
                open=price, high=price + 1, low=price - 1,
                close=price, volume=10,
            )
        )
    training = build_training_frame(build_feature_frame(candles))
    assert training.index[-1] == START + timedelta(minutes=43)
    assert training.loc[START + timedelta(minutes=41), "target"] == 1
    assert training.loc[START + timedelta(minutes=42), "target"] == 0  # empate
    assert training.loc[START + timedelta(minutes=43), "target"] == 0  # baja
    assert START + timedelta(minutes=44) not in training.index
    assert set(training["target"].unique()) == {0, 1}
    assert target_distribution(training) == {
        0: int((training["target"] == 0).sum()),
        1: int((training["target"] == 1).sum()),
    }
    assert training[list(FEATURE_COLUMNS)].notna().all().all()


def test_target_omits_rows_without_next_minute():
    index = pd.DatetimeIndex([START, START + timedelta(minutes=2)])
    frame = pd.DataFrame(
        {"close": [100.0, 101.0], **{name: [1.0, 1.0] for name in FEATURE_COLUMNS}},
        index=index,
    )
    assert build_training_frame(frame).empty


def test_chronological_split_purges_boundary_labels():
    index = pd.date_range(START, periods=100, freq="min")
    frame = pd.DataFrame({"target": [i % 2 for i in range(100)]}, index=index)
    split = chronological_split(frame)
    assert (len(split.train), len(split.validation), len(split.test)) == (69, 14, 15)
    assert split.train.index[-1] == index[68]
    assert split.validation.index[0] == index[70]
    assert split.validation.index[-1] == index[83]
    assert split.test.index[0] == index[85]
    assert index[69] not in split.train.index
    assert index[84] not in split.validation.index


def test_chronological_split_rejects_insufficient_rows():
    frame = pd.DataFrame({"target": [0] * 10}, index=pd.date_range(START, periods=10, freq="min"))
    with pytest.raises(ValueError, match="suficientes"):
        chronological_split(frame)

