from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.repositories.influx import InfluxMarketRepository


OPEN_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def sample_candle(open_time=OPEN_TIME):
    return Candle(
        symbol="BTCUSDT",
        interval="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=12.5,
    )


class FakeWriteApi:
    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)


class FakeRecord:
    def __init__(self, candle):
        self.values = {
            "symbol": candle.symbol,
            "interval": candle.interval,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "close_time_ms": int(candle.close_time.timestamp() * 1000),
        }
        self._time = candle.open_time

    def get_time(self):
        return self._time


class FakeTable:
    def __init__(self, records):
        self.records = records


class FakeQueryApi:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeTable(self.records)]


class FakeClient:
    def __init__(self, records=()):
        self.writer = FakeWriteApi()
        self.querier = FakeQueryApi(records)
        self.closed = False

    def write_api(self, **kwargs):
        return self.writer

    def query_api(self):
        return self.querier

    def close(self):
        self.closed = True


def test_write_uses_stable_series_and_open_time():
    fake = FakeClient()
    repo = InfluxMarketRepository(fake, "alpha-engine", "market")
    candle = sample_candle()

    assert repo.upsert_candles([candle]) == 1
    assert repo.upsert_candles([candle]) == 1
    first = fake.writer.calls[0]
    second = fake.writer.calls[1]
    assert first["bucket"] == "market"
    assert first["org"] == "alpha-engine"
    assert first["record"][0].to_line_protocol() == second["record"][0].to_line_protocol()
    assert "symbol=BTCUSDT" in first["record"][0].to_line_protocol()
    assert "interval=1m" in first["record"][0].to_line_protocol()
    assert str(int(OPEN_TIME.timestamp() * 1000)) in first["record"][0].to_line_protocol()


def test_read_reconstructs_and_sorts_candles():
    first = sample_candle()
    second = sample_candle(OPEN_TIME + timedelta(minutes=1))
    fake = FakeClient([FakeRecord(second), FakeRecord(first)])
    repo = InfluxMarketRepository(fake, "alpha-engine", "market")

    result = repo.get_candles("BTCUSDT", "1m", OPEN_TIME, OPEN_TIME + timedelta(minutes=2))
    assert result == [first, second]
    assert 'r.symbol == "BTCUSDT"' in fake.querier.calls[0]["query"]
    assert 'r.interval == "1m"' in fake.querier.calls[0]["query"]


def test_latest_returns_none_when_empty():
    fake = FakeClient()
    repo = InfluxMarketRepository(fake, "alpha-engine", "market")
    assert repo.get_latest_candle("BTCUSDT", "1m") is None
    repo.close()
    assert fake.closed


def test_range_requires_aware_ordered_times():
    repo = InfluxMarketRepository(FakeClient(), "alpha-engine", "market")
    with pytest.raises(ValueError, match="zona horaria"):
        repo.get_candles("BTCUSDT", "1m", OPEN_TIME.replace(tzinfo=None), OPEN_TIME)


def test_candle_rejects_inconsistent_prices():
    with pytest.raises(ValueError, match="high"):
        Candle(**{**sample_candle().model_dump(), "high": 100.0})

