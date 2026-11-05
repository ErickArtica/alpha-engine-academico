from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.services.ingestion import (
    MarketIngestionService,
    closed_boundary,
    interval_duration,
)


START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def candle(minute: int) -> Candle:
    open_time = START + timedelta(minutes=minute)
    return Candle(
        symbol="BTCUSDT", interval="1m", open_time=open_time,
        close_time=open_time + timedelta(milliseconds=59_999),
        open=100, high=102, low=99, close=101, volume=2,
    )


class FakeSource:
    def __init__(self, candles, page_size=2):
        self.candles = candles
        self.page_size = page_size
        self.calls = []

    def fetch_page(self, symbol, interval, start, stop, limit=1000):
        self.calls.append((start, stop))
        return [c for c in self.candles if start <= c.open_time < stop][:self.page_size]


class FakeRepository:
    def __init__(self):
        self.candles = {}
        self.calls = 0

    def upsert_candles(self, candles):
        self.calls += 1
        for item in candles:
            self.candles[(item.symbol, item.interval, item.open_time)] = item
        return len(candles)

    def get_latest_candle(self, symbol, interval):
        matching = [
            item for key, item in self.candles.items()
            if key[0] == symbol and key[1] == interval
        ]
        return max(matching, key=lambda item: item.open_time, default=None)

    def close(self):
        pass


def test_closed_boundary_uses_utc_minute():
    now = START + timedelta(minutes=2, seconds=42)
    assert closed_boundary(now, "1m") == START + timedelta(minutes=2)
    assert interval_duration("1m") == timedelta(minutes=1)
    with pytest.raises(ValueError, match="no soportado"):
        interval_duration("1M")


def test_backfill_paginates_and_repeated_run_is_idempotent():
    source = FakeSource([candle(0), candle(1), candle(2)], page_size=2)
    repo = FakeRepository()
    service = MarketIngestionService(source, repo)

    assert service.ingest_range("BTCUSDT", "1m", START, START + timedelta(minutes=3)) == 3
    assert len(repo.candles) == 3
    assert source.calls[1][0] == START + timedelta(minutes=2)
    assert service.ingest_range("BTCUSDT", "1m", START, START + timedelta(minutes=3)) == 3
    assert len(repo.candles) == 3


def test_fetch_latest_starts_after_latest_and_skips_when_current():
    source = FakeSource([candle(0), candle(1), candle(2)])
    repo = FakeRepository()
    repo.upsert_candles([candle(0)])
    service = MarketIngestionService(source, repo)
    now = START + timedelta(minutes=3, seconds=17)

    assert service.fetch_latest("BTCUSDT", "1m", now=now) == 2
    assert source.calls[0][0] == START + timedelta(minutes=1)
    assert service.fetch_latest("BTCUSDT", "1m", now=now) == 0
    assert len(source.calls) == 1  # la segunda ejecución no consulta Binance
    assert len(repo.candles) == 3


def test_backfill_uses_configured_days_and_closed_end():
    source = FakeSource([])
    repo = FakeRepository()
    service = MarketIngestionService(source, repo)
    now = START + timedelta(minutes=3, seconds=10)
    assert service.backfill("BTCUSDT", "1m", days=1, now=now) == 0
    assert source.calls[0] == (START + timedelta(minutes=3) - timedelta(days=1), START + timedelta(minutes=3))


def test_service_rejects_unclosed_candle_from_source():
    item = candle(0).model_copy(update={"close_time": START + timedelta(minutes=2)})
    service = MarketIngestionService(FakeSource([item]), FakeRepository())
    with pytest.raises(ValueError, match="velas abiertas"):
        service.ingest_range("BTCUSDT", "1m", START, START + timedelta(minutes=1))
