from datetime import datetime, timedelta, timezone

import httpx
import pytest

from alpha_engine.data.binance import BinanceKlineClient


START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def kline(minute: int, close_time_ms: int | None = None) -> list:
    open_ms = int((START + timedelta(minutes=minute)).timestamp() * 1000)
    return [
        open_ms,
        "100.0",
        "102.0",
        "99.0",
        "101.0",
        "12.5",
        close_time_ms if close_time_ms is not None else open_ms + 59_999,
    ]


def test_fetch_page_requests_expected_range_and_excludes_open_candles():
    requests = []

    def handler(request):
        requests.append(request)
        open_second = kline(1)
        open_second[6] = int((START + timedelta(minutes=2)).timestamp() * 1000)
        return httpx.Response(200, json=[kline(0), open_second, kline(2)])

    http_client = httpx.Client(
        base_url="https://api.binance.com", transport=httpx.MockTransport(handler)
    )
    source = BinanceKlineClient(client=http_client)
    result = source.fetch_page("BTCUSDT", "1m", START, START + timedelta(minutes=2))

    assert [c.open_time for c in result] == [START]
    assert result[0].close == 101.0
    assert requests[0].url.path == "/api/v3/klines"
    assert requests[0].url.params["startTime"] == str(int(START.timestamp() * 1000))
    assert requests[0].url.params["endTime"] == str(
        int((START + timedelta(minutes=2)).timestamp() * 1000) - 1
    )
    assert requests[0].url.params["limit"] == "1000"
    http_client.close()


def test_fetch_page_propagates_http_error():
    client = httpx.Client(
        base_url="https://api.binance.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(429)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        BinanceKlineClient(client=client).fetch_page(
            "BTCUSDT", "1m", START, START + timedelta(minutes=1)
        )
    client.close()


def test_fetch_page_rejects_malformed_payload():
    client = httpx.Client(
        base_url="https://api.binance.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[[1, 2]])),
    )
    with pytest.raises(ValueError, match="incompleta"):
        BinanceKlineClient(client=client).fetch_page(
            "BTCUSDT", "1m", START, START + timedelta(minutes=1)
        )
    client.close()

