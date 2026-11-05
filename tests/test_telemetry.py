from datetime import datetime, timedelta, timezone

import httpx
import pytest

from alpha_engine.data.book_ticker import BinanceBookTickerClient, BookTickerQuote
from alpha_engine.observability.report import build_latency_report
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.services.quote_capture import capture_quote

OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(milliseconds=59_999)


def test_book_ticker_records_local_request_and_receipt_times(tmp_path):
    times = iter([CLOSE + timedelta(seconds=5), CLOSE + timedelta(seconds=5, milliseconds=80)])

    def respond(request):
        assert request.url.path == "/api/v3/ticker/bookTicker"
        assert request.url.params["symbol"] == "BTCUSDT"
        return httpx.Response(200, json={
            "symbol": "BTCUSDT", "bidPrice": "100.00", "bidQty": "2",
            "askPrice": "100.02", "askQty": "3",
        })

    client = httpx.Client(base_url="https://api.binance.com", transport=httpx.MockTransport(respond))
    source = BinanceBookTickerClient(client=client, clock=lambda: next(times))
    store = TelemetryStore(tmp_path)
    quote = capture_quote(source, store, "BTCUSDT", "trace-1", "prediction-1")
    assert quote.bid_price == 100.0
    assert quote.ask_price == 100.02
    assert quote.received_at - quote.requested_at == timedelta(milliseconds=80)
    assert store.read_quotes()[0]["prediction_id"] == "prediction-1"
    assert store.read_events()[0]["stage"] == "quote_received"
    client.close()


def test_book_ticker_rejects_invalid_spread_and_capture_does_not_raise(tmp_path):
    times = iter([CLOSE, CLOSE + timedelta(milliseconds=1)])
    client = httpx.Client(base_url="https://api.binance.com", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={
            "symbol": "BTCUSDT", "bidPrice": "101", "bidQty": "1",
            "askPrice": "100", "askQty": "1",
        })
    ))
    source = BinanceBookTickerClient(client=client, clock=lambda: next(times))
    store = TelemetryStore(tmp_path)
    assert capture_quote(source, store, "BTCUSDT", "trace-1") is None
    assert store.read_quotes() == []
    assert store.read_events()[0]["stage"] == "quote_failed"
    client.close()


def test_latency_report_uses_only_recorded_times(tmp_path):
    store = TelemetryStore(tmp_path)
    store.event("data_ready", "cycle-1", "BTCUSDT", at=CLOSE + timedelta(seconds=5), details={"candles_sent": 1})
    store.event("prediction_ready", "cycle-1", "BTCUSDT", at=CLOSE + timedelta(seconds=6),
                candle_open_time=OPEN, prediction_id="pred-1", duration_ms=1000)
    store.event("context_ready", "rec-1", "BTCUSDT", at=CLOSE + timedelta(seconds=8),
                candle_open_time=OPEN, prediction_id="pred-1", duration_ms=200)
    store.event("llm_finished", "rec-1", "BTCUSDT", at=CLOSE + timedelta(seconds=9),
                candle_open_time=OPEN, prediction_id="pred-1", duration_ms=900,
                details={"attempt": 1, "status": "ok"})
    store.event("recommendation_ready", "rec-1", "BTCUSDT", at=CLOSE + timedelta(seconds=10),
                candle_open_time=OPEN, prediction_id="pred-1", duration_ms=3000,
                details={"final_signal": "BUY", "expires_at": (CLOSE + timedelta(minutes=1)).isoformat()})
    store.event("server_rendered", "rec-1", "BTCUSDT", at=CLOSE + timedelta(seconds=11),
                candle_open_time=OPEN, prediction_id="pred-1")
    store.quote(BookTickerQuote(
        symbol="BTCUSDT", requested_at=CLOSE + timedelta(seconds=11),
        received_at=CLOSE + timedelta(seconds=12), bid_price=100, bid_qty=1,
        ask_price=100.02, ask_qty=1,
    ), "rec-1", "pred-1")
    report = build_latency_report(store.read_events(), store.read_quotes())
    assert report["prediction_cycles"] == 1
    assert report["predictions_ready_before_expiry_fraction"] == 1
    assert report["measurements"]["prediction_remaining_lifetime_ms"]["median"] == pytest.approx(54_000)
    assert report["recommendations_with_prediction"] == 1
    assert report["recommendations_before_expiry_fraction"] == 1
    assert report["actionable_recommendations_before_expiry"] == 1
    assert report["quotes_received_after_ready_before_expiry"] == 1
    assert report["measurements"]["candle_close_to_data_ready_ms"]["median"] == pytest.approx(5000)
    assert report["measurements"]["llm_call_ms"]["median"] == 900
    assert report["measurements"]["server_render_after_ready_ms"]["median"] == 1000
    assert report["measurements"]["observed_spread_bps"]["unit"] == "bps"


def test_empty_latency_report_does_not_invent_samples():
    report = build_latency_report([], [])
    assert report["recommendations_before_expiry_fraction"] is None
    assert report["measurements"]["llm_call_ms"]["samples"] == 0
