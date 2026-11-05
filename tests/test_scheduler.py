from datetime import datetime, timezone
from types import SimpleNamespace

from alpha_engine.jobs import scheduler as scheduler_module
from alpha_engine.jobs.scheduler import build_scheduler


def test_scheduler_registers_one_minute_cycle():
    scheduler = build_scheduler(lambda: None)
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "fetch_latest_candles"
    assert jobs[0].max_instances == 1
    assert jobs[0].coalesce is True
    assert jobs[0].misfire_grace_time == 30
    assert "second='5'" in str(jobs[0].trigger)


def test_scheduler_delegates_to_common_cycle(monkeypatch):
    calls = []
    result = SimpleNamespace(status="NO_NEW_CANDLE", candle=None, prediction=None,
                             prediction_id=None, trace_id=None)
    monkeypatch.setattr(scheduler_module, "run_live_cycle", lambda **kwargs: calls.append(kwargs) or result)
    assert scheduler_module.scheduled_fetch() is True
    assert calls == [{"telemetry": None}]


def test_scheduler_uses_persistent_cycle_and_captures_quote(monkeypatch):
    prediction = SimpleNamespace(symbol="BTCUSDT", prediction_id="pred-1")
    result = SimpleNamespace(status="SUCCESS", candle=SimpleNamespace(open_time=datetime.now(timezone.utc)),
                             prediction=prediction, prediction_id="pred-1", trace_id="trace-1")
    cycle = SimpleNamespace(run=lambda: result)
    calls = []
    monkeypatch.setattr(scheduler_module, "capture_quote", lambda *args: calls.append(args))
    telemetry = object()
    quote_client = object()
    assert scheduler_module.scheduled_fetch(telemetry, quote_client, "BTCUSDT", cycle=cycle) is True
    assert calls == [(quote_client, telemetry, "BTCUSDT", "trace-1", "pred-1")]


def test_scheduler_reports_cycle_failure(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("cycle failed")
    monkeypatch.setattr(scheduler_module, "run_live_cycle", fail)
    assert scheduler_module.scheduled_fetch() is False


def test_scheduler_forwards_telemetry_to_common_cycle(monkeypatch):
    telemetry = object()
    seen = []
    result = SimpleNamespace(status="NO_NEW_CANDLE", candle=None, prediction=None,
                             prediction_id=None, trace_id=None)
    monkeypatch.setattr(scheduler_module, "run_live_cycle",
                        lambda **kwargs: seen.append(kwargs) or result)
    assert scheduler_module.scheduled_fetch(telemetry=telemetry) is True
    assert seen == [{"telemetry": telemetry}]
