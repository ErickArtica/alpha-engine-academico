import json
from datetime import datetime, timezone
from types import SimpleNamespace

from alpha_engine.jobs import run_live_cycle as cli
from alpha_engine.services.errors import MarketDataError


def test_one_shot_returns_zero_and_structured_summary(monkeypatch, capsys):
    result = SimpleNamespace(completed=True, status="NO_NEW_CANDLE",
                             run_id="run-1", duration_ms=12.5,
                             as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
                             candles_ingested=0, new_candle=False, candle=None,
                             prediction_id=None, prediction_created=False)
    monkeypatch.setattr(cli, "run_live_cycle", lambda: result)
    monkeypatch.setattr(cli, "configure_logging", lambda: None)
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NO_NEW_CANDLE"


def test_one_shot_reports_sanitized_market_error(monkeypatch, capsys):
    def fail():
        raise MarketDataError("sensitive upstream detail")
    monkeypatch.setattr(cli, "run_live_cycle", fail)
    monkeypatch.setattr(cli, "configure_logging", lambda: None)
    assert cli.main() == 2
    message = capsys.readouterr().err
    assert json.loads(message)["error_type"] == "MarketDataError"
    assert "sensitive" not in message
