"""El corte público sólo contiene datos auditables seleccionados."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import Candle
from scripts.export_public_snapshot import build_public_snapshot


OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def candle(opened: datetime) -> Candle:
    return Candle(
        symbol="BTCUSDT", interval="1m", open_time=opened,
        open=100, high=101, low=99, close=100, volume=2,
        close_time=opened + timedelta(minutes=1) - timedelta(milliseconds=1),
    )


def reports() -> dict[str, dict]:
    return {
        "model": {
            "symbol": "BTCUSDT", "trained_at": "2026-01-01T00:00:00Z",
            "target_definition": "siguiente cierre", "data_range": {},
            "evaluation": {"test": {"accuracy": 0.51}},
        },
        "baseline": {"test": {"accuracy": 0.52}},
        "walk": {
            "symbol": "BTCUSDT", "input_sha256": "same", "generated_at": "2026-01-01T00:00:00Z",
            "summary": {}, "limits": "diagnóstico", "windows": [{}, {}],
        },
        "economics": {
            "symbol": "BTCUSDT", "source_input_sha256": "same",
            "observed_data": "velas", "assumptions": {"costs": {}}, "limits": "sin fills",
            "runs": [{"cost_scenario": "base_estimate", "latency_seconds_assumed": 0,
                      "total": {"executed_buy_trades": 1, "trades": [{"secret": "do-not-export"}]}}],
        },
        "prescription": {
            "trained_at": "2026-01-01T00:00:00Z", "test_rows": 1,
            "eligible_rows": 0, "eligible_fraction": 0, "direction_accuracy_eligible": None,
            "limits": "sin operaciones",
        },
        "latency": {
            "generated_at": "2026-01-01T00:00:00Z", "prediction_cycles": 1,
            "recommendations_with_prediction": 1, "recommendations_before_expiry": 0,
            "actionable_recommendations_before_expiry": 0, "measurements": {},
            "limits": "sin navegador",
        },
    }


def test_public_snapshot_excludes_open_candles_context_and_trades():
    secret = "do-not-export"
    history = [{
        "symbol": "BTCUSDT", "candle_open_at": OPEN.isoformat(),
        "generated_at": OPEN.isoformat(), "final_signal": "HOLD",
        "guardrail_reasons": ["HIGH_RISK"], "counted_for_metric": False,
        "context_json": secret, "response_json": secret,
    }]
    result = build_public_snapshot(
        candles=[candle(OPEN), candle(OPEN + timedelta(minutes=1))],
        prediction=None, history=history, reports=reports(),
        exported_at=OPEN + timedelta(minutes=1, seconds=30),
    )
    assert len(result["candles"]) == 1
    assert result["history"][0]["final_signal"] == "HOLD"
    assert "trades" not in result["economics"]["optimistic_zero_latency"]
    assert secret not in json.dumps(result)


def test_public_snapshot_rejects_mismatched_economic_source():
    incompatible = reports()
    incompatible["economics"]["source_input_sha256"] = "different"
    with pytest.raises(ValueError, match="otras velas"):
        build_public_snapshot(
            candles=[candle(OPEN)], prediction=None, history=[], reports=incompatible,
            exported_at=OPEN + timedelta(minutes=2),
        )
