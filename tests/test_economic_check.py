from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.data.schemas import Candle
from alpha_engine.ml.economic_check import EstimatedCosts, evaluate_economics
from alpha_engine.ml.walk_forward import REPORT_VERSION, candle_digest

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sample_input():
    candles = []
    for i in range(5):
        timestamp = START + timedelta(minutes=i)
        price = 100.0 + i
        candles.append(Candle(
            symbol="BTCUSDT", interval="1m", open_time=timestamp,
            close_time=timestamp + timedelta(milliseconds=59_999),
            open=price, high=price + 2, low=price - 1,
            close=price + 0.5, volume=10,
        ))
    signals = ("BUY", "SELL", "HOLD", "BUY")
    decisions = [
        {"timestamp": (START + timedelta(minutes=i)).isoformat(), "prob_up": 0.6, "signal": signal}
        for i, signal in enumerate(signals)
    ]
    report = {
        "report_version": REPORT_VERSION,
        "input_sha256": candle_digest(candles),
        "symbol": "BTCUSDT",
        "data_range": {"closed_candles": 5},
        "summary": {"total_test_rows": 4},
        "windows": [{
            "window": 1,
            "test": {"rows": 4, "start": decisions[0]["timestamp"], "end": decisions[-1]["timestamp"]},
            "signal": {"counts": {"BUY": 2, "SELL": 1, "HOLD": 1}},
            "test_decisions": decisions,
        }],
    }
    return candles, report


def test_economic_simulation_uses_next_open_and_estimated_costs():
    candles, temporal = sample_input()
    report = evaluate_economics(temporal, candles, EstimatedCosts(10, 2, 1), latencies=[0])
    zero, base, stress = report["runs"]
    assert [row["cost_scenario"] for row in report["runs"]] == ["zero_cost", "base_estimate", "double_estimate"]
    assert base["total"]["signals"] == {"BUY": 2, "SELL": 1, "HOLD": 1}
    assert base["total"]["executed_buy_trades"] == 2
    assert base["total"]["sell_without_initial_inventory"] == 1
    assert base["total"]["trade_coverage"] == 0.5
    assert base["total"]["signal_before_expiry_fraction"] == 1
    trade = base["total"]["trades"][0]
    assert trade["entry_time"] == candles[1].open_time.isoformat()
    assert trade["entry_open_observed"] == candles[1].open
    assert trade["exit_close_observed"] == candles[1].close
    assert trade["entry_open_observed"] != candles[0].close
    assert zero["total"]["net_compounded_return_estimated"] == pytest.approx(zero["total"]["gross_compounded_return"])
    assert stress["total"]["net_compounded_return_estimated"] < base["total"]["net_compounded_return_estimated"] < zero["total"]["net_compounded_return_estimated"]


def test_positive_latency_has_no_observed_entry_and_expiry_is_separate():
    candles, temporal = sample_input()
    report = evaluate_economics(temporal, candles, latencies=[1, 60])
    one_second = next(row for row in report["runs"] if row["cost_scenario"] == "base_estimate" and row["latency_seconds_assumed"] == 1)
    expired = next(row for row in report["runs"] if row["cost_scenario"] == "base_estimate" and row["latency_seconds_assumed"] == 60)
    assert one_second["total"]["signal_before_expiry_fraction"] == 1
    assert one_second["total"]["buy_without_observed_quote_after_generation"] == 2
    assert one_second["total"]["executed_buy_trades"] == 0
    assert expired["total"]["expired_buy"] == 2
    assert expired["total"]["signal_before_expiry_fraction"] == 0
    assert expired["total"]["net_compounded_return_estimated"] == 0


def test_economic_simulation_rejects_revised_prices():
    candles, temporal = sample_input()
    candles[1] = candles[1].model_copy(update={"close": 105.0, "high": 106.0})
    with pytest.raises(ValueError, match="no coinciden"):
        evaluate_economics(temporal, candles)
