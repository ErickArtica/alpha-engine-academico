"""Accounting, timing and validation-only selection checks for the offline policy."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.backtest import BacktestConfig, confidence_bins, execution_grid, simulate
from scripts.run_v3_policy_backtest import choose_validation_candidate


def candles(n=6, opens=None):
    return pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=n, freq="min", tz="UTC"),
                         "open": opens if opens is not None else [100.0] * n})


def test_signals_fill_two_minutes_later_and_never_at_source_close():
    bars = candles(opens=[100, 100, 100, 101, 102, 102])
    signals = pd.DataFrame({"timestamp": bars.timestamp.iloc[:2], "p_up": [0.9, 0.1]})
    grid = execution_grid(bars, signals)
    result, trades, curve = simulate(grid, BacktestConfig(fee=0, slippage=0))
    assert result["BUY_count"] == result["SELL_count"] == 1
    assert trades.entry_time.iloc[0] == bars.timestamp.iloc[2]
    assert trades.exit_time.iloc[0] == bars.timestamp.iloc[3]
    assert trades.entry_source_time.iloc[0] == bars.timestamp.iloc[0]
    assert trades.gross_pnl.iloc[0] == pytest.approx(2)
    assert result["net_pnl"] == pytest.approx(2)
    assert curve.action.tolist() == ["", "", "BUY", "SELL", "", ""]


def test_round_trip_costs_and_final_liquidation_reconcile():
    bars = candles(n=5, opens=[100] * 5)
    signals = pd.DataFrame({"timestamp": bars.timestamp.iloc[[0]], "p_up": [0.9]})
    result, trades, curve = simulate(execution_grid(bars, signals),
                                     BacktestConfig(fee=0.001, slippage=0.0005))
    assert result["trades"] == 1
    assert trades.exit_reason.iloc[0] == "period_end"
    assert result["net_pnl"] == pytest.approx(result["gross_pnl"] - result["fees_paid"] - result["slippage_cost"])
    assert result["net_pnl"] == pytest.approx(trades.net_pnl.sum())
    assert result["final_capital"] == pytest.approx(curve.equity.iloc[-1])
    assert result["time_exposure_pct"] == pytest.approx(50)


def test_gap_rejected_instead_of_filling_at_wrong_bar():
    bars = candles().drop(index=2).reset_index(drop=True)
    signals = pd.DataFrame({"timestamp": bars.timestamp.iloc[[0]], "p_up": [0.9]})
    with pytest.raises(ValueError, match="complete"):
        execution_grid(bars, signals)


def test_confidence_bin_does_not_report_balanced_accuracy_for_one_class():
    rows = confidence_bins(np.array([0.9, 0.8]), np.array([0, 0]))
    high = rows[-1]
    assert high["n"] == 2
    assert high["balanced_accuracy"] is None
    assert high["predicted_UP"] == 2


def test_candidate_selection_requires_economic_filter_and_not_maximum_trades():
    gates = [{"threshold": 0.52, "trades": 100, "net_pnl": 2,
              "max_drawdown_pct": 4, "balanced_accuracy": 0.55, "coverage_pct": 40},
             {"threshold": 0.54, "trades": 40, "net_pnl": 1.5,
              "max_drawdown_pct": 3, "balanced_accuracy": 0.58, "coverage_pct": 30}]
    threshold, supported, _ = choose_validation_candidate(gates, {"net_pnl": 1, "max_drawdown_pct": 5})
    assert supported and threshold == 0.54
    _, supported, _ = choose_validation_candidate(gates, {"net_pnl": 3, "max_drawdown_pct": 5})
    assert not supported
