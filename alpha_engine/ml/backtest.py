"""Causal, long-only spot simulation on a precomputed signal and execution grid.

Each signal has a source candle open timestamp. Its fill timestamp must be at
least two whole minutes later; the source candle has closed before the fill.
No position or signal is carried between independent evaluation periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1000.0
    allocation: float = 0.20
    threshold: float = 0.55
    fee: float = 0.001
    slippage: float = 0.0002

    def __post_init__(self) -> None:
        if self.initial_capital <= 0 or not 0 < self.allocation <= 1:
            raise ValueError("Capital must be positive and allocation in (0, 1]")
        if not 0.5 < self.threshold < 1 or not 0 <= self.fee < 1 or not 0 <= self.slippage < 1:
            raise ValueError("Invalid threshold, fee or slippage")


def execution_grid(candles: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Join each prediction to the open exactly two minutes after its source open.

    `candles` contains *only* the desired evaluation period. Missing candles
    cause a hard failure; no synthetic prices or fills are created.
    """
    required_candles = {"timestamp", "open"}
    required_signals = {"timestamp", "p_up"}
    if not required_candles <= set(candles) or not required_signals <= set(signals):
        raise ValueError("Missing candle or signal columns")
    grid = candles[["timestamp", "open"]].copy().reset_index(drop=True)
    if (grid.empty or grid.timestamp.isna().any() or grid.open.isna().any()
        or (grid.open <= 0).any() or not np.isfinite(grid.open.to_numpy(dtype=float)).all()
        or grid.timestamp.duplicated().any() or not grid.timestamp.is_monotonic_increasing
        or not np.all(np.diff(grid.timestamp.astype("int64").to_numpy()) == 60_000_000_000)):
        raise ValueError("Execution candles must be complete, positive, ordered 1m bars")
    if not isinstance(grid.timestamp.dtype, pd.DatetimeTZDtype) or str(grid.timestamp.dt.tz) != "UTC":
        raise ValueError("UTC execution timestamps required")
    source = signals[["timestamp", "p_up"]].copy()
    if (source.timestamp.duplicated().any() or source.timestamp.isna().any()
        or not source.timestamp.is_monotonic_increasing
        or not isinstance(source.timestamp.dtype, pd.DatetimeTZDtype)
        or str(source.timestamp.dt.tz) != "UTC"
        or not np.isfinite(source.p_up.to_numpy(dtype=float)).all()
        or not source.p_up.between(0, 1).all()):
        raise ValueError("Signals must have ordered, unique UTC timestamps and valid probabilities")
    source["execution_time"] = pd.to_datetime(source.timestamp.astype("int64") + 120_000_000_000, utc=True)
    source = source.rename(columns={"timestamp": "source_time"})
    grid = grid.merge(source, left_on="timestamp", right_on="execution_time", how="left", validate="one_to_one")
    grid = grid.drop(columns="execution_time")
    return grid


def simulate(grid: pd.DataFrame, config: BacktestConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fill BUY/SELL at delayed opens; mark equity at each open after any fill."""
    if grid.empty or not {"timestamp", "open", "p_up", "source_time"} <= set(grid):
        raise ValueError("Invalid execution grid")
    cash = config.initial_capital
    qty = 0.0
    entry = None
    fees = gross_pnl = slippage_cost = 0.0
    counts = {name: 0 for name in ("BUY", "SELL", "HOLD", "WAIT")}
    trades = []
    curve = []
    minutes_long = 0

    def sell(time, price, reason):
        nonlocal cash, qty, entry, fees, gross_pnl, slippage_cost
        sale_price = price * (1 - config.slippage)
        proceeds = qty * sale_price
        sale_fee = proceeds * config.fee
        cash += proceeds - sale_fee
        fees += sale_fee
        gross = qty * (price - entry["reference_price"])
        gross_pnl += gross
        slip_cost = qty * ((entry["fill_price"] - entry["reference_price"]) + (price - sale_price))
        slippage_cost += slip_cost
        net = proceeds - sale_fee - entry["cash_debit"]
        trades.append({"entry_source_time": entry["source_time"], "entry_time": entry["time"],
                       "exit_time": time, "exit_reason": reason,
                       "entry_reference_price": entry["reference_price"], "entry_fill_price": entry["fill_price"],
                       "exit_reference_price": price, "exit_fill_price": sale_price,
                       "quantity": qty, "gross_pnl": gross, "net_pnl": net,
                       "fees": entry["fee"] + sale_fee, "slippage_cost": slip_cost,
                       "return_on_deployed_cash_pct": 100 * net / entry["cash_debit"]})
        qty = 0.0
        entry = None
        counts["SELL"] += 1

    n = len(grid)
    for index, row in enumerate(grid.itertuples(index=False)):
        time, price = row.timestamp, float(row.open)
        probability = row.p_up
        source_time = row.source_time
        action = ""
        if index == n - 1:
            if qty > 0:
                sell(time, price, "period_end")
                action = "SELL"
            # The final open is a liquidation point; its signal cannot open a trade.
        elif pd.notna(probability):
            if source_time.value + 120_000_000_000 != time.value:
                raise ValueError("Signal execution is not delayed by two minutes")
            if qty == 0:
                if probability >= config.threshold:
                    budget = cash * config.allocation
                    notional = budget / (1 + config.fee)
                    buy_price = price * (1 + config.slippage)
                    qty = notional / buy_price
                    buy_fee = notional * config.fee
                    debit = notional + buy_fee
                    cash -= debit
                    fees += buy_fee
                    entry = {"source_time": source_time, "time": time, "reference_price": price,
                             "fill_price": buy_price, "cash_debit": debit, "fee": buy_fee}
                    counts["BUY"] += 1
                    action = "BUY"
                else:
                    counts["WAIT"] += 1
                    action = "WAIT"
            elif 1 - probability >= config.threshold:
                sell(time, price, "signal")
                action = "SELL"
            else:
                counts["HOLD"] += 1
                action = "HOLD"
        # Exposure is the interval [this open, next open).
        equity = cash + qty * price
        curve.append({"timestamp": time, "equity": equity, "cash": cash,
                      "position_quantity": qty, "action": action})
        if index < n - 1 and qty > 0:
            minutes_long += 1
    equity_frame = pd.DataFrame(curve)
    peaks = np.maximum.accumulate(np.r_[config.initial_capital, equity_frame.equity.to_numpy()])[1:]
    equity_frame["drawdown_pct"] = 100 * (equity_frame.equity.to_numpy() / peaks - 1)
    trade_frame = pd.DataFrame(trades)
    wins = int((trade_frame.net_pnl > 0).sum()) if trades else 0
    losses = int((trade_frame.net_pnl < 0).sum()) if trades else 0
    result = {"initial_capital": config.initial_capital, "final_capital": cash,
              "return_pct": 100 * (cash / config.initial_capital - 1),
              "gross_pnl": gross_pnl, "net_pnl": cash - config.initial_capital,
              "trades": len(trades), "winning_trades": wins, "losing_trades": losses,
              "flat_trades": len(trades) - wins - losses,
              "win_rate_pct": 100 * wins / len(trades) if trades else None,
              "average_trade_net": float(trade_frame.net_pnl.mean()) if trades else None,
              "max_drawdown_pct": float(max(0.0, -equity_frame.drawdown_pct.min())),
              "fees_paid": fees, "slippage_cost": slippage_cost,
              "time_exposure_pct": 100 * minutes_long / max(1, n - 1),
              "BUY_count": counts["BUY"], "SELL_count": counts["SELL"],
              "HOLD_count": counts["HOLD"], "WAIT_count": counts["WAIT"]}
    return result, trade_frame, equity_frame


def buy_and_hold(grid: pd.DataFrame, initial_capital: float, fee: float, slippage: float) -> dict:
    """Invest all initial cash at the first open; liquidate at the last open."""
    first, last = float(grid.open.iloc[0]), float(grid.open.iloc[-1])
    notional = initial_capital / (1 + fee)
    buy_fee = notional * fee
    qty = notional / (first * (1 + slippage))
    proceeds = qty * last * (1 - slippage)
    sell_fee = proceeds * fee
    path = qty * grid.open.to_numpy(dtype=float)
    path[-1] = proceeds - sell_fee
    peak = np.maximum.accumulate(np.r_[initial_capital, path])[1:]
    return {"initial_capital": initial_capital, "final_capital": proceeds - sell_fee,
            "return_pct": 100 * ((proceeds - sell_fee) / initial_capital - 1),
            "gross_pnl": qty * (last - first), "net_pnl": proceeds - sell_fee - initial_capital,
            "fees_paid": buy_fee + sell_fee,
            "max_drawdown_pct": float(np.max(100 * (1 - path / peak))),
            "time_exposure_pct": 100.0, "trades": 1}


def confidence_bins(p_up: np.ndarray, target: np.ndarray, edges=(0.5, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 1.0)) -> list[dict]:
    """Directional reliability by max(P(UP), P(DOWN)); no threshold fitting."""
    p = np.asarray(p_up, dtype=float)
    y = np.asarray(target, dtype=int)
    if len(p) != len(y) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid probabilities or labels")
    direction = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        n = int(mask.sum())
        actual_up = int(y[mask].sum())
        actual_down = n - actual_up
        predicted_up = int(direction[mask].sum())
        predicted_down = n - predicted_up
        accuracy = float((direction[mask] == y[mask]).mean()) if n else None
        balanced = None
        if actual_up and actual_down:
            balanced = float(0.5 * ((direction[mask & (y == 1)] == 1).mean()
                                    + (direction[mask & (y == 0)] == 0).mean()))
        rows.append({"min_confidence": low, "max_confidence": high, "n": n,
                     "coverage_pct": 100 * n / len(y) if len(y) else 0,
                     "accuracy": accuracy, "balanced_accuracy": balanced,
                     "actual_UP": actual_up, "actual_DOWN": actual_down,
                     "predicted_UP": predicted_up, "predicted_DOWN": predicted_down})
    return rows
