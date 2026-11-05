"""Read-only, causal replay of persisted V1 signals with the LIVE paper rule.

The signal is decided at generated_at, while it is valid. An order submitted
then is filled at the next full minute's persisted open (source T + 2m).
Only the open price is used; the candle's later high/low/close never decide an
action. Missing fills are skipped rather than synthesized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.services.paper_policy import paper_decision

MODEL_VERSION = "xgboost_v1"
GATE = 0.60
WINDOWS = (60, 180, 360)


@dataclass(frozen=True)
class ReplayConfig:
    initial_capital: float = 1000.0
    allocation: float = 0.20
    fee: float = 0.001
    slippage: float = 0.0002

    def __post_init__(self):
        if (not isfinite(self.initial_capital) or self.initial_capital <= 0
                or not isfinite(self.allocation) or not 0 < self.allocation <= 1
                or not isfinite(self.fee) or not 0 <= self.fee < 1
                or not isfinite(self.slippage) or not 0 <= self.slippage < 1):
            raise ValueError("Invalid replay simulation parameters")


def load_window(predictions_repo, market_repo, symbol: str, interval: str,
                count: int, before: datetime | None = None) -> tuple[list[PredictionRecord], list[Candle]]:
    """One bounded prediction read and one candle range read; no per-row I/O."""
    if count not in WINDOWS or interval != "1m":
        raise ValueError("Invalid replay window")
    if before is None:
        raw = predictions_repo.list_recent(symbol, limit=min(1000, count * 2))
    else:
        if before.tzinfo is None:
            raise ValueError("Replay anchor must include timezone")
        raw = predictions_repo.list_recent_before(symbol, before, limit=min(1000, count * 2))
    predictions = [p for p in raw if p.model_version == MODEL_VERSION and p.interval == interval][:count]
    predictions.sort(key=lambda p: (p.timestamp, p.prediction_id))
    if not predictions:
        return [], []
    start = predictions[0].timestamp + timedelta(minutes=2)
    stop = predictions[-1].timestamp + timedelta(minutes=3)
    candles = market_repo.get_candles(symbol, interval, start, stop)
    return predictions, candles


def find_event_window(predictions_repo, symbol: str, count: int,
                      scan_limit: int = 100) -> tuple[datetime | None, str, int]:
    """Query real V1 entry signals once; anchor a window at the newest valid one.

    Returns (latest source plus 75% of the window, description, signals read).
    The rest of the window is still loaded by the normal two-read replay path.
    """
    if count not in WINDOWS or not 1 <= scan_limit <= 1000:
        raise ValueError("Invalid event search")
    entries = predictions_repo.list_entry_signals(symbol, MODEL_VERSION, GATE,
                                                   limit=scan_limit)
    for prediction in entries:
        if (prediction.generated_at is None or prediction.valid_until is None
                or not prediction.timestamp + timedelta(minutes=1) <= prediction.generated_at
                < prediction.timestamp + timedelta(minutes=2)):
            continue
        decision = paper_decision(prediction, has_long=False, threshold=GATE,
                                  now=prediction.generated_at)
        if decision.action != "BUY":
            continue
        before = prediction.timestamp + timedelta(minutes=count * 3 // 4)
        return before, "Ventana alrededor de una señal BUY real de V1.", len(entries)
    return None, "No se encontró una señal de entrada elegible en el histórico consultado.", len(entries)


@dataclass(frozen=True)
class ReplayResult:
    config: ReplayConfig
    steps: tuple[dict, ...]
    ledger: tuple[dict, ...]
    omitted: int

    def at(self, index: int) -> dict:
        """Return the displayed prefix; index zero is the untouched wallet."""
        if not 0 <= index <= len(self.steps):
            raise ValueError("Invalid replay step")
        if index == 0:
            initial = self.config.initial_capital
            return {"index": 0, "capital_initial": initial, "capital_final": initial,
                    "cash": initial, "btc": 0.0, "position": "SIN POSICIÓN",
                    "entry_price": None, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                    "total_pnl": 0.0, "fees": 0.0, "return_pct": 0.0,
                    "BUY": 0, "SELL": 0, "HOLD": 0, "WAIT": 0,
                    "closed_trades": 0, "wins": 0, "losses": 0, "win_rate_pct": None,
                    "max_drawdown_pct": 0.0, "exposure_pct": 0.0,
                    "no_trade_capital": initial, "buy_hold_capital": initial,
                    "current": None, "steps": (), "ledger": ()}
        steps = self.steps[:index]
        current = steps[-1]
        first, last = steps[0]["price"], current["price"]
        notional = self.config.initial_capital / (1 + self.config.fee)
        hold_qty = notional / (first * (1 + self.config.slippage))
        hold_value = hold_qty * last  # Marked to market; no retrospective liquidation.
        ledger = tuple(row for row in self.ledger if row["step"] <= index)
        closed = current["closed_trades"]
        return {"index": index, "capital_initial": self.config.initial_capital,
                "capital_final": current["equity"], "cash": current["cash"],
                "btc": current["btc"], "position": current["position"],
                "entry_price": current["entry_price"],
                "realized_pnl": current["realized_pnl"],
                "unrealized_pnl": current["unrealized_pnl"],
                "total_pnl": current["equity"] - self.config.initial_capital,
                "fees": current["fees"],
                "return_pct": 100 * (current["equity"] / self.config.initial_capital - 1),
                **current["counts"], "closed_trades": closed,
                "wins": current["wins"], "losses": current["losses"],
                "win_rate_pct": 100 * current["wins"] / closed if closed else None,
                "max_drawdown_pct": max(-step["drawdown_pct"] for step in steps),
                "exposure_pct": 100 * sum(step["btc"] > 0 for step in steps) / index,
                "no_trade_capital": self.config.initial_capital,
                "buy_hold_capital": hold_value,
                "current": current, "steps": steps, "ledger": ledger}


def build_replay(predictions: list[PredictionRecord], candles: list[Candle],
                 config: ReplayConfig = ReplayConfig()) -> ReplayResult:
    """Replay saved V1 decisions and later real opens, without database writes."""
    by_open = {c.open_time: c for c in candles}
    if len(by_open) != len(candles):
        raise ValueError("Duplicate execution candles")
    cash, btc, entry_cost, entry_price = config.initial_capital, 0.0, 0.0, None
    realized, fees, wins, losses = 0.0, 0.0, 0, 0
    peak = config.initial_capital
    counts = {action: 0 for action in ("BUY", "SELL", "HOLD", "WAIT")}
    steps, ledger = [], []
    omitted = 0
    for prediction in sorted(predictions, key=lambda p: (p.timestamp, p.prediction_id)):
        if prediction.model_version != MODEL_VERSION or prediction.interval != "1m":
            continue
        execution_time = prediction.timestamp + timedelta(minutes=2)
        candle = by_open.get(execution_time)
        if (candle is None or prediction.generated_at is None
                or prediction.valid_until is None or
                not prediction.timestamp + timedelta(minutes=1) <= prediction.generated_at < execution_time):
            omitted += 1
            continue
        # The policy sees only the prediction and current position at generated_at.
        decision = paper_decision(prediction, has_long=btc > 0, threshold=GATE,
                                  now=prediction.generated_at)
        action = decision.action
        if action in ("BUY", "SELL") and not decision.executable:
            raise ValueError("Inconsistent paper policy")
        price = float(candle.open)
        if not isfinite(price) or price <= 0:
            raise ValueError("Invalid execution price")
        if action == "BUY":
            budget = cash * config.allocation
            notional = budget / (1 + config.fee)
            fill = price * (1 + config.slippage)
            btc = notional / fill
            fee = notional * config.fee
            entry_cost = notional + fee
            entry_price = fill
            cash -= entry_cost
            fees += fee
            ledger.append({"step": len(steps) + 1, "timestamp": execution_time,
                           "action": action, "price": fill, "quantity_btc": btc,
                           "value_usd": notional, "fee": fee, "cash_after": cash,
                           "btc_after": btc, "realized_pnl": None,
                           "model_version": prediction.model_version,
                           "signal_probability": decision.signal_probability})
        elif action == "SELL":
            fill = price * (1 - config.slippage)
            notional = btc * fill
            fee = notional * config.fee
            pnl = notional - fee - entry_cost
            realized += pnl
            wins += pnl > 0
            losses += pnl < 0
            cash += notional - fee
            fees += fee
            ledger.append({"step": len(steps) + 1, "timestamp": execution_time,
                           "action": action, "price": fill, "quantity_btc": btc,
                           "value_usd": notional, "fee": fee, "cash_after": cash,
                           "btc_after": 0.0, "realized_pnl": pnl,
                           "model_version": prediction.model_version,
                           "signal_probability": decision.signal_probability})
            btc, entry_cost, entry_price = 0.0, 0.0, None
        counts[action] += 1
        equity = cash + btc * price
        peak = max(peak, equity)
        steps.append({"timestamp": execution_time, "source_timestamp": prediction.timestamp,
                      "generated_at": prediction.generated_at, "price": price,
                      "prob_up": prediction.prob_up, "prob_down": prediction.prob_down,
                      "predicted_class": prediction.predicted_class,
                      "action": action, "reason": decision.reason,
                      "cash": cash, "btc": btc,
                      "position": "LONG" if btc else "SIN POSICIÓN",
                      "entry_price": entry_price, "equity": equity,
                      "realized_pnl": realized,
                      "unrealized_pnl": btc * price - entry_cost if btc else 0.0,
                      "fees": fees, "counts": counts.copy(),
                      "closed_trades": counts["SELL"], "wins": wins, "losses": losses,
                      "drawdown_pct": 100 * (equity / peak - 1)})
    return ReplayResult(config, tuple(steps), tuple(ledger), omitted)
