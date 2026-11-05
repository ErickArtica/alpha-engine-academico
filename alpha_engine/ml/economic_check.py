"""Simulación diagnóstica de BUY Spot con velas de un minuto y costos estimados."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from alpha_engine.data.schemas import Candle
from alpha_engine.ml.trainer import _atomic_json
from alpha_engine.ml.walk_forward import REPORT_VERSION, candle_digest

CHECK_VERSION = "economic_check_v1"
MINUTE = timedelta(minutes=1)
LATENCY_SECONDS = (0, 1, 5, 30, 60)


@dataclass(frozen=True)
class EstimatedCosts:
    fee_bps_each_side: float
    spread_bps_total: float
    slippage_bps_each_side: float

    def __post_init__(self) -> None:
        values = (self.fee_bps_each_side, self.spread_bps_total, self.slippage_bps_each_side)
        if any(not 0 <= value < 10_000 for value in values):
            raise ValueError("Los costos estimados deben estar entre 0 y 10000 bps")
        if self.spread_bps_total / 20_000 + self.slippage_bps_each_side / 10_000 >= 1:
            raise ValueError("Spread y deslizamiento producen un precio de salida inválido")

    def as_dict(self) -> dict:
        return {
            "fee_bps_each_side": self.fee_bps_each_side,
            "spread_bps_total": self.spread_bps_total,
            "slippage_bps_each_side": self.slippage_bps_each_side,
        }


def _prepare_windows(report: dict, candles: Sequence[Candle]) -> list[list[tuple[dict, Candle, Candle]]]:
    if report.get("report_version") != REPORT_VERSION or not report.get("input_sha256"):
        raise ValueError("Genere de nuevo walk_forward_v1.json con decisiones y huella de datos")
    if candle_digest(candles) != report["input_sha256"]:
        raise ValueError("Las velas locales no coinciden con las usadas en la evaluación temporal")
    if len(candles) != report["data_range"]["closed_candles"]:
        raise ValueError("El número de velas no coincide con el informe temporal")
    candle_by_time = {candle.open_time: candle for candle in candles}
    if len(candle_by_time) != len(candles):
        raise ValueError("Hay velas duplicadas")
    prepared = []
    seen = set()
    for window in report["windows"]:
        decisions = window.get("test_decisions")
        if decisions is None or len(decisions) != window["test"]["rows"]:
            raise ValueError("Faltan decisiones individuales del bloque TEST")
        if decisions[0]["timestamp"] != window["test"]["start"] or decisions[-1]["timestamp"] != window["test"]["end"]:
            raise ValueError("Las fechas de decisiones no coinciden con TEST")
        counts = Counter(row["signal"] for row in decisions)
        if any(row["signal"] not in {"BUY", "SELL", "HOLD"} for row in decisions) or dict(counts) != {key: value for key, value in window["signal"]["counts"].items() if value}:
            raise ValueError("Las señales no coinciden con el informe temporal")
        rows = []
        previous = None
        for decision in decisions:
            timestamp = datetime.fromisoformat(decision["timestamp"])
            if timestamp.tzinfo is None or timestamp in seen or (previous is not None and timestamp <= previous):
                raise ValueError("Las decisiones deben tener fechas únicas y ordenadas")
            seen.add(timestamp)
            previous = timestamp
            candle = candle_by_time.get(timestamp)
            next_candle = candle_by_time.get(timestamp + MINUTE)
            if candle is None or next_candle is None:
                raise ValueError("Faltan velas consecutivas para valorar una decisión")
            rows.append((decision, candle, next_candle))
        prepared.append(rows)
    return prepared


def _simulate_window(
    rows: list[tuple[dict, Candle, Candle]], costs: EstimatedCosts, latency_seconds: int,
    include_trades: bool,
) -> dict:
    counts: Counter[str] = Counter()
    gross_equity = net_equity = peak_equity = 1.0
    max_drawdown = 0.0
    gross_sum = net_sum = cost_sum = 0.0
    net_wins = 0
    trades = []
    for decision, candle, next_candle in rows:
        signal = decision["signal"]
        if signal == "HOLD":
            counts["hold"] += 1
            continue
        generated_at = candle.close_time + timedelta(seconds=latency_seconds)
        if generated_at < next_candle.close_time:
            counts["signal_before_expiry"] += 1
        if signal == "SELL":
            counts["sell_without_initial_inventory"] += 1
            continue
        counts["buy_signal"] += 1
        if generated_at >= next_candle.close_time:
            counts["expired_buy"] += 1
            continue
        if generated_at >= next_candle.open_time:
            counts["buy_without_observed_quote_after_generation"] += 1
            continue

        entry_mid = next_candle.open
        exit_mid = next_candle.close
        half_spread = costs.spread_bps_total / 20_000
        slippage = costs.slippage_bps_each_side / 10_000
        fee = costs.fee_bps_each_side / 10_000
        entry_ask_estimate = entry_mid * (1 + half_spread + slippage)
        exit_bid_estimate = exit_mid * (1 - half_spread - slippage)
        gross_return = exit_mid / entry_mid - 1
        net_return = (exit_bid_estimate / entry_ask_estimate) * (1 - fee) ** 2 - 1
        gross_equity *= 1 + gross_return
        net_equity *= 1 + net_return
        peak_equity = max(peak_equity, net_equity)
        max_drawdown = max(max_drawdown, 1 - net_equity / peak_equity)
        gross_sum += gross_return
        net_sum += net_return
        cost_sum += gross_return - net_return
        net_wins += net_return > 0
        counts["executed_buy"] += 1
        if include_trades:
            trades.append({
                "signal_candle": decision["timestamp"],
                "generated_at_assumed": generated_at.isoformat(),
                "entry_time": next_candle.open_time.isoformat(),
                "exit_time": next_candle.close_time.isoformat(),
                "entry_open_observed": entry_mid,
                "exit_close_observed": exit_mid,
                "entry_ask_estimated": entry_ask_estimate,
                "exit_bid_estimated": exit_bid_estimate,
                "gross_return": gross_return,
                "net_return_estimated": net_return,
            })

    tested = len(rows)
    executed = counts["executed_buy"]
    directional = counts["buy_signal"] + counts["sell_without_initial_inventory"]
    result = {
        "tested_rows": tested,
        "signals": {"BUY": counts["buy_signal"], "SELL": counts["sell_without_initial_inventory"], "HOLD": counts["hold"]},
        "signal_coverage": directional / tested,
        "signal_before_expiry": counts["signal_before_expiry"],
        "signal_before_expiry_fraction": counts["signal_before_expiry"] / directional if directional else None,
        "executed_buy_trades": executed,
        "trade_coverage": executed / tested,
        "sell_without_initial_inventory": counts["sell_without_initial_inventory"],
        "buy_without_observed_quote_after_generation": counts["buy_without_observed_quote_after_generation"],
        "expired_buy": counts["expired_buy"],
        "gross_compounded_return": gross_equity - 1,
        "net_compounded_return_estimated": net_equity - 1,
        "max_drawdown_net_estimated": max_drawdown,
        "average_gross_return_bps": gross_sum / executed * 10_000 if executed else None,
        "average_net_return_bps_estimated": net_sum / executed * 10_000 if executed else None,
        "average_cost_drag_bps_estimated": cost_sum / executed * 10_000 if executed else None,
        "net_win_fraction_estimated": net_wins / executed if executed else None,
    }
    if include_trades:
        result["trades"] = trades
    return result


def evaluate_economics(
    temporal_report: dict, candles: Sequence[Candle],
    base_costs: EstimatedCosts = EstimatedCosts(10, 2, 1),
    latencies: Sequence[int] = LATENCY_SECONDS,
) -> dict:
    """Prueba sensibilidad sin presentar fills, spreads o latencias como observados."""
    if not latencies or any(not isinstance(value, int) or value < 0 for value in latencies):
        raise ValueError("Las latencias deben ser segundos enteros no negativos")
    windows = _prepare_windows(temporal_report, candles)
    scenarios = {
        "zero_cost": EstimatedCosts(0, 0, 0),
        "base_estimate": base_costs,
        "double_estimate": EstimatedCosts(
            base_costs.fee_bps_each_side * 2,
            base_costs.spread_bps_total * 2,
            base_costs.slippage_bps_each_side * 2,
        ),
    }
    runs = []
    for scenario, costs in scenarios.items():
        for latency in latencies:
            all_rows = [row for window in windows for row in window]
            detailed = scenario == "base_estimate" and latency == 0
            run = _simulate_window(all_rows, costs, latency, detailed)
            per_window = [_simulate_window(window, costs, latency, False) for window in windows]
            if sum(part["executed_buy_trades"] for part in per_window) != run["executed_buy_trades"]:
                raise AssertionError("Los conteos por ventana y total no coinciden")
            runs.append({
                "cost_scenario": scenario,
                "costs": costs.as_dict(),
                "latency_seconds_assumed": latency,
                "total": run,
                "windows": [dict(window=source["window"], **part) for source, part in zip(temporal_report["windows"], per_window)],
            })
    return {
        "check_version": CHECK_VERSION,
        "source_report_version": temporal_report["report_version"],
        "source_input_sha256": temporal_report["input_sha256"],
        "source_test_rows": temporal_report["summary"]["total_test_rows"],
        "symbol": temporal_report["symbol"],
        "observed_data": "OHLCV de velas cerradas y señales XGBoost seleccionadas en VALIDATION; no incluye decisiones del LLM ni guardrails de riesgo",
        "assumptions": {
            "decision_time": "close_time de vela t + latencia simulada",
            "first_available_entry": "open de la vela t+1 sólo si ocurre estrictamente después de generar la señal; sin cotización intraminuto posterior disponible",
            "exit": "close de vela t+1, proxy optimista de venta al cierre; no demuestra fill real",
            "spot_inventory": "capital inicial en efectivo; BUY entra y sale al horizonte de 1 minuto; SELL sin activo previo no abre corto",
            "position_size": "cada BUY ejecutable invierte el 100 % del capital de referencia y se liquida antes de la siguiente decisión; retornos compuestos sobre operaciones, sin financiación",
            "drawdown": "máxima caída de capital neto observada sólo tras cerrar cada operación; no mide pérdidas intraminuto",
            "costs": "spread, comisión y deslizamiento son bps estimados, no cotizaciones ni tarifas históricas observadas",
            "latency": "segundos fijos supuestos; no son mediciones del pipeline, LLM o interfaz",
        },
        "runs": runs,
        "limits": "Las velas 1m no permiten reconstruir bid/ask, precio después de la generación intraminuto ni ejecución de stops. Con latencia positiva se informa falta de cotización observable, no ausencia demostrada de oportunidad real. Ningún resultado valida rentabilidad operable.",
    }


def save_economic_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output_path)
    return output_path
