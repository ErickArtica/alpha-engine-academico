"""Ejecutar: python -m alpha_engine.ml.validate_economics [opciones]."""

import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from alpha_engine.config.settings import PROJECT_ROOT, get_settings
from alpha_engine.ml.economic_check import EstimatedCosts, LATENCY_SECONDS, evaluate_economics, save_economic_report
from alpha_engine.repositories.factory import create_market_repository
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simula BUY Spot con costos y latencias hipotéticos")
    parser.add_argument("--temporal-report", type=Path, default=PROJECT_ROOT / "models" / "walk_forward_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "economic_check_v1.json")
    parser.add_argument("--fee-bps", type=float, default=10.0, help="Comisión estimada por lado; default 10 bps")
    parser.add_argument("--spread-bps", type=float, default=2.0, help="Spread total estimado; default 2 bps")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Deslizamiento estimado por lado; default 1 bp")
    parser.add_argument("--latencies", type=int, nargs="+", default=list(LATENCY_SECONDS), help="Segundos simulados tras close_time")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    if settings.interval != "1m":
        parser.error("La simulación requiere INTERVAL=1m")
    try:
        costs = EstimatedCosts(args.fee_bps, args.spread_bps, args.slippage_bps)
        temporal_report = json.loads(args.temporal_report.read_text(encoding="utf-8"))
        start = datetime.fromisoformat(temporal_report["data_range"]["start"])
        stop = datetime.fromisoformat(temporal_report["data_range"]["end"]) + timedelta(minutes=1)
        if start.tzinfo is None or stop.tzinfo is None:
            raise ValueError("El rango del informe debe incluir zona horaria")
        if temporal_report["symbol"] != settings.symbol:
            raise ValueError("SYMBOL no coincide con el informe temporal")
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    repository = create_market_repository(settings)
    try:
        candles = repository.get_candles(settings.symbol, settings.interval, start, stop)
    finally:
        repository.close()
    report = evaluate_economics(temporal_report, candles, costs, args.latencies)
    path = save_economic_report(report, args.output)
    logger.info("Simulación guardada: %s", path)
    base = next(run for run in report["runs"] if run["cost_scenario"] == "base_estimate" and run["latency_seconds_assumed"] == 0)
    logger.info("Escenario base optimista: operaciones=%s retorno neto estimado=%.4f drawdown=%.4f", base["total"]["executed_buy_trades"], base["total"]["net_compounded_return_estimated"], base["total"]["max_drawdown_net_estimated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
