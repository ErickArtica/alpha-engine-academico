"""Scheduler local: ingesta y predicción cada minuto a los 5 segundos UTC."""

import argparse
import logging
from datetime import timezone
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from alpha_engine.config.settings import get_settings
from alpha_engine.data.book_ticker import BinanceBookTickerClient
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.services.live_cycle import LiveCycle, run_live_cycle
from alpha_engine.services.quote_capture import capture_quote
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def scheduled_fetch(
    telemetry: TelemetryStore | None = None,
    quote_client: BinanceBookTickerClient | None = None,
    symbol: str | None = None,
    *,
    cycle: LiveCycle | None = None,
) -> bool:
    try:
        result = cycle.run() if cycle is not None else run_live_cycle(telemetry=telemetry)
        logger.info("Ciclo LIVE: status=%s candle=%s prediction_id=%s",
                    result.status, result.candle.open_time.isoformat() if result.candle else None,
                    result.prediction_id)
        if result.prediction and telemetry and quote_client and result.trace_id:
            capture_quote(quote_client, telemetry, result.prediction.symbol,
                          result.trace_id, result.prediction_id)
        return True
    except Exception:
        logger.exception("Fallo la actualizacion programada; se reintentara el proximo minuto")
        return False


def build_scheduler(job: Callable[[], None] = scheduled_fetch) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=timezone.utc)
    scheduler.add_job(
        job,
        trigger=CronTrigger(second=5, timezone=timezone.utc),
        id="fetch_latest_candles",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    return scheduler


def main() -> int:
    parser = argparse.ArgumentParser(description="Actualización de velas cada minuto")
    parser.add_argument("--once", action="store_true", help="Ejecuta una sola ingesta, predicción y snapshot")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    telemetry = TelemetryStore()
    quote_client = BinanceBookTickerClient(settings.binance_base_url)
    try:
        with LiveCycle.from_settings(settings, telemetry=telemetry) as cycle:
            if args.once:
                return 0 if scheduled_fetch(telemetry, quote_client, settings.symbol, cycle=cycle) else 1
            scheduler = build_scheduler(
                lambda: scheduled_fetch(telemetry, quote_client, settings.symbol, cycle=cycle)
            )
            logger.info("Scheduler iniciado: ciclo LIVE cada minuto a los 5 segundos UTC")
            scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler detenido")
    finally:
        quote_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
