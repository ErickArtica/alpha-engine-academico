"""Ejecutar: python -m alpha_engine.jobs.backfill [--days N]."""

import argparse
import logging

from alpha_engine.config.settings import get_settings
from alpha_engine.jobs.common import ingestion_service
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill histórico de velas cerradas")
    parser.add_argument("--days", type=int, help="Sobrescribe BACKFILL_DAYS")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    days = args.days if args.days is not None else settings.backfill_days
    with ingestion_service(settings) as service:
        sent = service.backfill(settings.symbol, settings.interval, days)
    logger.info("Backfill completado: points_sent=%s", sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

