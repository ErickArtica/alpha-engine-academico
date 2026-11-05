"""Ejecutar: python -m alpha_engine.jobs.fetch_latest."""

import argparse
import logging

from alpha_engine.config.settings import get_settings
from alpha_engine.jobs.common import ingestion_service
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def run_once() -> int:
    settings = get_settings()
    with ingestion_service(settings) as service:
        sent = service.fetch_latest(settings.symbol, settings.interval)
    logger.info("Actualización completada: points_sent=%s", sent)
    return sent


def main() -> int:
    argparse.ArgumentParser(description="Obtiene velas cerradas nuevas").parse_args()
    configure_logging()
    run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

