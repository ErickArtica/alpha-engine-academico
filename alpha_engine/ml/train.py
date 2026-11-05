"""Ejecutar: python -m alpha_engine.ml.train [--days N]."""

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

from alpha_engine.config.settings import PROJECT_ROOT, get_settings
from alpha_engine.ml.trainer import save_training_outcome, train_from_candles
from alpha_engine.repositories.factory import create_market_repository
from alpha_engine.services.ingestion import closed_boundary
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena baseline y XGBoost con velas locales")
    parser.add_argument("--days", type=int, help="Ventana histórica; por defecto BACKFILL_DAYS")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    configure_logging()
    settings = get_settings()
    if settings.interval != "1m":
        raise ValueError("El modelo inicial requiere INTERVAL=1m")
    days = args.days if args.days is not None else settings.backfill_days
    if days < 1:
        raise ValueError("days debe ser >= 1")
    stop = closed_boundary(datetime.now(timezone.utc), settings.interval)
    start = stop - timedelta(days=days)
    repository = create_market_repository(settings)
    try:
        candles = repository.get_candles(settings.symbol, settings.interval, start, stop)
    finally:
        repository.close()
    logger.info("Velas cerradas leídas para entrenamiento: count=%s", len(candles))
    outcome = train_from_candles(candles)
    paths = save_training_outcome(outcome, args.output_dir)
    logger.info(
        "Entrenamiento completado: model=%s test_accuracy=%.4f test_f1=%.4f",
        paths["model"], outcome.metadata["evaluation"]["test"]["accuracy"],
        outcome.metadata["evaluation"]["test"]["f1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

