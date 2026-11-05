"""Ejecutar: python -m alpha_engine.jobs.predict_latest."""

import argparse
import logging

from alpha_engine.config.settings import get_settings
from alpha_engine.repositories.factory import (
    create_market_repository,
    create_prediction_repository,
)
from alpha_engine.services.prediction import ModelNotTrainedError, PredictionService
from alpha_engine.services.prediction import PredictionResult
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def run_once() -> PredictionResult:
    """Predice la última vela cerrada; reutiliza el registro si ya existe."""
    settings = get_settings()
    market = create_market_repository(settings)
    try:
        predictions = create_prediction_repository(settings)
        try:
            service = PredictionService(
                market, predictions, settings.model_path,
                lookback_candles=settings.prediction_lookback_candles,
            )
            return service.predict_latest(settings.symbol, settings.interval)
        finally:
            predictions.close()
    finally:
        market.close()


def main() -> int:
    argparse.ArgumentParser(description="Predice y guarda la última vela cerrada").parse_args()
    configure_logging()
    try:
        result = run_once()
    except ModelNotTrainedError as exc:
        logger.error("%s", exc)
        return 2
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
