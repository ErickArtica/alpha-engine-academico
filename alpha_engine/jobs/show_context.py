"""Imprime el contexto estructurado que recibirá el LLM, sin llamarlo."""

import argparse
import logging

from alpha_engine.config.settings import get_settings
from alpha_engine.repositories.factory import (
    create_market_repository,
    create_prediction_repository,
)
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import ModelNotTrainedError, PredictionService
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools import RepositoryAnalyticsTools
from alpha_engine.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    argparse.ArgumentParser(description="Imprime el contexto JSON para IA").parse_args()
    configure_logging()
    settings = get_settings()
    market = create_market_repository(settings)
    try:
        predictions = create_prediction_repository(settings)
        try:
            prediction_service = PredictionService(
                market, predictions, settings.model_path,
                lookback_candles=settings.prediction_lookback_candles,
            )
            tools = RepositoryAnalyticsTools(
                market, prediction_service, RiskService(settings),
                settings.symbol, settings.interval,
                lookback_candles=settings.prediction_lookback_candles,
            )
            try:
                print(RecommendationContextService(tools).build_context_json())
            except ModelNotTrainedError as exc:
                logger.error("%s", exc)
                return 2
        finally:
            predictions.close()
    finally:
        market.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

