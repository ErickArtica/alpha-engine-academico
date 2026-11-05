"""Genera o reutiliza la recomendación contextualizada de la última vela."""

import argparse

from alpha_engine.ai.providers.factory import create_llm_provider
from alpha_engine.ai.recommendation_service import RecommendationService
from alpha_engine.config.settings import get_settings
from alpha_engine.data.book_ticker import BinanceBookTickerClient
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.repositories.factory import (
    create_market_repository,
    create_prediction_repository,
    create_recommendation_repository,
)
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionService
from alpha_engine.services.quote_capture import capture_quote
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools import RepositoryAnalyticsTools
from alpha_engine.utils.logging import configure_logging


def main() -> int:
    argparse.ArgumentParser(description="Genera una recomendación académica validada").parse_args()
    configure_logging()
    settings = get_settings()
    telemetry = TelemetryStore()
    market = create_market_repository(settings)
    try:
        predictions = create_prediction_repository(settings)
        try:
            recommendations = create_recommendation_repository(settings)
            try:
                provider = create_llm_provider(settings)
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
                    service = RecommendationService(
                        RecommendationContextService(tools), recommendations,
                        provider, settings, telemetry=telemetry,
                    )
                    decision = service.recommend()
                    print(decision.model_dump_json(indent=2))
                    if decision.trace_id:
                        telemetry.event("cli_returned", decision.trace_id, settings.symbol,
                                        candle_open_time=decision.timestamp,
                                        prediction_id=decision.prediction_id)
                        quote_client = BinanceBookTickerClient(settings.binance_base_url)
                        try:
                            capture_quote(quote_client, telemetry, settings.symbol,
                                          decision.trace_id, decision.prediction_id)
                        finally:
                            quote_client.close()
                finally:
                    provider.close()
            finally:
                recommendations.close()
        finally:
            predictions.close()
    finally:
        market.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
