"""Valida el proveedor Ollama con contexto real; no publica una señal."""

from alpha_engine.ai.context_builder import build_context
from alpha_engine.ai.providers.factory import create_llm_provider
from alpha_engine.ai.schemas import parse_llm_recommendation
from alpha_engine.config.settings import get_settings
from alpha_engine.repositories.factory import create_market_repository, create_prediction_repository
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionService
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools import RepositoryAnalyticsTools


def main() -> None:
    settings = get_settings()
    market = create_market_repository(settings)
    predictions = create_prediction_repository(settings)
    provider = create_llm_provider(settings)
    try:
        prediction_service = PredictionService(market, predictions, settings.model_path, settings.prediction_lookback_candles)
        tools = RepositoryAnalyticsTools(
            market, prediction_service, RiskService(settings), settings.symbol, settings.interval,
            lookback_candles=settings.prediction_lookback_candles,
        )
        context = build_context(RecommendationContextService(tools).build_snapshot())
        result = parse_llm_recommendation(provider.generate_recommendation(context))
        print(f"Ollama validado: model={provider.model_name} signal={result.signal} confidence={result.confidence}")
        print("La salida de esta prueba no se guarda ni reemplaza los guardrails.")
    finally:
        provider.close()
        predictions.close()
        market.close()


if __name__ == "__main__":
    main()
