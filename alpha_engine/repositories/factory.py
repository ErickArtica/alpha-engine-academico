"""Punto único de selección del adaptador de persistencia."""

from alpha_engine.config.settings import Settings
from alpha_engine.repositories.influx3 import (
    Influx3MarketRepository, Influx3PredictionRepository, Influx3RecommendationRepository,
    Influx3PipelineRunRepository, Influx3LiveEvaluationRepository,
    Influx3RecommendationRequestRepository,
)
from alpha_engine.repositories.protocols import (
    MarketRepository, PredictionRepository, RecommendationRepository,
    PipelineRunRepository, LiveEvaluationRepository,
    RecommendationRequestRepository,
)


def create_market_repository(settings: Settings) -> MarketRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresMarketRepository
        return PostgresMarketRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabaseMarketRepository
        return SupabaseMarketRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3MarketRepository.from_settings(settings)
    if settings.storage_provider == "influxdb":
        from alpha_engine.repositories.influx import InfluxMarketRepository
        if not settings.influx_token.get_secret_value():
            raise ValueError("INFLUX_TOKEN debe configurarse para conectar con InfluxDB")
        return InfluxMarketRepository.from_settings(settings)
    raise NotImplementedError(
        f"STORAGE_PROVIDER={settings.storage_provider} está reservado para la migración futura"
    )


def create_prediction_repository(settings: Settings) -> PredictionRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresPredictionRepository
        return PostgresPredictionRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabasePredictionRepository
        return SupabasePredictionRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3PredictionRepository.from_settings(settings)
    if settings.storage_provider == "influxdb":
        from alpha_engine.repositories.influx_prediction import InfluxPredictionRepository
        if not settings.influx_token.get_secret_value():
            raise ValueError("INFLUX_TOKEN debe configurarse para conectar con InfluxDB")
        return InfluxPredictionRepository.from_settings(settings)
    raise NotImplementedError(
        f"STORAGE_PROVIDER={settings.storage_provider} está reservado para la migración futura"
    )


def create_recommendation_repository(settings: Settings) -> RecommendationRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresRecommendationRepository
        return PostgresRecommendationRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabaseRecommendationRepository
        return SupabaseRecommendationRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3RecommendationRepository.from_settings(settings)
    if settings.storage_provider == "influxdb":
        from alpha_engine.repositories.influx_recommendation import InfluxRecommendationRepository
        if not settings.influx_token.get_secret_value():
            raise ValueError("INFLUX_TOKEN debe configurarse para conectar con InfluxDB")
        return InfluxRecommendationRepository.from_settings(settings)
    raise NotImplementedError(
        f"STORAGE_PROVIDER={settings.storage_provider} está reservado para la migración futura"
    )


def create_recommendation_request_repository(settings: Settings) -> RecommendationRequestRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresRecommendationRequestRepository
        return PostgresRecommendationRequestRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabaseRecommendationRequestRepository
        return SupabaseRecommendationRequestRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3RecommendationRequestRepository.from_settings(settings)
    raise NotImplementedError("recommendation requests requieren STORAGE_PROVIDER=influxdb3 o supabase")



def create_pipeline_run_repository(settings: Settings) -> PipelineRunRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresPipelineRunRepository
        return PostgresPipelineRunRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabasePipelineRunRepository
        return SupabasePipelineRunRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3PipelineRunRepository.from_settings(settings)
    raise NotImplementedError("pipeline_runs requiere STORAGE_PROVIDER=influxdb3 o supabase")


def create_live_evaluation_repository(settings: Settings) -> LiveEvaluationRepository:
    if settings.storage_provider == "postgresql":
        from alpha_engine.repositories.postgresql import PostgresLiveEvaluationRepository
        return PostgresLiveEvaluationRepository.from_settings(settings)
    if settings.storage_provider == "supabase":
        from alpha_engine.repositories.supabase import SupabaseLiveEvaluationRepository
        return SupabaseLiveEvaluationRepository.from_settings(settings)
    if settings.storage_provider == "influxdb3":
        return Influx3LiveEvaluationRepository.from_settings(settings)
    raise NotImplementedError("live_evaluations requiere STORAGE_PROVIDER=influxdb3 o supabase")
