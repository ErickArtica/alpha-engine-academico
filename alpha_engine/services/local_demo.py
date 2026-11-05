"""Explicit local classroom explanation; never an operational recommendation."""

import re
from datetime import datetime, timezone
from uuid import uuid4

from alpha_engine.ai.context_builder import build_context
from alpha_engine.ai.providers.factory import create_llm_provider
from alpha_engine.ai.schemas import parse_llm_recommendation
from alpha_engine.config.settings import Settings
from alpha_engine.data.news_context import NewsContext
from alpha_engine.repositories.factory import create_market_repository, create_prediction_repository
from alpha_engine.repositories.postgresql import PostgresPrescriptiveNewsRepository
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionResult, PredictionService
from alpha_engine.services.prescriptive_news import PrescriptiveNewsContextService
from alpha_engine.services.recommendation_api import ApiReply
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools import RepositoryAnalyticsTools

_ID = re.compile(r"^[0-9a-f]{64}$")


def demo_enabled(settings: Settings) -> bool:
    return settings.app_env == "local" and settings.storage_provider == "postgresql" and settings.local_demo_mode


def _audit(settings: Settings, prediction_id: str, status: str, news_status: str | None,
           error_type: str | None = None) -> None:
    repo = PostgresPrescriptiveNewsRepository.from_settings(settings)
    try:
        repo._one(
            "INSERT INTO local_demo_requests "
            "(request_id,prediction_id,requested_at,status,news_status,provider,error_type) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING 1",
            (uuid4(), prediction_id, datetime.now(timezone.utc), status,
             news_status, settings.llm_provider, error_type),
        )
    finally:
        repo.close()


def handle_demo_explanation(payload: object, settings: Settings) -> ApiReply:
    if not demo_enabled(settings):
        return ApiReply(404, {"status": "NOT_FOUND"})
    if not isinstance(payload, dict) or set(payload) != {"prediction_id"} or not isinstance(payload["prediction_id"], str):
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    prediction_id = payload["prediction_id"]
    if not _ID.fullmatch(prediction_id):
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    if settings.llm_provider == "nvidia" and not settings.nvidia_api_key.get_secret_value():
        return ApiReply(200, {"status": "LLM_NOT_CONFIGURED", "demo_only": True})
    resources = []
    news = NewsContext(status="UNAVAILABLE", reason="NOT_CONFIGURED")
    try:
        market = create_market_repository(settings); resources.append(market)
        predictions = create_prediction_repository(settings); resources.append(predictions)
        record = predictions.get_by_id(prediction_id)
        if (record is None or record.symbol != settings.symbol
                or record.interval != settings.interval or record.model_version != "xgboost_v1"):
            return ApiReply(404, {"status": "PREDICTION_NOT_FOUND", "demo_only": True})
        # The demo exists precisely for a real prediction below the unchanged gate.
        if max(record.prob_up, record.prob_down) >= settings.model_confidence_threshold:
            return ApiReply(409, {"status": "USE_OPERATIONAL_FLOW", "demo_only": True})
        predictor = PredictionService(market, predictions, settings.model_path,
                                      lookback_candles=settings.prediction_lookback_candles)
        tools = RepositoryAnalyticsTools(
            market, predictor, RiskService(settings), settings.symbol,
            settings.interval, settings.prediction_lookback_candles,
        )
        snapshot = RecommendationContextService(tools).build_snapshot_for_prediction(
            PredictionResult.model_validate(record.model_dump())
        )
        context = build_context(snapshot, settings)
        context["local_demo"] = {
            "demo_only": True, "operational_recommendation": False,
            "gate_met": False, "instruction": "Explain the context; do not recommend an operation.",
        }
        if settings.prescriptive_news_enabled:
            news = PrescriptiveNewsContextService(settings).get_context()
        context["news_context"] = news.for_prompt()
        provider = create_llm_provider(settings); resources.append(provider)
        raw = provider.generate_recommendation(context, timeout_seconds=90)
        parsed = parse_llm_recommendation(raw)
        _audit(settings, prediction_id, "DEMO_ONLY", news.status)
        return ApiReply(200, {
            "status": "DEMO_ONLY", "demo_only": True, "prediction_id": prediction_id,
            "technical_direction": record.predicted_class,
            "gate_met": False, "news_context": news.model_dump(mode="json"),
            "explanation": {
                "summary": parsed.summary, "key_factors": parsed.key_factors,
                "risks": parsed.risks, "warning": parsed.warning,
            },
        })
    except Exception as exc:
        try:
            _audit(settings, prediction_id, "DEMO_FAILED", news.status, type(exc).__name__)
        except Exception:
            pass
        return ApiReply(503, {"status": "DEMO_UNAVAILABLE", "demo_only": True})
    finally:
        for resource in reversed(resources):
            resource.close()
