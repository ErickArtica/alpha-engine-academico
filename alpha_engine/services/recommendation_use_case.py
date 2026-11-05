"""Explicit, on-demand prescriptive requests. It never runs from LiveCycle."""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from alpha_engine.ai.context_builder import build_context, context_json
from alpha_engine.ai.guardrails import candidate_signal, pre_llm_guardrails, resolve_final_signal
from alpha_engine.ai.prompts.loader import PROMPT_VERSION
from alpha_engine.ai.providers.base import LLMProvider, LLMProviderError, LLMProviderTimeoutError
from alpha_engine.ai.schemas import InvalidLLMResponseError, LLMRecommendation, parse_llm_recommendation
from alpha_engine.config.settings import Settings, get_settings
from alpha_engine.data.news_context import NewsContext
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord, RecommendationRequestStatus
from alpha_engine.data.recommendation_response import RecommendationResult, RecommendationView
from alpha_engine.data.schemas import PredictionRecord, RecommendationRecord
from alpha_engine.repositories.factory import (
    create_market_repository, create_prediction_repository, create_recommendation_repository,
    create_recommendation_request_repository,
)
from alpha_engine.repositories.protocols import (
    MarketRepository, PredictionRepository, RecommendationRepository, RecommendationRequestRepository,
)
from alpha_engine.risk import RiskService
from alpha_engine.services.errors import StorageError
from alpha_engine.services.prediction import PredictionResult
from alpha_engine.services.prescriptive_news import PrescriptiveNewsContextService
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools import RepositoryAnalyticsTools
from alpha_engine.tools.market import AnalyticsDataError

logger = logging.getLogger(__name__)

_ERRORS = {
    "LLMProviderTimeoutError": "LLM request exceeded its deadline",
    "LLMProviderError": "LLM provider request failed",
    "InvalidLLMResponseError": "LLM response did not satisfy the required schema",
    "AnalyticsDataError": "Required historical context is unavailable",
    "ValidationError": "Prediction or context validation failed",
}


class RecommendationUseCase:
    def __init__(
        self, context_service: RecommendationContextService,
        predictions: PredictionRepository, recommendations: RecommendationRepository,
        requests: RecommendationRequestRepository, provider: LLMProvider, settings: Settings,
        *, clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = perf_counter,
        news_service: PrescriptiveNewsContextService | None = None,
    ):
        self._context = context_service
        self._predictions = predictions
        self._recommendations = recommendations
        self._requests = requests
        self._provider = provider
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic
        self._news_service = news_service

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RecommendationUseCase":
        selected = settings or get_settings()
        resources = []
        try:
            market = create_market_repository(selected); resources.append(market)
            predictions = create_prediction_repository(selected); resources.append(predictions)
            recommendations = create_recommendation_repository(selected); resources.append(recommendations)
            requests = create_recommendation_request_repository(selected); resources.append(requests)
            from alpha_engine.ai.providers.factory import create_llm_provider
            try:
                provider = create_llm_provider(selected)
            except Exception:
                if not (selected.app_env == "local" and selected.storage_provider == "postgresql"
                        and selected.llm_provider == "nvidia"
                        and not selected.nvidia_api_key.get_secret_value()):
                    raise
                from alpha_engine.ai.providers.base import LLMProviderError

                class _UnconfiguredLocalProvider:
                    model_name = selected.nvidia_model

                    def close(self):
                        pass

                    def generate_recommendation(self, *_args, **_kwargs):
                        raise LLMProviderError("LLM_NOT_CONFIGURED")

                provider = _UnconfiguredLocalProvider()
            resources.append(provider)
            from alpha_engine.services.prediction import PredictionService
            tools = RepositoryAnalyticsTools(
                market, PredictionService(market, predictions, selected.model_path,
                lookback_candles=selected.prediction_lookback_candles), RiskService(selected),
                selected.symbol, selected.interval, selected.prediction_lookback_candles,
            )
            instance = cls(RecommendationContextService(tools), predictions, recommendations,
                           requests, provider, selected,
                           news_service=(PrescriptiveNewsContextService(selected)
                                         if selected.prescriptive_news_enabled else None))
            instance._owned_resources = resources
            return instance
        except Exception:
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception:
                    logger.exception("Could not close prescriptive resource")
            raise

    def close(self) -> None:
        for resource in reversed(getattr(self, "_owned_resources", [])):
            try:
                resource.close()
            except Exception:
                logger.exception("Could not close prescriptive resource")
        self._owned_resources = []

    def __enter__(self) -> "RecommendationUseCase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("El reloj prescriptivo debe incluir zona horaria")
        return value.astimezone(timezone.utc)

    def _result(
        self, *, request_id: UUID, prediction_id: str | None, status: RecommendationRequestStatus,
        requested_at: datetime, started_tick: float, valid_until: datetime | None = None,
        generated_at: datetime | None = None, llm_latency_ms: float | None = None,
        cached: bool = False, recommendation: RecommendationView | None = None,
        guardrails: list[str] | None = None, error_type: str | None = None,
        error_message: str | None = None,
        news_context: NewsContext | None = None,
    ) -> RecommendationResult:
        finished_at = max(requested_at, self._now())
        timing = None
        if generated_at is not None and valid_until is not None:
            timing = "TIMELY" if generated_at < valid_until else "LATE"
        result = RecommendationResult(
            request_id=request_id, prediction_id=prediction_id, status=status,
            requested_at=requested_at, generated_at=generated_at, finished_at=finished_at,
            valid_until=valid_until, timing_status=timing, llm_latency_ms=llm_latency_ms,
            total_latency_ms=max(0.0, (self._monotonic() - started_tick) * 1000),
            cached=cached,
            actionable=status in ("SUCCESS", "CACHED") and timing == "TIMELY"
            and (valid_until is None or finished_at < valid_until),
            recommendation=recommendation, guardrails=guardrails or [],
            news_context=news_context,
        )
        # A cached recommendation can have been generated before this request.
        # The request audit records when *this* response became available, while
        # RecommendationResult preserves the original recommendation timestamp.
        audit_generated_at = generated_at
        if audit_generated_at is not None and audit_generated_at < requested_at:
            audit_generated_at = requested_at
        audit = RecommendationRequestRecord(
            request_id=request_id, prediction_id=prediction_id, symbol=self._settings.symbol,
            interval=self._settings.interval, provider=self._settings.llm_provider,
            llm_model=self._provider.model_name, prompt_version=PROMPT_VERSION,
            requested_at=requested_at, generated_at=audit_generated_at, finished_at=finished_at,
            valid_until=valid_until, llm_latency_ms=llm_latency_ms,
            total_latency_ms=result.total_latency_ms, status=status, timing_status=timing,
            cached=cached, error_type=error_type, error_message=error_message,
            created_at=finished_at,
            news_context_id=news_context.snapshot_id if news_context else None,
            news_context_used=news_context.available if news_context else False,
            news_context_fetched_at=news_context.fetched_at if news_context else None,
            news_context_count=news_context.news_count if news_context and news_context.available else None,
            news_context_sentiment=(news_context.btc_sentiment_score
                                    if news_context and news_context.available else None),
        )
        try:
            self._requests.save(audit)
        except Exception as exc:
            logger.exception("Could not persist recommendation request; review local logs")
            raise StorageError("No se pudo guardar la auditoria de recomendacion") from exc
        return result

    @staticmethod
    def _view(record: RecommendationRecord) -> RecommendationView:
        explanation = None
        if record.response_json:
            try:
                explanation = parse_llm_recommendation(record.response_json)
            except InvalidLLMResponseError:
                pass
        return RecommendationView(
            prediction_id=record.prediction_id, candidate_signal=record.candidate_signal,
            llm_signal=record.llm_signal, final_signal=record.final_signal,
            risk_level=record.risk_level, llm_model=record.llm_model,
            prompt_version=record.prompt_version,
            summary=explanation.summary if explanation else None,
            key_factors=explanation.key_factors if explanation else [],
            risks=explanation.risks if explanation else [],
            warning=explanation.warning if explanation else None,
        )

    def _resolve_prediction(self, prediction_id: str | None, now: datetime) -> PredictionRecord | None:
        if prediction_id is not None:
            return self._predictions.get_by_id(prediction_id)
        for prediction in self._predictions.list_recent(self._settings.symbol, limit=100):
            if prediction.interval == self._settings.interval and prediction.current_validity(now) == "ACTIVE":
                return prediction
        return None

    def request(self, prediction_id: str | None = None, *, now: datetime | None = None,
                portfolio_id: str | None = None) -> RecommendationResult:
        request_id = uuid4()
        if now is not None and now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        requested_at = (now.astimezone(timezone.utc) if now is not None else self._now())
        started_tick = self._monotonic()
        try:
            record = self._resolve_prediction(prediction_id, requested_at)
        except Exception as exc:
            return self._result(
                request_id=request_id, prediction_id=prediction_id, status="PERSISTENCE_ERROR",
                requested_at=requested_at, started_tick=started_tick,
                error_type=type(exc).__name__, error_message="Prediction lookup failed",
            )
        if record is None:
            return self._result(
                request_id=request_id, prediction_id=prediction_id,
                status="PREDICTION_NOT_FOUND" if prediction_id else "NO_ACTIVE_PREDICTION",
                requested_at=requested_at, started_tick=started_tick,
            )
        if (record.symbol != self._settings.symbol or record.interval != self._settings.interval
                or not record.model_version.strip() or record.generated_at is None
                or record.valid_until is None):
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PREDICTION_INVALID",
                requested_at=requested_at, started_tick=started_tick,
            )
        if record.current_validity(requested_at) != "ACTIVE":
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id,
                status="EXPIRED_BEFORE_REQUEST", requested_at=requested_at,
                started_tick=started_tick, valid_until=record.valid_until,
            )
        # The experimental gate is a prescriptive abstention, not a third ML
        # class. Audit it before building context or fetching optional news.
        if max(record.prob_up, record.prob_down) < self._settings.model_confidence_threshold:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id,
                status="GUARDRAIL_REJECTED", requested_at=requested_at,
                started_tick=started_tick, valid_until=record.valid_until,
                guardrails=["LOW_MODEL_CONFIDENCE"],
            )
        remaining = (record.valid_until - requested_at).total_seconds()
        effective_timeout = min(
            float(self._settings.llm_live_timeout_seconds),
            remaining - self._settings.llm_deadline_margin_seconds,
        )
        if effective_timeout <= 0:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id,
                status="DEADLINE_TOO_CLOSE", requested_at=requested_at,
                started_tick=started_tick, valid_until=record.valid_until,
            )
        try:
            prediction = PredictionResult.model_validate(record.model_dump())
            snapshot = self._context.build_snapshot_for_prediction(prediction)
            context = build_context(snapshot, self._settings)
            from alpha_engine.services.paper_policy import paper_decision

            has_long = False
            if portfolio_id is not None:
                from alpha_engine.repositories.paper import create_paper_repository

                paper = create_paper_repository(self._settings)
                try:
                    account = paper.get(UUID(portfolio_id))
                finally:
                    paper.close()
                if account is None:
                    raise ValueError("Virtual portfolio not found")
                has_long = float(account["btc"]) > 0
            action = paper_decision(record, has_long=has_long,
                                    threshold=self._settings.model_confidence_threshold,
                                    now=requested_at)
            context["decision_policy"]["action"] = action.action
            context["decision_policy"]["position"] = "LONG" if has_long else "SIN_POSICION"
            context["decision_policy"]["action_reason"] = action.reason
            mapped_action = action.action if action.action != "WAIT" else "HOLD"
        except (AnalyticsDataError, ValidationError, ValueError) as exc:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PREDICTION_INVALID",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                error_type=type(exc).__name__, error_message=_ERRORS.get(type(exc).__name__, "Context validation failed"),
            )
        except Exception as exc:
            logger.exception("Could not build prescriptive context")
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PERSISTENCE_ERROR",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                error_type=type(exc).__name__, error_message="Context retrieval failed",
            )
        pre_reasons = pre_llm_guardrails(snapshot, self._settings, requested_at)
        if pre_reasons:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="GUARDRAIL_REJECTED",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                guardrails=pre_reasons,
            )
        news_context = None
        if self._news_service is not None:
            try:
                news_context = self._news_service.get_context(deadline_at=record.valid_until)
            except Exception:
                # News is optional context; an unexpected adapter failure must
                # not turn a valid technical request into a failed request.
                logger.warning("Optional news context unavailable")
                news_context = NewsContext(status="UNAVAILABLE", reason="PROVIDER_UNAVAILABLE")
        if news_context is not None:
            context["news_context"] = news_context.for_prompt()
        serialized = context_json(context)
        if self._news_service is not None:
            effective_timeout = min(
                effective_timeout,
                (record.valid_until - self._now()).total_seconds()
                - self._settings.llm_deadline_margin_seconds,
            )
        if effective_timeout <= 0:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id,
                status="DEADLINE_TOO_CLOSE", requested_at=requested_at,
                started_tick=started_tick, valid_until=record.valid_until,
                news_context=news_context,
            )
        cached = self._recommendations.get_cached(record.prediction_id, PROMPT_VERSION, self._provider.model_name)
        if (cached is not None and cached.context_json == serialized and cached.response_json
                and cached.final_signal == mapped_action
                and cached.timing_status == "TIMELY"
                and cached.current_validity(requested_at) == "ACTIVE"):
            try:
                # Cached JSON was validated before storage; parse it again before use.
                parse_llm_recommendation(cached.response_json)
            except InvalidLLMResponseError:
                cached = None
            else:
                return self._result(
                    request_id=request_id, prediction_id=record.prediction_id, status="CACHED",
                    requested_at=requested_at, started_tick=started_tick,
                    valid_until=record.valid_until, generated_at=cached.generated_at,
                    llm_latency_ms=cached.llm_latency_ms, cached=True,
                    recommendation=self._view(cached),
                    news_context=news_context,
                )
        llm_started = self._monotonic()
        try:
            raw = self._provider.generate_recommendation(context, timeout_seconds=effective_timeout)
        except LLMProviderTimeoutError as exc:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="LLM_TIMEOUT",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                llm_latency_ms=max(0.0, (self._monotonic() - llm_started) * 1000),
                error_type=type(exc).__name__, error_message=_ERRORS[type(exc).__name__],
                news_context=news_context,
            )
        except LLMProviderError as exc:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PROVIDER_ERROR",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                llm_latency_ms=max(0.0, (self._monotonic() - llm_started) * 1000),
                error_type=type(exc).__name__, error_message=_ERRORS["LLMProviderError"],
                news_context=news_context,
            )
        except Exception as exc:
            logger.exception("Unexpected LLM provider failure")
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PROVIDER_ERROR",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                llm_latency_ms=max(0.0, (self._monotonic() - llm_started) * 1000),
                error_type=type(exc).__name__, error_message="LLM provider request failed",
                news_context=news_context,
            )
        llm_latency_ms = max(0.0, (self._monotonic() - llm_started) * 1000)
        try:
            llm = parse_llm_recommendation(raw)
        except InvalidLLMResponseError as exc:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="INVALID_RESPONSE",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                llm_latency_ms=llm_latency_ms, error_type=type(exc).__name__,
                error_message=_ERRORS[type(exc).__name__],
                news_context=news_context,
            )
        generated_at = max(requested_at, self._now())
        final, reasons = resolve_final_signal(mapped_action, llm, [])
        saved = RecommendationRecord(
            prediction_id=record.prediction_id, prompt_version=PROMPT_VERSION,
            llm_model=self._provider.model_name, timestamp=record.timestamp,
            requested_at=requested_at, generated_at=generated_at, valid_until=record.valid_until,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=max(0.0, (self._monotonic() - started_tick) * 1000),
            candidate_signal=mapped_action, llm_signal=llm.signal,
            final_signal=final, risk_level=snapshot.risk.risk_level,
            context_json=serialized, response_json=llm.model_dump_json(),
            guardrails_json=json.dumps(reasons, separators=(",", ":"), ensure_ascii=False),
        )
        try:
            # A late or expired cache entry is retained in history but cannot
            # block a fresh, currently useful result from replacing the cache.
            # With no prior entry, the repository's unique constraint/upsert is
            # the concurrency boundary.
            inserted = True
            if cached is not None:
                self._recommendations.save(saved)
            else:
                inserted = self._recommendations.save_if_absent(saved)
            if inserted is False:
                winner = self._recommendations.get_cached(
                    record.prediction_id, PROMPT_VERSION, self._provider.model_name
                )
                if winner is None:
                    raise RuntimeError("Concurrent recommendation was not readable")
                if winner.context_json != serialized or winner.final_signal != mapped_action:
                    # A cached result for another virtual position or an old
                    # interpretation cannot take authority over this action.
                    self._recommendations.save(saved)
                    winner = saved
                saved = winner
                generated_at = saved.generated_at
                if saved.timing_status == "TIMELY" and saved.current_validity(self._now()) == "ACTIVE":
                    return self._result(
                        request_id=request_id, prediction_id=record.prediction_id, status="CACHED",
                        requested_at=requested_at, started_tick=started_tick,
                        valid_until=record.valid_until, generated_at=generated_at,
                        llm_latency_ms=llm_latency_ms, cached=True, recommendation=self._view(saved),
                        news_context=news_context,
                    )
        except Exception as exc:
            return self._result(
                request_id=request_id, prediction_id=record.prediction_id, status="PERSISTENCE_ERROR",
                requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
                generated_at=generated_at, llm_latency_ms=llm_latency_ms,
                error_type=type(exc).__name__, error_message="Recommendation persistence failed",
                news_context=news_context,
            )
        timing = "TIMELY" if generated_at < record.valid_until else "LATE"
        status: RecommendationRequestStatus = (
            "LATE_RESULT" if timing == "LATE" else "GUARDRAIL_REJECTED" if reasons else "SUCCESS"
        )
        return self._result(
            request_id=request_id, prediction_id=record.prediction_id, status=status,
            requested_at=requested_at, started_tick=started_tick, valid_until=record.valid_until,
            generated_at=generated_at, llm_latency_ms=llm_latency_ms,
            recommendation=self._view(saved), guardrails=reasons,
            news_context=news_context,
        )
