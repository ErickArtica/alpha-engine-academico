"""Caché, validación LLM y decisión final con guardrails."""

import json
import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from alpha_engine.ai.context_builder import build_context, context_json
from alpha_engine.ai.guardrails import candidate_signal, pre_llm_guardrails, prediction_expires_at, resolve_final_signal
from alpha_engine.ai.prompts.loader import PROMPT_VERSION
from alpha_engine.ai.providers.base import LLMProvider, LLMProviderError
from alpha_engine.ai.schemas import (
    InvalidLLMResponseError,
    LLMRecommendation,
    parse_llm_recommendation,
)
from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import RecommendationRecord
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.repositories.protocols import RecommendationRepository
from alpha_engine.services.prediction import ModelNotTrainedError, PredictionResult, PredictionUnavailableError
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools.market import AnalyticsDataError

logger = logging.getLogger(__name__)


class RecommendationGenerationError(RuntimeError):
    """El proveedor o la validación fallaron; no hay contenido IA publicable."""


class RecommendationDecision(BaseModel):
    prediction_id: str | None
    timestamp: datetime | None
    generated_at: datetime
    candidate_signal: Literal["BUY", "SELL", "HOLD"]
    llm_signal: Literal["BUY", "SELL", "HOLD"] | None
    final_signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    llm_recommendation: LLMRecommendation | None
    guardrails: list[str]
    context_json: str | None
    llm_model: str
    prompt_version: str
    cached: bool
    trace_id: str | None = None


class RecommendationService:
    def __init__(
        self,
        context_service: RecommendationContextService,
        repository: RecommendationRepository,
        provider: LLMProvider,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
        telemetry: TelemetryStore | None = None,
    ):
        self._context_service = context_service
        self._repository = repository
        self._provider = provider
        self._settings = settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._telemetry = telemetry

    def _event(
        self, stage: str, trace_id: str | None, prediction: PredictionResult | None = None,
        *, duration_ms: float | None = None, details: dict | None = None,
    ) -> None:
        if self._telemetry and trace_id:
            self._telemetry.event(
                stage, trace_id, prediction.symbol if prediction else self._settings.symbol,
                candle_open_time=prediction.timestamp if prediction else None,
                prediction_id=prediction.prediction_id if prediction else None,
                duration_ms=duration_ms, details=details,
            )

    def recommend(self, now: datetime | None = None) -> RecommendationDecision:
        trace_id = uuid4().hex if self._telemetry else None
        context_started = perf_counter()
        self._event("recommendation_started", trace_id)
        current_time = now if now is not None else self._clock()
        if current_time.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        try:
            snapshot = self._context_service.build_snapshot()
        except (ModelNotTrainedError, PredictionUnavailableError, AnalyticsDataError) as exc:
            logger.warning("Recomendación HOLD sin predicción: %s", exc)
            self._event("recommendation_ready", trace_id,
                        duration_ms=(perf_counter() - context_started) * 1000,
                        details={"final_signal": "HOLD", "reason": "MISSING_PREDICTION"})
            return RecommendationDecision(
                prediction_id=None, timestamp=None, generated_at=current_time,
                candidate_signal="HOLD", llm_signal=None, final_signal="HOLD",
                confidence=None, risk_level="HIGH", llm_recommendation=None,
                guardrails=["MISSING_PREDICTION"], context_json=None,
                llm_model=self._provider.model_name, prompt_version=PROMPT_VERSION,
                cached=False, trace_id=trace_id,
            )

        self._event("context_ready", trace_id, snapshot.prediction,
                    duration_ms=(perf_counter() - context_started) * 1000)
        context = build_context(snapshot, self._settings)
        serialized = context_json(context)
        prediction = snapshot.prediction
        candidate = candidate_signal(snapshot, self._settings)
        pre_reasons = pre_llm_guardrails(snapshot, self._settings, current_time)
        cached_record = self._repository.get_cached(
            prediction.prediction_id, PROMPT_VERSION, self._provider.model_name
        )
        llm: LLMRecommendation | None = None
        used_cache = False
        if cached_record and cached_record.context_json == serialized and cached_record.response_json:
            try:
                llm = parse_llm_recommendation(cached_record.response_json)
                used_cache = True
                logger.info("Recomendación IA reutilizada: prediction_id=%s", prediction.prediction_id)
                self._event("llm_cached", trace_id, prediction)
            except InvalidLLMResponseError:
                logger.warning("Caché LLM inválida: prediction_id=%s", prediction.prediction_id)
        # Cualquier guardrail previo fija HOLD. Consultar al LLM no puede
        # cambiar la señal y consume parte de la corta vigencia restante.
        if llm is None:
            if pre_reasons:
                self._event("llm_skipped", trace_id, prediction,
                            details={"reasons": pre_reasons})
            else:
                llm = self._generate_validated(context, trace_id, prediction)

        # Una respuesta remota puede llegar después del cierre de la vela objetivo.
        # En pruebas, `now` fija deliberadamente el reloj simulado.
        if now is None:
            current_time = self._clock()
            pre_reasons = pre_llm_guardrails(snapshot, self._settings, current_time)
        final, reasons = resolve_final_signal(candidate, llm, pre_reasons)
        response_json = llm.model_dump_json() if llm else None
        reasons_json = json.dumps(reasons, separators=(",", ":"), ensure_ascii=False)
        unchanged = (
            cached_record is not None
            and cached_record.context_json == serialized
            and cached_record.candidate_signal == candidate
            and cached_record.llm_signal == (llm.signal if llm else None)
            and cached_record.final_signal == final
            and cached_record.risk_level == snapshot.risk.risk_level
            and cached_record.response_json == response_json
            and cached_record.guardrails_json == reasons_json
        )
        generated_at = cached_record.generated_at if unchanged else current_time
        if not unchanged:
            self._repository.save(RecommendationRecord(
                prediction_id=prediction.prediction_id,
                prompt_version=PROMPT_VERSION,
                llm_model=self._provider.model_name,
                timestamp=prediction.timestamp,
                generated_at=generated_at,
                candidate_signal=candidate,
                llm_signal=llm.signal if llm else None,
                final_signal=final,
                risk_level=snapshot.risk.risk_level,
                context_json=serialized,
                response_json=response_json,
                guardrails_json=reasons_json,
            ))
        if reasons:
            logger.info(
                "Guardrails activados: prediction_id=%s reasons=%s final=%s",
                prediction.prediction_id, reasons_json, final,
            )
        self._event("recommendation_ready", trace_id, prediction,
                    duration_ms=(perf_counter() - context_started) * 1000,
                    details={
                        "final_signal": final,
                        "expires_at": prediction_expires_at(snapshot.market.close_time, prediction.horizon).isoformat(),
                        "cached_llm": used_cache,
                        "guardrail_count": len(reasons),
                    })
        return RecommendationDecision(
            prediction_id=prediction.prediction_id,
            timestamp=prediction.timestamp,
            generated_at=generated_at,
            candidate_signal=candidate,
            llm_signal=llm.signal if llm else None,
            final_signal=final,
            confidence=max(prediction.prob_up, prediction.prob_down),
            risk_level=snapshot.risk.risk_level,
            llm_recommendation=llm,
            guardrails=reasons,
            context_json=serialized,
            llm_model=self._provider.model_name,
            prompt_version=PROMPT_VERSION,
            cached=used_cache, trace_id=trace_id,
        )

    def _generate_validated(
        self, context: dict, trace_id: str | None = None,
        prediction: PredictionResult | None = None,
    ) -> LLMRecommendation:
        for attempt in range(2):
            started = perf_counter()
            self._event("llm_started", trace_id, prediction, details={"attempt": attempt + 1})
            try:
                raw = self._provider.generate_recommendation(context)
            except LLMProviderError as exc:
                self._event("llm_finished", trace_id, prediction,
                            duration_ms=(perf_counter() - started) * 1000,
                            details={"attempt": attempt + 1, "status": "provider_error"})
                raise RecommendationGenerationError("Proveedor LLM no disponible") from exc
            try:
                parsed = parse_llm_recommendation(raw)
            except InvalidLLMResponseError as exc:
                self._event("llm_finished", trace_id, prediction,
                            duration_ms=(perf_counter() - started) * 1000,
                            details={"attempt": attempt + 1, "status": "invalid_json"})
                logger.warning("JSON LLM inválido: attempt=%s model=%s", attempt + 1, self._provider.model_name)
                if attempt == 1:
                    raise RecommendationGenerationError(
                        "El LLM devolvió JSON inválido en dos intentos"
                    ) from exc
            else:
                self._event("llm_finished", trace_id, prediction,
                            duration_ms=(perf_counter() - started) * 1000,
                            details={"attempt": attempt + 1, "status": "ok"})
                return parsed
        raise AssertionError("Bucle de generación inesperado")
