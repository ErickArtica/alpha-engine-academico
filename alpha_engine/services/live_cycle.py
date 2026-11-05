"""Provider-neutral LIVE cycle: closed Binance candles to one ML prediction."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Literal
from uuid import UUID, uuid4

from alpha_engine.config.settings import Settings, get_settings
from alpha_engine.data.binance import BinanceKlineClient
from alpha_engine.data.schemas import Candle
from alpha_engine.data.live_schemas import PipelineRunRecord
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.observability.process_memory import memory_event
from alpha_engine.repositories.factory import (
    create_market_repository, create_prediction_repository,
    create_pipeline_run_repository, create_live_evaluation_repository,
)
from alpha_engine.repositories.protocols import (
    MarketRepository, PredictionRepository, PipelineRunRepository, LiveEvaluationRepository,
)
from alpha_engine.services.errors import StorageError
from alpha_engine.services.live_evaluation import LiveEvaluationService
from alpha_engine.services.ingestion import KlineSource, MarketIngestionService, closed_boundary, interval_duration
from alpha_engine.services.prediction import (
    InsufficientHistoryError, PredictionResult, PredictionService, load_model_bundle,
)
from alpha_engine.services.v3_shadow import V3Shadow

CycleStatus = Literal[
    "SUCCESS", "NO_NEW_CANDLE", "PREDICTION_ALREADY_EXISTS", "INSUFFICIENT_HISTORY",
]
logger = logging.getLogger(__name__)
timing_logger = logging.getLogger("uvicorn.error")


class LiveCycleResources:
    """Process-owned resources used only by serialized LIVE cycles."""

    def __init__(self, settings: Settings):
        if settings.storage_provider != "supabase":
            raise ValueError("Shared LIVE resources require Supabase")
        from alpha_engine.repositories.supabase import _client_from_settings

        self.client = _client_from_settings(settings)
        self._bundle: dict | None = None
        self._model_path: Path | None = None
        self._lock = RLock()
        self._closed = False
        memory_event("supabase_client_initialized")

    def model_bundle(self, path: Path) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("LIVE resources are closed")
            resolved = path.resolve()
            if self._model_path is not None and resolved != self._model_path:
                raise ValueError("LIVE model path changed during process lifetime")
            if self._bundle is None:
                self._bundle = load_model_bundle(resolved)
                self._model_path = resolved
                memory_event("model_initialized")
            return self._bundle

    def run(self, settings: Settings, *, now: datetime | None = None,
            telemetry: TelemetryStore | None = None) -> "LiveCycleResult":
        # The Supabase sync client and estimator are never accessed concurrently.
        with self._lock:
            if self._closed:
                raise RuntimeError("LIVE resources are closed")
            with LiveCycle.from_settings(settings, telemetry=telemetry,
                                         resources=self) as cycle:
                return cycle.run(now=now)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.client.postgrest.aclose()


class _TimedIO:
    """Count time spent in external calls without changing repository behavior."""

    def __init__(self, target: object, field: str, counters: dict[str, float]):
        self._target = target
        self._field = field
        self._counters = counters

    def __getattr__(self, name: str):
        attribute = getattr(self._target, name)
        if not callable(attribute) or name == "close":
            return attribute

        def measured(*args, **kwargs):
            started = perf_counter()
            try:
                return attribute(*args, **kwargs)
            finally:
                self._counters[self._field] += (perf_counter() - started) * 1000

        return measured

_SAFE_ERRORS = {
    "MarketDataError": "Market data request failed",
    "StorageError": "Storage operation failed",
    "ModelNotTrainedError": "Model artifact unavailable",
    "PredictionUnavailableError": "Prediction validation failed",
    "ValueError": "Invalid market data or configuration",
}


@dataclass(frozen=True)
class LiveCycleResult:
    status: CycleStatus
    as_of: datetime
    candle: Candle | None
    latest_stored_candle: Candle | None
    prediction: PredictionResult | None
    candles_ingested: int
    new_candle: bool
    prediction_created: bool
    completed: bool = True
    trace_id: str | None = None
    run_id: str | None = None
    duration_ms: float | None = None

    @property
    def prediction_id(self) -> str | None:
        return self.prediction.prediction_id if self.prediction else None


class LiveCycle:
    """Reusable cycle. A persistent trigger may retain this instance and model."""

    def __init__(
        self,
        settings: Settings,
        source: KlineSource,
        market: MarketRepository,
        predictions: PredictionRepository,
        *,
        clock: Callable[[], datetime] | None = None,
        model_loader: Callable[[Path], dict] = load_model_bundle,
        telemetry: TelemetryStore | None = None,
        runs: PipelineRunRepository | None = None,
        evaluations: LiveEvaluationRepository | None = None,
        monotonic: Callable[[], float] = perf_counter,
    ):
        self.settings = settings
        self._io_ms = {"binance_ms": 0.0, "db_ms": 0.0}
        self._source = _TimedIO(source, "binance_ms", self._io_ms)
        self._market = _TimedIO(market, "db_ms", self._io_ms)
        self._predictions = _TimedIO(predictions, "db_ms", self._io_ms)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._telemetry = telemetry
        self._runs = _TimedIO(runs, "db_ms", self._io_ms) if runs else None
        self._evaluations = _TimedIO(evaluations, "db_ms", self._io_ms) if evaluations else None
        self._evaluation_service = (
            LiveEvaluationService(self._market, self._predictions, self._evaluations)
            if evaluations else None
        )
        self._monotonic = monotonic
        self._ingestion = MarketIngestionService(self._source, self._market)
        self._predictor = PredictionService(
            self._market, self._predictions, settings.model_path,
            lookback_candles=settings.prediction_lookback_candles,
            loader=model_loader,
        )
        self._v3_shadow = V3Shadow(self._source) if settings.v3_shadow_enabled else None
        self._closed = False

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, telemetry: TelemetryStore | None = None,
        clock: Callable[[], datetime] | None = None,
        model_loader: Callable[[Path], dict] = load_model_bundle,
        resources: LiveCycleResources | None = None,
    ) -> "LiveCycle":
        selected = settings or get_settings()
        opened = []
        try:
            if resources is not None:
                from alpha_engine.repositories.supabase import (
                    SupabaseMarketRepository, SupabasePredictionRepository,
                    SupabasePipelineRunRepository, SupabaseLiveEvaluationRepository,
                )
                market = SupabaseMarketRepository(resources.client)
                predictions = SupabasePredictionRepository(resources.client)
                runs = SupabasePipelineRunRepository(resources.client)
                evaluations = SupabaseLiveEvaluationRepository(resources.client)
                model_loader = resources.model_bundle
                opened.extend((market, predictions, runs, evaluations))
            else:
                market = create_market_repository(selected)
                opened.append(market)
                predictions = create_prediction_repository(selected)
                opened.append(predictions)
                runs = create_pipeline_run_repository(selected)
                opened.append(runs)
                evaluations = create_live_evaluation_repository(selected)
                opened.append(evaluations)
            source = BinanceKlineClient(selected.binance_base_url)
            opened.append(source)
            return cls(selected, source, market, predictions, clock=clock,
                       model_loader=model_loader, telemetry=telemetry,
                       runs=runs, evaluations=evaluations)
        except Exception:
            for resource in reversed(opened):
                try:
                    resource.close()
                except Exception:
                    logger.exception("Fallo al cerrar recurso durante la inicializacion")
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error = None
        for resource in (
            self._source, self._evaluations, self._runs, self._predictions, self._market
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.exception("Fallo al cerrar recurso del ciclo LIVE")
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "LiveCycle":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _event(self, stage: str, trace_id: str | None, *, at: datetime | None = None,
               candle: Candle | None = None, prediction_id: str | None = None,
               duration_ms: float | None = None, details: dict | None = None) -> None:
        if self._telemetry and trace_id:
            self._telemetry.event(
                stage, trace_id, self.settings.symbol, at=at,
                candle_open_time=candle.open_time if candle else None,
                prediction_id=prediction_id, duration_ms=duration_ms, details=details,
            )

    def _latest_eligible(self, latest: Candle | None, now: datetime) -> Candle | None:
        boundary = closed_boundary(now, self.settings.interval)

        def eligible(candle: Candle) -> bool:
            return (candle.symbol == self.settings.symbol
                    and candle.interval == self.settings.interval
                    and candle.open_time < boundary
                    and candle.close_time < boundary
                    and candle.close_time < now)

        if latest is not None and eligible(latest):
            return latest
        step = interval_duration(self.settings.interval)
        try:
            nearby = self._market.get_candles(
                self.settings.symbol, self.settings.interval, boundary - 2 * step, boundary
            )
        except Exception as exc:
            raise StorageError("No se pudieron consultar velas elegibles") from exc
        return max((c for c in nearby if eligible(c)), key=lambda c: c.open_time, default=None)

    def _utc_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("El reloj del ciclo debe incluir zona horaria")
        return value.astimezone(timezone.utc)

    def _finished_at(self, instant: datetime, fixed_now: bool) -> datetime:
        value = instant if fixed_now else self._utc_now()
        return max(instant, value)

    def _persist_run(
        self, *, run_id: UUID, started_at: datetime, finished_at: datetime,
        duration_ms: float, progress: dict, status: str,
        error_type: str | None = None, error_message: str | None = None,
        evaluation_error_type: str | None = None,
        evaluation_error_message: str | None = None,
    ) -> None:
        if self._runs is None:
            return
        record = PipelineRunRecord(
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            duration_ms=duration_ms, market_data_ms=progress.get("market_data_ms"),
            prediction_ms=progress.get("prediction_ms"),
            evaluation_ms=progress.get("evaluation_ms"), status=status,
            symbol=self.settings.symbol, interval=self.settings.interval,
            candle_open_time=progress.get("candle_open_time"),
            prediction_id=progress.get("prediction_id"),
            new_candle=progress.get("new_candle", False),
            new_prediction=progress.get("new_prediction", False),
            candles_ingested=progress.get("candles_ingested", 0),
            error_type=error_type, error_message=error_message,
            evaluation_error_type=evaluation_error_type,
            evaluation_error_message=evaluation_error_message,
            created_at=finished_at,
        )
        self._runs.save(record)

    def run(self, now: datetime | None = None) -> LiveCycleResult:
        if self._closed:
            raise RuntimeError("LiveCycle ya esta cerrado")
        self._io_ms.update(binance_ms=0.0, db_ms=0.0)
        self._predictor.last_features_ms = 0.0
        timing_started = perf_counter()
        started_tick = self._monotonic()
        run_id = uuid4()
        memory_event("cycle_begin", run_id=str(run_id))
        progress: dict = {}
        try:
            instant = now if now is not None else self._utc_now()
            if instant.tzinfo is None:
                raise ValueError("now debe incluir zona horaria")
            instant = instant.astimezone(timezone.utc)
        except Exception as exc:
            fallback = datetime.now(timezone.utc)
            try:
                self._persist_run(
                    run_id=run_id, started_at=fallback, finished_at=fallback,
                    duration_ms=max(0.0, (self._monotonic() - started_tick) * 1000),
                    progress=progress, status="FAILED", error_type=type(exc).__name__,
                    error_message="Invalid cycle clock or timestamp",
                )
            except Exception:
                logger.exception("No se pudo persistir el run fallido; revisar logs locales")
            self._log_timing(run_id, "FAILED", timing_started)
            memory_event("cycle_end", run_id=str(run_id))
            raise
        fixed_now = now is not None
        try:
            result = self._run_core(instant, progress, fixed_now=fixed_now, run_id=run_id)
        except Exception as exc:
            finished_at = self._finished_at(instant, fixed_now)
            duration_ms = max(0.0, (self._monotonic() - started_tick) * 1000)
            error_type = type(exc).__name__
            try:
                self._persist_run(
                    run_id=run_id, started_at=instant, finished_at=finished_at,
                    duration_ms=duration_ms, progress=progress, status="FAILED",
                    error_type=error_type,
                    error_message=_SAFE_ERRORS.get(error_type, "Unexpected cycle error"),
                )
            except Exception:
                logger.exception("No se pudo persistir el run fallido; revisar logs locales")
            self._log_timing(run_id, "FAILED", timing_started)
            memory_event("cycle_end", run_id=str(run_id))
            raise

        evaluation_error_type = evaluation_error_message = None
        if self._evaluation_service is not None:
            evaluation_started = self._monotonic()
            try:
                self._evaluation_service.evaluate_pending(
                    self.settings.symbol, self._finished_at(instant, fixed_now)
                )
            except Exception as exc:
                evaluation_error_type = type(exc).__name__
                evaluation_error_message = _SAFE_ERRORS.get(
                    evaluation_error_type, "Historical evaluation failed"
                )
                logger.exception("Fallo la evaluacion LIVE secundaria; la prediccion permanece guardada")
            finally:
                progress["evaluation_ms"] = max(
                    0.0, (self._monotonic() - evaluation_started) * 1000
                )
                memory_event("after_evaluation", run_id=str(run_id))
        finished_at = self._finished_at(instant, fixed_now)
        duration_ms = max(0.0, (self._monotonic() - started_tick) * 1000)
        try:
            self._persist_run(
                run_id=run_id, started_at=instant, finished_at=finished_at,
                duration_ms=duration_ms, progress=progress, status=result.status,
                evaluation_error_type=evaluation_error_type,
                evaluation_error_message=evaluation_error_message,
            )
        except Exception as exc:
            logger.exception("No se pudo persistir el pipeline_run; revisar logs locales")
            self._log_timing(run_id, "FAILED", timing_started, progress)
            memory_event("cycle_end", run_id=str(run_id))
            raise StorageError("No se pudo guardar la ejecucion del pipeline") from exc
        self._log_timing(run_id, result.status, timing_started, progress)
        memory_event("cycle_end", run_id=str(run_id))
        return replace(result, run_id=str(run_id), duration_ms=duration_ms)

    def _log_timing(self, run_id: UUID, status: str, started: float,
                    progress: dict | None = None) -> None:
        progress = progress or {}
        timing_logger.info(
            "RUN_CYCLE_TIMING run_id=%s status=%s total_ms=%.3f binance_ms=%.3f "
            "db_ms=%.3f features_ms=%.3f prediction_ms=%.3f evaluation_ms=%.3f",
            run_id, status, (perf_counter() - started) * 1000,
            self._io_ms["binance_ms"], self._io_ms["db_ms"],
            self._predictor.last_features_ms, progress.get("prediction_ms", 0.0),
            progress.get("evaluation_ms", 0.0),
        )

    def _run_core(
        self, instant: datetime, progress: dict, *, fixed_now: bool, run_id: UUID
    ) -> LiveCycleResult:
        trace_id = run_id.hex if self._telemetry else None
        ingestion_started = self._monotonic()
        self._event("ingestion_started", trace_id)
        try:
            try:
                previous = self._market.get_latest_candle(self.settings.symbol, self.settings.interval)
            except Exception as exc:
                raise StorageError("No se pudo consultar la ultima vela") from exc
            sent = self._ingestion.fetch_latest(
                self.settings.symbol, self.settings.interval, now=instant,
                bootstrap_candles=self.settings.prediction_lookback_candles,
            )
            try:
                latest = self._market.get_latest_candle(self.settings.symbol, self.settings.interval)
            except Exception as exc:
                raise StorageError("No se pudo consultar la vela tras la ingesta") from exc
            new_candle = bool(latest and (previous is None or latest.open_time > previous.open_time))
            progress["new_candle"] = new_candle
            progress["candles_ingested"] = sent
            progress["market_data_ms"] = max(0.0, (self._monotonic() - ingestion_started) * 1000)
            memory_event("after_market_data", run_id=str(run_id))
            self._event("data_ready", trace_id,
                        duration_ms=progress["market_data_ms"],
                        details={"candles_sent": sent})
            candle = self._latest_eligible(latest, instant)
            if candle is not None:
                progress["candle_open_time"] = candle.open_time
            if candle is None:
                if sent:
                    raise StorageError("La ingesta envio velas pero no hay ninguna elegible")
                self._event("prediction_skipped", trace_id, details={"reason": "NO_NEW_CANDLE"})
                return LiveCycleResult("NO_NEW_CANDLE", instant, None, latest, None, sent, False, False,
                                       trace_id=trace_id)

            prediction_started = self._monotonic()
            progress["prediction_started"] = prediction_started
            self._event("prediction_started", trace_id, candle=candle)
            try:
                execution = self._predictor.predict_candle_execution(
                    candle, self.settings.symbol, self.settings.interval, now=instant,
                    generated_at_clock=(lambda: instant) if fixed_now else self._utc_now,
                )
            except InsufficientHistoryError:
                progress["prediction_ms"] = max(0.0, (self._monotonic() - prediction_started) * 1000)
                self._event("prediction_skipped", trace_id, candle=candle,
                            duration_ms=progress["prediction_ms"],
                            details={"reason": "INSUFFICIENT_HISTORY"})
                return LiveCycleResult(
                    "INSUFFICIENT_HISTORY", instant, candle, latest, None, sent, new_candle, False,
                    trace_id=trace_id
                )
            prediction = execution.prediction
            if execution.created and self._v3_shadow is not None:
                try:
                    shadow = self._v3_shadow.predict(
                        self.settings.symbol, self.settings.interval, candle.open_time
                    )
                    logger.info(
                        "V3_SHADOW model_version=%s candle_open_time=%s prob_down=%.9f "
                        "prob_up=%.9f class=%s feature_ms=%.3f prediction_ms=%.3f",
                        shadow.model_version, shadow.candle_open_time.isoformat(),
                        shadow.prob_down, shadow.prob_up, shadow.predicted_class,
                        shadow.feature_ms, shadow.prediction_ms,
                    )
                except Exception:
                    logger.exception("V3 shadow failed; V1 prediction remains authoritative")
            progress["prediction_ms"] = max(0.0, (self._monotonic() - prediction_started) * 1000)
            memory_event("after_prediction", run_id=str(run_id))
            progress["prediction_id"] = prediction.prediction_id
            progress["new_prediction"] = execution.created
            self._event("prediction_ready", trace_id, candle=candle,
                        prediction_id=prediction.prediction_id,
                        duration_ms=progress["prediction_ms"],
                        details={"created": execution.created})
            status: CycleStatus = (
                "SUCCESS" if execution.created else
                "PREDICTION_ALREADY_EXISTS" if new_candle else "NO_NEW_CANDLE"
            )
            return LiveCycleResult(
                status, instant, candle, latest, prediction, sent, new_candle, execution.created,
                trace_id=trace_id
            )
        except Exception:
            if "market_data_ms" not in progress:
                progress["market_data_ms"] = max(0.0, (self._monotonic() - ingestion_started) * 1000)
            elif "prediction_ms" not in progress and "prediction_started" in progress:
                progress["prediction_ms"] = max(
                    0.0, (self._monotonic() - progress["prediction_started"]) * 1000
                )
            self._event("cycle_failed", trace_id)
            raise


def run_live_cycle(
    settings: Settings | None = None, *, now: datetime | None = None,
    telemetry: TelemetryStore | None = None,
) -> LiveCycleResult:
    """Execute exactly one cycle, using the repository factory and closing resources."""
    with LiveCycle.from_settings(settings, telemetry=telemetry) as cycle:
        return cycle.run(now=now)
