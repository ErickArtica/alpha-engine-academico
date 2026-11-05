"""Safe read models for the Streamlit dashboard.

The dashboard only reads persisted facts.  It deliberately has no Binance,
model-training, inference, or LLM-provider dependency.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from alpha_engine.config.settings import PROJECT_ROOT, Settings
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.repositories.factory import (
    create_live_evaluation_repository,
    create_market_repository,
    create_pipeline_run_repository,
    create_prediction_repository,
    create_recommendation_request_repository,
)
from alpha_engine.repositories.protocols import (
    LiveEvaluationRepository,
    MarketRepository,
    PipelineRunRepository,
    PredictionRepository,
    RecommendationRequestRepository,
)
from alpha_engine.services.live_evaluation import LiveEvaluationService, LiveMetrics
from alpha_engine.services.recommendation_observability import RecommendationObservabilityService
from alpha_engine.services.system_status import SystemStatusService
from alpha_engine.tools.market import AnalyticsDataError, TechnicalIndicators, get_technical_indicators


class DashboardUnavailableError(RuntimeError):
    """The configured persisted source could not be read safely."""


class PublicCandleView(BaseModel):
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_record(cls, value: Candle) -> "PublicCandleView":
        return cls.model_validate(value.model_dump())


class PublicPredictionView(BaseModel):
    prediction_id: str
    prediction_id_short: str
    symbol: str
    interval: str
    horizon: str = "1m"
    source_candle_open_time: datetime
    predicted_class: Literal["UP", "DOWN"]
    prob_up: float
    prob_down: float
    model_version: str
    trained_at: datetime | None
    generated_at: datetime | None
    valid_until: datetime | None
    timing_status: Literal["TIMELY", "LATE"] | None
    current_validity: Literal["ACTIVE", "EXPIRED"] | None

    def validity_at(self, now: datetime) -> Literal["ACTIVE", "EXPIRED"] | None:
        """Re-evaluate validity when the UI renders; the stored field is a read-time snapshot."""
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if self.valid_until is None:
            return None
        return "ACTIVE" if now < self.valid_until else "EXPIRED"

    @classmethod
    def from_record(cls, value: PredictionRecord, now: datetime) -> "PublicPredictionView":
        return cls(
            prediction_id=value.prediction_id,
            prediction_id_short=_short_id(value.prediction_id),
            symbol=value.symbol, interval=value.interval, horizon=value.horizon,
            source_candle_open_time=value.source_candle_open_time,
            predicted_class=value.predicted_class,
            prob_up=value.prob_up,
            prob_down=value.prob_down,
            model_version=value.model_version,
            trained_at=value.trained_at,
            generated_at=value.generated_at,
            valid_until=value.valid_until,
            timing_status=value.timing_status,
            current_validity=value.current_validity(now),
        )


class PublicPipelineRunView(BaseModel):
    run_id_short: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    candle_open_time: datetime | None
    prediction_id_short: str | None
    new_candle: bool
    new_prediction: bool
    candles_ingested: int
    error_category: str | None

    @classmethod
    def from_record(cls, value: PipelineRunRecord) -> "PublicPipelineRunView":
        return cls(
            run_id_short=_short_id(str(value.run_id)), status=value.status,
            started_at=value.started_at, finished_at=value.finished_at,
            duration_ms=value.duration_ms, candle_open_time=value.candle_open_time,
            prediction_id_short=_short_id(value.prediction_id) if value.prediction_id else None,
            new_candle=value.new_candle, new_prediction=value.new_prediction,
            candles_ingested=value.candles_ingested,
            error_category=_error_category(value.error_type),
        )


class PublicLiveEvaluationView(BaseModel):
    prediction_id_short: str
    evaluated_at: datetime
    source_close: float
    target_close: float
    observed_move: Literal["UP", "DOWN", "FLAT"]
    observed_class: Literal["UP", "DOWN"]
    correct: bool

    @classmethod
    def from_record(cls, value: LiveEvaluationRecord) -> "PublicLiveEvaluationView":
        return cls(
            prediction_id_short=_short_id(value.prediction_id), evaluated_at=value.evaluated_at,
            source_close=value.source_close, target_close=value.target_close,
            observed_move=value.observed_move, observed_class=value.observed_class,
            correct=value.correct,
        )


class PublicLiveMetrics(BaseModel):
    evaluated: int
    correct: int
    incorrect: int
    accuracy: float | None
    timely: int
    late: int
    unknown_timing: int
    source_start: datetime | None = None
    source_end: datetime | None = None
    scored_predictions: int = 0
    missing_predictions: int = 0
    inconsistent_correct_rows: int = 0
    pred_down_actual_down: int = 0
    pred_down_actual_up: int = 0
    pred_up_actual_down: int = 0
    pred_up_actual_up: int = 0
    balanced_accuracy: float | None = None
    precision_up: float | None = None
    recall_up: float | None = None
    f1_up: float | None = None
    precision_down: float | None = None
    recall_down: float | None = None
    f1_down: float | None = None
    macro_f1: float | None = None
    roc_auc_up: float | None = None
    always_down_accuracy: float | None = None
    predicted_up_fraction: float | None = None
    predicted_down_fraction: float | None = None
    actual_up_fraction: float | None = None
    actual_down_fraction: float | None = None

    @classmethod
    def from_metrics(cls, value: LiveMetrics) -> "PublicLiveMetrics":
        return cls.model_validate(value.__dict__)


class PublicRecommendationObservability(BaseModel):
    last_attempt_at: datetime | None
    last_attempt_status: str | None
    last_success_at: datetime | None
    timely_count: int
    late_count: int
    timeout_count: int
    provider_error_count: int
    average_llm_latency_ms: float | None


class PublicTestMetrics(BaseModel):
    model_version: str | None = None
    trained_at: datetime | None = None
    sample_count: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    baseline_accuracy: float | None = None
    target_definition: str | None = None


class PublicEconomicScenario(BaseModel):
    cost_scenario: str
    latency_seconds_assumed: float
    tested_rows: int | None = None
    executed_buy_trades: int | None = None
    net_compounded_return_estimated: float | None = None
    max_drawdown_net_estimated: float | None = None
    limits: str | None = None


class PublicDashboardData(BaseModel):
    mode: Literal["LIVE", "FALLBACK"]
    storage: str
    refreshed_at: datetime
    snapshot_at: datetime | None = None
    system_state: Literal["CURRENT", "STALE", "NO_DATA"]
    latest_run: PublicPipelineRunView | None = None
    last_successful_run: PublicPipelineRunView | None = None
    latest_candle: PublicCandleView | None = None
    latest_prediction: PublicPredictionView | None = None
    recent_runs: list[PublicPipelineRunView] = []
    recent_errors: list[PublicPipelineRunView] = []
    candles: list[PublicCandleView] = []
    indicators: TechnicalIndicators | None = None
    predictions: list[PublicPredictionView] = []
    evaluations: list[PublicLiveEvaluationView] = []
    live_metrics: PublicLiveMetrics | None = None
    recommendation_observability: PublicRecommendationObservability | None = None
    test_metrics: PublicTestMetrics | None = None
    economic_scenarios: list[PublicEconomicScenario] = []

    def recommendation_available_at(self, now: datetime, deadline_margin_seconds: float) -> bool:
        """Use the current clock for a button decision, never cached validity."""
        if now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if deadline_margin_seconds < 0:
            raise ValueError("deadline_margin_seconds no puede ser negativo")
        prediction = self.latest_prediction
        return bool(
            self.mode == "LIVE" and self.system_state == "CURRENT"
            and prediction is not None and prediction.valid_until is not None
            and prediction.validity_at(now) == "ACTIVE"
            and (prediction.valid_until - now).total_seconds() > deadline_margin_seconds
        )


def read_latest_public_prediction(
    settings: Settings, now: datetime | None = None,
) -> PublicPredictionView | None:
    """Read one sanitized prediction without loading every dashboard repository."""
    if now is not None and now.tzinfo is None:
        raise ValueError("now debe incluir zona horaria")
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    repository = None
    try:
        repository = create_prediction_repository(settings)
        record = repository.get_latest(settings.symbol)
    except Exception as exc:
        raise DashboardUnavailableError("No se pudo consultar la última predicción") from exc
    finally:
        if repository is not None:
            try:
                repository.close()
            except Exception:
                pass
    if record is None or record.interval != settings.interval:
        return None
    return PublicPredictionView.from_record(record, instant)


def read_public_live_metrics(settings: Settings, model_version: str | None = None) -> PublicLiveMetrics:
    """Read only the two repositories needed for one cached LIVE metric cohort."""
    resources = []
    try:
        predictions = create_prediction_repository(settings); resources.append(predictions)
        evaluations = create_live_evaluation_repository(settings); resources.append(evaluations)
        return PublicLiveMetrics.from_metrics(
            LiveEvaluationService.metrics_for(predictions, evaluations, settings.symbol, model_version)
        )
    except Exception as exc:
        raise DashboardUnavailableError("No se pudo consultar la evaluación LIVE") from exc
    finally:
        for resource in reversed(resources):
            try:
                resource.close()
            except Exception:
                pass


def _short_id(value: str) -> str:
    return value if len(value) <= 12 else f"{value[:6]}…{value[-4:]}"


def _error_category(error_type: str | None) -> str | None:
    if not error_type:
        return None
    normalized = error_type.lower()
    if "market" in normalized or "binance" in normalized:
        return "MARKET_DATA"
    if "storage" in normalized or "database" in normalized:
        return "STORAGE"
    if "model" in normalized or "prediction" in normalized:
        return "MODEL"
    if "validation" in normalized or "value" in normalized:
        return "VALIDATION"
    return "UNEXPECTED"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _safe_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _historical_reports(settings: Settings) -> tuple[PublicTestMetrics | None, list[PublicEconomicScenario]]:
    model_path = settings.model_path if settings.model_path.is_absolute() else PROJECT_ROOT / settings.model_path
    metadata = _safe_json(model_path.with_name(model_path.stem + "_metadata.json"))
    baseline = _safe_json(model_path.with_name("baseline_v1.json"))
    test = metadata.get("evaluation", {}).get("test", {}) if metadata else {}
    test_metrics = None
    if metadata and test:
        test_metrics = PublicTestMetrics(
            model_version=metadata.get("model_version"),
            trained_at=_utc(metadata["trained_at"]) if metadata.get("trained_at") else None,
            sample_count=test.get("sample_count"), accuracy=test.get("accuracy"),
            precision=test.get("precision"), recall=test.get("recall"), f1=test.get("f1"),
            baseline_accuracy=(baseline or {}).get("test", {}).get("accuracy"),
            target_definition=metadata.get("target_definition"),
        )
    economics = _safe_json(model_path.with_name("economic_summary_v1.json"))
    if not economics:
        economics = _safe_json(model_path.with_name("economic_check_v1.json"))
    scenarios = []
    if economics:
        for row in economics.get("runs", []):
            total = row.get("total", {})
            scenarios.append(PublicEconomicScenario(
                cost_scenario=str(row.get("cost_scenario", "unknown")),
                latency_seconds_assumed=float(row.get("latency_seconds_assumed", 0)),
                tested_rows=total.get("tested_rows"), executed_buy_trades=total.get("executed_buy_trades"),
                net_compounded_return_estimated=total.get("net_compounded_return_estimated"),
                max_drawdown_net_estimated=total.get("max_drawdown_net_estimated"),
                limits=economics.get("limits"),
            ))
    return test_metrics, scenarios


class PublicDashboardService:
    """Builds public DTOs from provider-neutral repository contracts."""

    def __init__(
        self, settings: Settings, market: MarketRepository, predictions: PredictionRepository,
        runs: PipelineRunRepository, evaluations: LiveEvaluationRepository,
        recommendation_requests: RecommendationRequestRepository,
    ):
        self._settings = settings
        self._market = market
        self._predictions = predictions
        self._runs = runs
        self._evaluations = evaluations
        self._requests = recommendation_requests
        self._status = SystemStatusService(market, predictions, runs, evaluations)
        self._evaluation_service = LiveEvaluationService(market, predictions, evaluations)
        self._recommendation_observability = RecommendationObservabilityService(recommendation_requests)

    @classmethod
    def from_settings(cls, settings: Settings) -> "PublicDashboardService":
        resources = []
        try:
            market = create_market_repository(settings); resources.append(market)
            predictions = create_prediction_repository(settings); resources.append(predictions)
            runs = create_pipeline_run_repository(settings); resources.append(runs)
            evaluations = create_live_evaluation_repository(settings); resources.append(evaluations)
            requests = create_recommendation_request_repository(settings); resources.append(requests)
            instance = cls(settings, market, predictions, runs, evaluations, requests)
            instance._owned_resources = resources
            return instance
        except Exception as exc:
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception:
                    pass
            raise DashboardUnavailableError("No se pudo abrir el almacenamiento configurado") from exc

    def close(self) -> None:
        for resource in reversed(getattr(self, "_owned_resources", [])):
            try:
                resource.close()
            except Exception:
                pass
        self._owned_resources = []

    def __enter__(self) -> "PublicDashboardService":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def dashboard(
        self, now: datetime | None = None, candle_limit: int = 240,
        section: Literal["all", "home", "descriptive", "predictive", "prescriptive", "evaluation"] = "all",
    ) -> PublicDashboardData:
        if now is not None and now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not 2 <= candle_limit <= 1000:
            raise ValueError("candle_limit debe estar entre 2 y 1000")
        if section not in {"all", "home", "descriptive", "predictive", "prescriptive", "evaluation"}:
            raise ValueError("section no reconocida")
        try:
            status = self._status.get(self._settings.symbol, self._settings.interval) if section in {"all", "home"} else None
            candles = []
            if section in {"all", "home", "descriptive"}:
                stop = instant + timedelta(minutes=1)
                candles = self._market.get_candles(
                    self._settings.symbol, self._settings.interval,
                    stop - timedelta(minutes=candle_limit), stop,
                )
            indicators = None
            if section in {"all", "descriptive"}:
                try:
                    indicators = get_technical_indicators(candles)
                except AnalyticsDataError:
                    pass
            predictions = (
                self._predictions.list_recent(self._settings.symbol, limit=20)
                if section in {"all", "predictive"} else []
            )
            evaluations = (
                self._evaluations.list_recent(self._settings.symbol, limit=50)
                if section in {"all", "evaluation"} else []
            )
            # Public pages share one short-lived cached metric cohort. Keep the
            # full-dashboard API for local callers and existing service tests.
            metrics = self._evaluation_service.metrics(self._settings.symbol) if section == "all" else None
            recommendation = (
                self._recommendation_observability.get(self._settings.symbol, limit=100)
                if section in {"all", "prescriptive"} else None
            )
            latest_candle = (
                status.last_candle if status else
                candles[-1] if candles else
                self._market.get_latest_candle(self._settings.symbol, self._settings.interval)
                if section == "prescriptive" else None
            )
            latest_prediction = (
                status.last_prediction if status else
                self._predictions.get_latest(self._settings.symbol)
                if section == "predictive" else None
            )
        except Exception as exc:
            raise DashboardUnavailableError("No se pudo consultar el estado persistido") from exc

        if latest_candle is None:
            system_state: Literal["CURRENT", "STALE", "NO_DATA"] = "NO_DATA"
        elif (instant - latest_candle.close_time).total_seconds() > self._settings.max_candle_age_seconds:
            system_state = "STALE"
        else:
            system_state = "CURRENT"
        test_metrics, scenarios = _historical_reports(self._settings) if section in {"all", "evaluation"} else (None, [])
        return PublicDashboardData(
            mode="LIVE", storage=self._settings.storage_provider, refreshed_at=instant,
            system_state=system_state,
            latest_run=PublicPipelineRunView.from_record(status.last_run) if status and status.last_run else None,
            last_successful_run=(PublicPipelineRunView.from_record(status.last_successful_run)
                                 if status and status.last_successful_run else None),
            latest_candle=PublicCandleView.from_record(latest_candle) if latest_candle else None,
            latest_prediction=(PublicPredictionView.from_record(latest_prediction, instant)
                               if latest_prediction else None),
            recent_runs=[PublicPipelineRunView.from_record(item) for item in status.recent_runs] if status else [],
            recent_errors=[PublicPipelineRunView.from_record(item) for item in status.recent_errors] if status else [],
            candles=[PublicCandleView.from_record(item) for item in candles], indicators=indicators,
            predictions=[PublicPredictionView.from_record(item, instant) for item in predictions],
            evaluations=[PublicLiveEvaluationView.from_record(item) for item in evaluations],
            live_metrics=PublicLiveMetrics.from_metrics(metrics) if metrics else None,
            recommendation_observability=PublicRecommendationObservability(
                last_attempt_at=(recommendation.last_attempt.requested_at if recommendation.last_attempt else None),
                last_attempt_status=(recommendation.last_attempt.status if recommendation.last_attempt else None),
                last_success_at=(recommendation.last_successful_attempt.finished_at
                                 if recommendation.last_successful_attempt else None),
                timely_count=recommendation.timely_count, late_count=recommendation.late_count,
                timeout_count=recommendation.timeout_count,
                provider_error_count=recommendation.provider_error_count,
                average_llm_latency_ms=recommendation.average_llm_latency_ms,
            ) if recommendation else None,
            test_metrics=test_metrics, economic_scenarios=scenarios,
        )


class PublicFallbackDashboardService:
    """Explicit offline fallback for the reproducible public snapshot."""

    def __init__(self, snapshot_path: Path):
        self._path = snapshot_path

    def dashboard(self, now: datetime | None = None) -> PublicDashboardData:
        payload = _safe_json(self._path)
        if payload is None:
            raise DashboardUnavailableError("No existe un fallback legible")
        if now is not None and now.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        try:
            symbol = str(payload["symbol"])
            interval = str(payload["interval"])
            candles = []
            for item in payload.get("candles", []):
                opened = _utc(item["open_time"])
                candles.append(PublicCandleView(
                    symbol=symbol, interval=interval, open_time=opened,
                    close_time=(_utc(item["close_time"]) if item.get("close_time")
                                else opened + timedelta(minutes=1) - timedelta(milliseconds=1)),
                    open=item["open"], high=item["high"], low=item["low"],
                    close=item["close"], volume=item["volume"],
                ))
            raw_prediction = payload.get("latest_prediction")
            prediction = None
            if raw_prediction:
                source = raw_prediction.get("timestamp") or raw_prediction.get("source_candle_open_time") or raw_prediction.get("candle_open_at")
                prediction = PublicPredictionView(
                    prediction_id=str(raw_prediction.get("prediction_id", "snapshot")),
                    prediction_id_short=_short_id(str(raw_prediction.get("prediction_id", "snapshot"))),
                    symbol=symbol, interval=interval,
                    horizon=str(raw_prediction.get("horizon") or interval),
                    source_candle_open_time=_utc(source),
                    predicted_class=raw_prediction["predicted_class"], prob_up=raw_prediction["prob_up"],
                    prob_down=raw_prediction["prob_down"], model_version=raw_prediction["model_version"],
                    trained_at=None, generated_at=None, valid_until=None,
                    timing_status=None, current_validity=None,
                )
            model = payload.get("model", {})
            test = model.get("evaluation", {}).get("test", {})
            test_metrics = PublicTestMetrics(
                model_version=model.get("model_version"),
                trained_at=_utc(model["trained_at"]) if model.get("trained_at") else None,
                sample_count=test.get("sample_count"), accuracy=test.get("accuracy"),
                precision=test.get("precision"), recall=test.get("recall"), f1=test.get("f1"),
                target_definition=model.get("target_definition"),
            ) if model and test else None
        except (TypeError, ValueError, KeyError) as exc:
            raise DashboardUnavailableError("El fallback no tiene el formato esperado") from exc
        return PublicDashboardData(
            mode="FALLBACK", storage="snapshot offline", refreshed_at=instant,
            snapshot_at=_utc(payload["exported_at"]) if payload.get("exported_at") else None,
            system_state="STALE" if candles else "NO_DATA", candles=candles,
            latest_candle=candles[-1] if candles else None, latest_prediction=prediction,
            test_metrics=test_metrics,
        )
