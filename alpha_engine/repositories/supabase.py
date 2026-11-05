"""Supabase Data API adapters for the existing repository contracts.

The server key belongs only in a trusted process. Public readers have SQL/RLS
access to candles and predictions, but never to private LLM records.
"""

from datetime import datetime, timezone
from typing import Any, Sequence
from urllib.parse import urlparse

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import (
    LiveEvaluationRecord, PendingLiveEvaluation, PipelineRunRecord, RunStatus,
)
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord, RecommendationRequestStatus

_PAGE_SIZE = 1000
_WRITE_BATCH = 500


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("Los timestamps de Supabase deben incluir zona horaria")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _number(value: float) -> str:
    # PostgREST accepts decimal strings for PostgreSQL numeric. Avoid another
    # binary float conversion on the way from the existing float domain model.
    return str(value)


def _client_from_settings(settings: Settings) -> Any:
    from supabase import create_client

    url = settings.supabase_url.strip()
    key = settings.supabase_key.get_secret_value().strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("SUPABASE_URL debe ser una URL HTTP(S) válida")
    if not key:
        raise ValueError("SUPABASE_KEY debe configurarse para Supabase")
    return create_client(url, key)


class _Repository:
    def __init__(self, client: Any, *, owns_client: bool = False):
        self._client = client
        self._owns_client = owns_client

    @classmethod
    def from_settings(cls, settings: Settings):
        return cls(_client_from_settings(settings), owns_client=True)

    def close(self) -> None:
        if self._owns_client:
            self._client.postgrest.aclose()
            self._owns_client = False


class SupabaseMarketRepository(_Repository):
    @staticmethod
    def _row(candle: Candle) -> dict:
        return {
            "symbol": candle.symbol,
            "interval": candle.interval,
            "open_time": _iso(candle.open_time),
            "close_time": _iso(candle.close_time),
            "open": _number(candle.open),
            "high": _number(candle.high),
            "low": _number(candle.low),
            "close": _number(candle.close),
            "volume": _number(candle.volume),
        }

    @staticmethod
    def _candle(row: dict) -> Candle:
        return Candle.model_validate({
            **row,
            "open_time": _utc(row["open_time"]),
            "close_time": _utc(row["close_time"]),
        })

    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0
        for start in range(0, len(candles), _WRITE_BATCH):
            rows = [self._row(c) for c in candles[start:start + _WRITE_BATCH]]
            self._client.table("candles").upsert(
                rows, on_conflict="symbol,interval,open_time", returning="minimal"
            ).execute()
        return len(candles)

    def get_candles(self, symbol: str, interval: str, start: datetime, stop: datetime) -> list[Candle]:
        if _utc(stop) <= _utc(start):
            raise ValueError("stop debe ser posterior a start")
        result: list[Candle] = []
        offset = 0
        while True:
            response = (self._client.table("candles").select("*")
                        .eq("symbol", symbol).eq("interval", interval)
                        .gte("open_time", _iso(start)).lt("open_time", _iso(stop))
                        .order("open_time").range(offset, offset + _PAGE_SIZE - 1)
                        .execute())
            rows = response.data or []
            result.extend(self._candle(row) for row in rows)
            if len(rows) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return result

    def get_latest_candle(self, symbol: str, interval: str) -> Candle | None:
        response = (self._client.table("candles").select("*")
                    .eq("symbol", symbol).eq("interval", interval)
                    .order("open_time", desc=True).limit(1).execute())
        return self._candle(response.data[0]) if response.data else None


class SupabasePredictionRepository(_Repository):
    @staticmethod
    def _row(prediction: PredictionRecord) -> dict:
        return {
            "prediction_id": prediction.prediction_id,
            "symbol": prediction.symbol,
            "interval": prediction.interval,
            "candle_open_time": _iso(prediction.timestamp),
            "horizon": prediction.horizon,
            "prob_up": _number(prediction.prob_up),
            "prob_down": _number(prediction.prob_down),
            "predicted_class": prediction.predicted_class,
            "model_version": prediction.model_version,
            "trained_at": _iso(prediction.trained_at) if prediction.trained_at else None,
            "generated_at": _iso(prediction.generated_at) if prediction.generated_at else None,
            "valid_until": _iso(prediction.valid_until) if prediction.valid_until else None,
        }

    @staticmethod
    def _prediction(row: dict) -> PredictionRecord:
        return PredictionRecord.model_validate({
            **row,
            "timestamp": _utc(row["candle_open_time"]),
            "trained_at": _utc(row["trained_at"]) if row.get("trained_at") else None,
            "generated_at": _utc(row["generated_at"]) if row.get("generated_at") else None,
            "valid_until": _utc(row["valid_until"]) if row.get("valid_until") else None,
        })

    def save(self, prediction: PredictionRecord) -> bool:
        # Immutable result: a retry with the same ID does nothing. The database
        # also rejects a different ID for the same candle and trained model.
        response = self._client.table("predictions").upsert(
            self._row(prediction), on_conflict="prediction_id",
            ignore_duplicates=True, returning="representation",
        ).execute()
        return bool(response.data)

    def get_by_id(self, prediction_id: str) -> PredictionRecord | None:
        response = (self._client.table("predictions").select("*")
                    .eq("prediction_id", prediction_id).limit(1).execute())
        return self._prediction(response.data[0]) if response.data else None

    def get_latest(self, symbol: str) -> PredictionRecord | None:
        response = (self._client.table("predictions").select("*")
                    .eq("symbol", symbol).order("candle_open_time", desc=True)
                    .order("prediction_id", desc=True).limit(1).execute())
        return self._prediction(response.data[0]) if response.data else None

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[PredictionRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit/offset fuera de rango")
        response = (self._client.table("predictions").select("*")
                    .eq("symbol", symbol).order("candle_open_time", desc=True)
                    .range(offset, offset + limit - 1).execute())
        return [self._prediction(row) for row in (response.data or [])]

    def list_recent_before(self, symbol: str, before: datetime, limit: int = 100) -> list[PredictionRecord]:
        if before.tzinfo is None or not 1 <= limit <= 1000:
            raise ValueError("invalid replay window")
        response = (self._client.table("predictions").select("*")
                    .eq("symbol", symbol).lte("candle_open_time", _iso(before))
                    .order("candle_open_time", desc=True).limit(limit).execute())
        return [self._prediction(row) for row in (response.data or [])]

    def list_entry_signals(self, symbol: str, model_version: str,
                           minimum_up: float, limit: int = 100) -> list[PredictionRecord]:
        if not 0.5 < minimum_up < 1 or not 1 <= limit <= 1000:
            raise ValueError("invalid entry search")
        response = (self._client.table("predictions").select("*")
                    .eq("symbol", symbol).eq("model_version", model_version)
                    .gte("prob_up", minimum_up)
                    .order("candle_open_time", desc=True).limit(limit).execute())
        return [self._prediction(row) for row in (response.data or [])]

    def list_evaluable_pending(
        self, symbol: str, now: datetime, limit: int = 100
    ) -> list[PendingLiveEvaluation]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit fuera de rango")
        response = self._client.rpc("pending_live_evaluations", {
            "p_symbol": symbol, "p_now": _iso(now), "p_limit": limit,
        }).execute()
        return [PendingLiveEvaluation.model_validate(row) for row in (response.data or [])]


class SupabaseRecommendationRepository(_Repository):
    @staticmethod
    def _row(record: RecommendationRecord) -> dict:
        return {
            "prediction_id": record.prediction_id,
            "prompt_version": record.prompt_version,
            "llm_model": record.llm_model,
            "candle_open_time": _iso(record.timestamp),
            "generated_at": _iso(record.generated_at),
            "requested_at": _iso(record.requested_at) if record.requested_at else None,
            "valid_until": _iso(record.valid_until) if record.valid_until else None,
            "llm_latency_ms": _number(record.llm_latency_ms) if record.llm_latency_ms is not None else None,
            "total_latency_ms": _number(record.total_latency_ms) if record.total_latency_ms is not None else None,
            "candidate_signal": record.candidate_signal,
            "llm_signal": record.llm_signal,
            "final_signal": record.final_signal,
            "risk_level": record.risk_level,
            "context_json": record.context_json,
            "response_json": record.response_json,
            "guardrails_json": record.guardrails_json,
        }

    @staticmethod
    def _recommendation(row: dict) -> RecommendationRecord:
        return RecommendationRecord.model_validate({
            **row,
            "timestamp": _utc(row["candle_open_time"]),
            "generated_at": _utc(row["generated_at"]),
            "requested_at": _utc(row["requested_at"]) if row.get("requested_at") else None,
            "valid_until": _utc(row["valid_until"]) if row.get("valid_until") else None,
        })

    def save(self, recommendation: RecommendationRecord) -> None:
        # The migration trigger appends a revision atomically when the cache changes.
        self._client.table("recommendations").upsert(
            self._row(recommendation),
            on_conflict="prediction_id,prompt_version,llm_model",
            returning="minimal",
        ).execute()

    def save_if_absent(self, recommendation: RecommendationRecord) -> bool:
        response = self._client.table("recommendations").upsert(
            self._row(recommendation),
            on_conflict="prediction_id,prompt_version,llm_model",
            ignore_duplicates=True, returning="representation",
        ).execute()
        return bool(response.data)

    def get_cached(self, prediction_id: str, prompt_version: str, llm_model: str) -> RecommendationRecord | None:
        response = (self._client.table("recommendations").select("*")
                    .eq("prediction_id", prediction_id)
                    .eq("prompt_version", prompt_version)
                    .eq("llm_model", llm_model).limit(1).execute())
        return self._recommendation(response.data[0]) if response.data else None

    def list_recent(self, limit: int = 20) -> list[RecommendationRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit debe estar entre 1 y 100")
        response = (self._client.table("recommendation_history").select("*")
                    .order("generated_at", desc=True)
                    .order("revision_id", desc=True).limit(limit).execute())
        return [self._recommendation(row) for row in (response.data or [])]


class SupabaseRecommendationRequestRepository(_Repository):
    @staticmethod
    def _row(record: RecommendationRequestRecord) -> dict:
        row = {
            "request_id": str(record.request_id), "prediction_id": record.prediction_id,
            "symbol": record.symbol, "interval": record.interval,
            "provider": record.provider, "llm_model": record.llm_model,
            "prompt_version": record.prompt_version,
            "requested_at": _iso(record.requested_at),
            "generated_at": _iso(record.generated_at) if record.generated_at else None,
            "finished_at": _iso(record.finished_at),
            "valid_until": _iso(record.valid_until) if record.valid_until else None,
            "llm_latency_ms": _number(record.llm_latency_ms) if record.llm_latency_ms is not None else None,
            "total_latency_ms": _number(record.total_latency_ms),
            "status": record.status, "timing_status": record.timing_status,
            "cached": record.cached, "error_type": record.error_type,
            "error_message": record.error_message,
            "created_at": _iso(record.created_at) if record.created_at else _iso(record.finished_at),
        }
        # Existing installations keep auditing technical-only requests before
        # the optional news migration is approved and applied.
        if record.news_context_id is not None or record.news_context_fetched_at is not None:
            row.update({
                "news_context_id": str(record.news_context_id) if record.news_context_id else None,
                "news_context_used": record.news_context_used,
                "news_context_fetched_at": _iso(record.news_context_fetched_at) if record.news_context_fetched_at else None,
                "news_context_count": record.news_context_count,
                "news_context_sentiment": record.news_context_sentiment,
            })
        return row

    @staticmethod
    def _record(row: dict) -> RecommendationRequestRecord:
        return RecommendationRequestRecord.model_validate({
            **row, "requested_at": _utc(row["requested_at"]),
            "generated_at": _utc(row["generated_at"]) if row.get("generated_at") else None,
            "finished_at": _utc(row["finished_at"]),
            "valid_until": _utc(row["valid_until"]) if row.get("valid_until") else None,
            "created_at": _utc(row["created_at"]) if row.get("created_at") else None,
        })

    def save(self, request: RecommendationRequestRecord) -> None:
        self._client.table("recommendation_requests").upsert(
            self._row(request), on_conflict="request_id", ignore_duplicates=True,
            returning="minimal",
        ).execute()

    def get_latest(
        self, symbol: str, status: RecommendationRequestStatus | None = None
    ) -> RecommendationRequestRecord | None:
        records = self.list_recent(symbol, limit=1, status=status)
        return records[0] if records else None

    def list_recent(
        self, symbol: str, limit: int = 20, status: RecommendationRequestStatus | None = None
    ) -> list[RecommendationRequestRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit debe estar entre 1 y 1000")
        query = self._client.table("public_recommendation_request_summary").select("*").eq("symbol", symbol)
        if status is not None:
            query = query.eq("status", status)
        response = query.order("requested_at", desc=True).limit(limit).execute()
        return [self._record(row) for row in (response.data or [])]


class SupabasePipelineRunRepository(_Repository):
    @staticmethod
    def _row(run: PipelineRunRecord) -> dict:
        return {
            "run_id": str(run.run_id), "started_at": _iso(run.started_at),
            "finished_at": _iso(run.finished_at), "duration_ms": _number(run.duration_ms),
            "market_data_ms": _number(run.market_data_ms) if run.market_data_ms is not None else None,
            "prediction_ms": _number(run.prediction_ms) if run.prediction_ms is not None else None,
            "evaluation_ms": _number(run.evaluation_ms) if run.evaluation_ms is not None else None,
            "status": run.status, "symbol": run.symbol, "interval": run.interval,
            "candle_open_time": _iso(run.candle_open_time) if run.candle_open_time else None,
            "prediction_id": run.prediction_id, "new_candle": run.new_candle,
            "new_prediction": run.new_prediction, "candles_ingested": run.candles_ingested,
            "error_type": run.error_type, "error_message": run.error_message,
            "evaluation_error_type": run.evaluation_error_type,
            "evaluation_error_message": run.evaluation_error_message,
            "created_at": _iso(run.created_at) if run.created_at else _iso(run.finished_at),
        }

    @staticmethod
    def _record(row: dict) -> PipelineRunRecord:
        return PipelineRunRecord.model_validate({
            **row, "started_at": _utc(row["started_at"]),
            "finished_at": _utc(row["finished_at"]),
            "candle_open_time": _utc(row["candle_open_time"]) if row.get("candle_open_time") else None,
            "created_at": _utc(row["created_at"]) if row.get("created_at") else None,
        })

    def save(self, run: PipelineRunRecord) -> None:
        self._client.table("pipeline_runs").upsert(
            self._row(run), on_conflict="run_id", ignore_duplicates=True,
            returning="minimal",
        ).execute()

    def get_latest(self, symbol: str, status: RunStatus | None = None) -> PipelineRunRecord | None:
        rows = self.list_recent(symbol, limit=1, status=status)
        return rows[0] if rows else None

    def list_recent(
        self, symbol: str, limit: int = 20, status: RunStatus | None = None
    ) -> list[PipelineRunRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit debe estar entre 1 y 1000")
        query = self._client.table("public_pipeline_runs").select("*").eq("symbol", symbol)
        if status is not None:
            query = query.eq("status", status)
        response = query.order("started_at", desc=True).limit(limit).execute()
        return [self._record(row) for row in (response.data or [])]


class SupabaseLiveEvaluationRepository(_Repository):
    @staticmethod
    def _row(record: LiveEvaluationRecord) -> dict:
        return {
            "evaluation_id": str(record.evaluation_id), "prediction_id": record.prediction_id,
            "symbol": record.symbol, "evaluated_at": _iso(record.evaluated_at),
            "source_close": _number(record.source_close),
            "target_open_time": _iso(record.target_open_time),
            "target_close_time": _iso(record.target_close_time),
            "target_close": _number(record.target_close),
            "observed_move": record.observed_move,
            "observed_class": record.observed_class, "correct": record.correct,
        }

    @staticmethod
    def _record(row: dict) -> LiveEvaluationRecord:
        return LiveEvaluationRecord.model_validate({
            **row, "evaluated_at": _utc(row["evaluated_at"]),
            "target_open_time": _utc(row["target_open_time"]),
            "target_close_time": _utc(row["target_close_time"]),
        })

    def save(self, evaluation: LiveEvaluationRecord) -> bool:
        response = self._client.table("live_evaluations").upsert(
            self._row(evaluation), on_conflict="prediction_id",
            ignore_duplicates=True, returning="representation",
        ).execute()
        return bool(response.data)

    def save_batch(self, evaluations: Sequence[LiveEvaluationRecord]) -> set[str]:
        inserted: set[str] = set()
        for start in range(0, len(evaluations), _WRITE_BATCH):
            rows = [self._row(record) for record in evaluations[start:start + _WRITE_BATCH]]
            response = self._client.table("live_evaluations").upsert(
                rows, on_conflict="prediction_id", ignore_duplicates=True,
                returning="representation",
            ).execute()
            inserted.update(row["prediction_id"] for row in (response.data or []))
        return inserted

    def get_by_prediction(self, prediction_id: str) -> LiveEvaluationRecord | None:
        response = (self._client.table("live_evaluations").select("*")
                    .eq("prediction_id", prediction_id).limit(1).execute())
        return self._record(response.data[0]) if response.data else None

    def get_latest(self, symbol: str) -> LiveEvaluationRecord | None:
        rows = self.list_recent(symbol, limit=1)
        return rows[0] if rows else None

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[LiveEvaluationRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit/offset fuera de rango")
        response = (self._client.table("live_evaluations").select("*")
                    .eq("symbol", symbol).order("evaluated_at", desc=True)
                    .range(offset, offset + limit - 1).execute())
        return [self._record(row) for row in (response.data or [])]
