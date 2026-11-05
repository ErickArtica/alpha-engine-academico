"""Adaptadores InfluxDB 3 Core; consultas SQL sin dependencia de Flux."""

from datetime import datetime, timezone
import re
from typing import Sequence
from uuid import uuid4

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord, RunStatus
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord, RecommendationRequestStatus
from alpha_engine.repositories.influx3_connection import Influx3Connection


def _tag(value: str) -> str:
    return value.replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")


def _field(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("El timestamp debe incluir zona horaria")
    return value.astimezone(timezone.utc).isoformat()


def _query_or_empty(connection: Influx3Connection, sql: str, params: dict) -> list[dict]:
    try:
        return connection.query(sql, params)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.lower()
        if exc.response.status_code in (400, 404) and (re.search(r"table .* not found", body) or "does not exist" in body):
            return []
        raise


class Influx3MarketRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3MarketRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0
        lines = [
            f"candles,symbol={_tag(c.symbol)},interval={_tag(c.interval)} "
            f"open={c.open},high={c.high},low={c.low},close={c.close},"
            f"volume={c.volume},close_time_ms={_ms(c.close_time)}i {_ms(c.open_time)}"
            for c in candles
        ]
        self._connection.write("\n".join(lines))
        return len(candles)

    def get_candles(self, symbol: str, interval: str, start: datetime, stop: datetime) -> list[Candle]:
        if start.tzinfo is None or stop.tzinfo is None or stop <= start:
            raise ValueError("start y stop deben tener zona horaria y stop > start")
        rows = _query_or_empty(
            self._connection,
            'SELECT * FROM candles WHERE symbol = $symbol AND "interval" = $interval '
            "AND time >= $start AND time < $stop ORDER BY time",
            {"symbol": symbol, "interval": interval, "start": start.isoformat(), "stop": stop.isoformat()},
        )
        return [self._candle(row) for row in rows]

    def get_latest_candle(self, symbol: str, interval: str) -> Candle | None:
        rows = _query_or_empty(
            self._connection,
            'SELECT * FROM candles WHERE symbol = $symbol AND "interval" = $interval ORDER BY time DESC LIMIT 1',
            {"symbol": symbol, "interval": interval},
        )
        return self._candle(rows[0]) if rows else None

    @staticmethod
    def _candle(row: dict) -> Candle:
        return Candle(
            symbol=row["symbol"], interval=row["interval"], open_time=_dt(row["time"]),
            open=row["open"], high=row["high"], low=row["low"], close=row["close"],
            volume=row["volume"],
            close_time=datetime.fromtimestamp(row["close_time_ms"] / 1000, tz=timezone.utc),
        )


class Influx3PredictionRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3PredictionRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def save(self, p: PredictionRecord) -> None:
        trained_at = f",trained_at={_field(p.trained_at.isoformat())}" if p.trained_at else ""
        generated_at = f",generated_at={_field(_iso_utc(p.generated_at))}" if p.generated_at else ""
        valid_until = f",valid_until={_field(_iso_utc(p.valid_until))}" if p.valid_until else ""
        self._connection.write(
            f"predictions,prediction_id={_tag(p.prediction_id)},symbol={_tag(p.symbol)},"
            f"model_version={_tag(p.model_version)} "
            f"horizon={_field(p.horizon)},prob_up={p.prob_up},prob_down={p.prob_down},"
            f"predicted_class={_field(p.predicted_class)}{trained_at}"
            f"{generated_at}{valid_until} {_ms(p.timestamp)}"
        )

    def get_by_id(self, prediction_id: str) -> PredictionRecord | None:
        return self._first("prediction_id = $prediction_id", {"prediction_id": prediction_id})

    def get_latest(self, symbol: str) -> PredictionRecord | None:
        return self._first("symbol = $symbol", {"symbol": symbol})

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[PredictionRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit/offset fuera de rango")
        rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM predictions WHERE symbol = $symbol ORDER BY time DESC LIMIT {limit} OFFSET {offset}",
            {"symbol": symbol},
        )
        return [self._record(row) for row in rows]

    def _first(self, where: str, params: dict) -> PredictionRecord | None:
        rows = _query_or_empty(self._connection, f"SELECT * FROM predictions WHERE {where} ORDER BY time DESC LIMIT 1", params)
        if not rows:
            return None
        return self._record(rows[0])

    @staticmethod
    def _record(row: dict) -> PredictionRecord:
        return PredictionRecord.model_validate({
            **row, "timestamp": _dt(row["time"]),
            "generated_at": _dt(row["generated_at"]) if row.get("generated_at") else None,
            "valid_until": _dt(row["valid_until"]) if row.get("valid_until") else None,
        })


class Influx3RecommendationRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3RecommendationRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def save(self, r: RecommendationRecord) -> None:
        timing = []
        if r.requested_at is not None:
            timing.append(f"requested_at={_field(_iso_utc(r.requested_at))}")
        if r.valid_until is not None:
            timing.append(f"valid_until={_field(_iso_utc(r.valid_until))}")
        if r.llm_latency_ms is not None:
            timing.append(f"llm_latency_ms={r.llm_latency_ms}")
        if r.total_latency_ms is not None:
            timing.append(f"total_latency_ms={r.total_latency_ms}")
        timing_fields = ("," + ",".join(timing)) if timing else ""
        fields = (
            f"candidate_signal={_field(r.candidate_signal)},llm_signal={_field(r.llm_signal or '')},"
            f"final_signal={_field(r.final_signal)},risk_level={_field(r.risk_level)},"
            f"context_json={_field(r.context_json)},response_json={_field(r.response_json or '')},"
            f"guardrails_json={_field(r.guardrails_json)}{timing_fields}"
        )
        tags = (
            f"prediction_id={_tag(r.prediction_id)},"
            f"prompt_version={_tag(r.prompt_version)},llm_model={_tag(r.llm_model)}"
        )
        current = (
            f"recommendations,prediction_id={_tag(r.prediction_id)},"
            f"prompt_version={_tag(r.prompt_version)},llm_model={_tag(r.llm_model)} "
            f"generated_at={_field(r.generated_at.isoformat())},{fields} {_ms(r.timestamp)}"
        )
        history = (
            f"recommendation_history,{tags},revision_id={uuid4().hex} "
            f"candle_open_at={_field(r.timestamp.isoformat())},"
            f"generated_at={_field(r.generated_at.isoformat())},{fields} {_ms(r.generated_at)}"
        )
        self._connection.write(current + "\n" + history)

    def save_if_absent(self, recommendation: RecommendationRecord) -> bool:
        if self.get_cached(
            recommendation.prediction_id, recommendation.prompt_version, recommendation.llm_model
        ) is not None:
            return False
        self.save(recommendation)
        return True

    def get_cached(self, prediction_id: str, prompt_version: str, llm_model: str) -> RecommendationRecord | None:
        rows = _query_or_empty(
            self._connection,
            "SELECT * FROM recommendations WHERE prediction_id = $prediction_id "
            "AND prompt_version = $prompt_version AND llm_model = $llm_model ORDER BY time DESC LIMIT 1",
            {"prediction_id": prediction_id, "prompt_version": prompt_version, "llm_model": llm_model},
        )
        return self._record(rows[0]) if rows else None

    def list_recent(self, limit: int = 20) -> list[RecommendationRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit debe estar entre 1 y 100")
        history_rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM recommendation_history ORDER BY time DESC LIMIT {limit}",
            {},
        )
        current_rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM recommendations ORDER BY time DESC LIMIT {limit}",
            {},
        )
        combined = [self._history_record(row) for row in history_rows if "candle_open_at" in row]
        combined.extend(self._record(row) for row in current_rows)
        unique = {}
        for record in combined:
            key = (record.prediction_id, record.prompt_version, record.llm_model, record.generated_at)
            unique[key] = record
        return sorted(unique.values(), key=lambda record: record.generated_at, reverse=True)[:limit]

    @staticmethod
    def _history_record(row: dict) -> RecommendationRecord:
        return RecommendationRecord.model_validate({
            **row, "timestamp": _dt(row["candle_open_at"]),
            "generated_at": _dt(row["generated_at"]),
            "requested_at": _dt(row["requested_at"]) if row.get("requested_at") else None,
            "valid_until": _dt(row["valid_until"]) if row.get("valid_until") else None,
            "llm_signal": row["llm_signal"] or None,
            "response_json": row["response_json"] or None,
        })

    @staticmethod
    def _record(row: dict) -> RecommendationRecord:
        return RecommendationRecord.model_validate({
            **row, "timestamp": _dt(row["time"]),
            "generated_at": _dt(row["generated_at"]),
            "requested_at": _dt(row["requested_at"]) if row.get("requested_at") else None,
            "valid_until": _dt(row["valid_until"]) if row.get("valid_until") else None,
            "llm_signal": row["llm_signal"] or None,
            "response_json": row["response_json"] or None,
        })


class Influx3RecommendationRequestRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3RecommendationRequestRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def save(self, request: RecommendationRequestRecord) -> None:
        fields = [
            f"finished_at={_field(_iso_utc(request.finished_at))}",
            f"interval={_field(request.interval)}",
            f"provider={_field(request.provider)}",
            f"llm_model={_field(request.llm_model)}",
            f"prompt_version={_field(request.prompt_version)}",
            f"total_latency_ms={request.total_latency_ms}",
            f"cached={str(request.cached).lower()}",
            f"created_at={_field(_iso_utc(request.created_at or request.finished_at))}",
        ]
        for name in ("prediction_id", "generated_at", "valid_until", "llm_latency_ms",
                     "timing_status", "error_type", "error_message"):
            value = getattr(request, name)
            if value is None:
                continue
            if isinstance(value, datetime):
                value = _iso_utc(value)
            fields.append(
                f"{name}={value}" if isinstance(value, (int, float)) else f"{name}={_field(str(value))}"
            )
        self._connection.write(
            f"recommendation_requests,request_id={_tag(str(request.request_id))},"
            f"symbol={_tag(request.symbol)},status={_tag(request.status)} "
            + ",".join(fields) + f" {_ms(request.requested_at)}"
        )

    @staticmethod
    def _record(row: dict) -> RecommendationRequestRecord:
        return RecommendationRequestRecord.model_validate({
            **row, "requested_at": _dt(row["time"]),
            "generated_at": _dt(row["generated_at"]) if row.get("generated_at") else None,
            "finished_at": _dt(row["finished_at"]),
            "valid_until": _dt(row["valid_until"]) if row.get("valid_until") else None,
            "created_at": _dt(row["created_at"]) if row.get("created_at") else None,
        })

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
        where = "symbol = $symbol"
        params = {"symbol": symbol}
        if status is not None:
            where += " AND status = $status"
            params["status"] = status
        rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM recommendation_requests WHERE {where} ORDER BY time DESC LIMIT {limit}",
            params,
        )
        return [self._record(row) for row in rows]


class Influx3PipelineRunRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3PipelineRunRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def save(self, run: PipelineRunRecord) -> None:
        fields = [
            f"finished_at={_field(_iso_utc(run.finished_at))}",
            f"duration_ms={run.duration_ms}",
            f"new_candle={str(run.new_candle).lower()}",
            f"new_prediction={str(run.new_prediction).lower()}",
            f"candles_ingested={run.candles_ingested}i",
            f"interval={_field(run.interval)}",
            f"created_at={_field(_iso_utc(run.created_at or run.finished_at))}",
        ]
        for name in ("market_data_ms", "prediction_ms", "evaluation_ms"):
            value = getattr(run, name)
            if value is not None:
                fields.append(f"{name}={value}")
        for name in (
            "prediction_id", "error_type", "error_message",
            "evaluation_error_type", "evaluation_error_message",
        ):
            value = getattr(run, name)
            if value is not None:
                fields.append(f"{name}={_field(value)}")
        if run.candle_open_time is not None:
            fields.append(f"candle_open_time={_field(_iso_utc(run.candle_open_time))}")
        self._connection.write(
            f"pipeline_runs,run_id={run.run_id},symbol={_tag(run.symbol)},status={run.status} "
            + ",".join(fields) + f" {_ms(run.started_at)}"
        )

    @staticmethod
    def _record(row: dict) -> PipelineRunRecord:
        return PipelineRunRecord.model_validate({
            **row, "started_at": _dt(row["time"]),
            "finished_at": _dt(row["finished_at"]),
            "candle_open_time": _dt(row["candle_open_time"]) if row.get("candle_open_time") else None,
            "created_at": _dt(row["created_at"]) if row.get("created_at") else None,
        })

    def get_latest(self, symbol: str, status: RunStatus | None = None) -> PipelineRunRecord | None:
        rows = self.list_recent(symbol, limit=1, status=status)
        return rows[0] if rows else None

    def list_recent(
        self, symbol: str, limit: int = 20, status: RunStatus | None = None
    ) -> list[PipelineRunRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit debe estar entre 1 y 1000")
        where = "symbol = $symbol"
        params = {"symbol": symbol}
        if status is not None:
            where += " AND status = $status"
            params["status"] = status
        rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM pipeline_runs WHERE {where} ORDER BY time DESC LIMIT {limit}",
            params,
        )
        return [self._record(row) for row in rows]


class Influx3LiveEvaluationRepository:
    def __init__(self, connection: Influx3Connection):
        self._connection = connection

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3LiveEvaluationRepository":
        return cls(Influx3Connection.from_settings(settings))

    def close(self) -> None:
        self._connection.close()

    def save(self, evaluation: LiveEvaluationRecord) -> bool:
        existing = self.get_by_prediction(evaluation.prediction_id)
        if existing is not None:
            return False
        fields = [
            f"evaluated_at={_field(_iso_utc(evaluation.evaluated_at))}",
            f"source_close={evaluation.source_close}",
            f"target_open_time={_field(_iso_utc(evaluation.target_open_time))}",
            f"target_close_time={_field(_iso_utc(evaluation.target_close_time))}",
            f"target_close={evaluation.target_close}",
            f"observed_move={_field(evaluation.observed_move)}",
            f"observed_class={_field(evaluation.observed_class)}",
            f"correct={str(evaluation.correct).lower()}",
        ]
        self._connection.write(
            f"live_evaluations,evaluation_id={evaluation.evaluation_id},"
            f"prediction_id={_tag(evaluation.prediction_id)},symbol={_tag(evaluation.symbol)} "
            + ",".join(fields) + f" {_ms(evaluation.target_close_time)}"
        )
        return True

    @staticmethod
    def _record(row: dict) -> LiveEvaluationRecord:
        return LiveEvaluationRecord.model_validate({
            **row, "evaluated_at": _dt(row["evaluated_at"]),
            "target_open_time": _dt(row["target_open_time"]),
            "target_close_time": _dt(row["target_close_time"]),
        })

    def get_by_prediction(self, prediction_id: str) -> LiveEvaluationRecord | None:
        rows = _query_or_empty(
            self._connection,
            "SELECT * FROM live_evaluations WHERE prediction_id = $prediction_id ORDER BY time DESC LIMIT 1",
            {"prediction_id": prediction_id},
        )
        return self._record(rows[0]) if rows else None

    def get_latest(self, symbol: str) -> LiveEvaluationRecord | None:
        rows = self.list_recent(symbol, limit=1)
        return rows[0] if rows else None

    def list_recent(self, symbol: str, limit: int = 100, offset: int = 0) -> list[LiveEvaluationRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("limit/offset fuera de rango")
        rows = _query_or_empty(
            self._connection,
            f"SELECT * FROM live_evaluations WHERE symbol = $symbol ORDER BY evaluated_at DESC LIMIT {limit} OFFSET {offset}",
            {"symbol": symbol},
        )
        unique = {row["prediction_id"]: self._record(row) for row in reversed(rows)}
        return list(reversed(list(unique.values())))
