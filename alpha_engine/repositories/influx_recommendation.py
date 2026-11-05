"""Caché y trazabilidad de recomendaciones en InfluxDB 2."""

import json
from datetime import datetime, timezone
from uuid import uuid4

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import RecommendationRecord


class InfluxRecommendationRepository:
    def __init__(self, client: InfluxDBClient, org: str, bucket: str):
        self._client = client
        self._org = org
        self._bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> "InfluxRecommendationRepository":
        client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token.get_secret_value(),
            org=settings.influx_org,
            timeout=10_000,
        )
        return cls(client, settings.influx_org, settings.influx_bucket)

    def close(self) -> None:
        self._client.close()

    def save(self, recommendation: RecommendationRecord) -> None:
        current = (
            Point("recommendations")
            .tag("prediction_id", recommendation.prediction_id)
            .tag("prompt_version", recommendation.prompt_version)
            .tag("llm_model", recommendation.llm_model)
            .field("generated_at", recommendation.generated_at.isoformat())
            .field("candidate_signal", recommendation.candidate_signal)
            .field("llm_signal", recommendation.llm_signal or "")
            .field("final_signal", recommendation.final_signal)
            .field("risk_level", recommendation.risk_level)
            .field("context_json", recommendation.context_json)
            .field("response_json", recommendation.response_json or "")
            .field("guardrails_json", recommendation.guardrails_json)
            .time(int(recommendation.timestamp.timestamp() * 1000), WritePrecision.MS)
        )
        history = (
            Point("recommendation_history")
            .tag("prediction_id", recommendation.prediction_id)
            .tag("prompt_version", recommendation.prompt_version)
            .tag("llm_model", recommendation.llm_model)
            .tag("revision_id", uuid4().hex)
            .field("candle_open_at", recommendation.timestamp.isoformat())
            .field("generated_at", recommendation.generated_at.isoformat())
            .field("candidate_signal", recommendation.candidate_signal)
            .field("llm_signal", recommendation.llm_signal or "")
            .field("final_signal", recommendation.final_signal)
            .field("risk_level", recommendation.risk_level)
            .field("context_json", recommendation.context_json)
            .field("response_json", recommendation.response_json or "")
            .field("guardrails_json", recommendation.guardrails_json)
            .time(int(recommendation.generated_at.timestamp() * 1000), WritePrecision.MS)
        )
        self._client.write_api(write_options=SYNCHRONOUS).write(
            bucket=self._bucket, org=self._org, record=[current, history]
        )

    def get_cached(
        self, prediction_id: str, prompt_version: str, llm_model: str
    ) -> RecommendationRecord | None:
        query = (
            f"from(bucket: {json.dumps(self._bucket)})"
            " |> range(start: 0)"
            ' |> filter(fn: (r) => r._measurement == "recommendations")'
            f" |> filter(fn: (r) => r.prediction_id == {json.dumps(prediction_id)}"
            f" and r.prompt_version == {json.dumps(prompt_version)}"
            f" and r.llm_model == {json.dumps(llm_model)})"
            ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
            ' |> group(columns: []) |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 1)"
        )
        tables = self._client.query_api().query(query=query, org=self._org)
        for table in tables:
            for record in table.records:
                return self._record(record)
        return None

    def save_if_absent(self, recommendation: RecommendationRecord) -> bool:
        if self.get_cached(
            recommendation.prediction_id, recommendation.prompt_version, recommendation.llm_model
        ) is not None:
            return False
        self.save(recommendation)
        return True

    def list_recent(self, limit: int = 20) -> list[RecommendationRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit debe estar entre 1 y 100")
        base = (
            f"from(bucket: {json.dumps(self._bucket)})"
            " |> range(start: 0)"
        )
        suffix = (
            ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
            ' |> group(columns: []) |> sort(columns: ["_time"], desc: true)'
            f" |> limit(n: {limit})"
        )
        history_query = base + ' |> filter(fn: (r) => r._measurement == "recommendation_history")' + suffix
        current_query = base + ' |> filter(fn: (r) => r._measurement == "recommendations")' + suffix
        history = self._client.query_api().query(query=history_query, org=self._org)
        current = self._client.query_api().query(query=current_query, org=self._org)
        combined = [self._history_record(record) for table in history for record in table.records if "candle_open_at" in record.values]
        combined.extend(self._record(record) for table in current for record in table.records)
        unique = {}
        for record in combined:
            key = (record.prediction_id, record.prompt_version, record.llm_model, record.generated_at)
            unique[key] = record
        return sorted(unique.values(), key=lambda record: record.generated_at, reverse=True)[:limit]

    @staticmethod
    def _history_record(record) -> RecommendationRecord:
        fields = record.values
        return RecommendationRecord(
            prediction_id=fields["prediction_id"],
            prompt_version=fields["prompt_version"],
            llm_model=fields["llm_model"],
            timestamp=datetime.fromisoformat(fields["candle_open_at"]),
            generated_at=datetime.fromisoformat(fields["generated_at"]),
            candidate_signal=fields["candidate_signal"],
            llm_signal=fields["llm_signal"] or None,
            final_signal=fields["final_signal"],
            risk_level=fields["risk_level"],
            context_json=fields["context_json"],
            response_json=fields["response_json"] or None,
            guardrails_json=fields["guardrails_json"],
        )

    @staticmethod
    def _record(record) -> RecommendationRecord:
        fields = record.values
        return RecommendationRecord(
            prediction_id=fields["prediction_id"],
            prompt_version=fields["prompt_version"],
            llm_model=fields["llm_model"],
            timestamp=record.get_time().astimezone(timezone.utc),
            generated_at=datetime.fromisoformat(fields["generated_at"]),
            candidate_signal=fields["candidate_signal"],
            llm_signal=fields["llm_signal"] or None,
            final_signal=fields["final_signal"],
            risk_level=fields["risk_level"],
            context_json=fields["context_json"],
            response_json=fields["response_json"] or None,
            guardrails_json=fields["guardrails_json"],
        )
