"""Adaptador InfluxDB 2 para predicciones identificadas por vela y modelo."""

import json
from datetime import timezone

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import PredictionRecord


class InfluxPredictionRepository:
    def __init__(self, client: InfluxDBClient, org: str, bucket: str):
        self._client = client
        self._org = org
        self._bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> "InfluxPredictionRepository":
        client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token.get_secret_value(),
            org=settings.influx_org,
            timeout=10_000,
        )
        return cls(client, settings.influx_org, settings.influx_bucket)

    def close(self) -> None:
        self._client.close()

    def save(self, prediction: PredictionRecord) -> None:
        point = (
            Point("predictions")
            .tag("prediction_id", prediction.prediction_id)
            .tag("symbol", prediction.symbol)
            .tag("model_version", prediction.model_version)
            .field("horizon", prediction.horizon)
            .field("prob_up", prediction.prob_up)
            .field("prob_down", prediction.prob_down)
            .field("predicted_class", prediction.predicted_class)
            .time(int(prediction.timestamp.timestamp() * 1000), WritePrecision.MS)
        )
        if prediction.trained_at is not None:
            point.field("trained_at", prediction.trained_at.isoformat())
        self._client.write_api(write_options=SYNCHRONOUS).write(
            bucket=self._bucket, org=self._org, record=point
        )

    def get_by_id(self, prediction_id: str) -> PredictionRecord | None:
        query = self._base_query()
        query += f' |> filter(fn: (r) => r.prediction_id == {json.dumps(prediction_id)})'
        return self._first(query)

    def get_latest(self, symbol: str) -> PredictionRecord | None:
        query = self._base_query()
        query += f' |> filter(fn: (r) => r.symbol == {json.dumps(symbol)})'
        return self._first(query)

    def _base_query(self) -> str:
        return (
            f"from(bucket: {json.dumps(self._bucket)})"
            " |> range(start: 0)"
            ' |> filter(fn: (r) => r._measurement == "predictions")'
        )

    def _first(self, query: str) -> PredictionRecord | None:
        query += (
            ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
            ' |> group(columns: []) |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 1)"
        )
        tables = self._client.query_api().query(query=query, org=self._org)
        for table in tables:
            for record in table.records:
                fields = record.values
                return PredictionRecord(
                    prediction_id=fields["prediction_id"],
                    symbol=fields["symbol"],
                    timestamp=record.get_time().astimezone(timezone.utc),
                    horizon=fields["horizon"],
                    prob_up=fields["prob_up"],
                    prob_down=fields["prob_down"],
                    predicted_class=fields["predicted_class"],
                    model_version=fields["model_version"],
                    trained_at=fields.get("trained_at"),
                )
        return None
