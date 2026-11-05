"""Adaptador InfluxDB 2 para velas. Una vela = serie y open_time únicos."""

import json
from datetime import datetime, timezone
from typing import Sequence

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle

class InfluxMarketRepository:
    def __init__(self, client: InfluxDBClient, org: str, bucket: str):
        self._client = client
        self._org = org
        self._bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> "InfluxMarketRepository":
        client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token.get_secret_value(),
            org=settings.influx_org,
            timeout=10_000,
        )
        return cls(client, settings.influx_org, settings.influx_bucket)

    def close(self) -> None:
        self._client.close()

    def upsert_candles(self, candles: Sequence[Candle]) -> int:
        if not candles:
            return 0
        points = [
            Point("candles")
            .tag("symbol", candle.symbol)
            .tag("interval", candle.interval)
            .field("open", candle.open)
            .field("high", candle.high)
            .field("low", candle.low)
            .field("close", candle.close)
            .field("volume", candle.volume)
            .field("close_time_ms", int(candle.close_time.timestamp() * 1000))
            .time(int(candle.open_time.timestamp() * 1000), WritePrecision.MS)
            for candle in candles
        ]
        self._client.write_api(write_options=SYNCHRONOUS).write(
            bucket=self._bucket, org=self._org, record=points
        )
        return len(points)

    def get_candles(
        self, symbol: str, interval: str, start: datetime, stop: datetime
    ) -> list[Candle]:
        if start.tzinfo is None or stop.tzinfo is None or stop <= start:
            raise ValueError("start y stop deben tener zona horaria y stop > start")
        query = self._base_query(symbol, interval, f"{self._time(start)}, stop: {self._time(stop)}")
        query += ' |> sort(columns: ["_time"])'
        return self._execute(query)

    def get_latest_candle(self, symbol: str, interval: str) -> Candle | None:
        query = self._base_query(symbol, interval, "0")
        query += ' |> sort(columns: ["_time"], desc: true) |> limit(n: 1)'
        candles = self._execute(query)
        return candles[0] if candles else None

    def _base_query(self, symbol: str, interval: str, range_args: str) -> str:
        # JSON quoting impide que valores de configuración o tags alteren la consulta Flux.
        return (
            f"from(bucket: {json.dumps(self._bucket)})"
            f" |> range(start: {range_args})"
            ' |> filter(fn: (r) => r._measurement == "candles")'
            f" |> filter(fn: (r) => r.symbol == {json.dumps(symbol)}"
            f" and r.interval == {json.dumps(interval)})"
            ' |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
        )

    @staticmethod
    def _time(value: datetime) -> str:
        return f"time(v: {json.dumps(value.astimezone(timezone.utc).isoformat())})"

    def _execute(self, query: str) -> list[Candle]:
        tables = self._client.query_api().query(query=query, org=self._org)
        result = []
        for table in tables:
            for record in table.records:
                fields = record.values
                result.append(
                    Candle(
                        symbol=fields["symbol"],
                        interval=fields["interval"],
                        open_time=record.get_time().astimezone(timezone.utc),
                        open=fields["open"],
                        high=fields["high"],
                        low=fields["low"],
                        close=fields["close"],
                        volume=fields["volume"],
                        close_time=datetime.fromtimestamp(
                            fields["close_time_ms"] / 1000, tz=timezone.utc
                        ),
                    )
                )
        return sorted(result, key=lambda candle: candle.open_time)
