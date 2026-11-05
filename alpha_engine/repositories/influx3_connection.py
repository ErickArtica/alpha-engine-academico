"""Cliente HTTP reutilizable para InfluxDB 3 Core: SQL y line protocol."""

from collections.abc import Mapping
from typing import Any

import httpx

from alpha_engine.config.settings import Settings


class Influx3Connection:
    def __init__(self, url: str, database: str, token: str, client: httpx.Client | None = None):
        if not token:
            raise ValueError("INFLUX3_TOKEN debe configurarse para conectar con InfluxDB 3")
        self.database = database
        self._client = client or httpx.Client(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Influx3Connection":
        return cls(
            settings.influx3_url,
            settings.influx3_database,
            settings.influx3_token.get_secret_value(),
        )

    def close(self) -> None:
        self._client.close()

    def write(self, line_protocol: str) -> None:
        response = self._client.post(
            "/api/v3/write_lp",
            params={"db": self.database, "precision": "millisecond"},
            content=line_protocol.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        response.raise_for_status()

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        response = self._client.post(
            "/api/v3/query_sql",
            json={"db": self.database, "q": sql, "params": dict(params or {})},
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError("InfluxDB 3 devolvió un formato SQL inesperado")
        return rows
