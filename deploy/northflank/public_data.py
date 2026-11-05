"""Narrow GET-only PostgREST reader. No SDK, RPC, auth flow or write API."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx

from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.data.live_schemas import LiveEvaluationRecord
from alpha_engine.services.live_evaluation import LiveEvaluationService
from alpha_engine.services.paper_policy import paper_decision
from deploy.northflank.public_config import PublicConfig


TABLES = {
    "candles": ("symbol,interval,open_time,close_time,open,high,low,close,volume", "open_time.desc"),
    "predictions": ("prediction_id,symbol,interval,horizon,candle_open_time,prob_up,prob_down,predicted_class,model_version,trained_at,generated_at,valid_until", "candle_open_time.desc,prediction_id.desc"),
    "live_evaluations": ("evaluation_id,prediction_id,symbol,evaluated_at,source_close,target_open_time,target_close_time,target_close,observed_move,observed_class,correct", "evaluated_at.desc,prediction_id.desc"),
}


class ReadUnavailable(RuntimeError):
    pass


class ReadOnlyViolation(PermissionError):
    pass


class ReadOnlyTransport(httpx.BaseTransport):
    """Enforce policy before any bytes/credentials reach the delegate transport."""

    def __init__(self, config: PublicConfig, delegate: httpx.BaseTransport):
        self._origin = httpx.URL(config.supabase_url)
        self._delegate = delegate
        self._symbol = "eq." + config.symbol

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        origin = (url.scheme, url.host, url.port)
        if request.method != "GET" or origin != (self._origin.scheme, self._origin.host, self._origin.port):
            raise ReadOnlyViolation("Only GET to the configured Supabase origin is allowed")
        path = url.raw_path.partition(b"?")[0]
        resource = next((name for name in TABLES if path == ("/rest/v1/" + name).encode()), None)
        if resource is None or request.content:
            raise ReadOnlyViolation("Resource, RPC, encoded path or body not allowed")
        params = list(url.params.multi_items())
        values = dict(params)
        keys = {"select", "symbol", "order", "limit"} | ({"interval", "close_time"} if resource == "candles" else {"interval", "model_version"} if resource == "predictions" else set())
        if len(params) != len(values) or set(values) != keys:
            raise ReadOnlyViolation("Unexpected/duplicate query parameters")
        columns, order = TABLES[resource]
        if values["select"] != columns or values["order"] != order or values["symbol"] != self._symbol:
            raise ReadOnlyViolation("Only reviewed columns, ordering and symbol filters are allowed")
        if not values["limit"].isdigit() or not 1 <= int(values["limit"]) <= 1000:
            raise ReadOnlyViolation("Read limit must be between 1 and 1000")
        if "interval" in values and values["interval"] != "eq.1m":
            raise ReadOnlyViolation("Only the 1m series is allowed")
        if resource == "predictions" and values["model_version"] != "eq.xgboost_v1":
            raise ReadOnlyViolation("Only persisted V1 predictions are allowed")
        if resource == "candles":
            try:
                cutoff = datetime.fromisoformat(values["close_time"].removeprefix("lte."))
            except ValueError:
                raise ReadOnlyViolation("Invalid closed-candle cutoff") from None
            if not values["close_time"].startswith("lte.") or cutoff.tzinfo is None or cutoff > datetime.now(timezone.utc):
                raise ReadOnlyViolation("Only closed candles may be read")
        return self._delegate.handle_request(request)

    def close(self) -> None:
        self._delegate.close()


class PublicReader:
    def __init__(self, config: PublicConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        headers = {"apikey": config.supabase_key, "Accept": "application/json"}
        if not config.supabase_key.startswith("sb_publishable_"):
            headers["Authorization"] = "Bearer " + config.supabase_key
        delegate = transport if transport is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        self._http = httpx.Client(base_url=config.supabase_url, headers=headers,
                                  transport=ReadOnlyTransport(config, delegate), trust_env=False,
                                  follow_redirects=False, timeout=httpx.Timeout(8, connect=3))

    def __enter__(self) -> "PublicReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self._http.close()

    def rows(self, resource: str, limit: int) -> list[dict]:
        if resource not in TABLES or type(limit) is not int or not 1 <= limit <= 1000:
            raise ReadOnlyViolation("Unknown resource or invalid limit")
        columns, order = TABLES[resource]
        params = dict(select=columns, symbol="eq." + self.config.symbol, order=order, limit=str(limit))
        if resource in {"candles", "predictions"}:
            params["interval"] = "eq.1m"
        if resource == "predictions":
            params["model_version"] = "eq.xgboost_v1"
        if resource == "candles":
            params["close_time"] = "lte." + datetime.now(timezone.utc).isoformat()
        try:
            # Stream the response with an explicit memory bound; never follow redirects.
            with self._http.stream("GET", "/rest/v1/" + resource, params=params) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(body) + len(chunk) > 4_000_000:
                        raise ReadUnavailable("Public read response exceeded its limit")
                    body.extend(chunk)
            import json
            rows = json.loads(body)
            if not isinstance(rows, list) or len(rows) > limit or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Unexpected response shape")
            # Extra server columns are discarded before they reach the UI/cache.
            names = set(columns.split(","))
            return [{k: v for k, v in row.items() if k in names} for row in rows]
        except (httpx.HTTPError, ValueError, ReadUnavailable):
            raise ReadUnavailable("No se pudo leer la información pública almacenada.") from None

    def candles(self, limit: int = 240) -> list[Candle]:
        candles = sorted((Candle.model_validate(row) for row in self.rows("candles", limit)), key=lambda c: c.open_time)
        if any(c.symbol != self.config.symbol or c.interval != "1m" or c.close_time > datetime.now(timezone.utc) for c in candles):
            raise ReadUnavailable("Unexpected candle series")
        return candles

    def predictions(self, limit: int = 20) -> list[PredictionRecord]:
        return [PredictionRecord.model_validate({**row, "timestamp": row["candle_open_time"]}) for row in self.rows("predictions", limit)]

    def evaluations(self, limit: int = 500) -> list[LiveEvaluationRecord]:
        return [LiveEvaluationRecord.model_validate(row) for row in self.rows("live_evaluations", limit)]

    def recent_metrics(self) -> dict:
        predictions = self.predictions(1000)
        evaluations = self.evaluations(500)
        # Reuse the existing pure calculation with bounded in-memory readers.
        # No persistent repository or evaluation workflow is constructed.
        def recent(items):
            return SimpleNamespace(list_recent=lambda symbol, limit=100, offset=0: items[offset:offset + limit])
        metrics = LiveEvaluationService.metrics_for(recent(predictions), recent(evaluations), self.config.symbol, "xgboost_v1")
        return asdict(metrics)


def prescriptive_scenarios(prediction: PredictionRecord | None, latest_candle: Candle | None,
                           now: datetime | None = None) -> list[dict]:
    instant = now or datetime.now(timezone.utc)
    current = latest_candle is not None and 0 <= (instant - latest_candle.close_time).total_seconds() <= 120
    usable = (prediction if current and prediction is not None and prediction.model_version == "xgboost_v1"
              and prediction.timestamp == latest_candle.open_time else None)
    scenarios = []
    for has_long in (False, True):
        decision = paper_decision(usable, has_long=has_long, threshold=0.60, now=instant)
        scenarios.append(dict(escenario="Sin posición" if not has_long else "Con posición LONG hipotética",
                              accion=decision.action, motivo=decision.reason if current else "Datos de mercado ausentes o desactualizados."))
    return scenarios
