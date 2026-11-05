"""Registro append-only de eventos y cotizaciones, separado de las señales."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from alpha_engine.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)
DEFAULT_DIR = PROJECT_ROOT / "runtime" / "telemetry"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Las marcas de telemetría requieren zona horaria")
    return value.astimezone(timezone.utc).isoformat()


class TelemetryStore:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self.directory = directory

    def _append(self, name: str, data: dict[str, Any]) -> bool:
        path = self.directory / name
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return True
        except OSError:
            logger.exception("No se pudo guardar telemetría en %s", path)
            return False

    def event(
        self, stage: str, trace_id: str, symbol: str,
        *, at: datetime | None = None, candle_open_time: datetime | None = None,
        prediction_id: str | None = None, duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        if not stage or not trace_id or not symbol:
            raise ValueError("stage, trace_id y symbol son obligatorios")
        if duration_ms is not None and duration_ms < 0:
            raise ValueError("duration_ms no puede ser negativo")
        return self._append("events.jsonl", {
            "stage": stage, "trace_id": trace_id, "symbol": symbol,
            "at": _iso(at or utc_now()),
            "candle_open_time": _iso(candle_open_time) if candle_open_time else None,
            "prediction_id": prediction_id,
            "duration_ms": duration_ms,
            "details": details or {},
        })

    def quote(self, quote: "BookTickerQuote", trace_id: str, prediction_id: str | None = None) -> bool:
        from alpha_engine.data.book_ticker import BookTickerQuote

        if not isinstance(quote, BookTickerQuote) or not trace_id:
            raise ValueError("Cotización o trace_id inválido")
        return self._append("quotes.jsonl", {
            "trace_id": trace_id, "prediction_id": prediction_id,
            **quote.model_dump(mode="json"),
        })

    def read_events(self) -> list[dict]:
        return self._read("events.jsonl")

    def read_quotes(self) -> list[dict]:
        return self._read("quotes.jsonl")

    def _read(self, name: str) -> list[dict]:
        path = self.directory / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
