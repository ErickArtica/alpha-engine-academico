"""Ingesta paginada e incremental de velas cerradas."""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Protocol

import httpx

from alpha_engine.data.schemas import Candle
from alpha_engine.repositories.protocols import MarketRepository
from alpha_engine.services.errors import MarketDataError, StorageError

logger = logging.getLogger(__name__)

_INTERVAL_RE = re.compile(r"^([1-9][0-9]*)([mhd])$")
_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def interval_duration(interval: str) -> timedelta:
    match = _INTERVAL_RE.fullmatch(interval)
    if match is None:
        raise ValueError(f"Intervalo no soportado para ingesta: {interval}")
    return timedelta(seconds=int(match.group(1)) * _SECONDS[match.group(2)])


def closed_boundary(now: datetime, interval: str) -> datetime:
    """Marca UTC de apertura de la vela en curso; las anteriores ya cerraron."""
    if now.tzinfo is None:
        raise ValueError("now debe tener zona horaria")
    step_seconds = int(interval_duration(interval).total_seconds())
    return datetime.fromtimestamp(
        int(now.timestamp()) // step_seconds * step_seconds, tz=timezone.utc
    )


class KlineSource(Protocol):
    def fetch_page(
        self, symbol: str, interval: str, start: datetime, stop: datetime, limit: int = 1000
    ) -> list[Candle]: ...


class MarketIngestionService:
    def __init__(self, source: KlineSource, repository: MarketRepository):
        self._source = source
        self._repository = repository

    def backfill(
        self, symbol: str, interval: str, days: int, now: datetime | None = None
    ) -> int:
        if days < 1:
            raise ValueError("days debe ser >= 1")
        stop = closed_boundary(now or datetime.now(timezone.utc), interval)
        start = stop - timedelta(days=days)
        return self.ingest_range(symbol, interval, start, stop)

    def fetch_latest(
        self, symbol: str, interval: str, now: datetime | None = None,
        *, bootstrap_candles: int = 2,
    ) -> int:
        if bootstrap_candles < 1:
            raise ValueError("bootstrap_candles debe ser >= 1")
        stop = closed_boundary(now or datetime.now(timezone.utc), interval)
        try:
            latest = self._repository.get_latest_candle(symbol, interval)
        except Exception as exc:
            raise StorageError("No se pudo consultar la ultima vela") from exc
        step = interval_duration(interval)
        start = latest.open_time + step if latest else stop - step * bootstrap_candles
        if start >= stop:
            logger.info("No hay velas cerradas nuevas para %s %s", symbol, interval)
            return 0
        return self.ingest_range(symbol, interval, start, stop)

    def ingest_range(
        self, symbol: str, interval: str, start: datetime, stop: datetime
    ) -> int:
        if start.tzinfo is None or stop.tzinfo is None or stop <= start:
            raise ValueError("Rango de ingesta inválido")
        step = interval_duration(interval)
        cursor = start
        total_sent = 0
        while cursor < stop:
            try:
                page = self._source.fetch_page(symbol, interval, cursor, stop)
            except (httpx.HTTPError, ValueError) as exc:
                raise MarketDataError("Binance no devolvio velas utilizables") from exc
            if not page:
                break
            if any(
                candle.symbol != symbol
                or candle.interval != interval
                or candle.open_time < cursor
                or candle.open_time >= stop
                or candle.close_time >= stop
                for candle in page
            ):
                raise ValueError("La fuente devolvió velas abiertas o fuera del rango solicitado")
            if any(left.open_time >= right.open_time for left, right in zip(page, page[1:])):
                raise ValueError("La fuente devolvió velas sin orden temporal estricto")
            next_cursor = page[-1].open_time + step
            if next_cursor <= cursor:
                raise ValueError("La paginación de Binance no avanzó")
            try:
                sent = self._repository.upsert_candles(page)
            except Exception as exc:
                raise StorageError("No se pudieron persistir las velas") from exc
            total_sent += sent
            logger.info(
                "Lote de velas enviado: symbol=%s interval=%s count=%s last=%s",
                symbol, interval, sent, page[-1].open_time.isoformat(),
            )
            cursor = next_cursor
        logger.info(
            "Ingesta terminada: symbol=%s interval=%s points_sent=%s",
            symbol, interval, total_sent,
        )
        return total_sent
