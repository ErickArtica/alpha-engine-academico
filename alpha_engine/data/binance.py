"""Cliente HTTP de Binance Public REST API. Solo devuelve velas cerradas."""

from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

from alpha_engine.data.schemas import Candle


class BinanceKlineClient:
    MAX_LIMIT = 1000

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        client: httpx.Client | None = None,
    ):
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=15.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_page(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        stop: datetime,
        limit: int = MAX_LIMIT,
    ) -> list[Candle]:
        """Consulta [start, stop). `stop` debe ser el fin de la última vela cerrada."""
        if start.tzinfo is None or stop.tzinfo is None or stop <= start:
            raise ValueError("start y stop deben tener zona horaria y stop > start")
        if not 1 <= limit <= self.MAX_LIMIT:
            raise ValueError("limit debe estar entre 1 y 1000")
        start_ms = int(start.timestamp() * 1000)
        stop_ms = int(stop.timestamp() * 1000)
        response = self._client.get(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": stop_ms - 1,
                "limit": limit,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Binance devolvió una respuesta de klines inesperada")
        candles_by_time: dict[datetime, Candle] = {}
        for row in payload:
            candle = self._parse_row(row, symbol, interval)
            if start <= candle.open_time < stop and candle.close_time < stop:
                candles_by_time[candle.open_time] = candle
        return sorted(candles_by_time.values(), key=lambda item: item.open_time)

    def fetch_v3_page(
        self, symbol: str, interval: str, start: datetime, stop: datetime,
        limit: int = MAX_LIMIT,
    ) -> pd.DataFrame:
        """Read full Binance Spot kline fields for isolated V3 shadow inference."""
        if start.tzinfo is None or stop.tzinfo is None or stop <= start:
            raise ValueError("Invalid V3 kline range")
        if not 1 <= limit <= self.MAX_LIMIT:
            raise ValueError("Invalid V3 kline limit")
        response = self._client.get(
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval,
                    "startTime": int(start.timestamp() * 1000),
                    "endTime": int(stop.timestamp() * 1000) - 1, "limit": limit},
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Invalid V3 kline response")
        rows = []
        for raw in payload:
            if not isinstance(raw, list) or len(raw) < 11:
                raise ValueError("Incomplete V3 kline")
            candle = self._parse_row(raw, symbol, interval)
            if start <= candle.open_time < stop and candle.close_time < stop:
                rows.append({"timestamp": candle.open_time, "open": candle.open,
                             "high": candle.high, "low": candle.low, "close": candle.close,
                             "volume": candle.volume, "trade_count": int(raw[8]),
                             "taker_buy_base": float(raw[9])})
        frame = pd.DataFrame(rows, columns=("timestamp", "open", "high", "low",
                                            "close", "volume", "trade_count", "taker_buy_base"))
        if not frame.empty and (frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing):
            raise ValueError("Unordered V3 klines")
        return frame

    @staticmethod
    def _parse_row(row: Any, symbol: str, interval: str) -> Candle:
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError("Binance devolvió una vela incompleta")
        try:
            return Candle(
                symbol=symbol,
                interval=interval,
                open_time=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=datetime.fromtimestamp(int(row[6]) / 1000, tz=timezone.utc),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Binance devolvió una vela inválida") from exc
