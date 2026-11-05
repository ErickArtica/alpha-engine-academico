"""Snapshot REST del mejor bid/ask Spot; hora de mercado no incluida."""

from collections.abc import Callable
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field, model_validator


class BookTickerQuote(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9]{3,20}$")
    requested_at: datetime
    received_at: datetime
    bid_price: float = Field(gt=0, allow_inf_nan=False)
    bid_qty: float = Field(ge=0, allow_inf_nan=False)
    ask_price: float = Field(gt=0, allow_inf_nan=False)
    ask_qty: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_quote(self) -> "BookTickerQuote":
        if self.requested_at.tzinfo is None or self.received_at.tzinfo is None:
            raise ValueError("La cotización requiere horas con zona horaria")
        if self.received_at < self.requested_at or self.ask_price < self.bid_price:
            raise ValueError("Cotización con horas o spread inválidos")
        return self


class BinanceBookTickerClient:
    def __init__(
        self, base_url: str = "https://api.binance.com",
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=3.0)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(self, symbol: str) -> BookTickerQuote:
        if not symbol or not symbol.isascii() or not symbol.isalnum() or symbol != symbol.upper():
            raise ValueError("Símbolo Spot inválido")
        started = self._clock()
        response = self._client.get("/api/v3/ticker/bookTicker", params={"symbol": symbol})
        received = self._clock()
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("symbol") != symbol:
            raise ValueError("Binance devolvió bookTicker de otro símbolo")
        try:
            return BookTickerQuote(
                symbol=symbol, requested_at=started, received_at=received,
                bid_price=float(payload["bidPrice"]), bid_qty=float(payload["bidQty"]),
                ask_price=float(payload["askPrice"]), ask_qty=float(payload["askQty"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Binance devolvió bookTicker inválido") from exc
