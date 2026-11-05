"""Captura opcional de cotizaciones; un fallo no altera la recomendación."""

import logging

import httpx

from alpha_engine.data.book_ticker import BinanceBookTickerClient, BookTickerQuote
from alpha_engine.observability.telemetry import TelemetryStore

logger = logging.getLogger(__name__)


def capture_quote(
    client: BinanceBookTickerClient, store: TelemetryStore,
    symbol: str, trace_id: str, prediction_id: str | None = None,
) -> BookTickerQuote | None:
    try:
        quote = client.fetch(symbol)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("No se pudo capturar bookTicker para %s: %s", symbol, exc)
        store.event("quote_failed", trace_id, symbol, prediction_id=prediction_id, details={"error_type": type(exc).__name__})
        return None
    store.quote(quote, trace_id, prediction_id)
    store.event("quote_received", trace_id, symbol, at=quote.received_at, prediction_id=prediction_id)
    return quote
