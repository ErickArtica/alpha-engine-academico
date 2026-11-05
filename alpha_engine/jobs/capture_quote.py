"""Ejecutar: python -m alpha_engine.jobs.capture_quote."""

import argparse
from uuid import uuid4

from alpha_engine.config.settings import get_settings
from alpha_engine.data.book_ticker import BinanceBookTickerClient
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.services.quote_capture import capture_quote
from alpha_engine.utils.logging import configure_logging


def main() -> int:
    argparse.ArgumentParser(description="Guarda un snapshot bid/ask Spot actual").parse_args()
    configure_logging()
    settings = get_settings()
    client = BinanceBookTickerClient(settings.binance_base_url)
    try:
        quote = capture_quote(client, TelemetryStore(), settings.symbol, uuid4().hex)
    finally:
        client.close()
    if quote is None:
        return 1
    print(quote.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
