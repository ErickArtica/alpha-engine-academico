"""Recursos de ingesta compartidos por comandos y scheduler."""

from contextlib import contextmanager
from typing import Iterator

from alpha_engine.config.settings import Settings
from alpha_engine.data.binance import BinanceKlineClient
from alpha_engine.repositories.factory import create_market_repository
from alpha_engine.services.ingestion import MarketIngestionService


@contextmanager
def ingestion_service(settings: Settings) -> Iterator[MarketIngestionService]:
    repository = create_market_repository(settings)
    try:
        source = BinanceKlineClient(base_url=settings.binance_base_url)
        try:
            yield MarketIngestionService(source, repository)
        finally:
            source.close()
    finally:
        repository.close()

