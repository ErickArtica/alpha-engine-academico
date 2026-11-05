"""Small error categories shared by the LIVE ingestion and prediction services."""


class MarketDataError(RuntimeError):
    """Binance did not provide usable market data for this cycle."""


class StorageError(RuntimeError):
    """The selected repository failed to read or persist domain data."""
