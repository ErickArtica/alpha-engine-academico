"""Common one-minute prediction horizon semantics."""

from datetime import datetime, timedelta


def prediction_expires_at(source_close_time: datetime, horizon: str) -> datetime:
    """End of the target candle, matching the existing history audit convention."""
    if horizon != "1m":
        raise ValueError(f"Horizonte no soportado: {horizon}")
    if source_close_time.tzinfo is None:
        raise ValueError("source_close_time debe incluir zona horaria")
    return source_close_time + timedelta(minutes=1)
