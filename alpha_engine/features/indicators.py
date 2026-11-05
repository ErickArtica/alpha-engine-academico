"""Features calculadas solo con la vela t y velas anteriores."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alpha_engine.data.schemas import Candle

FEATURE_COLUMNS = (
    "return_1m",
    "rsi_14",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rolling_volatility",
    "volume_change",
)


def build_feature_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    """Devuelve OHLCV e indicadores con índice open_time UTC.

    Exige una única serie BTC/USDT u otro símbolo, intervalo 1m y velas
    consecutivas. Las primeras filas sin historia suficiente conservan NaN;
    jamás se rellenan con valores futuros.
    """
    if not candles:
        raise ValueError("Se necesitan velas para calcular features")
    if len({c.symbol for c in candles}) != 1:
        raise ValueError("Las velas deben pertenecer al mismo símbolo")
    if any(c.interval != "1m" for c in candles):
        raise ValueError("Este modelo inicial requiere velas de 1m")

    frame = pd.DataFrame([c.model_dump() for c in candles])
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.set_index("open_time").sort_index()
    if not frame.index.is_unique:
        raise ValueError("Hay velas con open_time duplicado")
    if len(frame) > 1 and not frame.index.to_series().diff().iloc[1:].dt.total_seconds().eq(
        60.0
    ).all():
        raise ValueError("Hay huecos en las velas de 1m; complete el backfill")

    close = frame["close"].astype("float64")
    volume = frame["volume"].astype("float64")
    frame["return_1m"] = close.pct_change(fill_method=None)

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    relative_strength = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    frame["rsi_14"] = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    frame["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    frame["macd_histogram"] = frame["macd"] - frame["macd_signal"]
    frame["rolling_volatility"] = frame["return_1m"].rolling(
        window=20, min_periods=20
    ).std(ddof=0)
    frame["volume_change"] = volume.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    return frame
