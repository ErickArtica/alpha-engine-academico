"""Cálculos estructurados sobre velas cerradas del repositorio."""

from collections.abc import Sequence
from datetime import timedelta

import numpy as np

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from alpha_engine.tools.schemas import MarketSummary, TechnicalIndicators


class AnalyticsDataError(ValueError):
    """Faltan datos válidos para una herramienta analítica."""


def get_market_summary(candles: Sequence[Candle]) -> MarketSummary:
    if len(candles) < 2:
        raise AnalyticsDataError("Se necesitan al menos dos velas cerradas")
    if len({candle.symbol for candle in candles}) != 1 or any(
        candle.interval != "1m" for candle in candles
    ):
        raise AnalyticsDataError("Las velas deben ser de un símbolo e intervalo 1m")
    previous, latest = sorted(candles, key=lambda candle: candle.open_time)[-2:]
    if latest.open_time - previous.open_time != timedelta(minutes=1):
        raise AnalyticsDataError("Falta la vela del minuto anterior")
    return MarketSummary(
        symbol=latest.symbol,
        timestamp=latest.open_time,
        close_time=latest.close_time,
        price=latest.close,
        open=latest.open,
        high=latest.high,
        low=latest.low,
        change_1m_pct=(latest.close / previous.close - 1) * 100,
        volume=latest.volume,
    )


def get_technical_indicators(candles: Sequence[Candle]) -> TechnicalIndicators:
    try:
        frame = build_feature_frame(candles)
    except ValueError as exc:
        raise AnalyticsDataError(str(exc)) from exc
    row = frame.iloc[-1]
    if not np.isfinite(row[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise AnalyticsDataError("No hay historia suficiente para indicadores completos")
    return TechnicalIndicators(
        symbol=str(row["symbol"]),
        timestamp=frame.index[-1].to_pydatetime(),
        **{name: float(row[name]) for name in FEATURE_COLUMNS},
    )
