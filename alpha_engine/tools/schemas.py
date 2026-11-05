"""Salidas tipadas de las herramientas y validación de coherencia temporal."""

from datetime import datetime
from math import isclose

from pydantic import BaseModel, Field, model_validator

from alpha_engine.risk.service import RiskResult
from alpha_engine.services.prediction import PredictionResult


class MarketSummary(BaseModel):
    symbol: str
    timestamp: datetime
    close_time: datetime
    price: float = Field(gt=0, allow_inf_nan=False)
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    change_1m_pct: float = Field(allow_inf_nan=False)
    volume: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_times(self) -> "MarketSummary":
        if self.timestamp.tzinfo is None or self.close_time.tzinfo is None:
            raise ValueError("Las marcas de tiempo deben incluir zona horaria")
        if self.close_time <= self.timestamp:
            raise ValueError("close_time debe ser posterior a timestamp")
        return self


class TechnicalIndicators(BaseModel):
    symbol: str
    timestamp: datetime
    return_1m: float = Field(allow_inf_nan=False)
    rsi_14: float = Field(ge=0, le=100, allow_inf_nan=False)
    ema_20: float = Field(gt=0, allow_inf_nan=False)
    macd: float = Field(allow_inf_nan=False)
    macd_signal: float = Field(allow_inf_nan=False)
    macd_histogram: float = Field(allow_inf_nan=False)
    rolling_volatility: float = Field(ge=0, allow_inf_nan=False)
    volume_change: float = Field(allow_inf_nan=False)


class RecommendationSnapshot(BaseModel):
    market: MarketSummary
    indicators: TechnicalIndicators
    prediction: PredictionResult
    risk: RiskResult

    @model_validator(mode="after")
    def validate_alignment(self) -> "RecommendationSnapshot":
        if not (
            self.market.symbol == self.indicators.symbol == self.prediction.symbol
            and self.market.timestamp == self.indicators.timestamp == self.prediction.timestamp
        ):
            raise ValueError("Mercado, indicadores y predicción no corresponden a la misma vela")
        if self.risk.prediction_id != self.prediction.prediction_id:
            raise ValueError("El análisis de riesgo no corresponde a la predicción")
        if not isclose(self.risk.reference_price, self.market.price, rel_tol=1e-12):
            raise ValueError("El precio de riesgo no coincide con el mercado")
        return self

