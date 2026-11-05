"""Reglas ilustrativas de riesgo del PMV; pendientes de backtesting."""

import math
from typing import Literal

from pydantic import BaseModel, Field

from alpha_engine.config.settings import Settings
from alpha_engine.services.prediction import PredictionResult


class RiskResult(BaseModel):
    prediction_id: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    reference_price: float = Field(gt=0, allow_inf_nan=False)
    stop_loss: float = Field(gt=0, allow_inf_nan=False)
    take_profit: float = Field(gt=0, allow_inf_nan=False)
    triggers: list[str]
    rule_version: str = "risk_v1"


class RiskService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def analyze(
        self,
        prediction: PredictionResult,
        price: float,
        volatility: float,
        rsi: float,
        macd: float,
    ) -> RiskResult:
        if not all(math.isfinite(value) for value in (price, volatility, rsi, macd)):
            raise ValueError("Los insumos de riesgo deben ser finitos")
        if price <= 0 or volatility < 0 or not 0 <= rsi <= 100:
            raise ValueError("Precio, volatilidad o RSI fuera de rango")
        settings = self._settings
        high: list[str] = []
        medium: list[str] = []
        if volatility >= settings.risk_volatility_high:
            high.append("HIGH_VOLATILITY")
        elif volatility >= settings.risk_volatility_medium:
            medium.append("ELEVATED_VOLATILITY")
        if rsi <= settings.risk_rsi_high_low or rsi >= settings.risk_rsi_high_high:
            high.append("EXTREME_RSI")
        elif rsi <= settings.risk_rsi_medium_low or rsi >= settings.risk_rsi_medium_high:
            medium.append("ELEVATED_RSI")
        if max(prediction.prob_up, prediction.prob_down) < settings.model_confidence_threshold:
            high.append("LOW_MODEL_CONFIDENCE")
        if (prediction.predicted_class == "UP" and macd < 0) or (
            prediction.predicted_class == "DOWN" and macd > 0
        ):
            medium.append("MACD_CONTRADICTS_DIRECTION")
        level: Literal["LOW", "MEDIUM", "HIGH"] = (
            "HIGH" if high else "MEDIUM" if medium else "LOW"
        )

        stop_pct = min(
            max(settings.risk_min_stop_pct, volatility * settings.risk_stop_vol_multiplier),
            settings.risk_stop_cap_pct,
        )
        take_pct = stop_pct * settings.risk_reward_ratio
        if prediction.predicted_class == "UP":
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + take_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - take_pct)
        return RiskResult(
            prediction_id=prediction.prediction_id,
            risk_level=level,
            reference_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            triggers=high + medium,
        )

