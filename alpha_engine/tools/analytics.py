"""Implementación de herramientas mediante contratos del dominio."""

from datetime import datetime, timedelta, timezone
from typing import Protocol

from alpha_engine.data.schemas import Candle
from alpha_engine.repositories.protocols import MarketRepository
from alpha_engine.risk.service import RiskResult, RiskService
from alpha_engine.services.prediction import PredictionResult, PredictionService
from alpha_engine.tools.market import (
    AnalyticsDataError,
    get_market_summary,
    get_technical_indicators,
)
from alpha_engine.tools.schemas import MarketSummary, TechnicalIndicators


class AnalyticsTools(Protocol):
    def get_ml_prediction(self) -> PredictionResult: ...

    def get_market_summary(self, timestamp: datetime) -> MarketSummary: ...

    def get_technical_indicators(self, timestamp: datetime) -> TechnicalIndicators: ...

    def get_risk_analysis(
        self,
        prediction: PredictionResult,
        market: MarketSummary,
        indicators: TechnicalIndicators,
    ) -> RiskResult: ...


class FutureAnalyticsTools(Protocol):
    """Extensión prevista; sentimiento y backtest aún no están implementados."""

    def get_sentiment(self) -> dict: ...

    def get_backtest_summary(self) -> dict: ...


class RepositoryAnalyticsTools:
    def __init__(
        self,
        market_repository: MarketRepository,
        prediction_service: PredictionService,
        risk_service: RiskService,
        symbol: str,
        interval: str = "1m",
        lookback_candles: int = 120,
    ):
        if interval != "1m" or lookback_candles < 35:
            raise ValueError("Las herramientas iniciales requieren 1m y al menos 35 velas")
        self._market = market_repository
        self._prediction = prediction_service
        self._risk = risk_service
        self._symbol = symbol
        self._interval = interval
        self._lookback = lookback_candles
        self._cached_timestamp: datetime | None = None
        self._cached_candles: list[Candle] | None = None

    def get_ml_prediction(self) -> PredictionResult:
        return self._prediction.predict_latest(self._symbol, self._interval)

    def get_snapshot_for_prediction(self, prediction: PredictionResult):
        """Build context from the persisted source candle, never from later data."""
        if prediction.symbol != self._symbol or prediction.interval != self._interval:
            raise AnalyticsDataError("La prediccion no corresponde a la serie configurada")
        market = self.get_market_summary(prediction.timestamp)
        indicators = self.get_technical_indicators(prediction.timestamp)
        risk = self.get_risk_analysis(prediction, market, indicators)
        from alpha_engine.tools.schemas import RecommendationSnapshot
        return RecommendationSnapshot(
            market=market, indicators=indicators, prediction=prediction, risk=risk
        )

    def get_market_summary(self, timestamp: datetime) -> MarketSummary:
        return get_market_summary(self._candles_at(timestamp))

    def get_technical_indicators(self, timestamp: datetime) -> TechnicalIndicators:
        return get_technical_indicators(self._candles_at(timestamp))

    def get_risk_analysis(
        self,
        prediction: PredictionResult,
        market: MarketSummary,
        indicators: TechnicalIndicators,
    ) -> RiskResult:
        return self._risk.analyze(
            prediction=prediction,
            price=market.price,
            volatility=indicators.rolling_volatility,
            rsi=indicators.rsi_14,
            macd=indicators.macd,
        )

    def _candles_at(self, timestamp: datetime) -> list[Candle]:
        if timestamp.tzinfo is None:
            raise AnalyticsDataError("timestamp debe incluir zona horaria")
        timestamp = timestamp.astimezone(timezone.utc)
        if self._cached_timestamp == timestamp and self._cached_candles is not None:
            return self._cached_candles
        stop = timestamp + timedelta(minutes=1)
        start = stop - timedelta(minutes=self._lookback)
        candles = self._market.get_candles(self._symbol, self._interval, start, stop)
        if not candles or candles[-1].open_time != timestamp:
            raise AnalyticsDataError("Faltan velas para el timestamp de la predicción")
        self._cached_timestamp = timestamp
        self._cached_candles = candles
        return candles
