"""Orquestación determinista de herramientas; sin tool calling del LLM."""

from alpha_engine.ai.context_builder import build_context, context_json
from alpha_engine.tools.analytics import AnalyticsTools
from alpha_engine.tools.schemas import RecommendationSnapshot
from alpha_engine.services.prediction import PredictionResult


class RecommendationContextService:
    def __init__(self, tools: AnalyticsTools):
        self._tools = tools

    def build_snapshot(self) -> RecommendationSnapshot:
        prediction = self._tools.get_ml_prediction()
        timestamp = prediction.timestamp
        market = self._tools.get_market_summary(timestamp)
        indicators = self._tools.get_technical_indicators(timestamp)
        risk = self._tools.get_risk_analysis(prediction, market, indicators)
        return RecommendationSnapshot(
            market=market, indicators=indicators, prediction=prediction, risk=risk
        )

    def build_snapshot_for_prediction(self, prediction: PredictionResult) -> RecommendationSnapshot:
        """Explicit-prediction variant for the on-demand prescriptive use case."""
        method = getattr(self._tools, "get_snapshot_for_prediction", None)
        if method is not None:
            return method(prediction)
        market = self._tools.get_market_summary(prediction.timestamp)
        indicators = self._tools.get_technical_indicators(prediction.timestamp)
        risk = self._tools.get_risk_analysis(prediction, market, indicators)
        return RecommendationSnapshot(
            market=market, indicators=indicators, prediction=prediction, risk=risk
        )

    def build_context(self) -> dict:
        return build_context(self.build_snapshot())

    def build_context_json(self) -> str:
        return context_json(self.build_context())
