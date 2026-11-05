"""Reglas deterministas que limitan la autoridad del LLM."""

from datetime import datetime, timezone
from typing import Literal

from alpha_engine.ai.schemas import LLMRecommendation
from alpha_engine.config.settings import Settings
from alpha_engine.data.timing import prediction_expires_at
from alpha_engine.tools.schemas import RecommendationSnapshot

Signal = Literal["BUY", "SELL", "HOLD"]


def candidate_signal(snapshot: RecommendationSnapshot, settings: Settings) -> Signal:
    prediction = snapshot.prediction
    if max(prediction.prob_up, prediction.prob_down) < settings.model_confidence_threshold:
        return "HOLD"
    return "BUY" if prediction.predicted_class == "UP" else "SELL"


def pre_llm_guardrails(
    snapshot: RecommendationSnapshot, settings: Settings, now: datetime
) -> list[str]:
    if now.tzinfo is None:
        raise ValueError("now debe incluir zona horaria")
    reasons = []
    age_seconds = (now.astimezone(timezone.utc) - snapshot.market.close_time).total_seconds()
    if age_seconds < 0:
        reasons.append("CANDLE_NOT_CLOSED")
    else:
        if now.astimezone(timezone.utc) >= prediction_expires_at(
            snapshot.market.close_time, snapshot.prediction.horizon
        ):
            reasons.append("PREDICTION_EXPIRED")
        if age_seconds > settings.max_candle_age_seconds:
            reasons.append("STALE_MARKET_DATA")
    if max(snapshot.prediction.prob_up, snapshot.prediction.prob_down) < settings.model_confidence_threshold:
        reasons.append("LOW_MODEL_CONFIDENCE")
    if snapshot.risk.risk_level == "HIGH":
        reasons.append("HIGH_RISK")
    return reasons


def resolve_final_signal(
    candidate: Signal,
    llm: LLMRecommendation | None,
    pre_reasons: list[str],
) -> tuple[Signal, list[str]]:
    reasons = list(pre_reasons)
    if reasons:
        return "HOLD", reasons
    if llm is None:
        return candidate, ["LLM_UNAVAILABLE_EXPLANATION"]
    # The provider's legacy `signal` field is an echo for audit only. It has
    # no authority to change the deterministic candidate in either direction.
    return candidate, reasons
