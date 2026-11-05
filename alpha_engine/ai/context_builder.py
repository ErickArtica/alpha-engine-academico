"""Contexto JSON mínimo, determinista y auditable para el LLM."""

import json
from datetime import datetime, timezone

from alpha_engine.config.settings import Settings
from alpha_engine.tools.schemas import RecommendationSnapshot

CONTEXT_VERSION = "context_paper_explain_v1"


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_context(snapshot: RecommendationSnapshot, settings: Settings | None = None) -> dict:
    market = snapshot.market
    indicators = snapshot.indicators
    prediction = snapshot.prediction
    risk = snapshot.risk
    context = {
        "system": {
            "project": "Alpha Engine",
            "purpose": "Academic scalping decision support",
            "real_trading": False,
            "context_version": CONTEXT_VERSION,
        },
        "market": {
            "symbol": market.symbol,
            "timestamp": _utc_iso(market.timestamp),
            "close_time": _utc_iso(market.close_time),
            "price": market.price,
            "change_1m_pct": market.change_1m_pct,
            "volume": market.volume,
        },
        "indicators": {
            "return_1m": indicators.return_1m,
            "rsi_14": indicators.rsi_14,
            "ema_20": indicators.ema_20,
            "macd": indicators.macd,
            "macd_signal": indicators.macd_signal,
            "macd_histogram": indicators.macd_histogram,
            "rolling_volatility": indicators.rolling_volatility,
            "volume_change": indicators.volume_change,
        },
        "prediction": {
            "prediction_id": prediction.prediction_id,
            "horizon": prediction.horizon,
            "prob_up": prediction.prob_up,
            "prob_down": prediction.prob_down,
            "predicted_class": prediction.predicted_class,
            "model_version": prediction.model_version,
        },
        "risk": {
            "level": risk.risk_level,
            "reference_price": risk.reference_price,
            "stop_loss": risk.stop_loss,
            "take_profit": risk.take_profit,
            "triggers": risk.triggers,
            "rule_version": risk.rule_version,
        },
    }
    if settings is not None:
        from alpha_engine.ai.guardrails import candidate_signal
        from alpha_engine.services.paper_policy import paper_decision

        action = paper_decision(prediction, has_long=False,
                                threshold=settings.model_confidence_threshold,
                                now=prediction.generated_at or market.close_time).action
        context["decision_policy"] = {
            "candidate_signal": candidate_signal(snapshot, settings),
            "action": action,
            "position": "SIN_POSICION",
            "action_authority": "deterministic_rules_only",
            "minimum_model_probability": settings.model_confidence_threshold,
            "probability_gate_role": "experimental_prescriptive_guardrail",
            "probability_gate_quality_validated": False,
            "high_risk_requires_hold": True,
            "expired_prediction_requires_hold": True,
            "probabilities_calibrated": False,
        }
    return context


def context_json(context: dict) -> str:
    return json.dumps(
        context, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
