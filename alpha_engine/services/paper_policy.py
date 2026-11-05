"""Deterministic, position-aware suggestions for virtual spot only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

Action = Literal["BUY", "SELL", "HOLD", "WAIT"]


@dataclass(frozen=True)
class PaperDecision:
    action: Action
    executable: bool
    reason: str
    signal_probability: float | None
    confidence: float | None


def paper_decision(prediction, *, has_long: bool, threshold: float = 0.60,
                   now: datetime | None = None) -> PaperDecision:
    """The LLM and news never participate in this rule.

    An absent/expired prediction abstains. P(UP) controls entry; P(DOWN)
    controls exit. A high DOWN probability while flat remains WAIT.
    """
    if not 0.5 < threshold < 1:
        raise ValueError("paper threshold must be in (0.5, 1)")
    fallback: Action = "HOLD" if has_long else "WAIT"
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("now must include a timezone")
    if prediction is None:
        return PaperDecision(fallback, False, "No hay predicción vigente.", None, None)
    valid_until = getattr(prediction, "valid_until", None)
    generated_at = getattr(prediction, "generated_at", None)
    if (valid_until is None or generated_at is None or valid_until.tzinfo is None
            or generated_at.tzinfo is None or instant >= valid_until
            or instant < generated_at):
        return PaperDecision(fallback, False, "La predicción no está vigente.", None, None)
    up, down = float(prediction.prob_up), float(prediction.prob_down)
    if not (0 <= up <= 1 and 0 <= down <= 1 and abs(up + down - 1) <= 1e-6):
        raise ValueError("Invalid prediction probabilities")
    confidence = max(up, down)
    if has_long:
        if down >= threshold:
            return PaperDecision("SELL", True, "P(DOWN) supera el gate; considerar salida virtual.", down, confidence)
        return PaperDecision("HOLD", False, "Sin señal de salida; mantener y reevaluar.", down, confidence)
    if up >= threshold:
        return PaperDecision("BUY", True, "P(UP) supera el gate; considerar compra virtual.", up, confidence)
    return PaperDecision("WAIT", False, "Sin señal de entrada; esperar y no entrar.", up, confidence)
