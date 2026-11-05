"""Diagnóstico de cobertura y dirección de reglas prescriptivas sobre TEST."""

from collections import Counter
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionResult

RULE_SETTING_KEYS = (
    "model_confidence_threshold",
    "risk_volatility_medium", "risk_volatility_high",
    "risk_rsi_medium_low", "risk_rsi_medium_high",
    "risk_rsi_high_low", "risk_rsi_high_high",
    "risk_min_stop_pct", "risk_stop_vol_multiplier",
    "risk_stop_cap_pct", "risk_reward_ratio",
)


def rule_settings(settings: Settings) -> dict[str, float]:
    """Parámetros no secretos que identifican una evaluación de reglas."""
    return {name: float(getattr(settings, name)) for name in RULE_SETTING_KEYS}


def evaluate_prescriptive_rules(
    candles: Sequence[Candle], bundle: dict, metadata: dict, settings: Settings,
) -> dict:
    """Usa el modelo ya entrenado y exactamente el intervalo TEST original.

    Cada decisión usa features de t; target compara close(t+1) con close(t).
    No reproduce la respuesta del LLM ni modela órdenes o costes.
    """
    if bundle["trained_at"] != metadata["trained_at"] or bundle["features"] != metadata["features"]:
        raise ValueError("El modelo y metadata no corresponden al mismo entrenamiento")
    if bundle["features"] != list(FEATURE_COLUMNS):
        raise ValueError("Las features del artefacto no coinciden con el código actual")
    frame = build_feature_frame(candles)
    labeled = build_training_frame(frame)
    test_info = metadata["split"]["test"]
    start = datetime.fromisoformat(test_info["start"])
    end = datetime.fromisoformat(test_info["end"])
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("El rango TEST debe incluir zona horaria")
    test = labeled.loc[start:end]
    if len(test) != test_info["rows"] or test.empty:
        raise ValueError("El bloque TEST local no coincide con el entrenamiento")
    probabilities = np.asarray(bundle["estimator"].predict_proba(test[list(FEATURE_COLUMNS)]))
    classes = list(bundle["estimator"].classes_)
    if probabilities.shape != (len(test), 2) or set(classes) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Probabilidades del modelo inválidas")
    prob_up = probabilities[:, classes.index(1)]
    prob_down = probabilities[:, classes.index(0)]
    predicted_up = prob_up > prob_down
    actual = test["target"].to_numpy(dtype=int)
    raw_accuracy = float(np.mean(predicted_up.astype(int) == actual))
    if abs(raw_accuracy - metadata["evaluation"]["test"]["accuracy"]) > 1e-12:
        raise ValueError("La evaluación TEST cambió; revise el modelo o los datos históricos")

    risk_service = RiskService(settings)
    candidate_counts: Counter[str] = Counter()
    filtered_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    eligible_correct = 0
    eligible_total = 0
    for position, (timestamp, row) in enumerate(test.iterrows()):
        up = float(prob_up[position])
        down = float(prob_down[position])
        direction = "UP" if predicted_up[position] else "DOWN"
        prediction = PredictionResult(
            prediction_id=f"test-{position}", symbol=str(row["symbol"]),
            timestamp=timestamp.to_pydatetime(), horizon="1m",
            prob_up=up, prob_down=down, predicted_class=direction,
            model_version=bundle["model_version"],
        )
        risk = risk_service.analyze(
            prediction=prediction, price=float(row["close"]),
            volatility=float(row["rolling_volatility"]),
            rsi=float(row["rsi_14"]), macd=float(row["macd"]),
        )
        candidate = "HOLD" if max(up, down) < settings.model_confidence_threshold else "BUY" if direction == "UP" else "SELL"
        filtered = "HOLD" if risk.risk_level == "HIGH" else candidate
        candidate_counts[candidate] += 1
        filtered_counts[filtered] += 1
        risk_counts[risk.risk_level] += 1
        trigger_counts.update(risk.triggers)
        if filtered != "HOLD":
            eligible_total += 1
            eligible_correct += int(int(predicted_up[position]) == actual[position])

    return {
        "check_version": "prescriptive_rules_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": bundle["model_version"],
        "trained_at": bundle["trained_at"],
        "test_start": test_info["start"],
        "test_end": test_info["end"],
        "test_rows": len(test),
        "target_definition": metadata["target_definition"],
        "model_test_accuracy": raw_accuracy,
        "confidence_threshold": settings.model_confidence_threshold,
        "rule_parameters": rule_settings(settings),
        "candidate_counts": {key: candidate_counts[key] for key in ("BUY", "SELL", "HOLD")},
        "filtered_counts": {key: filtered_counts[key] for key in ("BUY", "SELL", "HOLD")},
        "risk_level_counts": {key: risk_counts[key] for key in ("LOW", "MEDIUM", "HIGH")},
        "risk_trigger_counts": dict(sorted(trigger_counts.items())),
        "eligible_rows": eligible_total,
        "eligible_fraction": eligible_total / len(test),
        "direction_accuracy_eligible": eligible_correct / eligible_total if eligible_total else None,
        "limits": "Sólo reglas deterministas sobre TEST; sin LLM, costes, órdenes, stop ejecutado ni rentabilidad.",
    }
