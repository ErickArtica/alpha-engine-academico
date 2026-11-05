"""Une recomendaciones persistidas con velas cerradas, sin anticipar resultados."""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from alpha_engine.data.timing import prediction_expires_at
from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.repositories.protocols import MarketRepository, PredictionRepository, RecommendationRepository


def _guardrail_reasons(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return ["GUARDRAILS_JSON_INVALIDO"]
    return parsed if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed) else ["GUARDRAILS_JSON_INVALIDO"]


def audit_history(
    recommendations: Sequence[RecommendationRecord],
    predictions: Mapping[str, PredictionRecord],
    candles: Sequence[Candle],
    now: datetime,
) -> list[dict]:
    """Devuelve filas auditables; sólo el primer BUY/SELL visible oportuno cuenta por vela.

    La exactitud de señal exige un movimiento estricto; un empate no acierta
    ni BUY ni SELL. El target binario del modelo sigue contando empate como DOWN.
    """
    if now.tzinfo is None:
        raise ValueError("now debe incluir zona horaria")

    candle_by_key = {(c.symbol, c.open_time): c for c in candles if c.interval == "1m"}
    rows: list[dict] = []
    groups: dict[tuple[str, datetime], list[dict]] = defaultdict(list)

    for item in recommendations:
        prediction = predictions.get(item.prediction_id)
        matching_prediction = prediction is not None and prediction.timestamp == item.timestamp
        symbol = prediction.symbol if matching_prediction else None
        source = candle_by_key.get((symbol, item.timestamp)) if symbol else None
        target = candle_by_key.get((symbol, item.timestamp + timedelta(minutes=1))) if source else None
        expiry = prediction_expires_at(source.close_time, prediction.horizon) if source and matching_prediction else None
        emitted_on_time = (
            source.close_time <= item.generated_at < expiry if source and expiry else None
        )
        latency_seconds = (item.generated_at - source.close_time).total_seconds() if source else None

        if expiry is None:
            validity = "sin datos para verificar"
        elif item.generated_at < source.close_time:
            validity = "antes del cierre"
        elif not emitted_on_time:
            validity = "emitida tarde"
        elif item.final_signal == "HOLD":
            validity = "HOLD"
        elif now < expiry:
            validity = "vigente"
        else:
            validity = "vencida ahora"

        if target and target.close_time < now:
            outcome_status = "observado"
            target_close = target.close
            observed_move = "UP" if target.close > source.close else "DOWN" if target.close < source.close else "FLAT"
            target_class = "UP" if target.close > source.close else "DOWN"
            model_target_correct = prediction.predicted_class == target_class
            signal_direction_correct = (
                item.final_signal == "BUY" and observed_move == "UP"
                or item.final_signal == "SELL" and observed_move == "DOWN"
            ) if emitted_on_time and item.final_signal in ("BUY", "SELL") else None
        else:
            # La ausencia de la vela tras su cierre previsto es distinta de una vela abierta.
            expected_close = expiry or item.timestamp + timedelta(minutes=2) - timedelta(milliseconds=1)
            outcome_status = "pendiente" if now <= expected_close else "sin dato"
            target_close = observed_move = target_class = model_target_correct = signal_direction_correct = None

        row = {
            "symbol": symbol,
            "candle_open_at": item.timestamp.isoformat(),
            "source_close_at": source.close_time.isoformat() if source else None,
            "source_close": source.close if source else None,
            "target_close": target_close,
            "target_close_at": target.close_time.isoformat() if target and outcome_status == "observado" else None,
            "outcome_status": outcome_status,
            "observed_move": observed_move,
            "target_class": target_class,
            "predicted_class": prediction.predicted_class if matching_prediction else None,
            "model_target_correct": model_target_correct,
            "signal_direction_correct": signal_direction_correct,
            "prediction_id": item.prediction_id,
            "model_version": prediction.model_version if matching_prediction else None,
            "prompt_version": item.prompt_version,
            "llm_model": item.llm_model,
            "generated_at": item.generated_at.isoformat(),
            "expires_at": expiry.isoformat() if expiry else None,
            "latency_seconds": latency_seconds,
            "emitted_on_time": emitted_on_time,
            "validity": validity,
            "candidate_signal": item.candidate_signal,
            "llm_signal": item.llm_signal,
            "final_signal": item.final_signal,
            "risk_level": item.risk_level,
            "guardrail_reasons": _guardrail_reasons(item.guardrails_json),
            "versions_in_candle": 1,
            "counted_for_metric": False,
        }
        rows.append(row)
        if symbol:
            groups[(symbol, item.timestamp)].append(row)

    for group in groups.values():
        for row in group:
            row["versions_in_candle"] = len(group)
        timely_signals = [
            row for row in group
            if row["emitted_on_time"] and row["final_signal"] in ("BUY", "SELL")
        ]
        if timely_signals:
            earliest = min(
                timely_signals,
                key=lambda row: (row["generated_at"], row["prediction_id"], row["prompt_version"], row["llm_model"]),
            )
            earliest["counted_for_metric"] = True

    return sorted(rows, key=lambda row: (row["candle_open_at"], row["generated_at"]), reverse=True)


def history_summary(rows: Sequence[dict]) -> dict:
    """Métricas únicas sobre señales oportunas con resultado realmente observado."""
    selected = [row for row in rows if row["counted_for_metric"]]
    observed = [row for row in selected if row["outcome_status"] == "observado"]
    correct = sum(row["signal_direction_correct"] is True for row in observed)
    return {
        "saved_recommendations": len(rows),
        "unique_timely_signals": len(selected),
        "observed_signals": len(observed),
        "directional_hits": correct,
        "directional_hit_rate": correct / len(observed) if observed else None,
    }


def load_history(
    recommendations: RecommendationRepository,
    predictions: PredictionRepository,
    market: MarketRepository,
    *,
    limit: int = 100,
    now: datetime,
) -> list[dict]:
    """Carga las versiones visibles y las velas necesarias para auditarlas."""
    records = recommendations.list_recent(limit)
    prediction_by_id = {
        prediction_id: predictions.get_by_id(prediction_id)
        for prediction_id in {record.prediction_id for record in records}
    }
    candles = []
    by_symbol: dict[str, list[datetime]] = defaultdict(list)
    for record in records:
        prediction = prediction_by_id[record.prediction_id]
        if prediction and prediction.timestamp == record.timestamp:
            by_symbol[prediction.symbol].append(record.timestamp)
    for symbol, times in by_symbol.items():
        candles.extend(market.get_candles(
            symbol, "1m", min(times), max(times) + timedelta(minutes=2)
        ))
    return audit_history(records, prediction_by_id, candles, now)
