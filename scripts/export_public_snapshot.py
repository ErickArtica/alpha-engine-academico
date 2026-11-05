"""Exporta un corte público de sólo lectura, sin claves ni contexto LLM."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpha_engine.config.settings import PROJECT_ROOT, get_settings
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.repositories.factory import (
    create_market_repository,
    create_prediction_repository,
    create_recommendation_repository,
)
from alpha_engine.services.history_audit import history_summary, load_history


DESTINATION = PROJECT_ROOT / "public_demo" / "data" / "snapshot.json"
HISTORY_FIELDS = (
    "symbol", "candle_open_at", "generated_at", "expires_at", "validity",
    "candidate_signal", "llm_signal", "final_signal", "risk_level",
    "guardrail_reasons", "predicted_class", "observed_move", "outcome_status",
    "source_close", "target_close", "signal_direction_correct", "counted_for_metric",
)


def read_report(name: str) -> dict:
    return json.loads((PROJECT_ROOT / "models" / name).read_text(encoding="utf-8"))


def build_public_snapshot(
    *, candles: list[Candle], prediction: PredictionRecord | None,
    history: list[dict], reports: dict[str, dict], exported_at: datetime,
) -> dict:
    """Selecciona campos públicos y rechaza artefactos de series incongruentes."""
    if exported_at.tzinfo is None:
        raise ValueError("exported_at debe incluir zona horaria")
    closed = sorted((c for c in candles if c.close_time < exported_at), key=lambda c: c.open_time)
    if not closed:
        raise ValueError("No hay velas cerradas para exportar")
    symbol, interval = closed[-1].symbol, closed[-1].interval
    if any(c.symbol != symbol or c.interval != interval for c in closed):
        raise ValueError("El corte mezcla símbolos o intervalos")
    model = reports["model"]
    baseline = reports["baseline"]
    walk = reports["walk"]
    economics = reports["economics"]
    prescription = reports["prescription"]
    latency = reports["latency"]
    if any(report["symbol"] != symbol for report in (model, walk, economics)):
        raise ValueError("Los informes pertenecen a otro símbolo")
    if model["trained_at"] != prescription["trained_at"]:
        raise ValueError("El chequeo de reglas pertenece a otro modelo")
    if walk["input_sha256"] != economics["source_input_sha256"]:
        raise ValueError("El chequeo económico usa otras velas")
    latest = closed[-1]
    symbol_history = [row for row in history if row.get("symbol") == symbol]
    public_history = [
        {field: row.get(field) for field in HISTORY_FIELDS}
        for row in symbol_history
    ]
    latest_prediction = None
    if prediction and prediction.symbol == symbol and prediction.timestamp == latest.open_time:
        latest_prediction = {
            "candle_open_at": prediction.timestamp.isoformat(),
            "predicted_class": prediction.predicted_class,
            "prob_up": prediction.prob_up,
            "prob_down": prediction.prob_down,
            "model_version": prediction.model_version,
        }
    base_cost_zero_latency = next(
        run for run in economics["runs"]
        if run["cost_scenario"] == "base_estimate" and run["latency_seconds_assumed"] == 0
    )
    return {
        "snapshot_version": 1,
        "exported_at": exported_at.astimezone(timezone.utc).isoformat(),
        "symbol": symbol,
        "interval": interval,
        "last_candle_close_at": latest.close_time.isoformat(),
        "candles": [
            {"open_time": c.open_time.isoformat(), "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in closed[-120:]
        ],
        "latest_prediction": latest_prediction,
        "model": {
            "trained_at": model["trained_at"],
            "target_definition": model["target_definition"],
            "data_range": model["data_range"],
            "test": model["evaluation"]["test"],
            "baseline_test": baseline["test"],
            "probabilities_calibrated": False,
        },
        "walk_forward": {
            "generated_at": walk["generated_at"],
            "window_count": len(walk["windows"]),
            "summary": walk["summary"],
            "limits": walk["limits"],
        },
        "prescriptive": {
            "test_rows": prescription["test_rows"],
            "eligible_rows": prescription["eligible_rows"],
            "eligible_fraction": prescription["eligible_fraction"],
            "direction_accuracy_eligible": prescription["direction_accuracy_eligible"],
            "limits": prescription["limits"],
        },
        "economics": {
            "observed_data": economics["observed_data"],
            "assumed_costs": economics["assumptions"]["costs"],
            "optimistic_zero_latency": {
                key: value for key, value in base_cost_zero_latency["total"].items()
                if key != "trades"
            },
            "limits": economics["limits"],
        },
        "latency": {
            "generated_at": latency["generated_at"],
            "prediction_cycles": latency["prediction_cycles"],
            "recommendations_with_prediction": latency["recommendations_with_prediction"],
            "recommendations_before_expiry": latency["recommendations_before_expiry"],
            "actionable_recommendations_before_expiry": latency["actionable_recommendations_before_expiry"],
            "measurements": latency["measurements"],
            "limits": latency["limits"],
        },
        "history_summary": history_summary(symbol_history),
        "history": public_history,
    }


def main() -> None:
    settings = get_settings()
    if settings.interval != "1m":
        raise ValueError("El corte público actual sólo admite velas de 1 minuto")
    market = create_market_repository(settings)
    predictions = create_prediction_repository(settings)
    recommendations = create_recommendation_repository(settings)
    try:
        latest = market.get_latest_candle(settings.symbol, settings.interval)
        if latest is None:
            raise ValueError("No hay velas locales para exportar")
        candles = market.get_candles(
            settings.symbol, settings.interval,
            latest.open_time - timedelta(minutes=119), latest.open_time + timedelta(minutes=1),
        )
        exported_at = datetime.now(timezone.utc)
        rows = load_history(recommendations, predictions, market, now=exported_at)
        report_names = {
            "model": "xgboost_v1_metadata.json", "baseline": "baseline_v1.json",
            "walk": "walk_forward_v1.json", "economics": "economic_check_v1.json",
            "prescription": "prescriptive_check_v1.json", "latency": "latency_report_v1.json",
        }
        reports = {key: read_report(name) for key, name in report_names.items()}
        snapshot = build_public_snapshot(
            candles=candles, prediction=predictions.get_latest(settings.symbol),
            history=rows, reports=reports, exported_at=exported_at,
        )
    finally:
        recommendations.close()
        predictions.close()
        market.close()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DESTINATION)
    print(f"Corte público: {DESTINATION} ({len(snapshot['candles'])} velas, {len(snapshot['history'])} decisiones)")


if __name__ == "__main__":
    main()
