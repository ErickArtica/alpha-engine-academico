"""Resumen de latencias realmente registradas; no estima tiempos ausentes."""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import median

import numpy as np

from alpha_engine.config.settings import PROJECT_ROOT
from alpha_engine.ml.trainer import _atomic_json
from alpha_engine.observability.telemetry import DEFAULT_DIR, TelemetryStore, utc_now

REPORT_VERSION = "latency_report_v1"


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("La telemetría requiere fechas con zona horaria")
    return parsed.astimezone(timezone.utc)


def _stats(values: list[float], unit: str) -> dict:
    if not values:
        return {"samples": 0, "unit": unit, "median": None, "p95": None, "min": None, "max": None}
    return {
        "samples": len(values),
        "unit": unit,
        "median": float(median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_latency_report(events: list[dict], quotes: list[dict]) -> dict:
    traces: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        traces[event["trace_id"]].append(event)
    durations: dict[str, list[float]] = defaultdict(list)
    cycle_count = timely_predictions = recommendation_count = timely_recommendations = actionable_count = 0
    quote_after_prediction = quote_after_ready = 0
    quotes_by_trace: dict[str, list[dict]] = defaultdict(list)
    for quote in quotes:
        quotes_by_trace[quote["trace_id"]].append(quote)
        mid = (float(quote["bid_price"]) + float(quote["ask_price"])) / 2
        durations["observed_spread_bps"].append((float(quote["ask_price"]) - float(quote["bid_price"])) / mid * 10_000)
        durations["quote_request_ms"].append((_time(quote["received_at"]) - _time(quote["requested_at"])).total_seconds() * 1000)

    for trace_id, rows in traces.items():
        by_stage: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_stage[row["stage"]].append(row)
            if row["stage"] == "llm_finished" and row.get("duration_ms") is not None:
                durations["llm_call_ms"].append(float(row["duration_ms"]))
        prediction = by_stage["prediction_ready"][-1] if by_stage["prediction_ready"] else None
        data_ready = by_stage["data_ready"][-1] if by_stage["data_ready"] else None
        if prediction:
            cycle_count += 1
            close = _time(prediction["candle_open_time"]) + timedelta(minutes=1) - timedelta(milliseconds=1)
            prediction_at = _time(prediction["at"])
            expiry = close + timedelta(minutes=1)
            durations["prediction_remaining_lifetime_ms"].append((expiry - prediction_at).total_seconds() * 1000)
            if prediction_at < expiry:
                timely_predictions += 1
            for quote in quotes_by_trace[trace_id]:
                quote_at = _time(quote["received_at"])
                if prediction_at <= quote_at < expiry:
                    quote_after_prediction += 1
            if prediction.get("duration_ms") is not None:
                durations["prediction_service_ms"].append(float(prediction["duration_ms"]))
            if data_ready and data_ready.get("details", {}).get("candles_sent", 0) > 0:
                durations["candle_close_to_data_ready_ms"].append((_time(data_ready["at"]) - close).total_seconds() * 1000)
        context = by_stage["context_ready"][-1] if by_stage["context_ready"] else None
        if context and context.get("duration_ms") is not None:
            durations["recommendation_context_ms"].append(float(context["duration_ms"]))
        ready = by_stage["recommendation_ready"][-1] if by_stage["recommendation_ready"] else None
        if not ready or not ready.get("candle_open_time"):
            continue
        recommendation_count += 1
        ready_at = _time(ready["at"])
        close = _time(ready["candle_open_time"]) + timedelta(minutes=1) - timedelta(milliseconds=1)
        durations["candle_close_to_recommendation_ready_ms"].append((ready_at - close).total_seconds() * 1000)
        if ready.get("duration_ms") is not None:
            durations["recommendation_service_ms"].append(float(ready["duration_ms"]))
        expiry_raw = ready.get("details", {}).get("expires_at")
        if expiry_raw:
            expires_at = _time(expiry_raw)
            if ready_at < expires_at:
                timely_recommendations += 1
                if ready.get("details", {}).get("final_signal") in ("BUY", "SELL"):
                    actionable_count += 1
            for quote in quotes_by_trace[trace_id]:
                quote_at = _time(quote["received_at"])
                if ready_at <= quote_at < expires_at:
                    quote_after_ready += 1
        rendered = by_stage["server_rendered"][-1] if by_stage["server_rendered"] else None
        if rendered:
            durations["server_render_after_ready_ms"].append((_time(rendered["at"]) - ready_at).total_seconds() * 1000)

    return {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now().isoformat(),
        "observed_event_rows": len(events),
        "observed_quote_rows": len(quotes),
        "prediction_cycles": cycle_count,
        "predictions_ready_before_expiry": timely_predictions,
        "predictions_ready_before_expiry_fraction": timely_predictions / cycle_count if cycle_count else None,
        "quotes_received_after_prediction_before_expiry": quote_after_prediction,
        "recommendations_with_prediction": recommendation_count,
        "recommendations_before_expiry": timely_recommendations,
        "recommendations_before_expiry_fraction": timely_recommendations / recommendation_count if recommendation_count else None,
        "actionable_recommendations_before_expiry": actionable_count,
        "quotes_received_after_ready_before_expiry": quote_after_ready,
        "measurements": {name: _stats(durations[name], "bps" if name == "observed_spread_bps" else "ms") for name in (
            "candle_close_to_data_ready_ms", "prediction_service_ms", "prediction_remaining_lifetime_ms",
            "recommendation_context_ms", "llm_call_ms", "recommendation_service_ms",
            "candle_close_to_recommendation_ready_ms", "server_render_after_ready_ms",
            "quote_request_ms", "observed_spread_bps",
        )},
        "limits": "Tiempos UTC y duraciones medidas sólo para eventos registrados. server_rendered marca la llamada Streamlit del servidor, no el pintado del navegador. bookTicker REST carece de hora de evento de mercado; received_at es hora local de recepción y no confirma fill.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume telemetría local de latencia y cotizaciones")
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "latency_report_v1.json")
    args = parser.parse_args()
    store = TelemetryStore(args.telemetry_dir)
    report = build_latency_report(store.read_events(), store.read_quotes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)
    print(json.dumps({
        "report": str(args.output),
        "prediction_cycles": report["prediction_cycles"],
        "recommendations_with_prediction": report["recommendations_with_prediction"],
        "quotes": report["observed_quote_rows"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
