"""Read-only, reproducible diagnostic of persisted XGBoost LIVE evaluations.

The report contains aggregates only. A Supabase publishable/read key can come
from SUPABASE_KEY or the already authenticated local Supabase CLI. No key or
individual market row is written to the report.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx

from alpha_engine.data.live_schemas import LiveEvaluationRecord
from alpha_engine.data.schemas import PredictionRecord

PAGE_SIZE = 1000
THRESHOLDS = (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65)
BUCKETS = (
    (0.50, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 0.58),
    (0.58, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, None),
)
EVALUATION_SELECT = (
    "evaluation_id,prediction_id,symbol,evaluated_at,source_close,"
    "target_open_time,target_close_time,target_close,observed_move,observed_class,correct,"
    "predictions(prediction_id,symbol,interval,candle_open_time,horizon,prob_up,"
    "prob_down,predicted_class,model_version,trained_at,generated_at,valid_until)"
)


@dataclass(frozen=True)
class Observation:
    prediction: PredictionRecord
    evaluation: LiveEvaluationRecord

    @property
    def confidence(self) -> float:
        return max(self.prediction.prob_up, self.prediction.prob_down)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Database timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _publishable_key(project_ref: str) -> str:
    command = shutil.which("pnpm")
    if command is None:
        raise RuntimeError("SUPABASE_KEY is unset and pnpm is unavailable")
    env = dict(os.environ)
    env["SUPABASE_TELEMETRY_DISABLED"] = "1"
    try:
        process = subprocess.run(
            [command, "supabase", "projects", "api-keys", "--project-ref", project_ref,
             "--output", "json"],
            capture_output=True, text=True, timeout=45, check=False, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Supabase CLI key lookup timed out") from exc
    if process.returncode:
        raise RuntimeError("Supabase CLI key lookup failed; authenticate locally")
    entries = json.loads(process.stdout)
    keys = [item["api_key"] for item in entries if item.get("type") == "publishable"]
    if len(keys) != 1:
        raise RuntimeError("Expected one publishable key; refusing a privileged key")
    return keys[0]


def _require_public_read_key(key: str) -> None:
    """Refuse privileged keys even though all queries below use GET."""
    if key.startswith("sb_publishable_"):
        return
    if key.startswith("sb_secret_"):
        raise ValueError("A server/secret key is not allowed for this diagnostic")
    try:
        payload = key.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        role = json.loads(base64.urlsafe_b64decode(padded)).get("role")
    except (IndexError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Use a Supabase publishable or legacy anon key") from exc
    if role != "anon":
        raise ValueError("A server/secret key is not allowed for this diagnostic")


def _fetch_all(
    client: httpx.Client, url: str, key: str, table: str,
    params: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    expected: int | None = None
    while True:
        query = [*params, ("limit", str(PAGE_SIZE)), ("offset", str(len(rows)))]
        response = client.get(
            f"{url.rstrip('/')}/rest/v1/{table}",
            headers={"apikey": key, "Prefer": "count=exact"}, params=query,
        )
        # Do not put server response bodies in exceptions: they may contain
        # infrastructure details unrelated to the aggregate report.
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Read-only {table} query failed: HTTP {response.status_code}")
        content_range = response.headers.get("content-range", "")
        if "/" not in content_range or content_range.rsplit("/", 1)[1] == "*":
            raise RuntimeError(f"Exact count unavailable for {table}")
        count = int(content_range.rsplit("/", 1)[1])
        if expected is None:
            expected = count
        elif count != expected:
            raise RuntimeError(f"{table} changed within the frozen query window")
        page = response.json()
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected {table} response")
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete {table} pagination: {len(rows)} of {expected}")
    return rows, expected


def _observations(rows: list[dict[str, Any]], model_version: str) -> tuple[list[Observation], Counter[str]]:
    seen_evaluations: set[str] = set()
    seen_predictions: set[str] = set()
    models: Counter[str] = Counter()
    result: list[Observation] = []
    for row in rows:
        prediction_row = row.get("predictions")
        if not isinstance(prediction_row, dict):
            raise ValueError("Evaluation has no readable prediction")
        model = prediction_row.get("model_version")
        models[model] += 1
        prediction = PredictionRecord.model_validate({
            **prediction_row,
            "timestamp": _utc(prediction_row["candle_open_time"]),
        })
        evaluation = LiveEvaluationRecord.model_validate(row)
        if str(evaluation.evaluation_id) in seen_evaluations or prediction.prediction_id in seen_predictions:
            raise ValueError("Duplicate evaluation or prediction in the frozen cohort")
        seen_evaluations.add(str(evaluation.evaluation_id))
        seen_predictions.add(prediction.prediction_id)
        if evaluation.prediction_id != prediction.prediction_id or evaluation.symbol != prediction.symbol:
            raise ValueError("Evaluation/prediction identity mismatch")
        if evaluation.target_open_time != prediction.timestamp + timedelta(minutes=1):
            raise ValueError("Target candle does not follow the source candle")
        if evaluation.correct != (prediction.predicted_class == evaluation.observed_class):
            raise ValueError("Stored correctness disagrees with predicted/observed classes")
        if prediction.generated_at is None or prediction.valid_until is None:
            raise ValueError("LIVE evaluation references a legacy prediction without timing")
        if prediction.generated_at > evaluation.evaluated_at:
            raise ValueError("Evaluation predates prediction generation")
        if model == model_version:
            result.append(Observation(prediction, evaluation))
    result.sort(key=lambda item: (item.prediction.timestamp, item.prediction.prediction_id))
    return result, models


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _class_metrics(observations: list[Observation], label: str) -> dict[str, Any]:
    tp = sum(o.prediction.predicted_class == label and o.evaluation.observed_class == label for o in observations)
    predicted = sum(o.prediction.predicted_class == label for o in observations)
    actual = sum(o.evaluation.observed_class == label for o in observations)
    precision = _rate(tp, predicted)
    recall = _rate(tp, actual)
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"support": actual, "predicted": predicted, "correct": tp,
            "precision": precision, "recall": recall, "f1": f1}


def _summary(observations: list[Observation], total: int) -> dict[str, Any]:
    count = len(observations)
    correct = sum(item.evaluation.correct for item in observations)
    return {"count": count, "coverage": _rate(count, total), "correct": correct,
            "incorrect": count - correct, "accuracy": _rate(correct, count),
            "precision_up": _class_metrics(observations, "UP")["precision"],
            "precision_down": _class_metrics(observations, "DOWN")["precision"]}


def analyze_rows(
    evaluation_rows: list[dict[str, Any]], candle_rows: list[dict[str, Any]],
    *, cutoff: datetime, model_version: str = "xgboost_v1",
) -> dict[str, Any]:
    """Pure aggregation; raises on inconsistent persisted joins or labels."""
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must have a timezone")
    observations, models = _observations(evaluation_rows, model_version)
    if not observations:
        raise ValueError(f"No evaluated LIVE predictions for {model_version}")
    n = len(observations)
    matrix = {
        predicted: {actual: sum(
            o.prediction.predicted_class == predicted and o.evaluation.observed_class == actual
            for o in observations
        ) for actual in ("DOWN", "UP")}
        for predicted in ("DOWN", "UP")
    }
    per_class = {label: _class_metrics(observations, label) for label in ("DOWN", "UP")}
    weighted_f1 = sum((per_class[label]["f1"] or 0) * per_class[label]["support"]
                      for label in ("DOWN", "UP")) / n
    confidences = [o.confidence for o in observations]
    prob_up = [o.prediction.prob_up for o in observations]
    prob_down = [o.prediction.prob_down for o in observations]
    distribution = {}
    for name, values in (("confidence", confidences), ("prob_up", prob_up), ("prob_down", prob_down)):
        distribution[name] = {"min": min(values), "max": max(values), "mean": mean(values),
                              "median": median(values), **{
                                  f"p{int(q * 100)}": _percentile(values, q)
                                  for q in (0.50, 0.75, 0.90, 0.95, 0.99)
                              }}
    buckets = []
    for lower, upper in BUCKETS:
        selected = [o for o in observations if o.confidence >= lower and (upper is None or o.confidence < upper)]
        row = _summary(selected, n)
        row.update({"lower": lower, "upper_exclusive": upper,
                    "mean_confidence": mean(o.confidence for o in selected) if selected else None,
                    "sample_insufficient": len(selected) < 30})
        buckets.append(row)
    thresholds = [{"threshold": threshold, **_summary(
        [o for o in observations if o.confidence >= threshold], n,
    )} for threshold in THRESHOLDS]
    quarters = []
    for i in range(4):
        selected = observations[(i * n) // 4:((i + 1) * n) // 4]
        row = _summary(selected, n)
        row.update({"quarter": i + 1,
                    "source_start": selected[0].prediction.timestamp.isoformat() if selected else None,
                    "source_end": selected[-1].prediction.timestamp.isoformat() if selected else None})
        quarters.append(row)
    hours = []
    for hour in range(24):
        selected = [o for o in observations if o.prediction.timestamp.hour == hour]
        row = _summary(selected, n)
        row.update({"hour_utc": hour, "sample_insufficient": len(selected) < 30})
        hours.append(row)
    timing = {}
    for status in ("TIMELY", "LATE", "UNKNOWN"):
        selected = [o for o in observations if (o.prediction.timing_status or "UNKNOWN") == status]
        timing[status] = _summary(selected, n)

    close_by_time = {_utc(row["open_time"]): float(row["close"]) for row in candle_rows}
    momentum_total = momentum_correct = model_matched_correct = source_mismatches = 0
    for item in observations:
        source_time = item.prediction.timestamp
        source_close = item.evaluation.source_close
        if source_time in close_by_time and abs(close_by_time[source_time] - source_close) > 1e-6:
            source_mismatches += 1
        previous_close = close_by_time.get(source_time - timedelta(minutes=1))
        if previous_close is None:
            continue
        momentum_total += 1
        # Same tie rule as the persisted target: flat is DOWN.
        predicted = "UP" if source_close > previous_close else "DOWN"
        momentum_correct += predicted == item.evaluation.observed_class
        model_matched_correct += item.evaluation.correct
    if source_mismatches:
        raise ValueError(f"{source_mismatches} evaluation source closes disagree with candles")

    eps = 1e-15
    brier = mean((p - (o.evaluation.observed_class == "UP")) ** 2 for p, o in zip(prob_up, observations))
    log_loss = mean(-math.log(max(min(p if o.evaluation.observed_class == "UP" else 1 - p, 1 - eps), eps))
                    for p, o in zip(prob_up, observations))
    observed_counts = Counter(o.evaluation.observed_class for o in observations)
    predicted_counts = Counter(o.prediction.predicted_class for o in observations)
    source_gaps = [
        (right.prediction.timestamp - left.prediction.timestamp).total_seconds() / 60
        for left, right in zip(observations, observations[1:])
        if right.prediction.timestamp - left.prediction.timestamp > timedelta(minutes=1)
    ]
    source_dates = Counter(o.prediction.timestamp.date().isoformat() for o in observations)
    return {
        "analyzed_at_utc": cutoff.astimezone(timezone.utc).isoformat(),
        "source": "Supabase public.predictions JOIN public.live_evaluations; public.candles for momentum baseline",
        "symbol": observations[0].prediction.symbol,
        "interval": observations[0].prediction.interval,
        "model_version": model_version,
        "persisted_evaluation_rows": len(evaluation_rows),
        "model_version_rows": dict(models),
        "source_candle_range_utc": [observations[0].prediction.timestamp.isoformat(), observations[-1].prediction.timestamp.isoformat()],
        "evaluated_at_range_utc": [min(o.evaluation.evaluated_at for o in observations).isoformat(),
                                    max(o.evaluation.evaluated_at for o in observations).isoformat()],
        "source_date_counts_utc": dict(sorted(source_dates.items())),
        "source_gap_count": len(source_gaps),
        "largest_source_gap_minutes": max(source_gaps, default=0),
        "evaluated": n, "correct": sum(o.evaluation.correct for o in observations),
        "incorrect": sum(not o.evaluation.correct for o in observations),
        "accuracy": _rate(sum(o.evaluation.correct for o in observations), n),
        "observed_move": dict(Counter(o.evaluation.observed_move for o in observations)),
        "confusion_matrix_predicted_rows_actual_columns": matrix,
        "per_class": per_class,
        "macro_f1": mean(per_class[label]["f1"] or 0 for label in ("DOWN", "UP")),
        "weighted_f1": weighted_f1,
        "balanced_accuracy": mean(per_class[label]["recall"] or 0 for label in ("DOWN", "UP")),
        "predicted_classes": {label: {"count": predicted_counts[label], "share": predicted_counts[label] / n} for label in ("DOWN", "UP")},
        "observed_classes": {label: {"count": observed_counts[label], "share": observed_counts[label] / n} for label in ("DOWN", "UP")},
        "probability_distribution": distribution,
        "confidence_buckets": buckets,
        "thresholds": thresholds,
        "temporal_quarters": quarters,
        "hourly_utc": hours,
        "timing": timing,
        "brier_up": brier,
        "log_loss_up": log_loss,
        "baselines": {
            "train_majority_down": {"count": n, "correct": observed_counts["DOWN"], "accuracy": observed_counts["DOWN"] / n},
            "always_up": {"count": n, "correct": observed_counts["UP"], "accuracy": observed_counts["UP"] / n},
            "previous_movement_same": {"count": momentum_total, "correct": momentum_correct,
                                       "accuracy": _rate(momentum_correct, momentum_total),
                                       "model_accuracy_same_rows": _rate(model_matched_correct, momentum_total),
                                       "missing_previous_candle": n - momentum_total},
        },
        "methodology": {
            "cohort": "All persisted BTCUSDT 1m live_evaluations with evaluated_at <= analyzed_at, joined to xgboost_v1 predictions",
            "class_rule": "UP iff target close > source close; DOWN includes ties (FLAT)",
            "confidence": "max(original XGBoost P(UP), P(DOWN)); no recalibration or threshold change",
            "buckets": "Lower bound inclusive, upper bound exclusive; >=70% has no upper bound",
            "small_sample": "Fewer than 30 rows: descriptive sample insufficient; no inferential claim",
            "probability_scores": "Brier=(P(UP)-actual_UP)^2; binary log loss uses original P(UP), clipped to [1e-15,1-1e-15] only for log",
            "momentum": "Predict target class from source close versus immediately previous closed candle; ties DOWN; matched-cohort accuracy shown",
            "temporal": "Four equal-sized blocks ordered by source candle open time, UTC hour based on source candle",
            "precision_null": "Precision is null when a class is never predicted in a subset",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref", default=os.getenv("SUPABASE_PROJECT_REF", ""))
    parser.add_argument("--url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--output", type=Path, default=Path("reports/live_model_diagnostic.json"))
    parser.add_argument("--model-version", default="xgboost_v1")
    args = parser.parse_args()
    if not args.url and not args.project_ref:
        parser.error("Set SUPABASE_URL or --project-ref")
    url = args.url or f"https://{args.project_ref}.supabase.co"
    key = os.getenv("SUPABASE_KEY", "")
    if not key:
        if not args.project_ref:
            parser.error("Set SUPABASE_KEY or --project-ref for local CLI lookup")
        key = _publishable_key(args.project_ref)
    _require_public_read_key(key)
    cutoff = datetime.now(timezone.utc)
    cutoff_iso = cutoff.isoformat()
    with httpx.Client(timeout=30.0) as client:
        evaluations, count = _fetch_all(client, url, key, "live_evaluations", [
            ("select", EVALUATION_SELECT), ("symbol", "eq.BTCUSDT"),
            ("evaluated_at", f"lte.{cutoff_iso}"),
            ("order", "evaluated_at.asc,evaluation_id.asc"),
        ])
        if not evaluations:
            raise RuntimeError("No persisted LIVE evaluations at the cutoff")
        observations, _ = _observations(evaluations, args.model_version)
        if not observations:
            raise RuntimeError("No observations for the requested model version")
        start = observations[0].prediction.timestamp - timedelta(minutes=1)
        stop = observations[-1].prediction.timestamp
        candles, _ = _fetch_all(client, url, key, "candles", [
            ("select", "open_time,close"), ("symbol", "eq.BTCUSDT"), ("interval", "eq.1m"),
            ("open_time", f"gte.{start.isoformat()}"), ("open_time", f"lte.{stop.isoformat()}"),
            ("order", "open_time.asc"),
        ])
    report = analyze_rows(evaluations, candles, cutoff=cutoff, model_version=args.model_version)
    if count != report["persisted_evaluation_rows"]:
        raise RuntimeError("Exact count differs from downloaded evaluation rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}; cutoff UTC: {cutoff_iso}; evaluated: {report['evaluated']}; accuracy: {report['accuracy']:.4f}")


if __name__ == "__main__":
    main()
