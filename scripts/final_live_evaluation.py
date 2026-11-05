"""Extend a frozen read-only LIVE diagnostic with discrimination and gate audit.

Run scripts.live_model_diagnostic first with a distinct output path. This tool
re-reads only the persisted rows at that report's cutoff and writes aggregates.
It does not train, select a candidate, or mutate any cloud record.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import auc, average_precision_score, balanced_accuracy_score, precision_recall_curve, roc_auc_score

from scripts import live_model_diagnostic as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/final_live_evaluation_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/final_live_evaluation_v1.md"


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def binary_metrics(observations: list[base.Observation]) -> dict:
    y = np.array([o.evaluation.observed_class == "UP" for o in observations], dtype=int)
    p = np.array([o.prediction.prob_up for o in observations], dtype=float)
    predicted = np.array([o.prediction.predicted_class == "UP" for o in observations], dtype=int)
    if len(set(y)) < 2:
        return {"roc_auc_up": None, "pr_auc_up": None, "average_precision_up": None,
                "balanced_accuracy": None, "predicted_up_fraction": float(predicted.mean()),
                "actual_up_fraction": float(y.mean())}
    precision, recall, _ = precision_recall_curve(y, p)
    return {"roc_auc_up": float(roc_auc_score(y, p)),
            "pr_auc_up": float(auc(recall, precision)),
            "average_precision_up": float(average_precision_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "predicted_up_fraction": float(predicted.mean()),
            "actual_up_fraction": float(y.mean())}


def gate_60(observations: list[base.Observation], candle_closes: dict) -> dict:
    selected = [o for o in observations if o.confidence >= .60]
    n = len(selected)
    correct = sum(o.evaluation.correct for o in selected)
    previous = []
    for o in selected:
        prior = candle_closes.get(o.prediction.timestamp - timedelta(minutes=1))
        if prior is not None:
            previous.append((o.evaluation.source_close > prior) == (o.evaluation.observed_class == "UP"))
    return {"threshold": .60, "count": n, "coverage": n / len(observations),
            "correct": correct, "accuracy": correct / n if n else None,
            "accuracy_wilson_95_independence_optimistic": wilson(correct, n),
            "predicted_up": sum(o.prediction.predicted_class == "UP" for o in selected),
            "predicted_down": sum(o.prediction.predicted_class == "DOWN" for o in selected),
            "actual_up": sum(o.evaluation.observed_class == "UP" for o in selected),
            "actual_down": sum(o.evaluation.observed_class == "DOWN" for o in selected),
            "always_down_accuracy_same_rows": (sum(o.evaluation.observed_class == "DOWN" for o in selected) / n if n else None),
            "always_up_accuracy_same_rows": (sum(o.evaluation.observed_class == "UP" for o in selected) / n if n else None),
            "previous_direction_rows": len(previous),
            "previous_direction_accuracy_same_rows": sum(previous) / len(previous) if previous else None,
            "caution": "Selection is post-hoc descriptive; Wilson assumes independence and is optimistic under minute autocorrelation."}


def extend(report: dict, observations: list[base.Observation], candle_rows: list[dict]) -> dict:
    if len(observations) != report["evaluated"]:
        raise ValueError("Frozen LIVE cutoff or persisted cohort changed")
    if observations[0].prediction.timestamp.isoformat() != report["source_candle_range_utc"][0] or observations[-1].prediction.timestamp.isoformat() != report["source_candle_range_utc"][1]:
        raise ValueError("LIVE source range differs from frozen report")
    output = dict(report)
    output["discrimination"] = binary_metrics(observations)
    n = len(observations)
    output["temporal_quarters_extended"] = []
    for quarter in range(4):
        selected = observations[quarter * n // 4:(quarter + 1) * n // 4]
        metrics = binary_metrics(selected)
        metrics.update({"quarter": quarter + 1, "count": len(selected),
                        "source_start_utc": selected[0].prediction.timestamp.isoformat(),
                        "source_end_utc": selected[-1].prediction.timestamp.isoformat(),
                        "correct": sum(o.evaluation.correct for o in selected),
                        "accuracy": sum(o.evaluation.correct for o in selected) / len(selected)})
        output["temporal_quarters_extended"].append(metrics)
    close_by_time = {base._utc(row["open_time"]): float(row["close"]) for row in candle_rows}
    output["gate_60_final"] = gate_60(observations, close_by_time)
    output["phase1_comparison_note"] = "FASE 1 is frozen; this later cutoff is descriptive, not used to select candidates."
    return output


def markdown(report: dict, phase1: dict) -> str:
    d = report["discrimination"]
    gate = report["gate_60_final"]
    fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
    lines = ["# Final LIVE evaluation — xgboost_v1", "",
             f"Frozen at **{report['analyzed_at_utc']} UTC**. Source: persisted Supabase predictions, live_evaluations and candles; no LIVE fitting.",
             f"Source candle range: {report['source_candle_range_utc'][0]} to {report['source_candle_range_utc'][1]}.", "",
             "| Metric | Final LIVE | Frozen FASE 1 |", "| --- | ---: | ---: |",
             f"| Evaluated | {report['evaluated']} | {phase1['evaluated']} |",
             f"| Accuracy | {report['accuracy']:.4f} | {phase1['accuracy']:.4f} |",
             f"| Balanced accuracy | {report['balanced_accuracy']:.4f} | {phase1['balanced_accuracy']:.4f} |",
             f"| Macro F1 | {report['macro_f1']:.4f} | {phase1['macro_f1']:.4f} |",
             f"| ROC AUC UP | {fmt(d['roc_auc_up'])} | N/A in FASE 1 JSON |",
             f"| PR AUC UP | {fmt(d['pr_auc_up'])} | N/A in FASE 1 JSON |",
             f"| Average precision UP | {fmt(d['average_precision_up'])} | N/A in FASE 1 JSON |",
             f"| Brier | {report['brier_up']:.4f} | {phase1['brier_up']:.4f} |",
             f"| Log Loss | {report['log_loss_up']:.4f} | {phase1['log_loss_up']:.4f} |", "",
             "## Temporal quarters (source UTC)", "",
             "| Quarter | N | Accuracy | Balanced accuracy | ROC AUC | Pred UP | Actual UP |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["temporal_quarters_extended"]:
        lines.append(f"| {row['quarter']} | {row['count']} | {row['accuracy']:.4f} | {fmt(row['balanced_accuracy'])} | {fmt(row['roc_auc_up'])} | {row['predicted_up_fraction']:.2%} | {row['actual_up_fraction']:.2%} |")
    lines += ["", "## Confidence gate 60%", "",
              f"{gate['count']} of {report['evaluated']} ({gate['coverage']:.2%}) passed; {gate['correct']} correct; conditional accuracy {fmt(gate['accuracy'])}.",
              f"Same selected rows: always DOWN {fmt(gate['always_down_accuracy_same_rows'])}, always UP {fmt(gate['always_up_accuracy_same_rows'])}, previous direction {fmt(gate['previous_direction_accuracy_same_rows'])} on {gate['previous_direction_rows']} matched rows.",
              f"Wilson 95% interval {gate['accuracy_wilson_95_independence_optimistic']}; minute autocorrelation makes this optimistic. No threshold was changed or validated.", "",
              "## Methodology", "",
              "A public/read-only key fetched the exact persisted cohort at the frozen cutoff. ROC AUC and average precision use original P(UP). DOWN includes ties. The four source-time quarters are descriptive only. No individual prediction or secret is published."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--project-ref", default="")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    project_ref = args.project_ref or (ROOT / "supabase/.temp/project-ref").read_text(encoding="utf-8").strip()
    key = base._publishable_key(project_ref)
    base._require_public_read_key(key)
    url = f"https://{project_ref}.supabase.co"
    cutoff = report["analyzed_at_utc"]
    with httpx.Client(timeout=30.0) as client:
        rows, count = base._fetch_all(client, url, key, "live_evaluations", [
            ("select", base.EVALUATION_SELECT), ("symbol", "eq.BTCUSDT"),
            ("evaluated_at", f"lte.{cutoff}"), ("order", "evaluated_at.asc,evaluation_id.asc")])
        observations, _ = base._observations(rows, "xgboost_v1")
        if count != report["persisted_evaluation_rows"]:
            raise ValueError("Frozen evaluation count changed")
        start = observations[0].prediction.timestamp - timedelta(minutes=1)
        stop = observations[-1].prediction.timestamp
        candles, _ = base._fetch_all(client, url, key, "candles", [
            ("select", "open_time,close"), ("symbol", "eq.BTCUSDT"), ("interval", "eq.1m"),
            ("open_time", f"gte.{start.isoformat()}"), ("open_time", f"lte.{stop.isoformat()}"),
            ("order", "open_time.asc")])
    final = extend(report, observations, candles)
    args.report.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    phase1 = json.loads((ROOT / "reports/live_model_diagnostic.json").read_text(encoding="utf-8"))
    DEFAULT_MARKDOWN.write_text(markdown(final, phase1), encoding="utf-8")
    print(f"Frozen LIVE: n={final['evaluated']}; ROC AUC={final['discrimination']['roc_auc_up']:.4f}; gate60_n={final['gate_60_final']['count']}")


if __name__ == "__main__":
    main()
