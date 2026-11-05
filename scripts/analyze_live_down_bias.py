"""Read-only diagnosis of the frozen xgboost_v1 LIVE DOWN bias.

Uses a public Supabase read key and the verified local training candles. Writes
aggregates only. No fitting, API writes, model changes, or raw row exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, log_loss, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score, auc,
)

from alpha_engine.data.schemas import Candle
from alpha_engine.data.timing import prediction_expires_at
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from alpha_engine.services.prediction import load_model_bundle
from scripts.analyze_feature_parity import _load_historical, percentiles, population_stability_index
from scripts.live_model_diagnostic import (
    EVALUATION_SELECT, _fetch_all, _observations, _publishable_key,
    _require_public_read_key, _utc,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/xgboost_v1.joblib"
EXPECTED_SHA = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
BUCKETS = ((None, .40), (.40, .45), (.45, .48), (.48, .50),
           (.50, .52), (.52, .55), (.55, .60), (.60, None))


def excess_down_bias(predicted_down: int, actual_down: int, n: int) -> float:
    if n <= 0:
        raise ValueError("Nonempty cohort required")
    return (predicted_down - actual_down) / n


def probability_bins(probabilities, actual_up, bounds=BUCKETS):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(actual_up, dtype=int)
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError("Aligned, finite probabilities required")
    result = []
    for low, high in bounds:
        mask = np.ones(len(p), dtype=bool)
        if low is not None:
            mask &= p >= low
        if high is not None:
            mask &= p < high
        n = int(mask.sum())
        result.append({"lower_inclusive": low, "upper_exclusive": high, "n": n,
                       "fraction": n / len(p) if len(p) else None,
                       "mean_probability_up": float(p[mask].mean()) if n else None,
                       "actual_up_rate": float(y[mask].mean()) if n else None})
    if sum(row["n"] for row in result) != len(p):
        raise ValueError("Probability bins do not cover every observation")
    return result


def feature_range(train, live):
    a = np.asarray(train, dtype=float)
    b = np.asarray(live, dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite, nonempty feature arrays required")
    p01, p99 = np.quantile(a, [.01, .99])
    return {"below_train_min": float(np.mean(b < a.min())),
            "above_train_max": float(np.mean(b > a.max())),
            "below_train_p01": float(np.mean(b < p01)),
            "above_train_p99": float(np.mean(b > p99)),
            "train_min": float(a.min()), "train_max": float(a.max()),
            "train_p01": float(p01), "train_p99": float(p99)}


def contribution_aggregate(values, names):
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(names) or not len(data):
        raise ValueError("Contribution matrix shape mismatch")
    return {name: {"mean": float(data[:, i].mean()),
                   "median": float(np.median(data[:, i])),
                   "mean_absolute": float(np.abs(data[:, i]).mean())}
            for i, name in enumerate(names)}


def confusion_predicted_rows(y, predicted):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if len(y) != len(predicted) or not len(y):
        raise ValueError("Aligned, nonempty labels required")
    return {"DOWN": {"DOWN": int(((predicted == 0) & (y == 0)).sum()),
                     "UP": int(((predicted == 0) & (y == 1)).sum())},
            "UP": {"DOWN": int(((predicted == 1) & (y == 0)).sum()),
                   "UP": int(((predicted == 1) & (y == 1)).sum())}}


def score(y, predicted, probability=None):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if not len(y) or len(y) != len(predicted):
        raise ValueError("Aligned, nonempty labels required")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    result = {"n": len(y), "accuracy": float(accuracy_score(y, predicted)),
              "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if len(set(y)) == 2 else None,
              "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)),
              "predicted_up_fraction": float(predicted.mean()),
              "actual_up_fraction": float(y.mean()),
              "matrix_predicted_rows_actual_columns": confusion_predicted_rows(y, predicted),
              "classes": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(("DOWN", "UP"))}}
    if probability is not None:
        p = np.asarray(probability, dtype=float)
        if len(p) != len(y) or not np.isfinite(p).all():
            raise ValueError("Aligned probabilities required")
        result.update({"brier": float(brier_score_loss(y, p)),
                       "log_loss": float(log_loss(y, p, labels=[0, 1]))})
        if len(set(y)) == 2:
            pr_precision, pr_recall, _ = precision_recall_curve(y, p)
            result.update({"roc_auc": float(roc_auc_score(y, p)),
                           "pr_auc_up": float(auc(pr_recall, pr_precision)),
                           "average_precision_up": float(average_precision_score(y, p))})
        else:
            result.update({"roc_auc": None, "pr_auc_up": None, "average_precision_up": None})
    return result


def temporal_subset(times, end, duration):
    """Return source times inside (end-duration, end], without assuming continuity."""
    if end.tzinfo is None:
        raise ValueError("UTC-aware end required")
    return np.asarray([end - duration < t <= end for t in times], dtype=bool)


def _market(observations):
    returns = np.array([(o.evaluation.target_close / o.evaluation.source_close - 1) * 10_000
                        for o in observations])
    first, last = observations[0], observations[-1]
    return {"return_bps": percentiles(returns),
            "cumulative_price_change": float(last.evaluation.target_close / first.evaluation.source_close - 1),
            "first_source_close": float(first.evaluation.source_close),
            "last_target_close": float(last.evaluation.target_close)}


def _time_row(indices, observations, y, predicted, p):
    selected = [observations[i] for i in indices]
    if not len(indices):
        return {"n": 0}
    result = score(y[indices], predicted[indices], p[indices])
    result["predicted_down_fraction"] = 1 - result["predicted_up_fraction"]
    result["actual_down_fraction"] = 1 - result["actual_up_fraction"]
    result["market"] = _market(selected)
    result["source_range_utc"] = [selected[0].prediction.timestamp.isoformat(),
                                  selected[-1].prediction.timestamp.isoformat()]
    return result


def _ece(p, y):
    bins = probability_bins(p, y, tuple((i / 10, (i + 1) / 10) for i in range(10)))
    return {"equal_width_10": sum(row["n"] / len(p) * abs(row["mean_probability_up"] - row["actual_up_rate"])
                                  for row in bins if row["n"]), "bins": bins}


def _calibration_slope_intercept(p, y):
    if len(set(y)) < 2 or np.ptp(p) < 1e-8:
        return None
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0]),
            "method": "Unpenalized logistic regression of observed UP on logit(original P(UP)); descriptive only"}


def _tree_thresholds(estimator, historic, live):
    nodes = estimator.get_booster().trees_to_dataframe()
    nodes = nodes[nodes.Feature.isin(FEATURE_COLUMNS)]
    answer = {}
    for name in FEATURE_COLUMNS:
        feature_nodes = nodes[nodes.Feature == name]
        train_values = historic[name].to_numpy(dtype=float)
        live_values = live[name].to_numpy(dtype=float)
        thresholds = feature_nodes.Split.to_numpy(dtype=float)
        if len(thresholds):
            train_left = np.array([(train_values < t).mean() for t in thresholds])
            live_left = np.array([(live_values < t).mean() for t in thresholds])
            gain = feature_nodes.Gain.to_numpy(dtype=float)
            answer[name] = {"splits": len(thresholds), "gain_sum": float(gain.sum()),
                            "mean_absolute_left_fraction_shift": float(np.abs(live_left-train_left).mean()),
                            "live_one_side_95pct_splits": int(((live_left < .05) | (live_left > .95)).sum()),
                            "train_one_side_95pct_splits": int(((train_left < .05) | (train_left > .95)).sum()),
                            "threshold_min": float(thresholds.min()), "threshold_max": float(thresholds.max())}
        else:
            answer[name] = {"splits": 0, "gain_sum": 0.0}
    return answer


def _markdown(report):
    m = report["live"]["metrics"]
    p = report["live"]["probability_up"]
    fmt = lambda value, digits=4: "N/A" if value is None else f"{value:.{digits}f}"
    lines = ["# Diagnóstico post-despliegue: exceso de DOWN", "",
             f"Corte UTC: **{report['cutoff_utc']}**. Fuente: Supabase público de solo lectura y modelo congelado. No se modificó producción.", "",
             f"Predicciones: {report['freeze']['prediction_count']}; evaluaciones: {m['n']}; rango de velas fuente: {report['live']['source_range_utc']}.",
             f"DOWN predicho: {1-m['predicted_up_fraction']:.2%}; DOWN real: {1-m['actual_up_fraction']:.2%}; exceso: {report['live']['excess_down_bias']:+.2%}.",
             f"Accuracy {m['accuracy']:.2%}; balanced accuracy {m['balanced_accuracy']:.2%}; ROC AUC {m['roc_auc']:.4f}; recall UP {m['classes']['UP']['recall']:.2%}.",
             f"P(UP) mediana {p['median']:.4f}; media {p['mean']:.4f}; margen mediano {report['live']['decision_margin']['median']:.4f}.", "",
             "## Matriz (filas predichas, columnas reales)", "", "| | DOWN real | UP real |", "|---|---:|---:|",
             f"| DOWN | {m['matrix_predicted_rows_actual_columns']['DOWN']['DOWN']} | {m['matrix_predicted_rows_actual_columns']['DOWN']['UP']} |",
             f"| UP | {m['matrix_predicted_rows_actual_columns']['UP']['DOWN']} | {m['matrix_predicted_rows_actual_columns']['UP']['UP']} |", "",
             "## Baselines, misma cohorte", "", "| Baseline | Accuracy | Balanced accuracy |", "|---|---:|---:|"]
    for key, val in report["live"]["baselines"].items():
        lines.append(f"| {key} | {val['accuracy']:.2%} | {val['balanced_accuracy']:.2%} |")
    lines += ["", "## Contribuciones nativas XGBoost (log-odds de UP)", "",
              "Valores negativos reducen P(UP). El término bias se informa separado; no demuestra causalidad externa.", "",
              "| Feature | Media LIVE | Media DOWN | Media UP | Media TEST |", "|---|---:|---:|---:|---:|"]
    c = report["contributions"]
    for name in FEATURE_COLUMNS:
        lines.append(f"| {name} | {c['live'][name]['mean']:+.4f} | {c['predicted_down'][name]['mean']:+.4f} | {c['predicted_up'][name]['mean']:+.4f} | {c['test'][name]['mean']:+.4f} |")
    lines += ["", "## Drift y rango", "", "| Feature | PSI TRAIN→LIVE | Bajo TRAIN min | Sobre TRAIN max | Fuera P01–P99 |", "|---|---:|---:|---:|---:|"]
    for name in FEATURE_COLUMNS:
        d = report["features"][name]
        r = d["out_of_range"]
        lines.append(f"| {name} | {d['psi_train_live']:.3f} | {r['below_train_min']:.2%} | {r['above_train_max']:.2%} | {r['below_train_p01']+r['above_train_p99']:.2%} |")
    lines += ["", "## Probabilidades y margen", "",
              f"P(UP): mínimo {p['min']:.4f}; P05 {p['p05']:.4f}; P10 {p['p10']:.4f}; P25 {p['p25']:.4f}; mediana {p['median']:.4f}; media {p['mean']:.4f}; P75 {p['p75']:.4f}; P90 {p['p90']:.4f}; P95 {p['p95']:.4f}; máximo {p['max']:.4f}.",
              f"Margen |P(UP)-0.5|: mediana {report['live']['decision_margin']['median']:.4f}; P75 {report['live']['decision_margin']['p75']:.4f}; P90 {report['live']['decision_margin']['p90']:.4f}; P95 {report['live']['decision_margin']['p95']:.4f}. Dentro de 0.01: {report['live']['decision_margin']['fraction_below_001']:.2%}; 0.02: {report['live']['decision_margin']['fraction_below_002']:.2%}; 0.05: {report['live']['decision_margin']['fraction_below_005']:.2%}.",
              "", "| Rango P(UP) | N | UP real |", "|---|---:|---:|"]
    for row in p["bins"]:
        label = f"{row['lower_inclusive'] if row['lower_inclusive'] is not None else '-∞'}–{row['upper_exclusive'] if row['upper_exclusive'] is not None else '+∞'}"
        lines.append(f"| {label} | {row['n']} | {fmt(row['actual_up_rate'])} |")
    lines += ["", "## Comparación histórica", "", "| Período | N | Predicho UP | UP real |", "|---|---:|---:|---:|"]
    for name, row in report["historical_comparison"].items():
        lines.append(f"| {name} | {row['n']} | {row['predicted_up_fraction']:.2%} | {row['actual_up_fraction']:.2%} |")
    lines += ["", "## Ventanas recientes al corte UTC", "", "| Ventana | N | DOWN predicho | DOWN real | Accuracy | Balanced acc. | ROC AUC | Cambio precio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["recent_windows"].items():
        if not row["n"]:
            lines.append(f"| {name} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {name} | {row['n']} | {row['predicted_down_fraction']:.2%} | {row['actual_down_fraction']:.2%} | {row['accuracy']:.2%} | {fmt(row['balanced_accuracy'])} | {fmt(row['roc_auc'])} | {row['market']['cumulative_price_change']:.2%} |")
    lines += ["", "## Serie diaria UTC", "", "| Día | N | DOWN predicho | DOWN real | Accuracy | ROC AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["timeline_utc"]["day"].items():
        lines.append(f"| {name} | {row['n']} | {row['predicted_down_fraction']:.2%} | {row['actual_down_fraction']:.2%} | {row['accuracy']:.2%} | {fmt(row['roc_auc'])} |")
    ema = report["features"]["ema_20"]
    ema_tree = report["tree_splits"]["ema_20"]
    lines += ["", "## EMA_20 y estructura del modelo", "",
              f"EMA_20 mediana TRAIN {ema['TRAIN']['median']:.2f}, TEST {ema['TEST']['median']:.2f}, LIVE {ema['LIVE']['median']:.2f}; LIVE sobre máximo TRAIN: {ema['out_of_range']['above_train_max']:.2%}. Umbrales de split EMA: {ema_tree['splits']} entre {ema_tree['threshold_min']:.2f} y {ema_tree['threshold_max']:.2f}; {ema_tree['live_one_side_95pct_splits']} dejan más de 95% de LIVE de un solo lado. Su contribución media LIVE es {c['live']['ema_20']['mean']:+.4f} log-odds de UP, frente a {c['test']['ema_20']['mean']:+.4f} en TEST.",
              f"En DOWN correctos la contribución EMA media es {c['down_correct']['ema_20']['mean']:+.4f}, y en DOWN incorrectos {c['down_incorrect']['ema_20']['mean']:+.4f}: empuja en ambos grupos de forma similar.",
              "Esto respalda que el nivel absoluto de EMA está asociado con el empuje hacia DOWN en estas observaciones. La contribución de árboles refleja asociación interna del modelo; no es una prueba de efecto causal del precio.",
              "", "## MACD y otras señales", ""]
    for name in ("macd", "macd_signal", "macd_histogram", "rsi_14", "return_1m"):
        d = report["features"][name]
        lines.append(f"- {name}: PSI {d['psi_train_live']:.3f}; mediana TRAIN {d['TRAIN']['median']:.6f}, LIVE {d['LIVE']['median']:.6f}; contribución media LIVE {c['live'][name]['mean']:+.4f} log-odds.")
    calibration = report["live"]["calibration"]
    lines += ["", "## Calibración y discriminación", "",
              f"Brier {m['brier']:.5f}; log loss {m['log_loss']:.5f}; ECE de 10 bins {calibration['ece']['equal_width_10']:.5f}; intercept {calibration['intercept_slope']['intercept']:.3f}; slope {calibration['intercept_slope']['slope']:.3f}.",
              f"ROC AUC {m['roc_auc']:.4f}; PR AUC UP {m['pr_auc_up']:.4f}; average precision UP {m['average_precision_up']:.4f}. La discriminación observada está cerca de azar; no se exploró ningún umbral alternativo.",
              "", "## Pipeline y prior", "",
              report["pipeline"]["interpretation"],
              f"Repetición de {report['pipeline']['replay_n']} predicciones: error máximo P(UP) {report['pipeline']['max_probability_replay_error']:.2e}, cambios de clase {report['pipeline']['replay_class_flips']}, no finitos {report['pipeline']['nonfinite_feature_values']}, desacuerdos de vela objetivo {report['pipeline']['target_candle_mismatches']}. Gaps de evaluaciones fuente: {report['live']['gap_count']}, mayor {report['live']['largest_gap_minutes']:.0f} minutos.",
              f"TRAIN UP {report['class_prior']['train_up']}, DOWN {report['class_prior']['train_down']}; base_score guardado {report['class_prior']['base_score_booster_config']}; sin pesos de clase ni sample weights en la ruta de entrenamiento.",
              "", "## Interpretación", "",
              "El mercado real no tuvo una proporción DOWN similar a la predicha. El modelo emite P(UP) mayoritariamente por debajo de 0.50 y su EMA_20 absoluta está fuera del rango de entrenamiento, con contribución negativa dominante en LIVE. La debilidad de ROC AUC y de balanced accuracy indica escasa capacidad para distinguir UP de DOWN. La repetición exacta del pipeline descarta una regresión de features/probabilidad en esta cohorte; no prueba ausencia de todo defecto imaginable.",
              "", "La caída neta de precio y la proporción de velas DOWN son medidas distintas; ambas se muestran por ventana. Los grupos por hora, distribuciones completas y detalles de árboles están en el JSON. PSI es descriptivo; una contribución interna no prueba causalidad. LIVE no se utilizó para entrenar ni elegir umbrales."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref", default=os.getenv("SUPABASE_PROJECT_REF", ""))
    parser.add_argument("--output", type=Path, default=ROOT / "reports/live_down_bias_diagnostic_v1.json")
    args = parser.parse_args()
    start = datetime.now(timezone.utc)
    sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise RuntimeError("Production model hash differs from the frozen reference")
    bundle = load_model_bundle(MODEL)
    if bundle["features"] != list(FEATURE_COLUMNS) or bundle["model_version"] != "xgboost_v1" or bundle["horizon"] != "1m":
        raise RuntimeError("Production metadata differs from xgboost_v1 H1")
    estimator = bundle["estimator"]
    metadata = json.loads((ROOT / "models/xgboost_v1_metadata.json").read_text(encoding="utf-8"))
    project_ref = args.project_ref or (ROOT / "supabase/.temp/project-ref").read_text(encoding="utf-8").strip()
    key = os.getenv("SUPABASE_KEY", "") or _publishable_key(project_ref)
    _require_public_read_key(key)
    url = f"https://{project_ref}.supabase.co"
    cutoff = start.isoformat()
    with httpx.Client(timeout=60) as client:
        eval_rows, eval_count = _fetch_all(client, url, key, "live_evaluations", [
            ("select", EVALUATION_SELECT), ("symbol", "eq.BTCUSDT"),
            ("evaluated_at", f"lte.{cutoff}"), ("order", "evaluated_at.asc,evaluation_id.asc")])
        observations, model_counts = _observations(eval_rows, "xgboost_v1")
        if not observations:
            raise RuntimeError("No evaluated xgboost_v1 rows")
        prediction_rows, pred_count = _fetch_all(client, url, key, "predictions", [
            ("select", "prediction_id,candle_open_time,generated_at,valid_until,model_version"),
            ("symbol", "eq.BTCUSDT"), ("model_version", "eq.xgboost_v1"),
            ("generated_at", f"lte.{cutoff}"), ("order", "candle_open_time.asc")])
        candle_start = observations[0].prediction.timestamp - timedelta(minutes=119)
        candle_end = max(observations[-1].prediction.timestamp,
                         _utc(prediction_rows[-1]["candle_open_time"])) + timedelta(minutes=1)
        candle_rows, candle_count = _fetch_all(client, url, key, "candles", [
            ("select", "symbol,interval,open_time,close_time,open,high,low,close,volume"),
            ("symbol", "eq.BTCUSDT"), ("interval", "eq.1m"),
            ("open_time", f"gte.{candle_start.isoformat()}"),
            ("open_time", f"lte.{candle_end.isoformat()}"), ("order", "open_time.asc")])
        run_response = client.get(f"{url}/rest/v1/public_pipeline_runs", headers={"apikey": key}, params=[
            ("select", "run_id,started_at,finished_at,status,candle_open_time,prediction_id"),
            ("symbol", "eq.BTCUSDT"), ("finished_at", f"lte.{cutoff}"),
            ("order", "started_at.desc"), ("limit", "1")])
        if run_response.status_code not in (200, 206):
            raise RuntimeError(f"Read-only public_pipeline_runs query failed: HTTP {run_response.status_code}")
        runs = run_response.json()
    if eval_count != len(eval_rows) or pred_count != len(prediction_rows):
        raise ValueError("Incomplete frozen cohort")
    candles = [Candle.model_validate(row) for row in candle_rows]
    candle_map = {c.open_time: i for i, c in enumerate(candles)}
    if len(candle_map) != candle_count:
        raise ValueError("Duplicate persisted candles")
    X_rows, missing_history, source_mismatch, target_mismatch, time_errors = [], [], [], [], []
    for o in observations:
        ts = o.prediction.timestamp
        index = candle_map.get(ts)
        if index is None or index < 119:
            missing_history.append(ts.isoformat()); continue
        window = candles[index - 119:index + 1]
        if any(right.open_time-left.open_time != timedelta(minutes=1) for left, right in zip(window,window[1:])):
            missing_history.append(ts.isoformat()); continue
        if abs(window[-1].close - o.evaluation.source_close) > 1e-6:
            source_mismatch.append(ts.isoformat())
        next_index = candle_map.get(ts + timedelta(minutes=1))
        if (next_index is None or
            abs(candles[next_index].close - o.evaluation.target_close) > 1e-6 or
            candles[next_index].close_time != o.evaluation.target_close_time):
            target_mismatch.append(ts.isoformat())
        if not (window[-1].close_time < o.prediction.generated_at <= o.evaluation.evaluated_at):
            time_errors.append(ts.isoformat())
        if o.prediction.valid_until != prediction_expires_at(window[-1].close_time, "1m"):
            time_errors.append(ts.isoformat())
        row = build_feature_frame(window).iloc[-1][list(FEATURE_COLUMNS)]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite LIVE feature replay")
        X_rows.append(row.to_numpy(dtype=float))
    if missing_history or source_mismatch or target_mismatch or time_errors:
        raise ValueError(f"Pipeline replay incomplete: missing={len(missing_history)}, source_mismatch={len(source_mismatch)}, target_mismatch={len(target_mismatch)}, time_errors={len(time_errors)}")
    live_x = pd.DataFrame(X_rows, columns=FEATURE_COLUMNS)
    live_p = estimator.predict_proba(live_x)[:, list(estimator.classes_).index(1)]
    persisted_p = np.array([o.prediction.prob_up for o in observations])
    max_replay_error = float(np.max(np.abs(live_p - persisted_p)))
    class_flips = int(np.sum((live_p > .5) != (persisted_p > .5)))
    y = np.array([o.evaluation.observed_class == "UP" for o in observations], dtype=int)
    predicted = np.array([o.prediction.predicted_class == "UP" for o in observations], dtype=int)
    if max_replay_error > 1e-5 or class_flips:
        raise ValueError(f"LIVE replay differs from persisted predictions: {max_replay_error}, {class_flips} flips")
    metrics = score(y, predicted, persisted_p)
    previous = np.array([int(o.evaluation.source_close > candles[candle_map[o.prediction.timestamp]-1].close)
                         for o in observations], dtype=int)
    baselines = {"xgboost_v1": metrics,
                 "always_DOWN": score(y, np.zeros(len(y), dtype=int)),
                 "always_UP": score(y, np.ones(len(y), dtype=int)),
                 "previous_movement": score(y, previous)}
    times = [o.prediction.timestamp for o in observations]
    recent = {}
    for hours in (1, 3, 6, 12, 24):
        indices = np.flatnonzero(temporal_subset(times, start, timedelta(hours=hours)))
        recent[f"{hours}h"] = _time_row(indices, observations, y, predicted, persisted_p)
    recent["full"] = _time_row(np.arange(len(y)), observations, y, predicted, persisted_p)
    grouped = {}
    for granularity in ("hour", "day"):
        groups = defaultdict(list)
        for i, ts in enumerate(times):
            key_time = ts.strftime("%Y-%m-%dT%H:00Z") if granularity == "hour" else ts.strftime("%Y-%m-%d")
            groups[key_time].append(i)
        grouped[granularity] = {key_time: _time_row(np.asarray(indices), observations,y,predicted,persisted_p)
                                for key_time,indices in sorted(groups.items())}
    train_candles, hist_features, parts = _load_historical(metadata)
    historical = {}
    for name, part in parts.items():
        hp = estimator.predict_proba(part[list(FEATURE_COLUMNS)])[:, list(estimator.classes_).index(1)]
        historical[name] = {"n": len(part), "predicted_up_fraction": float(np.mean(hp > .5)),
                            "actual_up_fraction": float(part.target.mean()),
                            "source_range_utc": [part.index[0].isoformat(), part.index[-1].isoformat()]}
    phase1 = json.loads((ROOT / "reports/live_model_diagnostic.json").read_text(encoding="utf-8"))
    historical["LIVE_PHASE1"] = {"n": phase1["evaluated"],
                                   "predicted_up_fraction": phase1["predicted_classes"]["UP"]["share"],
                                   "actual_up_fraction": phase1["observed_classes"]["UP"]["share"],
                                   "source_range_utc": phase1["source_candle_range_utc"]}
    historical["LIVE_CURRENT"] = {"n": len(y), "predicted_up_fraction": float(predicted.mean()),
                                   "actual_up_fraction": float(y.mean()),
                                   "source_range_utc": [times[0].isoformat(),times[-1].isoformat()]}
    last24_mask = temporal_subset(times,start,timedelta(hours=24))
    feature_report = {}
    for name in FEATURE_COLUMNS:
        train_values = parts["TRAIN"][name].to_numpy(dtype=float)
        live_values = live_x[name].to_numpy(dtype=float)
        feature_report[name] = {"TRAIN": percentiles(train_values),
                                "VALIDATION": percentiles(parts["VALIDATION"][name]),
                                "TEST": percentiles(parts["TEST"][name]),
                                "LIVE": percentiles(live_values),
                                "LIVE_LAST24H": percentiles(live_values[last24_mask]),
                                "psi_train_live": population_stability_index(train_values, live_values),
                                "psi_train_last24h": population_stability_index(train_values,live_values[last24_mask]) if last24_mask.any() else None,
                                "out_of_range": feature_range(train_values,live_values)}
    booster = estimator.get_booster()
    names = list(FEATURE_COLUMNS)+["BIAS"]
    live_contrib = booster.predict(xgb.DMatrix(live_x, feature_names=list(FEATURE_COLUMNS)), pred_contribs=True)
    test_x = parts["TEST"][list(FEATURE_COLUMNS)]
    test_contrib = booster.predict(xgb.DMatrix(test_x,feature_names=list(FEATURE_COLUMNS)),pred_contribs=True)
    contribution_replay_error = float(np.max(np.abs(expit(live_contrib.sum(axis=1))-live_p)))
    if contribution_replay_error > 1e-5:
        raise ValueError("Native contributions do not sum to the model margin")
    down_correct = (predicted == 0) & (y == 0)
    down_wrong = (predicted == 0) & (y == 1)
    contributions = {"unit": "raw XGBoost margin / log-odds of UP; sigmoid(sum of 8 feature contributions + BIAS) = P(UP)",
                     "max_sigmoid_reconstruction_error": contribution_replay_error,
                     "live": contribution_aggregate(live_contrib,names),
                     "predicted_down": contribution_aggregate(live_contrib[predicted==0],names),
                     "predicted_up": contribution_aggregate(live_contrib[predicted==1],names),
                     "down_correct": contribution_aggregate(live_contrib[down_correct],names),
                     "down_incorrect": contribution_aggregate(live_contrib[down_wrong],names),
                     "down_correct_n": int(down_correct.sum()), "down_incorrect_n": int(down_wrong.sum()),
                     "test": contribution_aggregate(test_contrib,names)}
    margin = np.abs(persisted_p - .5)
    margin_report = {**percentiles(margin), "p90": float(np.quantile(margin,.9)),
                     "fraction_below_001":float(np.mean(margin < .01)),
                     "fraction_below_002":float(np.mean(margin < .02)),
                     "fraction_below_005":float(np.mean(margin < .05))}
    prob_report = {**percentiles(persisted_p), "p10": float(np.quantile(persisted_p,.1)),
                   "p90": float(np.quantile(persisted_p,.9)),
                   "bins": probability_bins(persisted_p,y)}
    config = json.loads(booster.save_config())
    learner = config.get("learner", {})
    source_gaps = [(right-left).total_seconds()/60 for left,right in zip(times,times[1:]) if right-left>timedelta(minutes=1)]
    pipeline = {"feature_order":list(FEATURE_COLUMNS), "replay_n":len(live_x),
                "max_probability_replay_error":max_replay_error,"replay_class_flips":class_flips,
                "nonfinite_feature_values":int((~np.isfinite(live_x.to_numpy())).sum()),
                "missing_120_candle_history":len(missing_history),"source_close_mismatches":len(source_mismatch),
                "target_candle_mismatches":len(target_mismatch),
                "time_or_valid_until_errors":len(time_errors),"duplicate_predictions": pred_count-len({r['prediction_id'] for r in prediction_rows}),
                "implementation":"PredictionService.predict_candle_execution uses build_feature_frame(last 120 contiguous closed candles)",
                "interpretation":"La repetición exacta de las ocho features, la clase y P(UP) no muestra una regresión del pipeline."}
    report = {"cutoff_utc":cutoff, "source":"Supabase read-only public tables; frozen local Influx training candles; frozen xgboost_v1 artifact",
              "freeze":{"model_sha256":sha,"model_version":bundle["model_version"],"commit":os.popen("git rev-parse HEAD").read().strip(),
                        "prediction_count":pred_count,"evaluation_count_all_models":eval_count,"evaluation_count_xgboost_v1":len(y),
                        "model_counts":dict(model_counts),"latest_prediction":prediction_rows[-1],
                        "latest_pipeline_run":runs[0] if runs else None},
              "live":{"source_range_utc":[times[0].isoformat(),times[-1].isoformat()],
                      "evaluation_range_utc":[min(o.evaluation.evaluated_at for o in observations).isoformat(),max(o.evaluation.evaluated_at for o in observations).isoformat()],
                      "gap_count":len(source_gaps),"largest_gap_minutes":max(source_gaps,default=0),
                      "metrics":metrics,"excess_down_bias":excess_down_bias(int((predicted==0).sum()),int((y==0).sum()),len(y)),
                      "predicted_counts":{"DOWN":int((predicted==0).sum()),"UP":int((predicted==1).sum())},
                      "actual_counts":{"DOWN":int((y==0).sum()),"UP":int((y==1).sum())},
                      "baselines":baselines,"probability_up":prob_report,"decision_margin":margin_report,
                      "calibration":{"ece":_ece(persisted_p,y),"intercept_slope":_calibration_slope_intercept(persisted_p,y)}},
              "recent_windows":recent,"timeline_utc":grouped,"historical_comparison":historical,
              "features":feature_report,"contributions":contributions,
              "tree_splits":_tree_thresholds(estimator,parts["TRAIN"],live_x),
              "class_prior":{"train_up":int(parts["TRAIN"].target.sum()),
                             "train_down":int((parts["TRAIN"].target==0).sum()),
                             "base_score_param":estimator.get_params().get("base_score"),
                             "base_score_booster_config":learner.get("learner_model_param",{}).get("base_score"),
                             "scale_pos_weight":estimator.get_params().get("scale_pos_weight"),
                             "sample_weights_in_training_code":False},
              "pipeline":pipeline,"methodology":{"no_training":True,"no_threshold_selection":True,
                  "source_time":"Source candle open UTC; source close follows 59.999 s later; H1 target is next close",
                  "calibration":"ECE uses ten fixed width bins; slope/intercept are unpenalized logistic fit on logit P(UP), descriptive",
                  "psi":"TRAIN deciles; descriptive; sensitive to distinct ranges and large samples",
                  "tree_thresholds":"Unconditional distribution relative to each split threshold; routing also depends on preceding splits",
                  "market_return":"Per evaluation next_close/source_close-1; cumulative is last target close/first source close-1, descriptive only"}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report),encoding="utf-8")
    final_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if final_sha != sha:
        raise RuntimeError("Production model changed during read-only analysis")
    print(f"cutoff={cutoff} evaluations={len(y)} predicted_DOWN={(predicted==0).mean():.4f} actual_DOWN={(y==0).mean():.4f} replay_error={max_replay_error:.8f} model_sha_unchanged={final_sha==sha}")


if __name__ == "__main__":
    main()
