"""Evaluación temporal de probabilidades y señales, sin costes ni ejecución."""

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame
from alpha_engine.features.target import target_distribution
from alpha_engine.ml.baseline import MajorityBaseline
from alpha_engine.ml.evaluation import evaluate_binary
from alpha_engine.ml.trainer import XGB_PARAMS, _atomic_json

REPORT_VERSION = "walk_forward_v1"
DEPTH_CANDIDATES = (2, 3, 4)
THRESHOLD_CANDIDATES = (0.50, 0.52, 0.55, 0.60, 0.65)
CALIBRATION_BINS = 10


def candle_digest(candles: Sequence[Candle]) -> str:
    """Identifica los precios exactos usados por evaluación y simulación."""
    digest = hashlib.sha256()
    for candle in candles:
        values = [
            candle.symbol, candle.interval,
            candle.open_time.astimezone(timezone.utc).isoformat(),
            candle.close_time.astimezone(timezone.utc).isoformat(),
            candle.open, candle.high, candle.low, candle.close, candle.volume,
        ]
        digest.update(json.dumps(values, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _range(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
        "class_distribution": target_distribution(frame),
    }


def _probability_metrics(actual: np.ndarray, prob_up: np.ndarray) -> dict:
    if len(actual) != len(prob_up) or len(actual) == 0:
        raise ValueError("Las probabilidades deben corresponder a etiquetas observadas")
    if not np.isfinite(prob_up).all() or ((prob_up < 0) | (prob_up > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1")
    bins = np.minimum((prob_up * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    reliability = []
    for index in range(CALIBRATION_BINS):
        selected = bins == index
        reliability.append({
            "lower": index / CALIBRATION_BINS,
            "upper": (index + 1) / CALIBRATION_BINS,
            "rows": int(selected.sum()),
            "mean_prob_up": float(prob_up[selected].mean()) if selected.any() else None,
            "observed_up_fraction": float(actual[selected].mean()) if selected.any() else None,
        })
    return {
        "brier_score": float(brier_score_loss(actual, prob_up)),
        "log_loss": float(log_loss(actual, prob_up, labels=[0, 1])),
        "reliability": reliability,
    }


def _signal_metrics(actual: np.ndarray, prob_up: np.ndarray, threshold: float | None) -> dict:
    predicted = (prob_up > 0.5).astype(int)
    eligible = (np.maximum(prob_up, 1 - prob_up) >= threshold) if threshold is not None else np.zeros(len(actual), dtype=bool)
    correct = predicted == actual
    by_signal = {}
    for signal, direction in (("BUY", 1), ("SELL", 0)):
        selected = eligible & (predicted == direction)
        count = int(selected.sum())
        by_signal[signal] = {
            "rows": count,
            "correct": int((selected & correct).sum()),
            "accuracy": float(correct[selected].mean()) if count else None,
        }
    count = int(eligible.sum())
    return {
        "threshold": threshold,
        "counts": {"BUY": by_signal["BUY"]["rows"], "SELL": by_signal["SELL"]["rows"], "HOLD": len(actual) - count},
        "coverage": count / len(actual),
        "direction_accuracy_eligible": float(correct[eligible].mean()) if count else None,
        "directional_edge_per_row": (2 * int((eligible & correct).sum()) - count) / len(actual),
        "by_signal": by_signal,
    }


def _select_threshold(actual: np.ndarray, prob_up: np.ndarray) -> tuple[float | None, list[dict]]:
    candidates = [_signal_metrics(actual, prob_up, threshold) for threshold in THRESHOLD_CANDIDATES]
    # HOLD tiene ventaja cero. Una señal se elige sólo si la validación supera cero.
    best = max(candidates, key=lambda row: (row["directional_edge_per_row"], row["coverage"]))
    return (best["threshold"] if best["directional_edge_per_row"] > 0 else None), candidates


def evaluate_walk_forward(candles: Sequence[Candle], windows: int = 4) -> dict:
    """Entrena en pasado creciente; valida y prueba en bloques disjuntos.

    La última etiqueta de TRAIN y VALIDATION se purga porque mira el primer
    close del bloque siguiente. Las features se calculan causalmente.
    """
    if windows < 2 or windows > 8:
        raise ValueError("windows debe estar entre 2 y 8")
    feature_frame = build_feature_frame(candles)
    labeled = build_training_frame(feature_frame)
    total = len(labeled)
    initial_train = total // 2
    block_rows = (total - initial_train) // (2 * windows)
    if block_rows < 30 or initial_train < 100:
        raise ValueError("No hay suficientes velas etiquetadas para las ventanas solicitadas")

    features = list(FEATURE_COLUMNS)
    results = []
    for number in range(windows):
        boundary = initial_train + number * 2 * block_rows
        train = labeled.iloc[:boundary - 1]
        validation = labeled.iloc[boundary:boundary + block_rows - 1]
        test_end = boundary + 2 * block_rows if number < windows - 1 else total
        test = labeled.iloc[boundary + block_rows:test_end]
        if train["target"].nunique() != 2:
            raise ValueError(f"TRAIN de ventana {number + 1} requiere ambas clases")
        y_validation = validation["target"].to_numpy(dtype=int)
        y_test = test["target"].to_numpy(dtype=int)
        baseline = MajorityBaseline.fit(train["target"].to_numpy(dtype=int))
        prevalence = float(train["target"].mean())

        candidates = []
        fitted = []
        for depth in DEPTH_CANDIDATES:
            params = {**XGB_PARAMS, "max_depth": depth}
            model = XGBClassifier(**params)
            model.fit(train[features], train["target"])
            validation_prob = np.asarray(model.predict_proba(validation[features]))[:, list(model.classes_).index(1)]
            score = float(brier_score_loss(y_validation, validation_prob))
            candidates.append({"max_depth": depth, "validation_brier_score": score})
            fitted.append((model, validation_prob))
        selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation_brier_score"])
        model, validation_prob = fitted[selected]
        threshold, threshold_candidates = _select_threshold(y_validation, validation_prob)
        test_prob = np.asarray(model.predict_proba(test[features]))[:, list(model.classes_).index(1)]
        test_prediction = (test_prob > 0.5).astype(int)
        test_eligible = (np.maximum(test_prob, 1 - test_prob) >= threshold) if threshold is not None else np.zeros(len(test), dtype=bool)
        test_decisions = [
            {"timestamp": timestamp.isoformat(), "prob_up": float(test_prob[i]),
             "signal": "HOLD" if not test_eligible[i] else "BUY" if test_prediction[i] else "SELL"}
            for i, timestamp in enumerate(test.index)
        ]
        baseline_probability = np.full(len(test), prevalence)
        baseline_prediction = baseline.predict(len(test))

        results.append({
            "window": number + 1,
            "train": _range(train),
            "validation": _range(validation),
            "test": _range(test),
            "purge_rows_at_each_boundary": 1,
            "selection": {
                "model_candidates": candidates,
                "selected_max_depth": DEPTH_CANDIDATES[selected],
                "threshold_candidates": threshold_candidates,
                "selected_threshold": threshold,
                "criterion": "lowest validation Brier for model; highest (correct - incorrect) / all rows for threshold; HOLD if no positive edge",
            },
            "baseline": {
                "majority_class": baseline.majority_class,
                "train_up_fraction": prevalence,
                "direction": evaluate_binary(y_test, baseline_prediction),
                "probability": _probability_metrics(y_test, baseline_probability),
            },
            "model": {
                "direction": evaluate_binary(y_test, test_prediction),
                "probability": _probability_metrics(y_test, test_prob),
            },
            "signal": _signal_metrics(y_test, test_prob, threshold),
            "test_decisions": test_decisions,
            "hold_baseline": {"coverage": 0.0, "directional_edge_per_row": 0.0},
        })

    accuracy_delta = [row["model"]["direction"]["accuracy"] - row["baseline"]["direction"]["accuracy"] for row in results]
    brier_delta = [row["model"]["probability"]["brier_score"] - row["baseline"]["probability"]["brier_score"] for row in results]
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(feature_frame["symbol"].iloc[0]),
        "interval": "1m",
        "target_definition": "1 si close(t+1m) > close(t); 0 si empata o baja",
        "probabilities_calibrated": False,
        "features": features,
        "fixed_xgboost_parameters": {key: value for key, value in XGB_PARAMS.items() if key != "max_depth"},
        "depth_candidates": list(DEPTH_CANDIDATES),
        "threshold_candidates": list(THRESHOLD_CANDIDATES),
        "data_range": {"start": feature_frame.index[0].isoformat(), "end": feature_frame.index[-1].isoformat(), "closed_candles": len(feature_frame), "labeled_rows": total},
        "input_sha256": candle_digest(candles),
        "window_design": {"type": "expanding_train_rolling_validation_disjoint_test", "windows": windows, "block_rows": block_rows, "initial_train_rows_before_purge": initial_train},
        "windows": results,
        "summary": {
            "model_accuracy_minus_baseline": {"mean": float(np.mean(accuracy_delta)), "min": float(np.min(accuracy_delta)), "max": float(np.max(accuracy_delta))},
            "model_brier_minus_baseline": {"mean": float(np.mean(brier_delta)), "min": float(np.min(brier_delta)), "max": float(np.max(brier_delta))},
            "windows_beating_baseline_accuracy": sum(delta > 0 for delta in accuracy_delta),
            "windows_beating_baseline_brier": sum(delta < 0 for delta in brier_delta),
            "total_test_rows": sum(row["test"]["rows"] for row in results),
            "signal_rows": sum(row["signal"]["counts"]["BUY"] + row["signal"]["counts"]["SELL"] for row in results),
        },
        "library_versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "limits": "Acierto, calibración y cobertura sin LLM, spread, comisiones, deslizamiento ni rentabilidad. No se ajustó un calibrador; prob_up sigue sin calibrar.",
    }


def save_walk_forward_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output_path)
    return output_path
