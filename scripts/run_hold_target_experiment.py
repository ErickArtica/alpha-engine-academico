"""Three-stage offline HOLD experiment: development, candidate lock, TEST.

TRAIN/VALIDATION and expanding folds determine at most two candidate deltas.
The final stage refuses to score TEST unless candidates were locked first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)
from xgboost import XGBClassifier

from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.analyze_hold_target import (DELTAS, LABELS, META, PRODUCT_SHA, REPORT as DISTRIBUTION_REPORT,
                                         ROOT, analyze, load_frozen_cohort, sha, ternary_labels, write_json, markdown)
from scripts.run_tfi_ablation import expanding_folds

DEVELOPMENT = ROOT / "reports/target_hold_development_v1.json"
LOCK = ROOT / "reports/target_hold_candidate_lock_v1.json"
FINAL = ROOT / "reports/target_hold_experiment_v1.json"
WALK = ROOT / "reports/target_hold_walk_forward_v1.json"
MODELS = ROOT / "models/experiments/hold_target_v1"
DELTAS_EXPERIMENT = tuple(delta for delta in DELTAS if delta > 0)
FEATURES = list(FEATURE_COLUMNS)
PARAMS = {**XGB_PARAMS, "objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3}


def code_sha() -> str:
    digest = hashlib.sha256()
    for relative in ("scripts/analyze_hold_target.py", "scripts/run_hold_target_experiment.py",
                     "alpha_engine/features/indicators.py", "alpha_engine/features/split.py"):
        digest.update(relative.encode() + b"\0" + (ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, FEATURES]
    if list(result.columns) != FEATURES or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Model input feature order or values differ")
    return result


def fit(frame: pd.DataFrame, delta: float) -> XGBClassifier:
    target = ternary_labels(frame["future_return_bps"].to_numpy(), delta)
    if set(np.unique(target)) != {0, 1, 2}:
        raise ValueError("TRAIN must contain DOWN, HOLD, UP")
    model = XGBClassifier(**PARAMS)
    model.fit(model_inputs(frame), target)
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Multiclass label order differs")
    return model


def predict(model: XGBClassifier, frame: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(model_inputs(frame)), dtype=float)
    if proba.shape != (len(frame), 3) or not np.isfinite(proba).all() or not np.allclose(proba.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid DOWN/HOLD/UP probabilities")
    # XGBoost float32 rows can sum to 0.99999994; sklearn's multiclass
    # log loss requires normalized probability rows.
    return proba / proba.sum(axis=1, keepdims=True)


def multiclass_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(y), 3) or not np.isin(y, [0, 1, 2]).all() or not np.isfinite(proba).all():
        raise ValueError("Invalid multiclass evaluation vectors")
    pred = np.argmax(proba, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    signal = pred != 1
    real_move = y != 1
    n_signal = int(signal.sum())
    n_hold_pred = int((~signal).sum())
    real_move_count = int(real_move.sum())
    missed_up = int(((pred == 1) & (y == 2)).sum())
    missed_down = int(((pred == 1) & (y == 0)).sum())
    false_action = int((signal & (y == 1)).sum())
    one_hot = np.eye(3)[y]
    entropy = -(proba * np.log(np.clip(proba, 1e-15, 1))).sum(axis=1)
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "multiclass_log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "multiclass_brier_sum": float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))),
        "per_class": {LABELS[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])} for i in (0, 1, 2)},
        "confusion_pred_rows_down_hold_up_real_columns_down_hold_up": confusion_matrix(pred, y, labels=[0, 1, 2]).tolist(),
        "predicted_counts": {LABELS[i]: int((pred == i).sum()) for i in (0, 1, 2)},
        "actual_counts": {LABELS[i]: int((y == i).sum()) for i in (0, 1, 2)},
        "n_signals": n_signal, "signal_coverage": n_signal / len(y),
        "directional_accuracy": float((pred[signal] == y[signal]).mean()) if n_signal else None,
        "false_actionable_signals": false_action,
        "false_actionable_rate_among_signals": false_action / n_signal if n_signal else None,
        "missed_up": missed_up, "missed_down": missed_down,
        "missed_move_rate_among_true_moves": (missed_up + missed_down) / real_move_count if real_move_count else None,
        "missed_move_rate_among_abstentions": (missed_up + missed_down) / n_hold_pred if n_hold_pred else None,
        "mean_probability": {LABELS[i]: float(proba[:, i].mean()) for i in (0, 1, 2)},
        "mean_entropy_nats": float(entropy.mean()),
    }


def baseline_metrics(y: np.ndarray, train_y: np.ndarray) -> dict:
    majority = int(np.argmax(np.bincount(train_y, minlength=3)))
    result = {}
    for name, label in (("train_majority", majority), ("always_hold", 1)):
        pred = np.full(len(y), label, dtype=int)
        result[name] = {"class": LABELS[label], "accuracy": float(accuracy_score(y, pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                        "signal_coverage": float(np.mean(pred != 1))}
    return result


def development() -> dict:
    _, split, meta, phase4 = load_frozen_cohort()
    folds = list(expanding_folds(pd.concat([split.train, split.validation]).sort_index()))
    results = {}
    walk = {}
    for delta in DELTAS_EXPERIMENT:
        key = str(delta)
        model = fit(split.train, delta)
        train_y = ternary_labels(split.train["future_return_bps"].to_numpy(), delta)
        validation_y = ternary_labels(split.validation["future_return_bps"].to_numpy(), delta)
        results[key] = {
            "train": multiclass_metrics(train_y, predict(model, split.train)),
            "validation": multiclass_metrics(validation_y, predict(model, split.validation)),
            "validation_baselines": baseline_metrics(validation_y, train_y),
        }
        walk[key] = []
        for number, past, future in folds:
            fitted = fit(past, delta)
            future_y = ternary_labels(future["future_return_bps"].to_numpy(), delta)
            score = multiclass_metrics(future_y, predict(fitted, future))
            walk[key].append({"fold": number, "train_end": past.index[-1].isoformat(),
                              "validation_start": future.index[0].isoformat(),
                              "validation_end": future.index[-1].isoformat(),
                              "purged_rows": 1,
                              "metrics": score,
                              "baselines": baseline_metrics(future_y, ternary_labels(past["future_return_bps"].to_numpy(), delta))})
        print(f"Development delta {delta:g} bps done", flush=True)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "development_only",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_sha256": code_sha(), "source_dataset_sha256": phase4["dataset"]["sha256"],
              "production_model_sha256": PRODUCT_SHA, "target_binary": meta["target_definition"],
              "features": FEATURES, "hyperparameters": PARAMS, "class_order": LABELS,
              "delta_zero_descriptive_only": True, "no_test_metrics": True,
              "fixed_threshold": "argmax P(DOWN),P(HOLD),P(UP); no confidence gate",
              "results": results, "walk_forward": walk}
    write_json(DEVELOPMENT, report)
    return report


def freeze_candidates(candidates: list[float], reason: str) -> dict:
    if not DEVELOPMENT.exists():
        raise ValueError("Run development before locking candidates")
    values = tuple(float(value) for value in candidates)
    if not 1 <= len(values) <= 2 or len(values) != len(set(values)) or any(value not in DELTAS_EXPERIMENT for value in values):
        raise ValueError("Lock one or two distinct development deltas")
    if not reason.strip():
        raise ValueError("Candidate-selection rationale is required")
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if dev["code_sha256"] != code_sha() or dev["production_model_sha256"] != sha(ROOT / "models/xgboost_v1.joblib"):
        raise ValueError("Development code or product model changed")
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "candidates_bps": list(values),
            "reason_from_train_validation_walk_forward": reason,
            "development_report_sha256": sha(DEVELOPMENT), "code_sha256": code_sha(),
            "test_metrics_seen_at_lock": False}
    if LOCK.exists():
        raise ValueError("Candidate lock already exists; do not overwrite after TEST")
    write_json(LOCK, lock)
    return lock


def block_intervals(y: np.ndarray, proba: np.ndarray, *, block: int = 60, repeats: int = 500) -> dict:
    if len(y) < block * 2:
        raise ValueError("Too few observations for temporal bootstrap")
    rng = np.random.default_rng(42)
    samples = {key: [] for key in ("directional_accuracy", "signal_coverage", "balanced_accuracy", "macro_f1")}
    for _ in range(repeats):
        starts = rng.integers(0, len(y), size=int(np.ceil(len(y) / block)))
        indices = np.concatenate([(start + np.arange(block)) % len(y) for start in starts])[:len(y)]
        if set(y[indices]) != {0, 1, 2}:
            continue
        result = multiclass_metrics(y[indices], proba[indices])
        for key in samples:
            if result[key] is not None:
                samples[key].append(result[key])
    return {"method": "circular block bootstrap; 60 source minutes/block; 500 resamples; seed 42; descriptive 95% interval",
            "metrics": {key: {"lower_95": float(np.quantile(values, .025)),
                              "upper_95": float(np.quantile(values, .975)), "resamples": len(values)}
                        for key, values in samples.items() if values}}


def final_test() -> dict:
    if not LOCK.exists() or not DEVELOPMENT.exists():
        raise ValueError("Development and candidate lock required before TEST")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if sha(DEVELOPMENT) != lock["development_report_sha256"] or code_sha() != lock["code_sha256"]:
        raise ValueError("Development specification changed after candidate lock")
    if sha(ROOT / "models/xgboost_v1.joblib") != PRODUCT_SHA:
        raise ValueError("Production model changed")
    _, split, meta, phase4 = load_frozen_cohort()
    results = {}
    MODELS.mkdir(parents=True, exist_ok=True)
    for delta in lock["candidates_bps"]:
        train_y = ternary_labels(split.train["future_return_bps"].to_numpy(), delta)
        test_y = ternary_labels(split.test["future_return_bps"].to_numpy(), delta)
        model = fit(split.train, delta)
        proba = predict(model, split.test)
        result = multiclass_metrics(test_y, proba)
        filename = f"xgboost_hold_delta_{str(delta).replace('.', 'p')}bp.joblib"
        path = MODELS / filename
        joblib.dump({"estimator": model, "features": FEATURES, "labels": LABELS,
                     "delta_bps": delta, "horizon": "1m", "experimental_only": True}, path, compress=3)
        metadata = {"model": filename, "model_sha256": sha(path), "production_model_sha256": PRODUCT_SHA,
                    "dataset_sha256": phase4["dataset"]["sha256"], "code_sha256": code_sha(),
                    "commit": dev["commit"], "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                    "train_range": meta["split"]["train"], "features": FEATURES,
                    "hyperparameters": PARAMS, "class_order": LABELS, "delta_bps": delta,
                    "selection_lock_sha256": sha(LOCK), "test_metrics": result}
        write_json(MODELS / filename.replace(".joblib", "_metadata.json"), metadata)
        results[str(delta)] = {"test": result, "test_baselines": baseline_metrics(test_y, train_y),
                               "uncertainty": block_intervals(test_y, proba),
                               "experimental_model": str(path.relative_to(ROOT)), "model_sha256": sha(path)}
        print(f"TEST delta {delta:g} bps done", flush=True)
    descriptive = analyze(include_test=True)
    write_json(DISTRIBUTION_REPORT, descriptive)
    DISTRIBUTION_REPORT.with_suffix(".md").write_text(markdown(descriptive), encoding="utf-8")
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "locked_test",
              "candidate_lock": lock, "candidate_lock_sha256": sha(LOCK),
              "development_report_sha256": sha(DEVELOPMENT), "code_sha256": code_sha(),
              "production_model_sha256_before": PRODUCT_SHA,
              "production_model_sha256_after": sha(ROOT / "models/xgboost_v1.joblib"),
              "source_dataset_sha256": phase4["dataset"]["sha256"],
              "historical_test_limit": "Original TEST was disclosed in prior phases; paired retrospective result is not a fresh untouched holdout.",
              "results": results, "live_usage": "None. LIVE is not read for fit, selection, tuning, or post-hoc scoring."}
    if report["production_model_sha256_after"] != PRODUCT_SHA:
        raise ValueError("Production model changed during experiment")
    write_json(FINAL, report)
    walk_report = {"generated_at_utc": report["generated_at_utc"], "development_report_sha256": sha(DEVELOPMENT),
                   "folds": dev["walk_forward"], "candidates_bps": lock["candidates_bps"]}
    write_json(WALK, walk_report)
    FINAL.with_suffix(".md").write_text(final_markdown(report, dev, descriptive), encoding="utf-8")
    WALK.with_suffix(".md").write_text(walk_markdown(walk_report), encoding="utf-8")
    return report


def final_markdown(report: dict, dev: dict, descriptive: dict) -> str:
    lines = ["# Experimento ternario HOLD — FASE 5", "",
             f"Candidatos congelados antes de TEST: {report['candidate_lock']['candidates_bps']} bps. Motivo: {report['candidate_lock']['reason_from_train_validation_walk_forward']}", "",
             "Modelo: mismas ocho features productivas, 1m, XGBoost multiclase sin pesos nuevos. Sólo objetivo y métrica de entrenamiento cambian por el target ternario.", "",
             "| δ bps | Split | Accuracy | Balanced Acc | Macro F1 | Coverage | Directional Acc | Precision UP | Precision DOWN |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for delta in report["candidate_lock"]["candidates_bps"]:
        key = str(delta)
        for label, score in (("VALIDATION", dev["results"][key]["validation"]), ("TEST", report["results"][key]["test"])):
            directional = f"{score['directional_accuracy']:.4f}" if score["directional_accuracy"] is not None else "N/A"
            lines.append(f"| {delta:g} | {label} | {score['accuracy']:.4f} | {score['balanced_accuracy']:.4f} | {score['macro_f1']:.4f} | {score['signal_coverage']:.2%} | {directional} | {score['per_class']['UP']['precision']:.4f} | {score['per_class']['DOWN']['precision']:.4f} |")
    lines += ["", "Cada matriz de confusión y métrica por clase está en JSON. La fila representa la predicción y la columna la clase real, en orden DOWN/HOLD/UP.",
              "", "Directional accuracy es acierto exacto entre predicciones UP/DOWN; una predicción direccional contra HOLD real cuenta como error.",
              "", "Los intervalos se calcularon con bloques temporales de 60 minutos. TEST se había divulgado en fases anteriores; no equivale a holdout nuevo.",
              "", "Los deltas son umbrales estadísticos de movimiento: no incorporan fees, spread, slippage, latencia ni ejecución. Ningún modelo se promovió a producción.", ""]
    return "\n".join(lines)


def walk_markdown(report: dict) -> str:
    lines = ["# Walk-forward HOLD — FASE 5", "",
             "Cuatro folds expansivos TRAIN+VALIDATION, purga de una fila en cada frontera; TEST excluido.", "",
             "| δ bps | Fold | Validación UTC | Balanced Acc | Macro F1 | Coverage | Directional Acc | Precision UP | Precision DOWN |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for delta, folds in report["folds"].items():
        for fold in folds:
            score = fold["metrics"]
            directional = f"{score['directional_accuracy']:.4f}" if score["directional_accuracy"] is not None else "N/A"
            lines.append(f"| {delta} | {fold['fold']} | {fold['validation_start']}–{fold['validation_end']} | {score['balanced_accuracy']:.4f} | {score['macro_f1']:.4f} | {score['signal_coverage']:.2%} | {directional} | {score['per_class']['UP']['precision']:.4f} | {score['per_class']['DOWN']['precision']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "freeze", "final"))
    parser.add_argument("--candidates", type=float, nargs="+")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.stage == "development":
        development()
    elif args.stage == "freeze":
        freeze_candidates(args.candidates or [], args.reason)
    else:
        final_test()
