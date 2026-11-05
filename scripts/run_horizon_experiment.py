"""Offline H1/H3/H5 horizon comparison with a pre-TEST candidate lock.

Development evaluates TRAIN/VALIDATION and common-cohort expanding folds.
Final TEST scores only horizons locked before TEST. No LIVE or production writes.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.analyze_hold_target import PRODUCT_SHA, ROOT, sha, write_json
from scripts.analyze_horizons import (HORIZONS, DEVELOPMENT_DESCRIPTION, REPORT, analyze,
                                      build_horizon_frames, common_frames, common_splits,
                                      markdown as descriptive_markdown, split_horizon)
from scripts.run_tfi_ablation import metrics as binary_metrics, reproduction_gate

DEVELOPMENT = ROOT / "reports/horizon_development_v1.json"
LOCK = ROOT / "reports/horizon_candidate_lock_v1.json"
WALK = ROOT / "reports/horizon_walk_forward_v1.json"
MODELS = ROOT / "models/experiments/horizon_v1"
FEATURES = list(FEATURE_COLUMNS)
EXPECTED_H1_VALIDATION_ACCURACY = 0.5339100880580875
EXPECTED_H1_TEST_ACCURACY = 0.5118900555898703


def code_sha() -> str:
    digest = hashlib.sha256()
    for relative in ("scripts/analyze_horizons.py", "scripts/run_horizon_experiment.py",
                     "alpha_engine/features/indicators.py", "alpha_engine/features/target.py",
                     "alpha_engine/features/split.py", "alpha_engine/ml/trainer.py"):
        digest.update(relative.encode() + b"\0" + (ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def input_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.loc[:, FEATURES]
    if list(x.columns) != FEATURES or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Horizon X contains missing, future, or reordered features")
    return x


def fit(frame: pd.DataFrame) -> XGBClassifier:
    y = frame["target"].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Binary horizon TRAIN lacks a class")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(input_features(frame), y)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Class 1 must be UP")
    return model


def probability(model: XGBClassifier, frame: pd.DataFrame) -> np.ndarray:
    p = np.asarray(model.predict_proba(input_features(frame))[:, 1], dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid H1/H3/H5 probability vector")
    return p


def score(frame: pd.DataFrame, p: np.ndarray) -> dict:
    y = frame["target"].to_numpy(dtype=int)
    result = binary_metrics(y, p)
    confidence = np.maximum(p, 1 - p)
    result["confidence"].update({"maximum": float(confidence.max()),
                                  "ge_55_count": int((confidence >= .55).sum()),
                                  "ge_55_fraction": float((confidence >= .55).mean()),
                                  "ge_60_fraction": float((confidence >= .60).mean())})
    result["accuracy_by_confidence"] = {
        str(th): {"n": int((confidence >= th).sum()),
                  "accuracy": float((((p > .5).astype(int) == y)[confidence >= th]).mean())
                  if (confidence >= th).any() else None}
        for th in (.50, .55, .60)}
    return result


def simple_baselines(train: pd.DataFrame, frame: pd.DataFrame) -> dict:
    y = frame["target"].to_numpy(dtype=int)
    majority = int(train["target"].mode().iloc[0])
    guesses = {"train_majority": np.full(len(y), majority, dtype=int),
               "always_down": np.zeros(len(y), dtype=int),
               "always_up": np.ones(len(y), dtype=int),
               "previous_direction": (frame["return_1m"].to_numpy(dtype=float) > 0).astype(int)}
    return {name: {"class": "UP" if name == "train_majority" and majority else
                     "DOWN" if name == "train_majority" else None,
                   "accuracy": float(accuracy_score(y, pred)),
                   "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                   "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
            for name, pred in guesses.items()}


def movement_error(frame: pd.DataFrame, p: np.ndarray) -> dict:
    absolute = np.abs(frame["future_return_bps"].to_numpy(dtype=float))
    correct = (p > .5) == frame["target"].to_numpy(dtype=int)
    rows = []
    for low, high in ((0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)):
        mask = (absolute >= low) & (absolute < high)
        rows.append({"lower_bps": low, "upper_bps": high if np.isfinite(high) else None,
                     "n": int(mask.sum()), "fraction": float(mask.mean()),
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return {"bins": rows, "descriptive_only": True}


def gain(model: XGBClassifier) -> dict:
    booster = model.get_booster()
    values = booster.get_score(importance_type="gain")
    counts = booster.get_score(importance_type="weight")
    return {"gain": {name: float(values.get(name, 0)) for name in FEATURES},
            "split_count": {name: int(counts.get(name, 0)) for name in FEATURES}}


def common_expanding_folds(frame: pd.DataFrame, count: int = 4, max_horizon: int = 5):
    if count < 2 or max_horizon not in HORIZONS or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Invalid common expanding-fold specification")
    initial = len(frame) // 2
    block = (len(frame) - initial) // count
    if initial < 100 or block < 30:
        raise ValueError("Insufficient temporal folds")
    for number in range(count):
        boundary = initial + number * block
        end = initial + (number + 1) * block if number < count - 1 else len(frame)
        train = frame.iloc[:boundary - max_horizon]
        validation = frame.iloc[boundary:end]
        if validation.index[0].value - train.index[-1].value <= max_horizon * 60_000_000_000:
            raise ValueError("Fold purge allows a future target into validation")
        yield number + 1, train, validation


def compare_bootstrap(a_y: np.ndarray, a_p: np.ndarray, b_y: np.ndarray, b_p: np.ndarray,
                      *, block: int = 60, repetitions: int = 500) -> dict:
    """Paired source-minute blocks; labels may differ by horizon."""
    if not len(a_y) == len(a_p) == len(b_y) == len(b_p) or len(a_y) < 2 * block or block < max(HORIZONS):
        raise ValueError("Misaligned or too-small horizon bootstrap")
    rng = np.random.default_rng(42)
    deltas = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}
    for _ in range(repetitions):
        starts = rng.integers(0, len(a_y), size=int(np.ceil(len(a_y) / block)))
        idx = np.concatenate([(s + np.arange(block)) % len(a_y) for s in starts])[:len(a_y)]
        ay, by = a_y[idx], b_y[idx]
        if len(np.unique(ay)) != 2 or len(np.unique(by)) != 2:
            continue
        ap, bp = a_p[idx], b_p[idx]
        a_pred, b_pred = ap > .5, bp > .5
        deltas["accuracy"].append(accuracy_score(by, b_pred) - accuracy_score(ay, a_pred))
        deltas["balanced_accuracy"].append(balanced_accuracy_score(by, b_pred) - balanced_accuracy_score(ay, a_pred))
        deltas["macro_f1"].append(f1_score(by, b_pred, average="macro") - f1_score(ay, a_pred, average="macro"))
        deltas["roc_auc"].append(roc_auc_score(by, bp) - roc_auc_score(ay, ap))
    return {"method": "paired circular source-minute block bootstrap; 60-minute blocks (>=H5); 500 resamples; seed 42; descriptive 95% CI",
            "difference": {name: {"lower_95": float(np.quantile(values, .025)),
                                  "upper_95": float(np.quantile(values, .975)),
                                  "fraction_le_zero": float(np.mean(np.asarray(values) <= 0)),
                                  "resamples": len(values)} for name, values in deltas.items()}}


def development() -> dict:
    description = analyze(include_test=False)
    write_json(DEVELOPMENT_DESCRIPTION, description)
    frames, _ = build_horizon_frames()
    common = common_frames(frames)
    splits = common_splits(common)
    natural_h1 = split_horizon(frames[1], 1)
    original = joblib.load(ROOT / "models/xgboost_v1.joblib")["estimator"]
    experimental_h1 = fit(natural_h1.train)
    h1_val = probability(experimental_h1, natural_h1.validation)
    gate = reproduction_gate(h1_val, probability(original, natural_h1.validation), label="H1 natural VALIDATION")
    if abs(score(natural_h1.validation, h1_val)["accuracy"] - EXPECTED_H1_VALIDATION_ACCURACY) > 1e-12:
        raise RuntimeError("H1 baseline metrics are not reproduced")

    models = {h: fit(splits[h].train) for h in HORIZONS}
    probabilities = {h: probability(models[h], splits[h].validation) for h in HORIZONS}
    validation = {f"H{h}": {"metrics": score(splits[h].validation, probabilities[h]),
                              "baselines": simple_baselines(splits[h].train, splits[h].validation),
                              "movement_error": movement_error(splits[h].validation, probabilities[h]),
                              "gain": gain(models[h])} for h in HORIZONS}
    folds = []
    dev_frames = {h: pd.concat((splits[h].train, splits[h].validation)).sort_index() for h in HORIZONS}
    for fold_number, train1, valid1 in common_expanding_folds(dev_frames[1]):
        item = {"fold": fold_number, "validation_start": valid1.index[0].isoformat(),
                "validation_end": valid1.index[-1].isoformat(), "purged_source_rows": 5,
                "horizons": {}}
        for h in HORIZONS:
            train = dev_frames[h].loc[train1.index]
            valid = dev_frames[h].loc[valid1.index]
            m = fit(train)
            item["horizons"][f"H{h}"] = {"metrics": score(valid, probability(m, valid)),
                                          "target_UP_fraction": float(valid["target"].mean()),
                                          "train_end": train.index[-1].isoformat()}
        folds.append(item)
        print(f"Common walk-forward fold {fold_number}/4 done", flush=True)
    uncertainty = {}
    a_y = splits[1].validation["target"].to_numpy(dtype=int)
    for h in (3, 5):
        uncertainty[f"H{h}_minus_H1"] = compare_bootstrap(a_y, probabilities[1],
            splits[h].validation["target"].to_numpy(dtype=int), probabilities[h])
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "development_only",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "code_sha256": code_sha(), "production_model_sha256": PRODUCT_SHA,
              "source_dataset_sha256": description["source_dataset_sha256"],
              "common_cohort": description["common_cohort"], "features": FEATURES,
              "hyperparameters": XGB_PARAMS, "threshold": "P(UP)>0.50",
              "h1_natural_validation_reproduction": gate,
              "validation": validation, "walk_forward": folds,
              "validation_uncertainty": uncertainty, "test_not_scored": True}
    write_json(DEVELOPMENT, report)
    return report


def freeze_candidates(candidates: list[int], reason: str) -> dict:
    if not DEVELOPMENT.exists() or not DEVELOPMENT_DESCRIPTION.exists():
        raise ValueError("Development must be complete before locking candidates")
    if LOCK.exists():
        raise ValueError("Candidate lock already exists; never replace after TEST")
    values = [int(value) for value in candidates]
    if not 1 <= len(values) <= 2 or len(values) != len(set(values)) or any(h not in HORIZONS for h in values):
        raise ValueError("Lock one or two distinct horizons")
    if not reason.strip():
        raise ValueError("Selection rationale from development is required")
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if dev["code_sha256"] != code_sha() or sha(ROOT / "models/xgboost_v1.joblib") != PRODUCT_SHA:
        raise ValueError("Development code or production model changed")
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "candidates_minutes": values,
            "reason_from_train_validation_walk_forward": reason,
            "development_report_sha256": sha(DEVELOPMENT), "code_sha256": code_sha(),
            "test_metrics_seen_at_lock": False}
    write_json(LOCK, lock)
    return lock


def save_experiment_model(model: XGBClassifier, horizon: int, development_report: dict,
                          dataset_sha: str, test_metrics: dict) -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    name = f"xgboost_h{horizon}_binary_experimental.joblib"
    path = MODELS / name
    joblib.dump({"estimator": model, "features": FEATURES, "horizon": f"{horizon}m",
                 "model_version": name.removesuffix(".joblib"), "experimental_only": True}, path, compress=3)
    metadata = {"model_sha256": sha(path), "production_model_sha256": PRODUCT_SHA,
                "dataset_sha256": dataset_sha, "code_sha256": code_sha(),
                "code_commit": development_report["commit"],
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "features": FEATURES, "horizon_minutes": horizon,
                "target": "UP iff future_close > source_close; tie DOWN",
                "hyperparameters": XGB_PARAMS, "split": "common H5 cohort; purge 5 source rows at each boundary",
                "candidate_lock_sha256": sha(LOCK), "test_metrics": test_metrics}
    write_json(MODELS / name.replace(".joblib", "_metadata.json"), metadata)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def final_test() -> dict:
    if not LOCK.exists() or not DEVELOPMENT.exists():
        raise ValueError("Candidate lock and development report required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    dev = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if sha(DEVELOPMENT) != lock["development_report_sha256"] or code_sha() != lock["code_sha256"]:
        raise ValueError("Development specification changed after lock")
    if sha(ROOT / "models/xgboost_v1.joblib") != PRODUCT_SHA:
        raise ValueError("Product model changed before final TEST")
    description = analyze(include_test=True)
    frames, _ = build_horizon_frames()
    splits = common_splits(common_frames(frames))
    results = {}
    for h in lock["candidates_minutes"]:
        model = fit(splits[h].train)
        p = probability(model, splits[h].test)
        scored = score(splits[h].test, p)
        results[f"H{h}"] = {"metrics": scored,
                             "baselines": simple_baselines(splits[h].train, splits[h].test),
                             "movement_error": movement_error(splits[h].test, p),
                             "experimental_artifact": save_experiment_model(model, h, dev,
                                description["horizons"][f"H{h}"]["natural_dataset"]["sha256"], scored)}
        print(f"Locked H{h} TEST scored", flush=True)
    natural_h1 = split_horizon(frames[1], 1)
    original = joblib.load(ROOT / "models/xgboost_v1.joblib")["estimator"]
    original_test = score(natural_h1.test, probability(original, natural_h1.test))
    if abs(original_test["accuracy"] - EXPECTED_H1_TEST_ACCURACY) > 1e-12:
        raise RuntimeError("Original H1 natural TEST reference differs")
    final = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
             "stage": "locked_test_only", "candidate_lock": lock,
             "development_report_sha256": sha(DEVELOPMENT), "code_sha256": code_sha(),
             "natural_H1_original_TEST_reference": original_test,
             "common_TEST_scored_horizons": list(results), "common_TEST_results": results,
             "production_model_sha256_after": sha(ROOT / "models/xgboost_v1.joblib"),
             "test_limitation": "Original H1 TEST was disclosed in earlier phases; H3/H5 TEST was not used for candidate selection in this phase.",
             "live_usage": "None for fitting, selection, or scoring. Public UI read-only Playwright check only."}
    if final["production_model_sha256_after"] != PRODUCT_SHA:
        raise RuntimeError("Product model changed during experiment")
    description["development"] = {"validation": dev["validation"], "walk_forward": dev["walk_forward"],
                                  "validation_uncertainty": dev["validation_uncertainty"],
                                  "h1_natural_validation_reproduction": dev["h1_natural_validation_reproduction"]}
    description["candidate_lock"] = lock
    description["final_test"] = final
    write_json(REPORT, description)
    REPORT.with_suffix(".md").write_text(report_markdown(description), encoding="utf-8")
    walk_report = {"generated_at_utc": final["generated_at_utc"], "common_cohort": description["common_cohort"],
                   "purge_rows": 5, "folds": dev["walk_forward"],
                   "candidate_lock": lock["candidates_minutes"]}
    write_json(WALK, walk_report)
    WALK.with_suffix(".md").write_text(walk_markdown(walk_report), encoding="utf-8")
    return description


def report_markdown(report: dict) -> str:
    lines = [descriptive_markdown(report), "## Comparación principal: COMMON COHORT VALIDATION", "",
             "Mismas ocho features y parámetros, mismo split de 1m, purga común de cinco filas. Sólo cambia el cierre usado por el target binario.", "",
             "| Métrica | H1 | H3 | H5 |", "|---|---:|---:|---:|"]
    scores = report["development"]["validation"]
    for name in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up", "average_precision_up",
                 "precision_up", "recall_up", "f1_up", "precision_down", "recall_down", "f1_down",
                 "macro_f1", "brier", "log_loss"):
        values = [scores[f"H{h}"]["metrics"][name] for h in HORIZONS]
        lines.append(f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    lines += ["", f"Candidatos bloqueados antes de TEST: {report['candidate_lock']['candidates_minutes']}. {report['candidate_lock']['reason_from_train_validation_walk_forward']}", "",
              "## TEST final sólo de candidatos bloqueados", "",
              "| Métrica | " + " | ".join(report["final_test"]["common_TEST_scored_horizons"]) + " |",
              "|---|" + "---:|" * len(report["final_test"]["common_TEST_scored_horizons"])]
    for name in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up", "macro_f1", "recall_up", "recall_down", "brier", "log_loss"):
        values = [report["final_test"]["common_TEST_results"][key]["metrics"][name]
                  for key in report["final_test"]["common_TEST_scored_horizons"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "El TEST original H1 ya había sido divulgado; no es un holdout nuevo. H3/H5 tienen etiquetas temporalmente solapadas.",
              "No se aplicó tuning, calibración, HOLD, TFI ni selección con LIVE. Los costes económicos documentados son supuestos estimados, no tarifas observadas.", ""]
    return "\n".join(lines)


def walk_markdown(report: dict) -> str:
    lines = ["# Walk-forward de horizontes — FASE 6", "",
             "Cuatro folds expansivos sobre la cohorte común TRAIN+VALIDATION. Cada frontera purga 5 filas fuente; TEST no participa.", "",
             "| Fold | Validación UTC | H1/H3/H5 ROC AUC | H1/H3/H5 BalAcc | H1/H3/H5 Macro F1 |", "|---:|---|---|---|---|"]
    for row in report["folds"]:
        def triple(metric: str) -> str:
            return " / ".join(f"{row['horizons'][f'H{h}']['metrics'][metric]:.4f}" for h in HORIZONS)
        lines.append(f"| {row['fold']} | {row['validation_start']}–{row['validation_end']} | {triple('roc_auc')} | {triple('balanced_accuracy')} | {triple('macro_f1')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "freeze", "final"))
    parser.add_argument("--candidates", type=int, nargs="+")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.stage == "development":
        development()
    elif args.stage == "freeze":
        freeze_candidates(args.candidates or [], args.reason)
    else:
        final_test()
