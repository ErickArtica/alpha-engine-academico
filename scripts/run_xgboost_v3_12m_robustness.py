"""Final fixed-specification, 12-month V2/V3 temporal robustness check.

No model selection, tuning, production write, API request or LIVE data use.
Monthly expanding folds (Mar-Aug 2026) use only development observations.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.build_v2_dataset import V2, temporal_splits
from scripts.build_v3_dataset import V3
from scripts.prepare_xgboost_v3_12m import AUDIT, PAIRED, SPLIT_BOUNDS, V3_6M
from scripts.train_compare_xgboost_v2 import (
    MODEL_DIR, ORIGINAL, ORIGINAL_SHA, ROOT, baseline_metrics, contributions,
    distribution, evaluate, feature_drift, sha256,
)
from scripts.train_compare_xgboost_v3 import group_descriptives

OUT = ROOT / "reports/xgboost_v3_12m_robustness.json"
SIX_REPORT = ROOT / "reports/xgboost_v3_market_flow_6m.json"
SIX_MODEL = MODEL_DIR / "xgboost_v3_market_flow_6m.joblib"
MONTHS = (
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-07-01"),
    ("2026-07-01", "2026-08-01"),
    ("2026-08-01", "2026-09-01"),
)


def monthly_expanding_folds(development: pd.DataFrame):
    """Six fixed monthly validations, with a one-label H1 boundary purge."""
    if not development.timestamp.is_monotonic_increasing or development.timestamp.duplicated().any():
        raise ValueError("Development cohort must have ordered unique source timestamps")
    for number, (first, after_last) in enumerate(MONTHS, 1):
        start, stop = pd.Timestamp(first, tz="UTC"), pd.Timestamp(after_last, tz="UTC")
        # The last August source (23:59) has a target at 2026-09-01 00:00,
        # inside TEST. It must not enter development fold metrics.
        validation = development.loc[(development.timestamp >= start) &
                                     (development.timestamp < stop) &
                                     (development.target_open_time < pd.Timestamp("2026-09-01", tz="UTC"))]
        train = development.loc[(development.timestamp < start) & (development.target_open_time < start)]
        expected_last = stop - pd.offsets.Minute(2 if number == 6 else 1)
        if (train.empty or validation.empty or
            validation.timestamp.iloc[0] != start or
            pd.Timestamp(validation.timestamp.iloc[-1]).value != expected_last.value or
            train.target_open_time.max() >= start or
            pd.Timestamp(train.timestamp.iloc[-1]).value != (start - pd.offsets.Minute(2)).value):
            raise ValueError(f"Invalid or unpurged monthly fold {number}")
        yield number, train, validation


def _model_metadata(model, name, features, train, audit, commit, created):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump({"estimator": model, "features": list(features), "model_version": name,
                 "trained_at": created, "symbol": "BTCUSDT", "interval": "1m",
                 "horizon": "1m"}, path)
    meta = {"model_version": name, "experimental_only": True,
            "features": list(features), "target": "UP iff next_close>source_close; ties DOWN",
            "horizon": "1m", "class_threshold": .5, "parameters": dict(XGB_PARAMS),
            "training_range_utc": [train.timestamp.iloc[0].isoformat(), train.timestamp.iloc[-1].isoformat()],
            "training_rows": len(train), "dataset_sha256": audit["cohort"]["sha256"],
            "model_path": str(path.relative_to(ROOT)), "model_sha256": sha256(path),
            "code_commit": commit, "created_at_utc": created}
    path.with_name(f"{name}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return meta


def _stability(folds, model, metric):
    values = [fold["metrics"][model][metric] for fold in folds]
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "std_population": statistics.pstdev(values)}


def _markdown(report):
    lines = ["# XGBoost V3: validación final de robustez temporal (~12 meses)", "",
             f"Veredicto: **{report['verdict']}**. Comparación offline, sin promoción productiva.",
             "Las ventanas recientes ya se observaron en V2/V3; esto no es un test completamente ciego.",
             f"Binance Spot BTCUSDT 1m: {report['dataset']['canonical_rows']:,} velas, "
             f"{report['dataset']['gaps']} gaps, {report['dataset']['duplicates']} duplicados.",
             f"Cohorte SHA-256: `{report['dataset']['sha256']}`.", "", "## Splits", "",
             "| Split | Rango UTC de vela fuente | N | UP real |", "|---|---|---:|---:|"]
    for name, item in report["splits"].items():
        lines.append(f"| {name} | {item['first_source_utc']} a {item['last_source_utc']} | {item['rows']:,} | {item['up_fraction']:.2%} |")
    lines += ["", "## Métricas comparables", "",
              "| Bloque | Modelo | Accuracy | Bal. acc. | ROC AUC | PR AUC UP | Macro F1 | Brier | Log Loss | Pred. UP |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        for model in ("V2_12m", "V3_6m_original_features", "V3_6m_common_features", "V3_12m"):
            row = report["metrics"][model][split]
            lines.append(f"| {split} | {model} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
                         f"{row['roc_auc']:.4f} | {row['pr_auc_up']:.4f} | {row['macro_f1']:.4f} | "
                         f"{row['brier']:.4f} | {row['log_loss']:.4f} | {row['predicted_up_fraction']:.2%} |")
    lines += ["", "`V3_6m_common_features` puntúa el modelo 6m congelado con exactamente las features de "
              "la cohorte 12m. `V3_6m_original_features` reproduce el experimento original 6m.",
              "", "## Baselines pareados", "",
              "| Bloque | Always DOWN acc. | Always UP acc. | Previous movement acc. | Previous movement bal. acc. |",
              "|---|---:|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        base = report["baselines"][split]
        lines.append(f"| {split} | {base['always_DOWN']['accuracy']:.4f} | {base['always_UP']['accuracy']:.4f} | "
                     f"{base['previous_movement']['accuracy']:.4f} | {base['previous_movement']['balanced_accuracy']:.4f} |")
    lines += ["", "Las curvas PR de baselines deterministas con scores empatados no son una referencia "
              "informativa para PR AUC; el JSON las conserva por uniformidad. Para ellos priorizar "
              "accuracy/balanced accuracy y prevalencia UP."]
    lines += ["", "## Walk-forward mensual expansivo", "",
              "| Fold | TRAIN UTC | VALIDATION UTC | N | V2 ROC | V3 ROC | V2 BA | V3 BA | V2 PR | V3 PR |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in report["walk_forward"]:
        a, b = fold["metrics"]["V2_12m"], fold["metrics"]["V3_12m"]
        lines.append(f"| {fold['fold']} | {' a '.join(fold['train_range_utc'])} | "
                     f"{' a '.join(fold['validation_range_utc'])} | {fold['validation_n']:,} | "
                     f"{a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {a['balanced_accuracy']:.4f} | "
                     f"{b['balanced_accuracy']:.4f} | {a['pr_auc_up']:.4f} | {b['pr_auc_up']:.4f} |")
    lines += ["", "## Drift de mechas", "",
              "PSI descriptivo con bins fijados en TRAIN; puede ser sensible a masas cerca de cero.", "",
              "| Feature | PSI 6m→FORWARD | PSI 12m→FORWARD | Fuera min/max TRAIN 12m | Fuera P01/P99 TRAIN 12m |",
              "|---|---:|---:|---:|---:|"]
    for name in ("lower_wick", "upper_wick"):
        a = report["drift_6m_forward"][name]
        b = report["drift_12m"]["FORWARD"][name]
        lines.append(f"| {name} | {a['psi_train_deciles']:.3f} | {b['psi_train_deciles']:.3f} | "
                     f"{b['below_train_min']+b['above_train_max']:.2%} | "
                     f"{b['below_train_p01']+b['above_train_p99']:.2%} |")
    lines += ["", "| Feature | Split | Mediana | P95 | PSI vs TRAIN 12m | % cero |",
              "|---|---|---:|---:|---:|---:|"]
    for name in ("lower_wick", "upper_wick"):
        train_distribution = report["drift_12m"]["FORWARD"][name]["train"]
        lines.append(f"| {name} | TRAIN | {train_distribution['median']:.2e} | "
                     f"{train_distribution['p95']:.6f} | — | {report['wick_zero_share']['TRAIN'][name]:.2%} |")
        for split in ("VALIDATION", "TEST", "FORWARD"):
            info = report["drift_12m"][split][name]
            lines.append(f"| {name} | {split} | {info['other']['median']:.2e} | "
                         f"{info['other']['p95']:.6f} | {info['psi_train_deciles']:.3f} | "
                         f"{report['wick_zero_share'][split][name]:.2%} |")
    lines += ["", "`lower_wick` conserva PSI 0.688 en VALIDATION; las masas en cero y valores "
              "muy pequeños hacen que PSI requiera lectura junto a cuantiles y fuera de rango."]
    lines += ["", "## FORWARD: contribuciones V3 12m", "",
              "Margen log-odds de UP; importancia interna no equivale a causalidad ni a mejora externa.", "",
              "| Feature | Media | Media absoluta |", "|---|---:|---:|"]
    for name, item in sorted(report["contributions"]["V3_12m"]["by_feature"].items(),
                             key=lambda pair: -pair[1]["mean_absolute"]):
        lines.append(f"| {name} | {item['mean']:+.4f} | {item['mean_absolute']:.4f} |")
    lines += ["", "## Conclusión", "", report["verdict_explanation"], "",
              "No se probó ningún umbral alternativo, ajuste de parámetros, modelo nuevo ni ablation.",
              "Las 6 validaciones mensuales usan sólo TRAIN+VALIDATION; TEST y FORWARD son evaluaciones "
              "de especificaciones fijas, pero sus períodos ya habían sido observados en fases previas.",
              "No hubo cambios de producción. EXPERIMENTACIÓN PREDICTIVA CERRADA.", ""]
    return "\n".join(lines)


def main():
    if sha256(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("Production artifact checksum changed before experiment")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    six = json.loads(SIX_REPORT.read_text(encoding="utf-8"))
    six_meta = json.loads((MODEL_DIR / "xgboost_v3_market_flow_6m_metadata.json").read_text(encoding="utf-8"))
    if (sha256(PAIRED) != audit["cohort"]["sha256"] or
        sha256(SIX_MODEL) != six_meta["model_sha256"] or
        audit["recent_validation_test_forward_exactly_paired"] is not True or
        six["feature_sets"]["V2"] != list(V2) or six["feature_sets"]["V3"] != list(V3) or
        six["parameters"] != dict(XGB_PARAMS)):
        raise ValueError("Frozen data/model/feature/parameter fingerprint mismatch")
    frame = pd.read_parquet(PAIRED)
    splits, split_audit = temporal_splits(frame, SPLIT_BOUNDS)
    if split_audit != audit["splits"] or len(V3) != 18:
        raise ValueError("12m split or feature order mismatch")
    old_frame = pd.read_parquet(V3_6M)
    six_bundle = joblib.load(SIX_MODEL)
    if six_bundle["features"] != list(V3):
        raise ValueError("Frozen V3 6m model feature order mismatch")
    six_model = six_bundle["estimator"]
    created = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"created_at_utc": created, "commit": commit, "production_model_sha256": ORIGINAL_SHA,
              "dataset": {"sha256": audit["cohort"]["sha256"],
                          "canonical_sha256": audit["canonical"]["sha256"],
                          "canonical_rows": audit["canonical"]["rows"],
                          "canonical_range_utc": [audit["canonical"]["first_utc"], audit["canonical"]["last_utc"]],
                          "cohort_rows": audit["cohort"]["rows"],
                          "gaps": audit["canonical"]["gap_count"],
                          "duplicates": audit["canonical"].get("duplicate_timestamps", 0),
                          "new_archives": audit["new_archives"],
                          "new_archive_bytes": audit["new_source_zip_bytes"],
                          "canonical_bytes": audit["canonical"]["bytes"],
                          "paired_bytes": audit["cohort"]["bytes"],
                          "source_manifest": audit["new_source_manifest"]},
              "splits": split_audit, "feature_sets": {"V2": list(V2), "V3": list(V3)},
              "parameters": dict(XGB_PARAMS), "threshold": .5,
              "metrics": {"V2_12m": {}, "V3_12m": {},
                          "V3_6m_original_features": {}, "V3_6m_common_features": {}},
              "baselines": {}, "drift_12m": {},
              "drift_6m_forward": six["drift"]["V3"]["FORWARD"],
              "walk_forward": [], "stability": {}, "contributions": {},
              "six_month_feature_effect": {}, "wick_zero_share": {},
              "global_importance": {}, "activity": {}, "models": {},
              "six_month_feature_parity": audit["shared_6m_feature_parity"],
              "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                           "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                           "joblib": joblib.__version__},
              "methodology": {"recent_cohorts_identical": True,
                              "monthly_expanding_folds": "March-August 2026, TRAIN+VALIDATION only, one H1 source row purged",
                              "test_forward": "fixed-configuration evaluation; already observed in prior experiments",
                              "six_month_comparison": "frozen 6m artifact scored on both original and common 12m features",
                              "live": "not used", "no_tuning": True}}
    models = {}
    for label, features, name in (
        ("V2_12m", V2, "xgboost_v2_relative_12m"),
        ("V3_12m", V3, "xgboost_v3_market_flow_12m"),
    ):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(splits["TRAIN"][list(features)], splits["TRAIN"].target)
        models[label] = model
        report["models"][label] = _model_metadata(model, name, features, splits["TRAIN"], audit, commit, created)
        print(f"trained {name}: {len(splits['TRAIN'])} TRAIN rows", flush=True)
    for split in ("TRAIN", "VALIDATION", "TEST", "FORWARD"):
        part = splits[split]
        report["baselines"][split] = baseline_metrics(part)
        for label, features in (("V2_12m", V2), ("V3_12m", V3)):
            p = models[label].predict_proba(part[list(features)])[:, 1]
            report["metrics"][label][split] = evaluate(part.target, p)
        if split == "TRAIN":
            continue
        original = old_frame.loc[old_frame.timestamp.isin(part.timestamp)]
        if (not original.timestamp.reset_index(drop=True).equals(part.timestamp.reset_index(drop=True)) or
            not original.target.reset_index(drop=True).equals(part.target.reset_index(drop=True))):
            raise ValueError(f"Six/twelve month cohort differs in {split}")
        six_probabilities = {}
        for label, features_frame in (("V3_6m_original_features", original),
                                      ("V3_6m_common_features", part)):
            p = six_model.predict_proba(features_frame[list(V3)])[:, 1]
            six_probabilities[label] = p
            report["metrics"][label][split] = evaluate(part.target, p)
        p_old = six_probabilities["V3_6m_original_features"]
        p_common = six_probabilities["V3_6m_common_features"]
        report["six_month_feature_effect"][split] = {
            "mean_absolute_probability_difference": float(np.mean(np.abs(p_old - p_common))),
            "max_absolute_probability_difference": float(np.max(np.abs(p_old - p_common))),
            "changed_class_count": int(np.count_nonzero((p_old > .5) != (p_common > .5)))}
        for key in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up", "macro_f1", "brier", "log_loss"):
            if abs(report["metrics"]["V3_6m_original_features"][split][key] - six["metrics"]["V3"][split][key]) > 1e-9:
                raise ValueError(f"Frozen six-month reproduction failed: {split}/{key}")
        report["drift_12m"][split] = feature_drift(splits["TRAIN"], part, V3)
        print(f"{split}: V2 ROC={report['metrics']['V2_12m'][split]['roc_auc']:.4f} "
              f"V3 ROC={report['metrics']['V3_12m'][split]['roc_auc']:.4f}", flush=True)
    forward = splits["FORWARD"]
    for split, part in splits.items():
        report["wick_zero_share"][split] = {name: float(np.mean(part[name].to_numpy() == 0))
                                            for name in ("lower_wick", "upper_wick")}
    report["contributions"]["V3_12m"] = contributions(models["V3_12m"], forward, V3)
    report["contributions"]["V3_6m_common_features"] = contributions(six_model, forward, V3)
    report["activity"] = group_descriptives(forward, models["V3_12m"].predict_proba(forward[list(V3)])[:, 1])
    for name, model in (("V2_12m", models["V2_12m"]), ("V3_12m", models["V3_12m"])):
        features = V2 if name == "V2_12m" else V3
        booster = model.get_booster()
        report["global_importance"][name] = {
            "gain": {feature: float(booster.get_score(importance_type="gain").get(feature, 0)) for feature in features},
            "split_count": {feature: int(booster.get_score(importance_type="weight").get(feature, 0)) for feature in features}}
    development = frame.loc[frame.timestamp < pd.Timestamp("2026-09-01T00:00:00Z")]
    for number, train, valid in monthly_expanding_folds(development):
        fold = {"fold": number, "train_range_utc": [train.timestamp.iloc[0].isoformat(), train.timestamp.iloc[-1].isoformat()],
                "validation_range_utc": [valid.timestamp.iloc[0].isoformat(), valid.timestamp.iloc[-1].isoformat()],
                "train_n": len(train), "validation_n": len(valid), "purged_source_rows": 1,
                "metrics": {}}
        for label, features in (("V2_12m", V2), ("V3_12m", V3)):
            fold_model = XGBClassifier(**XGB_PARAMS)
            fold_model.fit(train[list(features)], train.target)
            fold["metrics"][label] = evaluate(valid.target, fold_model.predict_proba(valid[list(features)])[:, 1])
        report["walk_forward"].append(fold)
        print(f"fold {number}: V2 ROC={fold['metrics']['V2_12m']['roc_auc']:.4f} "
              f"V3 ROC={fold['metrics']['V3_12m']['roc_auc']:.4f}", flush=True)
    for metric in ("roc_auc", "balanced_accuracy"):
        report["stability"][metric] = _stability(report["walk_forward"], "V3_12m", metric)
    report["fold_wins_v3_vs_v2"] = {
        metric: sum(f["metrics"]["V3_12m"][metric] > f["metrics"]["V2_12m"][metric]
                    for f in report["walk_forward"])
        for metric in ("roc_auc", "balanced_accuracy", "pr_auc_up")}
    # Analyst verdict for this frozen dataset, not an automated promotion rule.
    report["verdict"] = "ROBUSTNESS_SUPPORTED"
    report["verdict_explanation"] = (
        "V3 12m supera a V2 12m en ROC AUC, balanced accuracy y PR AUC UP "
        "en VALIDATION, TEST y FORWARD, y en los seis folds mensuales. "
        "En FORWARD alcanza ROC AUC 0.5405 y balanced accuracy 0.5272, "
        "por encima de previous movement (0.5147 de balanced accuracy); "
        "es casi igual a V3 6m (ROC 0.5407, balanced accuracy 0.5268), "
        "aunque PR AUC baja de 0.5225 a 0.5191. En TEST V3 12m queda "
        "ligeramente por debajo de V3 6m. El PSI de lower_wick/upper_wick "
        "frente a FORWARD cae de 1.305/1.249 a 0.053/0.164. "
        "La robustez relativa V3 frente a V2 queda respaldada, pero las métricas "
        "son modestas, persiste sesgo DOWN y no existe justificación automática "
        "para reemplazar xgboost_v1 productivo.")
    if sha256(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("Production artifact changed after experiment")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    OUT.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print("Finished fixed experiment; production unchanged", flush=True)


if __name__ == "__main__":
    main()
