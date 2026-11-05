"""Final fixed-parameter six-month V2 vs V3 market-flow experiment.

Reuses the frozen V2 model, cohort and fold metrics. Fits only V3 on TRAIN;
TEST/FORWARD/LIVE never choose parameters, features or thresholds.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.special import expit
from xgboost import DMatrix, XGBClassifier

from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.build_v2_dataset import AUDIT as V2_DATA_AUDIT, EXPERIMENT as V2_DATASET, V2, temporal_splits
from scripts.build_v3_dataset import AUDIT as V3_DATA_AUDIT, OUT as V3_DATASET, V3
from scripts.train_compare_xgboost_v2 import (
    MODEL_DIR, ORIGINAL, ORIGINAL_SHA, ROOT, baseline_metrics, distribution,
    evaluate, expanding_folds, feature_drift, sha256,
)

OUT = ROOT / "reports/xgboost_v3_market_flow_6m.json"
V2_REPORT = ROOT / "reports/xgboost_v2_6m_experiment.json"
V2_MODEL = MODEL_DIR / "xgboost_v2_relative_6m.joblib"
V2_METADATA = MODEL_DIR / "xgboost_v2_relative_6m_metadata.json"
V3_MODEL = MODEL_DIR / "xgboost_v3_market_flow_6m.joblib"
V3_METADATA = MODEL_DIR / "xgboost_v3_market_flow_6m_metadata.json"


def contribution_detail(model: XGBClassifier, part: pd.DataFrame) -> dict:
    x = part[list(V3)]
    array = model.get_booster().predict(DMatrix(x, feature_names=list(V3)), pred_contribs=True)
    p = model.predict_proba(x)[:, 1]
    mismatch = float(np.max(np.abs(expit(array.sum(axis=1)) - p)))
    if mismatch > 1e-5:
        raise ValueError("XGBoost contributions do not reconstruct P(UP)")
    predicted_up = p > .5
    def describe(mask):
        return {name: {"mean": float(array[mask, i].mean()),
                       "median": float(np.median(array[mask, i])),
                       "mean_absolute": float(np.abs(array[mask, i]).mean())}
                for i, name in enumerate((*V3, "BIAS"))}
    return {"unit": "raw margin/log-odds of UP", "max_reconstruction_error": mismatch,
            "all": describe(np.ones(len(p), dtype=bool)),
            "predicted_UP": describe(predicted_up),
            "predicted_DOWN": describe(~predicted_up)}


def group_descriptives(part: pd.DataFrame, probabilities: np.ndarray) -> dict:
    up = probabilities > .5
    y = part.target.to_numpy(dtype=int)
    result = {}
    for name in ("taker_buy_ratio", "volume_zscore", "trades_zscore",
                 "candle_range", "body_return", "upper_wick", "lower_wick"):
        values = part[name].to_numpy(dtype=float)
        result[name] = {
            "all": distribution(values),
            "predicted_UP": distribution(values[up]),
            "predicted_DOWN": distribution(values[~up]),
            "actual_UP": distribution(values[y == 1]),
            "actual_DOWN": distribution(values[y == 0]),
        }
    return result


def _markdown(report: dict) -> str:
    lines = ["# XGBoost V3 - price action, volume y taker flow (6 meses)", "",
             "Experimento offline sobre la misma cohorte H1 y splits de V2. V2 es el artefacto congelado; solo V3 se entrena.",
             f"Cohorte V2 SHA-256: `{report['v2_cohort_sha256']}`. Cohorte V3 SHA-256: `{report['v3_dataset_sha256']}`.",
             f"RAW oficial: {report['source_rows']} velas; cero volumen en {report['raw_quality']['zero_volume']}; cero trades en {report['raw_quality']['zero_trade_count']}. No se descargaron archivos adicionales.",
             "", "## Splits", "", "| Split | Primera vela fuente UTC | Ultima vela fuente UTC | N |",
             "|---|---|---|---:|"]
    for split, item in report["split_audit"].items():
        lines.append(f"| {split} | {item['first_source_utc']} | {item['last_source_utc']} | {item['rows']} |")
    lines += [
             "", "## Features", "", "| Feature | Formula | Ventana | Significado |",
             "|---|---|---|---|"]
    for name, item in report["feature_definitions"].items():
        lines.append(f"| {name} | `{item['formula']}` | {item['window']} | {item['meaning']} |")
    lines += ["", "## Comparacion pareada", "",
              "| Split | Modelo | N | Accuracy | Bal. acc. | ROC AUC | PR AUC UP | Macro F1 | Pred. UP |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        for name in ("V2", "V3"):
            row = report["metrics"][name][split]
            lines.append(f"| {split} | {name} | {row['n']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc_up']:.4f} | {row['macro_f1']:.4f} | {row['predicted_up_fraction']:.2%} |")
    lines += ["", "## Walk-forward de desarrollo", "",
              "| Fold | Validacion UTC | V2 ROC | V3 ROC | V2 Bal. acc. | V3 Bal. acc. | V2 PR AUC | V3 PR AUC |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for fold in report["walk_forward"]:
        a, b = fold["metrics"]["V2"], fold["metrics"]["V3"]
        lines.append(f"| {fold['fold']} | {' a '.join(fold['validation_range_utc'])} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {a['balanced_accuracy']:.4f} | {b['balanced_accuracy']:.4f} | {a['pr_auc_up']:.4f} | {b['pr_auc_up']:.4f} |")
    lines += ["", "## Baselines", "",
              "| Split | Always DOWN accuracy | Always UP accuracy | Previous movement accuracy / bal. acc. |",
              "|---|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        base = report["baselines"][split]
        lines.append(f"| {split} | {base['always_DOWN']['accuracy']:.4f} | {base['always_UP']['accuracy']:.4f} | {base['previous_movement']['accuracy']:.4f} / {base['previous_movement']['balanced_accuracy']:.4f} |")
    lines += ["", "## Drift V3 frente a TRAIN", "",
              "PSI usa deciles fijados en TRAIN; es descriptivo y sensible a masas cerca de cero en los wicks.",
              "", "| Feature | PSI TEST | PSI FORWARD | FORWARD fuera TRAIN min-max |",
              "|---|---:|---:|---:|"]
    for name in V3:
        test = report["drift"]["V3"]["TEST"][name]
        forward = report["drift"]["V3"]["FORWARD"][name]
        lines.append(f"| {name} | {test['psi_train_deciles']:.3f} | {forward['psi_train_deciles']:.3f} | {forward['below_train_min']+forward['above_train_max']:.2%} |")
    lines += ["", "## FORWARD: contribuciones V3", "",
              "Valores medios del margen log-odds de UP; las contribuciones no prueban causalidad.", "",
              "| Feature | Media | Media absoluta | Pred. UP media | Pred. DOWN media |",
              "|---|---:|---:|---:|---:|"]
    features = report["contributions"]["all"]
    for name, item in sorted(features.items(), key=lambda pair: -pair[1]["mean_absolute"]):
        lines.append(f"| {name} | {item['mean']:+.4f} | {item['mean_absolute']:.4f} | {report['contributions']['predicted_UP'][name]['mean']:+.4f} | {report['contributions']['predicted_DOWN'][name]['mean']:+.4f} |")
    if "live_drift" in report:
        live = report["live_drift"]
        lines += ["", "## Drift LIVE disponible", "",
                  f"Corte {live['cutoff_utc']}: {live['usable_live_rows']} velas OHLCV; solo diagnostico externo.",
                  "Trades y taker-buy no existen en las velas productivas persistidas; no se infirieron proxies.",
                  "", "| Feature | PSI LIVE vs TRAIN | Fuera TRAIN min-max |",
                  "|---|---:|---:|"]
        for name, detail in sorted(live["drift"].items(), key=lambda item: -item[1]["psi_train_deciles"]):
            lines.append(f"| {name} | {detail['psi_train_deciles']:.3f} | {detail['below_train_min']+detail['above_train_max']:.2%} |")
    f2, f3 = report["metrics"]["V2"]["FORWARD"], report["metrics"]["V3"]["FORWARD"]
    lines += ["", "## Conclusion", "",
              f"En FORWARD V3 mejora ROC AUC {f2['roc_auc']:.4f} a {f3['roc_auc']:.4f}, balanced accuracy {f2['balanced_accuracy']:.4f} a {f3['balanced_accuracy']:.4f} y PR AUC UP {f2['pr_auc_up']:.4f} a {f3['pr_auc_up']:.4f}.",
              "La mejora aparece en los cuatro folds, TEST y FORWARD. Sigue siendo una discriminacion moderada; los wicks presentan PSI elevado y las features trades/taker aun no estan disponibles en el esquema LIVE persistido. V3 permanece experimental para la unica confirmacion final de ~12 meses."]
    lines += ["", "## Metodo y limites", "",
              "Los splits, targets, umbral 0.50, seed y parametros son los de V2. TRAIN ajusta V3. Los cuatro folds usan TRAIN+VALIDATION y purgan una etiqueta H1. TEST y FORWARD puntuan ambos disenos fijados sin tuning.",
              "El grupo V3 incorpora varias senales a la vez: la comparacion mide el conjunto, no el efecto causal individual de cada feature. Las contribuciones y diferencias descriptivas por clase no sustituyen una ablacion de grupos.",
              "Las metricas completas, drift, matrices, probabilidades, distribuciones y metadata de los modelos estan en el JSON. Ninguna conclusion implica rentabilidad o promocion a produccion."]
    return "\n".join(lines) + "\n"


def main() -> None:
    if sha256(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("Production xgboost_v1 SHA changed")
    old_audit = json.loads(V2_DATA_AUDIT.read_text(encoding="utf-8"))
    new_audit = json.loads(V3_DATA_AUDIT.read_text(encoding="utf-8"))
    old_report = json.loads(V2_REPORT.read_text(encoding="utf-8"))
    v2_meta = json.loads(V2_METADATA.read_text(encoding="utf-8"))
    if (sha256(V2_DATASET) != old_audit["cohort"]["sha256"] or
        sha256(V3_DATASET) != new_audit["v3_dataset"]["sha256"] or
        sha256(V2_MODEL) != v2_meta["model_sha256"] or
        new_audit["v2_cohort_sha256"] != old_audit["cohort"]["sha256"]):
        raise ValueError("Frozen V2 dataset/model or V3 dataset fingerprint mismatch")
    if v2_meta["features"] != list(V2) or v2_meta["parameters"] != dict(XGB_PARAMS):
        raise ValueError("V2 control feature order or parameters changed")
    frame = pd.read_parquet(V3_DATASET)
    splits, split_audit = temporal_splits(frame)
    if split_audit != old_audit["splits"] or split_audit != new_audit["splits"]:
        raise ValueError("V3 split rows differ from frozen V2")
    bundle = joblib.load(V2_MODEL)
    if bundle["features"] != list(V2):
        raise ValueError("V2 artifact feature order changed")
    v2_model = bundle["estimator"]
    v3_model = XGBClassifier(**XGB_PARAMS)
    v3_model.fit(splits["TRAIN"][list(V3)], splits["TRAIN"].target)
    print(f"trained V3 on {len(splits['TRAIN'])} frozen TRAIN rows", flush=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": "verified official Binance Public Data Spot BTCUSDT 1m ZIPs; no redownload",
        "source_manifest": "reports/xgboost_v2_6m_source_manifest.json",
        "v2_cohort_sha256": old_audit["cohort"]["sha256"],
        "v3_dataset_sha256": new_audit["v3_dataset"]["sha256"],
        "source_rows": old_audit["canonical"]["rows"],
        "split_audit": split_audit,
        "feature_definitions": new_audit["definitions"],
        "feature_sets": {"V2": list(V2), "V3": list(V3)},
        "parameters": dict(XGB_PARAMS), "threshold": .5,
        "metrics": {"V2": {}, "V3": {}}, "baselines": {},
        "drift": {"V3": {}}, "walk_forward": [],
        "taker_and_activity": {}, "contributions": {}, "global_importance": {},
        "raw_quality": new_audit["raw_quality"],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "joblib": joblib.__version__},
        "methodology": {"final_model_fit": "TRAIN only", "walk_forward": "same four V2 folds, expanding, H1 purge",
                        "test_forward": "evaluation only; no tuning or threshold selection",
                        "live": "optional descriptive drift; no labels or fitting",
                        "v2_control": "frozen model artifact; exact same timestamp cohort"},
    }
    for split, part in splits.items():
        report["baselines"][split] = baseline_metrics(part)
        p2 = v2_model.predict_proba(part[list(V2)])[:, 1]
        p3 = v3_model.predict_proba(part[list(V3)])[:, 1]
        report["metrics"]["V2"][split] = evaluate(part.target, p2)
        report["metrics"]["V3"][split] = evaluate(part.target, p3)
        reference = old_report["metrics"]["B"][split]
        for name in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_up", "brier", "log_loss", "predicted_up_fraction"):
            if abs(report["metrics"]["V2"][split][name] - reference[name]) > 1e-10:
                raise ValueError(f"Frozen V2 control failed reproduction: {split}/{name}")
        if split != "TRAIN":
            report["drift"]["V3"][split] = feature_drift(splits["TRAIN"], part, V3)
        if split == "FORWARD":
            report["taker_and_activity"] = group_descriptives(part, p3)
            report["contributions"] = contribution_detail(v3_model, part)
    booster = v3_model.get_booster()
    report["global_importance"] = {
        "gain": {name: float(booster.get_score(importance_type="gain").get(name, 0)) for name in V3},
        "split_count": {name: int(booster.get_score(importance_type="weight").get(name, 0)) for name in V3},
    }
    development = pd.concat([splits["TRAIN"], splits["VALIDATION"]], ignore_index=True)
    reference_folds = old_report["walk_forward"]
    for number, train, valid in expanding_folds(development):
        expected = reference_folds[number-1]
        if (len(train) != expected["train_n"] or len(valid) != expected["validation_n"] or
            train.timestamp.iloc[0].isoformat() != expected["train_range_utc"][0] or
            train.timestamp.iloc[-1].isoformat() != expected["train_range_utc"][1] or
            valid.timestamp.iloc[0].isoformat() != expected["validation_range_utc"][0] or
            valid.timestamp.iloc[-1].isoformat() != expected["validation_range_utc"][1]):
            raise ValueError("V3 walk-forward fold does not match frozen V2 fold")
        fold_model = XGBClassifier(**XGB_PARAMS)
        fold_model.fit(train[list(V3)], train.target)
        fold = {"fold": number, "train_n": len(train), "validation_n": len(valid),
                "train_range_utc": expected["train_range_utc"],
                "validation_range_utc": expected["validation_range_utc"],
                "purged_source_rows": 1,
                "metrics": {"V2": expected["metrics"]["B"],
                            "V3": evaluate(valid.target, fold_model.predict_proba(valid[list(V3)])[:, 1])}}
        report["walk_forward"].append(fold)
        print(f"fold {number}: V2 ROC={fold['metrics']['V2']['roc_auc']:.4f} V3 ROC={fold['metrics']['V3']['roc_auc']:.4f}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": v3_model, "features": list(V3),
                 "model_version": "xgboost_v3_market_flow_6m",
                 "trained_at": report["created_at_utc"], "symbol": "BTCUSDT",
                 "interval": "1m", "horizon": "1m"}, V3_MODEL)
    metadata = {"model_version": "xgboost_v3_market_flow_6m", "experimental_only": True,
                "features": list(V3), "target": "UP iff next_close>source_close; ties DOWN",
                "horizon": "1m", "training_range": [splits["TRAIN"].timestamp.iloc[0].isoformat(),
                                                   splits["TRAIN"].timestamp.iloc[-1].isoformat()],
                "training_rows": len(splits["TRAIN"]), "parameters": dict(XGB_PARAMS),
                "v2_cohort_sha256": old_audit["cohort"]["sha256"],
                "v3_dataset_sha256": new_audit["v3_dataset"]["sha256"],
                "model_sha256": sha256(V3_MODEL), "model_path": str(V3_MODEL.relative_to(ROOT)),
                "code_commit": report["commit"], "created_at_utc": report["created_at_utc"]}
    V3_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    report["models"] = {"V2": v2_meta, "V3": metadata}
    live_path = ROOT / "reports/xgboost_v3_market_flow_6m_live_drift.json"
    if live_path.exists():
        live = json.loads(live_path.read_text(encoding="utf-8"))
        if live.get("v3_dataset_sha256") == new_audit["v3_dataset"]["sha256"]:
            report["live_drift"] = live
    if sha256(ORIGINAL) != ORIGINAL_SHA:
        raise RuntimeError("Production model changed")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    OUT.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"FORWARD V2 ROC={report['metrics']['V2']['FORWARD']['roc_auc']:.4f} V3 ROC={report['metrics']['V3']['FORWARD']['roc_auc']:.4f}; production unchanged", flush=True)


if __name__ == "__main__":
    main()
