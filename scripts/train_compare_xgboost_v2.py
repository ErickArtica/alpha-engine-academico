"""Fixed-parameter, paired offline XGBoost A/B on the enlarged BTCUSDT history.

TRAIN alone fits the two final models. VALIDATION, TEST and FORWARD are scored
without refitting. Four expanding development folds use TRAIN+VALIDATION only.
No LIVE labels, tuning, threshold search or production model writes occur.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score, auc,
)
from xgboost import DMatrix, XGBClassifier

from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.analyze_feature_parity import population_stability_index
from scripts.build_v2_dataset import AUDIT, EXPERIMENT, V1, V2, temporal_splits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/xgboost_v2_6m_experiment.json"
MODEL_DIR = ROOT / "models/experiments"
ORIGINAL = ROOT / "models/xgboost_v1.joblib"
ORIGINAL_SHA = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
NAMES = {"A": "xgboost_v1_retrained_6m", "B": "xgboost_v2_relative_6m"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values) -> dict:
    array = np.asarray(values,dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Finite nonempty distribution required")
    return {"n":int(len(array)),"min":float(array.min()),"p01":float(np.quantile(array,.01)),
            "p05":float(np.quantile(array,.05)),"p10":float(np.quantile(array,.10)),
            "p25":float(np.quantile(array,.25)),"median":float(np.median(array)),
            "mean":float(array.mean()),"std":float(array.std()),
            "p75":float(np.quantile(array,.75)),"p90":float(np.quantile(array,.90)),
            "p95":float(np.quantile(array,.95)),"p99":float(np.quantile(array,.99)),
            "max":float(array.max())}


def evaluate(y, probability) -> dict:
    y = np.asarray(y,dtype=int)
    p = np.asarray(probability,dtype=float)
    if len(y)!=len(p) or not len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Aligned binary labels and probabilities required")
    predicted = (p>.5).astype(int)
    precision,recall,f1,support = precision_recall_fscore_support(
        y,predicted,labels=[0,1],zero_division=0)
    raw_matrix=confusion_matrix(y,predicted,labels=[0,1])
    # sklearn rows are actual; flip them for this public predicted-row convention.
    matrix=raw_matrix.T
    report={"n":int(len(y)),"accuracy":float(accuracy_score(y,predicted)),
            "balanced_accuracy":float(balanced_accuracy_score(y,predicted)) if len(set(y))==2 else None,
            "macro_f1":float(f1_score(y,predicted,average="macro",labels=[0,1],zero_division=0)),
            "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
            "predicted_up_fraction":float(predicted.mean()),
            "predicted_down_fraction":float(1-predicted.mean()),
            "actual_up_fraction":float(y.mean()),"actual_down_fraction":float(1-y.mean()),
            "confusion_predicted_rows_actual_columns":{
                "DOWN":{"DOWN":int(matrix[0,0]),"UP":int(matrix[0,1])},
                "UP":{"DOWN":int(matrix[1,0]),"UP":int(matrix[1,1])}},
            "classes":{name:{"precision":float(precision[i]),"recall":float(recall[i]),
                             "f1":float(f1[i]),"support":int(support[i])}
                       for i,name in enumerate(("DOWN","UP"))},
            "probability_up":distribution(p),
            "near_half":{"within_1pp":float(np.mean(np.abs(p-.5)<.01)),
                         "within_2pp":float(np.mean(np.abs(p-.5)<.02)),
                         "within_5pp":float(np.mean(np.abs(p-.5)<.05))}}
    if len(set(y))==2:
        pr,rc,_=precision_recall_curve(y,p)
        report.update({"roc_auc":float(roc_auc_score(y,p)),
                       "pr_auc_up":float(auc(rc,pr)),
                       "average_precision_up":float(average_precision_score(y,p))})
    else:
        report.update({"roc_auc":None,"pr_auc_up":None,"average_precision_up":None})
    return report


def baseline_metrics(part: pd.DataFrame) -> dict:
    y=part.target.to_numpy(dtype=int)
    previous=(part.close.to_numpy(dtype=float)>part.previous_close.to_numpy(dtype=float)).astype(int)
    if not np.isfinite(part.previous_close.to_numpy(dtype=float)).all():
        raise ValueError("Previous movement baseline lacks a causal close")
    return {name:evaluate(y,p) for name,p in (
        ("always_DOWN",np.zeros(len(y))),
        ("always_UP",np.ones(len(y))),
        ("previous_movement",previous.astype(float)))}


def feature_drift(train: pd.DataFrame, other: pd.DataFrame, features: tuple[str,...]) -> dict:
    result={}
    for name in features:
        a=train[name].to_numpy(dtype=float)
        b=other[name].to_numpy(dtype=float)
        left,right=distribution(a),distribution(b)
        result[name]={"train":left,"other":right,
                      "psi_train_deciles":population_stability_index(a,b),
                      "below_train_min":float(np.mean(b<left["min"])),
                      "above_train_max":float(np.mean(b>left["max"])),
                      "below_train_p01":float(np.mean(b<left["p01"])),
                      "above_train_p99":float(np.mean(b>left["p99"]))}
    return result


def expanding_folds(development: pd.DataFrame, n_folds: int=4):
    if n_folds<4 or len(development)<1000:
        raise ValueError("At least four reasonably populated folds required")
    first=int(len(development)*.5)
    edges=np.linspace(first,len(development),n_folds+1,dtype=int)
    for i in range(n_folds):
        train=development.iloc[:edges[i]-1]
        valid=development.iloc[edges[i]:edges[i+1]]
        if train.empty or valid.empty or train.target_open_time.max()>=valid.timestamp.min():
            raise ValueError("Walk-forward target leakage")
        yield i+1,train,valid


def contributions(model:XGBClassifier,part:pd.DataFrame,features:tuple[str,...]) -> dict:
    values=part[list(features)]
    array=model.get_booster().predict(DMatrix(values,feature_names=list(features)),pred_contribs=True)
    probability=model.predict_proba(values)[:,1]
    mismatch=float(np.max(np.abs(expit(array.sum(axis=1))-probability)))
    if mismatch>1e-5:
        raise ValueError("Native XGBoost contributions do not reconstruct P(UP)")
    return {"unit":"XGBoost raw margin/log-odds of UP",
            "max_probability_reconstruction_error":mismatch,
            "by_feature":{name:{"mean":float(array[:,i].mean()),
                                "median":float(np.median(array[:,i])),
                                "mean_absolute":float(np.abs(array[:,i]).mean())}
                          for i,name in enumerate((*features,"BIAS"))}}


def _model_output(model,features,name,train,audit):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    path=MODEL_DIR/f"{name}.joblib"
    bundle={"estimator":model,"features":list(features),"model_version":name,
            "trained_at":datetime.now(timezone.utc).isoformat(),"symbol":"BTCUSDT",
            "interval":"1m","horizon":"1m"}
    joblib.dump(bundle,path)
    metadata={"name":name,"experimental_only":True,"features":list(features),
              "target":"UP iff next_close>source_close; ties DOWN", "horizon":"1m",
              "training_range":[train.timestamp.iloc[0].isoformat(),train.timestamp.iloc[-1].isoformat()],
              "training_rows":len(train),"parameters":dict(XGB_PARAMS),
              "dataset_sha256":audit["cohort"]["sha256"],"model_sha256":sha256(path),
              "model_path":str(path.relative_to(ROOT))}
    (MODEL_DIR/f"{name}_metadata.json").write_text(
        json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    return metadata


def _markdown(report):
    lines = ["# XGBoost V2 - historico ampliado y features relativas", "",
             "Experimento offline: dos modelos, hiperparametros fijos y filas pareadas. Produccion intacta.",
             f"Fuente: {report['source']}. Cohorte SHA-256: `{report['dataset_sha256']}`.",
             f"TRAIN {report['splits']['TRAIN']['rows']}; VALIDATION {report['splits']['VALIDATION']['rows']}; TEST {report['splits']['TEST']['rows']}; FORWARD {report['splits']['FORWARD']['rows']}.",
             "", "| Split | Modelo | N | Accuracy | Bal. acc. | ROC AUC | PR AUC UP | Macro F1 | Pred. UP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        for name in ("A", "B"):
            row = report["metrics"][name][split]
            lines.append(f"| {split} | {NAMES[name]} | {row['n']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc_up']:.4f} | {row['macro_f1']:.4f} | {row['predicted_up_fraction']:.2%} |")
    lines += ["", "## Walk-forward de desarrollo", "",
              "| Fold | Periodo validacion | A ROC AUC | B ROC AUC | A Bal. acc. | B Bal. acc. |",
              "|---:|---|---:|---:|---:|---:|"]
    for fold in report["walk_forward"]:
        a, b = fold["metrics"]["A"], fold["metrics"]["B"]
        period = " a ".join(fold["validation_range_utc"])
        lines.append(f"| {fold['fold']} | {period} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {a['balanced_accuracy']:.4f} | {b['balanced_accuracy']:.4f} |")
    lines += ["", "## Drift de EMA y familia MACD", "",
              "| Par | FORWARD fuera de TRAIN min-max A | B | PSI A | B |",
              "|---|---:|---:|---:|---:|"]
    pairs = (("ema_20", "ema_distance"), ("macd", "macd_relative"),
             ("macd_signal", "macd_signal_relative"),
             ("macd_histogram", "macd_histogram_relative"))
    for old, new in pairs:
        a = report["drift"]["A"]["FORWARD"][old]
        b = report["drift"]["B"]["FORWARD"][new]
        lines.append(f"| {old} / {new} | {a['below_train_min']+a['above_train_max']:.2%} | {b['below_train_min']+b['above_train_max']:.2%} | {a['psi_train_deciles']:.3f} | {b['psi_train_deciles']:.3f} |")
    if "live_drift" in report:
        live = report["live_drift"]
        lines += ["", "## Drift LIVE (solo diagnostico externo)", "",
                  f"Corte {live['cutoff_utc']}; {live['usable_live_rows']} velas con replay causal de 120 velas. Ninguna etiqueta LIVE se uso para ajuste o seleccion.",
                  "", "| Par | LIVE fuera de TRAIN min-max A | B | PSI A | B |",
                  "|---|---:|---:|---:|---:|---:|"]
        for old, new in pairs:
            a, b = live["drift"]["A"][old], live["drift"]["B"][new]
            lines.append(f"| {old} / {new} | {a['below_train_min']+a['above_train_max']:.2%} | {b['below_train_min']+b['above_train_max']:.2%} | {a['psi_train_deciles']:.3f} | {b['psi_train_deciles']:.3f} |")
    lines += ["", "## Baselines y distribucion de probabilidades", "",
              "| Split | Always DOWN acc. | Always UP acc. | Previous movement acc. | A median P(UP) | B median P(UP) |",
              "|---|---:|---:|---:|---:|---:|"]
    for split in ("VALIDATION", "TEST", "FORWARD"):
        base = report["baselines"][split]
        a = report["metrics"]["A"][split]["probability_up"]
        b = report["metrics"]["B"][split]["probability_up"]
        lines.append(f"| {split} | {base['always_DOWN']['accuracy']:.4f} | {base['always_UP']['accuracy']:.4f} | {base['previous_movement']['accuracy']:.4f} | {a['median']:.4f} | {b['median']:.4f} |")
    lines += ["", "## FORWARD: clases y contribuciones", "",
              "Matriz con filas de prediccion y columnas de realidad:", "",
              "| Modelo/fila | Real DOWN | Real UP |", "|---|---:|---:|"]
    for model_name in ("A", "B"):
        matrix = report["metrics"][model_name]["FORWARD"]["confusion_predicted_rows_actual_columns"]
        for predicted in ("DOWN", "UP"):
            lines.append(f"| {model_name} pred. {predicted} | {matrix[predicted]['DOWN']} | {matrix[predicted]['UP']} |")
    lines += ["", "Contribuciones XGBoost en margen log-odds de UP (media en FORWARD):", "",
              "| Modelo | Feature | Media | Media absoluta |", "|---|---|---:|---:|"]
    for model_name in ("A", "B"):
        values = report["contributions"][model_name]["by_feature"]
        for feature, detail in sorted(values.items(), key=lambda item: -item[1]["mean_absolute"])[:5]:
            lines.append(f"| {model_name} | {feature} | {detail['mean']:+.4f} | {detail['mean_absolute']:.4f} |")
    reference = report["strictly_post_original_reference"]
    lines += ["", "## Referencia productiva en la misma porcion posterior", "",
              f"{reference['rows']} filas estrictamente posteriores al historico usado por xgboost_v1 original.", "",
              "| Modelo | Accuracy | Bal. acc. | ROC AUC |", "|---|---:|---:|---:|"]
    for name in ("original_xgboost_v1", "A", "B"):
        row = reference[name]
        lines.append(f"| {name} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "## Conclusion experimental", "",
              "La normalizacion relativa reduce mucho el drift de EMA; no mejora de forma consistente ROC AUC ni balanced accuracy. V2 gana ROC AUC en dos de cuatro folds, pero pierde en FORWARD. Ambos mantienen mayoria de predicciones DOWN. No hay evidencia suficiente para promover V2."]
    lines += ["", "## Metodo y limites", "",
              "TRAIN ajusta A/B. Cuatro folds se entrenan solo dentro de TRAIN+VALIDATION, con purga de una etiqueta H1 antes de cada bloque. TEST y FORWARD no intervienen en ajuste ni seleccion.",
              "La comparacion A/B aisla la representacion de features en el mismo historico. Comparar A con xgboost_v1 original no aisla rigurosamente mas datos: cambia tambien el periodo de entrenamiento. El original se puntua solo en FORWARD posterior a sus datos historicos.",
              "Menor drift o clases predichas mas equilibradas no prueba mejor prediccion. No se exploraron umbrales ni hiperparametros. Distribuciones, baselines, matrices y contribuciones estan en el JSON.",
              "", "## Reproduccion", "",
              "Ejecutar, en orden: `python -m scripts.download_binance_btcusdt_1m`, `python -m scripts.build_v2_dataset`, `python -m scripts.train_compare_xgboost_v2`. El diagnostico LIVE opcional, solo lectura, se genera con `python -m scripts.analyze_v2_live_drift` si existe una clave publica Supabase configurada.",
              "Los ZIP, Parquet y modelos joblib experimentales permanecen fuera de Git. El manifest, los reportes, las definiciones y los checksums pequenos quedan versionables."]
    return "\n".join(lines) + "\n"


def main():
    if sha256(ORIGINAL)!=ORIGINAL_SHA:
        raise RuntimeError("Product model SHA differs from frozen reference")
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    if sha256(EXPERIMENT)!=audit["cohort"]["sha256"]:
        raise ValueError("Experiment dataset fingerprint mismatch")
    frame=pd.read_parquet(EXPERIMENT)
    splits,split_audit=temporal_splits(frame)
    if split_audit!=audit["splits"] or tuple(audit["cohort"]["feature_set_v1"])!=V1 or tuple(audit["cohort"]["feature_set_v2"])!=V2:
        raise ValueError("Cohort, split or feature order differs from audit")
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "source":"Binance Public Data SPOT BTCUSDT 1m, verified ZIP CHECKSUMs",
            "source_manifest":"reports/xgboost_v2_6m_source_manifest.json",
            "dataset_sha256":audit["cohort"]["sha256"],"original_model_sha256":ORIGINAL_SHA,
            "features":{"A":list(V1),"B":list(V2)},"parameters":dict(XGB_PARAMS),
            "threshold":.5,"splits":split_audit,"metrics":{"A":{},"B":{}},
            "baselines":{},"drift":{"A":{},"B":{}},"walk_forward":[],
            "models":{},"contributions":{},
            "versions":{"python":platform.python_version(),"pandas":pd.__version__,
                        "sklearn":sklearn.__version__,"xgboost":xgboost.__version__,
                        "joblib":joblib.__version__}}
    for split,part in splits.items():
        report["baselines"][split]=baseline_metrics(part)
    for name,features in (("A",V1),("B",V2)):
        model=XGBClassifier(**XGB_PARAMS)
        model.fit(splits["TRAIN"][list(features)],splits["TRAIN"].target)
        print(f"trained {NAMES[name]} on {len(splits['TRAIN'])} rows",flush=True)
        for split,part in splits.items():
            p=model.predict_proba(part[list(features)])[:,1]
            report["metrics"][name][split]=evaluate(part.target,p)
            if split!="TRAIN":
                report["drift"][name][split]=feature_drift(splits["TRAIN"],part,features)
        report["contributions"][name]=contributions(model,splits["FORWARD"],features)
        report["models"][name]=_model_output(model,features,NAMES[name],splits["TRAIN"],audit)
    development=pd.concat([splits["TRAIN"],splits["VALIDATION"]],ignore_index=True)
    for number,train,valid in expanding_folds(development):
        fold={"fold":number,"train_range_utc":[train.timestamp.iloc[0].isoformat(),train.timestamp.iloc[-1].isoformat()],
              "validation_range_utc":[valid.timestamp.iloc[0].isoformat(),valid.timestamp.iloc[-1].isoformat()],
              "train_n":len(train),"validation_n":len(valid),"purged_source_rows":1,"metrics":{}}
        for name,features in (("A",V1),("B",V2)):
            model=XGBClassifier(**XGB_PARAMS)
            model.fit(train[list(features)],train.target)
            fold["metrics"][name]=evaluate(valid.target,model.predict_proba(valid[list(features)])[:,1])
        report["walk_forward"].append(fold)
        print(f"walk_fold={number} n={len(valid)} A_AUC={fold['metrics']['A']['roc_auc']:.4f} B_AUC={fold['metrics']['B']['roc_auc']:.4f}",flush=True)
    original_bundle=joblib.load(ORIGINAL)
    original_end=pd.Timestamp(json.loads((ROOT/"models/xgboost_v1_metadata.json").read_text(encoding="utf-8"))["data_range"]["end"])
    later=splits["FORWARD"].loc[splits["FORWARD"].timestamp>original_end]
    if later.empty:
        raise ValueError("No strictly later FORWARD observations for product reference")
    original=original_bundle["estimator"]
    strict={"rows":len(later),"first_source_utc":later.timestamp.iloc[0].isoformat(),
            "last_source_utc":later.timestamp.iloc[-1].isoformat(),
            "caution":"A has more but older training rows than the original; the difference is not a pure sample-size effect.",
            "original_xgboost_v1":evaluate(later.target,original.predict_proba(later[list(V1)])[:,1])}
    for name in ("A","B"):
        model=joblib.load(ROOT/report["models"][name]["model_path"])["estimator"]
        features=V1 if name=="A" else V2
        strict[name]=evaluate(later.target,model.predict_proba(later[list(features)])[:,1])
    strict["baselines"]=baseline_metrics(later)
    report["strictly_post_original_reference"]=strict
    report["methodology"]={"paired_rows":True,"same_hyperparameters":True,
                           "training":"Final A/B fit on TRAIN only; no class weighting, calibration or tuning",
                           "walk_forward":"Four expanding folds on TRAIN+VALIDATION only; purge one H1 label before every validation block",
                           "test_forward":"Both fixed specifications scored; neither used for selection",
                           "previous_movement":"UP iff source close > previous source close; ties DOWN",
                           "production_reference":"Original model evaluated only on source rows after original historical end, no fitting",
                           "no_live_fit_or_selection":True}
    live_path=ROOT/"reports/xgboost_v2_6m_live_drift.json"
    if live_path.exists():
        live=json.loads(live_path.read_text(encoding="utf-8"))
        if live.get("training_cohort_sha256")==audit["cohort"]["sha256"]:
            report["live_drift"]=live
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    OUT.with_suffix(".md").write_text(_markdown(report),encoding="utf-8")
    if sha256(ORIGINAL)!=ORIGINAL_SHA:
        raise RuntimeError("Production artifact changed")
    print(f"FORWARD A_ROC={report['metrics']['A']['FORWARD']['roc_auc']:.4f} B_ROC={report['metrics']['B']['FORWARD']['roc_auc']:.4f} original_unchanged=True",flush=True)


if __name__=="__main__":
    main()
