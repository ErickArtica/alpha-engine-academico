"""Offline, leakage-aware A/B/C experiment on the frozen BTCUSDT 1m cohort.

No LIVE reads, production writes, model promotion, calibration or threshold
optimization. TEST is scored after the specification and walk-forward folds.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, auc,
)
from xgboost import XGBClassifier

from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame, chronological_split
from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.analyze_feature_parity import _load_historical, population_stability_index

ROOT=Path(__file__).resolve().parents[1]
PROD=ROOT/"models/xgboost_v1.joblib"
META=ROOT/"models/xgboost_v1_metadata.json"
BACKFILL=ROOT/"data/experiments/tfi/joined_klines_tfi.jsonl"
BACKFILL_REPORT=ROOT/"reports/tfi_backfill_v1.json"
DATASET=ROOT/"data/experiments/tfi/experimental_labeled.jsonl"
EXPERIMENTS=ROOT/"models/experiments/tfi_v1"
REPORT=ROOT/"reports/model_ablation_tfi_v1.json"
WALK=ROOT/"reports/walk_forward_tfi_v1.json"
EXPECTED_PROD_SHA="8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
CODE_FILES=("scripts/backfill_trade_flow.py","scripts/run_tfi_ablation.py",
            "alpha_engine/experiments/microstructure.py","alpha_engine/features/indicators.py",
            "alpha_engine/features/target.py","alpha_engine/features/split.py",
            "alpha_engine/ml/trainer.py")
A_FEATURES=list(FEATURE_COLUMNS)
B_FEATURES=A_FEATURES+["trade_flow_imbalance"]
C_FEATURES=["trade_flow_imbalance"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_fingerprint() -> str:
    digest=hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode()+b"\0"+(ROOT/name).read_bytes()+b"\0")
    return digest.hexdigest()


def _json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def read_backfill(frame: pd.DataFrame) -> pd.Series:
    """Join exact source-minute TFI and reject any incomplete/future rows."""
    report=json.loads(BACKFILL_REPORT.read_text(encoding="utf-8"))
    if _sha(BACKFILL)!=report["dataset_sha256"]:
        raise ValueError("Backfill digest differs from validated report")
    rows=[json.loads(line) for line in BACKFILL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=report["rows"]:
        raise ValueError("Backfill row count differs from validation report")
    times=[];values=[]
    for row in rows:
        source=pd.Timestamp(row["source_candle_open_time"])
        close=pd.Timestamp(row["source_candle_close_time"])
        cutoff=pd.Timestamp(row["feature_cutoff_time"])
        if source.tz is None or source.tz_convert("UTC")!=source or close!=source+timedelta(minutes=1,milliseconds=-1) or cutoff>close:
            raise ValueError("Backfill has invalid UTC source-candle alignment")
        value=row["trade_flow_imbalance"]
        if value is None:
            raise ValueError("Null TFI: cannot reproduce original A/B cohort without dropping a row")
        number=float(value)
        if not np.isfinite(number) or not -1<=number<=1:
            raise ValueError("Invalid TFI")
        times.append(source);values.append(number)
    series=pd.Series(values,index=pd.DatetimeIndex(times),name="trade_flow_imbalance",dtype="float64")
    if not series.index.is_unique or not series.index.is_monotonic_increasing or not frame.index.equals(series.index):
        raise ValueError("TFI backfill does not join one-to-one with original candles")
    return series


def write_dataset(labeled: pd.DataFrame) -> str:
    """Ignored local dataset; X is always explicitly selected by A/B/C feature lists."""
    DATASET.parent.mkdir(parents=True,exist_ok=True)
    with DATASET.open("w",encoding="utf-8",newline="\n") as handle:
        for time,row in labeled.iterrows():
            item={"symbol":row["symbol"],"interval":row["interval"],
                  "source_candle_open_time":time.isoformat(),
                  "source_candle_close_time":row["close_time"].isoformat(),
                  "open":float(row["open"]),"high":float(row["high"]),
                  "low":float(row["low"]),"close":float(row["close"]),
                  "volume":float(row["volume"]),
                  "features":{name:float(row[name]) for name in B_FEATURES},
                  "target":int(row["target"])}
            handle.write(json.dumps(item,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
    return _sha(DATASET)


def train_model(frame: pd.DataFrame, features: list[str]) -> XGBClassifier:
    if frame["target"].nunique()!=2 or list(frame[features].columns)!=features:
        raise ValueError("Invalid training labels or feature order")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite training features")
    model=XGBClassifier(**XGB_PARAMS)
    model.fit(frame[features],frame["target"])
    if list(model.classes_)!=[0,1]:raise ValueError("Class 1 must be UP")
    return model


def probabilities(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    result=np.asarray(model.predict_proba(frame[features])[:,1],dtype=float)
    if not np.isfinite(result).all() or ((result<0)|(result>1)).any():
        raise ValueError("Invalid model probabilities")
    return result


def reproduction_gate(candidate: np.ndarray, frozen: np.ndarray, *, label: str) -> dict:
    candidate=np.asarray(candidate,dtype=float);frozen=np.asarray(frozen,dtype=float)
    if candidate.shape!=frozen.shape or not len(candidate) or not np.isfinite(candidate).all() or not np.isfinite(frozen).all():
        raise ValueError("Baseline reproduction arrays are unaligned or invalid")
    maximum=float(np.abs(candidate-frozen).max())
    flips=int(((candidate>.5)!=(frozen>.5)).sum())
    if maximum>1e-6 or flips:
        raise RuntimeError(f"Experimental baseline A fails {label} reproduction gate")
    return {"max_abs_prob_difference":maximum,"class_flips":flips}


def metrics(y, p) -> dict:
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if len(y)==0 or len(y)!=len(p) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all():
        raise ValueError("Invalid aligned evaluation vectors")
    predicted=(p>.5).astype(int)
    cm=confusion_matrix(y,predicted,labels=[0,1])
    confidence=np.maximum(p,1-p)
    precision,recall,_=precision_recall_curve(y,p)
    result={"n":int(len(y)),"accuracy":float(accuracy_score(y,predicted)),
        "balanced_accuracy":float(balanced_accuracy_score(y,predicted)),
        "roc_auc":float(roc_auc_score(y,p)),
        "pr_auc_up":float(auc(recall,precision)),
        "average_precision_up":float(average_precision_score(y,p)),
        "precision_up":float(precision_score(y,predicted,pos_label=1,zero_division=0)),
        "recall_up":float(recall_score(y,predicted,pos_label=1,zero_division=0)),
        "f1_up":float(f1_score(y,predicted,pos_label=1,zero_division=0)),
        "precision_down":float(precision_score(y,predicted,pos_label=0,zero_division=0)),
        "recall_down":float(recall_score(y,predicted,pos_label=0,zero_division=0)),
        "f1_down":float(f1_score(y,predicted,pos_label=0,zero_division=0)),
        "macro_f1":float(f1_score(y,predicted,average="macro",zero_division=0)),
        "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
        "confusion_real_rows_down_up_pred_columns_down_up":cm.tolist(),
        "predicted_up":int(predicted.sum()),"predicted_down":int(len(y)-predicted.sum()),
        "predicted_up_fraction":float(predicted.mean()),
        "actual_up":int(y.sum()),"actual_down":int(len(y)-y.sum()),
        "confidence":{"mean":float(confidence.mean()),"median":float(np.median(confidence)),
                      "p75":float(np.quantile(confidence,.75)),"p90":float(np.quantile(confidence,.90)),
                      "p95":float(np.quantile(confidence,.95)),"p99":float(np.quantile(confidence,.99)),
                      "ge_60_count":int((confidence>=.60).sum())}}
    return result


def expanding_folds(frame: pd.DataFrame, count: int = 4):
    """All folds lie in TRAIN+VALIDATION; purge one label before each boundary."""
    if count<2 or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Invalid temporal fold specification")
    initial=len(frame)//2
    block=(len(frame)-initial)//count
    if initial<100 or block<30:raise ValueError("Insufficient temporal fold size")
    for number in range(count):
        boundary=initial+number*block
        end=initial+(number+1)*block if number<count-1 else len(frame)
        train=frame.iloc[:boundary-1]
        validation=frame.iloc[boundary:end]
        if train.index[-1]>=validation.index[0] or len(train)<100 or len(validation)<30:
            raise ValueError("Temporal fold overlap")
        yield number+1,train,validation


def walk_forward(frame: pd.DataFrame) -> dict:
    rows=[]
    for number,train,validation in expanding_folds(frame):
        actual=validation["target"].to_numpy(dtype=int)
        a=train_model(train,A_FEATURES)
        b=train_model(train,B_FEATURES)
        am=metrics(actual,probabilities(a,validation,A_FEATURES))
        bm=metrics(actual,probabilities(b,validation,B_FEATURES))
        rows.append({"fold":number,"train":{"n":len(train),"start":train.index[0].isoformat(),
            "end":train.index[-1].isoformat()},
            "validation":{"n":len(validation),"start":validation.index[0].isoformat(),
                          "end":validation.index[-1].isoformat()},
            "purged_rows_before_validation":1,
            "A":am,"B":bm,
            "delta_B_minus_A":{name:bm[name]-am[name] for name in
                ("roc_auc","balanced_accuracy","accuracy","macro_f1","recall_up","recall_down")}})
    return {"method":"4 expanding folds on TRAIN+VALIDATION only; one source label purged before each fold; fixed XGB_PARAMS; P(UP)>0.5",
            "folds":rows}


def paired_block_bootstrap(y, a, b, *, block=60, repeats=1000, seed=42) -> dict:
    """Paired circular 60-row blocks preserve nearby minute dependence."""
    y=np.asarray(y,dtype=int);a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not len(y)==len(a)==len(b) or len(y)<2*block:raise ValueError("Bootstrap cohort too small")
    rng=np.random.default_rng(seed);records={name:[] for name in ("accuracy","balanced_accuracy","roc_auc")}
    for _ in range(repeats):
        starts=rng.integers(0,len(y),size=int(np.ceil(len(y)/block)))
        indices=np.concatenate([(start+np.arange(block))%len(y) for start in starts])[:len(y)]
        yi=y[indices]
        if len(np.unique(yi))!=2:continue
        pa=a[indices];pb=b[indices]
        records["accuracy"].append(accuracy_score(yi,pb>.5)-accuracy_score(yi,pa>.5))
        records["balanced_accuracy"].append(balanced_accuracy_score(yi,pb>.5)-balanced_accuracy_score(yi,pa>.5))
        records["roc_auc"].append(roc_auc_score(yi,pb)-roc_auc_score(yi,pa))
    return {"method":"paired circular moving-block bootstrap, 60 observations/block, 1000 resamples, seed 42; descriptive interval",
            "differences_B_minus_A":{name:{"lower_95":float(np.quantile(values,.025)),
                                       "upper_95":float(np.quantile(values,.975)),
                                       "fraction_le_zero":float(np.mean(np.asarray(values)<=0)),
                                       "resamples":len(values)} for name,values in records.items()}}


def tfi_target_distribution(train: pd.DataFrame) -> dict:
    result={}
    for label,name in ((0,"DOWN"),(1,"UP")):
        values=train.loc[train["target"]==label,"trade_flow_imbalance"].to_numpy(dtype=float)
        result[name]={"n":len(values),"mean":float(np.mean(values)),"median":float(np.median(values)),
                      "p05":float(np.quantile(values,.05)),"p25":float(np.quantile(values,.25)),
                      "p75":float(np.quantile(values,.75)),"p95":float(np.quantile(values,.95))}
    x=train["trade_flow_imbalance"].to_numpy(dtype=float)
    y=train["target"].to_numpy(dtype=int)
    result["pearson_train_only"]=float(np.corrcoef(x,y)[0,1])
    result["mutual_information_train_only"]=float(mutual_info_classif(x.reshape(-1,1),y,random_state=42)[0])
    return result


def test_movement_analysis(source_close: np.ndarray, next_close: np.ndarray,
                           y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Descriptive TEST-only move sizes; never an input to a model or selector."""
    source_close=np.asarray(source_close,dtype=float)
    next_close=np.asarray(next_close,dtype=float)
    y=np.asarray(y,dtype=int)
    if not(len(source_close)==len(next_close)==len(y)==len(a)==len(b)) or (source_close<=0).any():
        raise ValueError("Unaligned TEST movement observations")
    signed=(next_close/source_close-1)*10_000
    absolute=np.abs(signed)
    rows=[]
    for low,high in ((0,1),(1,3),(3,float("inf"))):
        mask=(absolute>=low)&(absolute<high)
        n=int(mask.sum())
        rows.append({"lower_bps":low,"upper_bps":high if np.isfinite(high) else None,
                     "n":n,"fraction":n/len(y),
                     "A_accuracy":float(((a[mask]>.5)==y[mask]).mean()) if n else None,
                     "B_accuracy":float(((b[mask]>.5)==y[mask]).mean()) if n else None})
    return {"median_abs_bps":float(np.median(absolute)),"p25_abs_bps":float(np.quantile(absolute,.25)),
            "p75_abs_bps":float(np.quantile(absolute,.75)),
            "flat_exact_close_count":int((signed==0).sum()),"bins":rows,
            "note":"Next close is used only after inference to describe errors; target and models were unchanged."}


def importance(model: XGBClassifier) -> dict:
    booster=model.get_booster()
    gain=booster.get_score(importance_type="gain")
    weight=booster.get_score(importance_type="weight")
    return {"gain":{name:float(gain.get(name,0)) for name in B_FEATURES},
            "split_count":{name:int(weight.get(name,0)) for name in B_FEATURES},
            "tfi_gain_share":float(gain.get("trade_flow_imbalance",0)/sum(gain.values())) if gain else 0.0}


def _save_model(model: XGBClassifier, name: str, features: list[str], dataset_sha: str,
                commit: str, code_sha: str, metrics_by_split: dict, trained_at: str) -> dict:
    EXPERIMENTS.mkdir(parents=True,exist_ok=True)
    path=EXPERIMENTS/f"{name}.joblib"
    joblib.dump({"estimator":model,"features":features,"model_version":name,
                "symbol":"BTCUSDT","interval":"1m","horizon":"1m"},path,compress=3)
    information={"name":name,"features":features,"trained_at_utc":trained_at,
        "training_range":{"start":metrics_by_split["train_range"]["start"],
                          "end":metrics_by_split["train_range"]["end"],
                          "rows":metrics_by_split["train_range"]["n"]},
        "split":"frozen xgboost_v1 chronological 70/15/15, purge 1 each boundary",
        "hyperparameters":XGB_PARAMS,"dataset_sha256":dataset_sha,
        "model_sha256":_sha(path),"code_commit":commit,"code_fingerprint_sha256":code_sha,
        "evaluation":{"validation":metrics_by_split["validation"],"test":metrics_by_split["test"]}}
    _json(EXPERIMENTS/f"{name}_metadata.json",information)
    return {"path":str(path.relative_to(ROOT)),"sha256":information["model_sha256"]}


def _markdown(report: dict, walk: dict) -> str:
    lines=["# XGBoost v1 + trade-flow: ablation experimental — FASE 4","",
        f"Generado: {report['generated_at_utc']}. Git `{report['code_commit']}`; workspace con cambios locales={report['git_worktree_dirty']}; fingerprint de código experimental `{report['code_fingerprint_sha256']}`. Modelo productivo `{report['production_model_sha256']}` sin cambios.","",
        "A: ocho features originales; B: mismas ocho más TFI; C: sólo TFI diagnóstico. Mismo target, horizonte, splits, parámetros y clasificación P(UP)>0.50. Sin LIVE en fit/selección.","",
        f"Baseline A reproduce exactamente xgboost_v1: máximo |ΔP(UP)| VALIDATION={report['baseline_reproduction']['validation_max_abs_prob_difference']:.3g}; flips={report['baseline_reproduction']['validation_class_flips']}. TEST se revisó al final y también coincide.","",
        "## TEST final","",
        "| Métrica | A | B | Δ B−A | C diagnóstico |","|---|---:|---:|---:|---:|"]
    for name in ("accuracy","balanced_accuracy","roc_auc","pr_auc_up","average_precision_up","precision_up","recall_up","f1_up","precision_down","recall_down","f1_down","macro_f1","brier","log_loss"):
        a=report["models"]["A"]["test"][name];b=report["models"]["B"]["test"][name];c=report["models"]["C"]["test"][name]
        lines.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {c:.4f} |")
    lines += ["", "## Walk-forward TRAIN+VALIDATION", "",
              "| Fold | Periodo VALIDATION UTC | A ROC AUC | B ROC AUC | Δ | A BalAcc | B BalAcc | Δ |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for fold in walk["folds"]:
        a=fold["A"];b=fold["B"];d=fold["delta_B_minus_A"]
        lines.append(f"| {fold['fold']} | {fold['validation']['start']}–{fold['validation']['end']} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {d['roc_auc']:+.4f} | {a['balanced_accuracy']:.4f} | {b['balanced_accuracy']:.4f} | {d['balanced_accuracy']:+.4f} |")
    movement=report["test_movement_analysis"]
    lines += ["", "## Tamaño del movimiento objetivo en TEST (diagnóstico)","",
              f"Mediana |movimiento siguiente|: {movement['median_abs_bps']:.3f} bps; empates exactos: {movement['flat_exact_close_count']}. No se cambió el target.",
              "| Magnitud absoluta | Casos | A Accuracy | B Accuracy |","|---|---:|---:|---:|"]
    for row in movement["bins"]:
        band=f"{row['lower_bps']}–{row['upper_bps']} bps" if row["upper_bps"] is not None else f">={row['lower_bps']} bps"
        lines.append(f"| {band} | {row['n']} | {row['A_accuracy']:.4f} | {row['B_accuracy']:.4f} |")
    lines += ["", "## L1 prospectivo: diseño pendiente","",
              "Para spread, L1 OBI y microprice se necesitarían snapshots/eventos con recepción UTC dentro de cada vela fuente, captura continua con reconexión, deduplicación, control de huecos y persistencia shadow independiente. Las 19 filas locales no alcanzan para training; no se activó captura en Render ni se creó servicio adicional.","",
              "Bootstrap pareado por bloques de 60 minutos y análisis de importancia figuran en JSON. TEST ya constaba en la metadata original y se ha consultado en fases previas: no es una reserva independiente nunca vista.",
              "Ningún artefacto experimental se promovió a producción. Las 19 filas L1 de FASE 3 no entran en entrenamiento.",""]
    return "\n".join(lines)


def run() -> dict:
    before=_sha(PROD)
    if before!=EXPECTED_PROD_SHA:raise ValueError("Production model SHA changed before experiment")
    metadata=json.loads(META.read_text(encoding="utf-8"))
    frozen_meta_sha=_sha(META)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    code_sha=_code_fingerprint()
    git_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    candles,frame,parts=_load_historical(metadata)
    tfi=read_backfill(frame)
    frame=frame.assign(trade_flow_imbalance=tfi)
    labeled=build_training_frame(frame)
    if labeled[B_FEATURES].isna().any().any() or not np.isfinite(labeled[B_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("A/B cohorts would differ because experimental features are missing")
    dataset_sha=write_dataset(labeled)
    split=chronological_split(labeled)
    for name in ("train","validation","test"):
        new=getattr(split,name);original=parts[name.upper()]
        if not new.index.equals(original.index) or not new["target"].equals(original["target"]):
            raise ValueError(f"Frozen {name} cohort or target differs after TFI join")
    trained_at=datetime.now(timezone.utc).isoformat()
    a=train_model(split.train,A_FEATURES)
    original=joblib.load(PROD)["estimator"]
    a_val=probabilities(a,split.validation,A_FEATURES)
    old_val=probabilities(original,split.validation,A_FEATURES)
    val_gate=reproduction_gate(a_val,old_val,label="VALIDATION")
    b=train_model(split.train,B_FEATURES)
    c=train_model(split.train,C_FEATURES)
    # No tuning or selection: folds are reported with the frozen specification.
    development=pd.concat((split.train,split.validation)).sort_index()
    walk=walk_forward(development)
    walk.update({"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_sha256":dataset_sha,"code_commit":commit,"code_fingerprint_sha256":code_sha})
    # TEST is evaluated only after the fixed specification and folds above.
    results={}
    test_probs={}
    for key,model,features in (("A",a,A_FEATURES),("B",b,B_FEATURES),("C",c,C_FEATURES)):
        pieces={}
        for name in ("train","validation","test"):
            part=getattr(split,name)
            p=probabilities(model,part,features)
            pieces[name]=metrics(part["target"].to_numpy(dtype=int),p)
            if name=="test":test_probs[key]=p
        results[key]=pieces
    old_test=probabilities(original,split.test,A_FEATURES)
    test_gate=reproduction_gate(test_probs["A"],old_test,label="TEST")
    if abs(results["A"]["test"]["accuracy"]-metadata["evaluation"]["test"]["accuracy"])>1e-12:
        raise RuntimeError("A does not reproduce the frozen original TEST reference")
    y_test=split.test["target"].to_numpy(dtype=int)
    intervals=paired_block_bootstrap(y_test,test_probs["A"],test_probs["B"])
    source_close=frame["close"].reindex(split.test.index).to_numpy(dtype=float)
    next_close=frame["close"].shift(-1).reindex(split.test.index).to_numpy(dtype=float)
    movement=test_movement_analysis(source_close,next_close,y_test,test_probs["A"],test_probs["B"])
    tfi_distribution=tfi_target_distribution(split.train)
    experimental_ema=(frame["close"]-frame["ema_20"])/frame["close"]
    relative_ema=experimental_ema.reindex(labeled.index)
    train_ema=relative_ema.reindex(split.train.index).to_numpy(dtype=float)
    ema_feasibility={"formula":"(close-ema_20)/close; analysis only; absent from A/B/C",
        "train_psi":population_stability_index(train_ema,train_ema),
        "validation_psi_against_train":population_stability_index(train_ema,relative_ema.reindex(split.validation.index).to_numpy(dtype=float)),
        "test_psi_against_train":population_stability_index(train_ema,relative_ema.reindex(split.test.index).to_numpy(dtype=float))}
    report={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"code_commit":commit,
        "code_fingerprint_sha256":code_sha,"git_worktree_dirty":git_dirty,
        "production_model_sha256":before,"production_metadata_sha256":frozen_meta_sha,
        "target_definition":metadata["target_definition"],"horizon":"1m","symbol":"BTCUSDT","interval":"1m",
        "features":{"A":A_FEATURES,"B":B_FEATURES,"C":C_FEATURES},"hyperparameters":XGB_PARAMS,
        "dataset":{"source_candles":len(candles),"labeled_rows":len(labeled),"sha256":dataset_sha,
            "backfill_sha256":_sha(BACKFILL),"tfi_missing":0,"target_future_in_features":False,
            "raw_path":"data/experiments/tfi/experimental_labeled.jsonl (gitignored)"},
        "split_ranges":{name:{"n":len(getattr(split,name)),"start":getattr(split,name).index[0].isoformat(),
            "end":getattr(split,name).index[-1].isoformat()} for name in ("train","validation","test")},
        "baseline_reproduction":{"validation_max_abs_prob_difference":val_gate["max_abs_prob_difference"],
            "validation_class_flips":val_gate["class_flips"],"test_max_abs_prob_difference":test_gate["max_abs_prob_difference"],
            "test_class_flips":test_gate["class_flips"],"original_validation_accuracy":metadata["evaluation"]["validation"]["accuracy"],
            "original_test_accuracy":metadata["evaluation"]["test"]["accuracy"]},
        "models":results,"tfi_target_train":tfi_distribution,"feature_importance_B":importance(b),
        "test_uncertainty":intervals,"ema_relative_feasibility":ema_feasibility,
        "test_movement_analysis":movement,
        "test_baselines":{"always_down_accuracy":float((y_test==0).mean()),
            "previous_movement_accuracy":float(((split.test["return_1m"].to_numpy()>0)==y_test).mean())},
        "live_usage":"None. No LIVE observations were read, fitted, selected, or evaluated in this experiment.",
        "historical_test_limitation":"Original TEST was reported by xgboost_v1 and inspected in prior phases; this is a paired final historical comparison, not a fresh independent holdout."}
    range_info=report["split_ranges"]["train"]
    save_context={"train_range":range_info,"validation":results["A"]["validation"],"test":results["A"]["test"]}
    artifacts={}
    for key,model,features,name in (("A",a,A_FEATURES,"xgboost_v1_baseline_retrained"),
                                    ("B",b,B_FEATURES,"xgboost_v1_plus_tfi"),
                                    ("C",c,C_FEATURES,"xgboost_v1_tfi_only_diagnostic")):
        save_context.update({"validation":results[key]["validation"],"test":results[key]["test"]})
        artifacts[key]=_save_model(model,name,features,dataset_sha,commit,code_sha,save_context,trained_at)
    report["experimental_artifacts"]=artifacts
    after=_sha(PROD)
    if after!=before or _sha(META)!=frozen_meta_sha:
        raise RuntimeError("Production model or metadata changed during experiment")
    report["production_model_sha256_after"]=after
    _json(WALK,walk)
    _json(REPORT,report)
    REPORT.with_suffix(".md").write_text(_markdown(report,walk),encoding="utf-8")
    lines=["# Walk-forward A vs B — FASE 4","",walk["method"],"",
        "| Fold | VALIDATION UTC | A ROC AUC | B ROC AUC | Δ | A Balanced Accuracy | B Balanced Accuracy | Δ |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for fold in walk["folds"]:
        a=fold["A"];b=fold["B"];d=fold["delta_B_minus_A"]
        lines.append(f"| {fold['fold']} | {fold['validation']['start']}–{fold['validation']['end']} | {a['roc_auc']:.4f} | {b['roc_auc']:.4f} | {d['roc_auc']:+.4f} | {a['balanced_accuracy']:.4f} | {b['balanced_accuracy']:.4f} | {d['balanced_accuracy']:+.4f} |")
    WALK.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    output=run()
    a=output["models"]["A"]["test"];b=output["models"]["B"]["test"]
    print(f"A/B TEST: AUC {a['roc_auc']:.5f}/{b['roc_auc']:.5f}; Balanced Acc {a['balanced_accuracy']:.5f}/{b['balanced_accuracy']:.5f}")
