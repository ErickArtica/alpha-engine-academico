"""Offline diagnosis of the frozen xgboost_v1 probabilities.

Only the early chronological VALIDATION block fits experimental calibrators.
TEST and LIVE are read-only evaluations; nothing here updates production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, log_loss, precision_recall_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score,
)

from alpha_engine.config.settings import Settings
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame, chronological_split
from alpha_engine.ml.walk_forward import candle_digest
from alpha_engine.repositories.influx3 import Influx3MarketRepository
from scripts.live_model_diagnostic import (
    EVALUATION_SELECT, _fetch_all, _observations, _publishable_key,
    _require_public_read_key,
)

ROOT = Path(__file__).resolve().parents[1]
P_UP_THRESHOLDS = (.35, .40, .42, .44, .46, .48, .50, .52, .54, .56, .58, .60)
CONFIDENCE_GATES = (.50, .52, .54, .55, .56, .58, .60)
P_UP_EDGES = (0, .42, .44, .46, .48, .50, .52, .54, .56, .58, 1.0000001)


def _arrays(y, p):
    actual = np.asarray(y, dtype=int)
    probability = np.asarray(p, dtype=float)
    if actual.ndim != 1 or probability.ndim != 1 or len(actual) != len(probability) or not len(actual):
        raise ValueError("Actual labels and probabilities must be nonempty, aligned vectors")
    if not set(np.unique(actual)).issubset({0, 1}) or not np.isfinite(probability).all():
        raise ValueError("Binary labels and finite probabilities required")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    return actual, probability


def reliability_bins(y, p, *, count=10, strategy="quantile"):
    """Return observed UP frequency per bin; empty fixed-width bins are explicit."""
    y, p = _arrays(y, p)
    if count < 2 or strategy not in ("quantile", "width"):
        raise ValueError("Invalid bins")
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, count + 1))
        edges = np.unique(edges)
    else:
        # Width across the observed support, rather than ten nearly empty
        # deciles over [0,1]. The support is recorded in the report.
        edges = np.linspace(float(p.min()), float(p.max()), count + 1)
    if len(edges) < 2 or edges[0] == edges[-1]:
        edges = np.array([float(p[0]), float(p[0])])
    result = []
    for index in range(len(edges) - 1):
        chosen = (p >= edges[index]) & ((p < edges[index+1]) if index < len(edges)-2 else (p <= edges[index+1]))
        n = int(chosen.sum())
        predicted = float(p[chosen].mean()) if n else None
        observed = float(y[chosen].mean()) if n else None
        result.append({"lower": float(edges[index]), "upper": float(edges[index+1]),
                       "n": n, "mean_predicted_up": predicted, "observed_up": observed,
                       "gap": observed-predicted if n else None})
    return result


def calibration_error(bins):
    total = sum(row["n"] for row in bins)
    if not total:
        raise ValueError("Empty reliability bins")
    gaps = [abs(row["gap"]) for row in bins if row["n"]]
    return {"ece": sum(row["n"]*abs(row["gap"]) for row in bins if row["n"])/total,
            "mce": max(gaps)}


def class_metrics(y, p, *, threshold=.50):
    y, p = _arrays(y, p)
    # Production uses UP only if P(UP) > P(DOWN); exact ties are DOWN.
    predicted = (p > threshold).astype(int)
    return {"n": len(y), "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "precision_up": float(precision_score(y, predicted, pos_label=1, zero_division=0)),
            "recall_up": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
            "f1_up": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
            "precision_down": float(precision_score(y, predicted, pos_label=0, zero_division=0)),
            "recall_down": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "predicted_up": int(predicted.sum()), "predicted_down": int(len(y)-predicted.sum()),
            "actual_up": int(y.sum()), "actual_down": int(len(y)-y.sum())}


def wilson(correct, n, z=1.959963984540054):
    if n == 0:
        return None
    fraction = correct/n
    denominator = 1+z*z/n
    center = (fraction+z*z/(2*n))/denominator
    radius = z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denominator
    return [float(center-radius), float(center+radius)]


def confidence_gates(y, p, gates=CONFIDENCE_GATES):
    y, p = _arrays(y, p)
    confidence = np.maximum(p, 1-p)
    predicted = p > .5
    rows = []
    for gate in gates:
        mask = confidence >= gate
        n = int(mask.sum())
        right = int((predicted[mask] == y[mask]).sum())
        up = mask & predicted
        down = mask & ~predicted
        rows.append({"threshold": gate, "n": n, "coverage": n/len(y),
                     "correct": right, "accuracy": right/n if n else None,
                     "always_down_accuracy_selected": float((y[mask]==0).mean()) if n else None,
                     "wilson_95": wilson(right,n),
                     "balanced_accuracy": float(balanced_accuracy_score(y[mask], predicted[mask])) if n and len(set(y[mask])) == 2 else None,
                     "precision_up": float(y[up].mean()) if up.any() else None,
                     "precision_down": float((1-y[down]).mean()) if down.any() else None,
                     "predicted_up": int(up.sum()), "predicted_down": int(down.sum())})
    return rows


def threshold_table(y, p):
    return [{"threshold": t, **class_metrics(y,p,threshold=t)} for t in P_UP_THRESHOLDS]


def _logit(p):
    p = np.clip(np.asarray(p,dtype=float), 1e-9, 1-1e-9)
    return np.log(p/(1-p))


def fit_calibrators(y, p):
    y, p = _arrays(y,p)
    if len(y) < 1000 or len(np.unique(y)) != 2:
        raise ValueError("Insufficient temporal calibration data")
    platt = LogisticRegression(C=np.inf, max_iter=1000)
    platt.fit(_logit(p).reshape(-1,1), y)
    if platt.coef_[0,0] <= 0:
        raise ValueError("Platt fit is non-monotonic; cannot use as calibrator")
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(p,y)
    return platt,isotonic


def apply_calibrator(name, p, platt=None, isotonic=None):
    p=np.asarray(p,dtype=float)
    if name == "uncalibrated": return p
    if name == "platt": return platt.predict_proba(_logit(p).reshape(-1,1))[:,1]
    if name == "isotonic": return isotonic.predict(p)
    raise ValueError("Unknown calibrator")


def probability_metrics(y,p):
    y,p=_arrays(y,p)
    width=reliability_bins(y,p,strategy="width")
    quantile=reliability_bins(y,p,strategy="quantile")
    point=class_metrics(y,p)
    result={**point,
        "brier": float(brier_score_loss(y,p)), "log_loss": float(log_loss(y,p,labels=[0,1])),
        "ece_width": calibration_error(width), "ece_quantile": calibration_error(quantile),
        "reliability_width": width, "reliability_quantile": quantile,
        "prob_up_range": [float(p.min()),float(p.max())],
        "mean_prob_up": float(p.mean()),
        "class_thresholds": threshold_table(y,p),
        "confidence_gates": confidence_gates(y,p),
        "confidence": {name: float(value) for name,value in zip(
            ("mean","median","p75","p90","p95","p99","max"),
            (np.mean(np.maximum(p,1-p)),*np.quantile(np.maximum(p,1-p),[.5,.75,.9,.95,.99]),np.max(np.maximum(p,1-p))))},
    }
    if len(np.unique(y))==2:
        precision,recall,_=precision_recall_curve(y,p)
        precision_down,recall_down,_=precision_recall_curve(1-y,1-p)
        result.update({"roc_auc":float(roc_auc_score(y,p)),
                       "average_precision_up":float(average_precision_score(y,p)),
                       "pr_auc_up":float(auc(recall,precision)),
                       "average_precision_down":float(average_precision_score(1-y,1-p)),
                       "pr_auc_down":float(auc(recall_down,precision_down))})
        # Descriptive logistic recalibration on this dataset, not a fitted
        # calibrator used elsewhere. Its values diagnose offset and slope.
        diagnostic=LogisticRegression(C=np.inf,max_iter=1000)
        diagnostic.fit(_logit(p).reshape(-1,1),y)
        result["calibration_intercept"] = float(diagnostic.intercept_[0])
        result["calibration_slope"] = float(diagnostic.coef_[0,0])
    edges=np.asarray(P_UP_EDGES)
    groups=[]
    for left,right in zip(edges[:-1],edges[1:]):
        mask=(p>=left)&(p<right)
        groups.append({"lower":float(left),"upper":min(float(right),1.0),"n":int(mask.sum()),
                       "observed_up":float(y[mask].mean()) if mask.any() else None,
                       "class_accuracy":float(((p[mask]>.5)==y[mask]).mean()) if mask.any() else None,
                       "mean_confidence":float(np.maximum(p[mask],1-p[mask]).mean()) if mask.any() else None})
    result["prob_up_groups"]=groups
    return result


def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_data(model_path, metadata_path):
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    from datetime import timedelta
    start=datetime.fromisoformat(metadata["data_range"]["start"])
    end=datetime.fromisoformat(metadata["data_range"]["end"])+timedelta(minutes=1)
    repository=Influx3MarketRepository.from_settings(Settings())
    try: candles=repository.get_candles("BTCUSDT","1m",start,end)
    finally: repository.close()
    if len(candles)!=metadata["data_range"]["closed_candles"]:
        raise ValueError("Historical candle count differs from model metadata")
    digest=candle_digest(candles)
    if digest!="b5138e394357b6a55133a730a402f03653c9b4046ddf630f9cb497ed2760904b":
        raise ValueError("Historical candles do not match the frozen walk-forward source")
    frame=build_training_frame(build_feature_frame(candles))
    split=chronological_split(frame)
    for name,part in (("train",split.train),("validation",split.validation),("test",split.test)):
        expected=metadata["split"][name]
        if len(part)!=expected["rows"] or part.index[0].isoformat()!=expected["start"] or part.index[-1].isoformat()!=expected["end"]:
            raise ValueError(f"{name} does not reproduce the frozen model split")
    bundle=joblib.load(model_path)
    if bundle["features"]!=list(FEATURE_COLUMNS) or bundle["model_version"]!="xgboost_v1":
        raise ValueError("Wrong model artifact or features")
    result={}
    for name,part in (("validation",split.validation),("test",split.test)):
        probability=bundle["estimator"].predict_proba(part[list(FEATURE_COLUMNS)])[:,1]
        result[name]={"y":part["target"].to_numpy(dtype=int),"p":probability,
                      "times":part.index,
                      "momentum":(part["return_1m"].to_numpy()>0).astype(int)}
    # Validate reconstructed XGBoost predictions against original metadata.
    for name in ("validation","test"):
        original=metadata["evaluation"][name]["accuracy"]
        reconstructed=class_metrics(result[name]["y"],result[name]["p"])["accuracy"]
        if abs(original-reconstructed)>1e-12:
            raise ValueError(f"{name} probabilities do not reproduce original model evaluation")
    return metadata,digest,result


def live_data(project_ref, cutoff):
    key=os.getenv("SUPABASE_KEY","") or _publishable_key(project_ref)
    _require_public_read_key(key)
    url=os.getenv("SUPABASE_URL","") or f"https://{project_ref}.supabase.co"
    with httpx.Client(timeout=30.0) as client:
        rows,count=_fetch_all(client,url,key,"live_evaluations",[
            ("select",EVALUATION_SELECT),("symbol","eq.BTCUSDT"),
            ("evaluated_at",f"lte.{cutoff.isoformat()}"),
            ("order","evaluated_at.asc,evaluation_id.asc")])
        observations,_=_observations(rows,"xgboost_v1")
        if not observations or len(rows)!=count:
            raise ValueError("No or incomplete LIVE cohort")
        from datetime import timedelta
        start=observations[0].prediction.timestamp-timedelta(minutes=1)
        stop=observations[-1].prediction.timestamp
        candles,_=_fetch_all(client,url,key,"candles",[
            ("select","open_time,close"),("symbol","eq.BTCUSDT"),("interval","eq.1m"),
            ("open_time",f"gte.{start.isoformat()}"),("open_time",f"lte.{stop.isoformat()}"),
            ("order","open_time.asc")])
    close={datetime.fromisoformat(row["open_time"].replace("Z","+00:00")):float(row["close"]) for row in candles}
    y=np.array([1 if o.evaluation.observed_class=="UP" else 0 for o in observations],dtype=int)
    p=np.array([o.prediction.prob_up for o in observations],dtype=float)
    cohort_digest=hashlib.sha256()
    for o in observations:
        cohort_digest.update(json.dumps([
            o.prediction.prediction_id,o.prediction.prob_up,o.prediction.prob_down,
            o.prediction.predicted_class,o.evaluation.observed_class,o.evaluation.correct,
            o.prediction.generated_at.isoformat(),o.evaluation.evaluated_at.isoformat(),
        ],separators=(",", ":")).encode("utf-8"))
        cohort_digest.update(b"\n")
    momentum=[]
    for o in observations:
        source=o.prediction.timestamp
        previous=close.get(source-timedelta(minutes=1))
        momentum.append(None if previous is None else int(o.evaluation.source_close>previous))
    return {"y":y,"p":p,"times":[o.prediction.timestamp for o in observations],
            "momentum":momentum,"timely":sum(o.prediction.timing_status=="TIMELY" for o in observations),
            "late":sum(o.prediction.timing_status=="LATE" for o in observations),
            "evaluated_at_end":max(o.evaluation.evaluated_at for o in observations).isoformat(),
            "persisted_rows":count,"cohort_sha256":cohort_digest.hexdigest()}


def _subset(data, start, stop):
    return {"y":data["y"][start:stop],"p":data["p"][start:stop],
            "times":data["times"][start:stop],"momentum":data["momentum"][start:stop]}


def temporal_calibration_split(data):
    """Chronological 2/3 fit, one purged label, final 1/3 holdout."""
    times=data["times"]
    if len(times)<1500 or any(a>=b for a,b in zip(times[:-1],times[1:])):
        raise ValueError("Calibration requires sufficient strictly ordered time data")
    boundary=len(times)*2//3
    fit=_subset(data,0,boundary-1)
    holdout=_subset(data,boundary,len(times))
    if fit["times"][-1]>=holdout["times"][0]:
        raise ValueError("Temporal calibration leakage")
    return fit,holdout


def _range(data):
    return {"n":len(data["y"]),"start":data["times"][0].isoformat(),"end":data["times"][-1].isoformat(),
            "actual_down":int((data["y"]==0).sum()),"actual_up":int(data["y"].sum())}


def _baselines(data, train_prior_up):
    y=data["y"]
    always_down=float((y==0).mean())
    pairs=[(a,b) for a,b in zip(y,data["momentum"]) if b is not None]
    return {"always_down_accuracy":always_down,
            "previous_movement_n":len(pairs),
            "previous_movement_accuracy":sum(a==b for a,b in pairs)/len(pairs) if pairs else None,
            "constant_train_prior_up":train_prior_up,
            "constant_train_prior_brier":float(brier_score_loss(y,np.full(len(y),train_prior_up))),
            "constant_train_prior_log_loss":float(log_loss(y,np.full(len(y),train_prior_up),labels=[0,1]))}


def analyze(model_path, metadata_path, *, project_ref, cutoff, phase1_cutoff):
    before=_sha256(model_path)
    metadata,digest,historical=historical_data(model_path,metadata_path)
    booster_config=json.loads(joblib.load(model_path)["estimator"].get_booster().save_config())["learner"]
    learner_params=booster_config["learner_model_param"]
    validation=historical["validation"]
    # Purge the last fit label: it uses the first holdout candle's close.
    calibration,validation_holdout=temporal_calibration_split(validation)
    platt,isotonic=fit_calibrators(calibration["y"],calibration["p"])
    datasets={"calibration_fit":calibration,"validation_holdout":validation_holdout,
              "test":historical["test"]}
    train_counts=metadata["split"]["train"]["class_distribution"]
    train_prior_up=train_counts["1"]/(train_counts["0"]+train_counts["1"])
    if project_ref:
        datasets["live_phase2"]=live_data(project_ref,cutoff)
    report={"analyzed_at_utc":datetime.now(timezone.utc).isoformat(),
            "phase1_cutoff_utc":phase1_cutoff,"phase2_cutoff_utc":cutoff.isoformat() if project_ref else None,
            "model_sha256":before,"historical_candle_digest":digest,
            "model_training_configuration":{
                "objective":metadata["parameters"]["objective"],
                "base_score":learner_params["base_score"],
                "scale_pos_weight":booster_config["objective"]["reg_loss_param"]["scale_pos_weight"],
                "fit_sample_weight":False,
                "note":"trainer.py calls fit(X_train, y_train) without sample_weight; no explicit class weighting",
            },
            "method":{
                "split":"Original chronological TRAIN/VALIDATION/TEST 70/15/15 with 1-row purge; first 2/3 VALIDATION fit Platt/isotonic, last 1/3 diagnostic holdout; additional 1-row purge between calibration fit and holdout",
                "model":"Frozen xgboost_v1 predict_proba; no model fit or mutation",
                "platt":"Logistic regression with no regularization on logit(original P_UP); monotonic positive slope required",
                "isotonic":"IsotonicRegression on original P_UP; clipping outside fitted range",
                "class_rule":"UP if P_UP > 0.50, exact 0.50 tie DOWN; threshold table uses P_UP > T",
                "ece":"10 bins; absolute observed minus mean predicted UP, weighted by bin count; equal-width on observed probability support and equal-frequency quantiles",
                "calibration_slope_intercept":"Descriptive logistic regression of label on logit(probability) within each evaluated dataset; not applied as calibrator",
                "confidence_gate":"max(P_UP,1-P_UP) >= T; accuracy on selected cases; 95% Wilson interval assumes independent rows and is optimistic under minute autocorrelation",
                "phase1_note":"FASE1 remains frozen. The Phase2 LIVE cutoff is independent; calibration and thresholds are never fitted/selected on LIVE.",
                "validation_reuse":"Original VALIDATION was already inspected for model diagnostics and related walk-forward research; holdout is exploratory, not an untouched independent selection set",
                "training_live_feature_difference":"Historical features use all preceding 30-day candles; LIVE PredictionService uses a rolling 120-candle window, affecting EWM warm-up. Production is unchanged."},
            "split_metadata":metadata["split"],
            "calibrators":{"platt":{"intercept":float(platt.intercept_[0]),"slope":float(platt.coef_[0,0])},
                           "isotonic":{"threshold_points":len(isotonic.X_thresholds_)}},
            "datasets":{}}
    for name,data in datasets.items():
        n=len(data["y"])
        dataset={"range":_range(data),"baselines":_baselines(data,train_prior_up),
                 "models":{},"temporal_halves":{}}
        if name=="live_phase2":
            dataset["timing"]={"timely":data["timely"],"late":data["late"]}
            dataset["persisted_rows"]=data["persisted_rows"]
            dataset["evaluated_at_end"]=data["evaluated_at_end"]
            dataset["cohort_sha256"]=data["cohort_sha256"]
        for model_name in ("uncalibrated","platt","isotonic"):
            p=apply_calibrator(model_name,data["p"],platt,isotonic)
            dataset["models"][model_name]=probability_metrics(data["y"],p)
            dataset["temporal_halves"][model_name]=[
                {"range":_range(_subset(data,a,b)),
                 "roc_auc":float(roc_auc_score(data["y"][a:b],p[a:b])) if len(np.unique(data["y"][a:b]))==2 else None,
                 "brier":float(brier_score_loss(data["y"][a:b],p[a:b])),
                 "accuracy":float(accuracy_score(data["y"][a:b],p[a:b]>.5))}
                for a,b in ((0,n//2),(n//2,n))]
        report["datasets"][name]=dataset
    if _sha256(model_path)!=before:
        raise RuntimeError("Production model artifact changed during analysis")
    return report


def _pct(value):
    return "N/A" if value is None else f"{100*value:.2f}%"


def markdown(report):
    lines=["# Diagnóstico de calibración XGBoost v1 — FASE 2", "",
           f"Corte FASE 1: {report['phase1_cutoff_utc']}; corte FASE 2: {report['phase2_cutoff_utc']}.",
           "", "Este estudio es OFFLINE. Ningún calibrador, threshold o dato LIVE alteró producción.",
           "", "## Metodología", ""]
    lines += [f"- **{key}**: {value}" for key,value in report["method"].items()]
    config=report["model_training_configuration"]
    lines += ["",f"Modelo: objective={config['objective']}; base_score={config['base_score']}; scale_pos_weight={config['scale_pos_weight']}; sin sample weights en `trainer.py`."]
    lines += ["", "## Lectura del resultado", ""]
    test=report["datasets"]["test"]["models"]
    lines.append(f"En TEST, XGBoost sin calibrar tiene ROC AUC {test['uncalibrated']['roc_auc']:.4f}, Brier {test['uncalibrated']['brier']:.5f} y ECE cuantil {test['uncalibrated']['ece_quantile']['ece']:.5f}. Platt cambia Brier a {test['platt']['brier']:.5f}, sin alterar el ROC AUC. Isotonic cambia Brier a {test['isotonic']['brier']:.5f}; sus saltos producen probabilidades extremas y deterioran Log Loss fuera del ajuste.")
    if "live_phase2" in report["datasets"]:
        live=report["datasets"]["live_phase2"]["models"]
        gate56=next(x for x in live["uncalibrated"]["confidence_gates"] if x["threshold"]==.56)
        gate60=next(x for x in live["uncalibrated"]["confidence_gates"] if x["threshold"]==.60)
        lines.append(f"En LIVE al corte: n={live['uncalibrated']['n']}, ROC AUC {live['uncalibrated']['roc_auc']:.4f}; filtro 56%: {gate56['n']} casos, {_pct(gate56['accuracy'])}; filtro 60%: {gate60['n']} casos, {_pct(gate60['accuracy'])}. El grupo 60% no tiene tamaño suficiente para respaldar una política.")
        historical60=next(x for x in report["datasets"]["validation_holdout"]["models"]["uncalibrated"]["confidence_gates"] if x["threshold"]==.60)
        lines.append(f"La composición del filtro 60% cambia: en el holdout de VALIDATION, {historical60['predicted_down']}/{historical60['n']} señales son DOWN; en LIVE, {gate60['predicted_up']}/{gate60['n']} son UP. No representa un subconjunto direccional estable.")
    lines += ["", "## Comparación por conjunto", "",
              "| Conjunto | Modelo | n | Brier | Log Loss | ECE cuantil | ROC AUC | AP UP | Accuracy | Balanced accuracy | Recall UP | Recall DOWN |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,data in report["datasets"].items():
        for model,metrics in data["models"].items():
            lines.append(f"| {name} | {model} | {metrics['n']} | {metrics['brier']:.5f} | {metrics['log_loss']:.5f} | {metrics['ece_quantile']['ece']:.5f} | {metrics.get('roc_auc',float('nan')):.4f} | {metrics.get('average_precision_up',float('nan')):.4f} | {_pct(metrics['accuracy'])} | {_pct(metrics['balanced_accuracy'])} | {_pct(metrics['recall_up'])} | {_pct(metrics['recall_down'])} |")
    for name,data in report["datasets"].items():
        lines += ["",f"## {name}","",f"Rango: {data['range']['start']} a {data['range']['end']} UTC; n={data['range']['n']}; UP real={data['range']['actual_up']}, DOWN real={data['range']['actual_down']}.","",
                  f"Baselines: siempre DOWN {_pct(data['baselines']['always_down_accuracy'])}; movimiento previo {_pct(data['baselines']['previous_movement_accuracy'])} (n={data['baselines']['previous_movement_n']}); probabilidad constante P(UP) TRAIN {_pct(data['baselines']['constant_train_prior_up'])} con Brier {data['baselines']['constant_train_prior_brier']:.5f} y Log Loss {data['baselines']['constant_train_prior_log_loss']:.5f}.", ""]
        for model,metrics in data["models"].items():
            confidence=metrics["confidence"]
            lines += [f"### {model}","", "| P(UP) threshold | Accuracy | Balanced | Precision UP | Recall UP | % pred UP |", "|---:|---:|---:|---:|---:|---:|"]
            for t in metrics["class_thresholds"]:
                lines.append(f"| {t['threshold']:.2f} | {_pct(t['accuracy'])} | {_pct(t['balanced_accuracy'])} | {_pct(t['precision_up'])} | {_pct(t['recall_up'])} | {_pct(t['predicted_up']/t['n'])} |")
            lines += ["", "| Confidence gate | n | Coverage | Accuracy | Siempre DOWN en seleccionadas | Wilson 95% |", "|---:|---:|---:|---:|---:|---|"]
            for g in metrics["confidence_gates"]:
                ci=g["wilson_95"]
                lines.append(f"| {g['threshold']:.2f} | {g['n']} | {_pct(g['coverage'])} | {_pct(g['accuracy'])} | {_pct(g['always_down_accuracy_selected'])} | {f'{_pct(ci[0])}–{_pct(ci[1])}' if ci else 'N/A'} |")
            lines += ["", "Reliability por cuantiles:", "", "| P(UP) intervalo | n | Media estimada | UP observado | Diferencia |", "|---|---:|---:|---:|---:|"]
            for b in metrics["reliability_quantile"]:
                lines.append(f"| {b['lower']:.4f}–{b['upper']:.4f} | {b['n']} | {_pct(b['mean_predicted_up'])} | {_pct(b['observed_up'])} | {_pct(b['gap'])} |")
            lines += ["",f"Confianza máxima media {_pct(confidence['mean'])}; mediana {_pct(confidence['median'])}; P75 {_pct(confidence['p75'])}; P90 {_pct(confidence['p90'])}; P95 {_pct(confidence['p95'])}; P99 {_pct(confidence['p99'])}; máximo {_pct(confidence['max'])}.",
                      "", "| P(UP) grupo | n | UP observado | Accuracy clase | Confianza media |", "|---|---:|---:|---:|---:|"]
            for g in metrics["prob_up_groups"]:
                lines.append(f"| {g['lower']:.2f}–{g['upper']:.2f} | {g['n']} | {_pct(g['observed_up'])} | {_pct(g['class_accuracy'])} | {_pct(g['mean_confidence'])} |")
            lines.append("")
        lines += ["Estabilidad por mitad temporal del conjunto:","", "| Modelo | Mitad | n | ROC AUC | Brier | Accuracy |", "|---|---:|---:|---:|---:|---:|"]
        for model,halves in data["temporal_halves"].items():
            for index,half in enumerate(halves,1):
                lines.append(f"| {model} | {index} | {half['range']['n']} | {half['roc_auc']:.4f} | {half['brier']:.5f} | {_pct(half['accuracy'])} |")
    lines += ["## Interpretación", "", "Los intervalos Wilson suponen independencia entre observaciones; en velas consecutivas pueden ser optimistas. El TEST y LIVE se reportan para evaluación, nunca para elegir un calibrador o threshold. P(UP) es probabilidad estimada de la clase, no probabilidad de rentabilidad.",""]
    return "\n".join(lines)


def coverage_svg(report):
    """Standalone, dependency-free coverage versus accuracy chart."""
    width,height=1000,530
    colors={"uncalibrated":"#2563eb","platt":"#16a34a","isotonic":"#db2777"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<text x="50" y="30" font-size="20" font-family="sans-serif">Cobertura vs. accuracy por filtro de confianza — FASE 2</text>',
           '<text x="50" y="52" font-size="12" font-family="sans-serif" fill="#555">Lectura descriptiva; los filtros no se seleccionaron con LIVE ni se aplicaron en producción</text>']
    for panel,(dataset,label) in enumerate((("test","TEST histórico"),("live_phase2","LIVE, corte FASE 2"))):
        if dataset not in report["datasets"]: continue
        left=65+panel*490; top=100; plot_width=400; plot_height=335
        data=report["datasets"][dataset]
        def xy(coverage,accuracy):
            return left+plot_width*coverage,top+plot_height*(.8-accuracy)/.4
        parts += [f'<text x="{left}" y="{top-12}" font-size="16" font-family="sans-serif">{label} (n={data["range"]["n"]})</text>',
                  f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#444"/>']
        for accuracy in (.4,.5,.6,.7,.8):
            y=xy(0,accuracy)[1]
            parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                      f'<text x="{left-32}" y="{y+4:.1f}" font-size="11" font-family="sans-serif">{accuracy:.0%}</text>']
        for coverage in (0,.25,.5,.75,1):
            x=xy(coverage,.4)[0]
            parts.append(f'<text x="{x-12:.1f}" y="{top+plot_height+20}" font-size="11" font-family="sans-serif">{coverage:.0%}</text>')
        baseline=data["baselines"]["always_down_accuracy"]
        baseline_y=xy(0,baseline)[1]
        parts.append(f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{left+plot_width}" y2="{baseline_y:.1f}" stroke="#777" stroke-dasharray="5 5"/>')
        for model,color in colors.items():
            rows=data["models"][model]["confidence_gates"]
            points=[xy(row["coverage"],row["accuracy"]) for row in rows if row["accuracy"] is not None]
            parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for row in rows:
                if row["threshold"] in (.56,.60) and row["accuracy"] is not None:
                    x,y=xy(row["coverage"],row["accuracy"])
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{left+145}" y="{top+plot_height+42}" font-size="12" font-family="sans-serif">Cobertura</text>')
    legend_x=300
    for model,color in colors.items():
        parts += [f'<line x1="{legend_x}" y1="490" x2="{legend_x+25}" y2="490" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{legend_x+31}" y="494" font-size="12" font-family="sans-serif">{model}</text>']
        legend_x+=155
    parts.append('</svg>')
    return "\n".join(parts)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref",default=os.getenv("SUPABASE_PROJECT_REF",""))
    parser.add_argument("--model",type=Path,default=ROOT/"models/xgboost_v1.joblib")
    parser.add_argument("--metadata",type=Path,default=ROOT/"models/xgboost_v1_metadata.json")
    parser.add_argument("--phase1",type=Path,default=ROOT/"reports/live_model_diagnostic.json")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/calibration_analysis_v1.json")
    parser.add_argument("--cutoff",help="Optional fixed UTC cutoff for reproducible LIVE cohort")
    args=parser.parse_args()
    phase1=json.loads(args.phase1.read_text(encoding="utf-8"))
    cutoff=datetime.fromisoformat(args.cutoff.replace("Z","+00:00")) if args.cutoff else datetime.now(timezone.utc)
    if cutoff.tzinfo is None:
        parser.error("--cutoff requires timezone")
    cutoff=cutoff.astimezone(timezone.utc)
    report=analyze(args.model,args.metadata,project_ref=args.project_ref,cutoff=cutoff,
                   phase1_cutoff=phase1["analyzed_at_utc"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report),encoding="utf-8")
    args.output.with_name("calibration_coverage_v1.svg").write_text(coverage_svg(report),encoding="utf-8")
    print(f"FASE2 cutoff={cutoff.isoformat()}; historical reproduced; datasets="+
          ",".join(f"{k}:{v['range']['n']}" for k,v in report["datasets"].items()))


if __name__=="__main__":
    main()
