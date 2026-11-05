"""Read-only TRAIN/VALIDATION/TEST/LIVE feature parity audit for xgboost_v1.

Requires the original local InfluxDB 3 candles and optionally a Supabase public
read key. Writes only aggregate reports, never raw Binance rows or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame, build_training_frame, chronological_split
from alpha_engine.ml.walk_forward import candle_digest
from alpha_engine.repositories.influx3 import Influx3MarketRepository
from scripts.live_model_diagnostic import _fetch_all, _publishable_key, _require_public_read_key

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/xgboost_v1.joblib"
METADATA = ROOT / "models/xgboost_v1_metadata.json"
HISTORICAL_DIGEST = "b5138e394357b6a55133a730a402f03653c9b4046ddf630f9cb497ed2760904b"
LOOKBACKS = (35, 60, 90, 120, 180, 240, 360, 720)
FEATURE_DEFINITIONS = {
    "return_1m": "close.pct_change(fill_method=None)",
    "rsi_14": "gain/loss EWM alpha=1/14, adjust=False, min_periods=14; zero-loss special cases",
    "ema_20": "close.ewm(span=20, adjust=False, min_periods=20).mean()",
    "macd": "EMA12(close)-EMA26(close); adjust=False; min_periods=12/26",
    "macd_signal": "MACD.ewm(span=9, adjust=False, min_periods=9).mean()",
    "macd_histogram": "MACD-MACD signal",
    "rolling_volatility": "return_1m.rolling(window=20,min_periods=20).std(ddof=0)",
    "volume_change": "volume.pct_change(fill_method=None), infinite values -> NaN",
}


def percentiles(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return {"n": int(len(array)), "mean": float(array.mean()), "std": float(array.std()),
            "median": float(np.median(array)), "p05": float(np.quantile(array,.05)),
            "p25": float(np.quantile(array,.25)), "p75": float(np.quantile(array,.75)),
            "p95": float(np.quantile(array,.95)), "min": float(array.min()), "max": float(array.max())}


def difference_summary(reference, replay, train_iqr):
    reference=np.asarray(reference,dtype=float); replay=np.asarray(replay,dtype=float)
    if reference.shape!=replay.shape or not len(reference):
        raise ValueError("Aligned, nonempty feature vectors required")
    finite=np.isfinite(reference)&np.isfinite(replay)
    if not finite.all():
        raise ValueError("Nonfinite feature parity row")
    absolute=np.abs(reference-replay)
    relative=absolute/np.maximum(np.abs(reference),1e-12)
    normalized=absolute/train_iqr if train_iqr>0 else np.full(len(absolute),np.nan)
    return {"n":len(absolute),"absolute":percentiles(absolute),"relative":percentiles(relative),
            "train_iqr_normalized":percentiles(normalized),
            "zero_difference":int((absolute==0).sum())}


def classify_difference(summary):
    if summary is None or summary["n"]<30:
        return "INDETERMINATE"
    if summary["absolute"]["max"]<=1e-10:
        return "PARITY_OK"
    scaled=summary["train_iqr_normalized"]
    if scaled is None:
        return "INDETERMINATE"
    return ("MINOR_NUMERICAL_DIFFERENCE" if scaled["p95"]<=.01 and scaled["max"]<=.05
            else "MATERIAL_MISMATCH")


def population_stability_index(train, other, bins=10):
    """Fixed TRAIN decile edges, clipped proportions; descriptive only."""
    train=np.asarray(train,dtype=float); other=np.asarray(other,dtype=float)
    edges=np.unique(np.quantile(train,np.linspace(0,1,bins+1)))
    if len(edges)<3:
        return None
    edges[0]=-np.inf;edges[-1]=np.inf
    first=np.histogram(train,bins=edges)[0]/len(train)
    second=np.histogram(other,bins=edges)[0]/len(other)
    first=np.maximum(first,1e-6);second=np.maximum(second,1e-6)
    return float(np.sum((second-first)*np.log(second/first)))


def distribution_shift(train, other):
    train=np.asarray(train,dtype=float);other=np.asarray(other,dtype=float)
    if not np.isfinite(train).all() or not np.isfinite(other).all() or not len(other):
        raise ValueError("Finite distributions required")
    iqr=float(np.quantile(train,.75)-np.quantile(train,.25))
    return {"summary":percentiles(other),"psi_train_deciles":population_stability_index(train,other),
            "ks_statistic":float(ks_2samp(train,other).statistic),
            "wasserstein":float(wasserstein_distance(train,other)),
            "wasserstein_over_train_iqr":float(wasserstein_distance(train,other)/iqr) if iqr>0 else None,
            "train_iqr":iqr}


def _load_historical(metadata):
    start=datetime.fromisoformat(metadata["data_range"]["start"])
    stop=datetime.fromisoformat(metadata["data_range"]["end"])+timedelta(minutes=1)
    repository=Influx3MarketRepository.from_settings(Settings())
    try:candles=repository.get_candles("BTCUSDT","1m",start,stop)
    finally:repository.close()
    if len(candles)!=metadata["data_range"]["closed_candles"] or candle_digest(candles)!=HISTORICAL_DIGEST:
        raise ValueError("Local historical candles do not reproduce the frozen model data")
    features=build_feature_frame(candles)
    split=chronological_split(build_training_frame(features))
    parts={"TRAIN":split.train,"VALIDATION":split.validation,"TEST":split.test}
    for name,part in parts.items():
        expected=metadata["split"][name.lower()]
        if len(part)!=expected["rows"] or part.index[0].isoformat()!=expected["start"] or part.index[-1].isoformat()!=expected["end"]:
            raise ValueError(f"{name} differs from frozen metadata")
    return candles,features,parts


def _historical_samples(candles,features,parts,estimator,iqr):
    indices=set(range(120,len(candles),30))
    for part in parts.values():
        boundary=features.index.get_loc(part.index[0])
        for offset in (-3,-2,-1,0,1,2,3,30,120):
            if 120<=boundary+offset<len(candles):indices.add(boundary+offset)
    rows=[]
    for index in sorted(indices):
        full=features.iloc[index][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(full).all():continue
        replay=build_feature_frame(candles[index-119:index+1]).iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(replay).all():raise ValueError("LIVE-style historical replay produced NaN/Inf")
        rows.append((index,full,replay))
    full_matrix=np.vstack([row[1] for row in rows]); replay_matrix=np.vstack([row[2] for row in rows])
    full_prob=estimator.predict_proba(pd.DataFrame(full_matrix,columns=FEATURE_COLUMNS))[:,1]
    replay_prob=estimator.predict_proba(pd.DataFrame(replay_matrix,columns=FEATURE_COLUMNS))[:,1]
    by_feature={}
    for position,name in enumerate(FEATURE_COLUMNS):
        entry=difference_summary(full_matrix[:,position],replay_matrix[:,position],iqr[name])
        entry["classification"]=classify_difference(entry)
        order=np.argsort(np.abs(full_matrix[:,position]-replay_matrix[:,position]))[-3:][::-1]
        entry["largest_examples"]=[{
            "source_candle_open_time":candles[rows[j][0]].open_time.isoformat(),
            "full":float(full_matrix[j,position]),"live_120":float(replay_matrix[j,position]),
            "absolute_difference":float(abs(full_matrix[j,position]-replay_matrix[j,position]))}
            for j in order]
        by_feature[name]=entry
    probability=difference_summary(full_prob,replay_prob,1.0)
    flips=int(np.sum((full_prob>.5)!=(replay_prob>.5)))
    return {"samples":len(rows),"start":candles[rows[0][0]].open_time.isoformat(),
            "end":candles[rows[-1][0]].open_time.isoformat(),"features":by_feature,
            "model_effect":{"prob_up_difference":probability,"class_flips":flips,
                            "class_flip_fraction":flips/len(rows),
                            "full_predicted_up":int((full_prob>.5).sum()),
                            "live_120_predicted_up":int((replay_prob>.5).sum())}},rows


def _convergence(candles,features,estimator,iqr):
    indices=list(range(720,len(candles),120))
    reference=np.vstack([features.iloc[i][list(FEATURE_COLUMNS)].to_numpy(dtype=float) for i in indices])
    rows=[]
    for lookback in LOOKBACKS:
        replay=[];valid=[]
        for index in indices:
            row=build_feature_frame(candles[index-lookback+1:index+1]).iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            replay.append(row);valid.append(np.isfinite(row).all())
        replay=np.vstack(replay);valid=np.asarray(valid,dtype=bool)
        entry={"lookback":lookback,"n":int(valid.sum()),"total":len(indices),"features":{}}
        if valid.any():
            for pos,name in enumerate(FEATURE_COLUMNS):
                entry["features"][name]=difference_summary(reference[valid,pos],replay[valid,pos],iqr[name])
            a=estimator.predict_proba(pd.DataFrame(reference[valid],columns=FEATURE_COLUMNS))[:,1]
            b=estimator.predict_proba(pd.DataFrame(replay[valid],columns=FEATURE_COLUMNS))[:,1]
            entry["prob_up_absolute"]=percentiles(np.abs(a-b))
            entry["class_flips"]=int(((a>.5)!=(b>.5)).sum())
        rows.append(entry)
    return rows


def _live_rows(project_ref,cutoff):
    key=os.getenv("SUPABASE_KEY","") or _publishable_key(project_ref)
    _require_public_read_key(key)
    url=os.getenv("SUPABASE_URL","") or f"https://{project_ref}.supabase.co"
    with httpx.Client(timeout=30.0) as client:
        candle_rows,candle_n=_fetch_all(client,url,key,"candles",[
            ("select","symbol,interval,open_time,close_time,open,high,low,close,volume"),
            ("symbol","eq.BTCUSDT"),("interval","eq.1m"),
            ("open_time",f"lte.{cutoff.isoformat()}"),("order","open_time.asc")])
        prediction_rows,prediction_n=_fetch_all(client,url,key,"predictions",[
            ("select","prediction_id,symbol,interval,candle_open_time,generated_at,prob_up,prob_down,predicted_class,model_version"),
            ("model_version","eq.xgboost_v1"),("generated_at",f"lte.{cutoff.isoformat()}"),
            ("order","candle_open_time.asc")])
    if len(candle_rows)!=candle_n or len(prediction_rows)!=prediction_n:
        raise ValueError("Incomplete frozen LIVE pagination")
    candles=[Candle.model_validate(row) for row in candle_rows]
    predictions=prediction_rows
    digest=hashlib.sha256()
    for row in candle_rows:
        digest.update(json.dumps(row,sort_keys=True,separators=(",",":"),default=str).encode()+b"\n")
    for row in prediction_rows:
        digest.update(json.dumps(row,sort_keys=True,separators=(",",":"),default=str).encode()+b"\n")
    return candles,predictions,digest.hexdigest()


def _contiguous_segments(candles):
    segments=[]; current=[]
    for candle in candles:
        if current and candle.open_time-current[-1].open_time!=timedelta(minutes=1):
            segments.append(current);current=[]
        current.append(candle)
    if current:segments.append(current)
    return segments


def _live_features(candles,predictions,estimator,train_iqr):
    segments=_contiguous_segments(candles)
    replay={};full={};candle_by_time={c.open_time:c for c in candles}
    for segment in segments:
        if len(segment)<120:continue
        full_frame=build_feature_frame(segment)
        for index in range(119,len(segment)):
            time=segment[index].open_time
            replay_frame=build_feature_frame(segment[index-119:index+1])
            replay[time]=replay_frame.iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            full[time]=full_frame.loc[time,list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    aligned=[]; missing=0
    for row in predictions:
        source=datetime.fromisoformat(row["candle_open_time"].replace("Z","+00:00"))
        if source not in replay:
            missing+=1;continue
        if row["predicted_class"]!=("UP" if float(row["prob_up"])>float(row["prob_down"]) else "DOWN"):
            raise ValueError("Persisted LIVE class/probabilities disagree")
        if not np.isfinite(replay[source]).all() or not np.isfinite(full[source]).all():
            raise ValueError("LIVE replay has nonfinite features")
        aligned.append((source,replay[source],full[source],float(row["prob_up"])))
    if not aligned:raise ValueError("No LIVE prediction has an available closed 120-candle window")
    matrix=np.vstack([row[1] for row in aligned]);full_matrix=np.vstack([row[2] for row in aligned])
    persisted=np.array([row[3] for row in aligned])
    calculated=estimator.predict_proba(pd.DataFrame(matrix,columns=FEATURE_COLUMNS))[:,1]
    full_prob=estimator.predict_proba(pd.DataFrame(full_matrix,columns=FEATURE_COLUMNS))[:,1]
    features={}
    for pos,name in enumerate(FEATURE_COLUMNS):
        item=difference_summary(full_matrix[:,pos],matrix[:,pos],train_iqr[name])
        item["classification"]=classify_difference(item)
        features[name]=item
    return {"persisted_predictions":len(predictions),"replayable_predictions":len(aligned),
            "missing_120_window":missing,"candle_segments":len(segments),
            "candle_segment_lengths":[len(segment) for segment in segments],
            "replayed_source_start":aligned[0][0].isoformat(),"replayed_source_end":aligned[-1][0].isoformat(),
            "persisted_vs_120_prob_up_abs":percentiles(np.abs(calculated-persisted)),
            "persisted_vs_120_class_flips":int(((calculated>.5)!=(persisted>.5)).sum()),
            "full_history_vs_120_prob_up_abs":percentiles(np.abs(full_prob-calculated)),
            "full_history_vs_120_class_flips":int(((full_prob>.5)!=(calculated>.5)).sum()),
            "feature_parity":features,
            "replayed_prob_up":persisted.tolist()},matrix,aligned


def _correlation(matrix,probabilities):
    matrix=np.asarray(matrix,dtype=float);probabilities=np.asarray(probabilities,dtype=float)
    return {name:float(np.corrcoef(matrix[:,i],probabilities)[0,1])
            for i,name in enumerate(FEATURE_COLUMNS)}


def analyze(*,project_ref="",cutoff=None):
    cutoff=(cutoff or datetime.now(timezone.utc)).astimezone(timezone.utc)
    metadata=json.loads(METADATA.read_text(encoding="utf-8"))
    bundle=joblib.load(MODEL)
    if bundle["features"]!=metadata["features"] or bundle["features"]!=list(FEATURE_COLUMNS):
        raise ValueError("Model feature names/order differs from metadata or production code")
    if bundle["model_version"]!="xgboost_v1":raise ValueError("Wrong model")
    if list(bundle["estimator"].classes_)!=[0,1]:
        raise ValueError("Probability column 1 must represent UP")
    model_digest=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    candles,feature_frame,parts=_load_historical(metadata)
    estimator=bundle["estimator"]
    train=parts["TRAIN"]
    iqr={name:float(train[name].quantile(.75)-train[name].quantile(.25)) for name in FEATURE_COLUMNS}
    parity,samples=_historical_samples(candles,feature_frame,parts,estimator,iqr)
    convergence=_convergence(candles,feature_frame,estimator,iqr)
    report={"analyzed_at_utc":datetime.now(timezone.utc).isoformat(),"live_cutoff_utc":cutoff.isoformat(),
            "historical_candle_digest":HISTORICAL_DIGEST,"model_sha256":model_digest,
            "model_version":bundle["model_version"],"feature_order":bundle["features"],
            "feature_contract":{name:{"implementation":"alpha_engine/features/indicators.py::build_feature_frame",
                "formula":FEATURE_DEFINITIONS[name],"input":"closed BTCUSDT 1m OHLCV in UTC",
                "train_history":"all earlier historical candles","live_history":"last 120 contiguous closed candles",
                "first_valid_zero_based_row":int(feature_frame.index.get_loc(feature_frame[name].first_valid_index())),
                "dtype":"float64","nan_handling":"no fill; TRAIN drops warmup rows; LIVE rejects non-finite final row"}
                for name in FEATURE_COLUMNS},
            "definition":"Identical build_feature_frame implementation; historical uses all preceding candles, LIVE exactly last 120 closed candles. Row timestamp is source candle open UTC.",
            "criteria":{
                "PARITY_OK":"maximum absolute difference <= 1e-10",
                "MINOR_NUMERICAL_DIFFERENCE":"not exact; P95 absolute difference / TRAIN IQR <= 1% and maximum <= 5%",
                "MATERIAL_MISMATCH":"otherwise (provided TRAIN IQR > 0)",
                "INDETERMINATE":"fewer than 30 matched rows or zero TRAIN IQR",
                "relative_difference":"absolute difference / max(abs(full-history value), 1e-12); near-zero reference values can inflate it",
                "convergence":"P95 absolute difference / TRAIN IQR <= 1% and no NaN for a lookback; model effect reported separately"},
            "historical":{"split_ranges":{name:{"n":len(part),"start":part.index[0].isoformat(),"end":part.index[-1].isoformat()} for name,part in parts.items()},
                          "parity_sample":parity,"lookback_convergence":convergence},
            "distribution_shift":{},"probability_shift":{},"feature_probability_correlation":{}}
    train_array=train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    for name,part in parts.items():
        values=part[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        probs=estimator.predict_proba(part[list(FEATURE_COLUMNS)])[:,1]
        report["distribution_shift"][name]={feature:distribution_shift(train_array[:,pos],values[:,pos])
                                            for pos,feature in enumerate(FEATURE_COLUMNS)}
        report["probability_shift"][name]={"summary":percentiles(probs),"predicted_up_fraction":float((probs>.5).mean()),
            "low_probability_lt_0_46":int((probs<.46).sum()),"balanced_0_48_to_0_52":int(((probs>=.48)&(probs<=.52)).sum())}
        report["feature_probability_correlation"][name]=_correlation(values,probs)
    if project_ref:
        live_candles,live_predictions,cohort_digest=_live_rows(project_ref,cutoff)
        live,matrix,aligned=_live_features(live_candles,live_predictions,estimator,iqr)
        probabilities=np.array([row[3] for row in aligned])
        report["live"]={key:value for key,value in live.items() if key!="replayed_prob_up"}
        report["live"].update({"candles":len(live_candles),"cohort_sha256":cohort_digest})
        report["distribution_shift"]["LIVE"]={feature:distribution_shift(train_array[:,pos],matrix[:,pos])
                                            for pos,feature in enumerate(FEATURE_COLUMNS)}
        report["probability_shift"]["LIVE"]={"summary":percentiles(probabilities),"predicted_up_fraction":float((probabilities>.5).mean()),
            "low_probability_lt_0_46":int((probabilities<.46).sum()),
            "balanced_0_48_to_0_52":int(((probabilities>=.48)&(probabilities<=.52)).sum())}
        report["feature_probability_correlation"]["LIVE"]=_correlation(matrix,probabilities)
    if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=model_digest:
        raise RuntimeError("Production model changed during read-only parity audit")
    return report


def markdown(report):
    lines=["# Feature parity TRAIN ↔ LIVE — FASE 3", "",
           f"Corte UTC: {report['live_cutoff_utc']}. Modelo: {report['model_version']}; SHA-256: `{report['model_sha256']}`.","",
           "No se cambió modelo, inferencia, target, horizonte ni storage productivo. El informe contiene agregados, no velas crudas.",
           "", "## Método", "", report["definition"],"",
           f"Muestra histórica: {report['historical']['parity_sample']['samples']} timestamps distribuidos por todo el rango. Fuente local verificada con digest `{report['historical_candle_digest']}`.","",
           "| Feature | Clasificación histórica | Mediana diferencia absoluta | P95 | Máximo | P95 / IQR TRAIN |", "|---|---|---:|---:|---:|---:|"]
    contract=["## Contrato de las ocho features","",
              "| Feature | Cálculo común TRAIN/LIVE | Primera fila válida (0-based) | Historia TRAIN / LIVE |",
              "|---|---|---:|---|"]
    for name,item in report["feature_contract"].items():
        contract.append(f"| {name} | {item['formula']} | {item['first_valid_zero_based_row']} | Completa / últimas 120 cerradas |")
    contract += ["", "La función y el orden float64 son idénticos. TRAIN elimina warm-up; LIVE rechaza NaN/Inf y exige 120 velas continuas. El target futuro no interviene.", ""]
    lines[-2:-2]=contract
    for name,item in report["historical"]["parity_sample"]["features"].items():
        a=item["absolute"];s=item["train_iqr_normalized"]
        lines.append(f"| {name} | {item['classification']} | {a['median']:.8g} | {a['p95']:.8g} | {a['max']:.8g} | {s['p95']:.4%} |")
    effect=report["historical"]["parity_sample"]["model_effect"]
    lines += ["",f"Impacto histórico del replay 120: {effect['class_flips']}/{report['historical']['parity_sample']['samples']} clases diferentes; P95 |ΔP(UP)|={effect['prob_up_difference']['absolute']['p95']:.6f}.",""]
    if "live" in report:
        live=report["live"]
        lines += ["## Replay de predicciones LIVE persistidas","",
                  f"Velas leídas: {live['candles']}; predicciones: {live['persisted_predictions']}; con 120 velas continuas: {live['replayable_predictions']}; sin ventana: {live['missing_120_window']}.",
                  f"Persistido vs replay 120: P95 |ΔP(UP)|={live['persisted_vs_120_prob_up_abs']['p95']:.8g}, máximo={live['persisted_vs_120_prob_up_abs']['max']:.8g}, flips={live['persisted_vs_120_class_flips']}.",
                  f"Historia disponible extensa vs replay 120: P95 |ΔP(UP)|={live['full_history_vs_120_prob_up_abs']['p95']:.6f}, flips={live['full_history_vs_120_class_flips']}.","",
                  "| Feature | Clasificación LIVE | P95 diferencia absoluta | Máximo |", "|---|---|---:|---:|"]
        for name,item in live["feature_parity"].items():
            lines.append(f"| {name} | {item['classification']} | {item['absolute']['p95']:.8g} | {item['absolute']['max']:.8g} |")
    lines += ["", "## Convergencia por lookback", "",
              "| Velas | Filas válidas | EMA20 P95 abs | RSI P95 abs | MACD P95 abs | Señal P95 abs | P95 |ΔP(UP)| | Flips |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["historical"]["lookback_convergence"]:
        features=row["features"]
        if not features:continue
        lines.append(f"| {row['lookback']} | {row['n']} | {features['ema_20']['absolute']['p95']:.7g} | {features['rsi_14']['absolute']['p95']:.7g} | {features['macd']['absolute']['p95']:.7g} | {features['macd_signal']['absolute']['p95']:.7g} | {row['prob_up_absolute']['p95']:.6f} | {row['class_flips']} |")
    lines += ["", "## Distribution shift respecto a TRAIN", "",
              "| Feature | VALIDATION PSI | TEST PSI | LIVE PSI | LIVE KS | LIVE Wasserstein / IQR TRAIN |",
              "|---|---:|---:|---:|---:|---:|"]
    for feature in FEATURE_COLUMNS:
        val=report["distribution_shift"]["VALIDATION"][feature]
        test=report["distribution_shift"]["TEST"][feature]
        live=report["distribution_shift"].get("LIVE",{}).get(feature)
        lines.append(f"| {feature} | {val['psi_train_deciles']:.3f} | {test['psi_train_deciles']:.3f} | {live['psi_train_deciles']:.3f} | {live['ks_statistic']:.3f} | {live['wasserstein_over_train_iqr']:.3f} |" if live else f"| {feature} | {val['psi_train_deciles']:.3f} | {test['psi_train_deciles']:.3f} | N/A | N/A | N/A |")
    lines += ["", "Los PSI y KS son descriptivos; sin p-values. Históricos usan features de historia extensa; LIVE usa las ventanas reales de 120. Correlaciones feature↔P(UP), cuantiles P05/P25/mediana/P75/P95, NaN y extremos están en el JSON.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref",default=os.getenv("SUPABASE_PROJECT_REF",""))
    parser.add_argument("--cutoff",help="UTC timestamp fixed for reproducible LIVE selection")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/feature_parity_v1.json")
    args=parser.parse_args()
    cutoff=datetime.fromisoformat(args.cutoff.replace("Z","+00:00")) if args.cutoff else datetime.now(timezone.utc)
    if cutoff.tzinfo is None:parser.error("--cutoff requires timezone")
    report=analyze(project_ref=args.project_ref,cutoff=cutoff)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report),encoding="utf-8")
    print(f"Parity report: historical={report['historical']['parity_sample']['samples']} LIVE={report.get('live',{}).get('replayable_predictions','N/A')} cutoff={cutoff.isoformat()}")


if __name__=="__main__":main()
