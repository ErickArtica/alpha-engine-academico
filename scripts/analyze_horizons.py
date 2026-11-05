"""Construct causal H1/H3/H5 binary targets from frozen source-minute features.

Only source-close and target-close enter a label. Future prices never enter X.
Private raw datasets remain under gitignored data/experiments/horizon/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.features import FEATURE_COLUMNS, chronological_split
from scripts.analyze_hold_target import BACKFILL, PRODUCT_SHA, ROOT, load_frozen_cohort, sha, write_json

HORIZONS = (1, 3, 5)
PRIVATE = ROOT / "data/experiments/horizon"
REPORT = ROOT / "reports/horizon_analysis_v1.json"
DEVELOPMENT_DESCRIPTION = ROOT / "reports/horizon_development_description_v1.json"


def future_target(source_close: np.ndarray, target_close: np.ndarray) -> np.ndarray:
    source = np.asarray(source_close, dtype=float)
    target = np.asarray(target_close, dtype=float)
    if source.shape != target.shape or not np.isfinite(source).all() or not np.isfinite(target).all() or (source <= 0).any():
        raise ValueError("Invalid aligned source and horizon closes")
    return (target > source).astype(np.int8)


def load_candle_closes() -> dict[pd.Timestamp, float]:
    rows = {}
    with BACKFILL.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            t = pd.Timestamp(item["source_candle_open_time"])
            if t.tzinfo is None or t.utcoffset() != timedelta(0) or t in rows:
                raise ValueError("Invalid source candle UTC or duplicate")
            if pd.Timestamp(item["source_candle_close_time"]) != t + timedelta(minutes=1, milliseconds=-1):
                raise ValueError("Invalid source close timestamp")
            rows[t] = float(item["close"])
    ordered = sorted(rows)
    if any(right - left != timedelta(minutes=1) for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Frozen candle sequence has gaps")
    return rows


def build_horizon_frames() -> tuple[dict[int, pd.DataFrame], dict]:
    source, _, metadata, previous = load_frozen_cohort()
    closes = load_candle_closes()
    if len(closes) != metadata["data_range"]["closed_candles"]:
        raise ValueError("Frozen candle count differs")
    output = {}
    for horizon in HORIZONS:
        rows = []
        for t, row in source.iterrows():
            target_time = t + timedelta(minutes=horizon)
            if target_time not in closes:
                continue
            source_close = float(row["source_close"])
            target_close = closes[target_time]
            source_close_time = t + timedelta(minutes=1, milliseconds=-1)
            target_close_time = target_time + timedelta(minutes=1, milliseconds=-1)
            if target_close_time <= source_close_time:
                raise ValueError("Target is not future to source")
            data = {"source_candle_open_time": t, "source_candle_close_time": source_close_time,
                    "target_candle_open_time": target_time, "target_candle_close_time": target_close_time,
                    "source_close": source_close, "future_close": target_close,
                    **{name: float(row[name]) for name in FEATURE_COLUMNS}}
            rows.append(data)
        frame = pd.DataFrame(rows).set_index("source_candle_open_time")
        if frame.empty or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError("Horizon dataset is empty or unsorted")
        frame["target"] = future_target(frame["source_close"], frame["future_close"])
        frame["future_return_bps"] = (frame["future_close"] / frame["source_close"] - 1) * 10_000
        if not np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite features")
        output[horizon] = frame
    if not output[5].index.equals(output[3].index[:len(output[5])]) or not output[5].index.equals(output[1].index[:len(output[5])]):
        raise ValueError("Natural horizons do not share the expected initial common cohort")
    if not output[1].index.equals(source.index) or not np.array_equal(output[1]["target"].to_numpy(), source["binary_target"].to_numpy()):
        raise ValueError("H1 does not reproduce productive binary target")
    return output, {"metadata": metadata, "phase4": previous, "source_rows": len(source), "candle_rows": len(closes)}


def split_horizon(frame: pd.DataFrame, horizon: int):
    """Purge h source rows so no training target touches validation/test sources."""
    if horizon not in HORIZONS:
        raise ValueError("Unsupported horizon")
    parts = chronological_split(frame, purge_rows=horizon)
    for past, future in ((parts.train, parts.validation), (parts.validation, parts.test)):
        if past.index[-1] + timedelta(minutes=horizon) >= future.index[0]:
            raise ValueError("Future label crosses split boundary")
    return parts


def common_frames(frames: dict[int, pd.DataFrame]):
    index = frames[5].index
    common = {h: frame.loc[index].copy() for h, frame in frames.items()}
    if any(not part.index.equals(index) for part in common.values()):
        raise ValueError("Common cohort timestamps differ")
    return common


def common_splits(common: dict[int, pd.DataFrame]):
    """Use max-horizon purge 5 for all models, ensuring identical X timestamps."""
    splits = {h: chronological_split(frame, purge_rows=max(HORIZONS)) for h, frame in common.items()}
    for name in ("train", "validation", "test"):
        indices = [getattr(split, name).index for split in splits.values()]
        if not indices[0].equals(indices[1]) or not indices[0].equals(indices[2]):
            raise ValueError("Common split timestamps differ")
    for horizon, split in splits.items():
        for past, future in ((split.train, split.validation), (split.validation, split.test)):
            if past.index[-1] + timedelta(minutes=horizon) >= future.index[0]:
                raise ValueError("Common purge fails horizon boundary")
    return splits


def fingerprint_dataset(frame: pd.DataFrame, horizon: int) -> dict:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = PRIVATE / f"h{horizon}_binary.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for t, row in frame.iterrows():
            item = {"symbol": "BTCUSDT", "interval": "1m", "horizon_minutes": horizon,
                    "source_candle_open_time": t.isoformat(),
                    "source_candle_close_time": row["source_candle_close_time"].isoformat(),
                    "target_candle_open_time": row["target_candle_open_time"].isoformat(),
                    "target_candle_close_time": row["target_candle_close_time"].isoformat(),
                    "source_close": float(row["source_close"]), "future_close": float(row["future_close"]),
                    "target": int(row["target"]),
                    "features": {name: float(row[name]) for name in FEATURE_COLUMNS}}
            handle.write(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "rows": len(frame),
            "start": frame.index[0].isoformat(), "end": frame.index[-1].isoformat()}


def describe_part(frame: pd.DataFrame) -> dict:
    bps = frame["future_return_bps"].to_numpy(dtype=float)
    absolute = np.abs(bps)
    y = frame["target"].to_numpy(dtype=int)
    return {"rows": len(frame), "first_source": frame.index[0].isoformat(),
            "last_source": frame.index[-1].isoformat(),
            "DOWN": int((y == 0).sum()), "UP": int((y == 1).sum()),
            "DOWN_fraction": float((y == 0).mean()), "UP_fraction": float((y == 1).mean()),
            "exact_ties": int((frame["future_close"] == frame["source_close"]).sum()),
            "return_bps": {"mean": float(np.mean(bps)), "median": float(np.median(bps)),
                           "median_absolute": float(np.median(absolute)), "p25": float(np.quantile(bps, .25)),
                           "p75": float(np.quantile(bps, .75)), "p90_absolute": float(np.quantile(absolute, .9)),
                           "p95_absolute": float(np.quantile(absolute, .95))},
            "absolute_below_bps": {str(th): {"count": int((absolute < th).sum()),
                                               "fraction": float((absolute < th).mean())}
                                   for th in (1, 2, 3, 5)}}


def target_concordance(common: dict[int, pd.DataFrame], index: pd.Index) -> dict:
    values = {h: common[h].loc[index, "target"].to_numpy(dtype=int) for h in HORIZONS}
    result = {}
    for left, right in ((1, 3), (1, 5), (3, 5)):
        a, b = values[left], values[right]
        result[f"H{left}_vs_H{right}"] = {
            "same_fraction": float((a == b).mean()),
            "real_rows_down_up_future_columns_down_up": [[int(((a == x) & (b == y)).sum()) for y in (0, 1)] for x in (0, 1)],
            "P_future_UP_given_source_UP": float(b[a == 1].mean()) if (a == 1).any() else None,
            "P_future_DOWN_given_source_DOWN": float((b[a == 0] == 0).mean()) if (a == 0).any() else None,
        }
    return result


def analyze(*, include_test: bool = False) -> dict:
    frames, frozen = build_horizon_frames()
    common = common_frames(frames)
    natural_splits = {h: split_horizon(frame, h) for h, frame in frames.items()}
    aligned_splits = common_splits(common)
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "stage": "final_after_candidate_lock" if include_test else "development_only",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "production_model_sha256": PRODUCT_SHA,
              "source_dataset_sha256": frozen["phase4"]["dataset"]["sha256"],
              "features": list(FEATURE_COLUMNS), "source_interval": "1m",
              "target": "UP iff close(t+h)>close(t); tie or fall DOWN; h in 1,3,5 closed source minutes",
              "label_dependence": "Adjacent H3/H5 targets overlap future time intervals; rows are not independent.",
              "source_rows_with_H1": frozen["source_rows"], "source_closed_candles": frozen["candle_rows"],
              "common_cohort": {"rows": len(common[5]), "start": common[5].index[0].isoformat(),
                                "end": common[5].index[-1].isoformat(), "purge_rows_each_boundary": 5,
                                "timestamps_sha256": hashlib.sha256(("\n".join(t.isoformat() for t in common[5].index) + "\n").encode()).hexdigest()},
              "horizons": {}}
    for h, frame in frames.items():
        report["horizons"][f"H{h}"] = {
            "natural_dataset": fingerprint_dataset(frame, h),
            "lost_rows_from_H1": len(frames[1]) - len(frame),
            "lost_initial_rows_from_H1": 0,
            "lost_final_rows_from_H1": len(frames[1]) - len(frame),
            "common_rows": len(common[h]),
            "natural_purge_rows": h,
            "natural_splits": {name: describe_part(getattr(natural_splits[h], name)) for name in names},
            "common_splits": {name: describe_part(getattr(aligned_splits[h], name)) for name in names}}
    report["common_target_concordance"] = {
        name: target_concordance(common, aligned_splits[1].validation.index if name == "validation" else aligned_splits[1].test.index)
        for name in (("validation", "test") if include_test else ("validation",))}
    return report


def markdown(report: dict) -> str:
    lines = ["# Horizontes binarios H1, H3 y H5 — FASE 6", "",
             f"Generado: {report['generated_at_utc']} UTC; etapa {report['stage']}. Producción congelada: `{report['production_model_sha256']}`.", "",
             "H3 significa comparar el cierre fuente con el cierre de la vela situada tres minutos después; no significa tocar un máximo en los tres minutos. H5 sigue la misma semántica.", "",
             f"Common cohort: {report['common_cohort']['rows']} timestamps, {report['common_cohort']['start']} a {report['common_cohort']['end']}; purga de 5 minutos para todos en la comparación principal.", "",
             "| Horizonte | Filas naturales | Perdidas vs H1 | SHA-256 dataset |", "|---|---:|---:|---|"]
    for h in HORIZONS:
        row = report["horizons"][f"H{h}"]
        lines.append(f"| H{h} | {row['natural_dataset']['rows']} | {row['lost_rows_from_H1']} | `{row['natural_dataset']['sha256']}` |")
    lines += ["", "| Horizonte | Split común | UP % | Empates | Mediana |retorno| bps | |retorno| <1 bp |", "|---|---|---:|---:|---:|---:|"]
    for h in HORIZONS:
        for name, part in report["horizons"][f"H{h}"]["common_splits"].items():
            lines.append(f"| H{h} | {name.upper()} | {part['UP_fraction']:.2%} | {part['exact_ties']} | {part['return_bps']['median_absolute']:.3f} | {part['absolute_below_bps']['1']['fraction']:.2%} |")
    lines += ["", "H3/H5 tienen etiquetas solapadas; las features conservan exactamente las ocho columnas productivas y terminan en el cierre fuente. Ninguna feature futura entra en X.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-test", action="store_true", help="Only after candidate lock")
    args = parser.parse_args()
    output = analyze(include_test=args.include_test)
    destination = REPORT if args.include_test else DEVELOPMENT_DESCRIPTION
    write_json(destination, output)
    destination.with_suffix(".md").write_text(markdown(output), encoding="utf-8")
    print(output["stage"], {h: v["natural_dataset"]["rows"] for h, v in output["horizons"].items()})
