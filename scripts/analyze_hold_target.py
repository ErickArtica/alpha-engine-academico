"""Offline ternary-target cohort and descriptive movement analysis.

The eight model inputs come from the frozen Phase 4 feature frame. The next
close is joined only to construct labels, never to construct model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.features import FEATURE_COLUMNS, chronological_split

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data/experiments/tfi/experimental_labeled.jsonl"
BACKFILL = ROOT / "data/experiments/tfi/joined_klines_tfi.jsonl"
PHASE4 = ROOT / "reports/model_ablation_tfi_v1.json"
BACKFILL_REPORT = ROOT / "reports/tfi_backfill_v1.json"
META = ROOT / "models/xgboost_v1_metadata.json"
REPORT = ROOT / "reports/target_hold_analysis_v1.json"
DELTAS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
LABELS = {0: "DOWN", 1: "HOLD", 2: "UP"}
PRODUCT_SHA = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def future_return_bps(source_close: np.ndarray, next_close: np.ndarray) -> np.ndarray:
    source = np.asarray(source_close, dtype=float)
    target = np.asarray(next_close, dtype=float)
    if source.shape != target.shape or not np.isfinite(source).all() or not np.isfinite(target).all() or (source <= 0).any():
        raise ValueError("Invalid aligned source/next closes")
    return (target / source - 1.0) * 10_000.0


def ternary_labels(returns_bps: np.ndarray, delta: float) -> np.ndarray:
    values = np.asarray(returns_bps, dtype=float)
    if delta < 0 or not np.isfinite(delta) or not np.isfinite(values).all():
        raise ValueError("Delta and future returns must be finite; delta >= 0")
    return np.where(values > delta, 2, np.where(values < -delta, 0, 1)).astype(np.int8)


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError("Historical timestamp is not explicit UTC")
    return stamp.tz_convert("UTC")


def load_frozen_cohort():
    """Verify both frozen raw fingerprints and the original chronological splits."""
    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    backfill_report = json.loads(BACKFILL_REPORT.read_text(encoding="utf-8"))
    meta = json.loads(META.read_text(encoding="utf-8"))
    if sha(ROOT / "models/xgboost_v1.joblib") != PRODUCT_SHA:
        raise ValueError("Production model SHA differs from frozen reference")
    if sha(LABELED) != phase4["dataset"]["sha256"] or sha(BACKFILL) != backfill_report["dataset_sha256"]:
        raise ValueError("Frozen dataset fingerprint differs")
    if list(FEATURE_COLUMNS) != meta["features"] or meta["horizon"] != "1m":
        raise ValueError("Production feature order or horizon differs")

    candles = {}
    with BACKFILL.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stamp = _utc(row["source_candle_open_time"])
            if stamp in candles or _utc(row["source_candle_close_time"]) != stamp + timedelta(minutes=1, milliseconds=-1):
                raise ValueError("Duplicate or misaligned source candle")
            if _utc(row["feature_cutoff_time"]) > _utc(row["source_candle_close_time"]):
                raise ValueError("Future feature cutoff")
            candles[stamp] = float(row["close"])

    records = []
    with LABELED.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stamp = _utc(row["source_candle_open_time"])
            following = stamp + timedelta(minutes=1)
            if stamp not in candles or following not in candles:
                raise ValueError("Target candle missing or nonconsecutive")
            source = float(row["close"])
            if not np.isclose(source, candles[stamp], atol=1e-8, rtol=1e-12):
                raise ValueError("Source close differs from frozen backfill")
            features = row["features"]
            if any(name not in features for name in FEATURE_COLUMNS):
                raise ValueError("Frozen model feature missing")
            records.append({"source_candle_open_time": stamp, "source_close": source,
                            "next_close": candles[following], "binary_target": int(row["target"]),
                            **{name: float(features[name]) for name in FEATURE_COLUMNS}})
    frame = pd.DataFrame(records).set_index("source_candle_open_time")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or len(frame) != meta["data_range"]["labeled_rows"]:
        raise ValueError("Frozen labeled cohort differs")
    frame["future_return_bps"] = future_return_bps(frame["source_close"], frame["next_close"])
    if not np.array_equal(frame["binary_target"].to_numpy(), (frame["next_close"] > frame["source_close"]).astype(int).to_numpy()):
        raise ValueError("Reconstructed label differs from productive binary target")
    if not np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model features")
    split = chronological_split(frame)
    for name in ("train", "validation", "test"):
        part = getattr(split, name)
        frozen = meta["split"][name]
        if len(part) != frozen["rows"] or part.index[0].isoformat() != frozen["start"] or part.index[-1].isoformat() != frozen["end"]:
            raise ValueError(f"Frozen {name} split differs")
    return frame, split, meta, phase4


def describe_target(part: pd.DataFrame, delta: float) -> dict:
    movement = part["future_return_bps"].to_numpy(dtype=float)
    labels = ternary_labels(movement, delta)
    counts = {LABELS[i]: int((labels == i).sum()) for i in LABELS}
    return {"n": len(part), "counts": counts,
            "fractions": {name: n / len(part) for name, n in counts.items()},
            "inside_hold_count": counts["HOLD"], "inside_hold_fraction": counts["HOLD"] / len(part)}


def analyze(*, include_test: bool = False) -> dict:
    frame, split, meta, phase4 = load_frozen_cohort()
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    split_info = {}
    for name in names:
        part = getattr(split, name)
        movement = part["future_return_bps"].to_numpy(dtype=float)
        split_info[name] = {
            "n": len(part), "start": part.index[0].isoformat(), "end": part.index[-1].isoformat(),
            "binary_target": {"DOWN": int((part["binary_target"] == 0).sum()), "UP": int((part["binary_target"] == 1).sum())},
            "exact_ties": int((part["source_close"] == part["next_close"]).sum()),
            "abs_movement_bps": {"p25": float(np.quantile(np.abs(movement), .25)),
                                  "median": float(np.median(np.abs(movement))),
                                  "p75": float(np.quantile(np.abs(movement), .75))},
            "deltas": {str(d): describe_target(part, d) for d in DELTAS},
        }
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "final_descriptive" if include_test else "development_only",
            "commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "production_model_sha256": PRODUCT_SHA, "source_dataset_sha256": phase4["dataset"]["sha256"],
            "backfill_sha256": sha(BACKFILL), "target_binary": meta["target_definition"],
            "target_ternary": "future_return_bps=(next_close/source_close-1)*10000; DOWN if <-delta, UP if >+delta, HOLD otherwise",
            "target_only_next_close": True, "features": list(FEATURE_COLUMNS), "deltas_bps": DELTAS,
            "splits": split_info,
            "economic_note": "Movement thresholds are statistical; fees, spread, slippage, latency and execution are not incorporated."}


def markdown(report: dict) -> str:
    lines = ["# Target HOLD: distribución de movimiento y clases", "",
             f"Generado: {report['generated_at_utc']} UTC. Etapa: {report['stage']}. Dataset `{report['source_dataset_sha256']}`.", "",
             "Target: DOWN si retorno futuro < −δ; UP si > +δ; HOLD en [−δ,+δ]. Horizonte: siguiente cierre de 1m. Las ocho features no incluyen el siguiente cierre.", "",
             "| δ (bps) | Split | DOWN | HOLD | UP | HOLD % |", "|---:|---|---:|---:|---:|---:|"]
    for delta in DELTAS:
        for split_name, split in report["splits"].items():
            row = split["deltas"][str(delta)]
            count = row["counts"]
            lines.append(f"| {delta:g} | {split_name.upper()} | {count['DOWN']} | {count['HOLD']} | {count['UP']} | {row['inside_hold_fraction']:.2%} |")
    lines += ["", "Los umbrales son estadísticos, no umbrales demostrados de rentabilidad. TEST se describió sólo después de congelar los candidatos.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-test", action="store_true", help="Only after candidate lock: add TEST descriptive distribution")
    args = parser.parse_args()
    result = analyze(include_test=args.include_test)
    write_json(REPORT if args.include_test else REPORT.with_name("target_hold_development_distribution_v1.json"), result)
    if args.include_test:
        REPORT.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(result["stage"], {name: split["n"] for name, split in result["splits"].items()})
