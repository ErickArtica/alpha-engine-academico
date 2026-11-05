"""Paired, offline H1/H5 Alpha Vantage sentiment ablation.

Development never scores TEST. `freeze` locks candidate feature sets and hashes;
`final` verifies the lock before scoring TEST. LIVE and production are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.ml.trainer import XGB_PARAMS
from scripts.analyze_hold_target import PRODUCT_SHA, ROOT, sha, write_json
from scripts.analyze_horizons import build_horizon_frames, common_frames, common_splits
from scripts.build_news_sentiment_features import FEATURES as NEWS_FEATURES, FEATURE_REPORT, PRIVATE, load_articles
from scripts.run_horizon_experiment import common_expanding_folds, score

DEVELOPMENT = ROOT / "reports/news_sentiment_development_v1.json"
LOCK = ROOT / "reports/news_sentiment_candidate_lock_v1.json"
FINAL = ROOT / "reports/news_sentiment_ablation_v1.json"
WALK = ROOT / "reports/news_sentiment_walk_forward_v1.json"
TECHNICAL = tuple(FEATURE_COLUMNS)
PRESENCE = ("news_count_15m", "news_count_60m", "minutes_since_last_news")
SCORE_ONLY = ("btc_sentiment_15m", "btc_sentiment_60m")


def code_sha() -> str:
    digest = hashlib.sha256()
    for relative in ("scripts/fetch_alpha_vantage_news.py", "scripts/build_news_sentiment_features.py",
                     "scripts/run_news_sentiment_ablation.py", "alpha_engine/ml/trainer.py"):
        digest.update(relative.encode() + b"\0" + (ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def load_paired_frames() -> tuple[dict[int, pd.DataFrame], dict]:
    path = PRIVATE / "sentiment_common_cohort.jsonl"
    metadata = json.loads(FEATURE_REPORT.read_text(encoding="utf-8"))
    if sha(path) != metadata["dataset_sha256"]:
        raise ValueError("Private news feature dataset fingerprint changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    index = pd.DatetimeIndex([row["source_candle_open_time"] for row in rows])
    news = pd.DataFrame([{name: row["features"][name] for name in NEWS_FEATURES} for row in rows], index=index)
    cutoffs = pd.DatetimeIndex([row["sentiment_cutoff_time"] for row in rows])
    source_closes = pd.DatetimeIndex([row["source_candle_close_time"] for row in rows])
    if index.tz is None or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Invalid news dataset timeline")
    if (cutoffs[~cutoffs.isna()] > source_closes[~cutoffs.isna()]).any():
        raise ValueError("News feature cutoff uses future information")
    frames, _ = build_horizon_frames()
    common = common_frames(frames)
    for h in (1, 5):
        if not common[h].index.equals(news.index):
            raise ValueError("News and H1/H5 timestamp cohorts differ")
        common[h] = common[h].join(news, validate="one_to_one")
        if common[h][list(PRESENCE[:2])].isna().any().any():
            raise ValueError("News counts cannot be missing")
    return common, metadata


def x_for(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    x = frame.loc[:, list(columns)]
    if list(x.columns) != list(columns) or np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Feature order, infinity or unauthorized feature")
    if not set(columns).issubset(set(TECHNICAL) | set(NEWS_FEATURES)):
        raise ValueError("Unknown feature in news experiment")
    if x[list(set(columns) & set(TECHNICAL))].isna().any().any():
        raise ValueError("Technical feature missing")
    return x


def fit(frame: pd.DataFrame, columns: tuple[str, ...]) -> XGBClassifier:
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(x_for(frame, columns), frame["target"].to_numpy(dtype=int))
    return model


def predict(model: XGBClassifier, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    probabilities = model.predict_proba(x_for(frame, columns))[:, 1]
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability output")
    return probabilities


def model_result(train: pd.DataFrame, validation: pd.DataFrame, columns: tuple[str, ...]) -> tuple[dict, np.ndarray]:
    model = fit(train, columns)
    p = predict(model, validation, columns)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    count = booster.get_score(importance_type="weight")
    return {"features": list(columns), "metrics": score(validation, p),
            "gain": {name: float(gain.get(name, 0)) for name in columns},
            "split_count": {name: int(count.get(name, 0)) for name in columns}}, p


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, *, repetitions: int = 400,
                     block: int = 120) -> dict:
    if len(y) != len(a) or len(y) != len(b) or len(y) < 2 * block:
        raise ValueError("Unpaired or insufficient bootstrap inputs")
    rng = np.random.default_rng(51)
    values = {name: [] for name in ("roc_auc", "balanced_accuracy", "macro_f1", "accuracy")}
    for _ in range(repetitions):
        starts = rng.integers(0, len(y), size=int(np.ceil(len(y) / block)))
        idx = np.concatenate([(start + np.arange(block)) % len(y) for start in starts])[:len(y)]
        yy, aa, bb = y[idx], a[idx], b[idx]
        if len(set(yy)) != 2:
            continue
        for name, func in (("roc_auc", lambda p: roc_auc_score(yy, p)),
                           ("balanced_accuracy", lambda p: balanced_accuracy_score(yy, p > .5)),
                           ("macro_f1", lambda p: f1_score(yy, p > .5, average="macro")),
                           ("accuracy", lambda p: accuracy_score(yy, p > .5))):
            values[name].append(float(func(bb) - func(aa)))
    return {"method": "paired circular block bootstrap; 120-minute blocks (>=H5 and 60m news window); 400 draws; seed 51",
            "delta_B_minus_A": {name: {"low_95": float(np.quantile(data, .025)),
                                        "high_95": float(np.quantile(data, .975)),
                                        "resamples": len(data)} for name, data in values.items()}}


def models() -> dict[str, tuple[str, ...]]:
    return {"A_control": TECHNICAL, "B_sentiment": TECHNICAL + NEWS_FEATURES,
            "C_sentiment_only": NEWS_FEATURES, "N_presence_only": TECHNICAL + PRESENCE}


def development() -> dict:
    frames, news_metadata = load_paired_frames()
    splits = common_splits(frames)
    report = {"stage": "development_only", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "product_sha256": sha(ROOT / "models/xgboost_v1.joblib"),
              "news_dataset_sha256": news_metadata["dataset_sha256"],
              "model_spec": {"hyperparameters": XGB_PARAMS, "threshold": 0.5,
                             "technical_features": TECHNICAL, "news_features": NEWS_FEATURES},
              "validation": {}, "walk_forward": []}
    if report["product_sha256"] != PRODUCT_SHA:
        raise ValueError("Product model hash changed")
    for h in (1, 5):
        train, valid = splits[h].train, splits[h].validation
        out, probs = {}, {}
        for name, columns in models().items():
            out[name], probs[name] = model_result(train, valid, columns)
        out["bootstrap_B_minus_A"] = paired_bootstrap(valid["target"].to_numpy(dtype=int),
                                                        probs["A_control"], probs["B_sentiment"])
        out["paired_rows"] = len(valid)
        out["validation_range"] = [valid.index[0].isoformat(), valid.index[-1].isoformat()]
        report["validation"][f"H{h}"] = out
    dev_frames = {h: pd.concat((splits[h].train, splits[h].validation)).sort_index() for h in (1, 5)}
    for n, sample_train, sample_valid in common_expanding_folds(dev_frames[1]):
        fold = {"fold": n, "validation_start": sample_valid.index[0].isoformat(),
                "validation_end": sample_valid.index[-1].isoformat(), "horizons": {}}
        for h in (1, 5):
            train = dev_frames[h].loc[sample_train.index]
            valid = dev_frames[h].loc[sample_valid.index]
            control, _ = model_result(train, valid, TECHNICAL)
            sentiment, _ = model_result(train, valid, TECHNICAL + NEWS_FEATURES)
            fold["horizons"][f"H{h}"] = {"rows": len(valid), "A": control["metrics"],
                                          "B": sentiment["metrics"]}
        report["walk_forward"].append(fold)
    write_json(DEVELOPMENT, report)
    print(json.dumps({"stage": "development", "validation_rows": report["validation"]["H1"]["paired_rows"],
                      "walk_folds": len(report["walk_forward"])}))
    return report


def freeze() -> None:
    report = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if report["stage"] != "development_only":
        raise ValueError("Development report is not TEST-blind")
    frames, metadata = load_paired_frames()
    if metadata["dataset_sha256"] != report["news_dataset_sha256"]:
        raise ValueError("News cohort changed since development")
    lock = {"locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates": {"H1": list(TECHNICAL + NEWS_FEATURES),
                           "H5": list(TECHNICAL + NEWS_FEATURES)},
            "controls": {"H1": list(TECHNICAL), "H5": list(TECHNICAL)},
            "reason": "Prespecified 5-feature package on both authorized horizons. H1 VALIDATION gains are small with bootstrap intervals crossing zero and 3/4 ROC folds negative; H5 VALIDATION and folds deteriorate. Lock both solely for a final paired falsification check, not promotion. No TEST-based redesign.",
            "news_dataset_sha256": metadata["dataset_sha256"],
            "development_sha256": sha(DEVELOPMENT), "code_sha256": code_sha(),
            "test_metrics_seen_at_lock": False}
    write_json(LOCK, lock)
    print(json.dumps({"stage": "locked", "candidates": list(lock["candidates"])}))


def final() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["test_metrics_seen_at_lock"] or lock["code_sha256"] != code_sha() or lock["development_sha256"] != sha(DEVELOPMENT):
        raise ValueError("Candidate lock invalid or code changed")
    frames, metadata = load_paired_frames()
    if lock["news_dataset_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("News cohort changed after lock")
    splits = common_splits(frames)
    report = {"stage": "final_after_candidate_lock", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_lock_sha256": sha(LOCK), "news_dataset_sha256": metadata["dataset_sha256"],
              "product_sha256": sha(ROOT / "models/xgboost_v1.joblib"),
              "test": {}, "development_report_sha256": sha(DEVELOPMENT),
              "availability_limitation": "publication-time only; historical first-seen not provided by Alpha Vantage"}
    for h in (1, 5):
        split = splits[h]
        # Use original TRAIN only, as in FASE 6; VALIDATION did not train final models.
        train, test = split.train, split.test
        control, pa = model_result(train, test, TECHNICAL)
        sentiment, pb = model_result(train, test, TECHNICAL + NEWS_FEATURES)
        report["test"][f"H{h}"] = {"rows": len(test), "range": [test.index[0].isoformat(), test.index[-1].isoformat()],
                                      "A": control, "B": sentiment,
                                      "bootstrap_B_minus_A": paired_bootstrap(test["target"].to_numpy(dtype=int), pa, pb)}
    if report["product_sha256"] != PRODUCT_SHA:
        raise ValueError("Product model hash changed")
    write_json(FINAL, report)
    write_json(WALK, {"generated_at_utc": report["generated_at_utc"],
                      "source": str(DEVELOPMENT.relative_to(ROOT)),
                      "folds": json.loads(DEVELOPMENT.read_text(encoding="utf-8"))["walk_forward"]})
    print(json.dumps({"stage": "final", "H1_rows": report["test"]["H1"]["rows"],
                      "H5_rows": report["test"]["H5"]["rows"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "freeze", "final"))
    stage = parser.parse_args().stage
    {"development": development, "freeze": freeze, "final": final}[stage]()
