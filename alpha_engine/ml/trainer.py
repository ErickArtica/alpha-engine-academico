"""Entrenamiento reproducible sin acoplamiento a InfluxDB ni a Streamlit."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import tempfile
from typing import Sequence

import joblib
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from alpha_engine.data.schemas import Candle
from alpha_engine.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    build_training_frame,
    chronological_split,
    target_distribution,
)
from alpha_engine.features.split import TemporalSplit
from alpha_engine.ml.baseline import MajorityBaseline
from alpha_engine.ml.evaluation import evaluate_binary

MODEL_VERSION = "xgboost_v1"
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 2,
}


@dataclass(frozen=True)
class TrainingOutcome:
    model: XGBClassifier
    metadata: dict
    baseline: dict


def _range_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
        "class_distribution": target_distribution(frame),
    }


def train_from_candles(
    candles: Sequence[Candle], trained_at: datetime | None = None
) -> TrainingOutcome:
    """Entrena solo con TRAIN. VALIDATION diagnostica; TEST queda intacto."""
    feature_frame = build_feature_frame(candles)
    labeled = build_training_frame(feature_frame)
    split: TemporalSplit = chronological_split(labeled)
    if len(set(split.train["target"])) != 2:
        raise ValueError("TRAIN debe contener ambas clases para XGBoost")

    features = list(FEATURE_COLUMNS)
    baseline_model = MajorityBaseline.fit(split.train["target"].to_numpy())
    baseline = {
        "name": "majority_class",
        "trained_on": "TRAIN",
        "majority_class": baseline_model.majority_class,
        "validation": evaluate_binary(
            split.validation["target"].to_numpy(),
            baseline_model.predict(len(split.validation)),
        ),
        "test": evaluate_binary(
            split.test["target"].to_numpy(),
            baseline_model.predict(len(split.test)),
        ),
    }

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(split.train[features], split.train["target"])
    validation_metrics = evaluate_binary(
        split.validation["target"].to_numpy(), model.predict(split.validation[features])
    )
    test_metrics = evaluate_binary(
        split.test["target"].to_numpy(), model.predict(split.test[features])
    )
    timestamp = trained_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("trained_at debe tener zona horaria")
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": timestamp.astimezone(timezone.utc).isoformat(),
        "symbol": str(feature_frame["symbol"].iloc[0]),
        "interval": "1m",
        "horizon": "1m",
        "target_definition": "1 si close(t+1m) > close(t); 0 si empata o baja",
        "features": features,
        "parameters": XGB_PARAMS.copy(),
        "data_range": {
            "start": feature_frame.index.min().isoformat(),
            "end": feature_frame.index.max().isoformat(),
            "closed_candles": len(feature_frame),
            "labeled_rows": len(labeled),
        },
        "split": {
            "method": "chronological",
            "fractions_before_purge": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "purge_rows_at_each_boundary": 1,
            "train": _range_summary(split.train),
            "validation": _range_summary(split.validation),
            "test": _range_summary(split.test),
        },
        "evaluation": {"validation": validation_metrics, "test": test_metrics},
        "baseline_file": "baseline_v1.json",
        "library_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    return TrainingOutcome(model=model, metadata=metadata, baseline=baseline)


def save_training_outcome(outcome: TrainingOutcome, output_dir: Path) -> dict[str, Path]:
    """Escribe cada archivo mediante reemplazo atómico dentro de output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / "xgboost_v1.joblib",
        "metadata": output_dir / "xgboost_v1_metadata.json",
        "baseline": output_dir / "baseline_v1.json",
    }
    bundle = {
        "estimator": outcome.model,
        "features": outcome.metadata["features"],
        "model_version": outcome.metadata["model_version"],
        "trained_at": outcome.metadata["trained_at"],
        "symbol": outcome.metadata["symbol"],
        "interval": outcome.metadata["interval"],
        "horizon": outcome.metadata["horizon"],
    }
    _atomic_joblib(bundle, paths["model"])
    _atomic_json(outcome.baseline, paths["baseline"])
    _atomic_json(outcome.metadata, paths["metadata"])
    return paths


def _atomic_json(data: dict, destination: Path) -> None:
    descriptor, name = tempfile.mkstemp(
        suffix=".tmp", prefix=destination.stem, dir=destination.parent
    )
    temp_path = Path(name)
    try:
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_joblib(data: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        suffix=".tmp", prefix=destination.stem,
        dir=destination.parent, delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        joblib.dump(data, temp_path, compress=3)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
