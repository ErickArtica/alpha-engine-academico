"""Opt-in V3 shadow inference; results stay outside production repositories."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from alpha_engine.config.settings import PROJECT_ROOT
from alpha_engine.features.v3 import V3_FEATURE_COLUMNS, build_v3_feature_frame


MODEL_NAME = "xgboost_v3_market_flow_12m"
MODEL_PATH = PROJECT_ROOT / "models" / "experiments" / f"{MODEL_NAME}.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "experiments" / f"{MODEL_NAME}_metadata.json"
LOOKBACK = 600


@dataclass(frozen=True)
class ShadowPrediction:
    model_version: str
    candle_open_time: datetime
    prob_down: float
    prob_up: float
    predicted_class: str
    feature_ms: float
    prediction_ms: float


class V3Shadow:
    def __init__(self, source, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self._source = source
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._bundle = None

    def _model(self):
        if self._bundle is None:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(self._model_path.read_bytes()).hexdigest()
            if metadata["model_sha256"] != digest or metadata["features"] != list(V3_FEATURE_COLUMNS):
                raise ValueError("V3 shadow artifact fingerprint or feature order mismatch")
            bundle = joblib.load(self._model_path)
            if (bundle["model_version"] != MODEL_NAME or
                bundle["features"] != list(V3_FEATURE_COLUMNS) or
                bundle["symbol"] != "BTCUSDT" or bundle["interval"] != "1m" or
                bundle["horizon"] != "1m" or
                list(bundle["estimator"].classes_) != [0, 1]):
                raise ValueError("V3 shadow artifact metadata mismatch")
            self._bundle = bundle
        return self._bundle

    def predict(self, symbol: str, interval: str, candle_open_time: datetime) -> ShadowPrediction:
        if symbol != "BTCUSDT" or interval != "1m" or candle_open_time.tzinfo is None:
            raise ValueError("V3 shadow requires BTCUSDT 1m and timezone-aware candle")
        stop = candle_open_time + timedelta(minutes=1)
        start = stop - timedelta(minutes=LOOKBACK)
        candles = self._source.fetch_v3_page(symbol, interval, start, stop, limit=LOOKBACK)
        if (len(candles) != LOOKBACK or
            candles.timestamp.iloc[-1] != pd.Timestamp(candle_open_time)):
            raise ValueError("V3 shadow requires 600 continuous closed Binance klines")
        started = perf_counter()
        row = build_v3_feature_frame(candles).iloc[[-1]][list(V3_FEATURE_COLUMNS)]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("V3 shadow has missing or nonfinite features")
        feature_ms = (perf_counter() - started) * 1000
        model = self._model()["estimator"]
        started = perf_counter()
        probabilities = np.asarray(model.predict_proba(row))
        prediction_ms = (perf_counter() - started) * 1000
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ValueError("V3 shadow returned invalid probabilities")
        down, up = (float(probabilities[0, i]) for i in (0, 1))
        return ShadowPrediction(MODEL_NAME, candle_open_time, down, up,
                                "UP" if up > down else "DOWN", feature_ms, prediction_ms)
