"""Predicción de la última vela cerrada, independiente del motor de almacenamiento."""

import hashlib
import logging
from time import perf_counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from alpha_engine.config.settings import PROJECT_ROOT
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.data.timing import prediction_expires_at
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from alpha_engine.observability.process_memory import memory_event
from alpha_engine.repositories.protocols import MarketRepository, PredictionRepository
from alpha_engine.services.errors import StorageError

logger = logging.getLogger(__name__)


class ModelNotTrainedError(FileNotFoundError):
    """No existe todavía un artefacto de entrenamiento local."""


class PredictionUnavailableError(ValueError):
    """Las velas o el artefacto no permiten producir una predicción válida."""


class InsufficientHistoryError(PredictionUnavailableError):
    """The latest closed candle lacks enough valid past features."""


class PredictionResult(PredictionRecord):
    """Salida Pydantic; timestamp es la apertura UTC de la vela evaluada."""


@dataclass(frozen=True)
class PredictionExecution:
    prediction: PredictionResult
    created: bool


def load_model_bundle(path: Path) -> dict:
    if not path.is_file():
        raise ModelNotTrainedError(f"Modelo pendiente de entrenamiento: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {
        "estimator", "features", "model_version", "trained_at", "symbol", "interval", "horizon"
    }.issubset(bundle):
        raise PredictionUnavailableError("El artefacto de modelo está incompleto")
    if bundle["features"] != list(FEATURE_COLUMNS):
        raise PredictionUnavailableError("Las features del modelo no coinciden con la versión actual")
    if not hasattr(bundle["estimator"], "predict_proba"):
        raise PredictionUnavailableError("El estimador no ofrece predict_proba")
    memory_event("after_model_load")
    return bundle


class PredictionService:
    def __init__(
        self,
        market_repository: MarketRepository,
        prediction_repository: PredictionRepository,
        model_path: Path,
        lookback_candles: int = 120,
        loader: Callable[[Path], dict] = load_model_bundle,
    ):
        if lookback_candles < 35:
            raise ValueError("Se requieren al menos 35 velas de lookback")
        self._market = market_repository
        self._predictions = prediction_repository
        self._model_path = model_path if model_path.is_absolute() else PROJECT_ROOT / model_path
        self._lookback = lookback_candles
        self._loader = loader
        self._bundle: dict | None = None
        self.last_features_ms = 0.0

    def _model_bundle(self) -> dict:
        # One service instance can reuse the validated model across cycles.
        if self._bundle is None:
            self._bundle = self._loader(self._model_path)
        return self._bundle

    def predict_latest(
        self, symbol: str, interval: str = "1m", now: datetime | None = None
    ) -> PredictionResult:
        return self.predict_latest_execution(symbol, interval, now).prediction

    def predict_latest_execution(
        self, symbol: str, interval: str = "1m", now: datetime | None = None
    ) -> PredictionExecution:
        try:
            latest = self._market.get_latest_candle(symbol, interval)
        except Exception as exc:
            raise StorageError("No se pudo consultar la ultima vela") from exc
        if latest is None:
            raise PredictionUnavailableError("No hay velas cerradas para predecir")
        return self.predict_candle_execution(latest, symbol, interval, now)

    def predict_candle_execution(
        self, candle: Candle, symbol: str, interval: str = "1m", now: datetime | None = None,
        generated_at_clock: Callable[[], datetime] | None = None,
    ) -> PredictionExecution:
        self.last_features_ms = 0.0
        if interval != "1m":
            raise PredictionUnavailableError("El modelo inicial predice solo velas de 1m")
        if candle.symbol != symbol or candle.interval != interval:
            raise PredictionUnavailableError("La vela no corresponde a la serie solicitada")
        bundle = self._model_bundle()
        if bundle["symbol"] != symbol or bundle["interval"] != interval or bundle["horizon"] != "1m":
            raise PredictionUnavailableError("El modelo fue entrenado para otra serie u horizonte")
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now debe incluir zona horaria")
        if candle.close_time >= current_time:
            raise PredictionUnavailableError("La última vela aún no está cerrada")
        timestamp = candle.open_time.astimezone(timezone.utc)
        identity = (
            f"{symbol}|{interval}|{timestamp.isoformat()}|"
            f"{bundle['model_version']}|{bundle['trained_at']}"
        )
        prediction_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        try:
            cached = self._predictions.get_by_id(prediction_id)
        except Exception as exc:
            raise StorageError("No se pudo consultar la prediccion") from exc
        if cached is not None:
            logger.info("Predicción reutilizada: prediction_id=%s", prediction_id)
            values = cached.model_dump()
            values["trained_at"] = datetime.fromisoformat(bundle["trained_at"].replace("Z", "+00:00"))
            return PredictionExecution(PredictionResult.model_validate(values), created=False)

        stop = timestamp + timedelta(minutes=1)
        start = stop - timedelta(minutes=self._lookback)
        try:
            candles = self._market.get_candles(symbol, interval, start, stop)
        except Exception as exc:
            raise StorageError("No se pudo consultar la historia de velas") from exc
        if not candles or candles[-1].open_time != candle.open_time:
            raise PredictionUnavailableError("Faltan las últimas velas cerradas")
        if len(candles) < self._lookback or any(
            right.open_time - left.open_time != timedelta(minutes=1)
            for left, right in zip(candles, candles[1:])
        ):
            raise InsufficientHistoryError("No hay historia suficiente y continua para la prediccion")
        features_started = perf_counter()
        try:
            frame = build_feature_frame(candles)
            memory_event("after_features")
        except ValueError as exc:
            raise PredictionUnavailableError(str(exc)) from exc
        finally:
            self.last_features_ms = (perf_counter() - features_started) * 1000
        row = frame.iloc[-1][list(FEATURE_COLUMNS)]
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise InsufficientHistoryError("No hay historia suficiente para todas las features")
        classes = list(bundle["estimator"].classes_)
        if set(classes) != {0, 1} or len(classes) != 2:
            raise PredictionUnavailableError("El modelo no contiene las clases DOWN y UP")
        input_frame = pd.DataFrame([row.to_dict()], columns=list(FEATURE_COLUMNS))
        probabilities = np.asarray(bundle["estimator"].predict_proba(input_frame))
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise PredictionUnavailableError("El modelo devolvió probabilidades inválidas")
        prob_down = float(probabilities[0, classes.index(0)])
        prob_up = float(probabilities[0, classes.index(1)])
        total = prob_up + prob_down
        if total <= 0 or prob_up < 0 or prob_down < 0:
            raise PredictionUnavailableError("El modelo devolvió probabilidades inválidas")
        prob_up /= total
        prob_down /= total
        generated_at = (generated_at_clock or (lambda: datetime.now(timezone.utc)))()
        if generated_at.tzinfo is None:
            raise ValueError("generated_at debe incluir zona horaria")
        generated_at = generated_at.astimezone(timezone.utc)
        result = PredictionResult(
            prediction_id=prediction_id,
            symbol=symbol,
            timestamp=timestamp,
            horizon="1m",
            prob_up=prob_up,
            prob_down=prob_down,
            predicted_class="UP" if prob_up > prob_down else "DOWN",
            model_version=bundle["model_version"],
            trained_at=datetime.fromisoformat(bundle["trained_at"].replace("Z", "+00:00")),
            generated_at=generated_at,
            valid_until=prediction_expires_at(candle.close_time, bundle["horizon"]).astimezone(timezone.utc),
        )
        try:
            inserted = self._predictions.save(result)
        except Exception as exc:
            raise StorageError("No se pudo guardar la prediccion") from exc
        if inserted is False:
            try:
                cached = self._predictions.get_by_id(prediction_id)
            except Exception as exc:
                raise StorageError("No se pudo recuperar la prediccion concurrente") from exc
            if cached is None:
                raise StorageError("La prediccion concurrente no pudo recuperarse")
            values = cached.model_dump()
            values["trained_at"] = result.trained_at
            return PredictionExecution(PredictionResult.model_validate(values), created=False)
        logger.info(
            "Predicción guardada: symbol=%s timestamp=%s prediction_id=%s",
            symbol, timestamp.isoformat(), prediction_id,
        )
        return PredictionExecution(result, created=True)
