"""Target binario: close(t+1m) > close(t) => 1; empate o caída => 0."""

import numpy as np
import pandas as pd

from alpha_engine.features.indicators import FEATURE_COLUMNS


def build_training_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta solo filas con la siguiente vela de 1m y features completas.

    La última fila no tiene close(t+1), por lo que nunca recibe etiqueta.
    """
    required = {"close", *FEATURE_COLUMNS}
    missing = required.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(missing))}")
    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise ValueError("Se requiere un índice temporal")
    if not feature_frame.index.is_monotonic_increasing or not feature_frame.index.is_unique:
        raise ValueError("El índice temporal debe estar ordenado y sin duplicados")

    next_close = feature_frame["close"].shift(-1)
    next_timestamp = feature_frame.index.to_series().shift(-1)
    has_next_minute = (
        (next_timestamp - feature_frame.index.to_series()).dt.total_seconds().eq(60.0)
    )
    usable = has_next_minute & next_close.notna()
    training = feature_frame.loc[usable].copy()
    training["target"] = (
        next_close.loc[usable].gt(feature_frame.loc[usable, "close"])
    ).astype("int8")
    training = training.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(FEATURE_COLUMNS)
    )
    return training


def target_distribution(training_frame: pd.DataFrame) -> dict[int, int]:
    """Conteos reales de clases para diagnóstico previo al entrenamiento."""
    if "target" not in training_frame:
        raise ValueError("Falta la columna target")
    if training_frame["target"].isna().any() or not set(
        training_frame["target"].unique()
    ).issubset({0, 1}):
        raise ValueError("target debe contener exclusivamente 0 o 1")
    counts = training_frame["target"].value_counts()
    return {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
