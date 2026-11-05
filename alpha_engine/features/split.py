"""Separación cronológica con purga de la etiqueta junto a cada frontera."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_rows: int = 1,
) -> TemporalSplit:
    """Divide 70/15/15 sin barajar y purga una etiqueta en cada frontera.

    El target de la última fila de train usa el close de la primera fila de
    validation. Se retira esa fila; se hace lo mismo entre validation y test.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Se requiere un índice temporal")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("El índice temporal debe estar ordenado y sin duplicados")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Las proporciones deben estar entre 0 y 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Debe quedar un conjunto test")
    if purge_rows < 1:
        raise ValueError("purge_rows debe ser >= 1 para evitar fuga entre conjuntos")

    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
    train = frame.iloc[: train_end - purge_rows].copy()
    validation = frame.iloc[train_end : validation_end - purge_rows].copy()
    test = frame.iloc[validation_end:].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("No hay suficientes filas para los tres conjuntos y la purga")
    return TemporalSplit(train=train, validation=validation, test=test)
