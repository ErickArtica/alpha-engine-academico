"""Métricas binarias calculadas exclusivamente sobre labels observados."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("Las etiquetas deben ser vectores del mismo tamaño")
    if len(actual) == 0:
        raise ValueError("No se pueden evaluar conjuntos vacíos")
    if not set(np.unique(actual)).issubset({0, 1}) or not set(
        np.unique(predicted)
    ).issubset({0, 1}):
        raise ValueError("Las etiquetas deben ser binarias: 0 o 1")
    return {
        "sample_count": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "confusion_labels": ["DOWN", "UP"],
    }

