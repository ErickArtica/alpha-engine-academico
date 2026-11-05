"""Referencia simple: clase más frecuente de TRAIN, empate -> DOWN."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MajorityBaseline:
    majority_class: int

    @classmethod
    def fit(cls, labels: Sequence[int]) -> "MajorityBaseline":
        values = np.asarray(labels, dtype=int)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("TRAIN debe contener etiquetas")
        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("Las etiquetas deben ser 0 o 1")
        count_down = int((values == 0).sum())
        count_up = int((values == 1).sum())
        return cls(majority_class=1 if count_up > count_down else 0)

    def predict(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("count debe ser >= 0")
        return np.full(count, self.majority_class, dtype=np.int8)

