"""Las métricas de reglas deben coincidir con TEST y rechazar artefactos inconsistentes."""

from datetime import datetime, timedelta, timezone
from math import sin

import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle
from alpha_engine.ml.prescriptive_check import evaluate_prescriptive_rules, rule_settings
from alpha_engine.ml.trainer import train_from_candles


def _sample(count: int = 180) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    for index in range(count):
        price = 100 + 0.02 * index + 2 * sin(index / 3)
        opened = start + timedelta(minutes=index)
        candles.append(Candle(
            symbol="BTCUSDT", interval="1m", open_time=opened,
            close_time=opened + timedelta(milliseconds=59_999),
            open=price, high=price + 1, low=price - 1, close=price,
            volume=10 + index % 5,
        ))
    return candles


def test_prescriptive_check_uses_original_test_block_and_guards_metadata() -> None:
    candles = _sample()
    outcome = train_from_candles(candles, trained_at=candles[-1].open_time)
    metadata = outcome.metadata
    bundle = {
        "estimator": outcome.model,
        "features": metadata["features"],
        "trained_at": metadata["trained_at"],
        "model_version": metadata["model_version"],
    }
    result = evaluate_prescriptive_rules(candles, bundle, metadata, Settings(_env_file=None))
    assert result["test_rows"] == metadata["split"]["test"]["rows"]
    assert result["model_test_accuracy"] == pytest.approx(metadata["evaluation"]["test"]["accuracy"])
    assert sum(result["filtered_counts"].values()) == result["test_rows"]
    assert result["eligible_rows"] == result["filtered_counts"]["BUY"] + result["filtered_counts"]["SELL"]
    assert result["rule_parameters"] == rule_settings(Settings(_env_file=None))
    assert result["rule_parameters"] != rule_settings(Settings(_env_file=None, model_confidence_threshold=0.8))

    changed = {**metadata, "evaluation": {"test": {"accuracy": -1}}}
    with pytest.raises(ValueError, match="evaluación TEST cambió"):
        evaluate_prescriptive_rules(candles, bundle, changed, Settings(_env_file=None))
