from datetime import datetime, timezone

import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionResult


def prediction(up=0.7):
    return PredictionResult(
        prediction_id="id-1", symbol="BTCUSDT",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), horizon="1m",
        prob_up=up, prob_down=1 - up,
        predicted_class="UP" if up > 0.5 else "DOWN",
        model_version="xgboost_v1",
    )


def test_risk_low_and_directional_levels():
    service = RiskService(Settings(_env_file=None))
    long = service.analyze(prediction(0.7), price=100, volatility=0.001, rsi=50, macd=1)
    short = service.analyze(prediction(0.3), price=100, volatility=0.001, rsi=50, macd=-1)
    assert long.risk_level == short.risk_level == "LOW"
    assert long.stop_loss < 100 < long.take_profit
    assert short.take_profit < 100 < short.stop_loss
    assert long.triggers == []


def test_risk_high_and_medium_rules():
    service = RiskService(Settings(_env_file=None))
    high = service.analyze(prediction(0.55), 100, 0.006, 80, -1)
    assert high.risk_level == "HIGH"
    assert {"HIGH_VOLATILITY", "EXTREME_RSI", "LOW_MODEL_CONFIDENCE"}.issubset(high.triggers)
    medium = service.analyze(prediction(0.7), 100, 0.003, 50, -1)
    assert medium.risk_level == "MEDIUM"
    assert "MACD_CONTRADICTS_DIRECTION" in medium.triggers


def test_risk_rejects_nonfinite_or_invalid_inputs():
    service = RiskService(Settings(_env_file=None))
    with pytest.raises(ValueError, match="finitos"):
        service.analyze(prediction(), 100, float("nan"), 50, 1)
    with pytest.raises(ValueError, match="fuera de rango"):
        service.analyze(prediction(), 100, 0.001, 101, 1)

