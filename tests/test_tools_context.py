from datetime import datetime, timedelta, timezone
from math import sin
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alpha_engine.ai.context_builder import build_context, context_json
from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle
from alpha_engine.features import FEATURE_COLUMNS
from alpha_engine.risk import RiskService
from alpha_engine.services.prediction import PredictionService
from alpha_engine.services.recommendation_context import RecommendationContextService
from alpha_engine.tools.analytics import RepositoryAnalyticsTools
from alpha_engine.tools.market import AnalyticsDataError, get_market_summary, get_technical_indicators
from alpha_engine.tools.schemas import RecommendationSnapshot


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candles(count=120):
    result = []
    for i in range(count):
        open_time = START + timedelta(minutes=i)
        price = 100 + 0.05 * i + sin(i / 4)
        result.append(Candle(
            symbol="BTCUSDT", interval="1m", open_time=open_time,
            close_time=open_time + timedelta(milliseconds=59_999),
            open=price, high=price + 1, low=price - 1,
            close=price, volume=10 + i % 4,
        ))
    return result


class FakeMarket:
    def __init__(self, records):
        self.records = records
        self.range_reads = 0

    def get_latest_candle(self, symbol, interval):
        return self.records[-1]

    def get_candles(self, symbol, interval, start, stop):
        self.range_reads += 1
        return [c for c in self.records if start <= c.open_time < stop]


class FakePredictions:
    def __init__(self):
        self.values = {}

    def get_by_id(self, prediction_id):
        return self.values.get(prediction_id)

    def save(self, prediction):
        self.values[prediction.prediction_id] = prediction


class FakeEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return np.array([[0.3, 0.7]])


def build_tools(records=None):
    market = FakeMarket(records or candles())
    predictions = FakePredictions()
    bundle = {
        "estimator": FakeEstimator(),
        "features": list(FEATURE_COLUMNS),
        "model_version": "xgboost_v1",
        "trained_at": START.isoformat(),
        "symbol": "BTCUSDT", "interval": "1m", "horizon": "1m",
    }
    service = PredictionService(
        market, predictions, Path("models/unused.joblib"),
        loader=lambda path: bundle,
    )
    tools = RepositoryAnalyticsTools(
        market, service, RiskService(Settings(_env_file=None)), "BTCUSDT"
    )
    return tools, market


def test_market_and_indicator_tools_return_structured_real_calculations():
    records = candles()
    summary = get_market_summary(records)
    indicators = get_technical_indicators(records)
    assert summary.price == records[-1].close
    assert summary.change_1m_pct == pytest.approx(
        (records[-1].close / records[-2].close - 1) * 100
    )
    assert indicators.timestamp == summary.timestamp
    assert 0 <= indicators.rsi_14 <= 100
    assert indicators.ema_20 > 0
    assert set(indicators.model_dump()) == {"symbol", "timestamp", *FEATURE_COLUMNS}


def test_orchestrator_aligns_tools_and_context_is_deterministic():
    tools, market = build_tools()
    service = RecommendationContextService(tools)
    snapshot = service.build_snapshot()
    first = context_json(build_context(snapshot))
    second = context_json(build_context(snapshot))
    assert first == second
    assert market.range_reads == 2  # una lectura para ML y una compartida por dos herramientas
    parsed = json.loads(first)
    assert parsed["system"]["real_trading"] is False
    assert parsed["market"]["timestamp"] == snapshot.prediction.timestamp.isoformat().replace("+00:00", "Z")
    assert parsed["prediction"]["prob_up"] == pytest.approx(0.7)
    assert parsed["risk"]["reference_price"] == parsed["market"]["price"]
    assert "influx_token" not in first
    assert "candles" not in first
    assert service.build_context_json() == first
    governed = build_context(snapshot, Settings(_env_file=None))
    assert governed["decision_policy"]["candidate_signal"] == "BUY"
    assert governed["decision_policy"]["minimum_model_probability"] == 0.6


def test_context_rejects_misaligned_timestamp_and_nonfinite_json():
    tools, _ = build_tools()
    snapshot = RecommendationContextService(tools).build_snapshot()
    wrong_market = snapshot.market.model_copy(
        update={"timestamp": snapshot.market.timestamp - timedelta(minutes=1)}
    )
    with pytest.raises(ValidationError, match="misma vela"):
        RecommendationSnapshot(
            market=wrong_market, indicators=snapshot.indicators,
            prediction=snapshot.prediction, risk=snapshot.risk,
        )
    with pytest.raises(ValueError):
        context_json({"bad": float("nan")})


def test_technical_tool_rejects_insufficient_history():
    with pytest.raises(AnalyticsDataError, match="historia suficiente"):
        get_technical_indicators(candles(20))
