import json
from datetime import datetime, timedelta, timezone

import pytest

from alpha_engine.ai.recommendation_service import (
    RecommendationGenerationError,
    RecommendationService,
)
from alpha_engine.config.settings import Settings
from alpha_engine.observability.telemetry import TelemetryStore
from alpha_engine.risk.service import RiskResult
from alpha_engine.services.prediction import ModelNotTrainedError, PredictionResult
from alpha_engine.tools.schemas import (
    MarketSummary,
    RecommendationSnapshot,
    TechnicalIndicators,
)


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
CLOSE_TIME = TIMESTAMP + timedelta(milliseconds=59_999)
NOW = CLOSE_TIME + timedelta(seconds=5)


def snapshot(prob_up=0.7, risk_level="LOW", price=100.0):
    prediction = PredictionResult(
        prediction_id="pred-1", symbol="BTCUSDT", timestamp=TIMESTAMP,
        horizon="1m", prob_up=prob_up, prob_down=1 - prob_up,
        predicted_class="UP" if prob_up > 0.5 else "DOWN",
        model_version="xgboost_v1",
    )
    return RecommendationSnapshot(
        market=MarketSummary(
            symbol="BTCUSDT", timestamp=TIMESTAMP, close_time=CLOSE_TIME,
            price=price, open=price, high=price + 1, low=price - 1,
            change_1m_pct=0.1, volume=10,
        ),
        indicators=TechnicalIndicators(
            symbol="BTCUSDT", timestamp=TIMESTAMP, return_1m=0.001,
            rsi_14=50, ema_20=price, macd=0.1, macd_signal=0.05,
            macd_histogram=0.05, rolling_volatility=0.001, volume_change=0.1,
        ),
        prediction=prediction,
        risk=RiskResult(
            prediction_id="pred-1", risk_level=risk_level,
            reference_price=price, stop_loss=price - 1, take_profit=price + 1,
            triggers=[],
        ),
    )


def llm_json(signal="BUY"):
    return json.dumps({
        "signal": signal, "confidence": 0.8, "risk_level": "LOW",
        "summary": "Se considera la señal con cautela.",
        "key_factors": ["Modelo ML"], "risks": ["Volatilidad"],
        "warning": "No garantiza rentabilidad.",
    })


class FakeContextService:
    def __init__(self, value):
        self.value = value

    def build_snapshot(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeRecommendations:
    def __init__(self):
        self.values = {}
        self.saves = 0

    def get_cached(self, prediction_id, prompt_version, llm_model):
        return self.values.get((prediction_id, prompt_version, llm_model))

    def save(self, record):
        self.values[(record.prediction_id, record.prompt_version, record.llm_model)] = record
        self.saves += 1


class FakeProvider:
    model_name = "qwen3:4b-instruct"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate_recommendation(self, context):
        self.calls += 1
        return next(self.responses)


def service(value=None, responses=None):
    context = FakeContextService(value if value is not None else snapshot())
    repo = FakeRecommendations()
    provider = FakeProvider(responses if responses is not None else [llm_json()])
    return RecommendationService(context, repo, provider, Settings(_env_file=None)), context, repo, provider


def test_recommendation_is_cached_without_second_llm_call():
    recommendation, _, repo, provider = service()
    first = recommendation.recommend(now=NOW)
    second = recommendation.recommend(now=NOW)
    assert first.final_signal == "BUY"
    assert first.candidate_signal == first.llm_signal == "BUY"
    assert first.confidence == pytest.approx(0.7)
    assert second.cached is True
    assert provider.calls == 1
    assert repo.saves == 1


def test_invalid_json_retries_once_then_returns_controlled_error():
    recommendation, _, repo, provider = service(responses=["not json", llm_json()])
    assert recommendation.recommend(now=NOW).final_signal == "BUY"
    assert provider.calls == 2
    assert repo.saves == 1
    bad, _, bad_repo, bad_provider = service(responses=["bad", "also bad"])
    with pytest.raises(RecommendationGenerationError, match="dos intentos"):
        bad.recommend(now=NOW)
    assert bad_provider.calls == 2
    assert bad_repo.saves == 0


def test_deterministic_hold_skips_llm_and_is_saved():
    for value, now, expected in [
        (snapshot(prob_up=0.55), NOW, "LOW_MODEL_CONFIDENCE"),
        (snapshot(), CLOSE_TIME + timedelta(seconds=121), "STALE_MARKET_DATA"),
        (snapshot(), CLOSE_TIME + timedelta(seconds=60), "PREDICTION_EXPIRED"),
        (snapshot(risk_level="HIGH"), NOW, "HIGH_RISK"),
    ]:
        recommendation, _, repo, provider = service(value=value)
        result = recommendation.recommend(now=now)
        assert result.final_signal == "HOLD"
        assert expected in result.guardrails
        assert provider.calls == 0
        assert result.llm_signal is None
        assert repo.saves == 1


def test_low_confidence_hold_is_reused_without_llm_call():
    recommendation, _, repo, provider = service(
        value=snapshot(prob_up=0.55), responses=[llm_json("BUY")]
    )
    first = recommendation.recommend(now=NOW)
    second = recommendation.recommend(now=NOW)
    assert first.candidate_signal == "HOLD"
    assert first.llm_signal is None
    assert first.final_signal == second.final_signal == "HOLD"
    assert second.cached is False
    assert provider.calls == 0
    assert repo.saves == 1


def test_deterministic_hold_records_llm_skip_reason(tmp_path):
    context = FakeContextService(snapshot(risk_level="HIGH"))
    repo = FakeRecommendations()
    provider = FakeProvider([llm_json()])
    telemetry = TelemetryStore(tmp_path)
    result = RecommendationService(
        context, repo, provider, Settings(_env_file=None), telemetry=telemetry,
    ).recommend(now=NOW)
    events = telemetry.read_events()
    assert [event["stage"] for event in events] == [
        "recommendation_started", "context_ready", "llm_skipped", "recommendation_ready",
    ]
    assert events[2]["details"]["reasons"] == ["HIGH_RISK"]
    assert result.final_signal == "HOLD"
    assert provider.calls == 0


def test_llm_echo_cannot_change_deterministic_action():
    wrong, _, _, _ = service(responses=[llm_json("SELL")])
    result = wrong.recommend(now=NOW)
    assert result.candidate_signal == "BUY"
    assert result.llm_signal == "SELL"
    assert result.final_signal == "BUY"
    assert result.guardrails == []
    cautious, _, _, _ = service(responses=[llm_json("HOLD")])
    assert cautious.recommend(now=NOW).final_signal == "BUY"


def test_cached_llm_is_rechecked_for_staleness_without_new_call():
    recommendation, _, repo, provider = service()
    assert recommendation.recommend(now=NOW).final_signal == "BUY"
    later = recommendation.recommend(now=CLOSE_TIME + timedelta(seconds=121))
    assert later.final_signal == "HOLD"
    assert later.llm_signal == "BUY"
    assert "STALE_MARKET_DATA" in later.guardrails
    assert provider.calls == 1
    assert repo.saves == 2


def test_cached_llm_is_not_actionable_after_prediction_horizon():
    recommendation, _, repo, provider = service()
    assert recommendation.recommend(now=NOW).final_signal == "BUY"
    later = recommendation.recommend(now=CLOSE_TIME + timedelta(seconds=60))
    assert later.final_signal == "HOLD"
    assert later.llm_signal == "BUY"
    assert "PREDICTION_EXPIRED" in later.guardrails
    assert provider.calls == 1
    assert repo.saves == 2


def test_slow_llm_cannot_publish_signal_after_horizon():
    times = iter([NOW, CLOSE_TIME + timedelta(seconds=61)])
    context = FakeContextService(snapshot())
    repo = FakeRecommendations()
    provider = FakeProvider([llm_json("BUY")])
    recommendation = RecommendationService(
        context, repo, provider, Settings(_env_file=None), clock=lambda: next(times)
    )
    result = recommendation.recommend()
    assert result.generated_at == CLOSE_TIME + timedelta(seconds=61)
    assert result.llm_signal == "BUY"
    assert result.final_signal == "HOLD"
    assert "PREDICTION_EXPIRED" in result.guardrails
    assert provider.calls == repo.saves == 1


def test_context_change_regenerates_llm_even_with_same_cache_key():
    recommendation, context, repo, provider = service(responses=[llm_json(), llm_json()])
    assert recommendation.recommend(now=NOW).final_signal == "BUY"
    context.value = snapshot(price=101.0)
    assert recommendation.recommend(now=NOW).final_signal == "BUY"
    assert provider.calls == 2
    assert repo.saves == 2


def test_missing_prediction_returns_hold_without_llm_or_cache():
    recommendation, _, repo, provider = service(value=ModelNotTrainedError("sin modelo"))
    result = recommendation.recommend(now=NOW)
    assert result.final_signal == "HOLD"
    assert result.prediction_id is None
    assert result.context_json is None
    assert result.guardrails == ["MISSING_PREDICTION"]
    assert provider.calls == repo.saves == 0


def test_recommendation_records_context_llm_and_ready_stages(tmp_path):
    context = FakeContextService(snapshot())
    repo = FakeRecommendations()
    provider = FakeProvider([llm_json()])
    telemetry = TelemetryStore(tmp_path)
    result = RecommendationService(
        context, repo, provider, Settings(_env_file=None), telemetry=telemetry,
    ).recommend(now=NOW)
    assert result.trace_id
    events = telemetry.read_events()
    assert [event["stage"] for event in events] == [
        "recommendation_started", "context_ready", "llm_started", "llm_finished", "recommendation_ready",
    ]
    assert {event["trace_id"] for event in events} == {result.trace_id}
    assert events[-1]["details"]["final_signal"] == "BUY"
