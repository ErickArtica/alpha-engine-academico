import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from alpha_engine.ai.providers.base import LLMProviderError, LLMProviderTimeoutError
from alpha_engine.ai.prompts.loader import PROMPT_VERSION
from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import PredictionRecord, RecommendationRecord
from alpha_engine.data.news_context import NewsContext
from alpha_engine.risk.service import RiskResult
from alpha_engine.services.recommendation_observability import RecommendationObservabilityService
from alpha_engine.services.recommendation_use_case import RecommendationUseCase
from alpha_engine.tools.schemas import MarketSummary, RecommendationSnapshot, TechnicalIndicators

OPEN = datetime(2026, 1, 1, 10, 4, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(minutes=1) - timedelta(milliseconds=1)
VALID_UNTIL = CLOSE + timedelta(minutes=1)
NOW = CLOSE + timedelta(seconds=5)


def prediction(prediction_id="p1"):
    return PredictionRecord(
        prediction_id=prediction_id, symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=0.7, prob_down=0.3, predicted_class="UP", model_version="xgboost_v1",
        generated_at=NOW, valid_until=VALID_UNTIL,
    )


def snapshot(value, risk_level="LOW"):
    runtime = value.model_copy()
    return RecommendationSnapshot(
        market=MarketSummary(symbol="BTCUSDT", timestamp=OPEN, close_time=CLOSE,
                             price=100, open=100, high=101, low=99, change_1m_pct=0.1, volume=10),
        indicators=TechnicalIndicators(symbol="BTCUSDT", timestamp=OPEN, return_1m=0.001,
                                       rsi_14=50, ema_20=100, macd=0.1, macd_signal=0.05,
                                       macd_histogram=0.05, rolling_volatility=0.001, volume_change=0.1),
        prediction=runtime,
        risk=RiskResult(prediction_id=runtime.prediction_id, risk_level=risk_level,
                        reference_price=100, stop_loss=99, take_profit=101, triggers=[]),
    )


def llm(signal="BUY"):
    return json.dumps({
        "signal": signal, "confidence": 0.8, "risk_level": "LOW",
        "summary": "Se considera la señal con cautela.", "key_factors": ["Modelo ML"],
        "risks": ["Volatilidad"], "warning": "No garantiza rentabilidad.",
    })


class Context:
    def __init__(self, source, risk_level="LOW"):
        self.source = source
        self.risk_level = risk_level
        self.calls = []

    def build_snapshot_for_prediction(self, value):
        self.calls.append(value)
        return snapshot(value, self.risk_level)


class Predictions:
    def __init__(self, records=()):
        self.records = {row.prediction_id: row for row in records}

    def get_by_id(self, value):
        return self.records.get(value)

    def list_recent(self, symbol, limit=100, offset=0):
        rows = [row for row in self.records.values() if row.symbol == symbol]
        return sorted(rows, key=lambda row: row.timestamp, reverse=True)[offset:offset + limit]


class Recommendations:
    def __init__(self):
        self.records = {}
        self.save_calls = 0
        self.force_race = False

    def get_cached(self, prediction_id, prompt_version, llm_model):
        return self.records.get((prediction_id, prompt_version, llm_model))

    def save_if_absent(self, record):
        key = (record.prediction_id, record.prompt_version, record.llm_model)
        self.save_calls += 1
        if self.force_race and key not in self.records:
            winner = record.model_copy(update={"final_signal": "HOLD", "llm_signal": "HOLD"})
            self.records[key] = winner
            return False
        if key in self.records:
            return False
        self.records[key] = record
        return True

    def save(self, record):
        key = (record.prediction_id, record.prompt_version, record.llm_model)
        self.save_calls += 1
        self.records[key] = record


class Requests:
    def __init__(self):
        self.records = []

    def save(self, record):
        self.records.append(record)

    def get_latest(self, symbol, status=None):
        rows = self.list_recent(symbol, status=status)
        return rows[0] if rows else None

    def list_recent(self, symbol, limit=20, status=None):
        return [row for row in reversed(self.records) if row.symbol == symbol
                and (status is None or row.status == status)][:limit]


class Provider:
    model_name = "test-llm"

    def __init__(self, response=llm()):
        self.response = response
        self.calls = 0
        self.contexts = []
        self.timeouts = []

    def generate_recommendation(self, context, *, timeout_seconds=None):
        self.calls += 1
        self.contexts.append(context)
        self.timeouts.append(timeout_seconds)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def use_case(*, value=None, provider=None, context=None, clock=None, requests=None, recommendations=None,
             risk_level="LOW", settings=None, news_service=None):
    value = value or prediction()
    provider = provider or Provider()
    context = context or Context(value, risk_level)
    requests = requests or Requests()
    recommendations = recommendations or Recommendations()
    settings = settings or Settings(_env_file=None, llm_live_timeout_seconds=45,
                                   llm_deadline_margin_seconds=1)
    return RecommendationUseCase(context, Predictions([value]), recommendations, requests, provider,
                                 settings, clock=clock or (lambda: NOW), news_service=news_service), provider, recommendations, requests, context


class News:
    def __init__(self, result=None, failure=None):
        self.result = result or NewsContext(status="NO_NEWS", fetched_at=NOW)
        self.failure = failure
        self.calls = []

    def get_context(self, *, deadline_at):
        self.calls.append(deadline_at)
        if self.failure:
            raise self.failure
        return self.result


def test_optional_news_is_added_before_mock_llm_and_audited():
    news = News(NewsContext(status="AVAILABLE", fetched_at=NOW, latest_news_at=NOW,
                           news_count=3, btc_sentiment_score=-.2, average_relevance=.6,
                           source_count=2))
    service, provider, recommendations, requests, _ = use_case(news_service=news)
    result = service.request("p1", now=NOW)
    assert result.status == "SUCCESS" and news.calls == [VALID_UNTIL]
    assert provider.contexts[0]["prediction"]["model_version"] == "xgboost_v1"
    assert provider.contexts[0]["news_context"]["role"] == "CONTEXT_ONLY"
    assert provider.contexts[0]["news_context"]["btc_sentiment_score"] == -.2
    assert requests.records[0].news_context_used
    assert requests.records[0].news_context_count == 3
    assert result.news_context.news_count == 3
    assert "news_context" in next(iter(recommendations.records.values())).context_json


def test_news_failure_keeps_technical_request_and_pre_guard_skips_news():
    news = News(failure=RuntimeError("provider secret details"))
    service, provider, _, requests, _ = use_case(news_service=news)
    result = service.request("p1", now=NOW)
    assert result.status == "SUCCESS" and provider.calls == 1
    assert result.news_context.status == "UNAVAILABLE"
    assert not requests.records[0].news_context_used
    assert "provider secret details" not in json.dumps(provider.contexts[0])
    other, other_provider, _, _, _ = use_case(news_service=news, risk_level="HIGH")
    assert other.request("p1", now=NOW).status == "GUARDRAIL_REJECTED"
    assert len(news.calls) == 1 and other_provider.calls == 0


@pytest.mark.parametrize("prob_up,technical_class", [(.547, "UP"), (.453, "DOWN")])
def test_below_experimental_gate_keeps_binary_class_and_skips_news_llm(
    prob_up, technical_class,
):
    value = PredictionRecord(
        prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=prob_up, prob_down=1 - prob_up, predicted_class=technical_class,
        model_version="xgboost_v1", generated_at=NOW, valid_until=VALID_UNTIL,
    )
    news = News()
    service, provider, recommendations, requests, context = use_case(
        value=value, news_service=news,
    )
    result = service.request("p1", now=NOW)

    assert value.predicted_class == technical_class
    assert result.status == requests.records[0].status == "GUARDRAIL_REJECTED"
    assert result.guardrails == ["LOW_MODEL_CONFIDENCE"]
    assert not result.actionable and result.recommendation is None
    assert provider.calls == 0 and news.calls == [] and context.calls == []
    assert not recommendations.records
    assert requests.records[0].news_context_used is False


def test_exact_gate_boundary_preserves_existing_news_and_llm_flow():
    value = PredictionRecord(
        prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=.60, prob_down=.40, predicted_class="UP",
        model_version="xgboost_v1", generated_at=NOW, valid_until=VALID_UNTIL,
    )
    news = News()
    service, provider, recommendations, requests, context = use_case(
        value=value, news_service=news,
    )
    result = service.request("p1", now=NOW)

    assert service._settings.model_confidence_threshold == .60
    assert result.status == "SUCCESS" and result.actionable
    assert len(context.calls) == len(news.calls) == provider.calls == 1
    assert len(recommendations.records) == len(requests.records) == 1


def test_prediction_class_boundary_remains_binary():
    common = dict(prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN,
                  horizon="1m", model_version="xgboost_v1",
                  generated_at=NOW, valid_until=VALID_UNTIL)
    assert PredictionRecord(**common, prob_up=.50, prob_down=.50,
                            predicted_class="DOWN").predicted_class == "DOWN"
    assert PredictionRecord(**common, prob_up=.500001, prob_down=.499999,
                            predicted_class="UP").predicted_class == "UP"
    with pytest.raises(ValidationError):
        PredictionRecord(**common, prob_up=.547, prob_down=.453,
                         predicted_class="HOLD")


def test_optional_news_never_uses_llm_after_prediction_deadline():
    clock = [NOW]

    class SlowNews(News):
        def get_context(self, *, deadline_at):
            clock[0] = deadline_at
            return super().get_context(deadline_at=deadline_at)

    service, provider, _, requests, _ = use_case(news_service=SlowNews(), clock=lambda: clock[0])
    result = service.request("p1", now=NOW)
    assert result.status == "DEADLINE_TOO_CLOSE"
    assert provider.calls == 0
    assert requests.records[0].news_context_fetched_at == NOW


def test_explicit_active_prediction_generates_timely_actionable_recommendation():
    service, provider, recommendations, requests, context = use_case()
    result = service.request("p1", now=NOW)
    assert result.status == "SUCCESS" and result.actionable
    assert result.timing_status == "TIMELY" and result.current_validity(NOW) == "ACTIVE"
    assert result.recommendation.final_signal == "BUY"
    assert provider.timeouts == [45]
    assert requests.records[0].prediction_id == "p1"
    assert recommendations.records
    assert context.calls[0].prediction_id == "p1"


def test_llm_receives_fixed_action_and_its_signal_echo_cannot_change_it():
    service, provider, recommendations, _, _ = use_case(provider=Provider(llm("SELL")))
    result = service.request("p1", now=NOW)
    assert provider.contexts[0]["decision_policy"]["action"] == "BUY"
    assert provider.contexts[0]["decision_policy"]["action_authority"] == "deterministic_rules_only"
    assert result.recommendation.final_signal == "BUY"
    assert result.recommendation.llm_signal == "SELL"
    assert next(iter(recommendations.records.values())).final_signal == "BUY"


def test_llm_receives_long_exit_action_without_authority_to_reverse_it(monkeypatch):
    value = PredictionRecord(
        prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=.3, prob_down=.7, predicted_class="DOWN", model_version="xgboost_v1",
        generated_at=NOW, valid_until=VALID_UNTIL,
    )
    class Paper:
        def get(self, _portfolio_id):
            return {"btc": .01}
        def close(self):
            pass
    monkeypatch.setattr("alpha_engine.repositories.paper.create_paper_repository", lambda _: Paper())
    service, provider, _, _, _ = use_case(value=value, provider=Provider(llm("BUY")))
    result = service.request("p1", now=NOW, portfolio_id=str(uuid4()))
    assert provider.contexts[0]["decision_policy"]["position"] == "LONG"
    assert provider.contexts[0]["decision_policy"]["action"] == "SELL"
    assert result.recommendation.final_signal == "SELL"
    assert result.recommendation.llm_signal == "BUY"


def test_flat_down_prediction_is_wait_in_prompt_not_a_sell():
    value = PredictionRecord(
        prediction_id="p1", symbol="BTCUSDT", timestamp=OPEN, horizon="1m",
        prob_up=.3, prob_down=.7, predicted_class="DOWN", model_version="xgboost_v1",
        generated_at=NOW, valid_until=VALID_UNTIL,
    )
    service, provider, _, _, _ = use_case(value=value)
    result = service.request("p1", now=NOW)
    assert provider.contexts[0]["decision_policy"]["action"] == "WAIT"
    assert result.recommendation.final_signal == "HOLD"  # legacy audit enum; UI shows WAIT


def test_late_response_is_persisted_for_audit_but_not_actionable():
    after_deadline = VALID_UNTIL + timedelta(seconds=1)
    service, provider, recommendations, requests, _ = use_case(clock=lambda: after_deadline)
    result = service.request("p1", now=NOW)
    saved = next(iter(recommendations.records.values()))
    assert provider.calls == 1 and result.status == "LATE_RESULT"
    assert result.timing_status == saved.timing_status == "LATE"
    assert result.current_validity(after_deadline) == "EXPIRED"
    assert not result.actionable and requests.records[0].status == "LATE_RESULT"


def test_expired_or_missing_prediction_never_calls_provider_and_is_audited():
    service, provider, _, requests, _ = use_case()
    expired = service.request("p1", now=VALID_UNTIL)
    missing = service.request("missing", now=NOW)
    no_prediction_service, _, _, no_prediction_requests, _ = use_case(value=prediction("other"))
    no_prediction_service._predictions.records = {}
    none = no_prediction_service.request(now=NOW)
    assert [expired.status, missing.status, none.status] == [
        "EXPIRED_BEFORE_REQUEST", "PREDICTION_NOT_FOUND", "NO_ACTIVE_PREDICTION"
    ]
    assert provider.calls == 0
    assert [row.status for row in requests.records] == [
        "EXPIRED_BEFORE_REQUEST", "PREDICTION_NOT_FOUND"
    ]
    assert [row.status for row in no_prediction_requests.records] == ["NO_ACTIVE_PREDICTION"]


def test_dynamic_deadline_caps_timeout_and_too_short_request_skips_provider():
    settings = Settings(_env_file=None, llm_live_timeout_seconds=90, llm_deadline_margin_seconds=1)
    service, provider, _, requests, _ = use_case(settings=settings)
    near = VALID_UNTIL - timedelta(seconds=11)
    result = service.request("p1", now=near)
    assert result.status == "SUCCESS" and provider.timeouts == [10]
    separate, second_provider, _, second_requests, _ = use_case(settings=settings)
    too_late = separate.request("p1", now=VALID_UNTIL - timedelta(milliseconds=500))
    assert too_late.status == "DEADLINE_TOO_CLOSE" and second_provider.calls == 0
    assert second_requests.records[-1].status == "DEADLINE_TOO_CLOSE"


@pytest.mark.parametrize("failure,status", [
    (LLMProviderTimeoutError("late"), "LLM_TIMEOUT"),
    (LLMProviderError("provider"), "PROVIDER_ERROR"),
    ("not-json", "INVALID_RESPONSE"),
])
def test_provider_failures_are_sanitized_and_audited(failure, status):
    service, _, recommendations, requests, _ = use_case(provider=Provider(failure))
    result = service.request("p1", now=NOW)
    audit = requests.records[0]
    assert result.status == audit.status == status
    assert not result.actionable and not recommendations.records
    assert audit.error_type is not None
    assert audit.error_message in {
        "LLM request exceeded its deadline",
        "LLM provider request failed",
        "LLM response did not satisfy the required schema",
    }


def test_guardrail_rejection_does_not_call_provider_or_create_actionable_cache():
    service, provider, recommendations, requests, _ = use_case(risk_level="HIGH")
    result = service.request("p1", now=NOW)
    assert result.status == "GUARDRAIL_REJECTED"
    assert "HIGH_RISK" in result.guardrails
    assert provider.calls == 0 and not recommendations.records
    assert not result.actionable and requests.records[0].status == "GUARDRAIL_REJECTED"


def test_invalid_prediction_is_audited_without_calling_the_provider():
    corrupt = prediction().model_copy(update={"model_version": ""})
    service, provider, recommendations, requests, _ = use_case(value=corrupt)
    result = service.request("p1", now=NOW)
    assert result.status == "PREDICTION_INVALID"
    assert provider.calls == 0 and not recommendations.records
    assert requests.records[0].status == "PREDICTION_INVALID"


def test_cache_requires_matching_context_timely_and_active():
    service, provider, _, requests, _ = use_case()
    first = service.request("p1", now=NOW)
    second = service.request("p1", now=NOW + timedelta(seconds=1))
    assert first.status == "SUCCESS" and second.status == "CACHED"
    assert provider.calls == 1 and second.actionable
    assert requests.records[-1].cached
    expired = service.request("p1", now=VALID_UNTIL)
    assert expired.status == "EXPIRED_BEFORE_REQUEST" and provider.calls == 1


def test_late_cache_is_never_reused_as_actionable():
    service, provider, recommendations, _, _ = use_case()
    late = RecommendationRecord(
        prediction_id="p1", prompt_version=PROMPT_VERSION, llm_model="test-llm",
        timestamp=OPEN, requested_at=NOW, generated_at=VALID_UNTIL + timedelta(seconds=1),
        valid_until=VALID_UNTIL, llm_latency_ms=100, total_latency_ms=100,
        candidate_signal="BUY", llm_signal="BUY", final_signal="BUY", risk_level="LOW",
        context_json="different", response_json=llm(), guardrails_json="[]",
    )
    recommendations.records[("p1", PROMPT_VERSION, "test-llm")] = late
    result = service.request("p1", now=NOW)
    assert provider.calls == 1 and result.status == "SUCCESS"
    assert not result.cached


def test_concurrent_cache_race_keeps_one_cache_record_and_audits_each_attempt():
    recommendations = Recommendations()
    recommendations.force_race = True
    requests = Requests()
    service, provider, _, _, _ = use_case(recommendations=recommendations, requests=requests)
    result = service.request("p1", now=NOW)
    assert result.status == "CACHED" and result.cached and result.actionable
    assert len(recommendations.records) == 1 and len(requests.records) == 1
    assert provider.calls == 1


def test_context_has_no_future_evaluation_or_future_candle_data():
    service, provider, _, _, _ = use_case()
    service.request("p1", now=NOW)
    context = provider.contexts[0]
    encoded = json.dumps(context)
    assert "evaluation" not in encoded and "observed" not in encoded
    assert context["market"]["timestamp"] == OPEN.isoformat().replace("+00:00", "Z")
    assert context["market"]["close_time"] == CLOSE.isoformat().replace("+00:00", "Z")


def test_observability_reports_attempts_timing_and_failures_without_llm_accuracy():
    service, _, _, requests, _ = use_case()
    service.request("p1", now=NOW)
    service.request("p1", now=VALID_UNTIL)
    metrics = RecommendationObservabilityService(requests).get("BTCUSDT")
    assert metrics.last_attempt.status == "EXPIRED_BEFORE_REQUEST"
    assert metrics.last_successful_attempt.status == "SUCCESS"
    assert metrics.timely_count == 1 and metrics.late_count == 0
    assert metrics.timeout_count == metrics.provider_error_count == 0
    assert metrics.average_llm_latency_ms is not None


def test_naive_request_clock_is_rejected():
    service, _, _, _, _ = use_case()
    with pytest.raises(ValueError, match="zona horaria"):
        service.request("p1", now=NOW.replace(tzinfo=None))
