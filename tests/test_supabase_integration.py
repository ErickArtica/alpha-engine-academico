"""Opt-in tests against local Supabase or one explicitly selected cloud project."""

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from postgrest.exceptions import APIError
from supabase import create_client

from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.repositories.supabase import (
    SupabaseMarketRepository, SupabasePredictionRepository,
    SupabaseRecommendationRepository,
    SupabasePipelineRunRepository, SupabaseLiveEvaluationRepository,
    SupabaseRecommendationRequestRepository,
)

pytestmark = pytest.mark.integration


def _test_credentials(key_name):
    if os.getenv("ALPHA_SUPABASE_INTEGRATION") != "1":
        pytest.skip("Supabase integration no solicitado")
    url = os.getenv("SUPABASE_TEST_URL", "")
    key = os.getenv(key_name, "")
    host = urlparse(url).hostname
    local = host in {"127.0.0.1", "localhost"}
    project_ref = os.getenv("ALPHA_SUPABASE_REMOTE_PROJECT_REF", "")
    remote = (
        os.getenv("ALPHA_SUPABASE_REMOTE_INTEGRATION") == "1"
        and bool(project_ref)
        and host == f"{project_ref}.supabase.co"
    )
    if not (local or remote) or not key:
        pytest.skip("Configure un Supabase local o habilite explicitamente el proyecto remoto")
    return url, key


@pytest.fixture
def local_client():
    url, key = _test_credentials("SUPABASE_TEST_KEY")
    client = create_client(url, key)
    yield client
    client.postgrest.aclose()


@pytest.fixture
def local_anon_client():
    url, key = _test_credentials("SUPABASE_TEST_ANON_KEY")
    client = create_client(url, key)
    yield client
    client.postgrest.aclose()


def test_local_anon_can_read_projections_but_not_private_tables_or_write(local_anon_client):
    client = local_anon_client
    for relation in ("candles", "predictions", "live_evaluations",
                     "public_pipeline_runs", "public_recommendation_request_summary"):
        assert isinstance(client.table(relation).select("*").limit(1).execute().data, list)
    for relation in ("recommendations", "recommendation_history", "pipeline_runs",
                     "recommendation_requests"):
        with pytest.raises(APIError):
            client.table(relation).select("*").limit(1).execute()
    for relation in ("candles", "predictions", "pipeline_runs", "live_evaluations",
                     "recommendations", "recommendation_history", "recommendation_requests"):
        with pytest.raises(APIError) as error:
            client.table(relation).insert({}).execute()
        assert error.value.code == "42501"


def test_local_migrations_constraints_upserts_ranges_and_history(local_client):
    """A fresh db reset must have applied the migration before this test runs."""
    client = local_client
    symbol = "TST" + uuid4().hex[:8].upper()
    first_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    trained_at = first_time - timedelta(days=1)
    prediction_id = uuid4().hex
    market = SupabaseMarketRepository(client)
    predictions = SupabasePredictionRepository(client)
    recommendations = SupabaseRecommendationRepository(client)
    source = Candle(symbol=symbol, interval="1m", open_time=first_time,
                    close_time=first_time + timedelta(milliseconds=59_999),
                    open=100.123456789, high=101.0, low=99.0,
                    close=100.123456789, volume=0.00000001)
    next_candle = source.model_copy(update={
        "open_time": first_time + timedelta(minutes=1),
        "close_time": first_time + timedelta(minutes=1, milliseconds=59_999),
        "close": 100.25,
    })
    prediction = PredictionRecord(prediction_id=prediction_id, symbol=symbol,
                                  timestamp=first_time, horizon="1m",
                                  prob_up=0.625, prob_down=0.375,
                                  predicted_class="UP", model_version="xgboost_v1",
                                  trained_at=trained_at)
    recommendation = RecommendationRecord(
        prediction_id=prediction_id, prompt_version="recommendation_v3",
        llm_model="test-local", timestamp=first_time,
        generated_at=first_time + timedelta(seconds=65),
        candidate_signal="BUY", llm_signal="BUY", final_signal="BUY",
        risk_level="LOW", context_json='{"private":true}',
        response_json='{"summary":"local"}', guardrails_json="[]",
    )
    try:
        market.upsert_candles([source, next_candle])
        market.upsert_candles([source])
        assert len(client.table("candles").select("*").eq("symbol", symbol).execute().data) == 2
        with pytest.raises(APIError):
            client.table("candles").insert(SupabaseMarketRepository._row(source)).execute()
        assert market.get_latest_candle(symbol, "1m") == next_candle
        assert market.get_candles(symbol, "1m", first_time, first_time + timedelta(minutes=1)) == [source]
        predictions.save(prediction)
        predictions.save(prediction)
        assert len(client.table("predictions").select("*").eq("symbol", symbol).execute().data) == 1
        with pytest.raises(APIError):
            client.table("predictions").insert({
                **SupabasePredictionRepository._row(prediction),
                "prediction_id": uuid4().hex,
            }).execute()
        assert predictions.get_by_id(prediction_id) == prediction
        assert predictions.get_latest(symbol) == prediction
        recommendations.save(recommendation)
        revised = recommendation.model_copy(update={
            "generated_at": recommendation.generated_at + timedelta(seconds=2),
            "final_signal": "HOLD",
        })
        recommendations.save(revised)
        assert recommendations.get_cached(prediction_id, "recommendation_v3", "test-local") == revised
        history = (client.table("recommendation_history").select("*")
                   .eq("prediction_id", prediction_id).execute().data)
        assert len(history) == 2
        assert {row["final_signal"] for row in history} == {"BUY", "HOLD"}
    finally:
        client.table("recommendation_history").delete().eq("prediction_id", prediction_id).execute()
        client.table("recommendations").delete().eq("prediction_id", prediction_id).execute()
        client.table("predictions").delete().eq("prediction_id", prediction_id).execute()
        client.table("candles").delete().eq("symbol", symbol).execute()


def test_local_live_observability_migration_and_unique_evaluation(local_client):
    client = local_client
    symbol = "TST" + uuid4().hex[:8].upper()
    opened = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    close = opened + timedelta(minutes=1) - timedelta(milliseconds=1)
    target_open = opened + timedelta(minutes=1)
    target_close = target_open + timedelta(minutes=1) - timedelta(milliseconds=1)
    prediction_id = uuid4().hex
    market = SupabaseMarketRepository(client)
    predictions = SupabasePredictionRepository(client)
    runs = SupabasePipelineRunRepository(client)
    evaluations = SupabaseLiveEvaluationRepository(client)
    candles = [
        Candle(symbol=symbol, interval="1m", open_time=opened, close_time=close,
               open=100, high=101, low=99, close=100, volume=1),
        Candle(symbol=symbol, interval="1m", open_time=target_open, close_time=target_close,
               open=100, high=102, low=99, close=101, volume=1),
    ]
    prediction = PredictionRecord(
        prediction_id=prediction_id, symbol=symbol, timestamp=opened, horizon="1m",
        prob_up=0.7, prob_down=0.3, predicted_class="UP", model_version="xgboost_v1",
        generated_at=close + timedelta(seconds=5), valid_until=target_close,
    )
    run = PipelineRunRecord(
        run_id=uuid4(), started_at=close + timedelta(seconds=5),
        finished_at=close + timedelta(seconds=6), duration_ms=1000,
        market_data_ms=250, prediction_ms=500, status="SUCCESS",
        symbol=symbol, interval="1m", candle_open_time=opened,
        prediction_id=prediction_id, new_candle=True, new_prediction=True,
        candles_ingested=1,
    )
    evaluation = LiveEvaluationRecord(
        evaluation_id=uuid4(), prediction_id=prediction_id, symbol=symbol,
        evaluated_at=target_close + timedelta(seconds=1), source_close=100,
        target_open_time=target_open, target_close_time=target_close,
        target_close=101, observed_move="UP", observed_class="UP", correct=True,
    )
    try:
        market.upsert_candles(candles)
        assert predictions.save(prediction) is True
        assert predictions.get_by_id(prediction_id).timing_status == "TIMELY"
        runs.save(run)
        assert runs.get_latest(symbol).run_id == run.run_id
        assert evaluations.save(evaluation) is True
        assert evaluations.save(evaluation) is False
        with pytest.raises(APIError):
            client.table("live_evaluations").insert({
                **SupabaseLiveEvaluationRepository._row(evaluation),
                "evaluation_id": str(uuid4()),
            }).execute()
        assert evaluations.get_by_prediction(prediction_id) == evaluation
    finally:
        client.table("live_evaluations").delete().eq("prediction_id", prediction_id).execute()
        client.table("pipeline_runs").delete().eq("run_id", str(run.run_id)).execute()
        client.table("predictions").delete().eq("prediction_id", prediction_id).execute()
        client.table("candles").delete().eq("symbol", symbol).execute()


def test_prescriptive_audit_and_public_views_hide_private_errors(local_client, local_anon_client):
    symbol = "TST" + uuid4().hex[:8].upper()
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    run = PipelineRunRecord(
        run_id=uuid4(), started_at=now, finished_at=now + timedelta(seconds=1),
        duration_ms=1000, status="FAILED", symbol=symbol, interval="1m",
        error_type="StorageError", error_message="private-test-error",
    )
    request = RecommendationRequestRecord(
        request_id=uuid4(), symbol=symbol, interval="1m", provider="test",
        llm_model="test-model", prompt_version="test-v1", requested_at=now,
        finished_at=now + timedelta(seconds=1), total_latency_ms=1000,
        status="PROVIDER_ERROR", error_type="ProviderError",
        error_message="private-test-error",
    )
    runs = SupabasePipelineRunRepository(local_client)
    requests = SupabaseRecommendationRequestRepository(local_client)
    try:
        runs.save(run)
        requests.save(request)
        requests.save(request)
        private_requests = (local_client.table("recommendation_requests").select("*")
                            .eq("request_id", str(request.request_id)).execute().data)
        assert len(private_requests) == 1
        assert requests.get_latest(symbol).request_id == request.request_id
        for view, identity, expected_id in (
            ("public_pipeline_runs", "run_id", str(run.run_id)),
            ("public_recommendation_request_summary", "request_id", str(request.request_id)),
        ):
            rows = (local_anon_client.table(view).select("*")
                    .eq("symbol", symbol).execute().data)
            assert len(rows) == 1
            assert rows[0][identity] == expected_id
            assert rows[0]["error_message"] is None
            assert "private-test-error" not in json.dumps(rows)
            assert not {"context_json", "response_json", "guardrails_json"} & rows[0].keys()
    finally:
        local_client.table("recommendation_requests").delete().eq("request_id", str(request.request_id)).execute()
        local_client.table("pipeline_runs").delete().eq("run_id", str(run.run_id)).execute()
