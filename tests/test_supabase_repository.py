"""Supabase adapter tests with an in-memory Data API fake; no network required."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from supabase import ClientOptions, create_client

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle, PredictionRecord, RecommendationRecord
from alpha_engine.data.live_schemas import LiveEvaluationRecord, PipelineRunRecord
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.repositories import supabase as adapter
from alpha_engine.repositories.factory import create_market_repository

ROOT = Path(__file__).resolve().parents[1]
OPEN = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table = table
        self.action = ""
        self.payload = None
        self.options = {}
        self.filters = []
        self.orders = []
        self.sliced = None
        self.count = None

    def select(self, columns):
        self.action = "select"
        self.options["columns"] = columns
        return self

    def upsert(self, payload, **options):
        self.action = "upsert"
        self.payload = payload
        self.options = options
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def gte(self, column, value):
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column, value):
        self.filters.append(("lte", column, value))
        return self

    def lt(self, column, value):
        self.filters.append(("lt", column, value))
        return self

    def order(self, column, desc=False):
        self.orders.append((column, desc))
        return self

    def range(self, start, stop):
        self.sliced = (start, stop)
        return self

    def limit(self, count):
        self.count = count
        return self

    def execute(self):
        self.client.calls.append(self)
        if self.client.error:
            raise RuntimeError("Data API failed")
        if self.action == "upsert":
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            keys = self.options["on_conflict"].split(",")
            returned = []
            for row in rows:
                previous = next((old for old in self.client.rows[self.table]
                                 if all(old[key] == row[key] for key in keys)), None)
                if previous is None:
                    self.client.rows[self.table].append(dict(row))
                    returned.append(dict(row))
                elif not self.options.get("ignore_duplicates"):
                    previous.update(row)
                    returned.append(dict(previous))
            return SimpleNamespace(data=returned if self.options.get("returning") == "representation" else [])
        rows = list(self.client.rows[self.table])
        for operation, column, value in self.filters:
            if operation == "eq":
                rows = [row for row in rows if row[column] == value]
            elif operation == "gte":
                rows = [row for row in rows if (float(row[column]) if isinstance(value, float) else row[column]) >= value]
            elif operation == "lte":
                rows = [row for row in rows if row[column] <= value]
            else:
                rows = [row for row in rows if row[column] < value]
        for column, desc in reversed(self.orders):
            rows.sort(key=lambda row: row[column], reverse=desc)
        if self.sliced:
            rows = rows[self.sliced[0]:self.sliced[1] + 1]
        if self.count is not None:
            rows = rows[:self.count]
        return SimpleNamespace(data=rows)


class FakeClient:
    def __init__(self):
        self.rows = {"candles": [], "predictions": [], "recommendations": [],
                     "recommendation_history": [], "recommendation_requests": [],
                     "pipeline_runs": [], "live_evaluations": []}
        self.calls = []
        self.error = False
        self.closed = 0
        self.postgrest = SimpleNamespace(aclose=self.close)

    def table(self, name):
        name = {
            "public_pipeline_runs": "pipeline_runs",
            "public_recommendation_request_summary": "recommendation_requests",
        }.get(name, name)
        return FakeQuery(self, name)

    def close(self):
        self.closed += 1


def candle(at=OPEN, price=100.125):
    return Candle(symbol="BTCUSDT", interval="1m", open_time=at,
                  close_time=at + timedelta(milliseconds=59_999),
                  open=price, high=price + 1, low=price - 1,
                  close=price, volume=0.123456789)


def prediction(at=OPEN, prediction_id="p1"):
    return PredictionRecord(prediction_id=prediction_id, symbol="BTCUSDT", timestamp=at,
                            horizon="1m", prob_up=0.625, prob_down=0.375,
                            predicted_class="UP", model_version="xgboost_v1",
                            trained_at=OPEN - timedelta(days=2))


def test_replay_prediction_queries_filter_in_one_data_api_request_each():
    client = FakeClient()
    repo = adapter.SupabasePredictionRepository(client)
    earlier = prediction(OPEN - timedelta(minutes=2), "old")
    later = prediction(OPEN, "new")
    client.rows["predictions"] = [repo._row(earlier), repo._row(later)]
    assert repo.list_entry_signals("BTCUSDT", "xgboost_v1", .6, 100) == [later, earlier]
    assert len(client.calls) == 1
    assert ("gte", "prob_up", .6) in client.calls[-1].filters
    assert ("eq", "model_version", "xgboost_v1") in client.calls[-1].filters
    assert repo.list_recent_before("BTCUSDT", OPEN - timedelta(minutes=1), 60) == [earlier]
    assert len(client.calls) == 2
    assert ("lte", "candle_open_time", (OPEN - timedelta(minutes=1)).isoformat()) in client.calls[-1].filters


def recommendation(at=OPEN, generated_at=OPEN + timedelta(seconds=65)):
    return RecommendationRecord(prediction_id="p1", prompt_version="recommendation_v3",
                                llm_model="model", timestamp=at, generated_at=generated_at,
                                candidate_signal="BUY", llm_signal="BUY", final_signal="BUY",
                                risk_level="LOW", context_json='{"private":true}',
                                response_json='{"summary":"ok"}', guardrails_json="[]")


def recommendation_request(requested_at=OPEN + timedelta(seconds=5)):
    return RecommendationRequestRecord(
        request_id=UUID("00000000-0000-0000-0000-000000000010"), prediction_id="p1",
        symbol="BTCUSDT", interval="1m", provider="ollama", llm_model="model",
        prompt_version="recommendation_v3", requested_at=requested_at,
        generated_at=requested_at + timedelta(seconds=1),
        finished_at=requested_at + timedelta(seconds=1), valid_until=OPEN + timedelta(minutes=2),
        llm_latency_ms=900, total_latency_ms=1000, status="SUCCESS", timing_status="TIMELY",
    )


def test_candle_mapping_utc_numeric_ranges_and_upsert():
    client = FakeClient()
    repo = adapter.SupabaseMarketRepository(client)
    lima = timezone(timedelta(hours=-5))
    source = candle(OPEN.astimezone(lima))
    assert repo.upsert_candles([source, source]) == 2
    assert len(client.rows["candles"]) == 1
    row = client.rows["candles"][0]
    assert row["open_time"] == OPEN.isoformat()
    assert row["close_time"] == source.close_time.astimezone(timezone.utc).isoformat()
    assert row["volume"] == "0.123456789"
    assert repo.get_latest_candle("BTCUSDT", "1m") == candle()
    assert repo.get_candles("BTCUSDT", "1m", OPEN, OPEN + timedelta(minutes=1)) == [candle()]
    assert repo.get_candles("BTCUSDT", "1m", OPEN + timedelta(minutes=1), OPEN + timedelta(minutes=2)) == []
    assert client.calls[-1].sliced == (0, 999)
    with pytest.raises(ValueError, match="stop"):
        repo.get_candles("BTCUSDT", "1m", OPEN, OPEN)
    with pytest.raises(ValueError, match="zona horaria"):
        repo.get_candles("BTCUSDT", "1m", OPEN.replace(tzinfo=None), OPEN)


def test_candle_range_paginates_and_orders():
    client = FakeClient()
    repo = adapter.SupabaseMarketRepository(client)
    repo.upsert_candles([candle(OPEN + timedelta(minutes=i)) for i in range(1001)])
    result = repo.get_candles("BTCUSDT", "1m", OPEN, OPEN + timedelta(minutes=1001))
    assert len(result) == 1001
    assert result[0].open_time == OPEN
    assert result[-1].open_time == OPEN + timedelta(minutes=1000)
    assert [call.sliced for call in client.calls if call.action == "select"] == [(0, 999), (1000, 1999)]


def test_pending_rpc_and_evaluation_batch_use_two_data_api_round_trips():
    class RpcClient(FakeClient):
        def rpc(self, name, params):
            self.rpc_call = (name, params)
            self.calls.append((name, params))
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{
                "prediction_id": "p1", "symbol": "BTCUSDT", "predicted_class": "UP",
                "source_close": "100", "target_open_time": (OPEN + timedelta(minutes=1)).isoformat(),
                "target_close_time": (OPEN + timedelta(minutes=2) - timedelta(milliseconds=1)).isoformat(),
                "target_close": "101",
            }]))

    client = RpcClient()
    predictions = adapter.SupabasePredictionRepository(client)
    evaluations = adapter.SupabaseLiveEvaluationRepository(client)
    now = OPEN + timedelta(minutes=2)
    pending = predictions.list_evaluable_pending("BTCUSDT", now)
    assert len(pending) == 1 and pending[0].target_close == 101
    assert client.rpc_call == ("pending_live_evaluations", {
        "p_symbol": "BTCUSDT", "p_now": now.isoformat(), "p_limit": 100,
    })
    row = LiveEvaluationRecord(
        evaluation_id=UUID("00000000-0000-0000-0000-000000000001"),
        prediction_id="p1", symbol="BTCUSDT", evaluated_at=now,
        source_close=100, target_open_time=OPEN + timedelta(minutes=1),
        target_close_time=OPEN + timedelta(minutes=2) - timedelta(milliseconds=1),
        target_close=101, observed_move="UP", observed_class="UP", correct=True,
    )
    assert evaluations.save_batch([row]) == {"p1"}
    assert evaluations.save_batch([row]) == set()
    assert len([call for call in client.calls if getattr(call, "action", None) == "upsert"]) == 2
    assert evaluations.save_batch([]) == set()


def test_prediction_mapping_identity_and_latest():
    client = FakeClient()
    repo = adapter.SupabasePredictionRepository(client)
    first = prediction()
    second = prediction(OPEN + timedelta(minutes=1), "p2")
    assert repo.save(first) is True
    assert repo.save(first) is False
    assert repo.save(second) is True
    assert len(client.rows["predictions"]) == 2
    assert client.rows["predictions"][0]["prob_up"] == "0.625"
    assert client.calls[0].options["ignore_duplicates"] is True
    assert repo.get_by_id("p1") == first
    assert repo.get_latest("BTCUSDT") == second
    assert repo.get_by_id("missing") is None


def test_recommendation_mapping_cache_and_revision_query():
    client = FakeClient()
    repo = adapter.SupabaseRecommendationRepository(client)
    record = recommendation()
    repo.save(record)
    assert repo.get_cached("p1", "recommendation_v3", "model") == record
    assert client.calls[0].options["on_conflict"] == "prediction_id,prompt_version,llm_model"
    history = adapter.SupabaseRecommendationRepository._row(record)
    history["revision_id"] = "00000000-0000-0000-0000-000000000001"
    client.rows["recommendation_history"].append(history)
    assert repo.list_recent() == [record]
    with pytest.raises(ValueError, match="limit"):
        repo.list_recent(101)


def test_recommendation_request_mapping_is_private_and_idempotent():
    client = FakeClient()
    repo = adapter.SupabaseRecommendationRequestRepository(client)
    request = recommendation_request()
    repo.save(request)
    repo.save(request)
    assert len(client.rows["recommendation_requests"]) == 1
    loaded = repo.get_latest("BTCUSDT")
    assert loaded.request_id == request.request_id
    assert loaded.status == request.status and loaded.timing_status == "TIMELY"
    assert loaded.created_at == request.finished_at
    assert repo.list_recent("BTCUSDT", status="LLM_TIMEOUT") == []
    assert client.rows["recommendation_requests"][0]["request_id"] == str(request.request_id)


def test_client_errors_propagate_and_owned_client_closes_once():
    client = FakeClient()
    repo = adapter.SupabaseMarketRepository(client, owns_client=True)
    client.error = True
    with pytest.raises(RuntimeError, match="Data API"):
        repo.upsert_candles([candle()])
    repo.close()
    repo.close()
    assert client.closed == 1


def test_invalid_config_and_influx_factory_does_not_load_supabase():
    with pytest.raises(ValueError, match="SUPABASE_URL"):
        create_market_repository(Settings(_env_file=None, storage_provider="supabase"))
    assert Settings(_env_file=None, storage_provider="influxdb3").storage_provider == "influxdb3"
    with pytest.raises(ValueError, match="SUPABASE_KEY"):
        adapter._client_from_settings(Settings(
            _env_file=None, storage_provider="supabase", supabase_url="http://127.0.0.1:54321"
        ))


def test_migration_declares_database_uniqueness_and_private_llm_records():
    sql = next((ROOT / "supabase" / "migrations").glob("*_initial_alpha_engine_persistence.sql")).read_text()
    assert "primary key (symbol, interval, open_time)" in sql
    assert "unique nulls not distinct (symbol, interval, candle_open_time, model_version, trained_at)" in sql
    assert "alter table public.recommendation_history enable row level security" in sql
    assert "grant select on public.candles, public.predictions to anon, authenticated" in sql
    assert "create policy recommendations_public" not in sql


def test_live_observability_migration_keeps_runs_private_and_evaluations_unique():
    sql = next((ROOT / "supabase" / "migrations").glob("*_add_live_observability.sql")).read_text()
    assert "add column generated_at timestamptz" in sql
    assert "add column valid_until timestamptz" in sql
    assert "run_id uuid primary key" in sql
    assert "prediction_id text not null unique references public.predictions" in sql
    assert "alter table public.pipeline_runs enable row level security" in sql
    assert "alter table public.live_evaluations enable row level security" in sql
    assert "grant select on public.live_evaluations to anon, authenticated" in sql
    assert "grant select on public.pipeline_runs to anon" not in sql


def test_prescriptive_audit_migration_keeps_requests_and_private_context_private():
    sql = next((ROOT / "supabase" / "migrations").glob("*_add_recommendation_request_audit.sql")).read_text()
    assert "create table public.recommendation_requests" in sql
    assert "request_id uuid primary key" in sql
    assert "alter table public.recommendation_requests enable row level security" in sql
    assert "grant all on public.recommendation_requests to service_role" in sql
    assert "create policy" not in sql


def test_public_dashboard_views_expose_only_sanitized_read_projections():
    sql = next((ROOT / "supabase" / "migrations").glob("*_add_public_dashboard_views.sql")).read_text()
    assert "create view public.public_pipeline_runs" in sql
    assert "create view public.public_recommendation_request_summary" in sql
    assert "null::text as error_message" in sql
    assert "context_json" not in sql and "response_json" not in sql
    assert "grant select on public.public_pipeline_runs to anon, authenticated" in sql
    assert "grant select on public.public_recommendation_request_summary to anon, authenticated" in sql


def test_server_role_can_read_sanitized_dashboard_views():
    sql = next((ROOT / "supabase" / "migrations").glob("*_grant_server_read_public_views.sql")).read_text()
    assert "grant select on public.public_pipeline_runs to service_role" in sql
    assert "grant select on public.public_recommendation_request_summary to service_role" in sql
    assert "grant all" not in sql.lower()


def test_supabase_prediction_timing_and_old_rows_round_trip():
    client = FakeClient()
    repo = adapter.SupabasePredictionRepository(client)
    fresh = prediction()
    fresh.generated_at = OPEN + timedelta(seconds=5)
    fresh.valid_until = OPEN + timedelta(minutes=2) - timedelta(milliseconds=1)
    assert repo.save(fresh) is True
    assert repo.get_by_id("p1") == fresh
    assert repo.get_by_id("p1").timing_status == "TIMELY"
    old = prediction(OPEN + timedelta(minutes=1), "old")
    row = repo._row(old)
    row.pop("generated_at")
    row.pop("valid_until")
    client.rows["predictions"].append(row)
    assert repo.get_by_id("old").generated_at is None
    assert len(repo.list_recent("BTCUSDT")) == 2


def test_supabase_pipeline_run_and_evaluation_contracts():
    client = FakeClient()
    runs = adapter.SupabasePipelineRunRepository(client)
    evaluations = adapter.SupabaseLiveEvaluationRepository(client)
    run = PipelineRunRecord(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        started_at=OPEN, finished_at=OPEN + timedelta(seconds=1),
        duration_ms=950.5, market_data_ms=300, prediction_ms=200,
        status="SUCCESS", symbol="BTCUSDT", interval="1m",
        candle_open_time=OPEN, prediction_id="p1", new_candle=True,
        new_prediction=True, candles_ingested=1,
    )
    runs.save(run)
    runs.save(run)
    assert len(client.rows["pipeline_runs"]) == 1
    assert runs.get_latest("BTCUSDT").run_id == run.run_id
    assert runs.list_recent("BTCUSDT", status="FAILED") == []
    assert client.rows["pipeline_runs"][0]["duration_ms"] == "950.5"
    observed = LiveEvaluationRecord(
        evaluation_id=UUID("00000000-0000-0000-0000-000000000002"),
        prediction_id="p1", symbol="BTCUSDT",
        evaluated_at=OPEN + timedelta(minutes=2), source_close=100,
        target_open_time=OPEN + timedelta(minutes=1),
        target_close_time=OPEN + timedelta(minutes=2) - timedelta(milliseconds=1),
        target_close=101, observed_move="UP", observed_class="UP", correct=True,
    )
    assert evaluations.save(observed) is True
    assert evaluations.save(observed) is False
    assert evaluations.get_by_prediction("p1") == observed
    assert evaluations.get_latest("BTCUSDT") == observed
    assert len(evaluations.list_recent("BTCUSDT")) == 1


def test_official_client_sends_upserts_with_database_conflict_targets():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(201, content=b"", headers={"content-type": "application/json"})

    transport = httpx.MockTransport(respond)
    http_client = httpx.Client(transport=transport)
    client = create_client(
        "http://127.0.0.1:54321", "local-test-key",
        options=ClientOptions(httpx_client=http_client, persist_session=False,
                              auto_refresh_token=False),
    )
    try:
        adapter.SupabaseMarketRepository(client).upsert_candles([candle()])
        adapter.SupabasePredictionRepository(client).save(prediction())
        adapter.SupabaseRecommendationRepository(client).save(recommendation())
    finally:
        client.postgrest.aclose()
        http_client.close()
    assert [request.url.path for request in requests] == [
        "/rest/v1/candles", "/rest/v1/predictions", "/rest/v1/recommendations",
    ]
    assert [request.url.params.get("on_conflict") for request in requests] == [
        "symbol,interval,open_time", "prediction_id",
        "prediction_id,prompt_version,llm_model",
    ]
    assert "resolution=ignore-duplicates" in requests[1].headers["prefer"]
