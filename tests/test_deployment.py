"""Offline checks for cloud boundaries; no provider account is required."""

import hashlib
import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.data.recommendation_response import RecommendationResult
from alpha_engine.services.recommendation_api import (
    RecommendationApiClient, RecommendationApiError, handle_recommendation_request,
)
from scripts.package_azure_functions import package
from streamlit_app.ui import _is_supabase_write_key


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 21, 10, 0, tzinfo=timezone.utc)
PREDICTION_ID = "a" * 64


class Requests:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.closed = False

    def list_recent(self, symbol, limit=100):
        return self.rows[:limit]

    def save(self, row):
        self.rows.append(row)

    def close(self):
        self.closed = True


class UseCase:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def request(self, prediction_id):
        return RecommendationResult(
            request_id=uuid4(), prediction_id=prediction_id, status="SUCCESS",
            requested_at=NOW, generated_at=NOW + timedelta(seconds=2),
            finished_at=NOW + timedelta(seconds=2), valid_until=NOW + timedelta(minutes=1),
            timing_status="TIMELY", llm_latency_ms=1900, total_latency_ms=2000,
            cached=False, actionable=True,
        )


def settings(**kwargs):
    return Settings(_env_file=None, storage_provider="supabase", **kwargs)


def test_http_endpoint_rejects_extra_fields_before_opening_storage():
    for payload in ({"prediction_id": PREDICTION_ID, "prompt": "override"},
                    {"prediction_id": "bad"}, [], None):
        reply = handle_recommendation_request(payload, settings(),
            request_repository_factory=lambda _: pytest.fail("storage opened"))
        assert reply.status_code == 400


def test_http_endpoint_returns_only_sanitized_result():
    store = Requests()
    reply = handle_recommendation_request(
        {"prediction_id": PREDICTION_ID}, settings(),
        use_case_factory=lambda _: UseCase(), request_repository_factory=lambda _: store,
        now=lambda: NOW,
    )
    assert reply.status_code == 200 and store.closed
    assert reply.body["prediction_id"] == PREDICTION_ID
    assert not {"guardrails", "context_json", "raw_response", "prompt", "error_message"} & reply.body.keys()


def test_http_endpoint_limits_repeated_requests():
    from types import SimpleNamespace
    store = Requests(SimpleNamespace(prediction_id=PREDICTION_ID,
                                     requested_at=NOW - timedelta(seconds=1)) for _ in range(3))
    reply = handle_recommendation_request(
        {"prediction_id": PREDICTION_ID}, settings(),
        use_case_factory=lambda _: pytest.fail("LLM called"),
        request_repository_factory=lambda _: store, now=lambda: NOW,
    )
    assert reply.status_code == 429 and store.closed
    assert store.rows[-1].status == "RATE_LIMITED"


def test_http_endpoint_hides_storage_errors():
    reply = handle_recommendation_request(
        {"prediction_id": PREDICTION_ID}, settings(),
        request_repository_factory=lambda _: (_ for _ in ()).throw(RuntimeError("secret")),
        now=lambda: NOW,
    )
    assert reply.status_code == 503 and "secret" not in json.dumps(reply.body)


def test_client_uses_function_key_header_and_validates_response():
    seen = {}
    def respond(request):
        seen["header"] = request.headers.get("x-functions-key")
        seen["payload"] = request.content
        return httpx.Response(200, json=UseCase().request(PREDICTION_ID).model_dump(mode="json"))
    client = RecommendationApiClient(settings(
        recommendation_api_url="https://worker.example/api/recommendation",
        recommendation_api_key="test-function-key"), transport=httpx.MockTransport(respond))
    result = client.request(PREDICTION_ID)
    assert result.status == "SUCCESS"
    assert seen["header"] == "test-function-key"
    assert b"test-function-key" not in seen["payload"]


def test_client_uses_bearer_header_for_render_without_leaking_key_in_body():
    seen = {}

    def respond(request):
        seen["authorization"] = request.headers.get("authorization")
        seen["function_key"] = request.headers.get("x-functions-key")
        seen["payload"] = request.content
        return httpx.Response(200, json=UseCase().request(PREDICTION_ID).model_dump(mode="json"))

    client = RecommendationApiClient(settings(
        recommendation_api_url="https://render.example/api/recommendation",
        recommendation_api_key="test-render-key", recommendation_api_auth_scheme="bearer",
    ), transport=httpx.MockTransport(respond))
    assert client.request(PREDICTION_ID).status == "SUCCESS"
    assert seen["authorization"] == "Bearer test-render-key"
    assert seen["function_key"] is None
    assert b"test-render-key" not in seen["payload"]


def test_client_rejects_plain_http_and_sanitizes_http_failure():
    with pytest.raises(ValueError, match="HTTPS"):
        RecommendationApiClient(settings(
            recommendation_api_url="http://worker.example/api/recommendation",
            recommendation_api_key="key"))
    client = RecommendationApiClient(settings(
        recommendation_api_url="https://worker.example/api/recommendation",
        recommendation_api_key="key"),
        transport=httpx.MockTransport(lambda _: httpx.Response(500, text="private failure")))
    with pytest.raises(RecommendationApiError) as error:
        client.request(PREDICTION_ID)
    assert "private failure" not in str(error.value)


def test_cloud_key_guard_rejects_secret_and_legacy_service_role():
    import base64
    encoded = base64.urlsafe_b64encode(b'{"role":"service_role"}').decode().rstrip("=")
    assert _is_supabase_write_key("sb_secret_test")
    assert _is_supabase_write_key(f"x.{encoded}.y")
    assert not _is_supabase_write_key("sb_publishable_test")


@pytest.mark.parametrize("env,expected", [
    ({"SUPABASE_KEY": "sb_secret_test"}, "clave de escritura"),
    ({"SUPABASE_KEY": "sb_publishable_test", "DASHBOARD_FALLBACK_ENABLED": "true"},
     "Configuraci"),
])
def test_streamlit_cloud_rejects_unsafe_configuration(monkeypatch, env, expected):
    from streamlit.testing.v1 import AppTest
    from alpha_engine.config.settings import get_settings
    from streamlit_app.ui import _load_dashboard
    for key, value in {"PRESCRIPTIVE_BACKEND": "http", "STORAGE_PROVIDER": "supabase",
                       "DASHBOARD_FALLBACK_ENABLED": "false", **env}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    _load_dashboard.clear()
    try:
        app = AppTest.from_file(ROOT / "streamlit_app" / "Home.py").run()
        assert any(expected in item.value for item in app.error)
    finally:
        get_settings.cache_clear()
        _load_dashboard.clear()


def test_package_contains_exact_worker_artifact_without_snapshot_or_secrets():
    target = package()
    artifact = target / "models" / "xgboost_v1.joblib"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
    )
    assert (target / "function_app.py").is_file()
    assert not (target / ".env").exists()
    assert not (target / "public_demo" / "data" / "snapshot.json").exists()


def test_public_economic_artifact_preserves_aggregates_without_trade_prices(tmp_path, monkeypatch):
    from scripts import export_public_economic_summary as exporter
    source = {"runs": [{"total": {"tested_rows": 10, "trades": [
        {"entry_open_observed": 100, "entry_time": "private"}
    ]}}]}
    source_path = tmp_path / "source.json"
    target_path = tmp_path / "summary.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(exporter, "SOURCE", source_path)
    monkeypatch.setattr(exporter, "TARGET", target_path)
    exporter.summarize()
    assert json.loads(target_path.read_text(encoding="utf-8")) == {
        "runs": [{"total": {"tested_rows": 10}}]
    }
    summary = json.loads((ROOT / "models" / "economic_summary_v1.json").read_text(encoding="utf-8"))
    assert len(summary["runs"]) > 0
    def has_trade_rows(value):
        if isinstance(value, dict):
            return "trades" in value or any(has_trade_rows(item) for item in value.values())
        return isinstance(value, list) and any(has_trade_rows(item) for item in value)
    assert not has_trade_rows(summary)


def test_public_sql_views_are_explicit_and_do_not_expose_private_columns():
    sql = (ROOT / "supabase" / "migrations" /
           "20260922030000_add_public_dashboard_views.sql").read_text().lower()
    assert "security_invoker = false" in sql  # deliberate sanitized projection
    assert "select *" not in sql
    assert "null::text as error_message" in sql
    for private in ("context_json", "response_json", "raw_response", "prompt_text"):
        assert private not in sql
    for table in ("public.pipeline_runs", "public.recommendation_requests"):
        migrations = "\n".join(p.read_text().lower() for p in (ROOT / "supabase" / "migrations").glob("*.sql"))
        assert f"alter table {table} enable row level security" in migrations
        assert f"revoke all on {table}" in migrations or table == "public.pipeline_runs"


def test_render_cron_migration_keeps_endpoint_token_out_of_versioned_sql():
    sql = (ROOT / "supabase" / "migrations" /
           "20260923024505_schedule_render_live_cycle.sql").read_text(encoding="utf-8")
    assert "'* * * * *'" in sql
    assert "net.http_post(" in sql
    assert "vault.decrypted_secrets" in sql
    assert "alpha_engine_internal_cron_secret" in sql
    assert "alpha_engine_render_cycle_url" in sql
    assert "https://alpha-engine-api-" not in sql
    assert "sb_secret_" not in sql
    assert "revoke all on vault.secrets, vault.decrypted_secrets" in sql


def test_azure_wrapper_invokes_common_cycle_without_startup_run(monkeypatch):
    from alpha_engine.services.recommendation_api import ApiReply
    observed = {}
    class FakeApp:
        def timer_trigger(self, **kwargs):
            observed["timer"] = kwargs
            return lambda function: function
        def route(self, **kwargs):
            observed["route"] = kwargs
            return lambda function: function
    class FakeResponse:
        def __init__(self, body, status_code, mimetype):
            self.body, self.status_code, self.mimetype = body, status_code, mimetype
    fake_functions = types.ModuleType("azure.functions")
    fake_functions.FunctionApp = FakeApp
    fake_functions.AuthLevel = types.SimpleNamespace(FUNCTION="function")
    fake_functions.TimerRequest = type("TimerRequest", (), {})
    fake_functions.HttpRequest = type("HttpRequest", (), {})
    fake_functions.HttpResponse = FakeResponse
    fake_azure = types.ModuleType("azure")
    fake_azure.functions = fake_functions
    monkeypatch.setitem(sys.modules, "azure", fake_azure)
    monkeypatch.setitem(sys.modules, "azure.functions", fake_functions)
    path = ROOT / "deploy" / "azure_functions" / "function_app.py"
    spec = importlib.util.spec_from_file_location("test_function_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert observed["timer"]["schedule"] == "%LIVE_CYCLE_SCHEDULE%"
    assert observed["timer"]["run_on_startup"] is False
    assert observed["route"]["auth_level"] == "function"
    monkeypatch.setattr(module, "get_settings", settings)
    monkeypatch.setattr(module, "_verify_model", lambda: None)
    monkeypatch.setattr(module, "run_live_cycle", lambda selected: observed.update(
        {"selected": selected.storage_provider}) or types.SimpleNamespace(
            status="NO_NEW_CANDLE", run_id="run", prediction_id=None, candles_ingested=0))
    module.live_cycle_timer(None)
    assert observed["selected"] == "supabase"
    monkeypatch.setattr(module, "handle_recommendation_request",
                        lambda payload, selected: ApiReply(200, {"status": "CACHED"}))
    response = module.recommendation(types.SimpleNamespace(get_json=lambda: {
        "prediction_id": PREDICTION_ID}))
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "CACHED"}
