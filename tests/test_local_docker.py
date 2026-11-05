"""Local wiring checks; no Docker daemon, Binance, news API or LLM calls."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import Candle, PredictionRecord
from alpha_engine.jobs.local_bootstrap import bootstrap, causal_source_pairs
from alpha_engine.jobs.local_scheduler import run_once, seconds_to_next_tick
from alpha_engine.repositories.postgresql import PostgresMarketRepository, PostgresPrescriptiveNewsRepository
from alpha_engine.repositories.supabase import SupabaseMarketRepository
from alpha_engine.services.local_demo import demo_enabled, handle_demo_explanation
from deploy.render.app import RenderApp


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, query, params=()):
        self.query, self.params = query, params

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


def test_local_market_repository_reads_ordered_candle_cohort():
    start = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    candle = Candle(symbol="BTCUSDT", interval="1m", open_time=start,
                    close_time=start + timedelta(minutes=1) - timedelta(milliseconds=1),
                    open=100, high=101, low=99, close=100.5, volume=10)
    cursor = _Cursor([SupabaseMarketRepository._row(candle)])
    connection = SimpleNamespace(cursor=lambda: cursor)
    repository = PostgresMarketRepository(connection)
    assert repository.get_candles("BTCUSDT", "1m", start, start + timedelta(minutes=1)) == [candle]
    assert cursor.params == ("BTCUSDT", "1m", start, start + timedelta(minutes=1))
    assert repository.get_latest_candle("BTCUSDT", "1m") == candle


def test_local_news_articles_are_written_to_article_table(monkeypatch):
    repository = PostgresPrescriptiveNewsRepository(SimpleNamespace())
    captured = []
    monkeypatch.setattr(repository, "_insert", lambda row, **kwargs: captured.append((row, kwargs)))
    repository.save_articles([{"article_identifier": "one"}])
    assert captured == [({"article_identifier": "one"}, {"table": "prescriptive_news_articles"})]


def test_local_bootstrap_sources_require_120_contiguous_past_candles():
    start = datetime(2026, 9, 24, tzinfo=timezone.utc)
    candles = [SimpleNamespace(open_time=start + timedelta(minutes=i)) for i in range(125)]
    assert [c.open_time for c in causal_source_pairs(candles)] == [
        c.open_time for c in candles[119:]
    ]
    candles[70] = SimpleNamespace(open_time=start + timedelta(minutes=500))
    assert list(causal_source_pairs(candles)) == []


def test_local_bootstrap_skips_existing_database_without_ingestion():
    cursor = _Cursor([(1,)])
    market = SimpleNamespace(_connection=SimpleNamespace(cursor=lambda: cursor))
    settings = Settings(_env_file=None, app_env="local", storage_provider="postgresql")
    result = bootstrap(settings, market=market, predictions=object(), evaluations=object(),
                       source=object())
    assert result == {"status": "ALREADY_DONE"}
    assert "local_bootstrap_runs" in cursor.query


def test_scheduler_posts_only_to_run_cycle_on_next_minute():
    observed = []

    def respond(request):
        observed.append((request.method, request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"status": "SUCCESS"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert run_once(client, "http://api:8000", "test-secret") == 200
    assert observed == [("POST", "/internal/run-cycle", "Bearer test-secret")]
    assert seconds_to_next_tick(datetime(2026, 9, 24, 12, 0, 30, tzinfo=timezone.utc)) == 35


def test_demo_disabled_by_default_and_only_local_postgres():
    default = Settings(_env_file=None)
    assert default.local_demo_mode is False
    assert default.model_confidence_threshold == 0.60
    assert not demo_enabled(default)
    local = Settings(_env_file=None, app_env="local", storage_provider="postgresql",
                     local_demo_mode=True)
    assert demo_enabled(local)
    assert not demo_enabled(local.model_copy(update={"app_env": "cloud"}))


def test_demo_missing_llm_key_is_sanitized_and_makes_no_external_call():
    settings = Settings(_env_file=None, app_env="local", storage_provider="postgresql",
                        local_demo_mode=True, llm_provider="nvidia", nvidia_api_key="")
    response = handle_demo_explanation({"prediction_id": "a" * 64}, settings)
    assert response.status_code == 200
    assert response.body == {"status": "LLM_NOT_CONFIGURED", "demo_only": True}


@pytest.mark.parametrize("prob_up,technical_class", [(0.547, "UP"), (0.453, "DOWN")])
def test_local_demo_uses_real_prediction_and_separate_news_context_with_mocks(
    monkeypatch, prob_up, technical_class,
):
    import alpha_engine.services.local_demo as demo_module
    from alpha_engine.data.news_context import NewsContext

    instant = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    prediction = PredictionRecord(
        prediction_id="b" * 64, symbol="BTCUSDT", timestamp=instant,
        horizon="1m", prob_up=prob_up, prob_down=1 - prob_up,
        predicted_class=technical_class, model_version="xgboost_v1",
        generated_at=instant + timedelta(seconds=60),
        valid_until=instant + timedelta(seconds=120),
    )
    settings = Settings(_env_file=None, app_env="local", storage_provider="postgresql",
                        local_demo_mode=True, llm_provider="nvidia", nvidia_api_key="fake-key",
                        prescriptive_news_enabled=True)
    events = []
    resource = SimpleNamespace(close=lambda: None)
    predictions = SimpleNamespace(get_by_id=lambda _: prediction, close=lambda: None)
    monkeypatch.setattr(demo_module, "create_market_repository", lambda _: resource)
    monkeypatch.setattr(demo_module, "create_prediction_repository", lambda _: predictions)
    monkeypatch.setattr(demo_module, "PredictionService", lambda *args, **kwargs: object())
    monkeypatch.setattr(demo_module, "RepositoryAnalyticsTools", lambda *args, **kwargs: object())
    monkeypatch.setattr(demo_module, "RecommendationContextService", lambda _: SimpleNamespace(
        build_snapshot_for_prediction=lambda _: object()))
    monkeypatch.setattr(demo_module, "build_context", lambda *_: {"prediction": {"predicted_class": technical_class}})
    monkeypatch.setattr(demo_module, "PrescriptiveNewsContextService", lambda _: SimpleNamespace(
        get_context=lambda: (events.append("news") or NewsContext(status="NO_NEWS"))))

    class FakeProvider:
        def close(self):
            pass

        def generate_recommendation(self, context, *, timeout_seconds):
            events.append("llm")
            assert context["news_context"]["status"] == "NO_NEWS"
            assert context["local_demo"]["gate_met"] is False
            return ('{"signal":"HOLD","confidence":0.2,"risk_level":"LOW",'
                    '"summary":"Explicación de clase","key_factors":["Datos técnicos"],'
                    '"risks":["Incertidumbre"],"warning":"Sin operación"}')

    monkeypatch.setattr(demo_module, "create_llm_provider", lambda _: FakeProvider())
    monkeypatch.setattr(demo_module, "_audit", lambda *args: events.append("audit"))
    reply = demo_module.handle_demo_explanation({"prediction_id": prediction.prediction_id}, settings)
    assert reply.status_code == 200
    assert reply.body["status"] == "DEMO_ONLY"
    assert reply.body["technical_direction"] == technical_class
    assert reply.body["gate_met"] is False
    assert "signal" not in reply.body["explanation"]
    assert events == ["news", "llm", "audit"]


def test_asgi_demo_route_is_restricted_to_local_flag(monkeypatch):
    from alpha_engine.services import local_demo

    calls = []
    monkeypatch.setattr(local_demo, "handle_demo_explanation", lambda payload, settings: (
        calls.append(payload) or SimpleNamespace(status_code=200, body={"status": "DEMO_ONLY"})
    ))
    settings = Settings(_env_file=None, app_env="local", storage_provider="postgresql",
                        database_url="postgresql://local", local_demo_mode=False)
    app = RenderApp(
        settings_factory=lambda: settings,
        model_verifier=lambda: Path("models/xgboost_v1.joblib"),
        secrets={"INTERNAL_CRON_SECRET": "c" * 40, "RECOMMENDATION_API_SECRET": "r" * 40},
    )

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            return await client.post("/api/demo-explanation", json={"prediction_id": "a" * 64},
                                     headers={"Authorization": "Bearer " + "r" * 40})

    assert asyncio.run(request()).status_code == 404
    assert calls == []
    app._settings_factory = lambda: settings.model_copy(update={"local_demo_mode": True})
    assert asyncio.run(request()).json() == {"status": "DEMO_ONLY"}
    assert calls == [{"prediction_id": "a" * 64}]
