"""News remains optional, server-side context for an explicit request."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.data.news_context import NewsContext
from alpha_engine.news.alpha_vantage import AlphaVantageRateLimit, parse_article, validate_payload
from alpha_engine.repositories.supabase_news import NewsFetchRecord, SupabasePrescriptiveNewsRepository
from alpha_engine.repositories.supabase import SupabaseRecommendationRequestRepository
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.services.prescriptive_news import PrescriptiveNewsContextService, slot_start
from alpha_engine.services.prescriptive_news import AlphaVantageNewsClient


NOW = datetime(2026, 9, 24, 8, 10, tzinfo=timezone.utc)


def article(*, url="https://example.org/btc/1", time="20260924T080500", score="0.4", relevance="0.8"):
    return {"url": url, "time_published": time, "source": "Example",
            "overall_sentiment_score": "-0.9", "ticker_sentiment": [
                {"ticker": "CRYPTO:BTC", "ticker_sentiment_score": score,
                 "relevance_score": relevance, "ticker_sentiment_label": "Somewhat_Bullish"}]}


class MemoryNewsRepository:
    def __init__(self):
        self.slots = {}
        self.articles = {}
        self.closed = 0

    def latest_successful(self):
        records = [row for row in self.slots.values() if row.status in {"SUCCESS", "EMPTY"}]
        return max(records, key=lambda row: row.fetched_at) if records else None

    def claim_slot(self, start, attempted_at, snapshot_id):
        if start in self.slots:
            return False
        self.slots[start] = NewsFetchRecord(snapshot_id, attempted_at, attempted_at, "CLAIMED")
        return True

    def finish(self, snapshot_id, status, fetched_at, article_count):
        for start, row in self.slots.items():
            if row.snapshot_id == snapshot_id:
                self.slots[start] = NewsFetchRecord(snapshot_id, row.attempted_at,
                                                    fetched_at or row.attempted_at, status)
                return
        raise AssertionError("Slot not found")

    def save_articles(self, articles):
        for row in articles:
            self.articles.setdefault(row["article_identifier"], row)

    def articles_at(self, fetched_at):
        return [row for row in self.articles.values()
                if fetched_at - timedelta(minutes=60) < datetime.fromisoformat(row["published_at"]) <= fetched_at
                and datetime.fromisoformat(row["first_observed_at"]) <= fetched_at]

    def close(self):
        self.closed += 1


class Client:
    def __init__(self, feed=None, failure=None):
        self.feed = feed if feed is not None else [article()]
        self.failure = failure
        self.calls = []

    def fetch(self, now, *, timeout_seconds):
        self.calls.append((now, timeout_seconds))
        if self.failure:
            raise self.failure
        return self.feed


def settings(*, key="fake-for-mocked-client", enabled=True):
    return Settings(_env_file=None, storage_provider="supabase", prescriptive_news_enabled=enabled,
                    alpha_vantage_api_key=key)


def service(repo, client, clock, *, key="fake-for-mocked-client", enabled=True):
    return PrescriptiveNewsContextService(
        settings(key=key, enabled=enabled), repository_factory=lambda _: repo,
        client=client, clock=lambda: clock[0],
    )


def test_btc_specific_score_utc_and_provider_errors():
    parsed = parse_article(article())
    assert parsed["score"] == .4 and parsed["relevance"] == .8
    assert parsed["published_at_utc"].utcoffset() == timedelta(0)
    with pytest.raises(AlphaVantageRateLimit):
        validate_payload({"Information": "quota"})
    with pytest.raises(ValueError):
        validate_payload({"Error Message": "invalid"})
    with pytest.raises(ValueError):
        validate_payload({"feed": "invalid"})
    with pytest.raises(ValueError):
        parse_article(article(relevance="0"))


def test_real_client_shape_with_mocked_http_transport(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"feed": [article()]}

    class ClientContext:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(self, endpoint, *, params):
            captured.update({"endpoint": endpoint, "params": params})
            return Response()

    monkeypatch.setattr("alpha_engine.services.prescriptive_news.httpx.Client",
                        lambda **_: ClientContext())
    feed = AlphaVantageNewsClient("mock-only").fetch(NOW, timeout_seconds=3)
    assert len(feed) == 1
    assert captured["params"]["function"] == "NEWS_SENTIMENT"
    assert captured["params"]["tickers"] == "CRYPTO:BTC"
    assert captured["params"]["limit"] == 1000
    assert captured["params"]["time_from"] == "20260924T0710"
    assert captured["params"]["time_to"] == "20260924T0810"
    assert "apikey" not in captured["endpoint"]


def test_explicit_fetch_deduplicates_uses_btc_score_and_persistent_cache():
    repo, clock = MemoryNewsRepository(), [NOW]
    client = Client([article(), article(url="https://example.org/btc/1?tracking=2")])
    instance = service(repo, client, clock)
    first = instance.get_context(deadline_at=NOW + timedelta(seconds=30))
    assert first.status == "AVAILABLE" and first.news_count == 1
    assert first.btc_sentiment_score == pytest.approx(.4)
    assert first.average_relevance == pytest.approx(.8)
    assert len(repo.articles) == 1 and len(client.calls) == 1
    observed = next(iter(repo.articles.values()))["first_observed_at"]
    clock[0] += timedelta(minutes=10)
    second = service(repo, Client(failure=AssertionError("Must use cache")), clock).get_context()
    assert second.status == "AVAILABLE" and second.snapshot_id == first.snapshot_id
    assert next(iter(repo.articles.values()))["first_observed_at"] == observed


def test_stale_cache_and_slot_limit_do_not_repeat_provider_calls():
    repo, clock, client = MemoryNewsRepository(), [NOW], Client()
    instance = service(repo, client, clock)
    assert instance.get_context().status == "AVAILABLE"
    clock[0] += timedelta(minutes=61)
    assert instance.get_context().status == "STALE"
    assert len(client.calls) == 1
    clock[0] = NOW + timedelta(hours=2)
    client.feed = [article(time="20260924T100500", url="https://example.org/btc/2")]
    assert instance.get_context().status == "AVAILABLE"
    assert len(client.calls) == 2
    assert slot_start(NOW + timedelta(hours=2)) != slot_start(NOW)


def test_no_news_is_not_neutral_and_future_article_is_excluded():
    repo, clock = MemoryNewsRepository(), [NOW]
    client = Client([article(time="20260924T081100")])
    result = service(repo, client, clock).get_context()
    assert result.status == "NO_NEWS" and result.btc_sentiment_score is None
    assert not repo.articles
    assert "Absence of recent news is not confirmed neutral" in result.for_prompt()["limitation"]


@pytest.mark.parametrize("failure,reason", [
    (AlphaVantageRateLimit("quota"), "RATE_LIMIT"),
    (httpx.TimeoutException("slow"), "PROVIDER_UNAVAILABLE"),
    (ValueError("schema"), "INVALID_RESPONSE"),
])
def test_provider_failure_falls_back_and_consumes_only_one_slot(failure, reason):
    repo, clock, client = MemoryNewsRepository(), [NOW], Client(failure=failure)
    instance = service(repo, client, clock)
    assert instance.get_context().reason == reason
    assert instance.get_context().status == "UNAVAILABLE"
    assert len(client.calls) == 1


def test_missing_key_and_disabled_are_no_op():
    for key, enabled in (("", True), ("fake-for-mocked-client", False)):
        repo, clock, client = MemoryNewsRepository(), [NOW], Client()
        result = service(repo, client, clock, key=key, enabled=enabled).get_context()
        assert result.status == "UNAVAILABLE" and result.reason == "NOT_CONFIGURED"
        assert not client.calls and repo.closed == 0


def test_deadline_too_close_does_not_claim_slot_or_fetch():
    repo, clock, client = MemoryNewsRepository(), [NOW], Client()
    result = service(repo, client, clock).get_context(deadline_at=NOW + timedelta(seconds=2))
    assert result.status == "UNAVAILABLE" and not repo.slots and not client.calls


def test_public_context_never_contains_key_or_article_content():
    context = NewsContext(status="AVAILABLE", snapshot_id=uuid4(), fetched_at=NOW,
                          latest_news_at=NOW, news_count=1, btc_sentiment_score=.2,
                          average_relevance=.8, source_count=1)
    payload = context.model_dump_json()
    assert "api_key" not in payload and "raw" not in payload and "example.org" not in payload


def test_private_migration_and_unchanged_product_artifact():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "supabase/migrations/20260924093000_add_prescriptive_news_context.sql").read_text()
    assert "enable row level security" in migration
    assert "from public, anon, authenticated" in migration
    assert "to service_role" in migration
    assert "api_key" not in migration.lower()
    import hashlib
    digest = hashlib.sha256((root / "models/xgboost_v1.joblib").read_bytes()).hexdigest()
    assert digest == "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"


def test_technical_only_audit_stays_compatible_before_optional_migration():
    row = RecommendationRequestRecord(
        request_id=uuid4(), prediction_id=None, symbol="BTCUSDT", interval="1m",
        provider="nvidia", llm_model="mock", prompt_version="recommendation_v4",
        requested_at=NOW, finished_at=NOW, total_latency_ms=1, status="NO_ACTIVE_PREDICTION",
    )
    payload = SupabaseRecommendationRequestRepository._row(row)
    assert not any(name.startswith("news_context_") for name in payload)
    with_news = row.model_copy(update={"news_context_fetched_at": NOW})
    assert "news_context_fetched_at" in SupabaseRecommendationRequestRepository._row(with_news)


def test_article_upsert_uses_do_nothing_to_preserve_first_observed_at():
    class Table:
        def __init__(self):
            self.options = []

        def upsert(self, rows, **options):
            self.options.append(options)
            return self

        def execute(self):
            return None

    table = Table()

    class Client:
        def table(self, name):
            assert name == "prescriptive_news_articles"
            return table

    repository = SupabasePrescriptiveNewsRepository(Client())
    repository.save_articles([{"article_identifier": "a" * 64,
                               "first_observed_at": NOW.isoformat()}])
    assert table.options == [{"on_conflict": "article_identifier",
                              "ignore_duplicates": True, "returning": "minimal"}]


def test_refresh_button_only_clears_dashboard_caches(monkeypatch):
    from contextlib import nullcontext
    from streamlit_app import ui

    calls = []
    monkeypatch.setattr(ui.st, "container", lambda **kwargs: nullcontext())
    monkeypatch.setattr(ui.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(ui.st, "rerun", lambda: calls.append("rerun"))
    for name in ("_load_dashboard", "_load_latest_prediction", "_load_live_metrics"):
        monkeypatch.setattr(getattr(ui, name), "clear", lambda name=name: calls.append(name))
    ui.refresh_button()
    assert calls == ["_load_dashboard", "_load_latest_prediction", "_load_live_metrics", "rerun"]


def test_every_dashboard_page_uses_shared_refresh_helper_without_news_client():
    root = Path(__file__).resolve().parents[1]
    pages = [root / "streamlit_app/Home.py", *sorted(p for p in (root / "streamlit_app/pages").glob("*.py") if p.stem != "__init__")]
    assert len(pages) == 6
    for page in pages:
        source = page.read_text(encoding="utf-8")
        assert "refresh_button()" in source
        assert "AlphaVantageNewsClient" not in source
        assert "ALPHA_VANTAGE_API_KEY" not in source
    refresh_source = (root / "streamlit_app/ui.py").read_text(encoding="utf-8")
    assert ".st-key-refresh_control" in refresh_source
    assert ":hover" in refresh_source and ":focus-visible" in refresh_source
    assert ":disabled" in refresh_source
