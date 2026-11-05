"""Optional, persistent and rate-limited BTC news context for explicit requests."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from math import floor
from uuid import UUID, uuid4

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.data.news_context import NewsContext
from alpha_engine.news.alpha_vantage import (
    AlphaVantageRateLimit, parse_article, validate_payload,
)
from alpha_engine.repositories.supabase_news import NewsFetchRecord, SupabasePrescriptiveNewsRepository


NEWS_WINDOW = timedelta(minutes=60)
CACHE_TTL = timedelta(hours=2)
SLOT_SECONDS = 2 * 60 * 60
ENDPOINT = "https://www.alphavantage.co/query"


def slot_start(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("News request time must include timezone")
    stamp = floor(now.astimezone(timezone.utc).timestamp() / SLOT_SECONDS) * SLOT_SECONDS
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def _unavailable(reason: str) -> NewsContext:
    return NewsContext(status="UNAVAILABLE", reason=reason)


class AlphaVantageNewsClient:
    """Small HTTPS client; never logs request URLs because they carry the key."""

    def __init__(self, key: str):
        self._key = key

    def fetch(self, now: datetime, *, timeout_seconds: float) -> list[dict]:
        # Alpha Vantage requires the key in the query string. Suppress HTTP
        # client request logging so that URL is never emitted by this process.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        utc_now = now.astimezone(timezone.utc)
        params = {
            "function": "NEWS_SENTIMENT", "tickers": "CRYPTO:BTC",
            "time_from": (utc_now - NEWS_WINDOW).strftime("%Y%m%dT%H%M"),
            "time_to": utc_now.strftime("%Y%m%dT%H%M"),
            "sort": "LATEST", "limit": 1000, "apikey": self._key,
        }
        with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
            response = client.get(ENDPOINT, params=params)
            response.raise_for_status()
            feed = validate_payload(response.json())
        if len(feed) >= 1000:
            raise ValueError("News window may be truncated")
        return feed


class PrescriptiveNewsContextService:
    def __init__(
        self, settings: Settings, *,
        repository_factory: Callable[[Settings], object] | None = None,
        client: AlphaVantageNewsClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._settings = settings
        if repository_factory is None:
            if settings.storage_provider == "postgresql":
                from alpha_engine.repositories.postgresql import PostgresPrescriptiveNewsRepository
                repository_factory = PostgresPrescriptiveNewsRepository.from_settings
            else:
                repository_factory = SupabasePrescriptiveNewsRepository.from_settings
        self._repository_factory = repository_factory
        self._client = client
        self._clock = clock
        self._marketaux_service = None
        if settings.prescriptive_news_provider == "marketaux":
            from alpha_engine.services.marketaux_context import MarketauxContextService
            self._marketaux_service = MarketauxContextService(settings, repository_factory, client, clock)

    def _from_snapshot(self, repository, snapshot: NewsFetchRecord, now: datetime) -> NewsContext:
        age = max(0.0, (now - snapshot.fetched_at).total_seconds() / 60)
        if age > 60:
            return NewsContext(
                status="STALE", reason="TOO_OLD", snapshot_id=snapshot.snapshot_id,
                fetched_at=snapshot.fetched_at, age_minutes=age,
            )
        rows = repository.articles_at(snapshot.fetched_at)
        if not rows:
            return NewsContext(
                status="NO_NEWS", snapshot_id=snapshot.snapshot_id,
                fetched_at=snapshot.fetched_at, age_minutes=age,
            )
        scores = [float(row["btc_sentiment_score"]) for row in rows]
        weights = [float(row["btc_relevance_score"]) for row in rows]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("Invalid stored news relevance")
        published = [datetime.fromisoformat(row["published_at"].replace("Z", "+00:00"))
                     if isinstance(row["published_at"], str) else row["published_at"] for row in rows]
        return NewsContext(
            status="AVAILABLE", snapshot_id=snapshot.snapshot_id,
            fetched_at=snapshot.fetched_at, latest_news_at=max(published),
            news_count=len(rows), btc_sentiment_score=sum(s * w for s, w in zip(scores, weights)) / weight_sum,
            average_relevance=weight_sum / len(rows), age_minutes=age,
            source_count=len({row["source"] for row in rows if row["source"]}),
        )

    def get_context(self, *, deadline_at: datetime | None = None) -> NewsContext:
        """Called only after an explicit recommendation click and pre-LLM guards."""
        if self._marketaux_service is not None:
            return self._marketaux_service.get_context(deadline_at=deadline_at)
        if (not self._settings.prescriptive_news_enabled
                or not self._settings.alpha_vantage_api_key.get_secret_value()
                or self._settings.storage_provider not in {"supabase", "postgresql"}):
            return _unavailable("NOT_CONFIGURED")
        now = self._clock().astimezone(timezone.utc)
        if deadline_at is not None and deadline_at.tzinfo is None:
            raise ValueError("News deadline must include timezone")
        try:
            repository = self._repository_factory(self._settings)
            try:
                latest = repository.latest_successful()
                if latest and now - latest.fetched_at < CACHE_TTL:
                    return self._from_snapshot(repository, latest, now)
                seconds_left = ((deadline_at - now).total_seconds() - 2) if deadline_at else 30.0
                timeout = min(self._settings.alpha_vantage_timeout_seconds, seconds_left)
                if timeout < 1:
                    return _unavailable("PROVIDER_UNAVAILABLE")
                identifier = uuid4()
                if not repository.claim_slot(slot_start(now), now, identifier):
                    return self._from_snapshot(repository, latest, now) if latest else _unavailable("RATE_LIMIT")
                # Keep the short H1 deadline available for the LLM. Failed calls
                # consume the persistent slot and are not retried by rapid clicks.
                try:
                    client = self._client or AlphaVantageNewsClient(
                        self._settings.alpha_vantage_api_key.get_secret_value()
                    )
                    feed = client.fetch(now, timeout_seconds=timeout)
                    fetched_at = self._clock().astimezone(timezone.utc)
                    articles = {}
                    for item in feed:
                        parsed = parse_article(item)
                        published = parsed["published_at_utc"]
                        if not now - NEWS_WINDOW < published <= fetched_at:
                            continue
                        identifier_hash = hashlib.sha256(parsed["key"].encode("utf-8")).hexdigest()
                        articles[identifier_hash] = {
                            "article_identifier": identifier_hash, "provider": "ALPHA_VANTAGE",
                            "published_at": published.isoformat(), "fetched_at": fetched_at.isoformat(),
                            "first_observed_at": fetched_at.isoformat(),
                            "btc_sentiment_score": parsed["score"],
                            "btc_relevance_score": parsed["relevance"],
                            "source": parsed["source"][:120],
                        }
                    repository.save_articles(list(articles.values()))
                    repository.finish(identifier, "SUCCESS" if articles else "EMPTY", fetched_at, len(articles))
                    snapshot = NewsFetchRecord(identifier, now, fetched_at,
                                               "SUCCESS" if articles else "EMPTY")
                    return self._from_snapshot(repository, snapshot, fetched_at)
                except AlphaVantageRateLimit:
                    repository.finish(identifier, "RATE_LIMIT", None, 0)
                    return _unavailable("RATE_LIMIT")
                except (httpx.TimeoutException, httpx.HTTPError):
                    repository.finish(identifier, "FAILED", None, 0)
                    return _unavailable("PROVIDER_UNAVAILABLE")
                except (ValueError, TypeError, KeyError):
                    repository.finish(identifier, "FAILED", None, 0)
                    return _unavailable("INVALID_RESPONSE")
            finally:
                repository.close()
        except Exception:
            # A missing migration, RLS problem or provider exception must not
            # make the technical recommendation path unusable or expose details.
            return _unavailable("STORAGE_UNAVAILABLE")
