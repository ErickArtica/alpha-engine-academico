"""Optional Marketaux context, shared durable reservation, no product decisions."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from alpha_engine.data.news_context import NewsContext
from alpha_engine.news.marketaux import MarketauxNewsClient, MarketauxUnavailable, utc
from alpha_engine.repositories.supabase_news import MarketauxSnapshot

CACHE_TTL = timedelta(minutes=30)


def unavailable(reason: str) -> NewsContext:
    return NewsContext(provider="MARKETAUX", status="UNAVAILABLE", reason=reason)


def snapshot_context(snapshot: MarketauxSnapshot, now: datetime) -> NewsContext:
    return NewsContext(provider="MARKETAUX", status="UNAVAILABLE", snapshot_id=snapshot.snapshot_id,
                       fetched_at=snapshot.fetched_at, recent_articles=snapshot.articles).at_time(now)


class MarketauxContextService:
    def __init__(self, settings, repository_factory, client=None, clock=lambda:datetime.now(timezone.utc)):
        self.settings, self.repository_factory, self.client, self.clock = settings, repository_factory, client, clock

    def get_context(self, *, deadline_at=None) -> NewsContext:
        s = self.settings
        if (not s.prescriptive_news_enabled or not s.marketaux_api_token.get_secret_value()
                or s.storage_provider not in {"supabase", "postgresql"}):
            return unavailable("NOT_CONFIGURED")
        repository = None
        try:
            now = utc(self.clock())
            repository = self.repository_factory(s)
            latest = repository.latest_marketaux_snapshot()
            if latest and timedelta(0) <= now - latest.fetched_at < CACHE_TTL:
                return snapshot_context(latest, now)
            timeout = min(s.marketaux_timeout_seconds, (utc(deadline_at)-now).total_seconds()-2 if deadline_at else 30.)
            if timeout < 1:
                return unavailable("PROVIDER_UNAVAILABLE")
            # RPC commits budget and cooldown before ANY provider I/O. Unknown
            # reservation outcomes never cause a provider call or a second claim.
            claim = repository.claim_marketaux(uuid4(), s.marketaux_daily_budget)
            if claim.outcome != "CLAIMED":
                context = snapshot_context(latest, now) if latest else unavailable(claim.outcome)
                return context.model_copy(update={"reason":claim.outcome})
            if claim.snapshot_id is None or claim.attempted_at is None:
                return unavailable("STORAGE_UNAVAILABLE")
            try:
                client = self.client or MarketauxNewsClient(s.marketaux_api_token.get_secret_value())
                articles = client.fetch(min(now, claim.attempted_at), timeout_seconds=timeout)
                fetched = max(utc(self.clock()), claim.attempted_at)
                repository.finish_marketaux(claim.snapshot_id, articles, fetched)
                return snapshot_context(MarketauxSnapshot(claim.snapshot_id, claim.attempted_at, fetched, articles), fetched)
            except MarketauxUnavailable as error:
                repository.finish_marketaux(claim.snapshot_id, None, None)
                context = snapshot_context(latest, now) if latest else unavailable(error.reason)
                return context.model_copy(update={"reason":error.reason})
            except Exception:
                repository.finish_marketaux(claim.snapshot_id, None, None)
                return unavailable("PROVIDER_UNAVAILABLE")
        except Exception:
            return unavailable("STORAGE_UNAVAILABLE")
        finally:
            if repository is not None:
                try:
                    repository.close()
                except Exception:
                    pass
