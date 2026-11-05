"""Private Supabase cache for on-demand BTC news metadata and refresh slots."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from alpha_engine.config.settings import Settings
from alpha_engine.repositories.supabase import _client_from_settings, _iso, _utc
from alpha_engine.news.marketaux import NewsArticle


@dataclass(frozen=True)
class NewsFetchRecord:
    snapshot_id: UUID
    attempted_at: datetime
    fetched_at: datetime
    status: str


@dataclass(frozen=True)
class MarketauxSnapshot:
    snapshot_id: UUID
    attempted_at: datetime
    fetched_at: datetime
    articles: list[NewsArticle]


@dataclass(frozen=True)
class MarketauxClaim:
    outcome: str
    snapshot_id: UUID | None = None
    attempted_at: datetime | None = None


def marketaux_snapshot(row: dict) -> MarketauxSnapshot:
    payload = row["marketaux_payload"]
    if not isinstance(payload, list) or len(payload) > 3:
        raise ValueError("INVALID_SNAPSHOT")
    return MarketauxSnapshot(UUID(str(row["snapshot_id"])), _utc(row["attempted_at"]), _utc(row["fetched_at"]),
                             [NewsArticle.model_validate(a) for a in payload])


def marketaux_claim(row: dict) -> MarketauxClaim:
    outcome = row["outcome"]
    if outcome not in {"CLAIMED", "DAILY_BUDGET_EXHAUSTED", "RATE_LIMIT", "REFRESH_IN_PROGRESS"}:
        raise ValueError("INVALID_CLAIM")
    return MarketauxClaim(outcome, UUID(str(row["snapshot_id"])) if row.get("snapshot_id") else None,
                          _utc(row["attempted_at"]) if row.get("attempted_at") else None)


class SupabasePrescriptiveNewsRepository:
    def __init__(self, client, *, owns_client: bool = False):
        self._client = client
        self._owns_client = owns_client

    @classmethod
    def from_settings(cls, settings: Settings) -> "SupabasePrescriptiveNewsRepository":
        return cls(_client_from_settings(settings), owns_client=True)

    def close(self) -> None:
        if self._owns_client:
            self._client.postgrest.aclose()
            self._owns_client = False

    def latest_successful(self) -> NewsFetchRecord | None:
        response = (self._client.table("prescriptive_news_fetches")
                    .select("snapshot_id,attempted_at,fetched_at,status")
                    .eq("provider", "ALPHA_VANTAGE")
                    .in_("status", ["SUCCESS", "EMPTY"])
                    .order("fetched_at", desc=True).limit(1).execute())
        if not response.data:
            return None
        row = response.data[0]
        return NewsFetchRecord(
            snapshot_id=UUID(row["snapshot_id"]), attempted_at=_utc(row["attempted_at"]),
            fetched_at=_utc(row["fetched_at"]), status=row["status"],
        )

    def claim_slot(self, slot_start: datetime, attempted_at: datetime, snapshot_id: UUID) -> bool:
        response = self._client.table("prescriptive_news_fetches").upsert({
            "snapshot_id": str(snapshot_id), "slot_start": _iso(slot_start),
            "attempted_at": _iso(attempted_at), "status": "CLAIMED", "article_count": 0,
        }, on_conflict="slot_start", ignore_duplicates=True,
            returning="representation").execute()
        return bool(response.data)

    def finish(self, snapshot_id: UUID, status: str, fetched_at: datetime | None,
               article_count: int) -> None:
        (self._client.table("prescriptive_news_fetches")
         .update({"status": status,
                  "fetched_at": _iso(fetched_at) if fetched_at else None,
                  "article_count": article_count})
         .eq("snapshot_id", str(snapshot_id)).execute())

    def save_articles(self, articles: list[dict]) -> None:
        # DO NOTHING preserves first_observed_at when a later fetch sees an article again.
        for start in range(0, len(articles), 500):
            self._client.table("prescriptive_news_articles").upsert(
                articles[start:start + 500], on_conflict="article_identifier",
                ignore_duplicates=True, returning="minimal",
            ).execute()

    def articles_at(self, fetched_at: datetime) -> list[dict]:
        """Only articles already first seen at this snapshot's fetch time."""
        rows: list[dict] = []
        offset = 0
        while True:
            page = (self._client.table("prescriptive_news_articles")
                    .select("published_at,first_observed_at,btc_sentiment_score,btc_relevance_score,source")
                    .eq("provider", "ALPHA_VANTAGE")
                    .gt("published_at", _iso(fetched_at - timedelta(minutes=60)))
                    .lte("published_at", _iso(fetched_at))
                    .lte("first_observed_at", _iso(fetched_at))
                    .order("published_at", desc=True)
                    .range(offset, offset + 499).execute())
            rows.extend(page.data or [])
            if len(page.data or []) < 500:
                break
            offset += 500
        return rows

    def latest_marketaux_snapshot(self) -> MarketauxSnapshot | None:
        response = (self._client.table("prescriptive_news_fetches")
                    .select("snapshot_id,attempted_at,fetched_at,marketaux_payload")
                    .eq("provider", "MARKETAUX").in_("status", ["SUCCESS", "EMPTY"])
                    .order("fetched_at", desc=True).limit(1).execute())
        return marketaux_snapshot(response.data[0]) if response.data else None

    def claim_marketaux(self, snapshot_id: UUID, daily_budget: int) -> MarketauxClaim:
        response = self._client.rpc("claim_marketaux_news", {
            "p_snapshot_id":str(snapshot_id), "p_daily_budget":daily_budget}).execute()
        if not response.data or len(response.data) != 1:
            raise ValueError("INVALID_CLAIM")
        return marketaux_claim(response.data[0])

    def finish_marketaux(self, snapshot_id: UUID, articles: list[NewsArticle] | None,
                         fetched_at: datetime | None) -> None:
        payload = [a.model_dump(mode="json") for a in articles] if articles is not None else None
        response = (self._client.table("prescriptive_news_fetches").update({
            "status":"FAILED" if articles is None else "SUCCESS" if articles else "EMPTY",
            "fetched_at":_iso(fetched_at) if fetched_at else None,
            "marketaux_payload":payload, "article_count":len(articles or [])})
         .eq("provider", "MARKETAUX").eq("snapshot_id", str(snapshot_id)).eq("status", "CLAIMED").execute())
        if not response.data:
            raise ValueError("SNAPSHOT_NOT_FINISHED")
