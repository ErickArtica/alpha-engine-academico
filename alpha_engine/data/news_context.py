"""Sanitized, optional news snapshot for one prescriptive request."""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field
from alpha_engine.news.marketaux import NewsArticle


class NewsContext(BaseModel):
    provider: Literal["ALPHA_VANTAGE", "MARKETAUX"] = "ALPHA_VANTAGE"
    role: Literal["CONTEXT_ONLY"] = "CONTEXT_ONLY"
    status: Literal["AVAILABLE", "NO_NEWS", "STALE", "UNAVAILABLE"]
    snapshot_id: UUID | None = None
    window_minutes: int = 60
    fetched_at: datetime | None = None
    latest_news_at: datetime | None = None
    news_count: int = Field(default=0, ge=0)
    btc_sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    average_relevance: float | None = Field(default=None, gt=0, le=1)
    age_minutes: float | None = Field(default=None, ge=0)
    source_count: int = Field(default=0, ge=0)
    scored_news_count: int = Field(default=0, ge=0)
    aggregation_method: Literal["SIMPLE_MEAN"] | None = None
    recent_articles: list[NewsArticle] = Field(default_factory=list, max_length=3)
    historical_articles: list[NewsArticle] = Field(default_factory=list, max_length=3)
    reason: Literal["NOT_CONFIGURED", "STORAGE_UNAVAILABLE", "PROVIDER_UNAVAILABLE",
                    "RATE_LIMIT", "INVALID_RESPONSE", "TOO_OLD", "NO_RECENT_NEWS",
                    "NO_VALID_SENTIMENT", "NO_NEWS_IN_WINDOW", "CACHE_EXPIRED",
                    "DAILY_BUDGET_EXHAUSTED", "REFRESH_IN_PROGRESS", "AUTHENTICATION_FAILED",
                    "PLAN_RESTRICTION"] | None = None

    def at_time(self, now: datetime) -> "NewsContext":
        """Pure read-time freshness calculation. Never fetches or changes a decision."""
        if now.tzinfo is None:
            raise ValueError("UTC_REQUIRED")
        if self.provider != "MARKETAUX":
            return self
        if self.fetched_at is None:
            if self.status == "AVAILABLE" or self.btc_sentiment_score is not None or self.recent_articles or self.historical_articles:
                return self.model_copy(update={"status":"UNAVAILABLE", "reason":"INVALID_RESPONSE",
                    "btc_sentiment_score":None, "scored_news_count":0, "aggregation_method":None,
                    "recent_articles":[], "historical_articles":[]})
            return self
        now = now.astimezone(timezone.utc)
        if self.fetched_at > now:
            return self.model_copy(update={"status":"UNAVAILABLE", "reason":"INVALID_RESPONSE",
                                           "btc_sentiment_score":None, "scored_news_count":0,
                                           "recent_articles":[], "historical_articles":[]})
        rows = [a for a in [*self.recent_articles, *self.historical_articles]
                if now - timedelta(hours=24) < a.published_at <= now]
        recent = [a for a in rows if a.published_at > now - timedelta(minutes=60)]
        old = [a for a in rows if a.published_at <= now - timedelta(minutes=60)]
        scores = [a.sentiment_score for a in recent if a.sentiment_status == "VALID" and a.sentiment_score is not None]
        age = (now - self.fetched_at).total_seconds()/60
        if age >= 30:
            status, reason, scores = "STALE", "CACHE_EXPIRED", []
        elif scores:
            status, reason = "AVAILABLE", None
        elif recent:
            status, reason = "UNAVAILABLE", "NO_VALID_SENTIMENT"
        elif old:
            status, reason = "STALE", "NO_RECENT_NEWS"
        else:
            status, reason = "NO_NEWS", "NO_NEWS_IN_WINDOW"
        return self.model_copy(update={"status":status, "reason":reason, "age_minutes":age,
            "recent_articles":recent, "historical_articles":old, "news_count":len(recent),
            "latest_news_at":max((a.published_at for a in recent), default=None),
            "source_count":len({a.source for a in recent if a.source}), "scored_news_count":len(scores),
            "btc_sentiment_score":sum(scores)/len(scores) if scores else None,
            "average_relevance":None, "aggregation_method":"SIMPLE_MEAN" if scores else None})

    @property
    def available(self) -> bool:
        return self.status in {"AVAILABLE", "NO_NEWS"}

    def for_prompt(self) -> dict:
        """One compact snapshot; no provider raw response or article content."""
        result = {
            "role": self.role,
            "provider": self.provider,
            "status": self.status,
            "snapshot_id": str(self.snapshot_id) if self.snapshot_id else None,
            "window_minutes": self.window_minutes,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "latest_news_at": self.latest_news_at.isoformat() if self.latest_news_at else None,
            "news_count": self.news_count,
            "btc_sentiment_score": self.btc_sentiment_score,
            "average_relevance": self.average_relevance,
            "age_minutes": self.age_minutes,
            "source_count": self.source_count,
            "limitation": (
                "News sentiment is contextual information only; it did not show robust "
                "incremental predictive value in the historical experiment. Absence of "
                "recent news is not confirmed neutral sentiment."
            ),
        }
        if self.provider == "MARKETAUX":
            result.update(aggregation_method=self.aggregation_method, scored_news_count=self.scored_news_count,
                          historical_window_minutes=1440,
                          historical_context_role="HISTORICAL_INFORMATION_ONLY_NOT_CURRENT_SENTIMENT",
                          historical_news=[{"title":a.title,"source":a.source,"published_at":a.published_at.isoformat()}
                                           for a in self.historical_articles],
                          recent_news=[{"title":a.title,"source":a.source,"published_at":a.published_at.isoformat()}
                                       for a in self.recent_articles],
                          sample_limit=3)
        return result
