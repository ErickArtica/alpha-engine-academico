"""Pure NEWS_SENTIMENT parser reused by the frozen research workflow."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


class AlphaVantageRateLimit(RuntimeError):
    pass


def parse_time(value: str) -> datetime:
    if not isinstance(value, str) or len(value) != 15:
        raise ValueError("Invalid Alpha Vantage publication timestamp")
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def article_key(item: dict) -> str:
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        bits = urlsplit(url.strip())
        if bits.scheme not in {"http", "https"} or not bits.netloc:
            raise ValueError("Article URL is invalid")
        return urlunsplit((bits.scheme.lower(), bits.netloc.lower(), bits.path.rstrip("/"), "", ""))
    raise ValueError("Article missing stable URL")


def parse_article(item: dict) -> dict:
    published = parse_time(item.get("time_published"))
    ticker_items = item.get("ticker_sentiment")
    if not isinstance(ticker_items, list):
        raise ValueError("Missing ticker sentiment list")
    btc = [entry for entry in ticker_items if isinstance(entry, dict) and entry.get("ticker") == "CRYPTO:BTC"]
    if len(btc) != 1:
        raise ValueError("Missing or ambiguous BTC-specific sentiment")
    score = float(btc[0]["ticker_sentiment_score"])
    relevance = float(btc[0]["relevance_score"])
    if not math.isfinite(score) or not -1 <= score <= 1 or not math.isfinite(relevance) or not 0 < relevance <= 1:
        raise ValueError("Invalid BTC sentiment/relevance")
    return {"key": article_key(item), "published_at_utc": published,
            "score": score, "relevance": relevance,
            "source": str(item.get("source", "")),
            "label": str(btc[0].get("ticker_sentiment_label", ""))}


def validate_payload(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("Malformed JSON object")
    if any(key in payload for key in ("Note", "Information")):
        raise AlphaVantageRateLimit("ALPHA_VANTAGE_RATE_OR_PLAN_LIMIT")
    if "Error Message" in payload or "feed" not in payload or not isinstance(payload["feed"], list):
        raise ValueError("ALPHA_VANTAGE_API_ERROR_OR_SCHEMA")
    if any(not isinstance(article, dict) for article in payload["feed"]):
        raise ValueError("Malformed feed article")
    return payload["feed"]
