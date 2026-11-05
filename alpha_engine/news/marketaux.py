"""One bounded Marketaux request and pure metadata normalization. No storage/LLM."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from html import unescape
import ipaddress
import json
import logging
import math
import re
from time import perf_counter
from urllib.parse import parse_qsl, unquote_plus, urlencode, urlsplit, urlunsplit
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal

ENDPOINT = "https://api.marketaux.com/v1/news/all"
MAX_BYTES = 256 * 1024


class MarketauxUnavailable(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def clean_text(value: object, token: str = "", limit: int = 400) -> str | None:
    if not isinstance(value, str):
        return None
    value = unescape(value)
    probe = value
    for _ in range(4):
        if ((token and token.casefold() in probe.casefold())
                or re.search(r"(?:api[_-]?token|api[_-]?key|access[_-]?token|authorization)\s*[=:]", probe, re.I)):
            return None
        decoded = unquote_plus(probe)
        if decoded == probe:
            break
        probe = decoded
    value = re.sub(r"<[^>]*>", " ", value)
    return " ".join("".join(c for c in value if c.isprintable() or c.isspace()).split())[:limit] or None


def public_article_url(value: object, token: str = "") -> str | None:
    if not isinstance(value, str) or len(value) > 4096 or clean_text(value, token, 4096) is None:
        return None
    if any(c.isspace() or ord(c) < 32 for c in value) or "\\" in value:
        return None
    try:
        p = urlsplit(value)
        host = p.hostname
        if (p.scheme not in {"https", "http"} or not host or p.username is not None or p.password is not None
                or p.port not in {None, 80, 443} or host == "api.marketaux.com"):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            host = host.encode("idna").decode("ascii").lower()
            if ("." not in host or re.fullmatch(r"[0-9.]+", host) or host.endswith((".local", ".internal", ".localhost"))
                    or not re.fullmatch(r"[a-z0-9.-]+", host)
                    or any(not x or x.startswith("-") or x.endswith("-") for x in host.split("."))):
                return None
        pairs = parse_qsl(p.query, keep_blank_values=True, max_num_fields=100)
        if any(k.lower() in {"api_token", "apikey", "api_key", "token", "access_token", "key", "secret", "password", "authorization", "signature"} for k, _ in pairs):
            return None
        pairs = [(k, v) for k, v in pairs if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid"}]
        netloc = f"[{host}]" if ":" in host else host
        if p.port and p.port != (443 if p.scheme == "https" else 80):
            netloc += f":{p.port}"
        return urlunsplit((p.scheme, netloc, p.path or "/", urlencode(sorted(pairs)), ""))
    except (ValueError, UnicodeError):
        return None


def utc(value: str | datetime) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(instant, datetime) or instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("UTC_REQUIRED")
    return instant.astimezone(timezone.utc)


class NewsArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_identifier: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_uuid: UUID | None = None
    title: str = Field(min_length=1, max_length=400)
    source: str | None = Field(default=None, max_length=160)
    public_url: str
    published_at: datetime
    bitcoin_symbol: Literal["CC:BTC", "BTCUSD"] = "CC:BTC"
    bitcoin_name: Literal["Bitcoin"] = "Bitcoin"
    match_score: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    sentiment_score: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False, strict=True)
    sentiment_status: Literal["VALID", "MISSING", "INVALID", "AMBIGUOUS"]

    @field_validator("public_url")
    @classmethod
    def validate_url(cls, value):
        result = public_article_url(value)
        if result is None:
            raise ValueError("INVALID_PUBLIC_URL")
        return result

    @field_validator("published_at")
    @classmethod
    def validate_utc(cls, value):
        return utc(value)


def normalize_marketaux_article(item: object, cutoff: datetime, token: str = "", *,
                               min_match_score: float = 0) -> NewsArticle | None:
    if not isinstance(item, dict):
        return None
    title, url = clean_text(item.get("title"), token), public_article_url(item.get("url"), token)
    try:
        published = utc(item.get("published_at"))
    except (ValueError, TypeError, AttributeError):
        return None
    if not title or not url or not cutoff - timedelta(hours=24) < published <= cutoff:
        return None
    entities = item.get("entities")
    if not isinstance(entities, list):
        return None
    def is_bitcoin(entity):
        if not isinstance(entity, dict):
            return False
        name = (clean_text(entity.get("name"), token, 120) or "").casefold()
        legacy = entity.get("symbol") == "CC:BTC" and name == "bitcoin"
        documented = (entity.get("symbol") == "BTCUSD" and entity.get("exchange") == "CC"
                      and entity.get("type") == "cryptocurrency" and name in {"bitcoin", "bitcoin usd"})
        if not (legacy or documented):
            return False
        match = entity.get("match_score")
        return min_match_score == 0 or (type(match) in {int, float}
            and math.isfinite(match) and min_match_score <= match <= 100)
    bitcoin = [e for e in entities if is_bitcoin(e)]
    if not bitcoin:
        return None
    values = []
    invalid = False
    for entity in bitcoin:
        value = entity.get("sentiment_score")
        if value is None:
            continue
        try:
            valid = type(value) in {int, float} and math.isfinite(value) and -1 <= value <= 1
        except OverflowError:
            valid = False
        if valid:
            values.append(float(value))
        else:
            invalid = True
    status = "AMBIGUOUS" if len(set(values)) > 1 else "INVALID" if invalid else "VALID" if values else "MISSING"
    identifier = None
    try:
        raw = item.get("uuid")
        if isinstance(raw, str) and clean_text(raw, token):
            identifier = UUID(raw)
    except ValueError:
        pass
    key = "uuid:" + str(identifier) if identifier else "url:" + url
    return NewsArticle(article_identifier=hashlib.sha256(("MARKETAUX:" + key).encode()).hexdigest(),
                       provider_uuid=identifier, title=title, source=clean_text(item.get("source"), token, 160),
                       public_url=url, published_at=published,
                       bitcoin_symbol="BTCUSD" if any(e.get("symbol") == "BTCUSD" for e in bitcoin) else "CC:BTC",
                       match_score=max((e["match_score"] for e in bitcoin
                                        if type(e.get("match_score")) in {int,float}
                                        and math.isfinite(e["match_score"]) and 0 <= e["match_score"] <= 100), default=None),
                       sentiment_score=values[0] if status == "VALID" else None, sentiment_status=status)


class _DiscardHttpLogs(logging.Filter):
    def filter(self, record):
        return not (record.name == "httpx" or record.name.startswith(("httpx.", "httpcore")))


@contextmanager
def suppress_http_details():
    # Handler filters also cover child loggers configured at DEBUG. Distinct
    # filter instances can coexist safely during concurrent requests.
    log_filter = _DiscardHttpLogs()
    handlers = set(logging.getLogger().handlers)
    for logger in list(logging.root.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            handlers.update(logger.handlers)
    if logging.lastResort:
        handlers.add(logging.lastResort)
    for handler in handlers:
        handler.addFilter(log_filter)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(log_filter)


class MarketauxNewsClient:
    def __init__(self, token: str, *, transport: httpx.BaseTransport | None = None,
                 min_match_score: float = 0, allow_empty_filtered: bool = False):
        self._token, self._transport = token, transport
        self.min_match_score, self.allow_empty_filtered = min_match_score, allow_empty_filtered
        self.diagnostics = {}

    def fetch(self, now: datetime, *, timeout_seconds: float) -> list[NewsArticle]:
        if not self._token:
            raise MarketauxUnavailable("NOT_CONFIGURED")
        cutoff = utc(now)
        params = {"api_token": self._token, "symbols": "BTCUSD", "search": "Bitcoin", "limit": 3, "page": 1,
                  "filter_entities": "true", "must_have_entities": "true", "group_similar": "true",
                  "published_after": (cutoff - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
                  "published_before": cutoff.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            started = perf_counter()
            with suppress_http_details():
                transport = self._transport if self._transport is not None else httpx.HTTPTransport(retries=0, trust_env=False)
                with httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                                  timeout=httpx.Timeout(timeout_seconds, connect=min(3., timeout_seconds))) as client:
                    with client.stream("GET", ENDPOINT, params=params) as response:
                        self.diagnostics["http_status"] = response.status_code
                        if response.status_code != 200:
                            reason = {401:"AUTHENTICATION_FAILED", 402:"RATE_LIMIT", 403:"PLAN_RESTRICTION", 429:"RATE_LIMIT"}.get(response.status_code, "PROVIDER_UNAVAILABLE")
                            raise MarketauxUnavailable(reason)
                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            if perf_counter() - started > timeout_seconds:
                                raise MarketauxUnavailable("PROVIDER_UNAVAILABLE")
                            raw.extend(chunk)
                            if len(raw) > MAX_BYTES:
                                raise MarketauxUnavailable("INVALID_RESPONSE")
                        body = json.loads(raw)
            if not isinstance(body, dict) or not isinstance(body.get("data"), list) or "error" in body or len(body["data"]) > 3:
                raise MarketauxUnavailable("INVALID_RESPONSE")
            meta = body.get("meta")
            if (not isinstance(meta, dict) or type(meta.get("returned")) is not int
                    or type(meta.get("found")) is not int or meta["returned"] != len(body["data"])
                    or meta["found"] < meta["returned"]):
                raise MarketauxUnavailable("INVALID_RESPONSE")
            normalized = [normalize_marketaux_article(item, cutoff, self._token,
                          min_match_score=self.min_match_score) for item in body["data"]]
            # Exact snapshot membership: do not merge old cached article rows.
            articles, ids, urls = [], set(), set()
            for article in normalized:
                if article and article.article_identifier not in ids and article.public_url not in urls:
                    articles.append(article)
                    ids.add(article.article_identifier)
                    urls.add(article.public_url)
            self.diagnostics.update(provider_returned=len(body["data"]), normalized_articles=len(articles),
                                    provider_found=meta["found"], entity_symbol="BTCUSD", entity_exchange="CC")
            if body["data"] and not articles and not self.allow_empty_filtered:
                raise MarketauxUnavailable("INVALID_RESPONSE")
            return articles
        except MarketauxUnavailable:
            raise
        except httpx.HTTPError:
            raise MarketauxUnavailable("PROVIDER_UNAVAILABLE") from None
        except (ValueError, TypeError, RecursionError):
            raise MarketauxUnavailable("INVALID_RESPONSE") from None
