"""Read persisted V1 context; explain rules, never create predictions or trades."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import httpx

from pydantic import BaseModel, ConfigDict, Field

from deploy.render.academic_demo import canonical, fingerprint
from deploy.render.academic_nvidia import (
    BoundedNvidiaClient, GenerationPolicy, NvidiaTrialError, compact_context,
)


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: str = Field(pattern="^(BUY|SELL|HOLD|WAIT)$")
    decision_hash: str = Field(pattern="^[0-9a-f]{64}$")
    risk_level: str = Field(pattern="^(LOW|MEDIUM|HIGH)$")
    summary: str = Field(min_length=1, max_length=1200)
    key_factors: list[str] = Field(min_length=1, max_length=6)
    risks: list[str] = Field(min_length=1, max_length=6)
    warning: str = Field(min_length=1, max_length=400)


def persisted_snapshot(settings, prediction_id):
    # Existing domain services, explicit source timestamp, no get_ml_prediction().
    from alpha_engine.repositories.factory import create_market_repository, create_prediction_repository
    from alpha_engine.services.prediction import PredictionResult, PredictionService
    from alpha_engine.services.recommendation_context import RecommendationContextService
    from alpha_engine.tools import RepositoryAnalyticsTools
    from alpha_engine.risk import RiskService
    market = predictions = None
    try:
        predictions = create_prediction_repository(settings)
        record = predictions.get_by_id(prediction_id)
        if record is None or record.symbol != settings.symbol or record.interval != "1m" or record.model_version != "xgboost_v1":
            raise LookupError("Prediction unavailable")
        market = create_market_repository(settings)
        predictor = PredictionService(market, predictions, settings.model_path,
                                      lookback_candles=settings.prediction_lookback_candles)
        tools = RepositoryAnalyticsTools(market, predictor, RiskService(settings), settings.symbol,
                                         "1m", settings.prediction_lookback_candles)
        return RecommendationContextService(tools).build_snapshot_for_prediction(
            PredictionResult.model_validate(record.model_dump()))
    finally:
        for resource in (market, predictions):
            if resource is not None:
                resource.close()


def rule_context(snapshot, settings, has_long, now):
    from alpha_engine.ai.context_builder import build_context
    from alpha_engine.services.paper_policy import paper_decision
    if settings.model_confidence_threshold != 0.60:
        raise ValueError("Academic V1 gate must remain 0.60")
    decision = paper_decision(snapshot.prediction, has_long=has_long, threshold=0.60, now=now)
    fresh = 0 <= (now - snapshot.market.close_time).total_seconds() <= settings.max_candle_age_seconds
    if decision.executable and (snapshot.risk.risk_level == "HIGH" or not fresh):
        decision = replace(decision, action="HOLD" if has_long else "WAIT", executable=False,
                           reason="Riesgo alto o fuente vencida: abstenerse y reevaluar.")
    context = build_context(snapshot, settings)
    context["decision_policy"].update(action=decision.action,
        position="LONG_HIPOTETICO" if has_long else "SIN_POSICION_HIPOTETICA",
        reason=decision.reason, action_authority="deterministic_rules_only")
    generated, expires = snapshot.prediction.generated_at, snapshot.prediction.valid_until
    usable = generated is not None and expires is not None and generated <= now < expires
    context["academic_demo"] = {"operational": False, "real_trading": False,
        "source_status": "LIVE" if fresh and usable else "HISTORICO",
        "generated_at": snapshot.prediction.generated_at.isoformat() if snapshot.prediction.generated_at else None,
        "valid_until": snapshot.prediction.valid_until.isoformat() if snapshot.prediction.valid_until else None}
    return context


def news_view(cache, now):
    base = dict(provider="MARKETAUX", status="N/A", classification="N/A", score=None,
                considered=0, fetched_at=None, latest_news_at=None, method="SIMPLE_MEAN",
                reason="NO_AUTHORIZED_SNAPSHOT", scope="AUTHORIZED_DEMO_ONLY",
                classification_threshold=.05, max_article_age_seconds=3600, min_match_score=10)
    if not cache:
        return base
    fetched = cache["fetched_at"]
    observations = cache["observations"]
    recent = [item for item in observations if 0 <= now - item["at"] <= 3600]
    # Observations come only from the normalized Bitcoin entities, not general crypto sentiment.
    recent = list({item.get("id", "legacy:"+str(i)):item for i,item in enumerate(recent)}.values())
    scores = [item["score"] for item in recent if item["score"] is not None]
    score = sum(scores) / len(scores) if scores else None
    age = now - fetched
    status = ("STALE" if age >= 1800 else "LIVE") if scores else ("HISTORICO" if observations and not recent else "N/A")
    if age < 0:
        status, score = "N/A", None
    base.update(status=status, classification="N/A" if score is None else "POSITIVO" if score >= .05 else "NEGATIVO" if score <= -.05 else "NEUTRAL",
                score=score, considered=len(scores) if score is not None else 0,
                fetched_at=datetime.fromtimestamp(fetched, timezone.utc).isoformat(),
                latest_news_at=datetime.fromtimestamp(max(i["at"] for i in recent), timezone.utc).isoformat() if recent else None,
                reason="SNAPSHOT_EXPIRED" if status == "STALE" else "VALID_SCORES" if score is not None else "NO_RECENT_VALID_SCORE")
    return base


class AcademicService:
    def __init__(self, settings, store, clock, *, snapshot_loader=persisted_snapshot,
                 provider_factory=None, news_client=None):
        self.settings, self.store, self.clock = settings, store, clock
        self.snapshot_loader = snapshot_loader
        self.provider_factory, self.news_client = provider_factory, news_client
        self.generation_policy = GenerationPolicy.load()

    def _news(self, cfg):
        if not (cfg.news and cfg.news_rights):
            return {**news_view(None, self.clock()), "reason": "RIGHTS_OR_PROVIDER_NOT_APPROVED"}
        _, state = self.store.read()
        return news_view(state.get("news_cache"), self.clock())

    def handle(self, path, payload, claims, cfg, quotas):
        if path in {"news", "refresh-news"}:
            if payload:
                return 400, {"status": "INVALID_REQUEST"}
            if path == "news":
                return 200, {"status": "SNAPSHOT", "news": self._news(cfg)}
            return self.refresh_news(claims, cfg, quotas)
        if (path not in {"context", "explain"} or set(payload) not in (
                {"prediction_id", "has_long"}, {"prediction_id", "has_long", "historical"})
                or not isinstance(payload["prediction_id"], str)
                or re.fullmatch(r"[0-9a-f]{64}", payload["prediction_id"]) is None
                or type(payload["has_long"]) is not bool or type(payload.get("historical", False)) is not bool):
            return 400, {"status": "INVALID_REQUEST"}
        try:
            snapshot = self.snapshot_loader(self.settings, payload["prediction_id"])
        except LookupError:
            return 404, {"status": "PREDICTION_NOT_FOUND"}
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        historical = payload.get("historical", False)
        evaluated_at = snapshot.prediction.generated_at if historical else now
        if historical and (evaluated_at is None or evaluated_at > now
                or snapshot.prediction.valid_until is None
                or not snapshot.market.close_time <= evaluated_at < snapshot.prediction.valid_until):
            return 409, {"status": "NO_VALID_HISTORICAL_TIME"}
        context = rule_context(snapshot, self.settings, payload["has_long"], evaluated_at)
        if historical:
            context["academic_demo"]["source_status"] = "HISTORICO"
            context["academic_demo"]["historical_evaluated_at"] = evaluated_at.isoformat()
            context["news_context"] = {**news_view(None, self.clock()), "reason": "NO_POINT_IN_TIME_SNAPSHOT"}
        else:
            context["news_context"] = self._news(cfg)
        # Never refresh a paid/external source as a side effect of explanation.
        decision_hash = fingerprint(canonical(context))
        result = dict(status="CONTEXT", demo_only=True, operational=False,
                      prediction_id=payload["prediction_id"], decision=context["decision_policy"],
                      prediction=context["prediction"],
                      source=context["academic_demo"], risk=context["risk"],
                      news=context["news_context"], decision_hash=decision_hash)
        if path == "context":
            return 200, result
        s = self.settings
        if not cfg.llm:
            return 403, {"status": "LLM_NOT_AUTHORIZED"}
        if (s.llm_provider != "nvidia" or s.nvidia_model != "meta/muse-glimmer-30b"
                or s.nvidia_base_url.rstrip("/") != "https://integrate.api.nvidia.com/v1"
                or not s.nvidia_api_key.get_secret_value()):
            return 503, {"status": "MUSE_NOT_CONFIGURED"}
        cache_key = fingerprint(claims["sub"] + ":" + decision_hash + ":" + s.nvidia_model + ":academic_v1")
        _, state = self.store.read()
        cached = state.get("explanations", {}).get(cache_key)
        if cached and cached["until"] > self.clock():
            return 200, {**result, "status": "EXPLAINED", "cached": True, "model": s.nvidia_model,
                         "explanation": cached["explanation"]}
        ticket = quotas.reserve("llm", claims["sub"], claims["sid"], lease=True)
        if not ticket:
            return 429, {"status": "BUDGET_OR_CONCURRENCY_LIMIT"}
        provider, bounded_client, status = None, None, "PROVIDER_FAILED"
        try:
            if self.provider_factory:
                provider = self.provider_factory(s)
            else:
                from alpha_engine.ai.providers.nvidia import NvidiaProvider
                bounded_client = BoundedNvidiaClient(policy=self.generation_policy)
                provider = NvidiaProvider(s, Path(__file__).with_name("academic_prompt.txt").read_text(encoding="utf-8"),
                                          client=bounded_client)
            raw = provider.generate_recommendation(compact_context({**context, "decision_hash": decision_hash}),
                                                   timeout_seconds=self.generation_policy.total)
            if not isinstance(raw, str) or len(raw.encode()) > 12000:
                raise ValueError("Invalid explanation size")
            # Reject duplicate JSON keys and NaN before schema/coherence checks.
            def unique(pairs):
                output = {}
                for key, value in pairs:
                    if key in output:
                        raise ValueError("Duplicate field")
                    output[key] = value
                return output
            explanation = Explanation.model_validate(json.loads(raw, object_pairs_hook=unique)).model_dump()
            texts = [explanation["summary"], explanation["warning"],
                     *explanation["key_factors"], *explanation["risks"]]
            if (explanation["action"] != result["decision"]["action"]
                    or explanation["decision_hash"] != decision_hash
                    or explanation["risk_level"] != snapshot.risk.risk_level
                    or any(not text.strip() or len(text) > 1200 for text in explanation["key_factors"] + explanation["risks"])
                    or any(character.isdigit() for text in texts for character in text)
                    or any(phrase in " ".join(texts).casefold() for phrase in (
                        "ganancias garantizadas", "rentabilidad garantizada", "beneficio garantizado",
                        "sin riesgo", "guaranteed profit", "risk-free",
                    ))
                    or (result["decision"]["action"] in {"WAIT", "HOLD"} and any(
                        phrase in " ".join(texts).casefold() for phrase in
                        ("compra ahora", "vende ahora", "debes comprar", "debes vender", "buy now", "sell now")))
                    or s.nvidia_api_key.get_secret_value() in raw):
                status = "INCOHERENT_RESPONSE"
                raise ValueError("Explanation contradicts deterministic envelope")
            def save(state):
                cache = state.setdefault("explanations", {})
                for key in list(cache):
                    if cache[key]["until"] <= self.clock():
                        del cache[key]
                cache[cache_key] = {"until": self.clock() + 120, "explanation": explanation}
                # Bound state even under many distinct prediction IDs.
                while len(cache) > 20:
                    del cache[next(iter(cache))]
            self.store.update(save)
            status = "EXPLAINED"
            return 200, {**result, "status": status, "cached": False, "model": s.nvidia_model,
                         "explanation": explanation,
                         "provider_diagnostics": bounded_client.telemetry if bounded_client else None}
        except NvidiaTrialError as error:
            status = str(error)
            return 502, {**result, "status": status,
                         "provider_diagnostics": bounded_client.telemetry if bounded_client else None}
        except (ValueError, TypeError):
            status = "INVALID_EXPLANATION" if status == "PROVIDER_FAILED" else status
            return 502, {**result, "status": status,
                         "provider_diagnostics": bounded_client.telemetry if bounded_client else None}
        except Exception:
            return 502, {**result, "status": status,
                         "provider_diagnostics": bounded_client.telemetry if bounded_client else None}
        finally:
            try:
                if provider is not None:
                    provider.close()
            finally:
                if bounded_client is not None:
                    bounded_client.close()
                quotas.finish("llm", ticket, claims["sub"], status, decision_hash,
                              diagnostics=bounded_client.telemetry if bounded_client else None)

    def refresh_news(self, claims, cfg, quotas):
        if not (cfg.news and cfg.news_rights):
            return 403, {"status": "NEWS_NOT_AUTHORIZED", "news": self._news(cfg)}
        if not self.settings.marketaux_api_token.get_secret_value():
            return 503, {"status": "MARKETAUX_NOT_CONFIGURED"}
        now = self.clock()
        _, state = self.store.read()
        if state.get("news_next_at", 0) > now:
            return 200, {"status": "CACHED", "news": self._news(cfg)}
        ticket = quotas.reserve("news", claims["sub"], claims["sid"], lease=True)
        if not ticket:
            return 429, {"status": "BUDGET_OR_CONCURRENCY_LIMIT"}
        status, client = "PROVIDER_FAILED", None
        try:
            def claim(state):
                if state.get("news_next_at", 0) > now:
                    return False
                state["news_next_at"] = now + 1800
                return True
            if not self.store.update(claim):
                status = "CACHED"
                return 200, {"status": status, "news": self._news(cfg)}
            from alpha_engine.news.marketaux import MarketauxNewsClient
            client = self.news_client or MarketauxNewsClient(self.settings.marketaux_api_token.get_secret_value(),
                                                           min_match_score=10, allow_empty_filtered=True)
            articles = client.fetch(datetime.fromtimestamp(now, timezone.utc), timeout_seconds=6)
            # Existing client normalizes/deduplicates. Persist only authorized derived observations,
            # never titles, URLs, descriptions, snippets, source text or raw provider payloads.
            cache = {"fetched_at": self.clock(), "observations": [
                {"id": article.article_identifier, "at": article.published_at.timestamp(),
                 "score": article.sentiment_score, "match_score": article.match_score,
                 "entity": article.bitcoin_symbol, "source": article.source}
                for article in articles[:3]]}
            self.store.update(lambda state: state.update(news_cache=cache))
            status = "REFRESHED"
            return 200, {"status": status, "news": self._news(cfg),
                         "provider_diagnostics": getattr(client, "diagnostics", None)}
        except Exception:
            return 502, {"status": status, "news": self._news(cfg),
                         "provider_diagnostics": getattr(client, "diagnostics", None)}
        finally:
            quotas.finish("news", ticket, claims["sub"], status,
                          diagnostics=getattr(client, "diagnostics", None))
