"""Small server-to-server transport for the existing RecommendationUseCase."""

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import BoundedSemaphore
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4
from uuid import UUID

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.data.recommendation_response import RecommendationResult
from alpha_engine.data.prescriptive_schemas import RecommendationRequestRecord
from alpha_engine.ai.prompts.loader import PROMPT_VERSION
from alpha_engine.repositories.factory import create_recommendation_request_repository

if TYPE_CHECKING:
    from alpha_engine.services.recommendation_use_case import RecommendationUseCase


_PREDICTION_ID = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_SLOT = BoundedSemaphore(1)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiReply:
    status_code: int
    body: dict[str, Any]


def handle_recommendation_request(
    payload: object, settings: Settings, *,
    use_case_factory: Callable[[Settings], "RecommendationUseCase"] | None = None,
    request_repository_factory: Callable = create_recommendation_request_repository,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ApiReply:
    """Validate one narrow request and return only public DTO fields.

    The HTTP deployment wrapper authenticates before calling this function. The audit
    repository supplies a best-effort shared rate limit; the process slot
    prevents simultaneous local LLM calls on one instance.
    """
    if not isinstance(payload, dict) or set(payload) not in ({"prediction_id"}, {"prediction_id", "portfolio_id"}):
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    prediction_id = payload["prediction_id"]
    if not isinstance(prediction_id, str) or not _PREDICTION_ID.fullmatch(prediction_id):
        return ApiReply(400, {"status": "INVALID_REQUEST"})
    portfolio_id = payload.get("portfolio_id")
    if portfolio_id is not None:
        try:
            portfolio_id = str(UUID(portfolio_id))
        except (ValueError, TypeError, AttributeError):
            return ApiReply(400, {"status": "INVALID_REQUEST"})
    if not _PROCESS_SLOT.acquire(blocking=False):
        return ApiReply(429, {"status": "RATE_LIMITED"})
    try:
        instant = now().astimezone(timezone.utc)
        with_repository = request_repository_factory(settings)
        try:
            recent = with_repository.list_recent(settings.symbol, limit=100)
            attempted = sum(
                row.prediction_id == prediction_id
                and row.requested_at >= instant - timedelta(minutes=1)
                for row in recent
            )
            if attempted >= settings.recommendation_max_attempts_per_minute:
                with_repository.save(RecommendationRequestRecord(
                    request_id=uuid4(), prediction_id=prediction_id,
                    symbol=settings.symbol, interval=settings.interval,
                    provider=settings.llm_provider, llm_model=settings.llm_model_name,
                    prompt_version=PROMPT_VERSION, requested_at=instant,
                    finished_at=instant, total_latency_ms=0, status="RATE_LIMITED",
                    error_type="RateLimit", error_message="Prescriptive request limit reached",
                    created_at=instant,
                ))
                return ApiReply(429, {"status": "RATE_LIMITED"})
        finally:
            with_repository.close()
        if use_case_factory is None:
            from alpha_engine.services.recommendation_use_case import RecommendationUseCase
            use_case_factory = RecommendationUseCase.from_settings
        with use_case_factory(settings) as use_case:
            result = (use_case.request(prediction_id, portfolio_id=portfolio_id)
                      if portfolio_id else use_case.request(prediction_id))
        return ApiReply(200, result.model_dump(mode="json", exclude={"guardrails"}))
    except Exception as exc:
        # Never serialize provider, storage, or credential-bearing exceptions.
        logger.warning("Prescriptive endpoint failed: %s", type(exc).__name__)
        return ApiReply(503, {"status": "TEMPORARILY_UNAVAILABLE"})
    finally:
        _PROCESS_SLOT.release()


class RecommendationApiError(RuntimeError):
    pass


class RecommendationApiClient:
    """HTTP caller used only from the Streamlit server process."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        url = settings.recommendation_api_url.rstrip("/")
        local_compose = (settings.app_env == "local" and settings.storage_provider == "postgresql"
                         and url.startswith("http://api:8000/"))
        if not url.startswith("https://") and not url.startswith("http://127.0.0.1:") and not local_compose:
            raise ValueError("RECOMMENDATION_API_URL must use HTTPS")
        key = settings.recommendation_api_key.get_secret_value()
        if not key:
            raise ValueError("RECOMMENDATION_API_KEY is required")
        self._url = url
        self._key = key
        self._auth_scheme = settings.recommendation_api_auth_scheme
        self._timeout = settings.recommendation_api_timeout_seconds
        self._transport = transport

    def request(self, prediction_id: str, portfolio_id: str | None = None) -> RecommendationResult:
        try:
            headers = (
                {"Authorization": f"Bearer {self._key}"}
                if self._auth_scheme == "bearer" else {"x-functions-key": self._key}
            )
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(
                    self._url, json={"prediction_id": prediction_id, **({"portfolio_id": portfolio_id} if portfolio_id else {})},
                    headers=headers,
                )
            response.raise_for_status()
            return RecommendationResult.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise RecommendationApiError("Prescriptive service unavailable") from exc

    def demo_explanation(self, prediction_id: str) -> dict:
        """Explicit local-only explanation, separate from operational requests."""
        if not self._url.startswith("http://api:8000/"):
            raise RecommendationApiError("Local demo endpoint unavailable")
        try:
            with httpx.Client(timeout=max(self._timeout, 90), transport=self._transport) as client:
                response = client.post(
                    "http://api:8000/api/demo-explanation",
                    json={"prediction_id": prediction_id},
                    headers={"Authorization": f"Bearer {self._key}"},
                )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RecommendationApiError("Local demo explanation unavailable") from exc
