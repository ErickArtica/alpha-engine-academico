"""Streamlit-server transport to the authenticated virtual portfolio endpoint."""

from __future__ import annotations

import httpx

from alpha_engine.config.settings import Settings


class PaperApiError(RuntimeError):
    pass


class PaperApiClient:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        base = settings.recommendation_api_url.rstrip("/")
        if not base.endswith("/api/recommendation"):
            raise ValueError("RECOMMENDATION_API_URL must end with /api/recommendation")
        local_compose = (settings.app_env == "local" and settings.storage_provider == "postgresql"
                         and base.startswith("http://api:8000/"))
        if not base.startswith("https://") and not local_compose and not base.startswith("http://127.0.0.1:"):
            raise ValueError("Paper API requires HTTPS or local Compose")
        key = settings.recommendation_api_key.get_secret_value()
        if not key:
            raise ValueError("RECOMMENDATION_API_KEY is required")
        self.url = base.removesuffix("/recommendation") + "/paper"
        self.key = key
        self.scheme = settings.recommendation_api_auth_scheme
        self.timeout = settings.recommendation_api_timeout_seconds
        self.transport = transport

    def request(self, payload: dict) -> dict:
        headers = ({"Authorization": f"Bearer {self.key}"} if self.scheme == "bearer"
                   else {"x-functions-key": self.key})
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(self.url, json=payload, headers=headers)
            result = response.json()
            if not isinstance(result, dict) or "status" not in result:
                raise ValueError("Invalid paper response")
            if response.status_code not in (200, 400, 404, 409):
                raise PaperApiError("Virtual portfolio service unavailable")
            return result
        except (httpx.HTTPError, ValueError) as exc:
            raise PaperApiError("Virtual portfolio service unavailable") from exc


def paper_request(settings: Settings, payload: dict) -> dict:
    if settings.prescriptive_backend == "http":
        return PaperApiClient(settings).request(payload)
    from alpha_engine.services.paper_trading import handle_paper_request
    return handle_paper_request(payload, settings).body
