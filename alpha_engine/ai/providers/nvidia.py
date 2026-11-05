"""Adaptador opcional para el endpoint alojado de NVIDIA API Catalog."""

import logging
from time import perf_counter
from typing import Any, Mapping

import httpx

from alpha_engine.ai.context_builder import context_json
from alpha_engine.ai.providers.base import LLMProviderError, LLMProviderTimeoutError
from alpha_engine.config.settings import Settings

logger = logging.getLogger(__name__)


class NvidiaProviderError(LLMProviderError):
    """La API de NVIDIA no devolvió una respuesta utilizable."""


class NvidiaProvider:
    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        client: httpx.Client | None = None,
    ):
        api_key = settings.nvidia_api_key.get_secret_value().strip()
        if not api_key:
            raise NvidiaProviderError("Configure NVIDIA_API_KEY en .env antes de usar NVIDIA")
        self._api_key = api_key
        self._settings = settings
        self._system_prompt = system_prompt
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.nvidia_base_url.rstrip("/") + "/",
            timeout=float(settings.nvidia_timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        return self._settings.nvidia_model

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate_recommendation(
        self, context: Mapping[str, Any], *, timeout_seconds: float | None = None,
    ) -> str:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": context_json(dict(context))},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": 1024,
            "stream": False,
        }
        if (
            self.model_name == "meta/muse-glimmer-30b"
            and self._settings.nvidia_reasoning_effort == "low"
        ):
            body["reasoning_effort"] = self._settings.nvidia_reasoning_effort
        started = perf_counter()
        try:
            response = self._client.post(
                "chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                logger.warning("NVIDIA response truncated: model=%s", self.model_name)
                raise NvidiaProviderError("NVIDIA devolvió una respuesta truncada (finish_reason=length)")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise NvidiaProviderError("NVIDIA devolvió contenido vacío")
            return content
        except httpx.TimeoutException as exc:
            logger.warning("NVIDIA LIVE timeout: model=%s", self.model_name)
            raise LLMProviderTimeoutError("NVIDIA exceeded the LIVE timeout") from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error(
                "Fallo de NVIDIA: error_type=%s status=%s model=%s",
                type(exc).__name__,
                exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None,
                self.model_name,
            )
            raise NvidiaProviderError("Respuesta HTTP de NVIDIA incompleta o rechazada") from exc
        finally:
            logger.info(
                "Llamada LLM: provider=nvidia model=%s elapsed_ms=%.1f",
                self.model_name, (perf_counter() - started) * 1000,
            )
