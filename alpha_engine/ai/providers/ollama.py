"""Proveedor local Ollama mediante su API HTTP oficial."""

import logging
from time import perf_counter
from typing import Any, Mapping

import httpx

from alpha_engine.ai.context_builder import context_json
from alpha_engine.ai.providers.base import LLMProviderError, LLMProviderTimeoutError
from alpha_engine.ai.schemas import LLMRecommendation
from alpha_engine.config.settings import Settings

logger = logging.getLogger(__name__)


class OllamaProviderError(LLMProviderError):
    """Ollama no respondió con una estructura HTTP utilizable."""


class OllamaProvider:
    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        client: httpx.Client | None = None,
    ):
        self._settings = settings
        self._system_prompt = system_prompt
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=float(settings.ollama_timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        return self._settings.ollama_model

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
            "stream": False,
            "think": False,
            "format": LLMRecommendation.model_json_schema(),
            "options": {
                "temperature": self._settings.llm_temperature,
                "num_ctx": self._settings.ollama_num_ctx,
            },
        }
        started = perf_counter()
        try:
            response = self._client.post("/api/chat", json=body, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            content = payload["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise OllamaProviderError("Ollama devolvió contenido vacío")
            return content
        except httpx.TimeoutException as exc:
            logger.warning("Ollama LIVE timeout: model=%s", self.model_name)
            raise LLMProviderTimeoutError("Ollama exceeded the LIVE timeout") from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.error("Fallo de Ollama: error_type=%s model=%s", type(exc).__name__, self.model_name)
            raise OllamaProviderError("Respuesta HTTP de Ollama incompleta") from exc
        finally:
            logger.info(
                "Llamada LLM: provider=ollama model=%s elapsed_ms=%.1f",
                self.model_name, (perf_counter() - started) * 1000,
            )
