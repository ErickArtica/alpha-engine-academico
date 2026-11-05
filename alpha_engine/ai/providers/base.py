"""Contrato del proveedor LLM; la validación de JSON corresponderá al servicio de IA."""

from typing import Any, Mapping, Protocol


class LLMProviderError(RuntimeError):
    """Fallo controlado del proveedor de generación."""


class LLMProviderTimeoutError(LLMProviderError):
    """A provider did not complete before the LIVE request deadline."""


class LLMProvider(Protocol):
    def close(self) -> None: ...

    @property
    def model_name(self) -> str: ...

    def generate_recommendation(
        self, context: Mapping[str, Any], *, timeout_seconds: float | None = None,
    ) -> str:
        """Devuelve texto crudo; el servicio deberá validarlo antes de usarlo."""
