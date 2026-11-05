"""Selección del proveedor LLM sin cambiar la lógica de recomendación."""

from alpha_engine.ai.prompts.loader import load_recommendation_prompt
from alpha_engine.ai.providers.base import LLMProvider
from alpha_engine.ai.providers.nvidia import NvidiaProvider
from alpha_engine.ai.providers.ollama import OllamaProvider
from alpha_engine.config.settings import Settings


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings, load_recommendation_prompt())
    if settings.llm_provider == "nvidia":
        return NvidiaProvider(settings, load_recommendation_prompt())
    raise NotImplementedError(
        f"LLM_PROVIDER={settings.llm_provider} está reservado para la migración futura"
    )
