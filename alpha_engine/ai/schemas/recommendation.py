"""Validación estricta de la respuesta del LLM antes de mostrarla o guardarla."""

import json
import logging
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
logger = logging.getLogger(__name__)


class LLMRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    summary: NonEmptyText
    key_factors: list[NonEmptyText]
    risks: list[NonEmptyText]
    warning: NonEmptyText

    @model_validator(mode="after")
    def reject_unverified_numbers(self) -> "LLMRecommendation":
        texts = [self.summary, *self.key_factors, *self.risks, self.warning]
        if any(any(character.isdigit() for character in value) for value in texts):
            raise ValueError("El texto LLM contiene cifras no verificadas")
        return self


class InvalidLLMResponseError(ValueError):
    """La respuesta no cumple el esquema; nunca debe mostrarse sin validar."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"Constante JSON no permitida: {value}")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clave JSON duplicada: {key}")
        result[key] = value
    return result


def _remove_unverified_figures(payload: dict) -> dict:
    """Omite frases numéricas del LLM; no intenta corregirlas ni publicarlas."""
    cleaned = dict(payload)
    replacements = {
        "summary": "La explicación numérica de la IA se omitió; consulte los datos calculados en el panel.",
        "warning": "Recomendación académica sin garantía de resultados.",
    }
    redacted = False
    for field, replacement in replacements.items():
        value = cleaned.get(field)
        if isinstance(value, str) and any(char.isdigit() for char in value):
            cleaned[field] = replacement
            redacted = True
    for field in ("key_factors", "risks"):
        values = cleaned.get(field)
        if isinstance(values, list) and all(isinstance(item, str) for item in values):
            kept = [item for item in values if not any(char.isdigit() for char in item)]
            if len(kept) != len(values):
                cleaned[field] = kept or ["Se omitieron afirmaciones numéricas de la IA no verificadas."]
                redacted = True
    if redacted:
        logger.warning("Se omitieron cifras no verificadas del texto LLM")
    return cleaned


def parse_llm_recommendation(raw: str) -> LLMRecommendation:
    try:
        payload = json.loads(
            raw, parse_constant=_reject_constant, object_pairs_hook=_unique_keys
        )
        if not isinstance(payload, dict):
            raise ValueError("La respuesta debe ser un objeto JSON")
        return LLMRecommendation.model_validate(_remove_unverified_figures(payload))
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise InvalidLLMResponseError("Respuesta LLM inválida o fuera de esquema") from exc
