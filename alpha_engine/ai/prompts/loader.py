"""Carga el prompt versionado incluido con el paquete."""

from pathlib import Path

PROMPT_VERSION = "recommendation_explain_v1"


def load_recommendation_prompt() -> str:
    path = Path(__file__).with_name(f"{PROMPT_VERSION}.txt")
    return path.read_text(encoding="utf-8")
