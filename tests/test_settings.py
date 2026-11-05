import os

import pytest
from pydantic import ValidationError

from alpha_engine.config.settings import Settings
from alpha_engine.repositories.factory import (
    create_market_repository, create_prediction_repository, create_recommendation_repository,
)
from alpha_engine.ai.providers.factory import create_llm_provider


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.lower() in Settings.model_fields:
            monkeypatch.delenv(name)


def test_settings_defaults_without_env_file():
    settings = Settings(_env_file=None)
    assert settings.storage_provider == "influxdb"
    assert settings.llm_provider == "ollama"
    assert settings.model_confidence_threshold == 0.60
    assert settings.backfill_days == 30
    assert settings.nvidia_model == "google/gemma-4-31b-it"
    assert settings.nvidia_reasoning_effort is None
    assert settings.llm_temperature == 0.1


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("   ", None), ("low", "low"), (" low ", "low"),
])
def test_nvidia_reasoning_effort_is_optional(value, expected):
    assert Settings(_env_file=None, nvidia_reasoning_effort=value).nvidia_reasoning_effort == expected


@pytest.mark.parametrize("value", ["high", "medium", "LOW", "invalid", True, False])
def test_nvidia_reasoning_effort_rejects_unvalidated_values(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, nvidia_reasoning_effort=value)


@pytest.mark.parametrize("value,expected", [("low", "low"), ("", None), ("  ", None)])
def test_nvidia_reasoning_effort_reads_environment(monkeypatch, value, expected):
    monkeypatch.setenv("NVIDIA_REASONING_EFFORT", value)
    assert Settings(_env_file=None).nvidia_reasoning_effort == expected


def test_token_is_redacted_in_settings_repr():
    settings = Settings(_env_file=None, influx_token="super-secret-token")
    assert "super-secret-token" not in repr(settings)
    assert "**********" in repr(settings)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_confidence_threshold=1.1)


def test_factory_requires_token():
    with pytest.raises(ValueError, match="INFLUX_TOKEN"):
        create_market_repository(Settings(_env_file=None, influx_token=""))


def test_local_postgresql_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        create_market_repository(Settings(_env_file=None, storage_provider="postgresql"))


def test_prediction_factory_requires_token():
    with pytest.raises(ValueError, match="INFLUX_TOKEN"):
        create_prediction_repository(Settings(_env_file=None, influx_token=""))


def test_recommendation_factory_requires_token_and_cloud_llm_is_reserved():
    with pytest.raises(ValueError, match="INFLUX_TOKEN"):
        create_recommendation_repository(Settings(_env_file=None, influx_token=""))
    with pytest.raises(NotImplementedError, match="migración futura"):
        create_llm_provider(Settings(_env_file=None, llm_provider="gemini"))
