import json
import os

import httpx
import pytest

from alpha_engine.ai.providers.base import LLMProviderTimeoutError
from alpha_engine.ai.providers.factory import create_llm_provider
from alpha_engine.ai.providers.nvidia import NvidiaProvider, NvidiaProviderError
from alpha_engine.ai.schemas import parse_llm_recommendation
from alpha_engine.config.settings import Settings


VALID = {
    "signal": "HOLD", "confidence": 0.5, "risk_level": "HIGH",
    "summary": "La evidencia no es suficiente para actuar.",
    "key_factors": ["Señales mixtas"],
    "risks": ["Incertidumbre del modelo"],
    "warning": "Simulación académica sin garantía.",
}


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.lower() in Settings.model_fields:
            monkeypatch.delenv(name)


def test_nvidia_requires_key_without_contacting_api():
    settings = Settings(_env_file=None, llm_provider="nvidia", nvidia_api_key="")
    with pytest.raises(NvidiaProviderError, match="NVIDIA_API_KEY"):
        create_llm_provider(settings)


def test_nvidia_request_uses_secret_header_and_structured_context():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(VALID)}}]})

    settings = Settings(
        _env_file=None, llm_provider="nvidia", nvidia_api_key="test-private-key"
    )
    client = httpx.Client(
        base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler)
    )
    provider = NvidiaProvider(settings, "prompt", client=client)
    raw = provider.generate_recommendation({"market": {"price": 100.0}})

    assert parse_llm_recommendation(raw).signal == "HOLD"
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-private-key"
    body = json.loads(requests[0].content)
    assert body["model"] == settings.nvidia_model
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 1024
    assert body["stream"] is False
    assert "reasoning_effort" not in body
    assert body["messages"][0] == {"role": "system", "content": "prompt"}
    assert json.loads(body["messages"][1]["content"]) == {"market": {"price": 100.0}}
    assert settings.llm_model_name == settings.nvidia_model
    client.close()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_nvidia_http_error_is_controlled_without_leaking_secret(caplog, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "test-private-key"}})

    settings = Settings(_env_file=None, llm_provider="nvidia", nvidia_api_key="test-private-key")
    client = httpx.Client(
        base_url=settings.nvidia_base_url,
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaProvider(settings, "prompt", client=client)
    with pytest.raises(NvidiaProviderError, match="rechazada") as error:
        provider.generate_recommendation({"market": {"price": 100.0}})
    assert len(requests) == 1
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)
    assert error.value.__cause__.response.status_code == status
    assert "test-private-key" not in caplog.text
    client.close()


def test_factory_selects_muse_and_preserves_final_content(monkeypatch):
    requests = []
    content = json.dumps({**VALID, "risk_level": "MEDIUM"})

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"model": "meta/muse-glimmer-30b", "choices": [{
            "finish_reason": "stop",
            "message": {"content": content, "reasoning_content": "internal-placeholder"},
        }]})

    settings = Settings(
        _env_file=None, llm_provider="nvidia", nvidia_api_key="test-private-key",
        nvidia_model="meta/muse-glimmer-30b", nvidia_reasoning_effort="low", llm_temperature=0.95,
    )
    client = httpx.Client(base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler))
    monkeypatch.setattr("alpha_engine.ai.providers.nvidia.httpx.Client", lambda **_kwargs: client)
    provider = create_llm_provider(settings)
    try:
        raw = provider.generate_recommendation({"decision_policy": {"action": "WAIT"}})
        assert isinstance(provider, NvidiaProvider)
        assert provider.model_name == settings.llm_model_name == "meta/muse-glimmer-30b"
        assert raw == content
        assert "internal-placeholder" not in raw
        parsed = parse_llm_recommendation(raw)
        assert parsed.signal == "HOLD" and parsed.risk_level == "MEDIUM"
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body["model"] == "meta/muse-glimmer-30b"
        assert body["reasoning_effort"] == "low"
        assert body["temperature"] == 0.95
        assert body["max_tokens"] == 1024 and body["stream"] is False
        assert "top_p" not in body
        assert json.loads(body["messages"][1]["content"])["decision_policy"]["action"] == "WAIT"
    finally:
        provider.close()
    assert client.is_closed


@pytest.mark.parametrize("model,effort", [
    ("meta/muse-glimmer-30b", None),
    ("meta/muse-glimmer-30b", ""),
    ("google/gemma-4-31b-it", "low"),
    ("meta/llama-3.1-8b-instruct", "low"),
    ("meta/muse-glimmer-30b-other", "low"),
    ("META/MUSE-GLIMMER-30B", "low"),
])
def test_reasoning_effort_is_omitted_unless_explicitly_enabled_for_exact_muse(model, effort):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(VALID)}}]})

    settings = Settings(
        _env_file=None, nvidia_api_key="test-private-key", nvidia_model=model,
        nvidia_reasoning_effort=effort,
    )
    with httpx.Client(base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler)) as client:
        raw = NvidiaProvider(settings, "prompt", client=client).generate_recommendation({})
    assert parse_llm_recommendation(raw).signal == "HOLD"
    assert len(requests) == 1
    assert "reasoning_effort" not in json.loads(requests[0].content)


@pytest.mark.parametrize("content", [None, "", "   ", 123])
def test_empty_or_nontext_content_never_falls_back_to_reasoning(content, caplog):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop",
            "message": {"content": content, "reasoning_content": "test-private-key"},
        }]})

    settings = Settings(_env_file=None, nvidia_api_key="test-private-key")
    with httpx.Client(base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NvidiaProviderError, match="contenido"):
            NvidiaProvider(settings, "prompt", client=client).generate_recommendation({})
    assert len(requests) == 1
    assert "test-private-key" not in caplog.text


@pytest.mark.parametrize("content", [json.dumps(VALID), "", None])
def test_length_finish_reason_is_rejected_without_retry_even_with_valid_json(content, caplog):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "length",
            "message": {"content": content, "reasoning_content": "test-private-key"},
        }]})

    settings = Settings(_env_file=None, nvidia_api_key="test-private-key")
    with httpx.Client(base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NvidiaProviderError, match="finish_reason=length"):
            NvidiaProvider(settings, "prompt", client=client).generate_recommendation({})
    assert len(requests) == 1
    assert "test-private-key" not in caplog.text


@pytest.mark.parametrize("error_type", [
    httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
])
def test_timeout_preserves_original_cause_without_retry_or_secret_leak(error_type, caplog):
    requests = []
    failures = []

    def handler(request):
        requests.append(request)
        failure = error_type("test-private-key", request=request)
        failures.append(failure)
        raise failure

    settings = Settings(_env_file=None, nvidia_api_key="test-private-key")
    with httpx.Client(base_url=settings.nvidia_base_url, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMProviderTimeoutError) as error:
            NvidiaProvider(settings, "prompt", client=client).generate_recommendation({}, timeout_seconds=7.5)
    assert len(requests) == 1
    assert error.value.__cause__ is failures[0]
    assert requests[0].extensions["timeout"] == dict(connect=7.5, read=7.5, write=7.5, pool=7.5)
    assert "test-private-key" not in caplog.text


@pytest.mark.parametrize("payload", [
    {}, {"choices": []}, {"choices": [{}]}, {"choices": [None]},
    {"choices": [{"message": {}}]}, [],
])
def test_malformed_provider_payload_is_controlled(payload):
    settings = Settings(_env_file=None, nvidia_api_key="test-private-key")
    with httpx.Client(
        base_url=settings.nvidia_base_url,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    ) as client:
        with pytest.raises(NvidiaProviderError, match="incompleta"):
            NvidiaProvider(settings, "prompt", client=client).generate_recommendation({})


def test_non_json_http_body_is_controlled():
    settings = Settings(_env_file=None, nvidia_api_key="test-private-key")
    with httpx.Client(
        base_url=settings.nvidia_base_url,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not-json")),
    ) as client:
        with pytest.raises(NvidiaProviderError, match="incompleta"):
            NvidiaProvider(settings, "prompt", client=client).generate_recommendation({})
