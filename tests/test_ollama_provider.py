import json

import httpx
import pytest

from alpha_engine.ai.prompts.loader import PROMPT_VERSION, load_recommendation_prompt
from alpha_engine.ai.providers.ollama import OllamaProvider, OllamaProviderError
from alpha_engine.ai.schemas import InvalidLLMResponseError, parse_llm_recommendation
from alpha_engine.config.settings import Settings


VALID = {
    "signal": "BUY", "confidence": 0.7, "risk_level": "LOW",
    "summary": "El modelo favorece una subida.",
    "key_factors": ["Probabilidad superior al umbral"],
    "risks": ["Volatilidad presente"],
    "warning": "Simulación académica sin garantía.",
}


def test_llm_json_parser_rejects_invalid_values_and_extra_text():
    assert parse_llm_recommendation(json.dumps(VALID)).signal == "BUY"
    bad = [
        "```json\n" + json.dumps(VALID) + "\n```",
        json.dumps({**VALID, "confidence": 1.2}),
        json.dumps({**VALID, "confidence": float("nan")}),
        json.dumps({**VALID, "extra": "not allowed"}),
        '{"signal":"BUY","signal":"SELL"}',
    ]
    for raw in bad:
        with pytest.raises(InvalidLLMResponseError):
            parse_llm_recommendation(raw)


def test_llm_numeric_narrative_is_removed_before_display_or_cache():
    payload = {
        **VALID,
        "summary": "El stop está a 1.5%.",
        "key_factors": ["RSI 6:60.43", "MACD favorable"],
        "risks": ["Riesgo moderado"],
    }
    result = parse_llm_recommendation(json.dumps(payload))
    assert result.signal == "BUY"
    assert result.summary.startswith("La explicación numérica")
    assert result.key_factors == ["MACD favorable"]
    assert "1.5" not in result.model_dump_json()


def test_ollama_request_uses_prompt_schema_and_configured_options():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": json.dumps(VALID)}})

    client = httpx.Client(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    settings = Settings(_env_file=None)
    prompt = load_recommendation_prompt()
    provider = OllamaProvider(settings, prompt, client=client)
    result = provider.generate_recommendation({"market": {"price": 100.0}})
    assert parse_llm_recommendation(result).signal == "BUY"
    assert requests[0].url.path == "/api/chat"
    body = json.loads(requests[0].content)
    assert body["model"] == "qwen3:4b-instruct"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0.1, "num_ctx": 4096}
    assert body["format"]["type"] == "object"
    assert body["messages"][0] == {"role": "system", "content": prompt}
    assert json.loads(body["messages"][1]["content"]) == {"market": {"price": 100.0}}
    assert PROMPT_VERSION == "recommendation_explain_v1"
    client.close()


def test_ollama_http_failure_is_controlled():
    client = httpx.Client(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    provider = OllamaProvider(Settings(_env_file=None), "prompt", client=client)
    with pytest.raises(OllamaProviderError):
        provider.generate_recommendation({"a": 1})
    client.close()
