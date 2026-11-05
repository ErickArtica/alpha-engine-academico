"""Academic-only NVIDIA transport: one POST, bounded SSE, final content only."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from time import perf_counter

import httpx


class NvidiaTrialError(RuntimeError):
    """Safe diagnostic code; no provider body, headers, URL or credentials."""


@dataclass(frozen=True)
class GenerationPolicy:
    connect: float = 5
    read: float = 60
    total: float = 85
    max_tokens: int = 2048

    @classmethod
    def load(cls, env=None):
        env = os.environ if env is None else env
        policy = cls(float(env.get("ACADEMIC_NVIDIA_CONNECT_SECONDS", "5")),
                     float(env.get("ACADEMIC_NVIDIA_READ_SECONDS", "60")),
                     float(env.get("ACADEMIC_NVIDIA_TOTAL_SECONDS", "85")),
                     int(env.get("ACADEMIC_NVIDIA_MAX_TOKENS", "2048")))
        if not (1 <= policy.connect <= 10 and 30 <= policy.read <= 90
                and 60 <= policy.total <= 90 and policy.read <= policy.total
                and 1024 <= policy.max_tokens <= 4096):
            raise ValueError("Invalid bounded academic generation policy")
        return policy


def compact_context(context):
    """Keep the authoritative envelope; do not send history or redundant indicators."""
    return {
        "system": {"purpose": "Explain a precomputed academic decision", "real_trading": False},
        "market": {key: context["market"][key] for key in ("symbol", "close_time")},
        "prediction": context["prediction"],
        "decision_policy": {key: context["decision_policy"][key] for key in
                            ("action", "position", "minimum_model_probability", "reason", "action_authority")},
        "risk": {"level": context["risk"]["level"], "triggers": context["risk"]["triggers"]},
        "academic_demo": context["academic_demo"],
        "news_context": {key: context["news_context"].get(key) for key in
                         ("status", "classification", "score", "considered", "fetched_at", "reason")},
        "decision_hash": context["decision_hash"],
    }


class BoundedNvidiaClient:
    """Adapt the existing NvidiaProvider protocol, without changing the original backend."""

    def __init__(self, *, policy=None, transport=None):
        self.policy = policy or GenerationPolicy.load()
        self.transport = transport
        self.attempted = False
        self.telemetry = {"stream": True, "usage": None}

    @staticmethod
    def _usage(value):
        if not isinstance(value, dict):
            return None
        return {k: v for k, v in value.items() if k in
                {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(v) is int and 0 <= v <= 1_000_000_000} or None

    def post(self, path, *, json, headers, timeout):
        if self.attempted or path != "chat/completions":
            raise NvidiaTrialError("DUPLICATE_OR_UNEXPECTED_PROVIDER_REQUEST")
        self.attempted = True
        body = {**json, "max_tokens": self.policy.max_tokens, "temperature": .95,
                "top_p": 1, "reasoning_effort": "low", "stream": True, "tool_choice": "none"}
        return asyncio.run(self._request(body, headers, min(float(timeout), self.policy.total)))

    async def _request(self, body, headers, total):
        started = perf_counter()
        timeout = httpx.Timeout(self.policy.read, connect=self.policy.connect, write=5, pool=2)
        transport = self.transport or httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        try:
            async with asyncio.timeout(total):
                async with httpx.AsyncClient(transport=transport, timeout=timeout, trust_env=False,
                                             follow_redirects=False) as client:
                    async with client.stream("POST", "https://integrate.api.nvidia.com/v1/chat/completions",
                                             json=body, headers={**headers, "Accept": "text/event-stream"}) as response:
                        self.telemetry["http_status"] = response.status_code
                        self.telemetry["headers_seconds"] = round(perf_counter() - started, 3)
                        if response.status_code != 200:
                            reason = {202: "PROVIDER_PENDING_NO_RETRY", 429: "PROVIDER_RATE_LIMIT"}.get(
                                response.status_code, "PROVIDER_HTTP_ERROR")
                            raise NvidiaTrialError(reason)
                        if "text/event-stream" not in response.headers.get("content-type", ""):
                            raw = bytearray()
                            async for chunk in response.aiter_bytes():
                                raw.extend(chunk)
                                if len(raw) > 262144:
                                    raise NvidiaTrialError("PROVIDER_RESPONSE_TOO_LARGE")
                            data = json.loads(raw)
                            choice = data["choices"][0]
                            if choice.get("finish_reason") != "stop":
                                raise NvidiaTrialError("PROVIDER_INCOMPLETE_RESPONSE")
                            self.telemetry["usage"] = self._usage(data.get("usage"))
                            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                "message": {"content": choice["message"]["content"]}}],
                                "usage": self.telemetry["usage"]}, request=response.request)
                        content, finish, done, pending, event, size = [], None, False, b"", [], 0
                        def consume(lines):
                            nonlocal finish, done
                            data = b"\n".join(line[5:].lstrip() for line in lines if line.startswith(b"data:"))
                            if not data:
                                return
                            if done:
                                raise NvidiaTrialError("PROVIDER_DATA_AFTER_DONE")
                            if data == b"[DONE]":
                                done = True
                                return
                            item = json.loads(data)
                            self.telemetry["usage"] = self._usage(item.get("usage")) or self.telemetry["usage"]
                            for choice in item.get("choices", []):
                                if choice.get("index", 0) != 0 or choice.get("delta", {}).get("tool_calls"):
                                    raise NvidiaTrialError("PROVIDER_UNEXPECTED_CHOICE_OR_TOOL")
                                # reasoning_content is deliberately discarded, not saved or displayed.
                                value = choice.get("delta", {}).get("content")
                                if value is not None:
                                    if not isinstance(value, str):
                                        raise NvidiaTrialError("PROVIDER_INVALID_FINAL_CONTENT")
                                    content.append(value)
                                    if sum(map(len, content)) > 12000:
                                        raise NvidiaTrialError("PROVIDER_RESPONSE_TOO_LARGE")
                                if choice.get("finish_reason") is not None:
                                    finish = choice["finish_reason"]
                        async for chunk in response.aiter_bytes():
                            if "first_chunk_seconds" not in self.telemetry:
                                self.telemetry["first_chunk_seconds"] = round(perf_counter() - started, 3)
                            size += len(chunk)
                            if size > 262144:
                                raise NvidiaTrialError("PROVIDER_RESPONSE_TOO_LARGE")
                            pending += chunk
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                line = line.rstrip(b"\r")
                                if line:
                                    event.append(line)
                                else:
                                    consume(event)
                                    event = []
                        if pending.strip() or event or not done or finish != "stop" or not "".join(content).strip():
                            raise NvidiaTrialError("PROVIDER_INCOMPLETE_STREAM")
                        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                            "message": {"content": "".join(content)}}], "usage": self.telemetry["usage"]},
                            request=response.request)
        except NvidiaTrialError:
            raise
        except httpx.ConnectTimeout:
            raise NvidiaTrialError("PROVIDER_CONNECT_TIMEOUT") from None
        except httpx.ReadTimeout:
            raise NvidiaTrialError("PROVIDER_READ_TIMEOUT") from None
        except TimeoutError:
            raise NvidiaTrialError("PROVIDER_TOTAL_DEADLINE") from None
        except httpx.HTTPError:
            raise NvidiaTrialError("PROVIDER_NETWORK_ERROR") from None
        except (ValueError, KeyError, TypeError, IndexError, UnicodeError, AttributeError):
            raise NvidiaTrialError("PROVIDER_INVALID_RESPONSE") from None
        finally:
            self.telemetry["elapsed_seconds"] = round(perf_counter() - started, 3)

    def close(self):
        # Each request owns and closes its asynchronous client, including cancellation.
        pass
