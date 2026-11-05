"""Temporary, fail-closed HTTP diagnostic. Never uses production repositories."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

PATH = "/internal/diagnostics/muse"
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "muse3_context.json"
PROMPT = ROOT / "alpha_engine/ai/prompts/recommendation_explain_v1.txt"
MODEL = "meta/muse-glimmer-30b"
PROMPT_SHA256 = "ff327ebd683f81500edf3f58688c430e63114788d90219fb5b11e7841b02da43"
CONTEXT_SHA256 = "8760719d8c64c01aab4ca1fd5bf9501d72183b08f5aa451863584fe6eb002fae"
SOURCE_SHA256 = "c8eb21e13d922e7dace3773427e8bc60cbc4a992285d79b227de0b6cc30df602"
TOTAL_TIMEOUT_SECONDS = 40.0
MAX_WINDOW_SECONDS = 300
MAX_RECEIPT_BYTES = 16384
NOT_FOUND = (404, {"status": "NOT_FOUND"})


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(value: str | bytes) -> object:
    def invalid_constant(_value: str) -> None:
        raise ValueError("Non-finite JSON")
    return json.loads(value, object_pairs_hook=_unique, parse_constant=invalid_constant)


def pinned_inputs() -> tuple[dict, str]:
    """Keep the exact MUSE-3 messages; all new metadata lives in the receipt."""
    context = strict_json(FIXTURE.read_bytes())
    prompt_bytes = PROMPT.read_bytes()
    if (not isinstance(context, dict) or digest(canonical(context)) != CONTEXT_SHA256
            or digest(prompt_bytes) != PROMPT_SHA256):
        raise ValueError("Pinned diagnostic input mismatch")
    return context, prompt_bytes.decode("utf-8")


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TokenUsage(ClosedModel):
    prompt_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000)
    completion_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000)
    total_tokens: int | None = Field(default=None, ge=0, le=1_000_000_000)


class SemanticChecks(ClosedModel):
    signal_matches_action: bool
    risk_matches_context: bool
    historical_non_operational_context: bool
    narrative_without_digits: bool
    mentions_expiry: bool
    mentions_news_absence: bool


class ParsedFields(ClosedModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class WorkerReport(ClosedModel):
    """Only enums, numbers, booleans and hashes cross the child boundary."""
    status: Literal["PASS", "PARTIAL", "FAILED", "TIMEOUT", "CANCELLED", "WORKER_ERROR"] = "FAILED"
    stage: Literal["preflight", "provider", "parser", "complete", "timeout", "cancelled", "protocol"] = "preflight"
    network_attempts: int | None = Field(default=0, ge=0, le=1)
    http_status: int | None = Field(default=None, ge=100, le=599)
    reported_model: Literal["meta/muse-glimmer-30b", "OTHER"] | None = None
    model_matches: bool | None = None
    envelope_json_valid: bool | None = None
    finish_reason: Literal["stop", "length", "OTHER"] | None = None
    usage: TokenUsage | None = None
    provider_duration_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    final_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    final_json_valid: bool | None = None
    raw_schema_valid: bool | None = None
    parser_executed: bool = False
    parser_accepted: bool | None = None
    parser_modified_content: bool | None = None
    parsed_fields: ParsedFields | None = None
    semantic_checks: SemanticChecks | None = None
    technical_pass: bool = False
    semantic_contract_pass: bool | None = None
    child_reaped: bool | None = None


def _header(scope: dict, name: bytes) -> bytes | None:
    values = [v for k, v in scope.get("headers", ()) if k.lower() == name]
    return values[0] if len(values) == 1 else None


def _utc(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
        raise ValueError("UTC required")
    return instant


def _origin(env: Mapping[str, str]) -> dict:
    revision = env.get("RENDER_GIT_COMMIT", "")
    service = env.get("RENDER_SERVICE_ID", "")
    instance = env.get("RENDER_INSTANCE_ID", "")
    if (env.get("RENDER") != "true" or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not re.fullmatch(r"srv-[A-Za-z0-9_-]{1,100}", service)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", instance)):
        raise ValueError("Render metadata required")
    return {"platform": "Render", "revision": revision,
            "service_id_sha256": digest(service.encode()),
            "instance_id_sha256": digest(instance.encode()),
            "attestation": "platform_environment_not_independent_verification"}


def child_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Do not inherit Supabase, proxy, tracing, or production settings."""
    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "NVIDIA_API_KEY")
    result = {name: env[name] for name in allowed if name in env}
    result.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                  PYTHONNOUSERSITE="1", MUSE_DIAGNOSTIC_CHILD="1")
    return result


async def _kill_and_reap(process) -> None:
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    await asyncio.wait_for(process.wait(), timeout=5)


async def run_worker(env: Mapping[str, str], *, timeout_seconds: float = TOTAL_TIMEOUT_SECONDS,
                     command: list[str] | None = None) -> WorkerReport:
    """Kill and reap on timeout/cancellation; never launch a replacement."""
    started = perf_counter()
    command = command or [sys.executable, "-B", "-m", "deploy.render.muse_diagnostic_worker", "--execute"]
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        *command, cwd=ROOT, env=dict(env), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=(os.name == "posix"),
    ))
    process = None
    try:
        process = await asyncio.shield(spawn)
        remaining = max(0.001, timeout_seconds - (perf_counter() - started))
        raw, _ = await asyncio.wait_for(process.communicate(), timeout=remaining)
        if process.returncode == 124:
            return WorkerReport(status="TIMEOUT", stage="timeout", network_attempts=None, child_reaped=True)
        if process.returncode != 0 or len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError("Invalid worker protocol")
        report = WorkerReport.model_validate(strict_json(raw))
        return report.model_copy(update={"child_reaped": True})
    except asyncio.CancelledError:
        if process is None:
            process = await asyncio.shield(spawn)
        await asyncio.shield(_kill_and_reap(process))
        raise
    except TimeoutError:
        await _kill_and_reap(process)
        return WorkerReport(status="TIMEOUT", stage="timeout", network_attempts=None, child_reaped=True)
    except Exception:
        if process is not None:
            await _kill_and_reap(process)
        return WorkerReport(status="WORKER_ERROR", stage="protocol", network_attempts=None,
                            child_reaped=process is not None)


class MuseDiagnostic:
    def __init__(self, env: Mapping[str, str], *,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 runner: Callable[[Mapping[str, str]], Awaitable[WorkerReport]] = run_worker):
        self._env, self._clock, self._runner = env, clock, runner
        self._lock = Lock()
        self._run_id = uuid4().hex
        self._nonce = secrets.token_urlsafe(32)
        self._state = "NEW"
        self._receipt: dict | None = None

    def _configuration(self, scope: dict) -> tuple[datetime, dict] | None:
        try:
            env = self._env
            token = env.get("MUSE_DIAGNOSTIC_TOKEN", "")
            if (env.get("MUSE_DIAGNOSTIC_ENABLED") != "true"
                    or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token)
                    or not env.get("NVIDIA_API_KEY", "").strip()
                    or scope.get("query_string", b"")):
                return None
            for name in ("INTERNAL_CRON_SECRET", "RECOMMENDATION_API_SECRET", "NVIDIA_API_KEY", "SUPABASE_KEY"):
                if hmac.compare_digest(token.encode(), env.get(name, "").encode()):
                    return None
            start = _utc(env.get("MUSE_DIAGNOSTIC_STARTS_AT", ""))
            end = _utc(env.get("MUSE_DIAGNOSTIC_EXPIRES_AT", ""))
            if not 0 < (end - start).total_seconds() <= MAX_WINDOW_SECONDS:
                return None
            now = self._clock()
            if now.tzinfo is None or not start <= now < end:
                return None
            presented = _header(scope, b"authorization")
            if presented is None or not hmac.compare_digest(presented, f"Bearer {token}".encode()):
                return None
            return end, _origin(env)
        except (ValueError, TypeError, AttributeError, OverflowError):
            return None

    async def handle(self, scope: dict, read_payload: Callable[[], Awaitable[object]]) -> tuple[int, dict]:
        configuration = self._configuration(scope)
        if configuration is None:
            return NOT_FOUND
        method = scope.get("method")
        if method == "GET":
            with self._lock:
                if self._state == "FINISHED":
                    return 200, {"status": "FINISHED", "receipt": self._receipt}
                if self._state == "RUNNING":
                    return 200, {"status": "RUNNING", "run_id": self._run_id}
                try:
                    pinned_inputs()
                except Exception:
                    return 503, {"status": "DIAGNOSTIC_UNAVAILABLE"}
                self._state = "PREPARED"
                return 200, {"status": "PREPARED", "run_id": self._run_id, "nonce": self._nonce,
                             "expires_at": configuration[0].isoformat(),
                             "prompt_sha256": PROMPT_SHA256, "context_sha256": CONTEXT_SHA256}
        if method != "POST":
            return NOT_FOUND
        content_type = _header(scope, b"content-type")
        if content_type is None or content_type.split(b";", 1)[0].strip().lower() != b"application/json":
            return 415, {"status": "INVALID_REQUEST"}
        try:
            payload = await asyncio.wait_for(read_payload(), timeout=5)
        except (ValueError, UnicodeDecodeError, OverflowError, TimeoutError):
            return 400, {"status": "INVALID_REQUEST"}
        if (not isinstance(payload, dict) or set(payload) != {"run_id", "nonce"}
                or not all(isinstance(v, str) and v.isascii() for v in payload.values())):
            return 400, {"status": "INVALID_REQUEST"}
        with self._lock:
            configuration = self._configuration(scope)
            if configuration is None:
                return NOT_FOUND
            end, origin = configuration
            if (self._state != "PREPARED"
                    or not hmac.compare_digest(payload["run_id"], self._run_id)
                    or not hmac.compare_digest(payload["nonce"], self._nonce)):
                return 409, {"status": "PERMIT_UNAVAILABLE"}
            if (end - self._clock()).total_seconds() < TOTAL_TIMEOUT_SECONDS + 5:
                return NOT_FOUND
            # This transition is irreversible, including provider/worker failures.
            self._state = "RUNNING"
            started_at = self._clock().astimezone(timezone.utc).isoformat()
            env = child_environment(self._env)
        started = perf_counter()
        cancelled = False
        try:
            report = await self._runner(env)
        except asyncio.CancelledError:
            report = WorkerReport(status="CANCELLED", stage="cancelled", network_attempts=None,
                                  child_reaped=True)
            cancelled = True
        except Exception:
            report = WorkerReport(status="WORKER_ERROR", stage="protocol", network_attempts=None)
        receipt = {
            "schema_version": "muse_render_diagnostic_v1", "run_id": self._run_id,
            "diagnostic_only": True, "operational_recommendation": False,
            "origin": origin, "started_at": started_at,
            "finished_at": self._clock().astimezone(timezone.utc).isoformat(),
            "total_duration_ms": round((perf_counter() - started) * 1000, 3),
            "requested_model": MODEL, "prompt_version": "recommendation_explain_v1",
            "prompt_sha256": PROMPT_SHA256, "context_sha256": CONTEXT_SHA256,
            "fixture_source": "MUSE-3_LOCAL_HISTORICAL_NOT_RENDER_LIVE",
            "source_evidence_sha256": SOURCE_SHA256,
            "diagnostic_profile": {"temperature": 0.95, "reasoning_effort": "low",
                                   "max_tokens": 1024, "stream": False,
                                   "per_phase_timeout_seconds": 30, "total_timeout_seconds": TOTAL_TIMEOUT_SECONDS,
                                   "automatic_retries": 0, "news_enabled": False},
            "production_settings_modified": False,
            "persistence": "NO_REPOSITORIES_CONSTRUCTED_NO_DIAGNOSTIC_WRITES",
            "raw_response_saved": False, "reasoning_saved": False,
            "semantic_review_scope": "AUTOMATED_CONTRACT_AND_LEXICAL_CHECKS_NOT_FULL_REVIEW",
            "result": report.model_dump(mode="json"),
        }
        with self._lock:
            self._receipt, self._state = receipt, "FINISHED"
        if cancelled:
            raise asyncio.CancelledError
        return 200, {"status": "FINISHED", "receipt": receipt}
