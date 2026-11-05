"""Three narrow HTTP routes; analytic work remains in alpha_engine."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

from deploy.render.muse_diagnostic import PATH as MUSE_DIAGNOSTIC_PATH, MuseDiagnostic

from alpha_engine.observability.process_memory import memory_diagnostics_enabled, memory_event

memory_event("startup_begin")

from alpha_engine.config.settings import Settings, get_settings
from alpha_engine.services.live_cycle import LiveCycleResources, LiveCycleResult, run_live_cycle
from alpha_engine.services.recommendation_api import ApiReply, handle_recommendation_request
from alpha_engine.services.paper_trading import handle_paper_request

memory_event("startup_after_imports")


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "xgboost_v1.joblib"
METADATA = ROOT / "models" / "xgboost_v1_metadata.json"
CHECKSUM = ROOT / "models" / "xgboost_v1.sha256"
EXPECTED_MODEL_SHA256 = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"
MAX_REQUEST_BYTES = 2048
logger = logging.getLogger(__name__)


def verify_model_artifact() -> Path:
    """Fail closed if the deployed worker lacks the validated model bundle."""
    if not MODEL.is_file() or not METADATA.is_file() or not CHECKSUM.is_file():
        raise RuntimeError("Model bundle unavailable")
    declared = CHECKSUM.read_text(encoding="utf-8").strip()
    if declared != EXPECTED_MODEL_SHA256 or hashlib.sha256(MODEL.read_bytes()).hexdigest() != declared:
        raise RuntimeError("Model artifact checksum mismatch")
    return MODEL


async def _send_json(send: Callable, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body, separators=(",", ":"), allow_nan=False).encode("utf-8")
    await send({
        "type": "http.response.start", "status": status,
        "headers": [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(payload)).encode("ascii")),
            (b"cache-control", b"no-store"),
            (b"x-content-type-options", b"nosniff"),
        ],
    })
    await send({"type": "http.response.body", "body": payload})


def _single_header(scope: dict, name: bytes) -> bytes | None:
    values = [value for key, value in scope.get("headers", ()) if key.lower() == name]
    return values[0] if len(values) == 1 else None


async def _read_json(receive: Callable) -> object:
    body = bytearray()
    while True:
        event = await receive()
        if event["type"] == "http.disconnect":
            raise ValueError("Request disconnected")
        if event["type"] != "http.request":
            continue
        body.extend(event.get("body", b""))
        if len(body) > MAX_REQUEST_BYTES:
            raise OverflowError("Request too large")
        if not event.get("more_body", False):
            return json.loads(body)


class RenderApp:
    """ASGI transport with injectable dependencies for offline tests."""

    def __init__(
        self,
        *,
        settings_factory: Callable[[], Settings] = get_settings,
        cycle_runner: Callable[[Settings], LiveCycleResult] = run_live_cycle,
        recommendation_handler: Callable[[object, Settings], ApiReply] = handle_recommendation_request,
        paper_handler: Callable[[object, Settings], ApiReply] = handle_paper_request,
        model_verifier: Callable[[], Path] = verify_model_artifact,
        secrets: Mapping[str, str] = os.environ,
        muse_diagnostic: MuseDiagnostic | None = None,
        academic_demo=None,
    ) -> None:
        self._settings_factory = settings_factory
        self._cycle_runner = cycle_runner
        self._recommendation_handler = recommendation_handler
        self._paper_handler = paper_handler
        self._model_verifier = model_verifier
        self._secrets = secrets
        self._cycle_slot = BoundedSemaphore(1)
        self._resource_lock = Lock()
        self._live_resources: LiveCycleResources | None = None
        self._health_checks = 0
        self._muse_diagnostic = muse_diagnostic or MuseDiagnostic(secrets)
        from deploy.render.academic_demo import AcademicDemo
        self._academic_demo = academic_demo or AcademicDemo(secrets, self._settings)

    def _run_cycle(self, settings: Settings) -> LiveCycleResult:
        if self._cycle_runner is not run_live_cycle or settings.storage_provider != "supabase":
            return self._cycle_runner(settings)
        with self._resource_lock:
            if self._live_resources is None:
                self._live_resources = LiveCycleResources(settings)
            resources = self._live_resources
        return resources.run(settings)

    def _settings(self) -> Settings:
        settings = self._settings_factory()
        if settings.storage_provider == "supabase":
            if not settings.supabase_url or not settings.supabase_key.get_secret_value():
                raise RuntimeError("Cloud storage configuration missing")
        elif not (settings.storage_provider == "postgresql" and settings.app_env == "local"
                  and settings.database_url.get_secret_value()):
            raise RuntimeError("HTTP service requires configured cloud or local storage")
        model_path = self._model_verifier()
        return settings.model_copy(update={"model_path": model_path})

    def _ready(self) -> bool:
        if not self._secrets_valid():
            return False
        try:
            self._settings()
        except Exception as exc:
            logger.warning("Render readiness failed: %s", type(exc).__name__)
            return False
        return True

    def _secrets_valid(self) -> bool:
        cron = self._secrets.get("INTERNAL_CRON_SECRET", "")
        recommendation = self._secrets.get("RECOMMENDATION_API_SECRET", "")
        return (
            isinstance(cron, str) and isinstance(recommendation, str)
            and len(cron) >= 32 and len(recommendation) >= 32
            and not hmac.compare_digest(cron, recommendation)
        )

    def _authorized(self, scope: dict, secret_name: str) -> bool:
        expected = self._secrets.get(secret_name, "")
        presented = _single_header(scope, b"authorization")
        if not expected or presented is None:
            return False
        return hmac.compare_digest(presented, f"Bearer {expected}".encode("utf-8"))

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    memory_event("startup_ready")
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    with self._resource_lock:
                        if self._live_resources is not None:
                            self._live_resources.close()
                            self._live_resources = None
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return
        if scope["type"] != "http":
            return

        path = scope.get("path", "")
        method = scope.get("method", "")
        if path.startswith("/api/academic/"):
            status, body = await self._academic_demo.handle(scope, lambda: _read_json(receive))
            await _send_json(send, status, body)
            return
        if path == MUSE_DIAGNOSTIC_PATH:
            status, body = await self._muse_diagnostic.handle(scope, lambda: _read_json(receive))
            await _send_json(send, status, body)
            return
        allowed = {"/health": "GET", "/internal/run-cycle": "POST", "/api/recommendation": "POST",
                   "/api/demo-explanation": "POST", "/api/paper": "POST"}
        if path not in allowed:
            await _send_json(send, 404, {"status": "NOT_FOUND"})
            return
        if method != allowed[path]:
            await _send_json(send, 405, {"status": "METHOD_NOT_ALLOWED"})
            return
        if path == "/health":
            if memory_diagnostics_enabled():
                self._health_checks = (self._health_checks + 1) % 60
                if self._health_checks == 1:
                    memory_event("health_sample")
            if self._ready():
                await _send_json(send, 200, {"status": "READY", "scope": "http_process"})
            else:
                await _send_json(send, 503, {"status": "TEMPORARILY_UNAVAILABLE"})
            return

        secret_name = (
            "INTERNAL_CRON_SECRET" if path == "/internal/run-cycle"
            else "RECOMMENDATION_API_SECRET"
        )
        if not self._secrets_valid():
            await _send_json(send, 503, {"status": "TEMPORARILY_UNAVAILABLE"})
            return
        if not self._authorized(scope, secret_name):
            await _send_json(send, 401, {"status": "UNAUTHORIZED"})
            return

        try:
            settings = self._settings()
        except Exception as exc:
            logger.warning("Render configuration failed: %s", type(exc).__name__)
            await _send_json(send, 503, {"status": "TEMPORARILY_UNAVAILABLE"})
            return

        if path == "/api/demo-explanation" and not (
            settings.app_env == "local" and settings.storage_provider == "postgresql"
            and settings.local_demo_mode
        ):
            await _send_json(send, 404, {"status": "NOT_FOUND"})
            return

        if path == "/internal/run-cycle":
            if not self._cycle_slot.acquire(blocking=False):
                await _send_json(send, 409, {"status": "ALREADY_RUNNING"})
                return
            try:
                result = await asyncio.to_thread(self._run_cycle, settings)
                await _send_json(send, 200, {
                    "run_id": result.run_id, "status": result.status,
                    "prediction_id": result.prediction_id, "duration_ms": result.duration_ms,
                })
            except Exception as exc:
                logger.warning("Render cycle failed: %s", type(exc).__name__)
                await _send_json(send, 503, {"status": "TEMPORARILY_UNAVAILABLE"})
            finally:
                self._cycle_slot.release()
            return

        content_type = _single_header(scope, b"content-type")
        if content_type is None or content_type.split(b";", 1)[0].strip().lower() != b"application/json":
            await _send_json(send, 415, {"status": "UNSUPPORTED_MEDIA_TYPE"})
            return
        try:
            content_length = _single_header(scope, b"content-length")
            if content_length is not None and int(content_length) > MAX_REQUEST_BYTES:
                await _send_json(send, 413, {"status": "REQUEST_TOO_LARGE"})
                return
            payload = await _read_json(receive)
        except OverflowError:
            await _send_json(send, 413, {"status": "REQUEST_TOO_LARGE"})
            return
        except (ValueError, UnicodeDecodeError):
            await _send_json(send, 400, {"status": "INVALID_REQUEST"})
            return
        try:
            memory_event("paper_begin" if path == "/api/paper" else "recommendation_begin")
            if path == "/api/demo-explanation":
                from alpha_engine.services.local_demo import handle_demo_explanation
                reply = await asyncio.to_thread(handle_demo_explanation, payload, settings)
            elif path == "/api/paper":
                reply = await asyncio.to_thread(self._paper_handler, payload, settings)
            else:
                reply = await asyncio.to_thread(self._recommendation_handler, payload, settings)
            await _send_json(send, reply.status_code, reply.body)
        except Exception as exc:
            logger.warning("Render recommendation failed: %s", type(exc).__name__)
            await _send_json(send, 503, {"status": "TEMPORARILY_UNAVAILABLE"})
        finally:
            memory_event("paper_end" if path == "/api/paper" else "recommendation_end")


app = RenderApp()
