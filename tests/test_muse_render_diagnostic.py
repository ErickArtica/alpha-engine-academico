"""Deterministic offline coverage; no provider, Render, or database connections."""

import asyncio
import hashlib
import json
import os
import socket
import sys
from threading import local
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from deploy.render.app import RenderApp
from deploy.render import muse_diagnostic as diagnostic
from deploy.render.muse_diagnostic import (
    CONTEXT_SHA256, MODEL, PATH, PROMPT_SHA256, MuseDiagnostic, WorkerReport,
    canonical, child_environment, pinned_inputs, run_worker,
)
from deploy.render.muse_diagnostic_worker import MAX_BODY_BYTES, infer_once


NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
TOKEN = "diagnostic_" + "x" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SECRET = "fake-nvidia-secret-never-log"
VALID = {
    "signal": "HOLD", "confidence": 0.0, "risk_level": "MEDIUM",
    "summary": "La predicción está vencida; corresponde esperar.",
    "key_factors": ["Contexto histórico sin operación"],
    "risks": ["Sin noticias disponibles"], "warning": "Prueba académica sin operación.",
}


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    connect = socket.socket.connect
    socketpair = socket.socketpair
    internal = local()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Real networking is forbidden in diagnostic tests")
    def connect_internal(sock, address):
        # Windows asyncio implements its internal socketpair over loopback.
        # HTTP transports and create_connection remain forbidden unconditionally.
        if (getattr(internal, "socketpair", False) and isinstance(address, tuple)
                and address[0] in ("127.0.0.1", "::1")):
            return connect(sock, address)
        return forbidden()
    def internal_pair(*args, **kwargs):
        internal.socketpair = True
        try:
            return socketpair(*args, **kwargs)
        finally:
            internal.socketpair = False
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", connect_internal)
    monkeypatch.setattr(socket, "socketpair", internal_pair)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)


def environment(**updates):
    env = {
        "MUSE_DIAGNOSTIC_ENABLED": "true", "MUSE_DIAGNOSTIC_TOKEN": TOKEN,
        "MUSE_DIAGNOSTIC_STARTS_AT": NOW.isoformat(),
        "MUSE_DIAGNOSTIC_EXPIRES_AT": (NOW + timedelta(minutes=5)).isoformat(),
        "NVIDIA_API_KEY": SECRET, "RENDER": "true",
        "RENDER_SERVICE_ID": "srv-offline-test", "RENDER_INSTANCE_ID": "offline-instance",
        "RENDER_GIT_COMMIT": "a" * 40,
        "INTERNAL_CRON_SECRET": "c" * 48, "RECOMMENDATION_API_SECRET": "r" * 48,
    }
    env.update(updates)
    return env


def application(env=None, *, clock=lambda: NOW, runner=None):
    async def fake_runner(_env):
        return WorkerReport(status="PASS", stage="complete", network_attempts=1)
    env = environment() if env is None else env
    controller = MuseDiagnostic(env, clock=clock, runner=runner or fake_runner)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Production factory/handler was accessed")
    app = RenderApp(secrets=env, settings_factory=forbidden, cycle_runner=forbidden,
                    recommendation_handler=forbidden, paper_handler=forbidden,
                    model_verifier=forbidden, muse_diagnostic=controller)
    return app


async def request(app, method="GET", **kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://offline.test") as client:
        return await client.request(method, PATH, headers=kwargs.pop("headers", AUTH), **kwargs)


def call(app, method="GET", **kwargs):
    return asyncio.run(request(app, method, **kwargs))


def ticket(app):
    reply = call(app)
    assert reply.status_code == 200
    assert reply.json()["status"] == "PREPARED"
    return {name: reply.json()[name] for name in ("run_id", "nonce")}


@pytest.mark.parametrize("updates", [
    {"MUSE_DIAGNOSTIC_ENABLED": "false"}, {"MUSE_DIAGNOSTIC_ENABLED": ""},
    {"MUSE_DIAGNOSTIC_TOKEN": "short"}, {"MUSE_DIAGNOSTIC_TOKEN": "ñ" * 40},
    {"MUSE_DIAGNOSTIC_TOKEN": "c" * 48}, {"MUSE_DIAGNOSTIC_TOKEN": "r" * 48},
    {"NVIDIA_API_KEY": ""}, {"RENDER": "false"}, {"RENDER_GIT_COMMIT": "not-a-revision"},
    {"RENDER_INSTANCE_ID": ""}, {"RENDER_SERVICE_ID": "private secret"},
    {"MUSE_DIAGNOSTIC_STARTS_AT": "2026-10-22T12:00:00"},
    {"MUSE_DIAGNOSTIC_STARTS_AT": "2026-10-22T07:00:00-05:00"},
    {"MUSE_DIAGNOSTIC_EXPIRES_AT": (NOW + timedelta(minutes=6)).isoformat()},
    {"MUSE_DIAGNOSTIC_EXPIRES_AT": NOW.isoformat()},
])
def test_disabled_invalid_configuration_never_discloses_details(updates):
    app = application(environment(**updates))
    for method in ("GET", "POST", "DELETE"):
        response = call(app, method, json={"private": "ignored"})
        assert response.status_code == 404
        assert response.json() == {"status": "NOT_FOUND"}
        assert response.headers["cache-control"] == "no-store"


def test_default_is_disabled_without_touching_production():
    assert call(application({})).status_code == 404


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer " + "c" * 48},
    {"Authorization": "Bearer " + "r" * 48}, {"Authorization": TOKEN},
    [("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {TOKEN}")],
])
def test_exclusive_authentication(headers):
    assert call(application(), headers=headers).json() == {"status": "NOT_FOUND"}


def test_window_boundaries_and_query_strings():
    app = application(clock=lambda: NOW - timedelta(microseconds=1))
    assert call(app).status_code == 404
    app = application(clock=lambda: NOW + timedelta(minutes=5))
    assert call(app).status_code == 404
    assert call(application(), params={"token": TOKEN}).status_code == 404


def test_prepare_is_idempotent_without_inference_and_has_no_production_dependencies():
    calls = []
    async def runner(env):
        calls.append(env)
        return WorkerReport()
    app = application(runner=runner)
    first = call(app)
    assert first.json() == call(app).json()
    assert first.json()["prompt_sha256"] == PROMPT_SHA256
    assert first.json()["context_sha256"] == CONTEXT_SHA256
    assert calls == []


def test_input_and_prompt_hashes_are_pinned_and_source_context_is_non_operational():
    context, prompt = pinned_inputs()
    assert hashlib.sha256(canonical(context)).hexdigest() == CONTEXT_SHA256
    assert hashlib.sha256(prompt.encode()).hexdigest() == PROMPT_SHA256
    assert context["validation_context"]["operational_recommendation"] is False
    assert context["prediction"]["prob_down"] == 0.606224000453949
    assert context["decision_policy"]["action"] == "WAIT"
    assert "diagnostic_run_id" not in context


def test_tampered_input_fails_before_preparation(tmp_path, monkeypatch):
    fixture = tmp_path / "tampered.json"
    fixture.write_text('{"prediction": {"prob_up": 1}}')
    monkeypatch.setattr(diagnostic, "FIXTURE", fixture)
    response = call(application())
    assert response.status_code == 503
    assert response.json() == {"status": "DIAGNOSTIC_UNAVAILABLE"}


def test_one_use_and_receipt_retrieval_excludes_secrets():
    calls = []
    async def runner(env):
        calls.append(env)
        return WorkerReport(status="FAILED", stage="provider", network_attempts=1, http_status=429)
    app = application(runner=runner)
    prepared = ticket(app)
    reply = call(app, "POST", json=prepared)
    assert reply.status_code == 200
    receipt = reply.json()["receipt"]
    assert receipt["result"]["http_status"] == 429
    assert receipt["diagnostic_only"] is True
    assert receipt["operational_recommendation"] is False
    assert receipt["origin"]["revision"] == "a" * 40
    assert receipt["source_evidence_sha256"] == diagnostic.SOURCE_SHA256
    assert call(app, "POST", json=prepared).status_code == 409
    assert call(app).json()["receipt"] == receipt
    assert len(calls) == 1
    for private in (TOKEN, SECRET, prepared["nonce"], "srv-offline-test", "offline-instance"):
        assert private not in reply.text


def test_concurrent_requests_consume_before_runner_and_never_overlap():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def runner(env):
            calls.append(env)
            entered.set()
            await release.wait()
            return WorkerReport()
        app = application(runner=runner)
        prepared = (await request(app)).json()
        payload = {key: prepared[key] for key in ("run_id", "nonce")}
        first = asyncio.create_task(request(app, "POST", json=payload))
        await asyncio.wait_for(entered.wait(), timeout=2)
        second = await request(app, "POST", json=payload)
        assert second.status_code == 409
        assert (await request(app)).json()["status"] == "RUNNING"
        release.set()
        assert (await first).status_code == 200
        assert len(calls) == 1
    asyncio.run(scenario())


def test_restart_invalidates_old_ticket_but_is_not_a_durable_global_ledger():
    old = application()
    payload = ticket(old)
    assert call(old, "POST", json=payload).status_code == 200
    restarted = application()
    assert call(restarted, "POST", json=payload).status_code == 409
    new = ticket(restarted)
    assert new != payload
    assert call(restarted, "POST", json=payload).status_code == 409
    # A newly prepared ticket after restart is intentionally a new permission.
    assert call(restarted, "POST", json=new).status_code == 200


def test_time_is_rechecked_after_body_and_near_deadline_does_not_consume():
    clock = [NOW]
    app = application(clock=lambda: clock[0])
    prepared = ticket(app)
    clock[0] = NOW + timedelta(minutes=4, seconds=20)
    assert call(app, "POST", json=prepared).status_code == 404
    clock[0] = NOW + timedelta(minutes=5)
    assert call(app, "POST", json=prepared).status_code == 404


def test_expiry_during_body_read_prevents_dispatch():
    async def scenario():
        clock = [NOW]
        calls = []
        async def runner(env):
            calls.append(env)
            return WorkerReport()
        controller = MuseDiagnostic(environment(), clock=lambda: clock[0], runner=runner)
        scope = {"method": "GET", "headers": [(b"authorization", f"Bearer {TOKEN}".encode())]}
        async def read_payload():
            raise AssertionError("GET must not read a body")
        _, prepared = await controller.handle(scope, read_payload)
        scope.update(method="POST", headers=scope["headers"] + [(b"content-type", b"application/json")])
        async def delayed_body():
            clock[0] = NOW + timedelta(minutes=5)
            return {key: prepared[key] for key in ("run_id", "nonce")}
        assert (await controller.handle(scope, delayed_body))[0] == 404
        assert calls == []
    asyncio.run(scenario())


@pytest.mark.parametrize("payload", [{}, {"run_id": "x", "nonce": "x", "prompt": "override"},
                                     {"run_id": 1, "nonce": "x"}, {"run_id": "ñ", "nonce": "x"}])
def test_payload_cannot_override_inference(payload):
    app = application()
    prepared = ticket(app)
    assert call(app, "POST", json=payload).status_code == 400
    assert call(app, "POST", json=prepared).status_code == 200


def test_body_limits_and_no_content_type():
    app = application()
    ticket(app)
    assert call(app, "POST", content="{}", headers=AUTH).status_code == 415
    assert call(app, "POST", json={"padding": "x" * 3000}).status_code == 400
    assert call(app, "POST", content="invalid", headers={**AUTH, "Content-Type": "application/json"}).status_code == 400


def test_worker_failure_consumes_permission_and_exception_text_is_suppressed(caplog):
    async def runner(env):
        raise RuntimeError(SECRET + TOKEN)
    app = application(runner=runner)
    prepared = ticket(app)
    response = call(app, "POST", json=prepared)
    assert response.json()["receipt"]["result"]["status"] == "WORKER_ERROR"
    assert call(app, "POST", json=prepared).status_code == 409
    assert SECRET not in response.text + caplog.text
    assert TOKEN not in response.text + caplog.text


def test_child_environment_is_an_allowlist():
    env = environment(SUPABASE_KEY="private-db", PRESCRIPTIVE_NEWS_ENABLED="true",
                      HTTP_PROXY="https://private-proxy", PYTHONPATH="private-path")
    child = child_environment(env)
    assert child["NVIDIA_API_KEY"] == SECRET
    for forbidden in ("SUPABASE_KEY", "PRESCRIPTIVE_NEWS_ENABLED", "HTTP_PROXY", "PYTHONPATH",
                      "MUSE_DIAGNOSTIC_TOKEN", "RECOMMENDATION_API_SECRET", "INTERNAL_CRON_SECRET"):
        assert forbidden not in child


def local_child_environment():
    # Preserve only OS requirements, never a developer's real API credential.
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
           if name in os.environ}
    return child_environment({**env, "NVIDIA_API_KEY": SECRET})


def run_inference(content=None, *, status=200, model=MODEL, reason="stop", envelope=None):
    requests = []
    body = envelope if envelope is not None else {
        "model": model, "choices": [{"finish_reason": reason,
        "message": {"content": json.dumps(VALID if content is None else content, ensure_ascii=False),
                    "reasoning_content": SECRET}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                  "unknown_sensitive_field": SECRET},
    }
    def handler(req):
        requests.append(req)
        return httpx.Response(status, json=body, headers={"x-private-header": SECRET})
    return infer_once(SECRET, transport=httpx.MockTransport(handler)), requests


def test_real_provider_and_parser_are_used_with_exact_messages_and_no_writes(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Production storage/service construction is forbidden")
    from alpha_engine.repositories import factory
    for name in dir(factory):
        if name.startswith("create_"):
            monkeypatch.setattr(factory, name, forbidden)
    from alpha_engine.services.recommendation_use_case import RecommendationUseCase
    from alpha_engine.ai.recommendation_service import RecommendationService
    from alpha_engine.services.prescriptive_news import PrescriptiveNewsContextService
    from alpha_engine.observability.telemetry import TelemetryStore
    for target in (RecommendationUseCase, RecommendationService, PrescriptiveNewsContextService, TelemetryStore):
        monkeypatch.setattr(target, "__init__", forbidden)
    monkeypatch.setenv("PRESCRIPTIVE_NEWS_ENABLED", "true")
    monkeypatch.setenv("SUPABASE_KEY", "not-used")
    report, requests = run_inference()
    assert len(requests) == 1
    assert report.status == "PASS" and report.technical_pass and report.semantic_contract_pass
    assert report.parser_executed and report.parser_accepted
    assert report.raw_schema_valid and report.parser_modified_content is False
    body = json.loads(requests[0].content)
    assert body["model"] == MODEL and body["reasoning_effort"] == "low"
    assert body["temperature"] == 0.95 and body["max_tokens"] == 1024 and body["stream"] is False
    assert hashlib.sha256(body["messages"][0]["content"].encode()).hexdigest() == PROMPT_SHA256
    assert hashlib.sha256(body["messages"][1]["content"].encode()).hexdigest() == CONTEXT_SHA256
    assert requests[0].extensions["timeout"] == dict(connect=30, read=30, write=30, pool=30)
    assert report.usage.total_tokens == 150
    assert SECRET not in report.model_dump_json()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_provider_error_never_retries_or_leaks(status, caplog):
    report, requests = run_inference(status=status, envelope={"error": {"message": SECRET}})
    assert len(requests) == 1 and report.network_attempts == 1
    assert report.http_status == status and report.status == "FAILED"
    assert not report.parser_executed
    assert SECRET not in report.model_dump_json() + caplog.text


def test_mismatched_model_and_finish_reason_are_not_accepted():
    report, _ = run_inference(model=SECRET)
    assert report.reported_model == "OTHER" and report.model_matches is False
    assert report.status == "FAILED" and SECRET not in report.model_dump_json()
    report, _ = run_inference(reason="length")
    assert report.status == "FAILED" and report.finish_reason == "length"
    assert not report.parser_executed


@pytest.mark.parametrize("content", ["not-json", {**VALID, "extra": "not allowed"},
                                     {**VALID, "confidence": 2}, {**VALID, "risk_level": "INVALID"}])
def test_invalid_content_runs_existing_parser_and_is_rejected(content):
    report, requests = run_inference(content)
    assert len(requests) == 1
    assert report.parser_executed and report.parser_accepted is False
    assert report.status == "FAILED"


def test_parser_sanitization_and_semantic_mismatches_are_explicit():
    report, _ = run_inference({**VALID, "summary": "Ganará 99 por ciento."})
    assert report.parser_accepted and not report.raw_schema_valid
    assert report.parser_modified_content is True
    assert not report.semantic_checks.narrative_without_digits
    assert report.status == "PARTIAL"
    report, _ = run_inference({**VALID, "signal": "BUY", "risk_level": "LOW", "risks": ["Riesgo técnico"]})
    assert report.technical_pass and report.status == "PARTIAL"
    assert not report.semantic_checks.signal_matches_action
    assert not report.semantic_checks.risk_matches_context
    assert not report.semantic_checks.mentions_news_absence


def test_arbitrary_generated_text_headers_and_reasoning_are_never_in_report():
    report, _ = run_inference({**VALID, "warning": SECRET})
    assert report.parser_accepted
    raw = report.model_dump_json()
    assert SECRET not in raw and "reasoning_content" not in raw and "Authorization" not in raw


def test_oversized_and_non_json_provider_bodies_are_bounded():
    report = infer_once(SECRET, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"x" * (MAX_BODY_BYTES + 1))))
    assert report.status == "FAILED" and report.network_attempts == 1
    report = infer_once(SECRET, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"not json")))
    assert report.envelope_json_valid is False and not report.parser_executed


def test_http_timeout_is_not_retried():
    calls = []
    def timeout(req):
        calls.append(req)
        raise httpx.ReadTimeout(SECRET, request=req)
    report = infer_once(SECRET, transport=httpx.MockTransport(timeout))
    assert len(calls) == 1 and report.status == "FAILED"
    assert SECRET not in report.model_dump_json()


def test_no_key_fails_without_transport():
    calls = []
    report = infer_once("", transport=httpx.MockTransport(lambda req: calls.append(req)))
    assert report.network_attempts == 0 and calls == []


def test_worker_module_is_safe_without_explicit_execute_permission():
    report = asyncio.run(run_worker(local_child_environment(), command=[
        sys.executable, "-B", "-m", "deploy.render.muse_diagnostic_worker",
    ]))
    assert report.network_attempts == 0 and report.stage == "preflight"
    assert report.child_reaped is True


def test_total_timeout_really_kills_and_reaps_child(monkeypatch):
    processes = []
    create = asyncio.create_subprocess_exec
    async def track(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", track)
    report = asyncio.run(run_worker(local_child_environment(), timeout_seconds=0.15,
        command=[sys.executable, "-B", "-c", "import threading; threading.Event().wait()"] ))
    assert report.status == "TIMEOUT" and report.network_attempts is None
    assert report.child_reaped is True
    assert len(processes) == 1 and processes[0].returncode is not None


def test_cancellation_kills_and_reaps_and_consumes_permission(monkeypatch):
    async def scenario():
        processes = []
        entered = asyncio.Event()
        create = asyncio.create_subprocess_exec
        async def track(*args, **kwargs):
            process = await create(*args, **kwargs)
            processes.append(process)
            entered.set()
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", track)
        async def runner(env):
            return await run_worker(env, command=[sys.executable, "-B", "-c",
                                                "import threading; threading.Event().wait()"])
        app = application(runner=runner)
        prepare = (await request(app)).json()
        payload = {key: prepare[key] for key in ("run_id", "nonce")}
        task = asyncio.create_task(request(app, "POST", json=payload))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None
        assert (await request(app, "POST", json=payload)).status_code == 409
        result = (await request(app)).json()["receipt"]["result"]
        assert result["status"] == "CANCELLED" and result["child_reaped"] is True
    asyncio.run(scenario())


def test_invalid_child_protocol_is_suppressed():
    report = asyncio.run(run_worker(child_environment({}), command=[
        sys.executable, "-B", "-c", "print('private-exception-and-response')",
    ]))
    assert report.status == "WORKER_ERROR" and report.child_reaped
    assert "private-exception" not in report.model_dump_json()


def test_child_watchdog_is_armed_and_cancelled_without_running_inference(monkeypatch, capsys):
    from deploy.render import muse_diagnostic_worker as worker
    events = []
    class FakeWatchdog:
        def __init__(self, seconds, callback):
            events.append(("timeout", seconds))
            self.callback = callback
        def start(self):
            events.append("armed")
        def cancel(self):
            events.append("cancelled")
    def offline_infer(key):
        assert "armed" in events and "cancelled" not in events
        assert key == SECRET
        return WorkerReport()
    monkeypatch.setattr(worker, "Timer", FakeWatchdog)
    monkeypatch.setattr(worker, "infer_once", offline_infer)
    monkeypatch.setattr(sys, "argv", ["worker", "--execute"])
    monkeypatch.setenv("MUSE_DIAGNOSTIC_CHILD", "1")
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)
    previous = worker.logging.root.manager.disable
    try:
        worker.main()
    finally:
        worker.logging.disable(previous)
    assert events == [("timeout", diagnostic.TOTAL_TIMEOUT_SECONDS), "armed", "cancelled"]
    assert json.loads(capsys.readouterr().out)["network_attempts"] == 0


def test_child_watchdog_terminates_even_without_parent_timeout():
    # Replace inference BEFORE main; only a local Event.wait runs in this child.
    code = (
        "import sys,threading; from deploy.render import muse_diagnostic_worker as w; "
        "w.TOTAL_TIMEOUT_SECONDS=0.1; w.infer_once=lambda _:threading.Event().wait(); "
        "sys.argv=['worker','--execute']; w.main()"
    )
    report = asyncio.run(run_worker(local_child_environment(), timeout_seconds=5,
                                   command=[sys.executable, "-B", "-c", code]))
    assert report.status == "TIMEOUT" and report.child_reaped is True
    assert report.network_attempts is None


def test_timeout_result_stays_consumed():
    calls = []
    async def runner(env):
        calls.append(env)
        return WorkerReport(status="TIMEOUT", stage="timeout", network_attempts=None, child_reaped=True)
    app = application(runner=runner)
    prepared = ticket(app)
    assert call(app, "POST", json=prepared).json()["receipt"]["result"]["status"] == "TIMEOUT"
    assert call(app, "POST", json=prepared).status_code == 409
    assert len(calls) == 1
