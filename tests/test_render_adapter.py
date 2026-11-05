"""Render's HTTP edge is narrow, authenticated and independent of real cloud services."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.services.recommendation_api import ApiReply
from deploy.render.app import MODEL, RenderApp, verify_model_artifact


PREDICTION_ID = "a" * 64
CRON_SECRET = "c" * 48
LLM_SECRET = "l" * 48
SECRETS = {"INTERNAL_CRON_SECRET": CRON_SECRET, "RECOMMENDATION_API_SECRET": LLM_SECRET}


def _app(*, cycle_runner=None, recommendation_handler=None, paper_handler=None,
         secrets=None, verifier=None):
    settings = Settings(
        _env_file=None, storage_provider="supabase",
        supabase_url="https://example.supabase.co", supabase_key="test-server-key",
    )
    return RenderApp(
        settings_factory=lambda: settings,
        cycle_runner=cycle_runner or (lambda _: SimpleNamespace(
            run_id="run-1", status="SUCCESS", prediction_id=PREDICTION_ID,
            duration_ms=124.5,
        )),
        recommendation_handler=recommendation_handler or (
            lambda payload, _: ApiReply(200, {"status": "SUCCESS", "prediction_id": payload["prediction_id"]})
        ),
        paper_handler=paper_handler or (
            lambda payload, _: ApiReply(200, {"status": "OK", "op": payload["op"]})
        ),
        model_verifier=verifier or (lambda: Path("models/xgboost_v1.joblib")),
        secrets=SECRETS if secrets is None else secrets,
    )


def _request(app, method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_is_process_readiness_only_and_fails_closed():
    reply = _request(_app(), "GET", "/health")
    assert reply.status_code == 200
    assert reply.json() == {"status": "READY", "scope": "http_process"}
    assert reply.headers["cache-control"] == "no-store"
    unavailable = _request(_app(secrets={}), "GET", "/health")
    assert unavailable.status_code == 503
    assert _request(_app(secrets={"INTERNAL_CRON_SECRET": CRON_SECRET,
                                  "RECOMMENDATION_API_SECRET": CRON_SECRET}),
                    "GET", "/health").status_code == 503
    assert _request(_app(verifier=lambda: (_ for _ in ()).throw(RuntimeError("private"))),
                    "GET", "/health").status_code == 503


def test_only_three_routes_and_methods_are_exposed():
    app = _app()
    assert _request(app, "GET", "/anything").status_code == 404
    assert _request(app, "GET", "/internal/run-cycle").status_code == 405
    assert _request(app, "GET", "/api/recommendation").status_code == 405


def test_endpoints_require_distinct_bearer_secrets_before_work():
    called = []
    app = _app(
        cycle_runner=lambda _: called.append("cycle"),
        recommendation_handler=lambda *_: called.append("llm"),
    )
    for path in ("/internal/run-cycle", "/api/recommendation"):
        assert _request(app, "POST", path).status_code == 401
        assert _request(app, "POST", path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert _request(app, "POST", "/api/recommendation", headers={
        "Authorization": f"Bearer {CRON_SECRET}"}).status_code == 401
    assert _request(app, "POST", "/internal/run-cycle", headers={
        "Authorization": f"Bearer {LLM_SECRET}"}).status_code == 401
    assert _request(app, "POST", "/internal/run-cycle", headers=[
        ("Authorization", f"Bearer {CRON_SECRET}"),
        ("Authorization", f"Bearer {CRON_SECRET}"),
    ]).status_code == 401
    assert called == []


def test_authorized_cycle_calls_common_use_case_and_sanitizes_output():
    seen = []

    def run(settings):
        seen.append(settings.model_path)
        return SimpleNamespace(
            run_id="run-1", status="SUCCESS", prediction_id=PREDICTION_ID,
            duration_ms=124.5, secret="never public",
        )

    reply = _request(_app(cycle_runner=run, verifier=lambda: MODEL), "POST",
                     "/internal/run-cycle", headers={"Authorization": f"Bearer {CRON_SECRET}"})
    assert reply.status_code == 200
    assert reply.json() == {
        "run_id": "run-1", "status": "SUCCESS",
        "prediction_id": PREDICTION_ID, "duration_ms": 124.5,
    }
    assert seen == [MODEL]


def test_cycle_reentry_and_errors_are_controlled():
    app = _app()
    app._cycle_slot.acquire()
    try:
        busy = _request(app, "POST", "/internal/run-cycle",
                        headers={"Authorization": f"Bearer {CRON_SECRET}"})
        assert busy.status_code == 409 and busy.json() == {"status": "ALREADY_RUNNING"}
    finally:
        app._cycle_slot.release()

    def fail(_):
        raise RuntimeError("private storage detail")

    reply = _request(_app(cycle_runner=fail), "POST", "/internal/run-cycle",
                     headers={"Authorization": f"Bearer {CRON_SECRET}"})
    assert reply.status_code == 503
    assert "private" not in reply.text


def test_recommendation_restricts_json_and_returns_sanitized_result():
    seen = []

    def handle(payload, settings):
        seen.append((payload, settings.storage_provider))
        if not isinstance(payload, dict) or set(payload) != {"prediction_id"}:
            return ApiReply(400, {"status": "INVALID_REQUEST"})
        return ApiReply(200, {"status": "SUCCESS", "prediction_id": payload["prediction_id"]})

    app = _app(recommendation_handler=handle)
    auth = {"Authorization": f"Bearer {LLM_SECRET}"}
    assert _request(app, "POST", "/api/recommendation", headers=auth,
                    content="text").status_code == 415
    assert _request(app, "POST", "/api/recommendation", headers={**auth, "Content-Type": "application/json"},
                    content="not json").status_code == 400
    assert _request(app, "POST", "/api/recommendation", headers=auth,
                    json={"prediction_id": PREDICTION_ID, "prompt": "override"}).status_code == 400
    assert _request(app, "POST", "/api/recommendation", headers=auth,
                    json={"prediction_id": PREDICTION_ID, "padding": "x" * 3000}).status_code == 413
    good = _request(app, "POST", "/api/recommendation", headers=auth,
                    json={"prediction_id": PREDICTION_ID})
    assert good.status_code == 200 and good.json()["prediction_id"] == PREDICTION_ID
    assert seen[-1] == ({"prediction_id": PREDICTION_ID}, "supabase")


def test_recommendation_exception_never_reaches_http_response():
    def fail(*_):
        raise RuntimeError("private provider detail")

    reply = _request(_app(recommendation_handler=fail), "POST", "/api/recommendation",
                     headers={"Authorization": f"Bearer {LLM_SECRET}"},
                     json={"prediction_id": PREDICTION_ID})
    assert reply.status_code == 503
    assert "private" not in reply.text


def test_paper_route_requires_backend_secret_and_has_narrow_payload():
    seen = []
    app = _app(paper_handler=lambda payload, _: (
        seen.append(payload) or ApiReply(200, {"status": "OK"})))
    assert _request(app, "POST", "/api/paper", json={"op": "read"}).status_code == 401
    assert _request(app, "GET", "/api/paper").status_code == 405
    reply = _request(app, "POST", "/api/paper",
                     headers={"Authorization": f"Bearer {LLM_SECRET}"},
                     json={"op": "read", "portfolio_id": "00000000-0000-4000-8000-000000000101"})
    assert reply.status_code == 200 and reply.json() == {"status": "OK"}
    assert seen == [{"op": "read", "portfolio_id": "00000000-0000-4000-8000-000000000101"}]


def test_worker_model_bundle_checksum_matches_versioned_artifact():
    assert verify_model_artifact() == MODEL
