"""Smoke tests for every public frontend page in explicit offline fallback mode."""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from alpha_engine.config.settings import Settings, get_settings
from alpha_engine.data.recommendation_response import RecommendationResult
from alpha_engine.services.public_dashboard import (
    PublicCandleView, PublicDashboardData, PublicPredictionView, PublicRecommendationObservability,
)
from streamlit_app import ui
from streamlit_app.ui import _load_dashboard


ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "streamlit_app/Home.py",
    "streamlit_app/pages/1_Descriptivo.py",
    "streamlit_app/pages/2_Predictivo.py",
    "streamlit_app/pages/3_Prescriptivo.py",
    "streamlit_app/pages/4_Evaluacion.py",
)
PACKAGE_PAGES = PAGES + ("streamlit_app/pages/5_Simulacion.py",)


@pytest.mark.parametrize("relative", PAGES)
def test_streamlit_page_renders_explicit_fallback_without_live_side_effects(monkeypatch, relative):
    if not (ROOT / "public_demo" / "data" / "snapshot.json").is_file():
        pytest.skip("Snapshot histórico privado no incluido en Git")
    monkeypatch.setenv("DASHBOARD_FALLBACK_ENABLED", "true")
    get_settings.cache_clear()
    _load_dashboard.clear()
    try:
        app = AppTest.from_file(ROOT / relative, default_timeout=30).run()
        assert not app.exception
        assert not app.error
        assert any("MODO FALLBACK" in warning.value for warning in app.warning)
    finally:
        _load_dashboard.clear()
        get_settings.cache_clear()


@pytest.mark.parametrize("relative", PACKAGE_PAGES)
def test_streamlit_page_finds_local_package_without_editable_install(tmp_path, relative):
    """Cloud installs frontend dependencies but does not pip-install this repository."""
    script = """
import sys
from pathlib import Path

page = Path(sys.argv[1]).resolve()
source = page.read_text(encoding="utf-8")
bootstrap = source.split("import streamlit as st", 1)[0]
exec(compile(bootstrap, str(page), "exec"), {"__file__": str(page)})
import alpha_engine
import streamlit_app
assert Path(alpha_engine.__file__).resolve() == page.parents[1 if page.name == "Home.py" else 2] / "alpha_engine" / "__init__.py"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(ROOT / relative)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_live_auto_refresh_requests_one_full_rerun_after_a_minute(monkeypatch):
    ticks = []
    clock = [100.0]
    session = {}

    def fragment(*, run_every):
        assert run_every == 60
        def decorate(func):
            ticks.append(func)
            return func
        return decorate

    def rerun():
        raise RuntimeError("full rerun requested")

    monkeypatch.setattr(ui.st, "session_state", session)
    monkeypatch.setattr(ui.st, "fragment", fragment)
    monkeypatch.setattr(ui.st, "rerun", rerun)
    monkeypatch.setattr(ui.time, "monotonic", lambda: clock[0])

    ui.auto_refresh_live()
    clock[0] = 157.0
    ticks[0]()
    clock[0] = 160.0
    with pytest.raises(RuntimeError, match="full rerun requested"):
        ticks[0]()


@pytest.mark.parametrize("relative", [
    "streamlit_app/Home.py",
    "streamlit_app/pages/2_Predictivo.py",
    "streamlit_app/pages/3_Prescriptivo.py",
])
def test_latest_prediction_uses_five_second_fragment_without_heavy_queries(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    panels = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in {"_latest_prediction_panel", "_request_panel"}]
    assert len(panels) == 1
    panel = panels[0]
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "fragment"
        and any(keyword.arg == "run_every" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "5s" for keyword in decorator.keywords)
        for decorator in panel.decorator_list
    )
    names = {node.id for node in ast.walk(panel) if isinstance(node, ast.Name)}
    assert not names.intersection({"live_evaluation_metrics", "dashboard_data", "LiveCycle",
                                   "paper_request"})


def test_fast_prediction_read_uses_one_record_and_keeps_metrics_cache(monkeypatch):
    prediction = object()
    calls = []
    monkeypatch.setattr(ui, "_load_latest_prediction", lambda *args: calls.append(args) or prediction)
    settings = Settings(_env_file=None, storage_provider="supabase")
    live = PublicDashboardData(mode="LIVE", storage="supabase", refreshed_at=datetime.now(timezone.utc),
                               system_state="CURRENT")
    fallback = live.model_copy(update={"mode": "FALLBACK", "latest_prediction": prediction})
    assert ui.latest_visible_prediction(settings, live) is prediction
    assert calls == [(settings.storage_provider, settings.symbol, settings.interval)]
    assert ui.latest_visible_prediction(settings, fallback) is prediction
    assert len(calls) == 1

    source = (ROOT / "streamlit_app/ui.py").read_text(encoding="utf-8")
    assert "@st.cache_data(ttl=4, show_spinner=False)\ndef _load_latest_prediction" in source
    assert "@st.cache_data(ttl=55, show_spinner=False)\ndef _load_live_metrics" in source
    assert "@st.fragment(run_every=60)" in source


def test_cached_dashboard_recomputes_freshness_and_validity_on_navigation(monkeypatch):
    now = datetime.now(timezone.utc)
    candle = PublicCandleView(
        symbol="BTCUSDT", interval="1m", open_time=now - timedelta(seconds=181),
        close_time=now - timedelta(seconds=121), open=100, high=101,
        low=99, close=100, volume=10,
    )
    prediction = PublicPredictionView(
        prediction_id="a" * 64, prediction_id_short="aaaaaa…aaaa",
        symbol="BTCUSDT", interval="1m", source_candle_open_time=candle.open_time,
        predicted_class="UP", prob_up=0.7, prob_down=0.3, model_version="xgboost_v1",
        trained_at=None, generated_at=now - timedelta(minutes=1),
        valid_until=now - timedelta(seconds=1), timing_status="TIMELY",
        current_validity="ACTIVE",
    )
    cached = PublicDashboardData(
        mode="LIVE", storage="supabase", refreshed_at=now - timedelta(minutes=1),
        system_state="CURRENT", latest_candle=candle, latest_prediction=prediction,
    )
    monkeypatch.setattr(ui, "_load_dashboard", lambda *_args: cached)

    shown = ui.dashboard_data(Settings(_env_file=None, max_candle_age_seconds=120))

    assert shown.system_state == "STALE"
    assert shown.latest_prediction.current_validity == "EXPIRED"
    assert cached.system_state == "CURRENT"
    assert cached.latest_prediction.current_validity == "ACTIVE"


@pytest.mark.parametrize("offset_seconds, cached_validity, disabled", [
    (-28, "ACTIVE", True),
    (30, "EXPIRED", False),
])
def test_prescriptive_button_uses_deadline_instead_of_cached_validity(
    monkeypatch, offset_seconds, cached_validity, disabled,
):
    now = datetime.now(timezone.utc)
    prediction = PublicPredictionView(
        prediction_id="a" * 64, prediction_id_short="aaaaaa…aaaa",
        symbol="BTCUSDT", interval="1m", source_candle_open_time=now - timedelta(minutes=1),
        predicted_class="UP", prob_up=0.7, prob_down=0.3, model_version="xgboost_v1",
        trained_at=None, generated_at=now - timedelta(seconds=2),
        valid_until=now + timedelta(seconds=offset_seconds), timing_status="TIMELY",
        current_validity=cached_validity,
    )
    dashboard = PublicDashboardData(
        mode="LIVE", storage="supabase", refreshed_at=now - timedelta(minutes=1),
        system_state="CURRENT", latest_prediction=prediction,
    )
    monkeypatch.setattr(ui, "dashboard_data", lambda _settings, _section: dashboard)
    monkeypatch.setattr(ui, "latest_prescriptive_prediction", lambda _settings: prediction)
    monkeypatch.setattr(
        "alpha_engine.services.recommendation_api.RecommendationApiClient.request",
        lambda *_args: pytest.fail("El refresco visual no debe llamar al LLM ni a noticias"),
    )
    monkeypatch.setenv("PRESCRIPTIVE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_PROVIDER", "influxdb3")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(ROOT / "streamlit_app/pages/3_Prescriptivo.py", default_timeout=30).run()
        assert not app.exception
        action = next(button for button in app.button if button.label.startswith("Generar recomendación"))
        assert action.disabled is disabled
        if disabled:
            assert any("Umbral experimental cumplido" in item.value for item in app.info)
        else:
            assert any("ANÁLISIS PRESCRIPTIVO HABILITADO" in item.value for item in app.success)
    finally:
        get_settings.cache_clear()


def test_prescriptive_click_refreshes_audit_and_keeps_request_result(monkeypatch):
    now = datetime.now(timezone.utc)
    prediction = PublicPredictionView(
        prediction_id="b" * 64, prediction_id_short="bbbbbb…bbbb",
        symbol="BTCUSDT", interval="1m", source_candle_open_time=now - timedelta(minutes=1),
        predicted_class="UP", prob_up=0.7, prob_down=0.3, model_version="xgboost_v1",
        trained_at=None, generated_at=now - timedelta(seconds=2),
        valid_until=now + timedelta(seconds=45), timing_status="TIMELY",
        current_validity="ACTIVE",
    )
    old_audit = PublicRecommendationObservability(
        last_attempt_at=None, last_attempt_status="GUARDRAIL_REJECTED", last_success_at=None,
        timely_count=0, late_count=0, timeout_count=0, provider_error_count=0,
        average_llm_latency_ms=None,
    )
    dashboard = PublicDashboardData(
        mode="LIVE", storage="supabase", refreshed_at=now, system_state="CURRENT",
        latest_prediction=prediction, recommendation_observability=old_audit,
    )
    state = {"audited": False}
    new_prediction = prediction.model_copy(update={
        "prediction_id": "c" * 64, "prediction_id_short": "cccccc…cccc",
    })

    def dashboard_data(_settings, _section):
        if not state["audited"]:
            return dashboard
        new_audit = old_audit.model_copy(update={
            "last_attempt_at": now, "last_attempt_status": "EXPIRED_BEFORE_REQUEST",
        })
        return dashboard.model_copy(update={
            "latest_prediction": new_prediction, "recommendation_observability": new_audit,
        })

    def request(_client, prediction_id):
        assert prediction_id == prediction.prediction_id
        state["audited"] = True
        return RecommendationResult(
            request_id=uuid4(), prediction_id=prediction_id,
            status="EXPIRED_BEFORE_REQUEST", requested_at=now, generated_at=None,
            finished_at=now, valid_until=now - timedelta(seconds=1),
            timing_status=None, llm_latency_ms=None, total_latency_ms=333,
            cached=False, actionable=False,
        )

    monkeypatch.setattr(ui, "dashboard_data", dashboard_data)
    monkeypatch.setattr(
        ui, "latest_prescriptive_prediction",
        lambda _settings: prediction if not state["audited"] else new_prediction,
    )
    monkeypatch.setattr("alpha_engine.services.recommendation_api.RecommendationApiClient.request", request)
    monkeypatch.setenv("STORAGE_PROVIDER", "supabase")
    monkeypatch.setenv("PRESCRIPTIVE_BACKEND", "http")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb_publishable_test")
    monkeypatch.setenv("RECOMMENDATION_API_URL", "https://example.invalid/api/recommendation")
    monkeypatch.setenv("RECOMMENDATION_API_KEY", "test-only-secret")
    monkeypatch.setenv("RECOMMENDATION_API_AUTH_SCHEME", "bearer")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(ROOT / "streamlit_app/pages/3_Prescriptivo.py", default_timeout=30).run()
        action = next(button for button in app.button if button.label.startswith("Generar recomendación"))
        assert not action.disabled
        app = action.click().run()
        assert state["audited"] and not app.exception
        assert any(metric.label == "Estado" and metric.value == "EXPIRED_BEFORE_REQUEST" for metric in app.metric)
        assert any(metric.label == "Último intento" and metric.value == "EXPIRED_BEFORE_REQUEST" for metric in app.metric)
        assert app.session_state["recommendation_result"].prediction_id == prediction.prediction_id
        assert any("otra predicción" in item.value for item in app.info)
    finally:
        get_settings.cache_clear()


def test_prescriptive_labels_previous_guardrail_result_without_llm(monkeypatch):
    now = datetime.now(timezone.utc)
    prediction = PublicPredictionView(
        prediction_id="d" * 64, prediction_id_short="dddddd…dddd",
        symbol="BTCUSDT", interval="1m", source_candle_open_time=now - timedelta(minutes=1),
        predicted_class="UP", prob_up=0.7, prob_down=0.3, model_version="xgboost_v1",
        trained_at=None, generated_at=now - timedelta(seconds=2),
        valid_until=now + timedelta(seconds=45), timing_status="TIMELY",
        current_validity="ACTIVE",
    )
    dashboard = PublicDashboardData(
        mode="LIVE", storage="supabase", refreshed_at=now,
        system_state="CURRENT", latest_prediction=prediction,
    )
    prior = RecommendationResult(
        request_id=uuid4(), prediction_id="e" * 64,
        status="GUARDRAIL_REJECTED", requested_at=now - timedelta(minutes=30),
        generated_at=None, finished_at=now - timedelta(minutes=30),
        valid_until=now - timedelta(minutes=29), timing_status=None,
        llm_latency_ms=None, total_latency_ms=504,
        cached=False, actionable=False,
    )
    monkeypatch.setattr(ui, "dashboard_data", lambda _settings, _section: dashboard)
    monkeypatch.setattr(ui, "latest_prescriptive_prediction", lambda _settings: prediction)
    monkeypatch.setattr(
        "alpha_engine.services.recommendation_api.RecommendationApiClient.request",
        lambda *_args: pytest.fail("El refresco visual no debe solicitar una recomendación"),
    )
    monkeypatch.setenv("STORAGE_PROVIDER", "influxdb3")
    monkeypatch.setenv("PRESCRIPTIVE_BACKEND", "local")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(ROOT / "streamlit_app/pages/3_Prescriptivo.py", default_timeout=30)
        app.session_state["recommendation_result"] = prior
        app = app.run()
        assert not app.exception
        assert any("otra predicción" in item.value for item in app.info)
        assert any("No se llamó al LLM" in item.value for item in app.info)
        app = app.run()
        assert not app.exception
        assert app.session_state["recommendation_result"].request_id == prior.request_id
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("prob_up,technical_class", [(.547, "UP"), (.453, "DOWN")])
def test_prescriptive_low_gate_shows_no_action_without_turning_it_into_hold(
    monkeypatch, prob_up, technical_class,
):
    now = datetime.now(timezone.utc)
    prediction = PublicPredictionView(
        prediction_id="f" * 64, prediction_id_short="ffffffff…ffffffff",
        symbol="BTCUSDT", interval="1m", source_candle_open_time=now - timedelta(minutes=1),
        predicted_class=technical_class, prob_up=prob_up, prob_down=1 - prob_up,
        model_version="xgboost_v1", trained_at=None, generated_at=now - timedelta(seconds=2),
        valid_until=now + timedelta(seconds=45), timing_status="TIMELY",
        current_validity="ACTIVE",
    )
    dashboard = PublicDashboardData(
        mode="LIVE", storage="supabase", refreshed_at=now,
        system_state="CURRENT", latest_prediction=prediction,
    )
    calls = []

    def request(_client, prediction_id):
        calls.append(prediction_id)
        return RecommendationResult(
            request_id=uuid4(), prediction_id=prediction_id,
            status="GUARDRAIL_REJECTED", requested_at=now,
            generated_at=None, finished_at=now, valid_until=prediction.valid_until,
            timing_status=None, llm_latency_ms=None, total_latency_ms=1,
            cached=False, actionable=False, guardrails=["LOW_MODEL_CONFIDENCE"],
        )

    monkeypatch.setattr(ui, "dashboard_data", lambda _settings, _section: dashboard)
    monkeypatch.setattr(ui, "latest_prescriptive_prediction", lambda _settings: prediction)
    monkeypatch.setattr("alpha_engine.services.recommendation_api.RecommendationApiClient.request", request)
    monkeypatch.setenv("STORAGE_PROVIDER", "supabase")
    monkeypatch.setenv("PRESCRIPTIVE_BACKEND", "http")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sb_publishable_test")
    monkeypatch.setenv("RECOMMENDATION_API_URL", "https://example.invalid/api/recommendation")
    monkeypatch.setenv("RECOMMENDATION_API_KEY", "test-only-secret")
    monkeypatch.setenv("RECOMMENDATION_API_AUTH_SCHEME", "bearer")
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(ROOT / "streamlit_app/pages/3_Prescriptivo.py", default_timeout=30).run()
        assert not app.exception and not calls
        assert any(metric.label == "Dirección" and metric.value == technical_class for metric in app.metric)
        assert any(metric.label == "P(SUBE)" and metric.value == f"{prob_up:.1%}" for metric in app.metric)
        assert any(metric.label == "P(BAJA)" and metric.value == f"{1 - prob_up:.1%}" for metric in app.metric)
        assert any("ESPERAR / NO ENTRAR" in item.value for item in app.info)
        action = next(button for button in app.button if button.label == "Generar recomendación")
        assert not action.disabled
        app = action.click().run()
        assert not app.exception and calls == [prediction.prediction_id]
        assert any(metric.label == "Estado" and metric.value == "ESPERAR / SIN ACCIÓN" for metric in app.metric)
        assert all(metric.value != "HOLD" for metric in app.metric)
        assert any("No se consultaron noticias" in item.value for item in app.info)
    finally:
        get_settings.cache_clear()
