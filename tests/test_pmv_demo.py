"""PMV evidence from controlled fixtures only; all external transport is denied."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from alpha_engine.config.settings import Settings
from alpha_engine.data.binance import BinanceKlineClient
from alpha_engine.data.news_context import NewsContext
from alpha_engine.data.recommendation_response import RecommendationResult, RecommendationView
from alpha_engine.features import FEATURE_COLUMNS, build_feature_frame
from alpha_engine.services.live_cycle import LiveCycle
from alpha_engine.services.paper_policy import paper_decision
from alpha_engine.services.public_dashboard import (
    PublicCandleView, PublicDashboardData, PublicDashboardService, PublicLiveMetrics,
    PublicPredictionView,
)
from deploy.render.app import RenderApp, verify_model_artifact
from streamlit_app import ui
from test_live_cycle import Market, Predictions, Runs, Evaluations, candle
from test_marketaux_disabled_release import offline  # Existing autouse network guard.
from test_public_dashboard import Requests
from test_render_adapter import _request, SECRETS, CRON_SECRET

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("Home.py", "pages/1_Descriptivo.py", "pages/2_Predictivo.py", "pages/3_Prescriptivo.py")
MODEL_SHA = "8e23d25eeb7fcec82e1f71faf5d127ce4479318b9b5b1a76fc36b6691d147a8c"


@pytest.fixture
def selected(monkeypatch):
    value = Settings(_env_file=None, storage_provider="supabase",
                     supabase_url="https://synthetic.invalid", supabase_key="sb_publishable_synthetic",
                     prescriptive_backend="http", prescriptive_news_enabled=False)
    monkeypatch.setattr(ui, "get_settings", lambda: value)
    monkeypatch.setattr(ui, "live_evaluation_metrics", lambda *_: PublicLiveMetrics(
        evaluated=0, correct=0, incorrect=0, accuracy=None, timely=0, late=0, unknown_timing=0))
    def forbidden(*_args, **_kwargs):
        pytest.fail("Navigation must not request explanations, paper actions or news")
    monkeypatch.setattr("alpha_engine.services.recommendation_api.RecommendationApiClient.request", forbidden)
    monkeypatch.setattr("alpha_engine.services.paper_api.paper_request", forbidden)
    return value


def scenario(state="current", *, prob_up=.7):
    now = datetime.now(timezone.utc)
    base = now.replace(second=0, microsecond=0) - timedelta(minutes=120)
    rows = []
    for i in range(120):
        opened = base + timedelta(minutes=i)
        rows.append(PublicCandleView.from_record(candle(i).model_copy(update={
            "open_time": opened, "close_time": opened + timedelta(minutes=1, milliseconds=-1)})))
    prediction = PublicPredictionView(
        prediction_id="a" * 64, prediction_id_short="aaaaaa…aaaa", symbol="BTCUSDT", interval="1m",
        source_candle_open_time=rows[-1].open_time, predicted_class="UP" if prob_up > .5 else "DOWN",
        prob_up=prob_up, prob_down=1-prob_up, model_version="xgboost_v1", trained_at=None,
        generated_at=now - timedelta(seconds=2), valid_until=now + timedelta(seconds=50),
        timing_status="TIMELY", current_validity="ACTIVE")
    if state == "empty":
        rows, prediction = [], None
    elif state == "stale":
        rows = [row.model_copy(update={"open_time": row.open_time-timedelta(hours=1),
                                      "close_time": row.close_time-timedelta(hours=1)}) for row in rows]
    from alpha_engine.tools.market import get_technical_indicators
    return PublicDashboardData(
        mode="LIVE", storage="synthetic-memory", refreshed_at=now,
        system_state="NO_DATA" if not rows else "STALE" if state == "stale" else "CURRENT",
        candles=rows, latest_candle=rows[-1] if rows else None, latest_prediction=prediction,
        predictions=[prediction] if prediction else [],
        indicators=get_technical_indicators(rows) if rows else None)


def render(monkeypatch, data, relative, prior=None):
    monkeypatch.setattr(ui, "dashboard_data", lambda *_: data)
    monkeypatch.setattr(ui, "latest_prescriptive_prediction", lambda *_: data.latest_prediction)
    monkeypatch.setattr(ui, "latest_visible_prediction", lambda *_: data.latest_prediction)
    app = AppTest.from_file(ROOT / "streamlit_app" / relative, default_timeout=30)
    if prior is not None:
        app.session_state["recommendation_result"] = prior
    app.run()
    assert not app.exception and not app.error
    return app


def metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


@pytest.mark.parametrize("relative", PAGES)
@pytest.mark.parametrize("state", ["current", "empty", "stale"])
def test_four_pages_handle_current_missing_and_stale_data(monkeypatch, selected, relative, state):
    app = render(monkeypatch, scenario(state), relative)
    if state == "stale":
        assert any("STALE" in item.value for item in app.warning)
    if state == "empty":
        assert all(item.label not in {"P(SUBE)", "P(BAJA)", "Precio de cierre"} for item in app.metric)
    # These are rendered AppTest elements, not screenshots or production measurements.
    evidence = {"kind": "SYNTHETIC_APPTEST", "page": relative, "state": state,
                "metrics": [{"label": m.label, "value": m.value} for m in app.metric],
                "warnings": [w.value for w in app.warning],
                "info": [w.value for w in app.info], "exceptions": len(app.exception)}
    destination = ROOT / "runtime" / "pmv_ui_evidence"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{Path(relative).stem}_{state}.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


def test_descriptive_gap_is_not_reported_as_one_minute_return(monkeypatch, selected):
    data = scenario()
    data.candles.pop(-2)
    data.indicators = None
    app = render(monkeypatch, data, PAGES[1])
    assert metric(app, "Variación 1 minuto") == "N/A"
    assert any("consecutivas" in item.value for item in app.caption)
    assert any("historia continua" in item.value for item in app.info)


@pytest.mark.parametrize("prob_up,gate_met", [(.59, False), (.60, True), (.41, False), (.40, True)])
def test_predictive_gate_explains_abstention_without_changing_binary_class(
    monkeypatch, selected, prob_up, gate_met,
):
    app = render(monkeypatch, scenario(prob_up=prob_up), PAGES[2])
    assert metric(app, "P(SUBE)") == f"{prob_up:.1%}"
    assert metric(app, "P(BAJA)") == f"{1-prob_up:.1%}"
    assert metric(app, "Clase prevista") == ("UP" if prob_up > .5 else "DOWN")
    assert any("umbral ≥ 60%" in item.value for item in app.caption)
    assert any("Gate no alcanzado" in item.value for item in app.info) is not gate_met
    assert all(item.label != "Accuracy LIVE" for item in app.metric)  # No cohort, no invented KPI.


@pytest.mark.parametrize("context", ["stale", "fallback", "expired", "future", "empty", "aging"])
def test_prescriptive_never_presents_entry_from_unusable_data(monkeypatch, selected, context):
    data = scenario("stale" if context == "stale" else "empty" if context == "empty" else "current")
    if context == "fallback":
        data.mode = "FALLBACK"
    elif context == "expired":
        data.latest_prediction.valid_until = datetime.now(timezone.utc)-timedelta(seconds=1)
    elif context == "future":
        data.latest_prediction.generated_at = datetime.now(timezone.utc)+timedelta(seconds=20)
    elif context == "aging":
        # A fragment must notice aging even before the next full-page refresh.
        data.latest_candle.close_time = datetime.now(timezone.utc)-timedelta(minutes=3)
    app = render(monkeypatch, data, PAGES[3])
    assert metric(app, "Acción sugerida") == "WAIT"
    assert next(b for b in app.button if b.label == "Generar recomendación").disabled


def test_prescriptive_keeps_labelled_rules_when_llm_fails_and_hides_disabled_news(monkeypatch, selected):
    data = scenario()
    now = datetime.now(timezone.utc)
    prior = RecommendationResult(
        request_id=uuid4(), prediction_id=data.latest_prediction.prediction_id, status="PROVIDER_ERROR",
        requested_at=now, generated_at=None, finished_at=now, valid_until=data.latest_prediction.valid_until,
        timing_status=None, llm_latency_ms=None, total_latency_ms=0, cached=False, actionable=False,
        news_context=NewsContext(provider="MARKETAUX", status="AVAILABLE", fetched_at=now,
                                 btc_sentiment_score=.99))
    app = render(monkeypatch, data, PAGES[3], prior)
    assert metric(app, "Acción sugerida") == "BUY"
    assert any("Origen: política determinista, sin LLM" in item.value for item in app.caption)
    assert any("no evaluado" in item.value for item in app.caption)
    assert any("Noticias desactivadas" in item.value for item in app.info)
    assert all("Sentimiento" not in item.label for item in app.metric)
    assert not any(item.value == "Explicación LLM" for item in app.subheader)


def test_existing_api_risk_and_explanation_origin_are_visible(monkeypatch, selected):
    data = scenario()
    now = datetime.now(timezone.utc)
    view = RecommendationView(prediction_id=data.latest_prediction.prediction_id, candidate_signal="BUY",
        llm_signal="BUY", final_signal="BUY", risk_level="MEDIUM", llm_model="synthetic-explainer",
        prompt_version="synthetic", summary="CONTROLLED TEST EXPLANATION")
    prior = RecommendationResult(request_id=uuid4(), prediction_id=view.prediction_id,
        status="SUCCESS", requested_at=now, generated_at=now, finished_at=now,
        valid_until=data.latest_prediction.valid_until, timing_status="TIMELY", llm_latency_ms=1,
        total_latency_ms=2, cached=False, actionable=True, recommendation=view)
    app = render(monkeypatch, data, PAGES[3], prior)
    assert metric(app, "Riesgo evaluado") == "MEDIUM"
    assert any("synthetic-explainer" in item.value for item in app.caption)


def test_controlled_binance_to_real_v1_api_dashboard_and_rules(monkeypatch, selected):
    assert selected.model_confidence_threshold == .60
    assert FEATURE_COLUMNS == ("return_1m", "rsi_14", "ema_20", "macd", "macd_signal",
                               "macd_histogram", "rolling_volatility", "volume_change")
    path = verify_model_artifact()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA
    market, predictions, runs, evaluations = Market(), Predictions(), Runs(), Evaluations()
    base = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=120)
    rows = [candle(i).model_copy(update={
        "open_time": base+timedelta(minutes=i),
        "close_time": base+timedelta(minutes=i+1, milliseconds=-1)}) for i in range(121)]
    raw = [[int(c.open_time.timestamp()*1000), str(c.open), str(c.high), str(c.low),
            str(c.close), str(c.volume), int(c.close_time.timestamp()*1000)] for c in rows]
    calls = []
    def binance(request):
        calls.append(request.url.path)
        assert request.url.path == "/api/v3/klines"
        return httpx.Response(200, json=raw)
    now = base + timedelta(minutes=120, seconds=5)
    with httpx.Client(transport=httpx.MockTransport(binance), base_url="https://synthetic.invalid") as http:
        source = BinanceKlineClient(client=http)
        with LiveCycle(selected, source, market, predictions, runs=runs, evaluations=evaluations,
                       clock=lambda: now) as cycle:
            app = RenderApp(settings_factory=lambda: selected, cycle_runner=lambda _: cycle.run(now=now),
                            secrets=SECRETS)
            assert _request(app, "GET", "/health").status_code == 200
            reply = _request(app, "POST", "/internal/run-cycle",
                             headers={"Authorization": "Bearer " + CRON_SECRET})
            assert reply.status_code == 200 and reply.json()["status"] == "SUCCESS"
            assert len(market.rows) == 120 and len(predictions.rows) == 1
            assert rows[120].open_time not in [c.open_time for c in market.rows.values()]
            service = PublicDashboardService(selected, market, predictions, runs, evaluations, Requests([]))
            dashboard = service.dashboard(now=now, section="all")
            reloaded = PublicDashboardData.model_validate_json(dashboard.model_dump_json())
            pred = reloaded.latest_prediction
            assert pred.prediction_id == reply.json()["prediction_id"]
            assert pred.model_version == "xgboost_v1" and pred.validity_at(now) == "ACTIVE"
            assert pred.prob_up + pred.prob_down == pytest.approx(1)
            frame = build_feature_frame(sorted(market.rows.values(), key=lambda c: c.open_time))
            for feature in FEATURE_COLUMNS:
                assert getattr(reloaded.indicators, feature) == pytest.approx(frame.iloc[-1][feature])
            decision = paper_decision(pred, has_long=False, threshold=.60, now=now)
            assert decision.action == ("BUY" if pred.prob_up >= .60 else "WAIT")
            assert dashboard.live_metrics.evaluated == 0 and dashboard.live_metrics.accuracy is None
            for relative in PAGES:
                shown = render(monkeypatch, reloaded, relative)
                if relative in {PAGES[2], PAGES[3]}:
                    assert metric(shown, "P(SUBE)") == f"{pred.prob_up:.1%}"
                elif relative == PAGES[0]:
                    assert any(f"P(SUBE): {pred.prob_up:.1%}" in item.value for item in shown.caption)
    assert calls and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA
