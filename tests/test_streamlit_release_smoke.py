"""Six real page entrypoints with synthetic data and external services forbidden."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import socket

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from alpha_engine.config.settings import Settings
from alpha_engine.services.public_dashboard import (
    PublicCandleView, PublicDashboardData, PublicLiveMetrics, PublicPredictionView,
)
from streamlit_app import ui

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ('Home.py', 'Alpha Engine', 'Estado del sistema'),
    ('pages/1_Descriptivo.py', 'Descriptivo', 'Indicadores técnicos'),
    ('pages/2_Predictivo.py', 'Predictivo', 'Historial LIVE reciente'),
    ('pages/3_Prescriptivo.py', 'Prescriptivo', 'Posición y acción determinista'),
    ('pages/4_Evaluacion.py', 'Evaluación', 'Calidad por clase'),
    ('pages/5_Simulacion.py', 'Simulación', 'Crear o recuperar cartera'),
]


@pytest.fixture
def isolated_dashboard(monkeypatch):
    connect = socket.socket.connect

    def only_loopback(sock, address):
        if isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}:
            return connect(sock, address)
        raise AssertionError('EXTERNAL_NETWORK_FORBIDDEN')

    def forbidden(*args, **kwargs):
        raise AssertionError('PRODUCTION_SERVICE_OR_PROVIDER_FORBIDDEN')

    monkeypatch.setattr(socket.socket, 'connect', only_loopback)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', forbidden)
    monkeypatch.setattr('alpha_engine.services.recommendation_api.RecommendationApiClient.request', forbidden)
    monkeypatch.setattr('alpha_engine.services.paper_api.paper_request', forbidden)
    now = datetime.now(timezone.utc)
    source = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
    candles = [PublicCandleView(
        symbol='BTCUSDT', interval='1m',
        open_time=source - timedelta(minutes=19-i),
        close_time=source - timedelta(minutes=18-i, milliseconds=1),
        open=100+i, high=102+i, low=99+i, close=101+i, volume=1+i,
    ) for i in range(20)]
    prediction = PublicPredictionView(
        prediction_id='f'*64, prediction_id_short='ffffff…ffff',
        symbol='BTCUSDT', interval='1m', source_candle_open_time=source,
        predicted_class='UP', prob_up=.7, prob_down=.3, model_version='xgboost_v1',
        trained_at=None, generated_at=now, valid_until=now+timedelta(seconds=59),
        timing_status='TIMELY', current_validity='ACTIVE',
    )
    settings = Settings(
        _env_file=None, app_env='local', storage_provider='supabase',
        supabase_url='https://synthetic.invalid', supabase_key='sb_publishable_synthetic',
        prescriptive_backend='http', prescriptive_news_enabled=False,
    )
    data = PublicDashboardData(
        mode='LIVE', storage='synthetic-memory', refreshed_at=now,
        system_state='CURRENT', latest_candle=candles[-1], candles=candles,
        latest_prediction=prediction, predictions=[prediction],
    )
    metric_calls = []

    def metrics(*args):
        metric_calls.append(args)
        return PublicLiveMetrics(
            evaluated=1, correct=1, incorrect=0, accuracy=1.,
            timely=1, late=0, unknown_timing=0, scored_predictions=1,
            source_start=source, source_end=source,
        )

    monkeypatch.setattr(ui, 'get_settings', lambda: settings)
    monkeypatch.setattr(ui, 'dashboard_data', lambda *args: data)
    monkeypatch.setattr(ui, 'latest_visible_prediction', lambda *args: prediction)
    monkeypatch.setattr(ui, 'latest_prescriptive_prediction', lambda *args: prediction)
    monkeypatch.setattr(ui, 'live_evaluation_metrics', metrics)
    return metric_calls


@pytest.mark.parametrize('relative,title,landmark', PAGES)
def test_six_pages_render_without_external_operations(isolated_dashboard, relative, title, landmark):
    app = AppTest.from_file(ROOT/'streamlit_app'/relative, default_timeout=15).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == title
    assert any(x.value == landmark for x in app.subheader)
    if title in {'Alpha Engine', 'Descriptivo'}:
        assert len(app.get('plotly_chart')) >= 1
    if title == 'Predictivo':
        assert not isolated_dashboard
        assert len(app.dataframe) == 1
        assert any(b.label == 'Consultar desempeño LIVE' for b in app.button)
    if title == 'Evaluación':
        assert len(isolated_dashboard) == 1
    if title == 'Simulación':
        assert any(b.label == 'Crear cartera virtual' for b in app.button)
