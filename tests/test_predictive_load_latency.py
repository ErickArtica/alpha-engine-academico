"""Focused navigation regression checks; no real storage/provider I/O."""
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from alpha_engine.config.settings import Settings
from alpha_engine.services.live_evaluation import LiveEvaluationService
from alpha_engine.services.public_dashboard import (
    DashboardUnavailableError, PublicDashboardData, PublicPredictionView, PublicLiveMetrics,
)
from streamlit_app import ui

ROOT=Path(__file__).resolve().parents[1]
PREDICTIVE=ROOT/'streamlit_app/pages/2_Predictivo.py'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    connect=socket.socket.connect
    def local_event_loop_only(sock,address):
        # Windows asyncio creates its internal socketpair over loopback.
        if isinstance(address,tuple) and address[0] in {'127.0.0.1','::1'}:
            return connect(sock,address)
        raise AssertionError('EXTERNAL_NETWORK_FORBIDDEN')
    def denied(*a,**kw):
        raise AssertionError('REAL_NETWORK_OR_PROVIDER_FORBIDDEN')
    monkeypatch.setattr(socket.socket,'connect',local_event_loop_only)
    monkeypatch.setattr(socket,'getaddrinfo',denied)
    monkeypatch.setattr(httpx.HTTPTransport,'handle_request',denied)
    monkeypatch.setattr('alpha_engine.services.recommendation_api.RecommendationApiClient.request',denied)
    monkeypatch.setattr('alpha_engine.services.paper_api.paper_request',denied)


@pytest.fixture
def data(monkeypatch):
    now=datetime.now(timezone.utc)
    pred=PublicPredictionView(prediction_id='f'*64,prediction_id_short='ffffff…ffff',
        symbol='BTCUSDT',interval='1m',source_candle_open_time=now-timedelta(minutes=1),
        predicted_class='UP',prob_up=.7,prob_down=.3,model_version='xgboost_v1',
        trained_at=None,generated_at=now,valid_until=now+timedelta(seconds=59),
        timing_status='TIMELY',current_validity='ACTIVE')
    settings=Settings(_env_file=None,app_env='local',storage_provider='supabase',
        supabase_url='https://synthetic.invalid',supabase_key='sb_publishable_synthetic',
        prescriptive_backend='http',prescriptive_news_enabled=False)
    dashboard=PublicDashboardData(mode='LIVE',storage='offline-memory',refreshed_at=now,
        system_state='NO_DATA',latest_prediction=pred,predictions=[pred])
    monkeypatch.setattr(ui,'get_settings',lambda:settings)
    monkeypatch.setattr(ui,'dashboard_data',lambda *args:dashboard)
    monkeypatch.setattr(ui,'latest_visible_prediction',lambda *args:pred)
    monkeypatch.setattr(ui,'latest_prescriptive_prediction',lambda *args:pred)
    return dashboard


def metric_value():
    return PublicLiveMetrics(evaluated=1,correct=1,incorrect=0,accuracy=1.,timely=1,
        late=0,unknown_timing=0,scored_predictions=1,always_down_accuracy=0.)


def test_predictive_navigation_never_fetches_full_metric_cohort(monkeypatch,data):
    calls=[]
    def slow_query(*args):
        calls.append(args)
        raise AssertionError('ACCUMULATED_METRICS_BLOCKS_INITIAL_NAVIGATION')
    monkeypatch.setattr(ui,'live_evaluation_metrics',slow_query)
    app=AppTest.from_file(PREDICTIVE,default_timeout=15).run()
    assert not app.exception
    assert calls==[]
    assert any(m.label=='P(SUBE)' for m in app.metric)
    assert len(app.dataframe)==1  # Recent prediction history still renders.


def test_metrics_query_only_on_explicit_request(monkeypatch,data):
    calls=[]
    monkeypatch.setattr(ui,'live_evaluation_metrics',lambda *a:(calls.append(a) or metric_value()))
    app=AppTest.from_file(PREDICTIVE,default_timeout=15).run()
    assert not app.exception and calls==[]
    next(b for b in app.button if b.label=='Consultar desempeño LIVE').click().run()
    assert not app.exception and len(calls)==1
    assert any(m.label=='Accuracy LIVE' and m.value=='100.0%' for m in app.metric)


def test_optional_metrics_failure_keeps_prediction_and_history(monkeypatch,data):
    def failed(*a):raise DashboardUnavailableError('controlled timeout')
    monkeypatch.setattr(ui,'live_evaluation_metrics',failed)
    app=AppTest.from_file(PREDICTIVE,default_timeout=15).run()
    next(b for b in app.button if b.label=='Consultar desempeño LIVE').click().run()
    assert not app.exception and len(app.dataframe)==1
    assert any(m.label=='P(SUBE)' for m in app.metric)
    assert any('métricas LIVE no están disponibles' in x.value for x in app.info)


def test_prescriptive_predictive_switches_have_no_ui_import_keyerror(monkeypatch,data):
    monkeypatch.setattr(ui,'live_evaluation_metrics',lambda *a:metric_value())
    for relative in ['pages/3_Prescriptivo.py','pages/2_Predictivo.py']*2:
        app=AppTest.from_file(ROOT/'streamlit_app'/relative,default_timeout=15).run()
        assert not app.exception


def test_predictive_prescriptive_imports_in_clean_process():
    script = r'''
import ast, pathlib, socket, sys
sys.path.insert(0,sys.argv[1])
import httpx
def denied(*a,**kw):raise AssertionError('EXTERNAL_NETWORK_FORBIDDEN')
socket.socket.connect=denied
socket.getaddrinfo=denied
httpx.HTTPTransport.handle_request=denied
for name in ('2_Predictivo.py','3_Prescriptivo.py'):
    path=pathlib.Path(sys.argv[1])/'streamlit_app/pages'/name
    tree=ast.parse(path.read_text(encoding='utf-8'))
    imports=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))],type_ignores=[])
    exec(compile(imports,str(path),'exec'),{})
    assert 'streamlit_app.ui' in sys.modules
assert pathlib.Path(sys.modules['streamlit_app.ui'].__file__).resolve()==pathlib.Path(sys.argv[1])/'streamlit_app/ui.py'
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(ROOT)],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_metric_cohort_pages_all_rows_before_model_filter():
    records=[];predictions=[]
    for i in range(1001):
        direction='UP' if i%2 else 'DOWN'
        records.append(SimpleNamespace(prediction_id=str(i),correct=True,observed_class=direction,
            target_open_time=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=i)))
        predictions.append(SimpleNamespace(prediction_id=str(i),timing_status='TIMELY',
            predicted_class=direction,prob_up=.7 if i%2 else .3,
            model_version='other' if i<10 else 'xgboost_v1'))
    class Paged:
        def __init__(self,rows):self.rows=rows;self.calls=[]
        def list_recent(self,symbol,limit,offset):
            self.calls.append((limit,offset));return self.rows[offset:offset+limit]
    evaluations=Paged(records);predictions=Paged(predictions)
    result=LiveEvaluationService.metrics_for(predictions,evaluations,'BTCUSDT','xgboost_v1')
    assert result.evaluated==991
    assert evaluations.calls==predictions.calls==[(500,0),(500,500),(500,1000)]


def test_local_timing_is_opt_in_and_logs_no_credentials(monkeypatch,caplog):
    monkeypatch.delenv('ALPHA_UI_TIMINGS',raising=False)
    with caplog.at_level(logging.INFO,logger='alpha_engine.ui_timing'):
        with ui.section_timing('predictive.dashboard'):pass
    assert not caplog.records
    monkeypatch.setenv('ALPHA_UI_TIMINGS','1')
    with caplog.at_level(logging.INFO,logger='alpha_engine.ui_timing'):
        with ui.section_timing('predictive.dashboard'):pass
    assert 'section=predictive.dashboard' in caplog.text and 'elapsed_ms=' in caplog.text
    assert 'supabase' not in caplog.text.lower() and 'authorization' not in caplog.text.lower()
