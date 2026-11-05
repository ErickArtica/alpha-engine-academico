"""Release-only evidence: no network, news schema, provider or migration needed."""
import asyncio
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType,SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from alpha_engine.config.settings import Settings
from alpha_engine.data.news_context import NewsContext
from alpha_engine.data.recommendation_response import RecommendationResult
from alpha_engine.services.prescriptive_news import PrescriptiveNewsContextService
from alpha_engine.services.recommendation_use_case import RecommendationUseCase
from alpha_engine.services.public_dashboard import PublicDashboardData
from alpha_engine.services.prediction import PredictionService
from deploy.render.app import RenderApp,verify_model_artifact
from test_prediction_service import FakeMarket,FakePredictions,candles
from test_render_adapter import _request,SECRETS,CRON_SECRET,LLM_SECRET

ROOT=Path(__file__).resolve().parents[1]
NOW=datetime(2026,11,2,22,tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args,**kwargs): raise AssertionError('EXTERNAL_IO_FORBIDDEN')
    original_connect=socket.socket.connect
    def internal_socketpair_only(sock,address):
        # Windows asyncio implements its private self-pipe using loopback TCP.
        if isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
            return original_connect(sock,address)
        return denied()
    monkeypatch.setattr(socket.socket,'connect',internal_socketpair_only)
    monkeypatch.setattr(socket,'create_connection',denied)
    monkeypatch.setattr(socket,'getaddrinfo',denied)
    monkeypatch.setattr(httpx.HTTPTransport,'handle_request',denied)
    monkeypatch.setenv('PRESCRIPTIVE_NEWS_ENABLED','false')
    monkeypatch.setenv('MARKETAUX_API_TOKEN','synthetic-present-but-disabled')


def settings(**kwargs):
    return Settings(_env_file=None,storage_provider='supabase',supabase_url='https://synthetic.invalid',
        supabase_key='synthetic-only',prescriptive_news_enabled=False,**kwargs)


def result(news=None):
    return RecommendationResult(request_id=uuid4(),prediction_id='a'*64,status='GUARDRAIL_REJECTED',
        requested_at=NOW,generated_at=None,finished_at=NOW,valid_until=NOW+timedelta(minutes=1),
        timing_status=None,llm_latency_ms=None,total_latency_ms=0,cached=False,actionable=False,news_context=news)


@pytest.mark.parametrize('provider',['alpha_vantage','marketaux'])
def test_disabled_service_does_not_open_news_storage_even_with_token(provider):
    def forbidden(*args,**kwargs): pytest.fail('NEWS_STORAGE_OR_PROVIDER_OPENED')
    s=settings(prescriptive_news_provider=provider,alpha_vantage_api_key='synthetic-present')
    context=PrescriptiveNewsContextService(s,repository_factory=forbidden,client=SimpleNamespace(fetch=forbidden)).get_context()
    assert context.status=='UNAVAILABLE' and context.reason=='NOT_CONFIGURED'


@pytest.mark.parametrize('provider',['alpha_vantage','marketaux'])
def test_real_use_case_factory_does_not_construct_news_service(monkeypatch,provider):
    import alpha_engine.services.recommendation_use_case as uc
    import alpha_engine.ai.providers.factory as providers
    def forbidden(*args,**kwargs): pytest.fail('OPTIONAL_NEWS_IN_DISABLED_RELEASE')
    for name in ('create_market_repository','create_prediction_repository','create_recommendation_repository','create_recommendation_request_repository'):
        monkeypatch.setattr(uc,name,lambda _:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(providers,'create_llm_provider',lambda _:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(uc,'PrescriptiveNewsContextService',forbidden)
    with RecommendationUseCase.from_settings(settings(prescriptive_news_provider=provider)) as service:
        assert service._news_service is None


def test_backend_startup_health_and_real_frozen_model_prediction_without_news_tables():
    market=FakeMarket(candles()); predictions=FakePredictions()
    service=PredictionService(market,predictions,verify_model_artifact())
    def cycle(_settings):
        value=service.predict_latest('BTCUSDT')
        assert value.model_version=='xgboost_v1'
        assert 0<=value.prob_up<=1 and 0<=value.prob_down<=1
        return SimpleNamespace(run_id='synthetic',status='SUCCESS',prediction_id=value.prediction_id,duration_ms=1.)
    app=RenderApp(settings_factory=lambda:settings(prescriptive_news_provider='marketaux'),
                  cycle_runner=cycle,secrets=SECRETS)
    events=iter([{'type':'lifespan.startup'},{'type':'lifespan.shutdown'}]); replies=[]
    async def receive(): return next(events)
    async def send(event): replies.append(event)
    asyncio.run(app({'type':'lifespan'},receive,send))
    assert [x['type'] for x in replies]==['lifespan.startup.complete','lifespan.shutdown.complete']
    assert _request(app,'GET','/health').status_code==200
    reply=_request(app,'POST','/internal/run-cycle',headers={'Authorization':'Bearer '+CRON_SECRET})
    assert reply.status_code==200 and reply.json()['status']=='SUCCESS'
    assert predictions.saves==1
    # Existing protected routes reject unauthenticated calls without provider I/O.
    for path in ('/api/recommendation','/api/paper'):
        assert _request(app,'POST',path,json={}).status_code==401


def legacy_type(monkeypatch):
    name='_release_legacy_dto'
    path=ROOT/'tests/fixtures/legacy_prescriptive_dto.py'
    source=path.read_text(encoding='utf-8')
    module=ModuleType(name); monkeypatch.setitem(sys.modules,name,module)
    exec(compile(source,path,'exec'),module.__dict__)
    return module


def test_staggered_release_dto_both_directions_with_news_disabled(monkeypatch):
    legacy=legacy_type(monkeypatch)
    new=result().model_dump(mode='json')
    assert legacy.RecommendationResult.model_validate(new).news_context is None
    old=legacy.RecommendationResult.model_validate(new).model_dump(mode='json')
    assert RecommendationResult.model_validate(old).news_context is None
    alpha=legacy.NewsContext(status='NO_NEWS').model_dump(mode='json')
    assert NewsContext.model_validate(alpha).recent_articles==[]
    # Activation cannot precede frontend upgrade: legacy enum rejects Marketaux.
    with pytest.raises(ValueError): legacy.NewsContext(provider='MARKETAUX',status='UNAVAILABLE')


def test_authenticated_recommendation_gate_uses_existing_schema_without_llm_or_news(monkeypatch):
    import alpha_engine.services.recommendation_use_case as uc
    import alpha_engine.ai.providers.factory as providers
    from alpha_engine.services.recommendation_api import handle_recommendation_request
    from test_recommendation_use_case import prediction,Predictions,Recommendations,Requests,NOW as request_time
    class Audit(Requests):
        def close(self): pass
    value=prediction('a'*64).model_copy(update={'prob_up':.59,'prob_down':.41})
    prediction_repo=Predictions([value]); prediction_repo.close=lambda:None
    recommendations=Recommendations(); recommendations.close=lambda:None
    audit=Audit()
    def forbidden(*args,**kwargs): pytest.fail('DISABLED_GATE_CALLED_LLM_OR_NEWS')
    monkeypatch.setattr(uc,'create_market_repository',lambda _:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(uc,'create_prediction_repository',lambda _:prediction_repo)
    monkeypatch.setattr(uc,'create_recommendation_repository',lambda _:recommendations)
    monkeypatch.setattr(uc,'create_recommendation_request_repository',lambda _:audit)
    monkeypatch.setattr(uc,'PrescriptiveNewsContextService',forbidden)
    monkeypatch.setattr(providers,'create_llm_provider',lambda _:SimpleNamespace(model_name='synthetic-explanation-only',close=lambda:None,generate_recommendation=forbidden))
    def factory(selected):
        service=RecommendationUseCase.from_settings(selected); service._clock=lambda:request_time
        return service
    def handle(payload,selected):
        return handle_recommendation_request(payload,selected,use_case_factory=factory,
            request_repository_factory=lambda _:audit,now=lambda:request_time)
    app=RenderApp(settings_factory=lambda:settings(prescriptive_news_provider='marketaux'),
                  recommendation_handler=handle,secrets=SECRETS)
    response=_request(app,'POST','/api/recommendation',headers={'Authorization':'Bearer '+LLM_SECRET},json={'prediction_id':'a'*64})
    assert response.status_code==200
    assert response.json()['status']=='GUARDRAIL_REJECTED' and response.json()['news_context'] is None
    assert len(audit.records)==1 and audit.records[0].news_context_id is None


def test_marketaux_claim_without_capture_time_cannot_be_current():
    shown=NewsContext(provider='MARKETAUX',status='AVAILABLE',btc_sentiment_score=.8).at_time(NOW)
    assert shown.status=='UNAVAILABLE' and shown.reason=='INVALID_RESPONSE'
    assert shown.btc_sentiment_score is None and shown.scored_news_count==0
    disabled=NewsContext(provider='MARKETAUX',status='UNAVAILABLE',reason='NOT_CONFIGURED')
    assert disabled.at_time(NOW).reason=='NOT_CONFIGURED'


def test_unlicensed_marketaux_content_hidden_even_when_dto_claims_available(monkeypatch):
    from streamlit_app import news_context as ui
    calls=[]
    class Sink:
        def __getattr__(self,name): return lambda *args,**kwargs:calls.append((name,args))
    monkeypatch.setattr(ui,'st',Sink())
    ui.render_marketaux_context(NewsContext(provider='MARKETAUX',status='AVAILABLE',fetched_at=NOW,btc_sentiment_score=.8),now=NOW)
    assert [name for name,_ in calls]==['info']
    assert 'derechos' in calls[0][1][0]


def test_streamlit_navigation_and_rerun_have_no_extra_storage_or_news_calls(monkeypatch):
    from streamlit_app import ui
    from alpha_engine.services.recommendation_api import RecommendationApiClient
    def forbidden(*args,**kwargs): pytest.fail('NAVIGATION_CALLED_OPTIONAL_IO')
    selected=settings(prescriptive_news_provider='marketaux',prescriptive_backend='http')
    dashboard=PublicDashboardData(mode='LIVE',storage='supabase',refreshed_at=datetime.now(timezone.utc),system_state='CURRENT')
    business_reads=[]
    monkeypatch.setattr(ui,'get_settings',lambda:selected)
    monkeypatch.setattr(ui,'dashboard_data',lambda *args:business_reads.append('dashboard') or dashboard)
    monkeypatch.setattr(ui,'latest_prescriptive_prediction',lambda *args:business_reads.append('latest') or None)
    monkeypatch.setattr(RecommendationApiClient,'request',forbidden)
    monkeypatch.setattr(PrescriptiveNewsContextService,'get_context',forbidden)
    monkeypatch.setattr('alpha_engine.repositories.supabase_news.SupabasePrescriptiveNewsRepository.from_settings',forbidden)
    app=AppTest.from_file(ROOT/'streamlit_app/pages/3_Prescriptivo.py',default_timeout=25)
    app.session_state['recommendation_result']=result(NewsContext(provider='MARKETAUX',status='AVAILABLE',fetched_at=NOW,btc_sentiment_score=.8))
    app.run(); assert not app.exception and not app.error
    assert any('derechos' in item.value for item in app.info)
    first=list(business_reads); business_reads.clear()
    app.run(); assert not app.exception and not app.error
    assert business_reads==first  # No new reads caused by Marketaux presentation.


def test_frontend_dependencies_do_not_require_worker_or_new_packages():
    req=(ROOT/'streamlit_app/requirements.txt').read_text().lower()
    assert all(name in req for name in ('pydantic','pydantic-settings','httpx','streamlit','supabase'))
    assert all(name not in req for name in ('psycopg','xgboost','scikit-learn','nvidia'))
    packages=lambda path:[x.strip() for x in path.read_text().splitlines() if x.strip() and not x.startswith('#')]
    assert packages(ROOT/'requirements-streamlit.txt')==packages(ROOT/'streamlit_app/requirements.txt')


