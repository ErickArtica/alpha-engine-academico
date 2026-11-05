"""Offline classroom acceptance: fake provider responses, no external network/DB."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx

from alpha_engine.config.settings import Settings
from alpha_engine.data.schemas import PredictionRecord
from deploy.render.academic_demo import AcademicDemo, DemoConfig, Quotas, password_hash
from deploy.render.academic_service import AcademicService, news_view, rule_context
from deploy.render.academic_state import SQLiteState, StateUnavailable, SupabaseState


class TransportDeadlineTests(unittest.TestCase):
    def test_generation_connect_and_read_timeouts_are_explicit(self):
        from deploy.render.academic_nvidia import BoundedNvidiaClient
        calls=[]
        def respond(request):
            calls.append(request.extensions['timeout'])
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
        transport=BoundedNvidiaClient(transport=httpx.MockTransport(respond))
        try:transport.post('chat/completions',json={},headers={},timeout=85)
        finally:transport.close()
        self.assertEqual(calls[0]['read'],60)
        self.assertEqual(calls[0]['connect'],5)
        self.assertEqual(len(calls),1)

    def test_response_size_fails_without_retry(self):
        from deploy.render.academic_nvidia import BoundedNvidiaClient,NvidiaTrialError
        calls=[]
        def respond(request):
            calls.append(request)
            return httpx.Response(200,content=b'x'*262145)
        transport=BoundedNvidiaClient(transport=httpx.MockTransport(respond))
        with self.assertRaisesRegex(NvidiaTrialError,'PROVIDER_RESPONSE_TOO_LARGE'):
            transport.post('chat/completions',json={},headers={},timeout=85)
        self.assertEqual(len(calls),1)


NOW = datetime(2026, 11, 4, 22, 0, 20, tzinfo=timezone.utc)
HASH = password_hash("classroom-fixture-password", salt=b"a" * 24)


def environment(**changes):
    env = {"ACADEMIC_DEMO_ENABLED": "true", "ACADEMIC_DEMO_USERS_JSON": json.dumps({"profesor": HASH, "alumno": HASH}),
           "ACADEMIC_DEMO_SESSION_SECRET": "classroom-fixture-signing-value-" + "x" * 48,
           "ACADEMIC_DEMO_EXPIRES_AT": (NOW + timedelta(hours=2)).isoformat(),
           "ACADEMIC_DEMO_LLM_ENABLED": "true", "ACADEMIC_DEMO_NEWS_ENABLED": "true",
           "ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED": "true"}
    return {**env, **changes}


def settings():
    return Settings(_env_file=None, storage_provider="supabase", supabase_url="https://fixture.invalid",
                    supabase_key="fixture-server-value", llm_provider="nvidia", nvidia_model="meta/muse-glimmer-30b",
                    nvidia_api_key="fixture-provider-value")


def snapshot(up=0.7, risk="LOW", *, expired=False):
    prediction = PredictionRecord(prediction_id="a" * 64, symbol="BTCUSDT", timestamp=NOW - timedelta(seconds=80),
        horizon="1m", prob_up=up, prob_down=1-up, predicted_class="UP" if up > .5 else "DOWN",
        model_version="xgboost_v1", generated_at=NOW - timedelta(seconds=15),
        valid_until=NOW + timedelta(seconds=-1 if expired else 35))
    return NS(prediction=prediction,
        market=NS(symbol="BTCUSDT", timestamp=prediction.timestamp, close_time=NOW-timedelta(seconds=20),
                  price=100., change_1m_pct=.01, volume=3.),
        indicators=NS(**{k: .01 for k in ("return_1m", "rsi_14", "ema_20", "macd", "macd_signal", "macd_histogram", "rolling_volatility", "volume_change")}),
        risk=NS(risk_level=risk, reference_price=100., stop_loss=99., take_profit=102., triggers=[], rule_version="fixture"))


class Provider:
    def __init__(self, *, wrong=False, error=False):
        self.calls, self.closed, self.wrong, self.error = 0, False, wrong, error
    def generate_recommendation(self, context, *, timeout_seconds):
        self.calls += 1
        if self.error:
            raise httpx.ReadTimeout("fixture failure")
        assert timeout_seconds == 85
        return json.dumps(dict(action="SELL" if self.wrong else context["decision_policy"]["action"],
            decision_hash=context["decision_hash"], risk_level=context["risk"]["level"],
            summary="Las reglas determinan la acción académica.", key_factors=["Probabilidad y posición verificadas"],
            risks=["La predicción puede fallar"], warning="Demostración académica no operativa."))
    def close(self):
        self.closed = True


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.sqlite"
        self.store = SQLiteState(self.path)

    def test_atomic_budget_across_instances_and_restart(self):
        def reserve(_):
            q = Quotas(SQLiteState(self.path), lambda: NOW.timestamp())
            return q.reserve("llm", "profesor", "session", lease=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            reservations = list(pool.map(reserve, range(6)))
        self.assertEqual(sum(bool(x) for x in reservations), 1)
        q = Quotas(SQLiteState(self.path), lambda: NOW.timestamp())
        self.assertFalse(q.reserve("llm", "alumno", "other", lease=True))
        self.assertEqual(self.store.read()[1]["counters"][f"llm:global:86400:{int(NOW.timestamp())//86400}"][0], 1)

    def test_global_daily_budget_not_reset_by_new_sessions(self):
        tick = [NOW.timestamp()]
        q = Quotas(self.store, lambda: tick[0])
        for index in range(10):
            self.assertTrue(q.reserve("llm", "user"+str(index), "s"+str(index)))
            tick[0] += 61
        self.assertFalse(q.reserve("llm", "new-user", "new-session"))

    def test_rejected_reservation_does_not_write_unchanged_state(self):
        quota=Quotas(self.store,lambda:NOW.timestamp())
        self.assertTrue(quota.reserve("llm","profesor","session"))
        version=self.store.read()[0]
        self.assertFalse(quota.reserve("llm","profesor","session"))
        self.assertEqual(version,self.store.read()[0])

    def test_quota_failure_is_closed(self):
        store = SupabaseState(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(403)))
        with self.assertRaises(StateUnavailable):
            Quotas(store, lambda: NOW.timestamp()).reserve("llm", "profesor")

    def test_rpc_adapter_calls_only_reviewed_private_functions(self):
        paths = []
        def respond(request):
            paths.append(request.url.path)
            return httpx.Response(200, json={"version": 0, "payload": {}} if request.url.path.endswith("read") else True)
        state = SupabaseState(settings(), transport=httpx.MockTransport(respond))
        self.assertEqual(state.update(lambda s: s.update(check=True)), None)
        self.assertEqual(paths, ["/rest/v1/rpc/academic_demo_state_read", "/rest/v1/rpc/academic_demo_state_cas"])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        StateTests.setUp(self)
        self.tick = [NOW.timestamp()]
        self.clock = lambda: self.tick[0]
        self.cfg = DemoConfig.load(environment(), self.clock())
        self.quota = Quotas(self.store, self.clock)
        self.claims = {"sub": "profesor", "sid": "fixture-session"}
        self.payload = {"prediction_id": "a" * 64, "has_long": False}
        self.provider = Provider()
        self.source = snapshot()
        self.svc = AcademicService(settings(), self.store, self.clock,
            snapshot_loader=lambda *args: self.source, provider_factory=lambda s: self.provider)
        self.block_network = patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("External HTTP forbidden"))
        self.block_network.start()
        self.addCleanup(self.block_network.stop)

    def test_buy_sell_hold_wait_and_no_prediction_mutation(self):
        for up, has_long, expected in [(.7,False,"BUY"),(.3,True,"SELL"),(.7,True,"HOLD"),(.3,False,"WAIT"),(.55,False,"WAIT"),(.55,True,"HOLD")]:
            with self.subTest(expected=expected, up=up):
                source = snapshot(up)
                before = source.prediction.model_dump()
                context = rule_context(source, settings(), has_long, NOW)
                self.assertEqual(context["decision_policy"]["action"], expected)
                self.assertEqual(context["decision_policy"]["minimum_model_probability"], .60)
                self.assertEqual(before, source.prediction.model_dump())

    def test_each_action_has_coherent_mock_explanation(self):
        # G7 labels BUY/LOW, SELL/LOW, HOLD/MEDIUM, plus the position-aware WAIT.
        # Synthetic controlled values, not a replay or new proof of historical provider calls.
        for index, (up, position, action, risk) in enumerate([(.7,False,"BUY","LOW"),(.3,True,"SELL","LOW"),(.55,True,"HOLD","MEDIUM"),(.3,False,"WAIT","LOW")]):
            self.source = snapshot(up, risk)
            who = {"sub": "subject"+str(index), "sid": "session"+str(index)}
            code, result = self.svc.handle("explain", {**self.payload,"has_long":position}, who, self.cfg, self.quota)
            self.assertEqual(code, 200, result)
            self.assertEqual(result["decision"]["action"], action)
            self.assertEqual(result["explanation"]["action"], action)
            self.assertFalse(result["operational"])

    def test_expired_or_high_risk_never_yields_executable_entry(self):
        for source in (snapshot(expired=True), snapshot(risk="HIGH")):
            context = rule_context(source, settings(), False, NOW)
            self.assertEqual(context["decision_policy"]["action"], "WAIT")

    def test_context_reads_no_provider(self):
        code, result = self.svc.handle("context", self.payload, self.claims, self.cfg, self.quota)
        self.assertEqual(code, 200)
        self.assertEqual(result["news"]["score"], None)
        self.assertEqual(self.provider.calls, 0)

    def test_historical_mode_uses_real_generation_time_and_excludes_current_news(self):
        self.source=snapshot(expired=True)
        code,reply=self.svc.handle("context",{**self.payload,"historical":True},self.claims,self.cfg,self.quota)
        self.assertEqual(code,200)
        self.assertEqual(reply["decision"]["action"],"BUY")
        self.assertEqual(reply["source"]["source_status"],"HISTORICO")
        self.assertEqual(reply["news"]["reason"],"NO_POINT_IN_TIME_SNAPSHOT")
        self.assertEqual(reply["prediction"]["prob_up"],.7)
        self.source.prediction.generated_at=NOW
        self.assertEqual(self.svc.handle("context",{**self.payload,"historical":True},self.claims,self.cfg,self.quota)[0],409)

    def test_rejects_client_injected_action_price_or_gate(self):
        for key, value in [("action","BUY"),("price", 100),("threshold",.1),("portfolio_id","other")]:
            self.assertEqual(self.svc.handle("explain", {**self.payload,key:value}, self.claims,self.cfg,self.quota)[0],400)
        self.assertEqual(self.provider.calls, 0)

    def test_disabled_llm_and_wrong_model_do_not_call_provider(self):
        cfg = DemoConfig.load(environment(ACADEMIC_DEMO_LLM_ENABLED="false"), self.clock())
        self.assertEqual(self.svc.handle("explain",self.payload,self.claims,cfg,self.quota)[0],403)
        self.svc.settings = settings().model_copy(update={"nvidia_model":"different-model"})
        self.assertEqual(self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)[0],503)
        self.assertEqual(self.provider.calls,0)

    def test_cached_explanation_no_second_provider_call_and_scoped_to_user(self):
        first = self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)
        second = self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)
        self.assertEqual(first[0],200)
        self.assertTrue(second[1]["cached"])
        self.assertEqual(self.provider.calls,1)
        self.assertEqual(self.svc.handle("explain",self.payload,{"sub":"alumno","sid":"s2"},self.cfg,self.quota)[0],200)
        self.assertEqual(self.provider.calls,2)

    def test_provider_failure_consumes_budget_and_returns_safe_error(self):
        self.provider.error = True
        code, result = self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)
        self.assertEqual(code,502)
        self.assertNotIn("fixture failure",json.dumps(result))
        self.assertEqual(result["decision"]["action"],"BUY")
        self.assertTrue(self.provider.closed)
        self.assertEqual(self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)[0],429)

    def test_existing_nvidia_adapter_integrates_with_mock_http_and_schema(self):
        from alpha_engine.ai.providers.nvidia import NvidiaProvider
        calls=[]
        def response(request):
            body=json.loads(request.content)
            calls.append(body)
            self.assertEqual(body["model"],"meta/muse-glimmer-30b")
            self.assertEqual(body["max_tokens"],1024)
            context=json.loads(body["messages"][1]["content"])
            raw=self.provider.generate_recommendation(context,timeout_seconds=85)
            return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":raw}}]})
        client=httpx.Client(base_url="https://fixture-nvidia.invalid/",transport=httpx.MockTransport(response))
        self.addCleanup(client.close)
        prompt=Path('deploy/render/academic_prompt.txt').read_text(encoding='utf-8')
        self.svc.provider_factory=lambda s:NvidiaProvider(s,prompt,client=client)
        code,result=self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)
        self.assertEqual(code,200,result)
        self.assertEqual(len(calls),1)
        self.assertNotIn("fixture-provider-value",json.dumps(result))

    def test_incoherent_response_rejected_no_publication(self):
        self.provider.wrong = True
        code,result=self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)
        self.assertEqual(code,502)
        self.assertEqual(result["status"],"INCOHERENT_RESPONSE")
        self.assertNotIn("explanation",result)

    def test_unverified_numeric_claim_is_rejected_without_changing_rule(self):
        generate=self.provider.generate_recommendation
        def numeric(*args,**kwargs):
            result=json.loads(generate(*args,**kwargs));result['summary']='Rentabilidad garantizada del 90 por ciento.'
            return json.dumps(result)
        self.provider.generate_recommendation=numeric
        code,result=self.svc.handle('explain',self.payload,self.claims,self.cfg,self.quota)
        self.assertEqual(code,502)
        self.assertEqual(result['decision']['action'],'BUY')
        self.assertNotIn('explanation',result)

    def test_invalid_json_and_extra_fields_rejected(self):
        for raw in ("not json", '{"action":"BUY","action":"WAIT"}', '{"secret":"fixture"}'):
            with self.subTest(raw=raw):
                self.provider.generate_recommendation = lambda *a,**kw:raw
                self.assertEqual(self.svc.handle("explain",self.payload,self.claims,self.cfg,self.quota)[0],502)
                self.tick[0]+=61

    def test_news_requires_rights_no_implicit_refresh(self):
        cfg=DemoConfig.load(environment(ACADEMIC_DEMO_NEWS_RIGHTS_APPROVED="false"),self.clock())
        self.assertEqual(self.svc.handle("refresh-news",{},self.claims,cfg,self.quota)[0],403)
        self.assertEqual(self.svc.handle("news",{},self.claims,cfg,self.quota)[1]["news"]["score"],None)

    def test_marketaux_client_aggregation_cache_and_private_derivatives(self):
        from alpha_engine.news.marketaux import MarketauxNewsClient
        calls=[]
        def respond(request):
            calls.append(request)
            self.assertEqual(request.url.params["limit"],"3")
            self.assertEqual(request.url.params["symbols"],"BTCUSD")
            items=[dict(title="Fixture Bitcoin article",url="https://news.example.test/"+str(i),
                        published_at=(NOW-timedelta(minutes=5+i)).isoformat(),
                        entities=[{"symbol":"CC:BTC","name":"Bitcoin","sentiment_score":score}])
                   for i,score in enumerate([.4,-.2,0.])]
            return httpx.Response(200,json={"data":items,"meta":{"found":3,"returned":3}})
        self.svc.news_client=MarketauxNewsClient("fixture-marketaux-value",transport=httpx.MockTransport(respond))
        with patch.dict("os.environ", {"MARKETAUX_API_TOKEN":"fixture-marketaux-value"}):
            code,reply=self.svc.handle("refresh-news",{},self.claims,self.cfg,self.quota)
            self.assertEqual(code,200,reply)
            self.assertAlmostEqual(reply["news"]["score"],.2/3)
            self.assertEqual(reply["news"]["considered"],3)
            self.assertEqual(reply["news"]["classification"],"POSITIVO")
            self.assertEqual(self.svc.handle("refresh-news",{},self.claims,self.cfg,self.quota)[1]["status"],"CACHED")
        self.assertEqual(len(calls),1)
        stored=json.dumps(self.store.read()[1])
        self.assertNotIn("Fixture Bitcoin",stored)
        self.assertNotIn("news.example",stored)
        self.assertNotIn("fixture-marketaux-value",stored)
        self.tick[0]+=1801
        self.assertEqual(self.svc.handle("news",{},self.claims,self.cfg,self.quota)[1]["news"]["status"],"STALE")

    def test_neutral_missing_positive_negative_and_historical(self):
        now=self.clock()
        for scores,expected in [([.2],"POSITIVO"),([-.2],"NEGATIVO"),([0.],"NEUTRAL"),([None],"N/A"),([],"N/A")]:
            view=news_view({"fetched_at":now,"observations":[{"at":now-30,"score":s} for s in scores]},now)
            self.assertEqual(view["classification"],expected)
        self.assertEqual(news_view({"fetched_at":now,"observations":[{"at":now-4000,"score":.3}]},now)["status"],"HISTORICO")

    def test_news_failure_keeps_cooldown_and_no_retry(self):
        self.svc.news_client=NS(fetch=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("private failure")))
        with patch.dict("os.environ",{"MARKETAUX_API_TOKEN":"fixture-marketaux-value"}):
            self.assertEqual(self.svc.handle("refresh-news",{},self.claims,self.cfg,self.quota)[0],502)
            self.assertEqual(self.svc.handle("refresh-news",{},self.claims,self.cfg,self.quota)[1]["status"],"CACHED")


class AuthenticationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store=SQLiteState(Path(self.temp.name)/"auth.sqlite")
        self.tick=[NOW.timestamp()]
        self.env=environment()
        self.calls=[]
        service=NS(handle=lambda *args:(self.calls.append(args) or (200,{"status":"FIXTURE"})))
        self.app=AcademicDemo(self.env,settings,store=self.store,service=service,clock=lambda:self.tick[0])

    async def request(self,action,payload=None,token=None,method="POST",headers=None):
        values=[(b"content-type",b"application/json")]
        if token:values.append((b"authorization",("Bearer "+token).encode()))
        values.extend(headers or [])
        async def read(): return payload if payload is not None else {}
        return await self.app.handle({"path":"/api/academic/"+action,"method":method,"headers":values},read)

    async def login(self,user="profesor"):
        code,body=await self.request("login",{"user":user,"password":"classroom-fixture-password"})
        self.assertEqual(code,200,body)
        return body["access_token"]

    async def test_every_action_rejects_anonymous_shared_secret_and_duplicate_headers(self):
        for action in ("context","explain","news","refresh-news","logout"):
            self.assertEqual((await self.request(action))[0],401)
            self.assertEqual((await self.request(action,token="fixture-old-service-bearer"))[0],401)
        token=await self.login()
        self.assertEqual((await self.request("context",token=token,headers=[(b"authorization",b"Bearer other")]))[0],401)
        self.assertFalse(self.calls)

    async def test_login_signed_session_expiry_tampering_and_logout(self):
        token=await self.login()
        self.assertEqual((await self.request("context",token=token))[0],200)
        self.assertEqual((await self.request("context",token=token+"x"))[0],401)
        self.assertEqual((await self.request("logout",token=token))[0],200)
        self.assertEqual((await self.request("context",token=token))[0],401)
        token=await self.login("alumno")
        self.tick[0]+=901
        self.assertEqual((await self.request("context",token=token))[0],401)

    async def test_invalid_credentials_and_login_throttle(self):
        for _ in range(5):
            self.assertEqual((await self.request("login",{"user":"unknown","password":"wrong-fixture-password"}))[0],401)
        self.assertEqual((await self.request("login",{"user":"profesor","password":"classroom-fixture-password"}))[0],429)

    async def test_disabled_window_removed_user_unknown_endpoint_and_content_type(self):
        token=await self.login()
        self.env["ACADEMIC_DEMO_USERS_JSON"]=json.dumps({"alumno":HASH})
        self.assertEqual((await self.request("context",token=token))[0],401)
        self.env["ACADEMIC_DEMO_ENABLED"]="false"
        self.assertEqual((await self.request("context",token=token))[0],404)
        self.assertEqual((await self.request("paper",token=token))[0],404)
        self.assertEqual((await self.request("login",method="GET"))[0],405)

    async def test_backend_provider_and_storage_never_used_when_disabled(self):
        app=AcademicDemo({},lambda:(_ for _ in ()).throw(AssertionError("Should not load backend")))
        async def no_body():raise AssertionError("Should not read body")
        self.assertEqual((await app.handle({"path":"/api/academic/explain","method":"POST"},no_body))[0],404)

    async def test_process_ingress_guard_rejects_before_database_io(self):
        with patch.object(self.store,"read",side_effect=AssertionError("Anonymous request touched DB")):
            for _ in range(60):self.assertEqual((await self.request("explain"))[0],401)
            self.assertEqual((await self.request("explain"))[0],429)

    async def test_config_rejects_plaintext_hashes_shared_signing_key_and_unbounded_window(self):
        for values in ({"ACADEMIC_DEMO_USERS_JSON":'{"profesor":"plaintext"}'},
                       {"ACADEMIC_DEMO_SESSION_SECRET":"short"},
                       {"ACADEMIC_DEMO_EXPIRES_AT":(NOW+timedelta(days=3)).isoformat()}):
            with self.assertRaises(ValueError):DemoConfig.load(environment(**values),NOW.timestamp())

    async def test_actual_asgi_routing_body_limit_and_legacy_routes_unchanged(self):
        from deploy.render.app import RenderApp
        app=RenderApp(secrets={},academic_demo=self.app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture.invalid") as client:
            response=await client.post("/api/academic/explain",json={})
            self.assertEqual(response.status_code,401)
            response=await client.post("/api/academic/login",json={"user":"profesor","password":"classroom-fixture-password"})
            self.assertEqual(response.status_code,200)
            token=response.json()["access_token"]
            response=await client.post("/api/academic/context",json={},headers={"Authorization":"Bearer "+token})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers["cache-control"],"no-store")
            response=await client.post("/api/academic/context",content="x"*2049,
                headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
            self.assertEqual(response.status_code,413)
            self.assertEqual((await client.get("/api/paper")).status_code,405)
            self.assertEqual((await client.post("/api/paper",json={})).status_code,503)

    async def test_demo_session_cannot_reach_legacy_write_or_cron_handlers(self):
        from deploy.render.app import RenderApp
        from alpha_engine.services.recommendation_api import ApiReply
        called=[]
        secrets={"INTERNAL_CRON_SECRET":"fixture-cron-"+"c"*40,
                 "RECOMMENDATION_API_SECRET":"fixture-service-"+"r"*40}
        def reply(payload,settings):
            called.append(payload)
            return ApiReply(200,{"status":"FIXTURE_ONLY"})
        app=RenderApp(secrets=secrets,academic_demo=self.app,settings_factory=settings,
            model_verifier=lambda:Path("models/xgboost_v1.joblib"),recommendation_handler=reply,paper_handler=reply)
        token=await self.login()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture.invalid") as client:
            for path in ("/api/paper","/api/recommendation","/internal/run-cycle","/api/demo-explanation"):
                response=await client.post(path,json={},headers={"Authorization":"Bearer "+token})
                self.assertEqual(response.status_code,401)
            self.assertFalse(called)
            response=await client.post("/api/recommendation",json={"fixture":True},
                headers={"Authorization":"Bearer "+secrets["RECOMMENDATION_API_SECRET"]})
            self.assertEqual(response.status_code,200)
            self.assertEqual(called,[{"fixture":True}])

    async def test_independent_entrypoint_excludes_all_legacy_action_routes(self):
        from deploy.render.academic_only import AcademicOnly
        from deploy.render.app import RenderApp
        app=AcademicOnly(RenderApp(secrets={},academic_demo=self.app))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://fixture.invalid') as client:
            for path in ('/internal/run-cycle','/api/paper','/api/recommendation','/api/demo-explanation','/internal/diagnostics/muse'):
                self.assertEqual((await client.post(path,json={})).status_code,404)
            self.assertEqual((await client.post('/api/academic/explain',json={})).status_code,401)


if __name__ == "__main__":
    unittest.main()
