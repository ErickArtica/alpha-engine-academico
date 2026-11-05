"""Offline public/DEMO boundaries and Streamlit interactions; no provider calls."""
import json
import os
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlunsplit

import httpx
from streamlit.testing.v1 import AppTest

from deploy.northflank.academic_client import DemoClient, DemoUnavailable, endpoint
from deploy.northflank import academic_panel as panel
from deploy.northflank.public_config import PublicConfig, ConfigurationError
from public_test_support import PUBLIC_ENV


class DemoTransportTests(unittest.TestCase):
    def test_requires_https_and_never_follows_redirects(self):
        userinfo_fixture = urlunsplit(("https", "user:password@api.example.test", "", "", ""))
        for url in ("http://api.example.test", "https://api.example.test/path", userinfo_fixture, "https://api.example.test?token=x"):
            with self.assertRaises(DemoUnavailable):
                DemoClient(url)
        calls=[]
        def redirect(request):
            calls.append(request)
            return httpx.Response(302,headers={"Location":"https://other.example.test"})
        with self.assertRaises(DemoUnavailable):
            DemoClient("https://api.example.test",transport=httpx.MockTransport(redirect)).post("context",{},"fixture-session")
        self.assertEqual(len(calls),1)

    def test_no_action_transport_without_session_or_outside_allowlist(self):
        def forbidden(request):raise AssertionError("Anonymous action sent")
        client=DemoClient("https://api.example.test",transport=httpx.MockTransport(forbidden))
        self.assertEqual(client.post("explain",{})[0],401)
        with self.assertRaises(DemoUnavailable):client.post("paper",{},"fixture")

    def test_public_config_rejects_every_server_only_demo_variable(self):
        for name in ("ACADEMIC_DEMO_USERS_JSON","ACADEMIC_DEMO_SESSION_SECRET","ACADEMIC_DEMO_NEWS_ENABLED","ACADEMIC_DEMO_LLM_ENABLED"):
            with self.assertRaises(ConfigurationError):
                PublicConfig.from_environment({**PUBLIC_ENV,name:"fixture"})
        self.assertIsNotNone(PublicConfig.from_environment({**PUBLIC_ENV,"ACADEMIC_DEMO_ENABLED":"true","ACADEMIC_DEMO_API_URL":"https://api.example.test"}))

    def test_bounded_response_and_safe_exception(self):
        client=DemoClient("https://api.example.test",transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b"x"*32001)))
        with self.assertRaises(DemoUnavailable) as error:client.post("news",{},"fixture")
        self.assertNotIn("fixture",str(error.exception))


class DemoPanelTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.env=patch.dict(os.environ,{**PUBLIC_ENV,"ACADEMIC_DEMO_ENABLED":"true","ACADEMIC_DEMO_API_URL":"https://api.example.test"},clear=True)
        self.env.start();self.addCleanup(self.env.stop)
        calls=self.calls
        class Client:
            def __init__(self,origin):pass
            def post(self,action,payload,token=None):
                calls.append((action,payload,token))
                if action=="login":
                    return 200,{"status":"AUTHORIZED","access_token":"fixture-user-session","expires_at":time.time()+900}
                if action=="logout":return 200,{"status":"SIGNED_OUT"}
                if action in {"news","refresh-news"}:
                    return 200,{"status":"SNAPSHOT","news":{"status":"N/A","classification":"N/A","score":None,"considered":0}}
                return 200,{"status":"EXPLAINED","decision":{"action":"WAIT","reason":"Regla sin entrada"},
                    "source":{"source_status":"HISTORICO","valid_until":"2026-01-01T00:00:00+00:00"},
                    "decision_hash":"a"*64,"model":"fixture-provider-no-network", "cached":False,
                    "explanation":{"summary":"Esperar sin entrar.","key_factors":["Gate no superado"],"risks":["Incertidumbre"],"warning":"No operativo"}}
        self.client=patch.object(panel,"DemoClient",Client)
        self.client.start();self.addCleanup(self.client.stop)

    def app(self):
        return AppTest.from_string("from deploy.northflank.academic_panel import render_panel\nrender_panel('a'*64)")

    def button(self,app,label):return next(b for b in app.button if b.label==label)

    def test_anonymous_panel_has_no_provider_action(self):
        app=self.app().run()
        self.assertFalse(app.exception)
        self.assertNotIn("Generar explicación con IA",{b.label for b in app.button})
        self.assertFalse(self.calls)

    def test_public_default_has_no_login_or_requests(self):
        os.environ["ACADEMIC_DEMO_ENABLED"]="false"
        app=self.app().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button)
        self.assertFalse(self.calls)

    def test_login_explanation_news_logout_and_no_token_display(self):
        app=self.app().run()
        app.text_input[0].set_value("profesor")
        app.text_input[1].set_value("classroom-fixture-password")
        self.button(app,"Ingresar al DEMO").click().run()
        self.assertFalse(app.exception)
        # AppTest represents full-script reruns; execute once after fragment rerun.
        app.run()
        self.button(app,"Generar explicación con IA").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label=="Decisión determinista"),"WAIT")
        self.assertTrue(any("HISTÓRICO" in c.value for c in app.caption))
        visible=" ".join(e.value for kind in (app.text,app.caption,app.markdown) for e in kind)
        self.assertNotIn("fixture-user-session",visible)
        self.button(app,"Leer sentimiento guardado").click().run()
        self.assertFalse(app.exception)
        self.button(app,"Cerrar sesión DEMO").click().run()
        app.run()
        self.assertNotIn("Generar explicación con IA",{b.label for b in app.button})
        self.assertEqual([x[0] for x in self.calls],["login","explain","news","logout"])

    def test_expired_session_removed_before_any_action(self):
        app=self.app()
        app.session_state[panel.SESSION]={"token":"fixture-expired","expires_at":time.time()-1}
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("Generar explicación con IA",{b.label for b in app.button})
        self.assertFalse(self.calls)

    def test_provider_error_keeps_deterministic_decision_without_fake_explanation(self):
        class FailedClient:
            def __init__(self,origin):pass
            def post(self,*args):
                return 502,{"status":"PROVIDER_READ_TIMEOUT","decision":{"action":"WAIT"},
                            "prediction":{},"source":{},"risk":{},"news":{}}
        app=self.app()
        app.session_state[panel.SESSION]={"token":"fixture-signed-session","expires_at":time.time()+900}
        with patch.object(panel,"DemoClient",FailedClient):
            app.run()
            self.button(app,"Generar explicación con IA").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label=="Decisión determinista"),"WAIT")
        self.assertTrue(any("PROVIDER_READ_TIMEOUT" in w.value for w in app.warning))
        self.assertFalse(any("Explicación NVIDIA" in s.value for s in app.subheader))


if __name__=="__main__":unittest.main()
