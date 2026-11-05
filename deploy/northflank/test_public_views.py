"""Real Streamlit page tests with synthetic GET responses and network denied."""
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from deploy.northflank import public_views as views, public_app
from deploy.northflank.public_data import PublicReader
from public_test_support import PUBLIC_ENV, mock_transport, synthetic_rows


class PublicViewsTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.rows = synthetic_rows()
        self.unavailable = None
        self.environment = patch.dict(os.environ, PUBLIC_ENV, clear=True)
        self.environment.start()
        connect = socket.socket.connect
        def loopback_only(sock, address):
            if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
                return connect(sock, address)
            raise AssertionError("EXTERNAL_SOCKET_FORBIDDEN")
        self.sockets = patch.object(socket.socket, "connect", loopback_only)
        self.sockets.start()
        self.dns = patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS_FORBIDDEN"))
        self.dns.start()
        self.reader_patch = patch.object(views, "make_reader", side_effect=lambda: PublicReader(
            views.config(), transport=mock_transport(self.rows, self.requests, unavailable=self.unavailable)))
        self.reader_patch.start()
        for cache in (views.snapshot, views.latest_prediction, views.latest_candle, views.metrics):
            cache.clear()

    def tearDown(self):
        self.reader_patch.stop()
        self.dns.stop()
        self.sockets.stop()
        self.environment.stop()

    def page(self, route):
        return AppTest.from_string("from deploy.northflank.public_views import resolve_page\nresolve_page(" + repr(route) + ")()", default_timeout=15)

    def test_all_six_public_pages_render_without_actions_or_private_reads(self):
        for route, (title, _) in views.ROUTES.items():
            with self.subTest(route=route):
                app = self.page(route)
                app.session_state["paper_portfolio_id"] = "synthetic-ignored"
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.title[0].value, "Alpha Engine" if route == "inicio" else title)
                labels = [button.label for button in app.button]
                self.assertFalse(any(word in label for label in labels for word in ("Generar", "Crear cartera", "Aplicar operación", "Reiniciar simulación", "LLM")))
        self.assertTrue(self.requests)
        self.assertEqual({r.method for r in self.requests}, {"GET"})
        self.assertTrue({r.url.path for r in self.requests} <= {"/rest/v1/candles", "/rest/v1/predictions", "/rest/v1/live_evaluations"})

    def test_predictive_metrics_remain_opt_in_and_history_survives_read_error(self):
        app = self.page("predictivo").run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == "P(SUBE)" for m in app.metric))
        self.assertFalse(any(r.url.path.endswith("live_evaluations") for r in self.requests))
        self.unavailable = "live_evaluations"
        next(b for b in app.button if b.label == "Consultar desempeño LIVE").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertTrue(any("métricas LIVE no están disponibles" in i.value for i in app.info))

    def test_offline_metrics_click_only_reads_the_bounded_cohort(self):
        app = self.page("predictivo").run()
        next(b for b in app.button if b.label == "Consultar desempeño LIVE").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label == "Accuracy LIVE" and m.value == "100.0%" for m in app.metric))

    def test_router_rejects_original_pages_and_action_paths(self):
        self.assertEqual(views.ROUTES["inicio"][0], "Home")
        for route in ("5_Simulacion", "3_Prescriptivo", "../streamlit_app/pages/3_Prescriptivo.py", "/api/paper", "/api/recommendation"):
            with self.subTest(route=route), self.assertRaises(ValueError):
                views.resolve_page(route)
        self.assertEqual(self.requests, [])
        app = AppTest.from_file(public_app.__file__, default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Alpha Engine")

    def test_disabling_profile_stops_even_when_bypassing_the_launcher(self):
        os.environ["NORTHFLANK_PUBLIC_READ_ONLY"] = "false"
        app = AppTest.from_file(public_app.__file__, default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
