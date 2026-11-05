"""Offline interaction/security regressions for the safe visual parity update."""
from datetime import datetime, timedelta, timezone
import json
import unittest

from streamlit.testing.v1 import AppTest

from deploy.northflank import public_views as views
from deploy.northflank.public_replay import replay_from_sample
from alpha_engine.services.historical_replay import ReplayConfig
import test_public_views as original_checks


class PublicVisualParityTests(unittest.TestCase):
    setUp = original_checks.PublicViewsTests.setUp
    tearDown = original_checks.PublicViewsTests.tearDown
    page = original_checks.PublicViewsTests.page

    def replay_sample(self):
        source = datetime.fromisoformat(self.rows['predictions'][0]['candle_open_time'])
        template = self.rows['predictions'][0]
        predictions = []
        for index in range(3, 19):
            opened = source - timedelta(minutes=index)
            up = .7 if index % 4 in (0, 1) else .3
            predictions.append({**template, 'prediction_id':f'{index:064x}',
                                'candle_open_time':opened.isoformat(),
                                'generated_at':(opened+timedelta(minutes=1)).isoformat(),
                                'valid_until':(opened+timedelta(minutes=2)).isoformat(),
                                'prob_up':up,'prob_down':1-up,'predicted_class':'UP' if up>.5 else 'DOWN'})
        self.rows['predictions'] = predictions

    def test_descriptive_original_kpis_units_and_dark_ohlc_volume(self):
        app = self.page('descriptivo').run()
        self.assertFalse(app.exception)
        self.assertTrue({'Precio de cierre','Variación 1 minuto','Volumen','Última vela cerrada',
                         'RSI 14','EMA 20','MACD','Volatilidad móvil'} <= {m.label for m in app.metric})
        charts = app.get('plotly_chart')
        self.assertEqual(len(charts), 2)
        figure = json.loads(charts[0].proto.spec)
        self.assertEqual(figure['layout']['paper_bgcolor'], '#0F172A')
        self.assertEqual(figure['data'][0]['increasing']['line']['color'], '#22C55E')
        self.assertTrue(any('USDT' in m.value for m in app.metric))
        self.assertTrue(any('BTC' in m.value for m in app.metric))

    def test_home_has_recent_market_chart_but_no_private_audit_queries(self):
        app = self.page('inicio').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get('plotly_chart'))
        self.assertTrue(any(m.label=='Estado basado en datos' for m in app.metric))
        self.assertTrue(all(r.url.path in ('/rest/v1/predictions','/rest/v1/candles') for r in self.requests))

    def test_predictive_keeps_timing_gate_and_complete_recent_history(self):
        app = self.page('predictivo').run()
        self.assertFalse(app.exception)
        columns = set(app.dataframe[0].value.columns)
        self.assertTrue({'Generada (Lima)','Válida hasta','Timing','Vigencia','ID'} <= columns)
        self.assertTrue(any('Gate prescriptivo:' in c.value for c in app.caption))
        self.assertFalse(any(r.url.path.endswith('live_evaluations') for r in self.requests))

    def test_evaluation_separates_three_datasets_and_details_stay_opt_in(self):
        app = self.page('evaluacion').run()
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ['LIVE','TEST HISTÓRICO','SIMULACIÓN ECONÓMICA'])
        self.assertTrue(any(m.label=='Accuracy TEST' for m in app.metric))
        self.assertFalse(any(r.url.path.endswith('predictions') for r in self.requests))
        next(b for b in app.button if b.label=='Consultar desempeño LIVE').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(s.value=='Matriz de confusión LIVE' for s in app.subheader))
        self.assertTrue(any(m.label=='Macro F1' for m in app.metric))
        self.assertTrue(all(r.method=='GET' for r in self.requests))

    def test_prescriptive_hypothetical_position_never_uses_wallet_session(self):
        app = self.page('prescriptivo')
        app.session_state['paper_portfolio_id']='synthetic-private-wallet-must-be-ignored'
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label=='Acción de la regla' and m.value=='BUY' for m in app.metric))
        app.selectbox[0].select('Con posición LONG hipotética').run()
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label=='Acción de la regla' and m.value=='HOLD' for m in app.metric))
        self.assertFalse(any('paper' in r.url.path or '/rpc/' in r.url.path for r in self.requests))

    def test_replay_step_reset_event_search_are_memory_only(self):
        self.replay_sample()
        app = self.page('simulacion').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual({int(r.url.params['limit']) for r in self.requests}, {360,1000})
        next(b for b in app.button if b.label=='Siguiente minuto >').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['public_replay_index'], 1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(app.get('plotly_chart')), 2)
        next(b for b in app.button if b.label=='|◀ Reiniciar').click().run()
        self.assertEqual(app.session_state['public_replay_index'], 0)
        next(b for b in app.button if b.label=='Buscar período con eventos').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(r.method=='GET' for r in self.requests))

    def test_replay_uses_original_policy_and_future_close_cannot_change_decisions(self):
        self.replay_sample()
        with views.make_reader() as reader:
            predictions=reader.predictions(360);candles=reader.candles(1000)
        original=replay_from_sample(predictions,candles,60,ReplayConfig())
        changed=replay_from_sample(predictions,[c.model_copy(update={'close':c.close*5}) for c in candles],60,ReplayConfig())
        self.assertEqual([r['action'] for r in original.steps],[r['action'] for r in changed.steps])
        self.assertTrue(any(r['action']=='BUY' for r in original.steps))
        self.assertTrue(all(r['generated_at']<r['timestamp'] for r in original.steps))
        self.assertEqual(original.at(0)['capital_final'], 1000)

    def test_server_status_text_is_escaped_before_html(self):
        app=AppTest.from_string("from deploy.northflank.public_presentation import status_badge\nstatus_badge('<svg onload=synthetic>')").run()
        self.assertFalse(app.exception)
        self.assertIn('&lt;svg', app.markdown[0].value)
        self.assertNotIn('<svg', app.markdown[0].value)


if __name__=='__main__': unittest.main()
