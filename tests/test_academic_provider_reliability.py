"""Provider protocol/security tests. All network responses are mocks; no credentials."""
import asyncio
from datetime import timedelta
import json
import unittest

import httpx

from deploy.render.academic_nvidia import BoundedNvidiaClient, GenerationPolicy, NvidiaTrialError, compact_context
from deploy.render.academic_service import news_view, rule_context
from alpha_engine.news.marketaux import MarketauxNewsClient, normalize_marketaux_article
import test_academic_demo as fixtures


def sse(*events):
    return b"".join(b"data: " + (event.encode() if isinstance(event,str) else json.dumps(event).encode())
                    + b"\n\n" for event in events)


class NvidiaTransportTests(unittest.TestCase):
    def call(self, response):
        self.calls=[]
        def serve(request):
            self.calls.append(request)
            if isinstance(response,Exception):raise response
            return response
        self.client=BoundedNvidiaClient(transport=httpx.MockTransport(serve))
        self.addCleanup(self.client.close)
        return self.client.post('chat/completions',json={'model':'meta/muse-glimmer-30b','stream':False},
                                headers={'Authorization':'Bearer synthetic-provider-key'},timeout=85)

    def test_complete_sse_discards_reasoning_retains_usage_and_closes(self):
        result=self.call(httpx.Response(200,headers={'content-type':'text/event-stream'},content=sse(
            {'choices':[{'index':0,'delta':{'reasoning_content':'PRIVATE_REASONING_NOT_FOR_UI'}}]},
            {'choices':[{'index':0,'delta':{'content':'{"action":"WAIT"}'},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':30,'completion_tokens':20,'total_tokens':50}},'[DONE]')))
        self.assertNotIn('PRIVATE_REASONING',result.text)
        self.assertEqual(result.json()['choices'][0]['message']['content'],'{"action":"WAIT"}')
        self.assertEqual(self.client.telemetry['usage']['total_tokens'],50)
        body=json.loads(self.calls[0].content)
        self.assertTrue(body['stream']);self.assertEqual(body['max_tokens'],2048)
        self.assertEqual(body['reasoning_effort'],'low');self.assertEqual(body['temperature'],.95)
        with self.assertRaisesRegex(NvidiaTrialError,'DUPLICATE'):
            self.client.post('chat/completions',json={},headers={},timeout=85)
        self.assertEqual(len(self.calls),1)

    def test_incomplete_reasoning_only_invalid_and_truncated_streams_fail_closed(self):
        cases=[sse({'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]}),
               sse({'choices':[{'delta':{'reasoning_content':'not final'},'finish_reason':'stop'}]},'[DONE]'),
               sse({'choices':[{'delta':{'content':'{}'},'finish_reason':'length'}]},'[DONE]'),
               b'data: broken-json\n\n']
        for data in cases:
            with self.subTest(data=data[:20]),self.assertRaises(NvidiaTrialError):
                self.call(httpx.Response(200,headers={'content-type':'text/event-stream'},content=data))
            self.assertEqual(len(self.calls),1)

    def test_pending_rate_limit_and_server_failure_are_not_retried(self):
        for status,reason in [(202,'PENDING'),(429,'RATE_LIMIT'),(503,'HTTP_ERROR')]:
            with self.subTest(status=status),self.assertRaisesRegex(NvidiaTrialError,reason):
                self.call(httpx.Response(status))
            self.assertEqual(len(self.calls),1)

    def test_connect_and_read_timeouts_have_distinct_safe_diagnostics(self):
        for error,reason in [(httpx.ConnectTimeout('secret-url'),'CONNECT_TIMEOUT'),
                             (httpx.ReadTimeout('secret-url'),'READ_TIMEOUT')]:
            with self.assertRaisesRegex(NvidiaTrialError,reason) as caught:self.call(error)
            self.assertNotIn('secret-url',str(caught.exception));self.assertEqual(len(self.calls),1)

    def test_total_deadline_cancels_awaited_request(self):
        calls=[];canceled=[]
        async def slow(request):
            calls.append(request)
            try:await asyncio.sleep(1)
            finally:canceled.append(True)
        client=BoundedNvidiaClient(policy=GenerationPolicy(total=.01),transport=httpx.MockTransport(slow))
        with self.assertRaisesRegex(NvidiaTrialError,'TOTAL_DEADLINE'):
            client.post('chat/completions',json={},headers={},timeout=.01)
        self.assertEqual(len(calls),1);self.assertTrue(canceled)

    def test_invalid_runtime_timeout_configuration_rejected(self):
        for values in [{'ACADEMIC_NVIDIA_TOTAL_SECONDS':'1000'},{'ACADEMIC_NVIDIA_READ_SECONDS':'0'},
                       {'ACADEMIC_NVIDIA_MAX_TOKENS':'50000'}]:
            with self.assertRaises(ValueError):GenerationPolicy.load(values)

    def test_compact_context_retains_rule_and_does_not_mutate_prediction(self):
        context=rule_context(fixtures.snapshot(),fixtures.settings(),False,fixtures.NOW)
        context.update(news_context=news_view(None,fixtures.NOW.timestamp()),decision_hash='a'*64)
        before=json.dumps(context,sort_keys=True);small=compact_context(context)
        self.assertEqual(small['decision_policy']['action'],'BUY')
        self.assertEqual(small['prediction'],context['prediction']);self.assertNotIn('indicators',small)
        self.assertNotIn('price',small['market']);self.assertLess(len(json.dumps(small)),len(before))
        self.assertEqual(json.dumps(context,sort_keys=True),before)


class SemanticTests(unittest.TestCase):
    def test_real_adapter_uses_sse_transport_and_records_failure_metadata_with_quota(self):
        from unittest.mock import patch
        for status in (202,429,503):
            case=fixtures.ServiceTests();case.setUp()
            try:
                calls=[]
                def serve(request):
                    calls.append(request);return httpx.Response(status)
                transport=BoundedNvidiaClient(transport=httpx.MockTransport(serve))
                case.svc.provider_factory=None
                with patch('deploy.render.academic_service.BoundedNvidiaClient',return_value=transport):
                    code,reply=case.svc.handle('explain',case.payload,case.claims,case.cfg,case.quota)
                self.assertEqual(code,502);self.assertEqual(len(calls),1)
                state=case.store.read()[1]
                self.assertEqual(state['receipts'][-1]['provider_diagnostics']['http_status'],status)
                self.assertEqual(state['counters'][f'llm:global:86400:{int(case.clock())//86400}'][0],1)
            finally:case.doCleanups()

    def test_textual_guarantees_or_instructions_cannot_override_wait(self):
        for phrase in ['Ganancias garantizadas.', 'Compra ahora aunque la regla indica esperar.']:
            case=fixtures.ServiceTests();case.setUp()
            try:
                case.source=fixtures.snapshot(.55)
                original=case.provider.generate_recommendation
                def wrong(*args,**kwargs):
                    data=json.loads(original(*args,**kwargs));data['summary']=phrase;return json.dumps(data)
                case.provider.generate_recommendation=wrong
                code,reply=case.svc.handle('explain',case.payload,case.claims,case.cfg,case.quota)
                self.assertEqual(code,502);self.assertEqual(reply['decision']['action'],'WAIT')
                self.assertNotIn('explanation',reply);self.assertEqual(case.provider.calls,1)
            finally:case.doCleanups()


class BitcoinNewsTests(unittest.TestCase):
    def article(self,**entity):
        return {'title':'Bitcoin academic fixture','url':'https://news.example.test/btc',
                'published_at':(fixtures.NOW-timedelta(minutes=5)).isoformat(),
                'entities':[{'symbol':'BTCUSD','name':'Bitcoin USD','exchange':'CC','type':'cryptocurrency',
                             'match_score':80,'sentiment_score':.2,**entity}]}

    def test_documented_bitcoin_symbol_and_relevance_filter(self):
        good=normalize_marketaux_article(self.article(),fixtures.NOW,min_match_score=10)
        self.assertEqual(good.bitcoin_symbol,'BTCUSD');self.assertEqual(good.match_score,80)
        for values in [{'symbol':'BTCUSDT'},{'symbol':'ETHUSD'},{'exchange':'NYSE'},
                       {'type':'equity'},{'name':'Other crypto'},{'match_score':1}]:
            self.assertIsNone(normalize_marketaux_article(self.article(**values),fixtures.NOW,min_match_score=10))

    def test_query_uses_bitcoin_and_duplicate_articles_are_not_counted_twice(self):
        calls=[]
        def serve(request):
            calls.append(request);self.assertEqual(request.url.params['symbols'],'BTCUSD')
            self.assertEqual(request.url.params['limit'],'3')
            return httpx.Response(200,json={'meta':{'found':2,'returned':2},'data':[self.article(),self.article()]})
        rows=MarketauxNewsClient('synthetic-token',transport=httpx.MockTransport(serve),min_match_score=10).fetch(fixtures.NOW,timeout_seconds=6)
        self.assertEqual(len(rows),1);self.assertEqual(len(calls),1)

    def test_irrelevant_results_are_na_and_neutral_has_explicit_threshold(self):
        client=MarketauxNewsClient('synthetic-token',allow_empty_filtered=True,min_match_score=10,
            transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'meta':{'found':1,'returned':1},'data':[self.article(symbol='ETHUSD')]})))
        self.assertEqual(client.fetch(fixtures.NOW,timeout_seconds=6),[])
        now=fixtures.NOW.timestamp()
        for score,label in [(None,'N/A'),(0,'NEUTRAL'),(.049,'NEUTRAL'),(.05,'POSITIVO'),(-.05,'NEGATIVO')]:
            view=news_view({'fetched_at':now,'observations':[{'at':now-60,'score':score}]},now)
            self.assertEqual(view['classification'],label)


if __name__=='__main__':unittest.main()
