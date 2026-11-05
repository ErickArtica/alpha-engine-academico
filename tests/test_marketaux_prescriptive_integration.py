"""Mocked ingestion -> durable cache -> policy -> Muse context, without external I/O."""
from dataclasses import replace
from datetime import timedelta
import os
import unittest
from unittest.mock import patch
import httpx
import test_academic_demo as fixture
from alpha_engine.news.marketaux import MarketauxNewsClient

class NewsIntegrationTests(unittest.TestCase):
    def case(self,respond):
        case=fixture.ServiceTests();case.setUp();self.addCleanup(case.doCleanups)
        case.cfg=replace(case.cfg,news=True,news_rights=True)
        case.svc.news_client=MarketauxNewsClient('synthetic-fixture-token',min_match_score=10,
            allow_empty_filtered=True,transport=httpx.MockTransport(respond))
        return case

    @patch.dict(os.environ,{'MARKETAUX_API_TOKEN':'synthetic-fixture-token'})
    def test_valid_above_100_article_flows_to_muse_without_changing_policy(self):
        calls=[]
        def respond(request):
            calls.append(request)
            return httpx.Response(200,json={'meta':{'found':1,'returned':1},'data':[{
                'title':'Synthetic Bitcoin fixture','url':'https://fixture.example.test/bitcoin',
                'published_at':(fixture.NOW-timedelta(minutes=5)).isoformat(),
                'entities':[{'symbol':'BTCUSD','name':'Bitcoin USD','type':'cryptocurrency','exchange':'CC',
                             'match_score':112.536514,'sentiment_score':.13115}]}]})
        case=self.case(respond)
        _,before=case.svc.handle('context',case.payload,case.claims,case.cfg,case.quota)
        original_prediction=case.source.prediction.model_dump()
        code,result=case.svc.handle('refresh-news',{},case.claims,case.cfg,case.quota)
        self.assertEqual(code,200);self.assertEqual(result['news']['status'],'LIVE')
        self.assertEqual(result['news']['score'],.13115);self.assertEqual(result['news']['classification'],'POSITIVO')
        self.assertEqual(result['news']['considered'],1)
        cached=case.store.read()[1]['news_cache']['observations'][0]
        self.assertEqual(cached['match_score'],112.536514)
        for forbidden in ('title','url','description','raw','token'):self.assertNotIn(forbidden,cached)
        seen=[];original=case.provider.generate_recommendation
        def explain(context,**kwargs):seen.append(context);return original(context,**kwargs)
        case.provider.generate_recommendation=explain
        code,reply=case.svc.handle('explain',case.payload,case.claims,case.cfg,case.quota)
        self.assertEqual(code,200);self.assertEqual(reply['decision'],before['decision'])
        self.assertEqual(case.source.prediction.model_dump(),original_prediction)
        self.assertEqual(seen[0]['news_context']['score'],.13115)
        self.assertEqual(seen[0]['decision_policy']['minimum_model_probability'],.60)
        self.assertEqual(case.svc.handle('refresh-news',{},case.claims,case.cfg,case.quota)[1]['status'],'CACHED')
        self.assertEqual(len(calls),1);self.assertEqual(case.provider.calls,1)
        case.tick[0]+=1801
        self.assertEqual(case.svc.handle('news',{},case.claims,case.cfg,case.quota)[1]['news']['status'],'STALE')
        case.tick[0]+=3600
        old=case.svc.handle('news',{},case.claims,case.cfg,case.quota)[1]['news']
        self.assertEqual(old['status'],'HISTORICO');self.assertIsNone(old['score'])

    def test_unapproved_exposure_never_calls_news_or_reserves_provider_budget(self):
        case=self.case(lambda r:(_ for _ in ()).throw(AssertionError('Not authorized')))
        case.cfg=replace(case.cfg,news_rights=False)
        self.assertEqual(case.svc.handle('refresh-news',{},case.claims,case.cfg,case.quota)[0],403)
        view=case.svc.handle('news',{},case.claims,case.cfg,case.quota)[1]['news']
        self.assertEqual(view['status'],'N/A');self.assertIsNone(view['score'])
        self.assertNotIn('news_cache',case.store.read()[1])

    @patch.dict(os.environ,{'MARKETAUX_API_TOKEN':'synthetic-fixture-token'})
    def test_failed_provider_keeps_policy_and_cooldown_and_counts_failure(self):
        calls=[]
        def failure(request):calls.append(request);return httpx.Response(503)
        case=self.case(failure)
        code,reply=case.svc.handle('refresh-news',{},case.claims,case.cfg,case.quota)
        self.assertEqual(code,502);self.assertEqual(reply['provider_diagnostics']['http_status'],503)
        state=case.store.read()[1]
        self.assertEqual(state['counters'][f'news:global:86400:{int(case.clock())//86400}'][0],1)
        self.assertEqual(state['receipts'][-1]['provider_diagnostics']['http_status'],503)
        self.assertEqual(case.svc.handle('refresh-news',{},case.claims,case.cfg,case.quota)[1]['status'],'CACHED')
        self.assertEqual(len(calls),1)
        code,context=case.svc.handle('context',case.payload,case.claims,case.cfg,case.quota)
        self.assertEqual(code,200);self.assertEqual(context['decision']['action'],'BUY')
        self.assertEqual(context['news']['status'],'N/A');self.assertEqual(case.provider.calls,0)

if __name__=='__main__':unittest.main()
