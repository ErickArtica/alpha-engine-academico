"""Offline regression of unbounded relevance, safe diagnostics and age semantics."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import math
import unittest
import httpx
from pydantic import ValidationError
from alpha_engine.news.marketaux import MarketauxNewsClient,MarketauxUnavailable,NewsArticle,normalize_marketaux_article
from deploy.render.academic_service import news_view

NOW=datetime(2026,11,5,12,tzinfo=timezone.utc)

def article():
    return {'title':'Bitcoin synthetic fixture','url':'https://fixture.example.test/bitcoin',
            'published_at':(NOW-timedelta(minutes=5)).isoformat(),
            'entities':[{'symbol':'BTCUSD','exchange':'CC','type':'cryptocurrency','name':'Bitcoin USD',
                         'match_score':112.536514,'sentiment_score':.13115}]}

class DiagnosticsTests(unittest.TestCase):
    def normalize(self,item):
        self.diagnostics={}
        return normalize_marketaux_article(item,NOW,min_match_score=10,diagnostics=self.diagnostics)

    def test_documented_relevance_above_100_preserves_value(self):
        row=self.normalize(article())
        self.assertEqual(row.match_score,112.536514)
        self.assertEqual(row.sentiment_score,.13115)
        self.assertEqual(NewsArticle.model_validate(row.model_dump()).match_score,112.536514)
        self.assertEqual(self.diagnostics,{})

    def test_relevance_is_finite_nonnegative_and_respects_minimum(self):
        for value,code in [(-1,'RELEVANCE_INVALID'),(math.nan,'RELEVANCE_INVALID'),
                           (math.inf,'RELEVANCE_INVALID'),(True,'RELEVANCE_INVALID'),
                           ('112.5','RELEVANCE_INVALID'),(9.99,'RELEVANCE_INSUFFICIENT'),(None,'RELEVANCE_MISSING')]:
            with self.subTest(value=str(value)):
                item=article();item['entities'][0]['match_score']=value
                self.assertIsNone(self.normalize(item));self.assertEqual(self.diagnostics['reason_counts'],{code:1})
        for value in [10,100,112.536514,1000,1e200]:
            item=article();item['entities'][0]['match_score']=value
            self.assertEqual(self.normalize(item).match_score,value)
        good=self.normalize(article()).model_dump()
        for value in [-1,math.inf,math.nan,True]:
            with self.assertRaises(ValidationError):NewsArticle.model_validate({**good,'match_score':value})

    def test_sentiment_boundaries_missing_invalid_and_ambiguous(self):
        for value in [-1,0,.13115,1]:
            item=article();item['entities'][0]['sentiment_score']=value
            self.assertEqual(self.normalize(item).sentiment_score,value)
        for value,code in [(None,'SENTIMENT_MISSING'),(1.01,'SENTIMENT_INVALID'),(-1.01,'SENTIMENT_INVALID'),
                           (math.nan,'SENTIMENT_INVALID'),(math.inf,'SENTIMENT_INVALID'),(True,'SENTIMENT_INVALID')]:
            item=article();item['entities'][0]['sentiment_score']=value
            row=self.normalize(item);self.assertIsNotNone(row);self.assertIsNone(row.sentiment_score)
            self.assertEqual(self.diagnostics['reason_counts'],{code:1})
        item=article();item['entities'].append({**item['entities'][0],'sentiment_score':-.2})
        self.assertIsNone(self.normalize(item).sentiment_score)
        self.assertEqual(self.diagnostics['reason_counts'],{'SENTIMENT_AMBIGUOUS':1})

    def test_all_rejection_codes_are_sanitized(self):
        changes=[({'published_at':'invalid'},'DATE_INVALID'),
                 ({'published_at':(NOW-timedelta(hours=25)).isoformat()},'DATE_EXPIRED_24H'),
                 ({'published_at':(NOW+timedelta(minutes=1)).isoformat()},'DATE_FUTURE'),
                 ({'title':None},'TITLE_INVALID'),({'url':'file:///private/file'},'URL_INVALID'),
                 ({'entities':{}},'STRUCTURE_ERROR')]
        for values,code in changes:
            with self.subTest(code=code):
                self.assertIsNone(self.normalize({**article(),**values}))
                self.assertIn(code,self.diagnostics['reason_counts'])
        for key,value,code in [('symbol','ETHUSD','SYMBOL_INCORRECT'),('name','Other','BITCOIN_ENTITY_UNRECOGNIZED'),
                                ('exchange','NYSE','BITCOIN_ENTITY_UNRECOGNIZED')]:
            item=article();item['entities'][0][key]=value
            self.assertIsNone(self.normalize(item));self.assertIn(code,self.diagnostics['reason_counts'])
        self.assertIsNone(self.normalize(None));self.assertEqual(self.diagnostics['reason_counts'],{'STRUCTURE_ERROR':1})

    def fetch(self,items):
        client=MarketauxNewsClient('fixture-secret-only',min_match_score=10,allow_empty_filtered=True,
          transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'meta':{'found':len(items),'returned':len(items)},'data':items})))
        return client.fetch(NOW,timeout_seconds=6),client.diagnostics

    def test_http_200_discard_has_exact_code_and_never_copies_payload(self):
        item=article();item['title']='fixture-secret-only';item['url']='https://' + 'owner:fixture-secret-only@private.example.test/btc'
        rows,diagnostics=self.fetch([item])
        self.assertEqual(rows,[]);self.assertEqual(diagnostics['http_status'],200)
        self.assertEqual(diagnostics['rejected_articles'],1)
        self.assertEqual(diagnostics['reason_counts'],{'TITLE_INVALID':1,'URL_INVALID':1})
        safe=json.dumps(diagnostics)
        for text in ['fixture-secret-only','private.example.test','Bitcoin synthetic fixture','published_at']:
            self.assertNotIn(text,safe)

    def test_duplicate_is_counted_and_not_aggregated_twice(self):
        other=article();other['url']='https://fixture.example.test/other';other['entities'][0]['sentiment_score']=-.13115
        rows,diagnostics=self.fetch([article(),article(),other])
        self.assertEqual(len(rows),2);self.assertEqual(diagnostics['reason_counts']['ARTICLE_DUPLICATE'],1)
        view=news_view({'fetched_at':NOW.timestamp(),'observations':[
            {'id':r.article_identifier,'at':r.published_at.timestamp(),'score':r.sentiment_score} for r in rows]},NOW.timestamp())
        self.assertEqual(view['considered'],2);self.assertEqual(view['score'],0);self.assertEqual(view['classification'],'NEUTRAL')

    def test_24h_valid_does_not_mean_recent_or_live(self):
        old=article();old['published_at']=(NOW-timedelta(hours=2)).isoformat()
        rows,diagnostics=self.fetch([old])
        self.assertEqual(len(rows),1);self.assertEqual(diagnostics['accepted_within_24h'],1)
        self.assertEqual(diagnostics['recent_within_1h'],0);self.assertEqual(diagnostics['historical_1_to_24h'],1)
        view=news_view({'fetched_at':NOW.timestamp(),'observations':[
            {'id':rows[0].article_identifier,'at':rows[0].published_at.timestamp(),'score':rows[0].sentiment_score}]},NOW.timestamp())
        self.assertEqual(view['status'],'HISTORICO');self.assertIsNone(view['score'])

    def test_empty_and_missing_sentiment_are_na_not_neutral(self):
        for items in [[],[{**article(),'entities':[{**article()['entities'][0],'sentiment_score':None}]}]]:
            rows,diagnostics=self.fetch(items)
            view=news_view({'fetched_at':NOW.timestamp(),'observations':[
                {'at':r.published_at.timestamp(),'score':r.sentiment_score} for r in rows]},NOW.timestamp())
            self.assertEqual(view['status'],'N/A');self.assertIsNone(view['score'])
        rows,diagnostics=self.fetch([article()])
        self.assertEqual(diagnostics['recent_within_1h'],1)

    def test_malformed_response_has_structure_code(self):
        client=MarketauxNewsClient('synthetic',transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'data':[]})))
        with self.assertRaises(MarketauxUnavailable):client.fetch(NOW,timeout_seconds=6)
        self.assertEqual(client.diagnostics['reason_counts'],{'STRUCTURE_ERROR':1})

if __name__=='__main__':unittest.main()
