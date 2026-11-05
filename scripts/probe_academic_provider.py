"""One explicit protected operation on YOUR OWN academic backend; no automatic retry."""
import argparse
from getpass import getpass
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import urlsplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.northflank.academic_client import DemoClient

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin',required=True)
    parser.add_argument('--user',required=True)
    parser.add_argument('--action',choices=['context','explain','refresh-news'],default='context')
    parser.add_argument('--prediction-id')
    parser.add_argument('--expected-action',choices=['BUY','SELL','HOLD','WAIT'],default='WAIT')
    parser.add_argument('--confirm-own-service',action='store_true')
    parser.add_argument('--allow-one-provider-request',action='store_true')
    parser.add_argument('--confirm-free-plan',action='store_true')
    args=parser.parse_args()
    host=urlsplit(args.origin).hostname
    if not args.confirm_own_service or host in {'alpha-engine-frontend.onrender.com','alpha-engine-academic-api.onrender.com','alpha-engine-api-jggl.onrender.com'}:
        parser.error('Use your own verified backend, never the presentation services')
    if args.action!='context' and not (args.allow_one_provider_request and args.confirm_free_plan):
        parser.error('Explicit one-request authorization and verified free balance/plan required')
    if args.action!='refresh-news' and not re.fullmatch('[0-9a-f]{64}',args.prediction_id or ''):
        parser.error('Use a persisted prediction ID from your own database')
    client=DemoClient(args.origin);token=None
    try:
        password=getpass('Your private DEMO password: ')
        code,login=client.post('login',{'user':args.user,'password':password});del password
        if code!=200 or login.get('status')!='AUTHORIZED':raise RuntimeError('LOGIN_NOT_AUTHORIZED')
        token=login['access_token']
        payload={'prediction_id':args.prediction_id,'has_long':False,'historical':True}
        if args.action!='refresh-news':
            code,context=client.post('context',payload,token)
            if code!=200 or context.get('decision',{}).get('action')!=args.expected_action:
                raise RuntimeError('CONTROLLED_CONTEXT_NOT_VALIDATED_NO_GENERATION')
            if args.action=='context':
                print(json.dumps({'http':code,'status':context['status'],'action':args.expected_action,'provider_requests':0}));return
        started=perf_counter()
        code,result=client.post(args.action,{} if args.action=='refresh-news' else payload,token)
        report={'http':code,'status':result.get('status'),'seconds':round(perf_counter()-started,3),
                'maximum_protected_provider_operations':1,'automatic_retries':0,
                'model':result.get('model'),'cached':result.get('cached'),'provider_diagnostics':result.get('provider_diagnostics')}
        if args.action=='explain' and code==200:
            assert result['explanation']['action']==context['decision']['action']
            assert result['decision_hash']==context['decision_hash']
            report['deterministic_envelope_preserved']=True
        print(json.dumps(report,indent=2))
        if code!=200:raise SystemExit(1)
    finally:
        if token:
            try:client.post('logout',{},token)
            except Exception:pass

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'STOPPED_NO_RETRY','error_type':type(error).__name__}));raise SystemExit(1)
