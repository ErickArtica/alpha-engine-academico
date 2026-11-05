"""Read-only verification of this clone's local Compose stack. No provider calls."""
import argparse
import http.client
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
QUERY="SELECT json_build_object('candles',(SELECT count(*) FROM candles),'predictions',(SELECT count(*) FROM predictions),'evaluations',(SELECT count(*) FROM live_evaluations),'runs',(SELECT count(*) FROM pipeline_runs),'bootstrap',(SELECT count(*) FROM local_bootstrap_runs),'latest_prediction',(SELECT max(generated_at) FROM predictions))"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-scheduler',action='store_true')
    args=parser.parse_args()
    config=dict(line.split('=',1) for line in (ROOT/'.env').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#') and '=' in line)
    project=config['COMPOSE_PROJECT_NAME']
    command=['docker','compose','--project-name',project,'--env-file','.env']
    def execute(*extra):
        result=subprocess.run(command+list(extra),cwd=ROOT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('LOCAL_CONTAINER_QUERY_FAILED; inspect this project only')
        return result.stdout.strip()
    def data():return json.loads(execute('exec','-T','postgres','psql','-U','alpha_engine','-d','alpha_engine','-Atc',QUERY))
    report={'project':project,'provider_generations':0,'cloud_credentials_required':False}
    for name,port,path in [('api',int(config['LOCAL_API_PORT']),'/health'),('streamlit',int(config['LOCAL_STREAMLIT_PORT']),'/_stcore/health')]:
        deadline=time.monotonic()+180
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}'+path,timeout=5) as response:status=response.status
                if status==200:break
            except (OSError,http.client.HTTPException):pass
            if time.monotonic()>=deadline:raise RuntimeError('LOCAL_HEALTH_TIMEOUT_'+name)
            time.sleep(2)
        report[name+'_http']=status
    report['initial']=data()
    assert report['initial']['bootstrap']==1 and report['initial']['candles']>=120 and report['initial']['predictions']>0 and report['initial']['evaluations']>0,'NO_BOOTSTRAP_DATA: inspect bootstrap logs and Binance access'
    model_code="import json;from deploy.render.app import verify_model_artifact;from alpha_engine.config.settings import get_settings;s=get_settings();verify_model_artifact();m=json.load(open('/app/models/xgboost_v1_metadata.json'));assert s.storage_provider=='postgresql' and s.model_confidence_threshold==.60 and len(m['features'])==8;print('MODEL_V1_EIGHT_FEATURES_GATE_060_PASS')"
    report['model']=execute('exec','-T','api','python','-c',model_code)
    request=urllib.request.Request(f"http://127.0.0.1:{int(config['LOCAL_API_PORT'])}/internal/run-cycle",data=b'{}',headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=10) as response:status=response.status
    except urllib.error.HTTPError as error:status=error.code
    assert status==401,'UNAUTHORIZED_CYCLE_NOT_REJECTED'
    report['unauthorized_cycle_http']=status
    if args.observe_scheduler:
        deadline=time.monotonic()+160
        while time.monotonic()<deadline:
            after=data()
            if after['runs']>report['initial']['runs'] and after['predictions']>report['initial']['predictions']:break
            time.sleep(5)
        report['after_scheduler']=data()
        assert report['after_scheduler']['runs']>report['initial']['runs'] and report['after_scheduler']['predictions']>report['initial']['predictions'],'SCHEDULER_DID_NOT_CREATE_NEW_PREDICTION'
    report['status']='PASS'
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error) if isinstance(error,(RuntimeError,AssertionError)) else type(error).__name__}))
        raise SystemExit(1)
