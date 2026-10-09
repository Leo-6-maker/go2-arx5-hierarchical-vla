"""One read-only fixed-endpoint metadata snapshot; no monitor, model or launch."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def read(point):
    assert point in ('C0_REPLAY1024-1024','C1_CORRECT1024-1024')
    code=r'''
import json,time,hashlib
from pathlib import Path
R=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');point=POINT
arm=point.rsplit('-',1)[0];p=R/'delivery'/point/'READY.json';a=R/'delivery/audits'/(point+'.json')
v=dict(epoch=time.time(),point=point,READY=p.exists(),TRAINING_CPU=a.exists(),phases={})
for suffix in ('continuous-0-8','split-0-4','split-4-8','CANARY-continuous','CANARY-split','main-0-1024'):
 n=arm+'-'+suffix;d=R/'runtime'/(n+'.done.json');b=R/'runtime'/(n+'.start.json')
 if d.exists():
  z=json.loads(d.read_text());v['phases'][n]=dict(returncode=z['returncode'],elapsed_s=z['elapsed_s'])
 elif b.exists():v['phases'][n]=dict(started=True,pid=json.loads(b.read_text())['pid'])
if p.exists() and a.exists():
 ready=json.loads(p.read_text());audit=json.loads(a.read_text());exp=R/'delivery'/point/'EXPORT.json';h=hashlib.sha256(exp.read_bytes()).hexdigest()
 assert ready['status']=='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT' and ready['point']==point and not ready['optimizer_transfer']
 assert audit['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE' and audit['point']==point and audit['export_sha256']==h and audit['checkpoint']==ready['full_recovery']
 checkpoint=Path(ready['full_recovery']);assert checkpoint.resolve()==(R/'learning'/arm/'main/checkpoint-1024').resolve()
 recovery=json.loads((checkpoint/'PI0282_RECOVERY.json').read_text());assert recovery['cursor']==8192
 assert json.loads((R/'learning'/arm/'CANARY_COMPARISON.json').read_text())['status']=='PASS_FULL_RESUME_CANARY'
 v['endpoint_actual_ready_CPU_PASS']=True;v['READY_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();v['TRAINING_CPU_sha256']=hashlib.sha256(a.read_bytes()).hexdigest();v['EXPORT_sha256']=h
print(json.dumps(v))
'''.replace('POINT',repr(point),1)
    return json.loads(remote('hop',code))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',default='C0_REPLAY1024-1024');a=p.parse_args();v=read(a.point)
    out=Path(__file__).resolve().parents[2]/'reports/PI0286/preparation'/('C0_READY_DISCOVERY_LATEST.json' if a.point.startswith('C0_') else 'C1_READY_DISCOVERY_LATEST.json')
    out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
