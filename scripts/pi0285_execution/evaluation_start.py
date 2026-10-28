"""Start/adopt one finite fixed endpoint through the existing Corelab entry."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def start(point):
    assert point in ('PLACE4096-1024','PLACE4096-2048','PLACE4096-4096','REPAIR_A2048-2048','SHARED4096-2048','SHARED4096-4096','REPAIR_B2048-2048')
    code=r'''import fcntl,json,os,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0285');point=__POINT__
with (r/'runtime/EVALUATION_START.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 command=[str(r.parent/'PI0250/runtime/venv/bin/python'),'-u',str(r/'code/evaluate.py'),'--point',point]
 active=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:tokens=[v.decode() for v in p.read_bytes().split(bytes([0])) if v]
  except (OSError,UnicodeError):continue
  if str(r/'code/evaluate.py') in tokens:active.append(dict(pid=int(p.parent.name),argv=tokens))
 if active:
  assert len(active)==1 and active[0]['argv']==command,('Other endpoint retains sole evaluation ownership',active)
  print(json.dumps(dict(status='ADOPT_ACTUAL_RUNNING_ENDPOINT',point=point,**active[0])));raise SystemExit(0)
 closed=r/'runtime'/(point+'_EVALUATION_CLOSED.json')
 if closed.exists():
  v=json.loads(closed.read_text());assert v['status']!='ENGINEERING_PENDING_ORIGINAL_RESULTS_PRESERVED',('Inspect original engineering error before restart',v)
  print(json.dumps(dict(status='ADOPT_COMPLETED_ENDPOINT',point=point,receipt=v)));raise SystemExit(0)
 record=r/'runtime'/(point+'_EVALUATION_START.json');assert not record.exists(),'Recorded process missing: reconcile exact original attempt'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PI0285_ROOT=str(r),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 env['PYTHONPATH']=':'.join([str(r/'code'),str(r.parent/'PI0283/code'),str(r.parent/'PI0280/official'),str(r/'project-src'),str(r/'project-src/src')])
 with (r/'runtime'/(point+'_EVALUATION_OWNER.log')).open('ab') as log:p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 v=dict(status='ACTUAL_CHILD_STARTED_NOT_QUALITY_COMPLETE',point=point,pid=p.pid,argv=command,started_epoch=time.time(),selected='NONE',deployment='NONE');record.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
'''.replace('__POINT__',repr(point))
    value=json.loads(remote('core',code));path=Path('reports/PI0285/r3/WP2')/(point+'_EVALUATION_START.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);a=p.parse_args();start(a.point)
