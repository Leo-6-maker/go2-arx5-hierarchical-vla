"""One exact registered panel process; reuse its result or live PID."""
import argparse,hashlib,inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pi0276_execution'))
from coordinator import remote,phase_status
ROOT='/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2'

def call(point,stage):
    assert point in [f'{a}{n}' for a in ('GRIP','FOCUS') for n in (512,1024,2048,4096)]
    replay=stage.startswith(('replay-','delivery-'));lane=stage.removeprefix('replay-').removeprefix('delivery-')
    assert lane in ('admit','TRAIN_PHYSICS','TEST','CONFIRM') and not (replay and lane=='admit')
    target=ROOT+'/code/'+('deliver_panel.py' if stage.startswith('delivery-') else 'replay_panel.py' if replay else 'panel.py')
    args=[point,lane] if replay else [lane,'--point',point]
    python='/home/liuyu-nfs/Projects/go2-arx5/'+('PI0250/runtime/venv/bin/python' if replay or lane=='admit' else 'PI0276/venv/bin/python')
    name=point+'-'+stage
    worker=f'''import json,os,subprocess
from pathlib import Path
r=Path({ROOT!r})/'runtime'
env=dict(os.environ,CUDA_VISIBLE_DEVICES={('' if replay or lane=='admit' else '0')!r},OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
with (r/{(name+'.log')!r}).open('ab') as log:
 result=subprocess.run({[python,'-u',target,*args]!r},env=env,stdout=log,stderr=subprocess.STDOUT)
p=r/{(name+'.done.json')!r};tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(returncode=result.returncode,target={target!r},arguments={args!r}))+chr(10));tmp.replace(p)
'''
    program=f'''import fcntl,json,subprocess,sys
from pathlib import Path
{inspect.getsource(phase_status)}
r=Path({ROOT!r})/'runtime';r.mkdir(exist_ok=True)
start=r/{(name+'.start.json')!r};done=r/{(name+'.done.json')!r}
with (r/{(name+'.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=phase_status(start,done,{target!r},{args!r},None)
 if state['status']=='NEW':
  argv=[sys.executable,'-u','-c',{worker!r}]
  with (r/{(name+'.worker.log')!r}).open('ab') as log:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  tmp=start.with_suffix('.partial');tmp.write_text(json.dumps(dict(pid=child.pid,target={target!r},arguments={args!r},wrapper_argv=argv))+chr(10));tmp.replace(start)
  state=dict(status='RUNNING',pids=[child.pid])
 print(json.dumps(state))
'''
    print(remote('core',program),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('stage');a=p.parse_args();call(a.point,a.stage)
