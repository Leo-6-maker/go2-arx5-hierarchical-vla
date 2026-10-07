"""Start/adopt one finite sealed phase, using the existing process identity check."""
import argparse,inspect,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote,phase_status
ROOT='/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2'

def call(script,args,native=False):
    target=ROOT+'/code/'+script
    phase=(args[-1]+'-'+args[0]) if script=='panel.py' else script.removesuffix('.py')+'-'+'-'.join(args)
    python='/home/liuyu-nfs/Projects/go2-arx5/PI0276/venv/bin/python' if native else '/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
    worker=f'''import json,os,subprocess,time
from pathlib import Path
root=Path({ROOT!r});target={target!r};args={args!r}
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',CUDA_VISIBLE_DEVICES={'0' if native else ''!r})
started=time.time()
with (root/'runtime'/{(phase+'.log')!r}).open('ab') as log:r=subprocess.run([{python!r},'-u',target,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
value=dict(returncode=r.returncode,target=target,arguments=args,wall_s=time.time()-started)
p=root/'runtime'/{(phase+'.done.json')!r};tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value)+'\\n');tmp.replace(p)
'''
    program=f'''import fcntl,json,subprocess,sys
from pathlib import Path
{inspect.getsource(phase_status)}
root=Path({ROOT!r});runtime=root/'runtime';runtime.mkdir(exist_ok=True)
start=runtime/{(phase+'.start.json')!r};done=runtime/{(phase+'.done.json')!r}
with (runtime/{(phase+'.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 value=phase_status(start,done,{target!r},{args!r},None)
 if value['status']=='NEW':
  argv=[sys.executable,'-u','-c',{worker!r}]
  with (runtime/{(phase+'.worker.log')!r}).open('ab') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start.write_text(json.dumps(dict(pid=p.pid,target={target!r},arguments={args!r},wrapper_argv=argv))+'\\n')
  value=dict(status='RUNNING',pid=p.pid,phase={phase!r})
print(json.dumps(value))
'''
    compile(worker,'finite-worker','exec');compile(program,'finite-launch','exec')
    value=json.loads(remote('core',program));print(json.dumps(value),flush=True);assert value['status']!='FAILED',value
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('--native',action='store_true');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    call(a.script,a.args,a.native)
