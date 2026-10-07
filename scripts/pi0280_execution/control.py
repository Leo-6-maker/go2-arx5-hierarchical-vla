"""Finite, resumable Corelab commands using the existing SSH/process helper."""
import argparse,base64,inspect,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote,phase_status
ROOT='/home/liuyu-nfs/Projects/go2-arx5/PI0280'
PYTHON='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'

def run(phase):
    args=[phase];target=ROOT+'/code/native.py';name='native-'+phase
    program=f'''import fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
{inspect.getsource(phase_status)}
r=Path({ROOT!r});runtime=r/'runtime';runtime.mkdir(parents=True,exist_ok=True)
assert json.loads((r.parent/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
start=runtime/{(name+'.start.json')!r};done=runtime/{(name+'.done.json')!r}
with (runtime/{(name+'.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 value=phase_status(start,done,{target!r},{args!r},None)
 if value['status']=='NEW':
  seal=json.loads((r/'design/IMPLEMENTATION.json').read_text())
  for name,sha in seal['files'].items():assert hashlib.sha256((r/'code'/name).read_bytes()).hexdigest()==sha,name
  worker="""import json,os,subprocess,time
from pathlib import Path
r=Path({ROOT!r});begin=time.time()
env=dict(os.environ,CUDA_VISIBLE_DEVICES={'0' if phase=='reproduce' else ''!r},PYTHONPATH=str(r/'official'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HOME='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/hf_home',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (r/'runtime'/{(name+'.log')!r}).open('ab') as log:result=subprocess.run([{PYTHON!r},'-u',{target!r},*{args!r}],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=12*3600)
p=r/'runtime'/{(name+'.done.json')!r};p.write_text(json.dumps(dict(target={target!r},arguments={args!r},returncode=result.returncode,wall_s=time.time()-begin))+'\\n')
"""
  argv=[sys.executable,'-u','-c',worker]
  with (runtime/{(name+'.worker.log')!r}).open('ab') as log:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start.write_text(json.dumps(dict(pid=child.pid,target={target!r},arguments={args!r},wrapper_argv=argv))+'\\n');value=dict(status='RUNNING',pid=child.pid)
print(json.dumps(value))
'''
    print(remote('core',program))

def stage():
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in Path(__file__).parent.glob('*.py')}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    print(remote('core',f'''import base64,hashlib,json
from pathlib import Path
r=Path({ROOT!r});(r/'code').mkdir(exist_ok=True);(r/'design').mkdir(exist_ok=True)
assert not list((r/'runtime').glob('native-*.start.json')),'Adopt or close actual phase before restaging'
hashes={{}}
for name,data in {files!r}.items():
 b=base64.b64decode(data);(r/'code'/name).write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
(r/'design/IMPLEMENTATION.json').write_text(json.dumps(dict(commit={commit!r},files=hashes,training_updates=0))+'\\n')
print('PI0280_CODE_STAGED')
'''))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['stage','prepare','reproduce']);a=p.parse_args()
    stage() if a.phase=='stage' else run(a.phase)
