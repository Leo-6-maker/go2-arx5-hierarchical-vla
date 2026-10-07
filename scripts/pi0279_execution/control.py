"""Stage and adopt finite source work through the existing SSH/process helpers."""
import argparse,base64,inspect,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote,phase_status
ROOT='/home/liuyu-nfs/Projects/go2-arx5/PI0279'
RELEASE='4afdb1401650353092428dcb69d0cb353ee74164'


def stage():
    files={p.resolve().relative_to(Path.cwd()).as_posix():base64.b64encode(p.read_bytes()).decode() for p in [*Path(__file__).parent.glob('*.py'),Path('docs/protocols/PI0279_SINGLE_TASK_RELEASE_CLEAR_FEEDBACK_FINETUNING_V1.md'),Path('reports/PI0279/design/REVIEW_AND_EXECUTION_LIMITS.json')]}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    script='''import base64,hashlib,json,os
from pathlib import Path
root=Path(ROOT);old=root.with_name('PI0278')/'R2';raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0279/source')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def freeze(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,sort_keys=True)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==data,str(p)
 else:p.write_bytes(data)
assert not list((root/'runtime').glob('*.start.json')),'Adopt actual phase before restaging'
for n in ['code','authority','design','evidence','runtime']:(root/n).mkdir(parents=True,exist_ok=True)
for name,b in FILES.items():
 p=root/('code' if name.startswith('scripts/') else 'authority')/Path(name).name;data=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==data,str(p)
 else:p.write_bytes(data)
for n in ['project-src','ManiSkill']:
 p=root/n;target=root.with_name('PI0276')/n
 if not p.exists():p.symlink_to(target,target_is_directory=True)
 assert p.resolve()==target.resolve()
seal=read(old/'design/EVALUATION_FILESET.json');selected=[r for r in seal['files'] if Path(r['path']).name in ['source_worker.py','evaluate.py','replay.py','evaluation_card.py']]
for r in selected:assert sha(r['path'])==r['sha256'],r['path']
history=root.with_name('PI0277')/'design/SOURCE_FILESET.json';reconciliation=root.with_name('PI0277')/'R3/evidence/PREDECESSOR_SOURCE_MANIFEST_RECONCILIATION.json'
prior=read(history);native=[]
for r in prior['files']:
 if Path(r['path']).name in ['run.py','source_worker.py'] and '/PI0276/code/' in r['path']:continue
 assert sha(r['path'])==r['sha256'],r['path'];native.append(r)
freeze(root/'design/SOURCE_FILESET.json',dict(files=native+selected,historical_manifest=str(history),historical_manifest_sha256=sha(history),reconciliation=str(reconciliation),reconciliation_sha256=sha(reconciliation),original_manifest_preserved=True))
fileset=list((root/'code').glob('*.py'))+list((root/'authority').glob('*'))+[root/'design/SOURCE_FILESET.json']
rows=[dict(path=str(p),sha256=sha(p)) for p in fileset]+selected
freeze(root/'design/ADOPTION.json',dict(release=RELEASE,canonical=6036062568,implementation_commit=COMMIT,source_files=rows,old_terminal_preserved=True,formal_updates=0,selected_checkpoint='NONE',deployment='NONE'))
(raw/'evidence').mkdir(parents=True,exist_ok=True);ledger=root/'evidence/budget.jsonl';ledger.touch(exist_ok=True)
link=raw/'evidence/budget.jsonl'
if not link.exists():link.symlink_to(ledger)
assert link.resolve()==ledger.resolve()
print(json.dumps(dict(status='STAGED_ACTUAL_CURRENT_SEAL',implementation_commit=COMMIT,protected_files=len(native),root=str(root),adoption_sha256=sha(root/'design/ADOPTION.json'),new_physics=0,new_neural_queries=0)))
'''
    prefix='\n'.join(f'{k}={v!r}' for k,v in [('ROOT',ROOT),('RELEASE',RELEASE),('COMMIT',commit),('FILES',files)])
    print(remote('core',prefix+'\n'+script))


def call(args,native=False,script='pilot.py'):
    assert script in ('pilot.py','data.py','source_replay.py')
    target=ROOT+'/code/'+script;phase=script.removesuffix('.py')+'-'+'-'.join(args)
    python='/home/liuyu-nfs/Projects/go2-arx5/PI0276/venv/bin/python' if native else '/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
    worker=f'''import json,os,subprocess,time
from pathlib import Path
root=Path({ROOT!r});started=time.time()
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',CUDA_VISIBLE_DEVICES={'0' if native else ''!r})
with (root/'runtime'/{(phase+'.log')!r}).open('ab') as log:r=subprocess.run([{python!r},'-u',{target!r},*{args!r}],env=env,stdout=log,stderr=subprocess.STDOUT)
value=dict(returncode=r.returncode,target={target!r},arguments={args!r},wall_s=time.time()-started)
p=root/'runtime'/{(phase+'.done.json')!r};tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value)+'\\n');tmp.replace(p)
'''
    program=f'''import fcntl,json,subprocess,sys
from pathlib import Path
{inspect.getsource(phase_status)}
root=Path({ROOT!r});runtime=root/'runtime';start=runtime/{(phase+'.start.json')!r};done=runtime/{(phase+'.done.json')!r}
with (runtime/{(phase+'.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 value=phase_status(start,done,{target!r},{args!r},None)
 if value['status']=='NEW':
  argv=[sys.executable,'-u','-c',{worker!r}]
  with (runtime/{(phase+'.worker.log')!r}).open('ab') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start.write_text(json.dumps(dict(pid=p.pid,target={target!r},arguments={args!r},wrapper_argv=argv))+'\\n');value=dict(status='RUNNING',pid=p.pid,phase={phase!r})
print(json.dumps(value))
'''
    compile(program,'finite-launch','exec');compile(worker,'finite-worker','exec')
    value=json.loads(remote('core',program));print(json.dumps(value));assert value['status']!='FAILED',value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage','check','run','audit','verdict','build','codec','replay','views']);a=p.parse_args()
    if a.stage=='stage':stage()
    elif a.stage in ('build','codec','views'):call([a.stage],script='data.py')
    elif a.stage=='replay':call([],script='source_replay.py')
    else:call([a.stage],a.stage=='run')
