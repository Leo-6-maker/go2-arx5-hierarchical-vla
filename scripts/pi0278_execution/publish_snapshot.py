"""Close an actual screen, then deliver the newest retained unscreened milestone."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

def publish(previous):
    assert previous in [f'COVER93{n}' for n in (512,1024,2048,4096)]
    proof=json.loads(remote('core',f'''
import hashlib,json
from pathlib import Path
root=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2');fast=root.parent/'R3/evidence'/('FAST_'+{previous!r}+'_TRAIN_PHYSICS.json')
v=json.loads(fast.read_text());assert all(v['checks'].values()) and v['task']['denominator']==4
done=root/'runtime'/({previous!r}+'-TRAIN_PHYSICS.done.json');assert json.loads(done.read_text())['returncode']==0
fit=root/'evidence'/('FIT16_'+{previous!r}+'.json');f=json.loads(fit.read_text());assert f['actual_rows']==48 and len(f['results'])==48
for row in f['results']:assert hashlib.sha256(Path(row['output']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(status='ACTUAL_LOCAL_SCREEN_COMPLETE',point={previous!r},fast_sha256=hashlib.sha256(fast.read_bytes()).hexdigest(),fit_sha256=hashlib.sha256(fit.read_bytes()).hexdigest(),persistent_stacks=v['task']['persistent_stacks'],full_S4_replay=v['full_S4_replay'])))
'''))
    result=remote('hop',f'''
import ast,fcntl,hashlib,json,os
from pathlib import Path
root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(value,sort_keys=True)+chr(10));tmp.replace(p)
with (root/'runtime/publish.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 marker=root/'snapshots'/{previous!r}/'SCREEN_COMPLETE.json';value={proof!r}
 if marker.exists():assert json.loads(marker.read_text())==value
 else:write(marker,value)
 pending=[p.parent.name for p in (root/'snapshots').glob('*/READY.json') if not p.with_name('SCREEN_COMPLETE.json').exists()]
 if pending:assert len(pending)==1;point=pending[0]
 else:
  steps=[n for n in (1024,2048,4096) if (root/'COVER93/main/training'/('checkpoint-'+str(n))/'PI0277_RUN.json').exists() and not (root/'snapshots'/('COVER93'+str(n))/'SCREEN_COMPLETE.json').exists()]
  if steps:
   step=max(steps);point='COVER93'+str(step);checkpoint=root/'COVER93/main/training'/('checkpoint-'+str(step));run=json.loads((checkpoint/'PI0277_RUN.json').read_text());assert run['release']=='83af9099f3bab9a4ac08eaa85380c2cfd6351f34' and run['arm']=='COVER93'
   node=next(n for n in ast.parse((root/'code/persistence.py').read_text()).body if getattr(n,'name',None)=='snapshot')
   exec(compile(ast.Module(body=[node],type_ignores=[]),'qualified_snapshot','exec'));snapshot(root,'COVER93',step,checkpoint)
   for n in (1024,2048):
    if n<step and not (root/'snapshots'/('COVER93'+str(n))/'READY.json').exists():write(root/'evidence'/('COVER93'+str(n)+'_UNMEASURED_BACKLOG_SKIP.json'),dict(status='EXPLICIT_UNMEASURED_SUPERSEDED_MILESTONE_SKIP',point='COVER93'+str(n),superseded_by=point,valid_results_removed=False,inflight_episode_removed=False))
  else:point=None
 print(json.dumps(dict(screen=value,next_point=point)))
''')
    print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous');publish(p.parse_args().previous)
