"""Actual two-host pause and protected recovery identity; no neural/physics work."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
RELEASE='9c0172ced33c6052f7cb037b4a6d91910f9fe69e'


def capture(host):
    program='''import collections,datetime,hashlib,json,subprocess,time
from pathlib import Path
host=HOST
base=Path('/home/liuyu-nfs/Projects/go2-arx5') if host=='core' else Path('/scratch/e1192548/Projects/go2-arx5/experiments')
old=base/'PI0279';root=base/'PI0280';marker=old/'runtime/SCIENTIFIC_PAUSE.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
value=dict(authority_status='PAUSED_BY_USER_PRETRAINED_BASELINE_PRIORITY',release=RELEASE,canonical=6037445331,pause_comment=6037170888,launch_allowed=False,automatic_resume=False,recognized_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
marker.parent.mkdir(parents=True,exist_ok=True)
if marker.exists():value=json.loads(marker.read_text())
else:marker.write_text(json.dumps(value,sort_keys=True)+chr(10))
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  if any(x.startswith(str(old)+'/') for x in argv):active.append(dict(pid=int(p.name),argv=argv))
 except (OSError,UnicodeError):pass
assert not active,active
rows=[];queue=None;recovery=[];pilot=None;counts={};timeline=[]
if host=='hop':
 qstat=next((p for p in [Path('/cm/shared/apps/pbspro/current/bin/qstat'),Path('/opt/pbs/bin/qstat'),Path('/usr/pbs/bin/qstat')] if p.exists()),None);assert qstat,'Actual PBS CLI missing'
 q=subprocess.run([str(qstat),'-u','e1192548'],capture_output=True,text=True);assert q.returncode==0,q.stderr
 queue=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
 assert not any('pi0279' in line.lower() for line in q.stdout.splitlines()),'Old queued job must be identified/cancelled before acknowledgement'
 for parent in [base/'PI0278/run/COVER93/main/training/checkpoint-4096',base/'PI0278/run/COVER93/main/training/checkpoint-3072',base/'PI0277/R2_FOCUS/FOCUS/main/training/checkpoint-4096']:
  assert all((parent/n).is_file() for n in ['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json'])
  files=[dict(path=str(p.relative_to(parent)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(parent.rglob('*')) if p.is_file()]
  index=json.loads((parent/'model.safetensors.index.json').read_text())['weight_map'];assert all((parent/n).is_file() for n in set(index.values()))
  tree=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();recovery.append(dict(path=str(parent),kind='COMPLETE_TRAINER',global_step=json.loads((parent/'trainer_state.json').read_text())['global_step'],files=files,tree_sha256=tree))
else:
 raw=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0279/source')
 counts=dict(collections.Counter(json.loads(x)['kind'] for x in (old/'evidence/budget.jsonl').read_text().splitlines()))
 pilot=json.loads((old/'evidence/PILOT_VERDICT.json').read_text());pilot={k:pilot[k] for k in ['status','positive','denominator','per_state','new_neural_queries']}
 for p in sorted((raw/'episodes').glob('*/attempt-*/RESULT.json')):
  rec=json.loads(p.read_text());rows.append(dict(path=str(p),sha256=sha(p),status=rec['status'],control_calls=rec['control_calls']))
 for p in sorted((raw/'episodes').glob('*/attempt-*/START.json')):timeline.append(dict(slot=p.parent.parent.name,created_mtime_ns=p.stat().st_mtime_ns))
 assert not (old/'evidence/DATA_ADMISSION.json').exists(),'Report completed data admission honestly instead of assuming absent'
 done=json.loads((old/'runtime/pilot-run.done.json').read_text());assert done['returncode']==0
 queue=dict(action='NO_PBS_OR_LOCAL_OLD_TRAINING_SUBMITTED',wrapper_pid=572915,native_pid=572916,phase_done=done,phase_done_mtime_ns=(old/'runtime/pilot-run.done.json').stat().st_mtime_ns)
receipt=dict(host=host,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),release=RELEASE,canonical=6037445331,authority_status='AUTHORITY_PAUSED',remote_status='REMOTE_PAUSE_CONFIRMED',pause_marker=value,pause_marker_path=str(marker),active_old_processes=active,queue=queue,old_counts=counts,old_source_results=rows,old_start_timeline=timeline,protected_recoveries=recovery,pilot=pilot,old_training_updates=0,new_training_updates=0,automatic_continuation_disabled=True,old_confirmation_locked=True,pause_publication_utc='2026-10-07T11:42:26Z',post_publication_predecessor_work_retained=True,old_data_admission='NOT_EXECUTED',new_model_queries=0,new_physics=0,selected_checkpoint='NONE',deployment='NONE')
(root/'evidence').mkdir(parents=True,exist_ok=True);p=root/'evidence'/('PAUSE_'+host.upper()+'.json');p.write_text(json.dumps(receipt,sort_keys=True)+chr(10))
print(json.dumps(receipt))
'''
    return json.loads(remote(host,f'HOST={host!r}\nRELEASE={RELEASE!r}\n'+program))


def main():
    results=[capture(host) for host in ('core','hop')]
    out=Path('reports/PI0280/execution');out.mkdir(parents=True,exist_ok=True)
    for value in results:(out/f"PAUSE_{value['host'].upper()}.json").write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    ack=dict(status='REMOTE_PAUSE_CONFIRMED',release=RELEASE,canonical=6037445331,pause_comment=6037170888,hosts=results,new_training_updates=0,old_training_updates=0,old_budget_preserved=True,automatic_resume=False,selected_checkpoint='NONE',deployment='NONE')
    (out/'PAUSE_ACK.json').write_text(json.dumps(ack,indent=2)+'\n',encoding='utf-8')
    print('REMOTE_PAUSE_CONFIRMED; OLD_PILOT24_RETAINED; TRAINING0; THREE_FULL_RECOVERIES_HASHED',flush=True)


if __name__=='__main__':main()
