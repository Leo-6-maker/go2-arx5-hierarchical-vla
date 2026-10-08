"""Submit exactly the remaining PLACE segment after both earlier fixed gates close."""
import argparse,base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def allowed(routes,candidate):
    assert not candidate,'Preserve the unique early candidate; do not extend PLACE'
    assert set(routes)=={'PLACE4096-1024','PLACE4096-2048'}
    for point,v in routes.items():
        assert v['point']==point and v['next_action']=='CONTINUE_SAME_PHASE_4096'
        assert v['proofs']['TRAIN'] and v['confirmation'] is None


def submit():
    payload=json.loads(remote('core',r'''import base64,json,hashlib
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0285');routes={};files={}
for point in ('PLACE4096-1024','PLACE4096-2048'):
 p=r/'qualification'/(point+'_ROUTE.json');v=json.loads(p.read_text());routes[point]=v
 assert json.loads((r/'runtime'/(point+'_EVALUATION_CLOSED.json')).read_text())['status']=='CONTINUE_SAME_PHASE_4096'
 for proof in v['proofs'].values():
  if proof is None:continue
  for role in ('source','CPU'):
   p=Path(proof[role+'_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==proof[role+'_sha256'];files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
 files['qualification/'+point+'_ROUTE.json']=base64.b64encode((r/'qualification'/(point+'_ROUTE.json')).read_bytes()).decode()
print(json.dumps(dict(routes=routes,candidate=(r/'qualification/CONFIRM_A_CANDIDATE.json').exists(),files=files)))
'''))
    allowed(payload['routes'],payload['candidate'])
    code=r'''import fcntl,json,base64,re,time,hashlib,subprocess
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285');record=r/'runtime/PLACE4096_RESUME2048_REQUEST.json'
with (r/'runtime/SUBMISSION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if record.exists():
  v=json.loads(record.read_text());assert v['status']=='ACCEPTED','Reconcile original nonaccepted request before another submission'
  print(json.dumps(dict(status='ADOPT_RECORDED_RESUME_NO_NEW_QSUB',receipt=v)));raise SystemExit(0)
 q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-fx','655273'],capture_output=True,text=True)
 assert q.returncode==0 and re.search(r'job_state\s*=\s*F\b',q.stdout)
 assert json.loads((r/'runtime/recovery/2048-finalizer-JSON/RECONCILIATION.json').read_text())['status']=='PASS_READ_ONLY_FINALIZER_RECHECK'
 assert json.loads((r/'runtime/PLACE4096-main-0-2048.done.json').read_text())['returncode']==0
 audit=json.loads((r/'evidence/PLACE4096-2048_TRAINING_CPU.json').read_text());assert audit['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
 assert audit['checkpoint']==str(r/'learning/PLACE4096/main/checkpoint-2048')
 assert json.loads((r/'learning/PLACE4096/CANARY_COMPARISON.json').read_text())['status']=='PASS_FULL_RESUME_CANARY'
 rows=[json.loads(x) for x in (r/'learning/completed_updates.jsonl').read_text().splitlines()]
 assert sum(x['work_kind']=='formal' for x in rows)==2048 and not (r/'runtime/PLACE4096-main-2048-4096.start.json').exists()
 for n,b in __FILES__.items():
  p=r/'handoff/Corelab_fixed_gate'/n;p.parent.mkdir(parents=True,exist_ok=True);blob=base64.b64decode(b)
  if p.exists():assert p.read_bytes()==blob
  else:p.write_bytes(blob)
 p=r/'code/train_resume2048.pbs';text=(r/'code/train.pbs').read_text();assert text.count('--stop 2048')==1
 text=text.replace('--stop 2048','--resume 2048 --stop 4096');assert '#PBS -q auto' in text and '#PBS -m n' in text and 'gpu_model=H200' in text
 p.write_text(text);v=dict(status='RECORDED_BEFORE_SOLE_QSUB',requested_epoch=time.time(),resume=2048,stop=4096,card_sha256=hashlib.sha256((r/'design/PLACE4096_CARD.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),existing_canary_reused=True,formal_updates_repeated=0,early_routes=__ROUTES__,selected='NONE',deployment='NONE')
 record.write_text(json.dumps(v,indent=2)+'\n')
 result=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub','-P','CFP05-CF-052',str(p)],capture_output=True,text=True)
 v.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,status='ACCEPTED' if result.returncode==0 else 'REJECTED_ENGINEERING_NO_UPDATES')
 if result.returncode==0:
  v['PBS_JOBID']=result.stdout.strip();assert v['PBS_JOBID'].split('.')[0].isdigit()
 record.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));assert result.returncode==0
'''.replace('__FILES__',repr(payload['files'])).replace('__ROUTES__',repr(payload['routes']))
    v=json.loads(remote('hop',code));p=Path('reports/PI0285/r3/WP1/RESUME2048_REQUEST.json');p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');print(json.dumps(v))


def self_check():
    rows={p:dict(point=p,next_action='CONTINUE_SAME_PHASE_4096',proofs=dict(TRAIN={'source_sha256':'fixed'}),confirmation=None) for p in ('PLACE4096-1024','PLACE4096-2048')}
    allowed(rows,False)
    for r,c in [(rows,True),({k:v for k,v in rows.items() if k.endswith('1024')},False)]:
        try:allowed(r,c)
        except AssertionError:pass
        else:raise AssertionError('Missing early-gate or candidate guard')
    print('PASS_EARLY_FIXED_GATES_AND_UNIQUE_CANDIDATE_RESUME_GUARD; model/updates/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');a=p.parse_args();self_check() if a.self_check else submit()
