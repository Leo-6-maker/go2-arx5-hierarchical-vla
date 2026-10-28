"""One original reserved DOSE phase, after actual parent and capacity closure."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

PROGRAM=r'''
import fcntl,json,time,subprocess,hashlib,re,shlex
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (r/'runtime/SUBMISSION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 record=r/'runtime/DOSE_JOB_SUBMISSION.json'
 if record.exists():
  old=json.loads(record.read_text());assert old['status']=='ACCEPTED',('Reconcile original request before another submission',old)
  print(json.dumps(dict(status='ADOPT_EXISTING_ACCEPTED_REQUEST_NO_NEW_QSUB',record=old)));raise SystemExit(0)
 reclaim=r/'archive/DOSE_CAPACITY_RECLAIM_EXECUTED.json';capacity=json.loads(reclaim.read_text())
 assert capacity['status']=='EXACT_RETIREMENT_COMPLETE_CURRENT4096_PRESERVED' and capacity['logical_deleted_bytes']>57_000_000_000
 oldjob='655301.hopper-m-02';q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-fx',oldjob],capture_output=True,text=True)
 assert q.returncode==0 and re.search(r'job_state\s*=\s*F\b',q.stdout) and re.search(r'Exit_status\s*=\s*0\b',q.stdout)
 assert json.loads((r/'runtime'/('ALLOCATION_CLOSED_'+oldjob+'.json')).read_text())['status']=='FIXED_SEGMENT_COMPLETE'
 now=time.time();assert now<1791498600 # original06:30 new-phase cutoff
 # Measured2048 segment1973.925s and TRAIN episodep90<=40s; conservative actual-next-gate reserve.
 reserve=dict(canary_and_load_save_s=1200,training_s=2400,DIAG_CPU_s=900,TRAIN8_DEV16_CPU_s=1800,CONFIRM_A_s=2700)
 assert now+sum(reserve.values())<1791510300 and now+3600<1791505800
 ledger=[json.loads(x) for x in (r/'learning/completed_updates.jsonl').read_text().splitlines()] if (r/'learning/completed_updates.jsonl').exists() else None
 assert ledger is not None,'Bind the actual cumulative ledger, never default zero'
 assert sum(x['work_kind']=='formal' for x in ledger)==4096 and sum(x['work_kind']=='canary' for x in ledger)==16
 prospective=r/'data/DOSE_PREPARED/REPAIR_A2048_CARD.prospective.json';v=json.loads(prospective.read_text())
 assert not v['launch_allowed'] and v['branch']=='DOSE2048' and v['parent_global_update']==4096
 card=r/'design/REPAIR_A2048_CARD.json'
 v.update(launch_allowed=True,status='OWNER_ADMITTED_ORIGINAL_SINGLE_DOSE_PHASE_ACTUAL_PARENT_AND_CAPACITY',capacity_receipt=str(reclaim),capacity_receipt_sha256=sha(reclaim),time_reserve=reserve,admission_epoch=now,canary_status='ACTUAL_NEW_PARENT_CANARY_REQUIRED_BEFORE_MAIN')
 if card.exists():
  saved=json.loads(card.read_text());assert all(saved[k]==v[k] for k in v if k!='admission_epoch');v=saved
 else:card.write_text(json.dumps(v,indent=2)+'\n')
 shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+shlex.quote("from pathlib import Path;from contract import guard;from authority import stage_card;guard();r=Path("+repr(str(r))+");stage_card(r,r/'design/REPAIR_A2048_CARD.json');print('PASS_ACTUAL_BRANCH_ENTRY_CPU_ONLY')")
 pre=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True);assert pre.returncode==0,(pre.stdout,pre.stderr[-5000:])
 p=r/'code/train_dose.pbs';assert not p.exists();text=(r/'code/train.pbs').read_text().replace('pi0285_place','pi0285_dose').replace('PLACE4096_CARD.json','REPAIR_A2048_CARD.json').replace('--arm PLACE4096','--arm REPAIR_A2048')
 assert '#PBS -q auto' in text and '#PBS -m n' in text and 'gpu_model=H200' in text and '--arm REPAIR_A2048 --stop 2048' in text and '#PBS -A' not in text
 p.write_text(text)
 request=dict(status='REQUEST_RECORDED_BEFORE_SOLE_QSUB',requested_epoch=time.time(),script=str(p),script_sha256=sha(p),card=str(card),card_sha256=sha(card),branch='DOSE2048',arm='REPAIR_A2048',parent_global_update=4096,new_formal_updates=2048,new_canary_updates=16,project='CFP05-CF-052',queue='auto',mail='n',reserve=reserve,capacity_receipt_sha256=sha(reclaim),selected='NONE',deployment='NONE')
 record.write_text(json.dumps(request,indent=2)+'\n')
 out=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub','-P','CFP05-CF-052',str(p)],capture_output=True,text=True)
 request.update(status='ACCEPTED' if out.returncode==0 else 'REJECTED_ENGINEERING_NO_SCIENTIFIC_UPDATES',returncode=out.returncode,stdout=out.stdout,stderr=out.stderr)
 if out.returncode==0:
  job=out.stdout.strip();assert job.split('.')[0].isdigit();request['PBS_JOBID']=job
 record.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request));assert out.returncode==0
'''

if __name__=='__main__':
 if '--self-check' in sys.argv:
  compile(PROGRAM,'remote_DOSE_submit','exec');assert PROGRAM.count("'/cm/shared/apps/pbspro/current/bin/qsub'")==1
  assert PROGRAM.index('REQUEST_RECORDED_BEFORE_SOLE_QSUB')<PROGRAM.index("'/cm/shared/apps/pbspro/current/bin/qsub'")
  assert 'ADOPT_EXISTING_ACCEPTED_REQUEST_NO_NEW_QSUB' in PROGRAM and "sum(x['work_kind']=='formal' for x in ledger)==4096" in PROGRAM
  print('PASS_SOLE_RECORDED_SUBMISSION_EXACT_PARENT_BUDGET_AND_CPU_SYNTAX; remote/model/physics=0')
 else:
  value=json.loads(remote('hop',PROGRAM))
  p=Path('reports/PI0285/r3/WP6/DOSE_ACTUAL_JOB_SUBMISSION.json');p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
