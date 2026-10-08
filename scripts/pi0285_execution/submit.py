"""One first legal auto/H200 request only after real direct-data CPU admission."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def submit():
    response=remote('hop',r'''import fcntl,json,time,subprocess,hashlib
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285');record=r/'runtime/JOB_SUBMISSION.json'
with (r/'runtime/SUBMISSION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if record.exists():
  old=json.loads(record.read_text());assert old['status']=='ACCEPTED','Resolve actual original scheduler request before any resubmission'
  x=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-f',old['PBS_JOBID']],capture_output=True,text=True)
  print(json.dumps(dict(status='ADOPT_EXISTING_ACCEPTED_REQUEST_NO_NEW_QSUB',record=old,scheduler=x.stdout,scheduler_stderr=x.stderr)));raise SystemExit(0)
 assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'Adopt existing allocation'
 proof=json.loads((r/'evidence/DIRECT_DATA_READBACK.json').read_text())
 assert proof['status']=='PASS_DIRECT_ALL_ADMITTED_DATA_MEMBERS_AND_CROSS_HOST_NATIVE_TWO_WORKER_INPUTS'
 assert proof['new_model_queries']==proof['new_optimizer_updates']==proof['new_PBS_submissions']==0
 # Actual same-host authority/plan/import check under the future job environment.
 shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+__import__('shlex').quote("from pathlib import Path;from contract import guard;from authority import stage_card;guard();r=Path("+repr(str(r))+");stage_card(r,r/'design/PLACE4096_CARD.json');from native_model import local_processor;print('PASS_FROZEN_ACTUAL_FIRST_TRAINING_ENTRY_NO_MODEL_OR_OPTIMIZER_QUERY')")
 preflight=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True);assert preflight.returncode==0,(preflight.stdout,preflight.stderr[-5000:])
 p=r/'code/train.pbs';text=p.read_text();assert '#PBS -q auto' in text and '#PBS -m n' in text and 'gpu_model=H200' in text and 'PLACE4096 --stop 2048' in text and '#PBS -A' not in text
 request=dict(status='REQUEST_RECORDED_BEFORE_SOLE_QSUB',requested_epoch=time.time(),script=str(p),script_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data_readback_sha256=hashlib.sha256((r/'evidence/DIRECT_DATA_READBACK.json').read_bytes()).hexdigest(),queue='auto',project='AUTO_NO_PROJECT_OVERRIDE',quota_headroom='UNKNOWN',first_segment='PLACE4096_0_TO_2048_AFTER_ACTUAL_CANARY',mail='n',selected='NONE',deployment='NONE')
 record.write_text(json.dumps(request,indent=2)+'\n')
 out=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub',str(p)],capture_output=True,text=True)
 request.update(returncode=out.returncode,stdout=out.stdout,stderr=out.stderr,status='ACCEPTED' if out.returncode==0 else 'REJECTED_ENGINEERING_NO_SCIENTIFIC_UPDATES')
 if out.returncode==0:
  job=out.stdout.strip();assert job and job.split('.')[0].isdigit(),('Ambiguous scheduler return; inspect existing request',out.stdout)
  request['PBS_JOBID']=job
 record.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request))
 assert out.returncode==0
''')
    result=json.loads(response);p=Path('reports/PI0285/execution/FIRST_JOB_SUBMISSION.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(response)


if __name__=='__main__':submit()
