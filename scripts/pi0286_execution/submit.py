"""One owner-coordinated PI0286 lease request, C0 then conditional C1; no second job."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def submit():
    script=r'''
import fcntl,json,time,subprocess,hashlib
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');record=r/'runtime/JOB_SUBMISSION.json'
with (r/'runtime/SUBMISSION.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if record.exists():
  value=json.loads(record.read_text());print(json.dumps(dict(status='ADOPT_EXISTING_REQUEST_NO_SECOND_QSUB',record=value)));raise SystemExit(0)
 assert time.time()<1791516600,'No optimizer arm starts after11:30 SGT'
 assert not (r/'runtime/TRAINING_ALLOCATION.json').exists()
 capacity=json.loads((r/'design/CAPACITY_ADMISSION.json').read_text())
 assert capacity['status']=='OWNER_ADMITTED_PI0286_OUTPUT_HEADROOM' and capacity['remaining_headroom_bytes']>=capacity['required_peak_bytes']
 assert time.time()<=capacity['valid_until_epoch']
 code=json.loads((r/'evidence/TRAIN_CODE_READY.json').read_text());assert code['status']=='PASS_PI0286_ISOLATED_TRAIN_STAGE_CPU_ONLY'
 p=r/'code/C0_REPLAY1024.pbs';text=p.read_text();assert '#PBS -q auto' in text and '#PBS -m n' in text and 'gpu_model=H200' in text and '#PBS -A' not in text
 shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+__import__('shlex').quote("from pathlib import Path;from contract import guard;from authority import stage_card;guard();r=Path("+repr(str(r))+");stage_card(r,r/'design/C0_REPLAY1024_CARD.json');print('PASS_ACTUAL_C0_PARENT_DATA_CARD_CPU_NO_MODEL')")
 check=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True);assert check.returncode==0,(check.stdout,check.stderr)
 old=r.parent/'PI0285/runtime/JOB_SUBMISSION.json';prior=json.loads(old.read_text());project=prior.get('project')
 assert project=='CFP05-CF-052','Existing verified project identity only'
 request=dict(status='REQUEST_RECORDED_BEFORE_SOLE_QSUB',requested_epoch=time.time(),script=str(p),script_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),project=project,
  canonical=6072813602,release='eae6dbc1f9281b204a4e28e5f7c100d705837318',one_lease_driver='C0_then_C1_if_admitted_before1130_same_full6144_parent',
  capacity_sha256=hashlib.sha256((r/'design/CAPACITY_ADMISSION.json').read_bytes()).hexdigest(),queue='auto',mail='n',selected='NONE',deployment='NONE')
 record.write_text(json.dumps(request,indent=2)+'\n')
 out=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub','-P',project,str(p)],capture_output=True,text=True)
 request.update(returncode=out.returncode,stdout=out.stdout,stderr=out.stderr,status='ACCEPTED' if out.returncode==0 else 'REJECTED_ENGINEERING_NO_SCIENTIFIC_UPDATES')
 if out.returncode==0:
  job=out.stdout.strip();assert job and job.split('.')[0].isdigit(),'Ambiguous accepted scheduler return, audit existing request';request['PBS_JOBID']=job
 record.write_text(json.dumps(request,indent=2)+'\n');print(json.dumps(request));assert out.returncode==0
'''
    value=json.loads(remote('hop',script));p=Path('reports/PI0286/execution/FIRST_JOB_SUBMISSION.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':print(json.dumps(submit()))
