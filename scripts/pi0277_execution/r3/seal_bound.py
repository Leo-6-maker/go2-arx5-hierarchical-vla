"""Seal the actual stopped boundary, then submit its one remaining-budget continuation."""
import argparse,base64,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pi0276_execution'))
from coordinator import remote

def seal():
    paths=['docs/protocols/PI0277_R3_STAGE_RESOLVED_FOCUS_AND_LOCAL_FEEDBACK.md','docs/protocols/PI0277_R3_LATEST_RESULT_AND_CONDITIONAL_GRIPPER_BOUND_ADDENDUM.md','reports/PI0277/design/R3_REVIEW_AND_EXECUTION_LIMITS.json','reports/PI0277/design/R3_LATEST_ADOPTION_AND_BOUND_DELTA.json']
    files={Path(p).name:base64.b64encode(subprocess.check_output(['git','show','a5a7e569684d52170f02b0819ae006a7587800dd:'+p])).decode() for p in paths}
    script=r'''import os,json,hashlib,subprocess,base64
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS');phase=r.with_name('R3_BOUND');(phase/'authority').mkdir(exist_ok=True);(phase/'runtime').mkdir(exist_ok=True)
for name,payload in FILES.items():(phase/'authority'/name).write_bytes(base64.b64decode(payload))
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
p=phase/'evidence/OBJECTIVE_SEAL.json'
if p.exists():
 v=read(p);assert v['phase']=='FOCUS_GRIP_BOUND_R3';print('ADOPT_EXISTING_SEAL',sha(p));raise SystemExit(0)
q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-xf','651332'],capture_output=True,text=True);assert 'job_state = F' in q.stdout and 'Exit_status = 0' in q.stdout
complete=read(r/'FOCUS/main/training/SEGMENT_COMPLETE.json');resume=complete['actual_update'];assert resume==1802<4096 and complete['actual_microbatches']==2*resume
cp=r/'FOCUS/main/training'/f'checkpoint-{resume}';assert read(cp/'trainer_state.json')['global_step']==resume and read(cp/'pi0277_provenance/CURSOR.json')['next_unconsumed_micro']==2*resume
for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','model.safetensors.index.json'):assert (cp/name).is_file()
for name in ('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl'):assert sum(1 for _ in (cp.parent/name).open())==2*resume
trigger=read(phase/'evidence/TRIGGER_RECEIPT.json');assert not trigger['qualified_candidate_exists'] and len(trigger['receipt']['suffix_only_terminal_failures'])>=2
assert trigger['CPU_box_check']['source_sha256']==sha(phase/'code/bound_train.py')
updates=sum(1 for _ in (r.parent/'raw_completed_updates.jsonl').open());assert updates+(4096-resume)<=9048
pbs="""#!/bin/bash
#PBS -N pi0277_r3_bound
#PBS -P CFP05-CF-052
#PBS -q auto
#PBS -l select=1:ngpus=1:ncpus=12:mem=225GB:gpu_model=H200
#PBS -l walltime=04:00:00
#PBS -m n
set -euo pipefail
source /scratch/e1192548/Projects/go2-arx5/experiments/PI0273/code/environment.sh
export PYTHONPATH=/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS/code:/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R3_BOUND/code:"$PYTHONPATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_CORETYPE=Prescott
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS
exec "$venv/bin/python" -u /scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R3_BOUND/code/bound_train.py --root /scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS --resume 1802 --seal /scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R3_BOUND/evidence/OBJECTIVE_SEAL.json
"""
p=phase/'code/continue.pbs';p.write_text(pbs)
paths=list((phase/'code').glob('*.py'))+[p,*list((phase/'authority').glob('*')),r/'EXECUTION_FILESET.json',phase/'evidence/TRIGGER_RECEIPT.json']
state=[dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(cp.rglob('*')) if p.is_file()]
v=dict(phase='FOCUS_GRIP_BOUND_R3',release='a5a7e569684d52170f02b0819ae006a7587800dd',canonical=6021627390,launch_allowed=True,resume_update=resume,remaining_original_updates=4096-resume,trigger=trigger['receipt'],trigger_source_sha256=trigger['sha256'],qualified_candidate_exists=False,old_train_sha256=sha(r/'code/train.py'),resume_checkpoint=str(cp),resume_state=state,old_job_terminal_qstat=q.stdout,shared_completed_at_boundary=updates,optimizer='INHERITED_COMPLETE_STATE_NOT_RESET',scheduler='INHERITED_COSINE4096_NOT_RESET',RNG='INHERITED_FULL_STATE_AND_UNCHANGED_NAMESPACE',micro_cursor=2*resume,
 objective=dict(old_coefficient=1,generated_gripper_coefficient=.1,box_coefficient=.1,box_scale=.05,box_interval=[0,1],uniform_valid_H40=True),CPU_check=trigger['CPU_box_check'],files=[dict(path=str(p),sha256=sha(p)) for p in paths],selected_checkpoint='NONE',deployment='NONE')
p=phase/'evidence/OBJECTIVE_SEAL.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,sort_keys=True)+chr(10));tmp.replace(p)
stop=r/'runtime/STOP_TRAINING.json';assert read(stop)['reason']=='R3_REGISTERED_FOCUS1024_SUFFIX_RECURRENCE';stop.rename(phase/'evidence/STOP_651332_CONSUMED.json')
print(json.dumps(dict(status='SEALED_EXACT_RECOVERY_BOUNDARY',resume=resume,remaining=4096-resume,checkpoint=str(cp),complete_files=len(state),seal_sha256=sha(p),shared_completed=updates)))
'''
    print(remote('hop','FILES='+repr(files)+'\n'+script),flush=True)

def submit():
    print(remote('hop',r'''import json,fcntl,subprocess,hashlib
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R3_BOUND');seal=r/'evidence/OBJECTIVE_SEAL.json';v=json.loads(seal.read_text());assert v['launch_allowed'] and v['resume_update']==1802
with (r/'runtime/launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=r/'runtime/PBS_LAUNCH.json';intent=r/'runtime/SUBMISSION_INTENT.json'
 if receipt.exists():print(receipt.read_text());raise SystemExit(0)
 assert not intent.exists(),'Unknown previous submission outcome; inspect qstat, never duplicate'
 intent.write_text(json.dumps(dict(state='OUTCOME_UNKNOWN_UNTIL_QSUB_RETURN',source_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest()))+chr(10))
 p=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub','-o',str(r/'runtime/pbs-01.out'),'-e',str(r/'runtime/pbs-01.err'),str(r/'code/continue.pbs')],capture_output=True,text=True)
 value=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,source_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),resume=1802,remaining_original_updates=2294)
 if p.returncode==0:value['job']=p.stdout.strip();assert '.hopper-m-02' in value['job']
 receipt.write_text(json.dumps(value)+chr(10));print(json.dumps(value));assert p.returncode==0
'''),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('seal','submit'));a=p.parse_args()
    seal() if a.stage=='seal' else submit()
