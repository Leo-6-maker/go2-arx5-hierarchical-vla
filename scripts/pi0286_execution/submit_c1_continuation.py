"""Stage one explicit human-authorized second arm after C0 closes."""
import base64,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution')
from coordinator import remote

folder=Path('scripts/pi0286_execution')
r2=(folder/'r2.py').read_bytes()
helper=(folder/'launch_c1_after_c0.py').read_bytes()
authority=dict(status='EXPLICIT_HUMAN_BOTH_ARMS_CONTINUATION',launch_allowed=True,
 release='eae6dbc1f9281b204a4e28e5f7c100d705837318',canonical=6072813602,
 human_instructions=['先跑完实验，时限放宽到2pm','让C0,C1都跑完，不要让工程问题阻碍科学主线'],
 formal_cap=2048,canary_cap=32,all_updates_cap=2112,technical_recovery_cap=2,
 C0_job='656152.hopper-m-02',C1_arm='C1_CORRECT1024',
 science_stop_epoch=1791525600,optimizer_stop_epoch=1791524700,
 original_parent_global_update=6144,original_recipe_and_budgets_unchanged=True,
 r2_source_base64=base64.b64encode(r2).decode(),r2_sha256=hashlib.sha256(r2).hexdigest(),
 C1_helper_sha256=hashlib.sha256(helper).hexdigest(),created_epoch=time.time(),
 selected='NONE',deployment='NONE')
payload=dict(authority=authority,helper=base64.b64encode(helper).decode())
script=r"""
import base64,hashlib,json,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
payload=PAYLOAD
q=subprocess.check_output(['/cm/shared/apps/pbspro/current/bin/qstat','-xf','656152.hopper-m-02'],text=True)
assert 'job_state = F' in q and 'Exit_status = 0' in q
assert json.loads((r/'runtime/C0_REPLAY1024_SEGMENT_0_1024.json').read_text())['status']=='FIXED_SEGMENT_COMPLETE'
assert not (r/'runtime/C1_JOB_SUBMISSION.json').exists(),'Adopt existing submission; never duplicate'
assert not (r/'learning/C1_CORRECT1024/CANARY_COMPARISON.json').exists()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
write(r/'design/BOTH_ARMS_CONTINUATION_1400.json',payload['authority'])
work=r/'runtime/continuation-c1';work.mkdir(parents=True,exist_ok=True)
(work/'launch_c1_after_c0.py').write_bytes(base64.b64decode(payload['helper']))
cap=r/'design/CAPACITY_ADMISSION.json';old=cap.read_bytes();(work/'CAPACITY_ADMISSION.previous.json').write_bytes(old)
v=json.loads(old);v.update(epoch=time.time(),valid_until_epoch=1791525600,basis='Same admitted90GiB working envelope. C0 three canary full states retired after actual PASS; retained C0 full20.47GB plus export5.67GB plus C1 three canary states61.40GB plus C1 inputs1.31GB <97.89GB admitted envelope. Quota cached/inferred unchanged; no df claim or additional allocation probe.')
write(cap,v)
card=r/'design/C1_CORRECT1024_CARD.json';(work/'C1_CARD.prepared.json').write_bytes(card.read_bytes())
v=json.loads(card.read_text());v.update(owner_capacity_sha256=hashlib.sha256(cap.read_bytes()).hexdigest(),owner_admitted_epoch=time.time(),time_reserve='Explicit human both-arms priority; original fresh canary plus1024; optimizer13:45, science14:00SGT; confirmation45min unchanged.')
write(card,v)
pbs='''#!/bin/bash
#PBS -N PI0286_C1
#PBS -q auto
#PBS -l select=1:ncpus=12:mem=192gb:ngpus=1:gpu_model=H200
#PBS -l walltime=02:00:00
#PBS -m n
#PBS -o /scratch/e1192548/Projects/go2-arx5/experiments/PI0286/runtime/continuation-c1/pbs.out
#PBS -e /scratch/e1192548/Projects/go2-arx5/experiments/PI0286/runtime/continuation-c1/pbs.err
set -euo pipefail
source /scratch/e1192548/Projects/go2-arx5/experiments/PI0286/code/environment.sh
exec "$venv/bin/python" -u "$root/runtime/continuation-c1/launch_c1_after_c0.py" --base /scratch/e1192548/Projects/go2-arx5/archive/stage_runs/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B
'''
(work/'C1.pbs').write_text(pbs)
receipt=dict(status='QSUB_REQUEST_STARTED',epoch=time.time(),arm='C1_CORRECT1024',C0_completed=True,scientific_arm_not_recovery=True,mail='n',pbs_sha256=hashlib.sha256(pbs.encode()).hexdigest(),authority_sha256=hashlib.sha256((r/'design/BOTH_ARMS_CONTINUATION_1400.json').read_bytes()).hexdigest())
write(r/'runtime/C1_JOB_SUBMISSION.json',receipt)
p=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub','-P','CFP05-CF-052',str(work/'C1.pbs')],text=True,capture_output=True)
receipt.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,status='ACCEPTED' if p.returncode==0 else 'REJECTED_NO_JOB')
if p.returncode==0:receipt['job']=p.stdout.strip()
write(r/'runtime/C1_JOB_SUBMISSION.json',receipt)
print(json.dumps(receipt));assert p.returncode==0
""".replace('PAYLOAD',repr(payload))
result=remote('hop',script)
out=Path('reports/PI0286/execution');out.mkdir(parents=True,exist_ok=True)
(out/'BOTH_ARMS_CONTINUATION_1400.json').write_text(json.dumps(authority,indent=2)+'\n',encoding='utf-8')
(out/'C1_JOB_SUBMISSION_READBACK.txt').write_text(result,encoding='utf-8')
print(result)
