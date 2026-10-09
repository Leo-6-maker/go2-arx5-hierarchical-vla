"""Use the existing14:00 authority at the stopped canary/formal boundary."""
import sys
sys.path.insert(0,'scripts/pi0276_execution')
from coordinator import remote
print(remote('hop',r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');runtime=r/'runtime'
names=['C1_CORRECT1024-continuous-0-8','C1_CORRECT1024-split-0-4','C1_CORRECT1024-split-4-8','C1_CORRECT1024-CANARY-continuous','C1_CORRECT1024-CANARY-split']
until=time.time()+600
while time.time()<until:
 assert not (runtime/'C1_CORRECT1024-main-0-1024.start.json').exists(),'Already healthy main: retain active code'
 proof=r/'learning/C1_CORRECT1024/CANARY_COMPARISON.json'
 if proof.exists() and all((runtime/(n+'.done.json')).exists() for n in names):break
 time.sleep(1)
else:raise RuntimeError('No closed canary boundary yet; no changes')
assert json.loads(proof.read_text())['status']=='PASS_FULL_RESUME_CANARY'
assert all(json.loads((runtime/(n+'.done.json')).read_text())['returncode']==0 for n in names)
assert not (runtime/'C1_CORRECT1024-main-0-1024.start.json').exists()
history=r/'evidence/time-binding/C1-canary-closed-formal-1400';history.mkdir(parents=True,exist_ok=False)
paths=[r/'code/r2.py',r/'design/CODE_SEAL.json',runtime/'TRAINING_ALLOCATION.json']
for p in paths:(history/p.name).write_bytes(p.read_bytes())
source=paths[0].read_bytes();assert source.count(b'UPDATE_STOP=1791524700')==1
source=source.replace(b'UPDATE_STOP=1791524700',b'UPDATE_STOP=1791525300')
assert b'MODEL_STOP=1791525600' in source
paths[0].write_bytes(source)
seal=json.loads(paths[1].read_text());seal['files']['r2.py']=hashlib.sha256(source).hexdigest();paths[1].write_text(json.dumps(seal,indent=2)+'\n')
allocation=json.loads(paths[2].read_text());assert allocation['PBS_JOBID']=='656178.hopper-m-02'
old=allocation['deadline_epoch'];allocation['deadline_epoch']=1791525420;paths[2].write_text(json.dumps(allocation,indent=2)+'\n')
receipt=dict(status='STOPPED_CANARY_TO_FORMAL_OWNER_TIME_RESERVE_BOUNDARY',epoch=time.time(),actual_canary_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),old_optimizer_stop_epoch=1791524700,optimizer_stop_epoch=1791525300,science_stop_epoch=1791525600,old_allocation_deadline=old,allocation_deadline=allocation['deadline_epoch'],authority='Existing explicit14:00 human extension plus latest both-arms completion priority; derived optimizer margin15min becomes5min based on actual slow I/O, no extension beyond14:00 authorized here',r2_sha256=seal['files']['r2.py'],healthy_model_or_training_child_active=False,recipe_and_cumulative_caps_unchanged=True,selected='NONE',deployment='NONE')
(history/'RESOLUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
'''))
