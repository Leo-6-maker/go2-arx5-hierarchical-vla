"""Bind the explicit no14:00 cutoff reply before the one original C1 main."""
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution')
from coordinator import remote
result=remote('hop',r'''
import datetime,hashlib,json,re,subprocess,time
from pathlib import Path
from zoneinfo import ZoneInfo
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');runtime=r/'runtime'
prior=r/'evidence/time-binding/C1-canary-closed-formal-1400/RESOLUTION.json'
until=time.time()+600
while not prior.exists():
 assert time.time()<until
 assert not (runtime/'C1_CORRECT1024-main-0-1024.start.json').exists(),'Never amend a healthy main'
 time.sleep(.5)
assert not (runtime/'C1_CORRECT1024-main-0-1024.start.json').exists()
proof=r/'learning/C1_CORRECT1024/CANARY_COMPARISON.json';assert json.loads(proof.read_text())['status']=='PASS_FULL_RESUME_CANARY'
q=subprocess.check_output(['/cm/shared/apps/pbspro/current/bin/qstat','-f','656178.hopper-m-02'],text=True);assert 'job_state = R' in q
start=datetime.datetime.strptime(re.search(r'stime\s*=\s*([^\n]+)',q).group(1).strip(),'%a %b %d %H:%M:%S %Y').replace(tzinfo=ZoneInfo('Asia/Singapore')).timestamp()
wall=tuple(map(int,re.search(r'Resource_List.walltime\s*=\s*(\d+):(\d+):(\d+)',q).groups()));end=int(start+wall[0]*3600+wall[1]*60+wall[2])
history=r/'evidence/time-binding/EXPLICIT_NO_1400_CUTOFF';history.mkdir(parents=True,exist_ok=False)
source=r/'code/r2.py';seal=r/'design/CODE_SEAL.json';allocation=runtime/'TRAINING_ALLOCATION.json'
for p in [source,seal,allocation]:(history/p.name).write_bytes(p.read_bytes())
code=source.read_bytes()
for old,new in [(1791525300,end-120),(1791525600,1791540000),(1791522900,1791537300)]:code=code.replace(str(old).encode(),str(new).encode())
assert ('UPDATE_STOP='+str(end-120)).encode() in code and b'MODEL_STOP=1791540000' in code
source.write_bytes(code);value=json.loads(seal.read_text());value['files']['r2.py']=hashlib.sha256(code).hexdigest();seal.write_text(json.dumps(value,indent=2)+'\n')
v=json.loads(allocation.read_text());assert v['PBS_JOBID']=='656178.hopper-m-02';v['deadline_epoch']=end;allocation.write_text(json.dumps(v,indent=2)+'\n')
receipt=dict(status='EXPLICIT_HUMAN_COMPLETION_PRIORITY_NO_1400_CUTOFF_BOUND_AT_STOPPED_CANARY_FORMAL_BOUNDARY',human_reply='完成两臂，允许越过14:00',epoch=time.time(),original_release='eae6dbc1f9281b204a4e28e5f7c100d705837318',original_canonical=6072813602,PBS_JOBID='656178.hopper-m-02',scheduler_record=q,legal_PBS_end_epoch=end,optimizer_stop_epoch=end-120,calendar_model_stop_epoch=1791540000,confirmation_latest_epoch=1791537300,actual_canary_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),r2_sha256=value['files']['r2.py'],Hopper_and_Core_each_GPU_cap_seconds=14400,formal_cap=2048,canary_cap=32,all_updates_cap=2112,all_original_scientific_and_quality_budgets_unchanged=True,healthy_model_or_training_child_active=False,selected='NONE',deployment='NONE')
(history/'RESOLUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(receipt=receipt,code=code.decode())),flush=True)
''')
v=json.loads(result);folder=Path('reports/PI0286/execution');(folder/'EXPLICIT_COMPLETION_PRIORITY_HOP_TIME_BINDING.json').write_text(json.dumps(v['receipt'],indent=2)+'\n',encoding='utf-8')
Path('scripts/pi0286_execution/r2.py').write_bytes(v['code'].encode())
print(json.dumps(v['receipt']))
