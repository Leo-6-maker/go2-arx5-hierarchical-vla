"""Change only future output placement after the complete, audited pilot closes."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in here.glob('*.py') if not p.name.startswith('stage')}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=f'FILES={files!r}\nCOMMIT={commit!r}\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,time
from collections import Counter
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0285')
closed=json.loads((r/'runtime/SOURCE_CLOSED.json').read_text());assert closed['split']=='PILOT'
args=subprocess.run(['ps','-p',str(closed['pid']),'-o','args='],capture_output=True,text=True).stdout
assert str(r/'code/source.py') not in args,'Healthy pilot is never edited'
panel=json.loads((r/'source/evidence/PILOT.json').read_text());assert len(panel['rows'])==16
assert panel['tasks']['T1']['admitted'],'A valid T2 alone does not authorize a different first-policy goal'
audit=json.loads((r/'audit/PILOT_CPU.json').read_text());assert audit['status'].startswith('PASS_INDEPENDENT_CPU_') and len(audit['rows'])==16
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert audit['source_sha256']==sha(r/'source/evidence/PILOT.json')
local=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0285/source')
assert local.resolve().is_relative_to(Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0285').resolve())
local.mkdir(parents=True,exist_ok=True)
assert not any((local/'episodes').glob('*/attempt-*/START.json'))
before=shutil.disk_usage(local)
assert before.free>=80*1024**3,'Measured working-set capacity is insufficient'
fstype=subprocess.check_output(['findmnt','-T',str(local),'-n','-o','FSTYPE'],text=True).strip()
assert fstype not in ('tmpfs','ramfs'),'Temporary memory is never the sole source'
source=r/'source/evidence/budget.jsonl';target=local/'evidence/budget.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():assert sha(target)==sha(source),'Never reset an existing continuation ledger'
else:
 shutil.copyfile(source,target)
 with target.open('rb') as f:os.fsync(f.fileno())
assert sha(source)==sha(target)
counts=dict(Counter(json.loads(x)['kind'] for x in target.read_text().splitlines()))
assert counts['starts']==16
oldseal=r/'design/CODE_SEAL.json';preserved=r/'design/CODE_SEAL_PILOT.json'
if preserved.exists():assert preserved.read_bytes()==oldseal.read_bytes()
else:shutil.copyfile(oldseal,preserved)
large={}
for name in ('data','models','transfers'):
 target=local.parent/name;target.mkdir(exist_ok=True);alias=r/name
 assert target.resolve().is_relative_to(local.parent.resolve())
 if alias.is_symlink():assert alias.resolve()==target.resolve()
 else:
  if alias.exists():assert alias.is_dir() and not any(alias.iterdir());alias.rmdir()
  alias.symlink_to(target,target_is_directory=True)
 large[name]=str(target)
tier=dict(status='PASS_CLOSED_PILOT_CUMULATIVE_LEDGER_DURABLE_LOCAL_TIER',record_root=str(local),large_output_roots=large,original_pilot_root=str(r/'source'),original_pilot_seal_sha256=sha(preserved),closed_pid=closed['pid'],PILOT_CPU_sha256=sha(r/'audit/PILOT_CPU.json'),prior_ledger_sha256=sha(source),initial_cumulative_counts=counts,fstype=fstype,measured_local_free_bytes=before.free,personal_quota='UNKNOWN',scope='FUTURE_OWN_RAW_DATA_MODEL_VIEWS_AND_TRANSFERS_NO_TEACHER_SCORER_CAMERA_PHYSICS_OR_SEED_CHANGE',pilot_evidence_moved_or_deleted=False)
(r/'design/RECORDING_TIER.json').write_text(json.dumps(tier,indent=2)+'\n')
for name,b in FILES.items():(r/'code'/name).write_bytes(base64.b64decode(b))
seal=json.loads(preserved.read_text());seal.update(files={n:sha(r/'code'/n) for n in FILES},published_commit=COMMIT,prior_pilot_seal_sha256=sha(preserved),reason='POST_CLOSED_PILOT_OUTPUT_TIER_AND_PREPARED_NATIVE_DATA_TRAINING_ENTRY')
oldseal.write_text(json.dumps(seal,indent=2)+'\n')
reservations=dict(query=32640,head=130560,GPU_seconds=57600,host='core',global_caps=dict(query=32768,head=131072,GPU_seconds=115200),hop=dict(query=128,head=512,GPU_seconds=57600))
(r/'design/RESOURCE_RESERVATIONS.json').write_text(json.dumps(reservations,indent=2)+'\n')
start=json.loads((r/'runtime/PILOT_PROCESS.json').read_text())['start_epoch']
processes=r/'runtime/GPU_PROCESSES.jsonl';assert not processes.exists()
processes.write_text(json.dumps(dict(event='end',pid=closed['pid'],start_epoch=start,epoch=closed['end_epoch'],device='CORE_GPU0',scope='ADOPT_ACTUAL_CLOSED_PILOT_CONTEXT_INCLUDING_IO_WAIT'))+'\n')
print(json.dumps(tier))
'''
    result=json.loads(remote('core',code))
    Path('reports/PI0285/adoption/RECORDING_TIER.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':stage()
