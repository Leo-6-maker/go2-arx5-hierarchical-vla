"""Stage isolated execution assets; never overwrite an active runtime."""
import base64,hashlib,json,sys,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
ROOTS={'core':'/home/liuyu-nfs/Projects/go2-arx5/PI0286','hop':'/scratch/e1192548/Projects/go2-arx5/experiments/PI0286'}
def stage(host):
 here=Path(__file__).resolve().parent
 files={p.name:base64.b64encode(p.read_bytes()).decode() for p in here.iterdir() if p.suffix in ('.py','.sh','.pbs') and p.name!='stage_common.py'}
 docs={name:base64.b64encode(Path(path).read_bytes()).decode() for name,path in {'PROTOCOL.md':'docs/protocols/PI0286_TO_1300_RECOVERY_VS_REPLAY_AND_REPORT_V1.md','PLAN_AND_LIMITS.json':'reports/PI0286/design/R4_NEW_SCIENTIFIC_AUTHORITY_AND_LIMITS.json','PUBLICATION_READBACK.json':'reports/PI0286/adoption/PUBLICATION_READBACK.json','CANONICAL_READBACK.json':'reports/PI0286/adoption/CANONICAL_READBACK.json'}.items()}
 code='FILES='+repr(files)+'\nDOCS='+repr(docs)+'\nROOT='+repr(ROOTS[host])+'\nHOST='+repr(host)+'\n'+r"""
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
r=Path(ROOT);old=r.parent/'PI0285'
assert not (r/'runtime/TRAINING_ALLOCATION.json').exists()
assert not any((r/'source/episodes').glob('*/attempt-*/START.json'))
for name in ('code','design','runtime','evidence','source/evidence','data','learning','delivery','models','audit'):(r/name).mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
for n,b in FILES.items():(r/'code'/n).write_bytes(base64.b64decode(b))
for n,b in DOCS.items():(r/'design'/n).write_bytes(base64.b64decode(b))
for n in ('SOURCE_FILESET.json','SCENE_V1.json'):
 if (old/'design'/n).exists():shutil.copyfile(old/'design'/n,r/'design'/n)
for n in ('statistics.json','processor_config.json','embodiment_id.json','WINDOW_INDEX.json','SOURCE_BANK.json','NATIVE_INPUTS_CHECK.json','R2_SCOPED_DATA_ADMISSION.json','STATISTICS_RELEASE.json','CANARY_INPUT.npz'):
 p=r/'data'/n
 if not p.exists():p.symlink_to(old/'data'/n)
for ns in ('policy_observation','action_target','R2_PROVENANCE'):
 (r/'data'/ns).mkdir(exist_ok=True)
 for source in (old/'data'/ns).iterdir():
  dest=r/'data'/ns/source.name
  if not dest.exists():dest.symlink_to(source,target_is_directory=source.is_dir())
p=r/'project-src'
if not p.exists():p.symlink_to(old/'project-src',target_is_directory=True)
deps={str(p):sha(p) for p in (r.parent/'PI0283/code').glob('*.py')}
write(r/'design/CODE_SEAL.json',dict(files={n:sha(r/'code'/n) for n in FILES},native_dependencies=deps,release='eae6dbc1f9281b204a4e28e5f7c100d705837318',canonical=6072813602))
write(r/'design/RESOURCE_RESERVATIONS.json',dict(query=9984 if HOST=='core' else 16,head=39936 if HOST=='core' else 64,GPU_seconds=14400,global_caps=dict(query=10000,head=40000,GPU_seconds=28800)))
write(r/'design/TRAINING_LAYOUT.json',dict(microbatch=4,accumulation=2,effective_batch=8,workers=2,status='PROTOCOL_FIXED_NATIVE_LOADER'))
if HOST=='hop':
 for n in ('design/PRISTINE_LOADING_REFERENCE.json','learning/WEIGHT_LOADING_REFERENCE.json'):shutil.copyfile(r.parent/'PI0283/learning/WEIGHT_LOADING_REFERENCE.json',r/n)
 shutil.copyfile(old/'runtime/rrsync',r/'runtime/rrsync')
else:
 p=r/'models/PARENT_DOSE6144'
 if not p.exists():p.symlink_to(old/'models/REPAIR_A2048-2048',target_is_directory=True)
write(r/'evidence/STAGED_COMMON.json',dict(status='PREPARED_ISOLATED_COMMON_NOT_SCIENCE_STARTED',epoch=time.time(),host=HOST,root=str(r),code_files=len(FILES),shared_free_bytes=shutil.disk_usage(r).free,personal_quota='UNKNOWN'))
print((r/'evidence/STAGED_COMMON.json').read_text())
"""
 result=json.loads(remote(host,code));out=Path('reports/PI0286/adoption')/(host.upper()+'_STAGED.json');out.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(stage,ROOTS):print(json.dumps(result))
