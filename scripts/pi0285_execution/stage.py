"""Stage only own sealed modules. No model, rendering, physics or training call."""
import base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    folder=Path(__file__).resolve().parent
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in folder.glob('*.py') if p.name!='stage.py'}
    documents={n:base64.b64encode(Path(p).read_bytes()).decode() for n,p in {
        'PROTOCOL.md':'docs/protocols/PI0285_FIRST_MANIPULATION_THEN_SHARED_TWO_TASK_V1.md',
        'PLAN_AND_LIMITS.json':'reports/PI0285/design/PLAN_AND_LIMITS.json',
        'PUBLICATION_READBACK.json':'reports/PI0285/adoption/PUBLICATION_READBACK.json'}.items()}
    for host,root in [('core','/home/liuyu-nfs/Projects/go2-arx5/PI0285'),('hop','/scratch/e1192548/Projects/go2-arx5/experiments/PI0285')]:
        code='FILES='+repr(files)+'\nDOCS='+repr(documents)+'\nROOT='+repr(root)+'\n'+r'''
import base64,hashlib,json,os,shutil,subprocess,re,time
from pathlib import Path
r=Path(ROOT);r.mkdir(parents=True,exist_ok=True)
assert not any((r/'source/episodes').glob('*/attempt-*/START.json')),'Do not restage executed source'
for name in ('code','design','runtime','evidence','source/evidence','data'): (r/name).mkdir(parents=True,exist_ok=True)
for name,b in FILES.items():(r/'code'/name).write_bytes(base64.b64decode(b))
for name,b in DOCS.items():(r/'design'/name).write_bytes(base64.b64decode(b))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=dict(files={n:sha(r/'code'/n) for n in FILES},release='7e840324c71aef47b401bd861b22710774334c97',canonical=6058329368)
(r/'design/CODE_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
old=r.parent/'PI0284';prior=json.loads((old/'evidence/TERMINAL_RECEIPT.json').read_text());assert prior['execution_closed'] and prior['execution']=='COMPLETE'
patterns=re.compile(r'(?<!\d)(?:28500[0-7]|2851(?:[0-5]\d|6[0-3])|28520\d|28521[0-5]|2853[0-2]\d|28533[01]|2854[0-2]\d|28543[01])(?!\d)')
checked=[];hits=[]
for previous in sorted(r.parent.glob('PI0*')):
 if previous==r or not previous.is_dir():continue
 paths=set()
 for pattern in ('evidence/*budget*.jsonl','source/evidence/*budget*.jsonl','source/episodes/*/attempt-*/START.json','design/*CONDITIONS*.json','design/*EXPOSURE*.json','data/*CONTRACT*.json'):
  paths.update(previous.glob(pattern))
 for p in sorted(paths):
  if not p.is_file():continue
  blob=p.read_bytes();matches=sorted(set(patterns.findall(blob.decode('utf8'))))
  checked.append(dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
  if matches:hits.append(dict(path=str(p),ids=matches))
assert not hits,('HOLDOUT_COLLISION',hits)
(r/'design/EXPOSURE.json').write_text(json.dumps(dict(status='PASS_ALL_RETAINED_PROJECT_LEDGER_AND_SOURCE_START_SCAN',checked=checked,hits=hits,confirmation_never_rendered=True),indent=2)+'\n')
if '/home/' in ROOT:
 source=r.parent/'PI0276/design/SOURCE_FILESET.json';shutil.copyfile(source,r/'design/SOURCE_FILESET.json')
 p=r/'project-src'
 if not p.exists():p.symlink_to(r.parent/'PI0276/project-src',target_is_directory=True)
 assert p.resolve()==(r.parent/'PI0276/project-src').resolve()
 scene=dict(status='FROZEN_BEFORE_TEACHER_SCORES',registry='PI0285PickPlaceStack-v1',origin_world=[0,0,0],plane_axes=[[1,0,0],[0,1,0],[0,0,1]],workspace_xy=[[-.2,-.2],[.2,.2]],table_z=0,cube_edge=.04,zone_edge=.16,zone_collision_shapes=0,zone_render_height_m=.00003,source_sim_freq=100,control_freq=20,original_Source_fileset_sha256=sha(source),scene_code_sha256=sha(r/'code/scene.py'),camera='EXACT_PINNED_STACK_PANDA_WRISTCAM',same_scene_for_both_instructions=True)
 (r/'design/SCENE_V1.json').write_text(json.dumps(scene,indent=2)+'\n')
df=subprocess.check_output(['df','-B1',str(r.resolve())],text=True)
capacity=dict(user_authorized_storage='LOCAL_THEN_HF_PUBLIC_USEFUL_ASSETS',owned_cap_GiB=240,personal_quota_headroom='UNKNOWN',shared_df_only=df,stage='CPU_AND_PILOT_ONLY_NO_TRAIN_ALLOCATION_YET',small_write_pass=True)
(r/'evidence/STORAGE_PREFLIGHT.json').write_text(json.dumps(capacity,indent=2)+'\n')
print(json.dumps(dict(status='PI0285_STAGED_AND_PRESERVED_PRIOR_NO_SCIENTIFIC_CALLS',root=str(r),code_seal=seal,exposure_files=len(checked),storage=capacity)))
'''
        response=json.loads(remote(host,code))
        out=Path('reports/PI0285/adoption')/(host.upper()+'_STAGED.json');out.write_text(json.dumps(response,indent=2)+'\n',encoding='utf8')
        print(json.dumps(response))

if __name__=='__main__':stage()
