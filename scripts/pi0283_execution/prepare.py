"""Isolated small files, immutable old raw data, sealed new runtime; no launch."""
import base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
HERE=Path(__file__).resolve().parent
REPORT=HERE.parents[1]/'reports/PI0283/execution'
ROOTS=dict(hop='/scratch/e1192548/Projects/go2-arx5/experiments/PI0283',core='/home/liuyu-nfs/Projects/go2-arx5/PI0283')


def stage():
    from build import build
    code=build();files={p.name:base64.b64encode(p.read_bytes()).decode() for p in code.glob('*.py')}
    files['environment.sh']=base64.b64encode((HERE.parent/'pi0282_execution/environment.sh').read_bytes().replace(b'/experiments/PI0282',b'/experiments/PI0283')).decode()
    pbs=(HERE.parent/'pi0282_execution/train.pbs').read_text().replace('pi0282_native_eef','pi0283_native_more').replace('/experiments/PI0282','/experiments/PI0283').replace('06:00:00','02:30:00').replace('training_launch.py','launch.py')
    files['train.pbs']=base64.b64encode(pbs.encode()).decode()
    for host,root in ROOTS.items():
        clock=base64.b64encode((REPORT/'CLOCK_RESOLUTION.json').read_bytes()).decode()
        script=f'''import base64,json,os,shutil,hashlib,subprocess
from pathlib import Path
r=Path({root!r});old=r.parent/'PI0282'
assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'NO_INFLIGHT_RUNTIME_EDITS'
for n in ('code','data','design','evidence','learning','runtime','source/evidence','source/noise'):(r/n).mkdir(parents=True,exist_ok=True)
for n,b in {files!r}.items():(r/'code'/n).write_bytes(base64.b64decode(b))
for n in ('DATA_CONTRACT.json','EXPERT_CODEC_ADMISSION.json','processor_config.json','statistics.json','embodiment_id.json','DIAG48.jsonl','PLAN16384.jsonl'):
 shutil.copyfile(old/'data'/n,r/'data'/n)
ep=r/'data/episodes'
if not ep.exists():ep.symlink_to(old/'data/episodes',target_is_directory=True)
assert ep.resolve()==(old/'data/episodes').resolve()
(r/'design/CLOCK_RESOLUTION.json').write_bytes(base64.b64decode({clock!r}))
authority=dict(release='508e3db7b63d3c47758794f3ffbad87b577e6e9d',canonical=6048886543,formal_cap=3072,all_updates_cap=3136,training_GPU_reservation_s=9000,evaluation_GPU_reservation_s=5400,deadline_epoch=None,training_cutoff_epoch=None,model_physics_cutoff_epoch=None,old_budget_sealed=2080,new_budget_separate=True,technical_recoveries_cap=2)
(r/'design/RUNTIME_AUTHORITY.json').write_text(json.dumps(authority,indent=2)+'\\n')
for n in ('TRAINING_LAYOUT.json',):shutil.copyfile(old/'design'/n,r/'design'/n)
shutil.copyfile(old/'learning/WEIGHT_LOADING_REFERENCE.json',r/'learning/WEIGHT_LOADING_REFERENCE.json')
if {host!r}=='hop':
 p=r/'project-src'
 if not p.exists():p.symlink_to(old/'project-src',target_is_directory=True)
 (r/'design/TRAINING_ALLOCATION.json').write_text(json.dumps(dict(kind='PBS_H200',training_GPU_reservation_s=9000,project='CFP05-CF-052',queue='auto',select='1:ngpus=1:ncpus=12:mem=225gb:gpu_model=H200',mail='n'),indent=2)+'\\n')
else:
 shutil.copyfile(old/'source/evidence/RESULT_EXPERT_TRAIN.json',r/'source/evidence/RESULT_EXPERT_TRAIN.json')
 for p in (old/'source/noise').glob('*.npy'):
  dest=r/'source/noise'/p.name
  if not dest.exists():os.link(p,dest)
 subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python',str(r/'code/plan.py'),str(r),'/home/liuyu-nfs/Projects/go2-arx5/PI0276/design/DATASET_CARD.json'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),check=True)
print(json.dumps(dict(status='SMALL_RUNTIME_STAGED_NO_MODEL_QUERY',host={host!r},files={{n:hashlib.sha256((r/'code'/n).read_bytes()).hexdigest() for n in {list(files)!r}}})))
'''
        value=remote(host,script);(REPORT/(host.upper()+'_STAGING.json')).write_text(value);print(value[-1000:])
    # Only new JSONL draw manifests cross the PC, never model/optimizer payloads.
    payload=json.loads(remote('core',"import json,base64\nfrom pathlib import Path\nr=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283/data')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ('MORE_PLAN.jsonl','EVENT_PLAN.jsonl','PLAN_AUDIT.json')}))"))
    print(remote('hop',f"import json,base64\nfrom pathlib import Path\nr=Path({ROOTS['hop']!r})/'data'\nfor n,b in {payload!r}.items():(r/n).write_bytes(base64.b64decode(b))\nprint('FROZEN_DRAW_MANIFEST_BYTES_COPIED_NO_MODEL')"))


if __name__=='__main__':stage()
