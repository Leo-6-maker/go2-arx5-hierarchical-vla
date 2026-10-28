"""Owner-invoked isolated PI0286 training stage; no qsub or scientific computation."""
import base64,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent
    names=['authority.py','train.py','launch.py','training_audit.py','prepare_training.py','submit.py','environment.sh','C0_REPLAY1024.pbs','C1_CORRECT1024.pbs']
    files={name:base64.b64encode((here/name).read_bytes().replace(b'\r\n',b'\n')).decode() for name in names}
    manifest=base64.b64encode(Path('reports/PI0285/r3/WP8/DOSE2048_FULL_READBACK.json').read_bytes()).decode()
    script=f'FILES={files!r}\nMANIFEST={manifest!r}\n'+r'''
import ast,base64,json,hashlib,subprocess
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
assert r.is_dir() and (r/'code/contract.py').is_file(),'Owner common stage must exist first'
assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'Do not change an imported healthy PI0286 runtime'
for name,encoded in FILES.items():(r/'code'/name).write_bytes(base64.b64decode(encoded))
(r/'design/PARENT_HF_READBACK.json').write_bytes(base64.b64decode(MANIFEST))
seal=json.loads((r/'design/CODE_SEAL.json').read_text());seal['files'].update({name:hashlib.sha256((r/'code'/name).read_bytes()).hexdigest() for name in FILES})
(r/'design/CODE_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+__import__('shlex').quote("from pathlib import Path;import ast;from contract import guard;from authority import self_check;guard();print(self_check());[ast.parse(p.read_text()) for p in Path("+repr(str(r/'code'))+").glob('*.py')]")
check=subprocess.run(['/bin/bash','-lc',shell],text=True,capture_output=True);assert check.returncode==0,(check.stdout,check.stderr)
value=dict(status='PASS_PI0286_ISOLATED_TRAIN_STAGE_CPU_ONLY',files={name:hashlib.sha256((r/'code'/name).read_bytes()).hexdigest() for name in FILES},CPU_check=check.stdout,model_queries=0,optimizer_updates=0,PBS_submissions=0)
(r/'evidence/TRAIN_CODE_READY.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''
    value=json.loads(remote('hop',script));p=Path('reports/PI0286/preparation/TRAIN_CODE_STAGED.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');return value


if __name__=='__main__':print(json.dumps(stage()))
