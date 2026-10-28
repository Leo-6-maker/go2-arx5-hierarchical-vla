"""Prepare own CPU-checked Hopper entry; data admission and qsub remain separate."""
import base64,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent;root='/scratch/e1192548/Projects/go2-arx5/experiments/PI0285'
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in here.glob('*.py') if not p.name.startswith('stage')}
    environment=(here.parent/'pi0282_execution/environment.sh').read_text().splitlines();environment[0]='root='+root
    environment.extend(['export PI0285_ROOT="$root"','export PYTHONPATH="$root/code:$root/../PI0283/code:$PI0282_OFFICIAL_DIR:$root/project-src/src"'])
    files['environment.sh']=base64.b64encode(('\n'.join(environment)+'\n').encode()).decode()
    pbs=(here.parent/'pi0282_execution/train.pbs').read_text().replace('pi0282_native_eef','pi0285_place').replace('/experiments/PI0282','/experiments/PI0285').replace('training_launch.py','launch.py').rstrip()+' --card "$root/design/PLACE4096_CARD.json" --arm PLACE4096 --stop 2048\n'
    files['train.pbs']=base64.b64encode(pbs.encode()).decode()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    script=f'''import base64,hashlib,json,os,shutil,subprocess
from pathlib import Path
r=Path({root!r});old=r.parent/'PI0283';assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'Adopt actual lease instead of restaging'
for name in ('code','design','runtime','evidence','data','learning','delivery'):(r/name).mkdir(exist_ok=True)
for name,b in {files!r}.items():(r/'code'/name).write_bytes(base64.b64decode(b))
for name in ('design/PRISTINE_LOADING_REFERENCE.json','learning/WEIGHT_LOADING_REFERENCE.json'):
 shutil.copyfile(old/'learning/WEIGHT_LOADING_REFERENCE.json',r/name)
p=r/'project-src'
if not p.exists():p.symlink_to(old/'project-src',target_is_directory=True)
assert p.resolve()==(old/'project-src').resolve()
shutil.copyfile(r.parent/'PI0284/runtime/rrsync',r/'runtime/rrsync')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(r/name).write_text(json.dumps(value,indent=2)+'\\n')
dependencies={{str(p):sha(p) for p in (old/'code').glob('*.py')}}
seal=dict(files={{name:sha(r/'code'/name) for name in {list(files)!r}}},release='7e840324c71aef47b401bd861b22710774334c97',canonical=6058329368,published_commit={commit!r},native_dependencies=dependencies,stage='PREPARED_CPU_ONLY_NO_DATA_ADMISSION_OR_QSUB')
write('design/CODE_SEAL.json',seal)
write('design/RESOURCE_RESERVATIONS.json',dict(query=128,head=512,GPU_seconds=57600,global_caps=dict(query=32768,head=131072,GPU_seconds=115200)))
write('design/TRAINING_LAYOUT.json',dict(microbatch=4,accumulation=2,effective_batch=8,workers=2,status='PROTOCOL_FIXED_NATIVE_LOADER'))
shell='source '+str(r/'code/environment.sh')+'; export CUDA_VISIBLE_DEVICES=""; "$venv/bin/python" -c '+__import__('shlex').quote("import ast,pathlib;from contract import guard;from authority import self_check;guard();self_check();[ast.parse(p.read_text()) for p in pathlib.Path("+repr(str(r/'code'))+").glob('*.py')];from policy import self_check;self_check();from native_model import local_processor;from scope import configure;print('PASS_ACTUAL_IMPORTS_PARSE_AND_FINITE_CPU_ENTRY_NO_MODEL_OR_OPTIMIZER_CALL')")
tested=subprocess.run(['/bin/bash','-lc',shell],capture_output=True,text=True);assert tested.returncode==0,(tested.stdout[-2000:],tested.stderr[-4000:])
value=dict(status='PASS_PREPARED_HOPPER_CPU_IMPORTS_AND_FINITE_ENTRY_NOT_DATA_OR_TRAIN_ADMISSION',published_commit={commit!r},code_files=seal['files'],native_dependencies=dependencies,CPU_check=tested.stdout,new_model_queries=0,new_optimizer_updates=0,new_PBS_submissions=0,quota_headroom='UNKNOWN')
write('evidence/PREPARED_CPU_ENTRY.json',value);print(json.dumps(value))
'''
    result=json.loads(remote('hop',script))
    path=Path('reports/PI0285/preparation/HOPPER_CPU_ENTRY.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('code_files','native_dependencies')}))


if __name__=='__main__':stage()
