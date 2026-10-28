"""Stage only owned small files and readonly references before the single lease."""
import base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

HERE=Path(__file__).resolve().parent
REPORT=HERE.parents[1]/'reports/PI0284/execution'
HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0284'
CORE='/home/liuyu-nfs/Projects/go2-arx5/PI0284'

def stage():
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in HERE.iterdir() if p.suffix in ('.py','.pbs')}
    environment=(HERE.parent/'pi0282_execution/environment.sh').read_text().splitlines()
    environment[0]='root='+HOP
    environment.append('export PYTHONPATH="$root/code:$root/../PI0283/code:$PI0282_OFFICIAL_DIR:$root/project-src/src"')
    files['environment.sh']=base64.b64encode(('\n'.join(environment)+'\n').encode()).decode()
    names=['design/PUBLICATION_RECEIPT.json','design/TRAINING_ROUTE.json','design/HOLDOUT_EXPOSURE_CLASSIFIED.json','design/SAMPLER_FROZEN.json','data/PAIRED_PLAN16384.jsonl','data/PAIRED_PLAN_AUDIT.json']
    payload=json.loads(remote('core',f"""import base64,json
from pathlib import Path
r=Path({CORE!r});v={{n:base64.b64encode((r/n).read_bytes()).decode() for n in {names!r}}}
first=json.loads((r/'design/PRE_SCORE_INPUTS.json').read_text())['rows'][0]
v['data/CANARY_INPUT.npz']=base64.b64encode(Path(first['input']).read_bytes()).decode()
v['data/CANARY_NOISE.npy']=base64.b64encode(Path(first['packet0']).read_bytes()).decode()
print(json.dumps(v))
"""))
    payload['design/PARENT_ACTUAL14_READBACK.json']=base64.b64encode((REPORT.parent/'adoption/PARENT_ACTUAL14_READBACK.json').read_bytes()).decode()
    out=remote('hop',f"""import base64,json,hashlib,shutil,subprocess
from pathlib import Path
r=Path({HOP!r});old=r.parent/'PI0283'
assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'NO_LIVE_RUNTIME_EDITS'
for n in ('code','data','design','evidence','learning','runtime','delivery'):(r/n).mkdir(parents=True,exist_ok=True)
for n,b in {files!r}.items():(r/'code'/n).write_bytes(base64.b64decode(b))
for n,b in {payload!r}.items():
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
for n in ('DATA_CONTRACT.json','EXPERT_CODEC_ADMISSION.json','processor_config.json','statistics.json','embodiment_id.json','DIAG48.jsonl'):shutil.copyfile(old/'data'/n,r/'data'/n)
for dest,source in [('data/episodes','data/episodes'),('project-src','project-src')]:
 p=r/dest
 if not p.exists():p.symlink_to(old/source,target_is_directory=True)
 assert p.resolve()==(old/source).resolve()
shutil.copyfile(old/'learning/WEIGHT_LOADING_REFERENCE.json',r/'learning/WEIGHT_LOADING_REFERENCE.json')
shutil.copyfile(old/'runtime/rrsync',r/'runtime/rrsync')
def write(n,v):(r/n).write_text(json.dumps(v,indent=2)+'\\n')
pub=json.loads((r/'design/PUBLICATION_RECEIPT.json').read_text())
write('design/RUNTIME_AUTHORITY.json',dict(release=pub['release'],canonical=pub['canonical'],formal_cap=4096,all_updates_cap=4208,canary_cap=32,memory_cap=16,technical_cap=64,technical_recoveries_cap=2,technical_recoveries_used=0,deadline_epoch=None,training_GPU_reservation_s=28800,old_budget_sealed=True))
write('design/TRAINING_ALLOCATION.json',dict(kind='PBS_H200',training_GPU_reservation_s=28800,project='AUTO_NO_PROJECT_OVERRIDE',quota_headroom='UNKNOWN',queue='auto',select='1:ngpus=1:ncpus=12:mem=225gb:gpu_model=H200',mail='n'))
write('design/TRAINING_LAYOUT.json',dict(microbatch=4,accumulation=2,effective_batch=8,workers=2,same_for_both_arms=True,status='FROZEN_PROVEN_NATIVE_LAYOUT'))
prior=json.loads((old/'design/IMPLEMENTATION.json').read_text())
dependencies={{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old/'code').glob('*.py')}}
own={{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').glob('*') if p.is_file()}}
write('design/IMPLEMENTATION.json',dict(release=prior['release'],runtime_ABI_only=True,authority_release=pub['release'],published_commit='PENDING_RUNTIME_PUSH',files=own,native_dependencies=dependencies,codec_identity=prior['codec_identity']))
write('design/RUNTIME_CARD.json',dict(parent_root=str(old),parent_code=str(old/'code'),dependencies=dependencies,host_caps=dict(query=128,head=2048,GPU_seconds=28800)))
readback=json.loads((r/'design/PARENT_ACTUAL14_READBACK.json').read_text());parent=old/'learning/EVENT/main/checkpoint-1024'
parentfiles={{v['path']:dict(bytes=v['bytes'],sha256=v['sha256']) for v in readback['members']}}
for arm,lrs in [('CTRL2048',[1e-4,1e-5]),('LR5X2048',[1e-4,5e-5])]:
 write('design/'+arm+'_CARD.json',dict(authority_release=pub['release'],canonical=pub['canonical'],stage=arm,launch_allowed=True,parent=str(parent),parent_export=str(old/'delivery/EVENT1024/EXPORT.json'),parent_full_master_sha256='ac8fad1a39aad656fac3f0bdafaa856b984d67cd55a0df111103ae79a80f8e2a',parent_phase_step=1024,parent_global_update=5120,parent_files=parentfiles,plan_path=str(r/'data/PAIRED_PLAN16384.jsonl'),plan_sha256=hashlib.sha256((r/'data/PAIRED_PLAN16384.jsonl').read_bytes()).hexdigest(),updates=2048,warmup=103,peak_LRs=lrs,state_dropout=0.,only_scheduler_reset=True,selected='NONE',deployment='NONE'))
print(json.dumps(dict(status='STAGED_OWN_RUNTIME_NO_MODEL_OR_OPTIMIZER_CALL',own_files=own,native_dependencies=dependencies,plan_sha256=hashlib.sha256((r/'data/PAIRED_PLAN16384.jsonl').read_bytes()).hexdigest())))
""")
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'HOPPER_STAGED.json').write_text(out)
    print(json.dumps({'status':'HOPPER_SMALL_RUNTIME_STAGED','scientific_launches':0}))

if __name__=='__main__':stage()
