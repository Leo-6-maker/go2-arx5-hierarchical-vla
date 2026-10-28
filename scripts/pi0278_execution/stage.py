"""Small immutable code/plan staging over existing SSH; no model payload on PC."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0278'
CORE='/home/liuyu-nfs/Projects/go2-arx5/PI0278'
RELEASE='83af9099f3bab9a4ac08eaa85380c2cfd6351f34'

def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.pbs')}
    names=['TRAIN_PLAN.json','DIAGNOSTIC48.json','PROSPECTIVE_COVERAGE.json','PREDECESSOR_ACTUAL_EXPOSURE.json','PREDECESSOR_DEV_EVENT_AUDIT.json','LATE_SAVED_OBSERVATION_DIVERSITY.json']
    plans=json.loads(remote('core',"import base64,json\nfrom pathlib import Path\nr=Path("+repr(CORE)+");print(json.dumps({n:base64.b64encode((r/'design'/n).read_bytes()).decode() for n in "+repr(names)+"}))"))
    authority={Path(p).name:base64.b64encode(Path(p).read_bytes()).decode() for p in ['docs/protocols/PI0278_SINGLE_TASK_FULL_SUPPORT_COVERAGE_FINETUNING_V1.md','reports/PI0278/design/REVIEW_AND_EXECUTION_LIMITS.json']}
    script='''import base64,hashlib,json,os
from pathlib import Path
base=Path(HOP);root=base/'run';old=base.with_name('PI0277');focus=old/'R2_FOCUS';donor=focus/'FOCUS/main/training/checkpoint-4096'
assert not (root/'COVER93/main/training/S4_audit.jsonl').exists(),'Adopt existing training instead of restaging'
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,sort_keys=True)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==data,p
 else:p.write_bytes(data)
for folder in ['code','design','evidence','runtime','cache','snapshots']:(root/folder).mkdir(parents=True,exist_ok=True)
for n,b in FILES.items():(root/'code'/n).write_bytes(base64.b64decode(b))
for n,b in PLANS.items():
 p=root/'design'/n;v=base64.b64decode(b)
 if p.exists():assert p.read_bytes()==v
 else:p.write_bytes(v)
for n,b in AUTHORITY.items():
 p=root/'authority'/n;p.parent.mkdir(exist_ok=True);p.write_bytes(base64.b64decode(b))
binding=read(focus/'design/INPUT_BINDING.json');write(root/'design/INPUT_BINDING.json',binding)
scope=read(focus/'DONOR_PARAMETER_SCOPE.json');write(root/'DONOR_PARAMETER_SCOPE.json',scope)
ready=read(old/'R2/snapshots/FOCUS4096/READY.json');model=[r for r in ready['files'] if r['path'] in ['config.json','model.safetensors.index.json'] or r['path'].endswith('.safetensors')]
for r in model:assert (donor/r['path']).stat().st_size==r['bytes']
identity='LOCAL_FOCUS4096@93e500e8ecb6a719d4a53be09d23b35bdeeb83fcc61e75416247a2c0380f4604'
donor_binding=dict(status='BOUND_EXISTING_FULL_READBACK_HASHES_REVALIDATE_ON_ALLOCATED_NODE',point='FOCUS4096',path=str(donor),revision=identity,files=model,optimizer_inherited=False,complete_trainer_bytes=sum(p.stat().st_size for p in donor.rglob('*') if p.is_file()))
write(root/'DONOR_LOCAL_READBACK.json',donor_binding);write(base/'DONOR_LOCAL_READBACK.json',donor_binding)
readout=read(donor/'PI0277_READOUT_CONTRACT.json');readout.update(arm='COVER93',initial_weights_prefix=str(donor),initial_weights_revision=identity,governing_release=RELEASE,objective_phase='PI0278_FINAL_PARENT_OBJECTIVE_FROM_FIRST_UPDATE')
write(root/'design/COVER93_READOUT.json',readout)
retired=read(base/'evidence/STORAGE_RETIREMENT_DONE.json');q=retired['quota_before'];raw=Path(q['source']);rows=[x.split() for x in raw.read_text().splitlines() if '/e1192548 ' in x]
if len(rows)==1:
 current=rows[0];used=int(current[4]);hard=int(current[5]);available=hard-used
 if used==q['used_bytes']:available+=retired['actual_unlinked_unique_inode_allocated_bytes']
 quota=dict(status='ACTUAL_ADMIN_QUOTA_WITH_EXACT_DELETION_ADJUSTMENT_IF_REPORT_STILL_UNCHANGED',used_bytes=used,hard_bytes=hard,report_mtime_ns=raw.stat().st_mtime_ns,pre_report_deletion_adjustment=used==q['used_bytes'])
else:
 available=q['hard_bytes']-q['used_bytes']+retired['actual_unlinked_unique_inode_allocated_bytes'];quota=dict(status='ADMIN_REPORT_REFRESH_CURRENT_UNKNOWN_PREVIOUS_ACTUAL_REPORT_AND_EXACT_UNLINKS_USED',previous=q)
owned=sum({(p.stat().st_dev,p.stat().st_ino):p.stat().st_size for p in base.rglob('*') if p.is_file() and not p.is_symlink()}.values())
assert available>=3*donor_binding['complete_trainer_bytes']+10*2**30,'ACTUAL_CANARY_ATOMIC_QUOTA_HEADROOM'
write(root/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=available,owned_bytes_at_freeze=owned,ownership_scope=str(base),quota=quota,unique_inode_reclaimed=retired['actual_unlinked_unique_inode_allocated_bytes']))
write(root/'ADOPTION.json',dict(release=RELEASE,canonical=6027913340,implementation_commit=COMMIT,completed_updates_at_adoption=0,new_budget=True,old_terminal_preserved=True,selected_checkpoint='NONE',deployment='NONE'))
caps=read(root/'authority/REVIEW_AND_EXECUTION_LIMITS.json')['new_hard_caps']
write(root/'RUN_CONFIG.json',dict(release=RELEASE,launch_allowed=True,donor_revision=identity,caps=caps,reuse_cache=str(old/'R2/cache'),bound_source=str(old/'R3_BOUND/code/bound_train.py'),canary_comparison_source=str(old/'code/qualify.py')))
write(root/'evidence/LOADER_SELECTION.json',dict(route='FAST_CACHE2',source=str(focus/'evidence/LOADER_SELECTION.json'),new_tournament=False,cold_cache_streaming=True))
shared=['contexts.py','gripper.py','persistence.py','bench_cpu.py']
for n in shared:
 if n not in FILES:
  src=focus/'code'/n;dst=root/'code'/n;dst.write_bytes(src.read_bytes())
paths=list((root/'code').glob('*'))+list((root/'design').glob('*.json'))+list((root/'authority').glob('*'))+[root/n for n in ['DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json','ADOPTION.json','RUN_CONFIG.json']]
inherited=read(focus/'EXECUTION_FILESET.json')['files'];rows={r['path']:r for r in inherited}
for p in paths+[old/'R3_BOUND/code/bound_train.py',old/'R3_BOUND/code/bound_loss.py']:
 if p.is_file():rows[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
write(root/'EXECUTION_FILESET.json',dict(release=RELEASE,implementation_commit=COMMIT,files=sorted(rows.values(),key=lambda r:r['path']),old_sources_modified=False,authority=6027913340,legacy_metadata_filenames_only=True))
write(base/'evidence/STAGED.json',dict(status='PASS_SOURCE_BOUND_CPU_PLAN_AND_LOCAL_DONOR_STAGING',root=str(root),implementation_commit=COMMIT,plan_sha256=hashlib.sha256((root/'design/TRAIN_PLAN.json').read_bytes()).hexdigest(),quota=quota,available_bytes_at_freeze=available,launch_allowed=True,training_actual='NOT_YET_SUBMITTED'))
print(json.dumps(read(base/'evidence/STAGED.json')))
'''
    script='\n'.join(f'{key}={value!r}' for key,value in [('HOP',HOP),('CORE',CORE),('FILES',files),('PLANS',plans),('AUTHORITY',authority),('RELEASE',RELEASE),('COMMIT',commit)])+'\n'+script
    print(remote('hop',script))
    out=Path('reports/PI0278/execution')
    for n,b in plans.items():
        if n not in ['TRAIN_PLAN.json']:(out/n).write_bytes(base64.b64decode(b))
    result=json.loads(remote('hop',f"from pathlib import Path\nprint((Path({HOP!r})/'evidence/STAGED.json').read_text())"));(out/'STAGED.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
