"""Materialize the sole contingency only after actual primary terminal quality FAIL."""
import argparse,hashlib,json,os,py_compile,shutil,subprocess
from pathlib import Path
from focus_plan import sha,freeze
RELEASE='1a20200ff066ef5a003650110cd20106f9a4bf64'

def read(p):return json.loads(p.read_text())
def owned(root):
    return sum({(p.stat().st_dev,p.stat().st_ino):p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()}.values())

def prepare(primary,train_receipt,dev_receipt,available_bytes):
    parent=primary.parent;root=parent/'R2_FOCUS';assert primary.name=='R2' and not os.environ.get('PBS_JOBID')
    counts=[json.loads(x) for x in (parent/'raw_completed_updates.jsonl').read_text().splitlines()]
    primary_job=next(r['job'] for r in reversed(counts) if r['arm']=='GRIP' and r['phase']=='main')
    terminal=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-xf',primary_job],capture_output=True,text=True)
    assert terminal.returncode==0 and 'job_state = F' in terminal.stdout and 'Exit_status = 0' in terminal.stdout
    completed=read(primary/'GRIP/main/training/SEGMENT_COMPLETE.json');assert completed['actual_update']==4096 and completed['actual_microbatches']==8192
    train=read(train_receipt);assert train['point']=='GRIP4096' and train['lane']=='TRAIN_PHYSICS' and train['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
    assert train['task']['denominator']==4
    failed=train['task']['persistent_stacks']<3
    if not failed:
        assert dev_receipt is not None
        dev=read(dev_receipt);assert dev['point']=='GRIP4096' and dev['lane']=='TEST' and dev['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
        assert dev['task']['denominator']==8;failed=dev['task']['persistent_stacks']<6
    assert failed,'A development-qualified primary must be frozen/confirmed, not adapted'
    for row in read(primary/'EXECUTION_FILESET.json')['files']:assert sha(Path(row['path']))==row['sha256']
    assert read(primary/'evidence/LOADER_EQUIVALENCE.json')['outputs_gradients_rng_scheduler_inputs_losses']=='EXACT'
    checkpoint=primary/'GRIP/main/training/checkpoint-4096'
    for name in ('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','PI0277_RUN.json','CURSOR.json'):
        path=checkpoint/name if name!='CURSOR.json' else checkpoint/'pi0277_provenance/CURSOR.json';assert path.is_file()
    assert read(checkpoint/'trainer_state.json')['global_step']==4096
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in [checkpoint/'config.json',checkpoint/'model.safetensors.index.json',*sorted(checkpoint.glob('model-*.safetensors'))]]
    assert len(files)==4
    identity='LOCAL_GRIP4096@'+hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    root.mkdir(exist_ok=True);(root/'code').mkdir(exist_ok=True);(root/'runtime').mkdir(exist_ok=True)
    assert not (root/'FOCUS/main/training/S4_audit.jsonl').exists(),'Adopt an existing lineage instead of re-preparing'
    copy_names=('cpu_inputs.py','train.py','contexts.py','gripper.py','persistence.py','bench_cpu.py','run.py')
    source_rows=[]
    for name in copy_names:
        source=primary/'code'/name;dest=root/'code'/name
        if not dest.exists():shutil.copyfile(source,dest)
        if name!='train.py':assert dest.read_bytes()==source.read_bytes()
        source_rows.append(dict(path=str(source),sha256=sha(source),bytes=source.stat().st_size))
    # Quota accounting covers the shared cache/snapshots and both lineage directories.
    source=(primary/'code/train.py').read_text();old="for p in root.rglob('*') if p.is_file() and not p.is_symlink()"
    assert source.count(old)==1
    updated=source.replace(old,"for p in budget_root.rglob('*') if p.is_file() and not p.is_symlink()")
    (root/'code/train.py').write_text(updated)
    for name in ('focus_train.py','focus_run.py'):
        source=Path(__file__).parent/name;dest=root/'code'/name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:shutil.copyfile(source,dest)
    source=Path(__file__).parent/'focus.pbs';dest=root/'code/focus.pbs'
    if dest.exists():assert dest.read_bytes()==source.read_bytes()
    else:shutil.copyfile(source,dest)
    for p in (root/'code').glob('*.py'):py_compile.compile(str(p),doraise=True)
    draft=parent/'r2_preparation/focus_design';assert read(draft/'PREPARATION.json')['launch_allowed'] is False
    for name in ('TRAIN_PLAN.json','ELIGIBLE_MEMBERSHIP.json'):
        freeze(root/'design'/name,read(draft/name))
    freeze(root/'design/INPUT_BINDING.json',read(primary/'design/INPUT_BINDING.json'))
    freeze(root/'DONOR_PARAMETER_SCOPE.json',read(primary/'DONOR_PARAMETER_SCOPE.json'))
    freeze(root/'DONOR_LOCAL_READBACK.json',dict(status='PASS_LOCAL_CURRENT_PRIMARY_MODEL_FILES',point='GRIP4096',path=str(checkpoint),revision=identity,
        files=files,complete_trainer_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()),optimizer_inherited=False,primary_complete_receipt_sha256=sha(primary/'GRIP/main/training/SEGMENT_COMPLETE.json')))
    readout=read(primary/'design/GRIP_READOUT.json');readout.update(arm='FOCUS',initial_weights_prefix=str(checkpoint),initial_weights_revision=identity)
    freeze(root/'design/FOCUS_READOUT.json',readout)
    noise=read(primary/'design/NOISE_CARD.json');noise.update(noise_namespace=read(root/'design/TRAIN_PLAN.json')['noise_namespace'],shared_with='FOCUS continuous/split/main')
    freeze(root/'design/NOISE_CARD.json',noise)
    freeze(root/'evidence/LOADER_SELECTION.json',dict(read(primary/'evidence/LOADER_SELECTION.json'),reused_from=str(primary/'evidence/LOADER_SELECTION.json')))
    for name in ('cache','snapshots'):
        link=root/name
        if not link.exists():link.symlink_to(primary/name,target_is_directory=True)
        assert link.resolve()==(primary/name).resolve()
    assert available_bytes>=3*read(root/'DONOR_LOCAL_READBACK.json')['complete_trainer_bytes']+10*2**30,'FOCUS_CANARY_STORAGE_HEADROOM'
    freeze(root/'STORAGE_ADMISSION.json',dict(available_bytes_at_freeze=available_bytes,owned_bytes_at_freeze=owned(parent),ownership_scope=str(parent),quota_report='See source-bound FOCUS_AUTHORITY receipt',qualification_payloads_retired_after_comparison=True))
    assert len(counts)+16+4096<=9048 and sum(r['phase']!='main' for r in counts)+16<=64
    assert len({(r['arm'],r['unit']) for r in counts if r['phase']=='main'})==4096 and not any(r['arm']=='FOCUS' for r in counts)
    authority=dict(status='ACTUAL_PRIMARY4096_DEVELOPMENT_FAIL',launch_allowed=True,release=RELEASE,canonical=6015575969,primary_job=primary_job,primary_terminal_qstat=terminal.stdout,
        train_CPU_receipt=str(train_receipt),train_CPU_receipt_sha256=sha(train_receipt),DEV_CPU_receipt=str(dev_receipt) if dev_receipt else None,
        DEV_CPU_receipt_sha256=sha(dev_receipt) if dev_receipt else None,initializer=identity,new_lineages=1,actual_completed_before_FOCUS=len(counts),available_bytes=available_bytes)
    freeze(root/'FOCUS_AUTHORITY.json',authority)
    adoption=read(primary/'ADOPTION.json');adoption.update(status='FOCUS_CONDITIONAL_ACTUAL_ADOPTION',initial_qualification_budget_origin_preserved=True,actual_completed_before_FOCUS=len(counts),initializer=identity)
    freeze(root/'ADOPTION.json',adoption)
    paths=list((root/'code').glob('*.py'))+list((root/'code').glob('*.pbs'))+list((root/'design').glob('*.json'))+[root/name for name in ('DONOR_PARAMETER_SCOPE.json','DONOR_LOCAL_READBACK.json','STORAGE_ADMISSION.json','ADOPTION.json','FOCUS_AUTHORITY.json')]
    inherited={r['path']:r for r in read(primary/'EXECUTION_FILESET.json')['files']}
    for row in source_rows:inherited[row['path']]=row
    for p in paths:inherited[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
    freeze(root/'EXECUTION_FILESET.json',dict(release=RELEASE,files=sorted(inherited.values(),key=lambda r:r['path']),qualified_math_changed=False,only_scientific_change='Authorized FOCUS source sampler and new branch RNG namespace; inherited primary weights'))
    print(json.dumps(dict(status='FOCUS_READY_AFTER_ACTUAL_PRIMARY_FAIL',root=str(root),launch_allowed=True,formal_updates_max=4096,qualification_updates=16)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--train-receipt',type=Path,required=True)
    p.add_argument('--dev-receipt',type=Path);p.add_argument('--available-bytes',type=int,required=True);a=p.parse_args();prepare(a.primary,a.train_receipt,a.dev_receipt,a.available_bytes)
