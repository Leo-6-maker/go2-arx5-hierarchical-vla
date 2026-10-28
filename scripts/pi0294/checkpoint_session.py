"""CPU artifact boundary in the same own allocation, no optimizer/inference."""
import argparse
from pathlib import Path
import resource
import shutil
import signal
import time

from scripts.pi0292.durable import read,write,OwnerLock
from .native_dataset import sha
from .neural import verify_allocation
from .checkpoint_delivery import publish,GIB
from .run_admission import DEADLINE


def retirement(root,checkpoint,delivery,replacement,*,expected_step):
    root,checkpoint,replacement=Path(root),Path(checkpoint),Path(replacement)
    if checkpoint!=root/'training/main'/('checkpoint-'+str(expected_step)) or checkpoint.is_symlink():
        raise PermissionError('ONLY_EXACT_OWN_GENERATED_MAIN_INTERMEDIATE')
    if expected_step not in (512,2048) or replacement==checkpoint or not replacement.resolve().is_relative_to(root/'training/main'):
        raise PermissionError('ONLY_OWN_INTERMEDIATE_AFTER_LATER_FULL_STATE')
    later=read(replacement/'CHECKPOINT_COMPLETE.json')
    if (later['global_step']<=expected_step or replacement!=root/'training/main'/('checkpoint-'+str(later['global_step']))
        or replacement.is_symlink() or later['status']!='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING'):
        raise PermissionError('LATER_COMPLETE_RESUME_MUST_BE_RETAINED')
    for name,digest in later['required_files'].items():
        if sha(replacement/name)!=digest:raise ValueError('RETAINED_LATER_FULL_STATE_CHANGED')
    if delivery['status']!='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK' or delivery['global_step']!=expected_step:
        raise PermissionError('ACTUAL_FIXED_REVISION_FULL_ARTIFACT_READBACK_REQUIRED')
    if expected_step==2048:
        panel=read(root/'evidence/FIXED_DEV_RESULTS.json')
        from .neural import dev_panel,choose_candidate
        dev_panel(panel['M2048']);dev_panel(panel['M4096'])
        if choose_candidate({'2048':panel['M2048'],'4096':panel['M4096']})==2048:
            raise PermissionError('EARLIER_DEV_WINNER_MUST_REMAIN_LOCAL')
        candidate=root/'evidence/CANDIDATE_SELECTION.json'
        if candidate.exists() and read(candidate)['selected_step']==2048:
            raise PermissionError('FROZEN_SELECTED_CANDIDATE_MUST_REMAIN_LOCAL')
    files={name:row for name,row in delivery['files'].items() if name not in ('READY.json','CHECKPOINT_MANIFEST.json')}
    actual={str(path.relative_to(checkpoint)) for path in checkpoint.rglob('*') if path.is_file()}
    if actual!=set(files):raise ValueError('EXACT_GENERATED_ARTIFACT_MEMBER_SET_REQUIRED')
    for name,row in files.items():
        path=checkpoint/name
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('ARTIFACT_CHANGED_AFTER_FIXED_REVISION_VERIFICATION')
    receipt=root/'evidence'/('RETIRED_GENERATED_MAIN_M'+str(expected_step)+'.json')
    if receipt.exists():raise FileExistsError('Never repeat a local artifact retirement')
    proof=dict(status='VALIDATED_OWN_INTERMEDIATE_FULL_HF_AND_LATER_LOCAL_RESUME',global_step=expected_step,
        revision=delivery['revision'],prefix=delivery['prefix'],files=files,later_complete_sha256=sha(replacement/'CHECKPOINT_COMPLETE.json'),
        removed_bytes=sum(row['bytes'] for row in files.values()),failed_checkpoint_deleted=False,historical_data_deleted=False)
    write(receipt,proof)
    shutil.rmtree(checkpoint)
    proof['status']='ACTUAL_OWN_INTERMEDIATE_LOCAL_COPY_REMOVED_FULL_HF_PRESERVED';write(receipt,proof)
    return proof


def run(args):
    root=Path(args.root);request=read(args.request)
    allocation=verify_allocation(root,'checkpoint',args.request)
    if request['purpose']!='FULL_CHECKPOINT_DELIVERY_BETWEEN_OPTIMIZATION_STAGES':
        raise PermissionError('PREREGISTERED_ARTIFACT_BOUNDARY_REQUIRED')
    if request['repo']!='leoyuer/pi0291-arx5-fixed-base':
        raise PermissionError('EXISTING_PRIVATE_HANDOFF_REPOSITORY_ONLY')
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if ledger.get('pending_updates',0):raise PermissionError('NO_CPU_ARTIFACT_WORK_DURING_UNKNOWN_OPTIMIZER_UPDATE')
    lock=OwnerLock(root/'NEURAL_OWNER.lock')
    output=Path(request['output']);checkpoint=Path(request['checkpoint'])
    deadline=min(DEADLINE,allocation['allocation_end_epoch']-120,time.time()+1800)
    def guard(signum=None,frame=None):
        owned=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if owned>180*GIB or time.time()>=deadline or shutil.disk_usage(root).free<10*GIB:
            raise PermissionError('ORIGINAL_ARTIFACT_PEAK_AND_TIME_BOUNDARY')
        signal.alarm(1)
    signal.signal(signal.SIGALRM,guard);guard()
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
    try:
        bindings=read(request['bindings'])
        delivery=publish(root,checkpoint,bindings['dataset'],output,request['repo'],
            expected_step=request['expected_step'],kind='main',existing_token_file=request['existing_token_file'],
            provenance=dict(origin_source_sha=request['source_sha'],origin_job_id=allocation['job_id'],
                original_request_sha256=sha(args.request),same_own_allocation_CPU_boundary=True,model_or_optimizer_executed=False))
        if request.get('retire_after_verified_delivery'):
            retirement(root,checkpoint,delivery,request['retained_later_checkpoint'],expected_step=request['expected_step'])
    finally:
        signal.alarm(0)
        after=resource.getrusage(resource.RUSAGE_SELF)
        path=Path(args.request).parent/'CPU_ARTIFACT_STAGE.json'
        write(path,dict(CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            wall_s=time.monotonic()-started,job_id=allocation['job_id'],
            accounting='CPU_ALREADY_INCLUDED_IN_NATIVE_PBS_ALLOCATION_RECEIPT_NOT_ADDITIVE',new_PBS=0,new_optimizer_updates=0))
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True)
    run(p.parse_args())
