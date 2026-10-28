"""Complete immutable Trainer artifact delivery, without model or pickle loading."""
import argparse
import os
from pathlib import Path
import resource
import signal
import time
from types import SimpleNamespace

from scripts.pi0292.checkpoint_artifacts import inventory_bound_sampler
from scripts.pi0292.durable import OwnerLock,read,write
from .native_dataset import sha
from .sampling import BalancedSampler

GIB=1024**3


def capacity(request,owned,free,now):
    if (request['max_owned_bytes']!=180*GIB or not 0<request['growth_reserved_bytes']<=32*GIB
        or owned>request['initial_owned_bytes']+request['growth_reserved_bytes']
        or owned>request['max_owned_bytes'] or free<10*GIB or now>=request['deadline_epoch']):
        raise PermissionError('CPU_ARTIFACT_SPACE_OR_ORIGINAL_TIME_BOUNDARY')


def inventory(checkpoint,dataset,*,expected_step,kind):
    if kind not in ('engineering','main'):raise ValueError('EXPLICIT_ARTIFACT_KIND_REQUIRED')
    checkpoint,dataset=Path(checkpoint),Path(dataset)
    manifest=read(dataset/'DATASET_MANIFEST.json');manifest_sha=sha(dataset/'DATASET_MANIFEST.json')
    stats_sha=sha(dataset/'statistics.json')
    if stats_sha!=manifest['statistics_sha256']:raise ValueError('ACTUAL_FROZEN_DATA_STATISTICS_REQUIRED')
    if checkpoint.is_symlink() or not checkpoint.is_dir():raise ValueError('REGULAR_CHECKPOINT_DIRECTORY_REQUIRED')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):raise ValueError('CHECKPOINT_SYMLINK_REFUSED')
    saved=read(checkpoint/'SAMPLER_STATE.json');recipe=saved['recipe']
    if (kind=='engineering' and (expected_step not in (2,4) or recipe['kind']!='smoke' or recipe['max_steps']!=4)
        or kind=='main' and (not 0<expected_step<=6144 or recipe['kind']!='main' or recipe['max_steps']!=6144)):
        raise ValueError('ORIGINAL_ENGINEERING_OR_MAIN_HORIZON_REQUIRED')
    if kind=='main' and (manifest.get('schema')!='PI0294_FROZEN_MULTITASK_NATIVE_V1' or not manifest.get('frozen')):
        raise ValueError('MAIN_REQUIRES_ACTUAL_FROZEN_MULTITASK_BANK')
    if recipe['effective_batch']!=32 or recipe['seed']!=29400:
        raise ValueError('REGISTERED_BATCH_AND_SAMPLER_SEED_REQUIRED')
    rows=[(row,anchor) for row in manifest['episodes'] if row['split']=='TRAIN' for anchor in range(row['windows'])]
    data=SimpleNamespace(rows=rows,manifest_sha=manifest_sha)
    expected=BalancedSampler(data,seed=29400,stop=recipe['max_steps']*32,smoke=kind=='engineering').state_dict(expected_step*32)
    result=inventory_bound_sampler(checkpoint,expected_step=expected_step,recipe=recipe,
        statistics_sha256=stats_sha,manifest_sha256=manifest_sha,expected_sampler=expected)
    result.update(schema='PI0294_COMPLETE_CHECKPOINT_INVENTORY_V1',kind=kind,
        policy_candidate=kind=='main' and expected_step in (2048,4096,6144),
        selection_qualified=False,selected='NONE',deployment='NONE')
    return result


def verify_remote(api,repo,prefix,revision,files,download,token):
    remote={row.path:row for row in api.list_repo_tree(repo,repo_type='dataset',revision=revision,
        path_in_repo=prefix,recursive=True) if hasattr(row,'size')}
    if set(remote)!={prefix+'/'+name for name in files}:raise ValueError('EXACT_FIXED_ARTIFACT_FILE_SET_REQUIRED')
    for name,expected in files.items():
        row=remote[prefix+'/'+name]
        if row.size!=expected['bytes']:raise ValueError('FIXED_ARTIFACT_BYTES_MISMATCH')
        if row.lfs:
            actual=row.lfs.sha256 if hasattr(row.lfs,'sha256') else row.lfs['sha256']
        else:
            if row.size>5*1024**2:raise PermissionError('LARGE_ARTIFACT_REDOWNLOAD_NOT_ALLOWED_FOR_VERIFICATION')
            actual=sha(download(repo,prefix+'/'+name,repo_type='dataset',revision=revision,token=token))
        if actual!=expected['sha256']:raise ValueError('FIXED_ARTIFACT_SHA_MISMATCH')


def publish(root,checkpoint,dataset,output,repo,*,expected_step,kind,existing_token_file,provenance=None):
    root,checkpoint,output=Path(root),Path(checkpoint),Path(output)
    if (not checkpoint.resolve().is_relative_to(root.resolve()) or not output.resolve().is_relative_to(root.resolve())
        or output.exists()):raise PermissionError('NEW_DELIVERY_INSIDE_OWN_REGISTERED_RUN_REQUIRED')
    lock=OwnerLock(root/'CHECKPOINT_DELIVERY.lock');started=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_SELF);journal=None
    try:
        output.mkdir(parents=True,exist_ok=False)
        journal=dict(status='INVENTORYING_ACTUAL_FULL_CHECKPOINT',kind=kind,global_step=expected_step,
            model_loaded=False,optimizer_pickle_loaded=False,local_deletion=False,selected='NONE',deployment='NONE')
        write(output/'DELIVERY.json',journal)
        bound=inventory(checkpoint,dataset,expected_step=expected_step,kind=kind)
        bound['provenance']=provenance
        write(output/'CHECKPOINT_MANIFEST.json',bound)
        os.environ['HF_HUB_OFFLINE']='0'
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        token=Path(existing_token_file).read_text().strip()
        api=HfApi(token=token);info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise PermissionError('EXISTING_PRIVATE_ARTIFACT_REPOSITORY_REQUIRED')
        identity=sha(output/'CHECKPOINT_MANIFEST.json')
        prefix='PI0294/'+root.name+'/checkpoints/'+kind+'-M'+str(expected_step)+'-'+identity[:12]
        if any(name.startswith(prefix+'/') for name in api.list_repo_files(repo,repo_type='dataset')):
            raise FileExistsError('FRESH_CHECKPOINT_PREFIX_REQUIRED_NO_OVERWRITE')
        files=dict(bound['files'])
        files['CHECKPOINT_MANIFEST.json']=dict(bytes=(output/'CHECKPOINT_MANIFEST.json').stat().st_size,sha256=identity)
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,
            path_or_fileobj=str(output/name if name=='CHECKPOINT_MANIFEST.json' else checkpoint/name)) for name in files]
        journal.update(status='COMMITTING_FULL_STATE_PAYLOAD_NO_LOCAL_ARCHIVE_COPY',prefix=prefix,repo_id=repo,
            manifest_sha256=identity,total_checkpoint_bytes=bound['total_bytes'],files=files)
        write(output/'DELIVERY.json',journal)
        commit=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='PI0294 complete '+kind+' checkpoint, not policy selection',operations=operations)
        revision=commit.oid
        journal.update(status='PAYLOAD_COMMITTED_FIXED_READBACK_PENDING',data_revision=revision)
        write(output/'DELIVERY.json',journal)
        verify_remote(api,repo,prefix,revision,files,hf_hub_download,token)
        # Recheck the source bytes after upload, not just a stale pre-upload hash.
        for name,expected in bound['files'].items():
            if sha(checkpoint/name)!=expected['sha256']:raise ValueError('LOCAL_CHECKPOINT_CHANGED_DURING_DELIVERY')
        ready=dict(schema='PI0294_CHECKPOINT_READY_V1',status='FULL_STATE_FIXED_REVISION_SHA_VERIFIED',
            private=True,repo_id=repo,prefix=prefix,data_revision=revision,files=files,kind=kind,
            global_step=expected_step,manifest_sha256=identity,model_loaded=False,optimizer_pickle_loaded=False,
            independent_resume_qualification=False,policy_selection_qualified=False,
            READY_uploaded_last=True,selected='NONE',deployment='NONE')
        write(output/'READY.json',ready)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=revision,
            commit_message='PI0294 checkpoint READY after complete fixed SHA readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        files['READY.json']=dict(bytes=(output/'READY.json').stat().st_size,sha256=sha(output/'READY.json'))
        verify_remote(api,repo,prefix,final,files,hf_hub_download,token)
        journal.update(status='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK',revision=final,
            READY_uploaded_last=True,independent_resume_qualification=False,policy_selection_qualified=False)
        return journal
    except BaseException as error:
        if journal is not None:journal.update(status='FAILED_OR_UNKNOWN_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__)
        raise
    finally:
        after=resource.getrusage(resource.RUSAGE_SELF)
        if journal is not None:
            journal.update(CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                wall_s=time.monotonic()-started)
            write(output/'DELIVERY.json',journal)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','checkpoint','dataset','output','repo','existing-token-file','request'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--kind',choices=['engineering','main'],required=True)
    p.add_argument('--expected-step',type=int,required=True)
    args=p.parse_args();root=Path(args.root)
    from .run_admission import validate,DEADLINE
    request_path=Path(args.request)
    if not request_path.resolve().is_relative_to(root.resolve()):raise PermissionError('OWN_CPU_ARTIFACT_REQUEST_REQUIRED')
    request=read(request_path)
    manifest=validate(root,profile='hopper-neural',closeout=True)
    if (request['run_uuid']!=root.name or request['source_sha']!=manifest['code']['vla_commit']
        or request['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
        or request['purpose']!='CPU_ONLY_COMPLETE_CHECKPOINT_DELIVERY_NO_MODEL'
        or any(request[k]!=getattr(args,k) for k in ('checkpoint','dataset','output','kind','expected_step'))):
        raise PermissionError('EXACT_FIXED_CPU_ARTIFACT_REQUEST_REQUIRED')
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if (ledger.get('CPU_artifact_request_sha256')!=sha(request_path) or ledger.get('CPU_reserved_s')!=1800
        or ledger.get('pending_updates',0) or ledger.get('GPU_reserved_s',0)):
        raise PermissionError('SOLE_OWNER_CPU_RESERVATION_REQUIRED_NO_UNRECONCILED_NEURAL_WORK')
    if sha(Path(args.checkpoint)/'CHECKPOINT_COMPLETE.json')!=request['checkpoint_complete_sha256']:
        raise PermissionError('EXACT_COMPLETE_CHECKPOINT_BOUNDARY_REQUIRED')
    origin_path=Path(request['origin_request']);allocation_path=Path(request['origin_allocation'])
    if any(not path.resolve().is_relative_to(root.resolve()) for path in (origin_path,allocation_path)):
        raise PermissionError('OWN_ORIGINAL_TRAINING_PROOF_REQUIRED')
    original=read(origin_path);allocation=read(allocation_path)
    entries=[row for row in ledger['PBS_submissions'] if row['request_sha256']==sha(origin_path)
        and row['status']=='RECONCILED_TERMINAL']
    if (len(entries)!=1 or entries[0]['source_sha']!=original['source_sha']
        or allocation['request_sha256']!=sha(origin_path) or allocation['job_id']!=entries[0]['job_id']):
        raise PermissionError('EXACT_KNOWN_TERMINAL_LEGAL_TRAINING_ORIGIN_REQUIRED')
    provenance=dict(origin_source_sha=original['source_sha'],origin_request_sha256=sha(origin_path),
        origin_allocation_sha256=sha(allocation_path),origin_job_id=allocation['job_id'],
        artifact_delivery_source_sha=manifest['code']['vla_commit'],protocol_release=manifest['authority']['protocol_ref'],
        original_T0_epoch=manifest['limits']['T0_epoch'],no_training_reexecuted=True)
    if not 0<request['deadline_epoch']-time.time()<=1800 or request['deadline_epoch']>DEADLINE:
        raise PermissionError('ORIGINAL_BOUNDED_CPU_DELIVERY_DEADLINE_REQUIRED')
    cpus=request['cpus']
    if len(cpus)!=1 or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise PermissionError('SINGLE_ALLOWED_CPU_REQUIRED')
    os.sched_setaffinity(0,set(cpus));os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_XET_CACHE=str(root/'delivery/hf-xet-cache'))
    def guard(signum=None,frame=None):
        import shutil
        owned=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        capacity(request,owned,shutil.disk_usage(root).free,time.time())
        signal.alarm(1)
    signal.signal(signal.SIGALRM,guard);guard()
    owner=OwnerLock(root/'NEURAL_OWNER.lock')
    try:
        publish(root,args.checkpoint,args.dataset,args.output,args.repo,expected_step=args.expected_step,
            kind=args.kind,existing_token_file=args.existing_token_file,provenance=provenance)
    finally:
        signal.alarm(0);owner.close()
