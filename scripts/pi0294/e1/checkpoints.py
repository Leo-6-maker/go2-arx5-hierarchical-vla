"""Private E1 checkpoint delivery with no model/pickle load or local archive copy."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.checkpoint_delivery import verify_remote
from .scope import validate,Ledger,STATS

LABELS={'M1-256':('M1',256),'M1-1024':('M1',1024),'M2-BC':('M2-BC',512),'M2-CORR':('M2-CORR',512)}


def publish(root,label,repo,token_file):
    root=Path(root);runtime=validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard();start=time.process_time()
    if label not in LABELS:raise PermissionError('E1_FIXED_CHECKPOINT_LABEL_ONLY')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('E1_CHECKPOINT_PUBLISH_CPU_ONLY')
    if read(ledger.path)['GPU_reserved_s'] or any(x['status']!='NATIVE_TERMINAL' for x in read(ledger.path)['PBS_submissions']):
        raise PermissionError('E1_CHECKPOINT_PUBLICATION_AFTER_NATIVE_GPU_SETTLEMENT')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    branch,step=LABELS[label];checkpoint=root/'training'/branch/('checkpoint-'+str(step))
    complete=read(checkpoint/'CHECKPOINT_COMPLETE.json');files=complete['files'];owner=OwnerLock(root/'E1_MODEL_HF_PUBLISH.lock')
    output=root/'artifacts'/label;output.mkdir(parents=True,mode=0o700);journal=dict(status='INVENTORYING_ACTUAL_STATE',label=label,READY_claim=False)
    try:
        if complete['global_step']!=step or complete['full_resume']!=(label!='M1-256'):raise ValueError('E1_CHECKPOINT_KIND_OR_STEP_MISMATCH')
        if any(p.is_symlink() for p in checkpoint.rglob('*')):raise PermissionError('NO_CHECKPOINT_SYMLINKS')
        if {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()}!=set(files)|{'CHECKPOINT_COMPLETE.json'}:raise ValueError('E1_COMPLETE_CHECKPOINT_FILE_SET')
        for name,digest in files.items():
            ledger.guard()
            if not (checkpoint/name).resolve().is_relative_to(checkpoint.resolve()) or sha(checkpoint/name)!=digest:raise ValueError('E1_CHECKPOINT_BYTES_CHANGED:'+name)
        if sha(checkpoint/'statistics.json')!=STATS:raise ValueError('E1_CHECKPOINT_STATS_CHANGED')
        sampler=read(checkpoint/'SAMPLER_STATE.json');state=read(checkpoint/'trainer_state.json')
        if state['global_step']!=step or sampler['sampler']['cursor']!=step*32 or sampler['recipe']['seed']!=20261016:
            raise ValueError('E1_ACTUAL_NATIVE_STEP_SAMPLER_RECIPE')
        inventory={n:dict(sha256=s,bytes=(checkpoint/n).stat().st_size) for n,s in files.items()}
        inventory['CHECKPOINT_COMPLETE.json']=dict(sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json'),bytes=(checkpoint/'CHECKPOINT_COMPLETE.json').stat().st_size)
        allocation_dir=root/'pbs'/('alloc-01' if branch=='M1' else 'alloc-02');origin=read(allocation_dir/'ALLOCATION_ADMISSION.json');request=read(allocation_dir/'REQUEST.json')
        if origin['request_sha256']!=sha(allocation_dir/'REQUEST.json'):raise ValueError('E1_CHECKPOINT_ORIGIN_REQUEST_CHANGED')
        manifest=dict(schema='E1_FIXED_CHECKPOINT_INVENTORY_V1',label=label,global_step=step,full_native_resume_state=label!='M1-256',files=inventory,
            statistics_sha256=STATS,recipe=sampler['recipe'],sampler=sampler['sampler'],model_loaded=False,optimizer_pickle_loaded=False,
            independent_resume_test=False,origin_job_id=origin['job_id'],origin_source_sha=request['source_sha'],origin_allocation_sha256=sha(allocation_dir/'ALLOCATION_ADMISSION.json'),
            artifact_source_sha=runtime['code']['vla_commit'],authority=runtime['authority']['protocol_ref'],selected='NONE',deployment='NONE')
        write(output/'CHECKPOINT_MANIFEST.json',manifest)
        inventory['CHECKPOINT_MANIFEST.json']=dict(sha256=sha(output/'CHECKPOINT_MANIFEST.json'),bytes=(output/'CHECKPOINT_MANIFEST.json').stat().st_size)
        os.environ.update(HF_HUB_OFFLINE='0',HF_XET_CACHE=str(root/'artifacts/hf-xet-cache'))
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise PermissionError('E1_NEW_CHECKPOINTS_PRIVATE_HF_ONLY')
        prefix='PI0294/'+root.name+'/checkpoints/'+label+'-'+sha(output/'CHECKPOINT_MANIFEST.json')[:12]
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(repo,repo_type='dataset')):raise FileExistsError('PRESERVE_CHECKPOINT_PREFIX_NO_BLIND_REPUBLISH')
        journal.update(status='UPLOAD_INTENT_UNKNOWN_NO_AUTOMATIC_RETRY',repo_id=repo,prefix=prefix,files=inventory,
            manifest_sha256=sha(output/'CHECKPOINT_MANIFEST.json'),total_checkpoint_bytes=sum(v['bytes'] for v in inventory.values()))
        write(output/'DELIVERY.private.json',journal)
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(output/name if name=='CHECKPOINT_MANIFEST.json' else checkpoint/name)) for name in inventory]
        revision=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,commit_message='E1 '+label+' immutable native state, not selection',operations=operations).oid
        journal.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=revision);write(output/'DELIVERY.private.json',journal)
        verify_remote(api,repo,prefix,revision,inventory,hf_hub_download,token)
        for name,digest in files.items():
            if sha(checkpoint/name)!=digest:raise ValueError('E1_LOCAL_CHECKPOINT_CHANGED_DURING_UPLOAD')
        ready=dict(status='E1_FIXED_CHECKPOINT_SHA_VERIFIED',label=label,global_step=step,full_native_resume_state=label!='M1-256',
            files=inventory,data_revision=revision,manifest_sha256=journal['manifest_sha256'],READY_uploaded_last=True,
            model_loaded=False,optimizer_pickle_loaded=False,independent_resume_test=False,selected='NONE',deployment='NONE')
        write(output/'READY.json',ready)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=revision,commit_message='E1 '+label+' READY after exact fixed SHA readback',operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        inventory['READY.json']=dict(sha256=sha(output/'READY.json'),bytes=(output/'READY.json').stat().st_size)
        verify_remote(api,repo,prefix,final,inventory,hf_hub_download,token)
        journal.update(status='PASS_E1_PRIVATE_CHECKPOINT_FIXED_REVISION_READY_LAST',revision=final,READY_claim=True,
            full_native_resume_state=label!='M1-256',independent_resume_test=False)
        write(output/'DELIVERY.private.json',journal);return journal
    finally:
        ledger.cpu(time.process_time()-start);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',required=True,choices=sorted(LABELS));p.add_argument('--repo',required=True);p.add_argument('--existing-token-file',required=True);a=p.parse_args()
    v=publish(a.root,a.label,a.repo,a.existing_token_file);print(json.dumps({k:x for k,x in v.items() if k in ['status','label','revision','prefix','total_checkpoint_bytes']}),flush=True)
