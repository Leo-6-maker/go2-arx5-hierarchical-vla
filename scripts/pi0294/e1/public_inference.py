"""Explicitly authorized public inference export; native training state stays private."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import struct
import time

from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.checkpoint_delivery import verify_remote
from scripts.pi0294.public_checkpoint import public_config,reject_private
from .checkpoints import checkpoint_spec
from .scope import validate,Ledger,STATS

PUBLIC_REPO='leoyuer/pi0294-public-artifacts'
BASE_REVISION='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
COSMOS_REVISION='9ce19a195e423419c349abfc86fd07178b230561'


def inference_members(files,index):
    names={'config.json','statistics.json','model.safetensors.index.json',
        'processor/embodiment_id.json','processor/processor_config.json','processor/statistics.json'}
    shards=set(index['weight_map'].values())
    if not shards or any(not re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors',n) for n in shards):
        raise ValueError('EXACT_SHARDED_INFERENCE_WEIGHT_INDEX_REQUIRED')
    names.update(shards)
    if not names<=set(files):raise ValueError('INCOMPLETE_NATIVE_INFERENCE_MEMBERS')
    return sorted(names)


def tensor_header(path):
    with Path(path).open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('TRUNCATED_SAFETENSORS_HEADER')
        length=struct.unpack('<Q',raw)[0]
        if not 0<length<=10*1024**2:raise ValueError('BOUNDED_SAFETENSORS_HEADER_REQUIRED')
        reject_private(json.loads(f.read(length)))


def publish(root,label,token_file):
    root=Path(root);runtime=validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard(closeout=True)
    amendment=read(root/'evidence/PUBLIC_INFERENCE_ARTIFACT_AUTHORITY.json')
    if amendment!=dict(authority='DIRECT_USER_HF_PUBLIC_STORAGE_PERMISSION',run_uuid=root.name,
            repo_id=PUBLIC_REPO,scope='SANITIZED_INFERENCE_ONLY_NO_OPTIMIZER_RNG_OR_PRIVATE_REPO_VISIBILITY_CHANGE',new_scientific_resources=False):
        raise PermissionError('EXPLICIT_BOUND_PUBLIC_INFERENCE_AUTHORITY_REQUIRED')
    l=read(ledger.path)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or l['GPU_reserved_s'] or any(r['status']!='NATIVE_TERMINAL' for r in l['PBS_submissions']):
        raise PermissionError('CPU_ONLY_AFTER_NATIVE_GPU_SETTLEMENT')
    branch,step,partial=checkpoint_spec(label)
    if partial:raise PermissionError('FIXED_PREDECLARED_INFERENCE_CHECKPOINT_ONLY')
    owner=OwnerLock(root/'E1_MODEL_HF_PUBLISH.lock');started=time.process_time();journal=dict(status='AUDITING_PUBLIC_INFERENCE_ALLOWLIST')
    out=root/'artifacts'/('public-inference-'+label)
    created=False;old_handler=signal.getsignal(signal.SIGALRM)
    def deadline(signum,frame):raise TimeoutError('ORIGINAL_E1_PUBLIC_ARTIFACT_HARD_CLOSEOUT')
    try:
        signal.signal(signal.SIGALRM,deadline)
        remaining=read(root/'E1_ADOPTION_AND_BINDING.json')['hard_closeout_epoch']-time.time()
        if remaining<=60:raise TimeoutError('ORIGINAL_E1_PUBLIC_ARTIFACT_CLOSEOUT_RESERVE')
        signal.alarm(int(remaining-30))
        out.mkdir(mode=0o700,exist_ok=False);created=True;checkpoint=root/'training'/branch/('checkpoint-'+str(step))
        complete=read(checkpoint/'CHECKPOINT_COMPLETE.json');files=complete['files']
        if complete['global_step']!=step or sha(checkpoint/'statistics.json')!=STATS:raise ValueError('EXACT_NATIVE_STEP_AND_STATISTICS')
        job=complete.get('origin_job_id')
        candidates=[p.parent for p in (root/'pbs').glob('alloc-*/ALLOCATION_ADMISSION.json') if read(p)['job_id']==job] if job else [root/'pbs'/('alloc-01' if branch=='M1' else 'alloc-02')]
        if len(candidates)!=1:raise ValueError('EXACT_NATIVE_SAVING_ALLOCATION')
        allocation=candidates[0];origin=read(allocation/'ALLOCATION_ADMISSION.json');request=read(allocation/'REQUEST.json')
        if origin['request_sha256']!=sha(allocation/'REQUEST.json') or not any(r['job_id']==origin['job_id'] and r['status']=='NATIVE_TERMINAL' for r in l['PBS_submissions']):
            raise ValueError('ORIGINAL_LEGAL_ALLOCATION_IDENTITY')
        if complete.get('origin_source_sha',request['source_sha'])!=request['source_sha']:raise ValueError('ORIGINAL_SAVING_SOURCE_IDENTITY')
        names=inference_members(files,read(checkpoint/'model.safetensors.index.json'));paths={};changes={}
        for name in names:
            ledger.guard(closeout=True);source=checkpoint/name
            if source.is_symlink() or not source.resolve().is_relative_to(checkpoint.resolve()) or sha(source)!=files[name]:
                raise ValueError('NATIVE_INFERENCE_MEMBER_CHANGED:'+name)
            if name in ['config.json','processor/processor_config.json']:
                target=out/name;target.parent.mkdir(exist_ok=True,parents=True)
                write(target,public_config(read(source),processor=name.startswith('processor/')))
                changes[name]=dict(original_sha256=files[name],change='ONLY_PRIVATE_COSMOS_MODEL_NAME_TO_PUBLIC_REPO_ID')
            else:
                target=source
                if name.endswith('.json'):reject_private(read(source))
                else:tensor_header(source)
            paths[name]=target
        b=read(root/'NEURAL_BINDINGS.private.json')
        if b['base_revision']!=BASE_REVISION or b['cosmos_revision']!=COSMOS_REVISION:raise ValueError('EXACT_LICENSED_UPSTREAM_REVISIONS')
        os.environ.update(HF_HUB_OFFLINE='0',HF_XET_CHUNK_CACHE_SIZE_BYTES='0',HF_XET_CACHE=str(out/'xet-cache'))
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download,ModelCard
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(PUBLIC_REPO,repo_type='dataset')
        if info.private:raise PermissionError('EXISTING_PUBLIC_REPOSITORY_ONLY_NO_VISIBILITY_CHANGE')
        for name,repo,revision,source_name in [
            ('LICENSE','nvidia/GR00T-N1.7-3B',BASE_REVISION,'LICENSE'),
            ('UPSTREAM_GR00T_README.md','nvidia/GR00T-N1.7-3B',BASE_REVISION,'README.md'),
            ('UPSTREAM_COSMOS_README.md','nvidia/Cosmos-Reason2-2B',COSMOS_REVISION,'README.md')]:
            paths[name]=Path(hf_hub_download(repo,source_name,revision=revision,token=token))
        agreement=ModelCard.load(paths['UPSTREAM_COSMOS_README.md']).data.to_dict()['extra_gated_prompt']
        (out/'LICENSE_COSMOS.txt').write_text(agreement+'\n');paths['LICENSE_COSMOS.txt']=out/'LICENSE_COSMOS.txt'
        (out/'NOTICE.txt').write_text('Licensed by NVIDIA Corporation under the NVIDIA Open Model License\nBuilt on NVIDIA Cosmos\nGR00T derivative: non-commercial research/evaluation only under LICENSE.\n')
        paths['NOTICE.txt']=out/'NOTICE.txt'
        (out/'README.md').write_text('# E1 '+label+' research inference checkpoint\n\nNon-commercial research/evaluation only. Built on NVIDIA Cosmos.\nLicensed by NVIDIA Corporation under the NVIDIA Open Model License.\n\nWeights are byte-identical to the native checkpoint. Only the Cosmos\nmodel-name binding in two configuration files is relocated. Native\noptimizer, scheduler, RNG, trainer arguments, private paths, datasets,\nWBC/vendor assets and credentials are not included. Full native training\nstate remains on its original host, not independently resumed here.\n\nThis artifact is not a qualified deployment or proof of task success.\nConsult fixed GitHub E1 scientific results; selected/deployment NONE.\n')
        paths['README.md']=out/'README.md'
        for name,p in paths.items():
            if name not in names:reject_private(p.read_text())
        inventory={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in paths.items()}
        manifest=dict(schema='E1_SANITIZED_PUBLIC_INFERENCE_V1',label=label,global_step=step,
            original_checkpoint_manifest_sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json'),
            original_inference_sha256={n:files[n] for n in names},metadata_changes=changes,files=inventory,
            upstream_revisions=dict(GR00T=BASE_REVISION,Cosmos=COSMOS_REVISION),
            delivery_source_sha=runtime['code']['vla_commit'],public_authority_sha256=sha(root/'evidence/PUBLIC_INFERENCE_ARTIFACT_AUTHORITY.json'),
            origin_job_id=origin['job_id'],origin_source_sha=request['source_sha'],origin_allocation_sha256=sha(allocation/'ALLOCATION_ADMISSION.json'),
            numerical_weights_unchanged=True,model_loaded=False,pickle_executed=False,
            full_native_training_state_public=False,independent_public_inference_or_resume_qualification=False,
            original_checkpoint_untouched=True,selected='NONE',deployment='NONE')
        reject_private(manifest);write(out/'INFERENCE_MANIFEST.json',manifest);paths['INFERENCE_MANIFEST.json']=out/'INFERENCE_MANIFEST.json'
        inventory['INFERENCE_MANIFEST.json']=dict(bytes=paths['INFERENCE_MANIFEST.json'].stat().st_size,sha256=sha(paths['INFERENCE_MANIFEST.json']))
        prefix='PI0294/'+root.name+'/public-inference-'+label+'-'+sha(out/'INFERENCE_MANIFEST.json')[:12]
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(PUBLIC_REPO,repo_type='dataset')):raise FileExistsError('NO_PUBLIC_PREFIX_OVERWRITE_OR_BLIND_RETRY')
        journal=dict(status='PUBLIC_UPLOAD_INTENT_UNKNOWN_NO_AUTOMATIC_RETRY',repo_id=PUBLIC_REPO,prefix=prefix,files=inventory,label=label)
        write(out/'DELIVERY.json',journal)
        rev=api.create_commit(PUBLIC_REPO,repo_type='dataset',parent_commit=info.sha,commit_message='E1 '+label+' sanitized inference, not deployment',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in paths.items()],num_threads=2).oid
        journal.update(status='PUBLIC_PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=rev);write(out/'DELIVERY.json',journal)
        def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=out/'small-readback')
        verify_remote(api,PUBLIC_REPO,prefix,rev,inventory,download,token)
        for n in names:
            if sha(checkpoint/n)!=files[n]:raise ValueError('ORIGINAL_CHANGED_DURING_PUBLIC_DELIVERY')
        ready=dict(status='PUBLIC_INFERENCE_FIXED_SHA_VERIFIED',manifest_sha256=sha(out/'INFERENCE_MANIFEST.json'),
            data_revision=rev,files=inventory,full_native_training_state_public=False,READY_uploaded_last=True,selected='NONE',deployment='NONE')
        write(out/'READY.json',ready)
        final=api.create_commit(PUBLIC_REPO,repo_type='dataset',parent_commit=rev,commit_message='E1 inference READY after fixed SHA verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(out/'READY.json'))]).oid
        inventory['READY.json']=dict(bytes=(out/'READY.json').stat().st_size,sha256=sha(out/'READY.json'))
        verify_remote(api,PUBLIC_REPO,prefix,final,inventory,download,token)
        journal.update(status='PASS_PUBLIC_INFERENCE_FIXED_REVISION_READY_LAST',revision=final);return journal
    except BaseException as exc:
        journal.update(status='FAILED_OR_UNKNOWN_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(exc).__name__)
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        if created:write(out/'DELIVERY.json',dict(journal,CPU_s=time.process_time()-started))
        ledger.cpu(time.process_time()-started);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',required=True);p.add_argument('--existing-token-file',required=True)
    a=p.parse_args();v=publish(a.root,a.label,a.existing_token_file);print(json.dumps(dict(status=v['status'],revision=v['revision'],label=a.label)))
