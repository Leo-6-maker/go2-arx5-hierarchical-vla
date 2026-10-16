"""Private fixed-revision transport of existing probe arrays only, no inference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from scripts.pi0292.durable import read,write
from scripts.pi0292.data_consumer import extract_checked
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger

LABELS=['M0','M1-256','M1-1024','M1-256-CANONICAL','M1-1024-CANONICAL','M2-BC','M2-CORR']


def publish(root,repo,token_file):
    root=Path(root);validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard(closeout=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('E1_PROBE_TRANSFER_CPU_ONLY')
    l=read(ledger.path)
    if l['GPU_reserved_s'] or any(x['status']!='NATIVE_TERMINAL' for x in l['PBS_submissions']):raise PermissionError('E1_TERMINAL_GPU_BEFORE_PROBE_EXPORT')
    started=time.process_time();output=root/'datasets/probe-output-delivery';output.mkdir(mode=0o700)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    try:
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(repo,repo_type='dataset')
        if not info.private:raise PermissionError('E1_PRIVATE_PROBE_OUTPUTS_ONLY')
        prefix='PI0294/'+root.name+'/E1_FIXED_PROBE_OUTPUTS_V1'
        if any(p.startswith(prefix+'/') for p in api.list_repo_files(repo,repo_type='dataset')):raise FileExistsError('E1_IMMUTABLE_PROBE_PREFIX')
        members={};available=[];missing=[]
        for label in LABELS:
            folder=root/'evidence'/label/'probes'
            if not (folder/'ON_EXPERT_RESULTS.json').exists() or not (folder/'ON_STUDENT_RESULTS.private.json').exists():missing.append(label);continue
            result=read(folder/'ON_EXPERT_RESULTS.json');student=read(folder/'ON_STUDENT_RESULTS.private.json')
            if not result['complete'] or len(result['records'])!=96 or len(student['records'])!=12:raise ValueError('E1_COMPLETE_FIXED_PROBE_OUTPUTS_REQUIRED')
            available.append(label)
            for p in sorted(folder.iterdir()):
                if p.is_symlink() or (p.suffix not in ['.json','.npz']):raise ValueError('E1_NO_UNEXPECTED_PROBE_MEMBER')
                members[str(p.relative_to(root))]=sha(p)
        if not available:raise ValueError('E1_NO_ACTUAL_PROBES')
        for name in ['plans/PROBE_MANIFEST.json','plans/STUDENT_PROBES.private.json','plans/SMALL8_BINDING.json']:
            members[name]=sha(root/name)
        total=sum((root/name).stat().st_size for name in members)
        if total>64*1024**2:raise PermissionError('E1_PROBE_EXPORT_64_MIB_CAP')
        archive=output/'fixed-probes.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for name in sorted(members):ledger.guard(closeout=True);tar.add(root/name,arcname=name,recursive=False)
        write(output/'MEMBERS.json',members)
        write(output/'INDEX.json',dict(run_uuid=root.name,available=available,not_available=missing,new_model_queries=0,old_results_preserved=True))
        files={n:dict(bytes=(output/n).stat().st_size,sha256=sha(output/n)) for n in ['fixed-probes.tar.gz','MEMBERS.json','INDEX.json']}
        write(output/'TRANSFER.private.json',dict(status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY',repo_id=repo,prefix=prefix,files=files))
        rev=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,commit_message='E1 immutable existing fixed probe arrays',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(output/n)) for n in files]).oid
        for n,row in files.items():
            path=hf_hub_download(repo,prefix+'/'+n,repo_type='dataset',revision=rev,token=token,local_dir=output/'readback')
            if sha(path)!=row['sha256']:raise ValueError('E1_PROBE_PAYLOAD_READBACK')
        ready=dict(status='READY_EXISTING_FIXED_PROBE_ARRAYS',run_uuid=root.name,files=files,data_revision=rev,available=available,not_available=missing,new_model_queries=0)
        write(output/'READY.json',ready)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=rev,commit_message='E1 fixed probes READY after byte verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        path=hf_hub_download(repo,prefix+'/READY.json',repo_type='dataset',revision=final,token=token,local_dir=output/'final-readback')
        if sha(path)!=sha(output/'READY.json'):raise ValueError('E1_PROBE_READY_READBACK')
        value=dict(status='PASS_PRIVATE_FIXED_PROBE_TRANSFER',repo_id=repo,prefix=prefix,revision=final,data_revision=rev,
            files=files,READY_sha256=sha(output/'READY.json'),available=available,not_available=missing)
        write(output/'TRANSFER.private.json',value);return value
    finally:ledger.cpu(time.process_time()-started)


def consume(root,transfer,token_file):
    root=Path(root);validate(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    started=time.process_time();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,hf_hub_download
    try:
        t=read(transfer)
        if t['status']!='PASS_PRIVATE_FIXED_PROBE_TRANSFER':raise PermissionError('E1_FIXED_PROBE_TRANSFER_REQUIRED')
        token=Path(token_file).read_text().strip();api=HfApi(token=token);info=api.repo_info(t['repo_id'],repo_type='dataset',revision=t['revision'])
        if not info.private or info.sha!=t['revision']:raise PermissionError('E1_PRIVATE_EXACT_PROBE_REVISION')
        out=root/'datasets/received-fixed-probes';out.mkdir(mode=0o700)
        def get(name,digest):
            p=Path(hf_hub_download(t['repo_id'],t['prefix']+'/'+name,repo_type='dataset',revision=t['revision'],token=token,local_dir=out/'download'))
            if sha(p)!=digest:raise ValueError('E1_PROBE_RECEIVED_HASH')
            return p
        ready=read(get('READY.json',t['READY_sha256']))
        if ready['run_uuid']!=root.name or ready['files']!=t['files'] or ready['data_revision']!=t['data_revision']:raise ValueError('E1_PROBE_READY_IDENTITY')
        members=read(get('MEMBERS.json',t['files']['MEMBERS.json']['sha256']));index=read(get('INDEX.json',t['files']['INDEX.json']['sha256']))
        archive=get('fixed-probes.tar.gz',t['files']['fixed-probes.tar.gz']['sha256'])
        def guard(extra):
            ledger.guard(closeout=True)
            if extra>64*1024**2 or shutil.disk_usage(root).free<extra+2*1024**3:raise PermissionError('E1_PROBE_TRANSFER_SPACE')
        extract_checked(archive,out/'native',members,guard)
        value=dict(status='PASS_ACTUAL_FIXED_PROBE_ARRAY_CONSUMPTION',revision=t['revision'],files=len(members),index=index,new_model_queries=0)
        write(out/'CONSUMPTION.private.json',value);return value
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['publish','consume']);p.add_argument('--root',required=True);p.add_argument('--repo');p.add_argument('--transfer');p.add_argument('--existing-token-file',required=True);a=p.parse_args()
    v=publish(a.root,a.repo,a.existing_token_file) if a.operation=='publish' else consume(a.root,a.transfer,a.existing_token_file)
    print(json.dumps({k:x for k,x in v.items() if k not in ['repo_id']}),flush=True)
