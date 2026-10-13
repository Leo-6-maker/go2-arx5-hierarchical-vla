"""Fetch only the fixed original TRAIN dependency; verify every archive member."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import resource
import shutil
import signal
import tarfile
import time

from .durable import OwnerLock,read,write


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def owned_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def extract_checked(path,target,expected,guard):
    with tarfile.open(path,'r:gz') as archive:
        guard(sum(member.size for member in archive.getmembers()))
    target.mkdir(parents=True,exist_ok=True)
    seen=set()
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in seen or member.name not in expected:
                raise ValueError('UNSAFE_OR_UNEXPECTED_ARCHIVE_MEMBER')
            guard(member.size)
            output=target/member.name
            if not output.resolve().is_relative_to(target.resolve()):raise ValueError('ARCHIVE_PATH_ESCAPE')
            output.parent.mkdir(parents=True,exist_ok=True)
            if not output.exists():
                with output.open('xb') as handle:shutil.copyfileobj(archive.extractfile(member),handle,1<<20)
            if sha(output)!=expected[member.name]:raise ValueError('EXTRACTED_MEMBER_HASH_MISMATCH')
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('INCOMPLETE_ARCHIVE')
    return len(seen)


def consume(args):
    from huggingface_hub import HfApi,hf_hub_download
    root=Path(args.root);adoption=read(root/'ADOPTION.json')
    statefile=root/'evidence/ORIGINAL_TRAIN_CONSUMER.json'
    lock=OwnerLock(root/'ORIGINAL_TRAIN_CONSUMER.lock')
    if statefile.exists():raise FileExistsError('Inspect existing attempt before explicit repair/resume')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    resource.setrlimit(resource.RLIMIT_CPU,(1800,1810))
    signal.alarm(max(1,min(3600,int(adoption['new_work_cutoff_epoch']-time.time()))))
    start=time.time();receipt=read(args.manifest)
    if receipt['revision']!='c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea':raise ValueError('ORIGINAL_REVISION_CHANGED')
    token=Path(args.existing_token_file).read_text().strip()
    api=HfApi(token=token)
    info=api.repo_info(receipt['repo_id'],repo_type='dataset',revision=receipt['revision'])
    if not info.private or info.sha!=receipt['revision']:raise ValueError('PRIVATE_FIXED_REVISION_REQUIRED')
    prefix=receipt['prefix']+'/'
    files={r['path'][len(prefix):]:r for r in receipt['files']}
    identifiers=[f'T291{i:03d}' for i in range(56)]
    wanted=['READY.json','ARCHIVE_MEMBER_MANIFEST.json','receipts/DATASET_ADMISSION.json','native_dataset.tar.gz']
    wanted += ['episodes/'+identifier+'.tar.gz' for identifier in identifiers]
    if any(name not in files for name in wanted):raise ValueError('ORIGINAL_DEPENDENCY_MISSING')
    target=root/'data/original_train';target.mkdir(parents=True,exist_ok=True)
    budget=16*1024**3
    if owned_bytes(target)+sum(files[n]['bytes'] for n in wanted)*4>budget:raise ValueError('DOWNLOAD_PEAK_PRECHECK')
    if shutil.disk_usage(root).free<budget+20*1024**3:raise ValueError('INSUFFICIENT_STAGE_MARGIN')
    state=dict(status='RUNNING',pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        started=start,revision=receipt['revision'],repo=receipt['repo_id'],completed=[],current=None,
        CPU_reserved_upper_s=1800,bytes_reserved_upper=budget,model_loads=0,physics_steps=0)
    write(statefile,state)
    accounted=owned_bytes(target)

    def guard(extra=0):
        nonlocal accounted
        if time.time()>=adoption['new_work_cutoff_epoch']:raise TimeoutError('NEW_WORK_CUTOFF')
        if accounted+extra>budget or shutil.disk_usage(root).free<extra+20*1024**3:raise ValueError('STORAGE_GUARD')

    def download(name):
        nonlocal accounted
        row=files[name];guard(row['bytes'])
        path=Path(hf_hub_download(receipt['repo_id'],row['path'],repo_type='dataset',revision=receipt['revision'],
            token=token,local_dir=target/'downloads'))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('DOWNLOADED_HASH_MISMATCH')
        accounted=owned_bytes(target)
        return path

    try:
        ready=read(download('READY.json'));proofs=read(download('ARCHIVE_MEMBER_MANIFEST.json'))
        if not ready['private'] or not ready['NATIVE_DATALOADER_PASS']:raise ValueError('ORIGINAL_READY_INVALID')
        admission=read(download('receipts/DATASET_ADMISSION.json'))
        admitted={r['id'] for r in admission['episodes'] if r['admitted'] and r['split']=='TRAIN'}
        if admitted!=set(identifiers):raise ValueError('ORIGINAL_56_SET_CHANGED')
        native=download('native_dataset.tar.gz')
        extract_checked(native,target/'native',proofs['native_dataset']['member_sha256'],guard)
        accounted=owned_bytes(target)
        for identifier in identifiers:
            state.update(current=identifier,updated=time.time());write(statefile,state)
            path=download('episodes/'+identifier+'.tar.gz')
            count=extract_checked(path,target/'raw',proofs[identifier]['member_sha256'],guard)
            accounted=owned_bytes(target)
            state['completed'].append(dict(id=identifier,members=count,archive_sha256=sha(path)))
            state.update(owned_bytes=accounted);write(statefile,state)
        bindings={identifier:str(target/'raw'/identifier) for identifier in identifiers}
        write(target/'RAW_BINDINGS.json',bindings)
        manifest=read(target/'native/DATASET_MANIFEST.json')
        if {r['id'] for r in manifest['episodes']}!=set(identifiers):raise ValueError('NATIVE_SET_CHANGED')
        for row in manifest['episodes']:
            if sha(target/'native'/row['file'])!=row['sha256']:raise ValueError('NATIVE_NUMERIC_HASH_CHANGED')
        state.update(status='ORIGINAL_56_COMPLETE_ALL_MEMBERS_VERIFIED_CPU_SAMPLE_PENDING',current=None,
            original_PI0291_status='PARTIAL',recollected=False,archives_preserved=True)
    except Exception as error:
        state.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        state.update(CPU_s=usage.ru_utime+usage.ru_stime,finished=time.time(),owned_bytes=owned_bytes(target))
        write(statefile,state);lock.close()
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','manifest','existing-token-file'):p.add_argument('--'+name,required=True)
    result=consume(p.parse_args());print(json.dumps(dict(status=result['status'],completed=len(result['completed']))))
