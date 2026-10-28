"""Consume fixed bank v2 once, reusing the fully verified original 56 dependency."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

from .durable import OwnerLock,read,write
from .data_consumer import extract_checked,sha,owned_bytes


def consume(args):
    os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,hf_hub_download
    root=Path(args.root);operations=Path(args.operations)
    lock=OwnerLock(root/'HOPPER_CPU_OWNER.lock')
    statefile=root/'evidence/BANK_V2_CONSUMER.json'
    if statefile.exists():raise FileExistsError('Inspect existing immutable consumer attempt')
    adoption=read(root/'ADOPTION.json');receipt=read(args.manifest)
    if receipt['status']!='BANK_V2_FIXED_HF_FULL_SET_AND_ACTUAL_NEW_EPISODE_READBACK_PASS':
        raise ValueError('PRODUCER_FIXED_READBACK_REQUIRED')
    original=read(root/'evidence/ORIGINAL_TRAIN_CONSUMER.json')
    if len(original['completed'])!=56 or original['status']!='ORIGINAL_56_COMPLETE_ALL_MEMBERS_VERIFIED_CPU_SAMPLE_PENDING':
        raise ValueError('ALL_ORIGINAL_56_ARCHIVES_REQUIRED')
    if read(root/'evidence/original_cpu/RESULT.json')['status']!='PASS_ACTUAL_CROSS_HOST_RGB_PROCESSOR_TENSORS':
        raise ValueError('ORIGINAL_ACTUAL_CONSUMPTION_REQUIRED')
    deadline=min(time.time()+3600,adoption['new_work_cutoff_epoch'])
    if deadline-time.time()<300:raise TimeoutError('INSUFFICIENT_BANK_TRANSFER_WINDOW')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');reserve=4*(deadline-time.time())
    if ledger['CPU_s']+sum(s.get('CPU_reserved_s',0) for s in ledger['stages'])+reserve>64*3600:
        raise PermissionError('HOPPER_CPU_BUDGET')
    budget=20*1024**3
    if owned_bytes(root)+budget>180*1024**3 or shutil.disk_usage(root).free<budget+20*1024**3:
        raise PermissionError('HOPPER_BANK_SPACE_PREFLIGHT')
    stage=dict(id='BANK_V2_CONSUMER',status='RUNNING',CPU_reserved_s=reserve,bytes_reserved=budget)
    ledger['stages'].append(stage);write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
    target=root/'data/bank_v2';target.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',started=time.time(),deadline=deadline,pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],revision=receipt['revision'],
        prefix=receipt['prefix'],completed=[],current=None,CPU_reserved_s=reserve,models_loaded=0)
    write(statefile,state)
    token=Path(args.existing_token_file).read_text().strip();api=HfApi(token=token)
    prefix=receipt['prefix']+'/'
    files={r['path'][len(prefix):]:r for r in receipt['files']}
    accounted=owned_bytes(target)

    def guard(extra=0):
        if time.time()>=deadline:raise TimeoutError('BANK_CONSUMER_CUTOFF')
        if accounted+extra>budget or shutil.disk_usage(root).free<extra+20*1024**3:
            raise PermissionError('BANK_CONSUMER_STORAGE_GUARD')

    def download(name):
        nonlocal accounted
        row=files[name];guard(row['bytes'])
        path=Path(hf_hub_download(receipt['repo_id'],row['path'],repo_type='dataset',revision=receipt['revision'],
            token=token,local_dir=target/'downloads'))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('FIXED_BANK_FILE_MISMATCH')
        accounted=owned_bytes(target)
        return path

    def expired(signum,frame):raise TimeoutError('BANK_CONSUMER_CUTOFF')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    child=None
    try:
        info=api.repo_info(receipt['repo_id'],repo_type='dataset',revision=receipt['revision'])
        if not info.private or info.sha!=receipt['revision']:raise ValueError('PRIVATE_FIXED_BANK_REVISION_REQUIRED')
        remote={r.path for r in api.list_repo_tree(receipt['repo_id'],repo_type='dataset',revision=receipt['revision'],
            path_in_repo=receipt['prefix'],recursive=True) if hasattr(r,'size')}
        if remote!={r['path'] for r in receipt['files']}:raise ValueError('EXACT_HF_BANK_FILE_SET_REQUIRED')
        ready=read(download('READY.json'));proofs=read(download('ARCHIVE_MEMBER_MANIFEST.json'))
        if not ready['READY_uploaded_last'] or ready['manifest_sha256']!=receipt['manifest_sha256']:
            raise ValueError('FIXED_BANK_READY_MISMATCH')
        dependency=read(download('ORIGINAL_DEPENDENCY.json'))
        if dependency['revision']!=original['revision'] or dependency['TRAIN_ids']!=[f'T291{i:03d}' for i in range(56)]:
            raise ValueError('ORIGINAL_FIXED_DEPENDENCY_CHANGED')
        expected=['C29200','C29201']+[f'D291{i:03d}' for i in range(16)]+[f'T291{i:03d}' for i in range(56,64)]
        if {name for name in files if name.startswith('episodes/')}!={'episodes/'+i+'.tar.gz' for i in expected}:
            raise ValueError('EXACT_NEW_26_RAW_ARCHIVES_REQUIRED')
        for identifier in expected:
            state['current']=identifier;write(statefile,state)
            archive=download('episodes/'+identifier+'.tar.gz')
            count=extract_checked(archive,target/'raw',proofs[identifier]['member_sha256'],guard)
            accounted=owned_bytes(target)
            state['completed'].append(dict(id=identifier,members=count,sha256=sha(archive)));write(statefile,state)
        for key,name,folder in (('native_dataset','native_dataset.tar.gz','native'),
            ('frozen_source','frozen_source.tar.gz','frozen_source'),('operations_source','operations_source.tar.gz','producer_operations')):
            extract_checked(download(name),target/folder,proofs[key]['member_sha256'],guard)
            accounted=owned_bytes(target)
        for name in sorted(files):
            if name.startswith('receipts/'):download(name)
        for name in files:
            p=target/'downloads'/prefix/name
            if not p.is_file() or sha(p)!=files[name]['sha256']:raise ValueError('FULL_DOWNLOADED_BANK_SET_REQUIRED')
        manifest=read(target/'native/DATASET_MANIFEST.json')
        if sha(target/'native/DATASET_MANIFEST.json')!=receipt['manifest_sha256'] or sha(target/'native/statistics.json')!=receipt['statistics_sha256']:
            raise ValueError('FROZEN_BANK_IDENTITY')
        bindings=read(root/'data/original_train/RAW_BINDINGS.json')
        bindings.update({i:str(target/'raw'/i) for i in expected})
        for row in manifest['episodes']:
            if sha(Path(bindings[row['id']])/'samples.json')!=row['samples_sha256'] or sha(target/'native'/row['file'])!=row['sha256']:
                raise ValueError('CROSS_AUTHORITY_BANK_SOURCE_CHANGED')
        write(target/'RAW_BINDINGS.json',bindings)
        evidence=root/'evidence/bank_v2_cpu';evidence.mkdir(exist_ok=False)
        env=dict(os.environ,PYTHONPATH=str(operations)+':'+str(target/'frozen_source/src'),PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        qualifications={}
        for split in ('TRAIN','DEV'):
            state['current']='CPU_'+split;write(statefile,state)
            reference=target/'downloads'/prefix/'receipts/evidence/adapter_bank_v2'/(split+'_reference.json')
            result=evidence/(split+'_QUALIFICATION.json')
            command=[args.python,'-m','scripts.pi0292.qualify_consumer','--dataset',str(target/'native'),
                '--source',args.source,'--cosmos',args.cosmos,'--raw-bindings',str(target/'RAW_BINDINGS.json'),
                '--reference',str(reference),'--split',split,'--output',str(result)]
            with (evidence/(split+'.stdout')).open('xb') as out,(evidence/(split+'.stderr')).open('xb') as err:
                child=subprocess.Popen(command,env=env,cwd=operations,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(child_pid=child.pid,child_start=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]);write(statefile,state)
                code=child.wait(timeout=min(300,max(1,deadline-time.time())))
            child=None
            if code or not result.exists():raise ValueError('ACTUAL_CROSS_HOST_CPU_FAILED:'+split)
            qualifications[split]=read(result)
            if qualifications[split]['status']!='PASS_ACTUAL_CROSS_HOST_RGB_PROCESSOR_TENSORS':raise ValueError('CPU_COMPARISON_FAILED')
        probe=read(target/'downloads'/prefix/'receipts/evidence/OPEN_LOOP_PROBE_SEAL.json')
        for name,digest in probe['evaluation_script_hashes'].items():
            if sha(operations/'scripts/pi0292'/name)!=digest:raise ValueError('SEALED_EVALUATION_CODE_CHANGED')
        write(root/'evidence/OPEN_LOOP_PROBE_SEAL.json',probe)
        state.update(status='PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE',current=None,child_pid=None,
            manifest_sha256=receipt['manifest_sha256'],statistics_sha256=receipt['statistics_sha256'],
            producer_transfer_sha256=sha(args.manifest),all_new_archive_members_verified=True,
            all_original_56_reused=True,original_archives_redownloaded=0,statistics_refitted=False,
            actual_TRAIN_CPU=qualifications['TRAIN']['status'],actual_DEV_CPU=qualifications['DEV']['status'],
            source_operations=read(operations/'OPERATIONS_SEAL.json')['source_sha'])
        write(root/'evidence/HOPPER_DATA_QUALIFICATION.json',state)
    except Exception as error:
        state.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
        used=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
        state.update(CPU_s=used,finished=time.time(),owned_bytes=owned_bytes(target));write(statefile,state)
        ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');ledger['CPU_s']+=used
        stage=next(s for s in ledger['stages'] if s['id']=='BANK_V2_CONSUMER')
        stage.update(status=state['status'],CPU_reserved_s=0,CPU_s=used,actual_bytes=state['owned_bytes'])
        write(root/'HOPPER_RESOURCE_LEDGER.json',ledger);signal.alarm(0);lock.close()
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','operations','manifest','existing-token-file','python','source','cosmos'):parser.add_argument('--'+name,required=True)
    result=consume(parser.parse_args());print(json.dumps({k:result[k] for k in ('status','CPU_s','owned_bytes')}))
