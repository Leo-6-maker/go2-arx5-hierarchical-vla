"""Private fixed-prefix bank v2 archive, READY-last and actual sample readback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import tarfile
import time

from .durable import OwnerLock,read,write
from .resource_guard import owned_bytes
from .train_adapter import NativeDataset,ProcessedCollator,sha
from .data_consumer import extract_checked

EPISODE_METADATA={'CARD.json','RESULT.json','SCORE.json','RENDER_PROCESS.json','samples.json','references.json',
    'physics.npz','highrate.npz','scene.xml.json','TEACHER_EVENTS.json','ERROR.json','ACTUAL_DUAL_RGB.mp4','VIDEO_RECEIPT.json'}
NON_PY_CODE={'scripts/pi0291_arx5_data/README.md'}|{'src/go2arx5_vla/target_mujoco/'+name for name in
    ('audit_home.cpp','native_home.hpp','audit_model.cpp','native_startup.hpp','observer.cpp','audit_start_boundary.cpp','compile_model.cpp')}


def allowed_code(name):
    return name in NON_PY_CODE or (name.endswith('.py') and name.startswith(('scripts/','src/')))


def contained(base,name):
    base=Path(base);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('UNSAFE_ARCHIVE_PATH')
    path=base/relative
    if not path.is_file() or not path.resolve().is_relative_to(base.resolve()):raise ValueError('UNBOUND_ARCHIVE_FILE')
    cursor=path
    while cursor!=base:
        if cursor.is_symlink():raise ValueError('ARCHIVE_SYMLINK_FORBIDDEN')
        cursor=cursor.parent
    return path


def verified_tar(path,members,reuse=False):
    if path.exists() and not reuse:raise FileExistsError('Preserve first archive, including partial failures')
    expected={name:sha(source) for name,source in members.items()}
    if not path.exists():
        with tarfile.open(path,'w:gz',compresslevel=3) as archive:
            for name,source in sorted(members.items()):archive.add(source,arcname=name,recursive=False)
    actual={}
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in actual or member.name not in expected:raise ValueError('ARCHIVE_MEMBER_SET')
            h=hashlib.sha256();handle=archive.extractfile(member)
            for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
            actual[member.name]=h.hexdigest()
    if actual!=expected:raise ValueError('ARCHIVE_MEMBER_HASH')
    return dict(files=len(actual),member_sha256=actual,sha256=sha(path),bytes=path.stat().st_size)


def transfer(args):
    os.environ['HF_HUB_OFFLINE']='0'
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url,hf_hub_download
    import requests
    root=Path(args.root);operations=Path(args.operations);native=root/'bank_v2_native'
    owner=OwnerLock(root/'PHYSICS_OWNER.lock')
    repair=args.repair_preserved_packaging
    journal_path=root/'evidence'/('BANK_TRANSFER_PROGRESS_repair1.json' if repair else 'BANK_TRANSFER_PROGRESS.json')
    if journal_path.exists():raise FileExistsError('Inspect existing HF commit journal before repair')
    if repair:
        failure=read(root/'evidence/BANK_TRANSFER_PROGRESS.json')
        if failure['status']!='FAILED_PRESERVED' or failure['error']!='NONCODE_CAPSULE_FILE' or failure['READY_claim'] or 'data_revision' in failure:
            raise PermissionError('ONLY_PROVEN_PRECOMMIT_CODE_WHITELIST_REPAIR_ALLOWED')
    adoption=read(root/'ADOPTION.json');deadline=min(time.time()+3600,adoption['new_work_cutoff_epoch'])
    if deadline-time.time()<300:raise TimeoutError('INSUFFICIENT_TRANSFER_WINDOW')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    def expired(signum,frame):raise TimeoutError('BANK_TRANSFER_CUTOFF')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    ledger=read(root/'RESOURCE_LEDGER.json');reserve=4*(deadline-time.time())
    expected_ids={'C29200','C29201'}|{f'D291{i:03d}' for i in range(16)}|{f'T291{i:03d}' for i in range(56,64)}
    if {r['id'] for r in ledger['starts']}!=expected_ids or len(ledger['starts'])!=26:
        raise ValueError('EXACT_DATA_STAGE_ATTEMPTS_REQUIRED')
    if any(r['status']=='STARTED' for r in ledger['starts']):raise ValueError('LIVE_PHYSICS_NOT_ARCHIVABLE')
    if ledger['CPU_s']+ledger['CPU_upper_s']+reserve>=96*3600:raise ValueError('CPU_TRANSFER_RESERVATION')
    ledger['CPU_upper_s']+=reserve;write(root/'RESOURCE_LEDGER.json',ledger)
    started=time.time();cpu0=resource.getrusage(resource.RUSAGE_SELF)
    output=root/'delivery/bank_v2';output.mkdir(parents=True,exist_ok=repair)
    if repair and {p.name for p in output.iterdir()}!={i+'.tar.gz' for i in expected_ids}|{'native_dataset.tar.gz'}:
        raise PermissionError('ONLY_PREVIOUS_VERIFIED_RAW_AND_NATIVE_ARCHIVES_MAY_BE_REUSED')
    journal=dict(status='PACKAGING',started=started,CPU_reserved_s=reserve,READY_claim=False,
        preserved_packaging_repair=repair,prior_failure_sha256=sha(root/'evidence/BANK_TRANSFER_PROGRESS.json') if repair else None)
    write(journal_path,journal)

    def guard(extra=0):
        if time.time()>=deadline:raise TimeoutError('BANK_TRANSFER_CUTOFF')
        if owned_bytes(root)+extra>=40*1024**3 or shutil.disk_usage(root).free<extra+10*1024**3:
            raise ValueError('BANK_TRANSFER_STORAGE_GUARD')

    try:
        gate=read(root/'evidence/TRAINER_ADAPTER_CPU_QUALIFICATION.json')
        if gate['status']!='PASS_BANK_V2_TRAIN_DEV_AND_TRAINER_LOADER_CPU':raise ValueError('BANK_CPU_GATE_REQUIRED')
        manifest=read(native/'DATASET_MANIFEST.json');admission=read(root/'BANK_V2_ADMISSION.json')
        if manifest['statistics_sha256']!=gate['statistics_sha256'] or sha(native/'DATASET_MANIFEST.json')!=gate['manifest_sha256']:
            raise ValueError('QUALIFIED_BANK_CHANGED')
        if admission['status']!='ARX5_BANK_V2_QUALIFIED':raise ValueError('RAW_ADMISSION_GATE')
        projected=sum(p.stat().st_size for p in (root/'episodes').rglob('*') if p.is_file())*2+512*1024**2
        guard(projected)
        token=Path(args.existing_token_file).read_text().strip();api=HfApi(token=token)
        info=api.repo_info(args.repo,repo_type='dataset')
        if not info.private:raise PermissionError('EXISTING_PRIVATE_REPO_REQUIRED')
        source=read(operations/'OPERATIONS_SEAL.json')['source_sha']
        prefix='PI0292/'+root.name+'/bank-v2-'+source[:8]
        if any(name.startswith(prefix+'/') for name in api.list_repo_files(args.repo,repo_type='dataset')):
            raise FileExistsError('Never overwrite a prior HF prefix')
        payload={};proofs={}
        def pack(key,name,members):
            guard(sum(p.stat().st_size for p in members.values())+1024**2)
            archive=output/Path(name).name
            proofs[key]=verified_tar(archive,members,reuse=repair);payload[name]=archive
        for row in ledger['starts']:
            folder=root/'episodes'/row['id'];entries=read(folder/'FILE_MANIFEST.json')
            names={r['path'] for r in entries}
            if any(name not in EPISODE_METADATA and not re.fullmatch(r'(?:body_rgb|wrist_rgb)/[0-9]{4}\.png',name) for name in names):
                raise ValueError('UNEXPECTED_EPISODE_FILE')
            actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
            if actual!=names|{'FILE_MANIFEST.json'}:raise ValueError('UNMANIFESTED_EPISODE_FILE')
            members={row['id']+'/'+name:contained(folder,name) for name in names|{'FILE_MANIFEST.json'}}
            for entry in entries:
                if sha(folder/entry['path'])!=entry['sha256']:raise ValueError('RAW_FILE_CHANGED')
            pack(row['id'],'episodes/'+row['id']+'.tar.gz',members)
        native_files={p.relative_to(native).as_posix():contained(native,p.relative_to(native).as_posix()) for p in native.rglob('*') if p.is_file()}
        allowed={'DATASET_MANIFEST.json','STATISTICS_PROVENANCE.json','statistics.json','modality.json','SPLIT.json','task_map.json','EXPORT_RECEIPT.json'}
        allowed|={r['file'] for r in manifest['episodes']}
        if set(native_files)!=allowed:raise ValueError('UNEXPECTED_NATIVE_PAYLOAD')
        for row in manifest['episodes']:
            if sha(native/row['file'])!=row['sha256']:raise ValueError('QUALIFIED_NATIVE_NUMERIC_CHANGED')
        pack('native_dataset','native_dataset.tar.gz',native_files)
        code_seal=read(root/'CODE_SEAL.json')
        code_members={name:contained(root/'code',name) for name in code_seal['files']}
        if any(not allowed_code(name) for name in code_members):raise ValueError('NONCODE_CAPSULE_FILE')
        if {k:sha(p) for k,p in code_members.items()}!=code_seal['files']:raise ValueError('PHYSICAL_CAPSULE_CHANGED')
        pack('frozen_source','frozen_source.tar.gz',code_members)
        seal=read(operations/'OPERATIONS_SEAL.json')
        members={name:contained(operations,name) for name in seal['files']}
        if {k:sha(p) for k,p in members.items()}!=seal['files']:raise ValueError('OPERATIONS_CAPSULE_CHANGED')
        members['scripts/__init__.py']=operations/'scripts/__init__.py'
        pack('operations_source','operations_source.tar.gz',members)
        receipt_names=['ADOPTION.json','CODE_SEAL.json','ATTEMPT_PLAN.json','ENV_CANARY_GATE.json','SCIENCE_SOURCE_EQUIVALENCE.json',
            'BANK_V2_ADMISSION.json','BANK_V2_MANIFEST.json','STATISTICS_PROVENANCE.json','RESOURCE_LEDGER.json',
            'SUPPLEMENT_ATTEMPT_LEDGER.json','DATASET_CARD.md',
            'evidence/DISPATCH_supplement.json','evidence/DATALOADER_SMOKE.json','evidence/TRAINER_ADAPTER_CPU_QUALIFICATION.json',
            'evidence/OPEN_LOOP_PROBE_SEAL.json','evidence/BANK_PIPELINE.json','evidence/AUDIT_C29200.json']
        receipt_names += ['evidence/AUDIT_BOUND_'+r['id']+'.json' for r in ledger['starts']]
        if repair:receipt_names.append('evidence/BANK_TRANSFER_PROGRESS.json')
        receipt_names += ['evidence/adapter_bank_v2/'+split+'_'+kind+'.json' for split in ('TRAIN','DEV') for kind in ('reference','adapter','COMPARISON')]
        for name in receipt_names:
            path=contained(root,name);destination=output/('receipt_'+name.replace('/','_'))
            shutil.copyfile(path,destination);payload['receipts/'+name]=destination
        dependency=read(args.original_manifest)
        if dependency['revision']!='c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea':raise ValueError('ORIGINAL_DEPENDENCY_REVISION')
        original=dict(repo_id=dependency['repo_id'],revision=dependency['revision'],prefix=dependency['prefix'],
            TRAIN_ids=[f'T291{i:03d}' for i in range(56)],files=[r for r in dependency['files'] if '/episodes/T291' in r['path'] or r['path'].endswith(('READY.json','ARCHIVE_MEMBER_MANIFEST.json'))],
            recollected=False,old_status='PARTIAL')
        write(output/'ORIGINAL_DEPENDENCY.json',original);payload['ORIGINAL_DEPENDENCY.json']=output/'ORIGINAL_DEPENDENCY.json'
        write(output/'ARCHIVE_MEMBER_MANIFEST.json',proofs);payload['ARCHIVE_MEMBER_MANIFEST.json']=output/'ARCHIVE_MEMBER_MANIFEST.json'
        files=[dict(path=prefix+'/'+name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(payload.items())]
        data_revision=api.create_commit(args.repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='PI0292 bank v2 actual data and CPU evidence before READY',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(p)) for name,p in payload.items()]).oid
        journal.update(status='DATA_COMMIT_CREATED',repo_id=args.repo,private=True,prefix=prefix,data_revision=data_revision,files=files,
            source_sha=source,physical_source_sha=code_seal['source_sha'],original_dependency=original,
            manifest_sha256=sha(native/'DATASET_MANIFEST.json'),statistics_sha256=sha(native/'statistics.json'))
        write(journal_path,journal)
        def verify(revision,expected):
            remote={r.path:r for r in api.list_repo_tree(args.repo,repo_type='dataset',revision=revision,path_in_repo=prefix,recursive=True) if hasattr(r,'size')}
            if set(remote)!={r['path'] for r in expected}:raise ValueError('HF_EXACT_FILE_SET')
            for row in expected:
                actual=remote[row['path']]
                if actual.size!=row['bytes']:raise ValueError('HF_SIZE')
                if actual.lfs:
                    digest=actual.lfs.sha256 if hasattr(actual.lfs,'sha256') else actual.lfs['sha256']
                else:
                    response=requests.get(hf_hub_url(args.repo,row['path'],repo_type='dataset',revision=revision),headers={'Authorization':'Bearer '+token},timeout=60)
                    response.raise_for_status();digest=hashlib.sha256(response.content).hexdigest()
                if digest!=row['sha256']:raise ValueError('HF_HASH')
        verify(data_revision,files);guard()
        ready=dict(status='READY_FOR_FIXED_REVISION_READBACK',repo_id=args.repo,repo_type='dataset',private=True,prefix=prefix,
            data_revision=data_revision,files=files,original_dependency=original,admitted_counts=admission['counts'],
            manifest_sha256=sha(native/'DATASET_MANIFEST.json'),statistics_sha256=sha(native/'statistics.json'),
            original_PI0291_status='PARTIAL',raw_attempts=len(ledger['starts']),source_sha=source,
            physical_source_sha=code_seal['source_sha'],READY_uploaded_last=True,complete_readback='PENDING',
            model_loads=0,selected='NONE',deployment='NONE')
        write(output/'READY.json',ready)
        revision=api.create_commit(args.repo,repo_type='dataset',parent_commit=data_revision,
            commit_message='PI0292 bank v2 READY after complete remote file verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
        files.append(dict(path=prefix+'/READY.json',bytes=(output/'READY.json').stat().st_size,sha256=sha(output/'READY.json')))
        verify(revision,files)
        journal.update(status='READY_LAST_FULL_SET_SHA_VERIFIED',revision=revision,files=files,READY_claim=True);write(journal_path,journal)
        download=root/'delivery/bank_v2_readback';download.mkdir(exist_ok=False)
        byname={r['path']:r for r in files}
        def get(name):
            path=Path(hf_hub_download(args.repo,prefix+'/'+name,repo_type='dataset',revision=revision,token=token,local_dir=download/'blobs'))
            row=byname[prefix+'/'+name]
            if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('ACTUAL_DOWNLOAD_HASH')
            return path
        identifier=next(r['id'] for r in admission['episodes'] if r['origin_authority']=='PI0292' and r['admitted'])
        get('READY.json')
        extract_checked(get('episodes/'+identifier+'.tar.gz'),download/'raw',proofs[identifier]['member_sha256'],guard)
        extract_checked(get('native_dataset.tar.gz'),download/'native',proofs['native_dataset']['member_sha256'],guard)
        split=next(r['split'] for r in manifest['episodes'] if r['id']==identifier)
        dataset=NativeDataset(download/'native',args.source,args.cosmos,split,raw_bindings={identifier:str(download/'raw'/identifier)})
        base=next(i for i,(row,anchor) in enumerate(dataset.rows) if row['id']==identifier and anchor==0)
        from .qualify_adapter import inventory
        batches=[]
        for anchor in (0,80,160,320,480,600):
            processed=dataset[base+anchor];batches.append(inventory(ProcessedCollator(dataset.processor)([processed])))
        journal.update(status='BANK_V2_FIXED_HF_FULL_SET_AND_ACTUAL_NEW_EPISODE_READBACK_PASS',
            readback_episode=identifier,readback_split=split,actual_RGB_processor_batches=batches,
            model_loads=0,physics_steps=0,original_PI0291_status='PARTIAL',selected='NONE',deployment='NONE')
    except Exception as error:
        journal.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error));raise
    finally:
        own=resource.getrusage(resource.RUSAGE_SELF);used=own.ru_utime+own.ru_stime-cpu0.ru_utime-cpu0.ru_stime
        journal.update(CPU_s=used,finished=time.time());write(journal_path,journal)
        if journal['status']=='BANK_V2_FIXED_HF_FULL_SET_AND_ACTUAL_NEW_EPISODE_READBACK_PASS':
            write(root/'evidence/BANK_TRANSFER_MANIFEST.json',journal)
        ledger=read(root/'RESOURCE_LEDGER.json');ledger['CPU_s']+=used;ledger['CPU_upper_s']-=reserve
        ledger['events'].append(dict(stage='BANK_PRIVATE_HF',CPU_s=used,status=journal['status'],physics_steps=0))
        ledger['owned_bytes']=owned_bytes(root);write(root/'RESOURCE_LEDGER.json',ledger);owner.close()
        signal.alarm(0)
    return journal


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','operations','repo','existing-token-file','original-manifest','source','cosmos'):p.add_argument('--'+name,required=True)
    p.add_argument('--repair-preserved-packaging',action='store_true')
    value=transfer(p.parse_args());print(json.dumps({k:value[k] for k in ('status','revision','CPU_s')}))
