"""Export only a qualified PV1 bank; old data, models and private paths stay put."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import tarfile
import threading
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.hf_bank import contained,verified_tar
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1.scope import STATS
from .dataset import admit_new_support
from .decision import AUTHORITY
from .scope import EXPORT_ROLE,validate

READY='READY.json'
INDEX='ARCHIVE_MEMBER_MANIFEST.json'
SOURCE_READY='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED'


def data_network():
    previous=os.environ.get('HF_HUB_OFFLINE')
    from huggingface_hub import constants
    original=constants.HF_HUB_OFFLINE
    os.environ['HF_HUB_OFFLINE']='0';constants.HF_HUB_OFFLINE=False
    def restore():
        constants.HF_HUB_OFFLINE=original
        if previous is None:os.environ.pop('HF_HUB_OFFLINE',None)
        else:os.environ['HF_HUB_OFFLINE']=previous
    return restore


def download_dataset(*args,**kwargs):
    restore=data_network()
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(*args,repo_type='dataset',**kwargs)
    finally:restore()


def existing_token(bindings):
    name=bindings.get('hf_token_file')
    if name is None:return None
    p=Path(name)
    if not p.is_absolute() or not p.is_file() or p.stat().st_uid!=os.getuid():
        raise PermissionError('PV1_ONLY_EXISTING_OWN_PRIVATE_HF_AUTH_LOCATOR')
    return p.read_text().strip()


def safe_json(value):
    if isinstance(value,dict):
        for key,item in value.items():safe_json(key);safe_json(item)
    elif isinstance(value,list):
        for item in value:safe_json(item)
    elif isinstance(value,str) and (value.startswith(('/home/','/scratch/','/lustre/','/mnt/','/opt/'))
            or re.search(r'\bhf_[A-Za-z0-9]{20,}\b',value)):
        raise PermissionError('PV1_PUBLIC_EXPORT_FORBIDS_HOST_PATHS_AND_CREDENTIALS')


def bank_members(bank,guard):
    bank=Path(bank)
    if bank.is_symlink() or not bank.is_dir():raise PermissionError('PV1_REAL_OWN_BANK_DIRECTORY_ONLY')
    manifest=read(bank/'DATASET_MANIFEST.json');ready=read(bank/'READY.json')
    admit_new_support(manifest,manifest['route']);safe_json(manifest);safe_json(ready)
    if (ready['status']!=SOURCE_READY or ready['dataset_manifest_sha256']!=sha(bank/'DATASET_MANIFEST.json')
            or ready['payload_manifest_sha256']!=sha(bank/'PAYLOAD_MANIFEST.json')
            or ready['statistics_sha256']!=STATS or sha(bank/'statistics.json')!=STATS
            or ready['later_rows_hot_injected'] is not False):
        raise PermissionError('PV1_DELIVERY_ONLY_ORIGINAL_ADMITTED_FROZEN_BANK')
    entries=read(bank/'PAYLOAD_MANIFEST.json');safe_json(entries)
    expected={'DATASET_MANIFEST.json','statistics.json'}
    for r in manifest['episodes']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']) or r['file']!='numeric/'+r['id']+'.npz':
            raise PermissionError('PV1_ONLY_EXACT_TRAIN_NUMERIC_PARENT_IDENTITIES')
        expected.add(r['file'])
        expected|={'raw/'+r['id']+'/'+camera+'/'+str(i).zfill(4)+'.png'
            for camera in ('body_rgb','wrist_rgb') for i in range(641)}
    if ({r['path'] for r in entries}!=expected or len(entries)!=len(expected)
            or ready['files']!=len(entries) or ready['logical_payload_bytes']!=sum(r['bytes'] for r in entries)):
        raise PermissionError('PV1_EXACT_NUMERIC_AND_641_ACTUAL_TWO_CAMERA_PARENTS_ONLY')
    members={}
    for r in entries:
        guard();p=contained(bank,r['path'])
        if p.stat().st_uid!=os.getuid() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('PV1_ADMITTED_BANK_MEMBER_CHANGED:'+r['path'])
        members[r['path']]=p
    if {p.relative_to(bank).as_posix() for p in bank.rglob('*') if p.is_file()}!=expected|{'PAYLOAD_MANIFEST.json','READY.json'}:
        raise PermissionError('PV1_NO_EXTRA_RAW_XML_VENDOR_WBC_OR_PRIVATE_FILES_IN_DELIVERY')
    for name in ('PAYLOAD_MANIFEST.json','READY.json'):members[name]=contained(bank,name)
    return manifest,ready,members


def pack_archive(path,members,guard):
    path=Path(path)
    if path.exists():raise FileExistsError('PV1_PRESERVE_FIRST_ARCHIVE_INCLUDING_PARTIAL_FAILURE')
    # Hardlinked source RGB must become regular archive files, never tar links.
    with tarfile.open(path,'w:gz',compresslevel=1,dereference=True) as archive:
        for name,p in sorted(members.items()):
            guard();archive.add(p,arcname=name,recursive=False)
    return verified_tar(path,members,reuse=True)


def package(root):
    root=Path(root);m,inputs=validate(root,role=EXPORT_ROLE)
    plan=read(root/'plans/SAVED_EXPORT_PLAN.private.json');ledger=read(root/'RESOURCE_LEDGER.json')
    if (plan['schema']!='PI0294_PV1_FROZEN_TRAIN_BANK_PACK_PLAN_V1' or ledger['status']!='RESERVED'
            or ledger['unknown_effects'] or any(m['limits'][k] for k in
                ('engine_starts','physics_ticks','GR00T_queries','training_updates'))
            or type(plan['maximum_delivery_bytes']) is not int
            or not 0<plan['maximum_delivery_bytes']<=min(m['limits']['output_bytes'],300*1024**3)
            or type(plan['minimum_free_bytes']) is not int or plan['minimum_free_bytes']<10*1024**3):
        raise PermissionError('PV1_REGISTERED_CPU_PACKER_ONLY_NO_SCIENTIFIC_EFFECTS')
    bank=Path(inputs['new_dataset']);source=bank.parent.parent;b=read(source/'OUTPUT_BINDING.private.json')
    if (bank.is_symlink() or not bank.resolve().is_relative_to(source.resolve()) or source.is_symlink()
            or source.parent.resolve()!=root.parent.resolve() or b['uid']!=os.getuid()
            or Path(b['run_root']).resolve()!=source.resolve() or read(b['lease'])['status']!='CLOSED'
            or read(source/'RESOURCE_LEDGER.json')['status']!='FROZEN_TRAIN_BANK_READY'
            or sha(bank/'DATASET_MANIFEST.json')!=plan['dataset_manifest_sha256']
            or sha(bank/'READY.json')!=plan['source_ready_sha256']):
        raise PermissionError('PV1_ONLY_SAME_OWNER_CLOSED_QUALIFIED_STREAM_BANK')
    lock=OwnerLock(root/'PV1_PACK_OWNER.lock');started=time.process_time()
    output=root/'delivery';output.mkdir(mode=0o700)
    os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));error=None
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-started>=m['limits']['CPU_s']-15
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):
            raise PermissionError('PV1_PACK_CPU_WALL_AND_ORIGINAL_STOP')
        if shutil.disk_usage(root).free<plan['minimum_free_bytes']:
            raise PermissionError('PV1_PACK_RESERVED_STORAGE_MARGIN')
    try:
        ledger['status']='RUNNING';write(root/'RESOURCE_LEDGER.json',ledger)
        manifest,ready,members=bank_members(bank,guard)
        if sum(p.stat().st_size for p in members.values())*2>plan['maximum_delivery_bytes']:
            raise PermissionError('PV1_PACK_KNOWN_NEW_STORAGE_UPPER_BOUND')
        index={};episodes=manifest['episodes']
        for start in range(0,len(episodes),8):
            ids={r['id'] for r in episodes[start:start+8]}
            selected={name:p for name,p in members.items() if any(name=='numeric/'+i+'.npz' or name.startswith('raw/'+i+'/') for i in ids)}
            name='parents-'+str(start//8).zfill(3)+'.tar.gz'
            index[name]=pack_archive(output/name,selected,guard)
            ledger.update(packed_parent_count=min(start+8,len(episodes)),worker_CPU_s=time.process_time()-started)
            write(root/'RESOURCE_LEDGER.json',ledger)
        metadata={name:p for name,p in members.items() if '/' not in name}
        index['metadata.tar.gz']=pack_archive(output/'metadata.tar.gz',metadata,guard)
        write(output/INDEX,dict(schema='PI0294_PV1_BANK_ARCHIVE_INDEX_V1',authority_release=AUTHORITY,
            dataset_manifest_sha256=sha(bank/'DATASET_MANIFEST.json'),source_ready_sha256=sha(bank/'READY.json'),
            statistics_sha256=STATS,archives=index,only_new_bank=True,original_and_C1_reuploaded=False))
        result=dict(status='PASS_NEW_TRAIN_BANK_PACKAGED_FULL_MEMBER_HASHES',archives=len(index),
            parent_episodes=len(episodes),archive_index_sha256=sha(output/INDEX),
            bytes=sum(p.stat().st_size for p in output.iterdir()),new_physics=0,new_queries=0,new_models=0)
        if result['bytes']>plan['maximum_delivery_bytes']:raise PermissionError('PV1_PACK_ACTUAL_STORAGE_BOUND')
        write(root/'evidence/PACK_RESULT.json',result)
        return result
    except BaseException as exc:error=type(exc).__name__+': '+str(exc);raise
    finally:
        ledger.update(status='FAILED_PRESERVED' if error else 'TERMINAL_PACKAGED',error=error,
            CPU_s=time.process_time()-started,CPU_reserved_s=0)
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()


def verify_remote(api,repo,prefix,revision,files,guard,readback_root):
    from huggingface_hub import hf_hub_download
    actual={r.path:r for r in api.list_repo_tree(repo,repo_type='dataset',revision=revision,path_in_repo=prefix,recursive=True)
        if hasattr(r,'size')}
    if set(actual)!={r['path'] for r in files}:raise ValueError('PV1_HF_FIXED_PREFIX_EXACT_FILE_SET')
    for r in files:
        guard();v=actual[r['path']]
        if v.size!=r['bytes']:raise ValueError('PV1_HF_FIXED_REVISION_SIZE')
        if v.lfs:
            h=v.lfs.sha256 if hasattr(v.lfs,'sha256') else v.lfs['sha256']
        else:
            p=hf_hub_download(repo,r['path'],repo_type='dataset',revision=revision,token=api.token,local_dir=readback_root)
            h=sha(p)
        if h!=r['sha256']:raise ValueError('PV1_HF_FIXED_REVISION_SHA256')


def upload(root,guard):
    """One public fixed-prefix transaction. Unknown commits require reconciliation."""
    root=Path(root);m,inputs=validate(root,role=EXPORT_ROLE)
    plan=read(root/'plans/SAVED_EXPORT_PLAN.private.json');output=root/'delivery'
    result=read(root/'evidence/PACK_RESULT.json');index=read(output/INDEX)
    if (result['status']!='PASS_NEW_TRAIN_BANK_PACKAGED_FULL_MEMBER_HASHES'
            or sha(output/INDEX)!=result['archive_index_sha256']):
        raise PermissionError('PV1_COMPLETE_PACK_PROOF_BEFORE_HF_MUTATION')
    repo='leoyuer/pi0294-public-artifacts';prefix=plan['HF_prefix']
    a=read(root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    if prefix!='PI0294/'+a['run_uuid']+'/PV1_NEW_TRAIN_BANK_V1':
        raise PermissionError('PV1_ONE_CAMPAIGN_QUALIFIED_NEW_DATA_PREFIX_ONLY')
    lock=OwnerLock(root/'PV1_HF_OWNER.lock');journal=root/'evidence/HF_TRANSFER.private.json'
    if journal.exists():lock.close();raise FileExistsError('PV1_RECONCILE_PRIOR_HF_INTENT_NO_BLIND_COMMIT_RETRY')
    state=dict(status='PRECOMMIT_READONLY',source_sha=m['code']['vla_commit'],repo_id=repo,prefix=prefix,
        READY_claim=False,unknown_mutation=False);write(journal,state)
    restore=None
    try:
        restore=data_network()
        from huggingface_hub import HfApi,CommitOperationAdd
        api=HfApi(token=existing_token(inputs));info=api.repo_info(repo,repo_type='dataset')
        if info.private:raise PermissionError('PV1_EXPLICITLY_AUTHORIZED_PUBLIC_SANITIZED_DATA_REPO_ONLY')
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(repo,repo_type='dataset')):
            raise FileExistsError('PV1_NEVER_OVERWRITE_EXISTING_HF_PREFIX')
        names=set(index['archives'])|{INDEX}
        if {p.name for p in output.iterdir()}!=names:raise PermissionError('PV1_HF_ONLY_PREVERIFIED_NEW_BANK_ARCHIVES')
        for name,p in index['archives'].items():
            if sha(output/name)!=p['sha256'] or (output/name).stat().st_size!=p['bytes']:
                raise ValueError('PV1_PACK_ARCHIVE_CHANGED_BEFORE_UPLOAD')
        files=[dict(path=prefix+'/'+n,bytes=(output/n).stat().st_size,sha256=sha(output/n)) for n in sorted(names)]
        safe_json(index);guard()
        state.update(status='DATA_COMMIT_INTENT',parent_revision=info.sha,files=files,unknown_mutation=True);write(journal,state)
        revision=api.create_commit(repo,repo_type='dataset',parent_commit=info.sha,
            commit_message='PV1 admitted new TRAIN data, immutable payload before READY',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(output/n)) for n in sorted(names)]).oid
        state.update(status='DATA_COMMIT_CREATED',data_revision=revision,unknown_mutation=False);write(journal,state)
        verify_remote(api,repo,prefix,revision,files,guard,root/'evidence/HF_readback')
        ready=dict(schema='PI0294_PV1_HF_NEW_BANK_READY_V1',status='READY_FOR_FIXED_REVISION_CONSUMPTION',
            authority_release=AUTHORITY,campaign_run_uuid=a['run_uuid'],repo_id=repo,repo_type='dataset',private=False,
            prefix=prefix,data_revision=revision,files=files,source_ready_sha256=index['source_ready_sha256'],
            dataset_manifest_sha256=index['dataset_manifest_sha256'],statistics_sha256=STATS,
            archive_index_sha256=sha(output/INDEX),READY_uploaded_last=True,
            original_and_C1_reuploaded=False,model_or_physics_run_by_delivery=False)
        safe_json(ready);write(output/READY,ready);guard()
        state.update(status='READY_COMMIT_INTENT',unknown_mutation=True);write(journal,state)
        final=api.create_commit(repo,repo_type='dataset',parent_commit=revision,
            commit_message='PV1 READY after exact fixed-revision full-file hash verification',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+READY,path_or_fileobj=str(output/READY))]).oid
        files=files+[dict(path=prefix+'/'+READY,bytes=(output/READY).stat().st_size,sha256=sha(output/READY))]
        state.update(status='READY_COMMIT_CREATED',revision=final,files=files,unknown_mutation=False);write(journal,state)
        verify_remote(api,repo,prefix,final,files,guard,root/'evidence/HF_readback')
        state.update(status='PASS_FIXED_REVISION_READY_LAST_EXACT_FULL_SET_SIZE_SHA256',READY_claim=True,
            data_bank_status=SOURCE_READY,dataset_manifest_sha256=index['dataset_manifest_sha256'])
        return state
    finally:
        if restore is not None:restore()
        write(journal,state);lock.close()


def run(root):
    root=Path(root);m,_=validate(root,role=EXPORT_ROLE)
    plan=read(root/'plans/SAVED_EXPORT_PLAN.private.json');began=time.process_time();failure=None
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-15
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()
                or shutil.disk_usage(root).free<plan['minimum_free_bytes']):
            raise PermissionError('PV1_TRANSFER_ORIGINAL_CPU_WALL_STORAGE_AND_STOP')
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise PermissionError('PV1_DELIVERY_REQUIRES_BOUNDED_OWN_MAIN_THREAD_TIMER')
    def expired(signum,frame):raise TimeoutError('PV1_FINITE_DELIVERY_WALL_NO_AUTOMATIC_RESTART')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,m['limits']['operation_deadline_epoch']-time.time()))
    try:
        package(root)
        result=upload(root,guard)
        write(root/'evidence/DELIVERY_RESULT.private.json',result)
        return {k:result[k] for k in ('status','revision','dataset_manifest_sha256')}
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);raise
    finally:
        ledger=read(root/'RESOURCE_LEDGER.json')
        journal=root/'evidence/HF_TRANSFER.private.json';transfer=read(journal) if journal.exists() else {}
        files=transfer.get('files',[])
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_HF_DELIVERED',error=failure,
            CPU_s=time.process_time()-began,CPU_reserved_s=0,
            outgoing_bytes_upper=sum(r['bytes'] for r in files),
            uncertain_HF_mutation=transfer.get('unknown_mutation',False),
            new_models=0,new_queries=0,new_integrations=0)
        write(root/'RESOURCE_LEDGER.json',ledger)
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def consume(root,allocation,guard):
    """Inside one admitted allocation, fetch this new bank only and verify it all."""
    if not os.environ.get('PBS_JOBID') or os.environ['PBS_JOBID']!=allocation['job_id']:
        raise PermissionError('PV1_LARGE_BANK_CONSUMPTION_ONLY_INSIDE_OWN_NATIVE_ALLOCATION')
    from .neural_scope import validate as validate_neural,inside
    from scripts.pi0292.data_consumer import extract_checked
    root=Path(root);m,bindings,inputs,stage,a=validate_neural(root)
    matches=[p.parent for p in (root/'pbs').glob('alloc-*/ALLOCATION_ADMISSION.json') if read(p)==allocation]
    from . import evaluation_platform as evaluation
    typed=evaluation.cold_profile(m) or 'platform_evaluation_version' in allocation
    expected=evaluation.STATUS if typed else 'PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
    if (len(matches)!=1
            or os.uname().nodename!=allocation['hostname'] or allocation['status']!=expected):
        raise PermissionError('PV1_EXACT_CONSUMER_CURRENT_ADMISSION_AND_HOST')
    directory=matches[0]
    if sha(directory/'REQUEST.json')!=allocation['request_sha256']:
        raise PermissionError('PV1_NATIVE_CONSUMER_REQUEST_HASH_CHANGED')
    if typed:
        evaluation.current_identity(m,read(directory/'REQUEST.json'),allocation,run_uuid=root.name,
            manifest_sha256=sha(root/'RUN_MANIFEST.json'),request_sha256=sha(directory/'REQUEST.json'),
            user=os.environ['USER'],hostname=os.uname().nodename,job_id=os.environ['PBS_JOBID'])
    binding=inside(root,'plans/PV1_HF_BANK_DELIVERY.json');receipt=read(binding)
    if m['execution']['sealed_inputs'].get(str(binding.relative_to(root)))!=sha(binding):
        raise PermissionError('PV1_FIXED_HF_REVISION_AND_FILE_LIST_BEFORE_DOWNLOAD')
    if (receipt['status']!='PASS_FIXED_REVISION_READY_LAST_EXACT_FULL_SET_SIZE_SHA256'
            or receipt['repo_id']!='leoyuer/pi0294-public-artifacts'
            or receipt['prefix']!='PI0294/'+a['run_uuid']+'/PV1_NEW_TRAIN_BANK_V1'
            or not re.fullmatch('[0-9a-f]{40}',receipt['revision'])
            or receipt['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']):
        raise PermissionError('PV1_ONLY_EXACT_QUALIFIED_NEW_DATA_HF_PREFIX')
    target=Path(inputs['new_dataset'])
    reuse=None
    if target.is_symlink():raise PermissionError('PV1_DOWNLOAD_ONLY_REGISTERED_CHILD_NEW_BANK')
    if not target.resolve().is_relative_to(root.resolve()):
        from .bank_reuse import validate as validate_reuse
        reuse=validate_reuse(root,target,receipt,stage)
    terminal=root/'evidence/NEW_BANK_CONSUMPTION_READY.private.json'
    if terminal.exists():
        proof=read(terminal)
        if (proof['delivery_binding_sha256']!=sha(binding) or proof['status']!='PASS_ACTUAL_FULL_NEW_BANK_CONSUMPTION'
                or proof['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']):
            raise PermissionError('PV1_PRIOR_CONSUMPTION_RECEIPT_IDENTITY_CHANGED')
        bank_members(target,guard)
        return proof
    if reuse is not None:
        guard();manifest,ready,members=bank_members(target,guard)
        value=official_consumption(root,allocation,guard,bindings,inputs,binding,target,receipt,manifest,members)
        value.update(qualified_closed_sibling_reuse=reuse,new_download_bytes=0,new_extraction_bytes=0)
        write(terminal,value);terminal.chmod(0o400)
        return value
    journal=root/'evidence/NEW_BANK_DOWNLOAD_INTENT.private.json'
    if journal.exists():raise FileExistsError('PV1_RECONCILE_EXISTING_DOWNLOAD_NO_BLIND_RESTART')
    write(journal,dict(job_id=allocation['job_id'],delivery_binding_sha256=sha(binding),started_epoch=time.time()))
    prefix=receipt['prefix']+'/';files={r['path'][len(prefix):]:r for r in receipt['files'] if r['path'].startswith(prefix)}
    if len(files)!=len(receipt['files']) or any('/' in name or name in ('.','..') for name in files):
        raise PermissionError('PV1_EXACT_SINGLE_LEVEL_NEW_BANK_DELIVERY_FILES')
    folder=root/'data/new_bank_download';folder.mkdir(parents=True,exist_ok=False)
    def get(name):
        guard();row=files[name]
        p=Path(download_dataset(receipt['repo_id'],row['path'],revision=receipt['revision'],local_dir=folder,
            token=existing_token(bindings)))
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('PV1_ACTUAL_FIXED_REVISION_DOWNLOAD_SIZE_AND_HASH')
        return p
    ready=read(get(READY));index_path=get(INDEX);index=read(index_path)
    if (ready['archive_index_sha256']!=sha(index_path) or ready['statistics_sha256']!=STATS
            or ready['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']
            or index['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']
            or index['source_ready_sha256']!=stage['new_bank_ready_sha256']
            or set(files)!=set(index['archives'])|{INDEX,READY}):
        raise PermissionError('PV1_DOWNLOADED_READY_ARCHIVE_INDEX_AND_FIXED_DATA_IDENTITY')
    seen=set()
    for name,proof in index['archives'].items():
        if not re.fullmatch(r'(?:metadata|parents-[0-9]{3})\.tar\.gz',name) or set(proof['member_sha256'])&seen:
            raise PermissionError('PV1_ONLY_DISJOINT_PRESEALED_NUMERIC_RGB_METADATA_SHARDS')
        p=get(name)
        if sha(p)!=proof['sha256'] or p.stat().st_size!=proof['bytes']:
            raise ValueError('PV1_ARCHIVE_PAYLOAD_PROOF_MISMATCH')
        def extract_guard(extra=0):
            guard()
            if shutil.disk_usage(root).free<extra+40*1024**3:
                raise PermissionError('PV1_NATIVE_EXTRACT_PRESERVES_FINAL_STORAGE_RESERVE')
        extract_checked(p,target,proof['member_sha256'],extract_guard)
        seen.update(proof['member_sha256'])
    manifest,ready,members=bank_members(target,guard)
    if seen!=set(members):raise PermissionError('PV1_DELIVERY_ARCHIVES_MUST_EXACTLY_COVER_ACTUAL_BANK_MEMBERS')
    value=official_consumption(root,allocation,guard,bindings,inputs,binding,target,receipt,manifest,members)
    value['new_download_bytes']=sum(r['bytes'] for r in files.values())
    write(terminal,value);terminal.chmod(0o400)
    return value


def official_consumption(root,allocation,guard,bindings,inputs,binding,target,receipt,manifest,members):
    from scripts.pi0294.native_dataset import NativeDataset
    from .export import batch_inventory
    dataset=NativeDataset(target,bindings['source'],bindings['cosmos'],for_optimization=True,
        raw_bindings=read(inputs['new_raw_bindings']))
    dataset.initialize();lookup={(r['id'],anchor):i for i,(r,anchor) in enumerate(dataset.rows)}
    parents=[]
    for task in ('T1','T2'):
        task_rows=[r for r in manifest['episodes'] if r['task']==task]
        parents.extend([task_rows[0]['id'],task_rows[-1]['id']])
    batches=[];mixed=[]
    for parent in dict.fromkeys(parents):
        for anchor in (0,320,600):
            guard();processed=dataset[lookup[parent,anchor]]['processed']
            batch=batch_inventory(dataset.processor.collator([processed]),1)
            batches.append(dict(parent=parent,anchor=anchor,batch=batch))
            if anchor==0:mixed.append(processed)
    mixed_batch=batch_inventory(dataset.processor.collator([mixed[0],mixed[-1]]),2)
    value=dict(status='PASS_ACTUAL_FULL_NEW_BANK_CONSUMPTION',delivery_binding_sha256=sha(binding),
        dataset_manifest_sha256=sha(target/'DATASET_MANIFEST.json'),source_ready_sha256=sha(target/'READY.json'),
        statistics_sha256=STATS,download_revision=receipt['revision'],job_id=allocation['job_id'],
        files=len(members),parent_episodes=len(manifest['episodes']),actual_payload_bytes=sum(p.stat().st_size for p in members.values()),
        full_member_hashes_verified=True,old_banks_reuploaded_or_redownloaded=False,new_queries=0,new_physics=0,
        actual_Hopper_official_processor_and_collator_batches=batches,mixed_CPU_batch=mixed_batch,
        native_loss_forward_or_optimizer_executed=False)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root)),flush=True)
