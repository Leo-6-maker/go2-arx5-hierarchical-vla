"""CR2-only immutable HF delivery; full archive readback precedes READY."""
from pathlib import Path
import re
import shutil
import signal
import threading
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.hf_bank import contained,verified_tar
from scripts.pi0294.data_cov1.bank import public_json
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.pv1.data_delivery import data_network,existing_token,verify_remote
from .bank import LOCAL,closed_child,raw_file
from .plan import AUTH,R1
from .runtime import bind_cpu,wait_priority
from .scope import sealed_file,validate

REPO='leoyuer/pi0294-public-artifacts'
INDEX='ARCHIVE_MEMBER_MANIFEST.json'


def checked_package(source, result_sha256, guard):
    source=Path(source);result=source/'evidence/PACK_RESULT.private.json';sealed_file(result,result_sha256)
    actual=read(result);output=source/'delivery';index=read(output/INDEX)
    bank=source/'datasets/CR2_FIXED_R1_P12';local=read(bank/'LOCAL_FROZEN.json')
    if (actual['status']!='PASS_CR2_LOCAL_BANK_AND_PACK_NOT_HF_READY'
            or actual['gate']['status']!='PASS_CR2_DATA_BANK_GATE' or actual['HF_READY'] is not False
            or index['schema']!='PI0294_CR2_ARCHIVE_INDEX_V1' or index['statistics_sha256']!=R1
            or index['dataset_manifest_sha256']!=sha(bank/'DATASET_MANIFEST.json')
            or actual['dataset_manifest_sha256']!=index['dataset_manifest_sha256']
            or actual['archive_index_sha256']!=sha(output/INDEX)
            or local['status']!=LOCAL or local['statistics_sha256']!=R1
            or index['source_local_frozen_sha256']!=sha(bank/'LOCAL_FROZEN.json')
            or local['payload_manifest_sha256']!=sha(bank/'PAYLOAD_MANIFEST.json')):
        raise PermissionError('CR2_ONLY_CLOSED_FULLY_QUALIFIED_FIXED_R1_PACKAGE')
    public_json(index);public_json(local)
    if {p.name for p in output.iterdir()}!=set(index['archives'])|{INDEX}:
        raise PermissionError('CR2_NO_EXTRA_FILES_OR_PRIOR_READY_IN_SOURCE_DELIVERY')
    covered=set();members={}
    for name,proof in index['archives'].items():
        if (not re.fullmatch(r'(?:metadata|parents-[0-9]{3})\.tar\.gz',name)
                or set(proof['member_sha256'])&covered):
            raise PermissionError('CR2_ONLY_DISJOINT_EXPECTED_ARCHIVE_NAMES_AND_MEMBERS')
        payload={n:contained(bank,n) for n in proof['member_sha256']}
        for n,p in payload.items():guard();sealed_file(p,proof['member_sha256'][n])
        guard();path=output/name;raw_file(path,proof['sha256'])
        if path.stat().st_size!=proof['bytes'] or verified_tar(path,payload,reuse=True)!=proof:
            raise ValueError('CR2_SOURCE_ARCHIVE_FULL_MEMBER_IDENTITY')
        covered.update(payload);members.update(payload)
    actual_members={p.relative_to(bank).as_posix() for p in bank.rglob('*') if p.is_file()}
    if covered!=actual_members:raise PermissionError('CR2_ARCHIVES_MUST_COVER_FULL_BANK_EXACTLY')
    return index,members


def run(root):
    r=Path(root);m,inputs=validate(r,'PACK');work=read(r/'plans/WORK_PLAN.private.json')
    ledger=read(r/'RESOURCE_LEDGER.json');assert ledger['status']=='RESERVED' and not ledger['unknown_effects']
    if (work.get('operation')!='CR2_FIXED_REVISION_HF_DELIVERY'
            or work.get('repo_id')!=REPO or work.get('maximum_transfer_bytes')!=16*1024**3):
        raise PermissionError('CR2_EXPLICIT_EXISTING_REPO_AND_ORIGINAL_TRANSFER_CAP')
    source=closed_child(r.parent.parent,work['closed_pack_root'],'PACK','TERMINAL_PACK')
    parent=read(r.parent.parent/'CR2_ADOPTION_AND_SCOPE.json')
    prefix='PI0294/'+parent['run_uuid']+'/CR2_FIXED_R1_P12'
    if work.get('HF_prefix')!=prefix:raise PermissionError('CR2_ONE_NEW_ISOLATED_PREFIX')
    lock=OwnerLock(r/'CR2_WORKER.lock');began=time.process_time();failure=None;restore=None
    bind_cpu(m);journal=r/'evidence/HF_TRANSFER.private.json'
    if journal.exists():lock.close();raise FileExistsError('CR2_RECONCILE_OLD_HF_INTENT_NO_BLIND_RETRY')
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        lock.close();raise PermissionError('CR2_FINITE_DELIVERY_OWN_MAIN_THREAD_ONLY')
    def guard():
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-60
                or any((p/'STOP.json').exists() for p in (r,r.parent.parent))
                or shutil.disk_usage(r).free<40*1024**3):
            raise PermissionError('CR2_DELIVERY_FINITE_ORIGINAL_CPU_WALL_STORAGE_OR_STOP')
    state=dict(status='PRECOMMIT_VALIDATING',READY_claim=False,unknown_mutation=False,
        repo_id=REPO,prefix=prefix,new_models=0,new_queries=0,new_physics=0)
    def expired(*unused):raise TimeoutError('CR2_ORIGINAL_DELIVERY_CUTOFF')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,m['limits']['operation_deadline_epoch']-time.time()))
    write(journal,state)
    try:
        ledger['status']='RUNNING';write(r/'RESOURCE_LEDGER.json',ledger);wait_priority(r,guard)
        index,members=checked_package(source,work['source_pack_result_sha256'],guard)
        output=source/'delivery';names=set(index['archives'])|{INDEX}
        files=[dict(path=prefix+'/'+n,bytes=(output/n).stat().st_size,sha256=sha(output/n)) for n in sorted(names)]
        payload_bytes=sum(x['bytes'] for x in files)
        # Includes payload upload + independent payload readback and a finite
        # metadata/control allowance; not a claim of measured TCP wire bytes.
        upper=2*payload_bytes+64*1024**2
        previous_transfer=read(r.parent.parent/'RESOURCE_LEDGER.json')['network_bytes_upper']
        if previous_transfer+upper>work['maximum_transfer_bytes']:
            raise PermissionError('CR2_COMPLETE_PAYLOAD_UPLOAD_AND_READBACK_TRANSFER_BOUND')
        seen=set();resident=0
        for p in r.parent.parent.rglob('*'):
            if p.is_symlink() or not p.is_file():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:resident+=max(s.st_size,s.st_blocks*512);seen.add(key)
        resident_upper=resident+payload_bytes+64*1024**2
        if resident_upper>48*1024**3:
            raise PermissionError('CR2_FULL_READBACK_MUST_FIT_ORIGINAL_ALL_OUTPUT_PEAK')
        if not inputs.get('hf_token_file'):
            raise PermissionError('CR2_EXPLICIT_EXISTING_WRITER_LOCATOR_REQUIRED_BEFORE_REMOTE_INTENT')
        token=existing_token(inputs)
        if not token:raise PermissionError('CR2_EXISTING_WRITER_CREDENTIAL_EMPTY_NO_DEFAULT_PROFILE_FALLBACK')
        restore=data_network()
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        api=HfApi(token=token);info=api.repo_info(REPO,repo_type='dataset')
        if info.private:raise PermissionError('CR2_ONLY_EXPLICITLY_AUTHORIZED_SANITIZED_PUBLIC_ARTIFACT_REPO')
        if any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
            raise FileExistsError('CR2_NEVER_OVERWRITE_OR_BLINDLY_RETRY_EXISTING_PREFIX')
        state.update(status='DATA_COMMIT_INTENT',unknown_mutation=True,parent_revision=info.sha,
            files=files,logical_transfer_upper_bytes=upper,resident_with_full_readback_upper_bytes=resident_upper)
        write(journal,state);guard()
        revision=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='CR2 fixed R1 qualified TRAIN payload before READY',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(output/n)) for n in sorted(names)]).oid
        state.update(status='DATA_COMMIT_CREATED',data_revision=revision,unknown_mutation=False);write(journal,state)
        verify_remote(api,REPO,prefix,revision,files,guard,r/'evidence/metadata_readback')
        readback=r/'evidence/full_payload_readback';readback.mkdir()
        verified=[]
        for row in files:
            guard();path=Path(hf_hub_download(REPO,row['path'],repo_type='dataset',revision=revision,
                token=api.token,local_dir=readback,force_download=True))
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('CR2_ACTUAL_INDEPENDENT_FULL_PAYLOAD_DOWNLOAD_IDENTITY')
            name=Path(row['path']).name
            if name in index['archives']:
                proof=index['archives'][name]
                actual=verified_tar(path,{n:members[n] for n in proof['member_sha256']},reuse=True)
                if actual!=proof:raise ValueError('CR2_DOWNLOADED_ARCHIVE_EVERY_REAL_MEMBER_IDENTITY')
            verified.append(dict(path=row['path'],sha256=row['sha256'],bytes=row['bytes'],full_payload_read=True))
        local=r/'delivery';local.mkdir()
        ready=dict(schema='PI0294_CR2_HF_READY_V1',status='READY_FIXED_REVISION_FULL_MEMBER_READBACK',
            authority_release=AUTH,run_uuid=parent['run_uuid'],repo_id=REPO,repo_type='dataset',private=False,
            prefix=prefix,data_revision=revision,files=files,statistics_sha256=R1,
            dataset_manifest_sha256=index['dataset_manifest_sha256'],archive_index_sha256=sha(output/INDEX),
            complete_payload_readback_before_READY=verified,READY_uploaded_last=True,
            new_CR2_only=True,PV1_changed=False,training_or_model_execution=False)
        public_json(ready);put(local/'READY.json',ready);guard()
        state.update(status='READY_COMMIT_INTENT',unknown_mutation=True);write(journal,state)
        final=api.create_commit(REPO,repo_type='dataset',parent_commit=revision,
            commit_message='CR2 READY after actual full archive and member readback',
            operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(local/'READY.json'))]).oid
        final_files=files+[dict(path=prefix+'/READY.json',bytes=(local/'READY.json').stat().st_size,sha256=sha(local/'READY.json'))]
        state.update(status='READY_COMMIT_CREATED',revision=final,files=final_files,unknown_mutation=False);write(journal,state)
        verify_remote(api,REPO,prefix,final,final_files,guard,r/'evidence/final_metadata_readback')
        downloaded=Path(hf_hub_download(REPO,prefix+'/READY.json',repo_type='dataset',revision=final,
            token=api.token,local_dir=r/'evidence/final_READY_readback',force_download=True))
        if downloaded.read_bytes()!=(local/'READY.json').read_bytes():raise ValueError('CR2_FINAL_FIXED_REVISION_READY_BYTES')
        state.update(status='PASS_CR2_FIXED_REVISION_FULL_PAYLOAD_MEMBER_READBACK_READY_LAST',READY_claim=True,
            dataset_manifest_sha256=index['dataset_manifest_sha256'],full_payload_readback=verified,
            final_payload_content_hashes_equal_verified_data_revision=True)
        put(r/'evidence/DELIVERY_RESULT.private.json',state)
        return dict(status=state['status'],revision=final,prefix=prefix,dataset_manifest_sha256=index['dataset_manifest_sha256'])
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);raise
    finally:
        if restore is not None:restore()
        write(journal,state)
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_DELIVERY',CPU_s=time.process_time()-began,
            CPU_reserved_s=0,failure=failure,logical_transfer_upper_bytes=state.get('logical_transfer_upper_bytes',0),
            output_peak_bytes=state.get('resident_with_full_readback_upper_bytes',0),
            unknown_effects=['HF_MUTATION_OUTCOME_UNKNOWN'] if state.get('unknown_mutation') else [])
        write(r/'RESOURCE_LEDGER.json',ledger)
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous);lock.close()
