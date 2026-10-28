"""Exact CORR inference-only delta; no tensors, model or optimizer are loaded.

The unchanged first shard is an explicit fixed-HF dependency, not another
upload. A consumer must hash actual bytes, including its reused local shard.
Publication alone is not numerical qualification, a full resume or task proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.checkpoint_delivery import verify_remote
from scripts.pi0294.e1.public_inference import inference_members, tensor_header
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import public_config, reject_private
from .authority import AUTHORITY, PARENT_AUTHORITY, PARENT, R0, HARD_CLOSE

SCHEMA = 'DEC1_CORR_INFERENCE_DELTA_V1'
REPO = 'leoyuer/pi0294-public-artifacts'
FIRST = 'model-00001-of-00003.safetensors'
FIRST_SHA = 'cbcaea5ee88f1e0f1465043920a2647c67e7de17d24adfd1c477742a6168edec'
BASE_REV = '4c3097652bb81e43ea40a2b0b423086ec6ea7091'
BASE_PREFIX = 'PI0294/20261014T121640Z-incoming-pi0294-r3/public-M2048-0449992fddbc'
LICENSE_PREFIX = 'PI0294/20261016T150228Z-localpc-e1/public-inference-M2-CORR-4969f0180474'
LICENSE_REV = '79efde7c62634aed79c0dba11aad76b8f9141727'
LICENSE_NAMES = ('LICENSE', 'LICENSE_COSMOS.txt', 'NOTICE.txt',
                 'UPSTREAM_GR00T_README.md', 'UPSTREAM_COSMOS_README.md')
PREFIX = 'PI0294/20261018T182956Z-localpc-pv1/DEC1_CORR2048_INFERENCE_DELTA_V1'
LIMIT = 8*1024**3


def fail(message):
    raise PermissionError('DEC1_ASSET_' + message)


def dependency():
    return dict(repo_id=REPO, revision=BASE_REV, prefix=BASE_PREFIX,
                name=FIRST, sha256=FIRST_SHA, bytes=4986649584)


def verify_dependency(api):
    rows = [r for r in api.list_repo_tree(REPO,repo_type='dataset',revision=BASE_REV,
        path_in_repo=BASE_PREFIX,recursive=True) if getattr(r,'path',None)==BASE_PREFIX+'/'+FIRST]
    if len(rows)!=1 or rows[0].size!=dependency()['bytes'] or not rows[0].lfs:
        fail('EXACT_FIXED_HF_DEPENDENCY_MEMBER')
    value=rows[0].lfs
    if (value.sha256 if hasattr(value,'sha256') else value['sha256']) != FIRST_SHA:
        fail('FIXED_HF_DEPENDENCY_SHA256')


def authentication(bindings, mode):
    """Explicit public reads never inherit an unrelated site's credential path."""
    if bindings.get('public_anonymous_read') is True:
        if mode!='consume' or bindings.get('existing_token_file') is not None:
            fail('ANONYMOUS_IS_EXPLICIT_FIXED_PUBLIC_CONSUMER_ONLY')
        return False
    path=Path(bindings['existing_token_file'])
    if not path.is_file() or path.stat().st_uid!=os.getuid():
        fail('LOCAL_EXISTING_WRITER_LOCATOR_REQUIRED_BEFORE_DISPATCH')
    token=path.read_text().strip()
    if not token:fail('EMPTY_EXISTING_WRITER_CREDENTIAL')
    return token


def validate_manifest(value):
    reject_private(value)
    if (value.get('schema') != SCHEMA or value.get('authority_release') != AUTHORITY
            or value.get('original_checkpoint_manifest_sha256') != PARENT
            or value.get('statistics_sha256') != R0 or value.get('global_step') != 2048
            or value.get('dependency') != dependency()
            or value.get('full_resume') is not False or value.get('model_loaded') is not False
            or value.get('selected') != 'NONE' or value.get('deployment') != 'NONE'):
        fail('EXACT_CORR_R0_INFERENCE_ONLY_NOT_A_NEW_MODEL')
    files = value['files']; original = value['original_inference_sha256']
    names = {'config.json', 'statistics.json', 'model.safetensors.index.json',
             'processor/embodiment_id.json', 'processor/processor_config.json',
             'processor/statistics.json'} | {
                 'model-%05d-of-00003.safetensors' % i for i in (1, 2, 3)}
    if (set(original) != names or set(files) != names - {FIRST} | set(LICENSE_NAMES)
            or original[FIRST] != FIRST_SHA or original['statistics.json'] != R0
            or sum(r['bytes'] for r in files.values()) > LIMIT):
        fail('EXACT_NINE_INFERENCE_FILES_NO_OPTIMIZER_OR_EXTRA_DATA')
    relocated = {'config.json', 'processor/processor_config.json'}
    if set(value['metadata_changes']) != relocated:
        fail('EXACT_TWO_MODEL_NAME_RELOCATIONS')
    for name in names - {FIRST} - relocated:
        if files[name]['sha256'] != original[name]:
            fail('UNCHANGED_TENSORS_INDEX_AND_STATISTICS')
    for name in relocated:
        if value['metadata_changes'][name] != dict(original_sha256=original[name],
                change='ONLY_PRIVATE_COSMOS_MODEL_NAME_TO_PUBLIC_REPO_ID'):
            fail('RELOCATION_PROVENANCE')
    for row in files.values():
        if (type(row['bytes']) is not int or row['bytes'] <= 0
                or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])):
            fail('POSITIVE_BYTES_AND_FULL_SHA256')
    return value


def guard(root, mode):
    root = Path(root)
    m = read(root/'RUN_MANIFEST.json'); b = read(root/'OUTPUT_BINDING.private.json')
    lease = read(b['lease']); registry = read(b['registry'])
    matches = [r for r in registry['entries'] if r.get('role') == b['role']]
    if (root.is_symlink() or root.stat().st_uid != os.getuid()
            or (root/'RUN_MANIFEST.json').stat().st_mode & 0o222
            or (root/'OUTPUT_BINDING.private.json').stat().st_mode & 0o222
            or b.get('run_manifest_sha256')!=sha(root/'RUN_MANIFEST.json')
            or b.get('uid')!=os.getuid()
            or m.get('schema') != 'DEC1_INFERENCE_ASSET_CHILD_V1'
            or m.get('run_uuid') != root.name or m.get('owner_uid') != os.getuid()
            or m.get('mode') != mode or mode not in ('publish', 'consume', 'ready')
            or m.get('authority_release') != AUTHORITY
            or m.get('parent_authority_release') != PARENT_AUTHORITY
            or m.get('campaign_run_uuid') != '20261018T182956Z-localpc-pv1'
            or m.get('original_T0_epoch') != 1792348196
            or not time.time() < m['deadline_epoch'] <= HARD_CLOSE
            or m.get('CPU_upper_s') != (180 if mode=='ready' else 4500) or m.get('GPU_PBS') != 0
            or m.get('new_models') != 0 or m.get('transfer_upper_bytes') != 8*1024**3
            or m.get('execution_profile') != ('vanda-neural-probe' if mode=='consume' else 'hopper-neural')
            or not root.resolve().is_relative_to(Path(m['registered_runs_root']).resolve())
            or len(matches) != 1 or Path(matches[0]['path']).resolve() != root.resolve()
            or lease.get('status') != 'ACTIVE' or lease.get('run_uuid') != root.name
            or lease.get('authority_release') != AUTHORITY
            or lease.get('deadline_epoch') != m['deadline_epoch']
            or (root/'STOP.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        fail('STANDARD_OWNED_REGISTERED_CPU_CHILD_LIVE_AUTHORITY_AND_BOUNDS')
    for name, checksum in m['sealed_inputs'].items():
        p = root/name
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_mode & 0o222 or sha(p) != checksum):
            fail('SEALED_INPUT_BYTES')
    if 'BINDINGS.private.json' not in m['sealed_inputs']:
        fail('EXPLICIT_PRIVATE_INPUT_BINDINGS')
    if 'CPU_BUDGET_PREPAID.json' not in m['sealed_inputs']:
        fail('NO_CPU_DISPATCH_BEFORE_SAME_CAMPAIGN_PREPAYMENT')
    paid=read(root/'CPU_BUDGET_PREPAID.json')
    if (paid.get('authority_release')!=AUTHORITY or paid.get('original_T0_epoch')!=1792348196
            or paid.get('campaign_run_uuid')!=m['campaign_run_uuid']
            or paid.get('costs_inside_original_PV1_and_DEC1') is not True
            or paid.get('CPU_upper_s_by_mode',{}).get(mode)!=m['CPU_upper_s']
            or paid.get('total_Hopper_and_alternate_CPU_prepaid_s')!=9180
            or paid.get('transfer_reserved_bytes')!=16*1024**3
            or paid.get('unknown_effects')!=[]):
        fail('EXPLICIT_NONADDITIVE_CPU_AND_TRANSFER_PREPAYMENT')
    bindings=read(root/'BINDINGS.private.json')
    authentication(bindings,mode)
    return m, bindings


def publish(root, bindings, *, api, add, download):
    out = root/'payload'; out.mkdir(mode=0o700)
    ck = Path(bindings['checkpoint'])
    if ck.is_symlink() or sha(ck/'CHECKPOINT_COMPLETE.json') != PARENT:
        fail('EXACT_ORIGINAL_COMPLETE_MANIFEST')
    complete = read(ck/'CHECKPOINT_COMPLETE.json')
    if complete['global_step'] != 2048 or not complete['full_resume']:
        fail('ORIGINAL_NATIVE_STEP')
    files = complete['files']; names = inference_members(files, read(ck/'model.safetensors.index.json'))
    paths = {}; changes = {}
    for name in names:
        p = ck/name
        if p.is_symlink() or not p.resolve().is_relative_to(ck.resolve()) or sha(p) != files[name]:
            fail('ORIGINAL_MEMBER_CHANGED')
        if name in ('config.json', 'processor/processor_config.json'):
            q = out/name; q.parent.mkdir(parents=True, exist_ok=True)
            write(q, public_config(read(p), processor=name.startswith('processor/')))
            changes[name] = dict(original_sha256=files[name],
                change='ONLY_PRIVATE_COSMOS_MODEL_NAME_TO_PUBLIC_REPO_ID')
        else:
            q = p
            if name.endswith('.json'): reject_private(read(p))
            else: tensor_header(p)
        if name != FIRST: paths[name] = q
    for name in LICENSE_NAMES:
        paths[name] = Path(download(REPO, LICENSE_PREFIX+'/'+name, repo_type='dataset', revision=LICENSE_REV))
        reject_private(paths[name].read_text())
    inventory = {n: dict(bytes=p.stat().st_size, sha256=sha(p)) for n,p in paths.items()}
    value = dict(schema=SCHEMA, authority_release=AUTHORITY, global_step=2048,
        original_checkpoint_manifest_sha256=PARENT, statistics_sha256=R0,
        original_inference_sha256={n:files[n] for n in names}, files=inventory,
        metadata_changes=changes, dependency=dependency(), full_resume=False, model_loaded=False,
        optimizer_or_rng_included=False, selected='NONE', deployment='NONE')
    validate_manifest(value); write(out/'INFERENCE_MANIFEST.json',value)
    paths['INFERENCE_MANIFEST.json'] = out/'INFERENCE_MANIFEST.json'
    inventory = dict(inventory, **{'INFERENCE_MANIFEST.json':
        dict(bytes=paths['INFERENCE_MANIFEST.json'].stat().st_size, sha256=sha(paths['INFERENCE_MANIFEST.json']))})
    info = api.repo_info(REPO, repo_type='dataset')
    if info.private or any(n.startswith(PREFIX+'/') for n in api.list_repo_files(REPO,repo_type='dataset',revision=info.sha)):
        fail('PUBLIC_EXISTING_REPO_FRESH_PREFIX_NO_BLIND_REUPLOAD')
    verify_dependency(api)
    journal = dict(schema=SCHEMA, status='UPLOAD_INTENT_UNKNOWN_NO_RETRY', repo_id=REPO,
        prefix=PREFIX, parent_revision=info.sha, files=inventory, dependency=dependency(),
        manifest_sha256=sha(out/'INFERENCE_MANIFEST.json'), logical_upload_upper_bytes=sum(r['bytes'] for r in inventory.values()))
    write(root/'TRANSFER.private.json',journal)
    revision = api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
        commit_message='DEC1 fixed CORR2048 inference delta, original frozen shard dependency, not deployment',
        operations=[add(path_in_repo=PREFIX+'/'+n,path_or_fileobj=str(p)) for n,p in paths.items()],num_threads=2).oid
    journal.update(status='PAYLOAD_COMMITTED_PENDING_INDEPENDENT_CONSUMER_READBACK', data_revision=revision)
    write(root/'TRANSFER.private.json',journal)
    verify_remote(api,REPO,PREFIX,revision,inventory,download,api.token)
    for name in names:
        if sha(ck/name) != files[name]: fail('ORIGINAL_CHANGED_DURING_DELIVERY')
    return journal


def consume(root, bindings, *, download):
    transfer = bindings['transfer']
    if (transfer['schema'] != SCHEMA or transfer['repo_id'] != REPO or transfer['prefix'] != PREFIX
            or transfer['status'] != 'PAYLOAD_COMMITTED_PENDING_INDEPENDENT_CONSUMER_READBACK'
            or transfer['dependency'] != dependency()
            or sum(r['bytes'] for r in transfer['files'].values()) > LIMIT):
        fail('EXACT_FIXED_UNREADY_PAYLOAD_FOR_INDEPENDENT_READBACK')
    def get(name):
        row = transfer['files'][name]
        p = Path(download(REPO,PREFIX+'/'+name,repo_type='dataset',revision=transfer['data_revision'],local_dir=root/'download'))
        if p.is_symlink() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            fail('FULL_ACTUAL_CONSUMER_MEMBER_READBACK')
        return p
    value = validate_manifest(read(get('INFERENCE_MANIFEST.json')))
    if (sha(root/'download'/PREFIX/'INFERENCE_MANIFEST.json') != transfer['manifest_sha256']
            or value['files'] != {n:r for n,r in transfer['files'].items() if n!='INFERENCE_MANIFEST.json'}):
        fail('EXACT_MANIFEST_TRANSFER_IDENTITY')
    frozen = Path(bindings['frozen_local_member'])
    if (frozen.is_symlink() or frozen.stat().st_uid != os.getuid()
            or frozen.stat().st_size != dependency()['bytes'] or sha(frozen) != FIRST_SHA):
        fail('REUSED_LOCAL_FROZEN_BYTES_NOT_JUST_HF_OR_OLD_RECEIPT')
    paths = {n:get(n) for n in value['files']}
    checkpoint = root/'inference'; checkpoint.mkdir(mode=0o700)
    # These are immutable inference inputs, never Trainer output destinations.
    # Preserve source modes: chmod on a shared inode would alter the old asset.
    for name in value['original_inference_sha256']:
        p = checkpoint/name; p.parent.mkdir(parents=True,exist_ok=True)
        os.link(frozen if name==FIRST else paths[name],p)
    actual = {n:sha(checkpoint/n) for n in value['original_inference_sha256']}
    if (actual[FIRST] != FIRST_SHA or actual['statistics.json'] != R0
            or any(actual[n] != value['files'][n]['sha256'] for n in actual if n!=FIRST)):
        fail('MATERIALIZED_EXACT_NINE_MEMBER_INFERENCE')
    result = dict(schema=SCHEMA,status='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL',
        repo_id=REPO,prefix=PREFIX,data_revision=transfer['data_revision'],
        manifest_sha256=transfer['manifest_sha256'],dependency=dependency(),inference_files_sha256=actual,
        original_checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,
        logical_download_upper_bytes=sum(r['bytes'] for r in transfer['files'].values()),
        reused_local_bytes=dependency()['bytes'],new_GPU=0,new_PBS=0,new_models=0,new_queries=0,
        full_resume=False,selected='NONE',deployment='NONE')
    reject_private(result); write(root/'CONSUMPTION.json',result)
    return result


def ready(root, bindings, *, api, add, download):
    transfer=bindings['transfer']; consumer=read(root/'CONSUMPTION.json')
    if (consumer.get('schema')!=SCHEMA
            or consumer.get('status')!='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL'
            or consumer.get('data_revision')!=transfer['data_revision']
            or consumer.get('manifest_sha256')!=transfer['manifest_sha256']
            or consumer.get('dependency')!=dependency()
            or consumer.get('original_checkpoint_manifest_sha256')!=PARENT
            or consumer.get('statistics_sha256')!=R0
            or any(consumer.get(n)!=0 for n in ('new_GPU','new_PBS','new_models','new_queries'))):
        fail('INDEPENDENT_ACTUAL_CONSUMER_READBACK_REQUIRED_BEFORE_READY')
    if (transfer.get('repo_id')!=REPO or transfer.get('prefix')!=PREFIX
            or transfer.get('status')!='PAYLOAD_COMMITTED_PENDING_INDEPENDENT_CONSUMER_READBACK'):
        fail('EXACT_COMMITTED_PAYLOAD_NOT_AN_UNKNOWN_UPLOAD')
    p=Path(download(REPO,PREFIX+'/INFERENCE_MANIFEST.json',repo_type='dataset',revision=transfer['data_revision']))
    if sha(p)!=transfer['manifest_sha256']:fail('FIXED_MANIFEST_BYTES')
    manifest=validate_manifest(read(p))
    expected={n:r['sha256'] for n,r in manifest['files'].items() if n in manifest['original_inference_sha256']}
    expected[FIRST]=FIRST_SHA
    if consumer['inference_files_sha256']!=expected:fail('ALL_NINE_CONSUMED_MEMBERS')
    verify_remote(api,REPO,PREFIX,transfer['data_revision'],transfer['files'],download,api.token)
    verify_dependency(api)
    head=api.repo_info(REPO,repo_type='dataset')
    if head.private or head.sha!=transfer['data_revision']:
        fail('HF_HEAD_MOVED_RECONCILE_NO_OVERWRITE')
    value=dict(schema=SCHEMA,status='READY_INDEPENDENT_FULL_CONSUMER_BYTES_NOT_NUMERICAL_QUALIFICATION',
        data_revision=transfer['data_revision'],files=transfer['files'],dependency=dependency(),
        manifest_sha256=transfer['manifest_sha256'],consumer_receipt_sha256=sha(root/'CONSUMPTION.json'),
        original_checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,model_loaded=False,
        full_resume=False,READY_uploaded_last=True,selected='NONE',deployment='NONE')
    reject_private(value);write(root/'READY.json',value)
    journal=dict(schema=SCHEMA,status='READY_COMMIT_INTENT_UNKNOWN_NO_RETRY',repo_id=REPO,prefix=PREFIX,
        data_revision=transfer['data_revision'],READY_sha256=sha(root/'READY.json'))
    write(root/'TRANSFER.private.json',journal)
    revision=api.create_commit(REPO,repo_type='dataset',parent_commit=head.sha,
        commit_message='DEC1 CORR inference READY after independent Vanda full byte readback, no model load',
        operations=[add(path_in_repo=PREFIX+'/READY.json',path_or_fileobj=str(root/'READY.json'))]).oid
    files=dict(transfer['files'],**{'READY.json':dict(bytes=(root/'READY.json').stat().st_size,sha256=sha(root/'READY.json'))})
    journal.update(status='READY_COMMITTED_READBACK_PENDING',revision=revision)
    write(root/'TRANSFER.private.json',journal)
    verify_remote(api,REPO,PREFIX,revision,files,download,api.token)
    journal.update(status='PASS_FIXED_CORR_DELTA_READY_AFTER_INDEPENDENT_CONSUMER_READBACK',files=files,
        dependency=dependency(),manifest_sha256=transfer['manifest_sha256'],consumer_receipt_sha256=sha(root/'CONSUMPTION.json'))
    write(root/'TRANSFER.private.json',journal)
    return journal


def run(root):
    root = Path(root); mode = read(root/'RUN_MANIFEST.json')['mode']; m,b = guard(root,mode)
    if (root/'RESULT.private.json').exists() or (root/'TRANSFER.private.json').exists():
        fail('EXISTING_RESULT_OR_INTENT_REQUIRES_READONLY_RECONCILIATION')
    owner = OwnerLock(root/'ASSET_OWNER.lock')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    resource.setrlimit(resource.RLIMIT_CPU,(m['CPU_upper_s'],m['CPU_upper_s']))
    def stop(signum,frame): raise TimeoutError('DEC1_ASSET_BOUNDED_CPU_WALL')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop)
    signal.alarm(max(1,min(3600,int(m['deadline_epoch']-time.time()-30))))
    started=time.process_time(); result=dict(status='FAILED_OR_UNKNOWN_PRESERVED_NO_RETRY',mode=mode)
    try:
        from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
        token = authentication(b,mode)
        api = HfApi(token=token)
        if mode=='publish': result=publish(root,b,api=api,add=CommitOperationAdd,download=hf_hub_download)
        elif mode=='consume': result=consume(root,b,download=lambda *a,**kw:hf_hub_download(*a,token=token,**kw))
        else:
            if 'CONSUMPTION.json' not in m['sealed_inputs']:fail('SEALED_REAL_CONSUMER_RECEIPT')
            result=ready(root,b,api=api,add=CommitOperationAdd,download=hf_hub_download)
        return result
    finally:
        signal.alarm(0)
        result=dict(result,CPU_s=time.process_time()-started,completed_epoch=time.time())
        write(root/'RESULT.private.json',result)
        owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    print(json.dumps(run(parser.parse_args().root)))
