"""Parent-bound inference export; full resume archives remain with the producer.

This module does not submit work, transfer credentials, or turn a partial
training checkpoint into the predeclared 8192 policy.
"""
import json
import shutil
import time
from pathlib import Path

from scripts.pi0294.native_dataset import sha
from .shared_checkpoint import validate_header,raw_reference

SCHEMA='SHARED4_CLOSED8192_INFERENCE_DELIVERY_V1'
ARCHIVE_ONLY={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'}
WITNESSES=('RUN_MANIFEST.json','RESULT.json','control/LEASE.json',
           'RELAY_CLOSED.json','ACTUAL_DELTA_CHECKPOINT_RESTORE.json')


def validate(root,*,manifest_sha256,checkpoint_manifest_sha256):
    root=Path(root)
    def read(name):
        f=root/name
        if f.is_symlink()or not f.resolve().is_relative_to(root.resolve()):
            raise PermissionError('INFERENCE_DELIVERY_REGULAR_LOCAL_MEMBER')
        return json.loads(f.read_bytes())
    if sha(root/'INFERENCE_PACKAGE.json')!=manifest_sha256:
        raise PermissionError('EXACT_INFERENCE_DELIVERY_MANIFEST')
    package=read('INFERENCE_PACKAGE.json')
    if (package.get('schema')!=SCHEMA or package.get('global_step')!=8192
            or package.get('checkpoint_manifest_sha256')!=checkpoint_manifest_sha256
            or package.get('full_training_archive_transferred')is not False
            or package.get('full_training_archive_retained_at_producer')is not True):
        raise PermissionError('CLOSED8192_ONLY_INFERENCE_PACKAGE_NOT_RESUME')
    for name,pin in package['files'].items():
        f=root/name
        if f.is_symlink()or not f.resolve().is_relative_to(root.resolve())or sha(f)!=pin:
            raise PermissionError('CHANGED_INFERENCE_DELIVERY_MEMBER:'+name)
    header=validate_header(read('CHECKPOINT_COMPLETE.json'))
    if (sha(root/'CHECKPOINT_COMPLETE.json')!=checkpoint_manifest_sha256
            or header['global_step']!=8192
            or header['recipe']['capacity_qualification_only']
            or package['files']!={**{n:p for n,p in header['files'].items()if n not in ARCHIVE_ONLY},
                'CHECKPOINT_COMPLETE.json':checkpoint_manifest_sha256,
                **{'producer/'+n:package['files'].get('producer/'+n)for n in WITNESSES}}):
        raise PermissionError('EXACT_FULL_HEADER_AND_INFERENCE_SUBSET')
    m=read('producer/RUN_MANIFEST.json');result=read('producer/RESULT.json')
    lease=read('producer/control/LEASE.json');closed=read('producer/RELAY_CLOSED.json')
    restore=read('producer/ACTUAL_DELTA_CHECKPOINT_RESTORE.json')
    if (m.get('recipe')!=header['recipe'] or m.get('science_run_id')!=package['science_run_id']
            or result.get('status')!='COMPLETE_SHARED_RAW_FINITE_LEARNING'
            or result.get('optimizer_updates')!=8192 or result.get('TRAIN_visits')!=8192*32
            or result.get('error')is not None or lease.get('status')!='CLOSED'
            or lease.get('manifest_sha256')!=package['files']['producer/RUN_MANIFEST.json']
            or lease.get('result_sha256')!=package['files']['producer/RESULT.json']
            or closed.get('effects_closed')is not True
            or closed.get('producer_run_uuid')!=m.get('run_uuid')
            or restore.get('checkpoint_manifest_sha256')!=checkpoint_manifest_sha256
            or any(restore.get(k)!=v for k,v in dict(model_reconstruction=True,
                optimizer_state_entries=42,optimizer_updates=8192,scheduler_restored=True,
                RNG_restored=True,sampler_cursor=8192*32,frozen_parent_unchanged=True).items())
            or read('RAW_PARENT_REFERENCE.json')!=json.loads(json.dumps(raw_reference()))):
        raise PermissionError('CLOSED_PRODUCER_AND_ACTUAL_FULL_STATE_RESTORE_REQUIRED')
    return package


def export(producer,destination,*,guard):
    """One immutable export after native close; no optimizer state duplication."""
    producer=Path(producer);destination=Path(destination)
    checkpoint=producer/'training/checkpoint-8192'
    header=validate_header(json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes()))
    # Verify the retained training archive once at this transfer boundary.
    for name,pin in header['files'].items():
        guard();path=checkpoint/name
        if path.is_symlink()or not path.resolve().is_relative_to(checkpoint.resolve())or sha(path)!=pin:
            raise PermissionError('FULL_SOURCE_TRAINING_ARCHIVE_CHANGED:'+name)
    destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    files={}
    sources={n:checkpoint/n for n in header['files']if n not in ARCHIVE_ONLY}
    sources['CHECKPOINT_COMPLETE.json']=checkpoint/'CHECKPOINT_COMPLETE.json'
    sources.update({'producer/'+n:producer/n for n in WITNESSES})
    for name,path in sources.items():
        guard();out=destination/name;out.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb')as src,out.open('xb')as dst:shutil.copyfileobj(src,dst,1024**2)
        files[name]=sha(out)
    m=json.loads((producer/'RUN_MANIFEST.json').read_bytes())
    package=dict(schema=SCHEMA,science_run_id=m['science_run_id'],global_step=8192,
        checkpoint_manifest_sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json'),files=files,
        full_training_archive_transferred=False,full_training_archive_retained_at_producer=True,
        source_training_archive=str(checkpoint),standalone_model=False,
        required_raw_parent=raw_reference(),default_behavioral_endpoint=8192,
        DEV_conditions=68,intermediate_behavioral_panels=0)
    from scripts.pi0292.durable import write
    write(destination/'INFERENCE_PACKAGE.json',package)
    validate(destination,manifest_sha256=sha(destination/'INFERENCE_PACKAGE.json'),
        checkpoint_manifest_sha256=package['checkpoint_manifest_sha256'])
    return dict(manifest_sha256=sha(destination/'INFERENCE_PACKAGE.json'),
        checkpoint_manifest_sha256=package['checkpoint_manifest_sha256'],
        package_root=str(destination),files=len(files),payload_bytes=sum((destination/n).stat().st_size for n in files))


def hf_api(route):
    from huggingface_hub import HfApi
    # Use the pre-existing station credential in memory; never copy it into a
    # capsule, result, environment dump or the model artifact.
    token=Path(route['token_path']).read_text().strip()
    api=HfApi(token=token)
    if api.repo_info(repo_id=route['repo'],repo_type='dataset').private is not True:
        raise PermissionError('PRIVATE_AUTHORIZED_MODEL_ARTIFACT_ROUTE_REQUIRED')
    return api,token


def publish_hf(c,exported):
    from scripts.pi0294.night10 import native_relay as relay
    route=c['hf_route'];api,token=hf_api(route);root=Path(exported['package_root'])
    marker=dict(status='COMPLETE_CLOSED8192_PRIVATE_DELIVERY',science_run_id=c['science_run_id'],
        manifest_sha256=exported['manifest_sha256'],checkpoint_manifest_sha256=exported['checkpoint_manifest_sha256'])
    relay.exclusive(root/'DELIVERY_COMPLETE.json',marker)
    parent=api.repo_info(repo_id=route['repo'],repo_type='dataset').sha
    intent=dict(repo=route['repo'],prefix=route['prefix'],parent_commit=parent,
        manifest_sha256=exported['manifest_sha256'],epoch=time.time(),automatic_write_retry=False)
    relay.exclusive(Path(c['root'])/'HF_PUBLISH_INTENT.json',intent)
    # An expected parent and immutable unique prefix prevent a second commit
    # after an ambiguous write. An exception leaves this intent for readback.
    commit=api.upload_folder(repo_id=route['repo'],repo_type='dataset',folder_path=str(root),
        path_in_repo=route['prefix'],parent_commit=parent,
        commit_message='Closed shared8192 inference '+c['science_run_id'])
    receipt=dict(intent,revision=commit.oid,private=True,completed_epoch=time.time())
    relay.exclusive(Path(c['root'])/'HF_PUBLISH_COMPLETE.json',receipt)
    return receipt


def receive_hf(c):
    """Finite CPU input boundary, before neural ownership or model loading."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from scripts.pi0294.night10 import native_relay as relay
    route=c['hf_route'];api,token=hf_api(route);destination=Path(c['shared_delivery_root'])
    cache=str(Path(c['root'])/'cache/hf-delivery')
    while time.time()+120<c['dispatch_deadline_epoch']:
        info=api.repo_info(repo_id=route['repo'],repo_type='dataset')
        if info.private is not True:raise PermissionError('PRIVATE_ROUTE_CHANGED')
        def fetch(name):
            if Path(name).is_absolute()or '..'in Path(name).parts:
                raise PermissionError('PRIVATE_ARTIFACT_MEMBER_PATH')
            return Path(hf_hub_download(repo_id=route['repo'],repo_type='dataset',revision=info.sha,
                filename=route['prefix']+'/'+name,token=token,cache_dir=cache))
        try:marker=json.loads(fetch('DELIVERY_COMPLETE.json').read_bytes())
        except EntryNotFoundError:
            relay.update(Path(c['root'])/'SHARED_DELIVERY_STATE.json',dict(status='WAITING_VANDA_CLOSED8192',epoch=time.time()))
            time.sleep(60);continue
        if (marker.get('status')!='COMPLETE_CLOSED8192_PRIVATE_DELIVERY'
                or marker.get('science_run_id')!=c['shared_science_run_id']):
            raise PermissionError('PREBOUND_EXACT_SCIENTIFIC_MODEL_DELIVERY')
        manifest=fetch('INFERENCE_PACKAGE.json')
        if sha(manifest)!=marker['manifest_sha256']:raise PermissionError('EXACT_TRANSFER_MANIFEST')
        package=json.loads(manifest.read_bytes());destination.mkdir(mode=0o700,exist_ok=False)
        for name,pin in package['files'].items():
            f=fetch(name)
            if sha(f)!=pin:raise PermissionError('PRIVATE_TRANSFER_MEMBER_CHANGED:'+name)
            out=destination/name
            if not out.resolve().is_relative_to(destination.resolve()):raise PermissionError('DELIVERY_PATH_ESCAPE')
            out.parent.mkdir(parents=True,exist_ok=True)
            with f.open('rb')as src,out.open('xb')as dst:shutil.copyfileobj(src,dst,1024**2)
        (destination/'INFERENCE_PACKAGE.json').write_bytes(manifest.read_bytes())
        validate(destination,manifest_sha256=marker['manifest_sha256'],checkpoint_manifest_sha256=marker['checkpoint_manifest_sha256'])
        relay.exclusive(destination/'DELIVERY_COMPLETE.json',dict(marker,HF_revision=info.sha))
        return dict(marker,status='COMPLETE_CORE_PRIVATE_SHARED8192_INPUT',effects_closed=True,HF_revision=info.sha)
    return dict(status='UNRUN_SHARED8192_NOT_DELIVERED_BEFORE_CUTOFF',effects_closed=True)
