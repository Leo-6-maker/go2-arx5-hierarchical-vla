"""Validate and atomically copy full-state cold continuations between siblings."""
import hashlib
import os
from pathlib import Path
import re

from scripts.pi0292.durable import read,write
from scripts.pi0294.e1.journal import process_is_same
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest
from .decision import AUTHORITY
from .recipe import run_key,validate_primary
from .training import checked_resume
from .neural_scope import ROLE

PLAN='plans/REUSED_CHECKPOINT_BINDING.private.json'
INTENT='evidence/CHECKPOINT_REUSE_COPY_INTENT.private.json'
RECEIPT='evidence/CHECKPOINT_REUSE_COPY_RECEIPT.private.json'


def member(root,name):
    root=Path(root);relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise PermissionError('PV1_CHECKPOINT_MEMBER_MUST_STAY_IN_OWN_ROOT')
    p=root/relative
    if not p.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_CHECKPOINT_MEMBER_ESCAPE')
    if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
        raise PermissionError('PV1_CHECKPOINT_REUSE_NO_SYMLINK_COMPONENTS')
    return p


def sealed(root,manifest,name):
    p=member(root,name)
    if (not p.is_file() or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o222
            or manifest['execution']['sealed_inputs'].get(name)!=sha(p)):
        raise PermissionError('PV1_CHECKPOINT_REUSE_REQUIRES_PRESEALED_INPUT:'+name)
    return read(p)


def validate(root,recipe,dataset,*,allow_existing_destination=False):
    """Return a copy plan; do not copy, submit, change counters or reset cursors."""
    root=Path(root)
    if root.is_symlink() or not root.is_absolute() or root.resolve()!=root or root.parent.name!='children':
        raise PermissionError('PV1_CHECKPOINT_DESTINATION_REGISTERED_CHILD_ROOT_ONLY')
    parent=root.parent.parent;m=read(member(root,'RUN_MANIFEST.json'))
    plan=sealed(root,m,PLAN);source=Path(plan['source_child_root'])
    if (plan['schema']!='PI0294_PV1_CLOSED_SIBLING_CHECKPOINT_REUSE_V1'
            or source==root or not source.is_absolute() or source.resolve()!=source
            or source.parent!=root.parent or source.is_symlink() or not source.is_dir()
            or source.stat().st_uid!=os.getuid() or root.stat().st_uid!=os.getuid()):
        raise PermissionError('PV1_CHECKPOINT_REUSE_SAME_CAMPAIGN_OWN_CLOSED_SIBLING_ONLY')
    adoption=read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    source_manifest=member(source,'RUN_MANIFEST.json');sm=read(source_manifest)
    if (adoption['authority_release']!=AUTHORITY or plan['authority_release']!=AUTHORITY
            or plan['campaign_run_uuid']!=adoption['run_uuid']
            or sha(source_manifest)!=plan['source_manifest_sha256']
            or sm['authority']['protocol_ref']!=AUTHORITY or m['authority']['protocol_ref']!=AUTHORITY
            or sm['run_uuid']!=source.name or m['run_uuid']!=root.name
            or sm['code']['vla_commit']!=plan['source_code_sha']
            or not re.fullmatch('[0-9a-f]{40}',plan['source_code_sha'])
            or not re.fullmatch('[0-9a-f]{40}',m['code']['vla_commit'])):
        raise PermissionError('PV1_CHECKPOINT_REUSE_FIXED_AUTHORITY_AND_SOURCE_IDENTITY')
    parent_binding=read(parent/'OUTPUT_BINDING.private.json')
    registry_path=Path(parent_binding['registry'])
    if registry_path.is_symlink() or registry_path.stat().st_uid!=os.getuid():
        raise PermissionError('PV1_CHECKPOINT_REUSE_OWN_CANONICAL_REGISTRY_ONLY')
    entries=read(registry_path)['entries']
    for child,manifest in ((source,sm),(root,m)):
        binding_path=member(child,'OUTPUT_BINDING.private.json');binding=read(binding_path)
        if (sha(binding_path)!=manifest['execution']['output_binding_sha256']
                or binding['run_uuid']!=child.name or Path(binding['run_root'])!=child
                or binding['uid']!=os.getuid() or Path(binding['registry'])!=registry_path
                or binding.get('role')!=ROLE+'__'+child.name or manifest.get('scope')!=ROLE):
            raise PermissionError('PV1_CHECKPOINT_REUSE_REGISTERED_OUTPUT_BINDINGS_ONLY')
        matches=[e for e in entries if e.get('run_uuid')==child.name or e.get('role')==binding['role']]
        if (len(matches)!=1 or matches[0].get('run_uuid')!=child.name
                or matches[0].get('role')!=binding['role'] or matches[0].get('scope')!=ROLE
                or Path(matches[0]['path'])!=child):
            raise PermissionError('PV1_CHECKPOINT_REUSE_ONE_CANONICAL_REGISTERED_SIBLING')
        if child==source:
            lease_path=Path(binding['lease'])
            if lease_path.is_symlink() or lease_path.stat().st_uid!=os.getuid():
                raise PermissionError('PV1_CHECKPOINT_REUSE_OWN_SOURCE_LEASE_ONLY')
            lease=read(lease_path)
            if (lease['status']!='CLOSED' or lease['authority_release']!=AUTHORITY
                    or lease['run_uuid']!=source.name or lease['all_registered_workers_terminal'] is not True):
                raise PermissionError('PV1_CHECKPOINT_REUSE_SOURCE_NOT_CLOSED')
    key=run_key(recipe)
    original=sealed(root,m,'plans/FROZEN_SOURCE_PAIR_RECIPE.json');validate_primary(recipe,original)
    for child,manifest in ((source,sm),(root,m)):
        recipes=sealed(child,manifest,'plans/PRIMARY_LEARNING_RECIPES.json')
        if recipes['recipes'].get(key)!=recipe:
            raise PermissionError('PV1_CHECKPOINT_REUSE_EXACT_FIXED_LOCAL_RECIPE_ONLY')
    campaign=read(parent/'HOPPER_RESOURCE_LEDGER.json')
    jobs=[v for v in campaign['PBS_submissions'] if v['run_uuid']==source.name]
    if (campaign['campaign_run_uuid']!=adoption['run_uuid'] or campaign['authority_release']!=AUTHORITY
            or campaign['unknown_effects'] or campaign['pending_updates']
            or not jobs or any(v['status']!='NATIVE_TERMINAL' for v in jobs)):
        raise PermissionError('PV1_CHECKPOINT_REUSE_TERMINAL_NATIVE_JOBS_NO_UNKNOWN_EFFECTS')
    for folder in ('worker_processes','aux_worker_processes'):
        directory=member(source,folder)
        if not directory.is_dir() or directory.stat().st_uid!=os.getuid():
            raise PermissionError('PV1_CHECKPOINT_REUSE_COMPLETE_REGISTERED_PROCESS_INVENTORY')
        for path in directory.glob('*.json'):
            if path.is_symlink() or process_is_same(read(path)):
                raise PermissionError('PV1_CHECKPOINT_REUSE_REGISTERED_PROCESS_STILL_ACTIVE_OR_UNBOUND')
    checkpoint=member(source,plan['source_checkpoint_relative'])
    complete_path=member(checkpoint,'CHECKPOINT_COMPLETE.json');complete=read(complete_path)
    step=complete['global_step'];relative=Path('training')/key/('checkpoint-'+str(step))
    destination=member(root,relative)
    if (type(step) is not int or not 0<step<4096 or complete['full_resume'] is not True
            or Path(plan['source_checkpoint_relative'])!=relative
            or plan['destination_checkpoint_relative']!=relative.as_posix()
            or plan['run_key']!=key or plan['global_step']!=step
            or plan['recipe_sha256']!=digest(recipe) or plan['statistics_sha256']!=STATS
            or plan['dataset_manifest_sha256']!=dataset.manifest_sha
            or sha(complete_path)!=plan['source_checkpoint_manifest_sha256']
            or complete['origin_source_sha']!=plan['source_code_sha']
            or not any(v.get('job_id')==complete['origin_job_id'] and v.get('training_run_key')==key for v in jobs)
            or campaign.get('training_runs',{}).get(key)!=dict(recipe_sha256=digest(recipe),last_native_step=step)
            or campaign['optimizer_updates']<step
            or (destination.exists() and not allow_existing_destination)
            or plan['copy_full_resume_state'] is not True or plan['additional_optimizer_updates']!=0):
        raise PermissionError('PV1_CHECKPOINT_REUSE_EXACT_ALREADY_CHARGED_STEP_AND_ABSENT_DESTINATION')
    for name in complete['files']:
        p=member(checkpoint,name)
        if not p.is_file() or p.stat().st_uid!=os.getuid() or name=='CHECKPOINT_COMPLETE.json':
            raise PermissionError('PV1_CHECKPOINT_REUSE_ONLY_OWN_REGULAR_COMPLETE_MEMBERS')
    if checked_resume(source,key,checkpoint,recipe,dataset)!=step:
        raise PermissionError('PV1_CHECKPOINT_REUSE_NATIVE_RESUME_STEP_MISMATCH')
    files=dict(complete['files'],**{'CHECKPOINT_COMPLETE.json':sha(complete_path)})
    return dict(status='PASS_CLOSED_SIBLING_FULL_RESUME_COPY_PLAN_NOT_COPIED',
        binding_sha256=sha(root/PLAN),source_child_uuid=source.name,source_code_sha=plan['source_code_sha'],
        destination_code_sha=m['code']['vla_commit'],training_run_key=key,global_step=step,
        resume_sample_cursor=step*32,optimizer_updates_already_charged=step,additional_optimizer_updates=0,
        source_checkpoint=str(checkpoint),destination_checkpoint=str(destination),
        checkpoint_manifest_sha256=sha(complete_path),files=files,
        copy_logical_bytes=sum((checkpoint/name).stat().st_size for name in files),
        actual_copy_executed=False,current_destination_full_hash_verification_still_required=True,
        current_native_permissions_and_resource_admission_still_required=True)


def _actual_files(checkpoint):
    checkpoint=Path(checkpoint);names=[]
    for p in checkpoint.rglob('*'):
        if p.is_symlink():
            raise PermissionError('PV1_CHECKPOINT_COPY_NO_SYMLINK_OUTPUT')
        if p.is_file():names.append(p.relative_to(checkpoint).as_posix())
        elif not p.is_dir():raise PermissionError('PV1_CHECKPOINT_COPY_REGULAR_FILES_ONLY')
    return set(names)


def _verify(checkpoint,files):
    checkpoint=Path(checkpoint)
    if (checkpoint.is_symlink() or not checkpoint.is_dir()
            or _actual_files(checkpoint)!=set(files)):
        raise PermissionError('PV1_CHECKPOINT_COPY_EXACT_MEMBER_SET_REQUIRED')
    for name,expected in files.items():
        p=member(checkpoint,name)
        if not p.is_file() or p.is_symlink() or p.stat().st_uid!=os.getuid() or sha(p)!=expected:
            raise PermissionError('PV1_CHECKPOINT_COPY_MEMBER_READBACK:'+name)


def _copy_one(source,destination,expected):
    destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if destination.exists():
        if (destination.is_symlink() or not destination.is_file()
                or destination.stat().st_uid!=os.getuid() or sha(destination)!=expected):
            raise PermissionError('PV1_CHECKPOINT_PARTIAL_COPY_CHANGED:'+destination.name)
        return False
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    if hasattr(os,'O_NOFOLLOW'):flags|=os.O_NOFOLLOW
    source_flags=os.O_RDONLY|(os.O_NOFOLLOW if hasattr(os,'O_NOFOLLOW') else 0)
    source_fd=os.open(source,source_flags);destination_fd=None
    try:
        destination_fd=os.open(destination,flags,0o400)
        digest_=hashlib.sha256()
        while True:
            chunk=os.read(source_fd,8*1024*1024)
            if not chunk:break
            digest_.update(chunk)
            view=memoryview(chunk)
            while view:
                written=os.write(destination_fd,view)
                if written<=0:raise OSError('PV1_CHECKPOINT_COPY_SHORT_WRITE')
                view=view[written:]
        os.fsync(destination_fd)
        if digest_.hexdigest()!=expected:
            raise ValueError('PV1_CHECKPOINT_SOURCE_CHANGED_DURING_COPY:'+source.name)
    except BaseException:
        if destination_fd is not None:os.close(destination_fd);destination_fd=None
        if destination.exists() and not destination.is_symlink():destination.unlink()
        raise
    finally:
        os.close(source_fd)
        if destination_fd is not None:os.close(destination_fd)
    return True


def execute(root,recipe,dataset):
    """Copy once without changing optimizer/update accounting or source bytes."""
    root=Path(root);result=validate(root,recipe,dataset,allow_existing_destination=True)
    source=Path(result['source_checkpoint']);destination=Path(result['destination_checkpoint'])
    receipt_path=member(root,RECEIPT);intent_path=member(root,INTENT)
    if receipt_path.exists():
        receipt=read(receipt_path)
        if (receipt.get('status')!='PASS_FULL_STATE_CHECKPOINT_COPIED_AND_REHASHED'
                or receipt.get('binding_sha256')!=result['binding_sha256']
                or receipt.get('checkpoint_manifest_sha256')!=result['checkpoint_manifest_sha256']):
            raise PermissionError('PV1_CHECKPOINT_COPY_RECEIPT_IDENTITY')
        _verify(destination,result['files'])
        if checked_resume(root,result['training_run_key'],destination,recipe,dataset)!=result['global_step']:
            raise PermissionError('PV1_CHECKPOINT_COPIED_RESUME_READBACK')
        return receipt
    intent=dict(schema='PI0294_PV1_CHECKPOINT_REUSE_COPY_INTENT_V1',authority_release=AUTHORITY,
        binding_sha256=result['binding_sha256'],source_child_uuid=result['source_child_uuid'],
        source_checkpoint=result['source_checkpoint'],destination_checkpoint=result['destination_checkpoint'],
        checkpoint_manifest_sha256=result['checkpoint_manifest_sha256'],files=result['files'],
        optimizer_updates_already_charged=result['optimizer_updates_already_charged'],
        additional_optimizer_updates=0)
    intent_path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if intent_path.exists():
        if read(intent_path)!=intent:raise PermissionError('PV1_CHECKPOINT_COPY_INTENT_IDENTITY')
    else:write(intent_path,intent);intent_path.chmod(0o400)
    destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    staging=destination.with_name('.'+destination.name+'.copying-'+result['binding_sha256'][:16])
    copied=0;resumed_partial=staging.exists()
    if destination.exists():
        if staging.exists():raise PermissionError('PV1_CHECKPOINT_DESTINATION_AND_STAGING_BOTH_EXIST')
        recovered_existing=True
    else:
        recovered_existing=False
        staging.mkdir(mode=0o700,parents=False,exist_ok=True)
        unexpected=_actual_files(staging)-set(result['files'])
        if unexpected:raise PermissionError('PV1_CHECKPOINT_PARTIAL_COPY_UNEXPECTED_MEMBERS')
        for name in sorted(result['files'],key=lambda n:(n=='CHECKPOINT_COMPLETE.json',n)):
            copied+=_copy_one(member(source,name),member(staging,name),result['files'][name])
        _verify(staging,result['files'])
        os.replace(staging,destination)
        directory_fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    _verify(destination,result['files'])
    step=checked_resume(root,result['training_run_key'],destination,recipe,dataset)
    if step!=result['global_step']:
        raise PermissionError('PV1_CHECKPOINT_COPIED_RESUME_READBACK')
    receipt=dict(schema='PI0294_PV1_CHECKPOINT_REUSE_COPY_RECEIPT_V1',
        status='PASS_FULL_STATE_CHECKPOINT_COPIED_AND_REHASHED',authority_release=AUTHORITY,
        binding_sha256=result['binding_sha256'],source_child_uuid=result['source_child_uuid'],
        training_run_key=result['training_run_key'],global_step=step,
        resume_sample_cursor=result['resume_sample_cursor'],checkpoint_manifest_sha256=result['checkpoint_manifest_sha256'],
        files=result['files'],logical_bytes=result['copy_logical_bytes'],members_copied_this_call=copied,
        resumed_partial_staging=resumed_partial,recovered_existing_destination=recovered_existing,
        source_unchanged=True,additional_optimizer_updates=0,ledger_mutations=0,
        current_native_permissions_and_resource_admission_still_required=True)
    write(receipt_path,receipt);receipt_path.chmod(0o400)
    return receipt
