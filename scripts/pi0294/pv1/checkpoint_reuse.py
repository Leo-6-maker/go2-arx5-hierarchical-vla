"""Plan a full-state cold continuation from a closed registered sibling only."""
import os
from pathlib import Path
import re

from scripts.pi0292.durable import read
from scripts.pi0294.e1.journal import process_is_same
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest
from .decision import AUTHORITY
from .recipe import run_key,validate_primary
from .training import checked_resume
from .neural_scope import ROLE

PLAN='plans/REUSED_CHECKPOINT_BINDING.private.json'


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


def validate(root,recipe,dataset):
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
            or campaign['optimizer_updates']<step or destination.exists()
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
