"""PV1-only cold neural-root admission; this is not permission to load a model."""
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.e1.scope import DATA,STATS
from scripts.pi0294.fd1 import sha
from .dataset import POPULATION,admit_new_support
from .decision import AUTHORITY
from .recipe import PARENT
from .scope import SCOPE,PROTOCOL

ROLE='PI0294_PV1_SHARED_POLICY_HOPPER_V1'


def inside(root,name):
    root,name=Path(root),Path(name)
    p=root/name
    if name.is_absolute() or '..' in name.parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_NEURAL_REGISTERED_RELATIVE_INPUT_ONLY')
    return p


def validate(root,*,closeout=False):
    root=Path(root);m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json')
    if (root.is_symlink() or root.stat().st_uid!=os.getuid()
            or m['schema']!='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1' or m['scope']!=ROLE
            or m['owner']!='EXISTING_LOCAL_PC' or m['run_uuid']!=root.name
            or m['execution']['profile']!='hopper-neural' or m['execution']['uid']!=os.getuid()
            or b['uid']!=os.getuid() or b['role']!=ROLE+'__'+root.name
            or m['execution']['private_path_role']!=b['role']
            or Path(b['run_root']).resolve()!=root.resolve()
            or Path(b['registered_parent']).resolve()!=root.parent.resolve()
            or sha(root/'OUTPUT_BINDING.private.json')!=m['execution']['output_binding_sha256']):
        raise PermissionError('PV1_COLD_HOPPER_CHILD_UID_OWNER_PROFILE_AND_OUTPUT')
    code=inside(root,m['code']['capsule_relative']);seal=inside(root,m['code']['source_seal_relative'])
    worker=code.parent/'WORKER_RUN_MANIFEST.json'
    if (os.environ.get('PV1_RUN_MANIFEST')!=str(worker) or worker.stat().st_mode&0o222
            or read(worker)!=m or Path(__file__).resolve()!=(code/'scripts/pi0294/pv1/neural_scope.py').resolve()
            or sha(seal)!=m['code']['source_seal_sha256']):
        raise PermissionError('PV1_ACTUAL_IMPORTED_IMMUTABLE_NEURAL_CAPSULE')
    parent=root.parent.parent;a=read(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
    s=read(code/SCOPE);stage_path=root/'PV1_STAGE_BINDING.json';stage=read(stage_path)
    if (a['authority_release']!=AUTHORITY or a['bounds']!=s['new_campaign_budget']
            or any(a[key]!=a['T0_epoch']+hours*3600 for key,hours in
                [('stop_new_jobs_epoch',144),('stop_new_training_epoch',156),
                 ('stop_new_physics_epoch',160),('hard_closeout_epoch',168)])
            or m['authority']['protocol_ref']!=AUTHORITY or m['limits']['PV1_T0_epoch']!=a['T0_epoch']
            or m['limits']['PV1_hard_close_epoch']!=a['hard_closeout_epoch']
            or sha(code/SCOPE)!=m['authority']['scope_sha256'] or sha(code/PROTOCOL)!=m['authority']['protocol_sha256']
            or stage['schema']!='PI0294_PV1_ADMITTED_SHARED_POLICY_STAGE_V1'
            or stage['authority_release']!=AUTHORITY or stage['campaign_run_uuid']!=a['run_uuid']
            or stage['population_plan_sha256']!=POPULATION or stage['Teacher_takeover'] is not False
            or stage['normalizer_refit'] is not False or stage['evaluation_data_for_training'] is not False):
        raise PermissionError('PV1_EXACT_AUTHORITY_ORIGINAL_WINDOW_AND_FROZEN_STAGE')
    lease=read(b['lease']);pb=read(parent/'OUTPUT_BINDING.private.json');pl=read(pb['lease'])
    now=time.time();cutoff=a['hard_closeout_epoch'] if closeout else a['stop_new_physics_epoch']
    if (not a['T0_epoch']<=now<cutoff or lease['status']!='ACTIVE'
            or lease['run_uuid']!=root.name or lease['owner']!='EXISTING_LOCAL_PC'
            or lease['authority_release']!=AUTHORITY
            or lease['deadline_epoch']!=m['limits']['operation_deadline_epoch']
            or not now<lease['deadline_epoch']<=a['hard_closeout_epoch']
            or pl['status']!='ACTIVE_CONTROL_PLANE_ONLY' or pl['authority_release']!=AUTHORITY
            or pl['run_uuid']!=a['run_uuid'] or pl['owner']!='EXISTING_LOCAL_PC'
            or pl['T0_epoch']!=a['T0_epoch'] or pl['deadline_epoch']!=a['hard_closeout_epoch']
            or pb['run_uuid']!=a['run_uuid']):
        raise PermissionError('PV1_NEURAL_CHILD_AND_ORIGINAL_PARENT_LEASE')
    registry=read(b['registry'])['entries'];match=[e for e in registry if e.get('role')==b['role']]
    if len(match)!=1 or Path(match[0]['path']).resolve()!=root.resolve() or match[0]['run_uuid']!=root.name:
        raise PermissionError('PV1_ONE_CANONICAL_HOPPER_ROLE_PER_REGISTERED_CHILD')
    if not os.environ.get('PBS_JOBID'):
        if (os.uname().nodename not in m['execution']['control_hostnames']
                or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
            raise PermissionError('PV1_LOGIN_CONTROL_IS_MODEL_FREE_AND_CUDA_HIDDEN')
    members=read(seal)
    if members['source_sha']!=m['code']['vla_commit']:
        raise ValueError('PV1_NEURAL_FIXED_SOURCE_SHA')
    for name,digest in members['files'].items():
        if sha(inside(code,name))!=digest:raise ValueError('PV1_NEURAL_SOURCE_MEMBER_CHANGED:'+name)
    frozen=m['code']['frozen_numerical_files']
    if len(frozen)!=15 or any(members['files'].get(n)!=h or sha(code/n)!=h for n,h in frozen.items()):
        raise ValueError('PV1_ORIGINAL_15_NUMERICAL_FILES_REQUIRED')
    for name,digest in m['execution']['sealed_inputs'].items():
        if sha(inside(root,name))!=digest:raise ValueError('PV1_NEURAL_SEALED_INPUT_CHANGED:'+name)
    bindings=read(root/'NEURAL_BINDINGS.private.json');inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    original=Path(bindings['dataset']);new=Path(inputs['new_dataset']);correction=Path(inputs['C1_dataset'])
    if (m['code']['official_gr00t_commit']!='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
            or sha(original/'DATASET_MANIFEST.json')!=DATA or sha(original/'statistics.json')!=STATS
            or sha(new/'DATASET_MANIFEST.json')!=stage['new_bank_manifest_sha256']
            or sha(new/'statistics.json')!=STATS or sha(new/'READY.json')!=stage['new_bank_ready_sha256']
            or read(new/'READY.json')['status']!='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED'
            or read(new/'READY.json')['dataset_manifest_sha256']!=stage['new_bank_manifest_sha256']
            or sha(correction/'DATASET_MANIFEST.json')!=stage['C1_manifest_sha256']
            or sha(correction/'statistics.json')!=STATS
            or sha(Path(inputs['CORR_parent_checkpoint'])/'CHECKPOINT_COMPLETE.json')!=PARENT):
        raise PermissionError('PV1_ORIGINAL_DATA_CORR_PARENT_C1_AND_ACTUALLY_ADMITTED_NEW_BANK')
    admit_new_support(read(new/'DATASET_MANIFEST.json'),stage['route'])
    for p in [root/'STOP.json',parent/'STOP.json',root/'SOURCE_PHYSICS_STOP.json',root/'SOURCE_SAFETY_STOP_FROM_CORE.json']:
        if p.exists():raise PermissionError('PV1_PARENT_OR_SOURCE_SAFETY_STOP_NO_NEW_NEURAL_WORK')
    return m,bindings,inputs,stage,a
