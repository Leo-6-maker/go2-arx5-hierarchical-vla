"""Independent approval, canonical-entry and immutable-input admission."""
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import sha
from scripts.pi0294.e1.scope import STATS
from .plan import CARD,CARD_SHA256,PLAN_RELEASE,protocol_card,NEAR_REVISION,AMENDMENT_CARD,AMENDMENT_CARD_SHA256

ROLES=dict(STATIC='4080-simulation',CAPTURE='4080-simulation',EXPORT='4080-cpu-data',PACK='4080-cpu-data')
SCOPE='PI0294_DATA_COV1_CPU_CHILD_V1'


def sealed_file(path,expected=None):
    p=Path(path)
    if (p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid()
            or p.stat().st_mode&0o777!=0o400):
        raise PermissionError('DATA_COV1_IMMUTABLE_0400_OWNED_REGULAR_FILE')
    actual=sha(p)
    if expected is not None and actual!=expected:
        raise ValueError('DATA_COV1_IMMUTABLE_MEMBER_HASH')
    return actual


def approval(adoption):
    card=protocol_card()
    if (adoption.get('status')!='DATA_COV1_NATIVE_ADOPTED'
            or adoption.get('protocol_release')!=PLAN_RELEASE
            or adoption.get('protocol_card_sha256')!=CARD_SHA256
            or adoption.get('approval_level') not in ('PILOT_ONLY','FULL_CONDITIONAL_CAMPAIGN')
            or adoption.get('owner')!='EXISTING_LOCAL_PC' or not adoption.get('approval_ref')
            or adoption.get('D0_lease_reused') is not False or adoption.get('PV1_budget_reassigned') is not False
            or adoption.get('approved_limits')!=card['stages'][0 if adoption['approval_level']=='PILOT_ONLY' else -1]['proposed_cumulative_limits']):
        raise PermissionError('DATA_COV1_EXPLICIT_GO_AND_INDEPENDENT_NATIVE_ADOPTION')
    t0=adoption.get('T0_epoch')
    if type(t0) not in (int,float) or not 0<t0<=time.time():
        raise PermissionError('DATA_COV1_ONE_VALID_ORIGINAL_T0')
    return adoption


def geometry_amendment(root,pool):
    if pool.get('geometry_revision') is None:return None
    if pool['geometry_revision']!=NEAR_REVISION:
        raise PermissionError('DATA_COV1_UNKNOWN_GEOMETRY_AMENDMENT')
    root=Path(root);parent=root.parent.parent;work=read(root/'plans/WORK_PLAN.private.json')
    proof_path=work.get('geometry_amendment_path')
    if not proof_path or not work.get('geometry_amendment_sha256'):
        raise PermissionError('DATA_COV1_AMENDED_POOL_REQUIRES_SEALED_EXPLICIT_GO')
    sealed_file(proof_path,work['geometry_amendment_sha256']);proof=read(proof_path)
    adoption=read(parent/'DATA_COV1_ADOPTION_AND_SCOPE.json')
    original=parent/'plans/DATA_COV1_CANDIDATE_POOL.private.json'
    m=read(root/'RUN_MANIFEST.json');code=root/m['code']['capsule_relative']
    sealed_file(code/AMENDMENT_CARD,AMENDMENT_CARD_SHA256)
    if (proof.get('schema')!='PI0294_DATA_COV1_NATIVE_GEOMETRY_AMENDMENT_V2'
            or proof.get('revision')!=NEAR_REVISION or not proof.get('approval_ref')
            or proof.get('original_adoption_sha256')!=sealed_file(parent/'DATA_COV1_ADOPTION_AND_SCOPE.json')
            or proof.get('original_T0_epoch')!=adoption['T0_epoch']
            or proof.get('original_pool_sha256')!=sealed_file(original)
            or pool.get('parent_pool_sha256')!=proof['original_pool_sha256']
            or pool.get('geometry_amendment_approval_ref')!=proof['approval_ref']
            or proof.get('public_amendment_card_sha256')!=AMENDMENT_CARD_SHA256
            or proof.get('budgets_and_stage_gates_unchanged') is not True
            or proof.get('physical_replacements')!=0):
        raise PermissionError('DATA_COV1_NARROW_AMENDMENT_NO_NEW_CLOCK_BUDGET_OR_RETRY')
    return proof


def validate(root,role):
    if role not in ROLES:raise PermissionError('DATA_COV1_VERSIONED_CPU_ROLE')
    root=Path(root)
    for name in ('RUN_MANIFEST.json','OUTPUT_BINDING.private.json','INPUT_BINDINGS.private.json'):
        sealed_file(root/name)
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json');inputs=read(root/'INPUT_BINDINGS.private.json')
    if (root.is_symlink() or root.resolve()!=Path(b['run_root']).resolve()
            or root.parent.resolve()!=Path(b['registered_parent']).resolve()
            or b['uid']!=os.getuid() or b['host']!=os.uname().nodename
            or m.get('schema')!=SCOPE or m.get('scope')!=SCOPE or m.get('role')!=role
            or m.get('owner')!='EXISTING_LOCAL_PC' or m.get('run_uuid')!=root.name
            or m['execution']['host']!=os.uname().nodename or m['execution']['uid']!=os.getuid()
            or m['execution']['profile']!=ROLES[role] or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or 'PBS_JOBID' in os.environ):
        raise PermissionError('DATA_COV1_REGISTERED_NATIVE_HOST_AND_CPU_ONLY_IDENTITY')
    parent=root.parent.parent;adoption=approval(read(parent/'DATA_COV1_ADOPTION_AND_SCOPE.json'))
    if (m['authority']['protocol_release']!=PLAN_RELEASE
            or m['authority']['adoption_sha256']!=sealed_file(parent/'DATA_COV1_ADOPTION_AND_SCOPE.json')
            or m['authority']['approval_ref']!=adoption['approval_ref']
            or m['limits']['T0_epoch']!=adoption['T0_epoch']):
        raise PermissionError('DATA_COV1_CHILD_EXACT_APPROVAL_AND_ORIGINAL_CLOCK')
    if adoption['approval_level']=='PILOT_ONLY' and m['stage']!='P24':
        raise PermissionError('DATA_COV1_PILOT_DOES_NOT_ADMIT_LATER_STAGES')
    config=next((s for s in protocol_card()['stages'] if s['id']==m['stage']),None)
    if config is None:raise PermissionError('DATA_COV1_FIXED_STAGE')
    ceiling=adoption['T0_epoch']+3600*config['proposed_cumulative_limits']['wall_hours_from_campaign_T0']
    lease=read(b['lease']);sealed_file(b['lease'])
    if (lease['status']!='ACTIVE' or lease['owner']!='EXISTING_LOCAL_PC' or lease['run_uuid']!=root.name
            or lease['deadline_epoch']!=m['limits']['operation_deadline_epoch']
            or not time.time()<m['limits']['operation_deadline_epoch']<=ceiling):
        raise PermissionError('DATA_COV1_ACTIVE_BOUNDED_CHILD_LEASE')
    registry=read(b['registry'])['entries'];matches=[r for r in registry if r.get('role')==b['role']]
    if (len(matches)!=1 or Path(matches[0]['path']).resolve()!=root.resolve()
            or matches[0]['run_uuid']!=root.name or matches[0].get('scope')!=SCOPE):
        raise PermissionError('DATA_COV1_UNIQUE_CANONICAL_REGISTRY_BINDING')
    code=root/m['code']['capsule_relative'];seal_path=root/m['code']['source_seal_relative']
    if (code.is_symlink() or not code.resolve().is_relative_to(root.resolve())
            or not seal_path.resolve().is_relative_to(root.resolve())
            or Path(__file__).resolve()!=(code/'scripts/pi0294/data_cov1/scope.py').resolve()
            or os.environ.get('DATA_COV1_RUN_MANIFEST')!=str(code.parent/'WORKER_RUN_MANIFEST.json')
            or read(code.parent/'WORKER_RUN_MANIFEST.json')!=m):
        raise PermissionError('DATA_COV1_ACTUAL_IMMUTABLE_CAPSULE_NOT_PV1_LAUNCHER')
    sealed_file(seal_path,m['code']['source_seal_sha256']);seal=read(seal_path)
    if seal['source_sha']!=m['code']['vla_commit']:
        raise ValueError('DATA_COV1_EXACT_SOURCE_RELEASE')
    for relative,expected in seal['files'].items():
        p=code/relative
        if Path(relative).is_absolute() or not p.resolve().is_relative_to(code.resolve()):
            raise PermissionError('DATA_COV1_SOURCE_MEMBER_ESCAPE')
        sealed_file(p,expected)
    sealed_file(code/CARD,CARD_SHA256)
    for filename,key in [('OUTPUT_BINDING.private.json','output_binding_sha256'),('INPUT_BINDINGS.private.json','input_binding_sha256')]:
        sealed_file(root/filename,m['execution'][key])
    for relative,expected in m['code']['frozen_numerical_files'].items():sealed_file(code/relative,expected)
    if len(m['code']['frozen_numerical_files'])!=15:raise ValueError('DATA_COV1_ORIGINAL_15_NUMERICAL_ASSETS')
    asset_file=root/'plans/ASSET_IDENTITY.private.json';sealed_file(asset_file,m['data']['asset_identity_sha256'])
    assets=read(asset_file)['absolute_files']
    if len(assets)!=26 or any(sha(path)!=h for path,h in assets.items()):
        raise ValueError('DATA_COV1_FROZEN_NATIVE_ASSETS')
    infra=Path(b['registry']).parent
    for name,key in [('enter.sh','entry_sha256'),('doctor.py','doctor_sha256')]:
        if sha(infra/name)!=m['execution'][key]:raise ValueError('DATA_COV1_EXISTING_ENTRY_IDENTITY')
    sealed_file(root/'plans/WORK_PLAN.private.json',m['data']['plan_sha256'])
    proof_path=root/'plans/SOURCE_PUBLICATION_AND_CI.json'
    sealed_file(proof_path,m['code']['publication_and_CI_sha256']);proof=read(proof_path)
    if (proof['source_sha']!=m['code']['vla_commit'] or proof['CI_head_sha']!=m['code']['vla_commit']
            or proof['CI_conclusion']!='success' or proof['CI_status']!='completed'
            or not proof['files_exact_readback']):
        raise PermissionError('DATA_COV1_EXACT_PUBLISHED_SOURCE_CI_BEFORE_NATIVE_DISPATCH')
    sealed_file(root/'plans/statistics.json',STATS)
    if m['interfaces']['statistics_sha256']!=STATS or m['interfaces']['action_schema']!='ORIGINAL_EEF10_H40_PREFIX8':
        raise PermissionError('DATA_COV1_ORIGINAL_CODEC_AND_ACTION_SCHEMA')
    if set(m['limits']['cpus'])&set(adoption['PV1_reserved_CPUs']) or len(m['limits']['cpus'])!=1:
        raise PermissionError('DATA_COV1_SINGLE_CORE_DISJOINT_FROM_PV1')
    if (not set(m['limits']['cpus'])<=os.sched_getaffinity(0)
            or type(m['limits']['CPU_s']) is not int or not 30<m['limits']['CPU_s']<=3600*config['proposed_cumulative_limits']['CoreLab_CPU_core_hours']):
        raise PermissionError('DATA_COV1_ACTUAL_ALLOWED_CPU_AND_FINITE_RESERVATION')
    if role in ('STATIC','CAPTURE'):
        if time.time()>=adoption['T0_epoch']+3600*config['proposed_cumulative_limits']['stop_new_attempt_hours_from_T0']:
            raise PermissionError('DATA_COV1_NO_NEW_ATTEMPTS_IN_PROTECTED_CLOSEOUT')
        if m['limits']['operation_deadline_epoch']>adoption['T0_epoch']+3600*config['proposed_cumulative_limits']['stop_physics_hours_from_T0']:
            raise PermissionError('DATA_COV1_NATIVE_WORK_MUST_STOP_BEFORE_CLOSEOUT')
    if role in ('EXPORT','PACK') and any(m['limits'].get(k)!=0 for k in ('engine_starts','physics_ticks','Teacher_episodes')):
        raise PermissionError('DATA_COV1_SAVED_DATA_WORK_FORBIDS_NEW_CONSTRUCTORS_OR_PHYSICS')
    if any((p/'STOP.json').exists() for p in (root,parent)):
        raise PermissionError('DATA_COV1_STOP_PRESERVED')
    if any(m['limits'].get(k)!=0 for k in ('GR00T_queries','training_updates','PBS_submissions')):
        raise PermissionError('DATA_COV1_NO_NEURAL_TRAINING_OR_PBS')
    return m,inputs


def dispatch_priority(root):
    root=Path(root);parent=root.parent.parent;state=read(parent/'PV1_PRIORITY_STATE.private.json')
    if (state['owner']!='EXISTING_LOCAL_PC' or not state.get('job_id')
            or not 0<=time.time()-state['observed_epoch']<=180
            or state['native_job_state'] not in ('Q','NO_ACTIVE_PV1_GPU_ALLOCATION')
            or state['allow_future_data_dispatch'] is not True):
        raise PermissionError('DATA_COV1_WAIT_PV1_PRIORITY_OR_STALE_NATIVE_READ')
    if (Path(state['PV1_native_root'])/'STOP.json').exists():
        raise PermissionError('DATA_COV1_PV1_SAFETY_STOP')
    return state
