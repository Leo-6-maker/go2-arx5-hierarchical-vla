"""N1's single confirmation child delegates remaining resources, not a new clock."""
from pathlib import Path
import re

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import confirmation_plan as plan,n1_limits as n1
from .n1_boundary import native_history

SCHEMA=plan.SCHEMA
MODE='MP1_FIXED_SOURCE_CONFIRM_V1'
SCOPE='MP1_SINGLE_SOURCE_CONFIRMATION_NO_TRAINING'
CAPS=dict(main_starts=160,recovery_forks=0,live_and_recovery_integrations=1280000,
    shadow_integrations=1280000,generation_queries=6400,optimizer_updates=0)
CPU_CAPS=dict(core=144000,hopper=115200)
CARD='plans/SOURCE_CONFIRM_SCOPE.json'
CARD_SHA='00881d04a20fd033c80bbd2cad9323c608e86808f59ba5524e28034ba2f78b3e'
AUTHORITY_CARD='plans/CONFIRM_N1_AUTHORITY_CARD.json'
PRIOR='plans/CONFIRM_PRIOR_SOURCE.private.json'
PRIOR_HOPPER='plans/CONFIRM_PRIOR_HOPPER_LEDGER.json'
PRIOR_NATIVE='plans/CONFIRM_PRIOR_SOURCE_NATIVE_TERMINALS.json'
PLAN='plans/SOURCE_EVALUATION_PLAN.json'
MATRIX='plans/CONFIRM_DEVELOPMENT_MATRIX.json'
STATIC_INPUTS='plans/CONFIRM_SCENE_INPUTS.private.json'
CANDIDATE_INPUTS='plans/CONFIRM_CANDIDATE_BINDINGS.private.json'
STATIC_DELIVERY='plans/CONFIRM_STATIC_HF_BINDING.json'
FILES=[CARD,AUTHORITY_CARD,PRIOR,PRIOR_HOPPER,PRIOR_NATIVE,MATRIX,STATIC_INPUTS,CANDIDATE_INPUTS]
BINDING=dict(release=n1.RELEASE,adoption_delta_sha256=n1.ADOPTED_DELTA_SHA)


def is_confirmation(stage):return stage.get('schema')==SCHEMA


def checked_prior(past,native):
    native_history(past,native,11,dict(generation_queries=12000,GPU_s=129600))
    if (past['GPU_allocated_s']!=42833 or len(past['PBS_submissions'])!=10
            or past['generation_queries']!=2406 or past['optimizer_updates']!=4096
            or past['GPU_reserved_s'] or past['pending_updates']):
        raise PermissionError('ONLY_ACTUAL_CLOSED_SOURCE10_NI16_BASELINE_CAN_DELEGATE_CONFIRMATION')


def authority(root,manifest=None):
    root=Path(root);parent=root.parent.parent;m=read(root/'RUN_MANIFEST.json') if manifest is None else manifest
    stage=read(root/'MP1_STAGE_BINDING.json')
    if (root.name!=plan.CHILD or parent.name!=n1.PARENT_RUN or root.is_symlink()
            or m['run_uuid']!=root.name or m['execution'].get('confirmation_n1_binding')!=BINDING
            or not is_confirmation(stage) or m['scope']!=SCOPE
            or stage['models']!=[plan.CANDIDATE] or stage['updates_per_branch']!=0
            or stage['stage_caps']!=CAPS or stage['CPU_core_s']!=CPU_CAPS
            or 'n1_authority_binding' in m['execution'] or 'cold_engineering_binding_relative' in m['execution']):
        raise PermissionError('EXACT_SEPARATE_CONFIRMATION_CHILD_NOT_SOURCE_PAIR_REPURPOSING')
    if m['execution'].get('confirmation_phase') not in ['STATIC_PREPARATION','POLICY_FROZEN']:
        raise PermissionError('EXPLICIT_STATIC_PREPARATION_OR_FROZEN_POLICY_PHASE')
    first=parent/n1.FIRST_ADOPTION
    if first.is_symlink() or first.stat().st_mode&0o222 or sha(first)!=n1.ADOPTED_DELTA_SHA:
        raise PermissionError('ORIGINAL_SINGLE_N1_ADOPTION_AND_CLOCK_REQUIRED')
    sealed=m['execution']['sealed_inputs']
    for name in FILES:
        p=root/name
        if (p.is_symlink() or p.stat().st_mode&0o222 or not p.resolve().is_relative_to(root.resolve())
                or sealed.get(name)!=sha(p)):
            raise PermissionError('SEALED_CONFIRMATION_AUTHORITY_PRIORS_AND_CANDIDATE')
    if sha(root/CARD)!=CARD_SHA or sha(root/AUTHORITY_CARD)!=n1.BUNDLE[list(n1.BUNDLE)[2]]:
        raise PermissionError('EXACT_PUBLISHED_CONFIRM_SCOPE_AND_ORIGINAL_N1_CARD')
    result=n1.effective(read(parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'),read(root/AUTHORITY_CARD),read(first))
    checked_prior(read(root/PRIOR_HOPPER),read(root/PRIOR_NATIVE))
    q=read(root/PRIOR);source=parent/'children/source-pair-01'
    required={'RUN_MANIFEST.json','MP1_STAGE_BINDING.json',
        'HOPPER_RESOURCE_LEDGER.json' if q.get('role')=='hopper' else 'RESOURCE_LEDGER.json'}
    if (q['source_child']!='source-pair-01' or q['prior_H200_s']!=42833 or q['prior_PBS']!=10
            or q['prior_queries']!=2406 or q['prior_formal_updates']!=4096
            or not q['all_recorded_source_processes_terminal'] or q['unknown_effects']
            or q.get('role') not in ['core','hopper'] or not required<=set(q['files'])
            or Path(q['source_root']).resolve()!=source.resolve() or source.is_symlink()
            or (source/'SOURCE_PHYSICS_STOP.json').exists() or (source/'SOURCE_SAFETY_STOP_FROM_CORE.json').exists()):
        raise PermissionError('ALL_PRIOR_SOURCE_EFFECTS_CLOSED_WITH_FROZEN_OWNED_ROOT')
    for name,checksum in q['files'].items():
        p=source/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                or not p.resolve().is_relative_to(source.resolve()) or sha(p)!=checksum):
            raise PermissionError('PRIOR_SOURCE_COUNTERS_MANIFEST_AND_STOPS_IMMUTABLE')
    actual_stops={p.name for p in source.glob('RELAY_ENGINEERING_STOP*.json')}
    if actual_stops!=set(q['original_engineering_STOP_hashes']):
        raise PermissionError('ALL_OLD_SOURCE_ENGINEERING_STOPS_RETAINED')
    for name,checksum in q['original_engineering_STOP_hashes'].items():
        if sha(source/name)!=checksum:raise PermissionError('OLD_SOURCE_STOP_BYTES_CHANGED')
    original=read(source/'RUN_MANIFEST.json');seal=read(root/m['code']['source_seal_relative'])
    for key,count in [('unchanged_numerical_files',15),('unchanged_learning_files',12)]:
        frozen=original['code'][key]
        if (len(frozen)!=count or m['code'].get(key)!=frozen
                or any(seal['files'].get(name)!=checksum for name,checksum in frozen.items())):
            raise PermissionError('CONFIRMATION_PRESERVES_ALL_ORIGINAL_NUMERICAL_AND_LEARNING_FILE_HASHES')
    selected=plan.rank_candidate(read(root/MATRIX))
    if q['candidate']!=selected or selected['candidate']!=plan.CANDIDATE:
        raise PermissionError('ONE_ACTUAL_PREDECLARED_RANKED_ENDPOINT_CANDIDATE')
    result['confirmation']=dict(GPU_s=57600,PBS=2,prior_GPU_s=42833,prior_PBS=10,prior_queries=2406,
        generation_queries=6400,job_wall_s_max=28800,output_bytes=8*1024**3)
    if 'confirmation_engineering_binding_relative' in m['execution']:
        from .confirmation_boundary import validate
        validate(root,m)
    return result


def limits(root):
    bound=authority(root)
    return dict(mode=MODE,GPU_s=bound['confirmation']['GPU_s'],PBS=2,output_bytes=8*1024**3,job_wall_s_max=28800)


def validate_plan(root):
    root=Path(root);m=read(root/'RUN_MANIFEST.json');p=root/PLAN
    if (m['execution'].get('confirmation_phase')!='POLICY_FROZEN'
            or not p.exists() or p.is_symlink() or p.stat().st_mode&0o222 or m['execution']['sealed_inputs'].get(PLAN)!=sha(p)):
        raise PermissionError('FINAL_STATIC_QUALIFIED_CONFIRM_PLAN_MUST_BE_SEALED_BEFORE_MODEL')
    value=read(p);inputs=read(root/STATIC_INPUTS)
    if value['candidate']!=read(root/PRIOR)['candidate']:
        raise PermissionError('FROZEN_PLAN_MUST_USE_ITS_ACTUAL_PREDECLARED_RANKING')
    expected=plan.evaluation_plan(value['scene_groups'],value['candidate'],value['static_receipts'],inputs['exclusion_layouts'])
    if value!=expected:raise PermissionError('EXACT_SINGLE_CANDIDATE_40_CONDITIONS_NO_POSTRESULT_SELECTION')
    binding=root/STATIC_DELIVERY
    if (not binding.exists() or binding.is_symlink() or binding.stat().st_mode&0o222
            or m['execution']['sealed_inputs'].get(STATIC_DELIVERY)!=sha(binding)):
        raise PermissionError('ACTUAL_STATIC_FIXED_HF_DELIVERY_REQUIRED_BEFORE_POLICY')
    transferred=read(binding)
    if (transferred['status']!='PASS_SOURCE_CONFIRM_STATIC_FIXED_REVISION_READBACK'
            or transferred['repo_id']!='leoyuer/pi0294-public-artifacts'
            or re.fullmatch('[0-9a-f]{40}',transferred['revision']) is None
            or transferred['prefix']!='PI0294/'+n1.PARENT_RUN+'/SOURCE_CONFIRM_STATIC_TEN_GROUPS_V1'
            or transferred['files']['SOURCE_CONFIRM_PLAN.json']['sha256']!=sha(p)
            or transferred['READY_uploaded_last'] is not True):
        raise PermissionError('EXACT_ACTUAL_TEN_GROUP_STATIC_PLAN_FIXED_REVISION_HASH')
    return value


def candidate_checkpoint(root):
    root=Path(root);m=read(root/'RUN_MANIFEST.json')
    if m['execution']['profile']!='hopper-neural':raise PermissionError('NO_CONFIRMATION_MODEL_ON_CORE')
    b=read(root/CANDIDATE_INPUTS);p=Path(b['checkpoint'])
    wanted=root.parent.parent/'children/source-pair-01/training/S-CORR/checkpoint-2048'
    if (p.resolve()!=wanted.resolve() or p.is_symlink()
            or b['checkpoint_manifest_sha256']!=plan.CHECKPOINTS[plan.CANDIDATE]
            or sha(p/'CHECKPOINT_COMPLETE.json')!=plan.CHECKPOINTS[plan.CANDIDATE]):
        raise PermissionError('ONLY_EXISTING_EXACT_CORR2048_NO_MODEL_DOWNLOAD_OR_REPLACEMENT')
    complete=read(p/'CHECKPOINT_COMPLETE.json')
    if complete['global_step']!=2048 or not complete['full_resume']:
        raise PermissionError('ORIGINAL_COMPLETE_2048_CANDIDATE')
    return p


def closure_path(number):
    if type(number) is not int or not 1<=number<=2:raise PermissionError('TWO_CONFIRMATION_ALLOCATIONS_ONLY')
    return 'plans/CONFIRM_PRIOR_CLOSURE_ALLOCATION'+str(number).zfill(2)+'.json'


def prior_closure(root,request,ledger):
    root=Path(root);relative=closure_path(request['submit_count']);p=root/relative
    if (p.is_symlink() or p.stat().st_mode&0o222 or request['sealed_inputs'].get(relative)!=sha(p)
            or request.get('confirmation_prior_closure_relative')!=relative
            or request.get('confirmation_prior_closure_sha256')!=sha(p)):
        raise PermissionError('SEALED_OWN_PRIOR_NATIVE_EFFECT_CLOSURE_BEFORE_EACH_JOB')
    q=read(p)
    if (q['schema']!='MP1_CONFIRM_PRIOR_ALLOCATION_EFFECT_CLOSURE_V1'
            or q['run_uuid']!=root.name or q['prior_allocations']!=request['submit_count']-1
            or q['prior_GPU_s']!=request['prior_GPU_allocated_s']
            or q['prior_generation_queries']!=ledger['generation_queries']
            or q['all_recorded_processes_terminal'] is not True or q['unknown_effects']
            or q['core']['status']!='PASS_ALL_CLOSED_CORE_EFFECTS_RECORDED'
            or q['hopper']['status']!='PASS_ALL_NATIVE_DRAWS_CLASSIFIED_AND_ARRAYS_VERIFIED'
            or q['hopper']['cumulative_generation_queries']!=ledger['generation_queries']
            or q['rows']!=q['core']['rows'] or q['new_physics'] or q['new_queries']
            or len({r['attempt_id'] for r in q['rows']})!=len(q['rows'])):
        raise PermissionError('EVERY_OLD_GENERATION_INSTALL_ACK_OUTCOME_AND_PROCESS_CLOSED')
    if q['prior_allocations']==0 and (q['rows'] or q['prior_generation_queries'] or q['prior_GPU_s']):
        raise PermissionError('NO_HISTORICAL_POLICY_COST_RESET_IN_FIRST_CONFIRMATION_JOB')
    return q


def fixture_allocation(root,manifest,number):
    authority(root,manifest)
    if type(number) is not int or not 1<=number<=2:
        raise PermissionError('ONLY_TWO_REMAINING_SOURCE_FAMILY_QUALIFICATION_ALLOCATIONS')
    return number


def request_contract(root,request,ledger):
    root=Path(root);bound=authority(root);l=read(root/'HOPPER_RESOURCE_LEDGER.json')
    number=request['submit_count'];wall=request['walltime_s']
    if (type(number) is not int or not 1<=number<=2 or type(wall) is not int or not 2760<wall<=28800
            or request['mode']!=MODE or request['additional_output_upper_bytes']!=8*1024**3
            or 42833+request['prior_GPU_allocated_s']+wall+21600>bound['source']['GPU_s']
            or 10+number>bound['source']['PBS'] or l['optimizer_updates'] or l['pending_updates']
            or l['generation_queries']+2406>bound['source']['generation_queries']
            or ledger['GPU_reserved_s']!=wall):
        raise PermissionError('CONFIRMATION_CUMULATIVE_SOURCE_FAMILY_AND_FINAL_RESERVE')
    validate_plan(root)
    prior_closure(root,request,l)
