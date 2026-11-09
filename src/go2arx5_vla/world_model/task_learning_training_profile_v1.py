"""Distinct continuous-task SFT recipe; the original Trainer math is unchanged."""
import math,re
from . import target_gr00t_profile_v1 as original
from .task_learning_export_v1 import PROFILE,DOMAIN

KIND='TASK_LEARNING_GR00T_FULL_ACTION_SIDE_FT_V1'
SEEDS=(20261109,20261110,20261111)
PROJECT_ROOT='/scratch/Projects/CFP-04/CFP04-CF-007/e1192548/Projects/GR00T-WBC'


def descriptor(value):
    return isinstance(value,dict)and set(value)=={'path','sha256'}and isinstance(value['path'],str)and isinstance(value['sha256'],str)and bool(
        re.fullmatch('[0-9a-f]{64}',value.get('sha256','')))


def recipe(dataset_sha256,statistics_sha256,*,endpoint,microbatch=1,seed=20261109):
    original.need(type(seed)is int and seed in SEEDS,'FIXED_TASK_LEARNING_SEEDS')
    original.need(microbatch==1,'ACTUALLY_MEASURED_SINGLE_H200_FULL_ACTION_MICROBATCH')
    p=original.recipe(dataset_sha256,statistics_sha256,mode='FORMAL',variant='FULL_ACTION_SIDE',
        endpoint=endpoint,microbatch=microbatch)
    p.update(schema='TASK_LEARNING_CONTINUOUS_ACTION_SIDE_RECIPE_V1',profile=PROFILE,kind=KIND,
        seed=seed,dataset_domain=DOMAIN,
        initial_optimizer='FRESH_GENERATION_FROM_ORIGINAL_CORR2048_OR_SAME_GENERATION_FULL_STATE_RECOVERY',
        policy_execution_prefix_atoms=40,policy_replan_period_s=2.,
        open_loop_probes='FIXED_REAL_PHASES_NORMAL_AND_RECOVERY_TRAIN40_DEV20',
        overlapping_future_preview_consistency='NOT_APPLICABLE_FULL_H40_EXECUTED_WITHOUT_OVERLAP')
    return p


def validate(spec):
    p=spec.get('recipe')or{};need=original.need
    expected=recipe(p.get('dataset_sha256'),p.get('statistics_sha256'),endpoint=p.get('endpoint'),
        microbatch=p.get('microbatch'),seed=p.get('seed'))
    start=spec.get('start_step')
    need(p==expected and spec.get('profile')==PROFILE and spec.get('kind')==KIND
        and spec.get('execution_site')=='HOPPER_SINGLE_GPU'
        and spec.get('physics')is False and spec.get('WBC_changes')is False
        and type(start)is int and 0<=start<p['endpoint']
        and spec.get('GR00T_updates')==p['endpoint']-start
        and spec.get('WM_updates')==spec.get('Actor_updates')==spec.get('TASK_ADAPTER_updates')==0
        and spec.get('parent_manifest_sha256')==original.PARENT
        and all(not spec.get(k)for k in('fusion_binding','pilot1_binding','next1_binding','next2_binding','night1_binding')),
        'DISTINCT_CONTINUOUS_TASK_FULL_ACTION_ORIGINAL_PARENT_AND_HOPPER_IDENTITY')
    if start:need(descriptor(spec.get('resume')),'SAME_GENERATION_FULL_STATE_RECOVERY')
    else:need(spec.get('resume')is None and p['endpoint']==2000,'FRESH_FIRST2000_NOT_OLD_FIT4_WEIGHTS')
    if p['endpoint']>2000:
        need(start>=2000 and descriptor(spec.get('endpoint_decision')),
            'SEALED_TRAIN_AND_NATIVE_DEV_DECISION_BEFORE_LONGER_SCHEDULE')
    if p['seed']!=SEEDS[0]:
        need(descriptor(spec.get('first_seed_learning_and_native_gain')),
            'ACTUAL_FIRST_SEED_GAINS_BEFORE_SEQUENTIAL_CONFIRMATION_SEEDS')
    need(descriptor(spec.get('task_learning_admission')),
        'ACTUAL_DATA_PROCESSOR_AND_HOPPER_QUALIFICATIONS_BEFORE_FORMAL_SFT')
    need(descriptor(spec.get('project_storage_admission')),
        'ACTUAL_OWN_PROJECT_STORAGE_BEFORE_FULL_STATE_CHECKPOINTS')
    need(type(spec.get('deadline_epoch'))in(int,float)and math.isfinite(spec['deadline_epoch']),
        'FINITE_SITE_DEADLINE')
    return p
