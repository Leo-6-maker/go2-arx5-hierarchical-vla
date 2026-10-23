"""Four remaining original target conditions, never a new commissioning pool.

Reuse only the presealed TRAIN geometry/static receipt parser. Source outcomes
are not inputs. Dispatch still needs the current target registration, complete
paired resources and the actual frozen whole-body prefix/feedback/STOP chain.
"""
from copy import deepcopy
import json
from . import source_commissioning_plan as geometry
from .authority import AUTHORITY,PARENT,R0
from scripts.pi0294.scene_cards import digest

SCHEMA='DEC1_ORIGINAL_TARGET_FOUR_COMMISSIONING_BINDING_V1'
FROZEN_CLOCK='82004c3171cc8cd2a0486f391595ac39424bad96882d12c5936c4eca55fe3b08'

def build(population_bytes,*,owner_release):
    fixed=geometry.build(population_bytes,owner_release=owner_release)
    conditions=[]
    for original in fixed['conditions']:
        c=deepcopy(original)
        c['condition_id']=c['condition_id'].replace('DEC1-SOURCE-','DEC1-TARGET-',1)
        c['card']['id']=c['condition_id'];c['card_sha256']=digest(c['card'])
        c.update(executor='E_TARGET',policy='CORR2048',purpose='ORIGINAL_TARGET_INPUT_DEVICE_SAFETY_QUALIFICATION_ONLY',
                 checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,noise_bank='R0',
                 task_clock_sha256=FROZEN_CLOCK,native_tick_origin=2500,
                 task_ticks=16000,physical_tick_upper_including_warmup=18500)
        conditions.append(c)
    return dict(schema=SCHEMA,status='PREDECLARED_NOT_NATIVE_ADMISSION',authority_release=AUTHORITY,
        owner_release=owner_release,original_population_sha256=fixed['original_population_sha256'],
        original_groups=fixed['source_groups'],conditions=conditions,
        total_original_commissioning_upper=8,original_source_slots=4,remaining_target_slots=4,
        source_outcomes_are_not_an_input=True,core64_cells=0,new_training_data=False,
        engineering_replacements='EXISTING_PER_CELL_ONE_AND_TOTAL_SIXTEEN_NO_RESET',
        new_GPU_or_PBS_authorized=False,target_task_native_qualified=False,
        upper_counts_requiring_fresh_full_resource_admission=dict(episode_starts=4,
            warmup_ticks=10000,live_task_ticks=64000,native_prefix_prediction_ticks_upper=64000,
            GR00T_queries=320),
        target_CPU_cost='UNKNOWN_UNTIL_CURRENT_NATIVE_COMMISSIONING_NOT_SOURCE_TIMING',
        native_dispatch_requires=[
            'CURRENT_STANDARD_ENTRY_AND_REGISTERED_DEC1_TARGET_ROLE_BEFORE_NATIVE_IMPORT',
            'FULL_FRESH_PAIRED_CORE_LOCAL_AND_VANDA_REMAINING_RESOURCES_INCLUDING_ALL_CLOSED_SDK_COSTS',
            'FIXED_ORIGINAL_CORR2048_R0_NO_AB_SUBSTITUTION_OR_TARGET_TRAINING',
            'EXPLICIT_FROZEN_82004C31_CLOCK_AND_FULL_ROBOT_ASSET_BINARY_CONFIG_BINDINGS',
            'ORIGINAL_RAW_AND_MAPPED_FULL_H40_AND_ACTUAL_TARGET_NATIVE_PREFIX_SAFETY',
            'ACTUAL_MEASURED_ROBOT_CAMERA_CLOCK_POLICY_INSTALL_ACK_AND_QUOTA_TRACE',
            'VERIFIABLE_HIGH_FREQUENCY_STOP_AND_ORIGINAL_TASK_SCORING',
            'ACTUAL_PARTIAL_EFFECTS_ONCE_SETTLEMENT_AND_LEASE_CLOSE'],
        SDK12_or_CPU_PASS_is_not_wholebody_prefix_safety=True,
        old_R4_run_and_expired_pilots_must_not_execute=True)

def validate(plan,population_bytes,*,owner_release):
    wanted=build(population_bytes,owner_release=owner_release)
    if json.dumps(plan,sort_keys=True,allow_nan=False)!=json.dumps(wanted,sort_keys=True,allow_nan=False):
        raise PermissionError('EXACT_ORIGINAL_TARGET_FOUR_NO_NEW_POOL_OR_SAFETY_SHORTCUT')
    return wanted
