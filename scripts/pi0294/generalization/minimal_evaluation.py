"""Matched first512 DEV contracts for the existing evaluator, without dispatch.

The old C/D contract requires1024 and a CORR parent. Neither is a valid alias
for the raw-parent E1/E2 study. Reuse its exact24 conditions and original noise
namespace, with a separate typed identity. These functions grant no runtime,
concurrency, model-load, transport-qualification or physics permission.
"""
from copy import deepcopy
import re
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.dataset import POPULATION
from scripts.pi0294.pv1.plan import evaluation_cards
from .minimal_training import RAW_PARENT_FILES_ID
from .minimal_study import R0

VERSION='ARX5_MINIMAL_CORE_FIXED512_ORIGINAL_DEV24_V1'
STAGE='ARX5_MINIMAL_MATCHED_RAW_PARENT_DEV24'
PLAN='plans/SOURCE_EVALUATION_PLAN.json'
POPULATION_FILE='plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'
SCHEMA='ARX5_MINIMAL_FIXED512_SINGLE_ORIGINAL_DEV24_PLAN_V1'
ARMS=('E1','E2')


def need(ok, why):
    if not ok: raise PermissionError('ARX5_MINIMAL_EVALUATION_'+why)


def typed(value):
    return value.get('platform_evaluation_version')==VERSION


def panel(groups, candidate):
    need(candidate in ARMS, 'ONLY_FIXED_E1_E2_NO_CANDIDATE_SELECTION')
    label=candidate+'-DEV-N294101'
    cards=[]
    for base in evaluation_cards(groups,'DEV'):
        if base['noise_bank']!=294101: continue
        card=deepcopy(base)
        card.update(id=candidate+'-'+base['id'],policy_candidate=candidate,
                    pv1_model_label=candidate,pv1_session_label=label)
        cards.append(card)
    need(len(cards)==24 and len({c['scene_group'] for c in cards})==12,
         'ORIGINAL_TWELVE_GEOMETRIES_TWO_TASKS')
    return dict(candidate=candidate,checkpoint_step=512,split='DEV',noise_bank=294101,
        noise_namespace='PI0294/PV1/DEV/N294101',cards=cards,
        Teacher_takeover=False,complete_task_duration_s=32)


def candidate_plan(groups, candidate, *, checkpoint_manifest_sha256):
    need(type(checkpoint_manifest_sha256) is str and
         re.fullmatch('[0-9a-f]{64}',checkpoint_manifest_sha256),
         'EXPLICIT_FIXED_ENDPOINT_MANIFEST')
    spec=panel(groups,candidate)
    return dict(schema=SCHEMA,platform_evaluation_version=VERSION,candidate=candidate,
        population_plan_sha256=POPULATION,statistics_sha256=R0,
        raw_parent_files_identity=RAW_PARENT_FILES_ID,checkpoint_step=512,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        scheduler_horizon=1024,panels={candidate+'-DEV-N294101':spec},
        Teacher_takeover=False,evaluation_data_for_training=False,
        original_S1_and_score=True,policy_selection='NONE')


def labels(request, plan, prior):
    """Same remaining-panel rule: a valid poor result is never a retry credit."""
    from scripts.pi0294.pv1.evaluation_plan import remaining_panel
    candidate=request.get('evaluation_candidate')
    label=str(candidate)+'-DEV-N294101'
    need(typed(request) and candidate in ARMS and request.get('stage')==STAGE
        and type(request.get('checkpoint_step')) is int and request['checkpoint_step']==512
        and request.get('noise_bank')==294101
        and type(request.get('optimizer_updates')) is int and request['optimizer_updates']==0
        and request.get('statistics_sha256')==R0
        and request.get('raw_parent_files_identity')==RAW_PARENT_FILES_ID,
        'FIXED_RAW_PARENT512_ORIGINAL_NOISE_ZERO_UPDATE_REQUEST')
    need(plan.get('schema')==SCHEMA and typed(plan) and plan.get('candidate')==candidate
        and plan.get('population_plan_sha256')==POPULATION
        and plan.get('statistics_sha256')==R0
        and plan.get('raw_parent_files_identity')==RAW_PARENT_FILES_ID
        and type(plan.get('checkpoint_step')) is int and plan['checkpoint_step']==512
        and plan.get('scheduler_horizon')==1024
        and plan.get('checkpoint_manifest_sha256')==request.get('checkpoint_manifest_sha256')
        and type(plan.get('checkpoint_manifest_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}',plan['checkpoint_manifest_sha256'])
        and set(plan.get('panels',{}))=={label}
        and plan.get('Teacher_takeover') is False
        and plan.get('evaluation_data_for_training') is False
        and plan.get('original_S1_and_score') is True and plan.get('policy_selection')=='NONE',
        'ONE_DISTINCT_PREFROZEN_E1_OR_E2_PLAN_NOT_CORR_OR_BEST_STEP')
    spec=plan['panels'][label]
    need(spec.get('candidate')==candidate and spec.get('checkpoint_step')==512
        and spec.get('split')=='DEV' and spec.get('noise_bank')==294101
        and spec.get('noise_namespace')=='PI0294/PV1/DEV/N294101'
        and spec.get('Teacher_takeover') is False and spec.get('complete_task_duration_s')==32
        and len(spec.get('cards',[]))==24,
        'ORIGINAL_FULL_DURATION_POPULATION_AND_PAIRED_NOISE')
    count=len(remaining_panel(prior,spec))
    need(count>0 and type(request.get('remaining_conditions')) is int
        and request['remaining_conditions']==count,
        'EXACT_REMAINING_CONDITIONS_NO_REPEAT_OF_VALID_OUTCOMES')
    return (label,)


def session_spec(root, label):
    """Regenerate from the original sealed population, never caller flags."""
    root=Path(root);manifest=read(root/'RUN_MANIFEST.json')
    need(typed(manifest) and sha(root/POPULATION_FILE)==POPULATION
        and manifest['execution']['sealed_inputs'].get(POPULATION_FILE)==POPULATION
        and manifest['execution']['sealed_inputs'].get(PLAN)==sha(root/PLAN),
        'ACTUAL_SEALED_ORIGINAL_POPULATION_AND_FIXED512_PLAN')
    plan=read(root/PLAN)
    expected=candidate_plan(read(root/POPULATION_FILE)['groups'],plan['candidate'],
        checkpoint_manifest_sha256=plan['checkpoint_manifest_sha256'])
    need(plan==expected and label in expected['panels'],
         'NO_CHANGED_GEOMETRY_CARD_DURATION_OR_ADAPTIVE_PANEL')
    return expected['panels'][label]
