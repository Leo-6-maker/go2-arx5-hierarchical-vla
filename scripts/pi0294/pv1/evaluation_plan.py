"""Predeclared PV1 DEV panels preserve conditions, noise and all valid negatives."""
from copy import deepcopy
from pathlib import Path
import re
import uuid

from scripts.pi0294.e1.control_read import read
from scripts.pi0294.native_dataset import sha
from .dataset import POPULATION
from .decision import AUTHORITY
from .plan import evaluation_cards,BANKS
from .recipe import PARENT
from .sampler import ARMS

PLAN='plans/SOURCE_EVALUATION_PLAN.json'
POPULATION_FILE='plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'


def development(groups,*,route,candidate,new_bank_manifest_sha256):
    allowed=('B0_CORR',)+ARMS.get(route,())+tuple('REPLICA_'+a for a in ARMS.get(route,()))
    if route not in ARMS or candidate not in allowed or not re.fullmatch('[0-9a-f]{64}',new_bank_manifest_sha256):
        raise PermissionError('PV1_ONLY_B0_OR_FIXED_TERMINAL_PRIMARY_RECIPE_DEV_CANDIDATES')
    panels={}
    for bank in BANKS:
        label=candidate+'-DEV-N'+str(bank);cards=[]
        for base in evaluation_cards(groups,'DEV'):
            if base['noise_bank']!=bank:continue
            c=deepcopy(base)
            c.update(id=candidate+'-'+base['id'],policy_candidate=candidate,
                pv1_model_label=candidate,pv1_session_label=label)
            cards.append(c)
        if len(cards)!=24:raise PermissionError('PV1_EXACT_12_GROUP_TWO_TASK_PANEL_PER_NOISE_BANK')
        panels[label]=dict(candidate=candidate,checkpoint_step=2048 if candidate=='B0_CORR' else 4096,
            split='DEV',noise_bank=bank,noise_namespace='PI0294/PV1/DEV/N'+str(bank),
            cards=cards,Teacher_takeover=False,complete_task_duration_s=32)
    return dict(schema='PI0294_PV1_PRESEALED_POLICY_EVALUATION_V1',authority_release=AUTHORITY,
        population_plan_sha256=POPULATION,new_bank_manifest_sha256=new_bank_manifest_sha256,
        route=route,candidate=candidate,parent_CORR_checkpoint_manifest_sha256=PARENT,
        panels=panels,regular_conditions=48,unique_scene_groups=12,
        evaluation_conditions_teacher_success_screened=False,original_S1_and_score=True,
        engineering_replacements_per_condition_max=1,valid_policy_outcomes_repeated=False,
        final_LOCKBOX_opened=False,policy_selection='NONE')


def session_spec(root,label):
    root=Path(root);m=read(root/'RUN_MANIFEST.json')
    if (sha(root/POPULATION_FILE)!=POPULATION
            or m['execution']['sealed_inputs'].get(POPULATION_FILE)!=POPULATION
            or m['execution']['sealed_inputs'].get(PLAN)!=sha(root/PLAN)):
        raise PermissionError('PV1_SEALED_ACTUAL_STATIC_POPULATION_AND_ALL_DEV_CONDITIONS')
    plan=read(root/PLAN);groups=read(root/POPULATION_FILE)['groups']
    expected=development(groups,route=plan['route'],candidate=plan['candidate'],
        new_bank_manifest_sha256=plan['new_bank_manifest_sha256'])
    if plan!=expected or label not in expected['panels']:
        raise PermissionError('PV1_EXACT_PREFROZEN_DEVELOPMENT_PANEL_NO_NEW_DISTRIBUTIONS')
    return expected['panels'][label]


def remaining_panel(rows,spec):
    remaining=[]
    for card in spec['cards']:
        matches=[r for r in rows if r['condition_id']==card['condition_id'] and r['model']==card['policy_candidate']]
        ids={r['attempt_id'] for r in matches};original=next((r for r in matches if r['attempt_id']==card['id']),None)
        replacement=card['id']+'-REPLACEMENT-01'
        if (len(matches)>2 or len(ids)!=len(matches) or ids-{card['id'],replacement}
                or any(r.get('live_native_failures') or r['status']=='ACTUAL_SAFETY_FAIL'
                    or r['failure_stage']=='S1_INTEGRITY' for r in matches)):
            raise PermissionError('PV1_SAFETY_IDENTITY_OR_ONE_REPLACEMENT_BOUND')
        valid=[r for r in matches if r['pipeline_trustworthy'] and r['status'] in ('SUCCESS','VALID_TASK_FAIL')]
        if len(valid)>1 or replacement in ids and (original is None or original['pipeline_trustworthy'] or original['status']!='ENGINEERING_INCOMPLETE'):
            raise PermissionError('PV1_VALID_TASK_RESULT_CANNOT_BE_REPEATED')
        if replacement in ids and (original.get('confirmed_engineering_failure') is not True
                or not re.fullmatch('[0-9a-f]{64}',original.get('engineering_diagnosis_sha256',''))):
            raise PermissionError('PV1_RECORDED_REPLACEMENT_REQUIRES_ORIGINAL_SETTLED_DIAGNOSIS')
        if valid or len(matches)==2:continue
        if not matches:remaining.append(deepcopy(card));continue
        if (original['status']!='ENGINEERING_INCOMPLETE' or original['pipeline_trustworthy']
                or original.get('confirmed_engineering_failure') is not True
                or not re.fullmatch('[0-9a-f]{64}',original.get('engineering_diagnosis_sha256',''))):
            raise PermissionError('PV1_ENGINEERING_REPLACEMENT_NEEDS_SETTLED_EVIDENCE_NOT_POLICY_FAIL')
        new=deepcopy(card);new.update(id=replacement,pv1_replacement_of=original['attempt_id'],
            engineering_diagnosis_sha256=original['engineering_diagnosis_sha256'])
        remaining.append(new)
    return remaining


def validate_registry(registry,spec,identity,*,run_uuid,job_id,peer,prior_rows):
    allowed={c['id']:c for c in remaining_panel(prior_rows,spec)}
    indices={identifier:i for i,identifier in enumerate(allowed)}
    if (registry['run_uuid']!=run_uuid or registry['checkpoint_identity']!=identity
            or registry['job_id']!=job_id or identity['global_step']!=spec['checkpoint_step']
            or any(registry[k]!=peer[k] for k in ('core_manifest_sha256','core_source_sha','hopper_manifest_sha256','hopper_source_sha'))
            or len(registry['cases'])!=len(allowed)
            or {c['attempt_id'] for c in registry['cases'].values()}!=set(allowed)):
        raise PermissionError('PV1_EXACT_SAME_NATIVE_PEERS_MODEL_AND_REMAINING_PANEL')
    for episode,case in registry['cases'].items():
        uuid.UUID(episode)
        card=case['card'];base={k:v for k,v in card.items() if not k.startswith('relay_request_')}
        if (base!=allowed[case['attempt_id']] or episode!=case['episode_uuid'] or case['run_uuid']!=run_uuid
                or case['identity']!=identity['relay_identity']
                or case['panel_index']!=indices[case['attempt_id']] or case['deadline_epoch']!=registry['deadline_epoch']
                or case['layout_id']!=card['scene_group']+'/'+card['task']
                or card['noise_bank']!=spec['noise_bank'] or card['teacher_takeover'] is not False
                or case['hopper_session_relative']!='evidence/'+card['pv1_session_label']+'/resident'):
            raise PermissionError('PV1_LANGUAGE_ONLY_POLICY_EXACT_TASK_AND_CANDIDATE_INDEPENDENT_NOISE')
    return True
