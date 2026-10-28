"""Bounded source component plan from the existing sealed TRAIN population.

No IO, admission, model call or physics. These four Teacher-driven CPU episodes
qualify source interface components; they cannot be CORR student/task results.
The owner fixes the balanced interface assignment before any DEC1 outcomes.
"""
from copy import deepcopy
import hashlib
import json
import re

from scripts.pi0294.scene_cards import digest,task_card
from .authority import AUTHORITY,PARENT,R0

POPULATION='288be5e878baf1b112190a8ac69c1b507336ca03f550ae9b4f4672f13ef90db1'
SCHEMA='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_PLAN_V1'
GROUPS=('PI0294/PV1/TRAIN/F0/000','PI0294/PV1/TRAIN/F1/000')
MODES=(('I_TIME','I_PROGRESS'),('I_PROGRESS','I_TIME'))


def build(population_bytes,*,owner_release):
    """Predeclare four existing TRAIN cards, without admitting their dispatch.

    Selection uses sealed geometry/static receipts only, never task outcomes.
    MODE assignment is an explicit new owner predeclaration within the existing
    commissioning scope, not a claim that the original plan specified it.
    """
    if (type(population_bytes) is not bytes
            or hashlib.sha256(population_bytes).hexdigest()!=POPULATION
            or not isinstance(owner_release,str) or re.fullmatch('[0-9a-f]{40}',owner_release) is None):
        raise PermissionError('EXACT_ORIGINAL_POPULATION_BYTES_AND_FIXED_OWNER_RELEASE')
    population=json.loads(population_bytes)
    if (population.get('schema')!='PI0294_PV1_DATA_PLAN_AND_SPLIT_SEAL_V1'
            or population.get('status')!='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS'
            or population.get('evaluation_success_screening') is not False):
        raise PermissionError('ORIGINAL_STATIC_POPULATION_WITHOUT_SUCCESS_SCREENING')
    conditions=[]
    for group_index,group_id in enumerate(GROUPS):
        matches=[g for g in population['groups'] if g['group_id']==group_id]
        if len(matches)!=1 or matches[0]['split']!='TRAIN':
            raise PermissionError('EXACT_EXISTING_TRAIN_GEOMETRY_REQUIRED')
        group=matches[0]
        for task_index,task in enumerate(('T1','T2')):
            original=task_card(group,task)
            receipts=[r for r in population['static_receipts']
                if r['scene_group']==group_id and r['task']==task]
            if (len(receipts)!=1 or receipts[0].get('status')!='STATIC_PASS_PENDING_LIVE_CONTACT'
                    or receipts[0].get('geometry_passed') is not True
                    or receipts[0].get('field_of_view_passed') is not True
                    or receipts[0].get('card_sha256')!=digest(original)
                    or not isinstance(receipts[0].get('receipt_sha256'),str)
                    or re.fullmatch('[0-9a-f]{64}',receipts[0]['receipt_sha256']) is None):
                raise PermissionError('ORIGINAL_GEOMETRY_FOV_STATIC_CARD_BINDING_REQUIRED')
            mode=MODES[group_index][task_index]
            condition_id=f'DEC1-SOURCE-COMMISSION-{group_id.split("/")[-2]}-000-{task}-{mode}'
            card=deepcopy(original);card['id']=condition_id
            conditions.append(dict(condition_id=condition_id,executor='E_SOURCE',interface=mode,
                task=task,scene_group=group_id,source_geometry_sha256=digest(group),
                original_TRAIN_card_sha256=digest(original),card_sha256=digest(card),card=card,
                original_static_receipt=deepcopy(receipts[0]),
                policy='ORIGINAL_FEEDBACK_TEACHER_NO_GR00T',
                purpose='SOURCE_INTERFACE_COMPONENT_QUALIFICATION_ONLY'))
    return dict(schema=SCHEMA,status='PREDECLARED_NOT_NATIVE_ADMISSION',
        authority_release=AUTHORITY,owner_release=owner_release,
        original_population_sha256=POPULATION,source_groups=list(GROUPS),
        fixed_CORR_for_later_student_matrix=dict(checkpoint_manifest_sha256=PARENT,statistics_sha256=R0,
            used_by_this_CPU_component_phase=False),
        mode_assignment='OWNER_PREDECLARED_BALANCED_WITHIN_EXISTING_FOUR_SOURCE_SLOTS',
        total_original_commissioning_upper=8,source_component_upper=4,target_component_reserved=4,
        conditions=conditions,
        upper_components=dict(episodes=4,live_ticks=64000,shadow_ticks=64000,
            prefix_clones=320,main_engine_starts=4,RGB_pairs=2564,
            extra_static_reconstruction_control_and_closeout='MUST_BE_ADDED_IN_FRESH_OWNER_BUDGET'),
        GR00T_queries=0,GPU_submissions=0,training_updates=0,
        new_training_data=False,core64_cells=0,student_neural_ACK_qualification=False,
        target_task_qualification=False,success_rate_gate_added=False,
        original_score_safety_and_32s_deadline_unchanged=True,
        native_dispatch_requires=['EXACT_PUBLISHED_SOURCE_CI_AND_STANDARD_ENTRY',
            'REGISTERED_CPU_CHILD_AND_PARENT_LEASES_AND_UNIQUE_OWNER',
            'FRESH_DEC1_AND_PARENT_REMAINING_RESOURCE_RESERVATION',
            'ORIGINAL_NUMERICS_ASSETS_COUNTER_AND_MESA_QUALIFICATION',
            'ACTUAL_SOURCE_STOP_METERS_PERSISTENCE_AND_PARTIAL_EFFECTS_CLOSEOUT'])


def validate(plan,population_bytes,*,owner_release):
    expected=build(population_bytes,owner_release=owner_release)
    # Canonical JSON comparison rejects bool-as-int substitutions as well.
    if json.dumps(plan,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise PermissionError('EXACT_PREDECLARED_SOURCE_COMPONENT_PLAN_NO_REPLACEMENT_OR_STUDENT_CLAIM')
    return deepcopy(expected)
