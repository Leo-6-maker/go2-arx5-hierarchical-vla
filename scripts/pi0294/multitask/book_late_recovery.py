"""Event-triggered held/carry recovery on existing TRAIN initial layouts.

The student must genuinely reach a safe held state. A missing event produces
no expert suffix; no DEV replay, Teacher prefix, reset or forced grasp fills it.
"""
import math
import re
import numpy as np
from .libero_book_teacher import seed_card
from .book_held_teacher import from_held_state

KIND='V7_BOOK_LATE_TRAIN_RECOVERY_PANEL'
VERSION='BOOK_PHASE4096_LATE_HELD_TRAIN_RECOVERY_V1'


def conditions(phase='all'):
    choices=[('GRASP_CONFIRMED',0.),('GRASP_CONFIRMED',0.),
             ('LIFT_CARRY',.02),('LIFT_CARRY',.035),('LIFT_CARRY',.05),('LIFT_CARRY',.06),
             ('NEAR_SUPPORT',.08),('NEAR_SUPPORT',.06)]
    result=[dict(id=f'BOOK_LATE_RECOVERY_TRAIN{i:03}',TRAIN_parent=i+5,event=event,
                 threshold_m=threshold,split='TRAIN',xy_offset_m=seed_card(i+5)['initial_xy_offset_m'])
            for i,(event,threshold)in enumerate(choices)]
    if phase=='pilot':return [result[i]for i in(0,2)]
    if phase=='remainder':return [result[i]for i in(1,3,4,5,6,7)]
    if phase!='all':raise ValueError('DECLARED_TWO_PILOTS_THEN_SIX_REMAINING')
    return result


def plan(checkpoint_sha256,phase):
    if type(checkpoint_sha256)is not str or re.fullmatch('[0-9a-f]{64}',checkpoint_sha256)is None:
        raise ValueError('ACTUAL_CLOSED_PHASE4096_CHECKPOINT_REQUIRED')
    if phase not in('pilot','remainder'):raise ValueError('FIXED_RECOVERY_PILOT_OR_REMAINDER')
    rows=conditions(phase)
    return dict(schema=VERSION,phase=phase,checkpoint_manifest_sha256=checkpoint_sha256,
        required_producer_recipe_schema='BOOK_PHASE4096_V1',required_producer_step=4096,
        conditions=rows,source_kind='ACTUAL_ON_POLICY_TRAIN_STUDENT_PREFIX',
        minimum_observed_grasp_seconds=.3,latest_takeover_index=480,
        takeover_only_original_source_Prefix8_boundaries=True,
        full_task_seconds=32,warmup_seconds=5,H40=40,prefix=8,
        original_common_home_asset_goal_native_safety_score_unchanged=True,
        DEV_or_TEST_states=False,student_prefix_is_expert=False,
        no_event_means_no_recovery_labels=True,full_state_and_controller_snapshot_required=True,
        full_safe_complete_success_required_for_suffix=True,automatic_retry=False,
        remainder_requires_two_qualified_pilot_recoveries=True,training_admitted=False)


def validate_manifest(m):
    p=m['recovery_plan'];expected=plan(m['delta_checkpoint_manifest_sha256'],p['phase']);n=len(expected['conditions'])
    if m['kind']!=KIND or p!=expected:raise PermissionError('FIXED_SAFE_LATE_TRAIN_RECOVERY_NOT_DEV')
    limits=dict(optimizer_updates=0,physics_integrations=n*18500,generation_queries=n*80,
                RGB_pairs=n*642,source_constructors=n,model_loads=1)
    if any(m['limits'].get(k)!=v for k,v in limits.items()):raise PermissionError('FINITE_ORIGINAL_FULL_EPISODES_FOR_LATE_RECOVERY')


def original_card(condition):
    if condition not in conditions():raise ValueError('FIXED_LATE_RECOVERY_EXISTING_TRAIN_ONLY')
    return seed_card(condition['TRAIN_parent'])


def recovery_card(condition):
    card=original_card(condition)
    card.update(id=condition['id'],study_split='TRAIN_LATE_HELD_RECOVERY_DIAGNOSTIC',
                original_TRAIN_parent=f"BOOK_SOURCE_SEED{condition['TRAIN_parent']:03}",
                student_prefix_is_expert=False)
    return card


def eligible_event(condition,state,*,source_index,grasp_since,initial_object_z):
    original_card(condition)
    if(source_index<8 or source_index>480 or source_index%8 or grasp_since is None
            or state.get('safety_valid')is not True or state.get('native_abort')
            or state.get('native_grasp')is not True or not state.get('left_object_contact')
            or not state.get('right_object_contact')or not state.get('opposing_contact')
            or not state.get('relative_motion_valid')or not state.get('gripper_state_valid')):return False
    elapsed=state['time_s']-grasp_since
    if not math.isfinite(elapsed)or elapsed<.3-1e-10:return False
    book=np.asarray(state['object_pose']);rise=float(book[2,3]-initial_object_z)
    if condition['event']=='GRASP_CONFIRMED':return True
    if condition['event']=='LIFT_CARRY':return rise>=condition['threshold_m']
    legal_support=(state.get('native_libero_goal')is True and state.get('object_supported_by_shelf')is True
        and np.linalg.norm(state['object_linear_velocity'])<=.02 and np.linalg.norm(state['object_angular_velocity'])<=.1)
    target=np.asarray(original_card(condition)['book']['target_book_world'])
    return bool(not legal_support and rise>=.04 and np.linalg.norm(book[:3,3]-target[:3,3])<=condition['threshold_m'])


def make_student(base,instance,identifier,record,guard):
    found=[r for r in conditions()if r['id']==identifier]
    if len(found)!=1:raise ValueError('DECLARED_LATE_RECOVERY_TRAIN_IDENTITY')
    condition=found[0]
    class Recovery(base):
        def __init__(self):
            super().__init__(instance,identifier,record,guard,seed_namespace=VERSION)
            self.expert=None;self.engine=None;self.initial_object_z=None;self.grasp_since=None;self.takeover=None
        def bind_engine(self,engine):
            if self.engine is not None:raise PermissionError('ONE_ORIGINAL_SOURCE_ENGINE')
            self.engine=engine
        def observe(self,state):
            super().observe(state)
            if self.initial_object_z is None:self.initial_object_z=float(np.asarray(state['object_pose'])[2,3])
            if state.get('native_grasp'):
                if self.grasp_since is None:self.grasp_since=state['time_s']
            else:self.grasp_since=None
            if self.expert is None and eligible_event(condition,state,source_index=self.index,
                    grasp_since=self.grasp_since,initial_object_z=self.initial_object_z):
                if self.engine is None:raise PermissionError('ORIGINAL_ENGINE_FULL_STATE_SNAPSHOT_REQUIRED')
                if self.queries!=self.completed_generations or self.queries!=self.index//8:
                    raise PermissionError('ONLY_COMPLETED_REAL_STUDENT_PREFIX_BEFORE_HANDOVER')
                from .recovery_state import capture_handover
                evidence=capture_handover(self.engine,state)
                self.expert=from_held_state(original_card(condition),state,source_index=self.index,
                    grasp_since=self.grasp_since,initial_object_z=self.initial_object_z)
                after=capture_handover(self.engine,state)
                if evidence['sha256']!=after['sha256']:raise PermissionError('TEACHER_INITIALIZATION_CHANGED_SOURCE_STATE')
                self.takeover=dict(sequence=self.index//8,index=self.index,condition=condition,
                    event=condition['event'],actual_state_eef10=self.state.tolist(),
                    student_completed_generations=self.completed_generations,student_prefix_is_expert=False,
                    observed_grasp_since=self.grasp_since,source_initial_object_z=self.initial_object_z,
                    full_state=evidence,full_state_unchanged_by_initialization=True,training_admitted=False)
                record('expert_takeover',self.takeover)
            if self.expert is not None:
                self.expert.observe(state);self.stage=self.expert.stage;self.candidate_complete=self.expert.candidate_complete
        def plan(self):
            if self.expert is None:return super().plan()
            self.guard();return self.expert.plan()
        def recovery_status(self):
            return dict(protocol=VERSION,takeover_index=None if self.takeover is None else self.takeover['index'],
                event=condition['event'],actual_student_queries=self.queries,
                actual_student_completed_generations=self.completed_generations,
                student_prefix_is_expert=False,held_state_snapshot_recorded=self.takeover is not None,
                no_event_no_expert_suffix=self.takeover is None)
    return Recovery()


def eligible_recovery_anchors(condition,takeover_index,completed_indices,*,full_task_success,native_safe):
    original_card(condition)
    if not full_task_success or not native_safe or takeover_index is None:return ()
    if(type(takeover_index)is not int or takeover_index%8 or not 8<=takeover_index<=480
            or list(completed_indices)!=list(range(640))):
        raise ValueError('ACTUAL_COMPLETE_LATE_HELD_EXPERT_SUFFIX_REQUIRED')
    return tuple(range(takeover_index,601))
