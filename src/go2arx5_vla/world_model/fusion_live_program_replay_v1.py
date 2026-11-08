"""One anchored program-conditional record from authenticated native artifacts.

File, spec, lease and artifact authentication belongs to the original reader.
This pure join cannot change a split or grant execution. An existing diagnostic
has no registered learning assignment and stays DIAGNOSTIC, never TRAIN.
"""
import numpy as np
from . import fusion_live_causal_features_v2 as feature
from .fusion_trajectory_adapter import _identity
from scripts.pi0294.task_clock import actual_source_age

SCHEMA='WM02_LIVE_PROGRAM_RESPONSE_RECORD_V1'
PHASE='collect-target-fusion-task-phase-live-skill-v9'
B=34303
FEATURE_SHAPES=dict(robot_history=(9,70),measured_EEF_history=(9,8),
    past_consumed_reference_error=(8,7),past_reference_age_at_atom_start_s=(8,),
    prospective_install_clock=(2,),planned_published_H40=(3,40,7),
    planned_install_offsets_s=(3,),planned_install_source_ages_s=(3,))


def _need(ok,why):
    if not ok:raise ValueError('LIVE_PROGRAM_REPLAY_'+why)


def assignment(spec):
    value=spec.get('response_learning_assignment')
    if value is None:return dict(split='DIAGNOSTIC',parent=None,panel_sha256=None)
    _need(isinstance(value,dict)and value.get('schema')=='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1'
          and value.get('split')in('TRAIN','DEV','TEST','DIAGNOSTIC')
          and isinstance(value.get('parent'),str)and bool(value['parent'])
          and isinstance(value.get('panel_sha256'),str)and len(value['panel_sha256'])==64
          and all(c in '0123456789abcdef'for c in value['panel_sha256']),
          'ORIGINAL_REGISTERED_SPEC_ASSIGNMENT_NOT_CALLER_RELABEL')
    return dict(value)


def extract(spec,prefix,decision,segment,*,install_atom=0):
    """Return numeric inputs/labels and separate nonstudent provenance.

    The first clock is a collector field written before the first effect. It
    is not inferred from wall time or a future achieved state. The entire
    fixed reference path must have been sealed in the original decision.
    """
    _need(spec.get('phase')==PHASE and spec.get('live_skill_start_step')==B
          and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
          'EXACT_OBSERVED_LIVE_SCOPE_NOT_OLD_FOUR_ACTION_DATA')
    _need(prefix.get('native_endpoint_step')==B and decision.get('start_step')==B
          and decision.get('end_step')==segment.get('native_endpoint_step')==spec.get('max_ticks')
          and segment.get('start_step')==B and segment.get('phase')==PHASE
          and decision.get('future_response_observed')is False
          and decision.get('Processor_decodes')==0
          and segment.get('adaptation')==decision.get('adaptation'),
          'SAME_SEALED_PRE_EFFECT_DECISION_AND_OBSERVED_PROGRAM')
    path=np.asarray(decision['nominal_reference_path'],float)
    _need(_identity(path)==decision['adaptation']['reference_path'],
          'EXACT_COMPLETE_PRE_EFFECT_REFERENCE_PATH')
    _need(np.array_equal(decision['raw_H40'],segment['original_raw_H40'])
          and np.asarray(decision['raw_H40']).shape==(40,10),
          'FULL_ORIGINAL_PROPOSAL_NOT_FUTURE_ACHIEVED_EXPERT_LABEL')
    _need(type(install_atom)is int and install_atom>=0 and install_atom%7==0,
          'ONLY_REAL_PUBLICATION_GRID_NOT_ARBITRARY_FUTURE_ANCHORS')
    observed=segment.get('windows',[])
    _need(install_atom==0 or install_atom<len(observed),'NO_UNOBSERVED_LATER_ANCHOR')
    history=(decision['actual_history']if install_atom==0
             else observed[install_atom]['actual_history_before_atom'])
    past=(prefix['windows']+observed[:install_atom])[-8:]
    _need(len(past)==8 and np.array_equal(history['actual_history_native_steps'][:-1],
        [w['native_start_step']for w in past])
        and np.array_equal(history['actual_history_native_steps'][1:],
                           [w['native_end_step']for w in past])
        and np.array_equal(history['actual_state_history'][:-1],
            [w['actual_history_before_atom']['actual_state_history'][-1]for w in past])
        and np.array_equal(history['past_consumed_EEF10_BASE7'],
                           [w['actual_installed_consumed_EEF10_BASE7']for w in past]),
          'ACTUAL_PREFIX_HISTORY_JOIN_NOT_SYNTHETIC_MIDPOINT_RESTORE')
    current=np.asarray(history['actual_state_history'][-1],float)
    tcp=np.asarray(past[-1]['actual_TCP'],float)
    _need(tcp.shape==(4,4)and np.array_equal(current[86:95],np.r_[tcp[:3,3],tcp[:3,0],tcp[:3,1]])
          and np.array_equal(current[104:106],past[-1]['gripper']['actual_finger_q_m']),
          'ACTUAL_ENDPOINT_TCP_AND_FINGERS_JOIN_CURRENT_HISTORY')
    if 'actual_state_vector'in past[-1]:
        _need(np.array_equal(current,past[-1]['actual_state_vector']),'FULL_OBSERVED_SUFFIX_ENDPOINT_JOIN')
    # Use the original ROS integer-nanosecond domain. Subtracting raw accumulated
    # mjData time can make a fresh source appear a few picoseconds in the future.
    # Raw timestamps remain in the source evidence; no TTL or time is renewed.
    age=[actual_source_age(w['clock']['native_start_s'],w['clock']['source_mpc_s'])for w in past]
    plan=feature.plan_from_complete_reference_path(path[min(install_atom,len(path)-1):])
    rows=observed[install_atom:install_atom+16]
    first_age=(actual_source_age(rows[0]['clock']['native_start_s'],rows[0]['clock']['source_mpc_s'])if rows else 0.)
    x=feature.causal_inputs(history,past_source_age_s=age,planned_H40=plan,
                            first_install_source_age_s=first_age)
    if rows:
        _need(rows[0].get('actual_history_before_atom')==history,
              'FIRST_ACTUAL_PRE_EFFECT_HISTORY_EQUALS_SAVED_DECISION')
        y=feature.response_targets(history,planned_H40=plan,first_install_source_age_s=first_age,
                                   observed_windows=rows)
    else:y=dict(TCP_SE3_and_finger_displacement=np.zeros((16,8)),
                observed_horizon_mask=np.zeros(16,dtype=bool),valid_observed_atoms=0,
                first_exposure_mismatch=dict(reason='NO_CONFIRMED_POST_DECISION_TRANSITIONS'))
    return dict(schema=SCHEMA,features=x,labels=y,assignment=assignment(spec),
        executed_expert_labels=False,full_MPC_hidden_state_restored=False,
        first_install_clock_observed_before_effect=bool(rows),
        forecast_native_step=B+25*install_atom,program_elapsed_atom=install_atom,
        overlapping_anchors_are_independent_physical_parents=False,
        complete_H40_consumed=segment.get('all_40_knots_consumed')is True,
        all_source_attempts_must_remain_in_manifest=True)


def flatten_features(record):
    """Stable numeric-only view; parent/split/task outcome cannot enter X."""
    _need(record.get('schema')==SCHEMA,'TYPED_PROGRAM_RECORD')
    _need(set(record['features'])==set(FEATURE_SHAPES),'ONLY_EXPLICIT_NUMERIC_FEATURE_WHITELIST')
    names=tuple(sorted(FEATURE_SHAPES))
    arrays=[np.asarray(record['features'][key],float)for key in names]
    _need(all(a.shape==FEATURE_SHAPES[key]and np.isfinite(a).all()for key,a in zip(names,arrays)),
          'FINITE_CAUSAL_FEATURES_WITH_FROZEN_LAYOUT')
    return np.concatenate([a.reshape(-1)for a in arrays]),names
