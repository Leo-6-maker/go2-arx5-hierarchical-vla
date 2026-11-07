"""Past-only live-trajectory features and separately supplied response labels.

No model, dataset split, native admission or artifact authentication is supplied
here. Callers must authenticate sealed collector records and join actual command
consumption. Object/freejoint XYZ, parent IDs and future outcomes never enter
the feature dictionary. The old four-action replay contracts are unchanged.
For replanned MPC, V2 additionally verifies the complete preview exposure and
its publication clock; matching V1's consumed first points alone is insufficient.
"""
import numpy as np

from .target_reference import require_install_clock
from .fusion_task_phase_pose_feedback_v6 import _log_so3 as log_so3

SCHEMA='WM02_LIVE_CAUSAL_TRAJECTORY_FEATURES_V1'
ROBOT_COLUMNS=tuple((*range(3,27),*range(34,60),*range(66,86)))


def _need(ok,why):
    if not ok:raise ValueError('LIVE_CAUSAL_FEATURES_'+why)


def _require_install_age(native_step,age_s):
    # Encode both endpoints in the original ROS nanosecond domain. A binary
    # float product such as40428*.002 is one ULP above80856000000/1e9; using
    # that product as a source timestamp falsely rejects an observed fresh age0.
    now_ns=int(native_step)*2_000_000
    source_ns=now_ns-int(round(float(age_s)*1e9))
    require_install_clock(source_ns/1e9,now_ns/1e9)


def _state(value,n):
    state=np.asarray(value,dtype=np.float64)
    _need(state.shape==(n,106),'ORIGINAL_106_STATE_LAYOUT')
    # Validate only fields consumed by this module, not privileged object data.
    columns=(*ROBOT_COLUMNS,*range(86,95),104,105)
    _need(np.isfinite(state[:,columns]).all(),'FINITE_WHITELISTED_OBSERVATIONS')
    return state


def _rotation6(value,*,columns):
    value=np.asarray(value,dtype=np.float64)
    _need(value.shape[-1:]==(6,)and np.isfinite(value).all(),'FINITE_ROTATION6')
    a,b=value[...,:3],value[...,3:]
    _need(np.allclose(np.linalg.norm(a,axis=-1),1.,atol=1e-6,rtol=0)
          and np.allclose(np.linalg.norm(b,axis=-1),1.,atol=1e-6,rtol=0)
          and np.allclose(np.sum(a*b,axis=-1),0.,atol=1e-6,rtol=0),'PROPER_MEASURED_OR_COMPILED_ROTATION')
    return np.stack((a,b,np.cross(a,b)),axis=-1 if columns else -2)


def _commands(value,shape):
    rows=np.asarray(value,dtype=np.float64)
    _need(rows.shape==shape and np.isfinite(rows).all(),'FINITE_COMPLETE_EEF10_COMMANDS')
    _need(np.all((rows[...,9]>=0)&(rows[...,9]<=1)),'MECHANICAL_GRIPPER_FRACTION')
    return rows,_rotation6(rows[...,3:9],columns=False)


def _eef_relative(state,anchor_position,anchor_rotation):
    position=(state[:,86:89]-anchor_position)@anchor_rotation
    rotation=_rotation6(state[:,89:95],columns=True)
    relative=np.einsum('ij,njk->nik',anchor_rotation.T,rotation)
    angles=np.stack([log_so3(matrix)for matrix in relative])
    return np.column_stack((position,angles,state[:,104:106]))


def causal_inputs(history,*,past_source_age_s,candidate_H40,proposal_source_age_s):
    """Numeric features only; history comes from the collector's original join.

    Ages are actual simulation-clock ages at each completed past atom's start,
    plus the prospective install clock observed before the new action. No wall
    clock, future reference age, future measured state or guessed TTL is used.
    BASE7 is retained by the source collector but excluded from these features:
    its physical-point relation to the measured freejoint root is unqualified.
    """
    steps=np.asarray(history['actual_history_native_steps'])
    _need(steps.shape==(9,)and steps.dtype.kind in 'iu'
          and np.array_equal(np.diff(steps),np.full(8,25)),'NINE_CONTIGUOUS_PAST_NATIVE_STEPS')
    _need(history.get('history_provenance')=='ACTUAL_SAMPLES_AND_EXACT_INSTALLED_SOLVER_JOIN_PER_ATOM'
          and history.get('full_MPC_hidden_state_restored')is False
          and np.array_equal(history['past_reference_native_end_steps'],steps[1:]),'ORIGINAL_PAST_JOIN_ABI_NOT_FULL_RESTORE')
    state=_state(history['actual_state_history'],9)
    past=np.asarray(history['past_consumed_EEF10_BASE7'],dtype=np.float64)
    _need(past.shape==(8,17),'EIGHT_CONSUMED_REFERENCES')
    command,command_rotation=_commands(past[:,:10],(8,10))
    candidate,candidate_rotation=_commands(candidate_H40,(40,10))
    age=np.asarray(past_source_age_s,dtype=np.float64)
    _need(age.shape==(8,)and np.isfinite(age).all()and np.all(age>=-1e-8)
          and np.all(age+.05<.4-1e-9),'PAST_ATOMS_WITHIN_ORIGINAL_TTL')
    _need(type(proposal_source_age_s)in(int,float)and np.isfinite(proposal_source_age_s),'ACTUAL_PRE_EFFECT_INSTALL_AGE')
    _require_install_age(steps[-1],proposal_source_age_s)
    anchor=state[-1,86:89];rotation=_rotation6(state[-1,89:95],columns=True)
    actual_rotation=_rotation6(state[:8,89:95],columns=True)
    errors=np.column_stack(((command[:,:3]-state[:8,86:89])@rotation,
        np.stack([log_so3(a.T@b)for a,b in zip(actual_rotation,command_rotation)]),
        command[:,9]-state[:8,104:106].mean(1)/.044))
    planned=np.column_stack(((candidate[:,:3]-anchor)@rotation,
        np.stack([log_so3(rotation.T@r)for r in candidate_rotation]),candidate[:,9]))
    return dict(robot_history=state[:,ROBOT_COLUMNS].copy(),
        measured_EEF_history=_eef_relative(state,anchor,rotation),
        past_consumed_reference_error=errors,past_reference_age_at_atom_start_s=age.copy(),
        planned_complete_H40=planned,
        prospective_install_clock=np.array([proposal_source_age_s,.4-proposal_source_age_s]))


def response_targets(history,*,candidate_H40,future_states,future_native_steps,future_consumed_EEF10):
    """Separate labels, including observed partial failed runs via a mask.

    A cancelled/changed command cannot supply a label for the old candidate.
    Future measured values are never returned from causal_inputs. These are
    dynamics observations, not expert actions or successful-task labels.
    """
    states=np.asarray(future_states,dtype=np.float64)
    _need(states.ndim==2 and 1<=len(states)<=16,'ONE_TO_SIXTEEN_OBSERVED_FUTURE_ATOMS')
    states=_state(states,len(states));past=_state(history['actual_state_history'],9)
    end=history['actual_history_native_steps'][-1]
    steps=np.asarray(future_native_steps)
    _need(steps.dtype.kind in 'iu'and np.array_equal(steps,end+25*np.arange(1,len(states)+1)),
          'ONLY_CONTIGUOUS_ACTUALLY_OBSERVED_FUTURE_STEPS')
    candidate,_=_commands(candidate_H40,(40,10))
    consumed,_=_commands(future_consumed_EEF10,(len(states),10))
    _need(np.array_equal(consumed,candidate[:len(states)]),'CHANGED_CANCELLED_OR_UNCONSUMED_CANDIDATE_HAS_NO_LABEL')
    rotation=_rotation6(past[-1,89:95],columns=True)
    observed=_eef_relative(states,past[-1,86:89],rotation)
    # Finger targets are displacements from current actual fingers, not commands.
    observed[:,-2:]-=past[-1,104:106]
    target=np.zeros((16,8));mask=np.zeros(16,dtype=bool)
    target[:len(states)]=observed;mask[:len(states)]=True
    return dict(TCP_SE3_and_finger_displacement=target,observed_horizon_mask=mask)
