"""Program-conditional H16 prediction with the actual full-preview schedule.

The original receiver sees all40 points and a fresh publication every7 atoms.
Matching only the consumed first points is insufficient: a changed unconsumed
suffix can alter MPC. This ABI includes all3 planned publications before effect,
then masks labels from the first observed exposure/clock/consumption mismatch.
No future observations, split/identity fields or object truth enter inputs.
Authentication of the sealed pre-effect plan and records remains the caller's
responsibility. This module cannot fit a model or admit native execution.
"""
import numpy as np
from . import fusion_live_causal_features_v1 as original

SCHEMA='WM02_LIVE_PROGRAM_CONDITIONAL_FEATURES_V2'
INSTALL_ATOMS=np.array([0,7,14],dtype=np.int64)
HORIZON=16


def _need(ok,why):
    if not ok:raise ValueError('LIVE_PROGRAM_FEATURES_'+why)


def plan_from_complete_reference_path(path):
    """Only a sealed pre-effect fixed program, never a future achieved path.

    The original preview's terminal HOLD padding is preserved. Selecting the
    three published previews is feature extraction, not cropping execution.
    The native collector still owns and executes the complete original path.
    """
    path=np.asarray(path,dtype=np.float64)
    _need(path.ndim==2 and len(path)>0,'NONEMPTY_COMPLETE_PRE_EFFECT_PATH')
    original._commands(path,(len(path),10))
    return np.stack([path[np.minimum(atom+np.arange(40),len(path)-1)]for atom in INSTALL_ATOMS])


def _plan(value):
    plan=np.asarray(value,dtype=np.float64)
    original._commands(plan,(3,40,10))
    _need(all(np.array_equal(plan[i,7:],plan[i+1,:33])for i in range(2)),
          'SAME_FIXED_PROGRAM_ACROSS_ALL_PLANNED_PUBLICATIONS')
    return plan


def causal_inputs(history,*,past_source_age_s,planned_H40,first_install_source_age_s):
    plan=_plan(planned_H40)
    features=original.causal_inputs(history,past_source_age_s=past_source_age_s,
        candidate_H40=plan[0],proposal_source_age_s=first_install_source_age_s)
    anchor=np.asarray(history['actual_state_history'],float)[-1]
    rotation=original._rotation6(anchor[89:95],columns=True)
    chunks=[]
    for chunk in plan:
        rows,rotations=original._commands(chunk,(40,10))
        chunks.append(np.column_stack(((rows[:,:3]-anchor[86:89])@rotation,
            np.stack([original.log_so3(rotation.T@r)for r in rotations]),rows[:,9])))
    features.pop('planned_complete_H40')
    features.update(planned_published_H40=np.stack(chunks),
        planned_install_offsets_s=INSTALL_ATOMS.astype(float)*.05,
        planned_install_source_ages_s=np.array([first_install_source_age_s,0.,0.]))
    return features


def response_targets(history,*,planned_H40,first_install_source_age_s,observed_windows):
    """Retain valid partial responses; never label an unobserved suffix.

    Rows must come from the original authenticated collector. A changed full
    preview, late source clock, cancel, missing solver/gripper confirmation or
    changed BASE reference stops label validity at that point. Subsequent rows
    remain in the source record, rather than becoming labels for this forecast.
    """
    plan=_plan(planned_H40)
    _need(type(first_install_source_age_s)in(int,float) and np.isfinite(first_install_source_age_s),
          'FINITE_PRE_EFFECT_FIRST_INSTALL_AGE')
    end=history['actual_history_native_steps'][-1]
    original._require_install_age(end,first_install_source_age_s)
    _need(1<=len(observed_windows)<=HORIZON,'OBSERVED_PARTIAL_HORIZON_ONLY')
    base=np.asarray(history['past_consumed_EEF10_BASE7'],float)[-1,10:]
    _need(base.shape==(7,)and np.isfinite(base).all(),'FROZEN_BASE_REFERENCE_REQUIRED_FOR_LABEL_JOIN')
    valid=[];mismatch=None
    for atom,row in enumerate(observed_windows):
        origin=7*(atom//7);index=atom//7;clock=row.get('clock',{})
        try:
            _need(row.get('native_start_step')==end+25*atom
                  and row.get('native_end_step')==end+25*(atom+1)
                  and row.get('actual_ticks')==25 and row.get('stage')=='ACTUAL_CONSUMPTION_CONFIRMED'
                  and bool(row.get('solver_policies_with_exact_installed_target'))
                  and row.get('gripper',{}).get('command_consumed')is True,
                  'CONTIGUOUS_CONFIRMED_ACTUAL_CONSUMPTION')
            installed=np.asarray(row['full_final_installed_H40'],float)
            consumed=np.asarray(row['actual_installed_consumed_EEF10_BASE7'],float)
            _need(np.array_equal(installed,plan[index])and consumed.shape==(17,)
                  and np.array_equal(consumed[:10],plan[0,atom])
                  and np.array_equal(consumed[10:],base), 'FULL_PREVIEW_OR_CONSUMED_REFERENCE_CHANGED')
            planned_age=first_install_source_age_s if index==0 else 0.
            _need(abs(float(clock['source_mpc_s'])-((end+25*origin)*.002-planned_age))<=1e-8
                  and abs(float(clock['native_start_s'])-(end+25*atom)*.002)<=1e-8,
                  'ACTUAL_SOURCE_CLOCK_DIFFERS_FROM_SAVED_FORECAST_PLAN')
            valid.append(row)
        except (ValueError,KeyError,TypeError)as error:
            mismatch=dict(observed_atom=atom,native_start_step=end+25*atom,reason=str(error));break
    if valid:
        labels=original.response_targets(history,candidate_H40=plan[0],
            future_states=[row['actual_state_vector']for row in valid],
            future_native_steps=[row['native_end_step']for row in valid],
            future_consumed_EEF10=[row['actual_installed_consumed_EEF10_BASE7'][:10]for row in valid])
    else:labels=dict(TCP_SE3_and_finger_displacement=np.zeros((HORIZON,8)),
                     observed_horizon_mask=np.zeros(HORIZON,dtype=bool))
    return dict(labels,valid_observed_atoms=len(valid),first_exposure_mismatch=mismatch,
        full_future_preview_schedule_checked=True,expert_action_labels=False)
