"""Causal response windows from sealed real FUSION runs, not teacher labels."""
import numpy as np
from .fusion_contract import OFFSETS


def response_window(episode, decision, native, score, states, native_steps, windows):
    """Validate one 8-history/16-response window, retaining actual task failures.

    All future states are targets. The only source-age value used by the ARX
    is the measured value at anchor8; unmeasured clock entries are masked.
    No proposed but unexecuted candidate becomes a physical label.
    """
    state=np.asarray(states,float);steps=np.asarray(native_steps)
    if (episode['split'] not in ('TRAIN','DEV')
            or episode['purpose'] not in ('TRAIN_FEEDBACK','EVALUATION')
            or episode['purpose'] == 'TRAIN_FEEDBACK' and episode['split'] != 'TRAIN'
            or episode['inference_mode'] != 'CALIBRATION_NO_VLA'
            or episode['desired_outcome_provenance'] != 'CALIBRATION_INTENT'
            or native.get('status') != 'NATIVE_STAGE_PASSED' or not native.get('shutdown_clean')
            or native.get('physics_stops') or native.get('actual_target_integrations') != 4103
            or state.shape != (25,106) or not np.isfinite(state).all()
            or not np.array_equal(steps,3503+25*np.arange(25))
            or decision.get('native_step') != 3703 or decision.get('future_response_observed') is not False
            or decision.get('split') != episode['split'] or decision.get('parent') != episode['parent']
            or type(decision.get('selected_action')) is not int or decision['selected_action'] not in range(4)):
        raise ValueError('COMPLETE_REAL_TRAIN_FEEDBACK_WINDOW_REQUIRED_NO_TEST_OR_PARTIAL_RUN')
    if (not np.allclose(state[:9],decision['actual_history'],atol=1e-10,rtol=0)
            or score.get('selected_action') != decision['selected_action']
            or score.get('split') != episode['split'] or score.get('parent') != episode['parent']):
        raise ValueError('ACTUAL_REPLAY_NATIVE_INPUT_ACTION_OR_PARENT_DIFFERS')
    if (type(score.get('local_XY_success')) is not bool
            or not np.isfinite(score['actual_XY_error_m']) or score['actual_XY_error_m'] < 0
            or np.asarray(decision['desired_xy_m']).shape != (2,)
            or not np.isfinite(decision['desired_xy_m']).all()):
        raise ValueError('ACTUAL_FINITE_RESPONSE_AND_TASK_SCORE_REQUIRED')
    actual=[w for w in windows if w['native_start_step'] >= 3703]
    if (len(actual) != 16 or [w['native_start_step']for w in actual] != list(range(3703,4103,25))
            or any(w['native_end_step']-w['native_start_step'] != 25 or w['mode'] != 'WBC'
                   or not w['reference_sha256'] or not w['target_trajectory_sha256']
                   or not w['solver_policies_with_exact_installed_target']for w in actual)):
        raise ValueError('INSTALLED_TARGET_AND_SOLVER_CONSUMPTION_EVIDENCE_REQUIRED')
    age=float(decision['source_age_s'])
    if not np.isfinite(age) or not 0 <= age < .4:raise ValueError('ACTUAL_VALID_DECISION_SOURCE_AGE_REQUIRED')
    goal=np.asarray(decision['past_reference'],float).copy()
    if goal.shape != (17,) or not np.isfinite(goal).all():raise ValueError('ACTUAL_PAST_EEF_AND_BASE_REFERENCE_REQUIRED')
    goal[:3]+=OFFSETS[decision['selected_action']]
    ages=np.zeros(25);ages[8]=age;age_mask=np.zeros(25,bool);age_mask[8]=True
    return dict(state=state.copy(),goal=goal,history_goal=np.repeat(np.asarray(decision['past_reference'])[None],8,axis=0),
        source_age=ages,source_age_valid=age_mask,future_prefix_remaining=7.,window_start_s=0.,
        observed_action=decision['selected_action'],parent=episode['parent'],split=episode['split'],
        task_success=bool(score['local_XY_success']),actual_XY_error_m=float(score['actual_XY_error_m']),
        nominal_action=decision['nominal_action'],desired_xy_m=np.asarray(decision['desired_xy_m']),
        desired_outcome_provenance='CALIBRATION_INTENT',physical_future_atoms=16,
        history_atoms=8,exact_Markov_state=False,expert_action_label=False,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        action_evidence='INSTALLED_REFERENCE_AND_SOLVER_TARGET_CONFIRMED',
        unmeasured_source_age_entries='ZERO_PLACEHOLDERS_WITH_EXPLICIT_FALSE_VALIDITY_MASK')
