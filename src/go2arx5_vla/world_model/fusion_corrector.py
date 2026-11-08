"""Four matched one-decision local correctors; no hidden action projection."""
import numpy as np
from .fusion_contract import OFFSETS, require_resolved_outcome

ARMS = ('B0_DIRECT', 'B1_TRAIN_STATIC_CALIBRATION', 'B2_MATCHED_FEEDBACK_ONLY',
        'B3_HISTORY_CONDITIONAL_WM_CORRECTION')


def choose(*, arm, nominal_action, desired_xy_m, provenance, actual_tcp_xy_m,
           static_response_xy_m, WM_endpoint_xy_m, nonhold_cost_m,
           observation_time_s, source_time_s):
    """Use identical observations, four legal actions and one actual boundary.

    B0 preserves the nominal command. B1 uses TRAIN response only. B2 applies
    ordinary unit-response feedback to the actual error, with no WBC gain change.
    B3 uses the independently frozen history-conditioned forecast. The model's
    scores are suggestions, never physical labels or a control-gain claim.
    """
    require_resolved_outcome(provenance)
    if arm not in ARMS or type(nominal_action) is not int or nominal_action not in range(4):
        raise ValueError('MATCHED_FUSION_ARM_AND_SUPPORTED_NOMINAL_REQUIRED_NO_PROJECTION')
    desired, current, static, predicted = map(lambda x: np.asarray(x, float),
        (desired_xy_m, actual_tcp_xy_m, static_response_xy_m, WM_endpoint_xy_m))
    if (desired.shape != (2,) or current.shape != (2,) or static.shape != (4, 2)
            or predicted.shape != (4, 2) or not all(np.isfinite(v).all() for v in
                (desired, current, static, predicted)) or not np.isfinite(nonhold_cost_m)
            or nonhold_cost_m < 0 or not np.isfinite([observation_time_s, source_time_s]).all()):
        raise ValueError('FINITE_MATCHED_ACTUAL_STATE_AND_FROZEN_PREDICTIONS_REQUIRED')
    age = observation_time_s-source_time_s
    if age < -1e-8 or age >= .4:
        return dict(status='EXPIRED_HOLD_NO_INSTALL', selected_action=0,
                    original_source_time_s=source_time_s, source_age_s=age,
                    timestamp_refreshed=False, prediction_is_physical_evidence=False)
    cost = nonhold_cost_m*np.array([0, 1, 1, 1])
    objectives = dict(B1_TRAIN_STATIC_CALIBRATION=np.linalg.norm(current+static-desired, axis=1)+cost,
        B2_MATCHED_FEEDBACK_ONLY=np.linalg.norm(current+OFFSETS[:, :2]-desired, axis=1)+cost,
        B3_HISTORY_CONDITIONAL_WM_CORRECTION=np.linalg.norm(predicted-desired, axis=1)+cost)
    choices = {k: int(v.argmin()) for k, v in objectives.items()}; choices['B0_DIRECT'] = nominal_action
    selected = choices[arm]
    return dict(status='LOCAL_CANDIDATE_SELECTED_NOT_INSTALLED', arm=arm,
        nominal_action=nominal_action, selected_action=selected,
        nominal_offset_xyz_m=OFFSETS[nominal_action].tolist(),
        corrected_offset_xyz_m=OFFSETS[selected].tolist(),
        correction_xyz_m=(OFFSETS[selected]-OFFSETS[nominal_action]).tolist(),
        desired_xy_m=desired.tolist(), desired_outcome_provenance=provenance,
        actual_decision_TCP_xy_m=current.tolist(), predicted_endpoint_xy_m=predicted.tolist(),
        static_response_xy_m=static.tolist(), objectives_m={k:v.tolist()for k,v in objectives.items()},
        all_arm_choices=choices, source_age_s=age, source_time_s=source_time_s,
        timestamp_refreshed=False, decision_opportunities=1, future_response_observed=False,
        prediction_is_physical_evidence=False, native_admission='NOT_PROVIDED_BY_SELECTOR')
