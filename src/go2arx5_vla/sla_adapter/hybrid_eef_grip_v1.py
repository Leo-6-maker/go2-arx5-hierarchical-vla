"""Fixed physical-H40 composition. No gate-dependent source selection."""
import numpy as np

VERSION = 'MFREE_HYBRID_EEF_GRIP_V1'


def compose(eef, grip):
    a, b = np.asarray(eef), np.asarray(grip)
    if a.shape != (40, 10) or b.shape != (40, 10) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('TWO_FINITE_CORRECTLY_DECODED_PHYSICAL_H40_REQUIRED')
    result = a.copy(); result[:, 9] = b[:, 9]
    if not np.array_equal(result[:, :9], a[:, :9]) or not np.array_equal(result[:, 9], b[:, 9]):
        raise ValueError('FIXED_EEF_AND_GRIP_SOURCES_REQUIRED')
    return result


def coordination(action, expert, state, previous_reference, policy_from_world):
    """Command timing on expert states, not object grasp or student closed-loop proof."""
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10
    from .gripper_time_v1 import reference_width
    width = .002 + .040 * np.asarray(action)[:, 9]
    truth = .002 + .040 * np.asarray(expert)[:, 9]
    previous = reference_width(previous_reference)
    first = lambda mask: next((int(k) for k, v in enumerate(mask) if v), None)
    start = first(width < previous - .0001) if width[-1] < previous - .001 else None
    close = first(width <= width[-1] + .001) if width[-1] < previous - .001 else None
    world = np.linalg.inv(np.asarray(policy_from_world)) @ matrix_from10(action)
    actual = np.linalg.inv(np.asarray(policy_from_world)) @ matrix_from10(state)
    lift = first(world[:, 2, 3] - actual[2, 3] >= .002)
    crossing = first(width <= .022)
    annotated = bool(truth[0] > .022 and truth[-1] <= .022)
    return dict(close_start_k=start, near_predicted_terminal_width_k=close, EEF_raise2mm_k=lift,
        lift_minus_near_close_s=None if close is None or lift is None else .05*(lift-close),
        EEF_expert_position_error_at_predicted_crossing_m=None if not annotated or crossing is None else
            float(np.linalg.norm(np.asarray(action)[crossing, :3] - np.asarray(expert)[crossing, :3])),
        previous_actual_command_to_first_speed_m_s=float((width[0]-previous)/.05),
        onset_width_change_threshold_m=.0001, near_width_threshold_m=.001, lift_threshold_m=.002,
        expert_state_and_expert_previous_command_only=True, student_closed_loop_continuity_proven=False,
        object_grasp_or_contact_inferred=False, metric_does_not_change_original_gate=True)
