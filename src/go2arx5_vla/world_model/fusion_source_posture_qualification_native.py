"""Known-reference posture commissioning, with actual source-frame residuals.

A source observation supplies a calibration goal, never an expert action label.
Closing the pose gap neither establishes visual equivalence nor solves a task.
"""
import numpy as np


def run(adapter):
    from .fusion_source_posture_qualification_contract import episode_config, make_program
    from .fusion_trajectory_qualification_native import run_program
    from ..trajectory.so3 import geodesic_angle
    episode, inputs = episode_config(adapter.spec)
    if inputs.get('raw_is_generated_calibration_reference') is not True:
        raise ValueError('SOURCE_POSTURE_GENERATED_CALIBRATION_REFERENCE_MUST_BE_EXPLICIT')
    goal = np.asarray(inputs['source_goal_TCP_source'], float)
    if goal.shape != (4, 4) or not np.isfinite(goal).all():
        raise ValueError('SOURCE_POSTURE_EXACT_SOURCE_GOAL_REQUIRED')
    context = dict(schema='WM02_NATIVE_SOURCE_POSTURE_QUALIFICATION_V1',
        experiment_role='NON_VLA_CALIBRATION',
        desired_outcome_provenance='CALIBRATION_INTENT',
        calibration_goal_provenance='KNOWN_SOURCE_OBSERVATION_POSE_USED_AS_CALIBRATION_REFERENCE',
        source_pose_is_expert_action_label=False,
        generated_reference_is_GR00T_action=False,
        actual_VLA_commands=0, VLA_commands=0, VLA_reference_replays=0,
        VLA_inferences=0, original_Processor_decodes=0,
        geometry_intervention_scale=episode['geometry_scale'],
        target_task_qualification=False, full_task_success=False,
        online_query=False, online_VLA=False)

    def finalizer(report):
        measured = np.asarray(report['actual_endpoint_TCP'], float)
        actual_source = np.asarray(inputs['frame_map'], float) @ measured
        translation = actual_source[:3, 3] - goal[:3, 3]
        rotation = float(geodesic_angle(goal[:3, :3], actual_source[:3, :3]))
        report['source_posture_diagnostic'] = dict(
            goal_TCP_source=goal.tolist(), actual_endpoint_TCP_source=actual_source.tolist(),
            actual_minus_source_goal_xyz_m=translation.tolist(),
            actual_source_goal_position_error_m=float(np.linalg.norm(translation)),
            actual_source_goal_rotation_error_rad=rotation,
            diagnostic_position_tolerance_m=.01, diagnostic_rotation_tolerance_rad=.05,
            full_scale_source_pose_close=bool(episode['geometry_scale'] == 1.
                and np.linalg.norm(translation) <= .01 and rotation <= .05),
            thresholds_are_native_safety=False, native_execution_authorized=False,
            full_policy_input_distribution_qualified=False, camera_equivalence=False,
            task_success=False, source_observation_used_as_expert_action_label=False,
            residual_basis='ACTUAL_MEASURED_ENDPOINT_NOT_REFERENCE_OR_MODEL_PREDICTION')

    return run_program(adapter, episode, inputs, program_factory=make_program,
                       report_context=context, report_finalizer=finalizer)
