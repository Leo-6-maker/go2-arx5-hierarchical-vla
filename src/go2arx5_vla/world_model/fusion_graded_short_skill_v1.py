"""A distinct measured short task at B34303, never a relabelled old score.

Pure scoring only: no native wrapper, dynamic admission, inference or outcome
is supplied here. A future native caller must bind real contiguous samples.
"""
import numpy as np
from . import fusion_short_skill as original

LANGUAGE = original.LANGUAGE
TASK_ID = 'FUSION_GRADED_TARGET_RED_CUBE_APPROACH_STEP_AT34303_V1'
TRACKING_TASK_ID = 'FUSION_GRADED_SHORT_LANGUAGE_FULL_H40_TRACKING_AT34303_DIAGNOSTIC_V1'
DECISION_STEP = 34303


def score_contract():
    # Retain every original geometric/safety requirement, while explicitly
    # registering a new task identity and decision clock before execution.
    return dict(original.score_contract(),
        schema='WM02_FUSION_GRADED_TARGET_APPROACH_STEP_SCORE_V1',
        task_id=TASK_ID, decision_native_step=DECISION_STEP,
        score_role='NEW_GRADED_TARGET_SHORT_SKILL_NOT_OLD_3703_OR_PICK_PLACE')


def task_document(scene_sha256):
    return dict(original.task_document(scene_sha256), task_id=TASK_ID,
        score_contract_sha256=original.document_hash(score_contract()),
        language_provenance='NEW_TARGET_SHORT_SKILL_PROMPT')


def score(samples, contract):
    value = original._score_measured_samples(samples, contract,
        expected_contract=score_contract())
    return dict(value, decision_native_step=DECISION_STEP,
        source_task_phase_equivalence=False, old_3703_score_relabelled=False,
        native_execution_authorized=False)


def screen_proposal(program, frozen_cube_pose):
    """All-path desired-intent screen; never an actual outcome or certificate."""
    from scripts.pi0294.dec1.target_observation import rigid
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    if type(program) is not TrajectoryAdaptation or program.geometry_scale != 1.:
        raise ValueError('GRADED_SHORT_FULL_UNCLIPPED_TRAJECTORY_REQUIRED')
    cube=rigid(frozen_cube_pose)[:3,3]
    anchor=program.decision_TCP_target[:3,3]
    ray=cube-anchor;length=float(np.linalg.norm(ray));contract=score_contract()
    if length<=contract['precontact_offset_from_cube_m']:
        raise ValueError('GRADED_SHORT_START_OUTSIDE_PRECONTACT_REQUIRED')
    direction=ray/length;path=program.desired_path
    delta=path[:,:3]-anchor;signed=delta@direction
    distance=np.linalg.norm(path[:,:3]-cube,axis=1)
    metrics=dict(final_signed_progress_m=float(signed[-1]),
        final_cube_distance_reduction_m=float(length-distance[-1]),
        maximum_lateral_drift_m=float(np.linalg.norm(delta-signed[:,None]*direction,axis=1).max()),
        maximum_TCP_excursion_m=float(np.linalg.norm(delta,axis=1).max()),
        minimum_finger_opening_m=float(path[:,9].min()*.044),
        minimum_desired_or_reference_cube_distance_m=float(min(distance.min(),
            np.linalg.norm(program.reference_path[:,:3]-cube,axis=1).min())))
    predicates=dict(signed_progress=metrics['final_signed_progress_m']>=contract['minimum_signed_progress_m'],
        distance_reduction=metrics['final_cube_distance_reduction_m']>=contract['minimum_distance_reduction_m'],
        lateral=metrics['maximum_lateral_drift_m']<=contract['maximum_lateral_drift_m'],
        excursion=metrics['maximum_TCP_excursion_m']<=contract['maximum_TCP_excursion_m'],
        open=metrics['minimum_finger_opening_m']>=contract['minimum_finger_opening_m'],
        noncontact_geometry=metrics['minimum_desired_or_reference_cube_distance_m']>=.14)
    return dict(schema='WM02_GRADED_SHORT_FULL_H40_DESIRED_INTENT_SCREEN_V1',
        score_contract_sha256=original.document_hash(contract),qualified=all(predicates.values()),
        raw_knots=40,desired_points_checked=len(path),metrics=metrics,predicates=predicates,
        actual_task_success_established=False,native_safety_established=False,
        native_execution_authorized=False,cube_use='SCORER_ONLY_FROZEN_ACTUAL_B_GEOMETRY_NOT_POLICY_INPUT')


def tracking_diagnostic_screen(program, frozen_cube_pose):
    """Separate tracking examination; the strict short-task screen is retained.

    A lateral task failure is observable here, not reclassified as a short
    success. Open/toward/bounded/noncontact requirements and every original
    native certificate remain required. This supplies no execution grant.
    """
    strict=screen_proposal(program,frozen_cube_pose)
    return dict(schema='WM02_GRADED_SHORT_FULL_H40_TRACKING_DIAGNOSTIC_SCREEN_V1',
        task_id=TRACKING_TASK_ID,strict_short_proposal_screen=strict,
        eligible_for_original_native_certificate=all(
            value for key,value in strict['predicates'].items() if key!='lateral'),
        lateral_task_failure_retained=not strict['predicates']['lateral'],
        strict_short_thresholds_changed=False,strict_short_task_success=False,
        native_safety_established=False,native_execution_authorized=False,
        role='EXPLORATORY_FULL_PROPOSAL_TRACKING_NOT_SHORT_TASK_QUALIFICATION')
