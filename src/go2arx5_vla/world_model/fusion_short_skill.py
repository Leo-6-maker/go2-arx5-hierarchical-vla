"""A new target approach-step score; never an old-task predicate repair.

Object truth is scorer-only. This file neither supplies policy observations nor
grants contact/motion permission. Passing a step is not arrival or pick/place.
"""
import hashlib
import json
import numpy as np

LANGUAGE = 'Move the open gripper a little toward the red cube, then hold without touching it.'
TASK_ID = 'FUSION_TARGET_RED_CUBE_APPROACH_STEP_V1'


def document_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def score_contract():
    return dict(schema='WM02_FUSION_TARGET_APPROACH_STEP_SCORE_V1', task_id=TASK_ID,
        language=LANGUAGE, score_role='NEW_TARGET_SHORT_SKILL_NOT_OLD_PICK_PLACE',
        physical_frame='ACTUAL_TARGET_WORLD', decision_native_step=3703,
        atom_ticks=25, native_dt_s=.002, precontact_offset_from_cube_m=.12,
        minimum_signed_progress_m=.002, minimum_distance_reduction_m=.002,
        maximum_lateral_drift_m=.003, maximum_TCP_excursion_m=.025,
        minimum_finger_opening_m=.038, maximum_object_displacement_m=.0005,
        maximum_base_displacement_m=.002, maximum_base_rotation_rad=float(np.deg2rad(.5)),
        final_hold_duration_s=.4, final_hold_TCP_diameter_m=.002,
        final_hold_TCP_max_speed_m_s=.025, original_native_safety_required=True,
        robot_object_contact_allowed=False, full_task_success_claim=False,
        thresholds_are_native_safety=False, privileged_geometry_is_policy_input=False)


def task_document(scene_sha256):
    if (not isinstance(scene_sha256, str) or len(scene_sha256) != 64
            or any(c not in '0123456789abcdef' for c in scene_sha256)):
        raise ValueError('SHORT_SKILL_ACTUAL_SCENE_SHA_REQUIRED')
    return dict(schema='WM02_FUSION_PAUSED_TARGET_TASK_V2', task_id=TASK_ID,
        language=LANGUAGE, scene_sha256=scene_sha256,
        score_contract_sha256=document_hash(score_contract()),
        language_provenance='NEW_TARGET_SHORT_SKILL_PROMPT')


def score(samples, contract):
    """Evaluate actual contiguous 50ms samples, starting at the real decision.

    No inference or desired/reference pose is used as an achieved pose. The
    precontact point and direction are fixed from the first actual sample.
    """
    if contract != score_contract() or not isinstance(samples, list) or len(samples) < 9:
        raise ValueError('SHORT_SKILL_FROZEN_CONTRACT_AND_ACTUAL_HOLD_WINDOW_REQUIRED')
    required = ('native_step', 'time_s', 'tcp', 'object_pose', 'qpos', 'finger_q',
                'safety_valid', 'native_abort', 'robot_object_contact')
    if not all(isinstance(s, dict) and all(k in s for k in required) for s in samples):
        raise ValueError('SHORT_SKILL_ACTUAL_MEASURED_FIELDS_REQUIRED')
    steps = np.asarray([s['native_step'] for s in samples])
    times = np.asarray([s['time_s'] for s in samples], float)
    tcp = np.asarray([s['tcp'] for s in samples], float)
    objects = np.asarray([s['object_pose'] for s in samples], float)
    base = np.asarray([s['qpos'][:7] for s in samples], float)
    fingers = np.asarray([s['finger_q'] for s in samples], float)
    if (steps.dtype.kind not in 'iu' or any(type(s['native_step']) is not int for s in samples)
            or steps[0] != 3703 or not np.all(np.diff(steps) == 25)
            or not np.allclose(times, steps*.002, rtol=0, atol=1e-8)
            or tcp.shape != (len(samples), 4, 4) or objects.shape != tcp.shape
            or base.shape != (len(samples), 7) or fingers.shape != (len(samples), 2)
            or not all(np.isfinite(x).all() for x in (times, tcp, objects, base, fingers))):
        raise ValueError('SHORT_SKILL_REAL_CLOCK_SHAPES_AND_FINITE_MEASUREMENTS_REQUIRED')
    from scripts.pi0294.dec1.target_observation import rigid
    for matrix in np.concatenate((tcp, objects)):
        rigid(matrix)
    quaternions = base[:, 3:7]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError('SHORT_SKILL_ACTUAL_UNIT_BASE_QUATERNIONS_REQUIRED')
    p, cube = tcp[:, :3, 3], objects[:, :3, 3]
    ray = cube[0]-p[0]
    length = float(np.linalg.norm(ray))
    if length <= contract['precontact_offset_from_cube_m']:
        raise ValueError('SHORT_SKILL_NEW_APPROACH_START_MUST_BE_OUTSIDE_PRECONTACT_REGION')
    direction = ray/length
    goal = cube[0]-contract['precontact_offset_from_cube_m']*direction
    movement = p-p[0]
    signed = movement@direction
    lateral = np.linalg.norm(movement-signed[:, None]*direction, axis=1)
    distance = np.linalg.norm(p-goal, axis=1)
    tail = p[-9:]
    hold_diameter = float(np.linalg.norm(tail[:, None]-tail[None], axis=2).max())
    hold_speed = float((np.linalg.norm(np.diff(tail, axis=0), axis=1)/.05).max())
    base_rotation = 2*np.arccos(np.clip(np.abs(quaternions@quaternions[0]), 0., 1.))
    safe = all(s['safety_valid'] is True and s['native_abort'] is False for s in samples)
    no_contact = all(s['robot_object_contact'] is False for s in samples)
    metrics = dict(signed_progress_m=float(signed[-1]),
        precontact_distance_reduction_m=float(distance[0]-distance[-1]),
        maximum_lateral_drift_m=float(lateral.max()),
        maximum_TCP_excursion_m=float(np.linalg.norm(movement, axis=1).max()),
        minimum_finger_opening_m=float(fingers.min()),
        maximum_object_displacement_m=float(np.linalg.norm(cube-cube[0], axis=1).max()),
        maximum_base_displacement_m=float(np.linalg.norm(base[:, :3]-base[0, :3], axis=1).max()),
        maximum_base_rotation_rad=float(base_rotation.max()),
        final_hold_TCP_diameter_m=hold_diameter, final_hold_TCP_max_speed_m_s=hold_speed)
    predicates = dict(signed_progress=metrics['signed_progress_m'] >= contract['minimum_signed_progress_m'],
        distance_reduction=metrics['precontact_distance_reduction_m'] >= contract['minimum_distance_reduction_m'],
        lateral=metrics['maximum_lateral_drift_m'] <= contract['maximum_lateral_drift_m'],
        excursion=metrics['maximum_TCP_excursion_m'] <= contract['maximum_TCP_excursion_m'],
        open=metrics['minimum_finger_opening_m'] >= contract['minimum_finger_opening_m'],
        object_still=metrics['maximum_object_displacement_m'] <= contract['maximum_object_displacement_m'],
        base_still=metrics['maximum_base_displacement_m'] <= contract['maximum_base_displacement_m'],
        base_orientation=metrics['maximum_base_rotation_rad'] <= contract['maximum_base_rotation_rad'],
        hold=hold_diameter <= contract['final_hold_TCP_diameter_m'] and hold_speed <= contract['final_hold_TCP_max_speed_m_s'],
        original_native_safety=safe, no_robot_object_contact=no_contact)
    return dict(schema=contract['schema'], score_contract_sha256=document_hash(contract),
        task_id=TASK_ID, metrics=metrics, predicates=predicates,
        approach_step_success=all(predicates.values()), full_task_success=False,
        precontact_arrival_claim=False, pick_place_success_claim=False,
        scorer_only_privileged_geometry=True, thresholds_are_native_safety=False)
