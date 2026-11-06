"""Explicit source action decoding and the qualified local reference envelope.

This module imports no ROS, Torch or simulator. A successfully decoded proposal
is not an execution grant. Runtime admission and the original SafetyGate remain
necessary; unresolved outcome semantics admit only shadow recording.
"""
from dataclasses import dataclass
import re
import numpy as np
from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.relative_se3 import compose_poses, invert_pose
from ..trajectory.so3 import quat_to_matrix, rot6d_to_matrix, rot6d_rows_to_matrix

PLAN = 'f965a2c568ec26e9cecb8301fac0785f96a575a8'
PROVENANCE = ('INTENT_POSE', 'SOURCE_OUTCOME', 'CALIBRATION_INTENT', 'SEMANTICS_UNRESOLVED')
OFFSETS = np.array([[0., 0., 0.], [.001, 0., 0.], [-.001, 0., 0.], [0., .001, 0.]])


@dataclass(frozen=True)
class ActionSemantics:
    frame: str
    pose_mode: str
    rotation: str
    gripper: str
    position_unit: str
    point_dt_s: float
    execution_points: int
    tcp_semantic_id: str
    evidence_sha256: str
    clock: str = 'SOURCE_MPC_SIM_S'
    point_time_origin: str = 'OBSERVATION_PLUS_DT'

    def validate(self, count):
        if (self.frame not in ('WORLD', 'BASE', 'CAMERA', 'TOOL')
                or self.pose_mode not in ('ABSOLUTE', 'OBSERVATION_LEFT', 'OBSERVATION_RIGHT',
                                         'INCREMENTAL_LEFT', 'INCREMENTAL_RIGHT')
                or self.rotation not in ('QUAT_XYZW', 'ROT6D_COLUMNS', 'ROT6D_ROWS')
                or self.gripper not in ('OPEN_FRACTION', 'CLOSED_FRACTION')
                or self.position_unit != 'm' or self.clock != 'SOURCE_MPC_SIM_S'
                or self.point_time_origin not in ('OBSERVATION_PLUS_DT','OBSERVATION_INCLUSIVE_ISSUE')
                or not np.isfinite(self.point_dt_s) or self.point_dt_s <= 0
                or type(self.execution_points) is not int or not 0 < self.execution_points <= count
                or not self.tcp_semantic_id
                or not re.fullmatch('[0-9a-f]{64}', self.evidence_sha256)):
            raise ValueError('EXPLICIT_FROZEN_ACTION_SEMANTICS_REQUIRED')


def decode(*, positions, orientations, gripper, timestamps, observation_timestamp,
           world_from_source, observed_tcp_world, semantics, provenance, proposal_id):
    """Decode a real or calibration proposal without inventing a fresh clock.

    LEFT applies delta @ previous; RIGHT applies previous @ delta. Observation
    modes always use the same actual observation pose; incremental modes carry
    the preceding decoded pose. Translation and rotation share this SE(3) rule.
    The caller must provide the actual transform for the named source frame.
    """
    xyz, rotation, grip, ts = map(lambda x: np.asarray(x, float),
                                (positions, orientations, gripper, timestamps))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('NONEMPTY_XYZ_ACTION_SEQUENCE_REQUIRED')
    n = len(xyz); semantics.validate(n)
    first=0 if semantics.point_time_origin=='OBSERVATION_INCLUSIVE_ISSUE' else 1
    if (rotation.shape != (n, 4 if semantics.rotation == 'QUAT_XYZW' else 6)
            or grip.shape != (n,) or ts.shape != (n,)
            or not all(np.isfinite(v).all() for v in (xyz, rotation, grip, ts))
            or not np.isfinite(observation_timestamp) or np.any(grip < 0) or np.any(grip > 1)
            or provenance not in PROVENANCE or not proposal_id
            or not np.allclose(ts, observation_timestamp + np.arange(first, n+first)*semantics.point_dt_s,
                               atol=1e-8, rtol=0)):
        raise ValueError('FINITE_ACTION_UNITS_RANGE_AND_ACTUAL_SOURCE_CLOCK_REQUIRED')
    # Validate both transforms with the existing strict SE(3) implementation.
    world_source = compose_poses(world_from_source)
    observed = compose_poses(observed_tcp_world)
    if world_source.shape != (4, 4) or observed.shape != (4, 4):
        raise ValueError('ONE_ACTUAL_FRAME_AND_TCP_TRANSFORM_REQUIRED')
    if semantics.frame == 'WORLD' and not np.allclose(world_source, np.eye(4), atol=1e-10, rtol=0):
        raise ValueError('WORLD_FRAME_TRANSFORM_MUST_BE_IDENTITY')
    conversion = dict(QUAT_XYZW=quat_to_matrix, ROT6D_COLUMNS=rot6d_to_matrix,
                      ROT6D_ROWS=rot6d_rows_to_matrix)[semantics.rotation]
    transforms = np.repeat(np.eye(4)[None], n, axis=0)
    transforms[:, :3, 3] = xyz; transforms[:, :3, :3] = conversion(rotation)
    anchor = compose_poses(invert_pose(world_source), observed)
    previous = anchor; output = []
    for delta in transforms:
        mode = semantics.pose_mode
        if mode == 'ABSOLUTE': pose = delta
        else:
            base = anchor if mode.startswith('OBSERVATION_') else previous
            pose = compose_poses(delta, base) if mode.endswith('LEFT') else compose_poses(base, delta)
        output.append(compose_poses(world_source, pose)); previous = pose
    from ..trajectory.relative_se3 import pose_position, pose_quaternion
    output = np.asarray(output)
    opening = grip if semantics.gripper == 'OPEN_FRACTION' else 1-grip
    return TrajectoryChunkV1(chunk_id=proposal_id, source='FUSION_DECODED_PROPOSAL_V2_ISSUE_CLOCK'if first==0 else'FUSION_DECODED_PROPOSAL_V1',
        observation_timestamp=float(observation_timestamp), point_timestamps=ts.copy(),
        eef_position=pose_position(output), eef_orientation=pose_quaternion(output),
        source_frame='world', tcp_semantic_id=semantics.tcp_semantic_id,
        validity_start=float(observation_timestamp), validity_end=float(observation_timestamp)+.4,
        gripper=opening.copy(), provenance=dict(desired_outcome_provenance=provenance,
            semantics_evidence_sha256=semantics.evidence_sha256, source_frame=semantics.frame,
            pose_mode=semantics.pose_mode, execution_points=semantics.execution_points,
            point_time_origin=semantics.point_time_origin,
            original_source_timestamps_preserved=True, runtime_admission='NOT_PROVIDED_BY_DECODER'))


def require_resolved_outcome(provenance):
    if provenance not in PROVENANCE or provenance == 'SEMANTICS_UNRESOLVED':
        raise PermissionError('UNRESOLVED_DESIRED_OUTCOME_SHADOW_ONLY')


def validate_local_preview(action, *, initial_tcp, chosen_action, renewed, base_pose, frozen_base_pose):
    """Check the ENTIRE installed H40 against one already qualified program.

    A small correction to an unsupported nominal path does not pass this check.
    No projection/clipping, new Z/rotation/gripper program or BASE motion occurs.
    This envelope supplements, and never replaces, the native SafetyGate.
    """
    from .target_reference import fixed_goal_poses
    from ..task_data.codec import pose10
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    initial = compose_poses(initial_tcp)
    if initial.shape != (4, 4) or type(chosen_action) is not int or chosen_action not in range(4):
        raise ValueError('QUALIFIED_LOCAL_ACTION_AND_REAL_INITIAL_TCP_REQUIRED')
    base, frozen = np.asarray(base_pose, float), np.asarray(frozen_base_pose, float)
    if (base.shape != (7,) or frozen.shape != (7,) or not np.isfinite([base, frozen]).all()
            or abs(np.linalg.norm(frozen[3:])-1) > 1e-6
            or not np.allclose(base, frozen, atol=1e-10, rtol=0)):
        raise ValueError('FROZEN_MEASURED_BASE_HOLD_REQUIRED')
    goal = initial.copy(); goal[:3, 3] += OFFSETS[chosen_action]
    poses = fixed_goal_poses(initial, goal, renewed=renewed)
    expected = np.stack([pose10(p, float(teacher_physical_fraction(1.))) for p in poses])
    action = np.asarray(action, float)
    if (action.shape != (40, 10) or not np.isfinite(action).all()
            or not np.allclose(action, expected, atol=1e-9, rtol=0)):
        raise ValueError('FULL_FINAL_H40_OUTSIDE_QUALIFIED_LOCAL_REFERENCE_DOMAIN')
    return dict(domain='SINGLE_DECISION_LOCAL_XY4_RAMP8_H16_OPEN_GRIPPER',
                full_preview_points_checked=40, chosen_action=chosen_action,
                physical_gradient_qualified=False, exact_Markov_state=False,
                native_safety='ORIGINAL_SAFETY_GATE_STILL_REQUIRED')
