"""Versioned, NumPy-only full-H40 reference adaptation; no execution or model.

The input is the *already decoded* physical native EEF10 (row-packed rot6d).
This module never invokes a Processor, accumulates relative actions, clips a
command, loads a predictor, or grants native/model-domain qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

from ..trajectory.so3 import (
    geodesic_angle, matrix_to_quat, quat_to_matrix,
    rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows,
)

SCHEMA = "WM02_FULL_H40_TRAJECTORY_ADAPTATION_V1"
RECIPE = "FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_V1"
LEGACY_SCHEMA, LEGACY_RECIPE = SCHEMA, RECIPE
SLOW_SCHEMA = "WM02_FULL_H40_TRAJECTORY_ADAPTATION_V2"
SLOW_RECIPE = "FULL_H40_RIGID_RELOCATE_SCALE_WORKING_GRIP_INTEGER_RETIME_2X_V2"
GEOMETRY_SCALES = (.03, .1, .25, .5, 1.)
ATOM_DT_S = .05
NATIVE_DT_S = .002
NATIVE_TICKS_PER_ATOM = 25
MAX_TRANSLATION_SPEED_M_S = .05
MAX_ANGULAR_SPEED_RAD_S = .25
MAX_FINGER_SPEED_M_S = .04
MAX_TRANSLATION_ACCELERATION_M_S2 = .25
MAX_ANGULAR_ACCELERATION_RAD_S2 = 1.
QUINTIC_PEAK_VELOCITY = 1.875
QUINTIC_PEAK_ACCELERATION = 10 * math.sqrt(3) / 3
MECHANICAL_TRAVEL_M = .044
# Retain the original task_gripper_contract arithmetic exactly: the existing
# guard compares endpoints strictly, and literal .042 is one ULP larger.
WORKING_CLOSED_M = MAX_FINGER_SPEED_M_S * ATOM_DT_S
WORKING_OPEN_M = MECHANICAL_TRAVEL_M - WORKING_CLOSED_M
WORKING_OPEN_FRACTION = WORKING_OPEN_M / MECHANICAL_TRAVEL_M


def _need(condition, message):
    if not condition:
        raise ValueError("TRAJECTORY_ADAPTER_" + message)


def _immutable(value):
    """A bytes-backed copy cannot be made writable by a caller."""
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _identity(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str, byte_order="C",
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def _rigid(value, name):
    result = np.asarray(value, dtype=np.float64)
    _need(result.shape == (4, 4) and np.isfinite(result).all(), name + "_FINITE_4X4")
    rotation = result[:3, :3]
    _need(np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
          and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
          and abs(float(np.linalg.det(rotation)) - 1) <= 1e-6,
          name + "_PROPER_RIGID_TRANSFORM")
    return _immutable(result)


def _inverse(value):
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def _poses(rows):
    result = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
    result[:, :3, :3] = rot6d_rows_to_matrix(rows[:, 3:9])
    result[:, :3, 3] = rows[:, :3]
    return result


def _rows(poses, grip):
    return np.column_stack((poses[:, :3, 3],
                            rotation_matrix_to_rot6d_rows(poses[:, :3, :3]), grip))


def _slerp(first, last, weight):
    # Exact slerp outside the coincident limit. The generic trajectory helper's
    # small-angle nlerp approximation can exceed an exactly saturated rate.
    first, last = np.broadcast_arrays(first, last)
    dot = np.sum(first * last, axis=-1, keepdims=True)
    last = np.where(dot < 0, -last, last)
    theta = np.arccos(np.clip(np.abs(dot), 0., 1.))
    sine = np.sin(theta)
    coincident = sine < 1e-12
    denominator = np.where(coincident, 1., sine)
    weight = np.asarray(weight)
    result = (np.sin((1 - weight) * theta) / denominator * first
              + np.sin(weight * theta) / denominator * last)
    result = np.where(coincident, (1 - weight) * first + weight * last, result)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _interval_ratios(rows):
    """Required intervals for each rest-to-rest quintic segment, not a grant."""
    poses = _poses(rows)
    distance = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1)
    angle = geodesic_angle(poses[:-1, :3, :3], poses[1:, :3, :3])
    return np.maximum.reduce((
        QUINTIC_PEAK_VELOCITY * distance
        / (MAX_TRANSLATION_SPEED_M_S * ATOM_DT_S),
        np.sqrt(QUINTIC_PEAK_ACCELERATION * distance
                / (MAX_TRANSLATION_ACCELERATION_M_S2 * ATOM_DT_S ** 2)),
        QUINTIC_PEAK_VELOCITY * angle
        / (MAX_ANGULAR_SPEED_RAD_S * ATOM_DT_S),
        np.sqrt(QUINTIC_PEAK_ACCELERATION * angle
                / (MAX_ANGULAR_ACCELERATION_RAD_S2 * ATOM_DT_S ** 2)),
        QUINTIC_PEAK_VELOCITY * MECHANICAL_TRAVEL_M * np.abs(np.diff(rows[:, 9]))
        / (MAX_FINGER_SPEED_M_S * ATOM_DT_S),
    ))


def _interpolate(first, last, intervals):
    """Exclude first, include last: quintic time law along unchanged geometry."""
    u = np.arange(1, intervals + 1, dtype=np.float64) / intervals
    weight = u ** 3 * (10 + u * (-15 + 6 * u))
    result = first[None] + weight[:, None] * (last - first)[None]
    rotations = rot6d_rows_to_matrix(np.stack((first[3:9], last[3:9])))
    quaternions = matrix_to_quat(rotations)
    result[:, 3:9] = rotation_matrix_to_rot6d_rows(quat_to_matrix(
        _slerp(quaternions[0], quaternions[1], weight[:, None])))
    result[-1] = last  # Retain every original adapted knot without roundoff drift.
    return result


def _elapsed(value):
    _need(type(value) is int and value >= 0, "NONNEGATIVE_INTEGER_ATOM_CLOCK")
    return value


def _geometry(rows, anchor):
    poses = _poses(rows)
    delta = rows[:, :3] - anchor[:3, 3]
    velocity = np.diff(rows[:, :3], axis=0) / ATOM_DT_S
    acceleration = np.diff(np.vstack((np.zeros(3), velocity, np.zeros(3))), axis=0) / ATOM_DT_S
    relative = poses[1:, :3, :3] @ np.swapaxes(poses[:-1, :3, :3], -1, -2)
    q = matrix_to_quat(relative)
    q = np.where(q[:, 3:] < 0, -q, q)
    sine = np.linalg.norm(q[:, :3], axis=1, keepdims=True)
    angular_velocity = q[:, :3] * (2 * np.arctan2(sine, q[:, 3:])) / np.where(sine > 0, sine, 1.) / ATOM_DT_S
    angular_acceleration = np.diff(np.vstack((np.zeros(3), angular_velocity, np.zeros(3))), axis=0) / ATOM_DT_S
    return dict(xyz_min_m=np.min(rows[:, :3], axis=0).tolist(),
                xyz_max_m=np.max(rows[:, :3], axis=0).tolist(),
                max_anchor_distance_m=float(np.max(np.linalg.norm(delta, axis=1))),
                path_length_m=float(np.sum(np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1))),
                max_anchor_rotation_rad=float(np.max(geodesic_angle(anchor[:3, :3], poses[:, :3, :3]))),
                gripper_mechanical_fraction_min=float(np.min(rows[:, 9])),
                gripper_mechanical_fraction_max=float(np.max(rows[:, 9])),
                maximum_translation_speed_m_s=float(np.max(np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1)) / ATOM_DT_S),
                maximum_angular_speed_rad_s=float(np.max(geodesic_angle(poses[:-1, :3, :3], poses[1:, :3, :3])) / ATOM_DT_S),
                maximum_finger_speed_m_s=float(MECHANICAL_TRAVEL_M * np.max(np.abs(np.diff(rows[:, 9]))) / ATOM_DT_S),
                maximum_discrete_translation_acceleration_m_s2=float(np.max(np.linalg.norm(acceleration, axis=1))),
                maximum_discrete_angular_acceleration_rad_s2=float(np.max(np.linalg.norm(angular_acceleration, axis=1))))


@dataclass(frozen=True, eq=False)
class TrajectoryAdaptation:
    """Frozen full trajectory, with desired motion and nominal reference separate.

    ``geometry_scale`` is a declared geometry-calibration intervention, shared
    by all comparison arms. Only 1.0 retains the full raw geometry. No scale
    constitutes a VLA result, model-domain grant, or native PASS.

    The keyword-only ``time_stretch_factor`` is exactly 1 or 2. It stretches
    every original segment's duration and resamples its original quintic;
    it never interpolates an already sampled path or changes the knots.
    """
    decoded_H40: object
    query_TCP_source: object
    T_source_from_target: object
    decision_TCP_target: object
    initial_reference_TCP_target: object
    base_pose_hold: object
    geometry_scale: float
    time_stretch_factor: int = field(default=1, kw_only=True)
    reference_intervention: object = field(default=None, kw_only=True)
    mapped_H40: np.ndarray = field(init=False, repr=False)
    desired_knots: np.ndarray = field(init=False, repr=False)
    reference_knots: np.ndarray = field(init=False, repr=False)
    desired_path: np.ndarray = field(init=False, repr=False)
    reference_path: np.ndarray = field(init=False, repr=False)
    mapped_query_TCP: np.ndarray = field(init=False, repr=False)
    query_to_decision: np.ndarray = field(init=False, repr=False)
    knot_indices: tuple = field(init=False)
    bridge_atoms: int = field(init=False)
    retime_factor: int = field(init=False)

    def __post_init__(self):
        raw = np.asarray(self.decoded_H40)
        _need(raw.shape == (40, 10) and raw.dtype.kind == "f"
              and np.isfinite(raw).all(), "ALREADY_DECODED_FINITE_FLOAT_H40_EEF10")
        _need(np.all((raw[:, 9] >= 0) & (raw[:, 9] <= 1)),
              "SOURCE_MECHANICAL_FRACTION_OUTSIDE_0_1_NO_CLAMP")
        object.__setattr__(self, "decoded_H40", _immutable(raw))
        _need(isinstance(self.geometry_scale, (int, float, np.integer, np.floating))
              and not isinstance(self.geometry_scale, (bool, np.bool_))
              and self.geometry_scale in GEOMETRY_SCALES, "EXPLICIT_VERSIONED_GEOMETRY_SCALE")
        object.__setattr__(self, "geometry_scale", float(self.geometry_scale))
        _need(type(self.time_stretch_factor) is int and self.time_stretch_factor in (1, 2),
              "EXPLICIT_INTEGER_TIME_STRETCH_FACTOR_1_OR_2")
        for name in ("query_TCP_source", "T_source_from_target", "decision_TCP_target",
                     "initial_reference_TCP_target"):
            object.__setattr__(self, name, _rigid(getattr(self, name), name))
        base = np.asarray(self.base_pose_hold, dtype=np.float64)
        _need(base.shape == (7,) and np.isfinite(base).all()
              and abs(float(np.linalg.norm(base[3:])) - 1) <= 1e-6,
              "FIXED_MEASURED_BASE_POSE7")
        object.__setattr__(self, "base_pose_hold", _immutable(base))

        source_poses = _poses(np.asarray(raw, dtype=np.float64))
        target_from_source = _inverse(self.T_source_from_target)
        mapped_poses = target_from_source @ source_poses
        mapped_query = target_from_source @ self.query_TCP_source
        relocation = self.decision_TCP_target @ _inverse(mapped_query)
        relocated = relocation @ mapped_poses
        decision = self.decision_TCP_target
        desired = relocated.copy()
        desired[:, :3, 3] = decision[:3, 3] + self.geometry_scale * (
            relocated[:, :3, 3] - decision[:3, 3])
        if self.geometry_scale != 1.:
            relative = decision[:3, :3].T @ relocated[:, :3, :3]
            scaled = quat_to_matrix(_slerp(
                np.array([0., 0., 0., 1.]), matrix_to_quat(relative), self.geometry_scale))
            desired[:, :3, :3] = decision[:3, :3] @ scaled
        grip = (WORKING_CLOSED_M + (WORKING_OPEN_M - WORKING_CLOSED_M)
                * np.asarray(raw[:, 9], dtype=np.float64)) / MECHANICAL_TRAVEL_M
        desired_knots = _rows(desired, grip)

        initial = self.initial_reference_TCP_target
        reference = desired.copy()
        # Declared world-displacement transport, deliberately not an SE(3)
        # left action on xyz. It never reanchors to subsequent achieved motion.
        reference[:, :3, 3] = initial[:3, 3] + desired[:, :3, 3] - decision[:3, 3]
        reference[:, :3, :3] = initial[:3, :3] @ decision[:3, :3].T @ desired[:, :3, :3]
        reference_knots = _rows(reference, grip)
        anchor = _rows(decision[None], [WORKING_OPEN_FRACTION])[0]
        bridge_atoms = max(1, math.ceil(float(_interval_ratios(
            np.stack((anchor, desired_knots[0])))[0])))
        retime_factor = max(1, math.ceil(float(np.max(_interval_ratios(desired_knots)))))
        bridge_atoms *= self.time_stretch_factor
        retime_factor *= self.time_stretch_factor
        indices = tuple(bridge_atoms + i * retime_factor for i in range(40))
        path = np.concatenate([anchor[None], _interpolate(anchor, desired_knots[0], bridge_atoms)]
                              + [_interpolate(desired_knots[i - 1], desired_knots[i], retime_factor)
                                 for i in range(1, 40)])
        path_poses = _poses(path)
        reference_path_poses = path_poses.copy()
        reference_path_poses[:, :3, 3] = initial[:3, 3] + path_poses[:, :3, 3] - decision[:3, 3]
        reference_path_poses[:, :3, :3] = initial[:3, :3] @ decision[:3, :3].T @ path_poses[:, :3, :3]
        reference_path = _rows(reference_path_poses, path[:, 9])
        # Preserve the declared reference knots exactly, including initial pose.
        reference_path[0] = _rows(initial[None], [WORKING_OPEN_FRACTION])[0]
        reference_path[list(indices)] = reference_knots
        if self.reference_intervention is not None:
            from .fusion_program_reference_intervention_v1 import parse,apply
            intervention=parse(self.reference_intervention)
            object.__setattr__(self,'reference_intervention',intervention)
            reference_path=apply(reference_path,intervention)
            reference_knots=reference_path[list(indices)].copy()
            limits=_geometry(reference_path,initial)
            _need(limits['maximum_translation_speed_m_s']<=MAX_TRANSLATION_SPEED_M_S+1e-12
                  and limits['maximum_discrete_translation_acceleration_m_s2']<=MAX_TRANSLATION_ACCELERATION_M_S2+1e-12,
                  'DECLARED_REFERENCE_INTERVENTION_EXCEEDS_UNCHANGED_ADAPTER_RATES')
        for name, value in dict(mapped_H40=_rows(mapped_poses, grip),
                                mapped_query_TCP=mapped_query, query_to_decision=relocation,
                                desired_knots=desired_knots, reference_knots=reference_knots,
                                desired_path=path, reference_path=reference_path).items():
            _need(np.isfinite(value).all(), "FINITE_FULL_ADAPTED_PATH")
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "bridge_atoms", bridge_atoms)
        object.__setattr__(self, "retime_factor", retime_factor)
        object.__setattr__(self, "knot_indices", indices)

    @property
    def end_atom(self):
        return len(self.desired_path) - 1

    @property
    def nominal_reference_path(self):
        return self.reference_path

    def preview(self, elapsed_atoms=0):
        """Forty nominal reference points at the original 50ms atomic clock."""
        start = min(_elapsed(elapsed_atoms), self.end_atom)
        indices = np.minimum(start + np.arange(40), self.end_atom)
        return _immutable(self.reference_path[indices])

    def desired_at(self, elapsed_atoms):
        return self.desired_path[min(_elapsed(elapsed_atoms), self.end_atom)]

    def evidence(self):
        """JSON-safe audit data only; no file writes or admission decision."""
        raw_xyz = np.asarray(self.decoded_H40[:, :3], dtype=np.float64)
        raw_rotations = rot6d_rows_to_matrix(self.decoded_H40[:, 3:9])
        value = dict(
            schema=SLOW_SCHEMA if self.time_stretch_factor == 2 else SCHEMA,
            recipe=SLOW_RECIPE if self.time_stretch_factor == 2 else RECIPE,
            intent_kind="INTENT_POSE_REFERENCE", desired_outcome_provenance="INTENT_POSE",
            experiment_role="NON_VLA_CALIBRATION", geometry_scale=self.geometry_scale,
            full_raw_geometry=self.geometry_scale == 1.,
            time_stretch_factor=self.time_stretch_factor,
            Processor_decodes=0, relative_action_accumulations=0, command_clips=0,
            learned_models_loaded=0, optimizer_updates=0, native_execution_authorized=False,
            native_qualification="NOT_ESTABLISHED_BY_PURE_ADAPTER",
            rotation_conversion="NATIVE_ROT6D_ROWS_GRAM_SCHMIDT_NOT_PROCESSOR_REDECODE",
            rotation_scaling="PER_KNOT_SHORTEST_SO3_GEODESIC_FROM_RELOCATED_QUERY",
            decoded_H40=_identity(self.decoded_H40),
            source_raw_geometry=dict(
                xyz_diameter_m=float(np.max(np.linalg.norm(raw_xyz[:, None] - raw_xyz[None], axis=-1))),
                path_length_m=float(np.sum(np.linalg.norm(np.diff(raw_xyz, axis=0), axis=1))),
                max_query_distance_m=float(np.max(np.linalg.norm(raw_xyz - self.query_TCP_source[:3, 3], axis=1))),
                max_query_rotation_rad=float(np.max(geodesic_angle(self.query_TCP_source[:3, :3], raw_rotations))),
                gripper_fraction_range=[float(np.min(self.decoded_H40[:, 9])), float(np.max(self.decoded_H40[:, 9]))]),
            input_bindings={name: _identity(getattr(self, name)) for name in (
                "query_TCP_source", "T_source_from_target", "decision_TCP_target",
                "initial_reference_TCP_target", "base_pose_hold")},
            mapped_H40=_identity(self.mapped_H40), desired_knots=_identity(self.desired_knots),
            mapped_H40_semantics="TARGET_WORLD_POSES_PLUS_EXPLICIT_AFFINE_WORKING_GRIP",
            reference_knots=_identity(self.reference_knots),
            desired_path=_identity(self.desired_path), reference_path=_identity(self.reference_path),
            mapped_query_TCP=self.mapped_query_TCP.tolist(),
            query_to_decision=self.query_to_decision.tolist(),
            source_from_target=self.T_source_from_target.tolist(),
            query_TCP_source=self.query_TCP_source.tolist(),
            actual_decision_TCP=self.decision_TCP_target.tolist(),
            initial_reference_TCP=self.initial_reference_TCP_target.tolist(),
            measured_fixed_BASE7=self.base_pose_hold.tolist(),
            reference_translation="INIT_p + (desired_p - decision_p)_WORLD_DISPLACEMENT",
            reference_rotation="INIT_R @ decision_R.T @ desired_R",
            subsequent_actual_state_reanchor=False,
            gripper=dict(source_units="ALREADY_DECODED_MECHANICAL_FRACTION_0_TO_1",
                         source_range=[0., 1.], source_mechanical_travel_m=MECHANICAL_TRAVEL_M,
                         source_mechanical_range_m=[0., MECHANICAL_TRAVEL_M],
                         target_mechanical_travel_m=MECHANICAL_TRAVEL_M,
                         target_working_range_m=[WORKING_CLOSED_M, WORKING_OPEN_M],
                         rule="(.002 + .040 * source_fraction) / .044",
                         anchor_mechanical_fraction=WORKING_OPEN_FRACTION,
                         new_task_adaptation_shared_by_all_arms=True, clamp=False,
                         native_gripper_suffix_consumption="REQUIRES_SEPARATE_ACTUAL_CONSUMPTION_EVIDENCE"),
            clock=dict(atom_dt_s=ATOM_DT_S, native_dt_s=NATIVE_DT_S,
                       native_ticks_per_atom=NATIVE_TICKS_PER_ATOM,
                       original_knot_atoms=list(range(40)), prepend_anchor_atom=0,
                       time_stretch_factor=self.time_stretch_factor,
                       unstretched_bridge_atoms=self.bridge_atoms // self.time_stretch_factor,
                       unstretched_uniform_retime_factor=self.retime_factor // self.time_stretch_factor,
                       bridge_atoms=self.bridge_atoms, uniform_retime_factor=self.retime_factor,
                       knot_indices=list(self.knot_indices),
                       knot_native_tick_offsets=[i * NATIVE_TICKS_PER_ATOM for i in self.knot_indices],
                       end_atom=self.end_atom, end_time_s=self.end_atom * ATOM_DT_S,
                       preview_points=40, preview_tail="REPEAT_FINAL_KNOT"),
            retiming=dict(time_law="REST_TO_REST_QUINTIC_10u3_MINUS_15u4_PLUS_6u5_EACH_SEGMENT",
                          time_stretch_factor=self.time_stretch_factor,
                          stretched_from="ORIGINAL_ADAPTED_KNOTS_NOT_RESAMPLED_OLD_PATH",
                          geometry_changed_by_time_stretch=False,
                          sampling="50MS_QUINTIC_SAMPLES_ORIGINAL_NATIVE_BETWEEN_GRID_INTERPOLATION_UNCHANGED",
                          path_smoothing=False, knot_deletion=False,
                          uniform_duration_after_first_knot=True,
                          peak_velocity_factor=QUINTIC_PEAK_VELOCITY,
                          peak_acceleration_factor=QUINTIC_PEAK_ACCELERATION,
                          joint_velocity_acceleration_collision_qualification="ORIGINAL_NATIVE_CERTIFIER_STILL_REQUIRED"),
            speed_limits=dict(translation_m_s=MAX_TRANSLATION_SPEED_M_S,
                              angular_rad_s=MAX_ANGULAR_SPEED_RAD_S, finger_m_s=MAX_FINGER_SPEED_M_S,
                              translation_acceleration_m_s2=MAX_TRANSLATION_ACCELERATION_M_S2,
                              angular_acceleration_rad_s2=MAX_ANGULAR_ACCELERATION_RAD_S2),
            whole_desired_path=_geometry(self.desired_path, self.decision_TCP_target),
            whole_reference_path=_geometry(self.reference_path, self.initial_reference_TCP_target),
            raw_knots_retained=40, adapted_knots_retained=40,
            MODEL_DOMAIN_COVERAGE="NOT_ESTABLISHED_OLD_XY_LOCAL_MODEL_CANNOT_QUALIFY_THIS_PATH")
        if self.time_stretch_factor == 1:
            # Existing sealed V1 programs and their readers compare the whole
            # evidence dictionary. Default timing retains even its exact keys.
            value.pop("time_stretch_factor")
            for key in ("time_stretch_factor", "unstretched_bridge_atoms", "unstretched_uniform_retime_factor"):
                value["clock"].pop(key)
            for key in ("time_stretch_factor", "stretched_from", "geometry_changed_by_time_stretch"):
                value["retiming"].pop(key)
        if self.reference_intervention is not None:
            hold=self.reference_intervention.candidate=='HOLD'
            value.update(schema='WM02_FULL_H40_INTENT_DECLARED_REFERENCE_INTERVENTION_V1',
                recipe='FULL_H40_INTENT_UNCHANGED_WITH_EXPLICIT_POST_QUERY_REFERENCE_INTERVENTION_V1',
                reference_intervention=self.reference_intervention.document(),
                raw_desired_path_unchanged=True,raw_H40_installed_as_skill=not hold,
                post_query_HOLD_control=hold,
                reference_intervention_is_native_qualification=False,
                raw_intent_is_executed_expert_label=False)
            value['reference_translation']=('FIXED_INITIAL_REFERENCE_HOLD_RAW_INTENT_UNEXECUTED'if hold
                else'INIT_p_PLUS_DESIRED_WORLD_DISPLACEMENT_PLUS_DECLARED_16_ATOM_QUINTIC_XY_OFFSET')
            if hold:
                value['reference_rotation']='FIXED_INITIAL_REFERENCE_ROTATION_RAW_INTENT_UNEXECUTED'
                value['gripper']['execution_rule']='FIXED_WORKING_OPEN_FRACTION_RAW_GRIPPER_INTENT_UNEXECUTED'
        return value


def trajectory_domain_envelope(adaptation, correction_xyz_candidates=()):
    """Offline bounds over the WHOLE nominal trajectory plus finite xyz shifts.

    Candidates are constant world-coordinate translations, not predictions or
    an executable correction policy. These bounds do not qualify rotation,
    grip, contact, model support, speeds at correction joins, or native safety.
    """
    _need(isinstance(adaptation, TrajectoryAdaptation), "VERSIONED_ADAPTATION_REQUIRED")
    candidates = np.asarray(correction_xyz_candidates, dtype=np.float64)
    if candidates.shape == (0,):
        candidates = np.empty((0, 3), dtype=np.float64)
    _need(candidates.ndim == 2 and candidates.shape[1] == 3
          and np.isfinite(candidates).all(), "FINITE_WORLD_XYZ_CORRECTION_CATALOGUE")
    offsets = np.vstack((np.zeros(3), candidates))
    result = dict(schema="WM02_FULL_TRAJECTORY_DOMAIN_ENVELOPE_V1",
                  recipe=SLOW_RECIPE if adaptation.time_stretch_factor == 2 else RECIPE,
                  correction_candidates_xyz_m=candidates.tolist(),
                  candidate_count=len(candidates), nominal_included=True,
                  all_path_points_checked=len(adaptation.reference_path),
                  rotations_and_gripper="ENTIRE_UNMODIFIED_NOMINAL_PATH_CHECKED",
                  native_execution_authorized=False, model_support="NOT_ESTABLISHED",
                  old_local_XY_qualification_reused=False,
                  correction_join_speed_qualification="NOT_ESTABLISHED_BY_CONSTANT_OFFSET_BOUNDS")
    if adaptation.time_stretch_factor == 2:
        result.update(schema="WM02_FULL_TRAJECTORY_DOMAIN_ENVELOPE_V2", time_stretch_factor=2)
    for name, path, anchor in (("desired", adaptation.desired_path, adaptation.decision_TCP_target),
                               ("reference", adaptation.reference_path, adaptation.initial_reference_TCP_target)):
        # Exact extrema for all time points and every constant catalogue member.
        summary = _geometry(path, anchor)
        summary.update(xyz_min_with_corrections_m=(np.min(path[:, :3], axis=0)
                                                  + np.min(offsets, axis=0)).tolist(),
                       xyz_max_with_corrections_m=(np.max(path[:, :3], axis=0)
                                                  + np.max(offsets, axis=0)).tolist())
        result[name] = summary
    return result
