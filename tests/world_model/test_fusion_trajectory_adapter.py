"""Synthetic offline geometry fixtures, never native/VLA qualification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from go2arx5_vla.trajectory.so3 import (
    geodesic_angle, quat_to_matrix, rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)
from go2arx5_vla.world_model import fusion_trajectory_adapter as a
from scripts.pi0294.task_gripper_contract import require_working_target, teacher_physical_fraction


BASE = [0., 0., .4, 0., 0., 0., 1.]


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    return quat_to_matrix(np.r_[axis * np.sin(angle / 2), np.cos(angle / 2)])


def pose(xyz=(0., 0., 0.), axis=(0., 0., 1.), angle=0.):
    value = np.eye(4)
    value[:3, :3] = rotation(axis, angle)
    value[:3, 3] = xyz
    return value


def rows(poses, grip=1.):
    poses = np.asarray(poses)
    return np.column_stack((poses[:, :3, 3], rotation_matrix_to_rot6d_rows(poses[:, :3, :3]),
                            np.broadcast_to(grip, (len(poses),))))


def fixture(scale=1., *, raw=None):
    query = pose([.2, -.1, .6], (1, 2, 3), .3)
    frame = pose([.1, .2, -.3], (2, -1, 3), .4)
    decision = pose([.3, .1, .5], (1, -3, 2), -.2)
    initial = pose([.27, .12, .54], (-1, 2, 1), .35)
    if raw is None:
        poses = []
        for i in range(40):
            p = query.copy()
            p[:3, 3] += [.003 * np.sin(i / 4), .0008 * i, -.0003 * i]
            p[:3, :3] = query[:3, :3] @ rotation([1, 2, -1], .001 * i)
            poses.append(p)
        raw = rows(poses, np.linspace(0, 1, 40))
    return a.TrajectoryAdaptation(raw, query, frame, decision, initial, BASE, scale)


def matrices(eef):
    result = np.broadcast_to(np.eye(4), (len(eef), 4, 4)).copy()
    result[:, :3, :3] = rot6d_rows_to_matrix(eef[:, 3:9])
    result[:, :3, 3] = eef[:, :3]
    return result


def test_noncommuting_frame_relocation_and_world_displacement_reference():
    p = fixture()
    raw = matrices(p.decoded_H40)
    target_from_source = np.linalg.inv(p.T_source_from_target)
    mapped_query = target_from_source @ p.query_TCP_source
    expected_mapped = target_from_source @ raw
    relocated = p.decision_TCP_target @ np.linalg.inv(mapped_query) @ expected_mapped
    np.testing.assert_allclose(matrices(p.mapped_H40), expected_mapped, atol=1e-14)
    np.testing.assert_allclose(matrices(p.desired_knots), relocated, atol=1e-14)
    expected_xyz = p.initial_reference_TCP_target[:3, 3] + relocated[:, :3, 3] - p.decision_TCP_target[:3, 3]
    expected_rotations = (p.initial_reference_TCP_target[:3, :3]
                          @ p.decision_TCP_target[:3, :3].T @ relocated[:, :3, :3])
    np.testing.assert_allclose(p.reference_knots[:, :3], expected_xyz, atol=1e-14)
    np.testing.assert_allclose(matrices(p.reference_knots)[:, :3, :3], expected_rotations, atol=1e-14)
    wrong_se3_transport = p.initial_reference_TCP_target @ np.linalg.inv(p.decision_TCP_target) @ relocated
    assert np.max(np.abs(expected_xyz - wrong_se3_transport[:, :3, 3])) > .001
    assert np.max(np.abs(expected_rotations - expected_rotations.transpose(0, 2, 1))) > .1


@pytest.mark.parametrize("scale", a.GEOMETRY_SCALES)
def test_each_grade_preserves_all40_knots_rotation_scale_and_entire_grip_range(scale):
    p = fixture(scale)
    full = fixture(1.)
    delta = full.desired_knots[:, :3] - full.decision_TCP_target[:3, 3]
    np.testing.assert_allclose(p.desired_knots[:, :3], p.decision_TCP_target[:3, 3] + scale * delta, atol=1e-14)
    angles = geodesic_angle(p.decision_TCP_target[:3, :3], matrices(p.desired_knots)[:, :3, :3])
    full_angles = geodesic_angle(full.decision_TCP_target[:3, :3], matrices(full.desired_knots)[:, :3, :3])
    np.testing.assert_allclose(angles, scale * full_angles, atol=3e-8)
    np.testing.assert_array_equal(p.desired_path[list(p.knot_indices)], p.desired_knots)
    np.testing.assert_array_equal(p.reference_path[list(p.knot_indices)], p.reference_knots)
    np.testing.assert_array_equal(p.desired_knots[:, 9], full.desired_knots[:, 9])
    np.testing.assert_allclose(p.desired_knots[:, 9], (.002 + .040 * p.decoded_H40[:, 9]) / .044)
    np.testing.assert_allclose(p.desired_knots[[0, -1], 9], np.array([.002, .042]) / .044)
    assert len(p.knot_indices) == 40
    assert p.end_atom == p.bridge_atoms + 39 * p.retime_factor
    assert len(p.desired_path) == p.end_atom + 1


def test_alpha_one_identity_when_live_query_and_decision_anchor_match():
    q = pose([.3, -.1, .4], (1, 3, -1), .5)
    raw = rows([q] * 40, np.linspace(0, 1, 40))
    p = a.TrajectoryAdaptation(raw, q, np.eye(4), q, q, BASE, 1.)
    np.testing.assert_allclose(matrices(p.desired_knots), matrices(raw), atol=1e-14)
    np.testing.assert_allclose(p.query_to_decision, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(p.reference_knots, p.desired_knots, atol=1e-14)


def test_uniform_integer_quintic_timing_and_separate_bridge_without_path_smoothing():
    poses = [pose([.02 + (i % 2) * .013, .001 * i, 0.], (0, 1, 0), .001 * i) for i in range(40)]
    p = a.TrajectoryAdaptation(rows(poses, 1.), np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, 1.)
    # Independent analytic bounds determine the common duration for all39 joins.
    distance = np.linalg.norm(np.diff(p.desired_knots[:, :3], axis=0), axis=1).max()
    expected_factor = int(np.ceil(max(1.875 * distance / (.05 * .05),
                                     np.sqrt((10 * np.sqrt(3) / 3) * distance / (.25 * .05 ** 2)))))
    assert p.retime_factor == expected_factor
    assert p.bridge_atoms != p.retime_factor
    np.testing.assert_array_equal(np.diff(p.knot_indices), np.full(39, expected_factor))
    i = 10; j = p.retime_factor // 3; u = j / p.retime_factor
    weight = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    expected = (1 - weight) * p.desired_knots[i - 1, :3] + weight * p.desired_knots[i, :3]
    np.testing.assert_allclose(p.desired_path[p.knot_indices[i - 1] + j, :3], expected, atol=1e-14)
    assert not np.isclose(weight, u)
    e = p.evidence()
    assert e['retiming']['path_smoothing'] is False
    assert e['retiming']['knot_deletion'] is False


@pytest.mark.parametrize("scale", a.GEOMETRY_SCALES)
def test_full_path_and_bridge_physical_rates_and_discrete_accelerations(scale):
    p = fixture(scale)
    for path in (p.desired_path, p.reference_path):
        velocity = np.diff(path[:, :3], axis=0) / .05
        acceleration = np.diff(np.vstack((np.zeros(3), velocity, np.zeros(3))), axis=0) / .05
        assert np.max(np.linalg.norm(velocity, axis=1)) <= .05 + 1e-12
        assert np.max(np.linalg.norm(acceleration, axis=1)) <= .25 + 1e-10
        assert np.max(.044 * np.abs(np.diff(path[:, 9])) / .05) <= .04 + 1e-12
        rotation_path = matrices(path)[:, :3, :3]
        assert np.max(geodesic_angle(rotation_path[:-1], rotation_path[1:]) / .05) <= .25 + 1e-7
    e = p.evidence()
    assert e['whole_desired_path']['maximum_discrete_angular_acceleration_rad_s2'] <= 1. + 1e-7
    assert e['whole_reference_path']['maximum_discrete_angular_acceleration_rad_s2'] <= 1. + 1e-7
    assert e['retiming']['joint_velocity_acceleration_collision_qualification'] == 'ORIGINAL_NATIVE_CERTIFIER_STILL_REQUIRED'


def test_near_pi_scaled_so3_branch_is_retimed_using_final_geometry():
    angles = np.deg2rad(np.resize([179., -179.], 40))
    raw = rows([pose(axis=(0, 0, 1), angle=t) for t in angles])
    p = a.TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, .5)
    full = a.TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, 1.)
    assert p.retime_factor > 50 * full.retime_factor
    e = p.evidence()['whole_desired_path']
    assert e['maximum_angular_speed_rad_s'] <= .25 + 1e-7
    assert e['maximum_discrete_angular_acceleration_rad_s2'] <= 1. + 1e-7


def test_preview_uses_immutable_original_clock_suffix_and_tail_holds():
    p = fixture(.1)
    for elapsed in [0, p.bridge_atoms - 1, p.bridge_atoms, p.bridge_atoms + 1,
                    7, 14, p.end_atom - 1, p.end_atom, p.end_atom + 100]:
        expected = p.reference_path[np.minimum(elapsed + np.arange(40), p.end_atom)]
        np.testing.assert_array_equal(p.preview(elapsed), expected)
        np.testing.assert_array_equal(p.desired_at(elapsed), p.desired_path[min(elapsed, p.end_atom)])
    np.testing.assert_allclose(p.desired_path[0, :3], p.decision_TCP_target[:3, 3])
    np.testing.assert_allclose(p.reference_path[0, :3], p.initial_reference_TCP_target[:3, 3])
    assert p.desired_path[0, 9] == float(teacher_physical_fraction(1.))
    np.testing.assert_array_equal(p.preview(7)[:33], p.preview()[7:])
    for invalid in [-1, 1.5, True, np.int64(2)]:
        with pytest.raises(ValueError, match='INTEGER_ATOM_CLOCK'):
            p.preview(invalid)
        with pytest.raises(ValueError):
            p.desired_at(invalid)


def test_raw_float32_noncontiguous_bytes_and_all_outputs_are_frozen():
    original = fixture().decoded_H40.astype('>f4')
    store = np.repeat(original, 2, axis=0)
    raw = store[::2]
    assert not raw.flags.c_contiguous
    expected_bytes = raw.tobytes(order='C')
    p = fixture(raw=raw)
    e = p.evidence()
    assert e['decoded_H40']['sha256'] == hashlib.sha256(expected_bytes).hexdigest()
    assert e['decoded_H40']['dtype'] == '>f4'
    before = p.preview().copy()
    store[:] = 0
    assert p.decoded_H40.tobytes(order='C') == expected_bytes
    np.testing.assert_array_equal(p.preview(), before)
    for value in [p.decoded_H40, p.mapped_H40, p.desired_knots, p.reference_knots,
                  p.desired_path, p.reference_path, p.query_TCP_source, p.base_pose_hold,
                  p.preview(), p.desired_at(0)]:
        with pytest.raises(ValueError):
            value.setflags(write=True)
        with pytest.raises(ValueError):
            value.flat[0] = 4
    e['clock']['knot_indices'][0] = -99
    assert p.evidence()['clock']['knot_indices'][0] == p.bridge_atoms
    json.dumps(p.evidence(), allow_nan=False)


def test_working_endpoints_match_original_strict_guard_bit_exactly():
    raw = rows([np.eye(4)] * 40, np.resize([0., .5, 1.], 40))
    p = fixture(raw=raw)
    np.testing.assert_array_equal(p.desired_knots[:, 9], teacher_physical_fraction(raw[:, 9]))
    assert p.desired_path[0, 9] == float(teacher_physical_fraction(1.))
    require_working_target(p.desired_path[:, 9])
    require_working_target(p.reference_path[:, 9])
    require_working_target(p.preview(0)[:, 9])


def test_entire_path_and_correction_catalogue_envelope_includes_z_rotation_grip():
    raw = rows([pose([0., 0., .03 if i == 35 else 0.], (1, 0, 0), .2 if i == 25 else 0.) for i in range(40)],
               np.linspace(0, 1, 40))
    p = a.TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, 1.)
    corrections = [[.01, -.02, .004], [-.006, .003, -.007]]
    e = a.trajectory_domain_envelope(p, corrections)
    for name, path in [('desired', p.desired_path), ('reference', p.reference_path)]:
        all_points = np.concatenate([path[:, :3]] + [path[:, :3] + c for c in corrections])
        np.testing.assert_allclose(e[name]['xyz_min_with_corrections_m'], all_points.min(axis=0))
        np.testing.assert_allclose(e[name]['xyz_max_with_corrections_m'], all_points.max(axis=0))
        assert e[name]['max_anchor_rotation_rad'] == pytest.approx(.2)
        assert e[name]['gripper_mechanical_fraction_min'] == pytest.approx(.002 / .044)
    assert e['desired']['xyz_max_with_corrections_m'][2] == pytest.approx(.034)
    assert not e['old_local_XY_qualification_reused']
    assert e['model_support'] == 'NOT_ESTABLISHED'
    assert not e['native_execution_authorized']
    for invalid in [[np.nan, 0., 0.], [0., 0., 0.], [[0., 0.]], [[]]]:
        with pytest.raises(ValueError):
            a.trajectory_domain_envelope(p, invalid)


@pytest.mark.parametrize('change', ['shape', 'nonfinite', 'grip_low', 'grip_high', 'rot_degenerate',
                                   'frame_reflection', 'frame_row', 'scale_zero', 'scale_bool', 'base_quaternion'])
def test_invalid_physical_input_rejected_without_clipping(change):
    p = fixture()
    args = dict(decoded_H40=p.decoded_H40.copy(), query_TCP_source=p.query_TCP_source.copy(),
                T_source_from_target=p.T_source_from_target.copy(), decision_TCP_target=p.decision_TCP_target.copy(),
                initial_reference_TCP_target=p.initial_reference_TCP_target.copy(), base_pose_hold=BASE.copy(), geometry_scale=1.)
    if change == 'shape': args['decoded_H40'] = args['decoded_H40'][:8]
    elif change == 'nonfinite': args['decoded_H40'][10, 0] = np.nan
    elif change == 'grip_low': args['decoded_H40'][20, 9] = -.00001
    elif change == 'grip_high': args['decoded_H40'][39, 9] = 1.00001
    elif change == 'rot_degenerate': args['decoded_H40'][15, 3:9] = 0
    elif change == 'frame_reflection': args['T_source_from_target'][:3, 0] *= -1
    elif change == 'frame_row': args['T_source_from_target'][3, 0] = .1
    elif change == 'scale_zero': args['geometry_scale'] = 0.
    elif change == 'scale_bool': args['geometry_scale'] = True
    elif change == 'base_quaternion': args['base_pose_hold'][6] = 2.
    with pytest.raises(ValueError):
        a.TrajectoryAdaptation(**args)


def test_geometry_relocation_and_fixed_init_do_not_change_integer_timing():
    p = fixture(.25)
    changed = a.TrajectoryAdaptation(p.decoded_H40, p.query_TCP_source, p.T_source_from_target,
                                   pose([-2, 4, 1], (1, 2, 3), .7), pose([3, -2, 1], (2, 3, -1), -.5), BASE, .25)
    assert changed.knot_indices == p.knot_indices
    assert changed.end_atom == p.end_atom


def test_import_constructs_no_model_and_imports_no_neural_or_physics_runtime():
    source = (f"import sys; sys.path.insert(0, {str(Path(a.__file__).parents[2])!r})\n"
              "from go2arx5_vla.world_model import fusion_trajectory_adapter\n"
              "if any(x in sys.modules for x in ['torch','mujoco','gr00t','transformers']):\n"
              "    raise SystemExit('unexpected neural/physics import')\n")
    completed = subprocess.run([sys.executable, '-c', source], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert not any(token in Path(a.__file__).read_text() for token in ['subprocess.', 'hf_hub_download(', 'decode_action('])
