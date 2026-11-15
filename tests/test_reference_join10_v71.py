import numpy as np
import pytest

from scripts.sla.reference_join10_v71 import connect, _exp
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows


def inputs():
    past = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    past[0, :3, 3] = [.2, -.1, .3]
    past[1, :3, 3] = [.2001, -.1002, .3003]
    past[0, :3, :3] = _exp(np.array([.1, -.03, .07]))
    past[1, :3, :3] = past[0, :3, :3] @ _exp(np.array([.0001, .0002, -.0003]))
    t = np.arange(1, 41)*.05
    raw = np.zeros((40, 10))
    raw[:, :3] = past[-1, :3, 3] + np.stack([.001*t*t, -.002*t, .003*np.sin(t)], axis=1)
    raw[:, 3:9] = rotation_matrix_to_rot6d_rows(np.stack([
        past[-1, :3, :3] @ _exp(np.array([.001*x, .002*x*x, 0])) for x in t]))
    raw[:, 9] = np.linspace(.1, .4, 40)
    return raw, past


def test_first_two_points_continue_reference_motion_and_keep_suffix_grip_and_inputs():
    raw, past = inputs(); saved, history = raw.copy(), past.copy()
    result = connect(raw, past)
    velocity_step = past[-1, :3, 3]-past[-2, :3, 3]
    np.testing.assert_allclose(result[:2, :3],
        past[-1, :3, 3]+np.arange(1, 3)[:, None]*velocity_step, rtol=0, atol=1e-14)
    increment = past[-2, :3, :3].T @ past[-1, :3, :3]
    desired = np.stack([past[-1, :3, :3] @ increment,
                        past[-1, :3, :3] @ increment @ increment])
    np.testing.assert_allclose(rot6d_rows_to_matrix(result[:2, 3:9]), desired, rtol=0, atol=1e-13)
    assert np.array_equal(result[9:], raw[9:]) and np.array_equal(result[:, 9], raw[:, 9])
    assert np.array_equal(raw, saved) and np.array_equal(past, history)


def test_connection_is_equivariant_to_a_rigid_world_frame_change():
    raw, past = inputs(); result = connect(raw, past)
    rotation = _exp(np.array([.3, -.2, .15])); translation = np.array([1., -2., .5])
    changed = raw.copy(); changed[:, :3] = raw[:, :3] @ rotation.T + translation
    changed[:, 3:9] = rotation_matrix_to_rot6d_rows(rotation @ rot6d_rows_to_matrix(raw[:, 3:9]))
    moved_history = past.copy(); moved_history[:, :3, 3] = past[:, :3, 3] @ rotation.T+translation
    moved_history[:, :3, :3] = rotation @ past[:, :3, :3]
    actual = connect(changed, moved_history)
    np.testing.assert_allclose(actual[:, :3], result[:, :3] @ rotation.T+translation, rtol=0, atol=2e-13)
    np.testing.assert_allclose(rot6d_rows_to_matrix(actual[:, 3:9]),
        rotation @ rot6d_rows_to_matrix(result[:, 3:9]), rtol=0, atol=2e-13)


@pytest.mark.parametrize('kind', ['short_horizon', 'one_reference', 'nan', 'reflection'])
def test_invalid_inputs_are_rejected_without_clipping(kind):
    raw, past = inputs()
    if kind == 'short_horizon': raw = raw[:10]
    if kind == 'one_reference': past = past[:1]
    if kind == 'nan': raw[1, 0] = np.nan
    if kind == 'reflection': past[0, 0, :3] *= -1
    with pytest.raises(ValueError): connect(raw, past)
