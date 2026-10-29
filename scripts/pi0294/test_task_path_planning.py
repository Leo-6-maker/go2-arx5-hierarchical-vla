"""Boundary checks with real MuJoCo kinematics; optional CPU dependency only."""
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from .task_path_planning import load_saved_native, native_configuration_witness, planned_future_configuration


def native_fixture():
    # A small actual MuJoCo model tests interpolation/collision accounting without
    # shipping private robot assets or pretending to be the full target robot.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="obstacle" type="sphere" pos="0.1 0 0" size="0.02"/>
      <body><joint type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="arm" type="sphere" size="0.02"/>
        <site name="tcp"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    return SimpleNamespace(model=model, data=data, qpos=np.array([-.1]),
        joint_addresses=np.zeros(6, dtype=int), arm_limits=np.tile([-1., 1.], (6, 1)), site=0)


def test_midpath_collision_is_not_hidden_by_clear_endpoints():
    native = native_fixture()
    # All six witness coordinates use the fixture's one slide; endpoint checks
    # alone miss the obstacle at x=.1. Dense actual FK must detect the crossing.
    q = np.tile([-.1, .3], (6, 1)).T
    before = native.qpos.copy()
    check, tcp = native_configuration_witness(native, q)
    assert check['status'] == 'DISCRETE_NATIVE_CONFIGURATION_REJECT'
    assert len(check['violations']) > 0 and check['samples'] >= 21
    assert tcp.shape == (check['samples'], 4, 4)
    np.testing.assert_array_equal(native.qpos, before)
    np.testing.assert_array_equal(native.data.qpos, before)
    assert check['continuous_swept_collision_proved'] is False
    assert check['physical_execution_qualified'] is False


def test_outside_joint_limit_rejects_without_clipping_or_mutation():
    native = native_fixture()
    with pytest.raises(ValueError, match='JOINT_LIMIT_REJECT'):
        native_configuration_witness(native, np.full((1, 6), 1.01))
    np.testing.assert_array_equal(native.qpos, [-.1])


def test_wrong_model_pin_rejected_before_binary_deserialization(tmp_path):
    path = tmp_path/'not-a-model'
    path.write_bytes(b'opaque private bytes')
    with pytest.raises(ValueError, match='MODEL_HASH_MISMATCH'):
        load_saved_native(path, '0'*64, {})


def test_full_native_dimensions_required_even_with_correct_pin(tmp_path):
    native = native_fixture()
    path = tmp_path/'fixture.mjb'
    mujoco.mj_saveModel(native.model, str(path), None)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='FULL_GO2_ARX5_TASK_MODEL_REQUIRED'):
        load_saved_native(path, sha, {})


def test_future_stage_witness_is_separate_and_cannot_masquerade_as_observed_state():
    native = native_fixture()
    native.mount = np.eye(4)
    native.state_role = 'SAVED_ACTUAL_OBSERVATION'
    before = native.qpos.copy()
    future = planned_future_configuration(native, np.full(6, -.05))
    assert future.state_role == 'PLANNED_FUTURE_CONFIGURATION_NOT_OBSERVED'
    assert future.data is not native.data
    np.testing.assert_array_equal(native.qpos, before)
    np.testing.assert_allclose(future.qpos, [-.05])
