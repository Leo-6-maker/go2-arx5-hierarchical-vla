import pytest

from go2arx5_vla.world_model.native_body_camera_capture_v1 import session_configuration
from go2arx5_vla.world_model.native_camera_transfer_v1 import CAMERA_ID
from go2arx5_vla.world_model.task_learning_grasp_v1 import CONSTRUCTION_ROLE


def spec():
    return dict(phase='collect-target-task-learning-grasp-v1', task_learning_role=CONSTRUCTION_ROLE,
                observation_camera_profile=CAMERA_ID,
                body_camera_session_v3=dict(version='BODY_BASE_EPISODE_SESSION_V3', mode='PERSISTENT_COMPATIBILITY'))


def test_explicit_new_construction_only():
    assert session_configuration({}) is None
    assert session_configuration(spec())['version'] == 'BODY_BASE_EPISODE_SESSION_V3'


@pytest.mark.parametrize('key,value', [('phase', 'other'), ('task_learning_role', 'SLA_TARGET500_NATIVE_EVALUATION_V1'),
                                      ('observation_camera_profile', 'legacy')])
def test_never_adopts_active_or_wrong_camera_role(key, value):
    item = spec(); item[key] = value
    with pytest.raises(ValueError):
        session_configuration(item)
