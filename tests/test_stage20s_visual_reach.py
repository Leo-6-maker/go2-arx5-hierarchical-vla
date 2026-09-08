import numpy as np

from go2arx5_vla.simulation import (
    INITIAL_LOCAL_POSITION,
    SIM_VISUAL_REACH_INSTRUCTION,
    build_visual_reach_raw,
    default_visual_reach_specs,
)
from go2arx5_vla.data import SynchronizerV1


def test_visual_reach_fixture_has_explicit_state_action_and_source_timing(tmp_path):
    spec = default_visual_reach_specs(1)[0]
    raw = build_visual_reach_raw(spec, tmp_path / "frames")
    synchronization = SynchronizerV1().synchronize(raw)

    assert raw.task["instruction"] == SIM_VISUAL_REACH_INSTRUCTION
    assert raw.state_eef_9d.shape == (61, 9)
    assert raw.desired_eef_9d.shape == (61, 9)
    assert raw.desired_gripper.shape == (61, 1)
    assert raw.achieved_eef_9d.shape == (61, 9)
    assert raw.camera_streams["body"].effective_rate_hz == 20.0
    assert synchronization.metrics["max_abs_skew_s"] == 0.0
    assert raw.metadata["hardware_control"] is False
    np.testing.assert_allclose(raw.state_eef_9d[0, :3], INITIAL_LOCAL_POSITION)
    np.testing.assert_allclose(raw.desired_eef_9d[0], raw.state_eef_9d[0])
    assert not np.array_equal(raw.desired_eef_9d[1:, :3], raw.achieved_eef_9d[1:, :3])


def test_progressive_fixture_increases_task_space_demand_without_initial_jump(tmp_path):
    spec = default_visual_reach_specs(1, family="PROGRESSIVE_BASE_RELOCATION")[0]
    raw = build_visual_reach_raw(spec, tmp_path / "frames")
    initial_jump = np.linalg.norm(raw.desired_eef_9d[0, :3] - raw.state_eef_9d[0, :3])
    target_distance = np.linalg.norm(raw.desired_eef_9d[39, :3] - raw.state_eef_9d[0, :3])

    assert initial_jump == 0.0
    assert target_distance > 0.07
    assert np.all(np.diff(raw.native_timestamps) > 0.0)
    assert raw.frame_graph["pose_frame"] == "NATIVE_WBC_RELATIVE_REFERENCE"
