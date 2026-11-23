import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from compiler import SourceCurve
from target_intent_comparison import errors
from target_labels import TargetExamples, history, task_labels


def test_action_time_pose_and_step_width_across_two_clocks():
    t = np.arange(40) / 20
    rot = Rotation.from_rotvec(np.column_stack([0 * t, 0 * t, t])).as_matrix()
    nominal = np.column_stack(
        [t, 2 * t, -t, rot[:, :2, :].reshape(40, 6), np.where(t < 0.05, 1.0, 0.0)]
    )
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_rotvec([0.25, 0.1, -0.2]).as_matrix()
    origin[:3, 3] = [0.2, -0.4, 0.7]
    tool = np.eye(4)
    tool[:3, 3] = [0.1424, 0, 0.0001057]
    raw, world = task_labels(
        nominal, origin, tool, np.array([0, 0, 0, 1, 0, 0, 0, 0.9])
    )
    source_t = np.arange(50) / 30
    assert np.allclose(world[:, :3, 3], np.array([source_t, 2 * source_t, -source_t]).T)
    assert np.allclose(world[:, :3, :3], Rotation.from_euler("z", source_t).as_matrix())
    assert raw[:3, 15].tolist() == [1.0, 1.0, 0.0]
    assert np.allclose(SourceCurve(raw, origin, tool).knots, world, atol=2e-7, rtol=0)
    nominal[-1, -1] = 1.01
    with pytest.raises(ValueError, match="nominal pose or width"):
        task_labels(nominal, origin, tool, raw[0, :8])


def test_policy_sample_keys_and_episode_bounded_history():
    assert history(5703) == [0, 1703, 2203, 2703, 3203, 3703, 4203, 4703, 5203, 5703]
    assert history(0) == [0] * 10
    ds = TargetExamples.__new__(TargetExamples)
    ds.meta = {"language": "original", "task_index": 0, "phase": "privileged"}
    ds.data = {
        "query_tick": [5703],
        "history_indices": [np.arange(10)],
        "stereo": np.full((10, 2, 2, 3, 3), 255, np.uint8),
        "state": np.ones((1, 16)),
        "nominal_action": np.ones((1, 50, 16)),
        "object_pose": "privileged",
    }
    x = ds[0]
    assert set(x) == {
        "observation.images.cam_high",
        "observation.images.cam_right_wrist",
        "observation.images.cam_left_wrist",
        "observation.state",
        "action",
        "prompt",
        "task_index",
    }
    assert x["observation.images.cam_high"].shape == (10, 2, 3, 3)
    assert x["observation.images.cam_left_wrist"].sum() == 0
    x["action"][:] = 22
    assert np.all(ds.data["nominal_action"] == 1)


def test_scored_translation_is_vector_rmse_and_short_window_is_half_open():
    goal = np.tile(np.eye(4), (50, 1, 1))
    moved = goal.copy()
    moved[15:, 0, 3] = 0.003
    moved[15:, 1, 3] = 0.004
    score = errors(moved, np.full(50, 0.04), goal, np.full(50, 0.03))
    assert score["position_mm"] == pytest.approx(5 * np.sqrt(35 / 50))
    assert score["first_15_position_mm"] == 0
    assert score["gripper_width_mm"] == pytest.approx(10)
    assert score["orientation_deg"] == 0
