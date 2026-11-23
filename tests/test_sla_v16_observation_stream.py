from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from observation_stream import ObservationStream


def stream():
    return ObservationStream(
        "episode",
        "Grasp the blue cube.",
        [0, 0, 0, 1, 0, 0, 0, 0.9],
        [0, 0, 0],
        np.eye(4),
    )


def sample(s, t, rgb, *, epoch="episode", finger=0.022, tip=None):
    img = np.full((256, 256, 3), rgb, np.uint8)
    s.observe(
        epoch,
        t,
        t * 0.002,
        img,
        img,
        np.eye(4),
        np.eye(4),
        np.eye(4) if tip is None else tip,
        [finger, finger],
    )
    img.fill(253)


def test_history_is_causal_exact_and_keeps_reset_after_eviction():
    s = stream()
    for i in range(11):
        sample(s, i * 500, i)
        if i == 1:
            assert s.policy_input()["images"]["cam_high"][:, 0, 0, 0].tolist() == [
                0
            ] * 9 + [1]
    p = s.policy_input()
    assert p["images"]["cam_high"][:, 0, 0, 0].tolist() == [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert (
        p["state"][15] == 0.5
        and p["state"][11] == 1
        and p["images"]["cam_left_wrist"].shape == (480, 640, 3)
    )
    p["images"]["cam_right_wrist"].fill(254)
    p["state"].fill(255)
    assert (
        s.policy_input()["images"]["cam_right_wrist"][0, 0, 0] == 10
        and s.policy_input()["state"][15] == 0.5
    )


def test_absent_lookback_and_other_epoch_are_not_substituted():
    s = stream()
    with pytest.raises(ValueError, match="reset"):
        sample(s, 1, 0)
    sample(s, 0, 0)
    sample(s, 503, 1)
    with pytest.raises(ValueError, match="tick 3"):
        s.policy_input()
    with pytest.raises(ValueError, match="reset"):
        sample(s, 504, 1, epoch="different")
    with pytest.raises(ValueError, match="clock"):
        sample(s, 502, 1)


def test_bad_transform_keeps_last_observation_and_actual_width_can_refuse():
    s = stream()
    sample(s, 0, 0)
    off = np.eye(4)
    off[0, 3] = 0.01
    with pytest.raises(ValueError, match="calibration"):
        sample(s, 5, 1, tip=off)
    assert s.policy_input()["images"]["cam_high"][:, 0, 0, 0].tolist() == [0] * 10
    sample(s, 5, 1, finger=0.045)
    with pytest.raises(ValueError, match="outside"):
        s.policy_input()


def test_stop_prevents_another_query_even_without_a_new_camera_frame():
    s = stream()
    sample(s, 0, 0)
    with pytest.raises(ValueError, match="wrong stop epoch"):
        s.stop("other")
    s.policy_input()
    s.stop("episode")
    s.stop("episode")
    with pytest.raises(ValueError, match="episode stopped"):
        s.policy_input()
    with pytest.raises(ValueError, match="episode stopped"):
        sample(s, 1, 0)
