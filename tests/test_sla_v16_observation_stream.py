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


def actual_actions(obs):
    return np.repeat(obs["state"][None], 50, axis=0)


def verified_intervals():
    return [
        dict(
            episode="native-one",
            epoch="episode",
            sequence=10 + i,
            start_tick=i * 25,
            end_tick=25 + i * 25,
            previous_student_command_m=0.038 + i * 0.0005,
            previous_gripper_wire_sha256=str(i),
            installed_reference_sha256=str(i + 5),
            safety_valid=True,
            terminal_at_end=False,
            controller_reference_evidence="derived",
        )
        for i in range(3)
    ]


def test_inflight_model_return_keeps_query_frame_and_binds_latest_receipt():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from policy_query import policy_response

    s = stream()
    sample(s, 0, 0)
    sample(s, 25, 1)
    key, data = s.begin_query()
    entered, release = Event(), Event()

    def predict(obs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("worker stalled")
        return actual_actions(obs)

    with ThreadPoolExecutor(1) as worker:
        job = worker.submit(policy_response, key, data, predict)
        try:
            assert entered.wait(10)
            img = np.zeros((256, 256, 3), np.uint8)
            base = np.eye(4)
            base[0, 3] = 0.07
            s.observe("episode", 75, 0.15, img, img, base, base, base, [0.039, 0.039])
        finally:
            release.set()
        s.finish_query(*job.result())
    ledger = verified_intervals()
    with pytest.raises(ValueError, match="feedback gap"):
        s.take_for_compile(75, [ledger[0], ledger[2]])
    with pytest.raises(ValueError, match="feedback gap"):
        s.take_for_compile(75, [dict(row, epoch="other") for row in ledger])
    with pytest.raises(ValueError, match="boundary"):
        s.take_for_compile(50, ledger[:2])
    result = s.take_for_compile(75, ledger)
    assert result["query_tick"] == 25 and result["latest_sensor_tick"] == 75
    assert np.allclose(result["curve"].knots[:, :3, 3], 0)
    assert result["feedback"]["consumed_after_query"] == 2
    assert result["feedback"]["previous_student_command_m"] == pytest.approx(0.039)
    assert s.take_proposal() is None


def test_invalid_response_does_not_displace_newer_valid_candidate():
    from policy_query import policy_response

    s = stream()
    sample(s, 0, 0)
    old, old_obs = s.begin_query()
    sample(s, 25, 1)
    new, new_obs = s.begin_query()
    s.finish_query(*policy_response(new, new_obs, actual_actions))
    with pytest.raises(ValueError, match="newer proposal"):
        s.finish_query(*policy_response(old, old_obs, actual_actions))
    key, obs = s.begin_query()
    obs["state"][15] = 0.21
    with pytest.raises(ValueError, match="different observation"):
        s.finish_query(*policy_response(key, obs, actual_actions))
    assert s.take_proposal()["query_id"] == new


def test_native_terminal_closes_other_pending_model_work():
    from policy_query import policy_response

    s = stream()
    sample(s, 0, 0)
    sample(s, 25, 1)
    first = s.begin_query()
    other = s.begin_query()
    s.finish_query(*policy_response(*first, actual_actions))
    ledger = verified_intervals()
    ledger[1]["terminal_at_end"] = True
    with pytest.raises(ValueError, match="terminal or unsafe"):
        s.take_for_compile(50, ledger[:2])
    with pytest.raises(ValueError, match="episode stopped"):
        s.finish_query(*policy_response(*other, actual_actions))


def test_stop_receipt_during_inference_is_retained_without_a_ready_proposal():
    from policy_query import policy_response

    s = stream()
    sample(s, 0, 0)
    sample(s, 25, 1)
    query = s.begin_query()
    intervals = verified_intervals()[:2]
    intervals[-1]["terminal_at_end"] = True
    with pytest.raises(ValueError, match="terminal or unsafe"):
        s.take_for_compile(50, intervals)
    with pytest.raises(ValueError, match="episode stopped"):
        s.finish_query(*policy_response(*query, actual_actions))
