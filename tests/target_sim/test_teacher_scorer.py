"""CPU synthetic contracts only: these fixtures never assert physical readiness."""

import copy
import json

import numpy as np
import pytest

from go2arx5_vla.target_sim.recorder import Recorder, command_target
from go2arx5_vla.target_sim.scorer import BOOL_FIELDS, physical_grasp, score_episode
from go2arx5_vla.target_sim.teacher import Teacher
from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_to_matrix


def pose(xyz=(0, 0, 0)):
    result = np.eye(4)
    result[:3, 3] = xyz
    return result


INITIAL = pose((-.16, 0, .42))
GOAL = np.asarray((.16, 0, .42))
GRASP_OFFSET = pose((0, 0, .005))


def physical_fixture():
    times = np.arange(641) / 20
    n = len(times)
    objects = np.tile(INITIAL, (n, 1, 1))
    grasp = (times >= 1) & (times <= 4)
    objects[(times >= 1.5) & (times <= 4), 2, 3] += .08
    objects[times >= 2.5, 0, 3] = GOAL[0]
    objects[times > 4, 2, 3] = GOAL[2]
    data = {"time_s": times, "object_pose": objects,
            "tcp_world": np.tile(pose((.16, 0, .55)), (n, 1, 1)),
            "object_linear_velocity": np.zeros((n, 3)), "object_angular_velocity": np.zeros((n, 3))}
    for key in BOOL_FIELDS:
        data[key] = np.zeros(n, dtype=bool)
    for key in ("object_present", "contact_valid", "gripper_state_valid", "safety_valid"):
        data[key][:] = True
    for key in ("native_grasp", "left_object_contact", "right_object_contact"):
        data[key] = grasp.copy()
    return data


def score(data, **kwargs):
    return score_episode(data, goal_center=GOAL, initial_object_pose=INITIAL,
                         table_height_m=.40, evidence_source="CPU_SYNTHETIC_NOT_PHYSICS", **kwargs)


def test_scorer_positive_and_exact_41_hold_samples():
    result = score(physical_fixture())
    assert result["persistent_success"]
    assert result["complete_32s"] and result["clock_valid"] and result["safety_valid"]
    assert result["hold_observed"] == result["hold_pass"] == 41
    assert all(result["history"].values())
    assert result["selected"] == result["deployment"] == "NONE"


@pytest.mark.parametrize("key", ["object_present", "contact_valid", "gripper_state_valid", "native_grasp", "left_object_contact", "right_object_contact"])
def test_grasp_requires_all_calibrated_physical_evidence(key):
    row = {name: True for name in ("object_present", "contact_valid", "gripper_state_valid", "native_grasp", "left_object_contact", "right_object_contact")}
    assert physical_grasp(row)
    row[key] = False
    assert not physical_grasp(row)


@pytest.mark.parametrize("scenario", ["push", "single_side", "no_lift", "brief_lift", "no_carry", "held", "no_withdraw", "wrong_zone", "fast_linear", "fast_angular", "high", "low", "rotated_corners", "unsafe", "abort", "joint_limit", "reset", "takeover", "unknown_safety", "missing_contact"])
def test_scorer_physical_negative_controls(scenario):
    data = physical_fixture()
    if scenario == "push":
        data["native_grasp"][:] = False
    elif scenario == "single_side":
        data["right_object_contact"][:] = False
    elif scenario in {"no_lift", "brief_lift"}:
        data["object_pose"][:, 2, 3] = .42
        if scenario == "brief_lift":
            data["object_pose"][30:35, 2, 3] = .50
    elif scenario == "no_carry":
        data["object_pose"][data["native_grasp"], 0, 3] = INITIAL[0, 3]
    elif scenario == "held":
        for key in ("native_grasp", "left_object_contact", "right_object_contact"):
            data[key][560] = True
    elif scenario == "no_withdraw":
        data["tcp_world"][560, :3, 3] = GOAL
    elif scenario == "wrong_zone":
        data["object_pose"][560, 0, 3] = -.16
    elif scenario == "fast_linear":
        data["object_linear_velocity"][560, 0] = .020001
    elif scenario == "fast_angular":
        data["object_angular_velocity"][600, 0] = .100001
    elif scenario in {"high", "low"}:
        data["object_pose"][560, 2, 3] += .005001 * (1 if scenario == "high" else -1)
    elif scenario == "rotated_corners":
        data["object_pose"][560, 0, 3] += .045
        theta = np.pi / 4
        data["object_pose"][560, :3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    elif scenario in {"unsafe", "abort", "joint_limit", "reset", "takeover"}:
        key = {"unsafe": "illegal_contact", "abort": "native_abort", "joint_limit": "joint_limit_penetration"}.get(scenario, scenario)
        data[key][620] = True  # Even after the entire hold interval, safety still matters.
    elif scenario == "unknown_safety":
        data["safety_valid"][2] = False
    elif scenario == "missing_contact":
        data["contact_valid"][560] = False
    assert not score(data)["persistent_success"]


@pytest.mark.parametrize("index", [560, 579, 600])
def test_every_hold_sample_including_endpoints_required(index):
    data = physical_fixture()
    data["object_pose"][index, 0, 3] = -.16
    result = score(data)
    assert not result["persistent_success"] and result["hold_pass"] == 40


def test_no_early_success_no_sample_interpolation_and_no_fake_complete():
    data = physical_fixture()
    assert not score(data, complete=False)["persistent_success"]
    for index in (579, 640):
        missing = {key: np.delete(value, index, axis=0) for key, value in data.items()}
        assert not score(missing)["persistent_success"]
    data["time_s"][600] = data["time_s"][599]
    with pytest.raises(ValueError):
        score(data)


@pytest.mark.parametrize("bad", ["nan", "rotation", "bool", "missing", "invalid_goal"])
def test_scorer_rejects_invalid_sensor_data(bad):
    data = physical_fixture()
    if bad == "nan":
        data["object_linear_velocity"][100, 0] = np.nan
    elif bad == "rotation":
        data["object_pose"][100, 0, 0] = 2
    elif bad == "bool":
        data["native_grasp"] = data["native_grasp"].astype(float)
    elif bad == "missing":
        del data["safety_valid"]
    with pytest.raises((ValueError, KeyError)):
        if bad == "invalid_goal":
            score_episode(data, goal_center=[.16, 0, .43], initial_object_pose=INITIAL, table_height_m=.40)
        else:
            score(data)


def test_scorer_initial_and_scene_identity_cannot_be_rewritten():
    data = physical_fixture()
    with pytest.raises(ValueError):
        score(data, complete="false")
    changed = INITIAL.copy()
    changed[2, 3] -= .1
    with pytest.raises(ValueError):
        score_episode(data, goal_center=GOAL, initial_object_pose=changed, table_height_m=.4)
    with pytest.raises(ValueError):
        score_episode(data, goal_center=[-.05, 0, .42], initial_object_pose=INITIAL, table_height_m=.4)


def feedback(tcp, obj, grasp=False, opening=1.):
    return {"tcp_world": tcp, "object_pose": obj, "gripper_fraction": opening,
            "object_linear_velocity": np.zeros(3), "object_angular_velocity": np.zeros(3),
            "object_present": True, "contact_valid": True, "gripper_state_valid": True,
            "native_grasp": grasp, "left_object_contact": grasp, "right_object_contact": grasp,
            "safety_valid": True, "native_abort": False, "illegal_contact": False,
            "joint_limit_penetration": False}


def teacher():
    return Teacher(INITIAL @ GRASP_OFFSET, INITIAL, GOAL, GRASP_OFFSET,
                   [-1, -1, 0], [1, 1, 1])


def test_teacher_all_stages_use_actual_feedback_without_restart():
    instance = teacher()
    visited = []
    # Explicit synthetic sensor controls test the state machine; no physics is run.
    for tick in range(641):
        stage = instance.stage
        obj = INITIAL.copy()
        tcp = INITIAL @ GRASP_OFFSET
        grasp = stage in {"close", "lift", "carry", "lower"}
        if stage == "approach":
            tcp[2, 3] += .05
        elif stage == "lift":
            obj[2, 3] += .08
            tcp = obj @ GRASP_OFFSET
        elif stage in {"carry", "lower", "release", "withdraw", "hold"}:
            obj[:3, 3] = GOAL
            if stage == "carry":
                obj[2, 3] += .08
            tcp = obj @ GRASP_OFFSET
            if stage in {"withdraw", "hold"}:
                tcp[2, 3] += .10
        command = instance.query(tick / 20, feedback(tcp, obj, grasp, 0. if grasp else 1.))
        assert command.stage_before == stage and not command.failed
        if not visited or visited[-1] != command.stage_before:
            visited.append(command.stage_before)
        assert command.gripper_fraction == (0. if stage in {"close", "lift", "carry", "lower"} else 1.)
    assert visited == list(Teacher.stages)
    assert instance.stage == "hold" and instance.last_time == 32
    assert len(instance.events) == 8


def test_teacher_arrival_and_grasp_not_inferred_from_issued_commands():
    instance = teacher()
    actual = INITIAL @ GRASP_OFFSET
    for tick in range(41):
        instance.query(tick / 20, feedback(actual, INITIAL))
    assert instance.stage == "approach"  # Command reaches approach; actual TCP never does.
    instance = teacher()
    instance.stage = "close"
    for tick in range(41):
        row = feedback(actual, INITIAL, grasp=True, opening=0.)
        row["right_object_contact"] = False
        command = instance.query(tick / 20, row)
    assert instance.stage == "close" and not command.failed
    command = instance.query(2.05, row)
    assert command.failed and command.failure_reason == "PHYSICAL_GRASP_TIMEOUT"


def test_teacher_object_relative_grasp_and_calibrated_feedback_relation():
    instance = teacher()
    shifted = INITIAL.copy()
    shifted[0, 3] += .01
    command = instance.query(0., feedback(INITIAL @ GRASP_OFFSET, shifted))
    assert command.tcp_world[0, 3] > INITIAL[0, 3]
    instance = teacher()
    instance.stage = "close"
    measured = GRASP_OFFSET.copy()
    measured[0, 3] = .003
    for tick in range(7):
        instance.query(tick / 20, feedback(INITIAL @ measured, INITIAL, True, .2))
    assert instance.stage == "lift"
    np.testing.assert_allclose(instance.object_to_tcp, measured, atol=1e-12)


def test_teacher_translation_rotation_speed_envelope_and_no_unobserved_lift():
    instance = teacher()
    previous = instance.command.copy()
    theta = .5
    obj = INITIAL.copy()
    obj[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    command = instance.query(0, feedback(previous, obj))
    assert np.linalg.norm(command.tcp_world[:3, 3] - previous[:3, 3]) <= .15 * .05 + 1e-12
    assert geodesic_angle(previous[:3, :3], command.tcp_world[:3, :3]) <= .05 + 1e-8
    instance = teacher()
    instance.stage = "lift"
    for tick in range(20):
        instance.query(tick / 20, feedback(INITIAL @ GRASP_OFFSET, INITIAL, True, .1))
    assert instance.stage == "lift"
    row = feedback(INITIAL @ GRASP_OFFSET, INITIAL, False, .1)
    command = instance.query(1., row)
    assert command.failed and command.failure_reason == "PHYSICAL_GRASP_LOST"
    assert command.gripper_fraction == .1
    np.testing.assert_equal(command.tcp_world, row["tcp_world"])


def test_teacher_rejects_outside_envelope_clock_nan_and_native_abort():
    instance = teacher()
    outside = INITIAL.copy()
    outside[0, 3] = 2.
    command = instance.query(0., feedback(INITIAL @ GRASP_OFFSET, outside))
    assert command.failed and command.failure_reason == "POSE_ENVELOPE_REJECTED"
    instance = teacher()
    row = feedback(INITIAL @ GRASP_OFFSET, INITIAL)
    row["native_abort"] = True
    assert instance.query(0., row).failure_reason == "NATIVE_SAFETY_STOP"
    for timestamp in (.1, np.nan, 32.05):
        with pytest.raises(ValueError):
            teacher().query(timestamp, row)
    row["native_abort"] = False
    row["object_present"] = False
    assert teacher().query(0., row).failure_reason == "PHYSICAL_SENSOR_INVALID"


def recorder():
    return Recorder("cpu-287000", 287000, "CPU", {key: "CPU_SYNTHETIC_NONE" for key in ("asset", "teacher", "controller", "codec")})


def sample(index):
    timestamp = index / 20
    camera = {"path": "real-runtime-must-save-this.png", "timestamp_s": timestamp, "valid": True, "source": "target_render"}
    observation = {"body_rgb": camera.copy(), "wrist_rgb": camera.copy(), "tcp_xyz": [0., 0., .6],
                   "tcp_quaternion_xyzw": [0., 0., 0., 1.], "gripper": .8, "instruction": "Put the red cube in the blue zone."}
    # Commanded and actual are deliberately different; achieved TCP is never a label.
    action = {"command_id": f"actual-{index}", "issue_time_s": timestamp + .005,
              "tcp_xyz": [index / 1000., 0., .5], "tcp_quaternion_xyzw": [0., 0., 0., 1.],
              "gripper": .25, "accepted": True, "valid": True, "actually_sent": True,
              "frame_id": "world", "tcp_id": "x5_gripper_center"}
    execution = {"command_id": action["command_id"], "achieved_time_s": timestamp + .025,
                 "tcp_xyz": [.99, .99, .99], "base": None,
                 "valid_mask": {"tcp_xyz": True, "base": False}}
    return {"observation_time_s": timestamp, "policy_observation": observation, "action_target": action,
            "execution_feedback": execution, "privileged_diagnostics": {"object_pose": INITIAL.copy(), "teacher_stage": "lift"}}


def test_recorder_H40_real_future_actions_and_prefix_is_eight():
    instance = recorder()
    for index in range(42):
        instance.append(**sample(index))
    windows = instance.windows()
    assert len(windows) == 2
    window = windows[0]
    assert window["action_target"].shape == (40, 10)
    assert window["execution_prefix"] == 8
    assert window["future_command_ids"] == [f"actual-{index}" for index in range(1, 41)]
    np.testing.assert_allclose(window["action_target"][:, 0], np.arange(1, 41) / 1000)
    np.testing.assert_equal(window["action_target"][:, -1], .25)
    np.testing.assert_allclose(rot6d_to_matrix(window["action_target"][:, 3:9]), np.tile(np.eye(3), (40, 1, 1)))
    assert window["rot6d_layout"] == "FIRST_TWO_COLUMNS"
    assert "privileged_diagnostics" not in window


@pytest.mark.parametrize("kind", ["reset", "abort", "takeover"])
def test_recorder_H40_cannot_cross_boundaries(kind):
    instance = recorder()
    for index in range(25):
        instance.append(**sample(index))
    instance.boundary(kind, 1.225)
    if kind in {"reset", "abort"}:
        with pytest.raises(ValueError):
            instance.append(**sample(25))
        with pytest.raises(ValueError):
            instance.boundary("takeover", 1.25)
    else:
        for index in range(25, 66):
            instance.append(**sample(index))
        assert len(instance.windows()) == 1
        assert instance.windows()[0]["anchor_index"] == 25
    assert all(window["anchor_index"] >= 25 for window in instance.windows())


def test_recorder_sample_cannot_be_backdated_before_takeover():
    instance = recorder()
    instance.append(**sample(0))
    instance.boundary("takeover", .075)
    with pytest.raises(ValueError):
        instance.append(**sample(1))


@pytest.mark.parametrize("bad", ["future_rgb", "oracle", "unsent", "rejected", "mismatched_feedback", "zero_missing", "early_feedback", "nonunit_quat", "duplicate"])
def test_recorder_trust_boundary_rejections(bad):
    instance = recorder()
    row = sample(0)
    if bad == "future_rgb":
        row["policy_observation"]["body_rgb"]["timestamp_s"] = .1
    elif bad == "oracle":
        row["policy_observation"]["object_pose"] = INITIAL
    elif bad == "unsent":
        row["action_target"]["actually_sent"] = False
    elif bad == "rejected":
        row["action_target"]["accepted"] = False
    elif bad == "mismatched_feedback":
        row["execution_feedback"]["command_id"] = "other"
    elif bad == "zero_missing":
        row["execution_feedback"]["base"] = [0., 0., 0.]
    elif bad == "early_feedback":
        row["execution_feedback"]["achieved_time_s"] = 0.
    elif bad == "nonunit_quat":
        row["action_target"]["tcp_quaternion_xyzw"] = [0, 0, 0, 2]
    elif bad == "duplicate":
        instance.append(**row)
        row = sample(1)
        row["action_target"]["command_id"] = "actual-0"
        row["execution_feedback"]["command_id"] = "actual-0"
    with pytest.raises(ValueError):
        instance.append(**row)


def test_recorder_snapshots_preserves_rejected_raw_and_no_cross_gap_labels():
    instance = recorder()
    original = sample(0)
    instance.append(**original)
    original["action_target"]["tcp_xyz"][0] = 999
    assert instance.rows[0]["action_target"]["tcp_xyz"][0] == 0
    for index in range(1, 42):
        row = sample(index)
        if index == 20:
            row["action_target"]["valid"] = row["action_target"]["accepted"] = False
        instance.append(**row)
    assert not instance.windows()
    instance = recorder()
    for index in range(42):
        instance.append(**sample(index if index < 20 else index + 1))
    assert not instance.windows()


def test_recorder_independent_physics_write_readback_and_no_overwrite(tmp_path):
    instance = recorder()
    data = physical_fixture()
    for index in range(641):
        instance.append_physics({key: value[index] for key, value in data.items()})
    assert score(instance.physics_arrays())["persistent_success"]
    instance.append(**sample(0))
    directory = tmp_path / "original"
    manifest = instance.write(directory)
    assert manifest["physics_count"] == 641 and manifest["physical_success"] is None
    assert manifest["selected"] == manifest["deployment"] == "NONE"
    stored = json.loads((directory / "physics.json").read_text(encoding="utf-8"))
    assert score(stored)["persistent_success"]
    row = json.loads((directory / "samples.jsonl").read_text(encoding="utf-8"))
    assert "object_pose" not in row["policy_observation"]
    assert row["execution_feedback"]["base"] is None
    assert row["execution_feedback"]["installed_reference"] is None
    assert row["execution_feedback"]["valid_mask"]["installed_reference"] is False
    with pytest.raises(FileExistsError):
        instance.write(directory)


def test_command_envelope_preserves_teacher_native_pose_without_sending():
    instance = teacher()
    command = instance.query(0, feedback(INITIAL @ GRASP_OFFSET, INITIAL))
    action = command_target(command, command_id="cpu", issue_time_s=.001, accepted=False,
                            actually_sent=False, tcp_id="gripper_center")
    assert not action["valid"] and not action["actually_sent"]
    np.testing.assert_equal(action["tcp_xyz"], command.tcp_world[:3, 3])
