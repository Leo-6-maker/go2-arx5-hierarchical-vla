"""CPU contracts only: synthetic tests never claim official processor PASS."""
import copy
import json

import numpy as np
import pytest

from go2arx5_vla.target_sim.dataset import (enforce_roundtrip, load_records, pose10,
                                         training_population, validate_manifest, windows)
from go2arx5_vla.trajectory.so3 import (quat_to_matrix, rotation_matrix_to_rot6d,
                                     rotation_matrix_to_rot6d_rows)


def records(tmp_path, count=42):
    from PIL import Image
    rows = []
    for index in range(count):
        time_s = index * .05
        pose = dict(tcp_xyz=[time_s * .01, .1, .5], tcp_quaternion_xyzw=[.1, .2, .3, np.sqrt(.86)], gripper=.5)
        obs = dict(pose, instruction="Place the red cube in the blue target area.")
        for camera in ("body_rgb", "wrist_rgb"):
            name = f"{camera}-{index}.png"
            Image.fromarray(np.full((2, 2, 3), index, dtype=np.uint8)).save(tmp_path / name)
            obs[camera] = dict(path=name, timestamp_s=time_s, valid=True, source="target_render")
        rows.append(dict(episode_id="TRAIN-287100", layout_id=287100, split="TRAIN", task_id="GO2_PLACE_V1",
                         identity=dict(asset="a", teacher="t", controller="c", codec="q"),
                         observation_time_s=time_s, segment_id=0, policy_observation=obs,
                         action_target=dict(pose, command_id=f"c{index}", issue_time_s=time_s,
                                            accepted=True, valid=True, actually_sent=True, frame_id="world", tcp_id="tcp"),
                         execution_feedback=dict(command_id=f"c{index}", base=None, valid_mask={"base": False}), privileged_diagnostics={}))
    return rows


def test_native_layout_and_roundtrip():
    value = dict(tcp_xyz=[.1, .2, .3], tcp_quaternion_xyzw=[.1, .2, .3, np.sqrt(.86)], gripper=.6)
    encoded = pose10(value)
    matrix = quat_to_matrix(value["tcp_quaternion_xyzw"])
    assert np.allclose(encoded[3:9], rotation_matrix_to_rot6d_rows(matrix))
    assert not np.allclose(encoded[3:9], rotation_matrix_to_rot6d(matrix))
    action = np.tile(encoded, (40, 1))
    assert enforce_roundtrip(action, action)["SO3_rad"] < 1e-7
    for bad in (np.zeros(6), [1, 0, 0, 2, 0, 0], [np.nan] * 6):
        broken = action.copy(); broken[0, 3:9] = bad
        with pytest.raises(ValueError):
            enforce_roundtrip(action, broken)
    broken = action.copy(); broken[0, 0] += 2e-5
    with pytest.raises(ValueError):
        enforce_roundtrip(action, broken)
    with pytest.raises(ValueError):
        pose10(dict(value, tcp_quaternion_xyzw=[0, 0, 0, 0]))


def test_actual_future_and_boundaries(tmp_path):
    rows = records(tmp_path)
    samples = windows(rows)
    assert len(samples) == 2 and samples[0]["action"].shape == (40, 10)
    assert np.array_equal(samples[0]["action"][0], pose10(rows[1]["action_target"]))
    changed = copy.deepcopy(rows)
    changed[20]["execution_feedback"]["tcp_xyz"] = [9, 9, 9]
    assert np.array_equal(windows(changed)[0]["action"], samples[0]["action"])
    for field in ("accepted", "valid", "actually_sent"):
        changed = copy.deepcopy(rows); changed[20]["action_target"][field] = False
        assert windows(changed) == []
    for field in ("reset", "native_abort", "takeover"):
        changed = copy.deepcopy(rows); changed[20]["privileged_diagnostics"][field] = True
        assert windows(changed) == []
    changed = copy.deepcopy(rows); changed[20]["segment_id"] = 1
    assert windows(changed) == []
    assert windows(rows[:40]) == []


def test_record_input_isolation_and_clock(tmp_path):
    rows = records(tmp_path)
    episode = {key: rows[0][key] for key in ("episode_id", "layout_id", "split")}
    path = tmp_path / "samples.jsonl"
    def save(material):
        path.write_text("".join(json.dumps(row) + "\n" for row in material), encoding="utf-8")
    save(rows)
    assert len(load_records(path, episode)) == len(rows)
    for field in ("object_pose", "teacher_stage", "BASE", "success"):
        changed = copy.deepcopy(rows); changed[0]["policy_observation"][field] = 1
        save(changed)
        with pytest.raises(ValueError):
            load_records(path, episode)
    changed = copy.deepcopy(rows); changed[1]["observation_time_s"] = 0
    save(changed)
    with pytest.raises(ValueError):
        load_records(path, episode)
    changed = copy.deepcopy(rows); changed[1]["policy_observation"]["body_rgb"]["timestamp_s"] = .1
    save(changed)
    with pytest.raises(ValueError):
        load_records(path, episode)
    report, samples = validate_manifest(tmp_path / "missing_actual_data")
    assert report["status"] == "NOT_READY" and samples == []


def test_train_only_population_and_no_mock_gate(tmp_path):
    train = dict(split="TRAIN", state=[1] * 10)
    dev = dict(split="DEV", state=[999] * 10)
    assert training_population([dev, train]) == [train]
    assert training_population([dev]) == []
    with pytest.raises(ValueError):
        training_population([dict(split="CONFIRM")])
    manifest = tmp_path / "DATA_MANIFEST.json"
    manifest.write_text(json.dumps(dict(source="target_sim", simulation_mode="NO_WBC_OR_PINNED_BASE", episodes=[])))
    with pytest.raises(ValueError, match="NO_WBC"):
        validate_manifest(tmp_path)
