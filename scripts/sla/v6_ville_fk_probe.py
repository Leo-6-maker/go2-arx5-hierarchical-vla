"""Compare one fixed SDK FK candidate with the two observed Ville TCP streams."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

from v6_ville_pilot import NATIVE_20HZ_TCP_CANDIDATES, need, read, write


SDK_REVISION = "36eabcb618c2668dfdff228b18b30507fde71784"
SDK_XML = f"https://raw.githubusercontent.com/real-stanford/arx5-sdk/{SDK_REVISION}/models/X5.urdf"
AUTHOR_PLUGIN_REVISION = "d87b5f1d8bba052abf4182cbd27f20a7cc66f76f"


def compare(root, output, *, numeric_output=None, extension=None, candidates=()):
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    with urlopen(SDK_XML, timeout=25) as response:
        encoded = response.read(300 * 1024)
    need(len(encoded) < 300 * 1024, "Bounded SDK XML exceeded")
    robot = ElementTree.fromstring(encoded)
    joints = [robot.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
    joints.append(robot.find("joint[@name='gripper_fixed_joint']"))
    need(all(joint is not None for joint in joints)
         and joints[-1].find("child").get("link") == "eef_link", "SDK chain changed")
    for previous, current in zip(joints[:-1], joints[1:], strict=True):
        need(previous.find("child").get("link") == current.find("parent").get("link"), "Discontinuous SDK chain")

    def fk(q):
        position = np.zeros((len(q), 3))
        rotation = np.broadcast_to(np.eye(3), (len(q), 3, 3))
        for index, joint in enumerate(joints):
            origin = joint.find("origin")
            translation = np.fromstring(origin.get("xyz"), sep=" ")
            rpy = np.fromstring(origin.get("rpy"), sep=" ")
            position += rotation @ translation
            rotation = rotation @ Rotation.from_euler("xyz", rpy).as_matrix()
            if index < 6:
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                rotation = rotation @ Rotation.from_rotvec(q[:, index, None] * axis).as_matrix()
        return position, rotation

    episodes = []
    offset = None
    if numeric_output is not None:
        numeric_output.mkdir(mode=0o700)
    population = [(root, index, phase) for index, phase in ((0, "FIT"), (1, "HELD_OUT_CHECK"))]
    if extension is not None:
        observations = read(extension / "ACTUAL.json")
        need(observations["repo"] == "villekuosmanen/drop_footbag_into_dice_tower"
             and observations["source_train_added"] == 0, "Unexpected single-arm validation source")
        population.extend((extension, row["episode"], row["proposed_split"]) for row in observations["scanned"])
    candidate_sources = {}
    for candidate in candidates:
        observations = read(candidate / "ACTUAL.json")
        expected = {f"villekuosmanen/{name}": revision for name, revision in NATIVE_20HZ_TCP_CANDIDATES}
        source = observations["repo"]
        need(observations["revision"] == expected.get(source) and observations["source_train_added"] == 0
             and observations["license"] == "Apache-2.0", "Unexpected 20 Hz TCP candidate")
        pilot = "episodes" in observations
        hashes = {row["path"]: row["sha256"] for row in observations["member_hashes" if pilot else "files"]}
        for row in observations["episodes" if pilot else "scanned"]:
            index = row["episode"]
            member = f"data/chunk-000/episode_{index:06d}.parquet"
            need(row["fps"] == 20 and row["eef_observation_span"] is not None
                 and hashlib.sha256((candidate / member).read_bytes()).hexdigest() == hashes[member],
                 "Candidate timing, observed TCP or numeric member differs")
            phase = "CANDIDATE_CHECK" if pilot else row["proposed_split"]
            population.append((candidate, index, phase))
        candidate_sources[candidate] = source
    need(not (candidate_sources and numeric_output), "Candidate comparison cannot export overlapping label filenames")
    for folder, index, phase in population:
        sample = folder / f"data/chunk-000/episode_{index:06d}.parquet"
        values = pq.read_table(sample, columns=["observation.state", "observation.eef_6d_pose", "action", "timestamp"])
        state = np.asarray(values["observation.state"].to_pylist())
        observed = np.asarray(values["observation.eef_6d_pose"].to_pylist())
        action = np.asarray(values["action"].to_pylist())
        n = len(state)
        need(state.shape == action.shape == (n, 7) and observed.shape == (n, 6)
             and all(np.isfinite(x).all() for x in (state, observed, action)), "Recorded sample ABI changed")
        prediction, rotation = fk(state[:, :6])
        predicted_rpy = Rotation.from_euler("xyz", observed[:, 3:]).as_matrix()
        orientation = Rotation.from_matrix(rotation @ predicted_rpy.transpose(0, 2, 1)).magnitude()
        translation = observed[:, :3] - prediction
        if offset is None:
            offset = (rotation.transpose(0, 2, 1) @ translation[:, :, None])[:, :, 0].mean(axis=0)
        raw_error = np.linalg.norm(translation, axis=1)
        adjusted_error = np.linalg.norm(translation - rotation @ offset, axis=1)
        episodes.append(dict(episode=index, samples=n, fit_role=phase,
             raw_position_median_mm=float(np.median(raw_error) * 1000),
             raw_position_max_mm=float(raw_error.max() * 1000),
             adjusted_position_median_mm=float(np.median(adjusted_error) * 1000),
             adjusted_position_max_mm=float(adjusted_error.max() * 1000),
             angle_median_deg=float(np.degrees(np.median(orientation))),
             angle_max_deg=float(np.degrees(orientation.max())),
             observed_gripper_range=[float(state[:, 6].min()), float(state[:, 6].max())],
             recorded_action_gripper_range=[float(action[:, 6].min()), float(action[:, 6].max())]))
        if folder in candidate_sources:
            episodes[-1]["source"] = candidate_sources[folder]
        if numeric_output is not None:
            dt = np.asarray(values["timestamp"].to_pylist(), dtype=np.float64)
            need(np.max(np.abs(dt - np.arange(n) / 20.0)) < 1e-5, "20 Hz recorded clock changed")
            command_p, command_r = fk(action[:, :6])
            command_p += command_r @ offset
            horizon = 40
            count = n - horizon + 1
            windows = np.arange(count)[:, None] + np.arange(horizon)[None, :]
            relative = command_r[windows] @ predicted_rpy[:count, None].swapaxes(-1, -2)
            rotvec = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(count, horizon, 3)
            labels = np.concatenate((command_p[windows] - observed[:count, None, :3],
                                     rotvec, action[windows, 6:7]), axis=-1).astype(np.float32)
            state_ee = np.column_stack((observed[:, :3],
                 Rotation.from_matrix(predicted_rpy).as_rotvec(), state[:, 6])).astype(np.float32)
            packed = numeric_output / f"episode_{index:06d}.npz"
            with packed.open("xb") as handle:
                np.savez(handle, state=state_ee[:count], action=labels, observation_timestamp_s=dt[:count],
                         label_timestamp_s=dt[windows], support=np.ones((count, horizon), dtype=bool),
                         raw_action=action, raw_joint_observation=state)
            episodes[-1].update(preview_label="FK_OF_RECORDED_SETPOINT_CANDIDATE",
                                preview_windows=count, preview_shape=list(labels.shape),
                                signed_grip_field_preserved=True,
                                preview_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
    write(output, dict(sdk_revision=SDK_REVISION, sdk_urdf_sha256=hashlib.sha256(encoded).hexdigest(),
         author_plugin_revision=AUTHOR_PLUGIN_REVISION, fitted_fixed_tool_offset_m=offset.tolist(),
         calibration_source="TWO_EXISTING_RECORDED_OBSERVATION_STREAMS", episodes=episodes,
         dataset_writer_revision_verified=False, physical_gripper_calibration_verified=False,
         source_outcome_verified=False, admitted_train=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--numeric-output", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--candidate", dest="candidates", type=Path, action="append", default=[])
    compare(**vars(parser.parse_args()))
