"""Four isolated PI0288 streams; record actual receipts without filling gaps."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

from ..target_sim.recorder import _jsonable, _native_pose, _time
from ..target_sim.dataset import POLICY_FIELDS, require, write_json
from .scorer import BOOL_FIELDS, finite_array, pose_matrix

IDENTITIES = ("core_wbc", "task_executor", "teacher", "clock", "scene", "codec")
LABEL_SOURCE = "ACTUALLY_COMMITTED_ACCEPTED_TEACHER_SETPOINTS"


def validate_sample(row):
    t = _time(row["observation_time_s"], "observation time")
    obs, sup, execution = row["policy_observation"], row["supervision"], row["execution"]
    require(set(obs) == POLICY_FIELDS, "PolicyObs allows only dual RGB, measured TCP/gripper and language")
    _native_pose(obs)
    require(isinstance(obs["instruction"], str) and bool(obs["instruction"].strip()), "Missing language")
    stamps = [t]
    for name in ("body_rgb", "wrist_rgb"):
        camera = obs[name]
        stamp = _time(camera["timestamp_s"], "RGB timestamp")
        require(stamp <= t + 1e-8, "Future RGB cannot enter a training record")
        require(isinstance(camera["valid"], bool) and camera["source"] == "mujoco_render", "Explicit real MuJoCo RGB validity required")
        require(not camera["valid"] or bool(camera["path"]), "Valid frame needs original path")
        stamps.append(stamp)
    require(max(stamps) - min(stamps) <= .025 + 1e-8, "Dual RGB/state snapshot skew exceeds0.025s")
    require(obs["body_rgb"]["path"] != obs["wrist_rgb"]["path"], "Dual views cannot alias one frame path")
    snapshot = execution.get("snapshot_id")
    require(bool(snapshot) and all(obs[name].get("snapshot_id") == snapshot for name in ("body_rgb", "wrist_rgb")), "RGB/state must identify one actual snapshot")
    require(sup["action_label_source"] == LABEL_SOURCE, "Declared committed label source required")
    require(abs(_time(sup["command_time_s"], "command time") - t) <= 1e-8, "Committed command must bind same sample time")
    require(_time(sup["issue_time_s"], "issue time") <= t + 1e-8, "Committed command cannot be issued in the future")
    require(bool(sup["ref_id"]) and isinstance(sup["sequence"], int), "Reference ID and sequence required")
    for kind in ("requested", "accepted", "installed", "committed"):
        if sup.get(kind) is not None:
            _native_pose(sup[kind])
    require(sup.get("requested") is not None, "Raw requested command must be preserved")
    require(isinstance(sup["valid"], bool) and isinstance(sup["actually_sent"], bool), "Explicit supervision validity required")
    if sup["valid"]:
        require(sup["actually_sent"] and all(sup.get(k) is not None for k in ("accepted", "installed", "committed")), "Missing accepted/installed evidence cannot become supervision")
    require(execution["ref_id"] == sup["ref_id"] and execution["sequence"] == sup["sequence"], "Execution must bind same reference/sequence")
    require(isinstance(execution["valid_mask"], dict), "Missing execution masks")
    for name, valid in execution["valid_mask"].items():
        require(isinstance(valid, bool) and ((execution.get(name) is not None) == valid), "Missing execution channel must be null and masked")
    require(set(row["times"]) == {"simulation", "ros", "wall_monotonic"}, "Raw sim/ROS/wall clocks required")
    for name, value in row["times"].items():
        _time(value, name)
    require(row["task_id"] == "GO2_MJ_PLACE_V1" and all(row["identity"].get(k) for k in IDENTITIES), "PI0288 task and frozen identities required")
    require(isinstance(row["privileged"], dict), "Privileged stream required")


class Recorder:
    def __init__(self, episode_id, layout_id, split, identity, *, evidence_kind="REAL_MUJOCO_RECORDING"):
        require(split in {"TRAIN", "DEV", "PILOT", "COMMISSIONING", "CPU"}, "Unsupported split")
        require(str(episode_id) and set(identity) == set(IDENTITIES) and all(identity.values()), "All frozen identities required")
        require(evidence_kind in {"REAL_MUJOCO_RECORDING", "CPU_SYNTHETIC_NOT_PHYSICS"}, "Explicit evidence kind required")
        self.metadata = dict(episode_id=str(episode_id), layout_id=int(layout_id), split=split,
            identity=_jsonable(identity), task_id="GO2_MJ_PLACE_V1", evidence_kind=evidence_kind)
        self.rows, self.physics, self.events = [], [], []
        self.segment, self.stopped = 0, False

    def append(self, *, observation_time_s, policy_observation, supervision, execution, privileged, times):
        require(not self.stopped, "Episode ended; construct a new Recorder after reset")
        row = _jsonable(dict(self.metadata, observation_time_s=observation_time_s, segment_id=self.segment,
            policy_observation=policy_observation, supervision=supervision, execution=execution,
            privileged=privileged, times=times))
        validate_sample(row)
        if self.rows:
            require(abs(row["observation_time_s"] - self.rows[-1]["observation_time_s"] - .05) <= 1e-8, "Recorder requires contiguous20Hz samples")
            require(row["supervision"]["sequence"] > self.rows[-1]["supervision"]["sequence"], "Duplicate or regressed setpoint sequence")
            require(row["policy_observation"]["instruction"] == self.rows[0]["policy_observation"]["instruction"], "Episode language changed")
            for clock in ("simulation", "ros", "wall_monotonic"):
                require(row["times"][clock] >= self.rows[-1]["times"][clock], "Clock rollback rejected")
            for camera in ("body_rgb", "wrist_rgb"):
                require(row["policy_observation"][camera]["frame_index"] > self.rows[-1]["policy_observation"][camera]["frame_index"], "Repeated renderer frame index")
        else:
            require(abs(row["observation_time_s"]) <= 1e-8, "First recorded sample must be task t=0")
        self.rows.append(copy.deepcopy(row))
        return copy.deepcopy(row)

    def boundary(self, kind, time_s):
        require(not self.stopped and kind in {"reset", "abort", "takeover", "invalid_reference"}, "Invalid or repeated episode boundary")
        t = _time(time_s, "boundary")
        require(not self.rows or t >= self.rows[-1]["observation_time_s"], "Boundary precedes records")
        self.events.append(dict(kind=kind, time_s=t, segment_before=self.segment))
        self.segment += 1
        self.stopped = True

    def append_physics(self, row):
        t = _time(row["time_s"], "physics time")
        require(not self.physics or abs(t - self.physics[-1]["time_s"] - .05) <= 1e-8, "Scorer physics must be contiguous20Hz")
        for name in ("tcp_world", "object_pose"):
            pose_matrix(row[name])
        for name in ("object_linear_velocity", "object_angular_velocity"):
            finite_array(row[name], (3,), name)
        for name in BOOL_FIELDS:
            require(isinstance(row[name], (bool, np.bool_)), f"Physical {name} must be explicit boolean")
        self.physics.append(copy.deepcopy(_jsonable(row)))

    def physics_arrays(self):
        return {key: np.asarray([row[key] for row in self.physics]) for key in self.physics[0]} if self.physics else {}

    def windows(self):
        from .dataset import windows
        return windows(self.rows)

    def write(self, folder):
        directory = Path(folder)
        # ponytail: caller owns the directory containing original camera files.
        directory.mkdir(parents=True, exist_ok=True)
        require(not any((directory / name).exists() for name in ("samples.jsonl", "physics.json", "RECORDER_MANIFEST.json")), "Preserve existing episode recording artifacts")
        with (directory / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(directory / "physics.json", _jsonable(self.physics_arrays()))
        manifest = dict(self.metadata, schema="PI0288_RECORDING_V1", samples="samples.jsonl", physics="physics.json",
            sample_count=len(self.rows), physics_count=len(self.physics), valid_H40_count=len(self.windows()),
            boundaries=self.events, physical_success=None, selected="NONE", deployment="NONE")
        write_json(directory / "RECORDER_MANIFEST.json", manifest)
        return manifest
