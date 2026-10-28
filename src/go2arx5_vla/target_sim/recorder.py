"""Four separated streams and H40 labels from subsequent actual commands."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..trajectory.so3 import matrix_to_quat, quat_to_matrix, rotation_matrix_to_rot6d
from .scorer import BOOL_FIELDS, finite_array, pose_matrix

OBSERVATION_FIELDS = {"body_rgb", "wrist_rgb", "tcp_xyz", "tcp_quaternion_xyzw", "gripper", "instruction"}
ACTION_FIELDS = {"command_id", "issue_time_s", "tcp_xyz", "tcp_quaternion_xyzw", "gripper",
                 "accepted", "valid", "actually_sent", "frame_id", "tcp_id"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("recording cannot contain nonfinite JSON values")
        return value
    raise TypeError(f"unsupported recording value {type(value).__name__}")


def _time(value, name):
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative timestamp")
    return result


def _native_pose(row):
    xyz = finite_array(row["tcp_xyz"], (3,), "tcp_xyz")
    quat = finite_array(row["tcp_quaternion_xyzw"], (4,), "tcp_quaternion_xyzw")
    if abs(np.linalg.norm(quat) - 1) > 1e-6:
        raise ValueError("native TCP quaternion must be unit xyzw")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = quat_to_matrix(quat), xyz
    gripper = float(row["gripper"])
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError("gripper fraction must be finite in [0,1]")
    return pose, gripper


def command_target(command, *, command_id, issue_time_s, accepted, actually_sent, tcp_id):
    """Envelope a teacher command after observing the actual send outcome."""
    if not isinstance(accepted, (bool, np.bool_)) or not isinstance(actually_sent, (bool, np.bool_)):
        raise ValueError("actual dispatch outcomes must be explicit booleans")
    pose = pose_matrix(command.tcp_world)
    return {"command_id": str(command_id), "issue_time_s": float(issue_time_s),
            "tcp_xyz": pose[:3, 3].tolist(), "tcp_quaternion_xyzw": matrix_to_quat(pose[:3, :3]).tolist(),
            "gripper": command.gripper_fraction, "accepted": bool(accepted),
            "valid": not command.failed and bool(accepted), "actually_sent": bool(actually_sent),
            "frame_id": "world", "tcp_id": str(tcp_id)}


class Recorder:
    """In-memory one-episode recording; explicit boundaries prevent label leaks.

    Images stay at their actual target-render paths. Missing feedback fields
    use None plus valid_mask=False; they are never zero-filled labels. This
    does not initialize cameras, physics, WBC, or a learned model.
    """

    def __init__(self, episode_id, layout_id, split, identity, task_id="GO2_PLACE_V1"):
        if not str(episode_id) or split not in {"TRAIN", "DEV", "PILOT", "COMMISSIONING", "CPU"}:
            raise ValueError("episode identity and declared split are required")
        if task_id != "GO2_PLACE_V1" or set(identity) != {"asset", "teacher", "controller", "codec"} or not all(identity.values()):
            raise ValueError("asset/teacher/controller/codec identities are required")
        self.metadata = {"episode_id": str(episode_id), "layout_id": int(layout_id), "split": split,
                         "task_id": task_id, "identity": _jsonable(identity)}
        self.rows = []
        self.physics = []
        self.events = []
        self.segment = 0
        self.stopped = False
        self._command_ids = set()

    def boundary(self, kind, time_s):
        if self.stopped:
            raise ValueError("terminated recorder cannot resume via another boundary")
        if kind not in {"reset", "abort", "takeover"}:
            raise ValueError("boundary must be reset, abort, or takeover")
        timestamp = _time(time_s, "boundary time")
        if self.events and timestamp < self.events[-1]["time_s"]:
            raise ValueError("boundary timestamps must be monotonic")
        if self.rows and timestamp < self.rows[-1]["observation_time_s"]:
            raise ValueError("boundary precedes already recorded samples")
        self.events.append({"kind": kind, "time_s": timestamp, "segment_before": self.segment})
        self.segment += 1
        # A reset creates a different episode: construct a new Recorder.
        self.stopped = kind in {"reset", "abort"}

    def append(self, *, observation_time_s, policy_observation, action_target,
               execution_feedback, privileged_diagnostics):
        if self.stopped:
            raise ValueError("terminated recorder cannot resume; construct a new episode")
        timestamp = _time(observation_time_s, "observation time")
        if self.rows and timestamp <= self.rows[-1]["observation_time_s"]:
            raise ValueError("observation timestamps must be strictly increasing")
        if self.events and timestamp < self.events[-1]["time_s"]:
            raise ValueError("sample timestamp precedes its recorded boundary")
        if set(policy_observation) != OBSERVATION_FIELDS:
            raise ValueError("policy observation must contain only dual RGB, actual TCP/gripper, language")
        observation = copy.deepcopy(_jsonable(policy_observation))
        _native_pose(observation)
        if not isinstance(observation["instruction"], str) or not observation["instruction"]:
            raise ValueError("instruction is required")
        for name in ("body_rgb", "wrist_rgb"):
            camera = observation[name]
            if set(camera) != {"path", "timestamp_s", "valid", "source"} or not isinstance(camera["valid"], bool):
                raise ValueError("RGB requires path/timestamp/valid/source")
            capture = _time(camera["timestamp_s"], "RGB time")
            if capture > timestamp + 1e-8 or camera["source"] != "target_render":
                raise ValueError("future or non-target RGB is not a policy observation")
            if camera["valid"] and not camera["path"]:
                raise ValueError("valid RGB requires the saved real frame path")
        if set(action_target) != ACTION_FIELDS:
            raise ValueError("action target must preserve the actual command envelope")
        action = copy.deepcopy(_jsonable(action_target))
        _native_pose(action)
        issue = _time(action["issue_time_s"], "issue time")
        if issue < timestamp - 1e-8 or action["frame_id"] != "world" or not action["tcp_id"]:
            raise ValueError("command requires world frame, TCP identity, and causal issue time")
        for name in ("accepted", "valid", "actually_sent"):
            if not isinstance(action[name], bool):
                raise ValueError(f"{name} must be boolean")
        if action["valid"] and not (action["accepted"] and action["actually_sent"]):
            raise ValueError("unsent/rejected command cannot become valid supervision")
        if not action["command_id"] or action["command_id"] in self._command_ids:
            raise ValueError("command IDs must be nonempty and unique per episode")
        feedback = copy.deepcopy(_jsonable(execution_feedback))
        if feedback.get("command_id") != action["command_id"]:
            raise ValueError("execution feedback must name the same actual command")
        if "valid_mask" not in feedback or not isinstance(feedback["valid_mask"], dict):
            raise ValueError("execution feedback must preserve explicit validity masks")
        for name in ("tcp_xyz", "tcp_quaternion_xyzw", "base", "torso", "installed_reference", "delay_s"):
            if name not in feedback:
                feedback[name] = None
                feedback["valid_mask"][name] = False
            elif name not in feedback["valid_mask"]:
                raise ValueError(f"execution channel {name} requires its own validity mask")
        for name, valid in feedback["valid_mask"].items():
            if not isinstance(valid, bool) or (valid and feedback.get(name) is None):
                raise ValueError("feedback validity must agree with measured values")
            if not valid and feedback.get(name) is not None:
                raise ValueError("missing feedback must be None, never a zero-filled truth")
        if feedback.get("achieved_time_s") is not None:
            if _time(feedback["achieved_time_s"], "achieved time") < issue - 1e-8:
                raise ValueError("achieved feedback cannot precede its issued command")
        row = {**copy.deepcopy(self.metadata), "observation_time_s": timestamp, "segment_id": self.segment,
               "policy_observation": observation, "action_target": action, "execution_feedback": feedback,
               "privileged_diagnostics": copy.deepcopy(_jsonable(privileged_diagnostics))}
        self.rows.append(row)
        self._command_ids.add(action["command_id"])
        return copy.deepcopy(row)

    def append_physics(self, row):
        """Append real synchronized state, including t=0 and terminal t=32."""
        timestamp = _time(row["time_s"], "physics time")
        if self.physics and timestamp <= self.physics[-1]["time_s"]:
            raise ValueError("physics timestamps must be strictly increasing")
        required = {"time_s", "tcp_world", "object_pose", "object_linear_velocity", "object_angular_velocity", *BOOL_FIELDS}
        if not required.issubset(row):
            raise ValueError(f"missing independent scorer physics fields: {sorted(required - set(row))}")
        for name in ("tcp_world", "object_pose"):
            pose_matrix(finite_array(row[name], (4, 4), name))
        for name in ("object_linear_velocity", "object_angular_velocity"):
            finite_array(row[name], (3,), name)
        for name in BOOL_FIELDS:
            if not isinstance(row[name], (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        self.physics.append(copy.deepcopy(_jsonable(row)))

    def physics_arrays(self):
        if not self.physics:
            return {}
        return {key: np.asarray([row[key] for row in self.physics]) for key in self.physics[0]}

    def windows(self):
        """H40 at observation i uses commands i+1..i+40; no padding/copying."""
        result = []
        for index in range(max(0, len(self.rows) - 40)):
            anchor = self.rows[index]
            future = self.rows[index + 1:index + 41]
            selected = [anchor, *future]
            if (any(row["segment_id"] != anchor["segment_id"] for row in future)
                    or any(not all(row["action_target"][key] for key in ("accepted", "valid", "actually_sent")) for row in selected)):
                continue
            times = np.asarray([row["observation_time_s"] for row in selected])
            issues = np.asarray([row["action_target"]["issue_time_s"] for row in selected])
            if not (np.all(np.abs(np.diff(times) - .05) <= 1e-8) and np.all(np.abs(np.diff(issues) - .05) <= 1e-8)):
                continue
            labels = []
            for row in future:
                pose, gripper = _native_pose(row["action_target"])
                labels.append(np.r_[pose[:3, 3], rotation_matrix_to_rot6d(pose[:3, :3]), gripper])
            result.append({"episode_id": self.metadata["episode_id"], "anchor_index": index,
                           "observation_time_s": anchor["observation_time_s"], "segment_id": anchor["segment_id"],
                           "action_target": np.asarray(labels), "valid_mask": np.ones((40, 10), dtype=bool),
                           "future_command_ids": [row["action_target"]["command_id"] for row in future],
                           "future_issue_time_s": issues[1:].copy(), "execution_prefix": 8,
                           "rot6d_layout": "FIRST_TWO_COLUMNS", "label_source": "SUBSEQUENT_ACTUAL_TEACHER_COMMANDS"})
        return result

    def write(self, folder):
        """Create one new artifact directory; never overwrite original evidence."""
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=False)
        samples = directory / "samples.jsonl"
        with samples.open("x", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        physics = self.physics_arrays()
        (directory / "physics.json").write_text(json.dumps(_jsonable(physics), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        manifest = {"schema": "PI0287_RECORDING_V1", "source": "target_sim", **copy.deepcopy(self.metadata),
                    "samples": "samples.jsonl", "physics": "physics.json", "sample_count": len(self.rows),
                    "physics_count": len(self.physics), "valid_H40_count": len(self.windows()),
                    "boundaries": self.events, "physical_success": None,
                    "selected": "NONE", "deployment": "NONE"}
        (directory / "RECORDER_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return manifest
