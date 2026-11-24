"""Bind physical right-jaw metrology to a frozen SLA observation and decoded return."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from source_calibration import GripperCalibration, SCHEMA, rigid
from source_custody import SOURCE_H5, digest, pin, read, require
from policy_query import input_identity, tensor_sha
from compiler import SourceCurve

COMMON = "SYMMETRIC_PAD_INNER_CENTRE_X_APPROACH_Y_OPEN_Z_RIGHT_HAND"
TARGET_INSTALL_RANGE_M = (.002, .042)
TARGET_MEASURED_RANGE_M = (0., .044)
SOURCE_ROWS, SOURCE_HZ = 50, 30


class InvertibleJaw(GripperCalibration):
    """Use the existing source measurement format with a unique physical inverse."""

    def __init__(self, measured):
        super().__init__(np.array(measured, dtype=float, copy=True))
        require(np.all(np.diff(self.gap_m) > 0), "physical opening requires a strictly increasing source table")
        self.code.setflags(write=False)
        self.gap_m.setflags(write=False)

    @staticmethod
    def interpolate(value, x, y, label):
        v = np.asarray(value, dtype=float)
        require(v.size and np.isfinite(v).all() and np.all((x[0] <= v) & (v <= x[-1])),
                f"{label}: observed range {v.min() if v.size else None} to {v.max() if v.size else None}; "
                f"measured interval {x[0]} to {x[-1]}")
        return np.interp(v, x, y)

    def code_from_gap(self, gap_m):
        return self.interpolate(gap_m, self.gap_m, self.code, "SOURCE_PHYSICAL_GAP_UNCOVERED")

    def gap_from_code(self, code):
        return self.interpolate(code, self.code, self.gap_m, "SOURCE_ACTION_CODE_UNCOVERED")


def fixed_pose(value):
    pose = rigid(value)
    pose.setflags(write=False)
    return pose


@dataclass(frozen=True)
class SourceTool:
    link6_from_common: np.ndarray
    jaw: InvertibleJaw
    recording_sha256: str
    calibration_sha256: str

    @classmethod
    def read_metrology(cls, path, sha256, episode):
        """Read the physical-right-arm subset of the existing source calibration bundle."""
        require(episode in SOURCE_H5, "unrecognized source recording")
        bundle = read(pin(path, sha256))
        require(bundle.get("schema") == SCHEMA and bundle.get("world_units") == "m"
                and bundle.get("common_tool_definition") == COMMON, "unknown source physical tool convention")
        row = bundle.get("episodes", {}).get(episode, {})
        require(row.get("archive_sha256") == SOURCE_H5[episode] and row.get("fps") == 30,
                "source tool is unbound to this actual recorder")
        right = row.get("arms", {}).get("right", {})
        tool, measured = (right.get(k) for k in ("link6_from_tcp", "measured_gripper_code_gap_m"))
        require(tool is not None and measured is not None, "SOURCE_PHYSICAL_TOOL_OR_PAD_MEASUREMENTS_MISSING")
        return cls(fixed_pose(tool), InvertibleJaw(measured), row["archive_sha256"], digest(path))


@dataclass(frozen=True)
class TargetTool:
    tcp_from_common: np.ndarray
    zero_gap_m: float

    def __post_init__(self):
        require(np.isfinite(self.zero_gap_m) and self.zero_gap_m >= 0, "invalid target jaw intercept")
        object.__setattr__(self, "tcp_from_common", fixed_pose(self.tcp_from_common))

    def gap_from_measured_slides(self, slides_m):
        v = np.asarray(slides_m, dtype=float)
        lo, hi = TARGET_MEASURED_RANGE_M
        require(v.shape == (2,) and np.isfinite(v).all() and np.all((lo <= v) & (v <= hi)),
                "invalid physical two-finger observation")
        return float(self.zero_gap_m + v.sum())

    def stroke_from_gap(self, gap_m):
        stroke = (np.asarray(gap_m, dtype=float)-self.zero_gap_m)/2
        lo, hi = TARGET_INSTALL_RANGE_M
        require(np.all((lo-1e-12 <= stroke) & (stroke <= hi+1e-12)),
                f"TARGET_WORKING_STROKE_UNREACHABLE: observed {stroke.min()} to {stroke.max()}; limits {lo} to {hi}")
        return np.clip(stroke, lo, hi)


@dataclass(frozen=True)
class PhysicalToolBridge:
    source: SourceTool
    target: TargetTool

    @property
    def source_link6_from_target_tcp(self):
        return self.source.link6_from_common @ np.linalg.inv(self.target.tcp_from_common)

    def right_observation(self, world_from_policy, world_from_tcp, measured_slides_m):
        origin, tcp = rigid(world_from_policy), rigid(world_from_tcp)
        pose = np.linalg.inv(origin) @ tcp @ np.linalg.inv(self.source_link6_from_target_tcp)
        wxyz = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if wxyz[0] < 0:
            wxyz = -wxyz
        gap = self.target.gap_from_measured_slides(measured_slides_m)
        return np.r_[pose[:3, 3], wxyz, self.source.jaw.code_from_gap(gap)].astype(np.float32)

    def decode(self, observation, decoded, *, expected_input_sha256, returned_input_sha256, decoded_sha256,
               world_from_policy, world_from_tcp, measured_slides_m):
        """Keep all 50 original rows. Supply physical TCP poses and per-finger metres."""
        require(input_identity(observation)[0] == expected_input_sha256 == returned_input_sha256,
                "ACTION_OBSERVATION_MISMATCH")
        require(decoded.shape == (SOURCE_ROWS, 16) and decoded.dtype == np.float32
                and np.isfinite(decoded).all() and tensor_sha(decoded) == decoded_sha256,
                "ACTION_RETURN_IDENTITY")
        actual = self.right_observation(world_from_policy, world_from_tcp, measured_slides_m)
        state = observation["state"]
        require(state.shape == (16,) and state.dtype == np.float32
                and np.allclose(actual, state[8:], atol=2e-7, rtol=0), "CALIBRATED_QUERY_TOOL_STATE_MISMATCH")
        raw = decoded.copy()
        raw.setflags(write=False)
        path = SourceCurve(raw, rigid(world_from_policy), self.source_link6_from_target_tcp)
        gaps = self.source.jaw.gap_from_code(raw[:, 15])
        strokes = self.target.stroke_from_gap(gaps)
        return dict(raw=raw, tcp_world=path.knots.copy(), physical_gap_m=gaps, per_finger_stroke_m=strokes,
                    source_times_s=np.arange(SOURCE_ROWS)/SOURCE_HZ)
