"""Deterministic simulation-only visual-reach fixtures for Stage20S.

The fixture is deliberately a small data-generation boundary, not a robot
driver or a simulator controller.  It creates a red target in an RGB frame,
an explicit ``StateV1``-compatible current EEF state, and a smooth desired
EEF trajectory.  The trajectory is expressed in the same local reference
coordinate convention consumed by the existing native WBC benchmark loader:
the native loader adds its reset-derived anchor before running the delivered
controller.

The initial local pose is copied from the first point of the Stage16 native
``E1_W0_00`` bound input.  This is a provenance anchor for a simulation
fixture, not a physical calibration value.  Production camera geometry,
action timing, and robot calibration remain explicitly outside this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.readiness import (
    CameraStreamV1,
    CalibrationBundleV1,
    RawEpisodeV1,
)
from ..trajectory.so3 import (
    quat_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


SIM_VISUAL_REACH_INSTRUCTION = "move the end effector to the red target"
SIM_VISUAL_REACH_TASK_ID = "SIM_VISUAL_REACH_V0"
SIM_SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
SIM_TCP_SEMANTIC_ID = "x5_link6_gripper_center"
SIM_ACTION_SAMPLE_PERIOD_S = 0.05
SIM_ACTION_SAMPLE_PERIOD_PROVENANCE = "TEST_ONLY_SIM_VISUAL_REACH_V0"
SIM_FRAME_SIZE = (96, 96)  # height, width; square avoids crop ambiguity.
SIM_FRAME_COUNT = 61  # 40-point action horizon plus post-goal stationary frames.

# Stage16 E1_W0_00 bound input, first point.  The values are immutable fixture
# provenance and intentionally remain local to the simulation coordinate
# convention; they are not asserted to be a real-robot reset pose.
INITIAL_LOCAL_POSITION = np.asarray(
    [-0.2107728312736583, 0.06955135606396019, 0.02102946484349144],
    dtype=np.float64,
)
INITIAL_QUATERNION_XYZW = np.asarray(
    [0.4187190666811181, 0.30525921000456596, 0.36333476370362805, 0.7742602969218494],
    dtype=np.float64,
)
INITIAL_ROT6D_ROWS = rotation_matrix_to_rot6d_rows(
    quat_to_matrix(INITIAL_QUATERNION_XYZW)
)


@dataclass(frozen=True)
class SimVisualReachSpec:
    """One deterministic target and its simulation-only visual metadata."""

    episode_id: str
    target_delta: np.ndarray
    target_pixel_xy: tuple[int, int]
    family: str = "EASY_VISUAL_REACH"
    seed: int = 0
    instruction: str = SIM_VISUAL_REACH_INSTRUCTION
    sample_period_s: float = SIM_ACTION_SAMPLE_PERIOD_S
    frame_count: int = SIM_FRAME_COUNT

    def __post_init__(self) -> None:
        delta = np.asarray(self.target_delta, dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("target_delta must be finite [3]")
        if not self.episode_id or not self.instruction:
            raise ValueError("episode_id and instruction are required")
        if self.family not in {"EASY_VISUAL_REACH", "PROGRESSIVE_BASE_RELOCATION"}:
            raise ValueError(f"unsupported visual reach family: {self.family}")
        if not np.isfinite(float(self.sample_period_s)) or float(self.sample_period_s) <= 0.0:
            raise ValueError("sample_period_s must be positive and finite")
        if int(self.frame_count) < 40:
            raise ValueError("frame_count must contain the 40-point action horizon")
        pixel = tuple(int(value) for value in self.target_pixel_xy)
        if len(pixel) != 2 or any(value < 0 or value >= SIM_FRAME_SIZE[1] for value in pixel):
            raise ValueError("target_pixel_xy must be inside the square fixture frame")
        object.__setattr__(self, "target_delta", delta.copy())
        object.__setattr__(self, "target_pixel_xy", pixel)

    @property
    def target_position(self) -> np.ndarray:
        return INITIAL_LOCAL_POSITION + np.asarray(self.target_delta, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "SimVisualReachSpecV1",
            "task_id": SIM_VISUAL_REACH_TASK_ID,
            "episode_id": self.episode_id,
            "family": self.family,
            "seed": int(self.seed),
            "instruction": self.instruction,
            "target_delta_m": np.asarray(self.target_delta).tolist(),
            "target_position_local_m": self.target_position.tolist(),
            "target_pixel_xy": list(self.target_pixel_xy),
            "sample_period_s": float(self.sample_period_s),
            "sample_period_provenance": SIM_ACTION_SAMPLE_PERIOD_PROVENANCE,
            "frame_count": int(self.frame_count),
            "frame_size_hw": list(SIM_FRAME_SIZE),
            "source_frame": SIM_SOURCE_FRAME,
            "tcp_semantic_id": SIM_TCP_SEMANTIC_ID,
            "native_anchor_provenance": "STAGE16_E1_W0_00_FIRST_BOUND_INPUT_POINT",
        }


def _target_pixel(delta: np.ndarray, *, width: int = SIM_FRAME_SIZE[1], height: int = SIM_FRAME_SIZE[0]) -> tuple[int, int]:
    """Map the local target displacement to a visible, bounded pixel."""

    # The mapping is a fixture rendering convention, not a camera calibration.
    # y controls horizontal position and z controls vertical position; x is
    # represented by the target's radius so all three target coordinates vary.
    u = int(round(width / 2.0 + float(delta[1]) * 900.0))
    v = int(round(height / 2.0 - float(delta[2]) * 900.0))
    return (
        int(np.clip(u, 10, width - 11)),
        int(np.clip(v, 10, height - 11)),
    )


def default_visual_reach_specs(
    count: int,
    *,
    family: str = "EASY_VISUAL_REACH",
    seed: int = 2001,
) -> tuple[SimVisualReachSpec, ...]:
    """Return deterministic target coverage for the training ladder."""

    if int(count) < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(int(seed))
    if family == "EASY_VISUAL_REACH":
        base = np.asarray(
            [
                [-0.045, -0.035, 0.020],
                [-0.030, 0.030, 0.025],
                [-0.010, -0.020, 0.040],
                [0.010, 0.040, 0.018],
                [0.030, -0.040, 0.030],
                [0.050, 0.015, 0.038],
                [0.065, -0.005, 0.022],
                [-0.055, 0.010, 0.045],
            ],
            dtype=np.float64,
        )
    elif family == "PROGRESSIVE_BASE_RELOCATION":
        base = np.asarray(
            [
                [0.080, 0.000, 0.025],
                [0.105, -0.020, 0.030],
                [0.130, 0.015, 0.035],
                [0.155, -0.010, 0.040],
                [0.180, 0.020, 0.030],
                [0.205, -0.015, 0.045],
                [0.230, 0.010, 0.035],
                [0.255, 0.000, 0.040],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported visual reach family: {family}")

    specs: list[SimVisualReachSpec] = []
    for index in range(int(count)):
        delta = base[index % len(base)].copy()
        cycle = index // len(base)
        if cycle:
            # Keep the image/label relationship varied but bounded and
            # deterministic; do not introduce arbitrary high-speed motion.
            delta += rng.normal(0.0, [0.004, 0.004, 0.003], size=3)
            if family == "EASY_VISUAL_REACH":
                delta = np.clip(delta, [-0.065, -0.055, 0.010], [0.075, 0.055, 0.055])
            else:
                delta = np.clip(delta, [0.070, -0.045, 0.015], [0.270, 0.045, 0.060])
        specs.append(
            SimVisualReachSpec(
                episode_id=f"sim_visual_reach_{family.lower()}_{index:04d}",
                target_delta=delta,
                target_pixel_xy=_target_pixel(delta),
                family=family,
                seed=int(seed) + index,
            )
        )
    return tuple(specs)


def _smoothstep(count: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, int(count), dtype=np.float64)
    return x * x * (3.0 - 2.0 * x)


def _write_ppm(path: Path, *, target_pixel_xy: tuple[int, int], radius: int, seed: int, wrist: bool) -> str:
    """Write one dependency-free RGB fixture frame and return its SHA256."""

    height, width = SIM_FRAME_SIZE
    image = np.zeros((height, width, 3), dtype=np.uint8)
    background = np.asarray([28, 34, 42] if not wrist else [38, 42, 34], dtype=np.uint8)
    image[...] = background
    # A subtle deterministic texture prevents the image from being a constant
    # black/flat frame while leaving the red target visually dominant.
    yy, xx = np.indices((height, width))
    image[..., 1] = np.clip(image[..., 1].astype(np.int16) + ((xx + yy + seed) % 5), 0, 255).astype(np.uint8)
    u, v = target_pixel_xy
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius * radius
    image[mask] = np.asarray([230, 24, 20], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose_9d(position: np.ndarray, count: int) -> np.ndarray:
    rotation = np.broadcast_to(INITIAL_ROT6D_ROWS, (int(count), 6)).copy()
    return np.concatenate([np.asarray(position, dtype=np.float64), rotation], axis=1)


def build_visual_reach_raw(spec: SimVisualReachSpec, frame_root: str | Path) -> RawEpisodeV1:
    """Build one immutable ``RawEpisodeV1`` for a simulation fixture."""

    root = Path(frame_root)
    root.mkdir(parents=True, exist_ok=True)
    frame_body = root / f"{spec.episode_id}_body.ppm"
    frame_wrist = root / f"{spec.episode_id}_wrist.ppm"
    body_sha = _write_ppm(
        frame_body,
        target_pixel_xy=spec.target_pixel_xy,
        radius=int(np.clip(3.0 + abs(float(spec.target_delta[0])) * 24.0, 3.0, 8.0)),
        seed=spec.seed,
        wrist=False,
    )
    wrist_sha = _write_ppm(
        frame_wrist,
        target_pixel_xy=spec.target_pixel_xy,
        radius=int(np.clip(3.0 + abs(float(spec.target_delta[0])) * 20.0, 3.0, 8.0)),
        seed=spec.seed + 17,
        wrist=True,
    )

    timestamps = np.arange(int(spec.frame_count), dtype=np.float64) * float(spec.sample_period_s)
    horizon_count = 40
    alpha = _smoothstep(horizon_count)
    target = spec.target_position
    horizon_position = INITIAL_LOCAL_POSITION[None, :] + alpha[:, None] * (target - INITIAL_LOCAL_POSITION)[None, :]
    stationary = np.broadcast_to(target, (int(spec.frame_count) - horizon_count, 3))
    desired_position = np.concatenate([horizon_position, stationary], axis=0)
    desired_eef = _pose_9d(desired_position, spec.frame_count)

    # The current/achieved state is a one-sample delayed realization.  This is
    # an audit stream only; the training label remains ``desired_eef``.
    achieved_position = np.concatenate(
        [
            np.broadcast_to(INITIAL_LOCAL_POSITION, (1, 3)),
            desired_position[:-1],
        ],
        axis=0,
    )
    achieved_eef = _pose_9d(achieved_position, spec.frame_count)
    state_eef = achieved_eef.copy()
    desired_gripper = np.full((int(spec.frame_count), 1), 0.5, dtype=np.float64)
    state_gripper = desired_gripper.copy()
    achieved_gripper = desired_gripper.copy()
    identity_base = np.eye(4, dtype=np.float64)
    base_pose = np.broadcast_to(identity_base, (int(spec.frame_count), 4, 4)).copy()
    cameras = {
        "body": CameraStreamV1(
            name="body",
            frame_paths=tuple(str(frame_body) for _ in timestamps),
            timestamps=timestamps,
            declared_fps=1.0 / float(spec.sample_period_s),
            frame="SIM_VISUAL_REACH_CAMERA",
            provenance={
                "renderer": "Stage20S_dependency_free_ppm_fixture",
                "target_pixel_xy": list(spec.target_pixel_xy),
                "sha256": body_sha,
                "calibration_status": "SIMULATION_ONLY",
            },
        ),
        "wrist": CameraStreamV1(
            name="wrist",
            frame_paths=tuple(str(frame_wrist) for _ in timestamps),
            timestamps=timestamps,
            declared_fps=1.0 / float(spec.sample_period_s),
            frame="SIM_VISUAL_REACH_WRIST_CAMERA",
            provenance={
                "renderer": "Stage20S_dependency_free_ppm_fixture",
                "target_pixel_xy": list(spec.target_pixel_xy),
                "sha256": wrist_sha,
                "calibration_status": "SIMULATION_ONLY",
            },
        ),
    }
    calibration = CalibrationBundleV1(
        bundle_id="SIM_VISUAL_REACH_CALIBRATION_BUNDLE_V1",
        version="SIM_VISUAL_REACH_V0",
        status="SIMULATION_ONLY",
        arx5_joint_zero_version="SIM_REFERENCE_URDF_ZERO_V0",
        gripper_mapping={"status": "SIMULATION_ONLY", "normalized_center": 0.5},
        clock_sync_version="SIM_EXPLICIT_SHARED_TIMESTAMPS_V0",
        provenance={
            "task": SIM_VISUAL_REACH_TASK_ID,
            "camera_intrinsics": "fixture_rendering_only_not_physical_calibration",
            "native_anchor": "STAGE16_E1_W0_00_FIRST_BOUND_INPUT_POINT",
        },
        real_robot_calibration=False,
    )
    return RawEpisodeV1(
        episode_id=spec.episode_id,
        trajectory_id=spec.episode_id,
        source="SIM_VISUAL_REACH_V0_PROCEDURAL_RENDER",
        sim_level="NATIVE_WBC_CROSS_SIM_REPLAY",
        task={
            "task_id": SIM_VISUAL_REACH_TASK_ID,
            "instruction": spec.instruction,
            "target_position_local_m": target.tolist(),
            "target_delta_m": np.asarray(spec.target_delta).tolist(),
            "target_pixel_xy": list(spec.target_pixel_xy),
            "family": spec.family,
        },
        native_timestamps=timestamps,
        state_eef_9d=state_eef,
        state_gripper=state_gripper,
        desired_eef_9d=desired_eef,
        desired_gripper=desired_gripper,
        achieved_eef_9d=achieved_eef,
        achieved_gripper=achieved_gripper,
        camera_streams=cameras,
        calibration_bundle_id=calibration.bundle_id,
        frame_graph={
            "pose_frame": SIM_SOURCE_FRAME,
            "tcp": SIM_TCP_SEMANTIC_ID,
            "camera_frame": "SIM_VISUAL_REACH_CAMERA",
            "camera_path": "procedural_fixture_not_native_wbc_camera",
        },
        lifecycle_events=(
            {"type": "episode_start", "timestamp": float(timestamps[0])},
            {"type": "reference_start", "timestamp": float(timestamps[0])},
            {"type": "reference_complete", "timestamp": float(timestamps[horizon_count - 1])},
            {"type": "episode_end", "timestamp": float(timestamps[-1])},
        ),
        go2_base_pose=base_pose,
        raw_whole_body_streams={
            "sim_target": {
                "position_local_m": target.tolist(),
                "pixel_xy": list(spec.target_pixel_xy),
            },
            "sim_base": {
                "pose": "identity_static_fixture",
                "motion_demand": "encoded_in_EEF_label_only",
            },
        },
        clock_domains={
            "state": "SIM_SHARED_CLOCK",
            "desired": "SIM_SHARED_CLOCK",
            "achieved": "SIM_SHARED_CLOCK",
            "body": "SIM_SHARED_CLOCK",
            "wrist": "SIM_SHARED_CLOCK",
        },
        metadata={
            "task_id": SIM_VISUAL_REACH_TASK_ID,
            "fixture_family": spec.family,
            "sample_period_s": float(spec.sample_period_s),
            "sample_period_provenance": SIM_ACTION_SAMPLE_PERIOD_PROVENANCE,
            "source_timestamps_authority": True,
            "desired_label_only": True,
            "achieved_stream_role": "audit_only_not_training_label",
            "hardware_control": False,
            "native_wbc_camera": False,
            "closed_loop_mode": "CROSS_SIM_REPLAY_UNTIL_NATIVE_CAMERA_PATH_EXISTS",
            "native_wbc_anchor_reference": "STAGE16_E1_W0_00_FIRST_BOUND_INPUT_POINT",
        },
    )


def visual_reach_manifest(specs: Sequence[SimVisualReachSpec], *, dataset_path: str | Path) -> dict[str, Any]:
    """Return a compact, machine-readable fixture manifest."""

    return {
        "schema": "STAGE20S_SIM_VISUAL_REACH_MANIFEST_V1",
        "task_id": SIM_VISUAL_REACH_TASK_ID,
        "dataset_path": str(dataset_path),
        "episode_count": len(specs),
        "episodes": [spec.as_dict() for spec in specs],
        "state_abi": "StateV1 = EEF XYZ 3 + GR00T_FIRST_TWO_ROWS rot6d 6 + gripper 1",
        "action_abi": "future desired EEF [40,9] + future desired gripper [40,1]",
        "action_sample_period_s": SIM_ACTION_SAMPLE_PERIOD_S,
        "action_sample_period_provenance": SIM_ACTION_SAMPLE_PERIOD_PROVENANCE,
        "native_wbc_reference_convention": "local trajectory positions relative to Stage16-derived reset anchor",
        "camera_mode": "PROCEDURAL_SIM_FIXTURE_CROSS_SIM_REPLAY",
        "hardware_control": False,
        "training_label": "desired",
        "achieved_label": "audit_only_not_training_label",
    }


def write_visual_reach_manifest(path: str | Path, specs: Sequence[SimVisualReachSpec], *, dataset_path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(visual_reach_manifest(specs, dataset_path=dataset_path), indent=2) + "\n", encoding="utf-8")
