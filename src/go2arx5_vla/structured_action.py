"""Stage38 model-facing StructuredActionV1 codec.

This module is deliberately independent of GR00T, IsaacGym, CUDA, and PBS.
It defines the CPU-side representation contract that is audited before any
model is constructed:

* K16 geometry is represented by the fifteen future control poses relative to
  the measured current EEF anchor;
* canonical rot6d (the first two *columns*) is used inside the model-facing
  tensor; native WBC row-packed rot6d is produced only by the existing adapter;
* timing is represented by centered log interval fractions and decoded with
  softmax/cumsum, so the law is positive, monotone, and endpoint exact; and
* the existing dense H40 absolute gripper channel remains unchanged.

The source trajectory and its timestamps are never rewritten by this codec.
The ``materialize_structured_action`` helper keeps the H40 gripper samples
separate from the K16 ``MotionIntentV2`` control poses when returning the
timestamped physical trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .contracts.motion_intent import (
    DEFAULT_MOTION_INTENT_V2_ANCHOR,
    DEFAULT_MOTION_INTENT_V2_FRAME,
    DEFAULT_MOTION_INTENT_V2_TCP,
    MotionIntentV2,
)
from .contracts.trajectory import TrajectoryChunkV1
from .motion_api.qualification import (
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    STAGE38_TIMING_KNOTS,
    build_geometry_candidate,
    build_timing_candidate_with_knots,
)
from .trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rot6d_to_matrix,
    rotation_matrix_to_rot6d,
)


STRUCTURED_ACTION_SCHEMA = "StructuredActionV1"
STRUCTURED_ACTION_HORIZON = 40
STRUCTURED_ACTION_DIM = 132
STRUCTURED_GEOMETRY_K = 16
STRUCTURED_GEOMETRY_FUTURE_COUNT = STRUCTURED_GEOMETRY_K - 1
STRUCTURED_GEOMETRY_DIM = 9
STRUCTURED_TIMING_DIM = 1
STRUCTURED_GRIPPER_DIM = 1
STRUCTURED_PADDING_START = 11
STRUCTURED_ACTIVE_DENSE_DIM = 10
STRUCTURED_TIMING_MODE = "T16"
STRUCTURED_TIMING_INTERVAL_COUNT = int(STRUCTURED_TIMING_MODE[1:]) - 1
STRUCTURED_SOURCE_TIMING_POLICY = "ORIGINAL_TIMESTAMPED_EXPERT_TIME_LAW"
STRUCTURED_ROT6D_LAYOUT = "CANONICAL_FIRST_TWO_COLUMNS"
STRUCTURED_NATIVE_ROT6D_LAYOUT = "NATIVE_WBC_FIRST_TWO_ROWS_AT_ADAPTER_BOUNDARY"


def _array(name: str, value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result.copy()


def _timing_count(timing_mode: str) -> int:
    if timing_mode not in STAGE38_TIMING_KNOTS:
        raise ValueError(
            f"timing_mode must be one of {tuple(sorted(STAGE38_TIMING_KNOTS, key=lambda x: int(x[1:])))}"
        )
    return int(timing_mode[1:])


def structured_action_mask(
    timing_mode: str = STRUCTURED_TIMING_MODE,
) -> np.ndarray:
    """Return the exact [40,132] StructuredActionV1 supervision mask."""

    timing_count = _timing_count(timing_mode)
    mask = np.zeros(
        (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM), dtype=np.float64
    )
    mask[:STRUCTURED_GEOMETRY_FUTURE_COUNT, :STRUCTURED_GEOMETRY_DIM] = 1.0
    mask[: timing_count - 1, STRUCTURED_GEOMETRY_DIM] = 1.0
    mask[:, STRUCTURED_GEOMETRY_DIM + STRUCTURED_TIMING_DIM] = 1.0
    return mask


def structured_active_scalar_count(timing_mode: str = STRUCTURED_TIMING_MODE) -> int:
    """Count active supervised scalars for one unbatched structured sample."""

    timing_count = _timing_count(timing_mode)
    return (
        STRUCTURED_GEOMETRY_FUTURE_COUNT * STRUCTURED_GEOMETRY_DIM
        + timing_count - 1
        + STRUCTURED_ACTION_HORIZON * STRUCTURED_GRIPPER_DIM
    )


def _physical_to_chunk(
    physical_action: np.ndarray,
    source_timestamps: np.ndarray,
    *,
    chunk_id: str,
    source_frame: str,
    tcp_semantic_id: str,
    observation_timestamp: float,
    provenance: Mapping[str, Any] | None = None,
) -> TrajectoryChunkV1:
    if physical_action.shape != (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTIVE_DENSE_DIM):
        raise ValueError(
            "physical_action must have shape "
            f"[{STRUCTURED_ACTION_HORIZON},{STRUCTURED_ACTIVE_DENSE_DIM}]"
        )
    timestamps = _array("source_timestamps", source_timestamps, shape=(STRUCTURED_ACTION_HORIZON,))
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source_timestamps must be strictly increasing")
    eef = physical_action[:, :STRUCTURED_GEOMETRY_DIM]
    orientation = matrix_to_quat(rot6d_rows_to_matrix(eef[:, 3:9]))
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="STRUCTURED_ACTION_SOURCE",
        observation_timestamp=float(observation_timestamp),
        point_timestamps=timestamps,
        eef_position=eef[:, :3],
        eef_orientation=orientation,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=physical_action[:, 9].copy(),
        provenance={} if provenance is None else dict(provenance),
    )


def _validate_anchor(
    physical_action: np.ndarray,
    state_v1: np.ndarray,
    *,
    position_tolerance_m: float = 1e-9,
    rotation_tolerance_rad: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    state = _array("state_v1", state_v1, shape=(STRUCTURED_ACTIVE_DENSE_DIM,))
    action0 = physical_action[0, :STRUCTURED_GEOMETRY_DIM]
    state_rotation = rot6d_rows_to_matrix(state[3:9])
    action_rotation = rot6d_rows_to_matrix(action0[3:9])
    position_error = float(np.linalg.norm(action0[:3] - state[:3]))
    rotation_error = float(geodesic_angle(state_rotation, action_rotation))
    if position_error > position_tolerance_m or rotation_error > rotation_tolerance_rad:
        raise ValueError(
            "structured action control pose 0 is not the measured runtime anchor: "
            f"position_error_m={position_error}, rotation_error_rad={rotation_error}"
        )
    return state[:3].copy(), state_rotation.copy()


def _relative_canonical_pose(
    positions: np.ndarray,
    orientations: np.ndarray,
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
) -> np.ndarray:
    absolute_rotation = quat_to_matrix(orientations)
    relative_position = np.einsum(
        "ij,nj->ni", anchor_rotation.T, positions - anchor_position[None, :]
    )
    relative_rotation = np.einsum(
        "ij,njk->nik", anchor_rotation.T, absolute_rotation
    )
    return np.concatenate(
        [relative_position, rotation_matrix_to_rot6d(relative_rotation)], axis=-1
    )


def _absolute_from_relative_canonical(
    relative_pose: np.ndarray,
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relative = _array("relative_geometry", relative_pose)
    if relative.shape != (STRUCTURED_GEOMETRY_FUTURE_COUNT, STRUCTURED_GEOMETRY_DIM):
        raise ValueError(
            "relative_geometry must have shape "
            f"[{STRUCTURED_GEOMETRY_FUTURE_COUNT},{STRUCTURED_GEOMETRY_DIM}]"
        )
    relative_rotation = rot6d_to_matrix(relative[:, 3:9])
    positions = anchor_position[None, :] + np.einsum(
        "ij,nj->ni", anchor_rotation, relative[:, :3]
    )
    rotations = np.einsum("ij,njk->nik", anchor_rotation, relative_rotation)
    return positions, rotations


def _softmax(values: np.ndarray) -> np.ndarray:
    logits = _array("timing_logits", values)
    shifted = logits - float(np.max(logits))
    exponent = np.exp(shifted)
    total = float(np.sum(exponent))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("timing softmax normalization is not finite and positive")
    probabilities = exponent / total
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
        raise ValueError("timing softmax produced invalid interval fractions")
    return probabilities


def _centered_log_intervals(
    timing_timestamps: np.ndarray,
    *,
    start: float,
    duration: float,
) -> np.ndarray:
    timestamps = _array("timing_timestamps", timing_timestamps)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timing_timestamps must have at least two values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timing_timestamps must be strictly increasing")
    if not np.isclose(timestamps[0], start, rtol=0.0, atol=1e-10) or not np.isclose(
        timestamps[-1], start + duration, rtol=0.0, atol=1e-10
    ):
        raise ValueError("timing_timestamps must span the source duration")
    fractions = np.diff(timestamps) / float(duration)
    if np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("timing intervals must be positive and sum to one")
    centered = np.log(fractions)
    centered -= float(np.mean(centered))
    return centered


@dataclass(frozen=True)
class StructuredActionV1:
    """Validated model-facing structured action and its source binding."""

    values: np.ndarray
    mask: np.ndarray
    source_timestamps: np.ndarray
    source_duration_s: float
    anchor_state_v1: np.ndarray
    geometry_u: np.ndarray
    timing_u: np.ndarray
    timing_timestamps: np.ndarray
    timing_mode: str = STRUCTURED_TIMING_MODE
    source_frame: str = DEFAULT_MOTION_INTENT_V2_FRAME
    tcp_semantic_id: str = DEFAULT_MOTION_INTENT_V2_TCP
    observation_timestamp: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _array(
            "values",
            self.values,
            shape=(STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM),
        )
        mask = _array(
            "mask",
            self.mask,
            shape=(STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM),
        )
        expected_mask = structured_action_mask(self.timing_mode)
        if not np.array_equal(mask, expected_mask):
            raise ValueError("StructuredActionV1 mask does not match the frozen layout")
        if not np.all(np.isin(mask, [0.0, 1.0])):
            raise ValueError("StructuredActionV1 mask must be binary")
        if not np.allclose(values[mask == 0.0], 0.0, rtol=0.0, atol=0.0):
            raise ValueError("inactive StructuredActionV1 values must be exactly zero")

        source_ts = _array(
            "source_timestamps",
            self.source_timestamps,
            shape=(STRUCTURED_ACTION_HORIZON,),
        )
        if np.any(np.diff(source_ts) <= 0.0):
            raise ValueError("source_timestamps must be strictly increasing")
        duration = float(self.source_duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("source_duration_s must be finite and positive")
        if not np.isclose(source_ts[-1] - source_ts[0], duration, rtol=0.0, atol=1e-10):
            raise ValueError("source_duration_s must equal the source timestamp span")

        anchor = _array(
            "anchor_state_v1",
            self.anchor_state_v1,
            shape=(STRUCTURED_ACTIVE_DENSE_DIM,),
        )
        rot6d_rows_to_matrix(anchor[3:9])
        geometry_u = _array("geometry_u", self.geometry_u, shape=(STRUCTURED_GEOMETRY_K,))
        timing_u = _array(
            "timing_u", self.timing_u, shape=(_timing_count(self.timing_mode),)
        )
        timing_ts = _array(
            "timing_timestamps",
            self.timing_timestamps,
            shape=(_timing_count(self.timing_mode),),
        )
        expected_geometry_u = np.linspace(0.0, 1.0, STRUCTURED_GEOMETRY_K)
        expected_timing_u = STAGE38_TIMING_KNOTS[self.timing_mode]
        if not np.allclose(geometry_u, expected_geometry_u, rtol=0.0, atol=1e-12):
            raise ValueError("geometry_u must be the uniform K16 normalized arc grid")
        if not np.allclose(timing_u, expected_timing_u, rtol=0.0, atol=1e-12):
            raise ValueError("timing_u must match the preregistered timing mode")
        if np.any(np.diff(timing_ts) <= 0.0):
            raise ValueError("timing_timestamps must be strictly increasing")
        if not np.isclose(timing_ts[0], source_ts[0], rtol=0.0, atol=1e-10) or not np.isclose(
            timing_ts[-1], source_ts[-1], rtol=0.0, atol=1e-10
        ):
            raise ValueError("timing_timestamps must span the source timestamp interval")
        interval_logits = values[: len(timing_ts) - 1, 9]
        if not np.isclose(np.mean(interval_logits), 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("timing logits must be centered to zero mean")
        expected_fractions = np.diff(timing_ts) / duration
        if not np.allclose(
            _softmax(interval_logits), expected_fractions, rtol=0.0, atol=1e-10
        ):
            raise ValueError("timing logits do not reproduce source interval fractions")
        if not str(self.source_frame).strip() or not str(self.tcp_semantic_id).strip():
            raise ValueError("source_frame and tcp_semantic_id are required")
        if not np.isfinite(float(self.observation_timestamp)):
            raise ValueError("observation_timestamp must be finite")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "source_timestamps", source_ts)
        object.__setattr__(self, "source_duration_s", duration)
        object.__setattr__(self, "anchor_state_v1", anchor)
        object.__setattr__(self, "geometry_u", geometry_u)
        object.__setattr__(self, "timing_u", timing_u)
        object.__setattr__(self, "timing_timestamps", timing_ts)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def active_scalar_count(self) -> int:
        return structured_active_scalar_count(self.timing_mode)

    @property
    def timing_interval_fractions(self) -> np.ndarray:
        return _softmax(self.values[: self.timing_u.size - 1, 9])

    def as_model_arrays(self, *, dtype: Any = np.float32) -> tuple[np.ndarray, np.ndarray]:
        """Return copies suitable for the official collator/model boundary."""

        return self.values.astype(dtype, copy=True), self.mask.astype(dtype, copy=True)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": STRUCTURED_ACTION_SCHEMA,
            "timing_mode": self.timing_mode,
            "action_horizon": STRUCTURED_ACTION_HORIZON,
            "universal_action_dim": STRUCTURED_ACTION_DIM,
            "active_scalar_count": self.active_scalar_count,
            "source_timestamps": self.source_timestamps.tolist(),
            "source_duration_s": float(self.source_duration_s),
            "anchor_state_v1": self.anchor_state_v1.tolist(),
            "geometry_u": self.geometry_u.tolist(),
            "timing_u": self.timing_u.tolist(),
            "timing_timestamps": self.timing_timestamps.tolist(),
            "timing_interval_fractions": self.timing_interval_fractions.tolist(),
            "source_frame": self.source_frame,
            "tcp_semantic_id": self.tcp_semantic_id,
            "observation_timestamp": float(self.observation_timestamp),
            "rot6d_layout": STRUCTURED_ROT6D_LAYOUT,
            "native_rot6d_layout": STRUCTURED_NATIVE_ROT6D_LAYOUT,
            "provenance": dict(self.provenance),
        }
        if include_arrays:
            result["values"] = self.values.tolist()
            result["mask"] = self.mask.tolist()
        return result


def encode_structured_action(
    physical_action: Any,
    source_timestamps: Any,
    state_v1: Any,
    *,
    timing_mode: str = STRUCTURED_TIMING_MODE,
    chunk_id: str = "structured-action-source",
    source_frame: str = DEFAULT_MOTION_INTENT_V2_FRAME,
    tcp_semantic_id: str = DEFAULT_MOTION_INTENT_V2_TCP,
    observation_timestamp: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> StructuredActionV1:
    """Encode one timestamped dense expert H40 window into StructuredActionV1."""

    timing_count = _timing_count(timing_mode)
    physical = _array(
        "physical_action",
        physical_action,
        shape=(STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTIVE_DENSE_DIM),
    )
    timestamps = _array(
        "source_timestamps", source_timestamps, shape=(STRUCTURED_ACTION_HORIZON,)
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source_timestamps must be strictly increasing")
    anchor_position, anchor_rotation = _validate_anchor(physical, state_v1)
    state = _array("state_v1", state_v1, shape=(STRUCTURED_ACTIVE_DENSE_DIM,))
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0.0:
        raise ValueError("source timestamp duration must be positive")
    chunk = _physical_to_chunk(
        physical,
        timestamps,
        chunk_id=chunk_id,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        observation_timestamp=float(timestamps[0] if observation_timestamp is None else observation_timestamp),
        provenance={
            "structured_codec_source": True,
            "source_timestamp_policy": STRUCTURED_SOURCE_TIMING_POLICY,
        },
    )
    geometry = build_geometry_candidate(
        chunk,
        STRUCTURED_GEOMETRY_K,
        allowed_k_values=(STRUCTURED_GEOMETRY_K,),
        lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
    )
    semantic = build_timing_candidate_with_knots(
        chunk,
        geometry,
        timing_mode,
        STAGE38_TIMING_KNOTS[timing_mode],
        lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
    )
    relative_geometry = _relative_canonical_pose(
        semantic.eef_position[1:],
        semantic.eef_orientation[1:],
        anchor_position,
        anchor_rotation,
    )
    interval_logits = _centered_log_intervals(
        semantic.timing_timestamps,
        start=float(timestamps[0]),
        duration=duration,
    )
    values = np.zeros((STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM), dtype=np.float64)
    values[:STRUCTURED_GEOMETRY_FUTURE_COUNT, :STRUCTURED_GEOMETRY_DIM] = relative_geometry
    values[:timing_count - 1, STRUCTURED_GEOMETRY_DIM] = interval_logits
    values[:, STRUCTURED_GEOMETRY_DIM + STRUCTURED_TIMING_DIM] = physical[:, 9]
    mask = structured_action_mask(timing_mode)
    metadata = {
        "codec": STRUCTURED_ACTION_SCHEMA,
        "geometry_k": STRUCTURED_GEOMETRY_K,
        "geometry_anchor": DEFAULT_MOTION_INTENT_V2_ANCHOR,
        "geometry_u_policy": "UNIFORM_NORMALIZED_SE3_ARC_INCLUDING_ENDPOINTS",
        "predicted_geometry_controls": STRUCTURED_GEOMETRY_FUTURE_COUNT,
        "timing_mode": timing_mode,
        "timing_parameterization": "CENTERED_LOG_INTERVALS_SOFTMAX_CUMSUM",
        "timing_chunk_duration_binding": "SOURCE_TIMESTAMP_SPAN",
        "timing_source_timestamps": semantic.timing_timestamps.tolist(),
        "source_timestamp_policy": STRUCTURED_SOURCE_TIMING_POLICY,
        "rot6d_layout": STRUCTURED_ROT6D_LAYOUT,
        "native_rot6d_layout": STRUCTURED_NATIVE_ROT6D_LAYOUT,
        "gripper_layout": "ABSOLUTE_H40",
        "source_frame": source_frame,
        "tcp_semantic_id": tcp_semantic_id,
    }
    if provenance is not None:
        metadata.update(dict(provenance))
    return StructuredActionV1(
        values=values,
        mask=mask,
        source_timestamps=timestamps,
        source_duration_s=duration,
        anchor_state_v1=state,
        geometry_u=np.linspace(0.0, 1.0, STRUCTURED_GEOMETRY_K, dtype=np.float64),
        timing_u=STAGE38_TIMING_KNOTS[timing_mode].copy(),
        timing_timestamps=semantic.timing_timestamps.copy(),
        timing_mode=timing_mode,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        observation_timestamp=float(
            timestamps[0] if observation_timestamp is None else observation_timestamp
        ),
        provenance=metadata,
    )


def decode_structured_action(
    encoded_or_values: StructuredActionV1 | Any,
    *,
    anchor_state_v1: Any | None = None,
    source_timestamps: Any | None = None,
    source_duration_s: float | None = None,
    timing_mode: str | None = None,
    source_frame: str | None = None,
    tcp_semantic_id: str | None = None,
    observation_timestamp: float | None = None,
    intent_id: str = "structured-action-decoded",
    provenance: Mapping[str, Any] | None = None,
) -> MotionIntentV2:
    """Decode unnormalized structured model values into a MotionIntentV2."""

    encoded = encoded_or_values if isinstance(encoded_or_values, StructuredActionV1) else None
    values = encoded.values if encoded is not None else _array(
        "values", encoded_or_values, shape=(STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM)
    )
    mode = encoded.timing_mode if timing_mode is None and encoded is not None else (
        STRUCTURED_TIMING_MODE if timing_mode is None else timing_mode
    )
    timing_count = _timing_count(mode)
    if encoded is not None:
        if anchor_state_v1 is None:
            anchor_state_v1 = encoded.anchor_state_v1
        if source_timestamps is None:
            source_timestamps = encoded.source_timestamps
        if source_duration_s is None:
            source_duration_s = encoded.source_duration_s
        if source_frame is None:
            source_frame = encoded.source_frame
        if tcp_semantic_id is None:
            tcp_semantic_id = encoded.tcp_semantic_id
        if observation_timestamp is None:
            observation_timestamp = encoded.observation_timestamp
    if anchor_state_v1 is None or source_timestamps is None:
        raise ValueError("anchor_state_v1 and source_timestamps are required for decoding")
    state = _array("anchor_state_v1", anchor_state_v1, shape=(STRUCTURED_ACTIVE_DENSE_DIM,))
    timestamps = _array(
        "source_timestamps", source_timestamps, shape=(STRUCTURED_ACTION_HORIZON,)
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source_timestamps must be strictly increasing")
    duration = float(timestamps[-1] - timestamps[0] if source_duration_s is None else source_duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("source_duration_s must be finite and positive")
    if not np.isclose(timestamps[-1] - timestamps[0], duration, rtol=0.0, atol=1e-10):
        raise ValueError("source_duration_s must equal the decode timestamp span")
    anchor_position = state[:3]
    anchor_rotation = rot6d_rows_to_matrix(state[3:9])
    future_positions, future_rotations = _absolute_from_relative_canonical(
        values[:STRUCTURED_GEOMETRY_FUTURE_COUNT, :STRUCTURED_GEOMETRY_DIM],
        anchor_position,
        anchor_rotation,
    )
    control_positions = np.concatenate([anchor_position[None, :], future_positions], axis=0)
    control_rotations = np.concatenate([anchor_rotation[None, :, :], future_rotations], axis=0)
    control_quaternions = matrix_to_quat(control_rotations)
    timing_fractions = _softmax(values[:timing_count - 1, STRUCTURED_GEOMETRY_DIM])
    timing_u = STAGE38_TIMING_KNOTS[mode]
    control_timestamps = timestamps[0] + duration * np.concatenate(
        [np.asarray([0.0]), np.cumsum(timing_fractions)]
    )
    control_timestamps[0] = timestamps[0]
    control_timestamps[-1] = timestamps[-1]
    gripper_controls = np.interp(
        control_timestamps,
        timestamps,
        values[:, STRUCTURED_GEOMETRY_DIM + STRUCTURED_TIMING_DIM],
    )
    metadata = {
        "codec": STRUCTURED_ACTION_SCHEMA,
        "decode": True,
        "geometry_k": STRUCTURED_GEOMETRY_K,
        "timing_mode": mode,
        "timing_parameterization": "CENTERED_LOG_INTERVALS_SOFTMAX_CUMSUM",
        "timing_interval_fractions": timing_fractions.tolist(),
        "source_timestamp_policy": STRUCTURED_SOURCE_TIMING_POLICY,
        "rot6d_layout": STRUCTURED_ROT6D_LAYOUT,
        "native_rot6d_layout": STRUCTURED_NATIVE_ROT6D_LAYOUT,
        "gripper_layout": "ABSOLUTE_H40",
    }
    if encoded is not None:
        metadata.update(dict(encoded.provenance))
    if provenance is not None:
        metadata.update(dict(provenance))
    return MotionIntentV2(
        intent_id=intent_id,
        observation_timestamp=float(
            timestamps[0] if observation_timestamp is None else observation_timestamp
        ),
        point_timestamps=control_timestamps,
        eef_position=control_positions,
        eef_orientation=control_quaternions,
        geometry_arc_fraction=np.linspace(0.0, 1.0, STRUCTURED_GEOMETRY_K, dtype=np.float64),
        source_frame=source_frame or DEFAULT_MOTION_INTENT_V2_FRAME,
        tcp_semantic_id=tcp_semantic_id or DEFAULT_MOTION_INTENT_V2_TCP,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        timing_mode=mode,
        timing_arc_fraction=timing_u.copy(),
        timing_timestamps=control_timestamps,
        gripper=gripper_controls,
        anchor_semantics=DEFAULT_MOTION_INTENT_V2_ANCHOR,
        provenance=metadata,
    )


def materialize_structured_action(
    encoded_or_values: StructuredActionV1 | Any,
    intent: MotionIntentV2,
    *,
    source_timestamps: Any | None = None,
) -> TrajectoryChunkV1:
    """Materialize the decoded pose path and preserve the dense H40 gripper."""

    encoded = encoded_or_values if isinstance(encoded_or_values, StructuredActionV1) else None
    if source_timestamps is None and encoded is not None:
        source_timestamps = encoded.source_timestamps
    if source_timestamps is None:
        raise ValueError("source_timestamps are required to materialize a structured action")
    timestamps = _array(
        "source_timestamps", source_timestamps, shape=(STRUCTURED_ACTION_HORIZON,)
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source_timestamps must be strictly increasing")
    if not np.isclose(timestamps[0], intent.validity_start, rtol=0.0, atol=1e-10) or not np.isclose(
        timestamps[-1], intent.validity_end, rtol=0.0, atol=1e-10
    ):
        raise ValueError("materialization timestamps must span the intent validity interval")
    physical = intent.materialize_trajectory(query_timestamps=timestamps)
    values = encoded.values if encoded is not None else _array(
        "values", encoded_or_values, shape=(STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM)
    )
    gripper = values[:, STRUCTURED_GEOMETRY_DIM + STRUCTURED_TIMING_DIM].copy()
    metadata = dict(physical.provenance)
    metadata.update(
        {
            "structured_gripper_materialization": "EXACT_H40_MODEL_CHANNEL",
            "structured_pose_materialization": "MOTION_INTENT_V2_U_OF_T_THEN_GAMMA",
        }
    )
    return TrajectoryChunkV1(
        chunk_id=physical.chunk_id,
        source="StructuredActionV1",
        observation_timestamp=physical.observation_timestamp,
        point_timestamps=physical.point_timestamps,
        eef_position=physical.eef_position,
        eef_orientation=physical.eef_orientation,
        source_frame=physical.source_frame,
        tcp_semantic_id=physical.tcp_semantic_id,
        validity_start=physical.validity_start,
        validity_end=physical.validity_end,
        gripper=gripper,
        provenance=metadata,
    )


def structured_action_contract_summary(timing_mode: str = STRUCTURED_TIMING_MODE) -> dict[str, Any]:
    """Return the frozen dimensional and supervision contract."""

    timing_count = _timing_count(timing_mode)
    return {
        "schema": STRUCTURED_ACTION_SCHEMA,
        "timing_mode": timing_mode,
        "action_horizon": STRUCTURED_ACTION_HORIZON,
        "universal_action_dim": STRUCTURED_ACTION_DIM,
        "geometry_k": STRUCTURED_GEOMETRY_K,
        "predicted_future_geometry_controls": STRUCTURED_GEOMETRY_FUTURE_COUNT,
        "geometry_u": "uniform_normalized_se3_arc_including_endpoints",
        "layout": {
            "relative_xyz": [0, 3],
            "canonical_rot6d": [3, 9],
            "timing_logit": 9,
            "absolute_gripper": 10,
            "padding": [STRUCTURED_PADDING_START, STRUCTURED_ACTION_DIM],
        },
        "timing_knots": STAGE38_TIMING_KNOTS[timing_mode].tolist(),
        "timing_interval_count": timing_count - 1,
        "timing_parameterization": "p=softmax(z); t=cumsum(T_chunk*p); z=log(p)-mean(log(p))",
        "timing_chunk_binding": "source_timestamp_span",
        "source_timing_policy": STRUCTURED_SOURCE_TIMING_POLICY,
        "rot6d_layout": STRUCTURED_ROT6D_LAYOUT,
        "native_rot6d_layout": STRUCTURED_NATIVE_ROT6D_LAYOUT,
        "gripper_layout": "absolute_h40",
        "mask_active_scalar_count": structured_active_scalar_count(timing_mode),
        "dense_h40_active_scalar_count": STRUCTURED_ACTION_HORIZON * STRUCTURED_ACTIVE_DENSE_DIM,
        "padding_contributes_to_loss": False,
    }


__all__ = [
    "STRUCTURED_ACTION_DIM",
    "STRUCTURED_ACTION_HORIZON",
    "STRUCTURED_ACTION_SCHEMA",
    "STRUCTURED_ACTIVE_DENSE_DIM",
    "STRUCTURED_GEOMETRY_DIM",
    "STRUCTURED_GEOMETRY_FUTURE_COUNT",
    "STRUCTURED_GEOMETRY_K",
    "STRUCTURED_GRIPPER_DIM",
    "STRUCTURED_NATIVE_ROT6D_LAYOUT",
    "STRUCTURED_PADDING_START",
    "STRUCTURED_ROT6D_LAYOUT",
    "STRUCTURED_SOURCE_TIMING_POLICY",
    "STRUCTURED_TIMING_INTERVAL_COUNT",
    "STRUCTURED_TIMING_MODE",
    "STRUCTURED_TIMING_DIM",
    "StructuredActionV1",
    "decode_structured_action",
    "encode_structured_action",
    "materialize_structured_action",
    "structured_action_contract_summary",
    "structured_action_mask",
    "structured_active_scalar_count",
]
