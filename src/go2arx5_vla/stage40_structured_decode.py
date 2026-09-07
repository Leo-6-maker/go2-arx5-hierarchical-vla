"""Stage40's single StructuredActionV1 live-decode authority.

This module is deliberately CPU-only and imports no model, simulator, CUDA,
or scheduler code.  The GR00T worker supplies raw ``[1,40,132]`` values; the
accepted Stage38 codec performs the only inverse-normalization and physical
materialization here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts.trajectory import TrajectoryChunkV1
from .structured_action import (
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    decode_structured_action,
    materialize_structured_action,
    structured_action_contract_summary,
    structured_action_mask,
)
from .trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows


DEFAULT_STRUCTURED_STATS_PATH = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/"
    "experiment_cfg/structured_action_statistics.json"
)
EXPECTED_STRUCTURED_STATS_SHA256 = (
    "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
)
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
TIMING_MODE = "T16"
PERIOD_S = 0.05
DURATION_S = 1.9500000000000002


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_structured_statistics(path: Path = DEFAULT_STRUCTURED_STATS_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load and validate the immutable Stage38-E statistics contract."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"structured statistics are not a regular file: {path}")
    stats_sha256 = _sha256_file(path)
    if stats_sha256 != EXPECTED_STRUCTURED_STATS_SHA256:
        raise ValueError(
            "STRUCTURED_STATS_SHA256_MISMATCH: "
            f"expected {EXPECTED_STRUCTURED_STATS_SHA256}, got {stats_sha256}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    statistics = document.get("statistics", document)
    lower_value = document.get("selected_lower", statistics.get("selected_lower"))
    upper_value = document.get("selected_upper", statistics.get("selected_upper"))
    if lower_value is None or upper_value is None:
        raise ValueError("structured statistics are missing selected bounds")
    lower = np.asarray(lower_value, dtype=np.float64)
    upper = np.asarray(upper_value, dtype=np.float64)
    mask = np.asarray(statistics.get("active_mask"), dtype=np.float64)
    if lower.shape != (STRUCTURED_ACTION_DIM,) or upper.shape != (STRUCTURED_ACTION_DIM,):
        raise ValueError(f"structured bounds must be [132], got {lower.shape} {upper.shape}")
    expected_mask = structured_action_mask(TIMING_MODE)
    if mask.shape != expected_mask.shape or not np.array_equal(mask, expected_mask):
        raise ValueError("STRUCTURED_ACTION_MASK_MISMATCH")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("structured selected bounds are not finite")
    if np.any(upper < lower):
        raise ValueError("structured selected bounds are not ordered")
    return lower, upper, mask, stats_sha256


def _validate_source_timestamps(source_timestamps: Any) -> np.ndarray:
    timestamps = np.asarray(source_timestamps, dtype=np.float64)
    if timestamps.shape != (STRUCTURED_ACTION_HORIZON,) or not np.isfinite(timestamps).all():
        raise ValueError("Stage40 StructuredActionV1 requires finite source timestamps [40]")
    if not np.allclose(np.diff(timestamps), PERIOD_S, rtol=0.0, atol=1e-9):
        raise ValueError("STAGE40_NONUNIFORM_REFERENCE_TIMESTAMPS")
    if not np.isclose(timestamps[-1] - timestamps[0], DURATION_S, rtol=0.0, atol=1e-9):
        raise ValueError("STAGE40_REFERENCE_DURATION_MISMATCH")
    return timestamps


def decode_raw_structured_action(
    raw_universal_action: Any,
    state_v1: Any,
    *,
    stats_path: Path = DEFAULT_STRUCTURED_STATS_PATH,
    source_timestamps: Any,
    observation_timestamp: float,
    chunk_id: str,
    source_frame: str = SOURCE_FRAME,
    tcp_semantic_id: str = TCP_SEMANTIC_ID,
) -> tuple[TrajectoryChunkV1, np.ndarray, dict[str, Any]]:
    """Decode one raw model output through the accepted Stage38-E contract."""

    raw = np.asarray(raw_universal_action)
    if raw.shape != (1, STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM):
        raise ValueError(f"raw StructuredActionV1 output must be [1,40,132], got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("raw StructuredActionV1 output is not finite numeric data")
    state = np.asarray(state_v1, dtype=np.float32).reshape(-1)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError(f"StateV1 must be finite [10], got {state.shape}")
    timestamps = _validate_source_timestamps(source_timestamps)
    if not np.isfinite(float(observation_timestamp)):
        raise ValueError("observation_timestamp must be finite")
    lower, upper, mask, stats_sha256 = load_structured_statistics(Path(stats_path))
    normalized = np.clip(raw[0].astype(np.float64), -1.0, 1.0)
    values = (normalized + 1.0) * 0.5 * (upper[None, :] - lower[None, :]) + lower[None, :]
    values[mask == 0.0] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("inverse-normalized StructuredActionV1 values are not finite")
    intent = decode_structured_action(
        values,
        anchor_state_v1=state.astype(np.float64),
        source_timestamps=timestamps,
        source_duration_s=DURATION_S,
        timing_mode=TIMING_MODE,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        observation_timestamp=float(observation_timestamp),
        intent_id=chunk_id,
        provenance={
            "stage40_decoder": "Stage38-E-authority",
            "structured_stats_sha256": stats_sha256,
            "no_dense_decode_fallback": True,
        },
    )
    chunk = materialize_structured_action(values, intent, source_timestamps=timestamps)
    rows = rotation_matrix_to_rot6d_rows(
        quat_to_matrix(np.asarray(chunk.eef_orientation, dtype=np.float64))
    )
    physical = np.concatenate(
        [
            np.asarray(chunk.eef_position, dtype=np.float64),
            rows,
            np.asarray(chunk.gripper, dtype=np.float64).reshape(STRUCTURED_ACTION_HORIZON, 1),
        ],
        axis=1,
    )
    if physical.shape != (STRUCTURED_ACTION_HORIZON, 10) or not np.isfinite(physical).all():
        raise ValueError(f"materialized StructuredActionV1 path is invalid: {physical.shape}")
    details: dict[str, Any] = {
        "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
        "contract": structured_action_contract_summary(TIMING_MODE),
        "stats_path": str(Path(stats_path)),
        "stats_sha256": stats_sha256,
        "mask_sha256_float64": _sha256_array(mask),
        "raw_sha256": _sha256_array(raw),
        "state_v1_sha256": _sha256_array(state),
        "raw_shape": list(raw.shape),
        "state_v1_shape": list(state.shape),
        "physical_shape": list(physical.shape),
        "physical_sha256_float64": _sha256_array(physical),
        "timing_fractions": (
            np.diff(np.asarray(intent.timing_timestamps, dtype=np.float64)) / DURATION_S
        ).tolist(),
        "timing_timestamps": np.asarray(intent.timing_timestamps, dtype=np.float64).tolist(),
        "values_active_inactive_contract": True,
        "values_finite": True,
        "source_timestamps": timestamps.tolist(),
        "source_duration_s": DURATION_S,
        "source_frame": source_frame,
        "tcp_semantic_id": tcp_semantic_id,
    }
    return chunk, physical, details


__all__ = [
    "DEFAULT_STRUCTURED_STATS_PATH",
    "EXPECTED_STRUCTURED_STATS_SHA256",
    "decode_raw_structured_action",
    "load_structured_statistics",
]
