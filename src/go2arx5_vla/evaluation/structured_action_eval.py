"""Stage45C's runtime-independent StructuredAction evaluation boundary.

The inverse normalization and K16/T16/D132 -> H40/D10 materialization are
delegated to the accepted Stage40 decoder.  This module only validates the
boundary and keeps the raw prediction separate from the decoded physical
trajectory; it never constructs a model or worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..stage40_structured_decode import (
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    decode_raw_structured_action,
)
from .trajectory_metrics import EvaluationContractError


RAW_ACTION_SHAPE = (1, 40, 132)
STATE_V1_SHAPE = (10,)
PHYSICAL_ACTION_SHAPE = (40, 10)
EXPECTED_MODEL_INPUT_KEYS = ("body_rgb", "wrist_rgb", "state")


@dataclass(frozen=True)
class DecodedStructuredPrediction:
    """Raw-to-decoded result retained by the evaluation pipeline."""

    physical_action: np.ndarray
    decoder_details: Mapping[str, Any]
    trajectory_chunk: Any


def _finite(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationContractError(f"{name} is not an array") from exc
    if tuple(array.shape) != tuple(shape):
        raise EvaluationContractError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise EvaluationContractError(f"{name} must be finite numeric data")
    return array


def validate_model_input_contract(
    model_input_keys: Sequence[str],
    *,
    target_xyz_in_model_input: bool,
) -> None:
    """Enforce the accepted Stage44R model-input boundary."""

    if tuple(model_input_keys) != EXPECTED_MODEL_INPUT_KEYS:
        raise EvaluationContractError(
            "model input keys must be exactly body_rgb,wrist_rgb,state"
        )
    if bool(target_xyz_in_model_input):
        raise EvaluationContractError("numeric target XYZ must remain outside model input")


def validate_observation_arrays(
    *,
    body_rgb: Any,
    wrist_rgb: Any,
    state_v1: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the sealed observation ABI without touching a model."""

    body = _finite(body_rgb, name="body_rgb", shape=(96, 96, 3))
    wrist = _finite(wrist_rgb, name="wrist_rgb", shape=(96, 96, 3))
    state = _finite(state_v1, name="state_v1", shape=STATE_V1_SHAPE)
    if not np.issubdtype(body.dtype, np.integer) or not np.issubdtype(wrist.dtype, np.integer):
        raise EvaluationContractError("RGB arrays must retain their integer storage ABI")
    if np.any(body < 0) or np.any(body > 255) or np.any(wrist < 0) or np.any(wrist > 255):
        raise EvaluationContractError("RGB arrays contain values outside uint8 range")
    return body, wrist, state.astype(np.float64, copy=False)


def decode_raw_prediction(
    raw_universal_action: Any,
    state_v1: Any,
    *,
    stats_path: Path,
    source_timestamps: Any,
    observation_timestamp: float,
    chunk_id: str,
    source_frame: str = SOURCE_FRAME,
    tcp_semantic_id: str = TCP_SEMANTIC_ID,
) -> DecodedStructuredPrediction:
    """Decode one externally supplied raw prediction through Stage40 authority.

    ``stats_path`` is deliberately required and has no node-local default;
    the Stage45C manifest must late-bind its exact hash and location.
    """

    raw = _finite(raw_universal_action, name="raw_universal_action", shape=RAW_ACTION_SHAPE)
    state = _finite(state_v1, name="state_v1", shape=STATE_V1_SHAPE)
    if not str(chunk_id).strip():
        raise EvaluationContractError("chunk_id must be non-empty")
    if not Path(stats_path).is_file() or Path(stats_path).is_symlink():
        raise EvaluationContractError("stats_path must be an existing regular file")
    if source_frame != SOURCE_FRAME or tcp_semantic_id != TCP_SEMANTIC_ID:
        raise EvaluationContractError("source frame/TCP contract drift")
    try:
        chunk, physical, details = decode_raw_structured_action(
            raw,
            state.astype(np.float32),
            stats_path=Path(stats_path),
            source_timestamps=source_timestamps,
            observation_timestamp=float(observation_timestamp),
            chunk_id=chunk_id,
            source_frame=source_frame,
            tcp_semantic_id=tcp_semantic_id,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationContractError(f"Stage40 StructuredAction decode rejected input: {exc}") from exc
    physical_array = _finite(physical, name="decoded_physical_action", shape=PHYSICAL_ACTION_SHAPE)
    return DecodedStructuredPrediction(
        physical_action=physical_array,
        decoder_details=dict(details),
        trajectory_chunk=chunk,
    )


__all__ = [
    "EXPECTED_MODEL_INPUT_KEYS",
    "PHYSICAL_ACTION_SHAPE",
    "RAW_ACTION_SHAPE",
    "STATE_V1_SHAPE",
    "DecodedStructuredPrediction",
    "decode_raw_prediction",
    "validate_model_input_contract",
    "validate_observation_arrays",
]
