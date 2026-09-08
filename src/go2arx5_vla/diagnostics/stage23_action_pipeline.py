"""Stage23 CPU-only action-pipeline and learnability diagnostics.

This module mirrors the already-frozen official GR00T N1.7 boundary without
importing the private/model runtime.  It is intentionally descriptive and
non-corrective: no smoothing, clipping, retiming, resampling, or training is
performed here.

The contract audited here is:

* physical action ``[H, 10]`` = EEF ``[H, 9]`` + gripper ``[H, 1]``;
* EEF absolute labels become one-window relative poses with respect to the
  last observed EEF state, using ``T_ref^-1 @ T_target``;
* the EEF rotation representation is the first two rows of the rotation
  matrix flattened in row-major order;
* the gripper remains an absolute, separate channel;
* the universal action is ``[H, 132]`` with only the first ten dimensions
  active for this embodiment; and
* the official training loss mask zeros the remaining 122 dimensions.

The implementation is kept independent of LeRobot and Isaac-GR00T so that the
contract can be tested on CPU and compared with the official source audit.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..contracts import TrajectoryChunkV1
from ..metrics import chunk_profile, vla_prediction_error
from ..safety import TrajectorySafetyGateV1
from ..trajectory.so3 import (
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


ACTION_HORIZON = 40
PHYSICAL_ACTION_DIM = 10
EEF_ACTION_DIM = 9
GRIPPER_ACTION_DIM = 1
UNIVERSAL_ACTION_DIM = 132
PADDING_ACTION_DIM = UNIVERSAL_ACTION_DIM - PHYSICAL_ACTION_DIM


def _finite_array(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def split_physical_action(action: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split physical ``[..., 10]`` action into EEF ``[..., 9]`` and gripper."""

    array = _finite_array("physical_action", action)
    if array.ndim < 1 or array.shape[-1] != PHYSICAL_ACTION_DIM:
        raise ValueError(
            f"physical_action must end in [{PHYSICAL_ACTION_DIM}], got {array.shape}"
        )
    return array[..., :EEF_ACTION_DIM].copy(), array[..., EEF_ACTION_DIM:].copy()


def _validate_eef(eef_action: Any) -> np.ndarray:
    array = _finite_array("eef_action", eef_action)
    if array.ndim < 2 or array.shape[-1] != EEF_ACTION_DIM:
        raise ValueError(f"eef_action must have shape [...,H,9], got {array.shape}")
    return array


def _validate_reference(reference_state: Any) -> np.ndarray:
    reference = _finite_array("reference_state", reference_state)
    if reference.shape != (EEF_ACTION_DIM,):
        raise ValueError(f"reference_state must have shape [9], got {reference.shape}")
    # Decode once at the boundary so an invalid row-packed rot6d cannot be
    # silently propagated into the relative transform.
    rot6d_rows_to_matrix(reference[3:9])
    return reference


def relative_eef_action(eef_action: Any, reference_state: Any) -> np.ndarray:
    """Convert absolute EEF XYZ+row-rot6d poses to one-reference relative poses.

    This is the exact geometric operation used by the official EEF path:
    ``T_relative = T_reference^-1 @ T_absolute``.  Every future point uses the
    same reference frame; this function does not compute frame-to-frame deltas.
    """

    absolute = _validate_eef(eef_action)
    reference = _validate_reference(reference_state)
    reference_rotation = rot6d_rows_to_matrix(reference[3:9])
    absolute_rotation = rot6d_rows_to_matrix(absolute[..., 3:9])
    translation_delta = absolute[..., :3] - reference[:3]
    relative_translation = np.einsum(
        "ij,...j->...i", reference_rotation.T, translation_delta
    )
    relative_rotation = np.einsum(
        "ij,...jk->...ik", reference_rotation.T, absolute_rotation
    )
    return np.concatenate(
        [relative_translation, rotation_matrix_to_rot6d_rows(relative_rotation)], axis=-1
    )


def absolute_eef_action(relative_action: Any, reference_state: Any) -> np.ndarray:
    """Invert :func:`relative_eef_action` with the same fixed reference pose."""

    relative = _validate_eef(relative_action)
    reference = _validate_reference(reference_state)
    reference_rotation = rot6d_rows_to_matrix(reference[3:9])
    relative_rotation = rot6d_rows_to_matrix(relative[..., 3:9])
    absolute_translation = np.einsum(
        "ij,...j->...i", reference_rotation, relative[..., :3]
    ) + reference[:3]
    absolute_rotation = np.einsum(
        "ij,...jk->...ik", reference_rotation, relative_rotation
    )
    return np.concatenate(
        [absolute_translation, rotation_matrix_to_rot6d_rows(absolute_rotation)], axis=-1
    )


def relative_eef_roundtrip(eef_action: Any, reference_state: Any) -> dict[str, float]:
    """Return numeric absolute→relative→absolute errors for one action window."""

    original = _validate_eef(eef_action)
    relative = relative_eef_action(original, reference_state)
    reconstructed = absolute_eef_action(relative, reference_state)
    original_rotation = rot6d_rows_to_matrix(original[..., 3:9])
    reconstructed_rotation = rot6d_rows_to_matrix(reconstructed[..., 3:9])
    position_error = np.linalg.norm(original[..., :3] - reconstructed[..., :3], axis=-1)
    rotation_error = np.asarray(
        geodesic_angle(original_rotation, reconstructed_rotation), dtype=np.float64
    )
    return {
        "max_position_error_m": float(np.max(position_error)),
        "p95_position_error_m": float(np.percentile(position_error, 95.0)),
        "max_rotation_error_rad": float(np.max(rotation_error)),
        "p95_rotation_error_rad": float(np.percentile(rotation_error, 95.0)),
        "finite": bool(np.isfinite(reconstructed).all()),
    }


def build_universal_action_and_mask(
    physical_action: Any,
    *,
    horizon: int = ACTION_HORIZON,
    universal_dim: int = UNIVERSAL_ACTION_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad physical actions and construct the official training loss mask.

    The returned mask has the same rank as the input action.  For an unbatched
    window this is ``[H,132]``; for a batch it is ``[B,H,132]``.  Only the
    first ten dimensions are active.  The 122 padded dimensions are explicitly
    zero, matching ``F.mse_loss(..., reduction='none') * action_mask``.
    """

    action = _finite_array("physical_action", physical_action)
    if action.ndim < 2 or action.shape[-1] != PHYSICAL_ACTION_DIM:
        raise ValueError(f"physical_action must have shape [...,H,10], got {action.shape}")
    if action.shape[-2] != int(horizon):
        raise ValueError(f"expected horizon {horizon}, got {action.shape[-2]}")
    if int(universal_dim) < PHYSICAL_ACTION_DIM:
        raise ValueError("universal_dim must contain all physical action dimensions")
    universal = np.zeros(action.shape[:-1] + (int(universal_dim),), dtype=np.float32)
    universal[..., :PHYSICAL_ACTION_DIM] = action.astype(np.float32)
    mask = np.zeros_like(universal, dtype=np.float32)
    mask[..., :PHYSICAL_ACTION_DIM] = 1.0
    return universal, mask


def action_mask_partition(mask: Any, *, physical_dim: int = PHYSICAL_ACTION_DIM) -> dict[str, Any]:
    """Summarize active physical and padding dimensions in a loss mask."""

    array = _finite_array("action_mask", mask)
    if array.ndim < 2 or array.shape[-1] < int(physical_dim):
        raise ValueError(f"action_mask must end in at least [{physical_dim}], got {array.shape}")
    active = array[..., :physical_dim]
    padding = array[..., physical_dim:]
    return {
        "shape": list(array.shape),
        "physical_dimensions": int(physical_dim),
        "padding_dimensions": int(array.shape[-1] - physical_dim),
        "physical_active_elements": int(np.count_nonzero(active)),
        "padding_active_elements": int(np.count_nonzero(padding)),
        "physical_mask_sum": float(np.sum(active)),
        "padding_mask_sum": float(np.sum(padding)),
        "padding_contributes_to_loss": bool(np.any(padding != 0.0)),
    }


def _chunk_from_physical(
    physical_action: np.ndarray,
    timestamps: np.ndarray,
    *,
    chunk_id: str,
    source_frame: str,
    tcp_semantic_id: str,
    observation_timestamp: float,
    state: np.ndarray,
) -> TrajectoryChunkV1:
    eef, gripper = split_physical_action(physical_action)
    orientation = matrix_to_quat(rot6d_rows_to_matrix(eef[:, 3:9]))
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="STAGE23_EXPERT_LABEL_AS_PREDICTION",
        observation_timestamp=float(observation_timestamp),
        point_timestamps=timestamps,
        eef_position=eef[:, :3],
        eef_orientation=orientation,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=gripper,
        provenance={
            "stage": "Stage23",
            "prediction_source": "expert_label_identity",
            "trajectory_correction": False,
            "row_rot6d_decoded_before_canonical_chunk": True,
            "state_v1_used_for_gate": True,
        },
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= 1e-12:
        return 1.0 if abs(float(numerator)) <= 1e-12 else None
    return float(numerator / denominator)


def label_as_prediction_metrics(
    *,
    physical_action: Any,
    timestamps: Any,
    state_v1: Any,
    chunk_id: str = "stage23-label-identity",
    source_frame: str = "NATIVE_WBC_RELATIVE_REFERENCE",
    tcp_semantic_id: str = "x5_link6_gripper_center",
    observation_timestamp: float | None = None,
) -> dict[str, Any]:
    """Run the exact evaluator with expert labels as the prediction.

    This is a contract sanity check, not model evidence.  If the identity
    prediction does not produce zero task-space error and unit profile ratios,
    the evaluator or action contract is inconsistent and Hopper training must
    remain locked.
    """

    physical = _finite_array("physical_action", physical_action)
    if physical.shape != (ACTION_HORIZON, PHYSICAL_ACTION_DIM):
        raise ValueError(f"physical_action must be [40,10], got {physical.shape}")
    point_timestamps = _finite_array("timestamps", timestamps)
    if point_timestamps.shape != (ACTION_HORIZON,):
        raise ValueError(f"timestamps must be [40], got {point_timestamps.shape}")
    if np.any(np.diff(point_timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    state = _finite_array("state_v1", state_v1)
    if state.shape != (PHYSICAL_ACTION_DIM,):
        raise ValueError(f"state_v1 must be [10], got {state.shape}")
    state_rotation = rot6d_rows_to_matrix(state[3:9])
    predicted_chunk = _chunk_from_physical(
        physical,
        point_timestamps,
        chunk_id=chunk_id,
        source_frame=source_frame,
        tcp_semantic_id=tcp_semantic_id,
        observation_timestamp=(
            float(point_timestamps[0])
            if observation_timestamp is None
            else float(observation_timestamp)
        ),
        state=state,
    )
    eef, _ = split_physical_action(physical)
    expected_rotation = rot6d_rows_to_matrix(eef[:, 3:9])
    prediction_error = vla_prediction_error(
        eef[:, :3],
        predicted_chunk.eef_position,
        expected_rotation,
        quat_to_matrix(predicted_chunk.eef_orientation),
        metadata={"prediction_source": "expert_label_identity"},
    )
    profile = chunk_profile(
        predicted_chunk,
        achieved_position=state[:3],
        achieved_quaternion=matrix_to_quat(state_rotation),
        provenance={"prediction_source": "expert_label_identity"},
    )
    gate = TrajectorySafetyGateV1().evaluate(
        predicted_chunk,
        now=float(point_timestamps[0]),
        achieved_position=state[:3],
        achieved_orientation=matrix_to_quat(state_rotation),
    )
    path = float(profile["path_length_m"])
    mean_speed = float(profile["linear_speed_mps"]["mean"] or 0.0)
    max_speed = float(profile["linear_speed_mps"]["max"] or 0.0)
    roughness = float(profile["roughness_position_second_difference_m"]["mean"] or 0.0)
    ratios = {
        "path_length_ratio": _safe_ratio(path, path),
        "mean_speed_ratio": _safe_ratio(mean_speed, mean_speed),
        "max_speed_ratio": _safe_ratio(max_speed, max_speed),
        "roughness_ratio": _safe_ratio(roughness, roughness),
    }
    position = prediction_error["position_m"]
    rotation = prediction_error["rotation_rad"]
    identity_pass = bool(
        prediction_error["sample_count"] == ACTION_HORIZON
        and position["max"] <= 1e-12
        and rotation["max"] <= 1e-12
        and all(value is not None and abs(float(value) - 1.0) <= 1e-12 for value in ratios.values())
        and gate.decision.value == "ACCEPT"
    )
    return {
        "schema": "STAGE23_LABEL_AS_PREDICTION_DIAGNOSTIC_V1",
        "identity_pass": identity_pass,
        "physical_action_shape": list(physical.shape),
        "prediction_error": prediction_error,
        "profile": profile,
        "profile_ratios": ratios,
        "safety_gate": gate.as_dict(),
        "trajectory_correction": False,
        "classification": (
            "LABEL_AS_PREDICTION_PASS"
            if identity_pass
            else "STAGE23_EVALUATOR_OR_ACTION_CONTRACT_BUG"
        ),
    }


def action_contract_summary() -> dict[str, Any]:
    """Return the frozen Stage23 physical/universal boundary summary."""

    return {
        "state_v1_dim": PHYSICAL_ACTION_DIM,
        "physical_action_dim": PHYSICAL_ACTION_DIM,
        "eef_action_dim": EEF_ACTION_DIM,
        "gripper_action_dim": GRIPPER_ACTION_DIM,
        "universal_action_dim": UNIVERSAL_ACTION_DIM,
        "padding_action_dim": PADDING_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "eef_reference": "last state[eef_9d] sample in the current observation window",
        "eef_transform": "T_reference^-1 @ T_absolute_target",
        "eef_transform_is_frame_to_frame_delta": False,
        "rot6d_layout": "GR00T_FIRST_TWO_ROWS",
        "gripper_representation": "ABSOLUTE / separate channel",
        "timestamp_role": "sample ordering and provenance; not a model tensor",
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
        "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        "loss_mask_active_dimensions": list(range(PHYSICAL_ACTION_DIM)),
        "loss_mask_padding_dimensions": list(range(PHYSICAL_ACTION_DIM, UNIVERSAL_ACTION_DIM)),
        "padding_contributes_to_loss": False,
    }
