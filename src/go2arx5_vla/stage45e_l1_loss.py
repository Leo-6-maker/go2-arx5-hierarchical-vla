"""Stage45E-L1 physical-error-aware loss binding.

This module owns the *only* scientific intervention in PI_GATE_STATE_0123.
The StructuredAction normalization, statistics, binary supervision mask, and
decoder remain untouched.  At runtime the pinned GR00T action-head returns its
usual elementwise masked MSE; this module multiplies only the active XYZ
geometry elements by the frozen inverse-physical-scale weights and divides by
the unchanged binary-mask count.

The module intentionally has no Torch or GR00T import at module load time.  It
can therefore be imported by CPU/static tests and by the pre-model admission
path without constructing a model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


STAGE = "Stage45E-L1"
INTERVENTION = "PHYSICAL_ERROR_AWARE_XYZ_LOSS"
ACTION_HORIZON = 40
ACTION_DIM = 132
GEOMETRY_FUTURE_COUNT = 15
GEOMETRY_DIM = 9
TIMING_INDEX = 9
GRIPPER_INDEX = 10
ACTIVE_SCALAR_COUNT = 190

# These values are part of the PI0123 scientific contract.  They are not
# recomputed from an observed run or from a model checkpoint.
XYZ_LOSS_WEIGHTS = {
    "x": 1.401885,
    "y": 0.472436,
    "z": 1.125678,
}


class L1LossContractError(ValueError):
    """Raised when an L1 loss tensor violates the frozen contract."""


def canonical_action_mask() -> np.ndarray:
    """Return the exact binary ``T16`` mask without importing project code."""

    mask = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    mask[:GEOMETRY_FUTURE_COUNT, :GEOMETRY_DIM] = 1.0
    mask[:GEOMETRY_FUTURE_COUNT, TIMING_INDEX] = 1.0
    mask[:, GRIPPER_INDEX] = 1.0
    return mask


def build_loss_weight_matrix() -> np.ndarray:
    """Build the frozen ``[40,132]`` multiplicative loss-weight tensor."""

    weights = np.ones((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    weights[:GEOMETRY_FUTURE_COUNT, 0] = XYZ_LOSS_WEIGHTS["x"]
    weights[:GEOMETRY_FUTURE_COUNT, 1] = XYZ_LOSS_WEIGHTS["y"]
    weights[:GEOMETRY_FUTURE_COUNT, 2] = XYZ_LOSS_WEIGHTS["z"]
    return weights


L1_LOSS_WEIGHT_MATRIX = build_loss_weight_matrix()
_RUNTIME_COUNTERS = {"action_head_forwards": 0}


def reset_runtime_counters() -> None:
    """Reset the in-process L1 action-head counter after admission."""

    _RUNTIME_COUNTERS["action_head_forwards"] = 0


def runtime_counters() -> dict[str, int]:
    """Return a copy of the guarded action-head counter."""

    return {key: int(value) for key, value in _RUNTIME_COUNTERS.items()}


def loss_weight_contract(mask: Any | None = None) -> dict[str, Any]:
    """Validate and describe the immutable L1 intervention.

    ``mask`` is optional for static callers.  When supplied it must be the
    exact binary T16 mask; accepting a fractional or folded mask would change
    the scientific denominator and is therefore fail-closed.
    """

    expected_mask = canonical_action_mask()
    if mask is not None:
        observed = np.asarray(mask)
        if observed.shape != expected_mask.shape:
            raise L1LossContractError(
                f"action mask shape drift: {observed.shape} != {expected_mask.shape}"
            )
        if observed.dtype.kind not in {"f"}:
            raise L1LossContractError("action mask must use a floating binary T16 representation")
        if not np.array_equal(observed, expected_mask):
            raise L1LossContractError("action mask is not the frozen binary T16 mask")
    if L1_LOSS_WEIGHT_MATRIX.shape != (ACTION_HORIZON, ACTION_DIM):
        raise L1LossContractError("L1 loss-weight shape drift")
    if not np.isfinite(L1_LOSS_WEIGHT_MATRIX).all():
        raise L1LossContractError("L1 loss weights are non-finite")
    if not np.all(L1_LOSS_WEIGHT_MATRIX[expected_mask == 0] == 1.0):
        raise L1LossContractError("inactive/padded dimensions must retain weight 1.0")
    if not np.all(
        L1_LOSS_WEIGHT_MATRIX[:GEOMETRY_FUTURE_COUNT, 0] == XYZ_LOSS_WEIGHTS["x"]
    ):
        raise L1LossContractError("x loss weight drift")
    if not np.all(
        L1_LOSS_WEIGHT_MATRIX[:GEOMETRY_FUTURE_COUNT, 1] == XYZ_LOSS_WEIGHTS["y"]
    ):
        raise L1LossContractError("y loss weight drift")
    if not np.all(
        L1_LOSS_WEIGHT_MATRIX[:GEOMETRY_FUTURE_COUNT, 2] == XYZ_LOSS_WEIGHTS["z"]
    ):
        raise L1LossContractError("z loss weight drift")
    if int(np.count_nonzero(expected_mask)) != ACTIVE_SCALAR_COUNT:
        raise L1LossContractError("active scalar count drift")
    return {
        "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
        "stage": STAGE,
        "intervention": INTERVENTION,
        "weight_shape": [ACTION_HORIZON, ACTION_DIM],
        "xyz_rows": [0, GEOMETRY_FUTURE_COUNT - 1],
        "xyz_channels": [0, 1, 2],
        "xyz_loss_weights": dict(XYZ_LOSS_WEIGHTS),
        "all_other_active_dimensions_weight": 1.0,
        "inactive_dimensions_weight": 1.0,
        "mask_schema": "T16_BINARY_UNCHANGED",
        "active_scalar_count": ACTIVE_SCALAR_COUNT,
        "denominator": "binary_action_mask.sum()",
        "mask_folding": False,
        "adaptive_renormalization": False,
        "mean_weight_normalization": False,
        "matrix_sha256": _array_sha256(L1_LOSS_WEIGHT_MATRIX),
    }


def _array_sha256(array: Any) -> str:
    import hashlib

    value = np.asarray(array, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_torch_tensors(torch_module: Any, action_loss: Any, action_mask: Any) -> None:
    if not torch_module.is_tensor(action_loss) or not torch_module.is_tensor(action_mask):
        raise L1LossContractError("action_loss and action_mask must be Torch tensors")
    if tuple(action_loss.shape) != tuple(action_mask.shape):
        raise L1LossContractError("action_loss/action_mask shape mismatch")
    if action_loss.ndim != 3 or tuple(action_loss.shape[1:]) != (ACTION_HORIZON, ACTION_DIM):
        raise L1LossContractError(
            f"expected [B,{ACTION_HORIZON},{ACTION_DIM}] loss tensors, got {tuple(action_loss.shape)}"
        )
    if not bool(torch_module.isfinite(action_loss).all().item()):
        raise L1LossContractError("unweighted masked action loss is non-finite")
    if not bool(torch_module.isfinite(action_mask).all().item()):
        raise L1LossContractError("action mask is non-finite")
    if not bool(torch_module.is_floating_point(action_mask)):
        raise L1LossContractError("action mask must use a floating binary T16 representation")
    expected = torch_module.as_tensor(
        canonical_action_mask(), dtype=action_mask.dtype, device=action_mask.device
    ).unsqueeze(0).expand(action_mask.shape[0], -1, -1)
    if not bool(torch_module.equal(action_mask, expected)):
        raise L1LossContractError("runtime action mask differs from frozen binary T16 mask")
    if not bool(torch_module.equal(action_mask.sum(dim=(1, 2)), torch_module.full(
        (action_mask.shape[0],), ACTIVE_SCALAR_COUNT, dtype=action_mask.dtype,
        device=action_mask.device
    ))):
        raise L1LossContractError("runtime action mask active count is not exactly 190")
    inactive = action_mask == 0
    if bool(torch_module.any(action_loss[inactive] != 0).item()):
        raise L1LossContractError("original masked action loss is nonzero in inactive dimensions")


def weighted_loss_from_masked_mse_torch(
    torch_module: Any, action_loss: Any, action_mask: Any
) -> tuple[Any, Any]:
    """Apply the frozen weights to an already binary-masked MSE tensor.

    The pinned GR00T action head computes ``per_element_mse * action_mask``
    before returning ``action_loss``.  Since the mask is validated exactly,
    multiplying that tensor by the weight matrix is algebraically identical to
    the PI contract ``per_element_mse * action_mask * l1_loss_weight``.
    """

    _validate_torch_tensors(torch_module, action_loss, action_mask)
    weights = torch_module.as_tensor(
        L1_LOSS_WEIGHT_MATRIX, dtype=action_loss.dtype, device=action_loss.device
    ).unsqueeze(0)
    weighted = action_loss * weights
    denominator = action_mask.sum()
    if not bool(torch_module.equal(
        denominator,
        torch_module.as_tensor(ACTIVE_SCALAR_COUNT * action_mask.shape[0],
                               dtype=denominator.dtype, device=denominator.device),
    )):
        raise L1LossContractError("L1 loss denominator is not the exact binary-mask sum")
    return weighted.sum() / denominator, weights


def install_l1_loss_patch() -> dict[str, Any]:
    """Patch the pinned action-head loss immediately before model construction.

    This function is intentionally called only after the runtime admission
    guard.  It imports the external GR00T class lazily, preserves the original
    action-head graph, and replaces only the scalar loss plus its diagnostic
    ``action_loss`` tensor.
    """

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    marker = "_stage45e_l1_original_forward"
    existing = getattr(Gr00tN1d7ActionHead, marker, None)
    if existing is not None:
        return {
            "status": "ALREADY_INSTALLED",
            "class": "gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead",
            "matrix_sha256": _array_sha256(L1_LOSS_WEIGHT_MATRIX),
        }
    original = Gr00tN1d7ActionHead.forward
    setattr(Gr00tN1d7ActionHead, marker, original)

    def l1_forward(self: Any, backbone_output: Any, action_input: Any) -> Any:
        _RUNTIME_COUNTERS["action_head_forwards"] += 1
        output = original(self, backbone_output, action_input)
        action_loss = output["action_loss"]
        action_mask = output["action_mask"]
        import torch

        loss, _weights = weighted_loss_from_masked_mse_torch(
            torch, action_loss, action_mask
        )
        output["action_loss"] = action_loss * _weights
        output["loss"] = loss
        return output

    setattr(Gr00tN1d7ActionHead, "forward", l1_forward)
    setattr(Gr00tN1d7ActionHead, "_stage45e_l1_patch_schema", "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1")
    return {
        "status": "INSTALLED",
        "class": "gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead",
        "matrix_sha256": _array_sha256(L1_LOSS_WEIGHT_MATRIX),
        "original_forward_preserved": True,
    }


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTIVE_SCALAR_COUNT",
    "INTERVENTION",
    "L1_LOSS_WEIGHT_MATRIX",
    "L1LossContractError",
    "STAGE",
    "XYZ_LOSS_WEIGHTS",
    "build_loss_weight_matrix",
    "canonical_action_mask",
    "install_l1_loss_patch",
    "loss_weight_contract",
    "reset_runtime_counters",
    "runtime_counters",
    "weighted_loss_from_masked_mse_torch",
]
