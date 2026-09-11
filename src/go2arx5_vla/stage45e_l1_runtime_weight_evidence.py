"""Preparation-only runtime evidence helper for future L1 canaries.

Historical job 612326 did not serialize the dtype/value of the effective L1
weight tensor under BF16 autocast. This helper is intended for a future
separately gated canary patch. It does not alter the loss, mask, model, or
backward graph: it only materializes the already-frozen float32 weight matrix to
the observed action-loss dtype/device and returns detached scalar evidence.

Importing this module does not import Torch. The caller passes its admitted Torch
module and action-loss tensor explicitly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .stage45e_l1_loss import (
    ACTIVE_SCALAR_COUNT,
    GEOMETRY_FUTURE_COUNT,
    L1_LOSS_WEIGHT_MATRIX,
    XYZ_LOSS_WEIGHTS,
    canonical_action_mask,
)


class L1RuntimeWeightEvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise L1RuntimeWeightEvidenceError(message)


def _sha256_float32(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()


def capture_effective_weight_evidence(
    torch_module: Any,
    action_loss: Any,
    action_mask: Any,
) -> dict[str, Any]:
    """Return detached evidence for the exact runtime L1 weight representation.

    No tensor participating in the caller's loss graph is modified. A new
    diagnostic weight tensor is constructed exactly as the L1 implementation
    does: float32 canonical matrix -> action_loss dtype/device.
    """

    require(torch_module.is_tensor(action_loss), "action_loss must be Torch tensor")
    require(torch_module.is_tensor(action_mask), "action_mask must be Torch tensor")
    require(tuple(action_loss.shape) == tuple(action_mask.shape), "action_loss/action_mask shape mismatch")
    require(action_loss.ndim == 3 and tuple(action_loss.shape[1:]) == (40, 132), "action_loss shape drift")
    expected_mask = torch_module.as_tensor(
        canonical_action_mask(), dtype=action_mask.dtype, device=action_mask.device
    ).unsqueeze(0).expand(action_mask.shape[0], -1, -1)
    require(bool(torch_module.equal(action_mask, expected_mask)), "runtime mask differs from frozen T16")
    require(int(action_mask.detach().float().sum().cpu().item()) == ACTIVE_SCALAR_COUNT * action_mask.shape[0], "runtime mask active count drift")

    weights = torch_module.as_tensor(
        L1_LOSS_WEIGHT_MATRIX,
        dtype=action_loss.dtype,
        device=action_loss.device,
    )
    effective = {
        "x": float(weights[0, 0].detach().float().cpu().item()),
        "y": float(weights[0, 1].detach().float().cpu().item()),
        "z": float(weights[0, 2].detach().float().cpu().item()),
    }
    # All future geometry rows must use the same effective coefficient per axis.
    for row in range(GEOMETRY_FUTURE_COUNT):
        require(float(weights[row, 0].detach().float().cpu().item()) == effective["x"], "effective x weight varies across geometry rows")
        require(float(weights[row, 1].detach().float().cpu().item()) == effective["y"], "effective y weight varies across geometry rows")
        require(float(weights[row, 2].detach().float().cpu().item()) == effective["z"], "effective z weight varies across geometry rows")

    return {
        "schema": "STAGE45E_L1_RUNTIME_EFFECTIVE_WEIGHT_EVIDENCE_V1",
        "status": "CAPTURED_READONLY_FROM_FORWARD_TENSOR_DTYPE",
        "action_loss_dtype": str(action_loss.dtype),
        "action_mask_dtype": str(action_mask.dtype),
        "weight_tensor_dtype": str(weights.dtype),
        "weight_tensor_device_type": str(weights.device.type),
        "configured_xyz_loss_weights_float32": dict(XYZ_LOSS_WEIGHTS),
        "effective_xyz_loss_weights_after_runtime_cast": effective,
        "configured_matrix_float32_sha256": _sha256_float32(L1_LOSS_WEIGHT_MATRIX),
        "active_scalar_count_per_sample": ACTIVE_SCALAR_COUNT,
        "batch_size": int(action_loss.shape[0]),
        "mask_binary_exact": True,
        "loss_tensor_mutated": False,
        "mask_tensor_mutated": False,
        "claim_boundary": "The effective coefficients are the detached values obtained by casting the frozen float32 matrix to the observed action-loss dtype/device. This artifact does not prove gradient correctness by itself and does not change the loss graph.",
    }


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "L1RuntimeWeightEvidenceError",
    "capture_effective_weight_evidence",
    "canonical_json_sha256",
]
