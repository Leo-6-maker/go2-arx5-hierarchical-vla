"""CPU-only tests for future L1 runtime effective-weight evidence."""

import pytest


torch = pytest.importorskip("torch")

from go2arx5_vla.stage45e_l1_loss import L1_LOSS_WEIGHT_MATRIX, canonical_action_mask
from go2arx5_vla.stage45e_l1_runtime_weight_evidence import (
    L1RuntimeWeightEvidenceError,
    capture_effective_weight_evidence,
)


def _tensors(dtype):
    action_loss = torch.zeros((1, 40, 132), dtype=dtype)
    action_mask = torch.as_tensor(canonical_action_mask(), dtype=torch.float32).unsqueeze(0)
    return action_loss, action_mask


def test_runtime_weight_evidence_is_readonly_and_tracks_action_loss_dtype() -> None:
    action_loss, action_mask = _tensors(torch.bfloat16)
    loss_before = action_loss.clone()
    mask_before = action_mask.clone()
    result = capture_effective_weight_evidence(torch, action_loss, action_mask)
    assert result["status"] == "CAPTURED_READONLY_FROM_FORWARD_TENSOR_DTYPE"
    assert result["action_loss_dtype"] == "torch.bfloat16"
    assert result["weight_tensor_dtype"] == "torch.bfloat16"
    assert result["active_scalar_count_per_sample"] == 190
    assert result["mask_binary_exact"] is True
    assert result["loss_tensor_mutated"] is False
    assert result["mask_tensor_mutated"] is False
    assert torch.equal(action_loss, loss_before)
    assert torch.equal(action_mask, mask_before)


def test_runtime_weight_evidence_reports_actual_cast_not_config_literal() -> None:
    action_loss, action_mask = _tensors(torch.bfloat16)
    result = capture_effective_weight_evidence(torch, action_loss, action_mask)
    weights = torch.as_tensor(L1_LOSS_WEIGHT_MATRIX, dtype=torch.bfloat16)
    expected = {
        "x": float(weights[0, 0].float().item()),
        "y": float(weights[0, 1].float().item()),
        "z": float(weights[0, 2].float().item()),
    }
    assert result["effective_xyz_loss_weights_after_runtime_cast"] == expected
    assert result["configured_xyz_loss_weights_float32"] == {
        "x": 1.401885,
        "y": 0.472436,
        "z": 1.125678,
    }


def test_runtime_weight_evidence_rejects_mask_drift() -> None:
    action_loss, action_mask = _tensors(torch.float32)
    action_mask[0, 0, 0] = 0.0
    with pytest.raises(L1RuntimeWeightEvidenceError, match="frozen T16"):
        capture_effective_weight_evidence(torch, action_loss, action_mask)
