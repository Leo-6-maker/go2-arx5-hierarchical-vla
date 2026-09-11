"""CPU/static tests for the PI0123 Stage45E-L1 intervention."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla import stage45e_l1_loss as l1


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_weight_matrix_and_binary_mask() -> None:
    contract = l1.loss_weight_contract(l1.canonical_action_mask())
    assert contract["active_scalar_count"] == 190
    assert contract["xyz_loss_weights"] == {
        "x": 1.401885,
        "y": 0.472436,
        "z": 1.125678,
    }
    assert l1.L1_LOSS_WEIGHT_MATRIX.shape == (40, 132)
    assert np.all(l1.L1_LOSS_WEIGHT_MATRIX[l1.canonical_action_mask() == 0] == 1.0)
    assert np.all(l1.canonical_action_mask()[:, 11:] == 0.0)


def test_fractional_boolean_and_string_mask_coercion_is_rejected() -> None:
    mask = l1.canonical_action_mask()
    fractional = mask.copy()
    fractional[0, 0] = 0.5
    with pytest.raises(l1.L1LossContractError, match="binary T16"):
        l1.loss_weight_contract(fractional)

    boolean = mask.astype(bool)
    with pytest.raises(l1.L1LossContractError, match="binary T16"):
        l1.loss_weight_contract(boolean)

    string = mask.astype(str)
    with pytest.raises(l1.L1LossContractError, match="binary T16"):
        l1.loss_weight_contract(string)


def test_static_surface_has_no_top_level_model_import() -> None:
    tree = ast.parse((ROOT / "src/go2arx5_vla/stage45e_l1_loss.py").read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    assert not top_level.intersection({"torch", "transformers", "gr00t", "cuda"})


def test_weighted_loss_matches_exact_mask_denominator() -> None:
    torch = pytest.importorskip("torch")
    mask = torch.from_numpy(l1.canonical_action_mask()).unsqueeze(0)
    masked_mse = torch.zeros((1, 40, 132), dtype=torch.float32)
    masked_mse[:, :15, 0] = 2.0
    masked_mse[:, :15, 1] = 3.0
    masked_mse[:, :15, 2] = 5.0
    masked_mse *= mask
    loss, weights = l1.weighted_loss_from_masked_mse_torch(torch, masked_mse, mask)
    expected = (15 * (2 * 1.401885 + 3 * 0.472436 + 5 * 1.125678)) / 190
    assert torch.allclose(loss, torch.tensor(expected), rtol=0.0, atol=1e-6)
    assert weights.shape == (1, 40, 132)


def test_runtime_loss_rejects_folded_or_nonbinary_mask() -> None:
    torch = pytest.importorskip("torch")
    good_mask = torch.from_numpy(l1.canonical_action_mask()).unsqueeze(0)
    masked_mse = torch.zeros((1, 40, 132), dtype=torch.float32)
    folded = good_mask.clone()
    folded[:, 0, 0] = 0.5
    with pytest.raises(l1.L1LossContractError, match="binary T16"):
        l1.weighted_loss_from_masked_mse_torch(torch, masked_mse, folded)
    boolean = good_mask.to(dtype=torch.bool)
    with pytest.raises(l1.L1LossContractError, match="binary T16"):
        l1.weighted_loss_from_masked_mse_torch(torch, masked_mse, boolean)

