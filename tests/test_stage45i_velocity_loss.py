"""CPU contract checks for the PI0209 velocity intervention."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla import stage45i_velocity_loss as velocity


ROOT = Path(__file__).resolve().parents[1]


def test_contract_and_finite_difference_are_frozen() -> None:
    contract = velocity.contract()
    assert contract["lambda_vel"] == 0.05
    assert contract["period_s"] == 0.05
    assert contract["velocity_steps"] == 39
    assert contract["structured_action_shape"] == [40, 132]

    physical = np.zeros((40, 10), dtype=np.float32)
    physical[:, 0] = np.arange(40, dtype=np.float32) * 0.01
    original = physical.copy()
    result = velocity.finite_difference_xyz(physical)
    assert result.shape == (39, 3)
    assert np.allclose(result[:, 0], 0.2, rtol=0.0, atol=1e-6)
    assert np.array_equal(physical, original)


def test_torch_loss_is_finite_and_differentiable() -> None:
    torch = pytest.importorskip("torch")
    lower = np.full(132, -1.0, dtype=np.float32)
    upper = np.full(132, 1.0, dtype=np.float32)
    clean = torch.zeros((1, 40, 132), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        clean[:, 0, 0] = 0.2
    state = torch.zeros((1, 1, 10), dtype=torch.float32)
    state[:, :, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    expert = np.zeros((40, 10), dtype=np.float32)

    loss = velocity.physical_xyz_velocity_loss_torch(
        torch, clean, expert, state, lower, upper
    )
    assert bool(torch.isfinite(loss).item())
    loss.backward()
    assert clean.grad is not None
    assert bool(torch.isfinite(clean.grad).all().item())


def test_static_surface_has_no_top_level_model_import() -> None:
    tree = ast.parse(
        (ROOT / "src/go2arx5_vla/stage45i_velocity_loss.py").read_text(
            encoding="utf-8"
        )
    )
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    assert not top_level.intersection({"torch", "transformers", "gr00t", "cuda"})
