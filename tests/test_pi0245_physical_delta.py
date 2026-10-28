#!/usr/bin/env python3
"""CPU-only PI0245 admission: math, autograd, and sealed decoder parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ("scripts" if (root / "scripts").is_dir() else "code")))
from pi0245_physical_delta import decode_xyz, physical_delta_loss


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_real(records: Path, binding: dict) -> tuple[int, float]:
    scale = torch.tensor(binding["scale_xyz_m"], dtype=torch.float64)
    offset = torch.tensor(binding["offset_xyz_m"], dtype=torch.float64)
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
    assert len(rows) == 16
    mismatch = 0.0
    for row in rows:
        path = Path(row["tensor_path"])
        assert sha(path) == row["tensor_sha256"]
        with np.load(path, allow_pickle=False) as npz:
            raw = torch.tensor(npz["raw"][:, :, :3], dtype=torch.float64)
            state = torch.tensor(npz["state"].reshape(1, 10)[:, :9], dtype=torch.float64)
            expected = npz["decoded"][0, :, :3]
        predicted = decode_xyz(raw, state, scale, offset).detach().numpy()[0]
        mismatch = max(mismatch, float(np.max(np.abs(predicted - expected))))
    assert mismatch < 1e-5, mismatch
    return len(rows), mismatch


def synthetic() -> None:
    state = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=torch.float64)
    scale = torch.full((40, 3), 0.1, dtype=torch.float64)
    scale[1] = 0.5
    offset = torch.arange(40, dtype=torch.float64)[:, None].expand(40, 3) * 0.01
    zero = torch.zeros((1, 40, 3), dtype=torch.float64)
    bias = torch.full_like(zero, 0.1)
    e = decode_xyz(bias, state, scale, offset) - decode_xyz(zero, state, scale, offset)
    assert abs(float((e[0, 1, 0] - e[0, 0, 0]) / 0.05) - 0.8) < 1e-6
    physical_bias = torch.full_like(zero, 0.02) / scale[None]
    e_constant = decode_xyz(physical_bias, state, scale, offset) - decode_xyz(zero, state, scale, offset)
    assert torch.max(torch.abs(torch.diff(e_constant, dim=1))).item() < 1e-7
    rotated = state.clone()
    rotated[:, 3:9] = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float64)
    er = decode_xyz(bias, rotated, scale, offset) - decode_xyz(zero, rotated, scale, offset)
    assert torch.allclose(torch.linalg.vector_norm(torch.diff(er, dim=1), dim=-1), torch.linalg.vector_norm(torch.diff(e, dim=1), dim=-1), atol=1e-7)
    outside = torch.full_like(zero, 2.0).requires_grad_()
    saturated = decode_xyz(outside, state, scale, offset).sum()
    assert torch.all(torch.autograd.grad(saturated, outside)[0] == 0)
    scale_zero = torch.zeros_like(scale)
    assert torch.allclose(decode_xyz(bias, state, scale_zero, offset), decode_xyz(zero, state, scale_zero, offset))
    std_zero = torch.zeros_like(scale)
    assert torch.allclose(decode_xyz(bias, state, std_zero, offset, mode="meanstd") - decode_xyz(zero, state, std_zero, offset, mode="meanstd"), bias, atol=1e-7)

    action = torch.zeros((1, 40, 132), dtype=torch.float64)
    action[0, :, 0] = torch.arange(40, dtype=torch.float64) * 0.01
    noise = torch.ones_like(action) * 0.03
    tau = torch.tensor([0.25], dtype=torch.float64)
    noisy = (1 - tau[:, None, None]) * noise + tau[:, None, None] * action
    mask = torch.zeros_like(action)
    mask[..., :10] = 1
    loss_oracle, audit = physical_delta_loss(predicted_velocity=action - noise, clean_action=action, noisy_trajectory=noisy, tau=tau, action_mask=mask, state_eef=state, scale=scale, offset=offset)
    assert loss_oracle.item() < 1e-9 and audit["valid_transitions"] == 39
    pred = torch.zeros_like(action, requires_grad=True)
    native = (pred - (action - noise)).square().mean()
    aux, audit = physical_delta_loss(predicted_velocity=pred, clean_action=action, noisy_trajectory=noisy, tau=tau, action_mask=mask, state_eef=state, scale=scale, offset=offset)
    assert aux.item() > 0 and audit["valid_transitions"] == 39
    direct_grad = torch.autograd.grad(native, pred, retain_graph=True)[0]
    zero_lambda_grad = torch.autograd.grad(native + 0.0 * aux, pred)[0]
    assert torch.equal(direct_grad, zero_lambda_grad)
    mask[:, 37, :3] = 0
    _, audit = physical_delta_loss(predicted_velocity=pred, clean_action=action, noisy_trajectory=noisy, tau=tau, action_mask=mask, state_eef=state, scale=scale, offset=offset)
    assert audit["valid_transitions"] == 37
    try:
        physical_delta_loss(predicted_velocity=pred, clean_action=action, noisy_trajectory=noisy, tau=tau, action_mask=torch.zeros_like(mask), state_eef=state, scale=scale, offset=offset)
    except ValueError as error:
        assert "no valid" in str(error)
    else:
        raise AssertionError("empty mask must fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--r0-records", type=Path, required=True)
    parser.add_argument("--r2-records", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    assert binding["schema"] == "PI0245_P0_EFFECTIVE_PROCESSOR_XYZ_BINDING_V1"
    synthetic()
    n0, m0 = check_real(args.r0_records, binding)
    n2, m2 = check_real(args.r2_records, binding)
    print(json.dumps({"status": "PASS_CPU_PI0245_PHYSICAL_DELTA", "real_rows": n0 + n2, "max_world_decode_abs_error_m": max(m0, m2), "model_forwards": 0, "optimizer_steps": 0}))


if __name__ == "__main__":
    main()
