#!/usr/bin/env python3
"""Thin differentiable XYZ mirror of the pinned dense-relative decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


DT = 0.05
V_REF = 1.0


def _check(tensor: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if tensor.ndim != len(shape) or any(want is not None and got != want for got, want in zip(tensor.shape, shape)):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains nonfinite values")


def decode_xyz(
    normalized_xyz: torch.Tensor,
    state_eef: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    *,
    mode: str = "minmax",
) -> torch.Tensor:
    """Original inverse normalization, then fixed-reference SE(3) composition."""
    batch, horizon, dimensions = normalized_xyz.shape
    if dimensions != 3:
        raise ValueError("XYZ dimension must equal 3")
    _check(state_eef, (batch, 9), "state_eef")
    _check(scale, (horizon, 3), "scale")
    _check(offset, (horizon, 3), "offset")
    _check(normalized_xyz, (batch, horizon, 3), "normalized_xyz")
    if mode == "minmax":
        relative = normalized_xyz.clamp(-1.0, 1.0) * scale + offset
    elif mode == "meanstd":
        relative = torch.where(scale == 0, normalized_xyz, normalized_xyz * scale + offset)
    else:
        raise ValueError(f"unknown normalization mode {mode}")
    rows = state_eef[:, 3:9].reshape(batch, 2, 3)
    row0 = F.normalize(rows[:, 0], dim=-1)
    row1 = F.normalize(rows[:, 1] - (row0 * rows[:, 1]).sum(-1, keepdim=True) * row0, dim=-1)
    rotation = torch.stack((row0, row1, torch.cross(row0, row1, dim=-1)), dim=1)
    return relative @ rotation.transpose(1, 2) + state_eef[:, None, :3]


def physical_delta_loss(
    *,
    predicted_velocity: torch.Tensor,
    clean_action: torch.Tensor,
    noisy_trajectory: torch.Tensor,
    tau: torch.Tensor,
    action_mask: torch.Tensor,
    state_eef: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Physical increment residual from the *same* native flow forward."""
    batch = clean_action.shape[0]
    for name, value in (("predicted_velocity", predicted_velocity), ("clean_action", clean_action), ("noisy_trajectory", noisy_trajectory), ("action_mask", action_mask)):
        _check(value, (batch, 40, 132), name)
    if tau.numel() != batch:
        raise ValueError("tau must have one value per batch item")
    _check(tau.reshape(batch, 1, 1), (batch, 1, 1), "tau")
    if torch.any((action_mask[..., :3] < 0) | (action_mask[..., :3] > 1)):
        raise ValueError("XYZ mask must be in [0,1]")
    valid_point = (action_mask[..., :3] > 0).all(dim=-1)
    valid = valid_point[:, 1:] & valid_point[:, :-1]
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("no valid XYZ transitions")
    with torch.autocast(device_type=predicted_velocity.device.type, enabled=False):
        clean = clean_action[..., :3].float()
        local = noisy_trajectory[..., :3].float() + (1.0 - tau.reshape(batch, 1, 1).float()) * predicted_velocity[..., :3].float()
        prediction_xyz = decode_xyz(local, state_eef.float(), scale.float(), offset.float())
        expert_xyz = decode_xyz(clean, state_eef.float(), scale.float(), offset.float())
        residual = prediction_xyz - expert_xyz
        velocity_error = (residual[:, 1:] - residual[:, :-1]) / DT / V_REF
        loss = (velocity_error.square().sum(-1) * valid.float()).sum() / valid.sum()
    return loss, {"valid_transitions": count, "saturated_xyz": int((local.abs() > 1).sum().item())}
