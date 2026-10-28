"""Differentiable PI0252 peak-linear-speed margin for one H40 S4 rollout."""
from __future__ import annotations

import torch


HORIZON = 40
MARGIN_Z = 0.81
LAMBDA = 0.1


def margin_from_peak_z(peak_z: torch.Tensor) -> torch.Tensor:
    return torch.relu(peak_z - MARGIN_Z)


def peak_speed_margin(
    positions_xyz: torch.Tensor,
    dt_s: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if positions_xyz.shape != (1, HORIZON, 3):
        raise ValueError("PI0252 speed margin requires one physical [1,40,3] XYZ rollout")
    if action_mask.shape != (1, HORIZON, 132):
        raise ValueError("PI0252 speed margin requires the unchanged [1,40,132] action mask")
    dt = torch.as_tensor(dt_s, device=positions_xyz.device, dtype=torch.float32).reshape(-1)
    if dt.shape != (HORIZON - 1,) or not torch.isfinite(dt).all() or torch.any(dt <= 0):
        raise ValueError("PI0252 speed margin requires 39 finite positive timestamp deltas")
    valid = (action_mask[..., :3] > 0).all(dim=-1)
    if not bool(valid.all()):
        raise ValueError("PI0252 speed margin requires all 40 XYZ points and 39 transitions")
    with torch.autocast(device_type=positions_xyz.device.type, enabled=False):
        xyz = positions_xyz.float()
        if not torch.isfinite(xyz).all():
            raise ValueError("PI0252 physical XYZ rollout contains nonfinite values")
        z = (xyz[:, 1:] - xyz[:, :-1]).square().sum(dim=-1) / (2.0 * dt.unsqueeze(0)).square()
        peak_z, segment = z.max(dim=-1)
        margin = margin_from_peak_z(peak_z)
        peak_speed_mps = 2.0 * torch.sqrt(peak_z)
    return margin[0], peak_z[0], segment[0], peak_speed_mps[0]
