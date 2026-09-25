#!/usr/bin/env python3
"""PI0246 P-arm same-forward physical-position loss overlay."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from pi0245_physical_delta import decode_xyz


P_REF_M = 0.05
LAMBDA_POS = 0.1


def physical_position_loss(
    *,
    predicted_velocity: torch.Tensor,
    clean_action: torch.Tensor,
    noisy_trajectory: torch.Tensor,
    tau: torch.Tensor,
    action_mask: torch.Tensor,
    state_eef: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Mean valid-point squared XYZ error after the pinned physical decoder."""
    batch = clean_action.shape[0]
    expected = (batch, 40, 132)
    for name, value in (
        ("predicted_velocity", predicted_velocity),
        ("clean_action", clean_action),
        ("noisy_trajectory", noisy_trajectory),
        ("action_mask", action_mask),
    ):
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {expected}")
    if tuple(state_eef.shape) != (batch, 9) or not torch.isfinite(state_eef).all():
        raise ValueError("state_eef must be finite with shape [batch,9]")
    if tuple(tau.shape) not in ((batch,), (batch, 1)) or not torch.isfinite(tau).all():
        raise ValueError("tau must be finite with shape [batch] or [batch,1]")
    tau = tau.reshape(batch, 1, 1)
    if torch.any((tau < 0) | (tau > 1)):
        raise ValueError("tau must be in [0,1]")
    if torch.any((action_mask[..., :3] < 0) | (action_mask[..., :3] > 1)):
        raise ValueError("XYZ mask must be in [0,1]")
    valid = (action_mask[..., :3] > 0).all(dim=-1)
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("no valid XYZ points")

    with torch.autocast(device_type=predicted_velocity.device.type, enabled=False):
        local_prediction = (
            noisy_trajectory[..., :3].float()
            + (1.0 - tau.float()) * predicted_velocity[..., :3].float()
        )
        predicted_world = decode_xyz(
            local_prediction, state_eef.float(), scale.float(), offset.float()
        )
        expert_world = decode_xyz(
            clean_action[..., :3].float(), state_eef.float(), scale.float(), offset.float()
        )
        residual_m = predicted_world - expert_world
        point_loss = (residual_m / P_REF_M).square().sum(dim=-1)
        loss = (point_loss * valid.float()).sum() / valid.sum()
    if not torch.isfinite(loss):
        raise ValueError("physical-position loss is nonfinite")
    return loss, {
        "valid_points": valid_count,
        "saturated_xyz": int((local_prediction.abs() > 1).sum().item()),
        "physical_residual_max_m": float(residual_m.detach().abs().max().item()),
    }


def combine_loss(native: torch.Tensor, position: torch.Tensor, coefficient: float = LAMBDA_POS) -> torch.Tensor:
    return native + coefficient * position


def install(
    head_cls: type,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    treatment: bool = True,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    if getattr(head_cls, "_pi0246_position_overlay", False):
        raise RuntimeError("PI0246 physical-position overlay already installed")
    if treatment and audit_path is None:
        raise RuntimeError("P treatment requires a same-forward audit path")
    if audit_path is not None and audit_path.exists():
        raise RuntimeError(f"refusing to append to an existing forward audit: {audit_path}")

    native_forward = head_cls.forward
    audit: dict[str, Any] = {"forward_calls": 0, "first_forward": None}

    def forward(self: Any, backbone_output: Any, action_input: Any) -> Any:
        actions = action_input.action
        state_norm = action_input.state[:, -1, :9]
        captured: dict[str, list[torch.Tensor]] = {"noise": [], "tau": [], "noisy": [], "pred": []}
        native_randn = torch.randn
        native_sample_time = self.sample_time
        had_instance_sample_time = "sample_time" in self.__dict__

        def randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = native_randn(*args, **kwargs)
            if tuple(value.shape) == tuple(actions.shape) and value.device == actions.device:
                captured["noise"].append(value)
            return value

        def sample_time(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = native_sample_time(*args, **kwargs)
            captured["tau"].append(value)
            return value

        def on_encoder(_module: Any, args: tuple[Any, ...]) -> None:
            captured["noisy"].append(args[0])

        def on_decoder(_module: Any, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            captured["pred"].append(output)

        handles = [
            self.action_encoder.register_forward_pre_hook(on_encoder),
            self.action_decoder.register_forward_hook(on_decoder),
        ]
        try:
            torch.randn = randn
            self.sample_time = sample_time
            output = native_forward(self, backbone_output, action_input)
        finally:
            torch.randn = native_randn
            if had_instance_sample_time:
                self.sample_time = native_sample_time
            else:
                del self.sample_time
            for handle in handles:
                handle.remove()

        counts = {key: len(values) for key, values in captured.items()}
        if counts != {"noise": 1, "tau": 1, "noisy": 1, "pred": 1}:
            raise RuntimeError(f"native forward observer cardinality drift: {counts}")
        noise, tau, noisy, pred = (captured[key][0] for key in ("noise", "tau", "noisy", "pred"))
        if not torch.allclose(
            noisy, (1 - tau[:, None, None]) * noise + tau[:, None, None] * actions,
            atol=1e-4, rtol=1e-4,
        ):
            raise RuntimeError("captured native noise/tau/noisy trajectory mismatch")

        state_scale = (state_max.to(state_norm.device).float() - state_min.to(state_norm.device).float()) / 2.0
        state_offset = (state_max.to(state_norm.device).float() + state_min.to(state_norm.device).float()) / 2.0
        state_eef = state_norm.float().clamp(-1, 1) * state_scale + state_offset
        position, details = physical_position_loss(
            predicted_velocity=pred[:, -actions.shape[1]:], clean_action=actions,
            noisy_trajectory=noisy, tau=tau, action_mask=output["action_mask"],
            state_eef=state_eef, scale=scale.to(actions.device), offset=offset.to(actions.device),
        )
        native = output["loss"]
        if treatment:
            output["loss"] = combine_loss(native, position)
        audit["forward_calls"] += 1
        row = {
            "forward_index": audit["forward_calls"],
            "native_loss": float(native.detach().float().item()),
            "physical_position_loss": float(position.detach().float().item()),
            "total_loss": float(output["loss"].detach().float().item()),
            "tau": tau.detach().float().cpu().reshape(-1).tolist(),
            **details,
        }
        if audit["first_forward"] is None:
            audit["first_forward"] = row
        if treatment:
            assert audit_path is not None
            with audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                if audit["forward_calls"] % 100 == 0:
                    os.fsync(stream.fileno())
        return output

    head_cls.forward = forward
    head_cls._pi0246_position_overlay = True
    return audit
