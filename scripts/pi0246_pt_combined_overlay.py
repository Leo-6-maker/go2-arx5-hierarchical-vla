#!/usr/bin/env python3
"""Single native-forward observer for PI0246's physical-position + delta arm."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from pi0245_physical_delta import physical_delta_loss
from pi0246_physical_position import physical_position_loss


LAMBDA_POS = 0.1
LAMBDA_DELTA = 0.1


def install(
    head_cls: type,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    audit_path: Path,
) -> dict[str, Any]:
    if getattr(head_cls, "_pi0246_pt_overlay", False):
        raise RuntimeError("PI0246 PT combined overlay already installed")
    if audit_path.exists():
        raise RuntimeError(f"refusing to append to an existing PT forward audit: {audit_path}")

    native_forward = head_cls.forward
    audit: dict[str, Any] = {"forward_calls": 0, "first_forward": None}

    def forward(self: Any, backbone_output: Any, action_input: Any) -> Any:
        actions = action_input.action
        state_norm = action_input.state[:, -1, :9]
        captured: dict[str, list[torch.Tensor]] = {key: [] for key in ("noise", "tau", "noisy", "pred")}
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

        state_scale = (state_max.to(state_norm.device).float() - state_min.to(state_norm.device).float()) / 2
        state_offset = (state_max.to(state_norm.device).float() + state_min.to(state_norm.device).float()) / 2
        state_eef = state_norm.float().clamp(-1, 1) * state_scale + state_offset
        kwargs = {
            "predicted_velocity": pred[:, -actions.shape[1]:],
            "clean_action": actions,
            "noisy_trajectory": noisy,
            "tau": tau,
            "action_mask": output["action_mask"],
            "state_eef": state_eef,
            "scale": scale.to(actions.device),
            "offset": offset.to(actions.device),
        }
        position, pos_details = physical_position_loss(**kwargs)
        delta, delta_details = physical_delta_loss(**kwargs)
        if pos_details["saturated_xyz"] != delta_details["saturated_xyz"]:
            raise RuntimeError("position/delta loss did not use the same decoded native prediction")
        native = output["loss"]
        total = native + LAMBDA_POS * position + LAMBDA_DELTA * delta
        output["loss"] = total

        audit["forward_calls"] += 1
        row = {
            "forward_index": audit["forward_calls"],
            "native_loss": float(native.detach().float().item()),
            "physical_position_loss": float(position.detach().float().item()),
            "physical_delta_loss": float(delta.detach().float().item()),
            "total_loss": float(total.detach().float().item()),
            "tau": tau.detach().float().cpu().reshape(-1).tolist(),
            "valid_points": pos_details["valid_points"],
            "valid_transitions": delta_details["valid_transitions"],
            "saturated_xyz": pos_details["saturated_xyz"],
            "physical_residual_max_m": pos_details["physical_residual_max_m"],
        }
        if audit["first_forward"] is None:
            audit["first_forward"] = row
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            if audit["forward_calls"] % 100 == 0:
                os.fsync(stream.fileno())
        return output

    head_cls.forward = forward
    head_cls._pi0246_pt_overlay = True
    return audit
