#!/usr/bin/env python3
"""Observe one pinned native action-head forward; add only the PI0245 loss."""

from __future__ import annotations

from typing import Any

import torch

from pi0245_physical_delta import physical_delta_loss


def install(head_cls: type, *, scale: torch.Tensor, offset: torch.Tensor, state_min: torch.Tensor, state_max: torch.Tensor, treatment: bool) -> dict[str, Any]:
    if getattr(head_cls, "_pi0245_overlay", False):
        raise RuntimeError("PI0245 overlay already installed")
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

        handles = [self.action_encoder.register_forward_pre_hook(on_encoder), self.action_decoder.register_forward_hook(on_decoder)]
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
        if {key: len(values) for key, values in captured.items()} != {"noise": 1, "tau": 1, "noisy": 1, "pred": 1}:
            raise RuntimeError(f"native forward observer cardinality drift: { {key: len(values) for key, values in captured.items()} }")
        noise, tau, noisy, pred = (captured[key][0] for key in ("noise", "tau", "noisy", "pred"))
        if not torch.allclose(noisy, (1 - tau[:, None, None]) * noise + tau[:, None, None] * actions, atol=1e-4, rtol=1e-4):
            raise RuntimeError("captured native noise/tau/noisy trajectory mismatch")
        state_scale = (state_max.to(state_norm.device).float() - state_min.to(state_norm.device).float()) / 2.0
        state_offset = (state_max.to(state_norm.device).float() + state_min.to(state_norm.device).float()) / 2.0
        state_eef = state_norm.float().clamp(-1, 1) * state_scale + state_offset
        auxiliary, details = physical_delta_loss(
            predicted_velocity=pred[:, -actions.shape[1]:], clean_action=actions,
            noisy_trajectory=noisy, tau=tau, action_mask=output["action_mask"],
            state_eef=state_eef, scale=scale.to(actions.device), offset=offset.to(actions.device),
        )
        native = output["loss"]
        if treatment:
            output["loss"] = native + 0.1 * auxiliary
        audit["forward_calls"] += 1
        if audit["first_forward"] is None:
            audit["first_forward"] = {
                "native_loss": float(native.detach().float().item()),
                "physical_delta_loss": float(auxiliary.detach().float().item()),
                "total_loss": float(output["loss"].detach().float().item()),
                "tau": tau.detach().float().cpu().reshape(-1).tolist(),
                "noise_shape": list(noise.shape), "noise_dtype": str(noise.dtype),
                "action_shape": list(actions.shape), "action_dtype": str(actions.dtype),
                **details,
            }
        return output

    head_cls.forward = forward
    head_cls._pi0245_overlay = True
    return audit
