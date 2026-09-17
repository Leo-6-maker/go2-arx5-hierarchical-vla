"""Stage45I expert-relative physical XYZ velocity auxiliary loss.

The module keeps the pinned GR00T action-head graph intact and adds one
small differentiable term after the flow prediction is converted back to a
clean StructuredActionV1 trajectory. It has no model/runtime imports at
module load time so the contract remains CPU-testable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_HORIZON = 40
ACTION_DIM = 132
GEOMETRY_K = 16
GEOMETRY_FUTURE_COUNT = GEOMETRY_K - 1
TIMING_COUNT = 16
PHYSICAL_DIM = 3
PERIOD_S = 0.05
DURATION_S = (ACTION_HORIZON - 1) * PERIOD_S
LAMBDA_VEL = 0.05


class VelocityLossContractError(ValueError):
    """Raised when the Stage45I velocity contract is violated."""


_CURRENT_CONTEXT: dict[str, Any] = {}
_RUNTIME_COUNTERS = {"action_head_forwards": 0, "velocity_terms": 0}


def contract() -> dict[str, Any]:
    return {
        "schema": "STAGE45I_EXPERT_RELATIVE_XYZ_VELOCITY_LOSS_V1",
        "intervention": "EXPERT_RELATIVE_PHYSICAL_XYZ_ADJACENT_STEP_VELOCITY",
        "lambda_vel": LAMBDA_VEL,
        "period_s": PERIOD_S,
        "duration_s": DURATION_S,
        "velocity_steps": ACTION_HORIZON - 1,
        "velocity_channels": [0, 1, 2],
        "structured_action_shape": [ACTION_HORIZON, ACTION_DIM],
        "physical_label_shape": [ACTION_HORIZON, 10],
        "formula": "mean(((p_pred[t+1]-p_pred[t])/0.05-(p_expert[t+1]-p_expert[t])/0.05)^2)",
        "applied_only_to": "rows with legitimately bound expert physical H40 labels",
        "target_leakage": False,
        "threshold_or_output_postprocessing": False,
    }


def finite_difference_xyz(physical: Any, *, period_s: float = PERIOD_S) -> np.ndarray:
    """Return physical XYZ adjacent-step velocities without mutating labels."""

    value = np.asarray(physical, dtype=np.float64)
    if value.shape[-2:] != (ACTION_HORIZON, 10):
        raise VelocityLossContractError(
            f"expert physical labels must end in [40,10], got {value.shape}"
        )
    if not np.isfinite(value).all() or not np.isfinite(period_s) or period_s <= 0.0:
        raise VelocityLossContractError("physical labels and period must be finite")
    return np.diff(value[..., :PHYSICAL_DIM], axis=-2) / float(period_s)


def reset_runtime_counters() -> None:
    _RUNTIME_COUNTERS.update(action_head_forwards=0, velocity_terms=0)


def runtime_counters() -> dict[str, int]:
    return {key: int(value) for key, value in _RUNTIME_COUNTERS.items()}


def set_current_expert_label(physical: Any, sample_id: str) -> None:
    """Set the out-of-band label for the next batch (batch1, no workers)."""

    value = np.asarray(physical, dtype=np.float32)
    if value.shape != (ACTION_HORIZON, 10) or not np.isfinite(value).all():
        raise VelocityLossContractError(f"expert label shape/finite drift: {sample_id}")
    # ponytail: one batch1/no-worker context avoids adding target fields to the model input ABI.
    _CURRENT_CONTEXT.clear()
    _CURRENT_CONTEXT.update({"sample_id": str(sample_id), "physical": value.copy()})


def _take_current_expert_label() -> tuple[np.ndarray, str]:
    value = _CURRENT_CONTEXT.pop("physical", None)
    sample_id = str(_CURRENT_CONTEXT.pop("sample_id", ""))
    if value is None:
        raise VelocityLossContractError("expert physical label context was not set")
    return value, sample_id


def _check_bounds(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=np.float32).reshape(-1)
    hi = np.asarray(upper, dtype=np.float32).reshape(-1)
    if lo.shape != (ACTION_DIM,) or hi.shape != (ACTION_DIM,):
        raise VelocityLossContractError("structured normalization bounds must have shape [132]")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise VelocityLossContractError("structured normalization bounds are invalid")
    return lo, hi


def _rot6d_rows_to_matrix_torch(torch_module: Any, value: Any) -> Any:
    first = value[..., :3]
    second = value[..., 3:]
    first = first / torch_module.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1e-12)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = second / torch_module.linalg.vector_norm(second, dim=-1, keepdim=True).clamp_min(1e-12)
    third = torch_module.linalg.cross(first, second, dim=-1)
    return torch_module.stack((first, second, third), dim=-2)


def _physical_xyz_from_clean_normalized(
    torch_module: Any,
    clean_normalized: Any,
    state: Any,
    lower: Any,
    upper: Any,
) -> Any:
    if clean_normalized.ndim != 3 or tuple(clean_normalized.shape[1:]) != (ACTION_HORIZON, ACTION_DIM):
        raise VelocityLossContractError("clean normalized action shape drift")
    # ponytail: keep the frozen model ABI, but do the geometric auxiliary in
    # fp32 so bf16 autocast cannot perturb the decoder-side physical target.
    work = clean_normalized.to(dtype=torch_module.float32)
    if state.ndim == 3:
        state = state.reshape(state.shape[0], -1)
    if state.ndim != 2 or state.shape[1] < 10:
        raise VelocityLossContractError("runtime StateV1 shape drift")
    state = state.to(dtype=torch_module.float32)
    lo = torch_module.as_tensor(lower, dtype=work.dtype, device=work.device)
    hi = torch_module.as_tensor(upper, dtype=work.dtype, device=work.device)
    values = (torch_module.clamp(work, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo) + lo
    anchor_position = state[:, :3]
    anchor_rotation = _rot6d_rows_to_matrix_torch(torch_module, state[:, 3:9])
    relative = values[:, :GEOMETRY_FUTURE_COUNT, :PHYSICAL_DIM]
    future = anchor_position[:, None, :] + torch_module.einsum(
        "bij,bnj->bni", anchor_rotation, relative
    )
    controls = torch_module.cat((anchor_position[:, None, :], future), dim=1)

    timing_logits = values[:, :TIMING_COUNT - 1, 9]
    fractions = torch_module.softmax(timing_logits, dim=-1)
    zero = torch_module.zeros((fractions.shape[0], 1), dtype=fractions.dtype, device=fractions.device)
    control_times = DURATION_S * torch_module.cumsum(torch_module.cat((zero, fractions), dim=1), dim=1)
    query = torch_module.arange(ACTION_HORIZON, dtype=clean_normalized.dtype, device=clean_normalized.device) * PERIOD_S
    query = query[None, :].expand(clean_normalized.shape[0], -1)
    indices = torch_module.searchsorted(control_times, query, right=True) - 1
    indices = indices.clamp(0, TIMING_COUNT - 2)
    left_t = torch_module.gather(control_times, 1, indices)
    right_t = torch_module.gather(control_times, 1, indices + 1)
    alpha = (query - left_t) / (right_t - left_t).clamp_min(torch_module.finfo(clean_normalized.dtype).eps)
    progress = (indices.to(clean_normalized.dtype) + alpha) / float(GEOMETRY_K - 1)
    geometry_index = torch_module.floor(progress * float(GEOMETRY_K - 1)).long().clamp(0, GEOMETRY_K - 2)
    geometry_left = torch_module.gather(controls, 1, geometry_index[..., None].expand(-1, -1, PHYSICAL_DIM))
    geometry_right = torch_module.gather(controls, 1, (geometry_index + 1)[..., None].expand(-1, -1, PHYSICAL_DIM))
    geometry_alpha = (progress * float(GEOMETRY_K - 1) - geometry_index.to(clean_normalized.dtype)).unsqueeze(-1)
    return (1.0 - geometry_alpha) * geometry_left + geometry_alpha * geometry_right


def physical_xyz_velocity_loss_torch(
    torch_module: Any,
    clean_normalized: Any,
    expert_physical: Any,
    state: Any,
    lower: Any,
    upper: Any,
) -> Any:
    """Compute the differentiable physical XYZ velocity MSE."""

    expert = torch_module.as_tensor(expert_physical, dtype=clean_normalized.dtype, device=clean_normalized.device)
    if expert.ndim == 2:
        expert = expert.unsqueeze(0)
    if tuple(expert.shape[1:]) != (ACTION_HORIZON, 10) or not bool(torch_module.isfinite(expert).all().item()):
        raise VelocityLossContractError("expert physical tensor shape/finite drift")
    predicted_xyz = _physical_xyz_from_clean_normalized(torch_module, clean_normalized, state, lower, upper)
    target_xyz = expert[..., :PHYSICAL_DIM].detach()
    target_velocity = torch_module.diff(target_xyz, dim=-2) / PERIOD_S
    predicted_velocity = torch_module.diff(predicted_xyz, dim=-2) / PERIOD_S
    loss = torch_module.mean((predicted_velocity - target_velocity) ** 2)
    if not bool(torch_module.isfinite(loss).item()):
        raise VelocityLossContractError("velocity auxiliary loss is non-finite")
    return loss


def install_velocity_loss_patch(*, lower: Any, upper: Any) -> dict[str, Any]:
    """Patch only the pinned action-head forward with the fixed auxiliary term."""

    lo, hi = _check_bounds(lower, upper)
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    marker = "_stage45i_velocity_original_forward"
    original = getattr(Gr00tN1d7ActionHead, marker, None)
    if original is None:
        original = Gr00tN1d7ActionHead.forward
        setattr(Gr00tN1d7ActionHead, marker, original)

        def velocity_forward(self: Any, backbone_output: Any, action_input: Any) -> Any:
            import torch

            _RUNTIME_COUNTERS["action_head_forwards"] += 1
            captured: dict[str, Any] = {}

            def capture_encoder(_module: Any, args: tuple[Any, ...]) -> None:
                captured["noisy"] = args[0]

            def capture_decoder(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                captured["pred"] = output

            encoder_hook = self.action_encoder.register_forward_pre_hook(capture_encoder)
            decoder_hook = self.action_decoder.register_forward_hook(capture_decoder)
            original_sample_time = self.sample_time

            def capture_sample_time(batch_size: int, device: Any, dtype: Any) -> Any:
                sampled = original_sample_time(batch_size, device, dtype)
                captured["time"] = sampled
                return sampled

            self.sample_time = capture_sample_time
            state = action_input.state
            try:
                output = original(self, backbone_output, action_input)
            finally:
                self.sample_time = original_sample_time
                encoder_hook.remove()
                decoder_hook.remove()

            expert, sample_id = _take_current_expert_label()
            noisy = captured.get("noisy")
            predicted = captured.get("pred")
            sampled_time = captured.get("time")
            if noisy is None or predicted is None or sampled_time is None:
                raise VelocityLossContractError(f"forward capture incomplete: {sample_id}")
            predicted = predicted[:, -noisy.shape[1] :]
            clean_normalized = noisy + (1.0 - sampled_time[:, None, None]) * predicted
            aux = physical_xyz_velocity_loss_torch(torch, clean_normalized, expert, state, lo, hi)
            output["velocity_auxiliary_loss"] = aux
            output["loss"] = output["loss"] + LAMBDA_VEL * aux
            _RUNTIME_COUNTERS["velocity_terms"] += 1
            return output

        Gr00tN1d7ActionHead.forward = velocity_forward
        setattr(Gr00tN1d7ActionHead, "_stage45i_velocity_patch_schema", contract()["schema"])
        status = "INSTALLED"
    else:
        status = "ALREADY_INSTALLED"
    return {
        "status": status,
        "schema": contract()["schema"],
        "lambda_vel": LAMBDA_VEL,
        "lower_sha256": _array_sha256(lo),
        "upper_sha256": _array_sha256(hi),
        "original_forward_preserved": True,
    }


def _array_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(np.asarray(value, dtype=np.float32)).tobytes()).hexdigest()


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "DURATION_S",
    "LAMBDA_VEL",
    "PERIOD_S",
    "VelocityLossContractError",
    "contract",
    "finite_difference_xyz",
    "install_velocity_loss_patch",
    "physical_xyz_velocity_loss_torch",
    "reset_runtime_counters",
    "runtime_counters",
    "set_current_expert_label",
]
