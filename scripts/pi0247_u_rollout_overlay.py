#!/usr/bin/env python3
"""PI0247 U: full differentiable S4 rollout endpoint loss."""

from __future__ import annotations

import json
import hashlib
import math
import os
import numpy as np
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.checkpoint import checkpoint

from pi0245_physical_delta import decode_xyz


N_STEPS = 4
DT_S = 0.05
P_REF_M = 0.05
V_REF_MPS = 1.0
LAMBDA_POS = 0.1
LAMBDA_DELTA = 0.1


@contextmanager
def _module_eval(module: Any, audit: dict[str, Any] | None = None, phase: str | None = None):
    modes = [(child, child.training) for child in module.modules()]
    previous_phase = audit.get("phase") if audit is not None else None
    try:
        for child, _ in modes:
            child.training = False
        if audit is not None and phase is not None:
            audit["phase"] = phase
        yield
    finally:
        for child, training in modes:
            child.training = training
        if audit is not None:
            audit["phase"] = previous_phase


def _euler_rollout(
    initial: torch.Tensor,
    step_fn: Callable[..., torch.Tensor],
    context: Iterable[torch.Tensor],
    *,
    steps: int,
    checkpointed: bool,
    context_fn: Callable[[], tuple[Any, Any]] | None = None,
    state_hook: Callable[[int, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    if steps < 1:
        raise ValueError("Euler rollout requires at least one step")
    context = tuple(context)
    actions = initial
    states: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    dt = 1.0 / steps
    for index in range(steps):
        if checkpointed:
            def run(current: torch.Tensor, *extra: torch.Tensor, _index: int = index) -> torch.Tensor:
                return step_fn(current, _index, *extra)

            velocity = checkpoint(
                run,
                actions,
                *context,
                use_reentrant=False,
                preserve_rng_state=True,
                context_fn=context_fn or (lambda: (nullcontext(), nullcontext())),
            )
        else:
            velocity = step_fn(actions, index, *context)
        actions = actions + dt * velocity * torch.ones_like(actions)
        if state_hook is not None:
            state_hook(index + 1, actions)
        states.append(actions)
        velocities.append(velocity)
    return actions, states, velocities


def _append_json(path: Path, row: dict[str, Any], count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        if count % 100 == 0:
            os.fsync(stream.fileno())


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().contiguous()
    metadata = json.dumps({
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
    }, sort_keys=True, separators=(",", ":")).encode()
    raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(metadata + b"\0" + raw).hexdigest(),
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
        "byte_count": len(raw),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    return _tensor_identity(value)["sha256"]


def _physical_losses(
    generated: torch.Tensor,
    expert: torch.Tensor,
    action_mask: torch.Tensor,
    state_eef: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    expected = (expert.shape[0], 40, 132)
    if tuple(generated.shape) != expected or tuple(expert.shape) != expected or tuple(action_mask.shape) != expected:
        raise ValueError("U physical loss requires [batch,40,132] actions and mask")
    if not torch.isfinite(generated).all() or not torch.isfinite(expert).all():
        raise ValueError("U actions contain nonfinite values")
    valid_points = (action_mask[..., :3] > 0).all(dim=-1)
    valid_transitions = valid_points[:, 1:] & valid_points[:, :-1]
    point_count = int(valid_points.sum().item())
    transition_count = int(valid_transitions.sum().item())
    if not point_count or not transition_count:
        raise ValueError("U physical objective has no valid XYZ points/transitions")
    with torch.autocast(device_type=generated.device.type, enabled=False):
        generated_xyz = decode_xyz(generated[..., :3].float(), state_eef.float(), scale.float(), offset.float())
        expert_xyz = decode_xyz(expert[..., :3].float(), state_eef.float(), scale.float(), offset.float())
        residual = generated_xyz - expert_xyz
        point_error = (residual / P_REF_M).square().sum(dim=-1)
        position = (point_error * valid_points.float()).sum() / valid_points.sum()
        velocity_error = (residual[:, 1:] - residual[:, :-1]) / DT_S / V_REF_MPS
        transition_error = velocity_error.square().sum(dim=-1)
        delta = (transition_error * valid_transitions.float()).sum() / valid_transitions.sum()
    if not torch.isfinite(position) or not torch.isfinite(delta):
        raise ValueError("U physical objective is nonfinite")
    return position, delta, {
        "valid_points": point_count,
        "valid_transitions": transition_count,
        "saturated_xyz": int((generated[..., :3].detach().abs() > 1).sum().item()),
        "physical_residual_max_m": float(residual.detach().abs().max().item()),
    }


def install(
    head_cls: type,
    *,
    scale: torch.Tensor,
    offset: torch.Tensor,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
    audit_path: Path,
    parity_path: Path,
    processor_getter: Callable[[], Any],
    embodiment_tag: str,
    resume_after_step: int = 0,
    expected_r_star: str = "S4",
) -> dict[str, Any]:
    if expected_r_star != "S4":
        raise RuntimeError("this PI0247 overlay is bound only to the frozen S4 runtime")
    existing_rows = []
    existing_parity = None
    if resume_after_step:
        if not audit_path.is_file() or not parity_path.is_file():
            raise RuntimeError("exact U resume requires the existing audit and parity receipt")
        existing_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        existing_parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if (len(existing_rows) != resume_after_step
                or [int(row["forward_index"]) for row in existing_rows] != list(range(1, resume_after_step + 1))
                or existing_parity.get("status") != "PASS_PI0247_U_S4_PARITY"):
            raise RuntimeError("existing U evidence does not bind the exact resume cursor")
    elif audit_path.exists() or parity_path.exists():
        raise RuntimeError("refusing to overwrite an existing PI0247 U audit")
    native_forward = head_cls.forward
    audit: dict[str, Any] = {
        "native_forwards": resume_after_step,
        "training_rollouts": resume_after_step,
        "parity_nonselection_rollouts": 2 if existing_parity else 0,
        "head_evaluations": {"parity_reference": 4 if existing_parity else 0,
                              "parity_candidate": 4 if existing_parity else 0,
                              "rollout_forward": 4 * resume_after_step,
                              "backward_recompute": 4 * resume_after_step},
        "parity": existing_parity,
        "first_update_euler_gradient_norms": {},
        "native_mode_snapshot": None,
        "mode_restoration_forward_checks": resume_after_step,
        "mode_restoration_after_backward_checks": 0,
        "forwards": resume_after_step,
    }

    def head_velocity(
        self: Any,
        actions: torch.Tensor,
        index: int,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: Any,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        phase = audit.get("phase") or "rollout_forward"
        audit["head_evaluations"][phase] = audit["head_evaluations"].get(phase, 0) + 1
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        t_discretized = int((index / float(self.num_inference_timesteps)) * self.num_timestep_buckets)
        timesteps = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = self.action_encoder(actions, timesteps, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
            )
        prediction = self.action_decoder(model_output, embodiment_id)
        return prediction[:, -self.action_horizon :]

    def rollout(
        self: Any,
        backbone_output: Any,
        vl_embeds: torch.Tensor,
        state_raw: torch.Tensor,
        embodiment_id: torch.Tensor,
        epsilon: torch.Tensor,
        *,
        checkpointed: bool,
        grad_hook: Callable[[int, torch.Tensor], None] | None = None,
        phase: str = "rollout_forward",
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        if int(self.num_inference_timesteps) != N_STEPS or int(self.action_horizon) != 40 or int(self.action_dim) != 132:
            raise RuntimeError("pinned action-head horizon/runtime differs from PI0247 S4/H40")
        with _module_eval(self):
            state_features = self.state_encoder(state_raw.view(state_raw.shape[0], 1, -1), embodiment_id)
        context = (state_features, vl_embeds)

        def step(actions: torch.Tensor, index: int, state: torch.Tensor, vl: torch.Tensor) -> torch.Tensor:
            return head_velocity(self, actions, index, state, vl, backbone_output, embodiment_id)

        def checkpoint_contexts() -> tuple[Any, Any]:
            return (
                _module_eval(self, audit, phase),
                _module_eval(self, audit, "backward_recompute"),
            )

        if checkpointed:
            return _euler_rollout(
                epsilon, step, context, steps=N_STEPS, checkpointed=True,
                context_fn=checkpoint_contexts, state_hook=grad_hook,
            )
        with _module_eval(self, audit, phase):
            return _euler_rollout(
                epsilon, step, context, steps=N_STEPS, checkpointed=False,
                state_hook=grad_hook,
            )

    def verify_parity(
        self: Any,
        raw_backbone: Any,
        processed_backbone: Any,
        state_raw: torch.Tensor,
        embodiment_id: torch.Tensor,
        native_epsilon: torch.Tensor,
        stock_noise: torch.Tensor,
    ) -> dict[str, Any]:
        from transformers.feature_extraction_utils import BatchFeature

        condition = BatchFeature(data={"state": state_raw, "embodiment_id": embodiment_id})
        if "action" in condition:
            raise RuntimeError("PI0247 U parity condition unexpectedly contains an expert action")
        raw_input = BatchFeature(data={key: value for key, value in raw_backbone.items()})
        reference_states: list[torch.Tensor] = []
        reference_velocities: list[torch.Tensor] = []
        h1 = self.action_encoder.register_forward_pre_hook(
            lambda _module, args: reference_states.append(args[0].detach().clone())
        )
        h2 = self.action_decoder.register_forward_hook(
            lambda _module, _args, output: reference_velocities.append(
                output[:, -self.action_horizon :].detach().clone()
            )
        )
        original_randn = torch.randn
        substitutions = 0
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        def replay_noise(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal substitutions
            requested = kwargs.get("size", args[0] if args else None)
            if requested is not None and tuple(requested) == tuple(stock_noise.shape):
                requested_device = kwargs.get("device", stock_noise.device)
                requested_dtype = kwargs.get("dtype", stock_noise.dtype)
                if torch.device(requested_device) == stock_noise.device and requested_dtype == stock_noise.dtype:
                    substitutions += 1
                    return stock_noise
            return original_randn(*args, **kwargs)

        try:
            torch.randn = replay_noise
            with _module_eval(self):
                reference = self.get_action(raw_input, condition, options=None).action_pred
        finally:
            torch.randn = original_randn
            h1.remove()
            h2.remove()
        reference_cpu_rng = torch.random.get_rng_state()
        reference_cuda_rng = torch.cuda.get_rng_state_all() if cuda_rng is not None else None
        if not torch.equal(cpu_rng, reference_cpu_rng):
            raise RuntimeError("PI0247 U native S4 reference consumed the CPU RNG beyond the shared epsilon")
        if cuda_rng is not None and any(
            not torch.equal(before, after)
            for before, after in zip(cuda_rng, reference_cuda_rng)
        ):
            raise RuntimeError("PI0247 U native S4 reference consumed the CUDA RNG beyond the shared epsilon")
        if substitutions != 1 or len(reference_states) != N_STEPS or len(reference_velocities) != N_STEPS:
            raise RuntimeError(
                f"PI0247 U reference parity cardinality mismatch: noise={substitutions}, "
                f"encoder={len(reference_states)}, decoder={len(reference_velocities)}"
            )
        with torch.no_grad():
            candidate, candidate_states, candidate_velocities = rollout(
                self, processed_backbone, processed_backbone.backbone_features, state_raw,
                embodiment_id, stock_noise, checkpointed=False, phase="parity_candidate",
            )
        candidate_cpu_rng = torch.random.get_rng_state()
        candidate_cuda_rng = torch.cuda.get_rng_state_all() if cuda_rng is not None else None
        if not torch.equal(reference_cpu_rng, candidate_cpu_rng):
            raise RuntimeError("PI0247 U S4 candidate parity consumed the CPU RNG")
        if cuda_rng is not None and any(
            not torch.equal(before, after)
            for before, after in zip(reference_cuda_rng, candidate_cuda_rng)
        ):
            raise RuntimeError("PI0247 U S4 candidate parity consumed the CUDA RNG")
        step_state_max = [
            float((reference_states[i].float() - (stock_noise if i == 0 else candidate_states[i - 1].detach()).float()).abs().max().item())
            for i in range(N_STEPS)
        ]
        step_velocity_max = [
            float((reference_velocities[i].float() - candidate_velocities[i].detach().float()).abs().max().item())
            for i in range(N_STEPS)
        ]
        raw_max = float((reference.float() - candidate.detach().float()).abs().max().item())
        state_eef = _state_eef(state_raw, state_min, state_max)
        scale_device = scale.to(device=reference.device, dtype=torch.float32)
        offset_device = offset.to(device=reference.device, dtype=torch.float32)
        physical_ref = decode_xyz(reference[..., :3].float(), state_eef, scale_device, offset_device)
        physical_candidate = decode_xyz(candidate.detach()[..., :3].float(), state_eef, scale_device, offset_device)
        xyz_max_m = float((physical_ref - physical_candidate).abs().max().item())
        processor = processor_getter()
        if processor is None:
            raise RuntimeError("the fixed TRAIN processor was not captured for full U decoder parity")
        from pi0228_train_calib_checkpoint_selection import _build_chunk
        from go2arx5_vla.safety import TrajectorySafetyGateV1
        from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
        from gr00t.data.embodiment_tags import EmbodimentTag

        tag = EmbodimentTag(embodiment_tag)
        processor_state = processor.state_action_processor
        state_np = state_raw.detach().float().cpu().numpy()
        state_groups: dict[str, np.ndarray] = {}
        state_start = 0
        for group in processor_state.modality_configs[tag.value]["state"].modality_keys:
            dim = int(processor_state.norm_params[tag.value]["state"][group]["dim"].item())
            state_groups[group] = state_np[..., state_start : state_start + dim]
            state_start += dim
        decode_state = processor_state.unapply_state(state_groups, tag.value)
        def decode_and_gate(raw: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
            raw_np = raw.detach().float().cpu().numpy()
            groups = processor.decode_action(raw_np, tag, decode_state)
            decoded = np.concatenate((np.asarray(groups["eef_9d"]), np.asarray(groups["gripper_position"])), axis=-1)
            physical_state = np.asarray(decode_state["eef_9d"], dtype=np.float64)[0, -1]
            chunk = _build_chunk(decoded, chunk_id="pi0247-u-stock-parity")
            gate_result = TrajectorySafetyGateV1().evaluate(
                chunk, now=0.0, achieved_position=physical_state[:3],
                achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(physical_state[3:9].reshape(1, 6)))[0],
            ).as_dict()
            return decoded, gate_result

        decoded_ref, gate_ref = decode_and_gate(reference)
        decoded_candidate, gate_candidate = decode_and_gate(candidate)
        decoded_max = float(np.max(np.abs(decoded_ref.astype(np.float64) - decoded_candidate.astype(np.float64))))
        gate_issues_equal = gate_ref.get("issues") == gate_candidate.get("issues")
        gate_equal = gate_ref.get("decision") == gate_candidate.get("decision") and gate_issues_equal
        step_records = []
        for index in range(N_STEPS):
            candidate_input = stock_noise if index == 0 else candidate_states[index - 1].detach()
            step_records.append({
                "step": index + 1,
                "reference_input": _tensor_identity(reference_states[index]),
                "candidate_input": _tensor_identity(candidate_input),
                "reference_velocity": _tensor_identity(reference_velocities[index]),
                "candidate_velocity": _tensor_identity(candidate_velocities[index]),
            })
        parity_ok = (raw_max <= 1e-4 and xyz_max_m <= 1e-5
                     and decoded_max <= 1e-4 and gate_equal
                     and max(step_state_max + step_velocity_max) <= 1e-4)
        result = {
            "status": "PASS_PI0247_U_S4_PARITY" if parity_ok else "FAIL_PI0247_U_S4_PARITY",
            "reference": "pinned Gr00tN1d7ActionHead.get_action; label-free condition; shared stock-dtype epsilon",
            "runtime_id": "S4", "steps": N_STEPS, "native_noise_substitutions": substitutions,
            "rng_audit": {"cpu_unchanged_reference": True, "cuda_unchanged_reference": True,
                          "cpu_unchanged_candidate": True, "cuda_unchanged_candidate": True},
            "expert_action_in_condition": False, "rtc_options": None,
            "native_epsilon": _tensor_identity(native_epsilon),
            "stock_noise": _tensor_identity(stock_noise), "step_tensor_identities": step_records,
            "step_input_max_abs": step_state_max, "step_velocity_max_abs": step_velocity_max,
            "raw_final_max_abs": raw_max, "decoded_xyz_max_abs_m": xyz_max_m,
            "decoded_action_max_abs": decoded_max,
            "gate_decision_equal": gate_ref.get("decision") == gate_candidate.get("decision"),
            "gate_issues_equal": gate_issues_equal, "reference_gate": gate_ref,
            "candidate_gate": gate_candidate,
            "nonselection_complete_rollouts": 2, "head_evaluations_reference": N_STEPS,
            "head_evaluations_candidate": N_STEPS, "global_rng_consumed": False,
        }
        if result["status"] != "PASS_PI0247_U_S4_PARITY":
            raise RuntimeError(f"PI0247 U exact-runtime parity failed: {result}")
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        with parity_path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        audit["parity"] = result
        audit["parity_nonselection_rollouts"] += 2
        audit["head_evaluations"]["parity_reference"] += N_STEPS
        return result

    def forward(self: Any, backbone_output: Any, action_input: Any) -> Any:
        from transformers.feature_extraction_utils import BatchFeature

        raw_backbone = BatchFeature(data={key: value for key, value in backbone_output.items()})
        state_raw = action_input.state
        embodiment_id = action_input.embodiment_id
        expert = action_input.action
        captured: list[torch.Tensor] = []
        captured_time: list[torch.Tensor] = []
        original_randn = torch.randn
        original_sample_time = self.sample_time
        had_instance_sample_time = "sample_time" in self.__dict__

        def capture_noise(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_randn(*args, **kwargs)
            if tuple(value.shape) == tuple(expert.shape) and value.device == expert.device and value.dtype == expert.dtype:
                captured.append(value)
            return value

        def capture_time(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_sample_time(*args, **kwargs)
            captured_time.append(value)
            return value

        try:
            torch.randn = capture_noise
            self.sample_time = capture_time
            native = native_forward(self, backbone_output, action_input)
        finally:
            torch.randn = original_randn
            if had_instance_sample_time:
                self.sample_time = original_sample_time
            else:
                del self.sample_time
        if len(captured) != 1 or len(captured_time) != 1:
            raise RuntimeError(
                f"PI0247 U expected one native epsilon/tau, observed noise={len(captured)} tau={len(captured_time)}"
            )
        epsilon = captured[0]
        vl_embeds = native["backbone_features"]
        stock_noise = epsilon.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
        mask = native["action_mask"]
        mode_snapshot = [bool(child.training) for child in self.modules()]
        if audit["native_mode_snapshot"] is None:
            audit["native_mode_snapshot"] = mode_snapshot
        elif audit["native_mode_snapshot"] != mode_snapshot:
            raise RuntimeError("PI0247 U action-head modes drifted between native forwards")
        processed_backbone = BatchFeature(data={key: value for key, value in backbone_output.items()})
        if "action" in processed_backbone:
            processed_backbone.pop("action")
        if audit["parity"] is None:
            verify_parity(self, raw_backbone, processed_backbone, state_raw, embodiment_id, epsilon, stock_noise)

        call_index = audit["forwards"] + 1
        audit["native_forwards"] += 1
        audit["training_rollouts"] += 1
        grad_hook = None
        if call_index == 1:
            def record_chain_grad(index: int, state: torch.Tensor) -> None:
                if not state.requires_grad:
                    raise RuntimeError(f"PI0247 U Euler state {index} is detached")

                def save_gradient(gradient: torch.Tensor) -> torch.Tensor:
                    norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
                    audit["first_update_euler_gradient_norms"][str(index)] = norm
                    return gradient

                state.register_hook(save_gradient)

            grad_hook = record_chain_grad
        generated, rollout_states, rollout_velocities = rollout(
            self, processed_backbone, native["backbone_features"], state_raw,
            embodiment_id, stock_noise, checkpointed=True, grad_hook=grad_hook,
        )
        if [bool(child.training) for child in self.modules()] != mode_snapshot:
            raise RuntimeError("PI0247 U rollout failed to restore the native action-head modes")
        audit["mode_restoration_forward_checks"] += 1
        state_norm = state_raw[:, -1, :9]
        state_scale = (state_max.to(state_norm.device).float() - state_min.to(state_norm.device).float()) / 2
        state_offset = (state_max.to(state_norm.device).float() + state_min.to(state_norm.device).float()) / 2
        state_eef = state_norm.float().clamp(-1, 1) * state_scale + state_offset
        position, delta, details = _physical_losses(
            generated, expert, mask, state_eef, scale.to(expert.device), offset.to(expert.device)
        )
        total = native["loss"] + LAMBDA_POS * position + LAMBDA_DELTA * delta
        if not torch.isfinite(total):
            raise RuntimeError("PI0247 U total loss is nonfinite")
        audit["forwards"] = call_index
        row = {
            "forward_index": call_index,
            "native_loss": float(native["loss"].detach().float().item()),
            "native_epsilon": _tensor_identity(epsilon),
            "stock_initial_noise": _tensor_identity(stock_noise),
            "stock_vl_features": _tensor_identity(vl_embeds),
            "euler_states": [_tensor_identity(value) for value in rollout_states],
            "euler_velocities": [_tensor_identity(value) for value in rollout_velocities],
            "native_epsilon_sha256": _tensor_sha256(epsilon),
            "native_tau": captured_time[0].detach().float().cpu().reshape(-1).tolist(),
            "physical_position_loss": float(position.detach().float().item()),
            "physical_delta_loss": float(delta.detach().float().item()),
            "total_loss": float(total.detach().float().item()),
            "lambda_position": LAMBDA_POS, "lambda_delta": LAMBDA_DELTA,
            "runtime_id": "S4", "rollout_steps": N_STEPS,
            "rollout_count": 1, "rollout_head_evaluations": N_STEPS,
            "checkpointed_euler_steps": N_STEPS, "backward_recomputation_head_evaluations_expected": N_STEPS,
            "expert_action_in_condition": False, "active_action_modified": False,
            **details,
        }
        _append_json(audit_path, row, call_index)
        return {**native, "loss": total}

    head_cls.forward = forward
    head_cls._pi0247_u_rollout_overlay = True
    return audit


def _state_eef(state_raw: torch.Tensor, state_min: torch.Tensor, state_max: torch.Tensor) -> torch.Tensor:
    state_norm = state_raw[:, -1, :9]
    state_scale = (state_max.to(state_norm.device).float() - state_min.to(state_norm.device).float()) / 2
    state_offset = (state_max.to(state_norm.device).float() + state_min.to(state_norm.device).float()) / 2
    return state_norm.float().clamp(-1, 1) * state_scale + state_offset


def self_check() -> dict[str, Any]:
    from torch import nn
    from pi0246_physical_position import physical_position_loss
    from pi0245_physical_delta import physical_delta_loss

    module = nn.Dropout(p=0.5)
    module.train()
    parameter = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    epsilon = torch.tensor(0.61, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(-0.23, dtype=torch.float64)

    @contextmanager
    def eval_dropout():
        was_training = module.training
        module.training = False
        try:
            yield
        finally:
            module.training = was_training

    def contexts() -> tuple[Any, Any]:
        return eval_dropout(), eval_dropout()

    def step(value: torch.Tensor, index: int, weight: torch.Tensor) -> torch.Tensor:
        return torch.tanh(weight * module(value) + 0.07 * (index + 1))

    gradients: dict[int, float] = {}

    def track(index: int, state: torch.Tensor) -> None:
        def record(grad: torch.Tensor) -> torch.Tensor:
            gradients[index] = float(grad.detach().abs().item())
            return grad

        state.register_hook(record)

    result, states, _ = _euler_rollout(
        epsilon, step, (parameter,), steps=N_STEPS, checkpointed=True,
        context_fn=contexts, state_hook=track,
    )
    if module.training is not True:
        raise AssertionError("U toy rollout did not restore module mode")
    loss = (result - target).square()
    loss.backward()
    if module.training is not True or len(gradients) != N_STEPS or any(value <= 0 for value in gradients.values()):
        raise AssertionError("U toy chain gradient or mode restoration failed")

    analytic = float(parameter.grad.item())
    delta = 1e-5

    def evaluate(weight: float, start: float) -> float:
        w = torch.tensor(weight, dtype=torch.float64)
        x = torch.tensor(start, dtype=torch.float64)
        with eval_dropout():
            out, _, _ = _euler_rollout(
                x, step, (w,), steps=N_STEPS, checkpointed=False,
                context_fn=contexts,
            )
        return float((out - target).square().item())

    finite_difference = (evaluate(0.37 + delta, 0.61) - evaluate(0.37 - delta, 0.61)) / (2 * delta)
    if not math.isclose(analytic, finite_difference, rel_tol=1e-5, abs_tol=1e-7):
        raise AssertionError(f"U toy finite difference mismatch: autograd={analytic}, fd={finite_difference}")
    stock_noise = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    stock_identity = _tensor_identity(stock_noise)
    if (stock_identity != _tensor_identity(stock_noise.clone())
            or stock_identity == _tensor_identity(stock_noise.float())
            or stock_identity["byte_count"] != stock_noise.numel() * stock_noise.element_size()):
        raise AssertionError("U noise identity lost its native dtype or raw bytes")
    for tensor in (parameter.grad, epsilon.grad, *[state for state in states]):
        if tensor is None or not torch.isfinite(tensor).all():
            raise AssertionError("U toy gradient is missing or nonfinite")
    generated = torch.linspace(-0.8, 0.8, 40 * 132).reshape(1, 40, 132)
    expert = generated.flip(1) * 0.3
    noisy = generated * 0.25
    tau = torch.tensor([0.4])
    velocity = (generated - noisy) / (1.0 - tau[:, None, None])
    mask = torch.zeros_like(generated)
    mask[..., :10] = 1
    state_eef = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    scale = torch.full((40, 3), 0.05)
    offset = torch.zeros((40, 3))
    u_position, u_delta, _ = _physical_losses(generated, expert, mask, state_eef, scale, offset)
    l_position, _ = physical_position_loss(
        predicted_velocity=velocity, clean_action=expert, noisy_trajectory=noisy, tau=tau,
        action_mask=mask, state_eef=state_eef, scale=scale, offset=offset,
    )
    l_delta, _ = physical_delta_loss(
        predicted_velocity=velocity, clean_action=expert, noisy_trajectory=noisy, tau=tau,
        action_mask=mask, state_eef=state_eef, scale=scale, offset=offset,
    )
    position_difference = float((u_position - l_position).abs().item())
    delta_difference = float((u_delta - l_delta).abs().item())
    if position_difference > 1e-6 or delta_difference > 1e-6:
        raise AssertionError(
            f"U generated-action objective differs from L local-estimate objective: "
            f"position={position_difference}, delta={delta_difference}"
        )
    return {
        "status": "PASS_PI0247_U_CPU_CHAIN_GRADIENT",
        "steps": N_STEPS,
        "checkpointed": True,
        "use_reentrant": False,
        "preserve_rng_state": True,
        "analytic_parameter_gradient": analytic,
        "finite_difference_parameter_gradient": finite_difference,
        "step_state_gradient_norms": gradients,
        "module_mode_restored": module.training,
        "native_dtype_noise_identity": stock_identity,
        "local_estimate_objective_match": {
            "status": "PASS_PI0247_U_MATCHES_L_PHYSICAL_LOSSES",
            "position_abs_difference": position_difference,
            "delta_abs_difference": delta_difference,
        },
    }


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, sort_keys=True))
