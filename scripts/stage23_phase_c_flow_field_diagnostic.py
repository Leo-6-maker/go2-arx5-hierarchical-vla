#!/usr/bin/env python3
"""Freeze the Stage23 Phase C fixed-noise flow-field diagnostic.

This file is a preparation harness, not an experiment launcher. The default
CLI is standard-library-only and never imports a model, constructs a model,
runs a forward pass, performs inference/training, submits PBS, or executes
WBC. Torch is imported lazily only by helpers that a separately authorized
future C1 runner may call.

The future C1 diagnostic is teacher-forced: BASE and O1@200 receive the same
TINY1_H2STATS action sample, the same full universal noise tensor, and the
same fixed time value. It evaluates the local flow vector field before any
free-running sampler is involved.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "STAGE23_PHASE_C_FLOW_FIELD_DIAGNOSTIC_PROTOCOL_V1"
HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
INACTIVE_DIM = UNIVERSAL_DIM - ACTIVE_DIM
NOISE_SEEDS = (2223, 2224, 2225)
T_QUANTILES = (0.10, 0.50, 0.90)
BETA_ALPHA = 1.5
BETA_BETA = 1.0
NOISE_S = 0.999
NUM_INFERENCE_STEPS = 4
ACTION_PERIOD_S = 0.05
TRAIN_SEED = 2222
SAMPLER_STEP_CANDIDATES = (4, 8, 16, 32)


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except AttributeError as exc:
        raise TypeError("expected an array-like value with a .shape attribute") from exc


def _validate_probe_time(t_value: float) -> float:
    value = float(t_value)
    if not math.isfinite(value) or value < 0.0 or value > NOISE_S:
        raise ValueError(f"t must be finite and in [0,{NOISE_S}], got {t_value!r}")
    return value


def official_t_quantile(
    quantile: float,
    *,
    beta_alpha: float = BETA_ALPHA,
    beta_beta: float = BETA_BETA,
    noise_s: float = NOISE_S,
) -> float:
    """Return a quantile of the actual official training time variable.

    Official GR00T samples ``b ~ Beta(alpha, beta)`` and then computes
    ``t = (1 - b) * noise_s``. For the frozen official ``Beta(1.5, 1.0)``
    distribution, ``F_b(x) = x**alpha`` and therefore the q-quantile of the
    transformed ``t`` variable is::

        t_q = noise_s * (1 - (1-q)**(1/alpha))

    The function intentionally refuses a different beta parameter rather than
    silently using a formula that would no longer be exact.
    """

    q = float(quantile)
    alpha = float(beta_alpha)
    beta = float(beta_beta)
    scale = float(noise_s)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0,1], got {quantile!r}")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("beta_alpha must be positive and finite")
    if not math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the exact closed-form quantile is frozen only for Beta(alpha, 1.0)")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("noise_s must be positive and finite")
    return scale * (1.0 - (1.0 - q) ** (1.0 / alpha))


def beta_sample_at_t_quantile(
    quantile: float, *, beta_alpha: float = BETA_ALPHA
) -> float:
    """Return the Beta sample b that induces the requested t quantile."""

    q = float(quantile)
    return (1.0 - q) ** (1.0 / float(beta_alpha))


def build_probe_grid(
    *,
    noise_seeds: Sequence[int] = NOISE_SEEDS,
    t_quantiles: Sequence[float] = T_QUANTILES,
) -> list[dict[str, Any]]:
    """Build the fixed 3-seed x 3-quantile C1 probe grid."""

    probes: list[dict[str, Any]] = []
    for noise_seed in noise_seeds:
        for quantile in t_quantiles:
            q = float(quantile)
            t_value = official_t_quantile(q)
            probes.append(
                {
                    "probe_id": f"c1_seed{int(noise_seed)}_tq{int(round(q * 100)):02d}",
                    "noise_seed": int(noise_seed),
                    "t_quantile": q,
                    "beta_sample_at_t_quantile": beta_sample_at_t_quantile(q),
                    "t_value": t_value,
                    "t_discretized": int(t_value * 1000.0),
                    "noise_shape": [1, HORIZON, UNIVERSAL_DIM],
                    "noise_generation": (
                        "torch.Generator(device='cpu').manual_seed(noise_seed); "
                        "torch.randn([1,40,132], generator=generator, device='cpu') "
                        "then transfer to the action device"
                    ),
                    "same_noise_across_base_and_o1": True,
                    "same_t_across_base_and_o1": True,
                    "official_target_time_variable": "t=(1-Beta(alpha=1.5,beta=1.0))*noise_s",
                }
            )
    return probes


def expected_mask_partition() -> dict[str, Any]:
    """Return the frozen universal-action shape and loss-mask partition."""

    return {
        "universal_shape": [1, HORIZON, UNIVERSAL_DIM],
        "physical_shape": [1, HORIZON, ACTIVE_DIM],
        "active_slice": [0, ACTIVE_DIM],
        "inactive_slice": [ACTIVE_DIM, UNIVERSAL_DIM],
        "active_dimensions": ACTIVE_DIM,
        "inactive_dimensions": INACTIVE_DIM,
        "active_elements_per_probe": HORIZON * ACTIVE_DIM,
        "inactive_elements_per_probe": HORIZON * INACTIVE_DIM,
        "inactive_loss_mask_value": 0.0,
        "inactive_dimensions_contribute_to_loss": False,
        "inactive_noise_intervention": "PROHIBITED_IN_PHASE_C",
    }


def _validate_flow_tensor_shape(value: Any, name: str) -> tuple[int, ...]:
    shape = _shape(value)
    if len(shape) != 3 or shape[1:] != (HORIZON, UNIVERSAL_DIM):
        raise ValueError(f"{name} must have shape [B,{HORIZON},{UNIVERSAL_DIM}], got {shape}")
    return shape


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "torch"


def _finite(value: Any) -> bool:
    """Check finiteness lazily for a future tensor/array call."""

    if _is_torch_tensor(value):
        import torch

        return bool(torch.isfinite(value).all().item())
    try:
        import numpy as np

        return bool(np.isfinite(value).all())
    except ImportError:
        return all(math.isfinite(float(item)) for item in value)


def build_fixed_flow_inputs(
    actions: Any,
    *,
    noise_seed: int,
    t_value: float,
) -> dict[str, Any]:
    """Build official-compatible fixed-noise teacher-forced flow inputs.

    This future-runner helper requires a Torch tensor and performs no model
    operation. It generates noise on a CPU ``torch.Generator`` so the random
    stream is explicit and can be shared between BASE and O1@200 before the
    tensor is moved to the model device.
    """

    _validate_flow_tensor_shape(actions, "actions")
    if not _is_torch_tensor(actions):
        raise TypeError("build_fixed_flow_inputs requires a torch.Tensor actions input")
    import torch

    if not actions.is_floating_point():
        raise TypeError("actions must use a floating-point dtype")
    if not bool(torch.isfinite(actions).all().item()):
        raise ValueError("actions contains NaN or Inf")
    t = _validate_probe_time(t_value)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    noise_cpu = torch.randn(
        tuple(int(dimension) for dimension in actions.shape),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    noise = noise_cpu.to(device=actions.device, dtype=actions.dtype)
    t_tensor = torch.full(
        (int(actions.shape[0]), 1, 1), t, device=actions.device, dtype=actions.dtype
    )
    noisy_action = (1.0 - t_tensor) * noise + t_tensor * actions
    target_velocity = actions - noise
    action_mask = torch.zeros_like(actions)
    action_mask[..., :ACTIVE_DIM] = 1.0
    return {
        "noise": noise,
        "noisy_action": noisy_action,
        "target_velocity": target_velocity,
        "action_mask": action_mask,
        "t": t_tensor,
        "t_discretized": torch.full(
            (int(actions.shape[0]),), int(t * 1000.0), device=actions.device, dtype=torch.long
        ),
        "noise_seed": int(noise_seed),
        "noise_generation": "cpu_generator_then_device_transfer",
    }


def validate_velocity_output(predicted_velocity: Any) -> dict[str, Any]:
    """Validate a future model-runner velocity output at the thin boundary."""

    shape = _validate_flow_tensor_shape(predicted_velocity, "predicted_velocity")
    return {"shape": list(shape), "finite": _finite(predicted_velocity)}


def implied_action_from_velocity(noise: Any, predicted_velocity: Any) -> Any:
    """Form ``action_hat = noise + predicted_velocity`` in universal space."""

    noise_shape = _validate_flow_tensor_shape(noise, "noise")
    velocity_shape = _validate_flow_tensor_shape(predicted_velocity, "predicted_velocity")
    if noise_shape != velocity_shape:
        raise ValueError(f"noise and predicted_velocity shapes differ: {noise_shape} vs {velocity_shape}")
    if not _finite(noise) or not _finite(predicted_velocity):
        raise ValueError("noise and predicted_velocity must be finite")
    implied = noise + predicted_velocity
    if not _finite(implied):
        raise ValueError("implied universal action contains NaN or Inf")
    return implied


def extract_active_action_for_official_decode(implied_universal_action: Any) -> Any:
    """Expose the active model-space ``[B,40,10]`` slice for official decode.

    The 122 padded dimensions are deliberately not decoded, scored, or used
    as a physical command. The returned slice is still in the official model
    action space; unnormalization and absolute EEF SO(3)/gripper decoding must
    be performed by the canonical processor/provider boundary in a future
    authorized runner.
    """

    shape = _validate_flow_tensor_shape(implied_universal_action, "implied_universal_action")
    if not _finite(implied_universal_action):
        raise ValueError("implied_universal_action contains NaN or Inf")
    return implied_universal_action[..., :ACTIVE_DIM]


# Keep a descriptive compatibility alias for callers that use the earlier
# draft name. It still returns the active model-space slice, never a hardware
# command and never a silently corrected trajectory.
decode_active_physical_action = extract_active_action_for_official_decode


def masked_flow_metrics(
    predicted_velocity: Any,
    target_velocity: Any,
    action_mask: Any,
) -> dict[str, Any]:
    """Compute active-loss and inactive-energy diagnostics for a future run."""

    shape = _validate_flow_tensor_shape(predicted_velocity, "predicted_velocity")
    if _shape(target_velocity) != shape or _shape(action_mask) != shape:
        raise ValueError("predicted_velocity, target_velocity, and action_mask must share a shape")
    if not _finite(predicted_velocity) or not _finite(target_velocity) or not _finite(action_mask):
        raise ValueError("flow metrics inputs must be finite")

    if _is_torch_tensor(predicted_velocity):
        import torch

        squared_error = (predicted_velocity - target_velocity).pow(2)
        active_mask = action_mask[..., :ACTIVE_DIM]
        active_error = squared_error[..., :ACTIVE_DIM] * active_mask
        active_denominator = active_mask.sum()
        active_mse = active_error.sum() / (active_denominator + 1e-6)
        inactive = predicted_velocity[..., ACTIVE_DIM:]
        inactive_target = target_velocity[..., ACTIVE_DIM:]
        return {
            "shape": list(shape),
            "active_mask_sum": float(active_denominator.detach().cpu().item()),
            "active_masked_mse": float(active_mse.detach().cpu().item()),
            "inactive_prediction_energy_mean": float(inactive.pow(2).mean().detach().cpu().item()),
            "inactive_target_energy_mean": float(inactive_target.pow(2).mean().detach().cpu().item()),
            "inactive_loss_mask_sum": float(action_mask[..., ACTIVE_DIM:].sum().detach().cpu().item()),
            "finite": True,
        }

    import numpy as np

    predicted = np.asarray(predicted_velocity)
    target = np.asarray(target_velocity)
    mask = np.asarray(action_mask)
    squared_error = (predicted - target) ** 2
    active_mask = mask[..., :ACTIVE_DIM]
    active_error = squared_error[..., :ACTIVE_DIM] * active_mask
    active_denominator = float(active_mask.sum())
    return {
        "shape": list(shape),
        "active_mask_sum": active_denominator,
        "active_masked_mse": float(active_error.sum() / (active_denominator + 1e-6)),
        "inactive_prediction_energy_mean": float(np.mean(predicted[..., ACTIVE_DIM:] ** 2)),
        "inactive_target_energy_mean": float(np.mean(target[..., ACTIVE_DIM:] ** 2)),
        "inactive_loss_mask_sum": float(mask[..., ACTIVE_DIM:].sum()),
        "finite": True,
    }


def protocol_plan() -> dict[str, Any]:
    """Return the complete machine-readable, non-executing Phase C plan."""

    probes = build_probe_grid()
    return {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_EXECUTED",
        "execution_boundary": {
            "model_construction": False,
            "model_forward": False,
            "model_inference": False,
            "training": False,
            "optimizer_step": False,
            "pbs_submission": False,
            "wbc_execution": False,
            "holdout_access": False,
            "inactive_noise_intervention": False,
        },
        "fixed_inputs": {
            "dataset": "TINY1_H2STATS",
            "sample": "first ordered snapshot in red_target_train_a",
            "model_comparison": ["BASE", "O1@200"],
            "train_seed_context": TRAIN_SEED,
            "action_horizon": HORIZON,
            "universal_action_dim": UNIVERSAL_DIM,
            "physical_action_dim": ACTIVE_DIM,
            "action_period_s": ACTION_PERIOD_S,
            "action_period_provenance": "TEST_ONLY_STAGE22_SIMULATION; not a production contract",
            "validation_target": "NOT READ",
            "locked_final_test_target": "NOT READ",
        },
        "official_flow_contract": {
            "time_distribution": "Beta(alpha=1.5,beta=1.0)",
            "time_transform": "t=(1-B)*noise_s, noise_s=0.999",
            "expert_action_space": "official processor-normalized universal action [1,40,132]",
            "noisy_action": "x_t=(1-t)*noise+t*expert_action",
            "target_velocity": "v_target=expert_action-noise",
            "t_discretized": "(t*1000).long()",
            "loss": "masked MSE over the first 10 physical dimensions",
            "padded_dimensions": "122 dimensions remain in the universal noisy tensor but have mask zero",
            "physical_decode": "active first 10 model-space dimensions -> official processor unapply_action -> physical [1,40,10]",
        },
        "probe_grid": probes,
        "probe_count": len(probes),
        "mask_partition": expected_mask_partition(),
        "c1_teacher_forced_local_field": {
            "status": "LOCKED_UNTIL_EXPLICIT_AUTHORIZATION",
            "comparison": "BASE vs O1@200",
            "same_training_sample": True,
            "same_full_noise_tensor": True,
            "same_t_value": True,
            "probe_count": 9,
            "runner_output": "predicted_velocity [1,40,132]",
            "implied_action": "action_hat=noise+predicted_velocity",
            "physical_decode": "take active first 10 dimensions -> [1,40,10]",
            "required_evaluation": [
                "active masked flow MSE",
                "finite output",
                "decoded position/rotation profile",
                "path-length ratio",
                "mean-speed ratio",
                "max-speed ratio",
                "roughness",
                "initial discontinuity",
                "TrajectorySafetyGate decision",
            ],
            "wbc": "NOT RUN",
            "interpretation_if_pass": "LOCAL_FLOW_OBJECTIVE_FIT_SUPPORTED",
            "interpretation_if_fail": "LOCAL_FLOW_OBJECTIVE_FIT_NOT_SUPPORTED",
        },
        "c2_sampler_diagnostic": {
            "status": "LOCKED_UNTIL_C1_PASS_AND_SAMPLER_PARITY_PASS",
            "candidate_steps": list(SAMPLER_STEP_CANDIDATES),
            "first_gate": "mirrored 4-step sampler must match official 4-step implementation",
            "physical_intervention": "none; no smoothing, clipping, retiming, or correction",
        },
        "stop_rules": [
            "C1 failure stops C2 and returns LOCAL_FLOW_OBJECTIVE_FIT_NOT_SUPPORTED",
            "C1 pass is not task-performance evidence and does not authorize WBC",
            "C2 is not started before official-vs-mirror 4-step parity",
            "do not change the inactive 122D noise or loss partition in Phase C",
            "do not read validation e or locked final f",
        ],
        "future_provenance_requirements": [
            "TINY1_H2STATS tree/hash and selected snapshot hash",
            "BASE checkpoint tree/hash and O1@200 checkpoint tree/hash",
            "GR00T source SHA and Cosmos revision/tree hash",
            "official modality/processor configuration hash",
            "noise seed and CPU-generator construction",
            "t quantile, transformed t, and t_discretized",
            "model runner output hash before any physical decode",
            "decoded [1,40,10] hash and all unmodified evaluator outputs",
        ],
    }


def assert_protocol_static(plan: Mapping[str, Any] | None = None) -> None:
    """Fail closed if the frozen non-execution protocol drifts."""

    candidate = protocol_plan() if plan is None else plan
    if candidate.get("schema") != SCHEMA:
        raise AssertionError("unexpected Phase C schema")
    if candidate.get("status") != "PREPARED_NOT_EXECUTED":
        raise AssertionError("Phase C preparation must remain non-executing")
    boundary = candidate.get("execution_boundary", {})
    if any(bool(value) for value in boundary.values()):
        raise AssertionError("Phase C preparation contains an enabled execution boundary")
    probes = list(candidate.get("probe_grid", []))
    if len(probes) != 9 or candidate.get("probe_count") != 9:
        raise AssertionError("Phase C requires exactly nine fixed probes")
    if {probe["noise_seed"] for probe in probes} != set(NOISE_SEEDS):
        raise AssertionError("Phase C noise seeds drifted")
    if {probe["t_quantile"] for probe in probes} != set(T_QUANTILES):
        raise AssertionError("Phase C t quantiles drifted")
    for seed in NOISE_SEEDS:
        seed_probes = [probe for probe in probes if probe["noise_seed"] == seed]
        values = [probe["t_value"] for probe in seed_probes]
        if values != sorted(values):
            raise AssertionError("Phase C t values are not ordered by quantile")
        for probe in seed_probes:
            expected = official_t_quantile(probe["t_quantile"])
            if not math.isclose(probe["t_value"], expected, rel_tol=0.0, abs_tol=1e-15):
                raise AssertionError("Phase C t quantile does not match official transform")
            if probe["noise_shape"] != [1, HORIZON, UNIVERSAL_DIM]:
                raise AssertionError("Phase C noise shape drifted")
            if not probe["same_noise_across_base_and_o1"] or not probe["same_t_across_base_and_o1"]:
                raise AssertionError("BASE/O1 fixed-input pairing is not enforced")
    mask = candidate.get("mask_partition", {})
    if mask.get("active_dimensions") != ACTIVE_DIM or mask.get("inactive_dimensions") != INACTIVE_DIM:
        raise AssertionError("Phase C action-mask partition drifted")
    if mask.get("inactive_dimensions_contribute_to_loss") is not False:
        raise AssertionError("inactive loss-mask contract drifted")
    c1 = candidate.get("c1_teacher_forced_local_field", {})
    if c1.get("comparison") != "BASE vs O1@200" or c1.get("wbc") != "NOT RUN":
        raise AssertionError("C1 scope drifted")
    c2 = candidate.get("c2_sampler_diagnostic", {})
    if c2.get("candidate_steps") != list(SAMPLER_STEP_CANDIDATES):
        raise AssertionError("C2 sampler candidates drifted")


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the non-executing Phase C protocol plan (the default)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run standard-library-only static protocol assertions",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        plan = protocol_plan()
        assert_protocol_static(plan)
        print(json.dumps({"schema": SCHEMA, "status": "STATIC_SELF_TEST_PASS", "probe_count": 9}, indent=2))
        return 0
    print(json.dumps(protocol_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
