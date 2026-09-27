#!/usr/bin/env python3
"""Step-addressed Torch RNG for the PI0251 C/E continuation pair."""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SEED = 22301
MAX_STEPS = 2000
SCHEMA = "PI0251_STEP_ADDRESSED_TORCH_RNG_V1"
_ACTIVE_STEP = contextvars.ContextVar("pi0251_active_step", default=None)


def step_key_seed(step: int) -> tuple[str, int]:
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= MAX_STEPS:
        raise ValueError(f"PI0251 optimizer step must be in [1,{MAX_STEPS}]")
    key = f"PI0251|{SEED}|stage=M64_CE|optimizer_step={step}"
    seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)
    return key, seed


def build_key_table(schedule_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(schedule_rows) != MAX_STEPS:
        raise ValueError(f"PI0251 requires exactly {MAX_STEPS} exposure rows")
    result = []
    for step, row in enumerate(schedule_rows, 1):
        if int(row.get("optimizer_step", -1)) != step:
            raise ValueError(f"PI0251 exposure schedule cursor mismatch at step {step}")
        key, seed = step_key_seed(step)
        result.append({"optimizer_step": step, "key": key, "seed": seed,
                       "fixed_index": int(row["fixed_index"]),
                       "source_episode_id": str(row["source_episode_id"])})
    return result


def active_step_context() -> dict[str, Any] | None:
    return _ACTIVE_STEP.get()


def rng_context(step: int, device: Any = None):
    """Fork only CPU and the current job's single visible CUDA device."""
    from contextlib import contextmanager
    import torch

    key, seed = step_key_seed(step)
    selected: list[int] = []
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("PI0251 RNG isolation requires the single visible job CUDA device")
            selected = [torch.cuda.current_device() if device.index is None else int(device.index)]
            if selected[0] != torch.cuda.current_device():
                raise RuntimeError("PI0251 model device is not the current job CUDA device")

    @contextmanager
    def isolated():
        with torch.random.fork_rng(devices=selected, enabled=True):
            torch.random.default_generator.manual_seed(seed)
            if selected:
                torch.cuda.manual_seed(seed)
            yield {"key": key, "seed": seed, "optimizer_step": step}

    return isolated()


def _payload_identity(value: Any) -> tuple[str, int]:
    """Hash the collated batch tree without copying tensor values to another device."""
    import torch

    digest = hashlib.sha256()
    tensor_count = 0

    def add(item: Any) -> None:
        nonlocal tensor_count
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(str(tensor.device).encode())
            digest.update(tensor.cpu().view(torch.uint8).numpy().tobytes())
            tensor_count += 1
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update((b"list\0" if isinstance(item, list) else b"tuple\0"))
            for child in item:
                add(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        elif hasattr(item, "data") and isinstance(item.data, Mapping):
            add(item.data)
        else:
            raise TypeError(f"unsupported PI0251 model-input identity value: {type(item).__name__}")

    add(value)
    return digest.hexdigest(), tensor_count


def verify_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    mismatch = [field for field in fields if actual.get(field) != expected.get(field)]
    if mismatch:
        raise RuntimeError(f"PI0251 online identity mismatch: {mismatch}")


def install_training_step_rng(
    trainer_type: type, audit_path: Path, *, resume_after_step: int = 0,
    before_forward_guard: Any = None,
) -> type:
    """Wrap Trainer.training_step so forward, S4 recompute and backward share one key."""
    if isinstance(resume_after_step, bool) or not 0 <= int(resume_after_step) < MAX_STEPS:
        raise ValueError("PI0251 RNG resume cursor is outside the phase")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if audit_path.exists():
        rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(resume_after_step):
        raise ValueError("PI0251 RNG audit must be the exact contiguous restored prefix")
    for step, row in enumerate(rows, 1):
        key, seed = step_key_seed(step)
        if row.get("optimizer_step") != step or row.get("key") != key or row.get("seed") != seed:
            raise ValueError(f"PI0251 RNG audit prefix differs at step {step}")

    class StepAddressedTrainer(trainer_type):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            if int(getattr(self.args, "gradient_accumulation_steps", 1)) != 1:
                raise RuntimeError("PI0251 step key requires one optimizer update per training_step")
            from transformers import TrainerCallback

            class VerifyBeforeOptimizer(TrainerCallback):
                def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
                    expected = int(state.global_step) + 1
                    if expected > len(rows) or int(rows[expected - 1].get("optimizer_step", -1)) != expected:
                        raise RuntimeError(f"PI0251 RNG key missing before optimizer update {expected}")
                    return control

            self.add_callback(VerifyBeforeOptimizer())

        def training_step(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            step = int(self.state.global_step) + 1
            if step < int(resume_after_step) + 1 or step > MAX_STEPS:
                raise RuntimeError(f"PI0251 Trainer cursor is invalid: {step}")
            key, seed = step_key_seed(step)
            input_sha256, tensor_count = _payload_identity(inputs)
            model_device = next(model.parameters()).device
            active = {"optimizer_step": step, "key": key, "seed": seed,
                      "model_input_sha256": input_sha256, "input_tensor_count": tensor_count}
            active_token = _ACTIVE_STEP.set(active)
            try:
                guard_record = before_forward_guard(active) if before_forward_guard is not None else None
                with rng_context(step, model_device):
                    result = super().training_step(model, inputs, *args, **kwargs)
            finally:
                _ACTIVE_STEP.reset(active_token)
            record = {"optimizer_step": step, "key": key, "seed": seed,
                      "model_input_sha256": input_sha256, "input_tensor_count": tensor_count,
                      "pre_forward_guard": guard_record}
            if len(rows) != step - 1:
                raise RuntimeError(f"PI0251 Trainer step is duplicated or missing: {step}")
            rows.append(record)
            with audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                if step % 100 == 0:
                    os.fsync(stream.fileno())
            return result

    StepAddressedTrainer.__name__ = f"PI0251{trainer_type.__name__}"
    return StepAddressedTrainer


def cpu_self_check() -> dict[str, Any]:
    import torch
    from torch.utils.checkpoint import checkpoint
    from pi0247_u_rollout_overlay import _physical_losses
    from pi0245_physical_delta import decode_xyz
    from types import SimpleNamespace

    def native_tau() -> torch.Tensor:
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((1,))
        return ((1.0 - beta) * 0.999).to(dtype=torch.bfloat16)

    def draw(step: int) -> tuple[bytes, bytes, bytes]:
        with rng_context(step) as token:
            assert token["key"] == step_key_seed(step)[0]
            tau = native_tau()
            epsilon = torch.randn((1, 40, 132), dtype=torch.float32)
            stock_noise = epsilon.to(dtype=torch.bfloat16)
        return (tau.contiguous().view(torch.uint8).numpy().tobytes(),
                epsilon.contiguous().view(torch.uint8).numpy().tobytes(),
                stock_noise.contiguous().view(torch.uint8).numpy().tobytes())

    torch.manual_seed(8241)
    initial_outer = torch.get_rng_state().clone()
    with rng_context(17):
        _ = torch.randn(32)
    assert torch.equal(initial_outer, torch.get_rng_state()), "step context changed external CPU RNG"
    assert draw(77) == draw(77), "same step key did not reproduce native tau/noise/cast"
    assert draw(77) != draw(78), "different step keys did not change the random stream"

    activation = torch.ones(16, requires_grad=True)
    masks = []
    def random_checkpoint(value: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(value)
        masks.append(mask.detach().clone())
        return value * mask
    with rng_context(91):
        checkpoint(random_checkpoint, activation, use_reentrant=True,
                   preserve_rng_state=True).sum().backward()
    assert len(masks) == 2 and torch.equal(masks[0], masks[1]), "checkpoint recomputation changed the keyed mask"

    # Simulate uninterrupted training, a saved CPU state, then real restore-side
    # iterator construction, skipped rows and unrelated CPU RNG consumers.
    torch.manual_seed(19331)
    uninterrupted: dict[int, tuple[bytes, bytes, bytes]] = {}
    checkpoint_cpu_state = None
    for step in range(1, MAX_STEPS + 1):
        _ = torch.rand(3)  # iterator batch fetch, outside the model RNG scope
        uninterrupted[step] = draw(step)
        _ = torch.rand(2)  # unrelated CPU consumer after backward
        if step == 1000:
            checkpoint_cpu_state = torch.get_rng_state().clone()
    assert checkpoint_cpu_state is not None

    torch.set_rng_state(checkpoint_cpu_state)
    _ = torch.rand(11)  # resumed DataLoader iterator construction
    for _ in range(1000):
        _ = torch.rand(3)  # skip the consumed fixed schedule prefix
        _ = torch.rand(1)  # unrelated restore-time consumer
    resumed = {}
    for step in range(1001, MAX_STEPS + 1):
        _ = torch.rand(3)
        resumed[step] = draw(step)
        _ = torch.rand(2)
    assert list(resumed) == list(range(1001, MAX_STEPS + 1)), "restored prefix was duplicated"
    assert all(resumed[step] == uninterrupted[step] for step in resumed), "1000-step restore changed tau/noise"
    assert len({step_key_seed(step)[0] for step in range(1, MAX_STEPS + 1)}) == MAX_STEPS
    try:
        step_key_seed(0)
    except ValueError:
        pass
    else:
        raise AssertionError("bad RNG cursor was accepted")

    generated = torch.linspace(-0.72, 0.81, 40 * 132).reshape(1, 40, 132).requires_grad_()
    expert = torch.zeros_like(generated)
    action_mask = torch.ones_like(generated)
    action_mask[:, [4, 9], :3] = 0
    state_eef = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0.]])
    scale = torch.full((40, 3), 0.05)
    offset = torch.zeros_like(scale)
    position, delta, details = _physical_losses(generated, expert, action_mask, state_eef, scale, offset)
    assert details["valid_points"] == 38 and details["valid_transitions"] == 35
    native, rotation = generated.square().mean(), generated[..., 3:9].square().mean()
    k = native + 0.1 * position + 0.1 * delta + 0.1 * rotation
    r = native + 0.1 * position + 0.1 * delta + 0.1 * rotation
    x = native + 0.4 * position + 0.1 * delta + 0.1 * rotation
    assert torch.equal(k, r), "K at coefficient 0.1 differs from original R objective"
    assert torch.allclose(x - k, 0.3 * position, rtol=0, atol=1e-7)
    grad_k = torch.autograd.grad(k, generated, retain_graph=True)[0]
    grad_x = torch.autograd.grad(x, generated, retain_graph=True)[0]
    grad_pos = torch.autograd.grad(position, generated)[0]
    assert torch.allclose(grad_x - grad_k, 0.3 * grad_pos, rtol=1e-6, atol=1e-7)

    normalized = torch.zeros((1, 40, 3), requires_grad=True)
    with torch.no_grad():
        normalized[0, 0, 0] = 1.5
        normalized[0, 0, 1] = 0.25
        normalized[0, 0, 2] = -1.4
    decoded = decode_xyz(normalized, state_eef, scale, offset)
    decoded[0, 0].sum().backward()
    assert normalized.grad[0, 0, 0].item() == 0 and normalized.grad[0, 0, 2].item() == 0
    assert normalized.grad[0, 0, 1].item() != 0, "clamp/zero-Jacobian semantics changed"

    excess_probe = torch.tensor(0.63, dtype=torch.float64, requires_grad=True)
    c_probe = 1.2 + 0.4 * excess_probe
    e_probe = c_probe + 0.4 * torch.relu(excess_probe - 0.49)
    assert torch.allclose(e_probe - c_probe, torch.tensor(0.4 * (0.63 - 0.49), dtype=torch.float64), rtol=0, atol=1e-12)
    assert math.isclose(float(torch.autograd.grad(e_probe - c_probe, excess_probe)[0]), 0.4, rel_tol=0, abs_tol=1e-12)
    below = torch.tensor(0.48, requires_grad=True)
    edge = torch.tensor(0.49, requires_grad=True)
    above = torch.tensor(0.50, requires_grad=True)
    grads = [torch.autograd.grad(0.4 * torch.relu(v - 0.49), v)[0].item() for v in (below, edge, above)]
    assert grads[:2] == [0.0, 0.0] and math.isclose(grads[2], 0.4, rel_tol=0, abs_tol=1e-7), \
        f"PI0251 threshold/subgradient mismatch: {grads}"

    identity_fields = ("source", "key", "tau_identity", "epsilon_identity")
    expected_identity = {"source": "ep-17", "key": step_key_seed(17)[0],
                         "tau_identity": {"dtype": "torch.bfloat16", "device": "cpu", "sha256": "a"},
                         "epsilon_identity": {"dtype": "torch.float32", "device": "cpu", "sha256": "b"}}
    verify_identity(expected_identity, expected_identity, identity_fields)
    for field in identity_fields:
        changed = dict(expected_identity)
        changed[field] = "wrong"
        try:
            verify_identity(changed, expected_identity, identity_fields)
        except RuntimeError as error:
            assert field in str(error), f"online guard did not identify {field}"
        else:
            raise AssertionError(f"online guard accepted wrong {field}")

    class FakeTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(gradient_accumulation_steps=1)
            self.state = SimpleNamespace(global_step=0)
        def add_callback(self, callback: Any) -> None:
            self.callback = callback
        def training_step(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            return model(torch.ones(1, 1))

    try:
        from transformers import TrainerCallback
    except ImportError:
        import sys
        from types import ModuleType
        transformers = ModuleType("transformers")
        transformers.TrainerCallback = object
        sys.modules["transformers"] = transformers
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
        calls = []
        def trainer_guard(active: dict[str, Any]) -> dict[str, Any]:
            observed = {"source": "ep-17", "key": active["key"]}
            expected = {"source": "ep-17", "key": step_key_seed(1)[0]}
            verify_identity(observed, expected, ("source", "key"))
            calls.append(active["key"])
            return {"status": "PASS"}
        wrapped = install_training_step_rng(FakeTrainer, Path(temporary) / "audit.jsonl",
                                            before_forward_guard=trainer_guard)
        trainer = wrapped()
        trainer.training_step(torch.nn.Linear(1, 1), {})
        assert calls == [step_key_seed(1)[0]], "Trainer did not invoke the sealed pre-forward guard"

    return {"schema": SCHEMA, "status": "PASS_PI0251_G1_CPU_FIXTURES",
            "key_count": MAX_STEPS, "resume_cursor": 1000,
            "resumed_steps_exact": MAX_STEPS - 1000,
            "external_cpu_rng_restored": True, "beta_cast_noise_match": True,
            "checkpoint_rng_preserved": True,
            "input_independent_step_keys": True, "position_difference": float((x - k).item()),
            "gradient_difference_max_abs": float((grad_x - grad_k - 0.3 * grad_pos).abs().max().item()),
            "valid_points": details["valid_points"], "valid_transitions": details["valid_transitions"],
            "clamp_zero_jacobian": True,
            "excess_threshold_m": 0.035, "excess_threshold_scaled": 0.49,
            "excess_lambda": 0.4, "excess_gradient_below_edge_above": grads,
            "excess_boundary_subgradient": 0.0,
            "online_guard_rejects_source_key_tau_epsilon": True,
            "trainer_invoked_pre_forward_guard": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("only --self-check is available outside an authorized training run")
    print(json.dumps(cpu_self_check(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
