#!/usr/bin/env python3
"""PI0244 R2: thin XYZ4 loss overlay on the sealed PI0243 fixed-view path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping


PROTOCOL_COMMIT = "984565af45097b547354e681fc0c63fc7fa304af"
AUDITED_PARENT = "48c732d2195af2820a804baaede68c96a0390dc3"
AUTHORIZATION = "PI0244_R2_XYZ4_RELEASE_GO"
EXPERIMENT = "PI0244X_XYZ4_M16_10K_V1"
FAMILY = EXPERIMENT
HF_REPO_ID = "leoyuer/pi0243-stage-a-admitted-v1"
HF_PREFIX = "pi0244x-xyz4-m16-10k-v1"
R0_EXPERIMENT = "PI0244_M16_10K_V1"
R0_CLOSURE_SHA256 = "4a64898d05377ad23a7022ae16921ceb2bb2d80e330cf7433e7c3595a5de9851"
R0_TRAINING_MANIFEST_SHA256 = "62487ffff722eb61e41e621044c40e8292d291aeb1583d3c48073f0620cf7b9a"
R0_UPDATE_AUDIT_SHA256 = "eb775e0e00108cb240e2060e052e2132d179436e4ef4ddd6d7f8a28aa8eaa7a5"
MAX_STEPS = 10000
CHECKPOINTS = [2500, 5000, 7500, 10000]
SCREENS = [0, 2500, 5000, 7500, 10000]
TRAINABLE_SLOTS = 10
ACTION_WIDTH = 132
HORIZON = 40


class R2Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any], *, once: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if once:
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            if path.read_text(encoding="utf-8") != data:
                raise R2Error(f"first-valid evidence already differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def xyz4_loss(
    action_loss: Any,
    action_mask: Any,
    *,
    xyz_weight: float = 4.0,
    capture_details: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if tuple(action_loss.shape) != tuple(action_mask.shape):
        raise R2Error(f"action loss/mask shape mismatch: {tuple(action_loss.shape)} != {tuple(action_mask.shape)}")
    if action_loss.ndim != 3 or action_loss.shape[1:] != (HORIZON, ACTION_WIDTH):
        raise R2Error(f"expected [B,40,132] action tensors; got {tuple(action_loss.shape)}")
    weights = action_loss.new_ones((1, 1, ACTION_WIDTH))
    weights[..., :3] = xyz_weight
    numerator = (weights * action_loss).sum()
    denominator = (weights * action_mask).sum()
    weighted = numerator / (denominator + 1e-6)
    if not capture_details:
        return weighted, {}
    mask = action_mask.detach()
    loss = action_loss.detach()
    if action_mask.shape[0] != 1:
        raise R2Error(f"R2 fixed16 recipe requires batch=1, got {action_mask.shape[0]}")
    if not bool(torch.all((mask == 0) | (mask == 1)).item()):
        raise R2Error("R2 action mask is not binary")
    actual_slots = mask.sum(dim=(0, 1))
    expected_slots = torch.zeros_like(actual_slots)
    expected_slots[:TRAINABLE_SLOTS] = HORIZON
    if not bool(torch.equal(actual_slots, expected_slots)):
        raise R2Error("R2 active action slots differ from the sealed slots 0..9 contract")
    values = {
        "formula": "sum(weight*already_masked_action_loss)/(sum(weight*action_mask)+1e-6)",
        "xyz_slots": [0, 1, 2],
        "xyz_weight": float(xyz_weight),
        "other_weight": 1.0,
        "action_shape": list(action_loss.shape),
        "action_loss_dtype": str(action_loss.dtype),
        "action_mask_dtype": str(action_mask.dtype),
        "action_device": str(action_loss.device),
        "mask_active_scalars": int(mask.sum().item()),
        "active_slot_counts": actual_slots.cpu().tolist(),
        "weighted_denominator": float(denominator.detach().float().item()),
        "weighted_numerator": float(numerator.detach().float().item()),
        "xyz_mask_scalars": int(mask[..., :3].sum().item()),
        "other_mask_scalars": int(mask[..., 3:].sum().item()),
        "xyz_action_loss_sum": float(loss[..., :3].sum().float().item()),
        "other_action_loss_sum": float(loss[..., 3:].sum().float().item()),
        "native_loss_from_components": float(
            (loss.sum() / (mask.sum() + 1e-6)).float().item()
        ),
        "weighted_loss": float(weighted.detach().float().item()),
    }
    return weighted, values


def _scalar(value: Any) -> float:
    return float(value.detach().float().cpu().item())


def patch_action_head(head_cls: type, evidence_path: Path, source_sha256: str) -> dict[str, Any]:
    if getattr(head_cls, "_pi0244_xyz4_patched", False):
        raise R2Error("XYZ4 action-head patch was already installed")
    original = head_cls.forward
    state: dict[str, Any] = {"forward_calls": 0, "first_forward": None}

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(self, *args, **kwargs)
        if not isinstance(output, Mapping) or not {"loss", "action_loss", "action_mask"}.issubset(output):
            raise R2Error("pinned native action-head output lacks loss/action_loss/action_mask")
        native = output["loss"]
        first = state["forward_calls"] == 0
        weighted, details = xyz4_loss(output["action_loss"], output["action_mask"], capture_details=first)
        output["loss"] = weighted
        state["forward_calls"] += 1
        if first:
            parity = abs(_scalar(native) - details["native_loss_from_components"])
            if not math.isfinite(parity) or parity > max(1e-7, abs(_scalar(native)) * 1e-5):
                raise R2Error(f"native loss no longer matches its returned action_loss/mask: delta={parity}")
            record = {
                "schema": "PI0244_R2_XYZ4_FIRST_NATIVE_FORWARD_V1",
                "status": "RECORDED",
                "job_id": os.environ.get("PBS_JOBID"),
                "head_class": f"{head_cls.__module__}.{head_cls.__qualname__}",
                "head_source_sha256": source_sha256,
                "native_loss": _scalar(native),
                "native_component_loss_delta": parity,
                **details,
            }
            write_json(evidence_path, record, once=True)
            state["first_forward"] = record
        return output

    head_cls.forward = forward
    head_cls._pi0244_xyz4_patched = True
    return state


def install_moment_callback(trainer_cls: type, evidence_path: Path) -> None:
    if getattr(trainer_cls, "_pi0244_xyz4_moments_patched", False):
        raise R2Error("first-update optimizer audit was already installed")
    from transformers import TrainerCallback

    class FirstUpdateMoments(TrainerCallback):
        def __init__(self, trainer: Any) -> None:
            self.trainer = trainer
            self.model = None
            self.written = False

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if int(state.global_step) != 0:
                raise R2Error("R2 must start from the initializer with a fresh optimizer at step zero")
            self.model = kwargs.get("model")
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if self.written or int(state.global_step) != 1:
                return control
            model = kwargs.get("model") or self.model
            optimizer = getattr(self.trainer, "optimizer", None)
            while optimizer is not None and not hasattr(optimizer, "state") and hasattr(optimizer, "optimizer"):
                optimizer = optimizer.optimizer
            if model is None or optimizer is None or not hasattr(optimizer, "state"):
                raise R2Error("cannot inspect the real optimizer state after R2 update one")
            named = dict(model.named_parameters())
            rows = []
            missing = []
            for name, parameter in sorted(named.items()):
                if not parameter.requires_grad:
                    continue
                values = optimizer.state.get(parameter, {})
                moments = {}
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = values.get(key)
                    if value is not None:
                        moments[key] = {
                            "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__,
                            "shape": list(value.shape) if hasattr(value, "shape") else [],
                            "value": float(value.item()) if key == "step" and hasattr(value, "item") else None,
                        }
                if not {"exp_avg", "exp_avg_sq"}.issubset(moments):
                    missing.append(name)
                rows.append(
                    {
                        "name": name,
                        "parameter_dtype": str(parameter.dtype),
                        "gradient_dtype": str(parameter.grad.dtype) if parameter.grad is not None else None,
                        "moments": moments,
                    }
                )
            report = {
                "schema": "PI0244_R2_FIRST_UPDATE_NAMED_OPTIMIZER_MOMENTS_V1",
                "status": "RECORDED",
                "job_id": os.environ.get("PBS_JOBID"),
                "global_step": int(state.global_step),
                "optimizer": type(optimizer).__name__,
                "named_parameter_count": len(named),
                "trainable_parameter_count": sum(int(p.requires_grad) for p in named.values()),
                "frozen_parameter_dtype_counts": {},
                "trainable_moment_records": rows,
                "trainable_without_exp_avg_or_exp_avg_sq": missing,
            }
            for parameter in named.values():
                if not parameter.requires_grad:
                    dtype = str(parameter.dtype)
                    report["frozen_parameter_dtype_counts"][dtype] = report["frozen_parameter_dtype_counts"].get(dtype, 0) + 1
            write_json(evidence_path, report, once=True)
            self.written = True
            return control

    original_init = trainer_cls.__init__

    def init_with_moment_callback(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.add_callback(FirstUpdateMoments(self))

    trainer_cls.__init__ = init_with_moment_callback
    trainer_cls._pi0244_xyz4_moments_patched = True


def self_test() -> None:
    import tempfile
    import torch

    torch.manual_seed(22301)
    prediction = torch.randn(1, HORIZON, ACTION_WIDTH, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.zeros_like(prediction)
    mask[..., :TRAINABLE_SLOTS] = 1.0
    per_element = (prediction - target).square()
    masked = per_element * mask
    native = masked.sum() / (mask.sum() + 1e-6)
    same, _ = xyz4_loss(masked, mask, xyz_weight=1.0)
    native_grad = torch.autograd.grad(native, prediction, retain_graph=True)[0]
    same_grad = torch.autograd.grad(same, prediction, retain_graph=True)[0]
    torch.testing.assert_close(same, native, rtol=0.0, atol=0.0)
    torch.testing.assert_close(same_grad, native_grad, rtol=0.0, atol=0.0)

    weighted, _ = xyz4_loss(masked, mask)
    gradient = torch.autograd.grad(weighted, prediction)[0]
    assert torch.isfinite(weighted) and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[..., TRAINABLE_SLOTS:]) == 0
    xyz_scale = (gradient[..., :3] / native_grad[..., :3]).mean()
    other_scale = (gradient[..., 3:TRAINABLE_SLOTS] / native_grad[..., 3:TRAINABLE_SLOTS]).mean()
    ratio = xyz_scale / other_scale
    torch.testing.assert_close(ratio, torch.tensor(4.0, dtype=ratio.dtype), rtol=1e-12, atol=1e-12)
    assert int(mask.sum().item()) == HORIZON * TRAINABLE_SLOTS

    class DummyHead:
        def forward(self, pred: Any, truth: Any, action_mask: Any) -> dict[str, Any]:
            action_loss = (pred - truth).square() * action_mask
            return {
                "loss": action_loss.sum() / (action_mask.sum() + 1e-6),
                "action_loss": action_loss,
                "action_mask": action_mask,
                "untouched": action_mask,
            }

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "first.json"
        state = patch_action_head(DummyHead, path, "self-test")
        dummy = DummyHead()
        out = dummy.forward(prediction.detach(), target, mask)
        expected, _ = xyz4_loss(out["action_loss"], mask)
        torch.testing.assert_close(out["loss"], expected, rtol=0.0, atol=0.0)
        assert out["untouched"] is mask and state["forward_calls"] == 1
        assert json.loads(path.read_text())["status"] == "RECORDED"
        del DummyHead._pi0244_xyz4_patched

        class DummyTrainer:
            def __init__(self) -> None:
                self.callbacks = []

            def add_callback(self, callback: Any) -> None:
                self.callbacks.append(callback)

        install_moment_callback(DummyTrainer, Path(td) / "moments.json")
        trainer = DummyTrainer()
        assert len(trainer.callbacks) == 1
        module = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(module.parameters())
        parameter = next(module.parameters())
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        trainer.optimizer = optimizer
        callback = trainer.callbacks[0]
        callback.on_train_begin(None, SimpleNamespace(global_step=0), None, model=module)
        callback.on_step_end(None, SimpleNamespace(global_step=1), None, model=module)
        moment_report = json.loads((Path(td) / "moments.json").read_text())
        assert moment_report["trainable_parameter_count"] == 1
        assert moment_report["trainable_without_exp_avg_or_exp_avg_sq"] == []
        assert moment_report["trainable_moment_records"][0]["moments"]["exp_avg"]["dtype"] == str(parameter.dtype)
        DummyTrainer.__init__ = object.__init__
        del DummyTrainer._pi0244_xyz4_moments_patched
    print(json.dumps({"status": "PASS_CPU_XYZ4_AUTOGRAD_AND_TRAINER_HOOKS", "w1_value_gradient_exact": True, "masked_inactive_zero": True, "xyz_vs_other_gradient_ratio": 4.0, "finite": True, "optimizer_step": 0, "groot_weight_loads": 0, "groot_forward_calls": 0}))


def load_base(code_root: Path) -> Any:
    path = code_root / "scripts" / "pi0243_staged_learning_to_sim.py"
    spec = importlib.util.spec_from_file_location("pi0244x_pi0243_base", path)
    if spec is None or spec.loader is None:
        raise R2Error(f"cannot load mature PI0243 stage controller: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    module.AUDITED_PARENT = AUDITED_PARENT
    module.AUTH = dict(module.AUTH)
    module.AUTH["a"] = AUTHORIZATION
    module.stage_spec = lambda stage: stage_spec(stage)
    return module


def stage_spec(stage: str) -> dict[str, Any]:
    if str(stage).lower() != "a":
        raise R2Error("R2 is the fixed16/10K stage only")
    return {
        "stage": "A",
        "view": "16",
        "max_steps": MAX_STEPS,
        "checkpoints": list(CHECKPOINTS),
        "screens": list(SCREENS),
    }


def _git_state(root: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if branch != "pi0244/r2-xyz4-release-20260923" or dirty.strip():
        raise R2Error(f"execution checkout is not the clean sealed R2 branch: {branch}, dirty={bool(dirty.strip())}")
    return {"branch": branch, "commit": head}


def _execution_binding(card: Mapping[str, Any], code_root: Path) -> dict[str, Any]:
    implementation = card["r2_implementation"]
    parent = card["r0_parent"]
    payload = {
        "schema": "PI0244_R2_EXECUTION_BINDING_V1",
        "protocol_commit": PROTOCOL_COMMIT,
        "git": _git_state(code_root),
        "runner_sha256": sha256_file(Path(implementation["runner_path"])),
        "pi0243_sha256": sha256_file(Path(implementation["pi0243_path"])),
        "pi0241_sha256": sha256_file(Path(implementation["pi0241_path"])),
        "pi0228_sha256": sha256_file(Path(implementation["pi0228_path"])),
        "persistence_sha256": sha256_file(Path(implementation["persistence_path"])),
        "storage_preflight_sha256": sha256_file(Path(implementation["storage_preflight_path"])),
        "groot_source_commit": card["paths"]["groot_source_commit"],
        "source_paths": {key: card["paths"][key] for key in ("code_root", "groot_source", "stage10_root", "train_dataset", "p237_root", "modality_config", "processor_base_model", "initializer", "cosmos_snapshot", "cosmos_cache_root", "output_root")},
        "initializer_file_inventory_sha256": digest_json(card["initializer"]["files"]),
        "hf_repo_id": card["storage"]["repo_id"],
        "hf_path_prefix": card["storage"]["path_prefix"],
        "checkpoint_family": card["storage"]["checkpoint_family"],
        "parent_card_sha256": parent["card_sha256"],
        "parent_schedule_sha256": parent["schedule_sha256"],
        "cohort_sha256": card["cohort"]["sha256"],
        "schedule_sha256": card["schedule"]["sha256"],
        "stats_binding_sha256": digest_json(card["r2_data_binding"]["stats_files"]),
        "resolved_config_sha256": digest_json(card["resolved_config"]),
        "loss": card["loss_intervention"],
    }
    payload["binding_id"] = digest_json(payload)
    return payload


def _source_schedule(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != MAX_STEPS:
        raise R2Error(f"parent 10K schedule is incomplete: {path}")
    return rows


def prepare(args: argparse.Namespace) -> None:
    parent_path = args.parent_card.resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("experiment") != R0_EXPERIMENT or parent.get("stage", {}).get("max_steps") != MAX_STEPS:
        raise R2Error("parent card is not the sealed PI0244 R0 fixed16/10K run")
    if parent.get("initializer", {}).get("path") != parent["paths"]["initializer"]:
        raise R2Error("R0 initializer path binding is inconsistent")
    if Path(parent["paths"]["initializer"]).resolve() != args.initializer.resolve():
        raise R2Error("R2 must use the exact original PI0226 initializer path")
    if args.output_root.exists():
        raise R2Error(f"refusing to reuse R2 output root: {args.output_root}")

    r0_root = args.r0_compute_root.resolve()
    r0_closure_path = r0_root / "pi0244_stage_closure.json"
    r0_training_path = r0_root / "pi0244_training_manifest.json"
    if sha256_file(r0_closure_path) != R0_CLOSURE_SHA256 or sha256_file(r0_training_path) != R0_TRAINING_MANIFEST_SHA256:
        raise R2Error("R0 closure/training-manifest SHA differs from the canonical valid-failure evidence")
    r0_closure = json.loads(r0_closure_path.read_text(encoding="utf-8"))
    r0_training = json.loads(r0_training_path.read_text(encoding="utf-8"))
    r0_update_audit_path = r0_root / "training" / "pi0241_update_audit.json"
    if sha256_file(r0_update_audit_path) != R0_UPDATE_AUDIT_SHA256:
        raise R2Error("R0 FP32/update evidence does not match the canonical source audit")
    if r0_closure.get("status") != "SCIENCE_FAIL_M16_10K_BUDGET" or r0_closure.get("screen_cells") != 80 or r0_closure.get("selected_checkpoint") != "NONE":
        raise R2Error("R0 scientific failure is not the canonical 80-cell final-only result")
    if r0_training.get("status") != "PASS_R0_TRAINING_CLOSED" or r0_training.get("canonical_optimizer_steps") != MAX_STEPS:
        raise R2Error("R0 training did not close at the authorized 10K budget")

    receipt_rows = {}
    for step in CHECKPOINTS:
        path = r0_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        inventory = receipt.get("source_record", {}).get("inventory", {})
        if receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("family") != R0_EXPERIMENT or int(receipt.get("checkpoint_step", -1)) != step or int(receipt.get("max_steps", -1)) != MAX_STEPS:
            raise R2Error(f"R0 checkpoint-{step} receipt is not a valid full Trainer-state receipt")
        if int(inventory.get("file_count", -1)) != 18 or int(inventory.get("total_bytes", 0)) <= 0:
            raise R2Error(f"R0 checkpoint-{step} receipt inventory is incomplete")
        receipt_rows[str(step)] = {
            "receipt_path": str(path),
            "receipt_sha256": sha256_file(path),
            "durable_path": str(receipt["durable_path"]),
            "bytes": int(inventory["total_bytes"]),
            "tree_sha256": inventory["tree_sha256"],
            "file_count": int(inventory["file_count"]),
        }

    base = load_base(args.code_root.resolve())
    p = parent["paths"]
    base_args = SimpleNamespace(
        stage="a",
        code_root=args.code_root.resolve(),
        groot_source=Path(p["groot_source"]),
        stage10_root=Path(p["stage10_root"]),
        train_dataset=Path(p["train_dataset"]),
        p237_root=Path(p["p237_root"]),
        modality_config=Path(p["modality_config"]),
        processor_base_model=Path(p["processor_base_model"]),
        initializer=args.initializer.resolve(),
        cosmos_snapshot=Path(p["cosmos_snapshot"]),
        cosmos_cache_root=Path(p["cosmos_cache_root"]),
        output_root=args.output_root.resolve(),
        durable_host=None,
        durable_root=None,
        hf_repo_id=HF_REPO_ID,
        hf_token_file=Path(parent["storage"]["token_file"]),
        hf_path_prefix=HF_PREFIX,
    )
    base.make_card(base_args)
    card_path = args.output_root / "pi0243_run_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    parent_rows = _source_schedule(Path(parent["schedule"]["path"]))
    current_rows = _source_schedule(Path(card["schedule"]["path"]))
    if current_rows != parent_rows:
        raise R2Error("new deterministic schedule differs from the full original R0 10K schedule")
    if card["fixed16_source_ids"] != parent["fixed16_source_ids"]:
        raise R2Error("R2 fixed16 cohort/order differs from the R0 sealed cohort")
    if card["initializer"]["files"] != parent["initializer"]["files"]:
        raise R2Error("R2 initializer inventory differs from the original PI0226 checkpoint")
    if card["resolved_config"] != parent["resolved_config"]:
        raise R2Error("R2 resolved trainer config differs from R0 beyond the sole loss intervention")
    if card["processor"] != parent["processor"] or card.get("target_xyz_in_model_input") is not False:
        raise R2Error("R2 processor/stats identity or target-input exclusion differs from R0")
    for key in ("train_dataset", "p237_root", "stage10_root", "groot_source", "modality_config", "processor_base_model", "initializer", "cosmos_snapshot", "cosmos_cache_root"):
        if card["paths"][key] != parent["paths"][key]:
            raise R2Error(f"R2 source path drift: {key}")

    groot_head = subprocess.check_output(["git", "-C", p["groot_source"], "rev-parse", "HEAD"], text=True).strip()
    if groot_head != base.GROOT_SHA:
        raise R2Error(f"GR00T source commit mismatch: {groot_head} != {base.GROOT_SHA}")
    groot_dirty = subprocess.check_output(["git", "-C", p["groot_source"], "status", "--porcelain"], text=True)
    if groot_dirty.strip():
        raise R2Error("pinned GR00T source has local modifications")

    card["experiment"] = EXPERIMENT
    card["protocol_commit"] = PROTOCOL_COMMIT
    card["audited_parent"] = AUDITED_PARENT
    card["stage_role"] = "PI0244_R2_XYZ4_FIXED16_10K"
    card["storage"]["checkpoint_family"] = FAMILY
    card["storage"]["path_prefix"] = HF_PREFIX
    card["storage"]["layout"] = "retain_all_local_hf_durable"
    card["storage"]["capacity_binding"] = {
        "source_checkpoint_receipts": receipt_rows,
        "full_checkpoints": 4,
        "atomic_write_peak_extra_checkpoints": 1,
        "screen_artifact_bytes_from_r0": sum(p.stat().st_size for p in (r0_root / "screen").rglob("*") if p.is_file()),
        "headroom_fraction": 0.20,
        "reservation_bytes": None,
        "proof_path": str(args.output_root / "pi0244x_xyz4_capacity_admission.json"),
        "proof_sha256": None,
        "status": "PLANNED",
    }
    card["loss_intervention"] = {
        "schema": "PI0244_XYZ4_NATIVE_LOSS_V1",
        "formula": "sum(weight*already_masked_action_loss)/(sum(weight*action_mask)+1e-6)",
        "weight": {"action_slots_0_to_2": 4.0, "other_slots": 1.0},
        "shape": [None, HORIZON, ACTION_WIDTH],
        "native_forward_calls_per_loss": 1,
        "new_noise_or_time_sample": False,
        "detach_or_reconstructed_loss": False,
        "additional_mask": False,
    }
    stats_files = {}
    for name in ("modality.json", "relative_stats.json", "stats.json"):
        path = Path(card["paths"]["train_dataset"]) / "meta" / name
        stats_files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    card["r2_data_binding"] = {
        "schema": "PI0244_R2_FIXED_TRAIN_DATA_BINDING_V1",
        "parent_processor_identity_equal": True,
        "processor": card["processor"],
        "cohort_sha256": card["cohort"]["sha256"],
        "schedule_sha256": card["schedule"]["sha256"],
        "stats_files": stats_files,
        "fixed16_source_ids_sha256": digest_json(card["fixed16_source_ids"]),
        "target_xyz_in_model_input": False,
    }
    code_root = args.code_root.resolve()
    r2_script = Path(__file__).resolve()
    card["r0_parent"] = {
        "compute_root": str(r0_root),
        "card_path": str(parent_path),
        "card_sha256": sha256_file(parent_path),
        "schedule_path": parent["schedule"]["path"],
        "schedule_sha256": parent["schedule"]["sha256"],
        "closure_path": str(r0_closure_path),
        "closure_sha256": R0_CLOSURE_SHA256,
        "training_manifest_path": str(r0_training_path),
        "training_manifest_sha256": R0_TRAINING_MANIFEST_SHA256,
        "update_dtype_audit_path": str(r0_update_audit_path),
        "update_dtype_audit_sha256": R0_UPDATE_AUDIT_SHA256,
        "r1_p32_decision": "NOT_APPLICABLE",
        "r1_evidence_boundary": "471 trainable tensors FP32 plus checkpoint-2500 FloatStorage; not a claim that every forward operation is FP32",
        "card_hf_repo_id": parent["storage"]["repo_id"],
        "receipt_path_repo_ids": sorted({str(row["durable_path"]).split("/")[0] + "/" + str(row["durable_path"]).split("/")[1] for row in receipt_rows.values()}),
        "screen_status": r0_closure["status"],
        "selected_checkpoint": "NONE",
    }
    card["r2_implementation"] = {
        "runner_path": str(r2_script),
        "runner_sha256": sha256_file(r2_script),
        "pi0243_path": str(code_root / "scripts" / "pi0243_staged_learning_to_sim.py"),
        "pi0241_path": str(code_root / "scripts" / "pi0241_fixed16_learning.py"),
        "pi0228_path": str(code_root / "scripts" / "pi0228_h200_large_adapt_v1.py"),
        "persistence_path": str(code_root / "scripts" / "pi0243_checkpoint_persistence.py"),
        "storage_preflight_path": str(code_root / "scripts" / "pi0243_storage_preflight.py"),
    }
    for key in ("pi0243_path", "pi0241_path", "pi0228_path", "persistence_path", "storage_preflight_path"):
        card["r2_implementation"][key.replace("_path", "_sha256")] = sha256_file(Path(card["r2_implementation"][key]))
    card["paths"]["groot_source_commit"] = groot_head
    card["r2_resolved_diff"] = {
        "schema": "PI0244_R2_RESOLVED_DIFF_V1",
        "status": "PASS",
        "parent_card_sha256": card["r0_parent"]["card_sha256"],
        "resolved_config_equal_to_r0": True,
        "schedule_equal_all_10000_rows": True,
        "cohort_equal_all_16_rows": True,
        "initializer_equal_file_inventory": True,
        "r2_hf_storage_resolution": {
            "parent_card_repo_id": parent["storage"]["repo_id"],
            "r2_repo_id": HF_REPO_ID,
            "r2_prefix": HF_PREFIX,
            "reason": "parent card repo ID returned 404; account-owned public repo with an unused R2 prefix was live-verified",
        },
        "only_scientific_change": card["loss_intervention"],
        "old_p1_exact_replay": "HISTORICAL_CT_REPLAY_UNAVAILABLE_FROM_SEALED_ARTIFACTS",
        "r1_p32": {"status": "NOT_APPLICABLE", "evidence_sha256": R0_UPDATE_AUDIT_SHA256},
        "new_ct_packet": "NOT_RUN_NONCRITICAL_DIAGNOSTIC",
    }
    diff_path = args.output_root / "pi0244x_xyz4_resolved_diff.json"
    write_json(diff_path, card["r2_resolved_diff"])
    card["r2_resolved_diff_path"] = str(diff_path)
    card["r2_resolved_diff_sha256"] = sha256_file(diff_path)
    card["selected_checkpoint"] = "NONE"
    card["forbidden_counts"].update(
        {
            "dev_e_reads": 0,
            "final_f_reads": 0,
            "wbc_steps": 0,
            "sim_steps": 0,
            "dagger_steps": 0,
        }
    )
    card["r2_execution_binding"] = _execution_binding(card, code_root)
    write_json(card_path, card)
    write_json(card_path.with_name("pi0243_run_card.seal.json"), {"schema": "PI0243_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(card_path)})
    print(json.dumps({"status": "PREPARED_CAPACITY_PENDING", "run_card": str(card_path), "run_card_sha256": sha256_file(card_path), "schedule_sha256": card["schedule"]["sha256"], "execution_binding_id": card["r2_execution_binding"]["binding_id"]}, sort_keys=True))


def verify_card(path: Path, *, require_capacity: bool = True) -> tuple[Any, dict[str, Any]]:
    card = json.loads(path.read_text(encoding="utf-8"))
    code_root = Path(card["paths"]["code_root"])
    base = load_base(code_root)
    if card.get("experiment") != EXPERIMENT or card.get("protocol_commit") != PROTOCOL_COMMIT or card.get("stage") != stage_spec("a"):
        raise R2Error("R2 run-card identity mismatch")
    if card.get("storage", {}).get("checkpoint_family") != FAMILY or card["storage"].get("repo_id") != HF_REPO_ID or card["storage"].get("path_prefix") != HF_PREFIX:
        raise R2Error("R2 HF family/prefix mismatch")
    if sha256_file(Path(card["r2_implementation"]["runner_path"])) != card["r2_implementation"]["runner_sha256"]:
        raise R2Error("R2 runner changed after run-card sealing")
    if Path(card["r2_implementation"]["runner_path"]).resolve() != Path(__file__).resolve():
        raise R2Error("run card does not bind the active R2 runner")
    for key in ("pi0243_path", "pi0241_path", "pi0228_path", "persistence_path", "storage_preflight_path"):
        path_key, sha_key = key, key.replace("_path", "_sha256")
        if sha256_file(Path(card["r2_implementation"][path_key])) != card["r2_implementation"][sha_key]:
            raise R2Error(f"mature implementation changed after R2 seal: {key}")
    if card.get("r2_execution_binding") != _execution_binding(card, code_root):
        raise R2Error("immutable R2 execution binding mismatch")
    if card.get("r2_data_binding", {}).get("target_xyz_in_model_input") is not False:
        raise R2Error("R2 data binding permits target XYZ in model inputs")
    for row in card["r2_data_binding"]["stats_files"].values():
        source = Path(row["path"])
        if not source.is_file() or source.stat().st_size != int(row["bytes"]) or sha256_file(source) != row["sha256"]:
            raise R2Error(f"R2 TRAIN statistics identity changed: {source.name}")
    diff_path = Path(card["r2_resolved_diff_path"])
    if sha256_file(diff_path) != card["r2_resolved_diff_sha256"]:
        raise R2Error("R2 resolved-diff hash mismatch")
    seal_path = path.with_name("pi0243_run_card.seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("run_card_sha256") != sha256_file(path):
        raise R2Error("R2 run-card seal mismatch")
    base_card = base.verify_card(path)
    cohort_path = base._mapped_stage_path(card, card["cohort"]["path"], "PI0243_METADATA_MIRROR_ROOT")
    cohort_rows = base.read_json(cohort_path)["rows"]
    if [str(row["source_episode_id"]) for row in cohort_rows] != card["fixed16_source_ids"]:
        raise R2Error("R2 cohort order differs from its sealed fixed16 source list")
    parent = card["r0_parent"]
    parent_card_path = Path(parent["card_path"])
    parent_schedule_path = Path(parent["schedule_path"])
    r0_root = Path(parent["compute_root"])
    if sha256_file(parent_card_path) != parent["card_sha256"] or sha256_file(parent_schedule_path) != parent["schedule_sha256"]:
        raise R2Error("R0 parent card or schedule changed after R2 preparation")
    if sha256_file(r0_root / "pi0244_stage_closure.json") != R0_CLOSURE_SHA256 or sha256_file(r0_root / "pi0244_training_manifest.json") != R0_TRAINING_MANIFEST_SHA256:
        raise R2Error("canonical R0 failure/training evidence changed")
    if sha256_file(r0_root / "training" / "pi0241_update_audit.json") != R0_UPDATE_AUDIT_SHA256:
        raise R2Error("canonical R0 FP32/update evidence changed")
    parent_rows = _source_schedule(parent_schedule_path)
    current_rows = _source_schedule(Path(base._mapped_stage_path(card, card["schedule"]["path"], "PI0243_METADATA_MIRROR_ROOT")))
    if current_rows != parent_rows:
        raise R2Error("R2 exposure schedule no longer matches all 10000 sealed R0 rows")
    for row in card["storage"]["capacity_binding"]["source_checkpoint_receipts"].values():
        receipt_path = Path(row["receipt_path"])
        if not receipt_path.is_file() or sha256_file(receipt_path) != row["receipt_sha256"]:
            raise R2Error("pinned R0 storage-capacity receipt changed")
    capacity = card["storage"]["capacity_binding"]
    if require_capacity:
        proof = Path(capacity["proof_path"])
        if capacity.get("status") != "PASS" or not proof.is_file() or sha256_file(proof) != capacity.get("proof_sha256"):
            raise R2Error("R2 non-sparse storage capacity admission is incomplete")
        proof_data = json.loads(proof.read_text(encoding="utf-8"))
        if proof_data.get("status") != "PASS_NONSPARSE_FSYNC_RESERVATION" or int(proof_data.get("reservation_bytes", -1)) != int(capacity.get("reservation_bytes", -2)) or proof_data.get("prepared_run_card_sha256") != card.get("capacity_preparation_card_sha256"):
            raise R2Error("R2 storage proof does not match the final sealed capacity binding")
    return base, card


def admit_capacity(run_card: Path) -> None:
    base, card = verify_card(run_card, require_capacity=False)
    binding = card["storage"]["capacity_binding"]
    proof_path = Path(binding["proof_path"])
    if binding.get("status") == "PASS":
        if not proof_path.is_file() or sha256_file(proof_path) != binding.get("proof_sha256"):
            raise R2Error("sealed R2 capacity proof is missing or changed")
        print(json.dumps({"status": "PASS_CAPACITY_ALREADY_SEALED", "proof_sha256": binding["proof_sha256"]}))
        return
    if proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        expected_prepared_sha = card.get("capacity_preparation_card_sha256") or sha256_file(run_card)
        if proof.get("prepared_run_card_sha256") != expected_prepared_sha:
            raise R2Error("unbound prior capacity proof exists; preserve it and inspect")
    else:
        r0_receipts = binding["source_checkpoint_receipts"]
        maximum = max(int(row["bytes"]) for row in r0_receipts.values())
        raw_bound = 5 * maximum + int(binding["screen_artifact_bytes_from_r0"])
        reservation = math.ceil(raw_bound * (1.0 + float(binding["headroom_fraction"])))
        helper = base.load_module(Path(card["r2_implementation"]["storage_preflight_path"]), "pi0244x_storage_preflight")
        root = Path(card["paths"]["output_root"])
        before = helper.snapshot(root)
        measured = helper.reserve_local(root / ".pi0244x_capacity_reservation", reservation)
        after = helper.snapshot(root)
        proof = {
            "schema": "PI0244X_STORAGE_CAPACITY_PROOF_V1",
            "status": "PASS_NONSPARSE_FSYNC_RESERVATION",
            "prepared_run_card_sha256": sha256_file(run_card),
            "capacity_formula": "ceil((5*largest_R0_full_checkpoint_bytes + R0_80cell_screen_artifact_bytes)*1.20)",
            "largest_full_checkpoint_bytes": maximum,
            "four_full_checkpoints": 4,
            "one_atomic_write_checkpoint": 1,
            "screen_artifact_bytes_from_r0": int(binding["screen_artifact_bytes_from_r0"]),
            "reservation_bytes": reservation,
            "reservation": measured,
            "quota_command_status": "UNKNOWN_NOT_EXPOSED",
            "statvfs_before": before["statvfs"],
            "statvfs_after": after["statvfs"],
            "same_fileset_as_output": True,
            "proof_method": "posix_fallocate + fsync + allocated st_blocks verification; temporary reservation files removed by the mature helper",
        }
        write_json(proof_path, proof, once=True)
    card["capacity_preparation_card_sha256"] = proof["prepared_run_card_sha256"]
    binding["reservation_bytes"] = int(proof["reservation_bytes"])
    binding["proof_sha256"] = sha256_file(proof_path)
    binding["status"] = "PASS"
    write_json(run_card, card)
    write_json(run_card.with_name("pi0243_run_card.seal.json"), {"schema": "PI0243_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(run_card)})
    verify_card(run_card, require_capacity=True)
    print(json.dumps({"status": "PASS_R2_CAPACITY_ADMISSION", "reservation_bytes": proof["reservation_bytes"], "proof_sha256": sha256_file(proof_path), "run_card_sha256": sha256_file(run_card)}, sort_keys=True))


def _safe_job_id() -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in os.environ.get("PBS_JOBID", "manual"))


def runtime_preflight(groot_source: Path) -> None:
    import inspect

    actual = subprocess.check_output(["git", "-C", str(groot_source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(groot_source), "status", "--porcelain"], text=True)
    if actual != "51d4c89f72fda44cbf77285c6a8114b52676b8a1" or dirty.strip():
        raise R2Error(f"exact GR00T source preflight failed: commit={actual}, dirty={bool(dirty.strip())}")
    source = str(groot_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    head_module = importlib.import_module("gr00t.model.gr00t_n1d7.gr00t_n1d7")
    experiment = importlib.import_module("gr00t.experiment.experiment")
    from transformers import TrainerCallback

    native = " ".join(inspect.getsource(head_module.Gr00tN1d7ActionHead.forward).split())
    required = "action_loss = F.mse_loss(pred_actions, velocity, reduction=\"none\") * action_mask"
    if " ".join(required.split()) not in native or "loss = action_loss.sum() / (action_mask.sum() + 1e-6)" not in native:
        raise R2Error("pinned official action-head loss contract differs from the reviewed native implementation")

    action_cls = head_module.Gr00tN1d7ActionHead
    original_forward = action_cls.forward
    had_marker = hasattr(action_cls, "_pi0244_xyz4_patched")
    marker = getattr(action_cls, "_pi0244_xyz4_patched", None)
    trainer_cls = experiment.Gr00tTrainer
    original_init = trainer_cls.__init__
    had_trainer_marker = hasattr(trainer_cls, "_pi0244_xyz4_moments_patched")
    trainer_marker = getattr(trainer_cls, "_pi0244_xyz4_moments_patched", None)
    try:
        state = patch_action_head(action_cls, Path(os.devnull).with_name("pi0244x_no_write.json"), sha256_file(Path(head_module.__file__)))
        install_moment_callback(trainer_cls, Path(os.devnull).with_name("pi0244x_no_write_moments.json"))
        if action_cls.forward is original_forward or trainer_cls.__init__ is original_init or state["forward_calls"] != 0:
            raise R2Error("runtime hook did not bind without running a model")
    finally:
        action_cls.forward = original_forward
        trainer_cls.__init__ = original_init
        if had_marker:
            action_cls._pi0244_xyz4_patched = marker
        elif hasattr(action_cls, "_pi0244_xyz4_patched"):
            del action_cls._pi0244_xyz4_patched
        if had_trainer_marker:
            trainer_cls._pi0244_xyz4_moments_patched = trainer_marker
        elif hasattr(trainer_cls, "_pi0244_xyz4_moments_patched"):
            del trainer_cls._pi0244_xyz4_moments_patched
    print(json.dumps({"status": "PASS_EXACT_STAGE10_NATIVE_TRAINER_HOOK_PREFLIGHT", "groot_commit": actual, "action_head_source_sha256": sha256_file(Path(head_module.__file__)), "native_loss_keys": ["loss", "action_loss", "action_mask"], "trainer_class": f"{trainer_cls.__module__}.{trainer_cls.__qualname__}", "weights_loaded": 0, "model_forwards": 0, "optimizer_steps": 0, "trainer_callback_type": TrainerCallback.__name__}))


def _install_training_interventions(base: Any, code_root: Path, output: Path) -> None:
    original_loader = base.pi0228_module

    def loader(root: Path) -> Any:
        module = original_loader(root)
        original_run = module.run_official

        def run_official(config: Any, groot_source: Path) -> Any:
            source = str(groot_source)
            if source not in sys.path:
                sys.path.insert(0, source)
            head_module = importlib.import_module("gr00t.model.gr00t_n1d7.gr00t_n1d7")
            experiment = importlib.import_module("gr00t.experiment.experiment")
            job = _safe_job_id()
            training_root = Path(os.environ.get("PI0244_R2_TRAINING_ROOT", ""))
            if not training_root.is_dir() or training_root != output.resolve():
                raise R2Error("R2 training-output binding does not match the sealed output path")
            head_file = Path(head_module.__file__).resolve()
            loss_state = patch_action_head(
                head_module.Gr00tN1d7ActionHead,
                training_root / f"pi0244_r2_first_forward_{job}.json",
                sha256_file(head_file),
            )
            install_moment_callback(
                experiment.Gr00tTrainer,
                training_root / f"pi0244_r2_first_update_moments_{job}.json",
            )
            try:
                result = original_run(config, groot_source)
            except BaseException:
                write_json(
                    training_root / f"pi0244_r2_loss_runtime_{job}.json",
                    {"schema": "PI0244_R2_XYZ4_LOSS_RUNTIME_V1", "status": "TRAINER_EXCEPTION", "forward_calls": loss_state["forward_calls"], "first_forward": loss_state["first_forward"]},
                    once=True,
                )
                raise
            write_json(
                training_root / f"pi0244_r2_loss_runtime_{job}.json",
                {"schema": "PI0244_R2_XYZ4_LOSS_RUNTIME_V1", "status": "TRAINER_RETURNED", "forward_calls": loss_state["forward_calls"], "first_forward": loss_state["first_forward"]},
                once=True,
            )
            if loss_state["forward_calls"] <= 0 or loss_state["first_forward"] is None:
                raise R2Error("official Trainer returned without consuming the XYZ4 loss hook")
            if not (training_root / f"pi0244_r2_first_update_moments_{job}.json").is_file():
                raise R2Error("first actual update did not record named optimizer moments")
            return result

        module.run_official = run_official
        return module

    base.pi0228_module = loader


def train(run_card: Path) -> None:
    base, card = verify_card(run_card)
    root = Path(card["paths"]["output_root"])
    output = root / "training"
    if output.exists():
        raise R2Error("R2 training output already exists; do not restart from the initializer")
    if not os.environ.get("PBS_JOBID"):
        raise R2Error("R2 training must run as the unique scheduler-owned training job")
    expected_repo = card["storage"]["repo_id"]
    expected_prefix = card["storage"]["path_prefix"]
    if os.environ.get("PI0243_HF_REPO_ID") != expected_repo or os.environ.get("PI0243_HF_PATH_PREFIX") != expected_prefix:
        raise R2Error("HF publisher environment does not match the sealed R2 family")
    os.environ["PI0244_R2_TRAINING_ROOT"] = str(output.resolve())
    _install_training_interventions(base, Path(card["paths"]["code_root"]), output.resolve())
    base.train_stage(SimpleNamespace(run_card=run_card, stage="a", authorization=AUTHORIZATION))
    close_training(run_card)


def close_training(run_card: Path) -> None:
    base, card = verify_card(run_card)
    root = Path(card["paths"]["output_root"])
    output = root / "training"
    manifest = json.loads((root / "pi0243_training_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "pi0241_update_audit.json").read_text(encoding="utf-8"))
    schedule = _source_schedule(Path(card["schedule"]["path"]))
    consumption = [json.loads(line) for line in (output / "pi0241_sample_consumption.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")
    if len(consumption) != MAX_STEPS or any(any(actual.get(key) != expected.get(key) for key in fields) for actual, expected in zip(consumption, schedule)):
        raise R2Error("actual R2 sample consumption differs from the sealed 10K schedule")
    if manifest.get("status") != "PASS_UPDATE_INTEGRITY" or int(manifest.get("actual_sample_consumption", -1)) != MAX_STEPS:
        raise R2Error("mature trainer did not close with 10K valid sample consumption/update integrity")
    if audit.get("status") != "COMPLETE" or int(audit.get("global_step", -1)) != MAX_STEPS:
        raise R2Error("mature update audit did not reach optimizer step 10000")
    job = os.environ.get("PBS_JOBID")
    matches = sorted(output.glob("pi0244_r2_first_update_moments_*.json"))
    forwards = sorted(output.glob("pi0244_r2_loss_runtime_*.json"))
    first = sorted(output.glob("pi0244_r2_first_forward_*.json"))
    if len(matches) != 1 or len(forwards) != 1 or len(first) != 1:
        raise R2Error("R2 first-update/loss-hook evidence is not a unique complete training attempt")
    moment = json.loads(matches[0].read_text(encoding="utf-8"))
    runtime = json.loads(forwards[0].read_text(encoding="utf-8"))
    first_forward = json.loads(first[0].read_text(encoding="utf-8"))
    if moment.get("global_step") != 1 or runtime.get("status") != "TRAINER_RETURNED" or int(runtime.get("forward_calls", 0)) <= 0:
        raise R2Error("R2 first-update or weighted-loss consumption evidence is invalid")
    moment_rows = moment.get("trainable_moment_records", [])
    moment_names = [row.get("name") for row in moment_rows]
    if (
        moment.get("trainable_parameter_count") != len(moment_rows)
        or len(moment_names) != len(set(moment_names))
        or not any({"exp_avg", "exp_avg_sq"}.issubset(row.get("moments", {})) for row in moment_rows)
        or any(
            row.get("parameter_dtype") is None
            or (
                row.get("moments")
                and (
                    not {"exp_avg", "exp_avg_sq"}.issubset(row["moments"])
                    or row["moments"].get("step", {}).get("value") != 1.0
                )
            )
            for row in moment_rows
        )
    ):
        raise R2Error("first real update did not capture complete named parameter/moment dtypes")
    if first_forward.get("status") != "RECORDED" or first_forward.get("native_component_loss_delta", 1.0) > 1e-5:
        raise R2Error("R2 actual first forward did not preserve native loss-component identity")
    checkpoints = {}
    for step in CHECKPOINTS:
        path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("family") != FAMILY or int(receipt.get("checkpoint_step", -1)) != step or int(receipt.get("max_steps", -1)) != MAX_STEPS:
            raise R2Error(f"R2 checkpoint-{step} is not durably verified in the new family")
        checkpoints[str(step)] = {"receipt_sha256": sha256_file(path), "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"], "bytes": receipt["source_record"]["inventory"]["total_bytes"], "hf_path": receipt["durable_path"]}
    closure = {
        "schema": "PI0244_R2_XYZ4_TRAINING_CLOSURE_V1",
        "status": "PASS_R2_TRAINING_CLOSED",
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "run_card_sha256": sha256_file(run_card),
        "execution_binding_id": card["r2_execution_binding"]["binding_id"],
        "schedule_sha256": card["schedule"]["sha256"],
        "actual_sample_consumption": len(consumption),
        "actual_consumption_matches_all_schedule_fields": True,
        "optimizer_steps": int(audit["global_step"]),
        "update_integrity": manifest.get("update_integrity"),
        "r2_loss_runtime_sha256": sha256_file(forwards[0]),
        "first_forward_sha256": sha256_file(first[0]),
        "first_update_named_moments_sha256": sha256_file(matches[0]),
        "checkpoint_receipts": checkpoints,
        "selected_checkpoint": "NONE",
    }
    write_json(root / "pi0244x_xyz4_training_closure.json", closure, once=True)
    print(json.dumps(closure, sort_keys=True))


def _record_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_screen_records(base: Any, root: Path, step: int, expected_ids: set[str]) -> list[dict[str, Any]]:
    records_path = root / "screen" / f"step-{step:05d}" / "records.jsonl"
    rows = _record_rows(records_path)
    ids = [str(row.get("source_episode_id")) for row in rows]
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids):
        raise R2Error(f"screen step {step} contains duplicate or foreign source rows")
    for row in rows:
        if int(row.get("screen_step", -1)) != step:
            raise R2Error(f"screen step binding mismatch at {step}")
        tensor = Path(row["tensor_path"]).resolve()
        if root.resolve() not in tensor.parents or not tensor.is_file() or sha256_file(tensor) != row.get("tensor_sha256"):
            raise R2Error(f"screen tensor failed its path/hash check at step {step}")
        with __import__("numpy").load(tensor, allow_pickle=False) as arrays:
            if base.sha256_array(arrays["raw"]) != row.get("raw_sha256") or base.sha256_array(arrays["decoded"]) != row.get("decoded_sha256"):
                raise R2Error(f"screen raw/decoded arrays differ from record at step {step}")
    return rows


def _complete_screen(run_card: Path) -> dict[str, Any]:
    base, card = verify_card(run_card)
    root = Path(card["paths"]["output_root"])
    train_closure = json.loads((root / "pi0244x_xyz4_training_closure.json").read_text(encoding="utf-8"))
    if train_closure.get("status") != "PASS_R2_TRAINING_CLOSED":
        raise R2Error("screen is blocked until R2 training closure is valid")
    expected_ids = set(card["fixed16_source_ids"])
    if len(expected_ids) != 16:
        raise R2Error("R2 fixed16 source set is not 16 unique rows")
    profiles = {}
    bindings_root = root / "r2_screen_bindings"
    for step in SCREENS:
        rows = _verify_screen_records(base, root, step, expected_ids)
        if len(rows) != 16 or set(row["source_episode_id"] for row in rows) != expected_ids:
            raise R2Error(f"screen profile {step} is incomplete")
        records_path = root / "screen" / f"step-{step:05d}" / "records.jsonl"
        receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json" if step else None
        model = (
            {"kind": "PI0226_initializer", "files": card["initializer"]["files"]}
            if step == 0
            else {
                "kind": FAMILY,
                "step": step,
                "tree_sha256": json.loads(receipt_path.read_text(encoding="utf-8"))["source_record"]["inventory"]["tree_sha256"],
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
        binding = {
            "schema": "PI0244_R2_SCREEN_PROFILE_BINDING_V1",
            "status": "COMPLETE",
            "experiment": EXPERIMENT,
            "run_card_sha256": sha256_file(run_card),
            "execution_binding_id": card["r2_execution_binding"]["binding_id"],
            "step": step,
            "model_identity": model,
            "record_count": len(rows),
            "records_sha256": sha256_file(records_path),
            "raw_sha256s": {str(row["source_episode_id"]): row["raw_sha256"] for row in rows},
            "decoded_sha256s": {str(row["source_episode_id"]): row["decoded_sha256"] for row in rows},
            "tensor_sha256s": {str(row["source_episode_id"]): row["tensor_sha256"] for row in rows},
        }
        write_json(bindings_root / f"step-{step:05d}.json", binding, once=True)
        profiles[str(step)] = binding
    final_rows = _record_rows(root / "screen" / f"step-{MAX_STEPS:05d}" / "records.jsonl")
    import numpy as np

    reject_count = sum(str(row.get("gate_decision")) == "REJECT" for row in final_rows)
    xyz = np.asarray([float(row["xyz_rmse_m"]) for row in final_rows], dtype=np.float64)
    so3 = np.asarray([float(row["so3_rmse_rad"]) for row in final_rows], dtype=np.float64)
    median_xyz = float(np.percentile(xyz, 50, method="linear"))
    p95_xyz = float(np.percentile(xyz, 95, method="linear"))
    median_so3 = float(np.percentile(so3, 50, method="linear"))
    criteria = {
        "gate_rejects_zero": reject_count == 0,
        "median_xyz_rmse_le_0p020": median_xyz <= 0.020,
        "p95_xyz_rmse_le_0p035": p95_xyz <= 0.035,
        "median_so3_rmse_le_0p050": median_so3 <= 0.050,
    }
    passed = all(criteria.values())
    closure = {
        "schema": "PI0244_R2_XYZ4_SCREEN_CLOSURE_V1",
        "status": "PASS_R2_XYZ4" if passed else "SCIENCE_FAIL_R2_XYZ4_FINAL_GATE",
        "terminal_token": None if passed else "PREAPPROVED_RECIPE_SET_EXHAUSTED — STOP_FOR_PI_NEW_HYPOTHESIS",
        "experiment": EXPERIMENT,
        "run_card_sha256": sha256_file(run_card),
        "profile_count": len(SCREENS),
        "screen_cells": sum(binding["record_count"] for binding in profiles.values()),
        "final_count": len(final_rows),
        "final_10000": {"gate_rejects": reject_count, "median_xyz_rmse_m": median_xyz, "p95_xyz_rmse_m": p95_xyz, "median_so3_rmse_rad": median_so3},
        "criteria": criteria,
        "profile_bindings": {key: sha256_file(bindings_root / f"step-{int(key):05d}.json") for key in profiles},
        "selected_checkpoint": "NONE",
    }
    path = root / "pi0244x_xyz4_screen_closure.json"
    write_json(path, closure, once=True)
    return closure


def screen(run_card: Path, device: str) -> None:
    base, card = verify_card(run_card)
    root = Path(card["paths"]["output_root"])
    if not (root / "pi0244x_xyz4_training_closure.json").is_file():
        raise R2Error("screen cannot start without the R2 10K training closure")
    expected_ids = set(card["fixed16_source_ids"])
    bindings_root = root / "r2_screen_expected"
    for step in SCREENS:
        if step == 0:
            model = {"kind": "PI0226_initializer", "files": card["initializer"]["files"]}
        else:
            receipt_path = root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            model = {"kind": FAMILY, "step": step, "tree_sha256": receipt["source_record"]["inventory"]["tree_sha256"], "receipt_sha256": sha256_file(receipt_path)}
        expected = {
            "schema": "PI0244_R2_SCREEN_EXPECTED_V1",
            "experiment": EXPERIMENT,
            "run_card_sha256": sha256_file(run_card),
            "execution_binding_id": card["r2_execution_binding"]["binding_id"],
            "step": step,
            "model_identity": model,
            "source_episode_ids": card["fixed16_source_ids"],
            "record_count": 16,
        }
        write_json(bindings_root / f"step-{step:05d}.json", expected, once=True)
        _verify_screen_records(base, root, step, expected_ids)
    closure_path = root / "pi0244x_xyz4_screen_closure.json"
    if closure_path.is_file():
        print(closure_path.read_text(encoding="utf-8"))
        return
    base.screen_stage(SimpleNamespace(run_card=run_card, stage="a", device=device))
    closure = _complete_screen(run_card)
    print(json.dumps(closure, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("self-test", "runtime-preflight", "prepare", "capacity", "verify", "train", "finalize-training", "screen", "close-screen"), required=True)
    p.add_argument("--run-card", type=Path)
    p.add_argument("--parent-card", type=Path)
    p.add_argument("--r0-compute-root", type=Path)
    p.add_argument("--code-root", type=Path)
    p.add_argument("--initializer", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--groot-source", type=Path)
    p.add_argument("--device", default="cuda:0")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.phase == "self-test":
        self_test()
        return 0
    if args.phase == "runtime-preflight":
        if args.groot_source is None:
            raise R2Error("runtime-preflight requires --groot-source")
        runtime_preflight(args.groot_source)
        return 0
    if args.phase == "prepare":
        if any(value is None for value in (args.parent_card, args.r0_compute_root, args.code_root, args.initializer, args.output_root)):
            raise R2Error("prepare requires parent card, R0 evidence root, code root, initializer, and new output root")
        prepare(args)
        return 0
    if args.run_card is None:
        raise R2Error(f"{args.phase} requires --run-card")
    if args.phase == "capacity":
        admit_capacity(args.run_card)
    elif args.phase == "verify":
        base, card = verify_card(args.run_card)
        print(json.dumps({"status": "PASS_R2_RUN_CARD", "run_card_sha256": sha256_file(args.run_card), "execution_binding_id": card["r2_execution_binding"]["binding_id"], "schedule_sha256": card["schedule"]["sha256"]}))
    elif args.phase == "train":
        train(args.run_card)
    elif args.phase == "finalize-training":
        close_training(args.run_card)
    elif args.phase == "screen":
        screen(args.run_card, args.device)
    elif args.phase == "close-screen":
        print(json.dumps(_complete_screen(args.run_card), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
