#!/usr/bin/env python3
"""Bind recovered 612326 training artifacts to the exact frozen L1 contract.

This CPU/filesystem-only layer sits after the basic read-only recovery audit.
It proves that the historical runtime binding and input audit carried the exact
PI0123 L1 intervention and frozen training recipe.  It does not mutate the
historical root, load a model, or authorize evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


BASE_ACCEPT_TOKEN = "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTED"
FINAL_ACCEPT_TOKEN = "STAGE45E_L1_612326_CONTRACT_BOUND_TRAINING_ACCEPTED"
TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
BRANCH = "stage45e/l1-physical-error-aware-loss"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
L1_MATRIX_SHA256 = "b844c1c9a0400ad25ed55d43eb9d5cfe932435bb52340539038e7f4d6ace93a9"
XYZ_WEIGHTS = {"x": 1.401885, "y": 0.472436, "z": 1.125678}
FORMULA = "sum((prediction-target)^2 * binary_action_mask * l1_loss_weight) / binary_action_mask.sum()"

EXPECTED_TRAINING_RECIPE = {
    "base_mode": "FRESH_BASE_NO_RESUME",
    "seed": 2222,
    "max_steps": 2000,
    "save_steps": [1000, 2000],
    "global_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": "adamw_torch",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "warmup_ratio": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "fp16": False,
    "tf32": True,
    "num_gpus": 1,
    "dataloader_num_workers": 0,
    "eval_strategy": "no",
    "save_only_model": False,
    "resume_from_checkpoint": False,
    "trainable_modules": ["projector", "diffusion_model"],
    "frozen_modules": ["visual", "llm", "vlln", "vl_self_attention"],
}


class L1ContractGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise L1ContractGateError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _exact_subset(value: Any, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} missing")
    observed: dict[str, Any] = {}
    for field, expected_value in expected.items():
        require(field in value, f"{label}.{field} missing")
        require(value[field] == expected_value, f"{label}.{field} differs from frozen contract")
        observed[field] = value[field]
    return observed


def _runtime_l1_contract(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
        "normalization_unchanged": True,
        "stats_sha256": FULL24_STATS_SHA256,
        "mask_schema": "T16_BINARY_UNCHANGED",
        "mask_shape": [40, 132],
        "active_scalar_count": 190,
        "xyz_rows": [0, 14],
        "xyz_channels": [0, 1, 2],
        "xyz_loss_weights": XYZ_WEIGHTS,
        "other_active_weight": 1.0,
        "inactive_weight": 1.0,
        "weight_matrix_sha256": L1_MATRIX_SHA256,
        "formula": FORMULA,
        "mask_folding": False,
        "adaptive_renormalization": False,
    }
    return _exact_subset(value, expected, label)


def _code_l1_contract(value: Any) -> dict[str, Any]:
    expected = {
        "schema": "STAGE45E_L1_PHYSICAL_ERROR_AWARE_XYZ_LOSS_V1",
        "weight_shape": [40, 132],
        "xyz_rows": [0, 14],
        "xyz_channels": [0, 1, 2],
        "xyz_loss_weights": XYZ_WEIGHTS,
        "all_other_active_dimensions_weight": 1.0,
        "inactive_dimensions_weight": 1.0,
        "mask_schema": "T16_BINARY_UNCHANGED",
        "active_scalar_count": 190,
        "denominator": "binary_action_mask.sum()",
        "mask_folding": False,
        "adaptive_renormalization": False,
        "mean_weight_normalization": False,
        "matrix_sha256": L1_MATRIX_SHA256,
    }
    return _exact_subset(value, expected, "input_audit.code_contract")


def _runtime_binding(path: Path) -> dict[str, Any]:
    document = read_json(path)
    result: dict[str, Any] = {"sha256": sha256_file(path), "sides": {}}
    require(isinstance(document, Mapping), "runtime binding must be object")
    for side_name in ("binding", "observed"):
        side = document.get(side_name)
        require(isinstance(side, Mapping), f"runtime binding.{side_name} missing")
        l1 = _runtime_l1_contract(side.get("l1_contract"), f"runtime_binding.{side_name}.l1_contract")
        recipe = _exact_subset(side.get("training_recipe"), EXPECTED_TRAINING_RECIPE, f"runtime_binding.{side_name}.training_recipe")
        result["sides"][side_name] = {"l1_contract": l1, "training_recipe": recipe}
    require(
        result["sides"]["binding"] == result["sides"]["observed"],
        "runtime binding and observed scientific contracts differ",
    )
    return result


def _input_audit(path: Path) -> dict[str, Any]:
    document = read_json(path)
    require(document.get("schema") == "VANDA_STAGE45E_L1_INPUT_AUDIT_V1", "input audit schema drift")
    require(document.get("status") == "PASS", "input audit did not PASS")
    code_contract = _code_l1_contract(document.get("code_contract"))
    recipe = _exact_subset(document.get("training_recipe"), EXPECTED_TRAINING_RECIPE, "input_audit.training_recipe")
    identities = document.get("identities")
    require(isinstance(identities, Mapping), "input audit identities missing")
    require(identities.get("full24_stats_sha256") == FULL24_STATS_SHA256, "input audit FULL24 stats SHA drift")
    require(identities.get("full24_stats_payload_sha256") == FULL24_STATS_PAYLOAD_SHA256, "input audit FULL24 stats payload drift")
    require(identities.get("normalization_unchanged") is True, "input audit normalization drift")
    return {
        "sha256": sha256_file(path),
        "code_contract": code_contract,
        "training_recipe": recipe,
        "full24_stats_sha256": FULL24_STATS_SHA256,
        "full24_stats_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
        "normalization_unchanged": True,
    }


def bind(base_acceptance_path: Path, training_root: Path, output_path: Path) -> dict[str, Any]:
    base_acceptance_path = Path(base_acceptance_path).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    require(training_root.is_dir() and not training_root.is_symlink(), "historical training root unavailable")
    base = read_json(base_acceptance_path)
    require(base.get("schema") == "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1", "base acceptance schema drift")
    require(base.get("token") == BASE_ACCEPT_TOKEN, "base acceptance token drift")
    require(base.get("status") == "PASS_READONLY_TRAINING_ARTIFACT_INTEGRITY", "base acceptance status drift")
    require(base.get("historical_training_head") == TRAINING_HEAD, "base acceptance training HEAD drift")
    require(base.get("branch") == BRANCH, "base acceptance branch drift")
    require(Path(str(base.get("training_root"))).resolve() == training_root, "base acceptance training root drift")
    require(base.get("structured_stats", {}).get("statistics_payload_sha256") == FULL24_STATS_PAYLOAD_SHA256, "base acceptance stats payload drift")
    require(base.get("structured_stats", {}).get("mask_active_scalar_count") == 190, "base acceptance mask drift")

    runtime = _runtime_binding(training_root / "stage45e_l1_runtime_binding.json")
    input_audit = _input_audit(training_root / "stage45e_l1_input_audit.json")
    runtime_contract = runtime["sides"]["binding"]["l1_contract"]
    require(runtime_contract["xyz_loss_weights"] == input_audit["code_contract"]["xyz_loss_weights"], "runtime/input XYZ loss weights disagree")
    require(runtime_contract["weight_matrix_sha256"] == input_audit["code_contract"]["matrix_sha256"], "runtime/input weight matrix SHA disagree")
    require(runtime_contract["active_scalar_count"] == input_audit["code_contract"]["active_scalar_count"] == 190, "runtime/input active mask count disagree")
    require(runtime_contract["mask_folding"] is False and input_audit["code_contract"]["mask_folding"] is False, "mask folding observed")
    require(runtime_contract["adaptive_renormalization"] is False and input_audit["code_contract"]["adaptive_renormalization"] is False, "adaptive renormalization observed")
    require(runtime["sides"]["binding"]["training_recipe"] == input_audit["training_recipe"], "runtime/input training recipe disagree")

    final = {
        "schema": "STAGE45E_L1_612326_CONTRACT_BOUND_ACCEPTANCE_V1",
        "status": "PASS_READONLY_TRAINING_AND_FROZEN_CONTRACT_INTEGRITY",
        "token": FINAL_ACCEPT_TOKEN,
        "training_root": str(training_root),
        "historical_training_head": TRAINING_HEAD,
        "branch": BRANCH,
        "base_acceptance": {
            "path": str(base_acceptance_path),
            "sha256": sha256_file(base_acceptance_path),
            "token": BASE_ACCEPT_TOKEN,
        },
        "scientific_contract": {
            "l1_runtime_contract": runtime_contract,
            "l1_code_contract": input_audit["code_contract"],
            "training_recipe": input_audit["training_recipe"],
            "full24_stats_sha256": FULL24_STATS_SHA256,
            "full24_stats_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "normalization_unchanged": True,
            "runtime_binding_sha256": runtime["sha256"],
            "input_audit_sha256": input_audit["sha256"],
            "runtime_effective_weight_dtype_captured": base.get("scientific_claim_boundary", {}).get("runtime_effective_weight_dtype_captured") is True,
            "runtime_effective_weight_evidence_status": base.get("scientific_claim_boundary", {}).get("runtime_effective_weight_evidence_status"),
        },
        "checkpoint_1000": base.get("checkpoint_1000"),
        "checkpoint_2000": base.get("checkpoint_2000"),
        "historical_root_mutated": False,
        "evaluation_executed": False,
        "execution_counters": {
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "gpu": 0,
            "pbs": 0,
            "evaluation": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        "claim_boundary": "This acceptance proves the completed 612326 artifacts were produced under the frozen PI0123 L1 contract and training recipe. It does not recover historically unrecorded effective runtime coefficient dtype/values and does not authorize evaluation.",
    }
    require(isinstance(final["checkpoint_1000"], Mapping) and isinstance(final["checkpoint_2000"], Mapping), "base acceptance checkpoint records missing")
    write_json(output_path, final)
    return final


def plan() -> dict[str, Any]:
    return {
        "schema": "STAGE45E_L1_612326_CONTRACT_GATE_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "base_accept_token": BASE_ACCEPT_TOKEN,
        "final_accept_token": FINAL_ACCEPT_TOKEN,
        "training_head": TRAINING_HEAD,
        "expected_xyz_loss_weights": XYZ_WEIGHTS,
        "expected_weight_matrix_sha256": L1_MATRIX_SHA256,
        "expected_training_recipe": EXPECTED_TRAINING_RECIPE,
        "historical_root_mutation_allowed": False,
        "evaluation_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--bind", action="store_true")
    p.add_argument("--base-acceptance", type=Path)
    p.add_argument("--training-root", type=Path)
    p.add_argument("--output", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.base_acceptance is not None and args.training_root is not None and args.output is not None, "--bind requires base-acceptance, training-root, output")
    result = bind(args.base_acceptance, args.training_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except L1ContractGateError as exc:
        print(f"STAGE45E_L1_612326_CONTRACT_GATE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
