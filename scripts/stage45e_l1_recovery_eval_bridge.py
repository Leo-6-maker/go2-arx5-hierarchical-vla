#!/usr/bin/env python3
"""CPU-only bridge from contract-bound 612326 recovery to L1 eval admission.

The bridge does not create a historical training PASS result. It validates the
fresh contract-bound read-only acceptance against the immutable training root,
rehashes both checkpoints, and emits a normalized evaluation input contract
that a later evaluator patch may consume. It never loads a model or launches
evaluation itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


ACCEPT_TOKEN = "STAGE45E_L1_612326_CONTRACT_BOUND_TRAINING_ACCEPTED"
TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
BRANCH = "stage45e/l1-physical-error-aware-loss"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
L1_MATRIX_SHA256 = "b844c1c9a0400ad25ed55d43eb9d5cfe932435bb52340539038e7f4d6ace93a9"
XYZ_WEIGHTS = {"x": 1.401885, "y": 0.472436, "z": 1.125678}
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch", "transformers", "gr00t", "cuda", "subprocess", "numpy",
}


class RecoveryEvalBridgeError(RuntimeError):
    """Raised when recovered training evidence is not evaluation-admissible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryEvalBridgeError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"file unavailable or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe checkpoint root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe checkpoint directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe checkpoint file: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    return {
        "schema": "STAGE45E_L1_RECOVERY_EVAL_BRIDGE_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "evaluation": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def _checkpoint_match(training_root: Path, acceptance_record: Mapping[str, Any], step: int) -> dict[str, Any]:
    checkpoint = training_root / f"checkpoint-{step}"
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"checkpoint-{step} missing")
    state = read_json(checkpoint / "trainer_state.json")
    require(state.get("global_step") == step, f"checkpoint-{step} global-step drift")
    accepted = acceptance_record.get(f"checkpoint_{step}")
    require(isinstance(accepted, Mapping), f"acceptance checkpoint-{step} record missing")
    require(Path(str(accepted.get("path"))).resolve() == checkpoint.resolve(), f"acceptance checkpoint-{step} path drift")
    require(accepted.get("global_step") == step, f"acceptance checkpoint-{step} step drift")
    accepted_tree = accepted.get("tree_sha256")
    require(isinstance(accepted_tree, str) and len(accepted_tree) == 64, f"acceptance checkpoint-{step} tree identity missing")
    required_state = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    require(all((checkpoint / name).is_file() and not (checkpoint / name).is_symlink() for name in required_state), f"checkpoint-{step} full-state closure missing")
    entries = _tree_entries(checkpoint, f"stage45e_l1/checkpoint-{step}")
    observed_tree = _tree_digest(entries)
    require(observed_tree == accepted_tree, f"checkpoint-{step} bytes changed after recovery acceptance")
    return {
        "path": str(checkpoint.resolve()),
        "tree_sha256": observed_tree,
        "global_step": step,
        "required_state_files": list(required_state),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "rehash_matches_acceptance": True,
    }


def _scientific_contract(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    contract = acceptance.get("scientific_contract")
    require(isinstance(contract, Mapping), "contract-bound acceptance scientific contract missing")
    runtime = contract.get("l1_runtime_contract")
    code = contract.get("l1_code_contract")
    recipe = contract.get("training_recipe")
    require(isinstance(runtime, Mapping) and isinstance(code, Mapping) and isinstance(recipe, Mapping), "contract-bound acceptance contract blocks missing")
    require(runtime.get("xyz_loss_weights") == XYZ_WEIGHTS, "contract-bound runtime XYZ weights drift")
    require(code.get("xyz_loss_weights") == XYZ_WEIGHTS, "contract-bound code XYZ weights drift")
    require(runtime.get("weight_matrix_sha256") == L1_MATRIX_SHA256, "contract-bound runtime matrix SHA drift")
    require(code.get("matrix_sha256") == L1_MATRIX_SHA256, "contract-bound code matrix SHA drift")
    require(runtime.get("active_scalar_count") == code.get("active_scalar_count") == 190, "contract-bound active mask drift")
    require(code.get("denominator") == "binary_action_mask.sum()", "contract-bound denominator drift")
    require(runtime.get("mask_folding") is False and code.get("mask_folding") is False, "contract-bound mask folding drift")
    require(runtime.get("adaptive_renormalization") is False and code.get("adaptive_renormalization") is False, "contract-bound adaptive renormalization drift")
    require(contract.get("full24_stats_sha256") == FULL24_STATS_SHA256, "contract-bound FULL24 stats SHA drift")
    require(contract.get("full24_stats_payload_sha256") == FULL24_STATS_PAYLOAD_SHA256, "contract-bound FULL24 stats payload drift")
    require(contract.get("normalization_unchanged") is True, "contract-bound normalization drift")
    require(recipe.get("seed") == 2222 and recipe.get("max_steps") == 2000, "contract-bound training seed/steps drift")
    require(recipe.get("global_batch_size") == 1 and recipe.get("gradient_accumulation_steps") == 1, "contract-bound batch semantics drift")
    require(recipe.get("optimizer") == "adamw_torch" and recipe.get("lr_scheduler_type") == "cosine", "contract-bound optimizer/scheduler drift")
    require(recipe.get("bf16") is True and recipe.get("fp16") is False and recipe.get("num_gpus") == 1, "contract-bound precision/resource semantics drift")
    return dict(contract)


def bridge(acceptance_path: Path, training_root: Path, output_path: Path) -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    acceptance_path = Path(acceptance_path).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    require(training_root.is_dir() and not training_root.is_symlink(), "historical training root unavailable")
    acceptance = read_json(acceptance_path)
    require(acceptance.get("schema") == "STAGE45E_L1_612326_CONTRACT_BOUND_ACCEPTANCE_V1", "contract-bound acceptance schema drift")
    require(acceptance.get("status") == "PASS_READONLY_TRAINING_AND_FROZEN_CONTRACT_INTEGRITY", "contract-bound acceptance status drift")
    require(acceptance.get("token") == ACCEPT_TOKEN, "contract-bound acceptance token drift")
    require(acceptance.get("historical_training_head") == TRAINING_HEAD, "recovery training HEAD drift")
    require(acceptance.get("branch") == BRANCH, "recovery branch drift")
    require(Path(str(acceptance.get("training_root"))).resolve() == training_root, "recovery training-root identity drift")
    require(acceptance.get("historical_root_mutated") is False, "recovery claims historical-root mutation")
    require(acceptance.get("evaluation_executed") is False, "contract-bound acceptance already claims evaluation")
    scientific_contract = _scientific_contract(acceptance)
    require(not (training_root / "stage45e_l1_training_result.json").exists(), "historical PASS training result unexpectedly exists")

    checkpoints = {
        "L1@1000": _checkpoint_match(training_root, acceptance, 1000),
        "L1@2000": _checkpoint_match(training_root, acceptance, 2000),
    }
    weight_evidence_status = scientific_contract.get("runtime_effective_weight_evidence_status")
    require(weight_evidence_status in {"CAPTURED", "NOT_CAPTURED_IN_HISTORICAL_CANARY"}, "runtime weight evidence status drift")

    result = {
        "schema": "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2",
        "status": "PASS_CONTRACT_BOUND_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION",
        "source_kind": "CONTRACT_BOUND_READONLY_RECOVERY_NOT_HISTORICAL_PASS_RESULT",
        "acceptance_path": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "training_root": str(training_root),
        "historical_training_head": TRAINING_HEAD,
        "branch": BRANCH,
        "checkpoints": checkpoints,
        "scientific_contract": scientific_contract,
        "structured_stats": {
            "stats_sha256": FULL24_STATS_SHA256,
            "statistics_payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
            "mask_active_scalar_count": 190,
            "normalization_unchanged": True,
        },
        "runtime_effective_weight_evidence_status": weight_evidence_status,
        "evaluation_authorized_by_this_bridge": False,
        "final_train24_token_emitted": False,
        "execution_counters": {
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "gpu": 0,
            "pbs": 0,
            "evaluation": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        "next_patch_contract": [
            "future evaluator must accept this V2 contract-bound recovered-input schema as an alternative to the missing historical training PASS result",
            "future evaluator execution HEAD is independent from historical model-training HEAD and both must be serialized",
            "future eval remains the frozen 360-call TRAIN24 E4 protocol only",
            "future evaluator must persist per-window row bundles for all three arms",
            "future evaluator must include L1@2000-minus-L1@1000 paired comparison",
            "future evidence inventory must be non-cyclic and immutable",
            "future row-bundle provenance must bind this bridge SHA, checkpoint tree SHA, eval seal SHA, normalization/sampler/decoder/metric identities",
        ],
    }
    write_json(output_path, result)
    return result


def plan() -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_L1_RECOVERY_EVAL_BRIDGE_PLAN_V2",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "accept_token": ACCEPT_TOKEN,
        "historical_training_head": TRAINING_HEAD,
        "contract_bound_required": True,
        "evaluation_authorized": False,
        "static_guard": guard,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--build-bridge", action="store_true")
    value.add_argument("--acceptance", type=Path)
    value.add_argument("--training-root", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.acceptance is not None and args.training_root is not None and args.output is not None, "--build-bridge requires acceptance, training-root, output")
    result = bridge(args.acceptance, args.training_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryEvalBridgeError as exc:
        print(f"STAGE45E_L1_RECOVERY_EVAL_BRIDGE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
