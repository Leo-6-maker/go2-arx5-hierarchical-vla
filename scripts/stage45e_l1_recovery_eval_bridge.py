#!/usr/bin/env python3
"""CPU-only bridge from a 612326 recovery acceptance to L1 eval admission.

The bridge does not create a historical training PASS result. It validates the
fresh read-only acceptance artifact against the immutable training root and
emits a normalized *evaluation input contract* that a later evaluator patch
may consume. It never loads a model or launches evaluation itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ACCEPT_TOKEN = "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTED"
TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
BRANCH = "stage45e/l1-physical-error-aware-loss"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
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
    temporary.replace(path)


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
    tree_sha = accepted.get("tree_sha256")
    require(isinstance(tree_sha, str) and len(tree_sha) == 64, f"acceptance checkpoint-{step} tree identity missing")
    required_state = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    require(all((checkpoint / name).is_file() and not (checkpoint / name).is_symlink() for name in required_state), f"checkpoint-{step} full-state closure missing")
    return {
        "path": str(checkpoint.resolve()),
        "tree_sha256": tree_sha,
        "global_step": step,
        "required_state_files": list(required_state),
    }


def bridge(acceptance_path: Path, training_root: Path, output_path: Path) -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    acceptance_path = Path(acceptance_path).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    require(training_root.is_dir() and not training_root.is_symlink(), "historical training root unavailable")
    acceptance = read_json(acceptance_path)
    require(acceptance.get("schema") == "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1", "recovery acceptance schema drift")
    require(acceptance.get("status") == "PASS_READONLY_TRAINING_ARTIFACT_INTEGRITY", "recovery acceptance status drift")
    require(acceptance.get("token") == ACCEPT_TOKEN, "recovery acceptance token drift")
    require(acceptance.get("historical_training_head") == TRAINING_HEAD, "recovery training HEAD drift")
    require(acceptance.get("branch") == BRANCH, "recovery branch drift")
    require(Path(str(acceptance.get("training_root"))).resolve() == training_root, "recovery training-root identity drift")
    require(acceptance.get("historical_root_mutated") is False, "recovery claims historical-root mutation")
    require(acceptance.get("scientific_claim_boundary", {}).get("train24_evaluation_executed") is False, "recovery acceptance already claims evaluation")
    require(acceptance.get("structured_stats", {}).get("statistics_payload_sha256") == FULL24_STATS_PAYLOAD_SHA256, "recovery FULL24 stats payload drift")
    require(acceptance.get("structured_stats", {}).get("mask_active_scalar_count") == 190, "recovery T16 mask drift")
    require(not (training_root / "stage45e_l1_training_result.json").exists(), "historical PASS training result unexpectedly exists")

    checkpoints = {
        "L1@1000": _checkpoint_match(training_root, acceptance, 1000),
        "L1@2000": _checkpoint_match(training_root, acceptance, 2000),
    }
    weight_evidence_status = acceptance.get("scientific_claim_boundary", {}).get("runtime_effective_weight_evidence_status")
    require(weight_evidence_status in {"CAPTURED", "NOT_CAPTURED_IN_HISTORICAL_CANARY"}, "runtime weight evidence status drift")

    result = {
        "schema": "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V1",
        "status": "PASS_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION",
        "source_kind": "READONLY_RECOVERY_ACCEPTANCE_NOT_HISTORICAL_PASS_RESULT",
        "acceptance_path": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "training_root": str(training_root),
        "historical_training_head": TRAINING_HEAD,
        "branch": BRANCH,
        "checkpoints": checkpoints,
        "structured_stats": {
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
            "future evaluator must accept this recovered-input schema as an alternative to the missing historical training PASS result",
            "future evaluator execution HEAD is independent from historical model-training HEAD and both must be serialized",
            "future eval remains the frozen 360-call TRAIN24 E4 protocol only",
            "future evaluator must persist per-window row bundles for all three arms",
            "future evaluator must include L1@2000-minus-L1@1000 paired comparison",
            "future evidence inventory must be non-cyclic and immutable",
        ],
    }
    write_json(output_path, result)
    return result


def plan() -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_L1_RECOVERY_EVAL_BRIDGE_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "accept_token": ACCEPT_TOKEN,
        "historical_training_head": TRAINING_HEAD,
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
