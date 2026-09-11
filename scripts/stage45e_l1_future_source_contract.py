#!/usr/bin/env python3
"""CPU/static source-contract gate for future corrected Stage45E-L1 code.

The current live PR #57 is expected to fail several checks. A later separately
gated patch should pass this contract before any H200 model execution resumes.
Only source text/AST is inspected; trainer/evaluator modules are never imported.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Sequence


RECOVERED_SCHEMA = "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2"
TRAINING_ONLY_TOKEN = "STAGE45E_L1_TRAINING_COMPLETE_FOR_EVAL"
FINAL_TOKEN = "STAGE45E_L1_VANDA_H200_TRAIN24_COMPLETE_FOR_PI_REVIEW"


class L1FutureSourceContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise L1FutureSourceContractError(message)


def read_source(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing source: {path}")
    return path.read_text(encoding="utf-8")


def _imports(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _call_names(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            result.add(target.id)
        elif isinstance(target, ast.Attribute):
            result.add(target.attr)
    return result


def audit(train_path: Path, eval_path: Path) -> dict[str, Any]:
    train_source = read_source(train_path)
    eval_source = read_source(eval_path)
    train_tree = ast.parse(train_source, filename=str(train_path))
    eval_tree = ast.parse(eval_source, filename=str(eval_path))
    train_imports = _imports(train_tree)
    eval_imports = _imports(eval_tree)
    train_calls = _call_names(train_tree)
    eval_calls = _call_names(eval_tree)

    # Training may prepare evaluation, but the training result must own a
    # distinct training-only completion token. Runtime weight evidence can be
    # serialized as one nested helper result; do not overfit field placement.
    require(TRAINING_ONLY_TOKEN in train_source, "training-only completion token missing")
    require("stage45e_l1_training_result.json" in train_source, "training result artifact marker missing")
    require("capture_effective_weight_evidence" in train_calls or "capture_effective_weight_evidence" in train_source, "future canary does not capture runtime effective L1 weights")
    require("runtime_weight_evidence" in train_source, "future canary does not serialize runtime weight evidence")

    # Evaluation must accept the recovered contract-bound input, rather than
    # requiring only the missing historical PASS training result.
    require(RECOVERED_SCHEMA in eval_source, "recovered V2 eval-input schema marker missing")
    require("recovered_eval_input" in eval_source or "recovered-eval-input" in eval_source, "recovered eval-input CLI/path binding missing")
    require("validate_recovered_input" in eval_calls or "validate_recovered_input" in eval_source, "recovered input validator not integrated")

    # Evidence completeness repairs.
    require("paired_summary_complete" in eval_calls or "L1@2000_minus_L1@1000" in eval_source, "L1 exposure paired comparison missing")
    require("persist_row_bundles" in eval_calls or "row_bundles" in eval_source, "per-window row bundle persistence missing")
    require("build_immutable_inventory" in eval_calls or "IMMUTABLE_SHA256SUMS" in eval_source, "non-cyclic immutable inventory helper missing")
    require(FINAL_TOKEN in eval_source or "admission.FULL_PASS_TOKEN" in eval_source, "final TRAIN24 completion token missing from evaluator")

    # Static governance markers must remain.
    for marker in ("Validation-E", "Holdout-F", "public_sampler_calls", "get_action_calls"):
        require(marker in eval_source, f"evaluation governance marker missing: {marker}")

    return {
        "schema": "STAGE45E_L1_FUTURE_SOURCE_CONTRACT_V2",
        "status": "PASS_STATIC_SOURCE_CONTRACT",
        "training_source": str(Path(train_path).resolve()),
        "evaluation_source": str(Path(eval_path).resolve()),
        "training_only_token_present": True,
        "runtime_effective_weight_evidence_present": True,
        "recovered_v2_input_supported": True,
        "l1_2000_minus_1000_pair_present": True,
        "per_window_row_bundles_present": True,
        "noncyclic_inventory_present": True,
        "final_completion_token_owned_by_evaluator": True,
        "train_import_count": len(train_imports),
        "eval_import_count": len(eval_imports),
        "model_imported": False,
        "model_forward": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-source", type=Path, required=True)
    p.add_argument("--eval-source", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(audit(args.train_source, args.eval_source), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except L1FutureSourceContractError as exc:
        print(f"STAGE45E_L1_FUTURE_SOURCE_CONTRACT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
