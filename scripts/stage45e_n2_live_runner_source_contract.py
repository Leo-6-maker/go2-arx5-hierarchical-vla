#!/usr/bin/env python3
"""CPU-only source contract for the PI0130 continuation runner.

The checker reads the live runner and manifest as text/AST only.  It does not
import the runner, Torch, GR00T, CUDA, or any dataset/checkpoint reader.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CORRECT_BUDGETS = {
    "G3_D1": {"conditioning_passes": 96, "action_head_calls": 864, "total_model_side_calls": 960},
    "G4_D2A": {"conditioning_passes": 96, "action_head_calls": 768, "total_model_side_calls": 864},
    "G5_D1": {"conditioning_passes": 72, "action_head_calls": 648, "total_model_side_calls": 720},
    "G5_D2A": {"conditioning_passes": 72, "action_head_calls": 576, "total_model_side_calls": 648},
}
LEGACY_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
FORBIDDEN_TOP_LEVEL_IMPORTS = {"torch", "transformers", "gr00t", "jax", "tensorflow", "isaacgym", "omni"}


class LiveRunnerSourceContractError(RuntimeError):
    """Raised when a continuation contract cannot be proven statically."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveRunnerSourceContractError(message)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _names_in_target(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}


def _function_nodes(tree: ast.AST) -> dict[str, ast.AST]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _d1_conditioning_reuse(tree: ast.AST) -> dict[str, Any]:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_name(node) == "_prepare_model_conditioning"]
    offending: list[dict[str, Any]] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor)):
            continue
        if "noise_seed" not in _names_in_target(loop.target):
            continue
        if any(isinstance(node, ast.Call) and _call_name(node) == "_prepare_model_conditioning" for node in ast.walk(loop)):
            offending.append({"line": getattr(loop, "lineno", None)})
    return {
        "prepare_conditioning_call_sites": len(calls),
        "conditioning_inside_noise_seed_loop": bool(offending),
        "offending_noise_seed_loops": offending,
        "pass": not offending,
    }


def _top_level_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    return sorted(set(imports))


def audit(runner_path: Path, manifest_path: Path) -> dict[str, Any]:
    runner_path = Path(runner_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    require(runner_path.is_file() and not runner_path.is_symlink(), f"runner unavailable: {runner_path}")
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"manifest unavailable: {manifest_path}")
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(runner_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("authority", {}).get("pi_gate") == "PI_GATE_STATE_0130", "PI0130 authority drift")
    require(manifest.get("authority", {}).get("issue_comment") == 5642258588, "PI0130 issue comment drift")
    continuation = manifest.get("continuation")
    require(isinstance(continuation, Mapping), "continuation manifest block missing")
    require(continuation.get("order") == ["G3_D1", "G4_D2A", "G5_D1", "G5_D2A"], "continuation order drift")
    forbidden = continuation.get("forbidden_phases")
    require(isinstance(forbidden, list), "continuation forbidden-phase list missing")
    for phase in ("G0_ADMISSION", "G1_REPLICATION_TRAINING", "G2_TRAIN24_E4"):
        require(phase in forbidden, f"continuation does not forbid {phase}")
    for phase, expected in CORRECT_BUDGETS.items():
        block = manifest.get({
            "G3_D1": "g3_d1_teacher_forced_local_flow",
            "G4_D2A": "g4_d2a_tf_vs_fr",
            "G5_D1": "g5_d1_n2_at_1000",
            "G5_D2A": "g5_d2a_n2_at_1000",
        }[phase])
        require(isinstance(block, Mapping), f"manifest block missing for {phase}")
        observed = {key: block.get(key) for key in expected}
        require(observed == expected, f"{phase} budget drift: {observed} != {expected}")

    functions = _function_nodes(tree)
    require("run_continuation_g3_g4_g5" in functions, "continuation driver function missing")
    continuation_calls = {_call_name(node) for node in ast.walk(functions["run_continuation_g3_g4_g5"]) if isinstance(node, ast.Call)}
    require(not continuation_calls.intersection({"admission", "train_one", "g2_eval", "run_sprint"}), "continuation driver can rerun G1/G2")
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    require("continue-g3-g4-g5" in literals, "continuation CLI mode marker missing")
    require("checkpoint-2000" in literals, "explicit legacy checkpoint namespace missing")
    require(LEGACY_TREE in source, "accepted legacy FULL24@2000 tree identity missing")
    require(N2_STATS_SHA in source and N2_STATS_PAYLOAD_SHA in source, "frozen N2 statistics identities missing")
    d1 = _d1_conditioning_reuse(tree)
    require(d1["pass"], "D1 conditioning remains inside noise_seed loop")
    require("_sync_failure_boundary" in source and "counters" in source, "phase-local failure provenance marker missing")
    top_level_forbidden = sorted(set(_top_level_imports(tree)).intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    require(not top_level_forbidden, f"forbidden top-level runtime imports: {top_level_forbidden}")
    return {
        "schema": "CORELAB_STAGE45E_N2_LIVE_RUNNER_SOURCE_CONTRACT_V2",
        "status": "PASS_STATIC_SOURCE_CONTRACT",
        "runner": str(runner_path),
        "manifest": str(manifest_path),
        "corrected_budgets": CORRECT_BUDGETS,
        "d1_conditioning_reuse": d1,
        "continuation_order": continuation["order"],
        "forbidden_continuation_phases": forbidden,
        "legacy_logical_prefix": "checkpoint-2000",
        "n2_stats_bound": True,
        "top_level_forbidden_imports": top_level_forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.runner, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LiveRunnerSourceContractError as exc:
        print(f"STAGE45E_N2_LIVE_RUNNER_SOURCE_CONTRACT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
