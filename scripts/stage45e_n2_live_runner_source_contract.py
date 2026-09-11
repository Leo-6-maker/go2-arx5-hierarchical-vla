#!/usr/bin/env python3
"""CPU/static source-contract gate for the future corrected Corelab N2 runner.

This tool is intentionally run *before* the corrected G3->G4->G5 continuation
is promoted into the live execution branch. It inspects source/manifest text and
AST only. It never imports the runner, Torch, GR00T, or CUDA.

The current PR #56 live runner is expected to FAIL this contract until the
PI0127 corrections are applied; that is deliberate.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


CORRECT_BUDGETS = {
    "G3_D1": {"conditioning_passes": 96, "action_head_calls": 864, "total_model_side_calls": 960},
    "G4_D2A": {"conditioning_passes": 96, "action_head_calls": 768, "total_model_side_calls": 864},
    "G5_D1": {"conditioning_passes": 72, "action_head_calls": 648, "total_model_side_calls": 720},
    "G5_D2A": {"conditioning_passes": 72, "action_head_calls": 576, "total_model_side_calls": 648},
}
LEGACY_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"


class LiveRunnerSourceContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveRunnerSourceContractError(message)


def read_text(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing source: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Call) and _call_name(item) == name for item in ast.walk(node))


def _loop_target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            names.add(item.id)
    return names


def _d1_conditioning_reuse(tree: ast.AST) -> dict[str, Any]:
    offending: list[dict[str, Any]] = []
    conditioning_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "_prepare_model_conditioning":
            conditioning_calls += 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            targets = _loop_target_names(node.target)
            if "noise_seed" in targets and _contains_call(node, "_prepare_model_conditioning"):
                offending.append({"line": getattr(node, "lineno", None), "target_names": sorted(targets)})
    return {
        "prepare_conditioning_call_sites": conditioning_calls,
        "conditioning_inside_noise_seed_loop": bool(offending),
        "offending_noise_seed_loops": offending,
        "pass": not offending,
    }


def _function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _string_literals(tree: ast.AST) -> set[str]:
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _manifest_budget_candidates(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    candidates: dict[str, Mapping[str, Any]] = {}
    aliases = {
        "G3_D1": ("g3_d1_teacher_forced_local_flow", "G3_D1"),
        "G4_D2A": ("g4_d2a_tf_vs_fr", "G4_D2A"),
        "G5_D1": ("g5_d1_n2_at_1000", "G5_D1"),
        "G5_D2A": ("g5_d2a_n2_at_1000", "G5_D2A"),
    }
    corrected = manifest.get("corrected_budgets")
    for phase, names in aliases.items():
        found = None
        if isinstance(corrected, Mapping) and isinstance(corrected.get(phase), Mapping):
            found = corrected[phase]
        if found is None:
            for name in names:
                if isinstance(manifest.get(name), Mapping):
                    found = manifest[name]
                    break
        require(isinstance(found, Mapping), f"manifest budget block missing for {phase}")
        candidates[phase] = found
    return candidates


def audit(runner_path: Path, manifest_path: Path) -> dict[str, Any]:
    source = read_text(runner_path)
    tree = ast.parse(source, filename=str(runner_path))
    manifest = read_json(manifest_path)
    require(isinstance(manifest, Mapping), "manifest must be object")

    budgets = _manifest_budget_candidates(manifest)
    budget_status: dict[str, Any] = {}
    for phase, expected in CORRECT_BUDGETS.items():
        observed = budgets[phase]
        normalized = {
            "conditioning_passes": observed.get("conditioning_passes"),
            "action_head_calls": observed.get("action_head_calls"),
            "total_model_side_calls": observed.get("total_model_side_calls"),
        }
        require(normalized == expected, f"{phase} manifest budget drift: {normalized} != {expected}")
        budget_status[phase] = normalized

    d1 = _d1_conditioning_reuse(tree)
    require(d1["pass"], "D1 conditioning is still recomputed inside the noise_seed loop")

    functions = _function_names(tree)
    literals = _string_literals(tree)
    require("run_continuation_g3_g4_g5" in functions, "continuation-safe G3/G4/G5 driver function missing")
    require("continue-g3-g4-g5" in literals, "continuation-safe CLI mode marker missing")
    require("checkpoint-2000" in literals, "explicit legacy logical checkpoint-2000 namespace missing")
    require(LEGACY_TREE in source, "accepted legacy FULL24@2000 tree identity missing")
    require(N2_STATS_SHA in source and N2_STATS_PAYLOAD_SHA in source, "frozen N2 stats identities missing")

    forbidden_continuation_markers = {
        "continuation_reruns_g1": "G1_REPLICATION_TRAINING",
        "continuation_reruns_g2": "G2_TRAIN24_E4",
    }
    manifest_forbidden = manifest.get("forbidden_continuation_phases", [])
    require(isinstance(manifest_forbidden, list), "forbidden_continuation_phases missing")
    for label, marker in forbidden_continuation_markers.items():
        require(marker in manifest_forbidden, f"manifest does not forbid {marker} during continuation")

    # Truthful failure provenance must be represented explicitly. The corrected
    # runner may use a helper or a named phase-local record; require both semantic
    # markers rather than depending on one implementation shape.
    require("phase_local" in source or "phase_counters" in source, "phase-local failure-counter source marker missing")
    require("FAILURE_SEAL" in source, "failure seal marker missing")

    return {
        "schema": "STAGE45E_N2_LIVE_RUNNER_SOURCE_CONTRACT_V1",
        "status": "PASS_STATIC_SOURCE_CONTRACT",
        "runner_path": str(Path(runner_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "corrected_budgets": budget_status,
        "d1_conditioning_reuse": d1,
        "continuation_driver_present": True,
        "continuation_cli_mode_present": True,
        "legacy_logical_prefix_bound": "checkpoint-2000",
        "n2_stats_identities_bound": True,
        "g1_g2_forbidden_in_continuation_manifest": True,
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
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(audit(args.runner, args.manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LiveRunnerSourceContractError as exc:
        print(f"STAGE45E_N2_LIVE_RUNNER_SOURCE_CONTRACT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
