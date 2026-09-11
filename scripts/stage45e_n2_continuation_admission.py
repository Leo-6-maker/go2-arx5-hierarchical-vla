#!/usr/bin/env python3
"""CPU/filesystem-only admission helper for the corrected N2 G3→G4→G5 continuation.

This prep tool verifies the exact legacy FULL24@2000 tree under its frozen
logical namespace and verifies all three completed N2 roots, including their
N2 statistics identity and training sidecar binding. It never imports model
libraries, never calls a model, and never launches the continuation itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_N2_CONTINUATION_REPAIR_PREP_V1.json"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "transformers",
    "gr00t",
    "jax",
    "tensorflow",
    "isaacgym",
    "omni",
    "subprocess",
    "numpy",
}


class ContinuationAdmissionError(RuntimeError):
    """Raised when a continuation input cannot be bound exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationAdmissionError(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "STAGE45E_N2_CONTINUATION_REPAIR_PREP_V1", "manifest schema drift")
    require(value.get("status") == "PREP_ONLY_NO_MODEL_EXECUTION", "manifest is not prep-only")
    return value


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree file: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def legacy_full24_2000_tree(path: Path) -> dict[str, Any]:
    cfg = manifest()["accepted_identities"]
    prefix = cfg["legacy_full24_2000_logical_prefix"]
    entries = _tree_entries(Path(path).resolve(), prefix)
    digest = _tree_digest(entries)
    require(digest == cfg["legacy_full24_2000_tree_sha256"], f"legacy FULL24@2000 tree mismatch: {digest}")
    return {
        "path": str(Path(path).resolve()),
        "logical_prefix": prefix,
        "tree_sha256": digest,
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "destination_name_invariant": True,
    }


def compare_statistics(left: Any, right: Any, *, atol: float) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def visit(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                mismatches.append({"path": path, "left_keys": sorted(map(str, a)), "right_keys": sorted(map(str, b))})
                return
            for key in sorted(a, key=str):
                visit(a[key], b[key], f"{path}/{key}")
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatches.append({"path": path, "left_length": len(a), "right_length": len(b)})
                return
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if (
            isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
        ):
            left_number = float(a)
            right_number = float(b)
            if not math.isfinite(left_number) or not math.isfinite(right_number):
                mismatches.append({"path": path, "left": a, "right": b, "reason": "nonfinite"})
                return
            delta = abs(left_number - right_number)
            if delta > atol:
                mismatches.append({"path": path, "left": a, "right": b, "abs_delta": delta})
            return
        if a != b:
            mismatches.append({"path": path, "left": a, "right": b})

    visit(left, right, "statistics")
    deltas = [float(item["abs_delta"]) for item in mismatches if "abs_delta" in item]
    return {
        "match": not mismatches,
        "numeric_atol": atol,
        "mismatch_count": len(mismatches),
        "max_abs_delta": max(deltas, default=0.0),
        "mismatches": mismatches,
    }


def _result_path(root: Path) -> Path:
    candidates = (
        root / "training_result.json",
        root / "stage45e_n2_training_result.json",
    )
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    require(len(present) == 1, f"N2 result identity is ambiguous or absent: {root}")
    return present[0]


def _checkpoint(root: Path, step: int, expected_tree: str | None) -> dict[str, Any]:
    path = root / f"checkpoint-{step}"
    require(path.is_dir() and not path.is_symlink(), f"N2 checkpoint-{step} missing: {root}")
    state = read_json(path / "trainer_state.json")
    require(state.get("global_step") == step, f"N2 checkpoint-{step} global-step drift")
    required_state = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    require(all((path / name).is_file() and not (path / name).is_symlink() for name in required_state), f"N2 checkpoint-{step} full-state closure missing")
    entries = _tree_entries(path, f"stage45e/checkpoint-{step}")
    digest = _tree_digest(entries)
    if expected_tree is not None:
        require(digest == expected_tree, f"N2 checkpoint-{step} tree differs from training result")
    return {
        "path": str(path.resolve()),
        "global_step": step,
        "tree_sha256": digest,
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
    }


def n2_root(root: Path, expected_seed: int, n2_stats_path: Path) -> dict[str, Any]:
    cfg = manifest()
    identities = cfg["accepted_identities"]
    compare_cfg = cfg["stats_sidecar_compare"]
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"N2 root unavailable: {root}")

    stats_path = Path(n2_stats_path).resolve()
    require(sha256_file(stats_path) == identities["n2_stats_sha256"], "accepted N2 statistics file SHA drift")
    stats_doc = read_json(stats_path)
    require(stats_doc.get("statistics_payload_sha256") == identities["n2_stats_payload_sha256"], "accepted N2 statistics payload field drift")
    frozen_stats = stats_doc.get("statistics")
    require(isinstance(frozen_stats, Mapping), "accepted N2 statistics payload missing")
    require(canonical_sha(frozen_stats) == identities["n2_stats_payload_sha256"], "accepted N2 statistics payload content drift")

    result_path = _result_path(root)
    result = read_json(result_path)
    require(result.get("seed") == expected_seed, f"N2 seed mismatch: expected {expected_seed}")
    require(result.get("starting_checkpoint") == "BASE", "N2 root did not start from BASE")
    require(result.get("base_tree_sha256") == identities["base_tree_sha256"], "N2 BASE identity drift")
    require(result.get("full24_tree_sha256") == identities["full24_tree_sha256"], "N2 FULL24 identity drift")
    require(result.get("training_steps", result.get("max_steps")) == identities["n2_training_steps"], "N2 training-step budget drift")
    require(result.get("n2_stats_sha256") == identities["n2_stats_sha256"], "N2 training result statistics file identity drift")
    require(result.get("n2_stats_payload_sha256") == identities["n2_stats_payload_sha256"], "N2 training result statistics payload identity drift")
    status = result.get("status")
    require(status in {"PASS", "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW"}, f"N2 training result is not complete: {status}")

    sidecar_path = root / "experiment_cfg" / "structured_action_statistics.json"
    sidecar = read_json(sidecar_path)
    sidecar_stats = sidecar.get("statistics")
    require(isinstance(sidecar_stats, Mapping), "N2 sidecar statistics missing")
    comparison = compare_statistics(sidecar_stats, frozen_stats, atol=float(compare_cfg["numeric_atol"]))
    require(comparison["match"], f"N2 sidecar statistics drift: {comparison['mismatches'][:10]}")

    checkpoints: dict[str, Any] = {}
    for step in (1000, 2000):
        result_record = result.get(f"checkpoint_{step}")
        require(isinstance(result_record, Mapping), f"N2 checkpoint-{step} record missing from training result")
        expected_tree = result_record.get("tree_sha256")
        require(isinstance(expected_tree, str) and len(expected_tree) == 64, f"N2 checkpoint-{step} training-result tree identity missing")
        checkpoints[str(step)] = _checkpoint(root, step, expected_tree)

    return {
        "root": str(root),
        "seed": expected_seed,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "n2_stats_sha256": identities["n2_stats_sha256"],
        "n2_stats_payload_sha256": identities["n2_stats_payload_sha256"],
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "sidecar_statistics_binding": comparison,
        "checkpoints": checkpoints,
    }


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
        "schema": "STAGE45E_N2_CONTINUATION_ADMISSION_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def plan() -> dict[str, Any]:
    cfg = manifest()
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_N2_CONTINUATION_REPAIR_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "authority": cfg["authority"],
        "continuation_order": cfg["continuation_order"],
        "forbidden_continuation_phases": cfg["forbidden_continuation_phases"],
        "corrected_budgets": cfg["corrected_budgets"],
        "d1_conditioning_semantics": cfg["d1_conditioning_semantics"],
        "runner_patch_requirements": [
            "move D1 conditioning construction outside the noise-seed loop and reuse across 3 seeds x 3 t",
            "use explicit legacy logical tree prefix checkpoint-2000, never destination Path.name",
            "add continuation-only entrypoint that cannot run G1/G2/training",
            "bind every N2 root to exact N2 stats file/payload and sidecar statistics before model load",
            "make phase-local counters authoritative in failure sealing after model execution begins",
            "update manifest authority/budgets without rewriting historical G1/G2 evidence",
        ],
        "execution_enabled": False,
        "static_guard": guard,
        "return_token": cfg["return_token"],
    }


def admit(
    legacy_full24_2000: Path,
    n2_2222_root: Path,
    n2_2223_root: Path,
    n2_2224_root: Path,
    n2_stats: Path,
    output: Path,
) -> dict[str, Any]:
    cfg = manifest()
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    legacy = legacy_full24_2000_tree(legacy_full24_2000)
    roots = {
        "2222": n2_root(n2_2222_root, 2222, n2_stats),
        "2223": n2_root(n2_2223_root, 2223, n2_stats),
        "2224": n2_root(n2_2224_root, 2224, n2_stats),
    }
    result = {
        "schema": "STAGE45E_N2_CONTINUATION_CPU_ADMISSION_V1",
        "status": "PASS_CPU_INPUT_IDENTITIES_ONLY",
        "execution_enabled": False,
        "legacy_full24_2000": legacy,
        "n2_roots": roots,
        "continuation_order": cfg["continuation_order"],
        "corrected_budgets": cfg["corrected_budgets"],
        "scientific_contract_changed": False,
        "execution_counters": {
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "gpu": 0,
            "pbs": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": 0,
            "simulator": 0,
            "hardware": 0,
        },
        "next_action": "Use only after live PR #56 implements PI0127 corrections and Corelab execution authority resumes; this artifact alone cannot start G3.",
    }
    write_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--admit-inputs", action="store_true")
    value.add_argument("--legacy-full24-2000", type=Path)
    value.add_argument("--n2-2222-root", type=Path)
    value.add_argument("--n2-2223-root", type=Path)
    value.add_argument("--n2-2224-root", type=Path)
    value.add_argument("--n2-stats", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    required = (
        args.legacy_full24_2000,
        args.n2_2222_root,
        args.n2_2223_root,
        args.n2_2224_root,
        args.n2_stats,
        args.output,
    )
    require(all(item is not None for item in required), "--admit-inputs requires all legacy/N2 roots, N2 stats, and output")
    result = admit(
        args.legacy_full24_2000,
        args.n2_2222_root,
        args.n2_2223_root,
        args.n2_2224_root,
        args.n2_stats,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContinuationAdmissionError as exc:
        print(f"STAGE45E_N2_CONTINUATION_ADMISSION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
