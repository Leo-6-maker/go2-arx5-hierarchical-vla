#!/usr/bin/env python3
"""PI0197 candidate late-binding over the accepted Stage43 runner.

This is deliberately a small Stage45F shim.  It owns only checkpoint
identity/tree closure, candidate late-binding, fixed matrix orchestration, and
post-run scorecard derivation.  The native simulator, worker protocol,
SafetyGate, reference handoff, WBC loop, timing, and condition semantics are
delegated to the immutable Stage43/Stage40 implementation.

The CPU admission path is safe to run under the exact native Python runtime
before a GPU is selected.  It imports the Stage43/Stage40 modules and verifies
all candidate overlays, but never starts a worker, simulator, model, or
forward.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests" / "STAGE45F_PI0196_CORELAB_STAGE43_EVAL_V1.json"
SCHEMA = "CORELAB_PI0196_STAGE43_REUSE_EVAL_V1"
CELL_SCHEMA = "CORELAB_PI0196_STAGE43_CELL_V1"
PI_GATE = "PI_GATE_STATE_0197"
ISSUE_COMMENT = 5680082654
PR_POINTER_COMMENT = 5680088636
BRANCH = "stage45f/corelab-fast-loop-readiness"
ADMISSION_TOKEN = "LOCALPC_PI0197_STAGE43_REUSE_EVAL_ADMISSION"
RETURN_TOKEN = "LOCALPC_PI0197_VANDA_E_S20_CLOSED_LOOP_SCORECARD"

STAGE43_ACCEPTED_HEAD = "b2a70dcec0765cf69819bb76c180e5c92ae3eac3"
STAGE43_BUNDLE_REPOSITORY_HEAD = "d042ecd51af437040103892d9b2e2a6acdd00a9b"
STAGE43_BUNDLE_SHA256 = "127d95a2a5b9a6a40e864e029d85fd09a9e4dd55441aae58071502aa76700da6"
STAGE43_RESULT_SHA256 = "e635ecaa321bde38903be45781e9b47063ba6b81c2bc810b1a1ae64925ecfcdb"
STAGE43_RESULT_SEAL_SHA256 = "668bee50dc38997403a065d5f4831662731715cf28c289af3982e4f9abbf7283"
STAGE43_COHORT_SHA256 = "62e95728ea6da7a0d90613e52adabc0737f3b116f63d4f01ebb680387ea2de48"
TRAJECTORY_SHA256 = "e09d03e4f837234a1ed7504f6d16f202a40d7fab2c3cbbc0f2e2fa17b0b33722"

DEV_SEED = 4242
TARGET_WORLD = (0.30, 0.10, 0.58)
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"
ACTION_CONTRACT = "STRUCTURED_ACTION_V1"
CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
CANDIDATE_ORDER = ("PARENT", "E", "S20")
CONDITION_DELTAS = {
    "C0_NOMINAL": (0.0, 0.0, 0.0),
    "C1_X_PLUS_5CM": (0.05, 0.0, 0.0),
    "C2_X_MINUS_5CM": (-0.05, 0.0, 0.0),
    "C3_Y_PLUS_5CM": (0.0, 0.05, 0.0),
    "C4_Y_MINUS_5CM": (0.0, -0.05, 0.0),
}
NATIVE_SCHEDULE = (49, 49, 49, 98)
FORWARDS_PER_CELL = 4
UPDATES_PER_CELL = 3
TICKS_PER_CELL = sum(NATIVE_SCHEDULE)
MAX_CELLS = len(CANDIDATE_ORDER) * len(CONDITION_ORDER)
MAX_FORWARDS = MAX_CELLS * FORWARDS_PER_CELL
MAX_UPDATES = MAX_CELLS * UPDATES_PER_CELL
MAX_TICKS = MAX_CELLS * TICKS_PER_CELL
CHECKPOINT_FILES = (
    "config.json",
    "embodiment_id.json",
    "experiment_cfg/conf.yaml",
    "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
    "experiment_cfg/structured_action_statistics.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "optimizer.pt",
    "processor_config.json",
    "rng_state.pth",
    "scheduler.pt",
    "statistics.json",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
)

CANDIDATES = {
    "PARENT": {
        "label": "PARENT L1(seed2223)@2000",
        "tree_sha256": "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16",
        "tree_prefix": "stage45e_l1/checkpoint-2000",
        "global_step": 2000,
        "stats_sha256": "d61cea2260998faa0d3d216368389cb68656556e0e9e89b15ac0902eb109ca23",
        "training_result_sha256": None,
        "training_invocation_sha256": None,
    },
    "E": {
        "label": "E@256",
        "tree_sha256": "f91f4fd3cf3c440127c57d49cdd2dde52ce6a1294bd17a05710fba1026b9361b",
        "tree_prefix": "617089/E/checkpoint-256",
        "global_step": 256,
        "stats_sha256": "430cfc259357259791e069e564fc4b5754cbeb144b720c5289c032fe62d80e3d",
        "training_result_sha256": "27fcf42478c77792d1314c91a0b311e367b192a778131234241ed844c88598f9",
        "training_invocation_sha256": "3ef11b3f0869893490daf6e55dcb874700f9cfdcab741fe85052f3f680bf6d08",
    },
    "S20": {
        "label": "S20@256",
        "tree_sha256": "64c082022bb6786efa0d8f706b33e36f6bfca8c4af13db41664dccff697c882d",
        "tree_prefix": "617089/S20/checkpoint-256",
        "global_step": 256,
        "stats_sha256": "430cfc259357259791e069e564fc4b5754cbeb144b720c5289c032fe62d80e3d",
        "training_result_sha256": "8b47dcffcf619aae9bda504e9616bfc1d620aaf1916bf11fab201d829a396e37",
        "training_invocation_sha256": "724c892587945cd4724bde5a59b5dc776a98a4f9b121f86b63cadb739f4a57a8",
    },
}

FROZEN_FILES = {
    "scripts/stage43_corelab_evidence.py": "4d85ebfb3c69ceb59bd8a7e22ea0c24f8a92f5af96ce1d3e5d78ecc8829edf73",
    "scripts/stage43_corelab_initial_condition.py": "f7318f2478cc295d405444cc6691d3522e10f49cfdb7c661ad102558406e9f7e",
    "scripts/stage40_corelab_phase.py": "24f069eaf640fac36d65c1ed5168a3ef30b2a2b06b26b5236a26b06682eeadbd",
    "scripts/stage40_live_one_handoff.py": "1df21f0ee2b907cd6c1b3c94a14023e7f23a5b24dc2cd383c377ed74f5594314",
    "scripts/stage21_same_scene_groot_worker.py": "311814d8a0119a9fc5d1d45a930260cdf46c8a6ce5e4b93932a7bae8214e797b",
    "scripts/stage21_same_scene_groot_inference.py": "fd7c32584a97bc14fa13d190043cffa27ab77aa8335189dfe40950a6827a2ae3",
    "scripts/stage21_native_receding_horizon.py": "753244da3f4cccce2cef94a6aba5a2fa37450499e324db7446b42841569abb00",
    "src/go2arx5_vla/evaluation/checkpoint_eval.py": "f65c38e3b7e03e3e0d788c4f91d101a1d243e1eccc2df9bd330bd5f124801536",
    "src/go2arx5_vla/stage40_structured_decode.py": "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae",
}


class AdapterError(RuntimeError):
    """A fail-closed provenance or execution contract error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def read_json(path: Path, *, canonical: bool = False) -> Dict[str, Any]:
    candidate = Path(path)
    require(candidate.is_file() and not candidate.is_symlink(), "JSON is not a regular file: %s" % candidate)
    value = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required: %s" % candidate)
    if canonical:
        require(candidate.read_bytes() == canonical_json(value), "JSON is not canonical: %s" % candidate)
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    destination = Path(path)
    require(overwrite or not destination.exists(), "refusing to overwrite evidence: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(value))
    return sha256_file(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_checkout() -> Dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-C", str(REPO)] + list(args), capture_output=True, text=True, check=False)
        require(completed.returncode == 0, "git command failed: %s" % " ".join(args))
        return completed.stdout.strip()

    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "clean": git("status", "--porcelain=v1", "--untracked-files=all") == "",
    }


def _regular(path: Path, label: str, directory: bool) -> Path:
    candidate = Path(path).expanduser()
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise AdapterError("%s is unavailable: %s" % (label, candidate)) from exc
    if directory:
        require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), "%s is not a regular directory: %s" % (label, candidate))
    else:
        require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), "%s is not a regular file: %s" % (label, candidate))
    return candidate.resolve()


def _tree_entries(root: Path, prefix: str) -> List[Dict[str, Any]]:
    candidate = _regular(root, "candidate", True)
    entries: List[Dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(str(candidate), followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), "symlink or non-directory candidate entry: %s" % child)
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), "symlink or non-file candidate entry: %s" % child)
            entries.append({
                "path": prefix + "/" + child.relative_to(candidate).as_posix(),
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(("%s\0%s\0%s\n" % (entry["path"], entry["bytes"], entry["sha256"])).encode("utf-8"))
    return digest.hexdigest()


def _relative_entry_path(path: str, prefix: str) -> str:
    marker = prefix + "/"
    require(path.startswith(marker), "candidate inventory path has unexpected prefix: %s" % path)
    return path[len(marker):]


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema") == SCHEMA, "PI0196 manifest schema drift")
    require(manifest.get("authority", {}).get("issue_comment") == ISSUE_COMMENT, "PI0196 Issue authority drift")
    require(manifest.get("authority", {}).get("pr_comment") == PR_POINTER_COMMENT, "PI0196 PR authority drift")
    task = manifest.get("task", {})
    require(task.get("dev_seed") == DEV_SEED and task.get("target_world_m") == list(TARGET_WORLD), "task identity drift")
    require(task.get("language") == LANGUAGE and task.get("source_frame") == SOURCE_FRAME and task.get("tcp_semantic_id") == TCP_ID, "task contract drift")
    require(task.get("target_xyz_in_model_input") is False and task.get("action_contract") == ACTION_CONTRACT, "model-input contract drift")
    matrix = manifest.get("matrix", {})
    require(tuple(matrix.get("candidate_order", ())) == CANDIDATE_ORDER, "candidate order drift")
    require(tuple(matrix.get("condition_order", ())) == CONDITION_ORDER, "condition order drift")
    require(matrix.get("condition_deltas_m") == {key: list(value) for key, value in CONDITION_DELTAS.items()}, "condition delta drift")
    require(matrix.get("gr00t_forwards_per_cell") == FORWARDS_PER_CELL and matrix.get("reference_updates_per_cell") == UPDATES_PER_CELL, "cell budget drift")
    require(matrix.get("native_ticks_per_cell") == list(NATIVE_SCHEDULE), "native schedule drift")
    require(matrix.get("maximum_cells") == MAX_CELLS and matrix.get("maximum_gr00t_forwards") == MAX_FORWARDS and matrix.get("maximum_reference_updates") == MAX_UPDATES and matrix.get("maximum_native_ticks") == MAX_TICKS, "matrix budget drift")
    for candidate in CANDIDATE_ORDER:
        item = manifest.get("candidates", {}).get(candidate, {})
        require(item.get("tree_sha256") == CANDIDATES[candidate]["tree_sha256"], "%s tree identity drift" % candidate)
        require(item.get("global_step") == CANDIDATES[candidate]["global_step"], "%s step identity drift" % candidate)
    historical = manifest.get("historical_stage43", {})
    require(historical.get("bundle_sha256") == STAGE43_BUNDLE_SHA256 and historical.get("result_sha256") == STAGE43_RESULT_SHA256, "Stage43 result authority drift")
    require(historical.get("result_seal_sha256") == STAGE43_RESULT_SEAL_SHA256 and historical.get("cohort_manifest_sha256") == STAGE43_COHORT_SHA256 and historical.get("trajectory_sha256") == TRAJECTORY_SHA256, "Stage43 seal/trajectory authority drift")
    require(manifest.get("frozen_files") == FROZEN_FILES, "frozen source hash map drift")


def static_contract(repo: Path = REPO) -> Dict[str, Any]:
    manifest = read_json(MANIFEST_PATH, canonical=False)
    _validate_manifest(manifest)
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(__file__))
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top_level_imports.intersection({"numpy", "torch", "transformers", "gr00t", "isaacgym"}))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in {"backward", "save_pretrained", "save_model", "save_checkpoint", "train", "get_action"}:
                forbidden_calls.append(name)
    observed = {}
    for relative, expected in FROZEN_FILES.items():
        path = Path(repo) / relative
        require(path.is_file() and not path.is_symlink(), "frozen file unavailable: %s" % path)
        digest = sha256_file(path)
        require(digest == expected, "frozen file changed: %s" % relative)
        observed[relative] = digest
    return {
        "schema": "CORELAB_PI0196_STATIC_CONTRACT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if not forbidden_imports and not forbidden_calls else "BLOCKED",
        "adapter_path": str(Path(__file__).relative_to(repo)),
        "adapter_sha256": sha256_file(Path(__file__)),
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(forbidden_calls),
        "frozen_files": observed,
        "stage43_runner_reused": True,
        "independent_native_loop": False,
        "model_calls_in_plan": 0,
        "simulator_steps_in_plan": 0,
        "training_in_plan": 0,
    }


def plan() -> Dict[str, Any]:
    return {
        "schema": "CORELAB_PI0196_STAGE43_REUSE_PLAN_V1",
        "status": "READY_NOT_EXECUTED",
        "admission_token": ADMISSION_TOKEN,
        "return_token": RETURN_TOKEN,
        "matrix": {
            "candidate_order": list(CANDIDATE_ORDER),
            "condition_order": list(CONDITION_ORDER),
            "cells": MAX_CELLS,
            "gr00t_forwards": MAX_FORWARDS,
            "reference_updates": MAX_UPDATES,
            "native_ticks": MAX_TICKS,
        },
        "cell_contract": {
            "raw_shape": [1, 40, 132],
            "decoded_shape": [40, 10],
            "gr00t_forwards": FORWARDS_PER_CELL,
            "reference_updates": UPDATES_PER_CELL,
            "native_tick_schedule": list(NATIVE_SCHEDULE),
            "model_side_metrics_from_same_q0_q3_output": True,
            "separate_model_only_inference": 0,
        },
        "ownership": {
            "checkpoint_identity_late_binding": True,
            "stage43_stage40_runtime": "delegated",
            "candidate_execution_loop": "stage43_corelab_initial_condition._run",
            "historical_stage43_files_modified": False,
        },
        "hard_zeroes": {
            "training": 0,
            "backward": 0,
            "optimizer": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "ICL": 0,
            "DAgger": 0,
            "new_visited_state_collection": 0,
            "hardware": 0,
            "checkpoint_writes": 0,
            "metric_dependent_retry": 0,
            "allocation_retry": 0,
        },
    }


def _verify_candidate_tree(candidate: Path, name: str, relay_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    spec = CANDIDATES[name]
    entries = _tree_entries(candidate, spec["tree_prefix"])
    actual_by_relative = {}
    for item in entries:
        relative = _relative_entry_path(str(item["path"]), spec["tree_prefix"])
        require(relative not in actual_by_relative, "duplicate candidate inventory entry: %s" % relative)
        actual_by_relative[relative] = item
    require(len(entries) == len(CHECKPOINT_FILES), "%s checkpoint file count drift: %d" % (name, len(entries)))
    require(set(actual_by_relative) == set(CHECKPOINT_FILES), "%s checkpoint extras/missing entries" % name)
    expected_entries = None
    if name in ("E", "S20"):
        expected_arm = relay_manifest.get("arms", {}).get(name)
        require(isinstance(expected_arm, dict), "relay manifest missing arm: %s" % name)
        expected_entries = {
            _relative_entry_path(str(item["path"]), spec["tree_prefix"]): item
            for item in expected_arm.get("checkpoint_entries", [])
        }
        require(expected_entries == actual_by_relative, "%s checkpoint file bytes/SHA drift" % name)
        require(expected_arm.get("checkpoint_tree_sha256") == spec["tree_sha256"], "%s relay tree identity drift" % name)
        require(expected_arm.get("global_step") == spec["global_step"], "%s relay global step drift" % name)
        require(expected_arm.get("training_result_sha256") == spec["training_result_sha256"], "%s result provenance drift" % name)
        require(expected_arm.get("training_invocation_sha256") == spec["training_invocation_sha256"], "%s invocation provenance drift" % name)
    else:
        require(set(actual_by_relative) == set(CHECKPOINT_FILES), "PARENT checkpoint closure drift")
    digest = _tree_digest(entries)
    require(digest == spec["tree_sha256"], "%s checkpoint tree SHA drift: %s" % (name, digest))
    stats_path = Path(candidate) / "experiment_cfg" / "structured_action_statistics.json"
    require(sha256_file(stats_path) == spec["stats_sha256"], "%s structured stats SHA drift" % name)
    return {
        "candidate": name,
        "label": spec["label"],
        "path": str(Path(candidate).resolve()),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "tree_sha256": digest,
        "global_step": spec["global_step"],
        "entries": entries,
        "structured_stats_sha256": spec["stats_sha256"],
        "training_result_sha256": spec["training_result_sha256"],
        "training_invocation_sha256": spec["training_invocation_sha256"],
        "symlink_closure": True,
        "extras_rejected": True,
    }


def _verify_bundle_paths(bundle: Mapping[str, Any]) -> None:
    files = {"structured_stats", "modality", "trajectory", "wbc_arm_checkpoint", "wbc_dog_checkpoint", "modality_config"}
    for key, item in bundle.get("artifacts", {}).items():
        require(isinstance(item, dict) and isinstance(item.get("path"), str), "Stage43 artifact path missing: %s" % key)
        _regular(Path(item["path"]), "Stage43 artifact " + key, key not in files)
        if key == "trajectory":
            require(sha256_file(Path(item["path"])) == TRAJECTORY_SHA256, "Stage43 trajectory SHA drift")


def _make_overlay(base: Mapping[str, Any], candidate: str, candidate_path: Path) -> Dict[str, Any]:
    spec = CANDIDATES[candidate]
    overlay = copy.deepcopy(dict(base))
    artifacts = overlay["artifacts"]
    stats_path = Path(candidate_path) / "experiment_cfg" / "structured_action_statistics.json"
    value = {"label": spec["label"], "path": str(Path(candidate_path).resolve()), "tree_sha256": spec["tree_sha256"]}
    artifacts["structured_checkpoint"] = dict(value)
    artifacts["model_path"] = dict(value)
    artifacts["structured_stats"] = {
        "path": str(stats_path.resolve()),
        "sha256": spec["stats_sha256"],
        "size_bytes": int(stats_path.stat().st_size),
    }
    overlay["stage45f_pi0196_overlay"] = {
        "schema": "CORELAB_PI0196_STAGE43_OVERLAY_V1",
        "base_bundle_sha256": STAGE43_BUNDLE_SHA256,
        "candidate": candidate,
        "allowlisted_changes": [
            "artifacts.structured_checkpoint",
            "artifacts.model_path",
            "artifacts.structured_stats",
        ],
        "historical_stage43_immutable": True,
    }
    return overlay


def _runtime_admission(
    binding: Mapping[str, Any],
    inventories: Mapping[str, Mapping[str, Any]],
    *,
    persist_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Import and validate the actual Stage43/Stage40 runtime, without GPU work."""

    paths = binding["paths"]
    bundle_path = _regular(Path(paths["stage43_bundle"]), "Stage43 bundle", False)
    base = read_json(bundle_path, canonical=True)
    require(sha256_file(bundle_path) == STAGE43_BUNDLE_SHA256, "Stage43 bundle SHA drift")
    # The sealed bundle was emitted at d042; accepted Stage43 HEAD b2a70 is
    # its subsequent lineage commit, not the byte-level bundle repository head.
    require(base.get("repository", {}).get("head") == STAGE43_BUNDLE_REPOSITORY_HEAD, "Stage43 bundle repository HEAD drift")
    _verify_bundle_paths(base)

    # These imports are intentionally inside admission.  They prove the
    # interpreter that will run science can import the native lane before any
    # worker, simulator, or checkpoint/model load is attempted.
    import stage40_corelab_phase
    import stage40_live_one_handoff
    import stage43_corelab_evidence as stage43_evidence
    import stage43_corelab_initial_condition as stage43_runner
    import go2arx5_vla.stage40_structured_decode as structured_decode

    stage43_evidence.verify_stage43_bundle_files(base, repo_root=REPO)
    overlay_records = []
    old_tree = stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256
    old_stats = stage43_evidence.STRUCTURED_STATS_SHA256
    old_decode_stats = structured_decode.EXPECTED_STRUCTURED_STATS_SHA256
    try:
        for candidate in CANDIDATE_ORDER:
            spec = CANDIDATES[candidate]
            stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = spec["tree_sha256"]
            stage43_evidence.STRUCTURED_STATS_SHA256 = spec["stats_sha256"]
            structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = spec["stats_sha256"]
            overlay = _make_overlay(base, candidate, Path(inventories[candidate]["path"]))
            stage43_evidence.validate_stage43_bundle(overlay)
            stage43_evidence.verify_stage43_bundle_files(overlay, repo_root=REPO)
            overlay_records.append({
                "candidate": candidate,
                "overlay_allowlist": overlay["stage45f_pi0196_overlay"]["allowlisted_changes"],
                "candidate_tree_sha256": spec["tree_sha256"],
                "candidate_stats_sha256": spec["stats_sha256"],
                "verified": True,
            })
            if persist_root is not None:
                write_json(Path(persist_root) / "overlays" / (candidate + ".json"), overlay)
    finally:
        stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = old_tree
        stage43_evidence.STRUCTURED_STATS_SHA256 = old_stats
        structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = old_decode_stats
    return {
        "schema": "CORELAB_PI0196_EXACT_RUNTIME_ADMISSION_V1",
        "status": "PASS_NO_MODEL_CALLS",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "imported_modules": {
            "stage40_corelab_phase": str(Path(stage40_corelab_phase.__file__).resolve()),
            "stage40_live_one_handoff": str(Path(stage40_live_one_handoff.__file__).resolve()),
            "stage43_corelab_evidence": str(Path(stage43_evidence.__file__).resolve()),
            "stage43_corelab_initial_condition": str(Path(stage43_runner.__file__).resolve()),
            "structured_decode": str(Path(structured_decode.__file__).resolve()),
        },
        "stage43_bundle_sha256": STAGE43_BUNDLE_SHA256,
        "overlay_records": overlay_records,
        "model_import_started": False,
        "model_loads": 0,
        "model_forwards": 0,
        "simulator_steps": 0,
        "wbc_steps": 0,
    }


def validate_binding(
    binding_path: Path,
    *,
    expected_head: Optional[str] = None,
    check_output_fresh: bool = True,
    exact_runtime: bool = False,
) -> Dict[str, Any]:
    binding = read_json(Path(binding_path), canonical=True)
    require(binding.get("schema") == "CORELAB_PI0196_RUNTIME_BINDING_V1", "PI0196 binding schema drift")
    require(binding.get("execution_authorized") is True, "PI0196 binding is not execution-authorized")
    checkout = binding.get("execution_checkout", {})
    current = _current_checkout()
    require(checkout.get("authorized_branch") == BRANCH and current["branch"] == BRANCH, "execution branch drift")
    require(isinstance(expected_head, str) and current["head"] == expected_head, "execution HEAD drift")
    require(checkout.get("authorized_head") == expected_head, "binding HEAD drift")
    require(current["clean"] and checkout.get("authorized_clean_worktree") is True, "worktree is not clean")
    paths = binding.get("paths", {})
    for key in ("stage43_bundle", "relay_manifest", "model_python", "output_root"):
        require(isinstance(paths.get(key), str) and paths[key], "binding path missing: %s" % key)
    if check_output_fresh:
        require(not Path(paths["output_root"]).exists(), "output root is not fresh: %s" % paths["output_root"])
    relay = read_json(_regular(Path(paths["relay_manifest"]), "relay manifest", False), canonical=True)
    require(relay.get("schema") == "VANDA_PI0191_RELAY_MANIFEST_V1" and relay.get("status") == "PASS", "relay manifest is not PASS")
    candidate_paths = binding.get("candidate_paths", {})
    require(set(candidate_paths) == set(CANDIDATE_ORDER), "candidate path set drift")
    inventories = {}
    for candidate in CANDIDATE_ORDER:
        inventories[candidate] = _verify_candidate_tree(Path(candidate_paths[candidate]), candidate, relay)
    _regular(Path(paths["model_python"]), "model Python", False)
    result = {
        "schema": "CORELAB_PI0196_BINDING_ADMISSION_V1",
        "status": "PASS_CPU_PROVENANCE_NOT_EXECUTED",
        "binding_path": str(Path(binding_path).resolve()),
        "binding_sha256": sha256_file(Path(binding_path)),
        "checkout": current,
        "inventories": inventories,
        "candidate_paths": {key: str(Path(value).resolve()) for key, value in candidate_paths.items()},
        "stage43_bundle_path": str(Path(paths["stage43_bundle"]).resolve()),
        "output_root": str(Path(paths["output_root"]).resolve()),
        "model_calls": 0,
        "simulator_steps": 0,
        "wbc_steps": 0,
    }
    if exact_runtime:
        result["runtime"] = _runtime_admission(binding, inventories)
    return result


def metric_regression() -> Dict[str, Any]:
    """Behavioral proof that path length is not tracking RMS."""

    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

    positions = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)
    desired = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.4, 0.0]], dtype=np.float64)
    target = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
    canonical = trajectory_metrics(positions, target, prefix="pi0196.synthetic")
    tracking_rms = float(np.sqrt(np.mean(np.sum((positions - desired) ** 2, axis=1))))
    require(abs(float(canonical["path_length_m"]) - 1.0) < 1.0e-12, "synthetic path length regression failed")
    require(abs(tracking_rms - 0.1) < 1.0e-12, "synthetic tracking RMS regression failed")
    require(abs(float(canonical["path_length_m"]) - tracking_rms) > 0.5, "path length was aliased to tracking RMS")
    require(canonical["roughness_position_second_difference_m"]["mean"] > 0.0, "synthetic roughness regression failed")
    return {
        "schema": "CORELAB_PI0196_METRIC_REGRESSION_V1",
        "status": "PASS",
        "achieved_path_length_m": canonical["path_length_m"],
        "achieved_roughness_position_second_difference_m": canonical["roughness_position_second_difference_m"],
        "tracking_rms_m": tracking_rms,
        "path_length_not_tracking_rms": True,
    }


def _numeric_summary(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _condition_rows(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    setup = read_json(root / "condition_setup.json")
    event = setup.get("event", setup)
    rows: List[Dict[str, Any]] = []
    for filename in ("rows_before_handoff.json", "rows_after_handoff.json"):
        payload = read_json(root / filename)
        for segment in payload.get("segments", []):
            require(isinstance(segment, list), "Stage43 row segment is not a list")
            rows.extend(dict(row) for row in segment)
    return event, rows


def _derive_achieved_metrics(root: Path) -> Dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

    event, rows = _condition_rows(root)
    initial = np.asarray(event["s0"]["eef_world_m"], dtype=np.float64)
    achieved = np.asarray([row["achieved_pos"] for row in rows], dtype=np.float64)
    desired = np.asarray([row["desired_pos"] for row in rows], dtype=np.float64)
    require(achieved.shape == desired.shape and achieved.ndim == 2 and achieved.shape[1] == 3, "achieved/desired EEF row ABI drift")
    path = np.vstack((initial[None, :], achieved))
    target_metrics = trajectory_metrics(path, np.asarray(TARGET_WORLD, dtype=np.float64), prefix="pi0196.achieved")
    errors = achieved - desired
    tracking = np.linalg.norm(errors, axis=1)
    require(np.isfinite(tracking).all(), "tracking metrics are non-finite")
    return {
        "initial_target_distance_m": target_metrics["initial_target_distance_m"],
        "minimum_target_distance_m": target_metrics["minimum_target_distance_m"],
        "final_target_distance_m": target_metrics["final_target_distance_m"],
        "absolute_progress_m": target_metrics["absolute_progress_m"],
        "best_progress_m": target_metrics["best_progress_m"],
        "progress_sign": target_metrics["progress_sign"],
        "achieved_path_length_m": target_metrics["path_length_m"],
        "achieved_roughness_position_second_difference_m": target_metrics["roughness_position_second_difference_m"],
        "tracking_rms_m": float(np.sqrt(np.mean(tracking * tracking))),
        "tracking_max_m": float(np.max(tracking)),
        "sample_count_including_S0": target_metrics["sample_count"],
        "eef_path_source": "condition_setup.s0 + Stage43 achieved_pos rows",
        "tracking_source": "Stage43 achieved_pos - desired_pos rows",
    }


def _derive_model_metrics(root: Path) -> List[Dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

    bind = read_json(root / "initial_bind.json")
    events = bind.get("events", [])
    require(events and isinstance(events[0].get("anchor_world_m"), list), "Stage43 anchor evidence missing")
    anchor = np.asarray(events[0]["anchor_world_m"], dtype=np.float64)
    raw_npz = np.load(str(root / "raw_action_d132.npz"), allow_pickle=False)
    decoded_npz = np.load(str(root / "decoded_action_d10_h40.npz"), allow_pickle=False)
    records = []
    try:
        require(sorted(raw_npz.files) == sorted(decoded_npz.files), "raw/decoded query key mismatch")
        for key in sorted(decoded_npz.files):
            raw = np.asarray(raw_npz[key])
            decoded = np.asarray(decoded_npz[key], dtype=np.float64)
            require(tuple(raw.shape) == (1, 40, 132) and np.isfinite(raw).all(), "raw model ABI drift: %s" % key)
            require(tuple(decoded.shape) == (40, 10) and np.isfinite(decoded).all(), "decoded model ABI drift: %s" % key)
            predicted_world = decoded[:, :3] + anchor[None, :]
            metrics = trajectory_metrics(predicted_world, np.asarray(TARGET_WORLD, dtype=np.float64), prefix="pi0196.model." + key)
            records.append({
                "request_id": key,
                "raw_shape": list(raw.shape),
                "raw_dtype": str(raw.dtype),
                "raw_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                "decoded_shape": list(decoded.shape),
                "decoded_dtype": str(decoded.dtype),
                "decoded_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
                "predicted_endpoint_target_distance_m": metrics["final_target_distance_m"],
                "predicted_target_progress_m": metrics["absolute_progress_m"],
                "predicted_path_length_m": metrics["path_length_m"],
                "predicted_roughness_position_second_difference_m": metrics["roughness_position_second_difference_m"],
                "target_xyz_in_model_input": False,
            })
    finally:
        raw_npz.close()
        decoded_npz.close()
    return records


def _classify_failure(result: Mapping[str, Any]) -> str:
    error = result.get("error") or {}
    message = str(error.get("message", ""))
    if "SAFETYGATE_REJECT" in message:
        return "SCIENTIFIC_REJECT"
    if "termination" in message.lower() or "WBC" in message.upper() or "NATIVE_TASK_TERMINATION" in message:
        return "SCIENTIFIC_ABORT"
    return "ENGINEERING_BLOCKED"


def _cell_result(root: Path, candidate: str, condition: str, runner_code: int) -> Dict[str, Any]:
    stage43_result = read_json(root / "condition_result.json")
    stage43_status = str(stage43_result.get("status"))
    if stage43_status == "PASS":
        status = "PASS"
        metrics = _derive_achieved_metrics(root)
        model_metrics = _derive_model_metrics(root)
    else:
        status = _classify_failure(stage43_result)
        metrics = None
        model_metrics = []
    payload = {
        "schema": CELL_SCHEMA,
        "status": status,
        "candidate": candidate,
        "condition_id": condition,
        "stage43_runner_status": stage43_status,
        "stage43_runner_return_code": int(runner_code),
        "stage43_condition_result_sha256": sha256_file(root / "condition_result.json"),
        "cases": int(stage43_result.get("cases", stage43_result.get("gr00t_forward_count", 0))),
        "model_worker_started": bool((root / "worker.log").is_file() or stage43_result.get("cases", 0)),
        "gr00t_forward_count": int(stage43_result.get("gr00t_forward_count", 0)),
        "reference_update_count": int(stage43_result.get("reference_update_count", 0)),
        "wbc_step_count": int(stage43_result.get("wbc_step_count", 0)),
        "metrics": metrics,
        "model_metrics": model_metrics,
        "error": stage43_result.get("error"),
    }
    write_json(root / "pi0196_cell_result.json", payload)
    return payload


def _aggregate_candidate(root: Path, candidate: str) -> Dict[str, Any]:
    cells = [read_json(root / condition / "pi0196_cell_result.json") for condition in CONDITION_ORDER if (root / condition / "pi0196_cell_result.json").is_file()]
    actual = [cell["metrics"] for cell in cells if isinstance(cell.get("metrics"), dict)]
    statuses = [str(cell.get("status")) for cell in cells]
    def values(key: str) -> List[float]:
        return [float(item[key]) for item in actual if item.get(key) is not None]
    def roughness_values() -> List[float]:
        return [float(item["achieved_roughness_position_second_difference_m"]["mean"]) for item in actual if item.get("achieved_roughness_position_second_difference_m", {}).get("mean") is not None]
    return {
        "candidate": candidate,
        "label": CANDIDATES[candidate]["label"],
        "cell_count": len(cells),
        "complete_cell_count": sum(status == "PASS" for status in statuses),
        "scientific_reject_count": sum(status == "SCIENTIFIC_REJECT" for status in statuses),
        "scientific_abort_count": sum(status == "SCIENTIFIC_ABORT" for status in statuses),
        "engineering_blocked_count": sum(status == "ENGINEERING_BLOCKED" for status in statuses),
        "model_worker_started_count": sum(bool(cell.get("model_worker_started")) for cell in cells),
        "model_forward_cell_count": sum(int(cell.get("cases", 0)) > 0 for cell in cells),
        "gr00t_forward_count": sum(int(cell.get("gr00t_forward_count", 0)) for cell in cells),
        "reference_update_count": sum(int(cell.get("reference_update_count", 0)) for cell in cells),
        "wbc_step_count": sum(int(cell.get("wbc_step_count", 0)) for cell in cells),
        "mean_progress_m": _numeric_summary(values("absolute_progress_m")),
        "mean_final_distance_m": _numeric_summary(values("final_target_distance_m")),
        "mean_minimum_distance_m": _numeric_summary(values("minimum_target_distance_m")),
        "mean_achieved_path_length_m": _numeric_summary(values("achieved_path_length_m")),
        "mean_achieved_roughness_position_second_difference_m": _numeric_summary(roughness_values()),
        "mean_tracking_rms_m": _numeric_summary(values("tracking_rms_m")),
        "mean_tracking_max_m": _numeric_summary(values("tracking_max_m")),
        "per_condition": {
            condition: {
                "status": next((cell.get("status") for cell in cells if cell.get("condition_id") == condition), "NOT_STARTED"),
                "final_target_distance_m": next((cell.get("metrics", {}).get("final_target_distance_m") for cell in cells if cell.get("condition_id") == condition and isinstance(cell.get("metrics"), dict)), None),
                "absolute_progress_m": next((cell.get("metrics", {}).get("absolute_progress_m") for cell in cells if cell.get("condition_id") == condition and isinstance(cell.get("metrics"), dict)), None),
            }
            for condition in CONDITION_ORDER
        },
        "cells": cells,
    }


def _classify_scorecard(aggregates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    e = aggregates["E"]
    s20 = aggregates["S20"]
    complete = all(
        item.get("complete_cell_count") == len(CONDITION_ORDER)
        and item.get("scientific_reject_count") == 0
        and item.get("scientific_abort_count") == 0
        and item.get("engineering_blocked_count") == 0
        for item in (e, s20)
    )
    wins = None
    e_wins = None
    if not complete:
        classification = "MIXED_FOR_PI_REVIEW"
        basis = "E/S20 matrix contains an incomplete or rejected/aborted cell"
    else:
        e_progress = float(e["mean_progress_m"]["mean"])
        s20_progress = float(s20["mean_progress_m"]["mean"])
        e_final = float(e["mean_final_distance_m"]["mean"])
        s20_final = float(s20["mean_final_distance_m"]["mean"])
        wins = sum(float(s20["per_condition"][condition]["final_target_distance_m"]) < float(e["per_condition"][condition]["final_target_distance_m"]) for condition in CONDITION_ORDER)
        e_wins = sum(float(e["per_condition"][condition]["final_target_distance_m"]) < float(s20["per_condition"][condition]["final_target_distance_m"]) for condition in CONDITION_ORDER)
        if s20_progress > e_progress and s20_final < e_final and wins >= 3:
            classification = "STATE_SUPPORT_SUPPORTED"
            basis = "S20 wins both coupled aggregate endpoint criteria and at least three matched conditions"
        elif e_progress > s20_progress and e_final < s20_final and e_wins >= 3:
            classification = "STATE_SUPPORT_NOT_SUPPORTED"
            basis = "E wins both coupled aggregate endpoint criteria and at least three matched conditions"
        else:
            classification = "MIXED_FOR_PI_REVIEW"
            basis = "frozen aggregate and matched-condition rule did not select one arm"
    return {
        "schema": "CORELAB_PI0196_SCORECARD_V1",
        "classification": classification,
        "basis": basis,
        "PARENT": aggregates["PARENT"],
        "E": e,
        "S20": s20,
        "matched_condition_s20_final_distance_wins": wins,
        "matched_condition_E_final_distance_wins": e_wins,
        "safety_gate_rejects_in_E": e.get("scientific_reject_count"),
        "safety_gate_rejects_in_S20": s20.get("scientific_reject_count"),
        "wbc_aborts_in_E": e.get("scientific_abort_count"),
        "wbc_aborts_in_S20": s20.get("scientific_abort_count"),
    }


def _gpu_uuid_snapshot(requested_uuid: str) -> Dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, "nvidia-smi failed: %s" % completed.stderr[-1000:])
    rows = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if not line.strip():
            continue
        require(len(fields) == 5, "malformed nvidia-smi row")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_free_mib": int(fields[4])})
    selected = [row for row in rows if row["uuid"] == requested_uuid]
    selected_name = selected[0]["name"] if len(selected) == 1 else ""
    allowed_name = "RTX 4080 SUPER" in selected_name or selected_name == "NVIDIA A40"
    require(len(selected) == 1 and allowed_name and selected[0]["memory_total_mib"] >= 30000, "requested RTX4080S/A40 is not uniquely present")
    return {"rows": rows, "selected": selected[0], "query_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest()}


def _write_evidence_manifest(root: Path) -> str:
    """Hash every produced raw/derived file before the final result seal."""

    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {"evidence_manifest.json", "result_seal.json"}:
            continue
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "evidence tree contains a symlink: %s" % path)
        if stat.S_ISREG(mode):
            files.append({
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            })
        elif not stat.S_ISDIR(mode):
            raise AdapterError("evidence tree contains a non-regular entry: %s" % path)
    return write_json(root / "evidence_manifest.json", {
        "schema": "CORELAB_PI0196_EVIDENCE_MANIFEST_V1",
        "status": "PASS",
        "root": str(root),
        "excluded_from_manifest": ["evidence_manifest.json", "result_seal.json"],
        "file_count": len(files),
        "files": files,
    })


def execute(binding_path: Path, *, expected_head: str, gpu_uuid: str, sim_device: str, seed: int) -> int:
    require(seed == DEV_SEED, "PI0196 seed is fixed at 4242")
    binding = read_json(Path(binding_path), canonical=True)
    admission = validate_binding(binding_path, expected_head=expected_head, check_output_fresh=True, exact_runtime=True)
    output_root = Path(binding["paths"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    admission["gpu"] = _gpu_uuid_snapshot(gpu_uuid)
    write_json(output_root / "admission.json", admission)

    import stage43_corelab_evidence as stage43_evidence
    import stage43_corelab_initial_condition as stage43_runner
    import go2arx5_vla.stage40_structured_decode as structured_decode

    base = read_json(Path(binding["paths"]["stage43_bundle"]), canonical=True)
    base_head = base["repository"]["head"]
    aggregates = {}
    execution_status = "COMPLETED"
    stop_reason = None
    old_tree = stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256
    old_stats = stage43_evidence.STRUCTURED_STATS_SHA256
    old_decode_stats = structured_decode.EXPECTED_STRUCTURED_STATS_SHA256
    try:
        for candidate in CANDIDATE_ORDER:
            candidate_root = output_root / candidate
            candidate_root.mkdir()
            spec = CANDIDATES[candidate]
            stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = spec["tree_sha256"]
            stage43_evidence.STRUCTURED_STATS_SHA256 = spec["stats_sha256"]
            structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = spec["stats_sha256"]
            overlay = _make_overlay(base, candidate, Path(binding["candidate_paths"][candidate]))
            overlay_path = output_root / "overlays" / (candidate + ".json")
            write_json(overlay_path, overlay)
            for condition in CONDITION_ORDER:
                condition_root = candidate_root / condition
                run_args = argparse.Namespace(
                    bundle=overlay_path,
                    output_dir=condition_root,
                    condition_id=condition,
                    repository_head=base_head,
                    gpu_uuid=gpu_uuid,
                    source_root=Path(base["artifacts"]["wbc_source"]["path"]),
                    trajectory=Path(base["artifacts"]["trajectory"]["path"]),
                    arm=Path(base["artifacts"]["wbc_arm_checkpoint"]["path"]),
                    dog=Path(base["artifacts"]["wbc_dog_checkpoint"]["path"]),
                    stage10_root=Path(base["artifacts"]["stage10_root"]["path"]),
                    base_model_path=Path(base["artifacts"]["base_model_path"]["path"]),
                    model_path=Path(binding["candidate_paths"][candidate]),
                    dataset_dir=Path(base["artifacts"]["dataset_dir"]["path"]),
                    modality_config=Path(base["artifacts"]["modality_config"]["path"]),
                    cosmos_snapshot=Path(base["artifacts"]["cosmos_snapshot_path"]["path"]),
                    structured_stats=Path(binding["candidate_paths"][candidate]) / "experiment_cfg" / "structured_action_statistics.json",
                    model_python=Path(binding["paths"]["model_python"]),
                    wbc_source_root=Path(base["artifacts"]["wbc_source"]["path"]),
                    sim_device=sim_device,
                    seed=seed,
                    response_timeout_s=300.0,
                    worker_ready_timeout_s=1200.0,
                )
                try:
                    runner_code = int(stage43_runner._run(run_args))
                except Exception as exc:
                    # Preserve an engineering failure as a cell record even if
                    # the mature runner fails before it can serialize its own
                    # pointer.  This branch never retries or fabricates science.
                    condition_root.mkdir(parents=True, exist_ok=True)
                    pointer = condition_root / "condition_result.json"
                    if not pointer.exists():
                        write_json(
                            pointer,
                            {
                                "schema": "CORELAB_STAGE43_CONDITION_RESULT_V1",
                                "status": "BLOCKED",
                                "cases": 0,
                                "gr00t_forward_count": 0,
                                "reference_update_count": 0,
                                "wbc_step_count": 0,
                                "error": {"type": type(exc).__name__, "message": str(exc)},
                            },
                        )
                    runner_code = 2
                cell = _cell_result(condition_root, candidate, condition, runner_code)
                if cell["status"] != "PASS":
                    execution_status = "PARTIAL" if cell["status"] in ("SCIENTIFIC_REJECT", "SCIENTIFIC_ABORT") else "BLOCKED"
                    stop_reason = {"candidate": candidate, "condition": condition, "status": cell["status"]}
                    break
            aggregates[candidate] = _aggregate_candidate(candidate_root, candidate)
            if stop_reason is not None:
                break
    finally:
        stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = old_tree
        stage43_evidence.STRUCTURED_STATS_SHA256 = old_stats
        structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = old_decode_stats
    for candidate in CANDIDATE_ORDER:
        if candidate not in aggregates:
            aggregates[candidate] = _aggregate_candidate(output_root / candidate, candidate) if (output_root / candidate).exists() else _aggregate_candidate(output_root / candidate, candidate)
    scorecard = _classify_scorecard(aggregates)
    scorecard["execution_status"] = execution_status
    scorecard["stop_reason"] = stop_reason
    write_json(output_root / "scorecard.json", scorecard)
    evidence_manifest_sha = _write_evidence_manifest(output_root)
    seal = {
        "schema": "CORELAB_PI0196_RESULT_SEAL_V1",
        "status": execution_status,
        "token": RETURN_TOKEN + " — " + scorecard["classification"],
        "execution_head": expected_head,
        "gpu": admission["gpu"],
        "scorecard_sha256": sha256_file(output_root / "scorecard.json"),
        "admission_sha256": sha256_file(output_root / "admission.json"),
        "evidence_manifest_sha256": evidence_manifest_sha,
        "counters": {
            "model_loads": sum(int(candidate.get("model_worker_started_count", 0)) for candidate in aggregates.values()),
            "model_forward_cells": sum(int(candidate.get("model_forward_cell_count", 0)) for candidate in aggregates.values()),
            "gr00t_forwards": sum(int(candidate.get("gr00t_forward_count", 0)) for candidate in aggregates.values()),
            "reference_updates": sum(int(candidate.get("reference_update_count", 0)) for candidate in aggregates.values()),
            "native_ticks": sum(int(candidate.get("wbc_step_count", 0)) for candidate in aggregates.values()),
            "training": 0,
            "backward": 0,
            "optimizer": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "hardware": 0,
        },
        "stop_reason": stop_reason,
    }
    write_json(output_root / "result_seal.json", seal)
    return 0 if execution_status == "COMPLETED" else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--metric-regression", action="store_true")
    parser.add_argument("--verify-binding", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    args = parser.parse_args(argv)
    selected = sum(bool(value) for value in (args.plan, args.static_check, args.metric_regression, args.verify_binding, args.execute))
    if selected != 1:
        parser.error("select exactly one mode")
    if args.plan:
        print(json.dumps(plan(), sort_keys=True), flush=True)
        return 0
    if args.static_check:
        print(json.dumps(static_contract(), sort_keys=True), flush=True)
        return 0
    if args.metric_regression:
        print(json.dumps(metric_regression(), sort_keys=True), flush=True)
        return 0
    require(args.expected_head is not None, "--expected-head is required")
    if args.verify_binding:
        result = validate_binding(args.verify_binding, expected_head=args.expected_head, check_output_fresh=True, exact_runtime=True)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    require(args.execute is not None and args.gpu_uuid is not None, "--execute requires --gpu-uuid")
    return execute(args.execute, expected_head=args.expected_head, gpu_uuid=args.gpu_uuid, sim_device=args.sim_device, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
