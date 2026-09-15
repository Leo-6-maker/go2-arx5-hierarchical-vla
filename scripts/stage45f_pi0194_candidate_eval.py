#!/usr/bin/env python3
"""PI0194 late-bound Stage43 candidate closed-loop evaluation.

The historical Stage43 bundle is an immutable external authority.  This file
only binds one of the three already-sealed candidate checkpoints to that
bundle's unchanged scene, trajectory, observation, timing, decoder, SafetyGate
and native WBC helpers.  CPU-only plan/admission paths deliberately use only
the Python standard library.  NumPy, IsaacGym and the GR00T worker are loaded
only from the guarded ``--execute`` path after every binding check has passed.

The execution matrix is PARENT, E, S20 x C0..C4.  A SafetyGate reject or native
WBC abort is a sealed scientific cell outcome; it is never retried.  Runtime
or provenance failures are engineering failures and stop the matrix while
preserving the evidence already written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts"
MANIFEST_PATH = REPO / "manifests" / "STAGE45F_PI0194_CANDIDATE_EVAL_V1.json"
SCHEMA = "CORELAB_STAGE45F_PI0194_CANDIDATE_EVAL_V1"
CELL_SCHEMA = "CORELAB_STAGE45F_PI0194_CANDIDATE_CELL_V1"
RETURN_TOKEN = "VANDA_PI0194_E_S20_CLOSED_LOOP_SCORECARD"

STAGE43_ACCEPTED_HEAD = "b2a70dcec0765cf69819bb76c180e5c92ae3eac3"
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
CONTAINER_MODEL_PYTHON = "/opt/miniconda3/envs/isaacgym/bin/python"
CANDIDATE_ORDER = ("PARENT", "E", "S20")
CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
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
MAX_FORWARDS = 60
MAX_UPDATES = 45
MAX_TICKS = 3675

CANDIDATES = {
    "PARENT": {
        "label": "PARENT L1(seed2223)@2000",
        "tree_sha256": "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16",
        "tree_prefix": "stage45e_l1/checkpoint-2000",
        "global_step": 2000,
        "arm_manifest": None,
    },
    "E": {
        "label": "E@256",
        "tree_sha256": "f91f4fd3cf3c440127c57d49cdd2dde52ce6a1294bd17a05710fba1026b9361b",
        "tree_prefix": "617089/E/checkpoint-256",
        "global_step": 256,
        "arm_manifest": "E",
        "sequence_sha256": "06297d681228c4192f708a73361a0bc1c1782021d20cbd6cfb67c19088f2dcda",
    },
    "S20": {
        "label": "S20@256",
        "tree_sha256": "64c082022bb6786efa0d8f706b33e36f6bfca8c4af13db41664dccff697c882d",
        "tree_prefix": "617089/S20/checkpoint-256",
        "global_step": 256,
        "arm_manifest": "S20",
        "sequence_sha256": "d30c73f50668c243078e3b70001a53989758ed33a45d228dc2a1801d07e97f2d",
    },
}

REQUIRED_PATHS = {
    "stage10_root",
    "base_model_path",
    "structured_checkpoint",
    "structured_stats",
    "dataset_dir",
    "modality_config",
    "cosmos_snapshot",
    "trajectory",
    "wbc_source_root",
    "wbc_arm",
    "wbc_dog",
    "model_python",
    "relay_manifest",
    "output_root",
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


class AdapterError(RuntimeError):
    """Fail-closed contract or provenance error."""


class ScientificCellFailure(RuntimeError):
    """A frozen cell outcome, not an engineering retry condition."""

    def __init__(self, classification: str, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _read_json(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    require(candidate.is_file() and not candidate.is_symlink(), f"JSON is not a regular file: {candidate}")
    value = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {candidate}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_checkout() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)
        require(completed.returncode == 0, f"git identity command failed: {' '.join(args)}")
        return completed.stdout.strip()

    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "clean": git("status", "--porcelain=v1", "--untracked-files=all") == "",
    }


def _regular(path: Path, kind: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise AdapterError(f"{kind} is unavailable: {candidate}") from exc
    if kind.endswith("directory") or kind.endswith("root") or kind in {"checkpoint", "candidate", "runtime"}:
        require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{kind} is not a regular directory: {candidate}")
    else:
        require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{kind} is not a regular file: {candidate}")
    return candidate.resolve()


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    candidate = _regular(Path(root), "checkpoint")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(candidate, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"symlink or non-directory checkpoint entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"symlink or non-file checkpoint entry: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(candidate).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_static_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema") == SCHEMA, "PI0194 manifest schema drift")
    require(manifest.get("status") == "READY_FOR_PI_REVIEW_NOT_EXECUTED", "PI0194 manifest is not static-only")
    require(manifest.get("return_token") == RETURN_TOKEN, "PI0194 return token drift")
    require(manifest.get("authority", {}).get("issue_11_comment") == 5677804074, "PI0194 Issue authority drift")
    require(manifest.get("authority", {}).get("pr_75_comment") == 5677809759, "PI0194 PR authority drift")
    task = manifest.get("task", {})
    require(task.get("dev_seed") == DEV_SEED and task.get("target_world_m") == list(TARGET_WORLD), "PI0194 task identity drift")
    require(task.get("language") == LANGUAGE and task.get("source_frame") == SOURCE_FRAME and task.get("tcp_semantic_id") == TCP_ID, "PI0194 language/frame/TCP drift")
    require(task.get("target_xyz_in_model_input") is False and task.get("action_contract") == ACTION_CONTRACT, "PI0194 model-input contract drift")
    matrix = manifest.get("matrix", {})
    require(tuple(matrix.get("candidate_order", ())) == CANDIDATE_ORDER, "PI0194 candidate order drift")
    require(tuple(matrix.get("condition_order", ())) == CONDITION_ORDER, "PI0194 condition order drift")
    require(matrix.get("condition_deltas_m") == {key: list(value) for key, value in CONDITION_DELTAS.items()}, "PI0194 condition delta drift")
    require(matrix.get("gr00t_forwards_per_cell") == FORWARDS_PER_CELL and matrix.get("reference_updates_per_cell") == UPDATES_PER_CELL, "PI0194 cell budget drift")
    require(matrix.get("native_ticks_per_cell") == list(NATIVE_SCHEDULE), "PI0194 native tick schedule drift")
    require(matrix.get("maximum_gr00t_forwards") == MAX_FORWARDS and matrix.get("maximum_reference_updates") == MAX_UPDATES and matrix.get("maximum_native_ticks") == MAX_TICKS, "PI0194 total budget drift")
    for candidate in CANDIDATE_ORDER:
        item = manifest.get("candidates", {}).get(candidate, {})
        require(item.get("tree_sha256") == CANDIDATES[candidate]["tree_sha256"], f"{candidate} tree identity drift")
        require(item.get("global_step") == CANDIDATES[candidate]["global_step"], f"{candidate} step identity drift")
    require(manifest.get("historical_stage43", {}).get("bundle_sha256") == STAGE43_BUNDLE_SHA256, "Stage43 bundle authority drift")
    require(manifest.get("historical_stage43", {}).get("result_sha256") == STAGE43_RESULT_SHA256, "Stage43 result authority drift")
    require(manifest.get("historical_stage43", {}).get("result_seal_sha256") == STAGE43_RESULT_SEAL_SHA256, "Stage43 seal authority drift")
    require(manifest.get("historical_stage43", {}).get("cohort_manifest_sha256") == STAGE43_COHORT_SHA256, "Stage43 cohort authority drift")
    require(manifest.get("historical_stage43", {}).get("trajectory_sha256") == TRAJECTORY_SHA256, "Stage43 trajectory authority drift")
    require(manifest.get("frozen_files") == FROZEN_FILES, "frozen source hash map drift")


def static_contract(repo: Path = REPO) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    _validate_static_manifest(manifest)
    template = _read_json(repo / "manifests" / "STAGE45F_PI0194_RUNTIME_BINDING_TEMPLATE_V1.json")
    require(template.get("schema") == "CORELAB_STAGE45F_PI0194_RUNTIME_BINDING_V1" and template.get("status") == "LATE_BOUND_NOT_EXECUTED", "runtime binding template is executable")
    require(template.get("execution_authorized") is False, "runtime binding template has execution authority")
    require(all(value is None for value in template.get("execution_checkout", {}).values() if value is not True), "runtime binding template contains checkout identity")
    require(all(value is None for value in template.get("paths", {}).values()), "runtime binding template contains a node-local path")
    require(all(value is None for value in template.get("candidate_paths", {}).values()), "runtime binding template contains a candidate path")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(__file__))
    top_level_imports: set[str] = set()
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
    observed_frozen: dict[str, str] = {}
    for relative, expected in FROZEN_FILES.items():
        path = Path(repo) / relative
        require(path.is_file() and not path.is_symlink(), f"frozen file unavailable: {path}")
        observed = sha256_file(path)
        require(observed == expected, f"frozen file changed: {relative}: {observed}")
        observed_frozen[relative] = observed
    return {
        "schema": "CORELAB_STAGE45F_PI0194_STATIC_CONTRACT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if not forbidden_imports and not forbidden_calls else "BLOCKED",
        "adapter_path": str(Path(__file__).relative_to(repo)),
        "adapter_sha256": sha256_file(Path(__file__)),
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(forbidden_calls),
        "frozen_files": observed_frozen,
        "historical_stage43_immutable": True,
        "model_calls_in_plan": 0,
        "simulator_steps_in_plan": 0,
        "training_in_plan": 0,
    }


def plan() -> dict[str, Any]:
    return {
        "schema": "CORELAB_STAGE45F_PI0194_CANDIDATE_EVAL_PLAN_V1",
        "status": "READY_NOT_EXECUTED",
        "return_token": RETURN_TOKEN,
        "historical_stage43": {
            "accepted_head": STAGE43_ACCEPTED_HEAD,
            "bundle_sha256": STAGE43_BUNDLE_SHA256,
            "result_sha256": STAGE43_RESULT_SHA256,
            "result_seal_sha256": STAGE43_RESULT_SEAL_SHA256,
            "cohort_manifest_sha256": STAGE43_COHORT_SHA256,
            "trajectory_sha256": TRAJECTORY_SHA256,
            "late_binding": "candidate checkpoint only; scene/trajectory/condition/cameras/timing/WBC unchanged",
        },
        "matrix": {
            "candidate_order": list(CANDIDATE_ORDER),
            "condition_order": list(CONDITION_ORDER),
            "cells": len(CANDIDATE_ORDER) * len(CONDITION_ORDER),
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
        "candidates": {key: {item: value for item, value in spec.items() if item != "arm_manifest"} for key, spec in CANDIDATES.items()},
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


def _verify_candidate_tree(candidate: Path, name: str, relay_manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = CANDIDATES[name]
    candidate = _regular(candidate, "candidate")
    actual_entries = _tree_entries(candidate, spec["tree_prefix"])
    actual_relative = sorted(str(item["path"]).removeprefix(spec["tree_prefix"] + "/") for item in actual_entries)
    require(actual_relative == list(CHECKPOINT_FILES), f"{name} checkpoint file closure has extra or missing files")
    if spec["arm_manifest"] is not None:
        arm = str(spec["arm_manifest"])
        arm_payload = relay_manifest.get("arms", {}).get(arm)
        require(isinstance(arm_payload, dict), f"relay manifest has no {arm} arm")
        expected_entries = arm_payload.get("checkpoint_entries")
        require(isinstance(expected_entries, list), f"relay manifest {arm} entries missing")
        expected_by_relative: dict[str, Mapping[str, Any]] = {}
        prefix = f"617089/{arm}/checkpoint-256/"
        for entry in expected_entries:
            path = str(entry.get("path", ""))
            require(path.startswith(prefix), f"relay manifest path outside {arm} checkpoint: {path}")
            relative = path[len(prefix):]
            require(relative not in expected_by_relative, f"relay manifest duplicate: {path}")
            expected_by_relative[relative] = entry
        require(sorted(expected_by_relative) == list(CHECKPOINT_FILES), f"relay manifest {arm} file closure drift")
        observed_entries: list[dict[str, Any]] = []
        for relative in CHECKPOINT_FILES:
            expected = expected_by_relative[relative]
            path = candidate / relative
            observed = {"path": f"{prefix}{relative}", "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
            require(observed["bytes"] == int(expected["bytes"]) and observed["sha256"] == str(expected["sha256"]), f"{name} file identity drift: {relative}")
            observed_entries.append(observed)
        tree_sha = _tree_digest(observed_entries)
        require(tree_sha == spec["tree_sha256"] and tree_sha == arm_payload.get("checkpoint_tree_sha256"), f"{name} checkpoint tree drift: {tree_sha}")
        training_result = _read_json(candidate.parent / "training_result.json")
        training_invocation = _read_json(candidate.parent / "training_invocation.json")
        require(training_result.get("status") == "PASS" and training_result.get("global_step") == 256 and training_result.get("optimizer_steps") == 256, f"{name} training result is not sealed 256/256")
        require(training_result.get("model_forward") == 256 and training_result.get("backward") == 256, f"{name} training counters drift")
        require(training_result.get("sequence_sha256") == spec["sequence_sha256"], f"{name} frozen sequence drift")
        require(training_invocation.get("arm") == arm and training_invocation.get("base_seed") == 2223, f"{name} invocation identity drift")
        training_result_sha = sha256_file(candidate.parent / "training_result.json")
        invocation_sha = sha256_file(candidate.parent / "training_invocation.json")
    else:
        tree_sha = _tree_digest(actual_entries)
        require(tree_sha == spec["tree_sha256"], f"{name} checkpoint tree drift: {tree_sha}")
        trainer_state = _read_json(candidate / "trainer_state.json")
        require(trainer_state.get("global_step") == 2000, "PARENT trainer global_step drift")
        training_result_sha = None
        invocation_sha = None
    return {
        "candidate": name,
        "label": spec["label"],
        "path": str(candidate),
        "tree_prefix": spec["tree_prefix"],
        "tree_sha256": tree_sha,
        "file_count": len(actual_entries),
        "total_bytes": sum(int(item["bytes"]) for item in actual_entries),
        "entries": actual_entries,
        "global_step": spec["global_step"],
        "training_result_sha256": training_result_sha,
        "training_invocation_sha256": invocation_sha,
    }


def validate_binding(binding_path: Path, *, expected_head: str | None = None, check_output_fresh: bool = True) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    _validate_static_manifest(manifest)
    binding = _read_json(binding_path)
    require(binding.get("schema") == "CORELAB_STAGE45F_PI0194_RUNTIME_BINDING_V1", "runtime binding schema drift")
    require(binding.get("execution_authorized") is True, "runtime binding is not execution-authorized")
    checkout = binding.get("execution_checkout", {})
    require(isinstance(checkout, dict), "execution checkout binding missing")
    live = _current_checkout()
    expected = expected_head or str(checkout.get("authorized_head", ""))
    require(len(expected) == 40 and live["head"] == expected, f"execution HEAD mismatch: live={live['head']} expected={expected}")
    require(checkout.get("authorized_head") == expected and checkout.get("authorized_clean_worktree") is True and live["clean"], "execution checkout is not clean/bound")
    require(checkout.get("authorized_branch") == live["branch"], "execution branch mismatch")
    historical = binding.get("historical_stage43", {})
    require(historical.get("accepted_head") == STAGE43_ACCEPTED_HEAD, "historical Stage43 HEAD drift")
    for key, value in {
        "bundle_sha256": STAGE43_BUNDLE_SHA256,
        "result_sha256": STAGE43_RESULT_SHA256,
        "result_seal_sha256": STAGE43_RESULT_SEAL_SHA256,
        "cohort_manifest_sha256": STAGE43_COHORT_SHA256,
        "trajectory_sha256": TRAJECTORY_SHA256,
    }.items():
        require(historical.get(key) == value, f"historical Stage43 {key} drift")
    paths = binding.get("paths", {})
    require(isinstance(paths, dict) and set(paths) == REQUIRED_PATHS, "runtime path key set drift")
    for key, value in paths.items():
        require(isinstance(value, str) and value, f"runtime path is missing: {key}")
        lowered = value.lower()
        require("validation-e" not in lowered and "holdout-f" not in lowered, f"locked data path: {key}")
    stage10_root = _regular(Path(paths["stage10_root"]), "stage10 root")
    base_model = _regular(Path(paths["base_model_path"]), "base model")
    structured_checkpoint = _regular(Path(paths["structured_checkpoint"]), "structured_checkpoint directory")
    dataset = _regular(Path(paths["dataset_dir"]), "dataset_dir directory")
    cosmos = _regular(Path(paths["cosmos_snapshot"]), "cosmos_snapshot directory")
    wbc_root = _regular(Path(paths["wbc_source_root"]), "wbc_source_root directory")
    relay_manifest_path = _regular(Path(paths["relay_manifest"]), "relay manifest")
    model_python_value = str(paths["model_python"])
    if model_python_value == CONTAINER_MODEL_PYTHON:
        # The PBS entrypoint runs inside the frozen Isaac image.  This path is
        # intentionally image-local and therefore cannot be checked from the
        # host filesystem before Apptainer starts.
        model_python = Path(CONTAINER_MODEL_PYTHON)
    else:
        model_python = _regular(Path(model_python_value), "model Python")
        require(os.access(model_python, os.X_OK), "model Python is not executable")
    structured_stats = _regular(Path(paths["structured_stats"]), "structured stats")
    modality = _regular(Path(paths["modality_config"]), "modality config")
    trajectory = _regular(Path(paths["trajectory"]), "trajectory")
    wbc_arm = _regular(Path(paths["wbc_arm"]), "WBC arm checkpoint")
    wbc_dog = _regular(Path(paths["wbc_dog"]), "WBC dog checkpoint")
    require(sha256_file(structured_stats) == "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206", "structured stats hash drift")
    require(sha256_file(modality) == "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031", "modality hash drift")
    require(sha256_file(trajectory) == TRAJECTORY_SHA256, "trajectory hash drift")
    require(sha256_file(wbc_arm) == "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a", "WBC arm hash drift")
    require(sha256_file(wbc_dog) == "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670", "WBC dog hash drift")
    runtime_identity = binding.get("frozen_runtime_identity")
    require(runtime_identity == manifest.get("frozen_runtime"), "frozen runtime identity binding drift")
    relay_manifest = _read_json(relay_manifest_path)
    require(relay_manifest.get("schema") == "VANDA_PI0191_RELAY_MANIFEST_V1" and relay_manifest.get("status") == "PASS", "relay manifest is not PASS")
    candidate_paths = binding.get("candidate_paths", {})
    require(isinstance(candidate_paths, dict) and set(candidate_paths) == set(CANDIDATE_ORDER), "candidate path key set drift")
    inventories: dict[str, Any] = {}
    for candidate in CANDIDATE_ORDER:
        candidate_path = _regular(Path(candidate_paths.get(candidate, "")), f"{candidate} candidate")
        inventories[candidate] = _verify_candidate_tree(candidate_path, candidate, relay_manifest)
    output_root = Path(paths["output_root"]).expanduser()
    if check_output_fresh:
        require(not output_root.exists() and not output_root.is_symlink(), "output root must be fresh")
    runtime = {
        "stage10_root": str(stage10_root),
        "base_model_path": str(base_model),
        "structured_checkpoint": str(structured_checkpoint),
        "dataset_dir": str(dataset),
        "cosmos_snapshot": str(cosmos),
        "wbc_source_root": str(wbc_root),
        "model_python": str(model_python),
        "structured_stats": str(structured_stats),
        "modality_config": str(modality),
        "trajectory": str(trajectory),
        "wbc_arm": str(wbc_arm),
        "wbc_dog": str(wbc_dog),
        "relay_manifest": str(relay_manifest_path),
        "output_root": str(output_root),
    }
    return {
        "schema": binding["schema"],
        "status": "PASS",
        "binding_path": str(Path(binding_path).resolve()),
        "binding_sha256": sha256_file(Path(binding_path)),
        "checkout": live,
        "historical_stage43": historical,
        "runtime": runtime,
        "inventories": inventories,
        "candidate_paths": {key: str(Path(value).resolve()) for key, value in binding.get("candidate_paths", {}).items()},
        "binding": binding,
    }


def _make_stage43_bundle(binding: Mapping[str, Any], candidate: str, candidate_path: Path, stage43: Any) -> dict[str, Any]:
    paths = binding["paths"]
    file_records = {
        "structured_stats": (paths["structured_stats"], "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"),
        "modality": (paths["modality_config"], "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"),
        "trajectory": (paths["trajectory"], TRAJECTORY_SHA256),
        "wbc_arm_checkpoint": (paths["wbc_arm"], "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a"),
        "wbc_dog_checkpoint": (paths["wbc_dog"], "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670"),
    }
    artifacts: dict[str, Any] = {}
    for key, (path, digest) in file_records.items():
        artifacts[key] = {"path": path, "sha256": digest}
    directory_records = {
        "structured_checkpoint": paths["structured_checkpoint"],
        "tiny1_dataset": paths["dataset_dir"],
        "cosmos_snapshot": paths["cosmos_snapshot"],
        "wbc_source": paths["wbc_source_root"],
        "stage10_root": paths["stage10_root"],
        "base_model_path": paths["base_model_path"],
        "model_path": str(candidate_path),
        "dataset_dir": paths["dataset_dir"],
        "cosmos_snapshot_path": paths["cosmos_snapshot"],
    }
    for key, path in directory_records.items():
        artifacts[key] = {"path": path}
    artifacts["modality_config"] = {"path": paths["modality_config"], "sha256": file_records["modality"][1]}
    artifacts["wbc_arm_checkpoint"]["sha256"] = file_records["wbc_arm_checkpoint"][1]
    artifacts["wbc_dog_checkpoint"]["sha256"] = file_records["wbc_dog_checkpoint"][1]
    return {
        "schema": "CORELAB_STAGE45F_PI0194_LATE_BOUND_STAGE43_TEMPLATE_V1",
        "stage_id": "Stage43",
        "candidate": candidate,
        "repository": {"head": binding["execution_checkout"]["authorized_head"]},
        "target": {"evaluation_world_m": list(TARGET_WORLD), "language": LANGUAGE, "source_frame": SOURCE_FRAME, "tcp_semantic_id": TCP_ID, "numeric_xyz_in_model_input": False},
        "protocol": {"handoff_s": 0.9750000000000001, "action_dt_s": 0.05, "segment_native_ticks": list(NATIVE_SCHEDULE), "gr00t_forward_count_per_condition": FORWARDS_PER_CELL, "reference_update_count_per_condition": UPDATES_PER_CELL},
        "artifacts": artifacts,
        "historical_stage43": {"accepted_head": STAGE43_ACCEPTED_HEAD, "bundle_sha256": STAGE43_BUNDLE_SHA256, "result_sha256": STAGE43_RESULT_SHA256, "result_seal_sha256": STAGE43_RESULT_SEAL_SHA256, "cohort_manifest_sha256": STAGE43_COHORT_SHA256},
        "forbidden_operations": {key: False for key in stage43.evidence.expected_forbidden_operations()},
    }


def _request_with_counters(phase: Any, stage43: Any, *, native: Any, mailbox: Path, request_id: str, seed: int, sim_time: float, gate: Any, adapter: Any, args: argparse.Namespace, cases: list[Mapping[str, Any]], counter_events: list[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    before = stage43._counter_snapshot(cases, native)
    chunk, case = phase._case_request(native, mailbox, request_id, seed, sim_time, gate, adapter, args)
    case = dict(case)
    cases.append(case)
    after = stage43._counter_snapshot(cases, native)
    response = case["response"]
    event = {
        "event": "GR00T_FORWARD",
        "request_id": request_id,
        "sim_time_s": float(sim_time),
        "before": before,
        "after": after,
        "gr00t_forward_delta_observed": int(after["gr00t_forward_count"] - before["gr00t_forward_count"]),
        "worker_forward_count_delta": int(response.get("worker_forward_count_delta", -1)),
        "raw_sha256": case.get("raw_sha256"),
        "state_v1_sha256": case.get("state_v1_sha256"),
        "safety_decision": case["safety_gate"]["decision"],
    }
    require(event["gr00t_forward_delta_observed"] == 1 and event["worker_forward_count_delta"] == 1, request_id + " did not produce exactly one worker forward")
    counter_events.append(event)
    return chunk, case


def _start_worker(phase: Any, args: argparse.Namespace, mailbox: Path, log_path: Path) -> Any:
    """Reuse the Stage40 worker command with an explicit late-bound loader path.

    Stage40 normally removes the native Isaac library path before launching its
    model worker.  PI0194 keeps that isolation and adds only an optional
    ``PI0194_WORKER_LD_LIBRARY_PATH`` supplied by the frozen runtime binding;
    no model or action semantics are changed.
    """

    require(not (mailbox / "STOP").exists(), "stale worker mailbox STOP marker")
    worker_env = os.environ.copy()
    worker_env.pop("LD_LIBRARY_PATH", None)
    worker_ld = os.environ.get("PI0194_WORKER_LD_LIBRARY_PATH")
    if worker_ld:
        worker_env["LD_LIBRARY_PATH"] = worker_ld
    worker_env["PYTHONPATH"] = os.pathsep.join([
        str(REPO / "src"),
        str(SCRIPT_DIR),
        str(args.stage10_root / "src" / "isaac-gr00t"),
    ])
    cache_root = Path(args.cosmos_snapshot).resolve().parents[2]
    worker_env["HF_HUB_CACHE"] = str(cache_root)
    worker_env["TRANSFORMERS_CACHE"] = str(cache_root)
    worker_env["HF_HOME"] = str(cache_root.parent)
    worker_env["HF_HUB_OFFLINE"] = "1"
    worker_env["TRANSFORMERS_OFFLINE"] = "1"
    worker_env["HF_DATASETS_OFFLINE"] = "1"
    worker_env["PYTHONNOUSERSITE"] = "1"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    mailbox.mkdir(parents=True, exist_ok=True)
    (mailbox / "requests").mkdir(parents=True, exist_ok=True)
    (mailbox / "responses").mkdir(parents=True, exist_ok=True)
    command = [
        str(args.model_python), "-B", str(SCRIPT_DIR / "stage21_same_scene_groot_worker.py"),
        "--stage10-root", str(args.stage10_root),
        "--base-model-path", str(args.base_model_path),
        "--model-path", str(args.model_path),
        "--dataset-dir", str(args.dataset_dir),
        "--modality-config", str(args.modality_config),
        "--cosmos-snapshot", str(args.cosmos_snapshot),
        "--mailbox", str(mailbox),
        "--device", "cuda:0", "--seed", str(args.seed),
        "--local-files-only", "--action-contract", ACTION_CONTRACT,
    ]
    handle = log_path.open("w", encoding="utf-8")
    worker = subprocess.Popen(command, cwd=str(REPO), env=worker_env, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + float(args.worker_ready_timeout_s)
    while time.monotonic() < deadline:
        if log_path.is_file() and '"status": "READY"' in log_path.read_text(encoding="utf-8", errors="replace"):
            return worker
        if worker.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise AdapterError(f"GR00T worker exited before READY: rc={worker.returncode}\n{tail}")
        time.sleep(0.25)
    worker.terminate()
    raise AdapterError("timed out waiting for GR00T worker READY")


def _row_failure(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in rows:
        if bool(row.get("done")) or bool(row.get("reset_buf")) or str(row.get("termination_code", "NONE")) != "NONE":
            return str(row.get("termination_code", "NATIVE_TASK_TERMINATION"))
    return None


def _model_metrics(cases: Sequence[Mapping[str, Any]], native: Any, np: Any, trajectory_metrics: Any) -> list[dict[str, Any]]:
    anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3).astype(np.float64)
    target = np.asarray(TARGET_WORLD, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        with np.load(case["response"]["artifact"], allow_pickle=False) as data:
            raw = np.asarray(data["raw_universal_action"])
        decoded = np.asarray(case["decoded"], dtype=np.float64)
        require(raw.shape == (1, 40, 132) and np.isfinite(raw).all(), f"q{index} raw ABI mismatch")
        require(decoded.shape == (40, 10) and np.isfinite(decoded).all(), f"q{index} decoded ABI mismatch")
        prediction_world = decoded[:, :3] + anchor[None, :]
        predicted = trajectory_metrics(prediction_world, target, prefix=f"candidate.q{index}")
        rows.append({
            "query_index": index,
            "request_id": str(case["request_id"]),
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "raw_finite": bool(np.isfinite(raw).all()),
            "raw_array_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            "decoded_shape": list(decoded.shape),
            "decoded_dtype": str(decoded.dtype),
            "decoded_finite": bool(np.isfinite(decoded).all()),
            "decoded_array_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
            "predicted_endpoint_world_m": prediction_world[-1].tolist(),
            "predicted_endpoint_target_distance_m": predicted["final_target_distance_m"],
            "predicted_target_progress_m": predicted["absolute_progress_m"],
            "predicted_path_length_m": predicted["path_length_m"],
            "predicted_roughness_position_second_difference_m": predicted["roughness_position_second_difference_m"],
            "safety_gate": case["safety_gate"],
            "body_array_sha256": case.get("body_sha256"),
            "wrist_array_sha256": case.get("wrist_sha256"),
            "state_v1_array_sha256": case.get("state_v1_sha256"),
            "target_xyz_in_model_input": False,
            "model_input_components": ["body_rgb", "wrist_rgb", "StateV1", "language", "embodiment"],
        })
    return rows


def _write_ppm(path: Path, array: Any, np: Any) -> None:
    value = np.asarray(array, dtype=np.uint8)
    require(value.ndim == 3 and value.shape[-1] == 3, f"replay RGB shape drift: {value.shape}")
    height, width, _ = value.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + value.tobytes())


def _write_replay(root: Path, cases: Sequence[Mapping[str, Any]], setup: Mapping[str, Any] | None, rows_before: Sequence[Sequence[Mapping[str, Any]]], model_metrics: Sequence[Mapping[str, Any]], binding: Mapping[str, Any], native: Any, np: Any, status: str) -> dict[str, Any]:
    if not cases:
        return {"status": "NO_FRAMES", "frame_count": 0}
    from stage45f_pi0179_replay_recorder import ReplayFrameV1, ReplayRecorderConfigV1, ReplayRecorderV1

    source_root = root / "replay_source"
    frame_rows: list[dict[str, Any]] = []
    anchor = np.asarray(native.stage21_anchor_world.detach().cpu().numpy().reshape(3), dtype=np.float64) if native is not None else np.zeros(3, dtype=np.float64)
    for index, (case, prediction) in enumerate(zip(cases, model_metrics)):
        with np.load(case["observation_npz"], allow_pickle=False) as data:
            body = np.asarray(data["body_rgb"], dtype=np.uint8)
            wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
        body_path = source_root / f"q{index:02d}_body.ppm"
        wrist_path = source_root / f"q{index:02d}_wrist.ppm"
        _write_ppm(body_path, body, np)
        _write_ppm(wrist_path, wrist, np)
        if index == 0 and setup is not None:
            achieved = tuple(float(item) for item in setup["s0"]["eef_world_m"])
        elif index > 0 and index - 1 < len(rows_before) and rows_before[index - 1]:
            achieved = tuple(float(item) for item in rows_before[index - 1][-1]["achieved_pos"])
        else:
            achieved = tuple(float(item) for item in prediction["predicted_endpoint_world_m"])
        tracking_rms = None
        if index > 0 and index - 1 < len(rows_before) and rows_before[index - 1]:
            segment = rows_before[index - 1]
            errors = np.asarray([np.asarray(row["achieved_pos"], dtype=np.float64) - np.asarray(row["desired_pos"], dtype=np.float64) for row in segment])
            tracking_rms = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
        frame_rows.append({
            "timestamp_s": float(case["observation_timestamp"]),
            "body_frame": str(body_path),
            "wrist_frame": str(wrist_path),
            "achieved_eef_xyz": list(achieved),
            "target_xyz": list(TARGET_WORLD),
            "predicted_h40_xyz": (
                np.asarray(case["decoded"], dtype=np.float64)[:, :3] + anchor[None, :]
            ).tolist(),
            "model_update_index": index,
            "safety_gate": str(prediction["safety_gate"]["decision"]),
            "wbc_tracking_rms_position_m": tracking_rms,
            "lifecycle_state": "COMPLETE" if index == len(cases) - 1 and status == "PASS" else "RUNNING",
            "abort_reason": None if status == "PASS" else status,
            "initial_target_distance_m": None if setup is None else float(np.linalg.norm(np.asarray(setup["s0"]["eef_world_m"]) - np.asarray(TARGET_WORLD))),
        })
    frames_path = root / "replay_frames.json"
    _write_json(frames_path, {"schema": "CORELAB_STAGE45F_PI0194_REPLAY_FRAMES_V1", "frames": frame_rows})
    ffmpeg_bin = os.environ.get("PI0194_FFMPEG_BIN", "ffmpeg")
    replay_root = root / "replay"
    replay = ReplayRecorderV1(ReplayRecorderConfigV1(
        output_dir=replay_root,
        source_identity={"kind": "PI0194_NATIVE_CLOSED_LOOP", "frames_json": str(frames_path), "cell_status": status},
        classification="CANDIDATE_REPLAY_POSTPROCESSING_READY",
        ffmpeg_bin=ffmpeg_bin,
        native_third_person_source_available=False,
        allow_missing_ffmpeg=True,
    )).render(tuple(ReplayFrameV1.from_dict(row) for row in frame_rows))
    return {"status": replay.get("render_status"), "manifest": str(replay_root / "manifest.json"), "manifest_sha256": sha256_file(replay_root / "manifest.json"), "frame_count": len(frame_rows), "frames_json": str(frames_path), "frames_json_sha256": sha256_file(frames_path)}


def _run_cell(*, binding: Mapping[str, Any], admission: Mapping[str, Any], candidate: str, condition: str, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    for import_root in (SCRIPT_DIR, REPO / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    import stage43_corelab_initial_condition as stage43
    import stage40_corelab_phase as phase
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics
    from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
    from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1

    output_dir.mkdir(parents=True, exist_ok=False)
    mailbox = output_dir / "mailbox"
    (mailbox / "requests").mkdir(parents=True)
    (mailbox / "responses").mkdir(parents=True)
    paths = binding["paths"]
    candidate_path = Path(binding["candidate_paths"][candidate])
    bundle = _make_stage43_bundle(binding, candidate, candidate_path, stage43)
    bundle_artifact_checks = stage43._artifact_checks(bundle)
    trajectory = stage43._load_pose_file(Path(paths["trajectory"]))
    worker = None
    native = env = None
    cases: list[Mapping[str, Any]] = []
    rows_before: list[list[Mapping[str, Any]]] = []
    rows_after: list[list[Mapping[str, Any]]] = []
    bind_events: list[Mapping[str, Any]] = []
    handoffs: list[Mapping[str, Any]] = []
    executor_events: list[Mapping[str, Any]] = []
    state_events: list[Mapping[str, Any]] = []
    counter_events: list[Mapping[str, Any]] = []
    telemetry: list[Mapping[str, Any]] = []
    setup = None
    completion = None
    metrics = None
    error: dict[str, Any] | None = None
    scientific_failure: dict[str, Any] | None = None
    status = "ENGINEERING_BLOCKED"
    worker_return_code = None
    model_load_attempts = 0
    model_metrics: list[dict[str, Any]] = []
    replay: dict[str, Any] = {"status": "NOT_STARTED"}
    try:
        stage43._telemetry_sample(output_dir, telemetry, "before_condition")
        worker_args = argparse.Namespace(**vars(args))
        worker_args.bundle = Path(binding["binding_path"])
        worker_args.stage10_root = Path(paths["stage10_root"])
        worker_args.base_model_path = Path(paths["base_model_path"])
        worker_args.model_path = candidate_path
        worker_args.dataset_dir = Path(paths["dataset_dir"])
        worker_args.modality_config = Path(paths["modality_config"])
        worker_args.cosmos_snapshot = Path(paths["cosmos_snapshot"])
        worker_args.structured_stats = Path(paths["structured_stats"])
        worker_args.trajectory = Path(paths["trajectory"])
        worker_args.wbc_source_root = Path(paths["wbc_source_root"])
        worker_args.arm = Path(paths["wbc_arm"])
        worker_args.dog = Path(paths["wbc_dog"])
        worker_args.source_root = Path(paths["wbc_source_root"])
        worker_args.mode = "model_receding"
        model_load_attempts = 1
        worker = _start_worker(phase, worker_args, mailbox, output_dir / "worker.log")
        native, env, actors, _, _ = phase._build_native(phase._native_args(worker_args), trajectory)
        require(float(native.dt) > 0.0 and np.isfinite(float(native.dt)), "native dt is invalid")
        handoff_steps = int(round(stage43.HANDOFF_S / float(native.dt)))
        require((handoff_steps, handoff_steps, handoff_steps, 2 * handoff_steps) == NATIVE_SCHEDULE, "observed native dt drifted from frozen schedule")
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(expected_source_frame=SOURCE_FRAME, wbc_frame=SOURCE_FRAME, tcp_semantic_id=TCP_ID, stale_timeout_s=0.75))
        setup = stage43._apply_initial_base_delta(native, env, condition, CONDITION_DELTAS[condition])
        sim_time = 0.0
        current_chunk, first_case = _request_with_counters(phase, stage43, native=native, mailbox=mailbox, request_id=condition + "-q0", seed=args.seed, sim_time=sim_time, gate=gate, adapter=adapter, args=worker_args, cases=cases, counter_events=counter_events)
        if first_case["safety_gate"]["decision"] == GateDecision.REJECT.value:
            scientific_failure = {"classification": "SAFETY_GATE_REJECT", "request_id": first_case["request_id"], "decision": first_case["safety_gate"]}
            status = "SCIENTIFIC_REJECT"
        else:
            executor, initial_bind_event, state_event = phase._initial_bind(native, env, current_chunk, sim_time, {"stage": "Stage43", "candidate": candidate, "condition_id": condition, "mode": "CORELAB_STAGE45F_PI0194_CONTROLLED_COHORT"})
            bind_events.append(initial_bind_event)
            state_events.append(state_event)
            executor_events.append(initial_bind_event.get("executor_initial"))
            status = "RUNNING"
            for handoff_index in range(UPDATES_PER_CELL):
                before_counter = stage43._counter_snapshot(cases, native)
                segment, sim_time, termination = phase._step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=handoff_steps, update_count=handoff_index)
                rows_before.append(segment)
                after_counter = stage43._counter_snapshot(cases, native)
                counter_events.append({"event": "WBC_SEGMENT", "condition_id": condition, "segment_index": handoff_index, "steps_expected": handoff_steps, "steps_observed": len(segment), "before": before_counter, "after": after_counter, "termination": termination})
                row_failure = _row_failure(segment)
                if termination is not None or row_failure is not None or len(segment) != handoff_steps:
                    scientific_failure = {"classification": "WBC_ABORT", "segment_index": handoff_index, "termination": termination or row_failure, "steps_observed": len(segment), "steps_expected": handoff_steps}
                    status = "SCIENTIFIC_ABORT"
                    break
                state_before = {"eef_world_m": phase._current_eef(native)[0].tolist(), "state_v1": phase._current_state_v1(native).as_array().tolist(), "state_v1_sha256": stage43._array_sha(phase._current_state_v1(native).as_array())}
                next_chunk, next_case = _request_with_counters(phase, stage43, native=native, mailbox=mailbox, request_id=condition + "-q" + str(handoff_index + 1), seed=args.seed, sim_time=sim_time, gate=gate, adapter=adapter, args=worker_args, cases=cases, counter_events=counter_events)
                if next_case["safety_gate"]["decision"] == GateDecision.REJECT.value:
                    scientific_failure = {"classification": "SAFETY_GATE_REJECT", "request_id": next_case["request_id"], "decision": next_case["safety_gate"]}
                    status = "SCIENTIFIC_REJECT"
                    break
                update_before = stage43._counter_snapshot(cases, native)
                update_event, update_state = phase._update_reference(native, env, executor, current_chunk, next_chunk, sim_time, handoff_steps, handoff_index + 1)
                update_after = stage43._counter_snapshot(cases, native)
                update_event = dict(update_event)
                update_event.update({"condition_id": condition, "observation_state_after_segment": state_before, "reference_counter_before": update_before["reference_update_count"], "reference_counter_after": update_after["reference_update_count"], "reference_update_delta_observed": update_after["reference_update_count"] - update_before["reference_update_count"]})
                require(update_event["reference_update_delta_observed"] == 1 and update_event.get("reference_only") is True, "reference update contract drift")
                handoffs.append(update_event)
                state_events.append(update_state)
                executor_events.append(update_event.get("executor_update"))
                current_chunk = next_chunk
            if status == "RUNNING":
                before_counter = stage43._counter_snapshot(cases, native)
                final_segment, sim_time, termination = phase._step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=2 * handoff_steps, update_count=UPDATES_PER_CELL)
                rows_after.append(final_segment)
                after_counter = stage43._counter_snapshot(cases, native)
                counter_events.append({"event": "WBC_COMPLETION_SEGMENT", "condition_id": condition, "steps_expected": 2 * handoff_steps, "steps_observed": len(final_segment), "before": before_counter, "after": after_counter, "termination": termination})
                row_failure = _row_failure(final_segment)
                if termination is not None or row_failure is not None or len(final_segment) != 2 * handoff_steps:
                    scientific_failure = {"classification": "WBC_ABORT", "segment_index": UPDATES_PER_CELL, "termination": termination or row_failure, "steps_observed": len(final_segment), "steps_expected": 2 * handoff_steps}
                    status = "SCIENTIFIC_ABORT"
                else:
                    native_time = float(native.traj_sim_time[0].detach().cpu().item())
                    ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
                    ref_L = float(native.traj_batch.L[0].detach().cpu().item())
                    ref_duration = float(current_chunk.validity_end - current_chunk.validity_start)
                    endpoint = bool(native_time >= ref_duration - max(float(native.dt) * 0.5, 1e-9) and ref_s >= ref_L - max(1e-4, 1e-3 * max(ref_L, 1.0)))
                    completion = {"native_time_s": native_time, "reference_duration_s": ref_duration, "traj_s_ref": ref_s, "traj_L": ref_L, "endpoint_reached": endpoint, "termination": termination, "native_handoff_steps": handoff_steps, "schedule_native_ticks": list(NATIVE_SCHEDULE), "sim_time_s": sim_time}
                    require(endpoint, "Stage43 reference endpoint was not reached")
                    require(len(cases) == FORWARDS_PER_CELL and int(native.stage21_reference_update_count) == UPDATES_PER_CELL and int(native.stage21_wbc_forward_count) == TICKS_PER_CELL, "cell counter contract mismatch")
                    status = "PASS"
        model_metrics = _model_metrics(cases, native, np, trajectory_metrics)
        if setup is not None and (rows_before or rows_after):
            flat_rows = [row for segment in rows_before + rows_after for row in segment]
            metrics = stage43._metrics(setup, flat_rows)
    except Exception as exc:
        if scientific_failure is None:
            error = {"type": type(exc).__name__, "message": str(exc), "classification": "ENGINEERING"}
            status = "ENGINEERING_BLOCKED"
    finally:
        try:
            if native is not None:
                stage43._telemetry_sample(output_dir, telemetry, "after_native")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc), "classification": "ENGINEERING"}
            status = "ENGINEERING_BLOCKED"
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        worker_return_code = phase._stop_worker(worker, mailbox)
        try:
            stage43._telemetry_sample(output_dir, telemetry, "after_condition")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc), "classification": "ENGINEERING"}
            status = "ENGINEERING_BLOCKED"
        try:
            materialized = stage43._materialize(args=args, bundle=bundle, root=output_dir, mailbox=mailbox, condition_id=condition, setup=setup, cases=cases, rows_before=rows_before, rows_after=rows_after, bind_events=bind_events, handoffs=handoffs, executor_events=executor_events, state_events=state_events, counter_events=counter_events, telemetry=telemetry, native=native, worker_return_code=worker_return_code, status=status, completion=completion, metrics=metrics, artifact_checks=bundle_artifact_checks, error=error or scientific_failure)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc), "classification": "ENGINEERING_MATERIALIZATION"}
            status = "ENGINEERING_BLOCKED"
            materialized = None
            _write_json(output_dir / "materialize_error.json", error)
        if status != "ENGINEERING_BLOCKED":
            try:
                replay = _write_replay(output_dir, cases, setup, rows_before, model_metrics, binding, native, np, status)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc), "classification": "ENGINEERING_REPLAY"}
                status = "ENGINEERING_BLOCKED"
                _write_json(output_dir / "replay_error.json", error)
    payload = {
        "schema": CELL_SCHEMA,
        "status": status,
        "candidate": candidate,
        "condition_id": condition,
        "candidate_tree_sha256": CANDIDATES[candidate]["tree_sha256"],
        "historical_stage43": {"accepted_head": STAGE43_ACCEPTED_HEAD, "bundle_sha256": STAGE43_BUNDLE_SHA256, "result_sha256": STAGE43_RESULT_SHA256, "result_seal_sha256": STAGE43_RESULT_SEAL_SHA256, "cohort_manifest_sha256": STAGE43_COHORT_SHA256},
        "condition_delta_base_xy_m": list(CONDITION_DELTAS[condition]),
        "gr00t_forward_count": len(cases),
        "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
        "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "model_load_attempts": model_load_attempts,
        "metrics": metrics,
        "model_metrics": model_metrics,
        "scientific_failure": scientific_failure,
        "error": error,
        "replay": replay,
        "materialized": None if materialized is None else {"condition_result_sha256": materialized["result_sha256"], "condition_manifest_sha256": materialized["manifest_sha256"]},
        "forbidden_operations": {"training": False, "backward": False, "optimizer": False, "Validation-E": False, "Holdout-F": False, "hardware": False, "checkpoint_writes": False, "metric_dependent_retry": False},
        "runtime": {"python": platform.python_version(), "hostname": platform.node(), "gpu_uuid": args.gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    _write_json(output_dir / "cell_result.json", payload)
    _write_json(output_dir / "cell_result.json.sha256.json", {"sha256": sha256_file(output_dir / "cell_result.json"), "path": "cell_result.json"})
    return payload


def _numeric_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mean = sum(ordered) / len(ordered)
    return {"count": len(ordered), "mean": mean, "min": ordered[0], "max": ordered[-1]}


def _aggregate_candidate(root: Path, candidate: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        cells.append(_read_json(root / condition / "cell_result.json"))
    actual = [cell["metrics"] for cell in cells if isinstance(cell.get("metrics"), dict)]
    by_condition = {cell["condition_id"]: cell for cell in cells}
    reject_count = sum(1 for cell in cells if cell.get("status") == "SCIENTIFIC_REJECT")
    abort_count = sum(1 for cell in cells if cell.get("status") == "SCIENTIFIC_ABORT")
    return {
        "candidate": candidate,
        "label": CANDIDATES[candidate]["label"],
        "cell_count": len(cells),
        "complete_cell_count": sum(cell.get("status") == "PASS" for cell in cells),
        "scientific_reject_count": reject_count,
        "scientific_abort_count": abort_count,
        "engineering_blocked_count": sum(cell.get("status") == "ENGINEERING_BLOCKED" for cell in cells),
        "gr00t_forward_count": sum(int(cell.get("gr00t_forward_count", 0)) for cell in cells),
        "reference_update_count": sum(int(cell.get("reference_update_count", 0)) for cell in cells),
        "wbc_step_count": sum(int(cell.get("wbc_step_count", 0)) for cell in cells),
        "mean_progress_m": _numeric_summary([float(row["absolute_progress_m"]) for row in actual]),
        "mean_final_distance_m": _numeric_summary([float(row["final_target_error_m"]) for row in actual]),
        "mean_minimum_distance_m": _numeric_summary([float(row["minimum_target_error_m"]) for row in actual]),
        "mean_path_length_m": _numeric_summary([float(row["tracking_rms_m"]) for row in actual]),
        "per_condition": {condition: {"status": by_condition[condition].get("status"), "final_target_error_m": None if by_condition[condition].get("metrics") is None else by_condition[condition]["metrics"].get("final_target_error_m"), "absolute_progress_m": None if by_condition[condition].get("metrics") is None else by_condition[condition]["metrics"].get("absolute_progress_m"), "safety_gate_reject": by_condition[condition].get("status") == "SCIENTIFIC_REJECT", "wbc_abort": by_condition[condition].get("status") == "SCIENTIFIC_ABORT"} for condition in CONDITION_ORDER},
        "cells": cells,
    }


def _classify_scorecard(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    parent = aggregates["PARENT"]
    e = aggregates["E"]
    s20 = aggregates["S20"]
    complete = all(item.get("complete_cell_count") == len(CONDITION_ORDER) and item.get("scientific_reject_count") == 0 and item.get("scientific_abort_count") == 0 and item.get("engineering_blocked_count") == 0 for item in (e, s20))
    if not complete:
        classification = "MIXED_FOR_PI_REVIEW"
        basis = "E/S20 matrix contains an incomplete or scientifically rejected/aborted cell"
        wins = None
    else:
        e_progress = float(e["mean_progress_m"]["mean"])
        s20_progress = float(s20["mean_progress_m"]["mean"])
        e_final = float(e["mean_final_distance_m"]["mean"])
        s20_final = float(s20["mean_final_distance_m"]["mean"])
        wins = sum(float(s20["per_condition"][condition]["final_target_error_m"]) < float(e["per_condition"][condition]["final_target_error_m"]) for condition in CONDITION_ORDER)
        e_wins = sum(float(e["per_condition"][condition]["final_target_error_m"]) < float(s20["per_condition"][condition]["final_target_error_m"]) for condition in CONDITION_ORDER)
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
        "classification": classification,
        "basis": basis,
        "parent_background": parent,
        "E": e,
        "S20": s20,
        "matched_condition_s20_final_distance_wins": wins,
        "safety_gate_rejects_in_E": e.get("scientific_reject_count"),
        "safety_gate_rejects_in_S20": s20.get("scientific_reject_count"),
        "wbc_aborts_in_E": e.get("scientific_abort_count"),
        "wbc_aborts_in_S20": s20.get("scientific_abort_count"),
    }


def execute(binding_path: Path, *, expected_head: str | None = None, gpu_uuid: str, sim_device: str, response_timeout_s: float, worker_ready_timeout_s: float, seed: int) -> int:
    require(seed == DEV_SEED, "PI0194 development seed is fixed at 4242")
    admission = validate_binding(binding_path, expected_head=expected_head, check_output_fresh=True)
    binding = admission["binding"]
    output_root = Path(admission["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "admission.json", {"schema": "CORELAB_STAGE45F_PI0194_ADMISSION_V1", "status": "PASS", "return_token": RETURN_TOKEN, "admission": {key: value for key, value in admission.items() if key != "binding"}, "execution_controls": {"model_only_smoke": 0, "training": 0, "retry": 0, "allocation_retry": 0}})
    args = argparse.Namespace(
        bundle=Path(binding_path),
        gpu_uuid=gpu_uuid,
        source_root=Path(admission["runtime"]["wbc_source_root"]),
        trajectory=Path(admission["runtime"]["trajectory"]),
        arm=Path(admission["runtime"]["wbc_arm"]),
        dog=Path(admission["runtime"]["wbc_dog"]),
        stage10_root=Path(admission["runtime"]["stage10_root"]),
        base_model_path=Path(admission["runtime"]["base_model_path"]),
        model_path=None,
        dataset_dir=Path(admission["runtime"]["dataset_dir"]),
        modality_config=Path(admission["runtime"]["modality_config"]),
        cosmos_snapshot=Path(admission["runtime"]["cosmos_snapshot"]),
        structured_stats=Path(admission["runtime"]["structured_stats"]),
        model_python=Path(admission["runtime"]["model_python"]),
        wbc_source_root=Path(admission["runtime"]["wbc_source_root"]),
        sim_device=sim_device,
        seed=seed,
        response_timeout_s=response_timeout_s,
        worker_ready_timeout_s=worker_ready_timeout_s,
    )
    cell_counters = {"model_load_attempts": 0, "gr00t_forward_count": 0, "reference_update_count": 0, "wbc_step_count": 0, "cells": 0}
    aggregates: dict[str, dict[str, Any]] = {}
    try:
        for candidate in CANDIDATE_ORDER:
            candidate_root = output_root / candidate
            candidate_root.mkdir()
            for condition in CONDITION_ORDER:
                cell = _run_cell(binding=binding, admission=admission, candidate=candidate, condition=condition, output_dir=candidate_root / condition, args=args)
                cell_counters["cells"] += 1
                cell_counters["model_load_attempts"] += int(cell.get("model_load_attempts", 0))
                cell_counters["gr00t_forward_count"] += int(cell.get("gr00t_forward_count", 0))
                cell_counters["reference_update_count"] += int(cell.get("reference_update_count", 0))
                cell_counters["wbc_step_count"] += int(cell.get("wbc_step_count", 0))
                if cell.get("status") == "ENGINEERING_BLOCKED":
                    raise AdapterError(f"engineering failure at {candidate}/{condition}: {cell.get('error')}")
                require(cell_counters["gr00t_forward_count"] <= MAX_FORWARDS, "GR00T forward ceiling exceeded")
                require(cell_counters["reference_update_count"] <= MAX_UPDATES, "reference update ceiling exceeded")
                require(cell_counters["wbc_step_count"] <= MAX_TICKS, "native tick ceiling exceeded")
            aggregates[candidate] = _aggregate_candidate(candidate_root, candidate)
        scorecard = _classify_scorecard(aggregates)
        result = {
            "schema": "CORELAB_STAGE45F_PI0194_E_S20_CLOSED_LOOP_RESULT_V1",
            "status": "PASS",
            "return_token": RETURN_TOKEN,
            "execution_head": admission["checkout"]["head"],
            "gpu_uuid": gpu_uuid,
            "sim_device": sim_device,
            "candidate_order": list(CANDIDATE_ORDER),
            "condition_order": list(CONDITION_ORDER),
            "cell_counters": cell_counters,
            "scorecard": scorecard,
            "hard_zeroes": {"training": 0, "backward": 0, "optimizer": 0, "Validation-E": 0, "Holdout-F": 0, "hardware": 0, "ICL": 0, "DAgger": 0, "new_visited_state_collection": 0, "checkpoint_writes": 0, "metric_dependent_retry": 0},
            "runtime_identity": binding.get("frozen_runtime_identity"),
        }
        _write_json(output_root / "scorecard.json", result)
        result_sha = sha256_file(output_root / "scorecard.json")
        seal = {"schema": "CORELAB_STAGE45F_PI0194_RESULT_SEAL_V1", "status": "PASS", "return_token": RETURN_TOKEN, "execution_head": admission["checkout"]["head"], "scorecard_sha256": result_sha, "admission_sha256": sha256_file(output_root / "admission.json"), "cell_count": cell_counters["cells"], "gr00t_forward_count": cell_counters["gr00t_forward_count"], "reference_update_count": cell_counters["reference_update_count"], "wbc_step_count": cell_counters["wbc_step_count"]}
        _write_json(output_root / "result_seal.json", seal)
        print(json.dumps({"status": "PASS", "token": RETURN_TOKEN, "classification": scorecard["classification"], "output_root": str(output_root), "scorecard_sha256": result_sha, "counters": cell_counters}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = {"schema": "CORELAB_STAGE45F_PI0194_EXECUTION_FAILURE_V1", "status": "BLOCKED", "error": {"type": type(exc).__name__, "message": str(exc)}, "execution_head": admission["checkout"]["head"], "cell_counters": cell_counters, "valid_raw_preserved": True}
        _write_json(output_root / "execution_failure.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--static-check", action="store_true")
    modes.add_argument("--verify-binding", type=Path)
    modes.add_argument("--execute", type=Path, metavar="BINDING_JSON")
    parser.add_argument("--expected-head")
    parser.add_argument("--gpu-uuid", default="unknown")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    if args.static_check:
        print(json.dumps(static_contract(), indent=2, sort_keys=True))
        return 0
    if args.verify_binding is not None:
        print(json.dumps({key: value for key, value in validate_binding(args.verify_binding, expected_head=args.expected_head, check_output_fresh=False).items() if key != "binding"}, indent=2, sort_keys=True))
        return 0
    return execute(args.execute, expected_head=args.expected_head, gpu_uuid=args.gpu_uuid, sim_device=args.sim_device, response_timeout_s=args.response_timeout_s, worker_ready_timeout_s=args.worker_ready_timeout_s, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
