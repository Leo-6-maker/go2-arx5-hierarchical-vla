#!/usr/bin/env python3
"""Audit matched Stage22 H2 R0/R1 run provenance before profile claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


EXPECTED_SCHEMA = "STAGE22_RECIPE_COMPARISON_V1"
EXPECTED_TRAIN_TARGETS = [
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
]
EXPECTED_R1_TRAINABLE_PARAMETERS = 1_419_078_784


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _flag(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _has_flag(command: list[str], name: str) -> bool:
    return name in command


def _parse_trainable_parameters(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Trainable parameters:\s*([0-9][0-9,]*)", text)
    return None if not matches else int(matches[-1].replace(",", ""))


def _add_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def audit_h2_run(payload: Mapping[str, Any], *, comparison_path: Path) -> dict[str, Any]:
    """Return an integrity audit; every required invariant is explicit."""

    checks: dict[str, dict[str, Any]] = {}
    _add_check(
        checks,
        "comparison_schema",
        payload.get("schema") == EXPECTED_SCHEMA,
        payload.get("schema"),
        EXPECTED_SCHEMA,
    )
    _add_check(
        checks,
        "comparison_status",
        payload.get("status") == "PASS",
        payload.get("status"),
        "PASS",
    )
    _add_check(
        checks,
        "simulation_only",
        payload.get("simulation_only") is True,
        payload.get("simulation_only"),
        True,
    )
    _add_check(
        checks,
        "hardware_control_disabled",
        payload.get("hardware_control") is False,
        payload.get("hardware_control"),
        False,
    )
    _add_check(
        checks,
        "training_targets_locked",
        payload.get("training_target_ids") == EXPECTED_TRAIN_TARGETS,
        payload.get("training_target_ids"),
        EXPECTED_TRAIN_TARGETS,
    )
    _add_check(
        checks,
        "r1_preflight_passed",
        payload.get("r1_preflight_failed") is False,
        payload.get("r1_preflight_failed"),
        False,
    )

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    r0_matches = [item for item in results if isinstance(item, Mapping) and item.get("label") == "r0"]
    r1_matches = [item for item in results if isinstance(item, Mapping) and item.get("label") == "r1"]
    r0_result = r0_matches[0] if len(r0_matches) == 1 else None
    r1_result = r1_matches[0] if len(r1_matches) == 1 else None
    _add_check(checks, "exactly_one_full_r0_result", r0_result is not None, len(r0_matches), 1)
    _add_check(checks, "exactly_one_full_r1_result", r1_result is not None, len(r1_matches), 1)

    if r0_result is None or r1_result is None:
        return {
            "schema": "STAGE22_H2_RUN_INTEGRITY_AUDIT_V1",
            "status": "FAIL",
            "comparison_manifest": str(comparison_path),
            "checks": checks,
            "blocking_checks": [name for name, check in checks.items() if not check["passed"]],
            "claim_boundary": "H2 integrity not established; no R0/R1 profile claim",
        }

    r0_command = [str(value) for value in r0_result.get("command", [])]
    r1_command = [str(value) for value in r1_result.get("command", [])]
    _add_check(checks, "r0_result_passed", r0_result.get("status") == "PASS", r0_result.get("status"), "PASS")
    _add_check(checks, "r1_result_passed", r1_result.get("status") == "PASS", r1_result.get("status"), "PASS")
    _add_check(checks, "r0_max_steps_50", r0_result.get("max_steps") == 50, r0_result.get("max_steps"), 50)
    _add_check(checks, "r1_max_steps_50", r1_result.get("max_steps") == 50, r1_result.get("max_steps"), 50)
    _add_check(
        checks,
        "matched_save_steps",
        r0_result.get("save_steps") == r1_result.get("save_steps") == 25,
        [r0_result.get("save_steps"), r1_result.get("save_steps")],
        [25, 25],
    )

    common_flags = (
        "--stage10-root",
        "--base-model-path",
        "--dataset-dir",
        "--modality-config",
        "--learning-rate",
        "--seed",
    )
    for name in common_flags:
        left = _flag(r0_command, name)
        right = _flag(r1_command, name)
        _add_check(checks, f"matched_command_{name[2:]}", left == right, [left, right], left)
    _add_check(
        checks,
        "matched_training_seed_2222",
        _flag(r0_command, "--seed") == _flag(r1_command, "--seed") == "2222",
        [_flag(r0_command, "--seed"), _flag(r1_command, "--seed")],
        ["2222", "2222"],
    )
    _add_check(
        checks,
        "local_files_only_both",
        _has_flag(r0_command, "--local-files-only") and _has_flag(r1_command, "--local-files-only"),
        [_has_flag(r0_command, "--local-files-only"), _has_flag(r1_command, "--local-files-only")],
        [True, True],
    )
    _add_check(
        checks,
        "r0_diffusion_flag_absent",
        not _has_flag(r0_command, "--tune-diffusion-model"),
        _has_flag(r0_command, "--tune-diffusion-model"),
        False,
    )
    _add_check(
        checks,
        "r1_diffusion_flag_present",
        _has_flag(r1_command, "--tune-diffusion-model"),
        _has_flag(r1_command, "--tune-diffusion-model"),
        True,
    )

    comparison_root = comparison_path.parent
    manifests: dict[str, dict[str, Any] | None] = {}
    for recipe, result in (("r0", r0_result), ("r1", r1_result)):
        manifest_path_value = result.get("run_manifest")
        manifest_path = _resolve(manifest_path_value, base=comparison_root) if manifest_path_value else None
        manifest = None if manifest_path is None else _load_json(manifest_path)
        manifests[recipe] = manifest
        _add_check(
            checks,
            f"{recipe}_run_manifest_present",
            manifest is not None,
            None if manifest_path is None else str(manifest_path),
            "JSON manifest",
        )

    r0_manifest = manifests["r0"]
    r1_manifest = manifests["r1"]
    if r0_manifest is None or r1_manifest is None:
        return {
            "schema": "STAGE22_H2_RUN_INTEGRITY_AUDIT_V1",
            "status": "FAIL",
            "comparison_manifest": str(comparison_path),
            "checks": checks,
            "blocking_checks": [name for name, check in checks.items() if not check["passed"]],
            "claim_boundary": "H2 integrity not established; no R0/R1 profile claim",
        }

    for field, expected in (("schema", "STAGE20S_GR00T_TRAINING_RUN_V1"), ("status", "PASS")):
        _add_check(checks, f"r0_{field}", r0_manifest.get(field) == expected, r0_manifest.get(field), expected)
        _add_check(checks, f"r1_{field}", r1_manifest.get(field) == expected, r1_manifest.get(field), expected)
    for field, expected in (("simulation_only", True), ("hardware_control", False)):
        _add_check(checks, f"r0_{field}", r0_manifest.get(field) is expected, r0_manifest.get(field), expected)
        _add_check(checks, f"r1_{field}", r1_manifest.get(field) is expected, r1_manifest.get(field), expected)

    _add_check(
        checks,
        "same_base_model_path",
        r0_manifest.get("base_model_path") == r1_manifest.get("base_model_path"),
        [r0_manifest.get("base_model_path"), r1_manifest.get("base_model_path")],
        "equal",
    )
    _add_check(
        checks,
        "same_base_model_index_sha256",
        bool(r0_manifest.get("base_model_index_sha256"))
        and r0_manifest.get("base_model_index_sha256") == r1_manifest.get("base_model_index_sha256"),
        [r0_manifest.get("base_model_index_sha256"), r1_manifest.get("base_model_index_sha256")],
        "equal non-empty hashes",
    )
    _add_check(
        checks,
        "same_dataset_path",
        r0_manifest.get("dataset_dir") == r1_manifest.get("dataset_dir"),
        [r0_manifest.get("dataset_dir"), r1_manifest.get("dataset_dir")],
        "equal",
    )
    r0_dataset_fingerprint = r0_result.get("dataset_fingerprint")
    r1_dataset_fingerprint = r1_result.get("dataset_fingerprint")
    top_dataset_fingerprint = payload.get("dataset_fingerprint")
    r0_dataset_tree_sha = (
        r0_dataset_fingerprint.get("tree_sha256")
        if isinstance(r0_dataset_fingerprint, Mapping)
        else None
    )
    r1_dataset_tree_sha = (
        r1_dataset_fingerprint.get("tree_sha256")
        if isinstance(r1_dataset_fingerprint, Mapping)
        else None
    )
    top_dataset_tree_sha = (
        top_dataset_fingerprint.get("tree_sha256")
        if isinstance(top_dataset_fingerprint, Mapping)
        else None
    )
    _add_check(
        checks,
        "same_dataset_tree_sha256",
        bool(r0_dataset_tree_sha)
        and r0_dataset_tree_sha == r1_dataset_tree_sha == top_dataset_tree_sha,
        [r0_dataset_tree_sha, r1_dataset_tree_sha, top_dataset_tree_sha],
        "equal non-empty hashes",
    )
    _add_check(
        checks,
        "same_groot_repo_sha",
        bool(r0_manifest.get("groot_repo_sha")) and r0_manifest.get("groot_repo_sha") == r1_manifest.get("groot_repo_sha"),
        [r0_manifest.get("groot_repo_sha"), r1_manifest.get("groot_repo_sha")],
        "equal non-empty SHAs",
    )
    _add_check(
        checks,
        "action_horizon_40_both",
        r0_manifest.get("action_horizon") == r1_manifest.get("action_horizon") == 40,
        [r0_manifest.get("action_horizon"), r1_manifest.get("action_horizon")],
        [40, 40],
    )
    _add_check(
        checks,
        "action_period_semantics_matched",
        r0_manifest.get("action_sample_period_s") == r1_manifest.get("action_sample_period_s") == 0.05
        and r0_manifest.get("action_sample_period_provenance") == r1_manifest.get("action_sample_period_provenance") == "TEST_ONLY_SIM_VISUAL_REACH_V0",
        [
            r0_manifest.get("action_sample_period_s"),
            r1_manifest.get("action_sample_period_s"),
            r0_manifest.get("action_sample_period_provenance"),
            r1_manifest.get("action_sample_period_provenance"),
        ],
        "0.05 s / TEST_ONLY_SIM_VISUAL_REACH_V0",
    )

    expected_flags = {
        "tune_projector": (True, True),
        "tune_diffusion_model": (False, True),
        "tune_llm": (False, False),
        "tune_visual": (False, False),
    }
    for field, (r0_expected, r1_expected) in expected_flags.items():
        _add_check(checks, f"r0_{field}", r0_manifest.get(field) is r0_expected, r0_manifest.get(field), r0_expected)
        _add_check(checks, f"r1_{field}", r1_manifest.get(field) is r1_expected, r1_manifest.get(field), r1_expected)

    for field in ("python", "cuda_visible_devices"):
        r0_value = r0_manifest.get("runtime", {}).get(field)
        r1_value = r1_manifest.get("runtime", {}).get(field)
        _add_check(checks, f"matched_runtime_{field}", r0_value == r1_value, [r0_value, r1_value], "equal")
    for field in ("device_name", "torch_version", "cuda_version"):
        r0_value = r0_manifest.get("cuda_telemetry", {}).get(field)
        r1_value = r1_manifest.get("cuda_telemetry", {}).get(field)
        _add_check(checks, f"matched_cuda_{field}", r0_value == r1_value and r0_value is not None, [r0_value, r1_value], "equal non-empty")
    r0_total = r0_manifest.get("cuda_telemetry", {}).get("before", {}).get("total_memory_bytes")
    r1_total = r1_manifest.get("cuda_telemetry", {}).get("before", {}).get("total_memory_bytes")
    _add_check(checks, "matched_cuda_total_memory", r0_total == r1_total and r0_total is not None, [r0_total, r1_total], "equal non-empty")

    resource = payload.get("resource_fingerprint")
    _add_check(
        checks,
        "resource_fingerprint_available",
        isinstance(resource, Mapping) and resource.get("available") is True and bool(resource.get("devices")),
        resource,
        "available GPU identity",
    )
    if isinstance(resource, Mapping):
        devices = resource.get("devices", [])
        _add_check(checks, "resource_fingerprint_has_uuid", all(item.get("uuid") for item in devices if isinstance(item, Mapping)), devices, "UUID for every device")

    for recipe, result in (("r0", r0_result), ("r1", r1_result)):
        log_value = result.get("launcher_log")
        log_path = _resolve(log_value, base=comparison_root) if log_value else None
        count = None if log_path is None else _parse_trainable_parameters(log_path)
        if recipe == "r1":
            _add_check(checks, "r1_trainable_parameter_count", count == EXPECTED_R1_TRAINABLE_PARAMETERS, count, EXPECTED_R1_TRAINABLE_PARAMETERS)
        else:
            _add_check(checks, "r0_trainable_parameter_count_recorded", count is not None, count, "recorded in launcher log")

    split_lock_value = payload.get("split_lock")
    split_lock_path = _resolve(split_lock_value, base=comparison_root) if split_lock_value else None
    split_lock = None if split_lock_path is None else _load_json(split_lock_path)
    actual_split_hash = None if split_lock_path is None or not split_lock_path.is_file() else _sha256(split_lock_path)
    _add_check(checks, "split_lock_present", split_lock is not None, None if split_lock_path is None else str(split_lock_path), "JSON split lock")
    _add_check(checks, "split_lock_hash_matches_top_level", actual_split_hash == payload.get("split_lock_sha256"), actual_split_hash, payload.get("split_lock_sha256"))
    _add_check(
        checks,
        "split_lock_is_train_only",
        isinstance(split_lock, Mapping)
        and split_lock.get("dataset_role") == "TRAIN"
        and split_lock.get("selected_split") == "train"
        and split_lock.get("selected_target_ids") == EXPECTED_TRAIN_TARGETS
        and split_lock.get("training_allowed") is True
        and split_lock.get("final_test_in_training") is False,
        split_lock,
        "locked train_a-d-only sidecar",
    )

    blocking = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema": "STAGE22_H2_RUN_INTEGRITY_AUDIT_V1",
        "status": "PASS" if not blocking else "FAIL",
        "comparison_manifest": str(comparison_path),
        "r0_result": {"label": r0_result.get("label"), "run_manifest": r0_result.get("run_manifest")},
        "r1_result": {"label": r1_result.get("label"), "run_manifest": r1_result.get("run_manifest")},
        "checks": checks,
        "blocking_checks": blocking,
        "claim_boundary": "H2 matched-run provenance audit only; profile quality still requires held-out paired evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = _load_json(args.comparison.resolve())
    if payload is None:
        raise SystemExit(f"comparison manifest is missing or invalid: {args.comparison}")
    result = audit_h2_run(payload, comparison_path=args.comparison.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
