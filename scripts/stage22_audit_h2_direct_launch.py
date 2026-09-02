#!/usr/bin/env python3
"""Audit the authoritative direct-launch Stage22 H2 execution contract.

This adapter intentionally consumes the two frozen Stage20S run manifests
directly.  It does not invoke a training launcher, infer a model, select a
checkpoint, or read validation/final-test profile data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from go2arx5_vla.data.stage22_snapshot_lock import (
    EXPECTED_INFERENCE_SEEDS,
    load_snapshot_lock,
    snapshot_lock_file_sha256,
)


FROZEN_LAUNCHER_GIT_SHA = "deb920a070141e6a32a88cd0da6719828647e1e6"
EXPECTED_BASE_MODEL_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_R1_TRAINABLE_PARAMETERS = 1_419_078_784
EXPECTED_TRAINING_SEED = 2222
EXPECTED_MAX_STEPS = 50
EXPECTED_SAVE_STEPS = 25
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_ACTION_HORIZON = 40
EXPECTED_ACTION_PERIOD_S = 0.05
EXPECTED_ACTION_PERIOD_PROVENANCE = "TEST_ONLY_SIM_VISUAL_REACH_V0"
EXPECTED_COSMOS_REPO_ID = "nvidia/Cosmos-Reason2-2B"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_MODULE_ROLES = ("projector", "state", "action", "diffusion", "visual", "llm", "vlln")
EXPECTED_TRAINABLE_ROLES = {
    "R0": {"projector", "state", "action"},
    "R1": {"projector", "state", "action", "diffusion"},
}
EXPECTED_CHECKPOINT_LINEAGE_FIELDS = {
    "base_model_tree_sha256",
    "train_fixture_tree_sha256",
    "split_lock_sha256",
    "modality_config_sha256",
    "groot_repo_sha",
    "cosmos_snapshot_revision",
    "cosmos_tree_sha256",
    "source_run_manifest_sha256",
    "checkpoint_tree_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve(path: str | Path, *, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _arg(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _has(command: list[str], name: str) -> bool:
    return name in command


def _check(
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


def _run_manifest_checks(
    checks: dict[str, dict[str, Any]],
    *,
    recipe: str,
    run: Mapping[str, Any],
    command: list[str],
    expected_groot_sha: str,
    expected_base_path: str | None,
    expected_dataset_path: str | None,
    expected_modality_path: str | None,
) -> None:
    expected_diffusion = recipe == "R1"
    _check(checks, f"{recipe}_schema", run.get("schema") == "STAGE20S_GR00T_TRAINING_RUN_V1", run.get("schema"), "STAGE20S_GR00T_TRAINING_RUN_V1")
    _check(checks, f"{recipe}_status", run.get("status") == "PASS", run.get("status"), "PASS")
    _check(checks, f"{recipe}_simulation_only", run.get("simulation_only") is True, run.get("simulation_only"), True)
    _check(checks, f"{recipe}_hardware_disabled", run.get("hardware_control") is False, run.get("hardware_control"), False)
    _check(checks, f"{recipe}_horizon_40", run.get("action_horizon") == EXPECTED_ACTION_HORIZON, run.get("action_horizon"), EXPECTED_ACTION_HORIZON)
    _check(checks, f"{recipe}_manifest_max_steps", run.get("max_steps") == EXPECTED_MAX_STEPS, run.get("max_steps"), EXPECTED_MAX_STEPS)
    _check(
        checks,
        f"{recipe}_timing_semantics",
        run.get("action_sample_period_s") == EXPECTED_ACTION_PERIOD_S
        and run.get("action_sample_period_provenance") == EXPECTED_ACTION_PERIOD_PROVENANCE,
        [run.get("action_sample_period_s"), run.get("action_sample_period_provenance")],
        [EXPECTED_ACTION_PERIOD_S, EXPECTED_ACTION_PERIOD_PROVENANCE],
    )
    _check(checks, f"{recipe}_diffusion_flag", run.get("tune_diffusion_model") is expected_diffusion, run.get("tune_diffusion_model"), expected_diffusion)
    _check(checks, f"{recipe}_projector_flag", run.get("tune_projector") is True, run.get("tune_projector"), True)
    _check(checks, f"{recipe}_visual_frozen", run.get("tune_visual") is False, run.get("tune_visual"), False)
    _check(checks, f"{recipe}_llm_frozen", run.get("tune_llm") is False, run.get("tune_llm"), False)
    _check(checks, f"{recipe}_vlln_frozen", run.get("tune_vlln") is False, run.get("tune_vlln"), False)
    _check(checks, f"{recipe}_base_index_present", bool(run.get("base_model_index_sha256")), run.get("base_model_index_sha256"), "non-empty SHA256")
    _check(checks, f"{recipe}_groot_sha", run.get("groot_repo_sha") == expected_groot_sha, run.get("groot_repo_sha"), expected_groot_sha)
    _check(checks, f"{recipe}_base_path", expected_base_path is None or run.get("base_model_path") == expected_base_path, run.get("base_model_path"), expected_base_path or "same across R0/R1")
    _check(checks, f"{recipe}_dataset_path", expected_dataset_path is None or run.get("dataset_dir") == expected_dataset_path, run.get("dataset_dir"), expected_dataset_path or "same across R0/R1")
    _check(checks, f"{recipe}_modality_path", expected_modality_path is None or run.get("modality_config") == expected_modality_path, run.get("modality_config"), expected_modality_path or "same across R0/R1")

    launcher_tokens = {Path(token).name for token in command}
    _check(checks, f"{recipe}_direct_frozen_launcher", "stage20s_train_gr00t.py" in launcher_tokens and "stage22_train_recipes.py" not in launcher_tokens, sorted(launcher_tokens), "direct stage20s_train_gr00t.py only")
    _check(checks, f"{recipe}_max_steps", _arg(command, "--max-steps") == str(EXPECTED_MAX_STEPS), _arg(command, "--max-steps"), str(EXPECTED_MAX_STEPS))
    _check(checks, f"{recipe}_save_steps", _arg(command, "--save-steps") == str(EXPECTED_SAVE_STEPS), _arg(command, "--save-steps"), str(EXPECTED_SAVE_STEPS))
    _check(checks, f"{recipe}_learning_rate", _arg(command, "--learning-rate") == str(EXPECTED_LEARNING_RATE), _arg(command, "--learning-rate"), str(EXPECTED_LEARNING_RATE))
    _check(checks, f"{recipe}_training_seed", _arg(command, "--seed") == str(EXPECTED_TRAINING_SEED), _arg(command, "--seed"), str(EXPECTED_TRAINING_SEED))
    _check(checks, f"{recipe}_local_files_only", _has(command, "--local-files-only"), _has(command, "--local-files-only"), True)
    _check(checks, f"{recipe}_diffusion_command_flag", _has(command, "--tune-diffusion-model") is expected_diffusion, _has(command, "--tune-diffusion-model"), expected_diffusion)


def _module_inventory_check(
    checks: dict[str, dict[str, Any]],
    *,
    recipe: str,
    inventory: Mapping[str, Any],
    run_manifest_sha256: str,
) -> None:
    _check(checks, f"{recipe}_inventory_schema", inventory.get("schema") == "STAGE22_TRAINABLE_MODULE_INVENTORY_V1", inventory.get("schema"), "STAGE22_TRAINABLE_MODULE_INVENTORY_V1")
    _check(checks, f"{recipe}_inventory_status", inventory.get("status") == "PASS", inventory.get("status"), "PASS")
    _check(checks, f"{recipe}_inventory_recipe", inventory.get("recipe") == recipe, inventory.get("recipe"), recipe)
    _check(checks, f"{recipe}_inventory_source_manifest", inventory.get("source_run_manifest_sha256") == run_manifest_sha256, inventory.get("source_run_manifest_sha256"), run_manifest_sha256)
    modules = inventory.get("modules")
    by_role = {
        item.get("role"): item
        for item in modules
        if isinstance(item, Mapping) and item.get("role")
    } if isinstance(modules, list) else {}
    _check(
        checks,
        f"{recipe}_inventory_roles_complete",
        isinstance(modules, list)
        and len(modules) == len(EXPECTED_MODULE_ROLES)
        and set(by_role) == set(EXPECTED_MODULE_ROLES),
        sorted(by_role),
        list(EXPECTED_MODULE_ROLES),
    )
    expected_trainable = EXPECTED_TRAINABLE_ROLES[recipe]
    role_checks = []
    for role in EXPECTED_MODULE_ROLES:
        item = by_role.get(role, {})
        expected = role in expected_trainable
        count = item.get("trainable_parameter_count")
        role_checks.append(
            item.get("name")
            and item.get("all_requires_grad") is expected
            and isinstance(item.get("parameter_count"), int)
            and item.get("parameter_count") >= 0
            and isinstance(count, int)
            and ((count > 0) if expected else count == 0)
        )
    _check(checks, f"{recipe}_inventory_requires_grad_roles", all(role_checks), {role: by_role.get(role) for role in EXPECTED_MODULE_ROLES}, "role-level requires_grad matches recipe")
    total = inventory.get("total_trainable_parameters")
    sum_count = sum(int(item.get("trainable_parameter_count", 0)) for item in by_role.values() if isinstance(item.get("trainable_parameter_count"), int))
    expected_total = EXPECTED_R1_TRAINABLE_PARAMETERS if recipe == "R1" else sum_count
    _check(checks, f"{recipe}_inventory_total_count", total == sum_count and (recipe != "R1" or total == expected_total), total, expected_total if recipe == "R1" else "sum of module counts")


def audit_direct_h2(
    *,
    r0_run_path: Path,
    r1_run_path: Path,
    execution_provenance_path: Path,
    validation_snapshot_manifest_path: Path,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    provenance = _load_json(execution_provenance_path)
    provenance_root = execution_provenance_path.parent
    _check(checks, "provenance_schema", provenance.get("schema") == "STAGE22_H2_DIRECT_EXECUTION_PROVENANCE_V1", provenance.get("schema"), "STAGE22_H2_DIRECT_EXECUTION_PROVENANCE_V1")
    _check(checks, "provenance_status", provenance.get("status") == "H2_DIRECT_LAUNCH_COMPLETE", provenance.get("status"), "H2_DIRECT_LAUNCH_COMPLETE")
    _check(checks, "provenance_simulation_only", provenance.get("simulation_only") is True, provenance.get("simulation_only"), True)
    _check(checks, "provenance_hardware_disabled", provenance.get("hardware_control") is False, provenance.get("hardware_control"), False)
    _check(checks, "frozen_launcher_commit", provenance.get("frozen_launcher_git_sha") == FROZEN_LAUNCHER_GIT_SHA, provenance.get("frozen_launcher_git_sha"), FROZEN_LAUNCHER_GIT_SHA)
    _check(checks, "frozen_training_contract", [provenance.get(key) for key in ("training_seed", "max_steps", "save_steps", "learning_rate", "action_horizon", "action_sample_period_s", "action_sample_period_provenance", "local_files_only")] == [EXPECTED_TRAINING_SEED, EXPECTED_MAX_STEPS, EXPECTED_SAVE_STEPS, EXPECTED_LEARNING_RATE, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_PERIOD_S, EXPECTED_ACTION_PERIOD_PROVENANCE, True], [provenance.get(key) for key in ("training_seed", "max_steps", "save_steps", "learning_rate", "action_horizon", "action_sample_period_s", "action_sample_period_provenance", "local_files_only")], "frozen direct-launch contract")

    cosmos = provenance.get("cosmos") if isinstance(provenance.get("cosmos"), Mapping) else {}
    h1 = provenance.get("h1") if isinstance(provenance.get("h1"), Mapping) else {}
    _check(checks, "h1_optimizer_step_completed", h1.get("status") == "PASS" and h1.get("optimizer_step_completed") is True, h1, "status=PASS and optimizer_step_completed=true")
    _check(checks, "cosmos_identity", [cosmos.get("repo_id"), cosmos.get("resolved_snapshot_revision"), cosmos.get("tree_sha256")] == [EXPECTED_COSMOS_REPO_ID, EXPECTED_COSMOS_REVISION, EXPECTED_COSMOS_TREE_SHA256], [cosmos.get("repo_id"), cosmos.get("resolved_snapshot_revision"), cosmos.get("tree_sha256")], [EXPECTED_COSMOS_REPO_ID, EXPECTED_COSMOS_REVISION, EXPECTED_COSMOS_TREE_SHA256])

    lock = load_snapshot_lock(validation_snapshot_manifest_path)
    lock_sha = snapshot_lock_file_sha256(validation_snapshot_manifest_path)
    _check(checks, "validation_snapshot_lock_sha256", provenance.get("validation_snapshot_manifest_sha256") == lock_sha, provenance.get("validation_snapshot_manifest_sha256"), lock_sha)
    _check(checks, "validation_snapshot_ids_hash", provenance.get("validation_snapshot_ids_sha256") == lock.get("snapshot_ids_sha256"), provenance.get("validation_snapshot_ids_sha256"), lock.get("snapshot_ids_sha256"))
    _check(checks, "EXACT_SEED_GATE_2223_2224_2225", provenance.get("inference_seeds") == list(EXPECTED_INFERENCE_SEEDS), provenance.get("inference_seeds"), list(EXPECTED_INFERENCE_SEEDS))
    _check(checks, "VALIDATION_SNAPSHOT_COMPLETENESS_GATE", provenance.get("validation_snapshot_ids") == lock.get("snapshot_ids") and lock.get("snapshot_count") == 6 and lock.get("expected_overlap_count") == 5, provenance.get("validation_snapshot_ids"), lock.get("snapshot_ids"))

    _check(checks, "BASE_TREE_HASH_GATE", provenance.get("base_model_tree_sha256") == EXPECTED_BASE_MODEL_TREE_SHA256, provenance.get("base_model_tree_sha256"), EXPECTED_BASE_MODEL_TREE_SHA256)
    train_tree = provenance.get("train_fixture_tree_sha256")
    _check(checks, "train_fixture_tree_hash_present", isinstance(train_tree, str) and len(train_tree) == 64, train_tree, "non-empty SHA256")
    _check(checks, "modality_hash_present", isinstance(provenance.get("modality_config_sha256"), str) and len(provenance.get("modality_config_sha256")) == 64, provenance.get("modality_config_sha256"), "non-empty SHA256")
    _check(checks, "split_lock_hash_present", isinstance(provenance.get("split_lock_sha256"), str) and len(provenance.get("split_lock_sha256")) == 64, provenance.get("split_lock_sha256"), "non-empty SHA256")
    expected_groot_sha = provenance.get("groot_repo_sha")
    _check(checks, "groot_sha_present", isinstance(expected_groot_sha, str) and len(expected_groot_sha) == 40, expected_groot_sha, "40-character SHA")

    gpu = provenance.get("gpu") if isinstance(provenance.get("gpu"), Mapping) else {}
    _check(checks, "SAME_GPU_UUID_GATE", bool(gpu.get("uuid")) and gpu.get("r0_uuid") == gpu.get("r1_uuid") == gpu.get("uuid"), {key: gpu.get(key) for key in ("uuid", "r0_uuid", "r1_uuid")}, "one shared UUID")
    runtime = provenance.get("runtime") if isinstance(provenance.get("runtime"), Mapping) else {}
    _check(checks, "same_runtime_identity", all(runtime.get(key) is not None for key in ("python", "torch", "cuda", "device_name", "memory_total_mib")), runtime, "complete runtime identity")

    run_paths = {"R0": r0_run_path.resolve(strict=True), "R1": r1_run_path.resolve(strict=True)}
    manifests: dict[str, dict[str, Any]] = {}
    run_hashes: dict[str, str] = {}
    commands: dict[str, list[str]] = {}
    provenance_runs = provenance.get("runs") if isinstance(provenance.get("runs"), Mapping) else {}
    for recipe, path in run_paths.items():
        run = _load_json(path)
        manifests[recipe] = run
        run_hash = _sha256(path)
        run_hashes[recipe] = run_hash
        record = provenance_runs.get(recipe) if isinstance(provenance_runs, Mapping) else {}
        record = record if isinstance(record, Mapping) else {}
        _check(checks, f"{recipe}_run_manifest_hash", record.get("run_manifest_sha256") == run_hash, record.get("run_manifest_sha256"), run_hash)
        command = [str(value) for value in record.get("command", [])]
        commands[recipe] = command
        _check(checks, f"{recipe}_command_present", bool(command), command, "direct launcher command")

    r0 = manifests["R0"]
    r1 = manifests["R1"]
    common_base = r0.get("base_model_path") if r0.get("base_model_path") == r1.get("base_model_path") else None
    common_dataset = r0.get("dataset_dir") if r0.get("dataset_dir") == r1.get("dataset_dir") else None
    common_modality = r0.get("modality_config") if r0.get("modality_config") == r1.get("modality_config") else None
    _run_manifest_checks(checks, recipe="R0", run=r0, command=commands["R0"], expected_groot_sha=expected_groot_sha, expected_base_path=common_base, expected_dataset_path=common_dataset, expected_modality_path=common_modality)
    _run_manifest_checks(checks, recipe="R1", run=r1, command=commands["R1"], expected_groot_sha=expected_groot_sha, expected_base_path=common_base, expected_dataset_path=common_dataset, expected_modality_path=common_modality)
    _check(
        checks,
        "same_base_model_path",
        common_base is not None,
        [r0.get("base_model_path"), r1.get("base_model_path")],
        "equal",
    )
    _check(checks, "same_base_index_sha256", bool(r0.get("base_model_index_sha256")) and r0.get("base_model_index_sha256") == r1.get("base_model_index_sha256"), [r0.get("base_model_index_sha256"), r1.get("base_model_index_sha256")], "equal non-empty SHA256")
    _check(checks, "same_dataset_path", common_dataset is not None, [r0.get("dataset_dir"), r1.get("dataset_dir")], "equal")
    _check(
        checks,
        "same_modality_config_path",
        common_modality is not None,
        [r0.get("modality_config"), r1.get("modality_config")],
        "equal",
    )
    _check(checks, "same_groot_repo_sha", bool(expected_groot_sha) and r0.get("groot_repo_sha") == r1.get("groot_repo_sha") == expected_groot_sha, [r0.get("groot_repo_sha"), r1.get("groot_repo_sha")], expected_groot_sha)
    runtime_match_rows = []
    for recipe, run in (("R0", r0), ("R1", r1)):
        run_runtime = run.get("runtime") if isinstance(run.get("runtime"), Mapping) else {}
        telemetry = run.get("cuda_telemetry") if isinstance(run.get("cuda_telemetry"), Mapping) else {}
        runtime_match_rows.append(
            {
                "recipe": recipe,
                "python": run_runtime.get("python") == runtime.get("python"),
                "torch": telemetry.get("torch_version") == runtime.get("torch"),
                "cuda": telemetry.get("cuda_version") == runtime.get("cuda"),
                "device_name": telemetry.get("device_name") == runtime.get("device_name"),
                "memory_total_mib": (
                    telemetry.get("before", {}).get("total_memory_bytes")
                    == int(runtime.get("memory_total_mib", -1)) * 1024 * 1024
                ),
            }
        )
    _check(
        checks,
        "same_runtime_identity_in_run_manifests",
        all(all(value for key, value in row.items() if key != "recipe") for row in runtime_match_rows),
        runtime_match_rows,
        "both direct run manifests match frozen runtime",
    )

    for recipe, run_path in run_paths.items():
        record = provenance_runs.get(recipe) if isinstance(provenance_runs, Mapping) else {}
        record = record if isinstance(record, Mapping) else {}
        inventory_value = record.get("module_inventory")
        inventory_path = _resolve(inventory_value, base=provenance_root) if inventory_value else None
        inventory = None if inventory_path is None else _load_json(inventory_path)
        inventory_sha = None if inventory_path is None else _sha256(inventory_path)
        _check(checks, f"{recipe}_module_inventory_hash", inventory is not None and record.get("module_inventory_sha256") == inventory_sha, record.get("module_inventory_sha256"), inventory_sha)
        _check(checks, f"{recipe}_provenance_gpu_uuid", record.get("gpu_uuid") == gpu.get("uuid"), record.get("gpu_uuid"), gpu.get("uuid"))
        if inventory is not None:
            _module_inventory_check(checks, recipe=recipe, inventory=inventory, run_manifest_sha256=run_hashes[recipe])

    lineage_contract = provenance.get("checkpoint50_lineage_contract")
    lineage_contract = lineage_contract if isinstance(lineage_contract, Mapping) else {}
    _check(checks, "CHECKPOINT50_LINEAGE_GATE", lineage_contract.get("required_global_step") == 50 and set(lineage_contract.get("required_fields", [])) == EXPECTED_CHECKPOINT_LINEAGE_FIELDS, lineage_contract, {"required_global_step": 50, "required_fields": sorted(EXPECTED_CHECKPOINT_LINEAGE_FIELDS)})

    blocking = [name for name, value in checks.items() if not value["passed"]]
    required_labels = {
        "DIRECT_LAUNCH_AUDIT_SUPPORTED": not blocking,
        "EXACT_SEED_GATE_2223_2224_2225": checks.get("EXACT_SEED_GATE_2223_2224_2225", {}).get("passed", False),
        "VALIDATION_SNAPSHOT_COMPLETENESS_GATE": checks.get("VALIDATION_SNAPSHOT_COMPLETENESS_GATE", {}).get("passed", False),
        "BASE_TREE_HASH_GATE": checks.get("BASE_TREE_HASH_GATE", {}).get("passed", False),
        "TRAINABLE_MODULE_INVENTORY_GATE": all(
            value.get("passed", False)
            for name, value in checks.items()
            if "inventory" in name or "requires_grad_roles" in name or "total_count" in name
        ),
        "SAME_GPU_UUID_GATE": checks.get("SAME_GPU_UUID_GATE", {}).get("passed", False),
        "CHECKPOINT50_LINEAGE_GATE": checks.get("CHECKPOINT50_LINEAGE_GATE", {}).get("passed", False),
    }
    return {
        "schema": "STAGE22_H2_DIRECT_LAUNCH_AUDIT_V1",
        "status": "PASS" if not blocking else "FAIL",
        "execution_provenance": str(execution_provenance_path),
        "r0_run_manifest": str(r0_run_path),
        "r1_run_manifest": str(r1_run_path),
        "validation_snapshot_manifest": str(validation_snapshot_manifest_path),
        "run_manifest_sha256": run_hashes,
        "lineage": {
            "base_model_tree_sha256": provenance.get("base_model_tree_sha256"),
            "train_fixture_tree_sha256": provenance.get("train_fixture_tree_sha256"),
            "split_lock_sha256": provenance.get("split_lock_sha256"),
            "modality_config_sha256": provenance.get("modality_config_sha256"),
            "groot_repo_sha": expected_groot_sha,
            "cosmos_snapshot_revision": cosmos.get("resolved_snapshot_revision"),
            "cosmos_tree_sha256": cosmos.get("tree_sha256"),
            "source_run_manifest_sha256": run_hashes,
            "checkpoint_tree_sha256": None,
        },
        "required_gate_labels": required_labels,
        "checks": checks,
        "blocking_checks": blocking,
        "claim_boundary": "direct frozen-launch H2 integrity only; no profile qualification or checkpoint selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-run", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--execution-provenance", type=Path, required=True)
    parser.add_argument("--validation-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_direct_h2(
        r0_run_path=args.r0_run.resolve(),
        r1_run_path=args.r1_run.resolve(),
        execution_provenance_path=args.execution_provenance.resolve(),
        validation_snapshot_manifest_path=args.validation_snapshot_manifest.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
