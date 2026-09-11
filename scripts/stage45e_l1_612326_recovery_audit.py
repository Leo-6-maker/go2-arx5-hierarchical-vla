#!/usr/bin/env python3
"""Read-only integrity audit for the completed Stage45E-L1 job 612326.

This preparation surface exists because job 612326 completed the frozen
2000-step training lineage and failed only during post-training sidecar audit.
The auditor never imports Torch/GR00T, never mutates the historical training
root, and never launches evaluation.  It materializes a new acceptance record
from immutable filesystem evidence only.
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


TRAINING_HEAD = "377b50125a9127fe65d7f3db896f9396c7c3fbc2"
AUDIT_REPAIR_HEAD = "f77f9e02161068ffd29539ae70ed8b69fb7fb4d8"
BRANCH = "stage45e/l1-physical-error-aware-loss"
EXPECTED_PBS_JOBID = "612326.hopper-m-02"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
ACTIVE_SCALAR_COUNT = 190
ACTION_HORIZON = 40
ACTION_DIM = 132
ACCEPT_TOKEN = "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTED"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "transformers",
    "gr00t",
    "cuda",
    "subprocess",
    "numpy",
}


class RecoveryAuditError(RuntimeError):
    """Raised when immutable 612326 evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryAuditError(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON is unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
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


def _checkpoint(path: Path, step: int) -> dict[str, Any]:
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    require(path.is_dir() and not path.is_symlink(), f"checkpoint-{step} missing")
    require(all((path / name).is_file() and not (path / name).is_symlink() for name in required), f"checkpoint-{step} full-state closure missing")
    state = read_json(path / "trainer_state.json")
    require(state.get("global_step") == step, f"checkpoint-{step} global_step drift")
    entries = _tree_entries(path, f"stage45e_l1/checkpoint-{step}")
    require(len(entries) >= len(required), f"checkpoint-{step} tree is unexpectedly small")
    return {
        "path": str(path.resolve()),
        "global_step": step,
        "tree_sha256": _tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "required_state_files": list(required),
    }


def _events(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "training event log missing")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), "training event is not an object")
        rows.append(dict(value))
    names = [str(row.get("event")) for row in rows]
    required_names = [
        "PRE_MODEL_ADMISSION_PASS",
        "MODEL_IMPORT_COMPLETE",
        "ONE_BATCH_BACKWARD_CANARY_PASS",
        "TRAINING_START",
        "TRAINING_RETURNED",
    ]
    positions: list[int] = []
    for name in required_names:
        require(name in names, f"required training event missing: {name}")
        positions.append(names.index(name))
    require(positions == sorted(positions) and len(set(positions)) == len(positions), "training event order drift")
    training_start = rows[positions[3]]
    require(training_start.get("optimizer_steps") == 2000, "TRAINING_START optimizer-step budget drift")
    require(training_start.get("checkpoint_steps") == [1000, 2000], "TRAINING_START checkpoint budget drift")
    return {
        "sha256": sha256_file(path),
        "event_count": len(rows),
        "required_event_order": required_names,
        "training_returned": True,
    }


def _canary(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema") == "VANDA_STAGE45E_L1_ONE_BATCH_BACKWARD_CANARY_V1", "canary schema drift")
    require(value.get("status") == "PASS", "canary did not PASS")
    require(value.get("active_scalar_count") == ACTIVE_SCALAR_COUNT, "canary active mask drift")
    require(value.get("optimizer_steps") == 0 and value.get("scheduler_steps") == 0, "canary updated optimizer/scheduler")
    require(value.get("checkpoint_writes") == 0, "canary wrote a checkpoint")
    require(value.get("parameter_delta", {}).get("pass") is True, "canary mutated trainable parameters")
    require(value.get("stats_unchanged") is True and value.get("mask_unchanged") is True, "canary contract drift")
    require(value.get("normalization_unchanged") is True, "canary normalization drift")
    loss = value.get("loss")
    require(isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(float(loss)), "canary loss is not finite")
    runtime_dtype = value.get("action_loss_dtype")
    effective_weights = value.get("effective_xyz_loss_weights")
    weight_evidence = {
        "status": "CAPTURED" if runtime_dtype is not None and effective_weights is not None else "NOT_CAPTURED_IN_HISTORICAL_CANARY",
        "action_loss_dtype": runtime_dtype,
        "effective_xyz_loss_weights": effective_weights,
        "claim_boundary": "Do not infer runtime coefficient dtype/value from bf16=True alone when the historical canary did not serialize it.",
    }
    return {
        "sha256": sha256_file(path),
        "loss": float(loss),
        "active_scalar_count": ACTIVE_SCALAR_COUNT,
        "parameter_delta_pass": True,
        "optimizer_steps": 0,
        "runtime_weight_evidence": weight_evidence,
    }


def _sidecar(path: Path) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("source_dataset_untouched_by_factory") is True, "sidecar dataset mutation flag drift")
    require(value.get("upstream_generate_stats_called") is False, "sidecar indicates stats regeneration")
    statistics_payload = value.get("statistics")
    require(isinstance(statistics_payload, Mapping), "sidecar statistics payload missing")
    observed_payload_sha = canonical_sha(statistics_payload)
    require(observed_payload_sha == FULL24_STATS_PAYLOAD_SHA256, "FULL24 statistics payload drift")
    contract = value.get("contract")
    require(isinstance(contract, Mapping), "sidecar contract missing")
    require(contract.get("mask_active_scalar_count") == ACTIVE_SCALAR_COUNT, "sidecar mask count drift")
    active_mask = statistics_payload.get("active_mask")
    require(isinstance(active_mask, list) and len(active_mask) == ACTION_HORIZON, "sidecar active-mask horizon drift")
    require(all(isinstance(row, list) and len(row) == ACTION_DIM for row in active_mask), "sidecar active-mask dimension drift")
    flat = [item for row in active_mask for item in row]
    require(all(type(item) in (int, float) and item in (0, 1) for item in flat), "sidecar active mask is not binary")
    require(sum(flat) == ACTIVE_SCALAR_COUNT, "sidecar serialized active-mask count drift")
    return {
        "sha256": sha256_file(path),
        "statistics_payload_sha256": observed_payload_sha,
        "mask_shape": [ACTION_HORIZON, ACTION_DIM],
        "mask_active_scalar_count": ACTIVE_SCALAR_COUNT,
        "mask_binary": True,
    }


def _failure(path: Path, expected_pbs_jobid: str) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("schema") == "VANDA_STAGE45E_L1_FAILURE_SEAL_V1", "failure seal schema drift")
    require(value.get("status") == "FAIL", "historical seal is not a failure seal")
    require(value.get("phase") == "POST_TRAINING_AUDIT", "612326 did not fail in post-training audit")
    require(value.get("execution_head") == TRAINING_HEAD, "612326 training HEAD drift")
    require(value.get("training_started") is True, "failure seal does not prove training started")
    require(value.get("action_head_forwards") == 2000, "failure seal action-head training budget drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked data was read")
    require(value.get("wbc_steps") == 0 and value.get("simulator_steps") == 0 and value.get("hardware_steps") == 0, "forbidden downstream execution observed")
    observed_job = value.get("pbs_jobid")
    if observed_job is not None:
        require(observed_job == expected_pbs_jobid, "PBS job identity drift")
    return {
        "sha256": sha256_file(path),
        "phase": "POST_TRAINING_AUDIT",
        "execution_head": TRAINING_HEAD,
        "pbs_jobid": observed_job,
        "action_head_forwards": 2000,
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
        "schema": "STAGE45E_L1_612326_RECOVERY_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def plan() -> dict[str, Any]:
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_L1_612326_RECOVERY_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "training_head": TRAINING_HEAD,
        "audit_repair_head": AUDIT_REPAIR_HEAD,
        "expected_pbs_jobid": EXPECTED_PBS_JOBID,
        "future_accept_token": ACCEPT_TOKEN,
        "historical_root_mutation_allowed": False,
        "evaluation_execution": False,
        "static_guard": guard,
    }


def audit(training_root: Path, output_root: Path, expected_pbs_jobid: str) -> dict[str, Any]:
    training_root = Path(training_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    require(training_root.is_dir() and not training_root.is_symlink(), "training root unavailable or unsafe")
    require(output_root != training_root and not output_root.is_relative_to(training_root), "recovery output must not be inside historical training root")
    require(not output_root.exists() and not output_root.is_symlink(), "recovery output root must be fresh")
    require(not (training_root / "stage45e_l1_training_result.json").exists(), "historical root already contains a normal PASS training result; provenance is ambiguous")

    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    failure = _failure(training_root / "stage45e_l1_failure.json", expected_pbs_jobid)
    events = _events(training_root / "stage45e_l1_events.jsonl")
    canary = _canary(training_root / "stage45e_l1_backward_canary.json")
    sidecar = _sidecar(training_root / "experiment_cfg" / "structured_action_statistics.json")
    checkpoint_1000 = _checkpoint(training_root / "checkpoint-1000", 1000)
    checkpoint_2000 = _checkpoint(training_root / "checkpoint-2000", 2000)

    supporting_files = {}
    for relative in (
        "stage45e_l1_runtime_binding.json",
        "stage45e_l1_pre_model_admission.json",
        "stage45e_l1_input_audit.json",
    ):
        path = training_root / relative
        read_json(path)
        supporting_files[relative] = sha256_file(path)

    output_root.mkdir(parents=True, exist_ok=False)
    acceptance = {
        "schema": "STAGE45E_L1_612326_READONLY_TRAINING_ACCEPTANCE_V1",
        "status": "PASS_READONLY_TRAINING_ARTIFACT_INTEGRITY",
        "token": ACCEPT_TOKEN,
        "training_root": str(training_root),
        "historical_training_head": TRAINING_HEAD,
        "audit_repair_head": AUDIT_REPAIR_HEAD,
        "branch": BRANCH,
        "pbs_jobid": expected_pbs_jobid,
        "failure_seal": failure,
        "events": events,
        "backward_canary": canary,
        "structured_stats": sidecar,
        "checkpoint_1000": checkpoint_1000,
        "checkpoint_2000": checkpoint_2000,
        "supporting_file_sha256": supporting_files,
        "historical_root_mutated": False,
        "scientific_claim_boundary": {
            "training_completed": True,
            "post_training_audit_bug_only": True,
            "train24_evaluation_executed": False,
            "runtime_effective_weight_dtype_captured": canary["runtime_weight_evidence"]["status"] == "CAPTURED",
            "runtime_effective_weight_evidence_status": canary["runtime_weight_evidence"]["status"],
            "automatic_winner": False,
        },
        "execution_counters_for_this_audit": {
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "gpu": 0,
            "pbs": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": 0,
            "simulator": 0,
            "hardware": 0,
        },
        "next_action": "PI0127 may consume this acceptance token for the separate eval-only path; this auditor itself launches nothing.",
    }
    write_json(output_root / "stage45e_l1_612326_readonly_acceptance.json", acceptance)
    return acceptance


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--audit", action="store_true")
    value.add_argument("--training-root", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--expected-pbs-jobid", default=EXPECTED_PBS_JOBID)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.training_root is not None and args.output_root is not None, "--audit requires --training-root and --output-root")
    result = audit(args.training_root, args.output_root, args.expected_pbs_jobid)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryAuditError as exc:
        print(f"STAGE45E_L1_612326_RECOVERY_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
