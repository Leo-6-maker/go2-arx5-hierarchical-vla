#!/usr/bin/env python3
"""Materialize observed Hopper admission, run one D1 smoke, then exact full D1.

The launcher is intentionally standard-library-only at module import time.
Torch, GR00T, Transformers, and CUDA Python modules are confined to the two
child adapter processes, each of which validates the freshly observed runtime
document before its late model import.  There is no retry path.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
SMOKE_ADAPTER = SCRIPTS / "stage45d_d1_corelab_runtime_adapter.py"
FULL_ADAPTER = SCRIPTS / "stage45d_d1_corelab_full_runtime_adapter.py"
SMOKE_PROFILE = REPO / "manifests" / "STAGE45D_D1_CORELAB_4080S_PREFLIGHT_PROFILE_V1.json"
FULL_PROFILE = REPO / "manifests" / "STAGE45D_D1_CORELAB_FULL_EXECUTION_PROFILE_V1.json"
EXPECTED_BRANCH = "stage45d/vanda-h200-d1-execution"
SOURCE_SCIENTIFIC_HEAD = "404d345ed81972fb84c5f25aa7e48245f4ec7c8c"
PI_GATE_COMMENT = 5629752688
PR51_POINTER_COMMENT = 5629755293
ISSUE50_POINTER_COMMENT = 5629756230
AUTHORITY_TOKEN = "PI_GATE_STATE_0117"
SMOKE_PASS_TOKEN = "STAGE45D_D1_H200_SMOKE_PASS_AUTO_CONTINUE_FULL"
FULL_PASS_TOKEN = "STAGE45D_D1_VANDA_H200_FULL_EXECUTION_COMPLETE_FOR_PI_REVIEW"
MIN_H200_VRAM_BYTES = 80 * 1024**3
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}
PATH_KEYS = {
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
    "modality_config",
    "base_processor_config",
    "base_processor_statistics",
    "structured_1000_statistics",
    "full24_runtime_statistics",
    "golden_release_manifest",
    "groot_runtime_manifest",
    "full24_manifest",
    "checkpoint_integrity_audit",
    "runtime_admission",
}


class LauncherError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LauncherError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {argv[0]}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def checkout_observation(expected_head: str) -> dict[str, Any]:
    root = run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    require(Path(root).resolve() == REPO.resolve(), "observed repository root mismatch")
    require(branch == EXPECTED_BRANCH, "observed execution branch mismatch")
    require(head == expected_head, "observed execution HEAD mismatch")
    require(not status, "execution worktree is not clean, including untracked files")
    return {
        "observed_repo_root": str(REPO.resolve()),
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
        "git_status_mode": "--porcelain=v1 --untracked-files=all",
    }


def gpu_observation() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    raw = run_text(query)
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, "unexpected nvidia-smi observation format")
        rows.append({
            "index": int(fields[0]),
            "gpu_name": fields[1],
            "gpu_uuid": fields[2],
            "total_vram_bytes": int(fields[3]) * 1024**2,
            "free_vram_bytes": int(fields[4]) * 1024**2,
            "driver_version": fields[5],
        })
    require(rows, "nvidia-smi returned no visible GPU")
    selected = rows[0]
    require("H200" in selected["gpu_name"].upper(), "selected visible GPU is not an H200")
    require(selected["total_vram_bytes"] >= MIN_H200_VRAM_BYTES, "selected H200 is below the 80GiB floor")
    return {
        "host": socket.gethostname(),
        **selected,
        "selected_device": "cuda:0",
        "visible_gpu_count": len(rows),
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "pbs_jobname": os.environ.get("PBS_JOBNAME"),
        "offline_mode": True,
        "nvidia_smi_query": query,
        "nvidia_smi_stdout_sha256": hashlib.sha256((raw + "\n").encode("utf-8")).hexdigest(),
    }


def runtime_identity(gpu: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        require(os.environ.get(key) == "1", f"offline environment is not asserted: {key}")
    require(os.environ.get("PBS_JOBID"), "PBS_JOBID is absent")
    return {
        "schema": "VANDA_STAGE45D_D1_H200_OBSERVED_RUNTIME_V1",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pbs_job_id": os.environ["PBS_JOBID"],
        "gpu_uuid": gpu["gpu_uuid"],
        "gpu_name": gpu["gpu_name"],
        "offline_environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }


def read_paths(path: Path) -> dict[str, str]:
    require(path.is_file() and not path.is_symlink(), "runtime paths JSON must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and set(value) == PATH_KEYS, "runtime paths JSON key set mismatch")
    for key, item in value.items():
        require(isinstance(item, str) and item.startswith("/"), f"runtime path must be absolute: {key}")
    return dict(value)


def static_guard() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    return {
        "schema": "VANDA_STAGE45D_D1_H200_LAUNCHER_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "FAIL",
        "top_level_forbidden_imports": forbidden,
        "repo_from_file": str(REPO),
        "foreign_cwd_supported": True,
        "empty_pythonpath_supported": True,
        "automatic_retry": False,
    }


def _import_static_runtime() -> tuple[Any, Any, Any, Any, Any]:
    for path in (SRC, SCRIPTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import stage26_h1_c1_fixed_noise_local_flow as stage26
    import stage38_structured_offline_evaluation as stage38
    import stage45c_h200_eval_runtime_adapter as stage45c
    import stage45d_d1_corelab_full_runtime_adapter as full
    import stage45d_d1_runtime_admission as admission

    found = admission.model_imports_present()
    require(not found, "model-side module imported during observed admission: " + ",".join(found))
    return admission, full, stage45c, stage26, stage38


def build_document(
    *,
    admission: Any,
    paths: Mapping[str, str],
    output_root: Path,
    checkout: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gpu: Mapping[str, Any],
    authority_token: str,
) -> dict[str, Any]:
    counters = {"model_construction": 0, "model_forward": 0, "gpu": 0, "pbs": 1, "training_steps": 0}
    binding = {
        "schema": "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V2",
        "execution_authorized": True,
        "pi_authority_token": authority_token,
        "protocol_provenance_head": admission.PROTOCOL_PROVENANCE_HEAD,
        "execution_checkout": {
            "binding_mode": "OBSERVED_RUNTIME_ONLY",
            "authorized_repo_root": checkout["observed_repo_root"],
            "authorized_branch": checkout["observed_branch"],
            "authorized_head": checkout["observed_head"],
            "authorized_clean_worktree": True,
            "protocol_provenance_head_may_substitute": False,
        },
        "runtime_identity": dict(runtime),
        "source_identities": copy.deepcopy(admission.SOURCE_IDENTITIES),
        "checkpoint_identities": copy.deepcopy(admission.CHECKPOINT_IDENTITIES),
        "statistics_routes": copy.deepcopy(admission.STATS_ROUTES),
        "dataset_identity": copy.deepcopy(admission.DATASET_IDENTITY),
        "sampler": copy.deepcopy(admission.SAMPLER_CONTRACT),
        "host_gpu_runtime": dict(gpu),
        "call_budget": copy.deepcopy(admission.CALL_BUDGET),
        "target_xyz_in_model_input": False,
        "locked_boundaries": copy.deepcopy(admission.LOCKED_BOUNDARIES),
        "zero_counters": counters,
        "output": {"root": str(output_root), "fresh_required": True},
        "paths": {**dict(paths), "output_root": str(output_root)},
        "device_policy": {
            "selected_device": "cuda:0",
            "selected_gpu_role": "HOPPER_H200_SINGLE_GPU",
            "minimum_vram_bytes": MIN_H200_VRAM_BYTES,
            "allow_16gb_secondary_fallback": False,
            "allow_multi_gpu": False,
            "allow_sharding": False,
            "allow_data_parallel": False,
            "oom_retry": False,
            "device_observation_required_before_model_import": True,
        },
    }
    observed = {
        "execution_checkout": dict(checkout),
        "runtime_identity": dict(runtime),
        "source_identities": copy.deepcopy(admission.SOURCE_IDENTITIES),
        "checkpoint_identities": copy.deepcopy(admission.CHECKPOINT_IDENTITIES),
        "statistics_routes": copy.deepcopy(admission.STATS_ROUTES),
        "dataset_identity": copy.deepcopy(admission.DATASET_IDENTITY),
        "sampler": copy.deepcopy(admission.SAMPLER_CONTRACT),
        "host_gpu_runtime": dict(gpu),
        "call_budget": copy.deepcopy(admission.CALL_BUDGET),
        "target_xyz_in_model_input": False,
        "locked_boundaries": copy.deepcopy(admission.LOCKED_BOUNDARIES),
        "zero_counters": counters,
        "output": {"root": str(output_root), "fresh": True},
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "offline_mode": True,
    }
    return {"binding": binding, "observed": observed}


def materialize_smoke_sidecars(
    root: Path,
    paths: Mapping[str, Path],
    artifacts: Mapping[str, Any],
    full: Any,
    stage26: Any,
    stage38: Any,
) -> dict[str, Any]:
    import numpy as np

    labels = full._normalized_labels(artifacts["snapshots"], paths, stage26, stage38, "FULL24@2000")
    record = artifacts["snapshots"][0]
    snapshot = stage26._load_snapshot(Path(record["path"]))
    sidecar_root = root / "smoke_sidecars"
    sidecar_root.mkdir(parents=True, exist_ok=False)
    normalized_path = sidecar_root / "normalized_x1.npy"
    physical_path = sidecar_root / "expert_physical_action.npy"
    np.save(normalized_path, np.asarray(labels[0], dtype=np.float32), allow_pickle=False)
    np.save(physical_path, np.asarray(snapshot.physical_action, dtype=np.float32), allow_pickle=False)
    for path in (normalized_path, physical_path):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    return {
        "index": 0,
        "window_id": "stage22_expert_red_target_train_a:expert-0000",
        "target_id": "red_target_train_a",
        "split": "TRAIN_ONLY",
        "target_xyz_in_model_input": False,
        "normalized_label_path": str(normalized_path),
        "normalized_label_sha256": sha256_file(normalized_path),
        "expert_physical_action_path": str(physical_path),
        "expert_physical_action_sha256": sha256_file(physical_path),
        "snapshot": {
            "state_v1": snapshot.state_v1.as_array().tolist(),
            "source_timestamps": np.asarray(snapshot.future_timestamps, dtype=np.float64).tolist(),
            "observation_timestamp": float(snapshot.observation_timestamp),
            "anchor_world_m": np.asarray(snapshot.state_v1.eef_xyz, dtype=np.float64).tolist(),
            "target_world_m": np.asarray(snapshot.target_position_world_m, dtype=np.float64).tolist(),
            "source_frame": snapshot.source_frame,
            "tcp_semantic_id": snapshot.tcp_semantic_id,
        },
    }


def child_environment(paths: Mapping[str, str], evidence_root: Path, authority_token: str) -> dict[str, str]:
    namespace = evidence_root / "runtime_namespace"
    hub = namespace / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    model_link = hub / "models--nvidia--Cosmos-Reason2-2B"
    os.symlink(paths["cosmos_root"], model_link, target_is_directory=True)
    env = dict(os.environ)
    env.update({
        "HF_HOME": str(namespace / "huggingface"),
        "HF_HUB_CACHE": str(hub),
        "HUGGINGFACE_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub),
        "HF_DATASETS_CACHE": str(namespace / "huggingface" / "datasets"),
        "TORCH_HOME": str(namespace / "torch"),
        "XDG_CACHE_HOME": str(namespace / "xdg_cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
        # The frozen checkpoint config names the pinned Cosmos repository by
        # Hub ID.  The outer checkpoint load's local_files_only flag is not
        # forwarded through the custom GR00T model constructor, so activate
        # the pinned source's local-first bridge for that nested load.  Keep
        # the Mistral metadata guard enabled as well: tokenizer construction
        # must remain fully offline even when the model is addressed by ID.
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "STAGE45D_D1_CORELAB_PREFLIGHT_RUNTIME": "authorized",
        "STAGE45D_D1_CORELAB_PREFLIGHT_AUTHORITY_TOKEN": authority_token,
    })
    return env


def run_child(label: str, argv: Sequence[str], *, env: Mapping[str, str], record_root: Path) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(list(argv), cwd=str(record_root), env=dict(env), capture_output=True, text=True, check=False)
    record = {
        "label": label,
        "argv": list(argv),
        "returncode": completed.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json_durable(record_root / f"{label}_child_process.json", record)
    require(completed.returncode == 0, f"{label} child failed: {completed.stderr.strip()}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LauncherError(f"{label} child did not return JSON") from exc
    require(isinstance(result, dict), f"{label} child returned a non-object")
    return result


def execute(paths_file: Path, evidence_root: Path, expected_head: str, authority_token: str, repair_cycle: int, smoke_attempt: int) -> dict[str, Any]:
    require(authority_token == AUTHORITY_TOKEN, "PI0117 authority token mismatch")
    require(type(repair_cycle) is int and 0 <= repair_cycle <= 2, "repair cycle exceeds PI0117 budget")
    require(type(smoke_attempt) is int and 1 <= smoke_attempt <= 6, "smoke attempt exceeds PI0117 budget")
    require(not evidence_root.exists(), "evidence root is not fresh")
    require(evidence_root.parent.is_dir(), "evidence parent does not exist")
    checkout = checkout_observation(expected_head)
    gpu = gpu_observation()
    runtime = runtime_identity(gpu)
    paths = read_paths(paths_file)
    evidence_root.mkdir(parents=False, exist_ok=False)
    phase = "OBSERVED_ADMISSION"
    try:
        admission, full, stage45c, stage26, stage38 = _import_static_runtime()
        path_objects = {key: Path(value) for key, value in paths.items()}
        artifacts = full._verify_runtime_artifacts(path_objects, stage45c)
        require(not admission.model_imports_present(), "artifact verification imported a model-side module")
        write_json_durable(evidence_root / "observed_checkout.json", checkout)
        write_json_durable(evidence_root / "observed_gpu_runtime.json", gpu)
        write_json_durable(evidence_root / "observed_artifact_verification.json", artifacts)

        smoke_root = evidence_root / "smoke"
        smoke_document = build_document(
            admission=admission,
            paths=paths,
            output_root=smoke_root,
            checkout=checkout,
            runtime=runtime,
            gpu=gpu,
            authority_token=authority_token,
        )
        pre_model = admission.validate_future_admission(
            smoke_document["binding"], smoke_document["observed"],
            authority_token=authority_token, imported_modules=sys.modules,
        )
        write_json_durable(evidence_root / "observed_pre_model_admission.json", pre_model)
        smoke_document["binding"]["selected_window"] = materialize_smoke_sidecars(
            evidence_root, path_objects, artifacts, full, stage26, stage38,
        )
        smoke_document["binding"]["execution_control"] = {
            "phase": "SMOKE", "repair_cycle": repair_cycle, "smoke_attempt": smoke_attempt,
            "automatic_retry": False,
        }
        smoke_document_path = evidence_root / "runtime_document_smoke.json"
        write_json_durable(smoke_document_path, smoke_document)
        env = child_environment(paths, evidence_root, authority_token)

        phase = "SMOKE_MODEL_EXECUTION"
        smoke = run_child(
            "smoke",
            [sys.executable, str(SMOKE_ADAPTER), "--execute", "--profile", str(SMOKE_PROFILE), "--runtime-binding", str(smoke_document_path)],
            env=env,
            record_root=evidence_root,
        )
        require(smoke.get("status") == "PASS" and smoke.get("token") == SMOKE_PASS_TOKEN, "smoke did not produce PI0117 PASS token")
        require(smoke.get("execution_head") == expected_head, "smoke HEAD mismatch")
        smoke_seal_path = smoke_root / "preflight_seal.json"
        require(smoke_seal_path.is_file(), "smoke seal is absent")
        print(SMOKE_PASS_TOKEN, flush=True)

        phase = "FULL_ADMISSION"
        checkout_after_smoke = checkout_observation(expected_head)
        require(checkout_after_smoke == checkout, "checkout changed after smoke")
        full_root = evidence_root / "full"
        full_document = build_document(
            admission=admission,
            paths=paths,
            output_root=full_root,
            checkout=checkout_after_smoke,
            runtime=runtime,
            gpu=gpu,
            authority_token=authority_token,
        )
        full_document["binding"]["execution_control"] = {
            "phase": "FULL",
            "repair_cycle": repair_cycle,
            "smoke_attempt": smoke_attempt,
            "automatic_retry": False,
            "smoke_pass_head": expected_head,
            "smoke_seal_path": str(smoke_seal_path),
            "smoke_seal_sha256": sha256_file(smoke_seal_path),
        }
        full_document["binding"]["full_execution"] = {
            "checkpoint_order": ["STRUCTURED@1000", "FULL24@1000", "FULL24@2000"],
            "window_count": 24,
            "window_order": "target-major then snapshot-major",
            "noise_seeds": [2223, 2224, 2225],
            "t_values": [0.06776241796562847, 0.36966943557751086, 0.7837719744658148],
            "t_buckets": [67, 369, 783],
            "target_xyz_in_model_input": False,
        }
        admission.validate_future_admission(
            full_document["binding"], full_document["observed"],
            authority_token=authority_token, imported_modules=sys.modules,
        )
        full_document_path = evidence_root / "runtime_document_full.json"
        write_json_durable(full_document_path, full_document)

        phase = "FULL_MODEL_EXECUTION"
        full_result = run_child(
            "full",
            [sys.executable, str(FULL_ADAPTER), "--execute", "--profile", str(FULL_PROFILE), "--runtime-binding", str(full_document_path)],
            env=env,
            record_root=evidence_root,
        )
        require(full_result.get("status") == "PASS" and full_result.get("token") == FULL_PASS_TOKEN, "full D1 did not produce PI0117 completion token")
        result = {
            "schema": "VANDA_STAGE45D_D1_H200_SMOKE_TO_FULL_LAUNCHER_SEAL_V1",
            "status": "PASS",
            "token": FULL_PASS_TOKEN,
            "authority": {
                "issue_11_comment": PI_GATE_COMMENT,
                "pr_51_comment": PR51_POINTER_COMMENT,
                "issue_50_comment": ISSUE50_POINTER_COMMENT,
            },
            "source_scientific_head": SOURCE_SCIENTIFIC_HEAD,
            "execution_head": expected_head,
            "branch": EXPECTED_BRANCH,
            "pbs_job_id": os.environ["PBS_JOBID"],
            "node": socket.gethostname(),
            "gpu": gpu,
            "smoke_seal_path": str(smoke_seal_path),
            "smoke_seal_sha256": sha256_file(smoke_seal_path),
            "full_seal_path": str(full_root / "full_seal.json"),
            "full_seal_sha256": sha256_file(full_root / "full_seal.json"),
            "repair_cycle": repair_cycle,
            "smoke_attempt": smoke_attempt,
            "automatic_retry": False,
            "locked_boundaries": dict(admission.LOCKED_BOUNDARIES),
        }
        write_json_durable(evidence_root / "launcher_seal.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema": "VANDA_STAGE45D_D1_H200_LAUNCHER_FAILURE_SEAL_V1",
            "status": "FAIL",
            "phase": phase,
            "reason": f"{type(exc).__name__}: {exc}",
            "execution_head": expected_head,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "node": socket.gethostname(),
            "repair_cycle": repair_cycle,
            "smoke_attempt": smoke_attempt,
            "automatic_retry": False,
        }
        write_json_durable(evidence_root / "LAUNCHER_FAILURE_SEAL.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--authority-token", default=AUTHORITY_TOKEN)
    parser.add_argument("--repair-cycle", type=int, default=0)
    parser.add_argument("--smoke-attempt", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            guard = static_guard()
            require(guard["status"] == "PASS", "launcher static guard failed")
            result = {**guard, "source_scientific_head": SOURCE_SCIENTIFIC_HEAD, "smoke_token": SMOKE_PASS_TOKEN, "full_token": FULL_PASS_TOKEN}
        else:
            require(args.paths is not None and args.evidence_root is not None and args.expected_head, "execute requires paths, evidence root, and expected HEAD")
            result = execute(args.paths, args.evidence_root, args.expected_head, args.authority_token, args.repair_cycle, args.smoke_attempt)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"VANDA_STAGE45D_D1_H200_LAUNCHER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
