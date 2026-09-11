#!/usr/bin/env python3
"""Observed-runtime bootstrap for the Stage45D-D2B Hopper execution.

This entrypoint resolves the repository from its own absolute file path, so
it is safe from a foreign CWD with an empty PYTHONPATH.  It observes the
actual checkout and H200 first, materializes a binding, runs one smoke child,
and only then runs the exact full study.  There is no retry or fallback path.
"""

from __future__ import annotations

import argparse
import ast
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
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45d_d1_corelab_runtime_adapter as common  # noqa: E402
import stage45d_d2b_runtime_admission as admission  # noqa: E402
import stage45d_d2b_solver_discretization as engine  # noqa: E402


LAUNCHER_SCHEMA = "VANDA_STAGE45D_D2B_H200_LAUNCHER_V1"
BRANCH = admission.BRANCH
BASE_EXECUTION_HEAD = admission.BASE_EXECUTION_HEAD
AUTHORITY_TOKEN = admission.AUTHORITY_TOKEN
PI_GATE_COMMENT = admission.PI_GATE_COMMENT
PR52_POINTER_COMMENT = admission.PR52_POINTER_COMMENT
DEVICE = "cuda:0"
GPU_ROLE = "HOPPER_H200_SINGLE_GPU"
MIN_VRAM_BYTES = 80 * 1024**3
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}


class D2BLauncherError(RuntimeError):
    """Raised when the observed Hopper launch cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D2BLauncherError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    _require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    common.write_json_durable(path, value)


def write_inventory(evidence_root: Path) -> tuple[Path, str]:
    inventory_path = evidence_root / "inventory" / "SHA256SUMS"
    inventory_path.parent.mkdir(parents=True, exist_ok=False)
    files = sorted(path for path in evidence_root.rglob("*") if path.is_file() and path != inventory_path)
    payload = ("\n".join(f"{sha256_file(path)}  {path.relative_to(evidence_root).as_posix()}" for path in files) + "\n").encode("utf-8")
    temporary = inventory_path.with_name(f".{inventory_path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, inventory_path)
    return inventory_path, hashlib.sha256(payload).hexdigest()


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    return {
        "schema": "VANDA_STAGE45D_D2B_H200_LAUNCHER_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "FAIL",
        "top_level_forbidden_imports": forbidden,
        "repo_from_file": str(REPO),
        "foreign_cwd_supported": True,
        "empty_pythonpath_supported": True,
        "observed_runtime_binding": True,
        "admission_before_model_import": True,
        "automatic_retry": False,
        "branch": BRANCH,
        "base_execution_head": BASE_EXECUTION_HEAD,
    }


def checkout_observation(expected_head: str) -> dict[str, Any]:
    root = _run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = _run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = _run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = _run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    _require(root == str(REPO), "observed repository root differs from launcher repository")
    _require(branch == BRANCH, f"observed branch is not D2B branch: {branch}")
    _require(head == expected_head, f"observed HEAD differs from requested HEAD: {head} != {expected_head}")
    _require(not status, "execution checkout is not clean, including untracked files")
    ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", BASE_EXECUTION_HEAD, head], check=False)
    _require(ancestor.returncode == 0, "D2B HEAD is not based on accepted D2A HEAD")
    return {
        "observed_repo_root": root,
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
        "clean_check_includes_untracked": True,
        "base_execution_head": BASE_EXECUTION_HEAD,
    }


def gpu_observation() -> dict[str, Any]:
    query = "index,name,uuid,memory.total"
    raw = _run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, f"unexpected nvidia-smi row: {line}")
        rows.append({"index": int(parts[0]), "gpu_name": parts[1], "gpu_uuid": parts[2], "total_vram_bytes": int(parts[3]) * 1024**2})
    _require(len(rows) == 1 and rows[0]["index"] == 0, "D2B requires exactly one visible GPU at index 0")
    selected = rows[0]
    _require("H200" in selected["gpu_name"].upper(), "observed GPU is not H200")
    _require(selected["total_vram_bytes"] >= MIN_VRAM_BYTES, "observed H200 is below the 80GiB floor")
    return {
        "schema": "VANDA_STAGE45D_D2B_H200_OBSERVED_GPU_V1",
        "host": socket.gethostname(),
        "gpu_name": selected["gpu_name"],
        "gpu_uuid": selected["gpu_uuid"],
        "selected_device": DEVICE,
        "selected_gpu_role": GPU_ROLE,
        "total_vram_bytes": selected["total_vram_bytes"],
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
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is absent")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        _require(os.environ.get(key) == "1", f"offline environment is not asserted: {key}")
    return {
        "schema": "VANDA_STAGE45D_D2B_H200_OBSERVED_RUNTIME_V1",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pbs_job_id": os.environ["PBS_JOBID"],
        "gpu_uuid": gpu["gpu_uuid"],
        "gpu_name": gpu["gpu_name"],
        "selected_device": DEVICE,
        "offline_environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
    }


def read_paths(path: Path) -> dict[str, str]:
    payload = _json(path)
    _require(set(payload) == set(admission.RUNTIME_PATH_KEYS), "D2B runtime path key set mismatch")
    for key, value in payload.items():
        _require(isinstance(value, str) and value.startswith("/"), f"D2B runtime path is not absolute: {key}")
    return {key: str(value) for key, value in payload.items()}


def _boundary() -> dict[str, Any]:
    return {
        "model_construction": "RUNTIME_ONLY",
        "model_forward": "RUNTIME_ONLY",
        "gpu": "RUNTIME_ONLY",
        "h200": 1,
        "pbs": 1,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "D2B_solver_comparison": "RUNTIME_ONLY",
    }


def build_document(*, paths: Mapping[str, str], output_root: Path, checkout: Mapping[str, Any], runtime: Mapping[str, Any], gpu: Mapping[str, Any], phase: str, repair_cycle: int, smoke_attempt: int, smoke_seal_path: Path | None = None, smoke_seal_sha256: str | None = None) -> dict[str, Any]:
    bound_paths = {**dict(paths), "output_root": str(output_root)}
    control: dict[str, Any] = {"phase": phase, "repair_cycle": repair_cycle, "smoke_attempt": smoke_attempt, "automatic_retry": False}
    if phase == "FULL":
        _require(smoke_seal_path is not None and smoke_seal_sha256 is not None, "full binding needs smoke seal")
        control.update({"smoke_seal_path": str(smoke_seal_path), "smoke_seal_sha256": smoke_seal_sha256})
    zero = {
        "model_construction": 0, "model_forward": 0, "gpu": 0, "pbs": 1,
        "training_steps": 0, "backward_steps": 0, "optimizer_steps": 0,
        "checkpoint_write": 0, "action_head_calls": 0, "conditioning_passes": 0,
    }
    common_fields = {
        "schema": admission.RUNTIME_BINDING_SCHEMA,
        "stage": admission.STAGE,
        "pi_gate": PI_GATE_COMMENT,
        "execution_authorized": True,
        "pi_authority_token": AUTHORITY_TOKEN,
        "base_execution_head": BASE_EXECUTION_HEAD,
        "execution_checkout": {
            "binding_mode": "OBSERVED_RUNTIME_ONLY",
            "authorized_repo_root": checkout["observed_repo_root"],
            "authorized_branch": checkout["observed_branch"],
            "authorized_head": checkout["observed_head"],
            "authorized_clean_worktree": True,
        },
        "runtime_identity": dict(runtime),
        "host_gpu_runtime": dict(gpu),
        "source_identities": admission.SOURCE_IDENTITIES,
        "checkpoint_identities": admission.CHECKPOINT_IDENTITIES,
        "statistics_routes": admission.STATS_ROUTES,
        "dataset_identity": admission.DATASET_IDENTITY,
        "sampler": admission.SAMPLER_CONTRACT,
        "call_budget": admission.CALL_BUDGET,
        "contract": admission.CONTRACT,
        "target_xyz_in_model_input": False,
        "locked_boundaries": admission.LOCKED_BOUNDARIES,
        "zero_counters": zero,
        "execution_control": control,
        "execution_boundary": _boundary(),
        "output": {"root": str(output_root), "fresh_required": True},
        "paths": bound_paths,
        "device_policy": {
            "selected_device": DEVICE,
            "selected_gpu_role": GPU_ROLE,
            "minimum_vram_bytes": MIN_VRAM_BYTES,
            "allow_multi_gpu": False,
            "allow_sharding": False,
            "allow_data_parallel": False,
            "oom_retry": False,
            "offline_only": True,
            "device_observation_required_before_model_import": True,
        },
    }
    observed = {
        "execution_checkout": dict(checkout),
        "runtime_identity": dict(runtime),
        "host_gpu_runtime": dict(gpu),
        "source_identities": admission.SOURCE_IDENTITIES,
        "checkpoint_identities": admission.CHECKPOINT_IDENTITIES,
        "statistics_routes": admission.STATS_ROUTES,
        "dataset_identity": admission.DATASET_IDENTITY,
        "sampler": admission.SAMPLER_CONTRACT,
        "call_budget": admission.CALL_BUDGET,
        "contract": admission.CONTRACT,
        "target_xyz_in_model_input": False,
        "locked_boundaries": admission.LOCKED_BOUNDARIES,
        "zero_counters": zero,
        "execution_control": control,
        "execution_boundary": _boundary(),
        "output": {"root": str(output_root), "fresh": True},
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "offline_mode": True,
    }
    return {"binding": common_fields, "observed": observed}


def child_environment(paths: Mapping[str, str], evidence_root: Path) -> dict[str, str]:
    namespace = evidence_root / "runtime_namespace"
    hub = namespace / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    os.symlink(paths["cosmos_root"], hub / "models--nvidia--Cosmos-Reason2-2B", target_is_directory=True)
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
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "STAGE45D_D2B_RUNTIME": "authorized",
        "STAGE45D_D2B_AUTHORITY_TOKEN": AUTHORITY_TOKEN,
    })
    return env


def _parse_child_json(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset in range(len(stdout) - 1, -1, -1):
        if stdout[offset] != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not stdout[offset + end :].strip():
            return value
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise D2BLauncherError("child did not emit a trailing JSON result")


def run_child(label: str, binding_path: Path, evidence_root: Path, env: Mapping[str, str]) -> dict[str, Any]:
    started = time.time()
    engine_path = REPO / "scripts" / "stage45d_d2b_solver_discretization.py"
    phase = "SMOKE" if label == "smoke" else "FULL"
    argv = [sys.executable, str(engine_path), "--execute", "--profile", str(engine.PROFILE_PATH), "--runtime-binding", str(binding_path), "--phase", phase]
    completed = subprocess.run(argv, cwd=str(evidence_root), env=dict(env), capture_output=True, text=True, check=False)
    record = {
        "schema": "VANDA_STAGE45D_D2B_CHILD_PROCESS_RECORD_V1",
        "label": label,
        "argv": argv,
        "returncode": completed.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json_durable(evidence_root / f"{label}_child_process.json", record)
    _require(completed.returncode == 0, f"{label} child failed: {completed.stderr.strip()}")
    return _parse_child_json(completed.stdout)


def execute(paths_file: Path, evidence_root: Path, expected_head: str, repair_cycle: int, smoke_attempt: int) -> dict[str, Any]:
    _require(os.environ.get("STAGE45D_D2B_AUTHORITY_TOKEN") == AUTHORITY_TOKEN, "D2B authority token is not set")
    _require(type(repair_cycle) is int and 0 <= repair_cycle <= 2, "D2B repair cycle out of range")
    _require(type(smoke_attempt) is int and 1 <= smoke_attempt <= 6, "D2B smoke attempt out of range")
    _require(not evidence_root.exists(), "D2B evidence root is not fresh")
    _require(evidence_root.parent.is_dir(), "D2B evidence parent is unavailable")
    checkout = checkout_observation(expected_head)
    _require(evidence_root.name == f"stage45d_d2b_solver_discretization_20260911_{expected_head[:8]}_{os.environ.get('PBS_JOBID')}", "D2B evidence root provenance name drift")
    paths = read_paths(paths_file)
    gpu = gpu_observation()
    runtime = runtime_identity(gpu)
    evidence_root.mkdir(parents=False, exist_ok=False)
    write_json_durable(evidence_root / "observed_checkout.json", checkout)
    write_json_durable(evidence_root / "observed_gpu_runtime.json", gpu)
    write_json_durable(evidence_root / "observed_runtime_identity.json", runtime)
    phase = "STATIC_ADMISSION"
    try:
        guard = static_guard()
        _require(guard["status"] == "PASS", "D2B launcher static guard failed")
        static = engine.validate_static_contract()
        _require(not admission.model_imports_present(), "model-side module imported in launcher admission")
        write_json_durable(evidence_root / "static_contract.json", static)
        env = child_environment(paths, evidence_root)

        phase = "SMOKE_MODEL_EXECUTION"
        smoke_root = evidence_root / "smoke"
        smoke_binding = build_document(paths=paths, output_root=smoke_root, checkout=checkout, runtime=runtime, gpu=gpu, phase="SMOKE", repair_cycle=repair_cycle, smoke_attempt=smoke_attempt)
        smoke_binding_path = evidence_root / "runtime_binding_smoke.json"
        write_json_durable(smoke_binding_path, smoke_binding)
        smoke = run_child("smoke", smoke_binding_path, evidence_root, env)
        _require(smoke.get("status") == "PASS" and smoke.get("token") == admission.SMOKE_PASS_TOKEN, "D2B smoke did not pass")
        _require(smoke.get("execution_head") == expected_head, "D2B smoke HEAD mismatch")
        smoke_seal_path = smoke_root / "d2b_smoke_seal.json"
        _require(smoke_seal_path.is_file(), "D2B smoke seal is absent")
        smoke_sha = sha256_file(smoke_seal_path)
        print(admission.SMOKE_PASS_TOKEN, flush=True)

        phase = "FULL_ADMISSION"
        checkout_after_smoke = checkout_observation(expected_head)
        _require(checkout_after_smoke == checkout, "D2B checkout changed after smoke")
        full_root = evidence_root / "full"
        full_binding = build_document(paths=paths, output_root=full_root, checkout=checkout_after_smoke, runtime=runtime, gpu=gpu, phase="FULL", repair_cycle=repair_cycle, smoke_attempt=smoke_attempt, smoke_seal_path=smoke_seal_path, smoke_seal_sha256=smoke_sha)
        full_binding_path = evidence_root / "runtime_binding_full.json"
        write_json_durable(full_binding_path, full_binding)

        phase = "FULL_MODEL_EXECUTION"
        full = run_child("full", full_binding_path, evidence_root, env)
        _require(full.get("status") == "PASS" and full.get("token") == admission.FULL_PASS_TOKEN, "D2B full execution did not pass")
        _require(full.get("execution_head") == expected_head, "D2B full HEAD mismatch")
        full_seal_path = full_root / "d2b_full_seal.json"
        _require(full_seal_path.is_file(), "D2B full seal is absent")
        inventory_path, inventory_sha = write_inventory(evidence_root)
        result = {
            "schema": "VANDA_STAGE45D_D2B_H200_LAUNCHER_SEAL_V1",
            "status": "PASS",
            "token": admission.FULL_PASS_TOKEN,
            "authority": {"issue_11_comment": PI_GATE_COMMENT, "pr_52_comment": PR52_POINTER_COMMENT},
            "branch": BRANCH,
            "base_execution_head": BASE_EXECUTION_HEAD,
            "execution_head": expected_head,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "node": socket.gethostname(),
            "gpu": gpu,
            "evidence_root": str(evidence_root),
            "smoke_seal_path": str(smoke_seal_path),
            "smoke_seal_sha256": smoke_sha,
            "full_seal_path": str(full_seal_path),
            "full_seal_sha256": sha256_file(full_seal_path),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "repair_cycle": repair_cycle,
            "smoke_attempt": smoke_attempt,
            "automatic_retry": False,
            "locked_boundaries": admission.LOCKED_BOUNDARIES,
        }
        write_json_durable(evidence_root / "launcher_seal.json", result)
        return result
    except Exception as exc:
        write_json_durable(evidence_root / "LAUNCHER_FAILURE_SEAL.json", {
            "schema": "VANDA_STAGE45D_D2B_H200_LAUNCHER_FAILURE_SEAL_V1",
            "status": "FAIL",
            "phase": phase,
            "reason": f"{type(exc).__name__}: {exc}",
            "execution_head": expected_head,
            "branch": BRANCH,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "node": socket.gethostname(),
            "model_execution_started": phase in {"SMOKE_MODEL_EXECUTION", "FULL_MODEL_EXECUTION"},
            "automatic_retry": False,
            "repair_cycle": repair_cycle,
            "smoke_attempt": smoke_attempt,
        })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--repair-cycle", type=int, default=0)
    parser.add_argument("--smoke-attempt", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            result = static_guard()
            _require(result["status"] == "PASS", "D2B launcher static guard failed")
            result.update({"smoke_token": admission.SMOKE_PASS_TOKEN, "full_token": admission.FULL_PASS_TOKEN})
        else:
            _require(args.paths is not None and args.evidence_root is not None and args.expected_head, "execute requires paths, evidence root, and expected HEAD")
            result = execute(args.paths, args.evidence_root, args.expected_head, args.repair_cycle, args.smoke_attempt)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"VANDA_STAGE45D_D2B_H200_LAUNCHER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
