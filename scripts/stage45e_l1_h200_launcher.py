#!/usr/bin/env python3
"""Observed-runtime bootstrap for the single authorized Stage45E-L1 job.

The launcher is the committed/tested qualification entrypoint.  It is safe to
invoke by absolute path from a foreign CWD with an empty ``PYTHONPATH``.  It
performs one static/observed-runtime admission, starts exactly one fresh-BASE
training child, and only after that child seals PASS starts the exact TRAIN24
E4 evaluation child.  There is no retry or alternate-resource path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
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

import stage45e_l1_runtime_admission as admission  # noqa: E402


LAUNCHER_SCHEMA = "VANDA_STAGE45E_L1_H200_LAUNCHER_V1"
PI_GATE = admission.PI_GATE_COMMENT
BRANCH = admission.BRANCH
DEVICE = admission.DEVICE
MIN_VRAM_BYTES = admission.MIN_VRAM_BYTES
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}


class L1LauncherError(RuntimeError):
    """Raised when the observed launch cannot continue fail-closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise L1LauncherError(message)


def _run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    _require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_durable(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    markers = {
        "repo_from_file": str(REPO) == str(Path(__file__).resolve().parents[1]),
        "absolute_child_paths": "str(TRAIN_SCRIPT)" in source and "str(EVAL_SCRIPT)" in source,
        "foreign_cwd": "cwd=str(Path(os.getenv" in source,
        "empty_pythonpath": '"PYTHONPATH": ""' in source,
        "training_then_eval": "run_child(\"train\"" in source and "run_child(\"eval\"" in source,
        "no_retry": "retry" in source and "automatic_retry" in source and "False" in source,
        "admission_before_child": "static_contract" in source and "checkout_observation" in source,
        "no_top_level_model_import": not forbidden_imports,
    }
    # Keep the marker intentionally source-based; plan-only never touches the
    # node GPU or constructs a model.
    status = "PASS" if not forbidden_imports and all(markers.values()) else "FAIL"
    return {
        "schema": "VANDA_STAGE45E_L1_H200_LAUNCHER_STATIC_GUARD_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden_imports,
        "markers": markers,
        "foreign_cwd_supported": True,
        "empty_pythonpath_supported": True,
        "automatic_retry": False,
        "pbs": 0,
        "h200": 0,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
    }


def checkout_observation(expected_head: str) -> dict[str, Any]:
    root = _run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = _run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = _run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = _run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    _require(root == str(REPO), f"repository root drift: {root}")
    _require(branch == BRANCH, f"branch drift: {branch}")
    _require(head == expected_head, f"HEAD drift: {head} != {expected_head}")
    _require(not status, "launch checkout is dirty, including untracked files")
    ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", admission.BASE_EXECUTION_HEAD, head], check=False)
    _require(ancestor.returncode == 0 and head != admission.BASE_EXECUTION_HEAD, "L1 HEAD ancestry drift")
    return {
        "observed_repo_root": root,
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
        "base_execution_head": admission.BASE_EXECUTION_HEAD,
    }


def gpu_observation() -> dict[str, Any]:
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required for L1 launch")
    query = "index,name,uuid,memory.total,memory.used"
    raw = _run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(",")]
        _require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4])})
    _require(len(rows) == 1 and rows[0]["index"] == 0, f"L1 requires one GPU at index 0: {rows!r}")
    _require("H200" in rows[0]["name"].upper(), f"L1 launch requires H200: {rows[0]['name']}")
    _require(rows[0]["memory_total_mib"] * 1024**2 >= MIN_VRAM_BYTES, "H200 is below 80 GiB floor")
    return {
        "schema": "VANDA_STAGE45E_L1_LAUNCH_OBSERVED_H200_V1",
        "host": socket.gethostname(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "selected_device": DEVICE,
        "gpu_role": admission.GPU_ROLE,
        "gpu_name": rows[0]["name"],
        "gpu_uuid": rows[0]["uuid"],
        "total_vram_bytes": rows[0]["memory_total_mib"] * 1024**2,
        "memory_used_mib_at_launch": rows[0]["memory_used_mib"],
        "visible_gpu_count": 1,
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_query": query,
        "nvidia_smi_stdout_sha256": hashlib.sha256((raw + "\n").encode("utf-8")).hexdigest(),
    }


def _parse_result(path: Path, expected_token: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"child result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"child result is not an object: {path}")
    _require(value.get("status") == expected_token and value.get("token") == expected_token, f"child did not PASS: {path}")
    return value


TRAIN_SCRIPT = REPO / "scripts" / "stage45e_l1_h200_train.py"
EVAL_SCRIPT = REPO / "scripts" / "stage45e_l1_train24_eval.py"


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "STAGE45E_L1_AUTHORITY_TOKEN": admission.AUTHORITY_TOKEN,
        "STAGE45E_EXPECTED_HEAD": os.environ.get("STAGE45E_L1_EXPECTED_HEAD", ""),
    })
    return environment


def run_child(label: str, argv: Sequence[str], record_path: Path, environment: Mapping[str, str], *, output_root: Path) -> int:
    started = time.time()
    completed = subprocess.run(list(argv), cwd=str(Path(os.getenv("TMPDIR", "/tmp")).resolve()), env=dict(environment), check=False)
    record = {
        "schema": "VANDA_STAGE45E_L1_CHILD_PROCESS_RECORD_V1",
        "label": label,
        "argv": list(argv),
        "returncode": completed.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "output_root": str(output_root),
    }
    write_json_durable(record_path, record)
    return completed.returncode


def execute(expected_head: str) -> int:
    _require(os.environ.get("STAGE45E_L1_AUTHORITY_TOKEN") == admission.AUTHORITY_TOKEN, "L1 authority token is not set")
    _require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required")
    _require(expected_head and len(expected_head) == 40, "expected HEAD is required")
    _require(not admission.model_imports_present(), "model module is already imported at launcher entry")
    guard = static_guard()
    _require(guard["status"] == "PASS", "launcher static guard failed")
    checkout = checkout_observation(expected_head)
    gpu = gpu_observation()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        _require(os.environ.get(key) == "1", f"offline mode is not asserted: {key}")
    _require(Path(sys.executable).resolve() == Path(os.environ["STAGE45E_L1_PYTHON"]).resolve(), "launcher Python differs from admitted Python")

    job_tag = os.environ["PBS_JOBID"].replace("/", "_")
    head8 = expected_head[:8]
    training_root = Path(f"/scratch/e1192548/stage45e_l1_h200_20260911_{head8}_{job_tag}")
    eval_root = Path(f"/scratch/e1192548/stage45e_l1_train24_eval_20260911_{head8}_{job_tag}")
    _require(not training_root.exists() and not eval_root.exists(), "L1 output roots must both be fresh")
    environment = child_environment()
    phase = "TRAIN_CHILD"
    try:
        train_argv = [
            str(sys.executable), str(TRAIN_SCRIPT), "--execute",
            "--output-root", str(training_root), "--expected-head", expected_head,
            "--python", str(sys.executable),
        ]
        train_rc = run_child("train", train_argv, training_root.with_name(training_root.name + ".child.json"), environment, output_root=training_root)
        _require(train_rc == 0, f"L1 training child failed with returncode={train_rc}")
        training_result = _parse_result(training_root / "stage45e_l1_training_result.json", admission.FULL_PASS_TOKEN)
        _require(training_result.get("execution_head") == expected_head, "training child HEAD drift")

        phase = "EVAL_CHILD"
        eval_argv = [
            str(sys.executable), str(EVAL_SCRIPT), "--execute",
            "--training-root", str(training_root), "--output-root", str(eval_root),
            "--expected-head", expected_head,
        ]
        eval_rc = run_child("eval", eval_argv, eval_root.with_name(eval_root.name + ".child.json"), environment, output_root=eval_root)
        _require(eval_rc == 0, f"L1 evaluation child failed with returncode={eval_rc}")
        evaluation_result = _parse_result(eval_root / "stage45e_l1_eval_result.json", admission.FULL_PASS_TOKEN)
        _require(evaluation_result.get("execution_head") == expected_head, "evaluation child HEAD drift")
        result = {
            "schema": LAUNCHER_SCHEMA,
            "status": "PASS",
            "token": admission.FULL_PASS_TOKEN,
            "pi_gate": PI_GATE,
            "authority_token": admission.AUTHORITY_TOKEN,
            "branch": BRANCH,
            "base_execution_head": admission.BASE_EXECUTION_HEAD,
            "execution_head": expected_head,
            "pbs_jobid": os.environ["PBS_JOBID"],
            "host": socket.gethostname(),
            "gpu": gpu,
            "training_root": str(training_root),
            "training_result": str(training_root / "stage45e_l1_training_result.json"),
            "training_result_sha256": sha256_file(training_root / "stage45e_l1_training_result.json"),
            "evaluation_root": str(eval_root),
            "evaluation_result": str(eval_root / "stage45e_l1_eval_result.json"),
            "evaluation_result_sha256": sha256_file(eval_root / "stage45e_l1_eval_result.json"),
            "checkout": checkout,
            "phases": ["PRE_MODEL_OBSERVED_ADMISSION", "FRESH_BASE_TRAINING", "TRAIN24_E4_EVALUATION"],
            "automatic_retry": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "n1": False,
            "n2": False,
            "n2_plus_l1": False,
            "d2b": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
        }
        write_json_durable(eval_root / "stage45e_l1_launcher_result.json", result)
        print(admission.FULL_PASS_TOKEN, flush=True)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        failure_root = training_root if training_root.exists() else training_root.parent
        try:
            write_json_durable(failure_root / f"{training_root.name}.LAUNCHER_FAILURE_SEAL.json", {
                "schema": "VANDA_STAGE45E_L1_LAUNCHER_FAILURE_SEAL_V1",
                "status": "FAIL",
                "phase": phase,
                "reason": f"{type(exc).__name__}: {exc}",
                "execution_head": expected_head,
                "branch": BRANCH,
                "pbs_jobid": os.environ.get("PBS_JOBID"),
                "model_execution_started": phase == "EVAL_CHILD" or phase == "TRAIN_CHILD",
                "training_child_started": phase in {"TRAIN_CHILD", "EVAL_CHILD"},
                "evaluation_child_started": phase == "EVAL_CHILD",
                "automatic_retry": False,
                "validation_e_read": False,
                "holdout_f_read": False,
            })
        except Exception:
            pass
        print(f"STAGE45E_L1_LAUNCHER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def plan() -> dict[str, Any]:
    guard = static_guard()
    _require(guard["status"] == "PASS", "launcher static guard failed")
    static = admission.validate_static_contract()
    return {
        "schema": LAUNCHER_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "static_guard": guard,
        "static_contract": static,
        "repo": str(REPO),
        "train_entrypoint": str(TRAIN_SCRIPT),
        "eval_entrypoint": str(EVAL_SCRIPT),
        "foreign_cwd": True,
        "empty_pythonpath": True,
        "training_then_evaluation": True,
        "automatic_retry": False,
        "pbs": 0,
        "h200": 0,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "return_token": admission.FULL_PASS_TOKEN,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-head", default=os.environ.get("STAGE45E_L1_EXPECTED_HEAD"))
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        _require(args.expected_head is not None, "--expected-head is required")
        return execute(args.expected_head)
    except Exception as exc:
        print(f"STAGE45E_L1_LAUNCHER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
