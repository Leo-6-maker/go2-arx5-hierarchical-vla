#!/usr/bin/env python3
"""Bootstrap the recovered-primary TRAIN24 evaluation only.

This entrypoint intentionally has no training child or training fallback.  It
is safe to invoke by absolute path from a foreign CWD with an empty
``PYTHONPATH``; the evaluator performs the H200 admission before importing
Torch/GR00T and consumes only the contract-bound 612326 recovered input.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
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


LAUNCHER_SCHEMA = "VANDA_STAGE45E_L1_RECOVERED_EVAL_LAUNCHER_V2"
EVAL_SCRIPT = REPO / "scripts" / "stage45e_l1_train24_eval.py"
FORBIDDEN_IMPORTS = {"torch", "transformers", "gr00t", "cuda"}


class RecoveredEvalLauncherError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveredEvalLauncherError(message)


def _run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    _require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def _write_fresh(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"launcher record must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    markers = {
        "repo_from_file": str(REPO) == str(Path(__file__).resolve().parents[1]),
        "absolute_eval_path": "str(EVAL_SCRIPT)" in source,
        "foreign_cwd": "cwd=str(Path(os.getenv" in source,
        "empty_pythonpath": '"PYTHONPATH": ""' in source,
        "recovered_input_binding": "--recovered-eval-input" in source,
        "no_training_child": ("TRAIN_" + "SCRIPT =") not in source,
        "no_retry": "automatic_retry" in source and "False" in source,
        "admission_before_child": "checkout_observation" in source and "static_guard" in source,
        "no_top_level_model_import": not forbidden,
    }
    status = "PASS" if not forbidden and all(markers.values()) else "FAIL"
    return {
        "schema": "VANDA_STAGE45E_L1_RECOVERED_EVAL_LAUNCHER_STATIC_GUARD_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden,
        "markers": markers,
        "foreign_cwd_supported": True,
        "empty_pythonpath_supported": True,
        "recovered_input_v2_required": True,
        "training_child": False,
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
    _require(root == str(REPO), f"launch checkout root drift: {root}")
    _require(branch == admission.BRANCH, f"launch branch drift: {branch}")
    _require(head == expected_head, f"launch HEAD drift: {head} != {expected_head}")
    _require(not status, "launch checkout is dirty, including untracked files")
    return {
        "observed_repo_root": root,
        "observed_branch": branch,
        "observed_head": head,
        "observed_clean_worktree": True,
    }


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "STAGE45E_L1_AUTHORITY_TOKEN": admission.AUTHORITY_TOKEN,
            "STAGE45E_EXPECTED_HEAD": os.environ.get("STAGE45E_L1_EXPECTED_HEAD", ""),
        }
    )
    return environment


def _parse_result(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"evaluation result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(
        value.get("status") == admission.FULL_PASS_TOKEN
        and value.get("token") == admission.FULL_PASS_TOKEN,
        "recovered evaluation did not complete",
    )
    _require(value.get("training") is False, "evaluation result claims training")
    _require(value.get("call_budget", {}).get("total_model_side_calls") == 360, "evaluation total budget drift")
    return value


def execute(
    expected_head: str,
    training_root: Path,
    recovered_eval_input: Path,
    output_root: Path,
) -> int:
    _require(os.environ.get("PBS_JOBID"), "--execute requires a PBS allocation")
    _require(os.environ.get("STAGE45E_L1_AUTHORITY_TOKEN") == admission.AUTHORITY_TOKEN, "authority token is not set")
    _require(expected_head and len(expected_head) == 40, "full expected HEAD is required")
    _require(Path(sys.executable).resolve() == Path(os.environ["STAGE45E_L1_PYTHON"]).resolve(), "launcher Python mismatch")
    _require(not admission.model_imports_present(), "model module is already imported at launcher entry")
    guard = static_guard()
    _require(guard["status"] == "PASS", "recovered eval launcher static guard failed")
    checkout = checkout_observation(expected_head)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        _require(os.environ.get(key) == "1", f"offline mode is not asserted: {key}")
    _require(training_root.is_dir() and not training_root.is_symlink(), "recovered training root is unavailable")
    _require(recovered_eval_input.is_file() and not recovered_eval_input.is_symlink(), "recovered V2 input is unavailable")
    _require(not output_root.exists() and not output_root.is_symlink(), "evaluation output root is not fresh")

    environment = child_environment()
    started = time.time()
    argv = [
        str(sys.executable),
        str(EVAL_SCRIPT),
        "--execute",
        "--training-root",
        str(training_root),
        "--recovered-eval-input",
        str(recovered_eval_input),
        "--output-root",
        str(output_root),
        "--expected-head",
        expected_head,
    ]
    completed = subprocess.run(
        argv,
        cwd=str(Path(os.getenv("TMPDIR", "/tmp")).resolve()),
        env=environment,
        check=False,
    )
    record_path = output_root.with_name(output_root.name + ".launcher.json")
    record = {
        "schema": LAUNCHER_SCHEMA,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "execution_head": expected_head,
        "checkout": checkout,
        "argv": argv,
        "returncode": completed.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "training_root": str(training_root),
        "recovered_eval_input": str(recovered_eval_input),
        "output_root": str(output_root),
        "training_child_started": False,
        "training": False,
        "automatic_retry": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if completed.returncode == 0:
        result = _parse_result(output_root / "stage45e_l1_eval_result.json")
        record["evaluation_result"] = result
        record["token"] = admission.FULL_PASS_TOKEN
    _write_fresh(record_path, record)
    _require(completed.returncode == 0, f"recovered evaluation child failed with returncode={completed.returncode}")
    print(admission.FULL_PASS_TOKEN, flush=True)
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


def plan() -> dict[str, Any]:
    guard = static_guard()
    _require(guard["status"] == "PASS", "recovered eval launcher static guard failed")
    return {
        "schema": LAUNCHER_SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED",
        "static_guard": guard,
        "eval_entrypoint": str(EVAL_SCRIPT),
        "recovered_input_v2_required": True,
        "foreign_cwd": True,
        "empty_pythonpath": True,
        "training_child": False,
        "automatic_retry": False,
        "call_budget": admission.EVAL_CALL_BUDGET,
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
    parser.add_argument("--training-root", type=Path, default=os.environ.get("STAGE45E_L1_RECOVERY_TRAINING_ROOT"))
    parser.add_argument("--recovered-eval-input", type=Path, default=os.environ.get("STAGE45E_L1_RECOVERED_EVAL_INPUT"))
    parser.add_argument("--output-root", type=Path, default=os.environ.get("STAGE45E_L1_EVAL_OUTPUT_ROOT"))
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        _require(args.expected_head is not None, "--expected-head is required")
        _require(args.training_root is not None, "--training-root is required")
        _require(args.recovered_eval_input is not None, "--recovered-eval-input is required")
        _require(args.output_root is not None, "--output-root is required")
        return execute(
            args.expected_head,
            Path(args.training_root).expanduser().resolve(),
            Path(args.recovered_eval_input).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
        )
    except Exception as exc:
        print(f"STAGE45E_L1_RECOVERED_EVAL_LAUNCHER_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
