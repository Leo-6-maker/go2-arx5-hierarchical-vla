#!/usr/bin/env python3
"""Run the frozen Corelab Stage45D-D1 smoke then full adapter exactly once.

This orchestrator is CPU/static until explicitly authorized by the caller.  A
successful smoke is bound to the same checkout HEAD before the full adapter is
started; there is no retry, adaptive extension, or fallback device path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "scripts" / "stage45d_d1_corelab_runtime_adapter.py"
FULL = REPO / "scripts" / "stage45d_d1_corelab_full_runtime_adapter.py"
SMOKE_PASS_TOKEN = "STAGE45D_D1_4080S_SMOKE_PASS_AUTO_CONTINUE_FULL"
FULL_TOKEN = "STAGE45D_D1_CORELAB_FULL_EXECUTION_COMPLETE_FOR_PI_REVIEW"


class OrchestratorError(RuntimeError):
    pass


def _json_stdout(completed: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise OrchestratorError(f"{label} failed: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"{label} did not return JSON") from exc
    if not isinstance(value, dict):
        raise OrchestratorError(f"{label} returned a non-object")
    return value


def _head() -> str:
    completed = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if completed.returncode:
        raise OrchestratorError("cannot determine checkout HEAD")
    return completed.stdout.strip()


def run(smoke_binding: Path, full_binding: Path, *, python: str, authority_token: str) -> dict[str, Any]:
    start_head = _head()
    env = dict(os.environ)
    env["STAGE45D_D1_CORELAB_PREFLIGHT_RUNTIME"] = "authorized"
    env["STAGE45D_D1_CORELAB_PREFLIGHT_AUTHORITY_TOKEN"] = authority_token
    smoke_completed = subprocess.run(
        [python, str(SMOKE), "--execute", "--runtime-binding", str(smoke_binding)],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
    )
    smoke = _json_stdout(smoke_completed, "smoke")
    if smoke.get("status") != "PASS" or smoke.get("token") != SMOKE_PASS_TOKEN:
        raise OrchestratorError("smoke did not produce the required auto-continuation PASS token")
    smoke_head = smoke.get("execution_head")
    if smoke_head != start_head:
        raise OrchestratorError("smoke execution HEAD differs from orchestrator start HEAD")
    if _head() != start_head:
        raise OrchestratorError("checkout HEAD changed between smoke and full")
    full_completed = subprocess.run(
        [python, str(FULL), "--execute", "--runtime-binding", str(full_binding)],
        cwd=str(REPO), env=env, capture_output=True, text=True, check=False,
    )
    full = _json_stdout(full_completed, "full")
    if full.get("status") != "PASS" or full.get("token") != FULL_TOKEN:
        raise OrchestratorError("full adapter did not produce the required completion token")
    return {"schema": "CORELAB_STAGE45D_D1_CORELAB_ORCHESTRATOR_RESULT_V1", "status": "PASS", "execution_head": start_head, "smoke": smoke, "full": full}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-binding", type=Path, required=True)
    parser.add_argument("--full-binding", type=Path, required=True)
    parser.add_argument("--authority-token", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.smoke_binding, args.full_binding, python=args.python, authority_token=args.authority_token), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"CORELAB_STAGE45D_D1_ORCHESTRATOR_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
