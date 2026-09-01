#!/usr/bin/env python3
"""Run a gated, simulation-only Stage20S native WBC rollout.

The native launcher remains an external-input integration harness.  This
wrapper adds the Stage20S boundary check: an actual-model candidate may reach
the delivered native WBC only when the corresponding offline interface seal
records an ``ACCEPT`` decision.  WARN/REJECT outputs are never clipped,
re-timed, or silently forwarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_seal(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "STAGE20S_ACTUAL_MODEL_INTERFACE_SEAL_V1":
        raise ValueError("interface seal has an unexpected schema")
    if payload.get("simulation_only") is not True or payload.get("hardware_control") is not False:
        raise ValueError("native closed loop requires simulation_only=true and hardware_control=false")
    if payload.get("actual_wbc_execution") is not False:
        raise ValueError("interface seal must be created before native WBC execution")
    decision = payload.get("safety_gate", {}).get("decision")
    if decision != "ACCEPT":
        raise RuntimeError(
            f"refusing native WBC execution because TrajectorySafetyGateV1 decision is {decision!r}"
        )
    candidate = Path(payload["candidate_trajectory"]).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"sealed candidate trajectory is missing: {candidate}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-seal", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")

    seal_path = args.interface_seal.expanduser().resolve()
    seal = _load_seal(seal_path)
    candidate = Path(seal["candidate_trajectory"]).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("native_wbc_benchmark.py").resolve()
    command = [
        sys.executable,
        str(launcher),
        "--source-root",
        str(args.source_root),
        "--arm",
        str(args.arm),
        "--dog",
        str(args.dog),
        "--reference-mode",
        "BENCHMARK_CONTINUOUS",
        "--trajectory",
        str(candidate),
        "--output-dir",
        str(args.output_dir),
        "--seconds",
        str(args.seconds),
        "--seed",
        str(args.seed),
        "--sim-device",
        str(args.sim_device),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (args.output_dir / "stage20s_native_closed_loop_stdout.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    native_summary_path = args.output_dir / "native_external_summary.json"
    native_summary = (
        json.loads(native_summary_path.read_text(encoding="utf-8"))
        if native_summary_path.is_file()
        else None
    )
    if native_summary is not None:
        termination = native_summary.get("termination_code") or native_summary.get("termination")
        rollout_status = (
            "REFERENCE_COMPLETE"
            if bool(native_summary.get("reference_completed"))
            else str(termination or "NATIVE_TERMINATION")
        )
        status = "PASS" if rollout_status == "REFERENCE_COMPLETE" else f"EXECUTED_{rollout_status}"
        process_status = "NATIVE_LAUNCHER_COMPLETED"
        wrapper_returncode = 0
    else:
        rollout_status = "LAUNCH_FAILED"
        status = "FAIL"
        process_status = "NATIVE_LAUNCHER_FAILED"
        wrapper_returncode = int(completed.returncode)
    result = {
        "schema": "STAGE20S_GATED_NATIVE_CLOSED_LOOP_V1",
        "status": status,
        "process_status": process_status,
        "rollout_status": rollout_status,
        "simulation_only": True,
        "hardware_control": False,
        "safety_gate_decision": seal["safety_gate"]["decision"],
        "interface_seal": str(seal_path),
        "interface_seal_sha256": _sha256(seal_path),
        "candidate_trajectory": str(candidate),
        "native_launcher": str(launcher),
        "command": command,
        "returncode": completed.returncode,
        "native_summary": native_summary,
        "actual_gr00t_closed_loop": True,
        "hardware_command_path": False,
        "correction_applied": False,
    }
    (args.output_dir / "stage20s_native_closed_loop.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return wrapper_returncode


if __name__ == "__main__":
    raise SystemExit(main())
