#!/usr/bin/env python3
"""Source-backed, read-only audit of superseded PI0160 attempt directories.

This report intentionally distinguishes a reported counter from a counter that
can be justified by an original result/admission artifact.  Missing v1/v5
artifacts stay UNKNOWN; no experiment is started and no historical directory
is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


EXPERT_GLOB = "/var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_expert_20260913_v*"
MODEL_GLOB = "/var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_model_20260913_v*"
SOURCE_FILES = (
    "scripts/stage45f_pi0160_matched_reset_expert.py",
    "scripts/stage40_live_one_handoff.py",
    "scripts/stage21_native_receding_horizon.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def file_inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    if not root.is_dir():
        return result
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        try:
            result.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        except OSError as exc:
            result.append({"path": str(path.relative_to(root)), "error": str(exc)})
    return result


def _expert_constructor_lower_bound(root: Path, result: Mapping[str, Any] | None) -> tuple[int | None, str]:
    if result is None:
        return None, "UNRESOLVED_NO_RESULT_ARTIFACT"
    status = str(result.get("status", "UNKNOWN"))
    conditions = result.get("conditions")
    if not isinstance(conditions, list):
        return None, "UNRESOLVED_CONDITION_RECORDS"
    # The result is written after each condition returns.  A completed or
    # partial condition with WBC/reference counters is downstream of
    # _build_native; a pre-import libpython failure is explicitly before the
    # constructor.  For any unrecognised error, stay conservative.
    entered = 0
    unknown = False
    for condition in conditions:
        if not isinstance(condition, Mapping):
            unknown = True
            continue
        wbc = int(condition.get("wbc_steps", 0) or 0)
        updates = int(condition.get("reference_updates", 0) or 0)
        rows = int(condition.get("rows", 0) or 0)
        error = str((condition.get("error") or {}).get("message", ""))
        if wbc > 0 or updates > 0 or rows > 0:
            entered += 1
        elif error.startswith("libpython3.8.so.1.0:") or "cannot open shared object file" in error:
            # The native import fails before EnvClass construction.
            continue
        elif "_refresh_eef_from_rigid_body" in error:
            # This failure is raised after _build_native in the old wrapper.
            entered += 1
        elif status in {"COMPLETED", "PARTIAL"} and not error:
            unknown = True
        else:
            unknown = True
    if unknown and entered == 0:
        return None, "UNRESOLVED_CONSTRUCTOR_BOUNDARY"
    return entered, "SOURCE_VERIFIED_FROM_RESULT_AND_FAILURE_LOCATION" if not unknown else "SOURCE_VERIFIED_LOWER_BOUND_WITH_UNRESOLVED_ATTEMPT"


def _audit_expert(root: Path) -> dict[str, Any]:
    admission_path = root / "admission.json"
    result_path = root / "pi0160_matched_expert_result.json"
    admission = read_json(admission_path) if admission_path.is_file() else None
    result = read_json(result_path) if result_path.is_file() else None
    constructor_lb, constructor_basis = _expert_constructor_lower_bound(root, result if isinstance(result, Mapping) else None)
    counters = result.get("counters", {}) if isinstance(result, Mapping) else {}
    return {
        "attempt_id": root.name,
        "kind": "expert_native",
        "root": str(root),
        "admission": {"present": admission_path.is_file(), "sha256": sha256_file(admission_path) if admission_path.is_file() else None, "status": admission.get("status") if isinstance(admission, Mapping) else None},
        "result": {"present": result_path.is_file(), "sha256": sha256_file(result_path) if result_path.is_file() else None, "status": result.get("status") if isinstance(result, Mapping) else None},
        "source_verified": {"native_constructor_entered_lower_bound": constructor_lb, "basis": constructor_basis, "wbc_ticks": int(counters.get("expert_wbc_steps", 0) or 0) if isinstance(counters, Mapping) else None, "reference_updates": int(counters.get("reference_updates", 0) or 0) if isinstance(counters, Mapping) else None},
        "reported": {"conditions_started": counters.get("conditions_started") if isinstance(counters, Mapping) else None, "conditions_passed": counters.get("conditions_passed") if isinstance(counters, Mapping) else None},
        "unknown": [] if admission_path.is_file() and result_path.is_file() else ["UNRESOLVED_NO_RELIABLE_SOURCE"],
        "files": file_inventory(root),
    }


def _audit_model(root: Path) -> dict[str, Any]:
    admission_path = root / "model_admission.json"
    result_path = root / "pi0160_matched_reset_expert_model_result.json"
    admission = read_json(admission_path) if admission_path.is_file() else None
    result = read_json(result_path) if result_path.is_file() else None
    counters = result.get("counters", {}) if isinstance(result, Mapping) else {}
    known = isinstance(result, Mapping) and isinstance(counters, Mapping)
    return {
        "attempt_id": root.name,
        "kind": "model_gr00t",
        "root": str(root),
        "admission": {"present": admission_path.is_file(), "sha256": sha256_file(admission_path) if admission_path.is_file() else None, "status": admission.get("status") if isinstance(admission, Mapping) else None},
        "result": {"present": result_path.is_file(), "sha256": sha256_file(result_path) if result_path.is_file() else None, "status": result.get("status") if isinstance(result, Mapping) else None},
        "source_verified": {"constructor_entered": int(counters.get("model_construction_attempts", 0)) if known and counters.get("model_construction_attempts") is not None else None, "conditioning_entered": int(counters.get("conditioning_entered", 0)) if known and counters.get("conditioning_entered") is not None else None, "action_head_calls": int(counters.get("action_head_calls", 0)) if known and counters.get("action_head_calls") is not None else None, "model_side_calls": int(counters.get("model_side_calls", 0)) if known and counters.get("model_side_calls") is not None else None},
        "reported": {"admission_present": admission_path.is_file()},
        "unknown": [] if known else ["UNRESOLVED_NO_RELIABLE_RESULT_COUNTERS"],
        "files": file_inventory(root),
    }


def _process_snapshot() -> dict[str, Any]:
    command = ["ps", "-eo", "pid=,ppid=,etimes=,stat=,args="]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        rows = [line.strip() for line in proc.stdout.splitlines() if "stage45f_pi0160" in line and "grep" not in line]
        return {"hostname": platform.node(), "command": command, "returncode": proc.returncode, "matching_processes": rows, "old_processes_stopped": not rows}
    except OSError as exc:
        return {"hostname": platform.node(), "command": command, "error": str(exc), "old_processes_stopped": None}


def build_report(repository_head: str) -> dict[str, Any]:
    expert_roots = sorted(Path(p) for p in __import__("glob").glob(EXPERT_GLOB))
    model_roots = sorted(Path(p) for p in __import__("glob").glob(MODEL_GLOB))
    sources = []
    for relative in SOURCE_FILES:
        path = Path(relative)
        sources.append({"path": relative, "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None})
    return {
        "schema": "CORELAB_PI0162_SOURCE_AUDIT_V1",
        "generated_at": time.time(),
        "hostname": platform.node(),
        "repository_head": repository_head,
        "expert_attempts": [_audit_expert(path) for path in expert_roots],
        "model_attempts": [_audit_model(path) for path in model_roots],
        "sources": sources,
        "process_snapshot": _process_snapshot(),
        "historical_policy": {"reported_preserved": True, "source_verified_lower_bounds_preserved": True, "unknown_not_zero": True, "no_historical_rerun": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.repository_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": sha256_file(args.output), "expert_attempts": len(report["expert_attempts"]), "model_attempts": len(report["model_attempts"]), "old_processes_stopped": report["process_snapshot"].get("old_processes_stopped")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
