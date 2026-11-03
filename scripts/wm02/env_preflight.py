"""WM02 offline environment preflight; never grants target execution permission.

Standard library only. Reads a WM02 resolved spec and optionally verifies the
pinned target members under one explicit, local root. It neither imports a native
controller nor launches ROS, MuJoCo, a training model, or a GPU computation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_members(members: list[dict], root: Path | None) -> list[dict]:
    """Verify real bytes, or explicitly mark them unverified; never trust a claimed match."""
    out = []
    root_resolved = root.resolve(strict=True) if root is not None else None
    for member in members:
        role = member.get("role")
        expected = member.get("expected")
        if not isinstance(role, str) or not role or not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise ValueError("Invalid pinned member role/digest")
        relative = Path(role)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or relative.parts[0] == ".":
            raise ValueError("Pinned member must be a safe relative role")
        item = {"role": role, "expected_sha256": expected, "status": "NOT_CHECKED_NO_ROOT"}
        if root_resolved is not None:
            path = root_resolved / relative
            if not path.exists():
                item["status"] = "MISSING"
            elif not path.is_file() or not path.resolve(strict=True).is_relative_to(root_resolved):
                item["status"] = "INVALID_FILE_OR_OUTSIDE_ROOT"
            else:
                actual = sha256(path)
                item["status"] = "MATCH" if actual == expected else "MISMATCH"
                item["actual_sha256"] = actual
                item["bytes"] = path.stat().st_size
        out.append(item)
    return out


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "numpy", "mujoco", "rclpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_metadata(enabled: bool) -> dict:
    if not enabled:
        return {"status": "NOT_REQUESTED"}
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=7)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "UNAVAILABLE_OR_TIMEOUT"}
    if result.returncode:
        return {"status": "QUERY_FAILED", "exit_code": result.returncode}
    return {"status": "READ_ONLY_QUERY_OK", "columns": ["gpu_index", "name", "total_MiB", "free_MiB", "driver"],
            "rows": [x.strip() for x in result.stdout.splitlines() if x.strip()],
            "not_a_reservation": True}


def inspect(spec: dict, *, root: Path | None = None, gpu: bool = False) -> dict:
    if spec.get("schema") != "WM02_RESOLVED_SPEC_V1":
        raise ValueError("Not a WM02 resolved spec")
    target = spec.get("target", {})
    if target.get("backend") != "GO2_ARX5_MUJOCO_OCS2":
        raise ValueError("Expected actual frozen Go2-ARX5 MuJoCo/OCS2 backend")
    frozen = inspect_members(target.get("frozen_members", []), root)
    if not frozen:
        raise ValueError("No frozen target members to inspect")
    problems = [x["role"] for x in frozen if x["status"] not in ("MATCH", "NOT_CHECKED_NO_ROOT")]
    blockers = list(dict.fromkeys(spec.get("target_blockers", [])))
    if root is None:
        blockers.append("NO_LOCAL_TARGET_ROOT_BYTES_CHECKED")
    if problems:
        blockers.append("TARGET_FROZEN_MEMBER_BYTES_NOT_MATCHED")
    return {
        "schema": "WM02_READ_ONLY_ENVIRONMENT_PREFLIGHT_V1",
        "purpose": "OFFLINE_IDENTITY_AND_SOFTWARE_AUDIT_ONLY",
        "status": "INSPECTION_DONE_TARGET_NOT_AUTHORIZED",
        "native_target_admission": "NOT_PERFORMED",
        "target_dynamic_authorized": False,
        "target_backend": target["backend"],
        "reported_native_permission": target.get("permission", {}).get("status"),
        "reported_native_quarantine": target.get("permission", {}).get("native_safety_quarantine_preserved"),
        "target_blockers": blockers,
        "full_control_restore_available_in_spec": target.get("full_control_restore_available") is True,
        "scorer_status": target.get("scorer", {}).get("actual_WM02_standing_task_binding", "UNKNOWN"),
        "frozen_member_check": frozen,
        "initialization_members_not_automatically_verified": len(target.get("initialization_members", [])),
        "environment": {"python": sys.version.split()[0], "ros_distro": os.environ.get("ROS_DISTRO"),
                        "ros_cli_on_path": __import__("shutil").which("ros2") is not None,
                        "packages": package_versions()},
        "gpu": gpu_metadata(gpu),
        "no_simulator_ros_wbc_or_training_process_started": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True, help="Current read-only WM02 RESOLVED_SPEC JSON")
    p.add_argument("--target-root", type=Path, help="Local private frozen target root (read-only)")
    p.add_argument("--output", type=Path, required=True, help="New private report; never overwritten")
    p.add_argument("--gpu-query", action="store_true", help="Optional read-only nvidia-smi query")
    args = p.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.target_root is not None and args.output.resolve().is_relative_to(args.target_root.resolve()):
        p.error("Report must not be written inside the frozen target root")
    report = inspect(spec, root=args.target_root, gpu=args.gpu_query)
    report["input_spec_sha256"] = sha256(args.spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"status": report["status"], "target_blockers": report["target_blockers"],
                      "report_saved": True, "physical_steps": 0}, ensure_ascii=False))
    return 0 if all(x["status"] in ("MATCH", "NOT_CHECKED_NO_ROOT") for x in report["frozen_member_check"]) else 4


if __name__ == "__main__":
    raise SystemExit(main())
