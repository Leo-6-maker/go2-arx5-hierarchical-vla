#!/usr/bin/env python3
"""Stage36 CPU authority, native-admission, and qualification preflight.

This program is the non-allocation half of the Stage36-RQ gate.  It binds the
redelivered WBC archive and the immutable Stage35 corpus, reconstructs the
exact Python/IsaacGym import surface, admits the 24 core and eight continuous
base cases without constructing an actor or simulator, and runs the frozen
geometry ladder.  It never submits PBS, loads a policy checkpoint, performs a
WBC rollout, imports GR00T, or edits any historical input.

The program is deliberately explicit about every private input.  In
particular, it never chooses a component by basename from an arbitrary lab
tree: the selected run and every selected file are bound by path and SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
PI_GATE_ID = "PI_GATE_STATE_0021"
STAGE = "Stage36-RQ"
PR_NUMBER = 22
BASE_HEAD = "7c1e864ee975e4e94196792ed8408f326fb4dfb9"
STAGE35_BASE_HEAD = "3a3db87bbbad64d73d9013493ad5779a519e021a"
STAGE36_BRANCH = "stage36/wbc-authority-oracle-qualification"

ARCHIVE_PATH = Path(
    "/scratch/e1192548/leon-storage/wbc-redelivery/WBCTrackingShare_20260831.zip"
)
ARCHIVE_SIZE_BYTES = 963451251
ARCHIVE_SHA256 = (
    "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167"
)
EXTRACTED_TREE_SHA256 = (
    "63311df752c9aaa5289287849635309e558f79284c91b80ddf07ec5638992567"
)
ARCHIVE_FILE_COUNT = 742
ARCHIVE_DIRECTORY_COUNT = 145
ARCHIVE_MEMBER_COUNT = 887

DEFAULT_EXTRACTED_ROOT = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_authority_v2_20260905_exact_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260905/stage36_cpu_20260905_v4"
)

STAGE35_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage35_motion_api_authority_closure_20260905/stage35_cpu_20260905_v2"
)
STAGE35_COMMITTED_PROTOCOL_PATH = REPO / "manifests/STAGE35_ORACLE_CORPUS_V1.json"
STAGE35_CORPUS_TREE_SHA256 = (
    "b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414"
)
STAGE35_MANIFEST_SHA256 = (
    "c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642"
)
STAGE35_PROTOCOL_SHA256 = (
    "81585c50b29ba67d65e346d3b6fb10420a52ca283d78e54d7476073ecacca22d"
)
STAGE35_SEAL_SHA256 = (
    "5ec9dc293546ea33c39c225d184921e77f1ae6f86756ec1bd0304cb19aa63e89"
)

RUNTIME_IMAGE = Path(
    "/scratch/e1192548/Apptainer/ubuntu-desktop_24.04-cu12.6.2-isaac.sif"
)
RUNTIME_SITE_PACKAGES = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_runtime_reconstruction_20260905/torch_cu121_v1/site-packages"
)
RUNTIME_NINJA_BIN = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_runtime_reconstruction_20260905/torch_cu121_v1/ninja_target_v4/bin/ninja"
)
RUNTIME_PYTHON = Path("/opt/miniconda3/envs/isaacgym/bin/python")

AUTHORITY_RUN_REL = Path(
    "runs/2026-08-05/"
    "stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018"
)
AUTHORITY_EXPECTED = {
    "arm_checkpoint": (
        "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a"
    ),
    "dog_checkpoint": (
        "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670"
    ),
    "native_config": (
        "56132d4ff9b866ce196bbc1b1353900b967c54f0f48b2d4e4b82c3555769a0bb"
    ),
    "native_urdf": (
        "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25"
    ),
    "native_reach_table": (
        "e34ca5c595b30b737209f5dcdd7f3f1e0b95cdf6232fbc0f6f50e3ee7140f5fa"
    ),
    "trajectory_module": (
        "8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a"
    ),
    "wbc_env": (
        "e39e1bb1dfe4c7a5c2b7309f4f158b6dcf1c9aac91ccbf8a60988ed8a64e992e"
    ),
    "legged_robot": (
        "4dc9107af7c1f77dc3b1b686911addf1a1d1a5179dfb194d0b1f4d0bdb8d8a60"
    ),
    "requirements": (
        "4f2f0ad68644f873483403eb86d17830c976058b9bd22b72789497859ec303ec"
    ),
    "wbc_wrapper": (
        "9fa688ce7c3f4665ec48752f0f31639f05ab3309b53d00d95e3d0afd0cb300c7"
    ),
}

CORE_FAMILIES = ("N", "T", "P")
ALL_FAMILIES = ("N", "T", "P", "C")
CASE_IDS = tuple(f"{family}{index:02d}" for family in ALL_FAMILIES for index in range(1, 9))
CORE_CASE_IDS = tuple(
    f"{family}{index:02d}" for family in CORE_FAMILIES for index in range(1, 9)
)
CONTINUOUS_CASE_IDS = tuple(f"C{index:02d}" for index in range(1, 9))
GEOMETRY_K_VALUES = (4, 6, 9, 12)
TIMING_MODES = ("T1", "T3", "T5", "T9")
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE36_LONG_HORIZON_LADDER"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON value at %s" % path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, "%s[%d]" % (path, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, "%s.%s" % (path, key))


def load_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    finite_json(payload, str(path))
    return payload


def write_json(path: Path, payload: Any) -> None:
    finite_json(payload, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_event(output_root: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(fields)
    with (output_root / "stage36_cpu_execution.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def run_capture(argv: Sequence[str], cwd: Path = REPO) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            list(argv), cwd=str(cwd), text=True, capture_output=True, check=False
        )
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_output(*args: str) -> Optional[str]:
    code, stdout, _ = run_capture(["git"] + list(args))
    return stdout if code == 0 else None


def is_safe_zip_member(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        return False
    return all(part not in ("", ".", "..") for part in path.parts)


def member_relative_name(name: str) -> str:
    return PurePosixPath(name).as_posix()


def archive_intake(archive_path: Path, extracted_root: Path, output_root: Path) -> Dict[str, Any]:
    """Independently bind ZIP bytes to the existing extracted tree, read-only."""

    result: Dict[str, Any] = {
        "schema": "STAGE36_IMMUTABLE_ARCHIVE_INTAKE_V2",
        "timestamp_utc": now_utc(),
        "archive_path": str(archive_path),
        "archive_exists": archive_path.is_file(),
        "archive_size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "archive_size_expected": ARCHIVE_SIZE_BYTES,
        "archive_sha256": None,
        "archive_sha256_expected": ARCHIVE_SHA256,
        "member_count": 0,
        "file_count": 0,
        "directory_count": 0,
        "safe_members": True,
        "unique_members": True,
        "symlink_members": [],
        "member_records": [],
        "extracted_root": str(extracted_root),
        "extracted_tree_sha256": None,
        "extracted_tree_sha256_expected": EXTRACTED_TREE_SHA256,
        "extracted_file_count": 0,
        "extracted_directory_count": 0,
        "missing_extracted_members": [],
        "extra_extracted_files": [],
        "extra_extracted_directories": [],
        "mismatches": [],
        "status": "BLOCKED",
    }
    if not archive_path.is_file() or not extracted_root.is_dir():
        return result

    result["archive_sha256"] = sha256_file(archive_path)
    member_map: Dict[str, Dict[str, Any]] = {}
    directory_names: List[str] = []
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        infos = archive.infolist()
        result["member_count"] = len(infos)
        for info in infos:
            name = member_relative_name(info.filename)
            safe = is_safe_zip_member(info.filename)
            if not safe:
                result["safe_members"] = False
            if name in member_map or name in directory_names:
                result["unique_members"] = False
            record: Dict[str, Any] = {
                "path": name,
                "kind": "directory" if info.is_dir() else "file",
                "size_bytes": int(info.file_size),
                "sha256": None,
            }
            if not safe:
                record["unsafe"] = True
            if info.is_dir():
                directory_names.append(name)
                result["directory_count"] += 1
            else:
                result["file_count"] += 1
                with archive.open(info, "r") as handle:
                    digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                record["sha256"] = digest.hexdigest()
                member_map[name] = record
            result["member_records"].append(record)

    extracted_files: Dict[str, Path] = {}
    extracted_dirs: List[str] = []
    symlinks: List[str] = []
    for current, dirs, files in os.walk(str(extracted_root), topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs: List[str] = []
        for dirname in sorted(dirs):
            candidate = current_path / dirname
            rel = candidate.relative_to(extracted_root).as_posix()
            if candidate.is_symlink():
                symlinks.append(rel)
            else:
                kept_dirs.append(dirname)
                extracted_dirs.append(rel)
        dirs[:] = kept_dirs
        for filename in sorted(files):
            candidate = current_path / filename
            rel = candidate.relative_to(extracted_root).as_posix()
            if candidate.is_symlink():
                symlinks.append(rel)
            elif candidate.is_file():
                extracted_files[rel] = candidate
    result["symlink_members"] = sorted(symlinks)
    result["extracted_file_count"] = len(extracted_files)
    result["extracted_directory_count"] = len(extracted_dirs)

    extracted_records: List[str] = []
    for path in sorted(extracted_files.values()):
        rel = path.relative_to(extracted_root).as_posix()
        extracted_records.append(
            "%s\t%d\t%s\n" % (rel, path.stat().st_size, sha256_file(path))
        )
    result["extracted_tree_sha256"] = sha256_bytes("".join(extracted_records).encode("utf-8"))

    archive_paths = set(member_map)
    extracted_paths = set(extracted_files)
    result["missing_extracted_members"] = sorted(archive_paths - extracted_paths)
    result["extra_extracted_files"] = sorted(extracted_paths - archive_paths)
    archive_dirs = set(directory_names)
    extracted_dir_set = set(extracted_dirs)
    result["extra_extracted_directories"] = sorted(extracted_dir_set - archive_dirs)

    for rel, record in sorted(member_map.items()):
        target = extracted_root / Path(*PurePosixPath(rel).parts)
        if not target.is_file() or target.is_symlink():
            result["mismatches"].append({"path": rel, "reason": "missing_or_symlink"})
            continue
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        if actual_size != record["size_bytes"] or actual_sha != record["sha256"]:
            result["mismatches"].append(
                {
                    "path": rel,
                    "reason": "size_or_hash",
                    "archive_size_bytes": record["size_bytes"],
                    "extracted_size_bytes": actual_size,
                    "archive_sha256": record["sha256"],
                    "extracted_sha256": actual_sha,
                }
            )

    result["counts_match"] = (
        result["member_count"] == ARCHIVE_MEMBER_COUNT
        and result["file_count"] == ARCHIVE_FILE_COUNT
        and result["directory_count"] == ARCHIVE_DIRECTORY_COUNT
        and result["extracted_file_count"] == ARCHIVE_FILE_COUNT
        and result["extracted_directory_count"] == ARCHIVE_DIRECTORY_COUNT
    )
    # The extraction root contains the archive's explicit WBCTrackingShare
    # directory, so the directory sets and counts must match exactly.
    result["status"] = (
        "PASS"
        if result["archive_size_bytes"] == ARCHIVE_SIZE_BYTES
        and result["archive_sha256"] == ARCHIVE_SHA256
        and result["member_count"] == ARCHIVE_MEMBER_COUNT
        and result["file_count"] == ARCHIVE_FILE_COUNT
        and result["directory_count"] == ARCHIVE_DIRECTORY_COUNT
        and result["safe_members"]
        and result["unique_members"]
        and not result["symlink_members"]
        and result["extracted_tree_sha256"] == EXTRACTED_TREE_SHA256
        and not result["missing_extracted_members"]
        and not result["extra_extracted_files"]
        and not result["mismatches"]
        and result["extracted_file_count"] == ARCHIVE_FILE_COUNT
        and set(extracted_dirs) == archive_dirs
        else "BLOCKED_ARCHIVE_OR_EXTRACTION_MISMATCH"
    )
    write_json(output_root / "archive_intake.json", result)
    write_json(
        output_root / "archive_member_listing.json",
        {
            "schema": "STAGE36_ARCHIVE_MEMBER_LISTING_V1",
            "archive_sha256": result["archive_sha256"],
            "members": [row["path"] for row in result["member_records"]],
        },
    )
    write_json(
        output_root / "archive_member_hash_manifest.json",
        {
            "schema": "STAGE36_ARCHIVE_MEMBER_HASH_MANIFEST_V1",
            "archive_sha256": result["archive_sha256"],
            "file_records": [row for row in result["member_records"] if row["kind"] == "file"],
        },
    )
    write_json(
        output_root / "archive_extracted_tree_manifest.json",
        {
            "schema": "STAGE36_ARCHIVE_EXTRACTED_TREE_MANIFEST_V1",
            "extracted_root": str(extracted_root),
            "tree_sha256": result["extracted_tree_sha256"],
            "tree_sha256_expected": EXTRACTED_TREE_SHA256,
            "file_count": result["extracted_file_count"],
            "directory_count": result["extracted_directory_count"],
            "read_only_comparison": True,
        },
    )
    return result


def file_record(path: Path, expected_sha256: Optional[str], logical_name: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "logical_name": logical_name,
        "path": str(path),
        "expected_sha256": expected_sha256,
        "exists": path.exists() or path.is_symlink(),
        "is_symlink": path.is_symlink(),
        "is_file": path.is_file() and not path.is_symlink(),
        "size_bytes": None,
        "sha256": None,
        "hash_status": "MISSING",
    }
    if record["is_file"]:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        if expected_sha256 is None:
            record["hash_status"] = "HASHED_NO_HISTORICAL_EXPECTATION"
        elif record["sha256"] == expected_sha256:
            record["hash_status"] = "MATCH"
        else:
            record["hash_status"] = "MISMATCH"
    elif record["is_symlink"]:
        record["hash_status"] = "SYMLINK_REQUIRES_EXPLICIT_BINDING"
        record["symlink_target"] = os.readlink(str(path))
    return record


def duplicate_scope(
    source_root: Path,
    basename: str,
    archive_hashes: Mapping[str, str],
    extracted_root: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(source_root.rglob(basename)):
        if path.name != basename or path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(extracted_root).as_posix()
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": archive_hashes.get(relative, sha256_file(path)),
                "size_bytes": path.stat().st_size,
            }
        )
    identities = sorted({row["sha256"] for row in rows})
    return {
        "basename": basename,
        "scope_root": str(source_root),
        "candidate_count": len(rows),
        "distinct_hash_count": len(identities),
        "byte_identical_duplicates": len(rows) > 1 and len(identities) == 1,
        "identities": identities,
        "records": rows,
    }


def authority_audit(
    extracted_root: Path, archive_result: Mapping[str, Any], output_root: Path
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    source_root = extracted_root / "WBCTrackingShare"
    selected_run = source_root / AUTHORITY_RUN_REL
    archive_hashes = {
        row["path"]: row["sha256"]
        for row in archive_result.get("member_records", [])
        if row.get("kind") == "file" and row.get("sha256")
    }
    selected_paths = {
        "arm_checkpoint": selected_run / "checkpoints_arm/ac_weights_039999.pt",
        "dog_checkpoint": selected_run / "checkpoints_dog/ac_weights_039999.pt",
        "native_config": selected_run / "scripts/config/wbc.py",
        "native_urdf": source_root / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
        "native_reach_table": source_root / "resources/reach_tables/go2_x5_2d.pt",
        "trajectory_module": source_root / "modules/trajectory.py",
        "wbc_env": source_root / "go1_gym/envs/roboduet/wbc_env.py",
        "legged_robot": source_root / "go1_gym/envs/roboduet/legged_robot.py",
        "requirements": source_root / "requirements.txt",
        "wbc_wrapper": source_root / "go1_gym/envs/roboduet/wbc_env_wrapper.py",
    }
    components = [
        file_record(path, AUTHORITY_EXPECTED[name], name)
        for name, path in selected_paths.items()
    ]
    duplicates = {
        basename: duplicate_scope(source_root, basename, archive_hashes, extracted_root)
        for basename in ("ac_weights_039999.pt", "wbc.py", "go2_x5_2d.pt", "trajectory.py", "wbc_env.py", "legged_robot.py", "requirements.txt")
    }
    source_exists = source_root.is_dir() and selected_run.is_dir()
    authority_pass = bool(
        archive_result.get("status") == "PASS"
        and source_exists
        and all(row["hash_status"] == "MATCH" and not row["is_symlink"] for row in components)
    )
    audit = {
        "schema": "STAGE36_DELIVERED_WBC_AUTHORITY_V2_AUDIT_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "archive_path": str(ARCHIVE_PATH),
        "extracted_root": str(extracted_root),
        "source_root": str(source_root),
        "selected_run": str(selected_run),
        "status": "PASS" if authority_pass else "BLOCKED",
        "historical_stage16_equivalence_claim": False,
        "components": components,
        "duplicate_scope": duplicates,
        "all_selection_paths_explicit": True,
        "no_mutation_performed": True,
        "read_only": True,
    }
    write_json(output_root / "stage36_a_authority_audit.json", audit)

    selected_hashes = {
        row["logical_name"]: row["sha256"]
        for row in components
        if row.get("sha256")
    }
    manifest = {
        "schema": "DELIVERED_WBC_AUTHORITY_V2_MANIFEST_V2",
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "status": audit["status"],
        "source_root": str(source_root),
        "selected_run": str(selected_run),
        "components": [
            {
                "logical_name": row["logical_name"],
                "path": row["path"],
                "sha256": row["sha256"],
                "expected_sha256": row["expected_sha256"],
            }
            for row in components
        ],
        "component_audit_path": str(output_root / "stage36_a_authority_audit.json"),
        "duplicate_scope_recorded": True,
        "no_mutation_performed": True,
    }
    write_json(output_root / "wbc_authority_v2_manifest.json", manifest)
    write_json(
        output_root / "wbc_authority_v2_tree_manifest.json",
        {
            "schema": "DELIVERED_WBC_AUTHORITY_V2_TREE_MANIFEST_V2",
            "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
            "status": audit["status"],
            "archive_tree_sha256": archive_result.get("extracted_tree_sha256"),
            "archive_tree_sha256_expected": EXTRACTED_TREE_SHA256,
            "selected_source_root": str(source_root),
            "selected_component_hashes": selected_hashes,
            "no_mutation_performed": True,
        },
    )
    write_json(
        output_root / "wbc_authority_v2_runtime_contract.json",
        {
            "schema": "DELIVERED_WBC_AUTHORITY_V2_RUNTIME_CONTRACT_V2",
            "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
            "status": audit["status"],
            "source_root": str(source_root),
            "selected_run": str(selected_run),
            "required_runtime_imports": [
                "isaacgym",
                "modules.trajectory",
                "modules.reachability",
                "go1_gym.envs.config",
                "go1_gym.envs.roboduet.wbc_env",
                "go1_gym.envs.roboduet.legged_robot",
                "go1_gym.envs.roboduet.wbc_env_wrapper",
            ],
            "native_wbc_runner": str(REPO / "scripts/native_wbc_benchmark.py"),
            "native_wbc_runner_sha256": sha256_file(REPO / "scripts/native_wbc_benchmark.py"),
            "model_or_simulator_constructed_in_this_phase": False,
            "no_mutation_performed": True,
        },
    )
    return audit, selected_hashes


def validate_case(case: Mapping[str, Any], expected_family: str) -> Dict[str, Any]:
    errors: List[str] = []
    if case.get("case_id") not in CASE_IDS or case.get("family") != expected_family:
        errors.append("case_id_or_family")
    if case.get("schema") != "STAGE35_ORACLE_TRAJECTORY_V1":
        errors.append("schema")
    if case.get("sample_count") != 121 or case.get("duration_s") != 2.0:
        errors.append("sample_count_or_duration")
    if case.get("source_frame") != "NATIVE_WBC_RELATIVE_REFERENCE":
        errors.append("source_frame")
    if case.get("tcp_semantic_id") != "x5_link6_gripper_center":
        errors.append("tcp_semantic_id")
    timestamps = case.get("timestamps")
    positions = case.get("eef_position_m")
    orientations = case.get("eef_orientation_xyzw")
    gripper = case.get("gripper")
    count = case.get("sample_count")
    if not isinstance(timestamps, list) or len(timestamps) != count:
        errors.append("timestamps_shape")
    else:
        if any(float(b) <= float(a) for a, b in zip(timestamps, timestamps[1:])):
            errors.append("timestamps_not_strictly_monotonic")
        if timestamps[0] != 0.0 or timestamps[-1] != 2.0:
            errors.append("timestamp_endpoints")
    if not isinstance(positions, list) or len(positions) != count:
        errors.append("position_shape")
    elif any(not isinstance(row, list) or len(row) != 3 for row in positions):
        errors.append("position_dimension")
    if not isinstance(orientations, list) or len(orientations) != count:
        errors.append("orientation_shape")
    elif any(not isinstance(row, list) or len(row) != 4 for row in orientations):
        errors.append("orientation_dimension")
    if not isinstance(gripper, list) or len(gripper) != count:
        errors.append("gripper_shape")
    try:
        for value in (timestamps, positions, orientations, gripper):
            finite_json(value, "case.%s" % expected_family)
    except ValueError:
        errors.append("nonfinite_payload")
    if case.get("validity_start") != 0.0 or case.get("validity_end") != 2.0:
        errors.append("validity_interval")
    provenance = case.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("provenance_type")
        provenance = {}
    for key, expected in (
        ("authoritative_stage16_source", False),
        ("native_wbc_inputs_bound", False),
        ("outcome_independent", True),
    ):
        if provenance.get(key) is not expected:
            errors.append("provenance_%s" % key)
    support = case.get("support_witness", {})
    if not isinstance(support, dict) or support.get("native_reach_table_required") is not True:
        errors.append("native_reach_table_required")
    if expected_family == "C":
        if support.get("base_witness_hidden_from_wbc") is not True:
            errors.append("continuous_hidden_witness")
        if support.get("wbc_hidden_fields") != ["base_position_m", "arm_relative_position_m"]:
            errors.append("continuous_hidden_fields")
    else:
        if support.get("base_witness_hidden_from_wbc") is not False:
            errors.append("noncontinuous_hidden_witness")
    return {
        "case_id": case.get("case_id"),
        "family": expected_family,
        "structural_pass": not errors,
        "errors": errors,
    }


def canonical_stage35_tree(manifest: Mapping[str, Any]) -> str:
    rows = [
        {"relative_path": row["relative_path"], "sha256": row["sha256"]}
        for row in manifest.get("case_records", [])
    ]
    return sha256_bytes(canonical_json(rows))


def validate_stage35_corpus(output_root: Path) -> Dict[str, Any]:
    manifest_path = STAGE35_ROOT / "stage35_oracle_corpus_manifest.json"
    generated_protocol_path = STAGE35_ROOT / "stage35_oracle_corpus_protocol.json"
    protocol_path = STAGE35_COMMITTED_PROTOCOL_PATH
    seal_path = STAGE35_ROOT / "stage35_artifact_seal.json"
    result: Dict[str, Any] = {
        "schema": "STAGE36_STAGE35_CORPUS_INPUT_AUDIT_V2",
        "input_root": str(STAGE35_ROOT),
        "manifest_path": str(manifest_path),
        "protocol_path": str(protocol_path),
        "generated_protocol_path": str(generated_protocol_path),
        "seal_path": str(seal_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "protocol_sha256": sha256_file(protocol_path) if protocol_path.is_file() else None,
        "generated_protocol_sha256": sha256_file(generated_protocol_path) if generated_protocol_path.is_file() else None,
        "seal_sha256": sha256_file(seal_path) if seal_path.is_file() else None,
        "manifest_hash_match": False,
        "protocol_hash_match": False,
        "seal_hash_match": False,
        "manifest_tree_hash": None,
        "manifest_tree_hash_expected": STAGE35_CORPUS_TREE_SHA256,
        "manifest_tree_hash_match": False,
        "case_count": 0,
        "case_ids_exact": False,
        "extra_case_files": [],
        "missing_case_files": [],
        "case_results": [],
        "core24_structural_pass": False,
        "continuous8_structural_pass": False,
        "regeneration_performed": False,
        "cases_edited": False,
        "analytic_support_promoted": False,
        "status": "BLOCKED",
    }
    if not manifest_path.is_file() or not protocol_path.is_file() or not seal_path.is_file():
        write_json(output_root / "stage36_b_corpus_admission.json", result)
        return result
    manifest = load_json(manifest_path)
    protocol = load_json(protocol_path)
    result["manifest_hash_match"] = result["manifest_sha256"] == STAGE35_MANIFEST_SHA256
    result["protocol_hash_match"] = result["protocol_sha256"] == STAGE35_PROTOCOL_SHA256
    result["seal_hash_match"] = result["seal_sha256"] == STAGE35_SEAL_SHA256
    result["manifest_tree_hash"] = canonical_stage35_tree(manifest)
    result["manifest_tree_hash_match"] = (
        result["manifest_tree_hash"] == STAGE35_CORPUS_TREE_SHA256
        and manifest.get("corpus_tree_sha256") == STAGE35_CORPUS_TREE_SHA256
    )
    manifest_rows = manifest.get("case_records", [])
    manifest_ids = [row.get("case_id") for row in manifest_rows]
    case_dir = STAGE35_ROOT / "corpus/cases"
    actual_files = {
        path.stem
        for path in case_dir.glob("*.json")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = set(CASE_IDS)
    result["case_count"] = len(manifest_rows)
    result["case_ids_exact"] = (
        manifest_ids == list(CASE_IDS) and actual_files == expected_files
    )
    result["missing_case_files"] = sorted(expected_files - actual_files)
    result["extra_case_files"] = sorted(actual_files - expected_files)
    for row in manifest_rows:
        relative = row.get("relative_path", "")
        path = STAGE35_ROOT / relative
        case_result: Dict[str, Any] = {
            "case_id": row.get("case_id"),
            "relative_path": relative,
            "manifest_sha256": row.get("sha256"),
            "manifest_size_bytes": row.get("size_bytes"),
            "exists": path.is_file() and not path.is_symlink(),
            "hash_match": False,
            "size_match": False,
            "structural_pass": False,
            "errors": [],
        }
        if case_result["exists"]:
            actual_sha = sha256_file(path)
            case_result["actual_sha256"] = actual_sha
            case_result["actual_size_bytes"] = path.stat().st_size
            case_result["hash_match"] = actual_sha == row.get("sha256")
            case_result["size_match"] = path.stat().st_size == row.get("size_bytes")
            if case_result["hash_match"] and case_result["size_match"]:
                case_result.update(validate_case(load_json(path), str(row.get("family"))))
            else:
                case_result["errors"] = ["case_hash_or_size"]
        else:
            case_result["errors"] = ["missing_case"]
        result["case_results"].append(case_result)
    by_family = {
        family: [row for row in result["case_results"] if row.get("family") == family]
        for family in ALL_FAMILIES
    }
    result["core24_structural_pass"] = (
        len(by_family["N"] + by_family["T"] + by_family["P"]) == 24
        and all(row.get("structural_pass") for family in CORE_FAMILIES for row in by_family[family])
    )
    result["continuous8_structural_pass"] = (
        len(by_family["C"]) == 8
        and all(row.get("structural_pass") for row in by_family["C"])
    )
    result["protocol_case_ids_match"] = protocol.get("case_ids") == list(CASE_IDS)
    result["status"] = (
        "PASS"
        if result["manifest_hash_match"]
        and result["protocol_hash_match"]
        and result["seal_hash_match"]
        and result["manifest_tree_hash_match"]
        and result["case_count"] == 32
        and result["case_ids_exact"]
        and result["protocol_case_ids_match"]
        and result["core24_structural_pass"]
        and result["continuous8_structural_pass"]
        else "BLOCKED_STAGE35_CORPUS_MISMATCH"
    )
    write_json(output_root / "stage36_b_corpus_admission.json", result)
    return result


def module_origin(module: Any) -> Optional[str]:
    value = getattr(module, "__file__", None)
    return None if value is None else str(Path(value).resolve())


def under_root(path: Optional[str], root: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def runtime_admission(extracted_root: Path, output_root: Path) -> Dict[str, Any]:
    """Import the exact runtime surface without constructing model/simulator."""

    source_root = extracted_root / "WBCTrackingShare"
    result: Dict[str, Any] = {
        "schema": "STAGE36_EXACT_RUNTIME_ADMISSION_V2",
        "timestamp_utc": now_utc(),
        "image": str(RUNTIME_IMAGE),
        "image_sha256": sha256_file(RUNTIME_IMAGE) if RUNTIME_IMAGE.is_file() else None,
        "site_packages": str(RUNTIME_SITE_PACKAGES),
        "ninja": str(RUNTIME_NINJA_BIN),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": None,
        "torch_cuda_version": None,
        "torchvision_version": None,
        "torchaudio_version": None,
        "gym_version": None,
        "cuda_available": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "import_origins": {},
        "expected_source_root": str(source_root),
        "native_wbc_runner_exists": (REPO / "scripts/native_wbc_benchmark.py").is_file(),
        "ninja_available": shutil.which("ninja") == str(RUNTIME_NINJA_BIN),
        "status": "BLOCKED",
        "error": None,
        "model_constructed": False,
        "simulator_constructed": False,
    }
    try:
        if RUNTIME_SITE_PACKAGES.is_dir() and str(RUNTIME_SITE_PACKAGES) not in sys.path:
            sys.path.insert(0, str(RUNTIME_SITE_PACKAGES))
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        # This import order is part of the recovered IsaacGym runtime contract.
        import isaacgym  # noqa: F401
        import torch
        import torchvision
        import torchaudio
        import gym
        trajectory = importlib.import_module("modules.trajectory")
        reachability = importlib.import_module("modules.reachability")
        config = importlib.import_module("go1_gym.envs.config")
        wbc_env = importlib.import_module("go1_gym.envs.roboduet.wbc_env")
        legged_robot = importlib.import_module("go1_gym.envs.roboduet.legged_robot")
        wrapper = importlib.import_module("go1_gym.envs.roboduet.wbc_env_wrapper")
        result.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "torchvision_version": str(torchvision.__version__),
                "torchaudio_version": str(torchaudio.__version__),
                "gym_version": str(gym.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "import_origins": {
                    "isaacgym": module_origin(isaacgym),
                    "modules.trajectory": module_origin(trajectory),
                    "modules.reachability": module_origin(reachability),
                    "go1_gym.envs.config": module_origin(config),
                    "go1_gym.envs.roboduet.wbc_env": module_origin(wbc_env),
                    "go1_gym.envs.roboduet.legged_robot": module_origin(legged_robot),
                    "go1_gym.envs.roboduet.wbc_env_wrapper": module_origin(wrapper),
                },
            }
        )
        source_imports_ok = all(
            under_root(path, source_root)
            for name, path in result["import_origins"].items()
            if name != "isaacgym"
        )
        version_ok = (
            result["python_version"] == "3.8.20"
            and result["torch_version"].startswith("2.3.1+cu121")
            and result["torch_cuda_version"] == "12.1"
            and result["torchvision_version"].startswith("0.18.1+cu121")
            and result["torchaudio_version"].startswith("2.3.1+cu121")
            and result["gym_version"] == "0.23.1"
        )
        result["source_imports_under_exact_root"] = source_imports_ok
        result["version_contract_match"] = version_ok
        result["status"] = (
            "PASS"
            if version_ok
            and source_imports_ok
            and result["ninja_available"]
            and result["native_wbc_runner_exists"]
            else "BLOCKED_RUNTIME_CONTRACT_MISMATCH"
        )
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_runtime_admission.json", result)
    return result


def case_to_chunk(payload: Mapping[str, Any], TrajectoryChunkV1: Any) -> Any:
    import numpy as np

    return TrajectoryChunkV1(
        chunk_id=str(payload["case_id"]),
        source="STAGE35_IMMUTABLE_CORPUS",
        observation_timestamp=0.0,
        point_timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
        eef_position=np.asarray(payload["eef_position_m"], dtype=np.float64),
        eef_orientation=np.asarray(payload["eef_orientation_xyzw"], dtype=np.float64),
        source_frame=str(payload["source_frame"]),
        tcp_semantic_id=str(payload["tcp_semantic_id"]),
        validity_start=0.0,
        validity_end=2.0,
        gripper=np.asarray(payload["gripper"], dtype=np.float64),
        provenance={"stage35_case_sha256": payload.get("case_sha256")},
    )


def load_corpus_payloads() -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for case_id in CASE_IDS:
        payloads[case_id] = load_json(STAGE35_ROOT / "corpus/cases" / (case_id + ".json"))
    return payloads


def core24_native_admission(
    payloads: Mapping[str, Mapping[str, Any]],
    extracted_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    import numpy as np

    result: Dict[str, Any] = {
        "schema": "STAGE36_CORE24_NATIVE_ADMISSION_V2",
        "timestamp_utc": now_utc(),
        "case_results": [],
        "core24_pass": False,
        "source_module": None,
        "model_constructed": False,
        "simulator_constructed": False,
        "status": "BLOCKED",
    }
    try:
        import torch
        from go2arx5_vla.contracts import TrajectoryChunkV1

        source_root = extracted_root / "WBCTrackingShare"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        delivered = importlib.import_module("modules.trajectory")
        result["source_module"] = module_origin(delivered)
        rows: List[Dict[str, Any]] = []
        for case_id in CORE_CASE_IDS:
            payload = payloads[case_id]
            case_result: Dict[str, Any] = {"case_id": case_id, "status": "BLOCKED"}
            try:
                chunk = case_to_chunk(payload, TrajectoryChunkV1)
                dt = float(np.median(np.diff(chunk.point_timestamps)))
                gamma, time_law = delivered.to_canonical(
                    torch.as_tensor(chunk.eef_position, dtype=torch.float32),
                    delivered.quat_to_mat(
                        torch.as_tensor(chunk.eef_orientation, dtype=torch.float32)
                    ),
                    dt,
                )
                finite = bool(
                    torch.isfinite(gamma.s_grid).all()
                    and torch.isfinite(gamma.p).all()
                    and torch.isfinite(gamma.R).all()
                    and torch.isfinite(time_law.t_grid).all()
                    and torch.isfinite(time_law.s_of_t).all()
                )
                case_result.update(
                    {
                        "status": "PASS" if finite and gamma.L > 0.0 and abs(float(time_law.T) - 2.0) <= 1e-6 else "BLOCKED",
                        "finite": finite,
                        "gamma_points": int(gamma.s_grid.shape[0]),
                        "gamma_length_m": float(gamma.L),
                        "time_law_points": int(time_law.t_grid.shape[0]),
                        "time_law_duration_s": float(time_law.T),
                    }
                )
            except Exception as exc:
                case_result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            rows.append(case_result)
        result["case_results"] = rows
        result["core24_pass"] = len(rows) == 24 and all(row["status"] == "PASS" for row in rows)
        result["status"] = "PASS" if result["core24_pass"] else "BLOCKED_CORE24_NATIVE_ADMISSION"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_core24_native_admission.json", result)
    return result


def continuous8_native_admission(
    payloads: Mapping[str, Mapping[str, Any]],
    extracted_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    import numpy as np

    result: Dict[str, Any] = {
        "schema": "STAGE36_CONTINUOUS8_NATIVE_ADMISSION_V2",
        "timestamp_utc": now_utc(),
        "reach_table_path": str(extracted_root / "WBCTrackingShare/resources/reach_tables/go2_x5_2d.pt"),
        "reach_table_sha256": None,
        "case_results": [],
        "continuous8_pass": False,
        "support_witness_is_not_wbc_input": True,
        "native_rollout_performed": False,
        "status": "BLOCKED",
    }
    try:
        import torch

        from modules.reachability import ReachabilityTable

        table_path = Path(result["reach_table_path"])
        result["reach_table_sha256"] = sha256_file(table_path)
        table = ReachabilityTable.load(str(table_path), device="cpu")
        rows: List[Dict[str, Any]] = []
        for case_id in CONTINUOUS_CASE_IDS:
            payload = payloads[case_id]
            support = payload["support_witness"]
            visible = np.asarray(payload["eef_position_m"], dtype=np.float64)
            base = np.asarray(support["base_position_m"], dtype=np.float64)
            relative = np.asarray(support["arm_relative_position_m"], dtype=np.float64)
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
            world = base + relative
            norms = np.linalg.norm(relative, axis=1)
            directions = relative / np.maximum(norms[:, None], 1e-12)
            rmax = table.query(torch.as_tensor(directions, dtype=torch.float32)).detach().cpu().numpy()
            rho = norms / np.maximum(rmax, 1e-12)
            speed = np.linalg.norm(np.diff(world, axis=0), axis=1) / np.diff(timestamps)
            start_reset_exact = bool(np.max(np.abs(visible[0] - world[0])) <= 1e-12)
            finite = bool(
                np.isfinite(world).all()
                and np.isfinite(rmax).all()
                and np.isfinite(rho).all()
                and np.isfinite(speed).all()
            )
            fixed_interval = support["fixed_base_support_interval_m"]
            fixed_impossible = bool(
                support.get("fixed_base_impossibility_proven_under_analytic_envelope") is True
                and float(np.ptp(world[:, 0])) > float(fixed_interval[1] - fixed_interval[0])
            )
            row = {
                "case_id": case_id,
                "all_points_reachable": bool(np.all(rho <= 1.0)),
                "rho_min": float(np.min(rho)),
                "rho_max": float(np.max(rho)),
                "rho_start": float(rho[0]),
                "finite": finite,
                "max_world_translation_speed_mps": float(np.max(speed)),
                "start_reset_visible_equals_base_plus_relative": start_reset_exact,
                "fixed_base_impossible_under_recorded_witness": fixed_impossible,
                "moving_base_witness": bool(np.ptp(base[:, 0]) > 0.0),
                "world_x_span_m": float(np.ptp(world[:, 0])),
                "native_reach_table_hash_match": result["reach_table_sha256"] == AUTHORITY_EXPECTED["native_reach_table"],
            }
            row["status"] = (
                "PASS"
                if row["all_points_reachable"]
                and row["finite"]
                and row["start_reset_visible_equals_base_plus_relative"]
                and row["fixed_base_impossible_under_recorded_witness"]
                and row["moving_base_witness"]
                and row["native_reach_table_hash_match"]
                else "BLOCKED"
            )
            rows.append(row)
        result["case_results"] = rows
        result["continuous8_pass"] = len(rows) == 8 and all(row["status"] == "PASS" for row in rows)
        result["status"] = "PASS" if result["continuous8_pass"] else "BLOCKED_CONTINUOUS8_NATIVE_ADMISSION"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_continuous8_native_admission.json", result)
    return result


def geometry_qualification(
    payloads: Mapping[str, Mapping[str, Any]], output_root: Path
) -> Dict[str, Any]:
    import numpy as np

    result: Dict[str, Any] = {
        "schema": "STAGE36_GEOMETRY_QUALIFICATION_V2",
        "timestamp_utc": now_utc(),
        "lambda_m_per_rad": 0.15,
        "anchor_count": 101,
        "thresholds": {
            "position_p95_m_strictly_less_than": 0.006767,
            "rotation_p95_rad_strictly_less_than": 0.038215,
            "endpoint_clamp_agreement_at_least": 0.99,
            "start_end_exact": True,
            "duration_exact": True,
        },
        "per_k": [],
        "k_star": None,
        "status": "BLOCKED",
    }
    try:
        from go2arx5_vla.contracts import TrajectoryChunkV1
        from go2arx5_vla.motion_api import (
            aggregate_evaluations,
            build_geometry_candidate,
            evaluate_preview_equivalence,
            public_evaluation,
        )

        chunks = {
            case_id: case_to_chunk(payloads[case_id], TrajectoryChunkV1)
            for case_id in CORE_CASE_IDS
        }
        for k in GEOMETRY_K_VALUES:
            rows: List[Dict[str, Any]] = []
            for case_id in CORE_CASE_IDS:
                dense = chunks[case_id]
                candidate = build_geometry_candidate(dense, k)
                evaluation = evaluate_preview_equivalence(
                    dense,
                    candidate,
                    case_id=case_id,
                    representation="DENSE_TO_K%d" % k,
                    anchor_count=101,
                )
                rows.append(evaluation)
            aggregate = aggregate_evaluations(rows, representation="DENSE_TO_K%d" % k)
            public_rows = [public_evaluation(row) for row in rows]
            qualifies = bool(
                all(row["passes_limits"] for row in rows)
                and aggregate["passes_limits"]
            )
            result["per_k"].append(
                {
                    "k": k,
                    "qualifies_strict_all_cases_and_aggregate": qualifies,
                    "aggregate": aggregate,
                    "per_case": public_rows,
                }
            )
            if qualifies and result["k_star"] is None:
                result["k_star"] = k
        result["status"] = "PASS" if result["k_star"] is not None else "PASS_NO_QUALIFIED_K"
        result["strict_selection_rule"] = "smallest K whose every case and the aggregate pass all limits"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_d1_geometry_qualification.json", result)
    return result


def timing_qualification(geometry: Mapping[str, Any], output_root: Path) -> Dict[str, Any]:
    if geometry.get("k_star") is None:
        result = {
            "schema": "STAGE36_TIMING_QUALIFICATION_V2",
            "timestamp_utc": now_utc(),
            "timing_modes": list(TIMING_MODES),
            "t_star": None,
            "status": "NOT_EVALUATED_K_STAR_NULL",
            "reason": "strict geometry qualification produced no K_STAR",
        }
    else:
        # This branch is intentionally prepared but not populated by a null
        # K_STAR run.  Future use must call the committed project timing API
        # with the already accepted K_STAR; no outcome-dependent choice is
        # permitted here.
        result = {
            "schema": "STAGE36_TIMING_QUALIFICATION_V2",
            "timestamp_utc": now_utc(),
            "timing_modes": list(TIMING_MODES),
            "k_star": geometry["k_star"],
            "t_star": None,
            "status": "NOT_EVALUATED_IMPLEMENTATION_GUARD",
        }
    write_json(output_root / "stage36_d2_timing_qualification.json", result)
    return result


def campaign_static_guard(output_root: Path) -> Dict[str, Any]:
    runner_path = REPO / "scripts/native_wbc_benchmark.py"
    campaign_path = REPO / "scripts/stage36_wbc_oracle_campaign.py"
    pbs_path = REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"
    runner = runner_path.read_text(encoding="utf-8") if runner_path.is_file() else ""
    campaign = campaign_path.read_text(encoding="utf-8") if campaign_path.is_file() else ""
    pbs = pbs_path.read_text(encoding="utf-8") if pbs_path.is_file() else ""
    forbidden_command_tokens = (
        "--train",
        "backward(",
        "optimizer.step(",
        "scheduler.step(",
        "resume_from_checkpoint",
        "--no-stop-at-reference-completion",
        "--disable-trajectory-termination",
    )
    command_contract = all(token not in campaign for token in forbidden_command_tokens)
    pbs_queue_contract = "#PBS -q auto" in pbs and all(
        token not in pbs for token in ("#PBS -q small", "#PBS -q medium", "#PBS -q large", "#PBS -q special")
    )
    result = {
        "schema": "STAGE36_E_COMMAND_STATIC_GUARD_V2",
        "timestamp_utc": now_utc(),
        "native_runner_path": str(runner_path),
        "native_runner_sha256": sha256_file(runner_path) if runner_path.is_file() else None,
        "campaign_path": str(campaign_path),
        "campaign_sha256": sha256_file(campaign_path) if campaign_path.is_file() else None,
        "pbs_path": str(pbs_path),
        "pbs_sha256": sha256_file(pbs_path) if pbs_path.is_file() else None,
        "campaign_exists": campaign_path.is_file(),
        "pbs_exists": pbs_path.is_file(),
        "selected_reference_mode": "BENCHMARK_CONTINUOUS",
        "forbidden_flags_absent_from_campaign": command_contract,
        "queue_auto_only": pbs_queue_contract,
        "selected_command_does_not_request_training_or_resume": command_contract,
        "selected_command_does_not_use_diagnostic_termination_flags": command_contract,
        "status": "PASS" if campaign_path.is_file() and pbs_path.is_file() and command_contract and pbs_queue_contract else "BLOCKED",
    }
    write_json(output_root / "stage36_e_static_command_guard.json", result)
    return result


def qstat_snapshot() -> Dict[str, Any]:
    code, stdout, stderr = run_capture(["qstat", "-u", os.environ.get("USER", "e1192548")])
    lower = (stdout or "").lower()
    return {
        "command_available": code != 127,
        "returncode": code,
        "stdout": stdout,
        "stderr": stderr,
        "stage36_jobs_visible": any(token in lower for token in ("stage36", "wbc-authority", "oracle")),
    }


def make_preflight(
    archive: Mapping[str, Any],
    authority: Mapping[str, Any],
    corpus: Mapping[str, Any],
    runtime: Mapping[str, Any],
    core: Mapping[str, Any],
    continuous: Mapping[str, Any],
    geometry: Mapping[str, Any],
    static_guard: Mapping[str, Any],
    output_root: Path,
    api_head: str,
    expected_head: str,
) -> Dict[str, Any]:
    status = git_output("status", "--porcelain")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    remote_head = git_output("rev-parse", "origin/" + STAGE36_BRANCH)
    scratch_usage = shutil.disk_usage("/scratch")
    tmp_usage = shutil.disk_usage("/tmp")
    qstat = qstat_snapshot()
    qsub_code, qsub_stdout, qsub_stderr = run_capture(["qsub", "--version"])
    qsub_available = qsub_code != 127
    geometry_ok = geometry.get("status") in ("PASS", "PASS_NO_QUALIFIED_K")
    continuous_ok_or_conditional = continuous.get("status") in (
        "PASS",
        "BLOCKED_CONTINUOUS8_NATIVE_ADMISSION",
        "BLOCKED_DEPENDENCY",
    )
    all_admission = bool(
        archive.get("status") == "PASS"
        and authority.get("status") == "PASS"
        and corpus.get("status") == "PASS"
        and runtime.get("status") == "PASS"
        and core.get("status") == "PASS"
        and continuous_ok_or_conditional
        and geometry_ok
        and static_guard.get("status") == "PASS"
        and qstat.get("command_available")
        and qstat.get("returncode") == 0
        and not qstat.get("stage36_jobs_visible")
        and qsub_available
        and scratch_usage.free > 100 * 1024 * 1024 * 1024
        and api_head == BASE_HEAD
        and remote_head == BASE_HEAD
        and git_output("rev-parse", "HEAD") == BASE_HEAD
        and status == ""
    )
    result = {
        "schema": "STAGE36_ZERO_MODEL_PREFLIGHT_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "active_stage": STAGE,
        "expected_head": expected_head,
        "api_head_observed_or_bound": api_head,
        "branch": branch,
        "head": head,
        "origin_stage36_head": remote_head,
        "head_matches_expected": head == expected_head,
        "api_remote_local_head_match": api_head == expected_head and remote_head == expected_head and head == expected_head,
        "base_is_ancestor": git_output("merge-base", "HEAD", STAGE35_BASE_HEAD) == STAGE35_BASE_HEAD,
        "worktree_clean": status == "",
        "status_porcelain": status,
        "project": "CFP03-CF-004",
        "selection_queue": "auto",
        "qsub_allowed_queue_only": True,
        "qstat": qstat,
        "qsub_probe": {
            "command_available": qsub_available,
            "returncode": qsub_code,
            "stdout": qsub_stdout,
            "stderr": qsub_stderr,
            "submission_performed": False,
        },
        "no_active_stage36_job": not qstat["stage36_jobs_visible"],
        "scratch_disk": {
            "free_bytes": scratch_usage.free,
            "total_bytes": scratch_usage.total,
            "sufficient": scratch_usage.free > 100 * 1024 * 1024 * 1024,
        },
        "tmp_disk": {
            "free_bytes": tmp_usage.free,
            "total_bytes": tmp_usage.total,
            "not_used_for_runtime": True,
        },
        "scratch_backed_runtime_paths": all(
            str(path).startswith("/scratch/")
            for path in (RUNTIME_SITE_PACKAGES, RUNTIME_NINJA_BIN, output_root)
        ),
        "admissions": {
            "archive": archive.get("status"),
            "authority": authority.get("status"),
            "stage35_corpus": corpus.get("status"),
            "runtime": runtime.get("status"),
            "core24_native": core.get("status"),
            "continuous8_native": continuous.get("status"),
            "geometry": geometry.get("status"),
            "static_guard": static_guard.get("status"),
        },
        "e_predicates_pass": all_admission,
        "execution_boundary": {
            "pbs_submitted": False,
            "gpu_used": False,
            "model_constructed": False,
            "gr00t_model_forward": False,
            "wbc_executed": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_written": False,
            "sampler": False,
            "get_action": False,
            "c2": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_hardware": False,
            "pr_merge": False,
        },
    }
    write_json(output_root / "stage36_preflight.json", result)
    return result


def materialize_campaign_plan(
    payloads: Mapping[str, Mapping[str, Any]],
    corpus: Mapping[str, Any],
    continuous: Mapping[str, Any],
    geometry: Mapping[str, Any],
    timing: Mapping[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    import numpy as np

    inputs_root = output_root / "campaign_inputs"
    rows: List[Dict[str, Any]] = []
    admitted_ids = list(CORE_CASE_IDS)
    if continuous.get("status") == "PASS":
        admitted_ids.extend(CONTINUOUS_CASE_IDS)
    for case_id in admitted_ids:
        payload = payloads[case_id]
        trajectory_path = inputs_root / (case_id + "_dense_seed13.npz")
        metadata_path = inputs_root / (case_id + "_dense_seed13.json")
        np.savez_compressed(
            str(trajectory_path),
            timestamp=np.asarray(payload["timestamps"], dtype=np.float64),
            position=np.asarray(payload["eef_position_m"], dtype=np.float64),
            quaternion=np.asarray(payload["eef_orientation_xyzw"], dtype=np.float64),
        )
        metadata = {
            "schema": "STAGE36_WBC_CAMPAIGN_CASE_METADATA_V1",
            "case_id": case_id,
            "family": payload["family"],
            "representation": "DENSE",
            "seed": 13,
            "reference_mode": "BENCHMARK_CONTINUOUS",
            "seconds": 2.0,
            "source_case_path": str(STAGE35_ROOT / "corpus/cases" / (case_id + ".json")),
            "source_case_sha256": sha256_file(STAGE35_ROOT / "corpus/cases" / (case_id + ".json")),
            "support_witness_used_by_wbc": False,
        }
        write_json(metadata_path, metadata)
        rows.append(
            {
                "row_id": "dense_%s_seed13" % case_id,
                "case_id": case_id,
                "family": payload["family"],
                "representation": "DENSE",
                "seed": 13,
                "reference_mode": "BENCHMARK_CONTINUOUS",
                "seconds": 2.0,
                "trajectory_path": str(trajectory_path),
                "metadata_path": str(metadata_path),
                "trajectory_sha256": sha256_file(trajectory_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    plan_status = (
        "READY_FOR_E"
        if corpus.get("status") == "PASS"
        and geometry.get("status") in ("PASS", "PASS_NO_QUALIFIED_K")
        and len(rows) <= 96
        and len({row["row_id"] for row in rows}) == len(rows)
        else "BLOCKED"
    )
    plan = {
        "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "active_stage": STAGE,
        "execution_head": git_output("rev-parse", "HEAD"),
        "accepted_start_head": BASE_HEAD,
        "plan_status": plan_status,
        "project": "CFP03-CF-004",
        "queue": "auto",
        "one_pbs_allocation_only": True,
        "maximum_rows": 96,
        "row_count": len(rows),
        "core24_rows": sum(1 for row in rows if row["case_id"] in CORE_CASE_IDS),
        "continuous8_rows": sum(1 for row in rows if row["case_id"] in CONTINUOUS_CASE_IDS),
        "k_star": geometry.get("k_star"),
        "t_star": timing.get("t_star"),
        "sparse_rows": 0,
        "geometry_status": geometry.get("status"),
        "timing_status": timing.get("status"),
        "reference_mode": "BENCHMARK_CONTINUOUS",
        "native_runner_path": str(REPO / "scripts/native_wbc_benchmark.py"),
        "native_runner_sha256": sha256_file(REPO / "scripts/native_wbc_benchmark.py"),
        "pbs_wrapper_path": str(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
        "pbs_wrapper_sha256": sha256_file(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
        "forbidden_modes": [
            "BENCHMARK_40POINT_RECONSTRUCTED",
            "NATIVE_GENERATOR",
            "free_running_sampler",
            "get_action",
            "training",
            "resume",
            "new_checkpoint",
        ],
        "rows": rows,
    }
    write_json(output_root / "stage36_campaign_plan.json", plan)
    return plan


def artifact_entries(output_root: Path, exclude: Iterable[str]) -> List[Dict[str, Any]]:
    excluded = set(exclude)
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            entries.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def seal_cpu_artifacts(output_root: Path) -> Dict[str, Any]:
    seal = {
        "schema": "STAGE36_CPU_ARTIFACT_SEAL_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "stage": STAGE,
        "output_root": str(output_root),
        "entries": artifact_entries(output_root, ("stage36_artifact_seal.json", "stage36_return.json")),
        "historical_inputs_mutated": False,
    }
    write_json(output_root / "stage36_artifact_seal.json", seal)
    seal["seal_sha256"] = sha256_file(output_root / "stage36_artifact_seal.json")
    return seal


def build_return(
    output_root: Path,
    preflight: Optional[Mapping[str, Any]],
    plan: Optional[Mapping[str, Any]],
    seal: Optional[Mapping[str, Any]],
    status: str,
    blockers: Sequence[str],
    detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": STAGE,
        "PR": PR_NUMBER,
        "ACCEPTED_START_HEAD": BASE_HEAD,
        "EXPECTED_HEAD": git_output("rev-parse", "HEAD"),
        "LOCAL_HEAD": git_output("rev-parse", "HEAD"),
        "BRANCH": git_output("branch", "--show-current"),
        "ARTIFACT_ROOT": str(output_root),
        "RESULT": status,
        "BLOCKERS": list(blockers),
        "NEXT_GATE": NEXT_GATE,
        "NEXT_ACTION": (
            "E0_COMMIT_PUSH_THEN_ONE_QSUB_AUTO"
            if status == "STAGE36_CPU_ADMISSION_PASS_READY_FOR_E"
            else "STOP_AND_RETURN_TO_GPT_PI"
        ),
        "PLAN_STATUS": None if plan is None else plan.get("plan_status"),
        "K_STAR": None if plan is None else plan.get("k_star"),
        "T_STAR": None if plan is None else plan.get("t_star"),
        "DETAIL_OUTPUTS": {
            "archive_intake": str(output_root / "archive_intake.json"),
            "authority_audit": str(output_root / "stage36_a_authority_audit.json"),
            "runtime_admission": str(output_root / "stage36_runtime_admission.json"),
            "corpus_admission": str(output_root / "stage36_b_corpus_admission.json"),
            "core24_native_admission": str(output_root / "stage36_core24_native_admission.json"),
            "continuous8_native_admission": str(output_root / "stage36_continuous8_native_admission.json"),
            "geometry": str(output_root / "stage36_d1_geometry_qualification.json"),
            "timing": str(output_root / "stage36_d2_timing_qualification.json"),
            "preflight": str(output_root / "stage36_preflight.json"),
            "campaign_plan": str(output_root / "stage36_campaign_plan.json"),
            "artifact_seal": str(output_root / "stage36_artifact_seal.json"),
        },
        "PBS_SUBMITTED": False,
        "GPU_USED": False,
        "MODEL_CONSTRUCTED": False,
        "FORWARD": False,
        "TRAINING": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
    }
    if preflight is not None:
        result["PREFLIGHT_E_PREDICATES"] = preflight.get("e_predicates_pass")
    if seal is not None:
        result["CPU_SEAL_SHA256"] = seal.get("seal_sha256")
    if detail is not None:
        result["DETAIL"] = dict(detail)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, default=REPO)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument("--stage35-root", type=Path, default=STAGE35_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-head", required=True, help="fresh direct API PR head observation")
    parser.add_argument("--expected-head", required=True, help="fresh checkout/API/remote head to bind")
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()

    if args.project_repo.resolve() != REPO:
        raise SystemExit("FAIL_CLOSED wrong repository: %s" % args.project_repo)
    if args.stage35_root.resolve() != STAGE35_ROOT:
        raise SystemExit("FAIL_CLOSED wrong Stage35 root: %s" % args.stage35_root)
    if output_root.exists():
        raise SystemExit("FAIL_CLOSED output already exists: %s" % output_root)
    if git_output("branch", "--show-current") != STAGE36_BRANCH:
        raise SystemExit("FAIL_CLOSED wrong Stage36 branch")
    if git_output("status", "--porcelain") != "":
        raise SystemExit("FAIL_CLOSED dirty worktree before Stage36")
    local_head = git_output("rev-parse", "HEAD")
    if local_head != args.expected_head:
        raise SystemExit("FAIL_CLOSED local HEAD is not the supplied live canonical HEAD")
    if args.api_head != args.expected_head:
        raise SystemExit("FAIL_CLOSED API head does not equal the supplied live canonical HEAD")
    if git_output("merge-base", "HEAD", BASE_HEAD) != BASE_HEAD:
        raise SystemExit("FAIL_CLOSED current HEAD is not descended from accepted Stage36 start")
    if git_output("merge-base", "HEAD", STAGE35_BASE_HEAD) != STAGE35_BASE_HEAD:
        raise SystemExit("FAIL_CLOSED Stage35 base is not an ancestor")

    output_root.mkdir(parents=True)
    log_event(output_root, "START", head=git_output("rev-parse", "HEAD"))
    archive_result = archive_intake(args.archive.resolve(), args.extracted_root.resolve(), output_root)
    log_event(output_root, "ARCHIVE_INTAKE", status=archive_result.get("status"))
    authority_result, _ = authority_audit(args.extracted_root.resolve(), archive_result, output_root)
    log_event(output_root, "AUTHORITY_AUDIT", status=authority_result.get("status"))
    corpus_result = validate_stage35_corpus(output_root)
    log_event(output_root, "CORPUS_AUDIT", status=corpus_result.get("status"))

    runtime_result = runtime_admission(args.extracted_root.resolve(), output_root)
    log_event(output_root, "RUNTIME_ADMISSION", status=runtime_result.get("status"))

    payloads: Dict[str, Dict[str, Any]] = {}
    core_result: Dict[str, Any] = {"status": "BLOCKED_DEPENDENCY"}
    continuous_result: Dict[str, Any] = {"status": "BLOCKED_DEPENDENCY"}
    geometry_result: Dict[str, Any] = {"status": "BLOCKED_DEPENDENCY", "k_star": None}
    timing_result: Dict[str, Any] = {"status": "BLOCKED_DEPENDENCY", "t_star": None}
    if corpus_result.get("status") == "PASS" and runtime_result.get("status") == "PASS":
        payloads = load_corpus_payloads()
        # Project source is intentionally imported only after exact runtime
        # admission.  These calls are CPU-only and do not construct GR00T or
        # a simulator.
        sys.path.insert(0, str(REPO / "src"))
        core_result = core24_native_admission(payloads, args.extracted_root.resolve(), output_root)
        continuous_result = continuous8_native_admission(payloads, args.extracted_root.resolve(), output_root)
        if core_result.get("status") == "PASS":
            geometry_result = geometry_qualification(payloads, output_root)
            timing_result = timing_qualification(geometry_result, output_root)
        else:
            write_json(output_root / "stage36_core24_native_admission.json", core_result)
            write_json(output_root / "stage36_continuous8_native_admission.json", continuous_result)
            write_json(output_root / "stage36_d1_geometry_qualification.json", geometry_result)
            write_json(output_root / "stage36_d2_timing_qualification.json", timing_result)
    else:
        for name, value in (
            ("stage36_core24_native_admission.json", core_result),
            ("stage36_continuous8_native_admission.json", continuous_result),
            ("stage36_d1_geometry_qualification.json", geometry_result),
            ("stage36_d2_timing_qualification.json", timing_result),
        ):
            write_json(output_root / name, value)
    log_event(
        output_root,
        "NATIVE_ADMISSION_AND_D_QUALIFICATION",
        core24=core_result.get("status"),
        continuous8=continuous_result.get("status"),
        k_star=geometry_result.get("k_star"),
        t_star=timing_result.get("t_star"),
    )

    static_guard = campaign_static_guard(output_root)
    preflight = make_preflight(
        archive_result,
        authority_result,
        corpus_result,
        runtime_result,
        core_result,
        continuous_result,
        geometry_result,
        static_guard,
        output_root,
        args.api_head,
        args.expected_head,
    )
    plan: Optional[Dict[str, Any]] = None
    if preflight.get("e_predicates_pass") and payloads:
        plan = materialize_campaign_plan(
            payloads,
            corpus_result,
            continuous_result,
            geometry_result,
            timing_result,
            output_root,
        )
    else:
        plan = {
            "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V2",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "plan_status": "BLOCKED",
            "row_count": 0,
            "rows": [],
            "k_star": geometry_result.get("k_star"),
            "t_star": timing_result.get("t_star"),
        }
        write_json(output_root / "stage36_campaign_plan.json", plan)

    blockers: List[str] = []
    for name, value in (
        ("archive", archive_result),
        ("authority", authority_result),
        ("stage35_corpus", corpus_result),
        ("runtime", runtime_result),
        ("core24_native", core_result),
        ("geometry", geometry_result),
        ("static_guard", static_guard),
    ):
        if value.get("status") not in ("PASS", "PASS_NO_QUALIFIED_K"):
            blockers.append("%s:%s" % (name, value.get("status")))
    if continuous_result.get("status") not in (
        "PASS",
        "BLOCKED_CONTINUOUS8_NATIVE_ADMISSION",
    ):
        blockers.append("continuous8:%s" % continuous_result.get("status"))
    if not preflight.get("head_matches_expected") or not preflight.get("worktree_clean") or not preflight.get("api_remote_local_head_match"):
        blockers.append("HEAD_OR_WORKTREE_MISMATCH")
    if not preflight.get("no_active_stage36_job"):
        blockers.append("ACTIVE_STAGE36_JOB_VISIBLE")
    if plan.get("plan_status") != "READY_FOR_E":
        blockers.append("CAMPAIGN_PLAN_NOT_READY")
    status = (
        "STAGE36_CPU_ADMISSION_PASS_READY_FOR_E"
        if not blockers and preflight.get("e_predicates_pass")
        else "BLOCKED"
    )
    seal = seal_cpu_artifacts(output_root)
    return_payload = build_return(output_root, preflight, plan, seal, status, blockers)
    write_json(output_root / "stage36_return.json", return_payload)
    log_event(output_root, "RETURN", result=status, blockers=blockers)
    # The log receives the final event after the seal; the seal intentionally
    # excludes the return and the post-seal append so the seal remains stable.
    return 0 if status == "STAGE36_CPU_ADMISSION_PASS_READY_FOR_E" else 2


if __name__ == "__main__":
    raise SystemExit(main())
