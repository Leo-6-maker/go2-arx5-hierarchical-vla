#!/usr/bin/env python3
"""Run the CPU-only Stage35 motion-API authority and semantic closure.

The program has no model, simulator, CUDA, WBC, or PBS execution path.  It
audits the historical Stage16 authority without changing it, generates a new
deterministic *candidate* corpus, qualifies geometry/timing independently, and
seals the evidence.  Native Stage16/WBC authority is never inferred from the
generated candidates; missing exact inputs therefore remains a hard stop.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from go2arx5_vla.contracts import MotionIntentV2, TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.motion_api import (  # noqa: E402
    DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
    DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
    DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    GEOMETRY_K_VALUES,
    TIMING_KNOTS,
    TIMING_MODES,
    aggregate_evaluations,
    build_geometry_candidate,
    build_timing_candidate,
    evaluate_preview_equivalence,
    public_evaluation,
)
from go2arx5_vla.oracle.stage35_corpus import (  # noqa: E402
    CASES_PER_CLASS,
    CORPUS_CLASSES,
    CORPUS_SAMPLE_COUNT,
    DURATION_S,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    build_case,
    case_ids,
    protocol_manifest,
)
from go2arx5_vla.trajectory.so3 import geodesic_angle, quat_to_matrix  # noqa: E402


PI_GATE_ID = "PI_GATE_STATE_0018"
BASE_HEAD = "6037c392966715d516035507ddef6c1e8b07de6d"
STAGE34_BRANCH = "stage34/wbc-aligned-motion-api"
STAGE35_BRANCH = "stage35/motion-api-authority-closure"
STAGE34_SEAL_SHA256 = "32f082525b33f74c7df3ecee93abc71b078ae30dfc822518caec8ff7ec1f63a6"
NEXT_GATE = "GPT_PI_REVIEW_BEFORE_STRUCTURED_GR00T_TRAINING"
PROJECT = "CFP03-CF-004"

EXPECTED_STAGE16_ROOT = Path("/var/cache/users/liuyu-nfs/go2arx5-wbc-stage16")
SEARCH_ROOTS = (
    Path("/var/cache/users/liuyu-nfs"),
    Path("/var/cache/users/liuyu-nfs/go2arx5-wbc-stage13"),
    Path("/var/cache/users/liuyu-nfs/go2arx5-experiment-e2"),
)
STAGE16_MANIFEST_RELATIVE_PATHS = (
    "benchmark/native_suite_manifest.json",
    "benchmark/first_eight_manifest.json",
    "progressive_relocation/progressive_cohort.json",
    "continuity/initial_continuity_analysis.json",
)
WBC_SEARCH_FILENAMES = (
    "WBCTrackingShare_20260831.zip",
    "go2_x5.urdf",
    "ac_weights_039999.pt",
    "wbc.py",
    "go2_x5_2d.pt",
)
EXPECTED_WBC_HASHES = {
    "delivered_wbc_archive": "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167",
    "native_urdf": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "arm_checkpoint": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "native_config": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
}
STAGE16_SOURCE_FILES = (
    "reports/STAGE16_FINAL_REPORT.md",
    "reports/STAGE16_PROMOTION_REVIEW.md",
    "reports/STAGE16_NATIVE_BENCHMARK_MANIFEST.md",
    "reports/STAGE16_CONTINUOUS_BASE_CHALLENGE_DECISION.md",
    "scripts/stage16_prepare.py",
    "src/go2arx5_vla/benchmark/native_suite.py",
    "src/go2arx5_vla/benchmark/progressive_relocation.py",
    "artifact_manifests/stage15_native_wbc.yaml",
)
V1_CONTRACT_FILES = (
    "src/go2arx5_vla/contracts/trajectory.py",
    "src/go2arx5_vla/contracts/trajectory_shape.py",
    "src/go2arx5_vla/trajectory/preview.py",
    "src/go2arx5_vla/adapters/groot_to_wbc.py",
    "src/go2arx5_vla/diagnostics/stage23_action_pipeline.py",
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage35_motion_api_authority_closure_20260905/stage35_cpu_20260905_v2"
)
PROTOCOL_MANIFEST_PATH = REPO / "manifests/STAGE35_ORACLE_CORPUS_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def git_output(repo: Path, *args: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_sha(repo: Path, revision: str = "HEAD") -> str | None:
    code, stdout, _ = git_output(repo, "rev-parse", revision)
    return stdout if code == 0 else None


def current_branch(repo: Path) -> str | None:
    code, stdout, _ = git_output(repo, "branch", "--show-current")
    return stdout if code == 0 else None


def worktree_status(repo: Path) -> list[str]:
    code, stdout, _ = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    return stdout.splitlines() if code == 0 and stdout else []


def git_blob_sha(repo: Path, revision: str, relative_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return sha256_bytes(result.stdout)


def file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "expected_sha256": expected_sha256,
    }
    if path.is_file():
        actual = sha256_file(path)
        record.update(
            {
                "size_bytes": int(path.stat().st_size),
                "sha256": actual,
                "hash_status": (
                    "MATCH"
                    if expected_sha256 is None or actual == expected_sha256
                    else "MISMATCH"
                ),
            }
        )
    else:
        record.update({"sha256": None, "hash_status": "MISSING"})
    return record


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted({Path(path) for path in paths}, key=str)


def _search_named_files(roots: Iterable[Path], names: Iterable[str]) -> list[Path]:
    """Search bounded, already-referenced lab roots without touching contents."""

    wanted = set(names)
    found: list[Path] = []
    for root in _unique_paths(roots):
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and path.name in wanted:
                    found.append(path)
        except (OSError, RuntimeError):
            # A permission or symlink-loop failure is evidence of an incomplete
            # audit, never a reason to claim that authority was recovered.
            continue
    return _unique_paths(found)


def _candidate_record(
    logical_name: str,
    candidates: Iterable[Path],
    expected_sha256: str | None,
) -> dict[str, Any]:
    paths = _unique_paths(candidates)
    existing = [path for path in paths if path.is_file()]
    records = [file_record(path, expected_sha256=expected_sha256) for path in existing]
    hashes = sorted({str(record["sha256"]) for record in records if record.get("sha256")})
    if not existing:
        status = "MISSING"
    elif len(hashes) > 1:
        status = "AMBIGUOUS_IDENTITY"
    elif expected_sha256 is not None and hashes[0] != expected_sha256:
        status = "MISMATCH"
    elif expected_sha256 is None:
        status = "PRESENT_UNVERIFIED_IDENTITY"
    else:
        status = "MATCH"
    return {
        "logical_name": logical_name,
        "candidate_paths": [str(path) for path in paths],
        "existing_records": records,
        "existing_count": len(existing),
        "distinct_hashes": hashes,
        "expected_sha256": expected_sha256,
        "hash_status": status,
        "selected_path": str(existing[0]) if existing else None,
    }


def _manifest_case_paths(root: Path, payload: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
    cases = payload.get("cases") or payload.get("entries") or payload.get("selected") or []
    paths: list[Path] = []
    forbidden: list[str] = []
    if not isinstance(cases, list):
        return [], ["manifest_cases_not_a_list"]
    for row in cases:
        if not isinstance(row, Mapping):
            continue
        for key in ("path", "trajectory_path", "shape_file", "bound_file", "original_file"):
            value = row.get(key)
            if not value:
                continue
            text = str(value)
            if "validation" in text.lower() or "holdout" in text.lower():
                forbidden.append(text)
                continue
            candidate = Path(text)
            paths.append(candidate if candidate.is_absolute() else root / candidate)
    return _unique_paths(paths), forbidden


def _audit_stage16_root(root: Path) -> dict[str, Any]:
    manifest_records: dict[str, Any] = {}
    all_case_paths: list[Path] = []
    forbidden: list[str] = []
    parse_errors: list[str] = []
    counts: dict[str, int | None] = {}
    for relative in STAGE16_MANIFEST_RELATIVE_PATHS:
        path = root / relative
        record = file_record(path)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                paths, forbidden_paths = _manifest_case_paths(root, payload)
                cases = payload.get("cases") or payload.get("entries") or payload.get("selected") or []
                all_case_paths.extend(paths)
                forbidden.extend(forbidden_paths)
                record["case_count"] = len(cases) if isinstance(cases, list) else None
                record["referenced_case_count"] = len(paths)
                record["referenced_case_paths_exist"] = bool(paths) and all(
                    path.exists() for path in paths
                )
                if relative.endswith("native_suite_manifest.json"):
                    counts["native_suite"] = record["case_count"]
                elif relative.endswith("first_eight_manifest.json"):
                    counts["first_eight"] = record["case_count"]
                elif relative.endswith("progressive_cohort.json"):
                    counts["progressive"] = int(
                        payload.get("selected_count", record["case_count"] or 0)
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                record["parse_error"] = str(exc)
                parse_errors.append(f"{path}: {exc}")
        manifest_records[relative] = record
    shape_pass = bool(
        counts.get("native_suite") == 28
        and counts.get("first_eight") == 8
        and counts.get("progressive") == 8
        and not parse_errors
    )
    cases_exist = bool(all_case_paths) and all(path.exists() for path in all_case_paths)
    return {
        "root": str(root),
        "root_record": file_record(root),
        "manifests": manifest_records,
        "manifest_shape": {
            "native_suite_count": counts.get("native_suite"),
            "first_eight_count": counts.get("first_eight"),
            "progressive_count": counts.get("progressive"),
            "expected_native_suite_count": 28,
            "expected_primary_nominal_count": 27,
            "expected_first_eight_count": 8,
            "expected_progressive_count": 8,
            "manifest_shape_pass": shape_pass,
            "all_referenced_case_paths_exist": cases_exist,
        },
        "forbidden_reference_paths": sorted(set(forbidden)),
        "parse_errors": parse_errors,
        "read_only": True,
        "complete_historical_corpus_shape": bool(shape_pass and cases_exist and not forbidden),
    }


def _audit_stage16_authority(stage16_root: Path) -> dict[str, Any]:
    search_roots = _unique_paths((*SEARCH_ROOTS, stage16_root.parent))
    discovered_named = _search_named_files(search_roots, WBC_SEARCH_FILENAMES)
    discovered_roots = [
        path.parent
        for path in discovered_named
        if path.name in {"native_suite_manifest.json", "first_eight_manifest.json"}
    ]
    root_candidates = _unique_paths(
        [stage16_root]
        + [root for root in discovered_roots if root.name == "go2arx5-wbc-stage16"]
    )
    root_audits = [_audit_stage16_root(root) for root in root_candidates]
    complete_roots = [
        audit for audit in root_audits if audit["complete_historical_corpus_shape"]
    ]
    root_hash_signatures = [
        sha256_bytes(canonical_json(audit["manifests"])) for audit in complete_roots
    ]
    root_ambiguous = len(set(root_hash_signatures)) > 1

    stage16_wbc_root = stage16_root / "resources"
    wbc_root = Path("/var/cache/users/liuyu-nfs/go2arx5-wbc-stage13/extracted/WBCTrackingShare")
    fixed_candidates = {
        "delivered_wbc_archive": [
            Path("/var/cache/users/liuyu-nfs/go2arx5-experiment-e2/intake/incoming/WBCTrackingShare_20260831.zip")
        ],
        "native_urdf": [
            wbc_root / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
        ],
        "arm_checkpoint": [
            stage16_root / "checkpoints/arm/ac_weights_039999.pt",
            wbc_root / "checkpoints/arm/ac_weights_039999.pt",
            wbc_root / "resources/checkpoints/arm/ac_weights_039999.pt",
        ],
        "dog_checkpoint": [
            stage16_root / "checkpoints/dog/ac_weights_039999.pt",
            wbc_root / "checkpoints/dog/ac_weights_039999.pt",
            wbc_root / "resources/checkpoints/dog/ac_weights_039999.pt",
        ],
        "native_config": [
            stage16_root / "config/wbc.py",
            wbc_root / "go1_gym/envs/roboduet/wbc.py",
            wbc_root / "config/wbc.py",
        ],
    }
    basename_candidates = {
        "delivered_wbc_archive": [
            path for path in discovered_named if path.name == "WBCTrackingShare_20260831.zip"
        ],
        "native_urdf": [path for path in discovered_named if path.name == "go2_x5.urdf"],
        "arm_checkpoint": [
            path for path in discovered_named if path.name == "ac_weights_039999.pt" and "arm" in str(path)
        ],
        "dog_checkpoint": [
            path for path in discovered_named if path.name == "ac_weights_039999.pt" and "dog" in str(path)
        ],
        "native_config": [path for path in discovered_named if path.name == "wbc.py"],
    }
    wbc_records: dict[str, Any] = {}
    for logical_name, candidates in fixed_candidates.items():
        wbc_records[logical_name] = _candidate_record(
            logical_name,
            [*candidates, *basename_candidates[logical_name]],
            EXPECTED_WBC_HASHES[logical_name],
        )
    reach_candidates = [
        wbc_root / "resources/reach_tables/go2_x5_2d.pt",
        stage16_root / "resources/reach_tables/go2_x5_2d.pt",
        *[path for path in discovered_named if path.name == "go2_x5_2d.pt"],
    ]
    wbc_records["reach_table"] = _candidate_record("reach_table", reach_candidates, None)
    wbc_records["reach_table"]["identity_binding"] = (
        "MISSING_EXPECTED_HISTORICAL_HASH" if wbc_records["reach_table"]["hash_status"] != "MISSING" else "MISSING"
    )

    source_records = {
        relative: file_record(REPO / relative) for relative in STAGE16_SOURCE_FILES
    }
    ambiguous = bool(root_ambiguous) or any(
        record["hash_status"] == "AMBIGUOUS_IDENTITY" for record in wbc_records.values()
    )
    exact_wbc = all(
        wbc_records[name]["hash_status"] == "MATCH" for name in EXPECTED_WBC_HASHES
    ) and wbc_records["reach_table"]["hash_status"] == "MATCH"
    recovered = bool(complete_roots) and exact_wbc and not ambiguous
    if ambiguous:
        status = "AMBIGUOUS_IDENTITY"
    elif recovered:
        status = "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
    else:
        status = "HISTORICAL_STAGE16_AUTHORITY_NOT_RECOVERED"
    blockers = []
    if not complete_roots:
        blockers.append("complete Stage16 28-case corpus shape and referenced paths not proven")
    if not exact_wbc:
        blockers.append("exact delivered WBC inputs and reach-table identity not proven")
    if any(record["hash_status"] == "MISMATCH" for record in wbc_records.values()):
        blockers.append("one or more candidate WBC inputs hash-mismatch the registered identity")
    if ambiguous:
        blockers.append("multiple distinct candidate identities were found for a required artifact")
    return {
        "schema": "STAGE35_A_STAGE16_AUTHORITY_AUDIT_V1",
        "status": status,
        "expected_stage16_root": str(stage16_root),
        "search_roots": [str(root) for root in search_roots],
        "root_candidates": root_audits,
        "complete_root_count": len(complete_roots),
        "root_identity_ambiguous": root_ambiguous,
        "wbc_inputs": wbc_records,
        "wbc_exact_identity_pass": bool(exact_wbc),
        "stage16_source_backed_repo_files": source_records,
        "source_backed_protocol_present": bool(all(record["is_file"] for record in source_records.values())),
        "read_only": True,
        "validation_e_read": False,
        "holdout_f_read": False,
        "blockers": blockers,
    }


def _case_to_chunk(payload: Mapping[str, Any]) -> TrajectoryChunkV1:
    timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
    positions = np.asarray(payload["eef_position_m"], dtype=np.float64)
    orientations = np.asarray(payload["eef_orientation_xyzw"], dtype=np.float64)
    gripper = np.asarray(payload["gripper"], dtype=np.float64)
    return TrajectoryChunkV1(
        chunk_id=str(payload["case_id"]),
        source="STAGE35_DETERMINISTIC_CANDIDATE_CORPUS",
        observation_timestamp=0.0,
        point_timestamps=timestamps,
        eef_position=positions,
        eef_orientation=orientations,
        source_frame=str(payload["source_frame"]),
        tcp_semantic_id=str(payload["tcp_semantic_id"]),
        validity_start=float(payload["validity_start"]),
        validity_end=float(payload["validity_end"]),
        gripper=gripper,
        provenance={
            "stage35_candidate_corpus": True,
            "authoritative_stage16_source": False,
            "native_wbc_inputs_bound": False,
            "family": payload["family"],
        },
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _arc_length(positions: np.ndarray, orientations: np.ndarray) -> float:
    translation = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    rotation = geodesic_angle(
        quat_to_matrix(orientations[:-1]), quat_to_matrix(orientations[1:])
    )
    return float(np.sum(np.sqrt(translation * translation + (DEFAULT_SE3_LAMBDA_M_PER_RAD * rotation) ** 2)))


def _geometry_row(
    dense: TrajectoryChunkV1,
    candidate: MotionIntentV2,
    *,
    case_id: str,
    k: int,
) -> dict[str, Any]:
    u_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    dense_geometry = MotionIntentV2.from_trajectory_chunk(dense)
    dense_position, dense_orientation = dense_geometry.pose_at_progress(u_grid)
    candidate_position, candidate_orientation = candidate.pose_at_progress(u_grid)
    position_errors = np.linalg.norm(dense_position - candidate_position, axis=-1)
    rotation_errors = geodesic_angle(
        quat_to_matrix(dense_orientation), quat_to_matrix(candidate_orientation)
    )
    dense_arc = _arc_length(dense.eef_position, dense.eef_orientation)
    candidate_chunk = candidate.to_trajectory_chunk()
    candidate_arc = _arc_length(candidate_chunk.eef_position, candidate_chunk.eef_orientation)
    preview = evaluate_preview_equivalence(
        dense, candidate, case_id=case_id, representation=f"K{k}", anchor_count=101
    )
    start_end_exact = bool(
        np.allclose(dense_position[0], candidate_position[0], atol=1e-12, rtol=0.0)
        and np.allclose(dense_position[-1], candidate_position[-1], atol=1e-12, rtol=0.0)
        and np.allclose(dense_orientation[0], candidate_orientation[0], atol=1e-12, rtol=0.0)
        and np.allclose(dense_orientation[-1], candidate_orientation[-1], atol=1e-12, rtol=0.0)
    )
    finite = bool(np.isfinite(position_errors).all() and np.isfinite(rotation_errors).all())
    return {
        "case_id": case_id,
        "K": int(k),
        "control_pose_count": candidate.point_count,
        "geometry_u_knots": candidate.geometry_arc_fraction.tolist(),
        "position_error_m": _summary(position_errors),
        "rotation_error_rad": _summary(rotation_errors),
        "se3_arc_length_ratio": float(candidate_arc / dense_arc) if dense_arc > 1e-12 else 1.0,
        "se3_arc_length_relative_error": float(abs(candidate_arc - dense_arc) / max(dense_arc, 1e-12)),
        "start_end_pose_exact": start_end_exact,
        "finite": finite,
        "preview_diagnostic": public_evaluation(preview),
        "_position_errors": position_errors,
        "_rotation_errors": rotation_errors,
        "_preview_position_errors": np.asarray(preview["_position_errors"]),
        "_preview_rotation_errors": np.asarray(preview["_rotation_errors"]),
        "_preview_progress_errors": np.asarray(preview["_progress_errors"]),
        "_preview_arc_relative_error": preview["_arc_relative_error"],
        "_preview_endpoint_clamp_agreement": preview["_endpoint_clamp_agreement"],
    }


def _aggregate_geometry(rows: Iterable[Mapping[str, Any]], k: int) -> dict[str, Any]:
    rows = list(rows)
    positions = np.concatenate([np.asarray(row["_position_errors"]) for row in rows])
    rotations = np.concatenate([np.asarray(row["_rotation_errors"]) for row in rows])
    arcs = np.asarray([row["se3_arc_length_relative_error"] for row in rows], dtype=np.float64)
    pass_limits = bool(
        all(bool(row["finite"]) and bool(row["start_end_pose_exact"]) for row in rows)
        and _summary(positions)["p95"] < DEFAULT_PREVIEW_POSITION_P95_LIMIT_M
        and _summary(rotations)["p95"] < DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD
    )
    preview_rows = [
        {
            "_position_errors": row["_preview_position_errors"],
            "_rotation_errors": row["_preview_rotation_errors"],
            "_progress_errors": row["_preview_progress_errors"],
            "_arc_relative_error": row["_preview_arc_relative_error"],
            "_endpoint_clamp_agreement": row["_preview_endpoint_clamp_agreement"],
            "finite": row["preview_diagnostic"]["finite"],
            "monotonic_timestamps": row["preview_diagnostic"]["monotonic_timestamps"],
            "start_end_pose_exact": row["preview_diagnostic"]["start_end_pose_exact"],
            "duration_exact": row["preview_diagnostic"]["duration_exact"],
            "passes_limits": row["preview_diagnostic"]["passes_limits"],
            "anchor_count": row["preview_diagnostic"]["anchor_count"],
            "se3_arc_length_ratio": row["preview_diagnostic"]["se3_arc_length_ratio"],
        }
        for row in rows
    ]
    preview_aggregate = aggregate_evaluations(preview_rows, representation=f"K{k}")
    return {
        "representation": f"K{k}",
        "case_count": len(rows),
        "geometry_position_error_m": _summary(positions),
        "geometry_rotation_error_rad": _summary(rotations),
        "se3_arc_length_relative_error": _summary(arcs),
        "finite": bool(all(bool(row["finite"]) for row in rows)),
        "start_end_pose_exact": bool(all(bool(row["start_end_pose_exact"]) for row in rows)),
        "passes_geometry_limits": pass_limits,
        "preview_diagnostic": {
            key: value
            for key, value in preview_aggregate.items()
            if key != "per_case"
        },
    }


def _public_geometry_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _run_geometry(chunks: Mapping[str, TrajectoryChunkV1], reference_status: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for k in GEOMETRY_K_VALUES:
        rows: list[dict[str, Any]] = []
        for case_id in case_ids():
            dense = chunks[case_id]
            candidate = build_geometry_candidate(dense, k)
            rows.append(_geometry_row(dense, candidate, case_id=case_id, k=k))
        all_rows[str(k)] = rows
        aggregate = _aggregate_geometry(rows, k)
        results[str(k)] = {
            "K": int(k),
            "case_count": len(rows),
            "aggregate": aggregate,
            "per_case": [_public_geometry_row(row) for row in rows],
            "promotion_status": (
                "BLOCKED_NATIVE_STAGE16_AUTHORITY"
                if reference_status != "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
                else "ELIGIBLE_FOR_REGISTERED_SELECTION"
            ),
        }
    diagnostic_winner = next(
        (
            int(k)
            for k in GEOMETRY_K_VALUES
            if results[str(k)]["aggregate"]["passes_geometry_limits"]
        ),
        None,
    )
    return {
        "schema": "STAGE35_D1_GEOMETRY_QUALIFICATION_V1",
        "method": {
            "se3_lambda_m_per_rad": DEFAULT_SE3_LAMBDA_M_PER_RAD,
            "K_values": list(GEOMETRY_K_VALUES),
            "knots": "uniform_normalized_se3_arc_including_endpoints",
            "interpolation": "linear_xyz_quaternion_slerp",
            "outcome_dependent_knot_choice": False,
            "smoothing": False,
            "spline_fitting": False,
            "trajectory_optimization": False,
            "manual_knot_placement": False,
            "comparison_domain": "same_normalized_geometry_progress_u",
        },
        "thresholds": {
            "preview_position_p95_limit_m": DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
            "preview_rotation_p95_limit_rad": DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
            "endpoint_clamp_agreement_min": DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
            "start_end_pose_exact": True,
            "duration_exact": True,
        },
        "reference_provenance_status": reference_status,
        "K_RESULTS_4_6_9_12": results,
        "diagnostic_winner_without_promotion": diagnostic_winner,
        "K_STAR": (
            diagnostic_winner
            if reference_status == "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
            else None
        ),
        "GEOMETRY_RESULT": (
            "GEOMETRY_QUALIFIED"
            if diagnostic_winner is not None
            and reference_status == "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
            else "INFRASTRUCTURE_PROVENANCE_BLOCKER"
        ),
    }, all_rows


def _timing_row(
    dense: TrajectoryChunkV1,
    geometry: MotionIntentV2,
    mode: str,
    *,
    case_id: str,
    k: int,
) -> dict[str, Any]:
    timing = build_timing_candidate(dense, geometry, mode)
    dense_v2 = MotionIntentV2.from_trajectory_chunk(dense)
    times = np.linspace(dense.validity_start, dense.validity_end, 101, dtype=np.float64)
    source_progress = np.asarray(dense_v2.progress_at(times), dtype=np.float64)
    candidate_progress = np.asarray(timing.progress_at(times), dtype=np.float64)
    progress_errors = np.abs(source_progress - candidate_progress)
    preview = evaluate_preview_equivalence(
        dense, timing, case_id=case_id, representation=f"K{k}+{mode}", anchor_count=101
    )
    return {
        "case_id": case_id,
        "K": int(k),
        "timing_mode": mode,
        "timing_u_knots": timing.timing_arc_fraction.tolist(),
        "timing_timestamps": timing.timing_timestamps.tolist(),
        "time_to_arc_progress_error": _summary(progress_errors),
        "monotonic": bool(np.all(np.diff(timing.timing_timestamps) > 0.0)),
        "finite": bool(np.isfinite(progress_errors).all()),
        "duration_exact": bool(
            abs((timing.validity_end - timing.validity_start) - (dense.validity_end - dense.validity_start))
            <= 1e-10
        ),
        "preview_diagnostic": public_evaluation(preview),
        "_progress_errors": progress_errors,
        "_preview_position_errors": np.asarray(preview["_position_errors"]),
        "_preview_rotation_errors": np.asarray(preview["_rotation_errors"]),
        "_preview_progress_errors": np.asarray(preview["_progress_errors"]),
        "_preview_arc_relative_error": preview["_arc_relative_error"],
        "_preview_endpoint_clamp_agreement": preview["_endpoint_clamp_agreement"],
    }


def _aggregate_timing(rows: Iterable[Mapping[str, Any]], mode: str, k: int) -> dict[str, Any]:
    rows = list(rows)
    progress = np.concatenate([np.asarray(row["_progress_errors"]) for row in rows])
    preview_rows = [
        {
            "_position_errors": row["_preview_position_errors"],
            "_rotation_errors": row["_preview_rotation_errors"],
            "_progress_errors": row["_preview_progress_errors"],
            "_arc_relative_error": row["_preview_arc_relative_error"],
            "_endpoint_clamp_agreement": row["_preview_endpoint_clamp_agreement"],
            "finite": row["preview_diagnostic"]["finite"],
            "monotonic_timestamps": row["preview_diagnostic"]["monotonic_timestamps"],
            "start_end_pose_exact": row["preview_diagnostic"]["start_end_pose_exact"],
            "duration_exact": row["preview_diagnostic"]["duration_exact"],
            "passes_limits": row["preview_diagnostic"]["passes_limits"],
            "anchor_count": row["preview_diagnostic"]["anchor_count"],
            "se3_arc_length_ratio": row["preview_diagnostic"]["se3_arc_length_ratio"],
        }
        for row in rows
    ]
    preview = aggregate_evaluations(preview_rows, representation=f"K{k}+{mode}")
    return {
        "representation": f"K{k}+{mode}",
        "case_count": len(rows),
        "time_to_arc_progress_error": _summary(progress),
        "finite": bool(all(bool(row["finite"]) for row in rows)),
        "monotonic": bool(all(bool(row["monotonic"]) for row in rows)),
        "duration_exact": bool(all(bool(row["duration_exact"]) for row in rows)),
        "preview_diagnostic": {key: value for key, value in preview.items() if key != "per_case"},
        "passes_timing_limits": bool(
            all(bool(row["finite"]) and bool(row["monotonic"]) and bool(row["duration_exact"]) for row in rows)
            and bool(preview["passes_limits"])
        ),
    }


def _run_timing(
    chunks: Mapping[str, TrajectoryChunkV1],
    geometry_rows: Mapping[str, list[Mapping[str, Any]]],
    reference_status: str,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    results: dict[str, Any] = {}
    raw_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for k in GEOMETRY_K_VALUES:
        mode_results: dict[str, Any] = {}
        mode_rows: dict[str, list[dict[str, Any]]] = {}
        for mode in TIMING_MODES:
            rows: list[dict[str, Any]] = []
            for case_id in case_ids():
                dense = chunks[case_id]
                geometry = build_geometry_candidate(dense, k)
                rows.append(_timing_row(dense, geometry, mode, case_id=case_id, k=k))
            mode_rows[mode] = rows
            mode_results[mode] = {
                "timing_mode": mode,
                "timing_knots": TIMING_KNOTS[mode].tolist(),
                "aggregate": _aggregate_timing(rows, mode, k),
                "per_case": [
                    {key: value for key, value in row.items() if not str(key).startswith("_")}
                    for row in rows
                ],
                "promotion_status": (
                    "BLOCKED_NATIVE_STAGE16_AUTHORITY"
                    if reference_status != "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
                    else "ELIGIBLE_AFTER_K_SELECTION"
                ),
            }
        diagnostic_winner = next(
            (mode for mode in TIMING_MODES if mode_results[mode]["aggregate"]["passes_timing_limits"]),
            None,
        )
        results[str(k)] = {
            "K": int(k),
            "T_RESULTS_T1_T3_T5_T9": mode_results,
            "diagnostic_winner_without_promotion": diagnostic_winner,
            "T_STAR": (
                diagnostic_winner
                if reference_status == "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
                else None
            ),
        }
        raw_rows[str(k)] = mode_rows
    return {
        "schema": "STAGE35_D2_TIMING_QUALIFICATION_V1",
        "method": {
            "timing_modes": list(TIMING_MODES),
            "timing_interpolation": "monotonic_piecewise_linear",
            "comparison_domain": "same_geometry_control_poses_with_independent_u_of_t",
            "K_values_reported": list(GEOMETRY_K_VALUES),
            "outcome_dependent_timing_choice": False,
        },
        "reference_provenance_status": reference_status,
        "results_by_K": results,
        "T_STAR": None,
        "TIMING_RESULT": "INFRASTRUCTURE_PROVENANCE_BLOCKER",
    }, raw_rows


def _run_disentanglement(
    chunks: Mapping[str, TrajectoryChunkV1],
) -> dict[str, Any]:
    case_id = "T01"
    dense = chunks[case_id]
    geometry = build_geometry_candidate(dense, 12)
    t1 = build_timing_candidate(dense, geometry, "T1")
    t9 = build_timing_candidate(dense, geometry, "T9")
    u_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    p1, q1 = t1.pose_at_progress(u_grid)
    p9, q9 = t9.pose_at_progress(u_grid)
    times = np.linspace(dense.validity_start, dense.validity_end, 101, dtype=np.float64)
    progress_diff = np.abs(np.asarray(t1.progress_at(times)) - np.asarray(t9.progress_at(times)))
    geometry_k4 = build_geometry_candidate(dense, 4)
    k4_t9 = build_timing_candidate(dense, geometry_k4, "T9")
    same_timing_progress_diff = np.max(
        np.abs(np.asarray(t9.progress_at(times)) - np.asarray(k4_t9.progress_at(times)))
    )
    changed_geometry = np.linalg.norm(
        geometry.pose_at_progress(u_grid)[0] - geometry_k4.pose_at_progress(u_grid)[0], axis=-1
    )
    return {
        "schema": "STAGE35_B_GEOMETRY_TIMING_DISENTANGLEMENT_V1",
        "case_id": case_id,
        "same_geometry_different_timing": {
            "geometry_u_grid_max_position_delta_m": float(np.max(np.linalg.norm(p1 - p9, axis=-1))),
            "geometry_u_grid_max_rotation_delta_rad": float(
                np.max(geodesic_angle(quat_to_matrix(q1), quat_to_matrix(q9)))
            ),
            "timing_law_max_progress_delta": float(np.max(progress_diff)),
            "geometry_is_identical_at_same_u": bool(
                np.allclose(p1, p9, atol=1e-12, rtol=0.0)
                and np.allclose(q1, q9, atol=1e-12, rtol=0.0)
            ),
            "timing_law_is_distinct_on_test_case": bool(np.max(progress_diff) > 1e-6),
        },
        "same_timing_different_geometry": {
            "timing_law_max_progress_delta": float(same_timing_progress_diff),
            "geometry_max_position_delta_m": float(np.max(changed_geometry)),
            "timing_is_identical": bool(same_timing_progress_diff <= 1e-12),
            "geometry_is_distinct_on_test_case": bool(np.max(changed_geometry) > 1e-6),
        },
        "causal_disentanglement_pass": bool(
            np.allclose(p1, p9, atol=1e-12, rtol=0.0)
            and np.allclose(q1, q9, atol=1e-12, rtol=0.0)
            and np.max(progress_diff) > 1e-6
            and same_timing_progress_diff <= 1e-12
            and np.max(changed_geometry) > 1e-6
        ),
        "model_constructed": False,
        "wbc_executed": False,
    }


def _check_raises(callable_obj: Any) -> bool:
    try:
        callable_obj()
    except (ValueError, TypeError):
        return True
    return False


def _run_semantic_closure(base_head: str | None, chunks: Mapping[str, TrajectoryChunkV1]) -> dict[str, Any]:
    dense = chunks["N01"]
    geometry = build_geometry_candidate(dense, 6)
    timing = build_timing_candidate(dense, geometry, "T1")
    query_times = np.linspace(dense.validity_start, dense.validity_end, 17, dtype=np.float64)
    materialized = timing.to_trajectory_chunk(
        expected_source_frame=SOURCE_FRAME,
        expected_tcp_semantic_id=TCP_SEMANTIC_ID,
        query_timestamps=query_times,
    )
    direct_progress = np.asarray(timing.progress_at(query_times))
    direct_position, direct_orientation = timing.pose_at_progress(direct_progress)
    materialization_exact = bool(
        np.allclose(materialized.eef_position, direct_position, atol=1e-12, rtol=0.0)
        and np.allclose(materialized.eef_orientation, direct_orientation, atol=1e-12, rtol=0.0)
    )
    full_density_v2 = MotionIntentV2.from_trajectory_chunk(dense, timing_mode="T0")
    full_density = full_density_v2.to_trajectory_chunk(
        expected_source_frame=SOURCE_FRAME,
        expected_tcp_semantic_id=TCP_SEMANTIC_ID,
        query_timestamps=dense.point_timestamps,
    )
    t0_source_reproduction = bool(
        np.allclose(full_density.eef_position, dense.eef_position, atol=1e-12, rtol=0.0)
        and np.allclose(full_density.eef_orientation, dense.eef_orientation, atol=1e-12, rtol=0.0)
        and np.allclose(full_density.point_timestamps, dense.point_timestamps, atol=1e-12, rtol=0.0)
    )
    try:
        adapter = GR00TToWBCAdapter(
            ReferenceAdapterConfig(
                expected_source_frame=SOURCE_FRAME,
                wbc_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_SEMANTIC_ID,
            )
        )
        adapter_preview, adapter_gripper = adapter.adapt(
            materialized, controller_timestamp=float(query_times[3])
        )
        adapter_pass = bool(
            np.isfinite(adapter_preview.eef_position).all()
            and np.isfinite(adapter_preview.eef_rot6d).all()
            and adapter_preview.source_frame == SOURCE_FRAME
            and adapter_preview.tcp_semantic_id == TCP_SEMANTIC_ID
            and adapter_gripper is not None
        )
        adapter_error = None
    except (ValueError, TypeError, RuntimeError) as exc:
        adapter_pass = False
        adapter_error = str(exc)
    endpoint_clamp = bool(
        float(timing.progress_at(-1.0)) == 0.0
        and float(timing.progress_at(3.0)) == 1.0
        and np.allclose(timing.pose_at_progress(-1.0)[0], timing.eef_position[0], atol=1e-12)
        and np.allclose(timing.pose_at_progress(2.0)[0], timing.eef_position[-1], atol=1e-12)
    )
    fields = dict(timing.__dict__)
    bad_timing = dict(fields)
    bad_timing.update(
        {
            "timing_arc_fraction": np.asarray([0.0, 0.5, 0.5]),
            "timing_timestamps": np.asarray([timing.validity_start, 1.0, 1.0]),
        }
    )
    bad_pose = dict(fields)
    bad_position = timing.eef_position.copy()
    bad_position[0, 0] = np.nan
    bad_pose["eef_position"] = bad_position
    frame_guard = _check_raises(
        lambda: timing.to_trajectory_chunk(expected_source_frame="WRONG_FRAME")
    )
    tcp_guard = _check_raises(
        lambda: timing.to_trajectory_chunk(expected_tcp_semantic_id="WRONG_TCP")
    )
    timing_guard = _check_raises(lambda: MotionIntentV2(**bad_timing))
    finite_guard = _check_raises(lambda: MotionIntentV2(**bad_pose))
    v1_files: dict[str, Any] = {}
    v1_unchanged = True
    if base_head is not None:
        for relative in V1_CONTRACT_FILES:
            current = REPO / relative
            current_sha = sha256_file(current) if current.is_file() else None
            base_sha = git_blob_sha(REPO, base_head, relative)
            unchanged = current_sha is not None and base_sha is not None and current_sha == base_sha
            v1_files[relative] = {
                "current_sha256": current_sha,
                "base_sha256": base_sha,
                "unchanged_from_stage34": unchanged,
            }
            v1_unchanged = v1_unchanged and unchanged
    else:
        v1_unchanged = False
    static_paths = [
        REPO / "scripts/stage35_motion_api_authority_closure.py",
        REPO / "src/go2arx5_vla/contracts/motion_intent.py",
        REPO / "src/go2arx5_vla/oracle/stage35_corpus.py",
    ]
    static_pass = True
    static_records = []
    forbidden_imports = {"torch", "transformers", "isaacgym", "omni"}
    for path in static_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        record = {
            "path": str(path),
            "imports": sorted(set(imports)),
            "forbidden_imports_absent": not bool(set(imports) & forbidden_imports),
            "forbidden_execution_calls_absent": not bool(
                called_names
                & {
                    "qsub",
                    "train",
                    "backward",
                    "step",
                    "resume",
                    "get_action",
                    "from_pretrained",
                }
            ),
        }
        static_records.append(record)
        static_pass = static_pass and bool(
            record["forbidden_imports_absent"] and record["forbidden_execution_calls_absent"]
        )
    contract_payload = timing.as_dict()
    contract_hash = sha256_bytes(canonical_json(contract_payload))
    tests_pass = bool(
        materialization_exact
        and t0_source_reproduction
        and adapter_pass
        and endpoint_clamp
        and frame_guard
        and tcp_guard
        and timing_guard
        and finite_guard
        and v1_unchanged
        and static_pass
    )
    return {
        "schema": "STAGE35_BC_MOTION_INTENT_V2_SEMANTIC_CLOSURE_V1",
        "status": "PASS" if tests_pass else "FAIL",
        "contract_hash": contract_hash,
        "materialization": {
            "query_count": int(len(query_times)),
            "u_then_gamma_exact": materialization_exact,
            "source_frame_guard": frame_guard,
            "tcp_semantic_guard": tcp_guard,
            "endpoint_clamp": endpoint_clamp,
            "t0_full_density_source_reproduction": t0_source_reproduction,
            "existing_groot_to_wbc_adapter_path": adapter_pass,
            "existing_groot_to_wbc_adapter_error": adapter_error,
        },
        "validation_guards": {
            "nonmonotonic_timing_rejected": timing_guard,
            "nonfinite_pose_rejected": finite_guard,
        },
        "v1_and_stage23_abi_unchanged": v1_unchanged,
        "v1_contract_files": v1_files,
        "static_zero_model_checks": {
            "pass": static_pass,
            "records": static_records,
        },
        "production_v1_replaced": False,
        "model_constructed": False,
        "gr00t_forward": False,
        "wbc_executed": False,
    }


def _write_corpus(output_root: Path) -> tuple[dict[str, TrajectoryChunkV1], dict[str, Any]]:
    protocol = protocol_manifest()
    committed_protocol = json.loads(PROTOCOL_MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol_match = canonical_json(protocol) == canonical_json(committed_protocol)
    if not protocol_match:
        raise RuntimeError("committed Stage35 corpus protocol does not match generator protocol")
    corpus_dir = output_root / "corpus/cases"
    chunks: dict[str, TrajectoryChunkV1] = {}
    case_records: list[dict[str, Any]] = []
    for family in CORPUS_CLASSES:
        for index in range(CASES_PER_CLASS):
            payload = build_case(index, family)
            case_id = str(payload["case_id"])
            if case_id not in case_ids() or case_id in chunks:
                raise RuntimeError(f"invalid or duplicate generated case: {case_id}")
            chunk = _case_to_chunk(payload)
            chunks[case_id] = chunk
            path = corpus_dir / f"{case_id}.json"
            write_json(path, payload)
            case_records.append(
                {
                    "case_id": case_id,
                    "family": family,
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "support_proof_status": payload["support_witness"]["support_proof_status"],
                }
            )
    case_order = {case_id: index for index, case_id in enumerate(case_ids())}
    case_records.sort(key=lambda row: case_order[row["case_id"]])
    tree_sha = sha256_bytes(
        canonical_json(
            [{"relative_path": row["relative_path"], "sha256": row["sha256"]} for row in case_records]
        )
    )
    generated_manifest = {
        "schema": "STAGE35_GENERATED_ORACLE_CORPUS_MANIFEST_V1",
        "protocol_schema": protocol["schema"],
        "generator_version": protocol["generator_version"],
        "case_count": len(case_records),
        "class_counts": {family: sum(row["family"] == family for row in case_records) for family in CORPUS_CLASSES},
        "case_ids": [row["case_id"] for row in case_records],
        "case_records": case_records,
        "corpus_tree_sha256": tree_sha,
        "authoritative_stage16_source": False,
        "native_wbc_inputs_bound": False,
        "promotion_status": "BLOCKED_NATIVE_STAGE16_AND_WBC_PROVENANCE",
    }
    write_json(output_root / "stage35_oracle_corpus_manifest.json", generated_manifest)
    protocol_output = {
        "schema": "STAGE35_ORACLE_CORPUS_PROTOCOL_EVIDENCE_V1",
        "committed_protocol_path": str(PROTOCOL_MANIFEST_PATH),
        "committed_protocol_sha256": sha256_file(PROTOCOL_MANIFEST_PATH),
        "generator_module": str(REPO / "src/go2arx5_vla/oracle/stage35_corpus.py"),
        "generator_module_sha256": sha256_file(REPO / "src/go2arx5_vla/oracle/stage35_corpus.py"),
        "protocol": protocol,
        "generated_manifest": str(output_root / "stage35_oracle_corpus_manifest.json"),
        "generated_manifest_sha256": sha256_file(output_root / "stage35_oracle_corpus_manifest.json"),
        "corpus_tree_sha256": tree_sha,
        "authoritative": False,
        "blocker": "native Stage16 reference and exact WBC/reach-table inputs are not yet bound",
    }
    write_json(output_root / "stage35_oracle_corpus_protocol.json", protocol_output)
    return chunks, generated_manifest


def _qstat_record() -> dict[str, Any]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    result = subprocess.run(
        ["qstat", "-u", user],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    stage35_jobs = [
        line for line in stdout.splitlines() if "stage35" in line.lower() or "motion_api" in line.lower()
    ]
    return {
        "command": f"qstat -u {user}",
        "returncode": int(result.returncode),
        "stdout": stdout,
        "stderr": result.stderr.strip(),
        "active_stage35_jobs": stage35_jobs,
        "no_active_stage35_job": not stage35_jobs,
    }


def _preflight(repo: Path, stage34_seal: Path, reference_status: str) -> dict[str, Any]:
    head = git_sha(repo)
    branch = current_branch(repo)
    status = worktree_status(repo)
    ancestor_code, _, ancestor_err = git_output(repo, "merge-base", "--is-ancestor", BASE_HEAD, "HEAD")
    stage34_remote = git_sha(repo, f"refs/remotes/origin/{STAGE34_BRANCH}")
    scratch = Path("/scratch/e1192548")
    disk = shutil.disk_usage(scratch) if scratch.exists() else None
    return {
        "schema": "STAGE35_ZERO_MODEL_PREFLIGHT_V1",
        "PI_GATE_READ": PI_GATE_ID,
        "project": PROJECT,
        "base_head": BASE_HEAD,
        "stage35_head": head,
        "branch": branch,
        "expected_branch": STAGE35_BRANCH,
        "base_is_ancestor": ancestor_code == 0,
        "ancestor_error": ancestor_err,
        "stage34_remote_head": stage34_remote,
        "stage34_remote_matches_base": stage34_remote == BASE_HEAD,
        "worktree_clean": not status,
        "worktree_status": status,
        "stage34_seal": file_record(stage34_seal, expected_sha256=STAGE34_SEAL_SHA256),
        "reference_provenance_status": reference_status,
        "pbs": {
            "project": PROJECT,
            "allocation_requested": False,
            "qsub_submitted": False,
            "active_stage35": _qstat_record(),
        },
        "storage": {
            "scratch_root": str(scratch),
            "free_bytes": None if disk is None else int(disk.free),
            "total_bytes": None if disk is None else int(disk.total),
            "output_root_scratch_backed": True,
        },
        "runtime": {
            "hostname": platform.node(),
            "user": os.environ.get("USER"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "execution_boundary": {
            "scientific_execution_begun": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "gr00t_model_constructed": False,
            "gr00t_forward": False,
            "gr00t_sampler_calls": 0,
            "gr00t_checkpoint_loads": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "pbs_submitted": False,
            "gpu_used": False,
            "wbc_executed": False,
            "pr_merge": False,
        },
    }


def _run_e0(reference_audit: Mapping[str, Any], generated_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "STAGE35_E0_CONTINUOUS_BASE_AUDIT_V1",
        "result": "NO_VALID_CONTINUOUS_BASE_COHORT_PROVENANCE",
        "support_authority_status": "SUPPORT_AUTHORITY_MISSING",
        "generated_candidate_cases": int(
            sum(1 for row in generated_manifest.get("case_records", []) if row.get("family") == "C")
        ),
        "analytic_support_witness_only": True,
        "native_reach_table_identity_bound": bool(
            reference_audit.get("wbc_inputs", {}).get("reach_table", {}).get("hash_status") == "MATCH"
        ),
        "source_backed_authority_recovered": reference_audit.get("status")
        == "HISTORICAL_STAGE16_AUTHORITY_RECOVERED",
        "new_wbc_rollouts": 0,
        "new_pbs_allocations": 0,
        "base_hidden_from_wbc_in_candidate_schema": True,
        "fixed_base_impossibility_not_promoted": True,
        "next_gate_required_for_any_continuous_base_rollout": True,
    }


def _seal(
    output_root: Path,
    *,
    stage34_seal: Path,
    generated_files: Iterable[Path],
) -> tuple[dict[str, Any], str]:
    source_files = [
        REPO / "scripts/stage35_motion_api_authority_closure.py",
        REPO / "src/go2arx5_vla/contracts/motion_intent.py",
        REPO / "src/go2arx5_vla/motion_api/qualification.py",
        REPO / "src/go2arx5_vla/oracle/stage35_corpus.py",
        REPO / "tests/test_stage35_motion_api_authority_closure.py",
        REPO / "manifests/STAGE35_ORACLE_CORPUS_V1.json",
        REPO / "reports/STAGE35_MOTION_API_AUTHORITY_CLOSURE.md",
        stage34_seal,
    ]
    paths = {Path(path) for path in source_files if Path(path).is_file()}
    paths.update(Path(path) for path in generated_files if Path(path).is_file())
    paths.discard(output_root / "stage35_artifact_seal.json")
    paths.discard(output_root / "stage35_return.json")
    entries = [
        {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for path in sorted(paths, key=str)
    ]
    payload = {
        "schema": "STAGE35_ARTIFACT_SEAL_V1",
        "stage": "Stage35",
        "pi_gate": PI_GATE_ID,
        "base_head": BASE_HEAD,
        "stage35_head": git_sha(REPO),
        "execution_class": "CPU_ZERO_MODEL_AUTHORITY_AND_SEMANTIC_AUDIT",
        "pbs_job_id": None,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
        "wbc_executed": False,
        "sealed_files": entries,
        "forbidden_actions": {
            "gr00t_model_constructed": False,
            "gr00t_forward": False,
            "gr00t_sampler_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_write": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "pr_merge": False,
        },
    }
    seal_path = output_root / "stage35_artifact_seal.json"
    write_json(seal_path, payload)
    return payload, sha256_file(seal_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.project_repo).resolve()
    stage16_root = Path(args.stage16_root).resolve()
    stage34_seal = Path(args.stage34_seal).resolve()
    output_root = Path(args.output_root).resolve()
    if repo != REPO:
        raise RuntimeError(f"project repo must be the checked-out repository: {REPO}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    reference_audit = _audit_stage16_authority(stage16_root)
    reference_status = str(reference_audit["status"])
    chunks, generated_manifest = _write_corpus(output_root)
    preflight = _preflight(repo, stage34_seal, reference_status)
    semantic = _run_semantic_closure(BASE_HEAD, chunks)
    disentanglement = _run_disentanglement(chunks)
    geometry, geometry_rows = _run_geometry(chunks, reference_status)
    timing, _ = _run_timing(chunks, geometry_rows, reference_status)
    e0 = _run_e0(reference_audit, generated_manifest)

    output_payloads: dict[str, Any] = {
        "stage35_preflight.json": preflight,
        "stage35_a0_stage16_authority_audit.json": reference_audit,
        "stage35_b_motion_intent_semantic_closure.json": semantic,
        "stage35_b_geometry_timing_disentanglement.json": disentanglement,
        "stage35_d1_geometry_qualification.json": geometry,
        "stage35_d2_timing_qualification.json": timing,
        "stage35_e0_continuous_base_audit.json": e0,
    }
    for name, payload in output_payloads.items():
        write_json(output_root / name, _jsonable(payload))

    log_lines = [
        f"{datetime.now(timezone.utc).isoformat()} Stage35 CPU zero-model start",
        f"PI_GATE_READ={PI_GATE_ID}",
        f"STAGE16_AUTHORITY={reference_status}",
        f"CORPUS_CASE_COUNT={generated_manifest['case_count']}",
        f"CORPUS_TREE_SHA256={generated_manifest['corpus_tree_sha256']}",
        f"MOTION_INTENT_V2_SEMANTIC_CLOSURE={semantic['status']}",
        f"CAUSAL_DISENTANGLEMENT={disentanglement['causal_disentanglement_pass']}",
        f"GEOMETRY_RESULT={geometry['GEOMETRY_RESULT']}",
        f"K_STAR={geometry['K_STAR']}",
        f"TIMING_RESULT={timing['TIMING_RESULT']}",
        "REFERENCE_CORPUS_AUTHORITATIVE=false",
        "WBC_EXECUTED=false",
        "PBS_SUBMITTED=false",
        "GPU_USED=false",
        "MODEL_CONSTRUCTED=false",
        "GR00T_FORWARD=false",
        "STAGE35 CPU zero-model stop",
    ]
    log_path = output_root / "stage35_cpu_execution.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    generated_files = [output_root / name for name in output_payloads] + [
        output_root / "stage35_oracle_corpus_protocol.json",
        output_root / "stage35_oracle_corpus_manifest.json",
        log_path,
        *sorted((output_root / "corpus/cases").glob("*.json")),
    ]
    seal_payload, seal_sha256 = _seal(
        output_root,
        stage34_seal=stage34_seal,
        generated_files=generated_files,
    )

    overall_result = (
        "AMBIGUOUS_IDENTITY"
        if reference_status == "AMBIGUOUS_IDENTITY"
        else "INFRASTRUCTURE_PROVENANCE_BLOCKER"
        if reference_status != "HISTORICAL_STAGE16_AUTHORITY_RECOVERED"
        or semantic["status"] != "PASS"
        or not disentanglement["causal_disentanglement_pass"]
        else "GPT_PI_REVIEW_BEFORE_ORACLE_WBC"
    )
    return_payload = {
        "schema": "STAGE35_HOPPER_RETURN_V1",
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "PI_GATE_READ": PI_GATE_ID,
        "BASE_HEAD": BASE_HEAD,
        "STAGE35_HEAD": preflight.get("stage35_head"),
        "BRANCH": preflight.get("branch"),
        "BASE_PR": 20,
        "REFERENCE_PROVENANCE_STATUS": reference_status,
        "REFERENCE_AUDIT_PATH": str(output_root / "stage35_a0_stage16_authority_audit.json"),
        "NEW_CORPUS_STATUS": "DETERMINISTIC_CANDIDATE_GENERATED_NOT_AUTHORITY_PROMOTED",
        "NEW_CORPUS_MANIFEST_PATH": str(output_root / "stage35_oracle_corpus_manifest.json"),
        "NEW_CORPUS_MANIFEST_SHA256": sha256_file(output_root / "stage35_oracle_corpus_manifest.json"),
        "NEW_CORPUS_TREE_SHA256": generated_manifest["corpus_tree_sha256"],
        "NEW_CORPUS_CASE_COUNT": generated_manifest["case_count"],
        "NEW_CORPUS_CLASS_COUNTS": generated_manifest["class_counts"],
        "MOTION_INTENT_V2_CONTRACT_HASH": semantic["contract_hash"],
        "MOTION_INTENT_V2_CONTRACT_STATUS": semantic["status"],
        "CAUSAL_DISENTANGLEMENT_PASS": disentanglement["causal_disentanglement_pass"],
        "K_RESULTS_4_6_9_12": {
            key: {
                "K": row["K"],
                "case_count": row["case_count"],
                "aggregate": row["aggregate"],
                "promotion_status": row["promotion_status"],
            }
            for key, row in geometry["K_RESULTS_4_6_9_12"].items()
        },
        "K_STAR": geometry["K_STAR"],
        "GEOMETRY_RESULT": geometry["GEOMETRY_RESULT"],
        "T_RESULTS_T1_T3_T5_T9": {
            key: {
                "K": row["K"],
                "T_STAR": row["T_STAR"],
                "diagnostic_winner_without_promotion": row["diagnostic_winner_without_promotion"],
                "results": {
                    mode: {
                        "aggregate": mode_row["aggregate"],
                        "promotion_status": mode_row["promotion_status"],
                    }
                    for mode, mode_row in row["T_RESULTS_T1_T3_T5_T9"].items()
                },
            }
            for key, row in timing["results_by_K"].items()
        },
        "T_STAR": timing["T_STAR"],
        "CONTINUOUS_BASE_E0_RESULT": e0["result"],
        "WBC_PAIRED_EXECUTED": False,
        "PBS_JOB_ID": None,
        "PBS_SUBMITTED": False,
        "GPU_USED": False,
        "MODEL_CONSTRUCTED": False,
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "BACKWARD": False,
        "OPTIMIZER_STEP": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "HARDWARE": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(output_root / "stage35_artifact_seal.json"),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "SEALED_FILES": seal_payload["sealed_files"],
        "DETAIL_OUTPUTS": {
            "preflight": str(output_root / "stage35_preflight.json"),
            "a0": str(output_root / "stage35_a0_stage16_authority_audit.json"),
            "semantic_closure": str(output_root / "stage35_b_motion_intent_semantic_closure.json"),
            "disentanglement": str(output_root / "stage35_b_geometry_timing_disentanglement.json"),
            "d1_geometry": str(output_root / "stage35_d1_geometry_qualification.json"),
            "d2_timing": str(output_root / "stage35_d2_timing_qualification.json"),
            "e0": str(output_root / "stage35_e0_continuous_base_audit.json"),
            "corpus_protocol": str(output_root / "stage35_oracle_corpus_protocol.json"),
            "corpus_manifest": str(output_root / "stage35_oracle_corpus_manifest.json"),
            "cpu_log": str(log_path),
        },
        "RESULT": overall_result,
        "NEXT_GATE": NEXT_GATE,
        "STOP_REASON": (
            "Stage16 native reference corpus and exact immutable WBC/reach-table inputs are not proven; "
            "generated corpus remains a deterministic candidate only."
        ),
    }
    write_json(output_root / "stage35_return.json", _jsonable(return_payload))
    return return_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, default=REPO)
    parser.add_argument("--stage16-root", type=Path, default=EXPECTED_STAGE16_ROOT)
    parser.add_argument(
        "--stage34-seal",
        type=Path,
        default=Path(
            "/scratch/e1192548/leon-storage/runs/stage34_wbc_aligned_motion_api_20260905/"
            "stage34_cpu_20260905_v2/stage34_artifact_seal.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
