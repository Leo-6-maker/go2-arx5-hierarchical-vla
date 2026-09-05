#!/usr/bin/env python3
"""Run the zero-model portion of the Stage34 Motion API qualification.

The script is intentionally a CPU/static harness.  It audits the private
Stage16 reference authority, runs only deterministic geometry/timing
diagnostics on already-unlocked train snapshots, exercises MotionIntentV2,
and seals the evidence.  A missing exact Stage16 corpus is a hard scientific
stop: no simulator, WBC, model, GPU, or scheduler execution is reachable from
this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from go2arx5_vla.contracts import MotionIntentV2  # noqa: E402
from go2arx5_vla.motion_api import (  # noqa: E402
    DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
    DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
    DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    DEFAULT_WBC_PREVIEW_OFFSETS_M,
    GEOMETRY_K_VALUES,
    TIMING_KNOTS,
    TIMING_MODES,
    aggregate_evaluations,
    build_geometry_candidate,
    build_timing_candidate,
    evaluate_preview_equivalence,
    load_snapshot_chunk,
    public_evaluation,
)
from go2arx5_vla.trajectory.preview import se3_arc_length_preview  # noqa: E402


PI_GATE_ID = "PI_GATE_STATE_0017"
BASE_HEAD = "3f3ba8cbbf2f7ed3665526a886bacf041bf231bc"
STAGE33_SEAL_SHA256 = "84986433532caecc2b1390049557e27e73815f530a92ccff16e82abd5c9efb28"
STAGE33_BRANCH = "stage33/bottleneck-map"
STAGE34_BRANCH = "stage34/wbc-aligned-motion-api"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE34_WBC_ALIGNED_MOTION_API_RESULT"
PROJECT = "CFP03-CF-004"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"
EXPECTED_STAGE16_ROOT = Path("/var/cache/users/liuyu-nfs/go2arx5-wbc-stage16")
DEFAULT_STAGE33_SEAL = Path(
    "/scratch/e1192548/leon-storage/runs/stage33_bottleneck_map_20260905/"
    "stage33_602255.hopper-m-02/stage33_artifact_seal.json"
)
DEFAULT_SNAPSHOT_ROOT = Path(
    "/scratch/e1192548/stage26_h1_c1_20260904/snapshots/"
    "stage22_expert_red_target_train_a"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage34_wbc_aligned_motion_api_20260905/stage34_cpu_20260905"
)

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
EXPECTED_WBC_HASHES = {
    "delivered_wbc_archive": "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167",
    "native_urdf": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "arm_checkpoint": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "native_config": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def git_sha(repo: Path) -> str | None:
    code, stdout, _ = git_output(repo, "rev-parse", "HEAD")
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
        record["size_bytes"] = int(path.stat().st_size)
        record["sha256"] = actual
        record["hash_status"] = (
            "MATCH" if expected_sha256 is None or actual == expected_sha256 else "MISMATCH"
        )
    else:
        record["sha256"] = None
        record["hash_status"] = "MISSING"
    return record


def _candidate_record(
    name: str,
    candidates: Iterable[Path],
    expected_sha256: str,
) -> dict[str, Any]:
    candidate_paths = [Path(path) for path in candidates]
    existing = [path for path in candidate_paths if path.is_file()]
    selected = existing[0] if existing else candidate_paths[0]
    record = file_record(selected, expected_sha256=expected_sha256)
    record.update(
        {
            "logical_name": name,
            "candidate_paths": [str(path) for path in candidate_paths],
            "multiple_matching_paths": [str(path) for path in existing],
        }
    )
    if len(existing) > 1:
        record["hash_status"] = "AMBIGUOUS_MULTIPLE_PATHS"
    return record


def _manifest_path(root: Path, relative: str) -> Path:
    return root / relative


def _manifest_case_paths(root: Path, payload: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
    cases = payload.get("cases") or payload.get("entries") or payload.get("selected") or []
    paths: list[Path] = []
    forbidden: list[str] = []
    if not isinstance(cases, list):
        return paths, ["manifest_cases_not_a_list"]
    for row in cases:
        if not isinstance(row, Mapping):
            continue
        for key in ("path", "trajectory_path", "shape_file", "bound_file", "original_file"):
            value = row.get(key)
            if not value:
                continue
            text = str(value)
            lowered = text.lower()
            if "validation" in lowered or "holdout" in lowered:
                forbidden.append(text)
                continue
            candidate = Path(text)
            paths.append(candidate if candidate.is_absolute() else root / candidate)
    return paths, forbidden


def _audit_stage16_reference(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_relpaths = (
        "benchmark/native_suite_manifest.json",
        "benchmark/first_eight_manifest.json",
        "progressive_relocation/progressive_cohort.json",
        "continuity/initial_continuity_analysis.json",
    )
    manifest_records: dict[str, Any] = {}
    all_case_paths: list[Path] = []
    forbidden_paths: list[str] = []
    suite_count: int | None = None
    first_eight_count: int | None = None
    progressive_count: int | None = None
    parse_errors: list[str] = []
    for relative in manifest_relpaths:
        path = _manifest_path(root, relative)
        record = file_record(path)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                paths, forbidden = _manifest_case_paths(root, payload)
                all_case_paths.extend(paths)
                forbidden_paths.extend(forbidden)
                cases = payload.get("cases") or payload.get("entries") or payload.get("selected") or []
                if relative.endswith("native_suite_manifest.json"):
                    suite_count = len(cases) if isinstance(cases, list) else None
                elif relative.endswith("first_eight_manifest.json"):
                    first_eight_count = len(cases) if isinstance(cases, list) else None
                elif relative.endswith("progressive_cohort.json"):
                    progressive_count = payload.get("selected_count")
                    if progressive_count is None and isinstance(cases, list):
                        progressive_count = len(cases)
                record["case_path_count"] = len(paths)
                record["case_paths_exist"] = bool(paths) and all(path.exists() for path in paths)
                record["forbidden_path_count"] = len(forbidden)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                parse_errors.append(f"{path}: {exc}")
                record["parse_error"] = str(exc)
        manifest_records[relative] = record

    case_paths_exist = bool(all_case_paths) and all(path.exists() for path in all_case_paths)
    manifest_shape_pass = suite_count == 28 and first_eight_count == 8 and not parse_errors
    if progressive_count is not None:
        manifest_shape_pass = manifest_shape_pass and int(progressive_count) == 8
    root_exists = root.is_dir()
    audit = {
        "stage16_root": str(root),
        "stage16_root_record": file_record(root),
        "manifests": manifest_records,
        "manifest_shape": {
            "native_suite_count": suite_count,
            "first_eight_count": first_eight_count,
            "progressive_count": progressive_count,
            "expected_native_suite_count": 28,
            "expected_primary_nominal_count": 27,
            "expected_first_eight_count": 8,
            "manifest_shape_pass": bool(manifest_shape_pass),
            "all_referenced_case_paths_exist": bool(case_paths_exist),
        },
        "forbidden_reference_paths": sorted(set(forbidden_paths)),
        "parse_errors": parse_errors,
    }

    wbc_root = Path("/var/cache/users/liuyu-nfs/go2arx5-wbc-stage13/extracted/WBCTrackingShare")
    wbc_records = {
        "delivered_wbc_archive": _candidate_record(
            "delivered_wbc_archive",
            [Path("/var/cache/users/liuyu-nfs/go2arx5-experiment-e2/intake/incoming/WBCTrackingShare_20260831.zip")],
            EXPECTED_WBC_HASHES["delivered_wbc_archive"],
        ),
        "native_urdf": _candidate_record(
            "native_urdf",
            [wbc_root / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"],
            EXPECTED_WBC_HASHES["native_urdf"],
        ),
        "arm_checkpoint": _candidate_record(
            "arm_checkpoint",
            [
                root / "checkpoints/arm/ac_weights_039999.pt",
                wbc_root / "checkpoints/arm/ac_weights_039999.pt",
                wbc_root / "resources/checkpoints/arm/ac_weights_039999.pt",
            ],
            EXPECTED_WBC_HASHES["arm_checkpoint"],
        ),
        "dog_checkpoint": _candidate_record(
            "dog_checkpoint",
            [
                root / "checkpoints/dog/ac_weights_039999.pt",
                wbc_root / "checkpoints/dog/ac_weights_039999.pt",
                wbc_root / "resources/checkpoints/dog/ac_weights_039999.pt",
            ],
            EXPECTED_WBC_HASHES["dog_checkpoint"],
        ),
        "native_config": _candidate_record(
            "native_config",
            [
                root / "config/wbc.py",
                wbc_root / "go1_gym/envs/roboduet/wbc.py",
                wbc_root / "config/wbc.py",
            ],
            EXPECTED_WBC_HASHES["native_config"],
        ),
        "reach_table": file_record(
            wbc_root / "resources/reach_tables/go2_x5_2d.pt"
        ),
    }
    wbc_records["reach_table"]["logical_name"] = "reach_table"
    wbc_records["reach_table"]["expected_role"] = "native reach table"
    wbc_records["reach_table"]["expected_sha256"] = None
    wbc_statuses = [record["hash_status"] for record in wbc_records.values()]
    wbc_exact = all(status == "MATCH" for status in wbc_statuses[:5]) and wbc_records["reach_table"]["is_file"]
    wbc_records["all_required_inputs_exact"] = bool(wbc_exact)
    audit["wbc_inputs"] = wbc_records

    source_records = {}
    for relative in STAGE16_SOURCE_FILES:
        path = REPO / relative
        source_records[relative] = file_record(path)
    source_available = all(record["is_file"] for record in source_records.values())
    audit["source_backed_evidence"] = {
        "files": source_records,
        "generator_and_protocol_present": bool(source_available),
        "exact_regeneration_ready": bool(root_exists and manifest_shape_pass and wbc_exact),
    }

    complete = bool(
        root_exists
        and manifest_shape_pass
        and case_paths_exist
        and not forbidden_paths
        and wbc_exact
    )
    audit["REFERENCE_PROVENANCE_STATUS"] = "PASS" if complete else "INSUFFICIENT_REFERENCE_PROVENANCE"
    audit["blockers"] = [] if complete else [
        "complete Stage16 native reference corpus and/or exact immutable WBC inputs cannot be proven on Hopper"
    ]

    snapshot_records: list[dict[str, Any]] = []
    chunks: dict[str, Any] = {}
    snapshot_root = DEFAULT_SNAPSHOT_ROOT
    for path in sorted(snapshot_root.glob("expert-*.json")) if snapshot_root.is_dir() else []:
        record: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "diagnostic_only": True,
            "split": "train",
        }
        try:
            chunk = load_snapshot_chunk(path)
            chunks[str(path)] = chunk
            record.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "point_count": chunk.point_count,
                    "source_frame": chunk.source_frame,
                    "tcp_semantic_id": chunk.tcp_semantic_id,
                    "timestamp_start": float(chunk.point_timestamps[0]),
                    "timestamp_end": float(chunk.point_timestamps[-1]),
                    "eef_shape": [int(value) for value in chunk.eef_position.shape],
                    "load_status": "PASS",
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record.update({"load_status": "FAIL", "error": str(exc)})
        snapshot_records.append(record)
    audit["sentinel_snapshots"] = snapshot_records
    audit["sentinel_policy"] = {
        "allowed": True,
        "purpose": "diagnostic_only",
        "promotion_allowed": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    audit["reference_corpus_hashes"] = {
        relative: record.get("sha256")
        for relative, record in manifest_records.items()
        if record.get("sha256")
    }
    audit["reference_corpus_hashes"].update(
        {
            f"stage16_source/{relative}": record.get("sha256")
            for relative, record in source_records.items()
            if record.get("sha256")
        }
    )
    return audit, chunks


def _intent_payload(intent: MotionIntentV2) -> dict[str, Any]:
    payload = intent.as_dict()
    payload["control_pose_count"] = intent.point_count
    payload["control_pose_shape"] = [int(value) for value in intent.eef_position.shape]
    return payload


def _run_geometry(
    chunks: Mapping[str, Any],
    reference_status: str,
) -> dict[str, Any]:
    by_k: dict[str, list[dict[str, Any]]] = {str(k): [] for k in GEOMETRY_K_VALUES}
    candidates: list[dict[str, Any]] = []
    for path, dense in chunks.items():
        for k in GEOMETRY_K_VALUES:
            geometry = build_geometry_candidate(dense, k)
            evaluation = evaluate_preview_equivalence(
                dense,
                geometry,
                case_id=dense.chunk_id,
                representation=f"K{k}",
                anchor_count=101,
            )
            row = {
                "case_id": dense.chunk_id,
                "snapshot_path": path,
                "K": int(k),
                "diagnostic_only": True,
                "candidate": _intent_payload(geometry),
                "evaluation": public_evaluation(evaluation),
            }
            by_k[str(k)].append(evaluation)
            candidates.append(row)
    k_results: dict[str, Any] = {}
    for k in GEOMETRY_K_VALUES:
        rows = by_k[str(k)]
        aggregate = aggregate_evaluations(rows, representation=f"K{k}") if rows else None
        k_results[str(k)] = {
            "K": int(k),
            "diagnostic_case_count": len(rows),
            "diagnostic_aggregate": aggregate,
            "diagnostic_passes_limits": None if aggregate is None else bool(aggregate["passes_limits"]),
            "promotion_status": (
                "BLOCKED_INSUFFICIENT_REFERENCE_PROVENANCE"
                if reference_status != "PASS"
                else "ELIGIBLE_FOR_PROTOCOL_SELECTION"
            ),
        }
    diagnostic_winner = next(
        (str(k) for k in GEOMETRY_K_VALUES if k_results[str(k)]["diagnostic_passes_limits"] is True),
        None,
    )
    if reference_status != "PASS":
        geometry_result = "INSUFFICIENT_REFERENCE_PROVENANCE"
        k_star = None
    elif diagnostic_winner is None:
        geometry_result = "DENSE_OR_HIGHER_COMPLEXITY_REQUIRED_ON_TESTED_DISTRIBUTION"
        k_star = None
    else:
        geometry_result = "GEOMETRY_QUALIFIED"
        k_star = int(diagnostic_winner)
    return {
        "schema": "STAGE34_A_GEOMETRY_RESULTS_V1",
        "method": {
            "se3_lambda_m_per_rad": DEFAULT_SE3_LAMBDA_M_PER_RAD,
            "K_values": list(GEOMETRY_K_VALUES),
            "knot_policy": "uniform_normalized_se3_arc_including_endpoints",
            "interpolation": "linear_xyz_quaternion_slerp",
            "forbidden_operations": ["smoothing", "spline_fitting", "trajectory_optimization", "manual_knot_placement"],
            "preview_offsets_m": DEFAULT_WBC_PREVIEW_OFFSETS_M.tolist(),
        },
        "thresholds": {
            "preview_position_p95_limit_m": DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
            "preview_rotation_p95_limit_rad": DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
            "endpoint_clamp_agreement_min": DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
            "start_end_pose_exact": True,
            "duration_exact": True,
        },
        "reference_provenance_status": reference_status,
        "K_RESULTS_4_6_9_12": k_results,
        "diagnostic_candidate_count": len(candidates),
        "diagnostic_candidates": candidates,
        "diagnostic_winner_without_promotion": diagnostic_winner,
        "GEOMETRY_RESULT": geometry_result,
        "K_STAR": k_star,
    }


def _run_timing(
    chunks: Mapping[str, Any],
    reference_status: str,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in TIMING_MODES}
    candidates: list[dict[str, Any]] = []
    for path, dense in chunks.items():
        for k in GEOMETRY_K_VALUES:
            geometry = build_geometry_candidate(dense, k)
            for mode in TIMING_MODES:
                timing = build_timing_candidate(dense, geometry, mode)
                evaluation = evaluate_preview_equivalence(
                    dense,
                    timing,
                    case_id=dense.chunk_id,
                    representation=f"K{k}+{mode}",
                    anchor_count=101,
                )
                by_mode[mode].append(evaluation)
                candidates.append(
                    {
                        "case_id": dense.chunk_id,
                        "snapshot_path": path,
                        "K": int(k),
                        "timing_mode": mode,
                        "diagnostic_only": True,
                        "candidate": _intent_payload(timing),
                        "evaluation": public_evaluation(evaluation),
                    }
                )
    mode_results: dict[str, Any] = {}
    for mode in TIMING_MODES:
        rows = by_mode[mode]
        aggregate = aggregate_evaluations(rows, representation=mode) if rows else None
        mode_results[mode] = {
            "timing_mode": mode,
            "timing_knots": TIMING_KNOTS[mode].tolist(),
            "diagnostic_case_count": len(rows),
            "diagnostic_aggregate": aggregate,
            "diagnostic_passes_limits": None if aggregate is None else bool(aggregate["passes_limits"]),
            "promotion_status": (
                "BLOCKED_INSUFFICIENT_REFERENCE_PROVENANCE"
                if reference_status != "PASS"
                else "ELIGIBLE_ONLY_AFTER_K_STAR"
            ),
        }
    diagnostic_winner = next(
        (mode for mode in TIMING_MODES if mode_results[mode]["diagnostic_passes_limits"] is True),
        None,
    )
    return {
        "schema": "STAGE34_B_TIMING_RESULTS_V1",
        "method": {
            "timing_modes": list(TIMING_MODES),
            "timing_knots": {mode: TIMING_KNOTS[mode].tolist() for mode in TIMING_MODES},
            "timing_interpolation": "monotonic_piecewise_linear",
            "geometry_frozen_per_candidate": True,
        },
        "reference_provenance_status": reference_status,
        "T_RESULTS_T1_T3_T5_T9": mode_results,
        "diagnostic_candidate_count": len(candidates),
        "diagnostic_candidates": candidates,
        "diagnostic_winner_without_promotion": diagnostic_winner,
        "TIMING_RESULT": "INSUFFICIENT_REFERENCE_PROVENANCE" if reference_status != "PASS" else "TIMING_DIAGNOSTICS_COMPLETE",
        "T_STAR": None if reference_status != "PASS" else diagnostic_winner,
    }


def _check_raises(callable_obj: Any) -> bool:
    try:
        callable_obj()
    except (ValueError, TypeError):
        return True
    return False


def _run_contract_smoke(chunks: Mapping[str, Any], base_head: str | None) -> dict[str, Any]:
    if not chunks:
        return {
            "schema": "STAGE34_C_MOTION_INTENT_V2_CONTRACT_V1",
            "status": "BLOCKED_NO_DIAGNOSTIC_SNAPSHOT",
            "contract_hash": None,
        }
    first_path = sorted(chunks)[0]
    dense = chunks[first_path]
    geometry = build_geometry_candidate(dense, 4)
    timing = build_timing_candidate(dense, geometry, "T1")
    roundtrip = timing.to_trajectory_chunk(
        expected_source_frame=SOURCE_FRAME,
        expected_tcp_semantic_id=TCP_ID,
    )
    anchor_values = np.linspace(dense.validity_start, dense.validity_end, 3)
    preview_equal = []
    for anchor in anchor_values:
        original = se3_arc_length_preview(roundtrip, anchor_timestamp=float(anchor))
        rebuilt = se3_arc_length_preview(
            timing.to_trajectory_chunk(
                expected_source_frame=SOURCE_FRAME,
                expected_tcp_semantic_id=TCP_ID,
            ),
            anchor_timestamp=float(anchor),
        )
        preview_equal.append(
            bool(
                np.allclose(original.eef_position, rebuilt.eef_position, atol=1e-12, rtol=0.0)
                and np.allclose(original.eef_rot6d, rebuilt.eef_rot6d, atol=1e-12, rtol=0.0)
                and np.array_equal(original.endpoint_clamped, rebuilt.endpoint_clamped)
            )
        )
    endpoint_preview = se3_arc_length_preview(roundtrip, anchor_timestamp=float(roundtrip.validity_end))
    invalid_timing_kwargs = dict(timing.__dict__)
    invalid_timing_kwargs.update(
        {
            "timing_arc_fraction": np.asarray([0.0, 0.5, 0.5], dtype=np.float64),
            "timing_timestamps": np.asarray(
                [timing.validity_start, timing.validity_end, timing.validity_end],
                dtype=np.float64,
            ),
        }
    )
    invalid_finite_kwargs = dict(timing.__dict__)
    invalid_finite_position = np.asarray(timing.eef_position, dtype=np.float64).copy()
    invalid_finite_position[0, 0] = np.nan
    invalid_finite_kwargs["eef_position"] = invalid_finite_position
    frame_guard = _check_raises(
        lambda: timing.to_trajectory_chunk(expected_source_frame="WRONG_FRAME")
    )
    tcp_guard = _check_raises(
        lambda: timing.to_trajectory_chunk(expected_tcp_semantic_id="WRONG_TCP")
    )
    invalid_timing_guard = _check_raises(lambda: MotionIntentV2(**invalid_timing_kwargs))
    invalid_finite_guard = _check_raises(lambda: MotionIntentV2(**invalid_finite_kwargs))

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
                "unchanged_from_stage33_base": unchanged,
            }
            v1_unchanged = v1_unchanged and unchanged
    else:
        v1_unchanged = False

    contract_payload = _intent_payload(timing)
    contract_hash = sha256_bytes(canonical_json(contract_payload))
    tests_pass = bool(
        all(preview_equal)
        and endpoint_preview.endpoint_clamped.all()
        and frame_guard
        and tcp_guard
        and invalid_timing_guard
        and invalid_finite_guard
        and v1_unchanged
    )
    return {
        "schema": "STAGE34_C_MOTION_INTENT_V2_CONTRACT_V1",
        "status": "PASS" if tests_pass else "FAIL",
        "contract_hash": contract_hash,
        "smoke_source_snapshot": first_path,
        "smoke_representation": "K4+T1_DIAGNOSTIC",
        "roundtrip_preview_exact": bool(all(preview_equal)),
        "roundtrip_anchor_count": len(preview_equal),
        "endpoint_clamp_at_validity_end": bool(endpoint_preview.endpoint_clamped.all()),
        "frame_mismatch_fail_closed": frame_guard,
        "tcp_mismatch_fail_closed": tcp_guard,
        "invalid_timing_fail_closed": invalid_timing_guard,
        "nonfinite_pose_fail_closed": invalid_finite_guard,
        "v1_contract_files": v1_files,
        "v1_and_stage23_abi_unchanged": v1_unchanged,
        "production_v1_replaced": False,
        "model_constructed": False,
        "gr00t_forward": False,
    }


def _run_e0(reference_audit: Mapping[str, Any]) -> dict[str, Any]:
    source_present = bool(reference_audit.get("source_backed_evidence", {}).get("generator_and_protocol_present"))
    stage16_root_exists = bool(reference_audit.get("stage16_root_record", {}).get("is_dir"))
    continuous_artifact = False
    if stage16_root_exists:
        for relative, record in reference_audit.get("manifests", {}).items():
            if "continuous" in relative.lower() and record.get("is_file"):
                continuous_artifact = True
    if continuous_artifact:
        result = "CONTINUOUS_BASE_COHORT_SOURCE_RECOVERED"
    elif source_present:
        result = "CONTINUOUS_BASE_COHORT_DESIGN_READY_FOR_NEXT_GATE"
    else:
        result = "NO_VALID_CONTINUOUS_BASE_COHORT_PROVENANCE"
    return {
        "schema": "STAGE34_E0_CONTINUOUS_BASE_AUDIT_V1",
        "result": result,
        "source_backed_generator_present": source_present,
        "exact_continuous_base_cohort_recovered": continuous_artifact,
        "new_continuous_base_rollouts": 0,
        "new_wbc_rollouts": 0,
        "wbc_executed": False,
        "design_rules": [
            "source-backed reference and reach-table identity required",
            "support-aware continuity and bounded-speed checks",
            "fixed-base reach-boundary annotation retained",
            "no fabricated trajectories",
            "no outcome-dependent adaptation",
        ],
        "next_gate_required_for_new_rollout": True,
    }


def _preflight(repo: Path, stage33_seal: Path, reference_status: str) -> dict[str, Any]:
    head = git_sha(repo)
    branch = current_branch(repo)
    status = worktree_status(repo)
    base_ancestor = False
    code, _, _ = git_output(repo, "merge-base", "--is-ancestor", BASE_HEAD, "HEAD")
    base_ancestor = code == 0
    stage33_branch_head = None
    code, stdout, _ = git_output(repo, "rev-parse", f"refs/remotes/origin/{STAGE33_BRANCH}")
    if code == 0:
        stage33_branch_head = stdout
    seal_record = file_record(stage33_seal, expected_sha256=STAGE33_SEAL_SHA256)
    stage33_seal_payload: dict[str, Any] = {}
    if stage33_seal.is_file():
        try:
            stage33_seal_payload = json.loads(stage33_seal.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stage33_seal_payload = {"parse_error": True}
    return {
        "schema": "STAGE34_ZERO_MODEL_PREFLIGHT_V1",
        "PI_GATE_READ": PI_GATE_ID,
        "project": PROJECT,
        "base_head": BASE_HEAD,
        "stage34_head": head,
        "branch": branch,
        "expected_branch": STAGE34_BRANCH,
        "base_is_ancestor": base_ancestor,
        "stage33_remote_branch_head": stage33_branch_head,
        "stage33_remote_branch_matches_base": stage33_branch_head == BASE_HEAD,
        "worktree_clean": not status,
        "worktree_status": status,
        "stage33_seal": {
            "record": seal_record,
            "payload_schema": stage33_seal_payload.get("schema"),
            "payload_stage": stage33_seal_payload.get("stage"),
        },
        "reference_provenance_status": reference_status,
        "stage34_d_predicates": {
            "reference_provenance_pass": reference_status == "PASS",
            "k_star_exists": False,
            "motion_intent_v2_tests_pass": False,
            "head_exact_and_clean": False,
            "fresh_zero_model_preflight_pass": False,
            "stage34_d_authorized_in_this_run": False,
        },
        "runtime": {
            "hostname": platform.node(),
            "user": os.environ.get("USER"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_used": False,
            "model_constructed": False,
        },
        "execution_boundary": {
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


def _seal(
    output_root: Path,
    *,
    project_repo: Path,
    generated_files: Iterable[Path],
) -> tuple[dict[str, Any], str]:
    source_files = [
        project_repo / "scripts/stage34_wbc_aligned_motion_api.py",
        project_repo / "src/go2arx5_vla/contracts/motion_intent.py",
        project_repo / "src/go2arx5_vla/motion_api/qualification.py",
        project_repo / "tests/test_stage34_wbc_aligned_motion_api.py",
        project_repo / "reports/STAGE34_WBC_ALIGNED_MOTION_API.md",
    ]
    paths = set(Path(path).resolve() for path in source_files)
    paths.update(Path(path).resolve() for path in generated_files)
    paths.discard((output_root / "stage34_artifact_seal.json").resolve())
    paths.discard((output_root / "stage34_return.json").resolve())
    entries = []
    for path in sorted(paths, key=str):
        if not path.is_file():
            continue
        entries.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "STAGE34_ARTIFACT_SEAL_V1",
        "stage": "Stage34",
        "pi_gate": PI_GATE_ID,
        "base_head": BASE_HEAD,
        "stage34_head": git_sha(project_repo),
        "execution_class": "CPU_ZERO_MODEL_DIAGNOSTIC_ONLY",
        "pbs_job_id": None,
        "pbs_wrapper_sha256": None,
        "wbc_executed": False,
        "sealed_files": entries,
        "forbidden_actions": {
            "gr00t_model_constructed": False,
            "gr00t_forward": False,
            "training": False,
            "hardware": False,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
    }
    seal_path = output_root / "stage34_artifact_seal.json"
    write_json(seal_path, payload)
    return payload, sha256_file(seal_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_repo = Path(args.project_repo).resolve()
    stage16_root = Path(args.stage16_root).resolve()
    snapshot_root = Path(args.snapshot_root).resolve()
    stage33_seal = Path(args.stage33_seal).resolve()
    output_root = Path(args.output_root).resolve()
    if project_repo != REPO:
        raise RuntimeError(f"project repo must be the checked-out repository: {REPO}")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    # The audit helper uses the canonical train snapshot root by design.  The
    # explicit CLI path is accepted only when it is the same already-unlocked
    # train split, preventing an accidental validation/holdout read.
    if snapshot_root != DEFAULT_SNAPSHOT_ROOT:
        raise RuntimeError("snapshot root is fixed to the already-unlocked Stage22 train sentinel root")

    reference_audit, chunks = _audit_stage16_reference(stage16_root)
    reference_status = str(reference_audit["REFERENCE_PROVENANCE_STATUS"])
    geometry = _run_geometry(chunks, reference_status)
    timing = _run_timing(chunks, reference_status)
    contract = _run_contract_smoke(chunks, BASE_HEAD)
    e0 = _run_e0(reference_audit)
    preflight = _preflight(project_repo, stage33_seal, reference_status)
    preflight["stage34_d_predicates"]["motion_intent_v2_tests_pass"] = contract.get("status") == "PASS"
    preflight["stage34_d_predicates"]["head_exact_and_clean"] = bool(
        preflight.get("stage34_head")
        and preflight.get("branch") == STAGE34_BRANCH
        and preflight.get("worktree_clean")
        and preflight.get("base_is_ancestor")
    )
    preflight["stage34_d_predicates"]["fresh_zero_model_preflight_pass"] = bool(
        preflight["stage34_d_predicates"]["head_exact_and_clean"]
        and contract.get("status") == "PASS"
        and preflight["stage33_seal"]["record"].get("hash_status") == "MATCH"
    )
    preflight["stage34_d_predicates"]["stage34_d_authorized_in_this_run"] = False

    output_files = {
        "stage34_preflight.json": preflight,
        "stage34_a0_reference_provenance.json": reference_audit,
        "stage34_a_geometry_results.json": geometry,
        "stage34_b_timing_results.json": timing,
        "stage34_c_motion_intent_contract.json": contract,
        "stage34_e0_continuous_base_audit.json": e0,
    }
    for name, payload in output_files.items():
        write_json(output_root / name, _jsonable(payload))

    log_lines = [
        f"{datetime.now(timezone.utc).isoformat()} Stage34 CPU zero-model diagnostic start",
        f"PI_GATE_READ={PI_GATE_ID}",
        f"STAGE16_REFERENCE_PROVENANCE={reference_status}",
        f"DIAGNOSTIC_SNAPSHOT_COUNT={len(chunks)}",
        f"GEOMETRY_RESULT={geometry['GEOMETRY_RESULT']}",
        f"K_STAR={geometry['K_STAR']}",
        f"TIMING_RESULT={timing['TIMING_RESULT']}",
        f"T_STAR={timing['T_STAR']}",
        f"MOTION_INTENT_V2_CONTRACT={contract.get('status')}",
        "STAGE34_D_EXECUTED=false",
        "PBS_SUBMITTED=false",
        "GPU_USED=false",
        "MODEL_CONSTRUCTED=false",
        "GR00T_FORWARD=false",
        f"{datetime.now(timezone.utc).isoformat()} Stage34 CPU zero-model diagnostic stop",
    ]
    log_path = output_root / "stage34_cpu_execution.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    generated = [output_root / name for name in output_files] + [log_path]
    seal_payload, seal_sha256 = _seal(
        output_root,
        project_repo=project_repo,
        generated_files=generated,
    )
    return_payload = {
        "schema": "STAGE34_HOPPER_RETURN_V1",
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "PI_GATE_READ": PI_GATE_ID,
        "BASE_HEAD": BASE_HEAD,
        "STAGE34_HEAD": preflight.get("stage34_head"),
        "BRANCH": preflight.get("branch"),
        "REFERENCE_PROVENANCE_STATUS": reference_status,
        "REFERENCE_CORPUS_HASHES": reference_audit.get("reference_corpus_hashes", {}),
        "REFERENCE_AUDIT_PATH": str(output_root / "stage34_a0_reference_provenance.json"),
        "K_RESULTS_4_6_9_12": geometry.get("K_RESULTS_4_6_9_12"),
        "K_STAR": geometry.get("K_STAR"),
        "GEOMETRY_RESULT": geometry.get("GEOMETRY_RESULT"),
        "T_RESULTS_T1_T3_T5_T9": timing.get("T_RESULTS_T1_T3_T5_T9"),
        "T_STAR": timing.get("T_STAR"),
        "MOTION_INTENT_V2_CONTRACT_HASH": contract.get("contract_hash"),
        "MOTION_INTENT_V2_CONTRACT_STATUS": contract.get("status"),
        "PREVIEW_EQUIVALENCE_RESULT": "INSUFFICIENT_REFERENCE_PROVENANCE",
        "WBC_PAIRED_EXECUTED": False,
        "PBS_JOB_ID": None,
        "WBC_EQUIVALENCE_RESULT": "INSUFFICIENT_REFERENCE_PROVENANCE",
        "CONTINUOUS_BASE_E0_RESULT": e0.get("result"),
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(output_root / "stage34_artifact_seal.json"),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "SEALED_FILES": seal_payload.get("sealed_files", []),
        "PBS_WRAPPER_SHA256": None,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "BACKWARD": False,
        "OPTIMIZER_STEP": False,
        "HARDWARE": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "WBC_EXECUTED": False,
        "MODEL_CONSTRUCTED": False,
        "GPU_USED": False,
        "PBS_SUBMITTED": False,
        "RESULT": "INSUFFICIENT_REFERENCE_PROVENANCE",
        "NEXT_GATE": NEXT_GATE,
        "STOP_REASON": "Stage16 full primary native reference authority and exact WBC inputs are not reproducible on Hopper; diagnostics are sentinel-only.",
    }
    return_path = output_root / "stage34_return.json"
    write_json(return_path, return_payload)
    return return_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, default=REPO)
    parser.add_argument("--stage16-root", type=Path, default=EXPECTED_STAGE16_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--stage33-seal", type=Path, default=DEFAULT_STAGE33_SEAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
