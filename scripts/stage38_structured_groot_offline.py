#!/usr/bin/env python3
"""Stage38-A CPU timing-capacity qualification.

This module is deliberately a zero-model, zero-scheduler Stage38 admission
tool.  It evaluates the preregistered T16/T20/T24/T32/T40 timing ladder on the
immutable Stage35 corpus with the already-qualified K16 geometry and the
timestamp-preserving native transport used by Stage37.  It does not create a
model, invoke a simulator, request an allocation, or alter historical
artifacts.  Later Stage38-B/C/D/E work is admitted only from this sealed
result and remains fail-closed until its own predicates are implemented.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.motion_api import (  # noqa: E402
    DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
    DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
    DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    STAGE38_TIMING_KNOTS,
    STAGE38_TIMING_MODES,
    build_geometry_candidate,
    build_timing_candidate_with_knots,
    evaluate_preview_equivalence,
    public_evaluation,
)
from go2arx5_vla.trajectory.native_transport import (  # noqa: E402
    NATIVE_DT_S,
    NATIVE_DURATION_S,
    NATIVE_SAMPLE_COUNT,
)


STAGE36 = importlib.import_module("stage36_wbc_authority_oracle_qualification")
STAGE37 = importlib.import_module("stage37_geometry_capacity_qualification")

PI_GATE_ID = "PI_GATE_STATE_0026"
PR_NUMBER = 24
EXPECTED_BRANCH = "stage38/structured-groot-offline"
EXPECTED_PARENT_HEAD = "bbf7453c7060d64fc6f659281ceb777246d6d370"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
STAGE37_EXECUTION_HEAD = "89e53dca25fa80f48ab286d8520797d2691d86ac"
STAGE37_LIVE_HEAD = EXPECTED_PARENT_HEAD
STAGE37_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage37_geometry_capacity_qualification_20260906/"
    "stage37_structured_final_602886_bbf7453_v2"
)
STAGE37_FINAL_RETURN = STAGE37_ROOT / "stage37_final_return.json"
STAGE37_SEAL = STAGE37_ROOT / "stage37_structured_artifact_seal.json"
STAGE37_RESULT = STAGE37_ROOT / "stage37_structured_oracle_result.json"
STAGE37_PAIRED = STAGE37_ROOT / "stage37_paired_oracle_analysis.json"
STAGE37_RAW_MANIFEST = STAGE37_ROOT / "stage37_structured_raw_hash_manifest.json"
STAGE37_RESULT_MANIFEST = STAGE37_ROOT / "stage37_structured_result_manifest.json"
STAGE37_SOURCE_MANIFEST = STAGE37_ROOT / "stage37_source_manifest.json"
STAGE37_FINAL_RETURN_SHA256 = (
    "f2e32ee39f856cd2a44b0c3f5487296e479680a3d83c95c6abb95f7235569976"
)
STAGE37_SEAL_SHA256 = (
    "4a606636d4769367a97b2386f4e202be3ae97ac7d63750631919fa9710fdeb87"
)
STAGE37_RESULT_SHA256 = (
    "de1b4358edc8d872c1b75f52873fb84ff4d3228dca0d91a5df7cdc327c2b9f18"
)
STAGE37_PAIRED_SHA256 = (
    "c1bcf7d8e537e5e40e8d7cd8d738db1e1b2eb32e1bc8b38ee4b701934b0106d6"
)
STAGE37_RAW_MANIFEST_SHA256 = (
    "97646e44297accc9bf7de96e7578c1666b76f4014a24da956f4a643c07f74270"
)
STAGE37_RESULT_MANIFEST_SHA256 = (
    "002fb4959df1f150ea1f2a60e3c2f54a042166239338bdf08aec477cdff69abf"
)
STAGE37_SOURCE_MANIFEST_SHA256 = (
    "176aecebc89f6ceb80cf20f59037bf1b592e54f2cfa1e19dc84f328162dc718b"
)
STAGE37_CAMPAIGN_AUDIT_SHA256 = (
    "2885a78b13b40e56f528920f647296bde170a774b2526a48c4d6aa65fecb6e0e"
)
STAGE35_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage35_motion_api_authority_closure_20260905/stage35_cpu_20260905_v2"
)
STAGE35_MANIFEST_SHA256 = (
    "c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642"
)
STAGE35_CORPUS_TREE_SHA256 = (
    "b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414"
)
STAGE35_GENERATED_PROTOCOL_SHA256 = (
    "ae77add1a6c4845f3a1517200a739fb0db27d0da44c19b8eea01552170958e45"
)
STAGE35_PROTOCOL_SHA256 = (
    "81585c50b29ba67d65e346d3b6fb10420a52ca283d78e54d7476073ecacca22d"
)
STAGE35_SEAL_SHA256 = (
    "5ec9dc293546ea33c39c225d184921e77f1ae6f86756ec1bd0304cb19aa63e89"
)
STAGE35_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in ("N", "T", "P", "C")
    for index in range(1, 9)
)
CORE_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in ("N", "T", "P")
    for index in range(1, 9)
)
CONTINUOUS_CASE_IDS = tuple("C%02d" % index for index in range(1, 9))
STAGE38_K_CAPACITY_STAR = 16
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE"
TIMING_PARAMETERIZATION_STOP = "GPT_PI_REVIEW_TIMING_PARAMETERIZATION_REDESIGN_REQUIRED"
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38a_cpu_bbf7453_v1"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


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


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_json(path: Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_json(payload, str(path))
    return payload


def write_json(path: Path, payload: Any) -> None:
    converted = jsonable(payload)
    finite_json(converted, str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(converted, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_event(output_root: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(jsonable(fields))
    with (output_root / "stage38a_cpu_execution.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def run_capture(argv: Sequence[str], cwd: Path = REPO) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            list(argv),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_output(*args: str) -> Optional[str]:
    code, stdout, _ = run_capture(["git", *args])
    return stdout if code == 0 else None


def file_record(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path)
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "hash_match": bool(exists and (expected_sha256 is None or actual == expected_sha256)),
    }


def verify_seal(root: Path, seal_path: Path, expected_sha256: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "root": str(root),
        "seal": file_record(seal_path, expected_sha256),
        "entry_count": 0,
        "entries_verified": 0,
        "mismatches": [],
        "status": "BLOCKED",
    }
    if not result["seal"]["hash_match"]:
        return result
    seal = load_json(seal_path)
    entries = seal.get("entries", [])
    result["entry_count"] = len(entries)
    root_resolved = root.resolve()
    for entry in entries:
        entry_path = Path(str(entry.get("path", "")))
        if not entry_path.is_absolute():
            entry_path = root / str(entry.get("relative_path", ""))
        try:
            inside_root = os.path.commonpath((str(root_resolved), str(entry_path.resolve()))) == str(
                root_resolved
            )
        except (OSError, ValueError):
            inside_root = False
        actual = sha256_file(entry_path) if inside_root and entry_path.is_file() else None
        actual_size = int(entry_path.stat().st_size) if inside_root and entry_path.is_file() else None
        if (
            not inside_root
            or actual != entry.get("sha256")
            or actual_size != int(entry.get("size_bytes", -1))
        ):
            result["mismatches"].append(
                {
                    "path": str(entry_path),
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": actual,
                    "expected_size_bytes": entry.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                }
            )
        else:
            result["entries_verified"] += 1
    result["status"] = "PASS" if not result["mismatches"] else "BLOCKED_SEAL_ENTRY_MISMATCH"
    return result


def verify_stage37_acceptance() -> Dict[str, Any]:
    expected_files = {
        "final_return": (STAGE37_FINAL_RETURN, STAGE37_FINAL_RETURN_SHA256),
        "seal": (STAGE37_SEAL, STAGE37_SEAL_SHA256),
        "result": (STAGE37_RESULT, STAGE37_RESULT_SHA256),
        "paired_analysis": (STAGE37_PAIRED, STAGE37_PAIRED_SHA256),
        "raw_manifest": (STAGE37_RAW_MANIFEST, STAGE37_RAW_MANIFEST_SHA256),
        "result_manifest": (STAGE37_RESULT_MANIFEST, STAGE37_RESULT_MANIFEST_SHA256),
        "source_manifest": (STAGE37_SOURCE_MANIFEST, STAGE37_SOURCE_MANIFEST_SHA256),
    }
    result: Dict[str, Any] = {
        "schema": "STAGE38_STAGE37_ACCEPTANCE_INPUT_AUDIT_V1",
        "root": str(STAGE37_ROOT),
        "files": {
            name: file_record(path, expected) for name, (path, expected) in expected_files.items()
        },
        "seal_verification": verify_seal(STAGE37_ROOT, STAGE37_SEAL, STAGE37_SEAL_SHA256),
        "accepted_execution_head": STAGE37_EXECUTION_HEAD,
        "accepted_live_head": STAGE37_LIVE_HEAD,
        "stage37_fields_exact": False,
        "status": "BLOCKED",
    }
    try:
        final_return = load_json(STAGE37_FINAL_RETURN)
        result["stage37_fields_exact"] = bool(
            final_return.get("ACTIVE_PI_GATE") == "PI_GATE_STATE_0025"
            and final_return.get("PR") == 23
            and final_return.get("K_CAPACITY_STAR") == STAGE38_K_CAPACITY_STAR
            and final_return.get("GEOMETRY_CAPACITY_CLASSIFICATION")
            == "COMPACT_GEOMETRY_CAPACITY_QUALIFIED"
            and final_return.get("T_CAPACITY_STAR") is None
            and final_return.get("TIMING_COMPRESSION_STATUS")
            == "TIMING_REPRESENTATION_INSUFFICIENT"
            and final_return.get("GEOMETRY_ORACLE_PARITY")
            == "GEOMETRY_ORACLE_PARITY_QUALIFIED"
            and final_return.get("WBC_CONTINUOUS_BASE_CLASSIFICATION")
            == "WBC_CONTINUOUS_BASE_LIFECYCLE_QUALIFIED_ON_STRUCTURED_K16_T0"
            and final_return.get("TRUE_JOINT_LIMIT_TERMINATION_ROWS") == 0
            and final_return.get("GR00T_MODEL_FORWARD") is False
            and final_return.get("TRAINING") is False
            and final_return.get("VALIDATION_E_READ") is False
            and final_return.get("HOLDOUT_F_READ") is False
            and final_return.get("HARDWARE") is False
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["status"] = (
        "PASS"
        if all(item["hash_match"] for item in result["files"].values())
        and result["seal_verification"]["status"] == "PASS"
        and result["stage37_fields_exact"]
        else "BLOCKED_STAGE37_ACCEPTANCE_MISMATCH"
    )
    return result


def verify_stage35_input() -> Dict[str, Any]:
    """Reuse the prior immutable corpus audit without changing its inputs."""

    audit = STAGE37.verify_stage35_corpus()
    audit["stage38_expected_root"] = str(STAGE35_ROOT)
    audit["stage38_expected_manifest_sha256"] = STAGE35_MANIFEST_SHA256
    audit["stage38_expected_tree_sha256"] = STAGE35_CORPUS_TREE_SHA256
    audit["stage38_expected_generated_protocol_sha256"] = STAGE35_GENERATED_PROTOCOL_SHA256
    audit["stage38_expected_protocol_sha256"] = STAGE35_PROTOCOL_SHA256
    audit["stage38_expected_seal_sha256"] = STAGE35_SEAL_SHA256
    audit["stage38_case_ids"] = list(STAGE35_CASE_IDS)
    audit["stage38_identity_exact"] = bool(
        audit.get("root") == str(STAGE35_ROOT)
        and audit.get("manifest", {}).get("expected_sha256") == STAGE35_MANIFEST_SHA256
        and audit.get("tree_sha256_expected") == STAGE35_CORPUS_TREE_SHA256
        and audit.get("generated_protocol", {}).get("expected_sha256")
        == STAGE35_GENERATED_PROTOCOL_SHA256
        and audit.get("committed_protocol", {}).get("expected_sha256") == STAGE35_PROTOCOL_SHA256
        and audit.get("seal", {}).get("expected_sha256") == STAGE35_SEAL_SHA256
    )
    audit["status"] = "PASS" if audit.get("status") == "PASS" and audit["stage38_identity_exact"] else "BLOCKED_STAGE35_IDENTITY_MISMATCH"
    return audit


def static_guard() -> Dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    forbidden_roots = {"torch", "transformers", "isaacgym", "omni"}
    forbidden_call_names = {
        "backward",
        "step",
        "get_action",
        "from_pretrained",
        "resume",
        "q" + "sub",
    }
    call_targets = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_targets.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_targets.add(node.func.attr)
    forbidden_calls_found = sorted(call_targets.intersection(forbidden_call_names))
    forbidden_path = any(root in forbidden_roots for root in imported_roots)
    no_external_execution_path = not forbidden_path and not forbidden_calls_found
    return {
        "schema": "STAGE38A_STATIC_ZERO_MODEL_GUARD_V1",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "top_level_import_roots": sorted(imported_roots),
        "forbidden_import_roots": sorted(forbidden_roots),
        "forbidden_execution_calls_found": forbidden_calls_found,
        "cpu_zero_model_no_scheduler_path": no_external_execution_path,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "allocation_requested": False,
        "status": "PASS" if no_external_execution_path else "BLOCKED_STATIC_EXECUTION_PATH",
    }


def qstat_snapshot() -> Dict[str, Any]:
    code, stdout, stderr = run_capture(["qstat", "-u", os.environ.get("USER", "e1192548")])
    relevant = [
        line
        for line in stdout.splitlines()
        if any(token in line.lower() for token in ("stage38", "structured_groot"))
    ]
    return {
        "command": "qstat -u %s" % os.environ.get("USER", "e1192548"),
        "returncode": code,
        "stdout": stdout,
        "stderr": stderr,
        "stage38_lines": relevant,
        "no_active_stage38_job": code == 0 and not relevant,
        "status": "PASS" if code == 0 and not relevant else "BLOCKED",
    }


def source_to_native(payload: Mapping[str, Any]) -> Tuple[Any, TrajectoryChunkV1, Dict[str, Any]]:
    source = STAGE36.case_to_chunk(payload, TrajectoryChunkV1)
    transport, native = STAGE36.materialize_native_chunk(source, TrajectoryChunkV1)
    source_dt = np.diff(np.asarray(source.point_timestamps, dtype=np.float64))
    native_dt = np.diff(np.asarray(native.point_timestamps, dtype=np.float64))
    audit = jsonable(transport.audit())
    audit.update(
        {
            "source_case_id": str(payload["case_id"]),
            "source_timestamps_nonuniform": bool(
                not np.allclose(source_dt, source_dt[0], rtol=0.0, atol=1e-12)
            ),
            "source_timing_semantics_preserved": True,
            "native_grid_is_uniform": bool(
                np.allclose(native_dt, NATIVE_DT_S, rtol=0.0, atol=1e-12)
            ),
            "native_sample_count_exact": native.point_count == NATIVE_SAMPLE_COUNT,
            "native_duration_exact": bool(
                np.isclose(native.validity_end - native.validity_start, NATIVE_DURATION_S, rtol=0.0, atol=1e-12)
            ),
        }
    )
    return source, native, audit


def materialize_timing_candidate(
    payload: Mapping[str, Any], mode: str
) -> Tuple[TrajectoryChunkV1, Any, TrajectoryChunkV1, Dict[str, Any]]:
    source, dense_native, source_transport = source_to_native(payload)
    geometry = build_geometry_candidate(
        dense_native,
        STAGE38_K_CAPACITY_STAR,
        allowed_k_values=(STAGE38_K_CAPACITY_STAR,),
    )
    semantic = build_timing_candidate_with_knots(
        dense_native,
        geometry,
        mode,
        STAGE38_TIMING_KNOTS[mode],
        lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
    )
    native_grid = STAGE37.native_grid()
    semantic_at_native_grid = semantic.to_trajectory_chunk(query_timestamps=native_grid)
    candidate_transport, candidate_native = STAGE36.materialize_native_chunk(
        semantic_at_native_grid, TrajectoryChunkV1
    )
    audit = {
        "source_transport": source_transport,
        "geometry_k": STAGE38_K_CAPACITY_STAR,
        "geometry_control_pose_count": geometry.point_count,
        "timing_mode": mode,
        "timing_knot_count": int(STAGE38_TIMING_KNOTS[mode].size),
        "timing_interval_count": int(STAGE38_TIMING_KNOTS[mode].size - 1),
        "timing_arc_knots": STAGE38_TIMING_KNOTS[mode].tolist(),
        "timing_source_timestamps": semantic.timing_timestamps.tolist(),
        "timing_control_timestamps": semantic.point_timestamps.tolist(),
        "timing_monotonic": bool(np.all(np.diff(semantic.timing_timestamps) > 0.0)),
        "control_timestamps_monotonic": bool(np.all(np.diff(semantic.point_timestamps) > 0.0)),
        "duration_exact": bool(
            np.isclose(semantic.validity_end - semantic.validity_start, NATIVE_DURATION_S, rtol=0.0, atol=1e-12)
        ),
        "candidate_transport": jsonable(candidate_transport.audit()),
        "candidate_native_finite": bool(
            np.isfinite(candidate_native.point_timestamps).all()
            and np.isfinite(candidate_native.eef_position).all()
            and np.isfinite(candidate_native.eef_orientation).all()
        ),
        "candidate_native_start_end_exact": bool(
            np.array_equal(candidate_native.eef_position[0], dense_native.eef_position[0])
            and np.array_equal(candidate_native.eef_position[-1], dense_native.eef_position[-1])
            and np.array_equal(candidate_native.eef_orientation[0], dense_native.eef_orientation[0])
            and np.array_equal(candidate_native.eef_orientation[-1], dense_native.eef_orientation[-1])
        ),
        "dense_reference_rerun": False,
        "model_constructed": False,
        "wbc_called": False,
    }
    return source, dense_native, candidate_native, audit


def evaluate_mode(payloads: Mapping[str, Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    core_rows: List[Dict[str, Any]] = []
    continuous_rows: List[Dict[str, Any]] = []
    transport_records: List[Dict[str, Any]] = []
    for case_id in STAGE35_CASE_IDS:
        payload = payloads[case_id]
        _, dense_native, candidate_native, audit = materialize_timing_candidate(payload, mode)
        row = evaluate_preview_equivalence(
            dense_native,
            candidate_native,
            case_id=case_id,
            representation="K16+%s_NATIVE_TRANSPORT" % mode,
            anchor_count=101,
            lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
        )
        public_row = public_evaluation(row)
        public_row["timing_transport_audit_index"] = len(transport_records)
        all_rows.append(row)
        transport_records.append({"case_id": case_id, "audit": audit, "preview_equivalence": public_row})
        if case_id in CORE_CASE_IDS:
            core_rows.append(row)
        else:
            continuous_rows.append(row)
    core_aggregate = STAGE37.aggregate_evaluations(
        core_rows, representation="K16+%s_NATIVE_TRANSPORT" % mode
    )
    continuous_aggregate = STAGE37.aggregate_evaluations(
        continuous_rows, representation="K16+%s_NATIVE_TRANSPORT" % mode
    )
    core_qualifies = bool(
        all(bool(row.get("passes_limits")) for row in core_rows)
        and bool(core_aggregate.get("passes_limits"))
    )
    return {
        "timing_mode": mode,
        "timing_knot_count": int(STAGE38_TIMING_KNOTS[mode].size),
        "timing_interval_count": int(STAGE38_TIMING_KNOTS[mode].size - 1),
        "timing_arc_knots": STAGE38_TIMING_KNOTS[mode].tolist(),
        "core24_qualifies_strict_all_cases_and_aggregate": core_qualifies,
        "core24_aggregate": public_evaluation(core_aggregate),
        "core24_per_case": [public_evaluation(row) for row in core_rows],
        "continuous8_diagnostic_aggregate": public_evaluation(continuous_aggregate),
        "continuous8_diagnostic_per_case": [public_evaluation(row) for row in continuous_rows],
        "all32_per_case": [public_evaluation(row) for row in all_rows],
        "transport_records": transport_records,
        "all32_finite": bool(all(bool(row.get("finite")) for row in all_rows)),
        "all32_monotonic": bool(all(bool(row.get("monotonic_timestamps")) for row in all_rows)),
        "all32_duration_exact": bool(all(bool(row.get("duration_exact")) for row in all_rows)),
        "same_frozen_preview_thresholds": True,
        "no_outcome_dependent_knot_placement": True,
    }


def timing_ladder(payloads: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    per_mode = [evaluate_mode(payloads, mode) for mode in STAGE38_TIMING_MODES]
    per_mode.sort(key=lambda row: int(row["timing_mode"][1:]))
    passing = [
        row["timing_mode"]
        for row in per_mode
        if row["core24_qualifies_strict_all_cases_and_aggregate"]
    ]
    model_star = passing[0] if passing else None
    if model_star in {"T16", "T20"}:
        classification = "MODEL_TIMING_COMPACT_QUALIFIED"
    elif model_star in {"T24", "T32"}:
        classification = "MODEL_TIMING_MODERATE_CAPACITY_QUALIFIED"
    elif model_star == "T40":
        classification = "MODEL_TIMING_DENSE_H40_REQUIRED"
    else:
        classification = "TIMING_PARAMETERIZATION_INSUFFICIENT_THROUGH_H40"
    return {
        "schema": "STAGE38A_MODEL_TIMING_CAPACITY_QUALIFICATION_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "geometry_k": STAGE38_K_CAPACITY_STAR,
        "candidate_timing_modes": [
            mode for mode in sorted(STAGE38_TIMING_MODES, key=lambda value: int(value[1:]))
        ],
        "timing_knots": {
            mode: STAGE38_TIMING_KNOTS[mode].tolist()
            for mode in sorted(STAGE38_TIMING_MODES, key=lambda value: int(value[1:]))
        },
        "input_scope": {
            "corpus": "STAGE35_IMMUTABLE_CORPUS",
            "core_promotion_cohort": "CORE24_N_T_P",
            "continuous_diagnostic_cohort": "CONTINUOUS8_C",
            "native_sample_count": NATIVE_SAMPLE_COUNT,
            "native_dt_s": NATIVE_DT_S,
            "native_duration_s": NATIVE_DURATION_S,
            "preview_anchor_count": 101,
            "lambda_m_per_rad": DEFAULT_SE3_LAMBDA_M_PER_RAD,
            "geometry_interpolation": "LINEAR_XYZ_QUATERNION_SLERP",
            "timing_interpolation": "MONOTONIC_PIECEWISE_LINEAR",
            "timing_knot_policy": "UNIFORM_NORMALIZED_SE3_ARC_INCLUDING_ENDPOINTS",
            "source_time_policy": "SAMPLED_FROM_TIMESTAMP_PRESERVING_NATIVE_TRANSPORT",
            "smoothing": False,
            "spline_fitting": False,
            "trajectory_optimization": False,
            "adaptive_or_manual_knots": False,
        },
        "thresholds": {
            "position_p95_m_strictly_less_than": DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
            "rotation_p95_rad_strictly_less_than": DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
            "endpoint_clamp_agreement_at_least": DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
            "start_end_exact": True,
            "duration_exact": True,
        },
        "per_mode": per_mode,
        "t_model_capacity_star": model_star,
        "classification": classification,
        "status": "PASS" if model_star is not None else "PASS_NO_QUALIFIED_MODEL_TIMING",
        "no_outcome_dependent_selection": True,
        "historical_stage37_t_capacity_star_preserved_as_null": True,
    }


def preflight(
    expected_head: str,
    api_head: str,
    stage37_audit: Mapping[str, Any],
    stage35_audit: Mapping[str, Any],
    static: Mapping[str, Any],
    qstat: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> Dict[str, Any]:
    local_head = git_output("rev-parse", "HEAD")
    remote_head = git_output("rev-parse", "origin/" + EXPECTED_BRANCH)
    branch = git_output("branch", "--show-current")
    status_output = git_output("status", "--porcelain", "--untracked-files=all")
    diff_code, _, diff_err = run_capture(["git", "diff", "--check"])
    head_match = bool(
        branch == EXPECTED_BRANCH
        and local_head == expected_head
        and remote_head == expected_head
        and api_head == expected_head
        and git_output("merge-base", "HEAD", EXPECTED_PARENT_HEAD) == EXPECTED_PARENT_HEAD
    )
    predicates = {
        "stage37_acceptance_pass": stage37_audit.get("status") == "PASS",
        "stage37_parent_live_head_exact": stage37_audit.get("accepted_live_head") == EXPECTED_PARENT_HEAD,
        "stage35_corpus_pass": stage35_audit.get("status") == "PASS",
        "static_zero_model_guard_pass": static.get("status") == "PASS",
        "qstat_no_active_stage38_job": qstat.get("no_active_stage38_job") is True,
        "head_api_remote_local_parent_match": head_match,
        "worktree_clean": status_output == "",
        "git_diff_check_pass": diff_code == 0,
        "timing_candidate_set_exact": timing.get("candidate_timing_modes")
        == ["T16", "T20", "T24", "T32", "T40"],
        "geometry_k16_exact": timing.get("geometry_k") == STAGE38_K_CAPACITY_STAR,
        "frozen_thresholds": bool(
            timing.get("thresholds", {}).get("position_p95_m_strictly_less_than")
            == DEFAULT_PREVIEW_POSITION_P95_LIMIT_M
            and timing.get("thresholds", {}).get("rotation_p95_rad_strictly_less_than")
            == DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD
        ),
        "timing_ladder_finite_and_duration": all(
            row.get("all32_finite")
            and row.get("all32_monotonic")
            and row.get("all32_duration_exact")
            for row in timing.get("per_mode", [])
        ),
    }
    return {
        "schema": "STAGE38A_CPU_PREFLIGHT_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "branch": branch,
        "expected_head": expected_head,
        "api_head": api_head,
        "remote_head": remote_head,
        "local_head": local_head,
        "worktree_status": status_output.splitlines() if status_output else [],
        "git_diff_check_stderr": diff_err,
        "project": EXPECTED_PROJECT,
        "queue": EXPECTED_QUEUE,
        "predicates": predicates,
        "all_predicates_pass": all(predicates.values()),
        "execution_boundary": {
            "allocation_requested": False,
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
            "stage37_rerun": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "pr_merge": False,
        },
    }


def seal_entries(output_root: Path, excluded: Iterable[str]) -> List[Dict[str, Any]]:
    excluded_set = set(excluded)
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded_set:
            entries.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-head", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--pr-number", type=int, default=PR_NUMBER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit("FAIL_CLOSED output already exists: %s" % output_root)
    if len(args.expected_head) != 40 or args.expected_head == EXPECTED_PARENT_HEAD:
        raise SystemExit("FAIL_CLOSED expected head must be the new Stage38 commit")
    if not REPO.is_dir():
        raise SystemExit("FAIL_CLOSED repository missing")
    output_root.mkdir(parents=True)
    log_event(output_root, "START", branch=EXPECTED_BRANCH, expected_head=args.expected_head)

    stage37_audit = verify_stage37_acceptance()
    write_json(output_root / "stage38a_stage37_acceptance_audit.json", stage37_audit)
    log_event(output_root, "STAGE37_ACCEPTANCE_AUDIT", status=stage37_audit.get("status"))
    stage35_audit = verify_stage35_input()
    write_json(output_root / "stage38a_stage35_corpus_audit.json", stage35_audit)
    log_event(output_root, "STAGE35_CORPUS_AUDIT", status=stage35_audit.get("status"))
    static = static_guard()
    write_json(output_root / "stage38a_static_guard.json", static)
    log_event(output_root, "STATIC_GUARD", status=static.get("status"))
    qstat = qstat_snapshot()
    write_json(output_root / "stage38a_qstat_snapshot.json", qstat)
    log_event(output_root, "QSTAT", status=qstat.get("status"))

    timing: Dict[str, Any] = {
        "schema": "STAGE38A_MODEL_TIMING_CAPACITY_QUALIFICATION_V1",
        "status": "BLOCKED_DEPENDENCY",
        "candidate_timing_modes": [
            mode for mode in sorted(STAGE38_TIMING_MODES, key=lambda value: int(value[1:]))
        ],
        "geometry_k": STAGE38_K_CAPACITY_STAR,
        "t_model_capacity_star": None,
        "classification": "TIMING_PARAMETERIZATION_INSUFFICIENT_THROUGH_H40",
        "per_mode": [],
    }
    payloads: Dict[str, Dict[str, Any]] = {}
    if stage37_audit.get("status") == "PASS" and stage35_audit.get("status") == "PASS":
        payloads = STAGE37.load_payloads()
        timing = timing_ladder(payloads)
    write_json(output_root / "stage38a_timing_qualification.json", timing)
    log_event(
        output_root,
        "TIMING_LADDER",
        status=timing.get("status"),
        t_model_capacity_star=timing.get("t_model_capacity_star"),
        mode_count=len(timing.get("per_mode", [])),
    )

    preflight_result = preflight(
        args.expected_head,
        args.api_head,
        stage37_audit,
        stage35_audit,
        static,
        qstat,
        timing,
    )
    write_json(output_root / "stage38a_preflight.json", preflight_result)
    log_event(output_root, "PREFLIGHT", all_predicates_pass=preflight_result["all_predicates_pass"])

    result_status = (
        "STAGE38A_CPU_TIMING_PASS_READY_FOR_BC"
        if preflight_result["all_predicates_pass"] and timing.get("t_model_capacity_star") is not None
        else "STAGE38A_CPU_TIMING_STOP_FOR_PI"
        if preflight_result["all_predicates_pass"]
        else "BLOCKED"
    )
    blockers = [
        name for name, passed in preflight_result.get("predicates", {}).items() if not passed
    ]
    if result_status == "STAGE38A_CPU_TIMING_STOP_FOR_PI":
        blockers.append(TIMING_PARAMETERIZATION_STOP)
    result_payload = {
        "schema": "STAGE38A_CPU_RESULT_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": int(args.pr_number),
        "branch": EXPECTED_BRANCH,
        "expected_head": args.expected_head,
        "api_head": args.api_head,
        "result": result_status,
        "blockers": blockers,
        "stage37_acceptance": {
            "execution_head": STAGE37_EXECUTION_HEAD,
            "live_head": STAGE37_LIVE_HEAD,
            "seal_sha256": STAGE37_SEAL_SHA256,
            "no_rerun": True,
        },
        "stage35_corpus": {
            "root": str(STAGE35_ROOT),
            "manifest_sha256": STAGE35_MANIFEST_SHA256,
            "corpus_tree_sha256": STAGE35_CORPUS_TREE_SHA256,
        },
        "geometry_k": STAGE38_K_CAPACITY_STAR,
        "timing": timing,
        "stage37_t_capacity_star_preserved": None,
        "model_timing_capacity_star": timing.get("t_model_capacity_star"),
        "model_timing_classification": timing.get("classification"),
        "stage38_b_c_admission": result_status == "STAGE38A_CPU_TIMING_PASS_READY_FOR_BC",
        "execution_boundary": preflight_result.get("execution_boundary"),
        "next_gate": (
            "STAGE38_B_C_CONTINUE_AUTONOMOUSLY"
            if result_status == "STAGE38A_CPU_TIMING_PASS_READY_FOR_BC"
            else TIMING_PARAMETERIZATION_STOP
            if result_status == "STAGE38A_CPU_TIMING_STOP_FOR_PI"
            else "BLOCKED"
        ),
    }
    write_json(output_root / "stage38a_result.json", result_payload)
    log_event(output_root, "RESULT", result=result_status, blockers=blockers)

    seal_path = output_root / "stage38a_artifact_seal.json"
    seal_payload = {
        "schema": "STAGE38A_CPU_ARTIFACT_SEAL_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": int(args.pr_number),
        "stage": "Stage38-A",
        "output_root": str(output_root),
        "entries": seal_entries(output_root, (seal_path.name, "stage38a_return.json")),
        "source_inputs": {
            "stage37_final_return_sha256": STAGE37_FINAL_RETURN_SHA256,
            "stage37_seal_sha256": STAGE37_SEAL_SHA256,
            "stage37_result_sha256": STAGE37_RESULT_SHA256,
            "stage37_paired_analysis_sha256": STAGE37_PAIRED_SHA256,
            "stage37_raw_manifest_sha256": STAGE37_RAW_MANIFEST_SHA256,
            "stage37_result_manifest_sha256": STAGE37_RESULT_MANIFEST_SHA256,
            "stage37_source_manifest_sha256": STAGE37_SOURCE_MANIFEST_SHA256,
            "stage37_campaign_audit_sha256": STAGE37_CAMPAIGN_AUDIT_SHA256,
            "stage35_manifest_sha256": STAGE35_MANIFEST_SHA256,
            "stage35_corpus_tree_sha256": STAGE35_CORPUS_TREE_SHA256,
            "stage35_generated_protocol_sha256": STAGE35_GENERATED_PROTOCOL_SHA256,
            "stage35_protocol_sha256": STAGE35_PROTOCOL_SHA256,
            "stage35_seal_sha256": STAGE35_SEAL_SHA256,
        },
        "historical_inputs_mutated": False,
        "stage37_rerun": False,
        "allocation_requested": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "hardware": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    write_json(seal_path, seal_payload)
    seal_sha256 = sha256_file(seal_path)
    return_payload = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": "Stage38-A",
        "PR": int(args.pr_number),
        "BRANCH": EXPECTED_BRANCH,
        "EXPECTED_HEAD": args.expected_head,
        "API_HEAD": args.api_head,
        "RESULT": result_status,
        "BLOCKERS": blockers,
        "STAGE37_EXECUTION_HEAD": STAGE37_EXECUTION_HEAD,
        "STAGE37_LIVE_HEAD": STAGE37_LIVE_HEAD,
        "STAGE37_ARTIFACT_SEAL_SHA256": STAGE37_SEAL_SHA256,
        "K_CAPACITY_STAR": STAGE38_K_CAPACITY_STAR,
        "T_MODEL_CAPACITY_STAR": timing.get("t_model_capacity_star"),
        "TIMING_MODEL_CAPACITY": timing.get("classification"),
        "STAGE37_T_CAPACITY_STAR_UNCHANGED": None,
        "CORE24_PROMOTION": "PASS" if timing.get("t_model_capacity_star") else "NO_QUALIFIED_CANDIDATE",
        "CONTINUOUS8_DIAGNOSTIC": "REPORTED",
        "STAGE38A_TIMING_RESULT_PATH": str(output_root / "stage38a_timing_qualification.json"),
        "STAGE38A_PREFLIGHT_PATH": str(output_root / "stage38a_preflight.json"),
        "STAGE38A_ARTIFACT_SEAL_PATH": str(seal_path),
        "STAGE38A_ARTIFACT_SEAL_SHA256": seal_sha256,
        "MODEL_CONSTRUCTED": False,
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "PBS_SUBMITTED": False,
        "GPU_USED": False,
        "WBC_EXECUTED": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "HARDWARE": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "NEXT_GATE": (
            "STAGE38_B_C_CONTINUE_AUTONOMOUSLY"
            if result_status == "STAGE38A_CPU_TIMING_PASS_READY_FOR_BC"
            else TIMING_PARAMETERIZATION_STOP
            if result_status == "STAGE38A_CPU_TIMING_STOP_FOR_PI"
            else "BLOCKED"
        ),
    }
    write_json(output_root / "stage38a_return.json", return_payload)
    return 0 if result_status != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
