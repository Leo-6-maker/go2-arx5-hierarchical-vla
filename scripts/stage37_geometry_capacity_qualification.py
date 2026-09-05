#!/usr/bin/env python3
"""Run the zero-model Stage37 geometry-capacity qualification ladder.

Stage37-G0/G1/T is deliberately CPU-only.  It verifies the immutable
Stage36-R and Stage35 inputs, converts the already sealed K={4,6,9,12}
diagnostics into a deterministic failure map, evaluates the preregistered
K={16,20,24,32,40} and T={T1,T3,T5,T9} ladders, and prepares (but does not
submit) the one authorized structured-oracle WBC campaign.  No model,
simulator, CUDA device, PBS allocation, or WBC call is reachable here.

The geometry qualification is intentionally performed on the frozen dense
trajectory after the 121-point/60-Hz native materialization.  The candidate
MotionIntentV2 is then queried on that same native grid and passed through the
same native transport used by the eventual WBC input.  This makes the G1
qualification and the structured campaign inputs share one transport path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
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
    TIMING_MODES,
    aggregate_evaluations,
    build_geometry_candidate,
    build_timing_candidate,
    evaluate_preview_equivalence,
    public_evaluation,
)
from go2arx5_vla.trajectory.native_transport import (  # noqa: E402
    NATIVE_DT_S,
    NATIVE_DURATION_S,
    NATIVE_SAMPLE_COUNT,
)


STAGE36 = importlib.import_module("stage36_wbc_authority_oracle_qualification")
CAMPAIGN = importlib.import_module("stage36_wbc_oracle_campaign")

PI_GATE_ID = "PI_GATE_STATE_0025"
PR_NUMBER = 22
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
EXPECTED_BRANCH = "stage37/geometry-capacity-qualification"
EXPECTED_BASE_HEAD = "f5506ff476323cf92853198e38bbe0a6b561bd85"
STAGE36_R_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_r_re_adjudication_20260906_f5506ff_v2"
)
STAGE36_R_RESULT = STAGE36_R_ROOT / "stage36_r_result.json"
STAGE36_R_SEAL = STAGE36_R_ROOT / "stage36_r_artifact_seal.json"
STAGE36_R_RAW_MANIFEST = STAGE36_R_ROOT / "stage36_r_raw_hash_manifest.json"
STAGE36_R_RESULT_SHA256 = (
    "9d86e5db2a89c1f9290338c90a93699e2fd6eecf8da9ae3540c5543d11a95e9f"
)
STAGE36_R_SEAL_SHA256 = (
    "b0d0930685d34d9d566f4721ced0f91a0347a869aae3998c1985e54acc0bcab1"
)
STAGE36_R_RAW_MANIFEST_SHA256 = (
    "33239f355182d83cdd6d4d0a38a198ffadaf980964700fea97dc8613accf5449"
)

STAGE36_ORIGINAL_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_eval_602744.hopper-m-02"
)
STAGE36_ORIGINAL_RESULT = STAGE36_ORIGINAL_ROOT / "stage36_result.json"
STAGE36_ORIGINAL_RAW_MANIFEST = STAGE36_ORIGINAL_ROOT / "stage36_raw_hash_manifest.json"
STAGE36_ORIGINAL_SEAL = STAGE36_ORIGINAL_ROOT / "stage36_artifact_seal.json"
STAGE36_ORIGINAL_RESULT_SHA256 = (
    "3dcf2b95af39f0c6069d540d385082812fff453eb6966d98b20aa42419d77ac1"
)
STAGE36_ORIGINAL_RAW_MANIFEST_SHA256 = (
    "e6b13ce28c48eef1fe16704f50a80ba90920e6d28bc6adcd2b2c3594b5f0b236"
)
STAGE36_ORIGINAL_SEAL_SHA256 = (
    "52f8cf662f9a0ce5ca0a0ec65d9e44cb8199382765610d32e71bc2fb64e0b77b"
)

G0_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_cpu_20260906_resumea1a07a3_v11"
)
G0_RESULT = G0_ROOT / "stage36_d1_geometry_qualification.json"
G0_SEAL = G0_ROOT / "stage36_artifact_seal.json"
G0_RESULT_SHA256 = (
    "bca0e41460a98efda3743221caeacc998c5862182c902db7f5d7715ae5ae12e9"
)
G0_SEAL_SHA256 = (
    "ea885f332c4cd13ba3daeb04650fee6adc7aa75dd6fe69b42bfe060686be7851"
)

STAGE35_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage35_motion_api_authority_closure_20260905/stage35_cpu_20260905_v2"
)
STAGE35_COMMITTED_PROTOCOL = REPO / "manifests/STAGE35_ORACLE_CORPUS_V1.json"
STAGE35_CORPUS_TREE_SHA256 = (
    "b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414"
)
STAGE35_MANIFEST_SHA256 = (
    "c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642"
)
STAGE35_PROTOCOL_SHA256 = (
    "81585c50b29ba67d65e346d3b6fb10420a52ca283d78e54d7476073ecacca22d"
)
STAGE35_GENERATED_PROTOCOL_SHA256 = (
    "ae77add1a6c4845f3a1517200a739fb0db27d0da44c19b8eea01552170958e45"
)
STAGE35_SEAL_SHA256 = (
    "5ec9dc293546ea33c39c225d184921e77f1ae6f86756ec1bd0304cb19aa63e89"
)

STAGE37_K_VALUES = (16, 20, 24, 32, 40)
STAGE37_TIMING_MODES = ("T1", "T3", "T5", "T9")
CASE_FAMILIES = ("N", "T", "P", "C")
CORE_FAMILIES = ("N", "T", "P")
CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in CASE_FAMILIES
    for index in range(1, 9)
)
CORE_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in CORE_FAMILIES
    for index in range(1, 9)
)
CONTINUOUS_CASE_IDS = tuple("C%02d" % index for index in range(1, 9))
NEXT_GATE = "GPT_PI_REVIEW_BEFORE_STRUCTURED_GR00T_DESIGN"
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage37_geometry_capacity_qualification_20260906/"
    "stage37_cpu_f5506ff_v1"
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
    payload = jsonable(payload)
    finite_json(payload, str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_event(output_root: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(jsonable(fields))
    with (output_root / "stage37_cpu_execution.log").open("a", encoding="utf-8") as handle:
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
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "expected_sha256": expected_sha256,
    }
    result["hash_match"] = bool(
        path.is_file() and (expected_sha256 is None or result["sha256"] == expected_sha256)
    )
    return result


def verify_seal(root: Path, seal_path: Path, expected_sha256: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "root": str(root),
        "seal_path": str(seal_path),
        "seal_file": file_record(seal_path, expected_sha256),
        "entry_count": 0,
        "entries_verified": 0,
        "mismatches": [],
        "status": "BLOCKED",
    }
    if not result["seal_file"]["hash_match"]:
        return result
    seal = load_json(seal_path)
    entries = seal.get("entries", [])
    result["entry_count"] = len(entries)
    for entry in entries:
        path = Path(str(entry.get("path", "")))
        if not path.is_absolute():
            path = root / str(entry.get("relative_path", ""))
        actual = sha256_file(path) if path.is_file() else None
        if (
            not path.is_file()
            or actual != entry.get("sha256")
            or int(path.stat().st_size) != int(entry.get("size_bytes", -1))
        ):
            result["mismatches"].append(
                {
                    "path": str(path),
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": actual,
                    "expected_size_bytes": entry.get("size_bytes"),
                    "actual_size_bytes": int(path.stat().st_size) if path.is_file() else None,
                }
            )
        else:
            result["entries_verified"] += 1
    result["status"] = "PASS" if not result["mismatches"] else "BLOCKED_SEAL_ENTRY_MISMATCH"
    return result


def verify_stage36_inputs() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "STAGE37_STAGE36_R_INPUT_AUTHORITY_AUDIT_V1",
        "stage36_r_root": str(STAGE36_R_ROOT),
        "stage36_r_result": file_record(STAGE36_R_RESULT, STAGE36_R_RESULT_SHA256),
        "stage36_r_raw_manifest": file_record(
            STAGE36_R_RAW_MANIFEST, STAGE36_R_RAW_MANIFEST_SHA256
        ),
        "stage36_r_seal": verify_seal(
            STAGE36_R_ROOT, STAGE36_R_SEAL, STAGE36_R_SEAL_SHA256
        ),
        "original_root": str(STAGE36_ORIGINAL_ROOT),
        "original_result": file_record(
            STAGE36_ORIGINAL_RESULT, STAGE36_ORIGINAL_RESULT_SHA256
        ),
        "original_raw_manifest": file_record(
            STAGE36_ORIGINAL_RAW_MANIFEST, STAGE36_ORIGINAL_RAW_MANIFEST_SHA256
        ),
        "original_seal": file_record(STAGE36_ORIGINAL_SEAL, STAGE36_ORIGINAL_SEAL_SHA256),
        "stage36_r_core24_pass": False,
        "stage36_r_continuous8_pass": False,
        "stage36_r_rows_exact": False,
        "original_evidence_hashes_exact": False,
        "status": "BLOCKED",
    }
    try:
        r_result = load_json(STAGE36_R_RESULT)
        original_result = load_json(STAGE36_ORIGINAL_RESULT)
        result["stage36_r_core24_pass"] = bool(
            r_result.get("core24", {}).get("status") == "PASS"
            and r_result.get("core24", {}).get("classification")
            == "WBC_CORE_LIFECYCLE_QUALIFIED_ON_STAGE35_CORE24"
        )
        result["stage36_r_continuous8_pass"] = bool(
            r_result.get("continuous8", {}).get("status") == "PASS"
            and r_result.get("continuous8", {}).get("classification")
            == "WBC_CONTINUOUS_BASE_LIFECYCLE_QUALIFIED_ON_TESTED_ENVELOPE"
        )
        row_ids = [str(row.get("row_id")) for row in r_result.get("rows", [])]
        result["stage36_r_rows_exact"] = bool(
            len(row_ids) == 32 and len(set(row_ids)) == 32 and all(row_ids)
            and r_result.get("corrected_tracking_pass_rows") == 32
            and r_result.get("true_joint_limit_termination_rows") == 0
        )
        result["original_evidence_hashes_exact"] = bool(
            result["original_result"]["hash_match"]
            and result["original_raw_manifest"]["hash_match"]
            and result["original_seal"]["hash_match"]
            and r_result.get("input_authority", {}).get(
                "original_evidence_verified_before_analysis"
            )
            is True
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["status"] = (
        "PASS"
        if result["stage36_r_result"]["hash_match"]
        and result["stage36_r_raw_manifest"]["hash_match"]
        and result["stage36_r_seal"]["status"] == "PASS"
        and result["stage36_r_core24_pass"]
        and result["stage36_r_continuous8_pass"]
        and result["stage36_r_rows_exact"]
        and result["original_evidence_hashes_exact"]
        else "BLOCKED_STAGE36_R_INPUT_MISMATCH"
    )
    return result


def canonical_stage35_tree(manifest: Mapping[str, Any]) -> str:
    rows = [
        {"relative_path": row.get("relative_path"), "sha256": row.get("sha256")}
        for row in manifest.get("case_records", [])
    ]
    return sha256_bytes(canonical_json(rows))


def verify_stage35_corpus() -> Dict[str, Any]:
    manifest_path = STAGE35_ROOT / "stage35_oracle_corpus_manifest.json"
    generated_protocol_path = STAGE35_ROOT / "stage35_oracle_corpus_protocol.json"
    seal_path = STAGE35_ROOT / "stage35_artifact_seal.json"
    result: Dict[str, Any] = {
        "schema": "STAGE37_STAGE35_CORPUS_INPUT_AUDIT_V1",
        "root": str(STAGE35_ROOT),
        "manifest": file_record(manifest_path, STAGE35_MANIFEST_SHA256),
        "committed_protocol": file_record(STAGE35_COMMITTED_PROTOCOL, STAGE35_PROTOCOL_SHA256),
        "generated_protocol": file_record(
            generated_protocol_path, STAGE35_GENERATED_PROTOCOL_SHA256
        ),
        "seal": file_record(seal_path, STAGE35_SEAL_SHA256),
        "tree_sha256": None,
        "tree_sha256_expected": STAGE35_CORPUS_TREE_SHA256,
        "case_count": 0,
        "case_ids_exact": False,
        "case_records": [],
        "status": "BLOCKED",
    }
    try:
        manifest = load_json(manifest_path)
        protocol = load_json(STAGE35_COMMITTED_PROTOCOL)
        rows = manifest.get("case_records", [])
        result["tree_sha256"] = canonical_stage35_tree(manifest)
        result["case_count"] = len(rows)
        manifest_ids = [row.get("case_id") for row in rows]
        case_dir = STAGE35_ROOT / "corpus/cases"
        actual_ids = sorted(
            path.stem
            for path in case_dir.glob("*.json")
            if path.is_file() and not path.is_symlink()
        )
        result["case_ids_exact"] = bool(
            manifest_ids == list(CASE_IDS)
            and actual_ids == sorted(CASE_IDS)
            and protocol.get("case_ids") == list(CASE_IDS)
        )
        for row in rows:
            relative = str(row.get("relative_path", ""))
            path = STAGE35_ROOT / relative
            case_result = file_record(path, row.get("sha256"))
            case_result.update(
                {
                    "case_id": row.get("case_id"),
                    "family": row.get("family"),
                    "relative_path": relative,
                    "size_match": bool(
                        path.is_file()
                        and int(path.stat().st_size) == int(row.get("size_bytes", -1))
                    ),
                }
            )
            if path.is_file() and case_result["hash_match"] and case_result["size_match"]:
                payload = load_json(path)
                timestamps = payload.get("timestamps", [])
                case_result["payload_contract"] = {
                    "case_id_match": payload.get("case_id") == row.get("case_id"),
                    "sample_count": len(timestamps),
                    "sample_count_exact": len(timestamps) == 121,
                    "start_exact": bool(timestamps and timestamps[0] == 0.0),
                    "end_exact": bool(timestamps and timestamps[-1] == 2.0),
                    "finite": bool(
                        np.isfinite(np.asarray(timestamps, dtype=np.float64)).all()
                    ),
                }
            result["case_records"].append(case_result)
        case_pass = all(
            row.get("hash_match")
            and row.get("size_match")
            and row.get("payload_contract", {}).get("case_id_match")
            and row.get("payload_contract", {}).get("sample_count_exact")
            and row.get("payload_contract", {}).get("start_exact")
            and row.get("payload_contract", {}).get("end_exact")
            and row.get("payload_contract", {}).get("finite")
            for row in result["case_records"]
        )
        result["status"] = (
            "PASS"
            if result["manifest"]["hash_match"]
            and result["committed_protocol"]["hash_match"]
            and result["generated_protocol"]["hash_match"]
            and result["seal"]["hash_match"]
            and result["tree_sha256"] == STAGE35_CORPUS_TREE_SHA256
            and result["case_count"] == 32
            and result["case_ids_exact"]
            and case_pass
            else "BLOCKED_STAGE35_CORPUS_MISMATCH"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def verify_g0_input() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "STAGE37_G0_FROZEN_DIAGNOSTIC_INPUT_AUDIT_V1",
        "root": str(G0_ROOT),
        "diagnostic": file_record(G0_RESULT, G0_RESULT_SHA256),
        "seal": file_record(G0_SEAL, G0_SEAL_SHA256),
        "k_values_exact": False,
        "status": "BLOCKED",
    }
    if not result["diagnostic"]["hash_match"] or not result["seal"]["hash_match"]:
        return result
    try:
        payload = load_json(G0_RESULT)
        result["k_values_exact"] = [row.get("k") for row in payload.get("per_k", [])] == [
            4,
            6,
            9,
            12,
        ]
        result["status"] = "PASS" if result["k_values_exact"] else "BLOCKED_G0_K_SCOPE"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def metric_value(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def predicate_failures(row: Mapping[str, Any], *, aggregate: bool = False) -> List[str]:
    failures: List[str] = []
    if row.get("finite") is not True:
        failures.append("FINITE")
    if row.get("monotonic_timestamps") is not True:
        failures.append("MONOTONIC_TIMESTAMPS")
    position_p95 = metric_value(row, "preview_position_error_m", "p95")
    rotation_p95 = metric_value(row, "preview_rotation_error_rad", "p95")
    if position_p95 is None or not position_p95 < DEFAULT_PREVIEW_POSITION_P95_LIMIT_M:
        failures.append("POSITION_P95")
    if rotation_p95 is None or not rotation_p95 < DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD:
        failures.append("ROTATION_P95")
    clamp = (
        metric_value(row, "endpoint_clamp_agreement", "min")
        if aggregate
        else metric_value(row, "endpoint_clamp_agreement")
    )
    if clamp is None or clamp < DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN:
        failures.append("ENDPOINT_CLAMP")
    if not aggregate:
        if row.get("start_end_pose_exact") is not True:
            failures.append("START_END_EXACT")
        if row.get("duration_exact") is not True:
            failures.append("DURATION_EXACT")
    return failures


def predicate_margins(row: Mapping[str, Any], *, aggregate: bool = False) -> Dict[str, Any]:
    position_p95 = metric_value(row, "preview_position_error_m", "p95")
    rotation_p95 = metric_value(row, "preview_rotation_error_rad", "p95")
    clamp = (
        metric_value(row, "endpoint_clamp_agreement", "min")
        if aggregate
        else metric_value(row, "endpoint_clamp_agreement")
    )
    return {
        "position_p95_margin_m": None
        if position_p95 is None
        else DEFAULT_PREVIEW_POSITION_P95_LIMIT_M - position_p95,
        "rotation_p95_margin_rad": None
        if rotation_p95 is None
        else DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD - rotation_p95,
        "endpoint_clamp_margin": None
        if clamp is None
        else clamp - DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
        "start_end_exact": row.get("start_end_pose_exact") if not aggregate else None,
        "duration_exact": row.get("duration_exact") if not aggregate else None,
    }


def make_g0_failure_map() -> Dict[str, Any]:
    source = load_json(G0_RESULT)
    per_k: List[Dict[str, Any]] = []
    for record in source.get("per_k", []):
        case_rows: List[Dict[str, Any]] = []
        for row in record.get("per_case", []):
            case_rows.append(
                {
                    "case_id": row.get("case_id"),
                    "family": str(row.get("case_id", ""))[:1],
                    "passes_limits_source": row.get("passes_limits"),
                    "failed_predicates": predicate_failures(row),
                    "predicate_margins": predicate_margins(row),
                    "preview_position_error_m": row.get("preview_position_error_m"),
                    "preview_rotation_error_rad": row.get("preview_rotation_error_rad"),
                    "endpoint_clamp_agreement": row.get("endpoint_clamp_agreement"),
                    "se3_arc_length_m": row.get("se3_arc_length_m"),
                    "se3_arc_length_relative_error": row.get(
                        "se3_arc_length_relative_error"
                    ),
                    "se3_arc_length_ratio": row.get("se3_arc_length_ratio"),
                    "time_to_arc_progress_error": row.get("time_to_arc_progress_error"),
                    "start_end_pose_exact": row.get("start_end_pose_exact"),
                    "duration_exact": row.get("duration_exact"),
                }
            )
        aggregate = record.get("aggregate", {})
        failed_cases = [row["case_id"] for row in case_rows if row["failed_predicates"]]
        per_k.append(
            {
                "k": record.get("k"),
                "qualifies_strict_all_cases_and_aggregate_source": record.get(
                    "qualifies_strict_all_cases_and_aggregate"
                ),
                "failed_case_ids": failed_cases,
                "failed_case_count": len(failed_cases),
                "failed_family_counts": dict(Counter(str(case_id)[0] for case_id in failed_cases)),
                "aggregate_failed_predicates": predicate_failures(aggregate, aggregate=True),
                "aggregate_predicate_margins": predicate_margins(aggregate, aggregate=True),
                "aggregate": aggregate,
                "per_case": case_rows,
            }
        )
    return {
        "schema": "STAGE37_G0_FROZEN_GEOMETRY_FAILURE_MAP_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "source_path": str(G0_RESULT),
        "source_sha256": sha256_file(G0_RESULT),
        "source_seal_path": str(G0_SEAL),
        "source_seal_sha256": sha256_file(G0_SEAL),
        "no_rerun": True,
        "thresholds": {
            "position_p95_m_strictly_less_than": DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
            "rotation_p95_rad_strictly_less_than": DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
            "endpoint_clamp_agreement_at_least": DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
            "start_end_exact": True,
            "duration_exact": True,
        },
        "per_k": per_k,
        "status": "PASS_FROZEN_DIAGNOSTIC_MAP",
    }


def load_payloads() -> Dict[str, Dict[str, Any]]:
    return {
        case_id: load_json(STAGE35_ROOT / "corpus/cases" / (case_id + ".json"))
        for case_id in CASE_IDS
    }


def native_grid() -> np.ndarray:
    return NATIVE_DURATION_S * np.arange(NATIVE_SAMPLE_COUNT, dtype=np.float64) / float(
        NATIVE_SAMPLE_COUNT - 1
    )


def structured_materialization(
    payload: Mapping[str, Any],
    geometry_k: int,
    timing_mode: Optional[str] = None,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    dense_source = STAGE36.case_to_chunk(payload, TrajectoryChunkV1)
    _, dense_native = STAGE36.materialize_native_chunk(dense_source, TrajectoryChunkV1)
    geometry = build_geometry_candidate(
        dense_native,
        geometry_k,
        allowed_k_values=STAGE37_K_VALUES,
    )
    semantic: Any = geometry
    if timing_mode is not None and timing_mode != "T0":
        semantic = build_timing_candidate(dense_native, geometry, timing_mode)
    semantic_native_grid = semantic.to_trajectory_chunk(query_timestamps=native_grid())
    transport, structured_native = STAGE36.materialize_native_chunk(
        semantic_native_grid, TrajectoryChunkV1
    )
    audit = jsonable(transport.audit())
    audit.update(
        {
            "semantic_description": (
                "MOTION_INTENT_V2_AT_NATIVE_GRID"
                if timing_mode is None or timing_mode == "T0"
                else "MOTION_INTENT_V2_TIMING_AT_NATIVE_GRID"
            ),
            "geometry_source": "DENSE_NATIVE_TRANSPORT",
            "geometry_k": int(geometry_k),
            "timing_mode": timing_mode,
            "native_sample_count": int(structured_native.point_count),
            "native_dt_s": float(NATIVE_DT_S),
            "native_duration_s": float(NATIVE_DURATION_S),
            "native_grid_formula": "tau_j=2.0*j/120, j=0..120",
            "finite": bool(
                np.isfinite(structured_native.point_timestamps).all()
                and np.isfinite(structured_native.eef_position).all()
                and np.isfinite(structured_native.eef_orientation).all()
            ),
            "start_end_pose_exact": bool(
                np.array_equal(
                    structured_native.eef_position[0], dense_native.eef_position[0]
                )
                and np.array_equal(
                    structured_native.eef_position[-1], dense_native.eef_position[-1]
                )
                and np.array_equal(
                    structured_native.eef_orientation[0], dense_native.eef_orientation[0]
                )
                and np.array_equal(
                    structured_native.eef_orientation[-1], dense_native.eef_orientation[-1]
                )
            ),
        }
    )
    return dense_native, geometry, structured_native, audit


def evaluate_geometry_candidate(
    payloads: Mapping[str, Mapping[str, Any]], k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_id in CORE_CASE_IDS:
        dense_native, _, structured_native, _ = structured_materialization(payloads[case_id], k)
        evaluation = evaluate_preview_equivalence(
            dense_native,
            structured_native,
            case_id=case_id,
            representation="DENSE_NATIVE_TO_K%d_NATIVE_TRANSPORT" % k,
            anchor_count=101,
        )
        rows.append(evaluation)
    aggregate = aggregate_evaluations(
        rows, representation="DENSE_NATIVE_TO_K%d_NATIVE_TRANSPORT" % k
    )
    return rows, aggregate


def geometry_ladder(payloads: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    per_k: List[Dict[str, Any]] = []
    k_star: Optional[int] = None
    for k in STAGE37_K_VALUES:
        rows, aggregate = evaluate_geometry_candidate(payloads, k)
        qualifies = bool(all(row.get("passes_limits") for row in rows) and aggregate.get("passes_limits"))
        per_k.append(
            {
                "k": k,
                "qualifies_strict_all_cases_and_aggregate": qualifies,
                "aggregate": public_evaluation(aggregate),
                "per_case": [public_evaluation(row) for row in rows],
            }
        )
        if qualifies and k_star is None:
            k_star = k
    if k_star in (16, 20):
        classification = "COMPACT_GEOMETRY_CAPACITY_QUALIFIED"
    elif k_star in (24, 32):
        classification = "WEAK_COMPRESSION_GEOMETRY_QUALIFIED"
    elif k_star == 40:
        classification = "NO_MEANINGFUL_POINT_COUNT_COMPRESSION"
    else:
        classification = "CURRENT_LINEAR_SLERP_GEOMETRY_PARAMETERIZATION_INSUFFICIENT"
    return {
        "schema": "STAGE37_G1_GEOMETRY_CAPACITY_QUALIFICATION_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "candidate_k_values": list(STAGE37_K_VALUES),
        "input_scope": {
            "corpus": "STAGE35_IMMUTABLE_CORPUS_CORE24",
            "geometry_source": "DENSE_NATIVE_TRANSPORT",
            "native_sample_count": NATIVE_SAMPLE_COUNT,
            "native_dt_s": NATIVE_DT_S,
            "native_duration_s": NATIVE_DURATION_S,
            "anchor_count": 101,
            "lambda_m_per_rad": DEFAULT_SE3_LAMBDA_M_PER_RAD,
            "interpolation": "LINEAR_XYZ_QUATERNION_SLERP",
            "knot_policy": "UNIFORM_NORMALIZED_SE3_ARC_INCLUDING_ENDPOINTS",
            "smoothing": False,
            "spline_fitting": False,
            "trajectory_optimization": False,
            "manual_knot_placement": False,
        },
        "thresholds": {
            "position_p95_m_strictly_less_than": DEFAULT_PREVIEW_POSITION_P95_LIMIT_M,
            "rotation_p95_rad_strictly_less_than": DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD,
            "endpoint_clamp_agreement_at_least": DEFAULT_ENDPOINT_CLAMP_AGREEMENT_MIN,
            "start_end_exact": True,
            "duration_exact": True,
        },
        "per_k": per_k,
        "k_capacity_star": k_star,
        "classification": classification,
        "status": "PASS" if k_star is not None else "PASS_NO_K_CAPACITY_STAR",
        "no_outcome_dependent_selection": True,
    }


def timing_ladder(
    payloads: Mapping[str, Mapping[str, Any]], geometry: Mapping[str, Any]
) -> Dict[str, Any]:
    k_star = geometry.get("k_capacity_star")
    result: Dict[str, Any] = {
        "schema": "STAGE37_T_TIMING_CAPACITY_QUALIFICATION_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "candidate_timing_modes": list(STAGE37_TIMING_MODES),
        "k_capacity_star": k_star,
        "per_mode": [],
        "t_capacity_star": None,
        "timing_compression_status": "NOT_EVALUATED",
        "status": "BLOCKED",
    }
    if k_star is None:
        result["status"] = "NOT_EVALUATED_K_CAPACITY_STAR_NULL"
        result["timing_compression_status"] = "NOT_EVALUATED_K_CAPACITY_STAR_NULL"
        return result
    if int(k_star) > 32:
        result["status"] = "NOT_EVALUATED_K_CAPACITY_STAR_GT_32"
        result["timing_compression_status"] = "NOT_EVALUATED_K_CAPACITY_STAR_GT_32"
        return result
    for mode in STAGE37_TIMING_MODES:
        rows: List[Dict[str, Any]] = []
        for case_id in CORE_CASE_IDS:
            dense_native, _, structured_native, _ = structured_materialization(
                payloads[case_id], int(k_star), mode
            )
            evaluation = evaluate_preview_equivalence(
                dense_native,
                structured_native,
                case_id=case_id,
                representation="K%d+%s_NATIVE_TRANSPORT" % (int(k_star), mode),
                anchor_count=101,
            )
            rows.append(evaluation)
        aggregate = aggregate_evaluations(
            rows, representation="K%d+%s_NATIVE_TRANSPORT" % (int(k_star), mode)
        )
        qualifies = bool(all(row.get("passes_limits") for row in rows) and aggregate.get("passes_limits"))
        result["per_mode"].append(
            {
                "timing_mode": mode,
                "qualifies_strict_all_cases_and_aggregate": qualifies,
                "aggregate": public_evaluation(aggregate),
                "per_case": [public_evaluation(row) for row in rows],
            }
        )
        if qualifies and result["t_capacity_star"] is None:
            result["t_capacity_star"] = mode
    result["timing_compression_status"] = (
        "TIMING_REPRESENTATION_QUALIFIED"
        if result["t_capacity_star"] is not None
        else "TIMING_REPRESENTATION_INSUFFICIENT"
    )
    result["status"] = "PASS" if result["t_capacity_star"] is not None else "PASS_NO_QUALIFIED_T"
    result["no_outcome_dependent_selection"] = True
    return result


def source_identity() -> Dict[str, Any]:
    runner = REPO / "scripts/native_wbc_benchmark.py"
    wrapper = REPO / "hpc/pbs/stage37_structured_wbc.pbs"
    return {
        "evaluation_runner": {
            "path": str(runner),
            "size_bytes": runner.stat().st_size if runner.is_file() else None,
            "sha256": sha256_file(runner) if runner.is_file() else None,
        },
        "campaign_runner": {
            "path": str(REPO / "scripts/stage36_wbc_oracle_campaign.py"),
            "size_bytes": (REPO / "scripts/stage36_wbc_oracle_campaign.py").stat().st_size,
            "sha256": sha256_file(REPO / "scripts/stage36_wbc_oracle_campaign.py"),
        },
        "pbs_wrapper": {
            "path": str(wrapper),
            "size_bytes": wrapper.stat().st_size if wrapper.is_file() else None,
            "sha256": sha256_file(wrapper) if wrapper.is_file() else None,
        },
    }


def make_campaign_inputs(
    payloads: Mapping[str, Mapping[str, Any]], output_root: Path, k_star: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inputs_root = output_root / "campaign_inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    transport_records: List[Dict[str, Any]] = []
    for case_id in CASE_IDS:
        payload = payloads[case_id]
        dense_native, geometry, structured_native, transport_audit = structured_materialization(
            payload, k_star, None
        )
        transport_eval = evaluate_preview_equivalence(
            dense_native,
            structured_native,
            case_id=case_id,
            representation="K%d+T0_NATIVE_TRANSPORT" % k_star,
            anchor_count=101,
        )
        if not transport_eval.get("passes_limits"):
            raise RuntimeError(
                "structured T0 transport failed for %s" % case_id
            )
        stem = "k%d_t0_%s_seed13_primary" % (k_star, case_id)
        trajectory_path = inputs_root / (stem + ".npz")
        metadata_path = inputs_root / (stem + ".json")
        np.savez_compressed(
            str(trajectory_path),
            timestamp=np.asarray(structured_native.point_timestamps, dtype=np.float64),
            position=np.asarray(structured_native.eef_position, dtype=np.float64),
            quaternion=np.asarray(structured_native.eef_orientation, dtype=np.float64),
        )
        metadata = {
            "schema": "STAGE37_STRUCTURED_ORACLE_CASE_METADATA_V1",
            "pi_gate": PI_GATE_ID,
            "case_id": case_id,
            "family": payload["family"],
            "representation": "K%d+T0" % k_star,
            "geometry_k": k_star,
            "timing_mode": "T0",
            "phase": "PRIMARY",
            "seed": 13,
            "reference_mode": "BENCHMARK_CONTINUOUS",
            "seconds": NATIVE_DURATION_S,
            "source_case_path": str(STAGE35_ROOT / "corpus/cases" / (case_id + ".json")),
            "source_case_sha256": sha256_file(
                STAGE35_ROOT / "corpus/cases" / (case_id + ".json")
            ),
            "geometry_source": "DENSE_NATIVE_TRANSPORT",
            "semantic_source": {
                "sample_count": int(geometry.point_count),
                "start_s": float(geometry.point_timestamps[0]),
                "end_s": float(geometry.point_timestamps[-1]),
                "geometry_arc_fraction_count": int(geometry.point_count),
            },
            "native_transport": transport_audit,
            "transport_preview_equivalence": public_evaluation(transport_eval),
            "support_witness_used_by_wbc": False,
            "dense_reference_rerun": False,
            "wbc_calls_in_cpu_ladder": False,
        }
        write_json(metadata_path, metadata)
        row_id = "%s_%s_seed13_primary" % ("k%d_t0" % k_star, case_id)
        rows.append(
            {
                "row_id": row_id,
                "case_id": case_id,
                "family": payload["family"],
                "representation": "K%d+T0" % k_star,
                "geometry_k": k_star,
                "timing_mode": "T0",
                "phase": "PRIMARY",
                "seed": 13,
                "reference_mode": "BENCHMARK_CONTINUOUS",
                "seconds": NATIVE_DURATION_S,
                "trajectory_path": str(trajectory_path),
                "trajectory_sha256": sha256_file(trajectory_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "native_transport": transport_audit,
            }
        )
        transport_records.append(
            {
                "case_id": case_id,
                "transport": transport_audit,
                "preview_equivalence": public_evaluation(transport_eval),
            }
        )
    return rows, {
        "schema": "STAGE37_STRUCTURED_T0_TRANSPORT_AUDIT_V1",
        "representation": "K%d+T0" % k_star,
        "case_count": len(transport_records),
        "all_pass": all(
            record["preview_equivalence"].get("passes_limits") for record in transport_records
        ),
        "records": transport_records,
    }


def make_campaign_plan(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    geometry: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> Dict[str, Any]:
    k_star = int(geometry["k_capacity_star"])
    t_star = timing.get("t_capacity_star")
    plan = {
        "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V3",
        "campaign_kind": "STAGE37_STRUCTURED_ORACLE",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "execution_head": git_output("rev-parse", "HEAD"),
        "plan_status": "READY_FOR_E",
        "project": EXPECTED_PROJECT,
        "queue": EXPECTED_QUEUE,
        "one_pbs_allocation_only": True,
        "row_count": len(rows),
        "primary_row_count": sum(row.get("phase") == "PRIMARY" for row in rows),
        "robustness_row_count": sum(row.get("phase") == "ROBUSTNESS" for row in rows),
        "sparse_rows": sum(row.get("representation") != "DENSE" for row in rows),
        "representations": ["K%d+T0" % k_star],
        "k_star": k_star,
        "t_star": t_star,
        "stage37_k_capacity_star": k_star,
        "stage37_t_capacity_star": t_star,
        "native_transport": {
            "transport": "TIMESTAMP_PRESERVING_UNIFORM_NATIVE_GRID_V1",
            "sample_count": NATIVE_SAMPLE_COUNT,
            "duration_s": NATIVE_DURATION_S,
            "dt_s": NATIVE_DT_S,
            "grid_formula": "tau_j=2.0*j/120, j=0..120",
            "shared_across_representations": True,
        },
        "stage37_scope": {
            "dense_stage36_rows_rerun": False,
            "structured_rows": 32,
            "core24_rows": 24,
            "continuous8_rows": 8,
            "geometry_only_t0": True,
            "timing_candidate_added": False,
            "gr00t_model": False,
            "training": False,
            "hardware": False,
        },
        "source_identity": source_identity(),
        "native_runner_path": str(REPO / "scripts/native_wbc_benchmark.py"),
        "native_runner_sha256": source_identity()["evaluation_runner"]["sha256"],
        "pbs_wrapper_path": str(REPO / "hpc/pbs/stage37_structured_wbc.pbs"),
        "pbs_wrapper_sha256": source_identity()["pbs_wrapper"]["sha256"],
        "authority_component_hashes": dict(
            getattr(STAGE36, "AUTHORITY_EXPECTED", {})
        ),
        "stage36_r_input_root": str(STAGE36_R_ROOT),
        "stage36_r_result_sha256": STAGE36_R_RESULT_SHA256,
        "stage36_r_seal_sha256": STAGE36_R_SEAL_SHA256,
        "stage35_root": str(STAGE35_ROOT),
        "stage35_manifest_sha256": STAGE35_MANIFEST_SHA256,
        "stage35_corpus_tree_sha256": STAGE35_CORPUS_TREE_SHA256,
        "authority": {
            "archive_sha256": getattr(STAGE36, "ARCHIVE_SHA256", None),
            "extracted_tree_sha256": getattr(STAGE36, "EXTRACTED_TREE_SHA256", None),
            "authority_expected": getattr(STAGE36, "AUTHORITY_EXPECTED", {}),
        },
        "rows": [dict(row) for row in rows],
        "no_dense_rows": True,
        "no_free_running_sampler": True,
        "no_new_seed_or_t": True,
    }
    errors = CAMPAIGN.validate_plan(plan, REPO)
    plan["validator_errors"] = errors
    plan["validator_status"] = "PASS" if not errors else "BLOCKED"
    if errors:
        plan["plan_status"] = "BLOCKED"
    return plan


def static_guard() -> Dict[str, Any]:
    campaign_path = REPO / "scripts/stage36_wbc_oracle_campaign.py"
    native_path = REPO / "scripts/native_wbc_benchmark.py"
    pbs_path = REPO / "hpc/pbs/stage37_structured_wbc.pbs"
    campaign = campaign_path.read_text(encoding="utf-8") if campaign_path.is_file() else ""
    native = native_path.read_text(encoding="utf-8") if native_path.is_file() else ""
    pbs = pbs_path.read_text(encoding="utf-8") if pbs_path.is_file() else ""
    forbidden = (
        "--train",
        "backward(",
        "optimizer.step(",
        "scheduler.step(",
        "resume_from_checkpoint",
        "--no-stop-at-reference-completion",
        "--disable-trajectory-termination",
        "--reference-mode NATIVE_GENERATOR",
    )
    command_forbidden_absent = all(token not in campaign for token in forbidden)
    cpu_source = Path(__file__).read_text(encoding="utf-8")
    cpu_forbidden_tokens = ("import " + "torch", "import " + "isaacgym", "q" + "sub")
    no_model_in_cpu = all(token not in cpu_source for token in cpu_forbidden_tokens)
    queue_only = "#PBS -q auto" in pbs and all(
        value not in pbs
        for value in ("#PBS -q small", "#PBS -q medium", "#PBS -q large", "#PBS -q special")
    )
    return {
        "schema": "STAGE37_STATIC_EXECUTION_GUARD_V1",
        "campaign_path": str(campaign_path),
        "campaign_sha256": sha256_file(campaign_path) if campaign_path.is_file() else None,
        "native_runner_path": str(native_path),
        "native_runner_sha256": sha256_file(native_path) if native_path.is_file() else None,
        "pbs_wrapper_path": str(pbs_path),
        "pbs_wrapper_sha256": sha256_file(pbs_path) if pbs_path.is_file() else None,
        "campaign_exists": campaign_path.is_file(),
        "native_runner_exists": native_path.is_file(),
        "pbs_exists": pbs_path.is_file(),
        "forbidden_execution_tokens_absent": command_forbidden_absent,
        "cpu_ladder_has_no_model_or_scheduler_path": no_model_in_cpu,
        "queue_auto_only": queue_only,
        "native_transport_path_present": "materialize_timestamped_pose_to_native" in native,
        "status": (
            "PASS"
            if campaign_path.is_file()
            and native_path.is_file()
            and pbs_path.is_file()
            and command_forbidden_absent
            and no_model_in_cpu
            and queue_only
            else "BLOCKED"
        ),
    }


def qstat_snapshot() -> Dict[str, Any]:
    code, stdout, stderr = run_capture(["qstat", "-u", os.environ.get("USER", "e1192548")])
    stage37_lines = [
        line for line in stdout.splitlines() if "stage37" in line.lower()
    ]
    return {
        "command": "qstat -u %s" % os.environ.get("USER", "e1192548"),
        "returncode": code,
        "stdout": stdout,
        "stderr": stderr,
        "stage37_lines": stage37_lines,
        "no_active_stage37_job": code == 0 and not stage37_lines,
        "status": "PASS" if code == 0 and not stage37_lines else "BLOCKED",
    }


def preflight(
    expected_head: str,
    api_head: str,
    stage36_inputs: Mapping[str, Any],
    stage35: Mapping[str, Any],
    g0: Mapping[str, Any],
    geometry: Mapping[str, Any],
    timing: Mapping[str, Any],
    guard: Mapping[str, Any],
    qstat: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    local_head = git_output("rev-parse", "HEAD")
    remote_head = git_output("rev-parse", "origin/" + EXPECTED_BRANCH)
    branch = git_output("branch", "--show-current")
    status_output = git_output("status", "--porcelain", "--untracked-files=all")
    diff_code, _, diff_err = run_capture(["git", "diff", "--check"])
    head_matches = bool(
        local_head == expected_head
        and api_head == expected_head
        and remote_head == expected_head
        and branch == EXPECTED_BRANCH
    )
    geometry_pass = geometry.get("status") == "PASS" and geometry.get("k_capacity_star") is not None
    timing_allowed = geometry.get("k_capacity_star") is not None and int(
        geometry["k_capacity_star"]
    ) <= 32
    plan_ready = bool(
        plan.get("plan_status") == "READY_FOR_E"
        and plan.get("validator_status") == "PASS"
        and plan.get("row_count") == 32
        and plan.get("representations") == [
            "K%d+T0" % int(geometry["k_capacity_star"])
        ]
    )
    predicates = {
        "stage36_r_input_authority_pass": stage36_inputs.get("status") == "PASS",
        "stage35_corpus_pass": stage35.get("status") == "PASS",
        "g0_frozen_input_pass": g0.get("status") == "PASS",
        "geometry_capacity_pass": geometry_pass,
        "timing_scope_allowed": timing_allowed,
        "static_guard_pass": guard.get("status") == "PASS",
        "qstat_no_active_stage37_job": qstat.get("no_active_stage37_job") is True,
        "head_api_remote_local_match": head_matches,
        "campaign_plan_head_binding": bool(
            plan.get("execution_head") == expected_head
            and plan.get("accepted_live_head") == expected_head
            and plan.get("api_head") == api_head
        ),
        "worktree_clean": status_output == "",
        "git_diff_check_pass": diff_code == 0,
        "campaign_plan_ready": plan_ready,
    }
    return {
        "schema": "STAGE37_CPU_ADMISSION_PREFLIGHT_V1",
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
        "geometry": {
            "k_capacity_star": geometry.get("k_capacity_star"),
            "classification": geometry.get("classification"),
        },
        "timing": {
            "t_capacity_star": timing.get("t_capacity_star"),
            "timing_compression_status": timing.get("timing_compression_status"),
        },
        "predicates": predicates,
        "all_predicates_pass": all(predicates.values()),
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
            "dense_stage36_rerun": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "pr_merge": False,
        },
    }


def seal_entries(output_root: Path, excluded: Iterable[str]) -> List[Dict[str, Any]]:
    excluded = set(excluded)
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            entries.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return entries


def build_return(
    output_root: Path,
    preflight_result: Mapping[str, Any],
    plan: Mapping[str, Any],
    seal_sha256: str,
    status: str,
    blockers: Sequence[str],
) -> Dict[str, Any]:
    geometry = plan.get("stage37_geometry", {})
    return {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": "Stage37-G0-G1-T",
        "PR": PR_NUMBER,
        "BRANCH": EXPECTED_BRANCH,
        "RESULT": status,
        "BLOCKERS": list(blockers),
        "NEXT_GATE": NEXT_GATE,
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(output_root / "stage37_cpu_artifact_seal.json"),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "K_CAPACITY_STAR": geometry.get("k_capacity_star"),
        "GEOMETRY_CAPACITY_CLASSIFICATION": geometry.get("classification"),
        "T_CAPACITY_STAR": plan.get("stage37_timing", {}).get("t_capacity_star"),
        "TIMING_COMPRESSION_STATUS": plan.get("stage37_timing", {}).get(
            "timing_compression_status"
        ),
        "DETAIL_OUTPUTS": {
            "stage36_r_input_audit": str(output_root / "stage37_stage36_r_input_audit.json"),
            "stage35_corpus_audit": str(output_root / "stage37_stage35_corpus_audit.json"),
            "g0_failure_map": str(output_root / "stage37_g0_failure_map.json"),
            "g1_geometry": str(output_root / "stage37_g1_geometry_qualification.json"),
            "timing": str(output_root / "stage37_timing_qualification.json"),
            "campaign_plan": str(output_root / "stage37_campaign_plan.json"),
            "preflight": str(output_root / "stage37_preflight.json"),
        },
        "PBS_SUBMITTED": False,
        "GPU_USED": False,
        "MODEL_CONSTRUCTED": False,
        "FORWARD": False,
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "HARDWARE": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
        "STAGE36_DENSE_RERUN": False,
        "PREFLIGHT_ALL_PREDICATES_PASS": preflight_result.get("all_predicates_pass"),
    }


def blocked_plan(geometry: Mapping[str, Any], timing: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V3",
        "campaign_kind": "STAGE37_STRUCTURED_ORACLE",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "plan_status": "BLOCKED",
        "project": EXPECTED_PROJECT,
        "queue": EXPECTED_QUEUE,
        "one_pbs_allocation_only": True,
        "row_count": 0,
        "primary_row_count": 0,
        "robustness_row_count": 0,
        "sparse_rows": 0,
        "representations": [],
        "k_star": geometry.get("k_capacity_star"),
        "t_star": timing.get("t_capacity_star"),
        "native_transport": {
            "sample_count": NATIVE_SAMPLE_COUNT,
            "duration_s": NATIVE_DURATION_S,
            "dt_s": NATIVE_DT_S,
            "shared_across_representations": True,
        },
        "rows": [],
        "validator_errors": ["structured_campaign_not_admitted"],
        "validator_status": "BLOCKED",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-head", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    output_root = args.output_root.absolute()
    if output_root.exists():
        raise SystemExit("FAIL_CLOSED output already exists: %s" % output_root)
    if args.expected_head != EXPECTED_BASE_HEAD and len(args.expected_head) != 40:
        raise SystemExit("FAIL_CLOSED expected head must be a full commit SHA")
    if not REPO.is_dir():
        raise SystemExit("FAIL_CLOSED repository missing")
    output_root.mkdir(parents=True)
    log_event(output_root, "START", branch=EXPECTED_BRANCH, expected_head=args.expected_head)

    stage36_inputs = verify_stage36_inputs()
    write_json(output_root / "stage37_stage36_r_input_audit.json", stage36_inputs)
    log_event(output_root, "STAGE36_R_INPUT_AUDIT", status=stage36_inputs.get("status"))

    stage35 = verify_stage35_corpus()
    write_json(output_root / "stage37_stage35_corpus_audit.json", stage35)
    log_event(output_root, "STAGE35_CORPUS_AUDIT", status=stage35.get("status"))

    g0_audit = verify_g0_input()
    write_json(output_root / "stage37_g0_input_audit.json", g0_audit)
    log_event(output_root, "G0_INPUT_AUDIT", status=g0_audit.get("status"))

    geometry: Dict[str, Any] = {"status": "BLOCKED", "k_capacity_star": None}
    timing: Dict[str, Any] = {
        "status": "BLOCKED_DEPENDENCY",
        "t_capacity_star": None,
        "timing_compression_status": "NOT_EVALUATED_DEPENDENCY",
    }
    payloads: Dict[str, Dict[str, Any]] = {}
    if (
        stage36_inputs.get("status") == "PASS"
        and stage35.get("status") == "PASS"
        and g0_audit.get("status") == "PASS"
    ):
        g0_map = make_g0_failure_map()
        write_json(output_root / "stage37_g0_failure_map.json", g0_map)
        payloads = load_payloads()
        geometry = geometry_ladder(payloads)
        write_json(output_root / "stage37_g1_geometry_qualification.json", geometry)
        log_event(
            output_root,
            "G1_GEOMETRY_LADDER",
            status=geometry.get("status"),
            k_capacity_star=geometry.get("k_capacity_star"),
        )
        timing = timing_ladder(payloads, geometry)
        write_json(output_root / "stage37_timing_qualification.json", timing)
        log_event(
            output_root,
            "T_TIMING_LADDER",
            status=timing.get("status"),
            t_capacity_star=timing.get("t_capacity_star"),
        )
    else:
        write_json(
            output_root / "stage37_g0_failure_map.json",
            {"schema": "STAGE37_G0_FROZEN_GEOMETRY_FAILURE_MAP_V1", "status": "BLOCKED"},
        )
        write_json(output_root / "stage37_g1_geometry_qualification.json", geometry)
        write_json(output_root / "stage37_timing_qualification.json", timing)

    guard = static_guard()
    write_json(output_root / "stage37_static_execution_guard.json", guard)
    qstat = qstat_snapshot()
    write_json(output_root / "stage37_qstat_snapshot.json", qstat)

    campaign_plan: Dict[str, Any]
    transport_audit: Dict[str, Any] = {
        "schema": "STAGE37_STRUCTURED_T0_TRANSPORT_AUDIT_V1",
        "status": "NOT_CREATED",
    }
    if (
        geometry.get("status") == "PASS"
        and geometry.get("k_capacity_star") is not None
        and int(geometry["k_capacity_star"]) <= 32
        and payloads
    ):
        rows, transport_audit = make_campaign_inputs(
            payloads, output_root, int(geometry["k_capacity_star"])
        )
        write_json(output_root / "stage37_structured_t0_transport_audit.json", transport_audit)
        campaign_plan = make_campaign_plan(rows, output_root, geometry, timing)
    else:
        campaign_plan = blocked_plan(geometry, timing)
        write_json(output_root / "stage37_structured_t0_transport_audit.json", transport_audit)
    campaign_plan["accepted_live_head"] = args.expected_head
    campaign_plan["api_head"] = args.api_head
    campaign_plan["stage37_geometry"] = geometry
    campaign_plan["stage37_timing"] = timing
    campaign_plan["stage37_transport_audit"] = transport_audit
    write_json(output_root / "stage37_campaign_plan.json", campaign_plan)
    log_event(
        output_root,
        "CAMPAIGN_PLAN",
        status=campaign_plan.get("plan_status"),
        validator_status=campaign_plan.get("validator_status"),
        row_count=campaign_plan.get("row_count"),
    )

    preflight_result = preflight(
        args.expected_head,
        args.api_head,
        stage36_inputs,
        stage35,
        g0_audit,
        geometry,
        timing,
        guard,
        qstat,
        campaign_plan,
    )
    write_json(output_root / "stage37_preflight.json", preflight_result)
    log_event(
        output_root,
        "PREFLIGHT",
        all_predicates_pass=preflight_result.get("all_predicates_pass"),
    )

    result_payload = {
        "schema": "STAGE37_GEOMETRY_CAPACITY_CPU_RESULT_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "stage": "Stage37-G0-G1-T",
        "stage36_r_core_lifecycle": stage36_inputs.get("stage36_r_core24_pass"),
        "stage36_r_continuous8_lifecycle": stage36_inputs.get("stage36_r_continuous8_pass"),
        "g0": {
            "status": g0_audit.get("status"),
            "source_sha256": G0_RESULT_SHA256,
            "no_rerun": True,
        },
        "g1": geometry,
        "timing": timing,
        "structured_t0_transport": transport_audit,
        "campaign_plan_path": str(output_root / "stage37_campaign_plan.json"),
        "preflight_path": str(output_root / "stage37_preflight.json"),
        "execution_boundary": preflight_result.get("execution_boundary"),
        "next_gate": NEXT_GATE,
    }
    write_json(output_root / "stage37_cpu_result.json", result_payload)
    blockers = [
        name for name, passed in preflight_result.get("predicates", {}).items() if not passed
    ]
    status = (
        "STAGE37_CPU_ADMISSION_PASS_READY_FOR_E"
        if preflight_result.get("all_predicates_pass")
        else "BLOCKED"
    )
    log_event(output_root, "RETURN", result=status, blockers=blockers)
    seal_payload = {
        "schema": "STAGE37_CPU_ARTIFACT_SEAL_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "stage": "Stage37-G0-G1-T",
        "output_root": str(output_root),
        "entries": seal_entries(
            output_root, ("stage37_cpu_artifact_seal.json", "stage37_return.json")
        ),
        "source_inputs": {
            "stage36_r_result_sha256": STAGE36_R_RESULT_SHA256,
            "stage36_r_seal_sha256": STAGE36_R_SEAL_SHA256,
            "stage35_manifest_sha256": STAGE35_MANIFEST_SHA256,
            "stage35_corpus_tree_sha256": STAGE35_CORPUS_TREE_SHA256,
            "g0_result_sha256": G0_RESULT_SHA256,
            "g0_seal_sha256": G0_SEAL_SHA256,
        },
        "historical_inputs_mutated": False,
        "stage36_dense_rerun": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "hardware": False,
    }
    seal_path = output_root / "stage37_cpu_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha = sha256_file(seal_path)
    return_payload = build_return(
        output_root, preflight_result, campaign_plan, seal_sha, status, blockers
    )
    write_json(output_root / "stage37_return.json", return_payload)
    return 0 if status == "STAGE37_CPU_ADMISSION_PASS_READY_FOR_E" else 2


if __name__ == "__main__":
    raise SystemExit(main())
