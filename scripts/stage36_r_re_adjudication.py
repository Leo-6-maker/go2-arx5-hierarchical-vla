#!/usr/bin/env python3
"""Re-adjudicate sealed Stage36 WBC evidence without rerunning WBC.

This tool is deliberately an evidence reader, not a simulator launcher.  It
verifies the accepted Stage36 seal and every raw/summary member referenced by
the sealed manifests before reading any row.  The resulting correction layer
is written to a new scratch root; the original Stage36 root and all inherited
row roots remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_eval_602744.hopper-m-02"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_r_re_adjudication_20260906_a1a07a3_v1"
)

PI_GATE_ID = "PI_GATE_STATE_0025"
STAGE = "Stage36-R"
PR_NUMBER = 22
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_ORIGINAL_SEAL_SHA256 = (
    "52f8cf662f9a0ce5ca0a0ec65d9e44cb8199382765610d32e71bc2fb64e0b77b"
)
EXPECTED_ORIGINAL_RESULT_SHA256 = (
    "3dcf2b95af39f0c6069d540d385082812fff453eb6966d98b20aa42419d77ac1"
)
EXPECTED_ORIGINAL_RAW_MANIFEST_SHA256 = (
    "e6b13ce28c48eef1fe16704f50a80ba90920e6d28bc6adcd2b2c3594b5f0b236"
)
EXPECTED_ORIGINAL_ROW_COUNT = 32
EXPECTED_FAMILIES = ("N", "T", "P", "C")
EXPECTED_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in EXPECTED_FAMILIES
    for index in range(1, 9)
)
EXPECTED_ROW_IDS = tuple(
    "dense_%s_seed13_primary" % case_id for case_id in EXPECTED_CASE_IDS
)
EXPECTED_URDF = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_authority_v2_20260906_clean_v1/WBCTrackingShare/"
    "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"
)
EXPECTED_URDF_SHA256 = (
    "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25"
)
FINALIZED_NATIVE_STATUSES = {
    "NATIVE_BENCHMARK_CONTINUOUS_PASS",
    "NATIVE_BENCHMARK_CONTINUOUS_PARTIAL",
}
TRUE_TERMINATION_PRECEDENCE = (
    "INVALID_STATE",
    "JOINT_LIMIT",
    "FALL",
    "TIME_LIMIT",
    "TRACKING_ABORT",
    "NATIVE_TASK_TERMINATION",
)
LIFECYCLE_POLICY = {
    "hard_safety_failure": (
        "fall OR invalid_state/non_finite OR timeout OR "
        "tracking_abort_before_reference_completion OR "
        "native_task_termination_before_reference_completion OR "
        "joint_limit/joint_limit_termination"
    ),
    "joint_limit_violation": "diagnostic_only",
    "tracking_profile": "descriptive_only_no_posthoc_absolute_threshold",
    "authority": "src/go2arx5_vla/benchmark/lifecycle.py",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_nonfinite_json(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite_json,
    )


def finite_value(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_value(item) for item in value)
    if isinstance(value, dict):
        return all(finite_value(item) for item in value.values())
    return True


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def append_event(path: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize(payload), sort_keys=True, allow_nan=False) + "\n")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO)] + list(args), text=True
    ).strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("FAIL_CLOSED %s" % message)


def _verify_file(path: Path, expected_sha256: str, expected_size: Optional[int] = None) -> Dict[str, Any]:
    record = {
        "path": str(path),
        "exists": path.is_file(),
        "is_symlink": path.is_symlink(),
        "expected_sha256": expected_sha256,
        "expected_size_bytes": expected_size,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else None,
    }
    record["status"] = (
        "PASS"
        if record["exists"]
        and not record["is_symlink"]
        and record["sha256"] == expected_sha256
        and (expected_size is None or record["size_bytes"] == expected_size)
        else "MISMATCH"
    )
    _require(record["status"] == "PASS", "sealed file mismatch: %s" % path)
    return record


def verify_original_evidence(original_root: Path) -> Dict[str, Any]:
    """Verify the entire accepted seal before reading row payloads."""

    _require(original_root.is_dir() and not original_root.is_symlink(), "original root is not a real directory")
    seal_path = original_root / "stage36_artifact_seal.json"
    result_path = original_root / "stage36_result.json"
    raw_manifest_path = original_root / "stage36_raw_hash_manifest.json"
    result_manifest_path = original_root / "stage36_result_manifest.json"
    for path in (seal_path, result_path, raw_manifest_path, result_manifest_path):
        _require(path.is_file() and not path.is_symlink(), "required original file missing: %s" % path)

    _require(sha256_file(seal_path) == EXPECTED_ORIGINAL_SEAL_SHA256, "original artifact seal SHA mismatch")
    _require(sha256_file(result_path) == EXPECTED_ORIGINAL_RESULT_SHA256, "original result SHA mismatch")
    _require(
        sha256_file(raw_manifest_path) == EXPECTED_ORIGINAL_RAW_MANIFEST_SHA256,
        "original raw manifest SHA mismatch",
    )
    seal = load_json(seal_path)
    result = load_json(result_path)
    raw_manifest = load_json(raw_manifest_path)
    result_manifest = load_json(result_manifest_path)
    _require(seal.get("schema") == "STAGE36_ARTIFACT_SEAL_V2", "unexpected original seal schema")
    _require(result.get("schema") == "STAGE36_WBC_ORACLE_RESULT_V2", "unexpected original result schema")
    _require(raw_manifest.get("schema") == "STAGE36_RAW_HASH_MANIFEST_V1", "unexpected raw manifest schema")
    _require(result_manifest.get("result_sha256") == EXPECTED_ORIGINAL_RESULT_SHA256, "result manifest result SHA mismatch")
    _require(
        result_manifest.get("raw_hash_manifest_sha256") == EXPECTED_ORIGINAL_RAW_MANIFEST_SHA256,
        "result manifest raw manifest SHA mismatch",
    )
    _require(result_manifest.get("row_count") == EXPECTED_ORIGINAL_ROW_COUNT, "result manifest row count mismatch")

    sealed_entries: Dict[str, Dict[str, Any]] = {}
    all_seal_entries = list(seal.get("entries", [])) + list(seal.get("inherited_entries", []))
    _require(len(all_seal_entries) > 0, "original seal has no entries")
    entry_checks: List[Dict[str, Any]] = []
    for entry in all_seal_entries:
        path = Path(str(entry.get("path", "")))
        _require(str(path) not in sealed_entries, "duplicate seal entry: %s" % path)
        check = _verify_file(path, str(entry.get("sha256")), entry.get("size_bytes"))
        entry_checks.append(check)
        sealed_entries[str(path)] = entry

    result_rows = result.get("rows")
    manifest_rows = raw_manifest.get("rows")
    _require(isinstance(result_rows, list), "original result rows are not a list")
    _require(isinstance(manifest_rows, list), "original raw manifest rows are not a list")
    result_ids = [str(row.get("row_id")) for row in result_rows]
    manifest_ids = [str(row.get("row_id")) for row in manifest_rows]
    _require(len(result_rows) == EXPECTED_ORIGINAL_ROW_COUNT, "original result row count mismatch")
    _require(len(manifest_rows) == EXPECTED_ORIGINAL_ROW_COUNT, "original raw manifest row count mismatch")
    _require(len(set(result_ids)) == EXPECTED_ORIGINAL_ROW_COUNT, "duplicate result row IDs")
    _require(len(set(manifest_ids)) == EXPECTED_ORIGINAL_ROW_COUNT, "duplicate raw manifest row IDs")
    _require(set(result_ids) == set(EXPECTED_ROW_IDS), "result row ID set mismatch")
    _require(set(manifest_ids) == set(EXPECTED_ROW_IDS), "raw manifest row ID set mismatch")

    plan_path = original_root / "stage36_campaign_plan_snapshot.json"
    if plan_path.is_file() and not plan_path.is_symlink():
        plan = load_json(plan_path)
        plan_ids = [str(row.get("row_id")) for row in plan.get("rows", [])]
        _require(set(plan_ids) == set(EXPECTED_ROW_IDS), "planned row ID set mismatch")

    result_by_id = {str(row.get("row_id")): row for row in result_rows}
    manifest_by_id = {str(row.get("row_id")): row for row in manifest_rows}
    row_checks: List[Dict[str, Any]] = []
    for row_id in EXPECTED_ROW_IDS:
        result_row = result_by_id[row_id]
        manifest_row = manifest_by_id[row_id]
        raw_path = Path(str(manifest_row.get("raw_rows_path", "")))
        summary_path = Path(str(manifest_row.get("summary_path", "")))
        _require(result_row.get("raw_rows_path") == str(raw_path), "result/raw path mismatch: %s" % row_id)
        _require(str(raw_path) in sealed_entries, "raw path absent from original seal: %s" % row_id)
        _require(str(summary_path) in sealed_entries, "summary path absent from original seal: %s" % row_id)
        raw_check = _verify_file(raw_path, str(manifest_row.get("raw_rows_sha256")))
        summary_check = _verify_file(summary_path, str(manifest_row.get("summary_sha256")))
        _require(sealed_entries[str(raw_path)].get("sha256") == raw_check["sha256"], "sealed raw hash mismatch: %s" % row_id)
        _require(sealed_entries[str(summary_path)].get("sha256") == summary_check["sha256"], "sealed summary hash mismatch: %s" % row_id)
        row_checks.append(
            {
                "row_id": row_id,
                "raw_rows_path": str(raw_path),
                "raw_rows_sha256": raw_check["sha256"],
                "summary_path": str(summary_path),
                "summary_sha256": summary_check["sha256"],
                "raw_seal_entry_sha256": sealed_entries[str(raw_path)].get("sha256"),
                "summary_seal_entry_sha256": sealed_entries[str(summary_path)].get("sha256"),
            }
        )

    return {
        "status": "PASS",
        "original_root": str(original_root),
        "original_seal_path": str(seal_path),
        "original_seal_sha256": sha256_file(seal_path),
        "original_result_path": str(result_path),
        "original_result_sha256": sha256_file(result_path),
        "original_raw_manifest_path": str(raw_manifest_path),
        "original_raw_manifest_sha256": sha256_file(raw_manifest_path),
        "original_result_manifest_path": str(result_manifest_path),
        "original_result_manifest_sha256": sha256_file(result_manifest_path),
        "original_seal_entry_count": len(all_seal_entries),
        "original_seal_entry_checks": entry_checks,
        "row_checks": row_checks,
        "original_result": result,
        "original_raw_manifest": raw_manifest,
        "original_seal": seal,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=reject_nonfinite_json)
            except Exception as exc:
                raise RuntimeError("invalid JSONL at %s:%d: %s" % (path, line_number, exc))
            _require(isinstance(value, dict), "raw row is not an object: %s:%d" % (path, line_number))
            rows.append(value)
    return rows


def flag(row: Mapping[str, Any], name: str) -> bool:
    value = row.get(name, False)
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def hard_flags_independent(row: Mapping[str, Any]) -> Dict[str, bool]:
    """Independent mirror of the frozen lifecycle policy for the report."""

    code = str(row.get("termination_code", "UNKNOWN"))
    complete = flag(row, "reference_complete")
    return {
        "fall": flag(row, "fall_flag") or flag(row, "fall") or flag(row, "body_height_buf") or code == "FALL",
        "invalid_state": (
            flag(row, "invalid_state")
            or flag(row, "non_finite_state")
            or not finite_value(dict(row))
            or code == "INVALID_STATE"
        ),
        "timeout": flag(row, "time_out_buf") or flag(row, "timeout") or code in ("TIME_LIMIT", "TIMEOUT"),
        "tracking_abort_before_reference_completion": (
            (flag(row, "traj_early_term") or flag(row, "tracking_abort") or code == "TRACKING_ABORT")
            and not complete
        ),
        "native_task_termination_before_reference_completion": (
            (flag(row, "native_task_termination") or flag(row, "reset_buf") or code == "NATIVE_TASK_TERMINATION")
            and not complete
        ),
        "joint_limit_termination": (
            flag(row, "joint_limit")
            or flag(row, "joint_limit_termination")
            or code == "JOINT_LIMIT"
        ),
    }


def true_termination_code(hard_by_sample: Sequence[Mapping[str, bool]], complete: bool) -> str:
    for name in TRUE_TERMINATION_PRECEDENCE:
        key = {
            "INVALID_STATE": "invalid_state",
            "JOINT_LIMIT": "joint_limit_termination",
            "FALL": "fall",
            "TIME_LIMIT": "timeout",
            "TRACKING_ABORT": "tracking_abort_before_reference_completion",
            "NATIVE_TASK_TERMINATION": "native_task_termination_before_reference_completion",
        }[name]
        if any(bool(flags.get(key)) for flags in hard_by_sample):
            return name
    return "REFERENCE_COMPLETE" if complete else "UNKNOWN"


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def base_participation(raw_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positions: List[Tuple[float, float, float]] = []
    for row in raw_rows:
        value = row.get("base_pos")
        if not isinstance(value, list) or len(value) < 3:
            return {"status": "UNAVAILABLE", "reason": "base_pos_missing_or_short"}
        coords = [number(item) for item in value[:3]]
        if any(item is None for item in coords):
            return {"status": "UNAVAILABLE", "reason": "base_pos_nonfinite"}
        positions.append((coords[0], coords[1], coords[2]))  # type: ignore[index]
    if not positions:
        return {"status": "UNAVAILABLE", "reason": "no_raw_rows"}
    x_values = [item[0] for item in positions]
    displacement = math.sqrt(sum((positions[-1][i] - positions[0][i]) ** 2 for i in range(3)))
    path_length = sum(
        math.sqrt(sum((positions[index][axis] - positions[index - 1][axis]) ** 2 for axis in range(3)))
        for index in range(1, len(positions))
    )
    x_span = max(x_values) - min(x_values)
    return {
        "status": "PASS",
        "finite": True,
        "sample_count": len(positions),
        "x_span_m": x_span,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "nontrivial_observed": len(positions) > 1 and x_span > 1e-3,
    }


def metric_value(metrics: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return number(value)


def metric_means(records: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    paths = {
        "position_mean_m": ("position_m", "mean"),
        "position_rmse_m": ("position_m", "rmse"),
        "position_p95_m": ("position_m", "p95"),
        "position_max_m": ("position_m", "max"),
        "rotation_mean_rad": ("rotation_rad", "mean"),
        "rotation_rmse_rad": ("rotation_rad", "rmse"),
        "rotation_p95_rad": ("rotation_rad", "p95"),
        "rotation_max_rad": ("rotation_rad", "max"),
        "time_in_tolerance_fraction_0.05m_0.20rad": (
            "time_in_tolerance_fraction_0.05m_0.20rad",
        ),
        "base_displacement_m": ("base_displacement_m",),
        "base_path_length_m": ("base_path_length_m",),
        "path_completion": ("path_completion",),
    }
    output: Dict[str, Optional[float]] = {}
    for name, path in paths.items():
        values = [
            value
            for record in records
            for value in [metric_value(record.get("profile_metrics", {}), path)]
            if value is not None
        ]
        output[name] = None if not values else sum(values) / len(values)
    return output


def parse_urdf_joints(path: Path) -> Dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "exact URDF is missing")
    _require(sha256_file(path) == EXPECTED_URDF_SHA256, "exact URDF hash mismatch")
    root = ET.parse(str(path)).getroot()
    joints: List[Dict[str, Any]] = []
    for joint in root.findall(".//joint"):
        joint_type = str(joint.get("type", ""))
        if joint_type == "fixed":
            continue
        limit_element = joint.find("limit")
        limit: Dict[str, Any] = {}
        if limit_element is not None:
            for key in ("lower", "upper", "effort", "velocity"):
                if limit_element.get(key) is not None:
                    parsed = number(float(limit_element.get(key)))
                    limit[key] = parsed
        joints.append(
            {
                "index": len(joints),
                "name": joint.get("name"),
                "type": joint_type,
                "limit": limit,
            }
        )
    _require(len(joints) == 20, "exact URDF non-fixed joint count is not 20")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "non_fixed_joint_count": len(joints),
        "joints": joints,
    }


def joint_forensics(
    row_payloads: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
    urdf: Mapping[str, Any],
) -> Dict[str, Any]:
    samples = [
        (row_id, sample_index, row)
        for row_id, rows in row_payloads
        for sample_index, row in enumerate(rows)
    ]
    joints = urdf["joints"]
    reports: List[Dict[str, Any]] = []
    for index, joint in enumerate(joints):
        values: List[float] = []
        negative: List[Tuple[str, int, Mapping[str, Any]]] = []
        malformed_count = 0
        for row_id, sample_index, row in samples:
            margins = row.get("joint_limit_margin")
            if not isinstance(margins, list) or index >= len(margins):
                malformed_count += 1
                continue
            margin = number(margins[index])
            if margin is None:
                malformed_count += 1
                continue
            values.append(margin)
            if margin < 0.0:
                negative.append((row_id, sample_index, row))
        negative_by_family: Dict[str, Dict[str, Any]] = {}
        for family in EXPECTED_FAMILIES:
            family_negative = [item for item in negative if "_%s" % family in item[0]]
            family_rows = {item[0] for item in family_negative}
            negative_by_family[family] = {
                "negative_sample_count": len(family_negative),
                "sample_count": sum(1 for row_id, _, _ in samples if "_%s" % family in row_id),
                "negative_fraction": (
                    len(family_negative)
                    / float(sum(1 for row_id, _, _ in samples if "_%s" % family in row_id))
                    if any("_%s" % family in row_id for row_id, _, _ in samples)
                    else None
                ),
                "affected_row_count": len(family_rows),
                "affected_rows": sorted(family_rows),
            }
        first_negative = None
        if negative:
            first_negative = min(
                negative,
                key=lambda item: (
                    number(item[2].get("sim_time"))
                    if number(item[2].get("sim_time")) is not None
                    else float("inf"),
                    item[1],
                    item[0],
                ),
            )
        reset_negative_count = sum(1 for _, _, row in negative if flag(row, "reset_buf"))
        initial_negative_count = sum(1 for _, sample_index, _ in negative if sample_index == 0)
        true_termination_count = sum(
            1
            for _, _, row in samples
            if flag(row, "joint_limit_termination") or flag(row, "joint_limit")
        )
        reports.append(
            {
                "index": index,
                "name": joint.get("name"),
                "type": joint.get("type"),
                "limit": joint.get("limit", {}),
                "margin_units": "native_joint_coordinate",
                "sample_count": len(samples),
                "valid_margin_sample_count": len(values),
                "malformed_margin_sample_count": malformed_count,
                "minimum_signed_margin": min(values) if values else None,
                "maximum_signed_margin": max(values) if values else None,
                "mean_signed_margin": sum(values) / len(values) if values else None,
                "negative_margin_sample_count": len(negative),
                "negative_margin_fraction": len(negative) / float(len(samples)) if samples else None,
                "first_negative_onset": (
                    {
                        "row_id": first_negative[0],
                        "step": first_negative[1],
                        "sim_time": number(first_negative[2].get("sim_time")),
                    }
                    if first_negative is not None
                    else None
                ),
                "negative_at_initial_sample_count": initial_negative_count,
                "negative_at_reset_buf_sample_count": reset_negative_count,
                "reset_buf_true_sample_count": sum(flag(row, "reset_buf") for _, _, row in samples),
                "joint_limit_termination_sample_count": true_termination_count,
                "by_family": negative_by_family,
            }
        )
    return {
        "schema": "STAGE36_R_JOINT_MARGIN_FORENSICS_V1",
        "urdf": dict(urdf),
        "total_rows": len(row_payloads),
        "total_samples": len(samples),
        "joint_limit_violation_flag_sample_count": sum(
            flag(row, "joint_limit_violation") for _, _, row in samples
        ),
        "joint_limit_termination_flag_sample_count": sum(
            flag(row, "joint_limit_termination") for _, _, row in samples
        ),
        "joints": reports,
    }


def row_adjudication(
    source_row: Mapping[str, Any],
    summary: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    raw_finite = finite_value(list(raw_rows))
    summary_finite = finite_value(dict(summary))
    finite = raw_finite and summary_finite
    complete = bool(summary.get("reference_completed", False)) or any(
        flag(row, "reference_complete") for row in raw_rows
    )
    hard_by_sample = [hard_flags_independent(row) for row in raw_rows]
    hard_failure_flags = {
        name: any(bool(item.get(name)) for item in hard_by_sample)
        for name in (
            "fall",
            "invalid_state",
            "timeout",
            "tracking_abort_before_reference_completion",
            "native_task_termination_before_reference_completion",
            "joint_limit_termination",
        )
    }
    hard_failure = any(hard_failure_flags.values())
    status = str(summary.get("status", "UNKNOWN"))
    finalized = bool(source_row.get("scientific_row_finalized", False)) and bool(raw_rows) and status in FINALIZED_NATIVE_STATUSES
    runner_pass = bool(source_row.get("runner_pass", False)) and finite and finalized
    base = base_participation(raw_rows)
    diagnostic_count = sum(flag(row, "joint_limit_violation") for row in raw_rows)
    true_joint_count = sum(
        flag(row, "joint_limit_termination") or flag(row, "joint_limit")
        for row in raw_rows
    )
    termination_codes = Counter(str(row.get("termination_code", "UNKNOWN")) for row in raw_rows)
    first_hard = None
    for index, (row, flags) in enumerate(zip(raw_rows, hard_by_sample)):
        if any(flags.values()):
            first_hard = {
                "step": row.get("step", index),
                "sample_index": index,
                "sim_time": number(row.get("sim_time")),
                "flags": dict(flags),
                "termination_code": row.get("termination_code"),
            }
            break
    return {
        "row_id": source_row.get("row_id"),
        "case_id": source_row.get("case_id"),
        "family": source_row.get("family"),
        "representation": source_row.get("representation"),
        "geometry_k": source_row.get("geometry_k"),
        "timing_mode": source_row.get("timing_mode"),
        "phase": source_row.get("phase"),
        "seed": source_row.get("seed"),
        "returncode": source_row.get("returncode"),
        "timed_out": source_row.get("timed_out"),
        "native_status": status,
        "native_termination": summary.get("termination"),
        "raw_termination_codes": dict(sorted(termination_codes.items())),
        "raw_rows_count": len(raw_rows),
        "finite": finite,
        "finite_flags": {
            "summary": summary_finite,
            "raw_rows": raw_finite,
            "metrics": finite_value(summary.get("metrics", {})),
        },
        "reference_complete": complete,
        "scientific_row_finalized": finalized,
        "runner_pass_source": bool(source_row.get("runner_pass", False)),
        "runner_pass_recomputed": runner_pass,
        "hard_safety_failure": hard_failure,
        "hard_failure_flags": hard_failure_flags,
        "true_termination_code": true_termination_code(hard_by_sample, complete),
        "first_hard_failure": first_hard,
        "diagnostic_joint_margin_violation": diagnostic_count > 0,
        "diagnostic_joint_margin_violation_sample_count": diagnostic_count,
        "true_joint_limit_termination_sample_count": true_joint_count,
        "lifecycle_qualified": bool(finalized and finite and complete and not hard_failure),
        "profile_metrics": sanitize(summary.get("metrics", {})),
        "base_participation": base,
    }


def group_profile(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "row_count": len(records),
        "finite_count": sum(bool(record.get("finite")) for record in records),
        "finalized_count": sum(bool(record.get("scientific_row_finalized")) for record in records),
        "reference_complete_count": sum(bool(record.get("reference_complete")) for record in records),
        "lifecycle_qualified_count": sum(bool(record.get("lifecycle_qualified")) for record in records),
        "hard_failure_count": sum(bool(record.get("hard_safety_failure")) for record in records),
        "diagnostic_joint_margin_violation_row_count": sum(
            bool(record.get("diagnostic_joint_margin_violation")) for record in records
        ),
        "true_joint_limit_termination_row_count": sum(
            bool(record.get("true_joint_limit_termination_sample_count")) for record in records
        ),
        "profile_metric_means": metric_means(records),
        "base_participation_nontrivial_count": sum(
            bool(record.get("base_participation", {}).get("nontrivial_observed"))
            for record in records
        ),
    }


def source_artifact_records() -> List[Dict[str, Any]]:
    paths = [
        ("stage36_campaign_evaluator", REPO / "scripts/stage36_wbc_oracle_campaign.py"),
        ("lifecycle_authority", REPO / "src/go2arx5_vla/benchmark/lifecycle.py"),
        ("stage36_r_readjudicator", Path(__file__).resolve()),
    ]
    records: List[Dict[str, Any]] = []
    for role, path in paths:
        _require(path.is_file() and not path.is_symlink(), "source artifact missing: %s" % path)
        records.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def local_seal_entries(output_root: Path, excluded: Iterable[str]) -> List[Dict[str, Any]]:
    excluded_set = set(excluded)
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded_set:
            entries.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def prepare_output_root(output_root: Path) -> None:
    _require(not output_root.exists(), "readjudication output root already exists: %s" % output_root)
    _require(not output_root.is_symlink(), "readjudication output root is a symlink")
    output_root.mkdir(parents=True)


def build_report(original_root: Path, output_root: Path, verification: Mapping[str, Any]) -> Dict[str, Any]:
    original_result = verification["original_result"]
    raw_manifest = verification["original_raw_manifest"]
    result_by_id = {str(row.get("row_id")): row for row in original_result.get("rows", [])}
    manifest_by_id = {str(row.get("row_id")): row for row in raw_manifest.get("rows", [])}
    row_payloads: List[Tuple[str, List[Dict[str, Any]]]] = []
    records: List[Dict[str, Any]] = []
    raw_manifest_rows: List[Dict[str, Any]] = []
    for row_id in EXPECTED_ROW_IDS:
        manifest_row = manifest_by_id[row_id]
        raw_path = Path(str(manifest_row["raw_rows_path"]))
        summary_path = Path(str(manifest_row["summary_path"]))
        raw_rows = read_jsonl(raw_path)
        summary = load_json(summary_path)
        row_payloads.append((row_id, raw_rows))
        record = row_adjudication(result_by_id[row_id], summary, raw_rows)
        records.append(record)
        raw_manifest_rows.append(
            {
                "row_id": row_id,
                "raw_rows_path": str(raw_path),
                "raw_rows_sha256": sha256_file(raw_path),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "raw_rows_count": len(raw_rows),
            }
        )

    urdf = parse_urdf_joints(EXPECTED_URDF)
    forensics = joint_forensics(row_payloads, urdf)
    core_records = [record for record in records if record.get("family") in ("N", "T", "P")]
    continuous_records = [record for record in records if record.get("family") == "C"]
    core_pass = (
        len(core_records) == 24
        and all(
            bool(record.get("scientific_row_finalized"))
            and bool(record.get("finite"))
            and bool(record.get("reference_complete"))
            and not bool(record.get("hard_safety_failure"))
            for record in core_records
        )
    )
    continuous_pass = (
        len(continuous_records) == 8
        and all(
            bool(record.get("scientific_row_finalized"))
            and bool(record.get("finite"))
            and bool(record.get("reference_complete"))
            and not bool(record.get("hard_safety_failure"))
            and bool(record.get("base_participation", {}).get("nontrivial_observed"))
            for record in continuous_records
        )
    )
    core_hard_failure = any(bool(record.get("hard_safety_failure")) for record in core_records)
    if core_pass:
        top_level_outcome = "WBC_CORE_LIFECYCLE_QUALIFIED_ON_STAGE35_CORE24"
        next_gate = "GPT_PI_REVIEW_BEFORE_STRUCTURED_GR00T_DESIGN"
    elif core_hard_failure:
        top_level_outcome = "TRUE_WBC_CORE_LIFECYCLE_LIMITATION"
        next_gate = "GPT_PI_REVIEW_TRUE_WBC_CORE_LIMITATION"
    else:
        top_level_outcome = "DELIVERED_WBC_CORE_LIFECYCLE_LIMITATION"
        next_gate = "GPT_PI_REVIEW_TRUE_WBC_CORE_LIMITATION"

    if continuous_pass:
        continuous_classification = "WBC_CONTINUOUS_BASE_LIFECYCLE_QUALIFIED_ON_TESTED_ENVELOPE"
    elif any(bool(record.get("hard_safety_failure")) for record in continuous_records):
        continuous_classification = "WBC_CONTINUOUS_BASE_TRUE_LIFECYCLE_FAILURE"
    elif not all(bool(record.get("base_participation", {}).get("nontrivial_observed")) for record in continuous_records):
        continuous_classification = "WBC_CONTINUOUS_BASE_REFERENCE_COMPLETE_WITHOUT_REQUIRED_BASE_PARTICIPATION"
    else:
        continuous_classification = "WBC_CONTINUOUS_BASE_INCOMPLETE_OR_NONFINITE"

    old_tracking_pass_count = sum(bool(row.get("tracking_pass")) for row in original_result.get("rows", []))
    corrected_tracking_pass_count = sum(bool(record.get("lifecycle_qualified")) for record in records)
    family_summary = {
        family: group_profile([record for record in records if record.get("family") == family])
        for family in EXPECTED_FAMILIES
    }
    seed_summary = {
        str(seed): group_profile([record for record in records if record.get("seed") == seed])
        for seed in sorted({record.get("seed") for record in records})
    }
    result_payload = {
        "schema": "STAGE36_R_RE_ADJUDICATION_RESULT_V1",
        "timestamp_utc": now_utc(),
        "role": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "stage": STAGE,
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "project": EXPECTED_PROJECT,
        "execution_context": "CPU_ONLY_SEALED_EVIDENCE_RE_ADJUDICATION",
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "worktree_clean": not bool(git_output("status", "--porcelain")),
        },
        "server": {
            "hostname": platform.node(),
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
        "input_authority": {
            "original_root": str(original_root),
            "original_job_id": verification["original_seal"].get("pbs_job_id"),
            "original_artifact_seal_sha256": verification["original_seal_sha256"],
            "original_result_sha256": verification["original_result_sha256"],
            "original_raw_hash_manifest_sha256": verification["original_raw_manifest_sha256"],
            "original_evidence_verified_before_analysis": True,
            "original_seal_entry_count_verified": verification["original_seal_entry_count"],
            "row_count": len(records),
            "row_ids": [record.get("row_id") for record in records],
        },
        "source_identity": {
            "source_artifacts": source_artifact_records(),
            "original_stage36_evaluation_runner": next(
                (
                    item
                    for item in verification["original_seal"].get("source_artifacts", [])
                    if item.get("role") == "evaluation_runner"
                ),
                None,
            ),
            "original_stage36_pbs_wrapper": next(
                (
                    item
                    for item in verification["original_seal"].get("source_artifacts", [])
                    if item.get("role") == "pbs_wrapper"
                ),
                None,
            ),
        },
        "lifecycle_semantic_policy": LIFECYCLE_POLICY,
        "tracking_profile_status": "DESCRIPTIVE_ONLY_NO_POSTHOC_ABSOLUTE_THRESHOLD",
        "rows": records,
        "per_family_summary": family_summary,
        "per_seed_summary": seed_summary,
        "profile_metric_means": metric_means(records),
        "joint_margin_forensics_path": str(output_root / "stage36_r_joint_margin_forensics.json"),
        "joint_margin_forensics": forensics,
        "core24": {
            "row_count": len(core_records),
            "lifecycle_qualified": core_pass,
            "status": "PASS" if core_pass else "FAIL_OR_PARTIAL",
            "classification": (
                "WBC_CORE_LIFECYCLE_QUALIFIED_ON_STAGE35_CORE24"
                if core_pass
                else "DELIVERED_WBC_CORE_LIFECYCLE_LIMITATION"
            ),
        },
        "continuous8": {
            "row_count": len(continuous_records),
            "lifecycle_qualified": continuous_pass,
            "status": "PASS" if continuous_pass else "FAIL_OR_PARTIAL",
            "classification": continuous_classification,
        },
        "diagnostic_joint_margin_violation_rows": sum(
            bool(record.get("diagnostic_joint_margin_violation")) for record in records
        ),
        "diagnostic_joint_margin_violation_samples": sum(
            int(record.get("diagnostic_joint_margin_violation_sample_count", 0)) for record in records
        ),
        "true_joint_limit_termination_rows": sum(
            bool(record.get("true_joint_limit_termination_sample_count")) for record in records
        ),
        "true_joint_limit_termination_samples": sum(
            int(record.get("true_joint_limit_termination_sample_count", 0)) for record in records
        ),
        "corrected_tracking_pass_rows": corrected_tracking_pass_count,
        "superseded_derived_fields": {
            "source_final_classification": original_result.get("final_classification"),
            "source_motion_api_classification": original_result.get("motion_api_classification"),
            "source_continuous_base_classification": original_result.get("continuous_base_classification"),
            "source_tracking_pass_rows": "%d/%d" % (old_tracking_pass_count, len(records)),
            "superseded": [
                "rows[*].tracking_pass",
                "core24.tracking_pass",
                "continuous8.tracking_pass",
                "motion_api_classification",
                "continuous_base_classification",
                "final_classification",
            ],
            "reason": (
                "The original evaluator promoted joint_limit_violation, a diagnostic soft-margin "
                "check, into hard safety failure. Stage36-R applies the existing lifecycle contract "
                "where only source joint_limit/joint_limit_termination is a true joint-limit outcome."
            ),
            "original_raw_evidence_preserved": True,
        },
        "forbidden_execution": {
            "stage36_wbc_rerun": False,
            "pbs_submitted": False,
            "gpu_requested": False,
            "gpu_used_for_readjudication": False,
            "model_constructed": False,
            "gr00t_model_constructed": False,
            "forward": False,
            "training": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "original_evidence_mutated": False,
        },
        "top_level_outcome": top_level_outcome,
        "next_gate": next_gate,
        "artifact_root": str(output_root),
        "raw_manifest_path": str(output_root / "stage36_r_raw_hash_manifest.json"),
    }
    return {"result": result_payload, "forensics": forensics, "raw_manifest_rows": raw_manifest_rows}


def write_artifacts(
    output_root: Path,
    verification: Mapping[str, Any],
    built: Mapping[str, Any],
) -> Dict[str, Any]:
    log_path = output_root / "stage36_r_execution.log"
    append_event(log_path, "START", original_root=str(verification["original_root"]))
    append_event(
        log_path,
        "ORIGINAL_EVIDENCE_VERIFIED",
        original_seal_sha256=verification["original_seal_sha256"],
        original_result_sha256=verification["original_result_sha256"],
        original_raw_manifest_sha256=verification["original_raw_manifest_sha256"],
        seal_entry_count=verification["original_seal_entry_count"],
        row_count=EXPECTED_ORIGINAL_ROW_COUNT,
    )
    result_payload = built["result"]
    forensics = built["forensics"]
    raw_manifest_rows = built["raw_manifest_rows"]
    raw_manifest_payload = {
        "schema": "STAGE36_R_RAW_HASH_MANIFEST_V1",
        "timestamp_utc": now_utc(),
        "source_original_raw_manifest_sha256": verification["original_raw_manifest_sha256"],
        "rows": raw_manifest_rows,
        "row_count": len(raw_manifest_rows),
        "all_input_hashes_verified": True,
        "rerun": False,
    }
    write_json(output_root / "stage36_r_joint_margin_forensics.json", forensics)
    write_json(output_root / "stage36_r_raw_hash_manifest.json", raw_manifest_payload)
    write_json(output_root / "stage36_r_result.json", result_payload)
    source_manifest = {
        "schema": "STAGE36_R_SOURCE_MANIFEST_V1",
        "timestamp_utc": now_utc(),
        "git_head": result_payload["git"]["head"],
        "source_artifacts": result_payload["source_identity"]["source_artifacts"],
        "historical_stage36_source_artifacts": {
            "evaluation_runner": result_payload["source_identity"]["original_stage36_evaluation_runner"],
            "pbs_wrapper": result_payload["source_identity"]["original_stage36_pbs_wrapper"],
        },
    }
    write_json(output_root / "stage36_r_source_manifest.json", source_manifest)
    append_event(
        log_path,
        "READJUDICATION_COMPLETE",
        core24=result_payload["core24"]["classification"],
        continuous8=result_payload["continuous8"]["classification"],
        top_level_outcome=result_payload["top_level_outcome"],
        diagnostic_joint_margin_violation_rows=result_payload["diagnostic_joint_margin_violation_rows"],
        true_joint_limit_termination_rows=result_payload["true_joint_limit_termination_rows"],
    )
    result_manifest = {
        "schema": "STAGE36_R_RESULT_MANIFEST_V1",
        "timestamp_utc": now_utc(),
        "result_path": str(output_root / "stage36_r_result.json"),
        "result_sha256": sha256_file(output_root / "stage36_r_result.json"),
        "raw_hash_manifest_path": str(output_root / "stage36_r_raw_hash_manifest.json"),
        "raw_hash_manifest_sha256": sha256_file(output_root / "stage36_r_raw_hash_manifest.json"),
        "joint_margin_forensics_path": str(output_root / "stage36_r_joint_margin_forensics.json"),
        "joint_margin_forensics_sha256": sha256_file(output_root / "stage36_r_joint_margin_forensics.json"),
        "source_manifest_path": str(output_root / "stage36_r_source_manifest.json"),
        "source_manifest_sha256": sha256_file(output_root / "stage36_r_source_manifest.json"),
        "row_count": len(result_payload["rows"]),
        "core24_classification": result_payload["core24"]["classification"],
        "continuous8_classification": result_payload["continuous8"]["classification"],
        "top_level_outcome": result_payload["top_level_outcome"],
    }
    write_json(output_root / "stage36_r_result_manifest.json", result_manifest)
    # The log must be complete before its hash enters the seal.  Do not append
    # any event after the seal is written; that would invalidate the seal.
    append_event(log_path, "SEAL_READY", result_sha256=result_manifest["result_sha256"])

    seal_payload = {
        "schema": "STAGE36_R_ARTIFACT_SEAL_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "stage": STAGE,
        "pr": PR_NUMBER,
        "output_root": str(output_root),
        "entries": local_seal_entries(
            output_root,
            (
                "stage36_r_artifact_seal.json",
                "stage36_r_return.json",
            ),
        ),
        "input_authority": {
            "original_root": verification["original_root"],
            "original_artifact_seal_sha256": verification["original_seal_sha256"],
            "original_result_sha256": verification["original_result_sha256"],
            "original_raw_manifest_sha256": verification["original_raw_manifest_sha256"],
            "original_seal_entry_count_verified": verification["original_seal_entry_count"],
        },
        "source_artifacts": result_payload["source_identity"]["source_artifacts"],
        "historical_inputs_mutated": False,
        "stage36_wbc_rerun": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
    }
    seal_path = output_root / "stage36_r_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha256 = sha256_file(seal_path)
    return_payload = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": STAGE,
        "PR": PR_NUMBER,
        "GIT_HEAD": result_payload["git"]["head"],
        "BRANCH": result_payload["git"]["branch"],
        "WORKTREE_CLEAN": result_payload["git"]["worktree_clean"],
        "RESULT": "STAGE36_R_PASS_CORE24_LIFECYCLE_QUALIFIED" if result_payload["core24"]["lifecycle_qualified"] else "STAGE36_R_STOP",
        "TOP_LEVEL_OUTCOME": result_payload["top_level_outcome"],
        "CORE24": result_payload["core24"],
        "CONTINUOUS8": result_payload["continuous8"],
        "DIAGNOSTIC_JOINT_MARGIN_VIOLATION_ROWS": result_payload["diagnostic_joint_margin_violation_rows"],
        "TRUE_JOINT_LIMIT_TERMINATION_ROWS": result_payload["true_joint_limit_termination_rows"],
        "TRACKING_PROFILE_STATUS": result_payload["tracking_profile_status"],
        "ORIGINAL_EVIDENCE_IMMUTABLE": True,
        "ORIGINAL_ARTIFACT_SEAL_SHA256": verification["original_seal_sha256"],
        "ORIGINAL_RESULT_SHA256": verification["original_result_sha256"],
        "ORIGINAL_RAW_MANIFEST_SHA256": verification["original_raw_manifest_sha256"],
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "DETAIL_OUTPUTS": {
            "result": str(output_root / "stage36_r_result.json"),
            "result_manifest": str(output_root / "stage36_r_result_manifest.json"),
            "raw_hash_manifest": str(output_root / "stage36_r_raw_hash_manifest.json"),
            "joint_margin_forensics": str(output_root / "stage36_r_joint_margin_forensics.json"),
            "source_manifest": str(output_root / "stage36_r_source_manifest.json"),
            "execution_log": str(log_path),
            "artifact_seal": str(seal_path),
        },
        "PBS_SUBMITTED": False,
        "GPU_USED": False,
        "MODEL_CONSTRUCTED": False,
        "FORWARD": False,
        "TRAINING": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
        "NEXT_GATE": result_payload["next_gate"],
    }
    write_json(output_root / "stage36_r_return.json", return_payload)
    return return_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    output_root = args.output_root.absolute()
    prepare_output_root(output_root)
    log_path = output_root / "stage36_r_execution.log"
    try:
        append_event(log_path, "START", original_root=str(args.original_root.absolute()))
        verification = verify_original_evidence(args.original_root.absolute())
        built = build_report(args.original_root.absolute(), output_root, verification)
        return_payload = write_artifacts(output_root, verification, built)
        print(json.dumps(return_payload, indent=2, sort_keys=True))
        return 0 if return_payload["RESULT"] == "STAGE36_R_PASS_CORE24_LIFECYCLE_QUALIFIED" else 2
    except Exception as exc:
        append_event(log_path, "BLOCKED", error_type=type(exc).__name__, error=str(exc))
        blocked = {
            "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
            "ACTIVE_PI_GATE": PI_GATE_ID,
            "ACTIVE_STAGE": STAGE,
            "PR": PR_NUMBER,
            "RESULT": "BLOCKED",
            "BLOCKER": {"type": type(exc).__name__, "message": str(exc)},
            "ARTIFACT_ROOT": str(output_root),
            "PBS_SUBMITTED": False,
            "GPU_USED": False,
            "MODEL_CONSTRUCTED": False,
            "FORWARD": False,
            "TRAINING": False,
            "HISTORICAL_ARTIFACTS_MUTATED": False,
        }
        write_json(output_root / "stage36_r_return.json", blocked)
        print(json.dumps(blocked, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
