#!/usr/bin/env python3
"""Independently validate and seal the Stage37 structured WBC campaign.

This program is a read-only adjudicator for one Stage37 PBS output root.  It
does not invoke the native runner, PBS, CUDA, IsaacGym, GR00T, or any model.
It verifies the campaign's own raw/summary/seal chain and compares the
structured K_CAPACITY_STAR+T0 rows with the immutable corrected Stage36-R
dense rows on the preregistered paired predicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import stage36_wbc_oracle_campaign as campaign  # noqa: E402


PI_GATE_ID = "PI_GATE_STATE_0025"
PR_NUMBER = 23
STAGE37_BRANCH = "stage37/geometry-capacity-qualification"
NEXT_GATE = "GPT_PI_REVIEW_BEFORE_STRUCTURED_GR00T_DESIGN"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_REFERENCE_MODE = "BENCHMARK_CONTINUOUS"
EXPECTED_REPRESENTATION = "K16+T0"
EXPECTED_K = 16
EXPECTED_SEED = 13
EXPECTED_CASE_FAMILIES = ("N", "T", "P", "C")
EXPECTED_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in EXPECTED_CASE_FAMILIES
    for index in range(1, 9)
)
CORE_CASE_IDS = tuple(
    "%s%02d" % (family, index)
    for family in ("N", "T", "P")
    for index in range(1, 9)
)
CONTINUOUS_CASE_IDS = tuple("C%02d" % index for index in range(1, 9))

STAGE36_R_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_r_re_adjudication_20260906_f5506ff_v2"
)
STAGE36_R_RESULT = STAGE36_R_ROOT / "stage36_r_result.json"
STAGE36_R_RESULT_SHA256 = (
    "9d86e5db2a89c1f9290338c90a93699e2fd6eecf8da9ae3540c5543d11a95e9f"
)
STAGE36_R_SEAL = STAGE36_R_ROOT / "stage36_r_artifact_seal.json"
STAGE36_R_SEAL_SHA256 = (
    "b0d0930685d34d9d566f4721ced0f91a0347a869aae3998c1985e54acc0bcab1"
)
STAGE36_ORIGINAL_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_eval_602744.hopper-m-02"
)

PAIR_THRESHOLDS = {
    "position_mean_m": 0.002387,
    "rotation_mean_rad": 0.015389,
    "position_p95_m": 0.006767,
    "rotation_p95_rad": 0.038215,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_json(path: Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_json(payload, str(path))
    return payload


def write_json(path: Path, payload: Any) -> None:
    finite_json(payload, str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path)
    record: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "expected_sha256": expected_sha256,
    }
    record["hash_match"] = bool(
        path.is_file() and (expected_sha256 is None or record["sha256"] == expected_sha256)
    )
    return record


def verify_seal(root: Path, seal_path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
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
    try:
        seal = load_json(seal_path)
        entries = seal.get("entries", [])
        result["entry_count"] = len(entries)
        for entry in entries:
            path = Path(str(entry.get("path", "")))
            if not path.is_absolute():
                path = root / str(entry.get("relative_path", ""))
            actual = sha256_file(path) if path.is_file() else None
            expected_size = int(entry.get("size_bytes", -1))
            actual_size = int(path.stat().st_size) if path.is_file() else None
            if not path.is_file() or actual != entry.get("sha256") or actual_size != expected_size:
                result["mismatches"].append(
                    {
                        "path": str(path),
                        "expected_sha256": entry.get("sha256"),
                        "actual_sha256": actual,
                        "expected_size_bytes": expected_size,
                        "actual_size_bytes": actual_size,
                    }
                )
            else:
                result["entries_verified"] += 1
        result["status"] = "PASS" if not result["mismatches"] else "BLOCKED_SEAL_ENTRY_MISMATCH"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def verify_stage36_r_baseline() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "STAGE37_STAGE36_R_BASELINE_AUDIT_V1",
        "root": str(STAGE36_R_ROOT),
        "result": file_record(STAGE36_R_RESULT, STAGE36_R_RESULT_SHA256),
        "seal": verify_seal(STAGE36_R_ROOT, STAGE36_R_SEAL, STAGE36_R_SEAL_SHA256),
        "rows_exact": False,
        "core24_pass": False,
        "status": "BLOCKED",
    }
    try:
        payload = load_json(STAGE36_R_RESULT)
        rows = payload.get("rows", [])
        row_ids = [str(row.get("case_id")) for row in rows]
        result["rows_exact"] = bool(
            len(rows) == 32
            and len(set(row_ids)) == 32
            and set(row_ids) == set(EXPECTED_CASE_IDS)
            and all(row.get("representation") == "DENSE" for row in rows)
            and all(row.get("lifecycle_qualified") is True for row in rows)
            and all(row.get("finite") is True for row in rows)
            and all(row.get("reference_complete") is True for row in rows)
        )
        result["core24_pass"] = bool(
            payload.get("core24", {}).get("status") == "PASS"
            and payload.get("core24", {}).get("classification")
            == "WBC_CORE_LIFECYCLE_QUALIFIED_ON_STAGE35_CORE24"
        )
        result["status"] = (
            "PASS"
            if result["result"]["hash_match"]
            and result["seal"]["status"] == "PASS"
            and result["rows_exact"]
            and result["core24_pass"]
            else "BLOCKED_STAGE36_R_BASELINE_MISMATCH"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def metric(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def structured_hard_flags(row: Mapping[str, Any]) -> Dict[str, bool]:
    safety = row.get("safety_gate", {})
    flags = safety.get("hard_failure_flags", {}) if isinstance(safety, Mapping) else {}
    return {
        "fall": bool(flags.get("fall", False)),
        "invalid_state": bool(flags.get("invalid_state", False)),
        "joint_limit_termination": bool(flags.get("joint_limit_termination", False)),
        "timeout": bool(flags.get("timeout", False)),
        "tracking_abort": bool(
            flags.get("tracking_abort_before_reference_completion", False)
        ),
        "native_task_termination": bool(
            flags.get("native_task_termination_before_reference_completion", False)
        ),
    }


def verify_campaign_output(plan: Mapping[str, Any], campaign_root: Path) -> Dict[str, Any]:
    required = {
        "result": campaign_root / "stage36_result.json",
        "raw_hash_manifest": campaign_root / "stage36_raw_hash_manifest.json",
        "result_manifest": campaign_root / "stage36_result_manifest.json",
        "campaign_plan_snapshot": campaign_root / "stage36_campaign_plan_snapshot.json",
        "gpu_preflight": campaign_root / "stage36_gpu_preflight.json",
        "runtime_qualification": campaign_root / "stage36_runtime_qualification.json",
        "seal": campaign_root / "stage36_artifact_seal.json",
        "return": campaign_root / "stage36_return.json",
    }
    result: Dict[str, Any] = {
        "schema": "STAGE37_STRUCTURED_WBC_CAMPAIGN_AUDIT_V1",
        "campaign_root": str(campaign_root),
        "required_files": {
            name: file_record(path) for name, path in required.items()
        },
        "seal": verify_seal(campaign_root, required["seal"]),
        "plan_snapshot_matches": False,
        "plan_contract_pass": False,
        "result_contract_pass": False,
        "raw_manifest_contract_pass": False,
        "rows": [],
        "row_ids": [],
        "valid_structured_rows": 0,
        "status": "BLOCKED",
    }
    if not campaign_root.is_absolute() or not str(campaign_root).startswith("/scratch/"):
        result["status"] = "BLOCKED_NON_SCRATCH_CAMPAIGN_ROOT"
        return result
    try:
        snapshot = load_json(required["campaign_plan_snapshot"])
        result["plan_snapshot_matches"] = canonical_json(snapshot) == canonical_json(plan)
        result["plan_contract_pass"] = bool(
            result["plan_snapshot_matches"]
            and snapshot.get("pi_gate") == PI_GATE_ID
            and snapshot.get("campaign_kind") == "STAGE37_STRUCTURED_ORACLE"
            and snapshot.get("project") == EXPECTED_PROJECT
            and snapshot.get("queue") == "auto"
            and snapshot.get("one_pbs_allocation_only") is True
            and snapshot.get("representations") == [EXPECTED_REPRESENTATION]
            and snapshot.get("row_count") == 32
            and snapshot.get("primary_row_count") == 32
            and snapshot.get("robustness_row_count") == 0
            and snapshot.get("no_dense_rows") is True
        )
        campaign_result = load_json(required["result"])
        result["result_contract_pass"] = bool(
            campaign_result.get("schema") == "STAGE36_WBC_ORACLE_RESULT_V2"
            and campaign_result.get("pi_gate") == PI_GATE_ID
            and campaign_result.get("project") == EXPECTED_PROJECT
            and campaign_result.get("reference_mode") == EXPECTED_REFERENCE_MODE
            and campaign_result.get("row_count") == 32
            and campaign_result.get("all_planned_rows_completed") is True
            and campaign_result.get("all_finite") is True
            and campaign_result.get("wbc_call_started") is True
            and campaign_result.get("runtime_qualification", {}).get("status") == "PASS"
            and campaign_result.get("forbidden_execution", {}).get("training") is False
            and campaign_result.get("forbidden_execution", {}).get("new_checkpoint") is False
            and campaign_result.get("forbidden_execution", {}).get("sampler") is False
            and campaign_result.get("forbidden_execution", {}).get("get_action") is False
            and campaign_result.get("forbidden_execution", {}).get("second_allocation") is False
        )
        rows = campaign_result.get("rows", [])
        result["row_ids"] = [str(row.get("row_id")) for row in rows]
        result["rows"] = rows
        result["valid_structured_rows"] = sum(
            bool(
                row.get("representation") == EXPECTED_REPRESENTATION
                and row.get("geometry_k") == EXPECTED_K
                and row.get("timing_mode") == "T0"
                and row.get("phase") == "PRIMARY"
                and row.get("seed") == EXPECTED_SEED
                and row.get("scientific_row_finalized") is True
                and row.get("finite") is True
                and row.get("reference_complete") is True
            )
            for row in rows
        )
        raw_manifest = load_json(required["raw_hash_manifest"])
        raw_rows = raw_manifest.get("rows", [])
        raw_by_id = {str(row.get("row_id")): row for row in raw_rows}
        raw_ok = bool(len(raw_rows) == 32 and len(raw_by_id) == 32)
        for row in rows:
            row_id = str(row.get("row_id"))
            raw_entry = raw_by_id.get(row_id)
            if raw_entry is None:
                raw_ok = False
                continue
            summary_path = Path(str(raw_entry.get("summary_path", "")))
            raw_path = Path(str(raw_entry.get("raw_rows_path", "")))
            summary_hash = sha256_file(summary_path) if summary_path.is_file() else None
            raw_hash = sha256_file(raw_path) if raw_path.is_file() else None
            if summary_hash != raw_entry.get("summary_sha256") or raw_hash != raw_entry.get("raw_rows_sha256"):
                raw_ok = False
            if str(row.get("raw_rows_path")) != str(raw_path):
                raw_ok = False
            if str(row.get("output_dir", "")) != str(summary_path.parent):
                raw_ok = False
            try:
                summary = load_json(summary_path)
                samples = campaign.read_rows(raw_path)
                safety = campaign.safety_profile(samples)
                finalized = campaign.finalized_native_output(
                    summary, samples, bool(row.get("timed_out", False))
                )
                returncode = int(row.get("returncode", 1))
                runner_pass = bool(
                    finalized
                    and campaign.finite_value(summary)
                    and campaign.finite_value(samples)
                    and (returncode == 0 or returncode == campaign.POST_RUN_COMPLETION_RETURN_CODE)
                )
                recomputed_tracking = bool(
                    runner_pass
                    and bool(
                        summary.get("reference_completed", False)
                        or safety.get("reference_complete_seen", False)
                    )
                    and not bool(safety.get("safety_violation", True))
                )
                row["independent_raw_audit"] = {
                    "summary_sha256": summary_hash,
                    "raw_rows_sha256": raw_hash,
                    "raw_rows_count": len(samples),
                    "safety_gate": safety,
                    "finalized_recomputed": finalized,
                    "runner_pass_recomputed": runner_pass,
                    "tracking_pass_recomputed": recomputed_tracking,
                    "finite": campaign.finite_value(summary) and campaign.finite_value(samples),
                }
                if not runner_pass or not recomputed_tracking:
                    raw_ok = False
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raw_ok = False
                row["independent_raw_audit"] = {
                    "status": "ERROR",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
        result["raw_manifest_contract_pass"] = bool(
            raw_ok
            and result["valid_structured_rows"] == 32
            and set(result["row_ids"]) == {
                "%s_%s_seed13_primary" % ("k16_t0", case_id)
                for case_id in EXPECTED_CASE_IDS
            }
        )
        return_payload = load_json(required["return"])
        result["return_contract"] = {
            "pbs_submitted": return_payload.get("PBS_SUBMITTED") is True,
            "job_id": return_payload.get("PBS_JOB_ID"),
            "result": return_payload.get("RESULT"),
            "gr00t_model_constructed": return_payload.get("GR00T_MODEL_CONSTRUCTED") is False,
            "training": return_payload.get("TRAINING") is False,
        }
        result["campaign_result_payload"] = campaign_result
        result["status"] = (
            "PASS"
            if all(record["exists"] for record in result["required_files"].values())
            and result["seal"]["status"] == "PASS"
            and result["plan_contract_pass"]
            and result["result_contract_pass"]
            and result["raw_manifest_contract_pass"]
            and result["return_contract"]["pbs_submitted"]
            and result["return_contract"]["gr00t_model_constructed"]
            and result["return_contract"]["training"]
            else "BLOCKED_STRUCTURED_CAMPAIGN_CONTRACT"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def pair_analysis(
    baseline_payload: Mapping[str, Any],
    campaign_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_rows = {
        str(row.get("case_id")): row for row in baseline_payload.get("rows", [])
    }
    structured_rows = {
        str(row.get("case_id")): row for row in campaign_audit.get("rows", [])
    }
    pairs: List[Dict[str, Any]] = []
    new_failure_counts = {
        "falls": 0,
        "invalid_states": 0,
        "true_joint_limit_terminations": 0,
        "timeouts": 0,
        "tracking_aborts": 0,
        "native_task_terminations": 0,
        "reference_completion_regressions": 0,
    }
    for case_id in EXPECTED_CASE_IDS:
        dense = baseline_rows.get(case_id, {})
        structured = structured_rows.get(case_id, {})
        dense_metrics = dense.get("profile_metrics", {})
        structured_metrics = structured.get("profile_metrics", {})
        deltas: Dict[str, Optional[float]] = {}
        metric_paths = {
            "position_mean_m": ("position_m", "mean"),
            "rotation_mean_rad": ("rotation_rad", "mean"),
            "position_p95_m": ("position_m", "p95"),
            "rotation_p95_rad": ("rotation_rad", "p95"),
            "position_rmse_m": ("position_m", "rmse"),
            "rotation_rmse_rad": ("rotation_rad", "rmse"),
            "position_max_m": ("position_m", "max"),
            "rotation_max_rad": ("rotation_rad", "max"),
        }
        for name, path in metric_paths.items():
            dense_value = metric(dense_metrics, *path)
            structured_value = metric(structured_metrics, *path)
            deltas[name] = (
                None
                if dense_value is None or structured_value is None
                else abs(structured_value - dense_value)
            )
        flags = structured_hard_flags(structured)
        new_failure_counts["falls"] += int(flags["fall"])
        new_failure_counts["invalid_states"] += int(flags["invalid_state"])
        new_failure_counts["true_joint_limit_terminations"] += int(
            flags["joint_limit_termination"]
        )
        new_failure_counts["timeouts"] += int(flags["timeout"])
        new_failure_counts["tracking_aborts"] += int(flags["tracking_abort"])
        new_failure_counts["native_task_terminations"] += int(flags["native_task_termination"])
        if dense.get("reference_complete") is True and structured.get("reference_complete") is not True:
            new_failure_counts["reference_completion_regressions"] += 1
        predicate_pass = {
            name: deltas.get(name) is not None and deltas[name] <= limit
            for name, limit in PAIR_THRESHOLDS.items()
        }
        pairs.append(
            {
                "case_id": case_id,
                "family": case_id[:1],
                "dense_row_id": dense.get("row_id"),
                "structured_row_id": structured.get("row_id"),
                "dense_reference_complete": dense.get("reference_complete"),
                "structured_reference_complete": structured.get("reference_complete"),
                "structured_hard_failure_flags": flags,
                "structured_diagnostic_joint_margin_violation": bool(
                    structured.get("independent_raw_audit", {})
                    .get("safety_gate", {})
                    .get("diagnostic_joint_margin_violation", False)
                ),
                "deltas": deltas,
                "paired_predicate_pass": predicate_pass,
                "all_paired_tracking_predicates_pass": all(predicate_pass.values()),
            }
        )
    all_pairs_present = bool(
        len(pairs) == 32
        and all(pair.get("dense_row_id") and pair.get("structured_row_id") for pair in pairs)
    )
    no_new_failures = all(value == 0 for value in new_failure_counts.values())
    all_pair_predicates = all(
        pair.get("all_paired_tracking_predicates_pass") is True for pair in pairs
    )
    structured_core_pass = bool(
        all(
            structured_rows.get(case_id, {}).get("tracking_pass") is True
            and structured_rows.get(case_id, {}).get("scientific_row_finalized") is True
            for case_id in CORE_CASE_IDS
        )
    )
    structured_c8_lifecycle = bool(
        all(
            structured_rows.get(case_id, {}).get("tracking_pass") is True
            and structured_rows.get(case_id, {}).get("scientific_row_finalized") is True
            and structured_rows.get(case_id, {}).get("finite") is True
            and structured_rows.get(case_id, {}).get("reference_complete") is True
            for case_id in CONTINUOUS_CASE_IDS
        )
    )
    structured_c8_support = bool(
        all(
            structured_rows.get(case_id, {})
            .get("base_participation", {})
            .get("nontrivial_observed")
            is True
            for case_id in CONTINUOUS_CASE_IDS
        )
    )
    geometry_parity = bool(
        all_pairs_present
        and structured_core_pass
        and no_new_failures
        and all_pair_predicates
    )
    if geometry_parity:
        geometry_status = "GEOMETRY_ORACLE_PARITY_QUALIFIED"
    else:
        geometry_status = "GEOMETRY_ORACLE_PARITY_FAILED"
    if structured_c8_lifecycle and structured_c8_support:
        c8_classification = "WBC_CONTINUOUS_BASE_LIFECYCLE_QUALIFIED_ON_STRUCTURED_K16_T0"
    elif structured_c8_lifecycle:
        c8_classification = "STRUCTURED_C8_REFERENCE_COMPLETION_WITHOUT_BASE_PARTICIPATION"
    else:
        c8_classification = "STRUCTURED_C8_LIFECYCLE_LIMITATION"
    return {
        "schema": "STAGE37_STRUCTURED_ORACLE_PAIRED_ANALYSIS_V1",
        "timestamp_utc": now_utc(),
        "baseline": {
            "stage36_r_root": str(STAGE36_R_ROOT),
            "stage36_r_result_sha256": STAGE36_R_RESULT_SHA256,
            "representation": "DENSE",
            "seed": EXPECTED_SEED,
        },
        "structured": {
            "representation": EXPECTED_REPRESENTATION,
            "geometry_k": EXPECTED_K,
            "timing_mode": "T0",
            "seed": EXPECTED_SEED,
        },
        "paired_predicate_thresholds": dict(PAIR_THRESHOLDS),
        "pairs": pairs,
        "new_failure_counts": new_failure_counts,
        "all_pairs_present": all_pairs_present,
        "no_new_hard_failures_or_completion_regressions": no_new_failures,
        "all_paired_tracking_predicates_pass": all_pair_predicates,
        "structured_core24_lifecycle_pass": structured_core_pass,
        "structured_continuous8_lifecycle_pass": structured_c8_lifecycle,
        "structured_continuous8_base_participation_pass": structured_c8_support,
        "geometry_oracle_parity": geometry_status,
        "wbc_continuous_base_classification": c8_classification,
        "status": "PASS" if geometry_parity else "FAIL",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    campaign_root = args.campaign_root.resolve()
    output_root = args.output_root.absolute()
    if output_root.exists():
        raise SystemExit("FAIL_CLOSED finalization output already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    log_path = output_root / "stage37_finalization.log"

    def event(name: str, **fields: Any) -> None:
        payload = {"timestamp_utc": now_utc(), "event": name}
        payload.update(fields)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    event("START", plan=str(plan_path), campaign_root=str(campaign_root))
    plan = load_json(plan_path)
    baseline_audit = verify_stage36_r_baseline()
    write_json(output_root / "stage37_stage36_r_baseline_audit.json", baseline_audit)
    event("STAGE36_R_BASELINE", status=baseline_audit.get("status"))

    campaign_audit = verify_campaign_output(plan, campaign_root)
    write_json(output_root / "stage37_structured_campaign_audit.json", campaign_audit)
    event("STRUCTURED_CAMPAIGN_AUDIT", status=campaign_audit.get("status"))

    baseline_payload = load_json(STAGE36_R_RESULT)
    paired = pair_analysis(baseline_payload, campaign_audit)
    write_json(output_root / "stage37_paired_oracle_analysis.json", paired)
    event("PAIRED_ANALYSIS", status=paired.get("status"))

    plan_hash = sha256_file(plan_path)
    campaign_result_path = campaign_root / "stage36_result.json"
    campaign_raw_path = campaign_root / "stage36_raw_hash_manifest.json"
    campaign_seal_path = campaign_root / "stage36_artifact_seal.json"
    source_records = []
    for role, path in (
        ("stage37_campaign_plan", plan_path),
        ("stage36_r_result", STAGE36_R_RESULT),
        ("stage36_r_seal", STAGE36_R_SEAL),
        ("stage37_campaign_result", campaign_result_path),
        ("stage37_campaign_raw_hash_manifest", campaign_raw_path),
        ("stage37_campaign_seal", campaign_seal_path),
        ("evaluation_runner", REPO / "scripts/native_wbc_benchmark.py"),
        ("campaign_runner", REPO / "scripts/stage36_wbc_oracle_campaign.py"),
        ("pbs_wrapper", REPO / "hpc/pbs/stage37_structured_wbc.pbs"),
        ("stage37_finalizer", Path(__file__).resolve()),
    ):
        record = file_record(path)
        record["role"] = role
        source_records.append(record)
    write_json(
        output_root / "stage37_source_manifest.json",
        {
            "schema": "STAGE37_STRUCTURED_SOURCE_MANIFEST_V1",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "records": source_records,
        },
    )

    geometry_parity = paired.get("geometry_oracle_parity") == "GEOMETRY_ORACLE_PARITY_QUALIFIED"
    timing_star = plan.get("stage37_timing", {}).get("t_capacity_star")
    if geometry_parity and timing_star is None:
        full_motion_api_parity = "GEOMETRY_ONLY_T0_ORACLE_PARITY_QUALIFIED_TIMING_CAPACITY_UNAVAILABLE"
    elif geometry_parity:
        full_motion_api_parity = "T0_QUALIFIED_T_STAR_REQUIRES_SEPARATE_AUTHORIZED_ARM"
    else:
        full_motion_api_parity = "GEOMETRY_ONLY_T0_ORACLE_PARITY_FAILED"
    final_result = {
        "schema": "STAGE37_STRUCTURED_ORACLE_FINAL_RESULT_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "branch": STAGE37_BRANCH,
        "stage": "Stage37-structured-oracle-K16-T0",
        "campaign_root": str(campaign_root),
        "campaign_job_id": campaign_audit.get("return_contract", {}).get("job_id"),
        "stage36_r_core_lifecycle": (
            "WBC_CORE_LIFECYCLE_QUALIFIED_ON_STAGE35_CORE24"
            if baseline_audit.get("core24_pass")
            else "BLOCKED_STAGE36_R_BASELINE"
        ),
        "stage36_r_continuous_base_lifecycle": baseline_payload.get("continuous8", {}).get(
            "classification"
        ),
        "true_joint_limit_termination_rows": baseline_payload.get(
            "true_joint_limit_termination_rows"
        ),
        "diagnostic_joint_margin_violation_rows": baseline_payload.get(
            "diagnostic_joint_margin_violation_rows"
        ),
        "k_capacity_star": plan.get("stage37_geometry", {}).get("k_capacity_star"),
        "geometry_capacity_classification": plan.get("stage37_geometry", {}).get(
            "classification"
        ),
        "t_capacity_star": timing_star,
        "timing_compression_status": plan.get("stage37_timing", {}).get(
            "timing_compression_status"
        ),
        "geometry_oracle_parity": paired.get("geometry_oracle_parity"),
        "full_motion_api_oracle_parity": full_motion_api_parity,
        "wbc_continuous_base_classification": paired.get(
            "wbc_continuous_base_classification"
        ),
        "paired_analysis_path": str(output_root / "stage37_paired_oracle_analysis.json"),
        "campaign_audit_path": str(output_root / "stage37_structured_campaign_audit.json"),
        "plan_sha256": plan_hash,
        "source_records": source_records,
        "scientific_contract_changed": False,
        "gr00t_model_forward": False,
        "training": False,
        "hardware": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "dense_stage36_rerun": False,
        "final_classification": (
            "STAGE37_STRUCTURED_ORACLE_GEOMETRY_PARITY_QUALIFIED"
            if geometry_parity
            else "STAGE37_STRUCTURED_ORACLE_GEOMETRY_PARITY_FAILED"
        ),
        "next_gate": NEXT_GATE,
    }
    write_json(output_root / "stage37_structured_oracle_result.json", final_result)
    write_json(
        output_root / "stage37_structured_raw_hash_manifest.json",
        {
            "schema": "STAGE37_STRUCTURED_RAW_HASH_MANIFEST_V1",
            "timestamp_utc": now_utc(),
            "campaign_root": str(campaign_root),
            "campaign_raw_manifest_path": str(campaign_raw_path),
            "campaign_raw_manifest_sha256": sha256_file(campaign_raw_path)
            if campaign_raw_path.is_file()
            else None,
            "rows": [
                {
                    "row_id": row.get("row_id"),
                    "summary_path": row.get("raw_rows_path", "").replace(
                        "native_external_rows.jsonl", "native_external_summary.json"
                    ),
                    "raw_rows_path": row.get("raw_rows_path"),
                    "summary_sha256": (
                        sha256_file(
                            Path(str(row.get("raw_rows_path"))).with_name(
                                "native_external_summary.json"
                            )
                        )
                        if Path(str(row.get("raw_rows_path"))).with_name(
                            "native_external_summary.json"
                        ).is_file()
                        else None
                    ),
                    "raw_rows_sha256": (
                        sha256_file(Path(str(row.get("raw_rows_path"))))
                        if Path(str(row.get("raw_rows_path"))).is_file()
                        else None
                    ),
                }
                for row in campaign_audit.get("rows", [])
            ],
        },
    )
    write_json(
        output_root / "stage37_structured_result_manifest.json",
        {
            "schema": "STAGE37_STRUCTURED_RESULT_MANIFEST_V1",
            "result_path": str(output_root / "stage37_structured_oracle_result.json"),
            "result_sha256": sha256_file(output_root / "stage37_structured_oracle_result.json"),
            "paired_analysis_path": str(output_root / "stage37_paired_oracle_analysis.json"),
            "paired_analysis_sha256": sha256_file(output_root / "stage37_paired_oracle_analysis.json"),
            "campaign_audit_path": str(output_root / "stage37_structured_campaign_audit.json"),
            "campaign_audit_sha256": sha256_file(output_root / "stage37_structured_campaign_audit.json"),
            "raw_hash_manifest_path": str(output_root / "stage37_structured_raw_hash_manifest.json"),
            "raw_hash_manifest_sha256": sha256_file(
                output_root / "stage37_structured_raw_hash_manifest.json"
            ),
            "source_manifest_path": str(output_root / "stage37_source_manifest.json"),
            "source_manifest_sha256": sha256_file(output_root / "stage37_source_manifest.json"),
            "row_count": len(campaign_audit.get("rows", [])),
            "final_classification": final_result["final_classification"],
        },
    )
    event(
        "RETURN",
        geometry_oracle_parity=final_result["geometry_oracle_parity"],
        final_classification=final_result["final_classification"],
    )
    seal_payload = {
        "schema": "STAGE37_STRUCTURED_ARTIFACT_SEAL_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "stage": final_result["stage"],
        "output_root": str(output_root),
        "entries": seal_entries(
            output_root, ("stage37_structured_artifact_seal.json", "stage37_final_return.json")
        ),
        "source_artifacts": source_records,
        "campaign_root_seal_sha256": campaign_audit.get("seal", {})
        .get("seal", {})
        .get("sha256"),
        "stage36_r_result_sha256": STAGE36_R_RESULT_SHA256,
        "stage36_r_seal_sha256": STAGE36_R_SEAL_SHA256,
        "historical_inputs_mutated": False,
        "dense_stage36_rerun": False,
        "pbs_submitted_by_finalizer": False,
        "gpu_used_by_finalizer": False,
        "model_constructed_by_finalizer": False,
        "forward_by_finalizer": False,
    }
    seal_path = output_root / "stage37_structured_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha = sha256_file(seal_path)
    final_return = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": final_result["stage"],
        "PR": PR_NUMBER,
        "RESULT": "STAGE37_STRUCTURED_ORACLE_FINALIZED_NEEDS_PI_REVIEW"
        if geometry_parity
        else "STAGE37_STRUCTURED_ORACLE_PARITY_FAILED_NEEDS_PI_REVIEW",
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "ARTIFACT_SEAL_SHA256": seal_sha,
        "STAGE36_R_CORE_LIFECYCLE": final_result["stage36_r_core_lifecycle"],
        "STAGE36_R_CONTINUOUS_BASE_LIFECYCLE": final_result[
            "stage36_r_continuous_base_lifecycle"
        ],
        "TRUE_JOINT_LIMIT_TERMINATION_ROWS": final_result[
            "true_joint_limit_termination_rows"
        ],
        "DIAGNOSTIC_JOINT_MARGIN_VIOLATION_ROWS": final_result[
            "diagnostic_joint_margin_violation_rows"
        ],
        "K_CAPACITY_STAR": final_result["k_capacity_star"],
        "GEOMETRY_CAPACITY_CLASSIFICATION": final_result[
            "geometry_capacity_classification"
        ],
        "T_CAPACITY_STAR": final_result["t_capacity_star"],
        "TIMING_COMPRESSION_STATUS": final_result["timing_compression_status"],
        "GEOMETRY_ORACLE_PARITY": final_result["geometry_oracle_parity"],
        "FULL_MOTION_API_ORACLE_PARITY": final_result["full_motion_api_oracle_parity"],
        "WBC_CONTINUOUS_BASE_CLASSIFICATION": final_result[
            "wbc_continuous_base_classification"
        ],
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "HARDWARE": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "DENSE_STAGE36_RERUN": False,
        "NEXT_GATE": NEXT_GATE,
    }
    write_json(output_root / "stage37_final_return.json", final_return)
    return 0 if geometry_parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
