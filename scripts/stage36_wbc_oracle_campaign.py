#!/usr/bin/env python3
"""Execute the single, pre-admitted Stage36 native WBC campaign.

The CPU authority harness creates a frozen plan.  This runner consumes that
plan inside one PBS allocation and invokes the existing native WBC launcher
once per admitted case.  It has no qsub path, no retry path, no GR00T path,
and no training path.  A failure preserves the partial evidence and exits;
the caller must not submit a replacement allocation after this script starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO = Path(__file__).resolve().parents[1]
PI_GATE_ID = "PI_GATE_STATE_0021"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE36_LONG_HORIZON_LADDER"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_REFERENCE_MODE = "BENCHMARK_CONTINUOUS"
EXPECTED_SEED = 13
EXPECTED_SECONDS = 2.0
MAX_ROWS = 96
EXPECTED_STATUS = "NATIVE_BENCHMARK_CONTINUOUS_PASS"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_event(output_root: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(fields)
    with (output_root / "stage36_campaign_execution.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize(payload), sort_keys=True, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gpu_preflight(output_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "STAGE36_GPU_RUNTIME_PREFLIGHT_V1",
        "timestamp_utc": now_utc(),
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "nvidia_smi": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "device_count": None,
        "device_names": [],
        "status": "BLOCKED",
        "model_constructed": False,
        "wbc_executed": False,
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        names = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
        result["nvidia_smi"] = {
            "returncode": smi.returncode,
            "stdout": smi.stdout.strip(),
            "stderr": smi.stderr.strip(),
        }
        result["device_names"] = names
        import isaacgym  # noqa: F401
        import torch

        result.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        )
        result["status"] = (
            "PASS"
            if smi.returncode == 0
            and len(names) == 1
            and "H200" in names[0]
            and result["cuda_available"]
            and result["device_count"] == 1
            else "BLOCKED_GPU_RUNTIME"
        )
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_gpu_preflight.json", result)
    return result


def validate_plan(plan: Mapping[str, Any], source_root: Path) -> List[str]:
    errors: List[str] = []
    if plan.get("schema") != "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V2":
        errors.append("plan_schema")
    if plan.get("pi_gate") != PI_GATE_ID:
        errors.append("pi_gate")
    if plan.get("plan_status") != "READY_FOR_E":
        errors.append("plan_not_ready")
    if plan.get("project") != EXPECTED_PROJECT or plan.get("queue") != "auto":
        errors.append("project_or_queue")
    if plan.get("one_pbs_allocation_only") is not True:
        errors.append("allocation_policy")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows_missing")
        return errors
    if len(rows) > MAX_ROWS:
        errors.append("row_budget")
    row_ids = []
    for row in rows:
        row_ids.append(row.get("row_id"))
        if row.get("representation") != "DENSE":
            errors.append("non_dense_row:%s" % row.get("row_id"))
        if row.get("reference_mode") != EXPECTED_REFERENCE_MODE:
            errors.append("reference_mode:%s" % row.get("row_id"))
        if row.get("seed") != EXPECTED_SEED:
            errors.append("seed:%s" % row.get("row_id"))
        if float(row.get("seconds", -1.0)) != EXPECTED_SECONDS:
            errors.append("duration:%s" % row.get("row_id"))
        trajectory = Path(str(row.get("trajectory_path", "")))
        metadata = Path(str(row.get("metadata_path", "")))
        if not trajectory.is_file() or not metadata.is_file():
            errors.append("missing_input:%s" % row.get("row_id"))
        if not str(trajectory).startswith("/scratch/") or not str(metadata).startswith("/scratch/"):
            errors.append("non_scratch_input:%s" % row.get("row_id"))
        if not source_root.is_dir():
            errors.append("source_root_missing")
    if len(set(row_ids)) != len(row_ids):
        errors.append("duplicate_row_id")
    keys = [(row.get("case_id"), row.get("representation"), row.get("seed")) for row in rows]
    if len(set(keys)) != len(keys):
        errors.append("duplicate_case_representation_seed")
    if int(plan.get("sparse_rows", 0)) != 0:
        errors.append("sparse_rows_present")
    return sorted(set(errors))


def native_command(
    runner: Path,
    source_root: Path,
    arm: Path,
    dog: Path,
    trajectory: Path,
    output_dir: Path,
    metadata: Path,
    seed: int,
    seconds: float,
) -> List[str]:
    # Keep this list literal and auditable.  There is exactly one permitted
    # reference mode, and no diagnostic termination or training switches are
    # ever appended by this runner.
    return [
        sys.executable,
        str(runner),
        "--source-root",
        str(source_root),
        "--arm",
        str(arm),
        "--dog",
        str(dog),
        "--reference-mode",
        EXPECTED_REFERENCE_MODE,
        "--trajectory",
        str(trajectory),
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(seconds),
        "--seed",
        str(seed),
        "--metadata-json",
        str(metadata),
    ]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safety_profile(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    previous: Optional[tuple] = None
    for row in rows:
        state = (
            row.get("termination_code"),
            bool(row.get("reference_complete", False)),
            bool(row.get("fall_flag", False)),
            bool(row.get("invalid_state", False)),
            bool(row.get("joint_limit_violation", False)),
        )
        if state != previous:
            transitions.append({"step": row.get("step"), "state": list(state)})
            previous = state
    violation = any(
        bool(row.get(name, False))
        for row in rows
        for name in ("fall_flag", "invalid_state", "joint_limit_violation", "joint_limit_termination")
    )
    finite = all(finite_value(dict(row)) for row in rows)
    return {
        "finite": finite,
        "safety_violation": violation,
        "transitions": transitions,
        "termination_codes": sorted({str(row.get("termination_code", "UNKNOWN")) for row in rows}),
        "row_count": len(rows),
        "reference_complete_seen": any(bool(row.get("reference_complete", False)) for row in rows),
    }


def row_result(plan_row: Mapping[str, Any], output_dir: Path, returncode: int, timed_out: bool) -> Dict[str, Any]:
    summary_path = output_dir / "native_external_summary.json"
    raw_rows_path = output_dir / "native_external_rows.jsonl"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    raw_rows = read_rows(raw_rows_path)
    finite = finite_value(summary) and finite_value(raw_rows)
    safety = safety_profile(raw_rows)
    runner_pass = bool(
        returncode == 0
        and summary.get("status") == EXPECTED_STATUS
        and finite
        and safety["finite"]
        and not timed_out
    )
    complete = bool(summary.get("reference_completed", False) or safety["reference_complete_seen"])
    result = {
        "row_id": plan_row["row_id"],
        "case_id": plan_row["case_id"],
        "family": plan_row["family"],
        "representation": plan_row["representation"],
        "seed": plan_row["seed"],
        "reference_mode": plan_row["reference_mode"],
        "trajectory_path": plan_row["trajectory_path"],
        "trajectory_sha256_planned": plan_row.get("trajectory_sha256"),
        "output_dir": str(output_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "native_status": summary.get("status"),
        "native_termination": summary.get("termination"),
        "native_summary": sanitize(summary),
        "raw_rows_path": str(raw_rows_path),
        "raw_rows_count": len(raw_rows),
        "finite": finite and safety["finite"],
        "finite_flags": {
            "summary": finite_value(summary),
            "raw_rows": finite_value(raw_rows),
            "metrics": finite_value(summary.get("metrics", {})),
        },
        "safety_gate": safety,
        "reference_complete": complete,
        "runner_pass": runner_pass,
        "tracking_pass": bool(runner_pass and complete and not safety["safety_violation"]),
    }
    metrics = summary.get("metrics", {})
    result["profile_metrics"] = sanitize(metrics)
    if raw_rows:
        base = [row.get("base_pos", []) for row in raw_rows]
        try:
            import numpy as np

            base_array = np.asarray(base, dtype=np.float64)
            result["base_participation"] = {
                "finite": bool(np.isfinite(base_array).all()),
                "x_span_m": float(np.ptp(base_array[:, 0])) if len(base_array) else 0.0,
                "path_length_m": float(np.linalg.norm(np.diff(base_array, axis=0), axis=1).sum()) if len(base_array) > 1 else 0.0,
                "nontrivial_observed": bool(len(base_array) > 1 and np.ptp(base_array[:, 0]) > 1e-3),
            }
        except Exception as exc:
            result["base_participation"] = {"status": "UNAVAILABLE", "error": str(exc)}
    else:
        result["base_participation"] = {"status": "NO_ROWS"}
    return result


def aggregate_metric(rows: Sequence[Mapping[str, Any]], metric_path: Sequence[str]) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value: Any = row.get("profile_metrics", {})
        for key in metric_path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return None if not values else float(sum(values) / len(values))


def profile_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "row_count": len(rows),
        "runner_pass_count": sum(bool(row.get("runner_pass")) for row in rows),
        "tracking_pass_count": sum(bool(row.get("tracking_pass")) for row in rows),
        "all_finite": bool(rows) and all(bool(row.get("finite")) for row in rows),
        "position_rmse_mean_m": aggregate_metric(rows, ("position_m", "rmse")),
        "position_p95_mean_m": aggregate_metric(rows, ("position_m", "p95")),
        "rotation_rmse_mean_rad": aggregate_metric(rows, ("rotation_rad", "rmse")),
        "rotation_p95_mean_rad": aggregate_metric(rows, ("rotation_rad", "p95")),
        "time_in_tolerance_mean": aggregate_metric(rows, ("time_in_tolerance_fraction_0.05m_0.20rad",)),
        "base_participation_x_span_mean_m": aggregate_metric(rows, ("base_participation", "x_span_m")),
    }


def seal_entries(output_root: Path, exclude: Iterable[str]) -> List[Dict[str, Any]]:
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


def make_return(
    output_root: Path,
    result: Mapping[str, Any],
    seal_path: Path,
    seal_sha256: Optional[str],
    status: str,
    error: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": "Stage36-RQ",
        "PR": 22,
        "PBS_JOB_ID": os.environ.get("PBS_JOBID"),
        "PBS_PROJECT": EXPECTED_PROJECT,
        "RESULT": status,
        "NEXT_GATE": NEXT_GATE,
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "DETAIL_OUTPUTS": {
            "gpu_preflight": str(output_root / "stage36_gpu_preflight.json"),
            "campaign_plan": str(output_root / "stage36_campaign_plan_snapshot.json"),
            "result": str(output_root / "stage36_result.json"),
            "raw_hash_manifest": str(output_root / "stage36_raw_hash_manifest.json"),
            "result_manifest": str(output_root / "stage36_result_manifest.json"),
            "artifact_seal": str(seal_path),
        },
        "PBS_SUBMITTED": True,
        "GPU_USED": True,
        "MODEL_CONSTRUCTED": True,
        "FORWARD": bool(result.get("scientific_execution_begun", False)),
        "TRAINING": False,
        "GR00T_MODEL_CONSTRUCTED": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
        "ERROR": None if error is None else dict(error),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, default=REPO / "scripts/native_wbc_benchmark.py")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit("FAIL_CLOSED output already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    scientific_execution_begun = False
    rows: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    status = "BLOCKED"
    log_event(output_root, "START", plan=str(args.plan), pbs_job_id=os.environ.get("PBS_JOBID"))
    try:
        plan = load_json(args.plan)
        write_json(output_root / "stage36_campaign_plan_snapshot.json", plan)
        errors = validate_plan(plan, args.source_root.resolve())
        if errors:
            raise RuntimeError("plan validation failed: %s" % ",".join(errors))
        gpu = run_gpu_preflight(output_root)
        if gpu.get("status") != "PASS":
            raise RuntimeError("GPU preflight failed")
        if not args.native_runner.is_file() or not args.arm.is_file() or not args.dog.is_file():
            raise RuntimeError("native runner or exact checkpoint input missing")
        if sha256_file(args.native_runner) != plan.get("native_runner_sha256", sha256_file(args.native_runner)):
            raise RuntimeError("native runner SHA does not match plan")
        rows_plan = plan["rows"]
        for index, plan_row in enumerate(rows_plan, start=1):
            row_output = output_root / "rows" / str(plan_row["row_id"])
            row_output.mkdir(parents=True)
            command = native_command(
                args.native_runner,
                args.source_root.resolve(),
                args.arm.resolve(),
                args.dog.resolve(),
                Path(plan_row["trajectory_path"]).resolve(),
                row_output,
                Path(plan_row["metadata_path"]).resolve(),
                int(plan_row["seed"]),
                float(plan_row["seconds"]),
            )
            command_text = " ".join(command)
            if "qsub" in command_text or "--reference-mode BENCHMARK_CONTINUOUS" not in command_text:
                raise RuntimeError("selected native command violates execution contract")
            scientific_execution_begun = True
            log_event(output_root, "ROW_START", index=index, row_id=plan_row["row_id"], command=command)
            run_log = row_output / "run.log"
            timed_out = False
            with run_log.open("w", encoding="utf-8") as handle:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(args.source_root.resolve()),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        timeout=float(args.timeout_seconds),
                    )
                    returncode = int(completed.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
                    handle.write("STAGE36_TIMEOUT\n")
            row = row_result(plan_row, row_output, returncode, timed_out)
            rows.append(row)
            log_event(
                output_root,
                "ROW_END",
                row_id=plan_row["row_id"],
                returncode=returncode,
                runner_pass=row["runner_pass"],
                tracking_pass=row["tracking_pass"],
            )
        core_rows = [row for row in rows if row["family"] in ("N", "T", "P")]
        continuous_rows = [row for row in rows if row["family"] == "C"]
        core_pass = len(core_rows) == 24 and all(row["tracking_pass"] for row in core_rows)
        continuous_pass = len(continuous_rows) == 8 and all(row["tracking_pass"] for row in continuous_rows)
        family_summary = {
            family: profile_summary([row for row in rows if row["family"] == family])
            for family in ("N", "T", "P", "C")
        }
        seed_summary = {str(EXPECTED_SEED): profile_summary(rows)}
        t_summary = {"duration_2.0s": profile_summary(rows)}
        geometry_status = "NOT_QUALIFIED" if plan.get("k_star") is None else "QUALIFIED_CANDIDATE_NOT_RUN"
        timing_status = "NOT_QUALIFIED" if plan.get("t_star") is None else "QUALIFIED_CANDIDATE_NOT_RUN"
        final_classification = (
            "GEOMETRY_REPRESENTATION_INSUFFICIENT"
            if core_pass and plan.get("k_star") is None
            else "WBC_BOTTLENECK_BEFORE_STRUCTURED_GR00T"
        )
        result_payload = {
            "schema": "STAGE36_WBC_ORACLE_RESULT_V2",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "project": EXPECTED_PROJECT,
            "reference_mode": EXPECTED_REFERENCE_MODE,
            "scientific_execution_begun": scientific_execution_begun,
            "row_count": len(rows),
            "maximum_rows": MAX_ROWS,
            "all_finite": bool(rows) and all(row["finite"] for row in rows),
            "core24": {
                "row_count": len(core_rows),
                "tracking_pass": core_pass,
                "status": "PASS" if core_pass else "FAIL_OR_PARTIAL",
            },
            "continuous8": {
                "row_count": len(continuous_rows),
                "tracking_pass": continuous_pass,
                "status": "PASS" if continuous_pass else "FAIL_OR_PARTIAL",
            },
            "geometry": {"k_star": plan.get("k_star"), "status": geometry_status},
            "timing": {"t_star": plan.get("t_star"), "status": timing_status},
            "motion_api_classification": geometry_status,
            "continuous_base_classification": "NATIVE_WBC_PASS" if continuous_pass else "NATIVE_WBC_FAIL_OR_PARTIAL",
            "final_classification": final_classification,
            "per_family_summary": family_summary,
            "per_seed_summary": seed_summary,
            "per_t_summary": t_summary,
            "profile_metrics": [row["profile_metrics"] for row in rows],
            "safety_gate_transitions": [
                {"row_id": row["row_id"], "transitions": row["safety_gate"]["transitions"]}
                for row in rows
            ],
            "finite_flags": [row["finite_flags"] for row in rows],
            "rows": rows,
            "forbidden_execution": {
                "training": False,
                "backward": False,
                "optimizer_step": False,
                "scheduler_step": False,
                "resume": False,
                "new_checkpoint": False,
                "sampler": False,
                "get_action": False,
                "gr00t": False,
                "hardware": False,
                "second_allocation": False,
            },
            "next_gate": NEXT_GATE,
        }
        write_json(output_root / "stage36_result.json", result_payload)
        write_json(
            output_root / "stage36_raw_hash_manifest.json",
            {
                "schema": "STAGE36_RAW_HASH_MANIFEST_V1",
                "timestamp_utc": now_utc(),
                "rows": [
                    {
                        "row_id": row["row_id"],
                        "summary_path": str(Path(row["output_dir"]) / "native_external_summary.json"),
                        "summary_sha256": sha256_file(Path(row["output_dir"]) / "native_external_summary.json")
                        if (Path(row["output_dir"]) / "native_external_summary.json").is_file()
                        else None,
                        "raw_rows_path": row["raw_rows_path"],
                        "raw_rows_sha256": sha256_file(Path(row["raw_rows_path"]))
                        if Path(row["raw_rows_path"]).is_file()
                        else None,
                    }
                    for row in rows
                ],
            },
        )
        write_json(
            output_root / "stage36_result_manifest.json",
            {
                "schema": "STAGE36_RESULT_MANIFEST_V1",
                "result_path": str(output_root / "stage36_result.json"),
                "result_sha256": sha256_file(output_root / "stage36_result.json"),
                "raw_hash_manifest_path": str(output_root / "stage36_raw_hash_manifest.json"),
                "raw_hash_manifest_sha256": sha256_file(output_root / "stage36_raw_hash_manifest.json"),
                "row_count": len(rows),
                "all_finite": result_payload["all_finite"],
                "final_classification": final_classification,
            },
        )
        status = "PASS_SCIENTIFIC_EXECUTION_NEEDS_PI_REVIEW" if core_pass else "PARTIAL_OR_FAILED_NEEDS_PI_REVIEW"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        log_event(output_root, "ERROR", **error)
        result_payload = {
            "schema": "STAGE36_WBC_ORACLE_RESULT_V2",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "scientific_execution_begun": scientific_execution_begun,
            "row_count": len(rows),
            "rows": rows,
            "all_finite": False,
            "status": "PARTIAL_OR_BLOCKED",
            "error": error,
            "next_gate": NEXT_GATE,
        }
        write_json(output_root / "stage36_result.json", result_payload)
        write_json(
            output_root / "stage36_raw_hash_manifest.json",
            {"schema": "STAGE36_RAW_HASH_MANIFEST_V1", "rows": [], "partial": True},
        )
        write_json(
            output_root / "stage36_result_manifest.json",
            {
                "schema": "STAGE36_RESULT_MANIFEST_V1",
                "result_path": str(output_root / "stage36_result.json"),
                "result_sha256": sha256_file(output_root / "stage36_result.json"),
                "partial": True,
            },
        )
    # The campaign log is complete before sealing.  No evidence is deleted or
    # rewritten if a row failed, and there is no replacement-run branch.
    seal_payload = {
        "schema": "STAGE36_ARTIFACT_SEAL_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "output_root": str(output_root),
        "entries": seal_entries(output_root, ("stage36_artifact_seal.json", "stage36_return.json")),
        "source_artifacts": [
            {
                "role": "evaluation_runner",
                "path": str(args.native_runner),
                "size_bytes": args.native_runner.stat().st_size,
                "sha256": sha256_file(args.native_runner),
            },
            {
                "role": "pbs_wrapper",
                "path": str(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
                "size_bytes": (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").stat().st_size,
                "sha256": sha256_file(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
            },
        ],
        "historical_inputs_mutated": False,
        "second_allocation_submitted": False,
    }
    seal_path = output_root / "stage36_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha = sha256_file(seal_path)
    final_return = make_return(output_root, result_payload, seal_path, seal_sha, status, error)
    write_json(output_root / "stage36_return.json", final_return)
    return 0 if status == "PASS_SCIENTIFIC_EXECUTION_NEEDS_PI_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
