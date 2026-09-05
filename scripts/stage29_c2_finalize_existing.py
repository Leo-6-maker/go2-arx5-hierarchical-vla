#!/usr/bin/env python3
"""Zero-forward closure for six preserved Stage29 sampler records.

This script is only for the post-sampler summary-key failure in PBS job
601708.hopper-m-02.  It validates and summarizes already-written records and
artifacts.  It has no Torch/Transformers import and no model, sampler, qsub,
training, WBC, or hardware path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage29_c2_free_running_transfer as stage29


SCHEMA = "STAGE29_C2_POSTHOC_EXISTING_RECORDS_V1"
EXECUTION_SHA = "07290d042c24996b93d046edf514f91a786f52ba"
PBS_JOB_ID = "601708.hopper-m-02"
EXPECTED_ORDER = tuple((seed, checkpoint) for seed in stage29.SEEDS for checkpoint in stage29.CHECKPOINTS)


class FinalizeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FinalizeError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _git_show_bytes(repo: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative_path}"])


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_existing_records(partial: Mapping[str, Any]) -> list[dict[str, Any]]:
    if partial.get("schema") != stage29.SCHEMA or partial.get("status") != "PARTIAL_EVIDENCE":
        raise FinalizeError("partial evidence schema/status mismatch")
    records = [dict(row) for row in partial.get("records", [])]
    if partial.get("record_count") != stage29.MAX_SAMPLER_CALLS or len(records) != stage29.MAX_SAMPLER_CALLS:
        raise FinalizeError("existing sampler record count is not exactly six")
    observed_order = tuple((int(row["seed"]), str(row["checkpoint"])) for row in records)
    if observed_order != EXPECTED_ORDER:
        raise FinalizeError(f"sampler record order drifted: {observed_order}")
    if [int(row["sampler_call_index"]) for row in records] != list(range(1, 7)):
        raise FinalizeError("sampler call indices are not exactly 1..6")
    if len({str(row["conditioning_sha256"]) for row in records}) != 1:
        raise FinalizeError("conditioning hashes differ across records")
    if not _all_finite(records):
        raise FinalizeError("existing records contain non-finite JSON numbers")
    for row in records:
        if row.get("metrics", {}).get("all_finite") is not True:
            raise FinalizeError("record finite flag is not true")
        if row.get("initial_noise", {}).get("finite") is not True:
            raise FinalizeError("initial noise finite flag is not true")
        if row.get("sampler", {}).get("steps") != 4 or row.get("sampler", {}).get("rtc") is not False:
            raise FinalizeError("stock sampler contract drifted")
        artifact = row["artifact"]
        path = Path(artifact["path"])
        if path.stat().st_size != int(artifact["bytes"]) or _sha256_file(path) != artifact["sha256"]:
            raise FinalizeError(f"sampler artifact drifted: {path}")
    stage29.validate_paired_noise(records)
    return records


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact_entry(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"role": role, "path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256_file(resolved)}


def _git_artifact_entry(
    repo: Path,
    revision: str,
    relative_path: str,
    expected_sha256: str,
    role: str,
) -> dict[str, Any]:
    content = _git_show_bytes(repo, revision, relative_path)
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise FinalizeError(f"historical Git artifact SHA mismatch for {relative_path}: {observed}")
    return {
        "role": role,
        "path": f"git:{revision}:{relative_path}",
        "bytes": len(content),
        "sha256": observed,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.pbs_job_id != PBS_JOB_ID or args.execution_sha != EXECUTION_SHA:
        raise FinalizeError("execution job/SHA binding mismatch")
    head = _git(args.project_repo, "rev-parse", "HEAD")
    if head != args.expected_posthoc_sha or _git(args.project_repo, "status", "--short"):
        raise FinalizeError("post-hoc checkout SHA/cleanliness mismatch")
    for path, expected, label in (
        (args.partial_records, args.expected_partial_sha256, "partial records"),
        (args.preflight_evidence, args.expected_preflight_sha256, "preflight evidence"),
        (args.pbs_log, args.expected_log_sha256, "PBS log"),
        (args.source_audit_report, args.expected_source_audit_sha256, "source audit"),
    ):
        observed = _sha256_file(path)
        if observed != expected:
            raise FinalizeError(f"{label} SHA mismatch: {observed}")
    execution_runner_entry = _git_artifact_entry(
        args.project_repo,
        EXECUTION_SHA,
        "scripts/stage29_c2_free_running_transfer.py",
        args.expected_execution_runner_sha256,
        "execution_runner_at_execution_sha",
    )
    execution_pbs_entry = _git_artifact_entry(
        args.project_repo,
        EXECUTION_SHA,
        "hpc/pbs/stage29_c2_free_running_transfer.pbs",
        args.expected_execution_pbs_sha256,
        "execution_pbs_at_execution_sha",
    )
    partial = _read_json(args.partial_records)
    records = validate_existing_records(partial)
    preflight = _read_json(args.preflight_evidence)
    if preflight.get("status") != "PASS_PREFLIGHT_NO_MODEL":
        raise FinalizeError("preflight evidence is not PASS")
    stage28_result = _read_json(args.stage28_result)
    historical = stage29.stage28_historical_context(stage28_result)
    paired_noise = stage29.validate_paired_noise(records)
    expert_profiles = [row["metrics"]["expert_profile"] for row in records]
    expert_hashes = {_canonical_hash(profile) for profile in expert_profiles}
    if len(expert_hashes) != 1:
        raise FinalizeError("expert metric profile differs across preserved records")
    result = {
        "schema": stage29.SCHEMA,
        "status": "PASS_SCIENTIFIC_EXECUTION_POSTHOC_CLOSURE",
        "stage": "Stage29 C2 official free-running sampler transfer gate",
        "pi_gate": stage29.PI_GATE_ID,
        "execution_project_sha": EXECUTION_SHA,
        "posthoc_repair_sha": head,
        "posthoc_repair_reason": "six sampler calls completed; original runner then read nonexistent Stage28 summary keys",
        "pbs": {
            "job_id": PBS_JOB_ID,
            "project": stage29.EXPECTED_PROJECT,
            "requested_gpu": "1x NVIDIA H200",
            "compute_host": "hopper-12",
            "exit_status": 2,
            "exit_status_interpretation": "post-sampler summary serialization failure",
        },
        "preflight": preflight,
        "conditioning": {
            "sha256": records[0]["conditioning_sha256"],
            "equal_across_all_six_calls": True,
        },
        "expert_physical_action": {
            "metrics": expert_profiles[0],
            "metrics_sha256": next(iter(expert_hashes)),
            "same_locked_TINY1_sample": True,
        },
        "paired_initial_noise": paired_noise,
        "records": records,
        "paired_aggregates": stage29.paired_aggregates(records),
        "stage28_teacher_forced_historical_context": historical,
        "execution": {
            "scientific_execution_begun": True,
            "sampler_calls": 6,
            "base_sampler_calls": 3,
            "h2_sampler_calls": 3,
            "maximum_sampler_calls": 6,
            "posthoc_model_calls": 0,
            "replacement_qsub": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "o1_rerun": False,
            "stage28_rerun": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "all_finite": True,
        "scientific_contract_changed": False,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "claim_boundary": "single locked TINY1 training sample; paired descriptive free-running transfer evidence only",
        "next_gate": stage29.NEXT_GATE,
    }
    result_path = args.output_root / "stage29_c2_free_running_result.json"
    _write_json_atomic(result_path, result)
    raw_manifest_path = args.output_root / "stage29_c2_raw_hash_manifest.json"
    _write_json_atomic(raw_manifest_path, {
        "schema": stage29.SCHEMA,
        "source": "six preserved records from 601708.hopper-m-02",
        "records": [
            {
                "checkpoint": row["checkpoint"],
                "seed": row["seed"],
                "initial_noise_raw_sha256": row["initial_noise"]["raw_sha256"],
                "raw_universal_sha256": row["raw_universal_action"]["sha256"],
                "decoded_physical_sha256": row["decoded_physical_action"]["sha256"],
                "artifact_sha256": row["artifact"]["sha256"],
            }
            for row in records
        ],
    })
    state_path = args.output_root / "stage29_c2_execution_state.json"
    state = _read_json(state_path)
    state.update({
        "status": "PASS_SCIENTIFIC_EXECUTION_POSTHOC_CLOSURE",
        "completed_sampler_calls": 6,
        "posthoc_model_calls": 0,
        "replacement_qsub": False,
        "posthoc_repair_sha": head,
        "updated_at_utc": _utc_now(),
    })
    _write_json_atomic(state_path, state)

    seal_inputs = [
        *sorted(path for path in args.output_root.iterdir() if path.is_file() and path.name != "stage29_c2_artifact_seal.json"),
        args.pbs_log,
        args.source_audit_report,
        Path(__file__).resolve(),
        args.finalizer_test,
    ]
    entries = [execution_runner_entry, execution_pbs_entry]
    seen = set()
    for path in seal_inputs:
        resolved = path.resolve(strict=True)
        if str(resolved) not in seen:
            seen.add(str(resolved))
            entries.append(_artifact_entry(resolved, "stage29_evidence"))
    seal = {
        "schema": "STAGE29_C2_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "execution_project_sha": EXECUTION_SHA,
        "posthoc_repair_sha": head,
        "pbs_job_id": PBS_JOB_ID,
        "artifact_count": len(entries),
        "artifacts": entries,
        "posthoc_model_calls": 0,
        "replacement_qsub": False,
        "scientific_contract_changed": False,
        "next_gate": stage29.NEXT_GATE,
    }
    seal_path = args.output_root / "stage29_c2_artifact_seal.json"
    _write_json_atomic(seal_path, seal)
    return {
        "result": result,
        "result_artifact": _artifact_entry(result_path, "result"),
        "raw_manifest_artifact": _artifact_entry(raw_manifest_path, "raw_hash_manifest"),
        "artifact_seal": {**_artifact_entry(seal_path, "artifact_seal"), "artifact_count": len(entries)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, required=True)
    parser.add_argument("--expected-posthoc-sha", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--pbs-job-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partial-records", type=Path, required=True)
    parser.add_argument("--preflight-evidence", type=Path, required=True)
    parser.add_argument("--stage28-result", type=Path, required=True)
    parser.add_argument("--pbs-log", type=Path, required=True)
    parser.add_argument("--source-audit-report", type=Path, required=True)
    parser.add_argument("--finalizer-test", type=Path, required=True)
    parser.add_argument("--expected-partial-sha256", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--expected-log-sha256", required=True)
    parser.add_argument("--expected-execution-runner-sha256", required=True)
    parser.add_argument("--expected-execution-pbs-sha256", required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = finalize(_parser().parse_args(argv))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "posthoc_model_calls": 0,
            "replacement_qsub": False,
        }, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
