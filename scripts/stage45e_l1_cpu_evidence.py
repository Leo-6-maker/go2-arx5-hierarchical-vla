#!/usr/bin/env python3
"""Validate a completed recovered-primary L1 evaluation using CPU/read-only I/O.

This validator reads the evaluator's final result, seal, CPU summary, raw-first
inventory, and recovered-input bridge.  It never imports Torch/GR00T, opens a
checkpoint payload, or executes a model.  Its output is written outside the
sealed evaluation root so the evaluator's immutable inventory remains valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


FINAL_TOKEN = "STAGE45E_L1_VANDA_H200_TRAIN24_COMPLETE_FOR_PI_REVIEW"
EVAL_SCHEMA = "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_V1"
EVAL_SEAL_SCHEMA = "VANDA_STAGE45E_L1_TRAIN24_E4_EVALUATION_SEAL_V1"
CPU_SUMMARY_SCHEMA = "VANDA_STAGE45E_L1_TRAIN24_E4_CPU_SUMMARY_V1"
RECOVERED_SCHEMA = "STAGE45E_L1_RECOVERED_TRAINING_EVAL_INPUT_V2"
ARM_ORDER = ("FULL24@2000", "L1@1000", "L1@2000")
EXPECTED_COUNTS = {
    "model_construction": 3,
    "conditioning_passes": 72,
    "action_head_calls": 288,
    "model_forward": 288,
    "raw_prediction_count": 72,
    "derived_row_count": 72,
    "public_sampler_calls": 0,
    "get_action_calls": 0,
    "training_steps": 0,
    "backward_steps": 0,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
    "retry_count": 0,
}


class CpuEvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CpuEvidenceError(message)


def read_json(path: Path) -> Any:
    path = Path(path).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"JSON unavailable or unsafe: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CpuEvidenceError(f"invalid JSON: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_fresh(path: Path, value: Any) -> None:
    path = Path(path).expanduser().resolve()
    require(not path.exists() and not path.is_symlink(), f"CPU evidence output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)


def _inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    inventory_path = inventory_path.expanduser().resolve()
    require(inventory_path.is_file() and not inventory_path.is_symlink(), "immutable inventory missing")
    lines = inventory_path.read_text(encoding="utf-8").splitlines()
    require(lines and all(line.strip() for line in lines), "immutable inventory is empty or malformed")
    entries: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, f"malformed inventory line: {line!r}")
        digest, relative = parts
        require(relative not in {"stage45e_l1_eval_seal.json", "stage45e_l1_eval_result.json", "inventory/IMMUTABLE_SHA256SUMS"}, "final artifacts must be excluded from inventory")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"inventory entry unavailable: {relative}")
        observed = sha256_file(path)
        require(observed == digest, f"immutable inventory hash drift: {relative}")
        entries.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return {
        "path": str(inventory_path),
        "sha256": sha256_file(inventory_path),
        "entry_count": len(entries),
        "entries": entries,
        "final_seal_excluded": True,
        "final_result_excluded": True,
    }


def _rows(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    bundles = summary.get("row_bundles")
    require(isinstance(bundles, Mapping), "CPU summary row-bundle block missing")
    require(tuple(bundles) == ARM_ORDER, "CPU summary row-bundle arm order drift")
    result: dict[str, Any] = {}
    reference_windows: list[str] | None = None
    for arm in ARM_ORDER:
        record = bundles[arm]
        require(isinstance(record, Mapping), f"row-bundle record missing: {arm}")
        path = root / str(record.get("path"))
        rows = read_json(path)
        require(isinstance(rows, list) and len(rows) == 24, f"row-bundle count drift: {arm}")
        require(sha256_file(path) == record.get("sha256"), f"row-bundle hash drift: {arm}")
        window_ids = [str(row.get("window_id")) for row in rows if isinstance(row, Mapping)]
        require(len(window_ids) == 24 and len(set(window_ids)) == 24, f"row-bundle window identity drift: {arm}")
        if reference_windows is None:
            reference_windows = window_ids
        else:
            require(window_ids == reference_windows, f"row-bundle window order drift: {arm}")
        result[arm] = {"path": str(path), "sha256": record.get("sha256"), "row_count": len(rows), "window_ids": window_ids}
    return result


def validate(eval_result_path: Path, output_path: Path, expected_head: str | None = None) -> dict[str, Any]:
    eval_result_path = Path(eval_result_path).expanduser().resolve()
    result = read_json(eval_result_path)
    require(result.get("schema") == EVAL_SCHEMA, "evaluation result schema drift")
    require(result.get("status") == FINAL_TOKEN and result.get("token") == FINAL_TOKEN, "evaluation final token/status missing")
    if expected_head is not None:
        require(result.get("execution_head") == expected_head, "evaluation execution HEAD drift")
    eval_root = eval_result_path.parent
    seal_path = Path(str(result.get("evaluation_seal"))).expanduser().resolve()
    require(seal_path == eval_root / "stage45e_l1_eval_seal.json", "evaluation seal path drift")
    seal = read_json(seal_path)
    require(seal.get("schema") == EVAL_SEAL_SCHEMA and seal.get("status") == "PASS", "evaluation seal is not PASS")
    require(result.get("evaluation_seal_sha256") == sha256_file(seal_path), "evaluation seal hash drift")
    require(seal.get("execution_head") == result.get("execution_head"), "seal/result HEAD mismatch")
    require(seal.get("branch") == "stage45e/l1-physical-error-aware-loss", "evaluation branch drift")
    require(seal.get("training") is False and seal.get("backward") is False, "evaluation seal claims update work")
    require(seal.get("optimizer_steps") == 0 and seal.get("checkpoint_writes") == 0, "evaluation seal update counters drift")
    require(seal.get("validation_e_read") is False and seal.get("holdout_f_read") is False, "locked split was read")

    counters = result.get("observed_counters")
    require(isinstance(counters, Mapping), "evaluation observed counters missing")
    for field, expected in EXPECTED_COUNTS.items():
        require(counters.get(field) == expected, f"evaluation counter drift: {field}")
    require(result.get("call_budget", {}).get("conditioning_passes") == 72, "conditioning budget drift")
    require(result.get("call_budget", {}).get("action_head_calls") == 288, "action-head budget drift")
    require(result.get("call_budget", {}).get("total_model_side_calls") == 360, "total model-side budget drift")
    require(result.get("training") is False and result.get("automatic_retry") is False, "evaluation result boundary drift")
    require(result.get("validation_e_read") is False and result.get("holdout_f_read") is False, "evaluation result locked-split drift")

    cpu_summary_path = Path(str(result.get("cpu_summary"))).expanduser().resolve()
    require(cpu_summary_path == eval_root / "cpu_summary.json", "CPU summary path drift")
    cpu_summary = read_json(cpu_summary_path)
    require(cpu_summary.get("schema") == CPU_SUMMARY_SCHEMA and cpu_summary.get("status") == "PASS_READONLY_DERIVED_FROM_SEALED_RAW", "CPU summary is not PASS")
    require(result.get("cpu_summary_sha256") == sha256_file(cpu_summary_path), "CPU summary hash drift")
    require(cpu_summary.get("checkpoint_order") == list(ARM_ORDER), "CPU checkpoint order drift")
    require(cpu_summary.get("window_count_per_checkpoint") == 24, "CPU window count drift")
    require(cpu_summary.get("seed") == 4242, "CPU evaluation seed drift")
    require(cpu_summary.get("t_values") == [0.0, 0.25, 0.5, 0.75], "CPU E4 t-grid drift")
    require(cpu_summary.get("dt") == 0.25, "CPU E4 dt drift")
    pairs = cpu_summary.get("paired_summary", {}).get("pairs")
    require(isinstance(pairs, Mapping), "paired summary missing")
    require(set(pairs) == {"L1@1000_minus_FULL24@2000", "L1@2000_minus_FULL24@2000", "L1@2000_minus_L1@1000"}, "paired summary pair set drift")
    for name, pair in pairs.items():
        require(set(pair) >= {"overall", "by_target", "by_window"}, f"paired summary granularity missing: {name}")
        require(pair["overall"].get("window_count") == 24, f"paired overall window count drift: {name}")
        require(len(pair["by_window"]) == 24, f"paired by-window count drift: {name}")

    rows = _rows(eval_root, cpu_summary)
    inventory_path = Path(str(result.get("inventory_path"))).expanduser().resolve()
    require(inventory_path == eval_root / "inventory/IMMUTABLE_SHA256SUMS", "inventory path drift")
    inventory = _inventory(eval_root, inventory_path)
    require(result.get("inventory_sha256") == inventory["sha256"], "inventory hash drift")
    require(result.get("inventory_entry_count") == inventory["entry_count"], "inventory entry count drift")
    require(seal.get("inventory_sha256") == inventory["sha256"], "seal inventory hash drift")
    require(seal.get("inventory_entry_count") == inventory["entry_count"], "seal inventory count drift")

    recovered_path = Path(str(result.get("recovered_eval_input"))).expanduser().resolve()
    recovered = read_json(recovered_path)
    require(recovered.get("schema") == RECOVERED_SCHEMA, "recovered-input schema drift")
    require(recovered.get("status") == "PASS_CONTRACT_BOUND_RECOVERED_TRAINING_INPUT_FOR_FUTURE_EVAL_ADMISSION", "recovered-input status drift")
    require(recovered.get("evaluation_authorized_by_this_bridge") is False, "bridge claimed evaluation authority")
    require(result.get("recovered_eval_input_sha256") == sha256_file(recovered_path), "recovered-input hash drift")

    evidence = {
        "schema": "STAGE45E_L1_TRAIN24_CPU_EVIDENCE_V1",
        "status": "PASS_READONLY_SEAL_AND_PAIRED_EVIDENCE",
        "evaluation_result": str(eval_result_path),
        "evaluation_result_sha256": sha256_file(eval_result_path),
        "execution_head": result.get("execution_head"),
        "branch": seal.get("branch"),
        "pbs_jobid": seal.get("pbs_jobid"),
        "gpu": seal.get("gpu"),
        "recovered_eval_input": str(recovered_path),
        "recovered_eval_input_sha256": sha256_file(recovered_path),
        "call_budget": dict(result["call_budget"]),
        "observed_counters": dict(counters),
        "row_bundles": rows,
        "paired_summary_keys": sorted(pairs),
        "cpu_summary": str(cpu_summary_path),
        "cpu_summary_sha256": sha256_file(cpu_summary_path),
        "evaluation_seal": str(seal_path),
        "evaluation_seal_sha256": sha256_file(seal_path),
        "inventory": inventory,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "automatic_retry": False,
        "claim_boundary": "CPU/read-only validation of the sealed 360-call evaluation; descriptive paired evidence only, with no automatic winner claim.",
    }
    write_fresh(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head")
    args = parser.parse_args(argv)
    try:
        result = validate(args.eval_result, args.output, args.expected_head)
    except Exception as exc:
        print(f"STAGE45E_L1_CPU_EVIDENCE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
