#!/usr/bin/env python3
"""PI0236 outer evaluator for the frozen PI0228 TRAIN-CALIB protocol.

The selector remains unchanged. This wrapper separates execution integrity
from the selector's scientific profile status, so a complete SafetyGate
REJECT is preserved as a negative result and does not stop later profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEPS = (1000, 2500, 5000, 7500, 10000)
FRESH_STEPS = (2500, 5000, 7500, 10000)
FRESH_ROOT = Path("/scratch/e1192548/pi0228_h200_large_adapt_v1_623593.hopper-m-02")
BASELINE = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/checkpoint-1000")
CALIB = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_calib_dataset")
TRAIN = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_dataset")
STAGE10 = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902")
GROOT = STAGE10 / "src" / "isaac-gr00t"
MODALITY = Path("/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py")
COSMOS = Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561")
STAGE38 = Path("/scratch/e1192548/pi0228_h200_prep_20260919/scripts/stage38_structured_train.py")
SELECTOR = Path("/scratch/e1192548/pi0228_h200_prep_20260919/scripts/pi0228_train_calib_checkpoint_selection.py")
SEAL = Path("/scratch/e1192548/pi0228_h200_prep_20260919/evidence/pi0228_train_calib_seal.json")
AUTHORITY = Path("/scratch/e1192548/pi0228_h200_prep_20260919/evidence/pi0228_train_calib_target_authority.json")
EXPECTED_FORWARD_COUNT = 1088
MODEL_INPUT_KEYS = ["body_rgb", "wrist_rgb", "state"]
PROFILE_SCHEMA = "PI0228_TRAIN_CALIB_PROFILE_V1"
DECISIONS = ("ACCEPT", "WARN", "REJECT")


class EvaluationContractError(RuntimeError):
    """A profile or admission is not safe to consume."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reject_locked(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in ("dev-e", "dev_e", "holdout-f", "holdout_f", "final-f", "final_f")):
        raise EvaluationContractError(f"locked DEV/F path is inadmissible: {path}")


def _hex_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise EvaluationContractError(f"{label} is not a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise EvaluationContractError(f"{label} is not hexadecimal") from exc


def _load_expected_rows(calib_dataset: Path, authority_path: Path | None = None) -> list[dict[str, Any]]:
    _reject_locked(calib_dataset)
    path = calib_dataset / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise EvaluationContractError("TRAIN-CALIB episode indices are not contiguous")
    if len(rows) != EXPECTED_FORWARD_COUNT:
        raise EvaluationContractError(f"TRAIN-CALIB row count is {len(rows)}, expected {EXPECTED_FORWARD_COUNT}")
    if authority_path is not None:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        targets = {str(item["source_rollout_id"]): str(item["target_id"]) for item in authority["rollouts"]}
        for row in rows:
            rollout_id, separator, _ = str(row["source_episode_id"]).partition(":obs-")
            if not separator or rollout_id not in targets:
                raise EvaluationContractError(f"missing target authority for {row['source_episode_id']}")
            row["source_rollout_id"] = rollout_id
            row["target_id"] = targets[rollout_id]
    return rows


def _profile_role(step: int) -> str:
    return "BASELINE_COMPARISON_ONLY" if step == 1000 else "FRESH_SELECTION_CANDIDATE"


def profile_is_admissible(payload: dict[str, Any]) -> bool:
    """Mirror only the selector's eligibility predicate; do not rank here."""
    return (
        payload.get("status") == "PASS"
        and payload.get("safety_gate_counts", {}).get("REJECT", 1) == 0
        and payload.get("forward_count") == payload.get("expected_forward_count")
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _check_stats(payload: dict[str, Any], records: list[dict[str, Any]]) -> None:
    accepted = [
        float(record["reference_min_target_distance_m"])
        for record in records
        if record["safety_gate"]["decision"] != "REJECT"
    ]
    summary = payload.get("reference_min_target_distance_m")
    if not isinstance(summary, dict) or summary.get("count") != len(accepted):
        raise EvaluationContractError("reference-distance summary count mismatch")
    if accepted:
        if not math.isclose(float(summary.get("median")), _median(accepted), rel_tol=1e-12, abs_tol=1e-12):
            raise EvaluationContractError("reference-distance median mismatch")
        if not math.isclose(float(summary.get("max")), max(accepted), rel_tol=1e-12, abs_tol=1e-12):
            raise EvaluationContractError("reference-distance max mismatch")
    elif summary.get("median") is not None or summary.get("max") is not None:
        raise EvaluationContractError("all-REJECT profile must retain null ranking statistics")

    cells: dict[str, list[float]] = {}
    for record in records:
        if record["safety_gate"]["decision"] != "REJECT":
            cells.setdefault(str(record["source_rollout_id"]), []).append(float(record["reference_min_target_distance_m"]))
    expected_cells = {key: _median(values) for key, values in cells.items()}
    if payload.get("cell_medians_reference_min_target_distance_m") != expected_cells:
        raise EvaluationContractError("source-rollout cell medians mismatch")
    expected_worst = max(expected_cells.values()) if expected_cells else None
    actual_worst = payload.get("worst_cell_median_reference_min_target_distance_m")
    if expected_worst is None:
        if actual_worst is not None:
            raise EvaluationContractError("all-REJECT profile must retain null worst-cell statistic")
    elif not math.isclose(float(actual_worst), expected_worst, rel_tol=1e-12, abs_tol=1e-12):
        raise EvaluationContractError("worst-cell median mismatch")


def validate_profile_payload(
    payload: dict[str, Any], *, step: int, expected_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fail closed on structure while preserving a complete scientific REJECT."""
    if payload.get("schema") != PROFILE_SCHEMA:
        raise EvaluationContractError(f"profile {step} schema mismatch")
    if int(payload.get("checkpoint_step", -1)) != step:
        raise EvaluationContractError(f"profile {step} checkpoint step mismatch")
    if payload.get("action_contract") != "official_dense_relative_H40" or payload.get("action_dt_s") != 0.05:
        raise EvaluationContractError(f"profile {step} action contract mismatch")
    if payload.get("seed") != 22301 or payload.get("target_xyz_in_model_input") is not False or payload.get("wbc_execution") is not False:
        raise EvaluationContractError(f"profile {step} scientific boundary mismatch")
    if payload.get("model_input_keys") != MODEL_INPUT_KEYS:
        raise EvaluationContractError(f"profile {step} model input contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(expected_rows):
        raise EvaluationContractError(f"profile {step} record count mismatch")
    if payload.get("forward_count") != len(expected_rows) or payload.get("expected_forward_count") != len(expected_rows):
        raise EvaluationContractError(f"profile {step} forward count mismatch")
    if [record.get("episode_index") for record in records] != list(range(len(expected_rows))):
        raise EvaluationContractError(f"profile {step} episode order mismatch")
    for expected, record in zip(expected_rows, records):
        for field in ("source_episode_id", "source_rollout_id", "target_id"):
            if str(record.get(field)) != str(expected.get(field, "")):
                raise EvaluationContractError(f"profile {step} {field} mismatch at episode {expected['episode_index']}")
        if record.get("target_xyz_in_model_input") is not False or record.get("wbc_execution") is not False:
            raise EvaluationContractError(f"profile {step} row boundary mismatch")
        for field in ("raw_sha256", "decoded_sha256"):
            _hex_digest(record.get(field), f"profile {step} row {field}")
        gate = record.get("safety_gate")
        if not isinstance(gate, dict) or gate.get("decision") not in DECISIONS:
            raise EvaluationContractError(f"profile {step} unknown SafetyGate decision")

    counts = {decision: sum(record["safety_gate"]["decision"] == decision for record in records) for decision in DECISIONS}
    if payload.get("safety_gate_counts") != counts:
        raise EvaluationContractError(f"profile {step} SafetyGate count mismatch")
    expected_status = "PASS" if counts["REJECT"] == 0 else "SAFETY_GATE_REJECT"
    if payload.get("status") != expected_status:
        raise EvaluationContractError(f"profile {step} status/count mismatch")
    _check_stats(payload, records)
    return {
        "step": step,
        "role": _profile_role(step),
        "scientific_profile_status": payload["status"],
        "safety_gate_counts": counts,
        "forward_count": len(records),
        "profile_is_admissible": profile_is_admissible(payload),
    }


def _source_ref_matches(ref: dict[str, Any]) -> None:
    path = Path(str(ref["path"]))
    _reject_locked(path)
    if not path.is_file():
        raise EvaluationContractError(f"admission source missing: {path}")
    if sha256_file(path) != ref["sha256"]:
        raise EvaluationContractError(f"admission source hash mismatch: {path}")


def validate_admission(admission_path: Path) -> dict[str, Any]:
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("status") != "PASS" or admission.get("schema") != "PI0236_S2_EVALUATOR_ADMISSION_V1":
        raise EvaluationContractError("PI0236 S2 admission is not PASS")
    if admission.get("scientific_contract_changed") is not False:
        raise EvaluationContractError("scientific contract is not frozen")
    runtime = admission.get("runtime_contract", {})
    expected_runtime = {
        "device": "cuda:0", "seed": 22301, "action_contract": "official_dense_relative_H40",
        "action_dt_s": 0.05, "model_input_keys": MODEL_INPUT_KEYS,
        "target_xyz_in_model_input": False, "wbc_execution": False,
        "dev_e_source_read_count": 0, "final_f_source_read_count": 0,
    }
    if runtime != expected_runtime:
        raise EvaluationContractError("runtime contract mismatch")
    if admission.get("evaluation_order") != list(STEPS) or admission.get("expected_forward_count_per_profile") != EXPECTED_FORWARD_COUNT:
        raise EvaluationContractError("evaluation order or forward count mismatch")
    _source_ref_matches(admission["runner"])
    if sha256_file(Path(__file__)) != admission["runner"]["sha256"]:
        raise EvaluationContractError("runner identity mismatch")
    for ref in admission.get("source_files", []):
        _source_ref_matches(ref)
    for ref in (admission["calib"]["seal"], admission["calib"]["authority"]):
        _source_ref_matches(ref)
    canary = admission.get("canary", {})
    if canary.get("status") != "PASS_GATE_INVOCATION_WITH_REJECT" or canary.get("safety_gate_decision") != "REJECT" or canary.get("forward_count") != 1:
        raise EvaluationContractError("623912 canary qualification mismatch")
    if canary.get("wbc_execution") is not False or admission.get("exact_child_preflight", {}).get("status") != "PASS":
        raise EvaluationContractError("canary boundary or exact-child preflight mismatch")
    if admission.get("cpu_regression", {}).get("status") != "PASS":
        raise EvaluationContractError("CPU regression is not PASS")
    baseline = admission["baseline"]
    baseline_path = Path(baseline["path"])
    if not baseline_path.is_dir() or not baseline_path.joinpath("trainer_state.json").is_file():
        raise EvaluationContractError("baseline checkpoint is missing")
    for profile in admission["profiles"]:
        if int(profile["checkpoint_step"]) not in FRESH_STEPS:
            raise EvaluationContractError("unexpected fresh checkpoint in admission")
        path = Path(profile["path"])
        if not path.is_dir() or not path.joinpath("trainer_state.json").is_file() or not list(path.glob("*.safetensors.index.json")):
            raise EvaluationContractError(f"fresh checkpoint is incomplete: {path}")
    return admission


def bind_cache(cache_root: Path) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"] + "/hub"
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(GROOT))
    spec = importlib.util.spec_from_file_location("pi0236_stage38", STAGE38)
    if spec is None or spec.loader is None:
        raise EvaluationContractError("cannot load Stage38 binding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"cache": module.bind_cosmos_cache(cache_root), "processor": module.bind_cosmos_processor()}


def run_profile(step: int, model_path: Path, output: Path, *, expected_rows: list[dict[str, Any]], state_path: Path) -> dict[str, Any]:
    command = [
        sys.executable, str(SELECTOR), "--mode", "evaluate", "--output", str(output),
        "--stage10-root", str(STAGE10), "--base-model", str(BASELINE), "--model-path", str(model_path),
        "--train-dataset", str(TRAIN), "--calib-dataset", str(CALIB), "--modality-config", str(MODALITY),
        "--cosmos-snapshot", str(COSMOS), "--seal", str(SEAL), "--target-authority", str(AUTHORITY),
        "--checkpoint-step", str(step), "--device", "cuda:0", "--seed", "22301",
    ]
    stdout_path = output.parent / f"profile-{step}.stdout.txt"
    stderr_path = output.parent / f"profile-{step}.stderr.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(state_path, {"status": "PROFILE_RUNNING", "step": step, "started_at": now_utc(), "command": command, "stdout": str(stdout_path), "stderr": str(stderr_path)})
    stdout_path.touch()
    stderr_path.touch()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, text=True, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise EvaluationContractError(f"profile {step} subprocess failed with returncode {result.returncode}")
    if not output.is_file():
        raise EvaluationContractError(f"profile {step} output is missing")
    payload = json.loads(output.read_text(encoding="utf-8"))
    summary = validate_profile_payload(payload, step=step, expected_rows=expected_rows)
    summary.update({"path": str(output), "sha256": sha256_file(output), "stdout": str(stdout_path), "stderr": str(stderr_path), "completed_at": now_utc()})
    _atomic_json(output.parent / f"profile-{step}.validation.json", summary)
    _atomic_json(state_path, {"status": "PROFILE_COMPLETE", **summary})
    return summary


def _failure_manifest(output_root: Path, *, step: int | None, error: BaseException, state_path: Path) -> None:
    _atomic_json(output_root / "pi0236_evaluation_failure.json", {
        "schema": "PI0236_TRAIN_CALIB_EVALUATION_FAILURE_V1", "status": "ENGINEERING_BLOCKED", "step": step,
        "error_type": type(error).__name__, "error": str(error), "known_forward_count": "UNKNOWN",
        "wbc_execution": False, "dev_e_source_read_count": 0, "final_f_source_read_count": 0, "created_at": now_utc(),
    })
    _atomic_json(state_path, {"status": "ENGINEERING_BLOCKED", "step": step, "error": str(error), "updated_at": now_utc()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise EvaluationContractError(f"evaluation output root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "pi0236_evaluation_state.json"
    step: int | None = None
    try:
        validate_admission(args.admission)
        expected_rows = _load_expected_rows(CALIB, AUTHORITY)
        binding = bind_cache(args.cache_root)
        profiles: list[dict[str, Any]] = []
        for step in STEPS:
            model_path = BASELINE if step == 1000 else FRESH_ROOT / "training" / f"checkpoint-{step}"
            profiles.append(run_profile(step, model_path, args.output_root / f"profile-{step}.json", expected_rows=expected_rows, state_path=state_path))
        payload = {
            "schema": "PI0236_TRAIN_CALIB_EVALUATION_RUN_V1", "status": "EVALUATION_COMPLETE", "execution_status": "COMPLETE",
            "admission": {"path": str(args.admission), "sha256": sha256_file(args.admission)}, "runtime_binding": binding,
            "evaluation_order": list(STEPS), "expected_forward_count_per_profile": EXPECTED_FORWARD_COUNT, "profiles": profiles,
            "safety_gate_reject_total": sum(item["safety_gate_counts"]["REJECT"] for item in profiles), "selection_pending": True,
            "wbc_execution": False, "target_xyz_in_model_input": False, "dev_e_source_read_count": 0, "final_f_source_read_count": 0,
        }
        final_path = args.output_root / "pi0236_train_calib_evaluation_run.json"
        _atomic_json(final_path, payload)
        _atomic_json(state_path, {"status": "EVALUATION_COMPLETE", "profiles": profiles, "updated_at": now_utc()})
        print(json.dumps({"status": payload["status"], "output": str(final_path), "sha256": sha256_file(final_path)}, indent=2))
        return 0
    except BaseException as exc:
        _failure_manifest(args.output_root, step=step, error=exc, state_path=state_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
