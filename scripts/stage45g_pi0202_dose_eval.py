#!/usr/bin/env python3
"""PI0202 dose scorecard using the existing Stage43 native evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(os.environ["PI0200_REPO"]).resolve()
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
import stage45f_pi0194_candidate_eval as native_eval
import stage43_corelab_evidence as stage43_evidence


_arm_text = os.environ.get("PI0208_ARMS", "C80,S25,S50")
ARMS = tuple(item.strip() for item in _arm_text.split(",") if item.strip())
if not ARMS or any(item not in {"C80", "S25", "S50"} for item in ARMS) or len(set(ARMS)) != len(ARMS):
    raise RuntimeError("PI0208_ARMS must contain unique values from C80,S25,S50")
CONDITIONS = tuple(native_eval.CONDITION_ORDER)
RUNNER = REPO / "scripts" / "stage43_corelab_initial_condition.py"
EVAL_ROOT = Path(os.environ["PI0200_EVAL_ROOT"]).resolve()
BASE_BUNDLE = Path(os.environ["PI0200_STAGE43_BUNDLE"]).resolve()
TRAINING_ROOT = Path(os.environ["PI0200_TRAINING_ROOT"]).resolve()
MODEL_PYTHON = Path(os.environ.get("PI0200_MODEL_PYTHON", str(REPO / "scripts" / "stage45f_pi0197_model_python.sh"))).resolve()
GPU_UUID = os.environ["PI0200_GPU_UUID"]
CURRENT_HEAD = os.environ["PI0200_HEAD"]
CORRECTED_RUNNER_SHA256 = os.environ["PI0200_CORRECTED_RUNNER_SHA256"]
SOURCE_SHA256 = os.environ["PI0200_STAGE43_SOURCE_SHA256"]
EXPECTED_BUNDLE_SHA256 = "127d95a2a5b9a6a40e864e029d85fd09a9e4dd55441aae58071502aa76700da6"
EXPECTED_SOURCE_SHA256 = "4d85ebfb3c69ceb59bd8a7e22ea0c24f8a92f5af96ce1d3e5d78ecc8829edf73"
EXPECTED_STATS_SHA256 = "430cfc259357259791e069e564fc4b5754cbeb144b720c5289c032fe62d80e3d"
FROZEN_SOURCE_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
EXPECTED_SCHEDULE_SHA256 = "03634d5adec0366f1f8550f8cb10f48cc0096dc6e2d69f4d9de4a90c10376bb7"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def plain_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def training_specs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    for arm in ARMS:
        arm_root = TRAINING_ROOT / arm
        result_path = arm_root / "training_result.json"
        invocation_path = arm_root / "training_invocation.json"
        result = plain_json(result_path)
        require(result.get("status") == "PASS" and result.get("arm") == arm, f"{arm} training result is not PASS")
        require(int(result.get("global_step", -1)) == 80 and int(result.get("updates", -1)) == 80, f"{arm} step count drift")
        require(result.get("schedule_sha256") == EXPECTED_SCHEDULE_SHA256, f"{arm} schedule binding drift")
        checkpoint = arm_root / "checkpoint-80"
        source_checkpoint = Path(str(result.get("final_checkpoint", "")))
        require(source_checkpoint.name == "checkpoint-80" and source_checkpoint.parent.name == arm, f"{arm} checkpoint identity path drift")
        require(invocation_path.is_file() and not invocation_path.is_symlink(), f"{arm} training invocation is missing")
        require(result.get("model_forward") == 80 and result.get("backward") == 80 and result.get("optimizer_steps") == 80, f"{arm} training counters drift")
        require(result.get("identities", {}).get("structured_stats_sha256") == FROZEN_SOURCE_STATS_SHA256, f"{arm} frozen statistics source drift")
        stats_path = checkpoint / "experiment_cfg" / "structured_action_statistics.json"
        stats = plain_json(stats_path)
        frozen_stats = stats.get("statistics", stats)
        require(stats.get("schema") == "STAGE38_D_STRUCTURED_TRAINING_SIDE_CAR_V1", f"{arm} structured statistics schema drift")
        require(frozen_stats.get("source_file_sha256") == FROZEN_SOURCE_STATS_SHA256, f"{arm} structured statistics provenance drift")
        require(frozen_stats.get("statistics_recomputed") is False, f"{arm} structured statistics were recomputed")
        require(native_eval.sha256_file(stats_path) == EXPECTED_STATS_SHA256, f"{arm} checkpoint statistics bytes drift")
        spec = {
            "label": f"{arm}@80",
            "tree_sha256": str(result["checkpoint_tree_sha256"]),
            "tree_prefix": str(result["checkpoint_tree_prefix"]),
            "global_step": 80,
            "stats_sha256": EXPECTED_STATS_SHA256,
            "training_result_sha256": native_eval.sha256_file(result_path),
            "training_invocation_sha256": native_eval.sha256_file(invocation_path),
        }
        specs[arm] = spec
        records[arm] = {
            "training_result_path": str(result_path),
            "training_result_sha256": spec["training_result_sha256"],
            "training_invocation_path": str(invocation_path),
            "training_invocation_sha256": spec["training_invocation_sha256"],
            "checkpoint_path": str(checkpoint),
            "source_checkpoint_path": str(source_checkpoint),
            "checkpoint_tree_sha256": spec["tree_sha256"],
            "checkpoint_tree_prefix": spec["tree_prefix"],
        }
    return specs, records


def child_command(*, base: Mapping[str, Any], overlay_path: Path, candidate: str, condition: str, condition_root: Path, binding: Mapping[str, Any], expected_head: str, gpu_uuid: str, sim_device: str, seed: int) -> list[str]:
    values = [
        "--bundle", str(overlay_path), "--output-dir", str(condition_root), "--condition-id", condition,
        "--repository-head", str(base["repository"]["head"]), "--gpu-uuid", gpu_uuid,
        "--source-root", str(base["artifacts"]["wbc_source"]["path"]), "--trajectory", str(base["artifacts"]["trajectory"]["path"]),
        "--arm", str(base["artifacts"]["wbc_arm_checkpoint"]["path"]), "--dog", str(base["artifacts"]["wbc_dog_checkpoint"]["path"]),
        "--stage10-root", str(base["artifacts"]["stage10_root"]["path"]), "--base-model-path", str(base["artifacts"]["base_model_path"]["path"]),
        "--model-path", str(binding["candidate_paths"][candidate]), "--dataset-dir", str(base["artifacts"]["dataset_dir"]["path"]),
        "--modality-config", str(base["artifacts"]["modality_config"]["path"]), "--cosmos-snapshot", str(base["artifacts"]["cosmos_snapshot_path"]["path"]),
        "--structured-stats", str(Path(binding["candidate_paths"][candidate]) / "experiment_cfg" / "structured_action_statistics.json"),
        "--model-python", str(binding["paths"]["model_python"]), "--wbc-source-root", str(base["artifacts"]["wbc_source"]["path"]),
        "--sim-device", sim_device, "--seed", str(seed), "--response-timeout-s", "300.0", "--worker-ready-timeout-s", "1200.0",
    ]
    argv = [str(RUNNER)] + values
    bootstrap = "\n".join([
        "import runpy, sys",
        "from pathlib import Path",
        "import stage43_corelab_evidence as _e",
        "import go2arx5_vla.stage40_structured_decode as _d",
        f"_e.STRUCTURED_CHECKPOINT_TREE_SHA256 = {native_eval.CANDIDATES[candidate]['tree_sha256']!r}",
        f"_e.STRUCTURED_STATS_SHA256 = {native_eval.CANDIDATES[candidate]['stats_sha256']!r}",
        f"_d.EXPECTED_STRUCTURED_STATS_SHA256 = {native_eval.CANDIDATES[candidate]['stats_sha256']!r}",
        "_original_verify = _e.verify_stage43_bundle_files",
        "def _verify(bundle, repo_root=None):",
        "    result = _original_verify(bundle, repo_root=None)",
        f"    if _e.sha256_file(Path({str(RUNNER)!r})) != {CORRECTED_RUNNER_SHA256!r}: raise RuntimeError('corrected runner SHA drift')",
        f"    if _e.sha256_file(Path({str(REPO / 'scripts' / 'stage43_corelab_evidence.py')!r})) != {SOURCE_SHA256!r}: raise RuntimeError('Stage43 evidence source SHA drift')",
        "    return result",
        "_e.verify_stage43_bundle_files = _verify",
        f"sys.argv = {argv!r}",
        f"runpy.run_path({str(RUNNER)!r}, run_name='__main__')",
    ])
    return [sys.executable, "-B", "-u", "-c", bootstrap]


def scorecard(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    c80 = aggregates.get("C80")
    s25 = aggregates.get("S25")
    s50 = aggregates.get("S50")
    if ARMS == ("S50",):
        s50_complete = (
            s50["cell_count"] == len(CONDITIONS)
            and s50["complete_cell_count"] == len(CONDITIONS)
            and s50["engineering_blocked_count"] == 0
        )
        return {
            "schema": "CORELAB_PI0208_G2_S50_DIAGNOSTIC_SCORECARD_V1",
            "classification": "S50_DESCRIPTIVE_COMPLETE" if s50_complete else "S50_DESCRIPTIVE_INCOMPLETE",
            "basis": "S50 is a descriptive dose diagnostic after the primary C80/S25 confirmation",
            "primary_comparison": None,
            "S50": s50,
            "S50_role": "descriptive_diagnostic_only",
            "automatic_promotion_gate": False,
        }
    require(c80 is not None and s25 is not None, "C80 and S25 are required for the primary scorecard")
    complete = all(
        item.get("cell_count") == len(CONDITIONS)
        and item.get("complete_cell_count") == len(CONDITIONS)
        and item.get("scientific_reject_count") == 0
        and item.get("scientific_abort_count") == 0
        and item.get("engineering_blocked_count") == 0
        for item in (c80, s25)
    )
    s25_wins = None
    c80_wins = None
    if complete:
        s25_wins = sum(float(s25["per_condition"][condition]["final_target_distance_m"]) < float(c80["per_condition"][condition]["final_target_distance_m"]) for condition in CONDITIONS)
        c80_wins = sum(float(c80["per_condition"][condition]["final_target_distance_m"]) < float(s25["per_condition"][condition]["final_target_distance_m"]) for condition in CONDITIONS)
        if float(s25["mean_progress_m"]["mean"]) > float(c80["mean_progress_m"]["mean"]) and float(s25["mean_final_distance_m"]["mean"]) < float(c80["mean_final_distance_m"]["mean"]) and s25_wins >= 3:
            classification = "LOW_DOSE_STATE_SUPPORT_SUPPORTED"
            basis = "S25 wins both coupled endpoint aggregates and at least three matched C0-C4 endpoints"
        elif float(s25["mean_progress_m"]["mean"]) < float(c80["mean_progress_m"]["mean"]) and float(s25["mean_final_distance_m"]["mean"]) > float(c80["mean_final_distance_m"]["mean"]) and c80_wins >= 3:
            classification = "LOW_DOSE_STATE_SUPPORT_NOT_SUPPORTED"
            basis = "C80 wins both coupled endpoint aggregates and at least three matched C0-C4 endpoints"
        else:
            classification = "MIXED_FOR_PI_REVIEW"
            basis = "frozen S25/C80 rule did not select one arm"
    else:
        classification = "MIXED_FOR_PI_REVIEW"
        basis = "C80 or S25 contains an incomplete, rejected, aborted, or engineering-blocked cell"
    return {
        "schema": "CORELAB_PI0208_G2_PRIMARY_SCORECARD_V1" if s50 is None else "CORELAB_PI0200_TERMINAL_SCORECARD_V1",
        "classification": classification,
        "basis": basis,
        "primary_comparison": "S25_vs_C80",
        "S25": s25,
        "C80": c80,
        "S50": s50,
        "matched_condition_S25_final_distance_wins": s25_wins,
        "matched_condition_C80_final_distance_wins": c80_wins,
        "S50_role": "descriptive_diagnostic_only" if s50 is not None else None,
        "automatic_promotion_gate": complete,
    }


def main() -> int:
    require(CURRENT_HEAD == os.environ["PI0202_EXPECTED_HEAD"], "PI0202 branch HEAD drift")
    require(native_eval.sha256_file(BASE_BUNDLE) == EXPECTED_BUNDLE_SHA256, "Stage43 bundle SHA drift")
    require(native_eval.sha256_file(REPO / "scripts" / "stage43_corelab_evidence.py") == EXPECTED_SOURCE_SHA256 == SOURCE_SHA256, "Stage43 evidence source drift")
    require(native_eval.sha256_file(RUNNER) == CORRECTED_RUNNER_SHA256, "corrected runner SHA drift")
    base = native_eval.read_json(BASE_BUNDLE, canonical=True)
    native_eval._verify_bundle_paths(base)
    stage43_evidence.verify_stage43_bundle_files(base, repo_root=None)
    specs, training_records = training_specs()
    native_eval.CANDIDATE_ORDER = ARMS
    native_eval.CANDIDATES = specs
    native_eval.MAX_CELLS = len(ARMS) * len(CONDITIONS)
    native_eval.MAX_FORWARDS = native_eval.MAX_CELLS * native_eval.FORWARDS_PER_CELL
    native_eval.MAX_UPDATES = native_eval.MAX_CELLS * native_eval.UPDATES_PER_CELL
    native_eval.MAX_TICKS = native_eval.MAX_CELLS * native_eval.TICKS_PER_CELL
    inventories = {
        arm: native_eval._verify_candidate_tree(TRAINING_ROOT / arm / "checkpoint-80", arm, {})
        for arm in ARMS
    }
    require(not EVAL_ROOT.exists(), f"evaluation root is not fresh: {EVAL_ROOT}")
    EVAL_ROOT.mkdir(parents=True)
    binding = {
        "schema": "CORELAB_PI0200_STAGE43_EVAL_BINDING_V1",
        "status": "READY",
        "execution_head": CURRENT_HEAD,
        "gpu_uuid": GPU_UUID,
        "paths": {"stage43_bundle": str(BASE_BUNDLE), "model_python": str(MODEL_PYTHON), "output_root": str(EVAL_ROOT)},
        "candidate_paths": {arm: str(TRAINING_ROOT / arm / "checkpoint-80") for arm in ARMS},
        "stage43_historical_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "stage43_historical_runner_sha256": base["executable_lineage"]["runner_sha256"],
        "authorized_corrected_runner_sha256": CORRECTED_RUNNER_SHA256,
        "stage43_evidence_source_sha256": SOURCE_SHA256,
        "training": training_records,
        "candidate_inventories": inventories,
        "model_loads": 0,
        "model_forwards": 0,
    }
    native_eval.write_json(EVAL_ROOT / "binding.json", binding)
    native_eval.write_json(EVAL_ROOT / "admission.json", {
        "schema": "CORELAB_PI0200_A40_EVAL_ADMISSION_V1",
        "status": "PASS",
        "gpu_uuid": GPU_UUID,
        "candidate_order": list(ARMS),
        "condition_order": list(CONDITIONS),
        "stage43_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "training": training_records,
        "runner_hash_transition": {"historical": base["executable_lineage"]["runner_sha256"], "authorized_corrected": CORRECTED_RUNNER_SHA256},
        "scientific_zeroes": {"training": 0, "backward": 0, "optimizer": 0, "Validation-E": 0, "Holdout-F": 0, "ICL": 0, "DAgger": 0, "hardware": 0},
    })
    old_tree = stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256
    old_stats = stage43_evidence.STRUCTURED_STATS_SHA256
    import go2arx5_vla.stage40_structured_decode as structured_decode
    old_decode_stats = structured_decode.EXPECTED_STRUCTURED_STATS_SHA256
    try:
        for arm in ARMS:
            stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = specs[arm]["tree_sha256"]
            stage43_evidence.STRUCTURED_STATS_SHA256 = specs[arm]["stats_sha256"]
            structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = specs[arm]["stats_sha256"]
            overlay = native_eval._make_overlay(base, arm, TRAINING_ROOT / arm / "checkpoint-80")
            stage43_evidence.validate_stage43_bundle(overlay)
            stage43_evidence.verify_stage43_bundle_files(overlay, repo_root=None)
            native_eval.write_json(EVAL_ROOT / "overlays" / f"{arm}.json", overlay)
    finally:
        stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = old_tree
        stage43_evidence.STRUCTURED_STATS_SHA256 = old_stats
        structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = old_decode_stats
    native_eval._child_command = child_command
    for arm in ARMS:
        (EVAL_ROOT / arm).mkdir(parents=True, exist_ok=True)
        stage43_evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = specs[arm]["tree_sha256"]
        stage43_evidence.STRUCTURED_STATS_SHA256 = specs[arm]["stats_sha256"]
        structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = specs[arm]["stats_sha256"]
        for condition in CONDITIONS:
            cell = native_eval._execute_cell(output_root=EVAL_ROOT, base=base, candidate=arm, condition=condition, binding=binding, gpu_uuid=GPU_UUID, sim_device="cuda:0", seed=4242, stage43_evidence=stage43_evidence)
            if cell.get("status") == "ENGINEERING_BLOCKED":
                raise RuntimeError(f"engineering cell block: {arm}/{condition}")
    aggregates = {arm: native_eval._aggregate_candidate(EVAL_ROOT / arm, arm) for arm in ARMS}
    card = scorecard(aggregates)
    card["training"] = training_records
    card["gpu_uuid"] = GPU_UUID
    card["execution_head"] = CURRENT_HEAD
    native_eval.write_json(EVAL_ROOT / "scorecard.json", card)
    evidence_sha = native_eval._write_evidence_manifest(EVAL_ROOT)
    seal = {
        "schema": "CORELAB_PI0200_RESULT_SEAL_V1",
        "status": "COMPLETED",
        "token": "LOCALPC_PI0200_E256_SUPPORT_DOSE_SCORECARD — " + card["classification"],
        "execution_head": CURRENT_HEAD,
        "gpu_uuid": GPU_UUID,
        "scorecard_sha256": native_eval.sha256_file(EVAL_ROOT / "scorecard.json"),
        "admission_sha256": native_eval.sha256_file(EVAL_ROOT / "admission.json"),
        "binding_sha256": native_eval.sha256_file(EVAL_ROOT / "binding.json"),
        "evidence_manifest_sha256": evidence_sha,
        "counters": {
            "model_loads": len(ARMS),
            "model_forward_cells": len(ARMS) * len(CONDITIONS),
            "gr00t_forwards": len(ARMS) * len(CONDITIONS) * native_eval.FORWARDS_PER_CELL,
            "reference_updates": len(ARMS) * len(CONDITIONS) * native_eval.UPDATES_PER_CELL,
            "native_ticks": len(ARMS) * len(CONDITIONS) * native_eval.TICKS_PER_CELL,
            "training": 0, "backward": 0, "optimizer": 0, "Validation-E": 0, "Holdout-F": 0,
            "WBC": 0, "simulator": 0, "hardware": 0,
        },
    }
    native_eval.write_json(EVAL_ROOT / "result_seal.json", seal)
    print(json.dumps({"status": card["classification"], "scorecard": str(EVAL_ROOT / "scorecard.json"), "evidence_manifest_sha256": evidence_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        raise
