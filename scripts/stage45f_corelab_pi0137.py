#!/usr/bin/env python3
"""PI0137 CPU-only pre-seal hardening and post-selection safeguards.

All campaign inputs accepted by this module are either explicitly marked
synthetic or immutable sealed metadata.  The module has no model, Torch,
NumPy, scheduler, GPU, Validation-E, Holdout-F, WBC, simulator, or hardware
execution surface.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import stage45f_corelab_evidence_convergence as convergence
import stage45f_corelab_pi0136 as pi0136


REPO = Path(__file__).resolve().parents[1]
HARDENING_MANIFEST = REPO / "manifests/STAGE45F_CORELAB_C10_VERIFIER_HARDENING_V1.json"
SHADOW_MANIFEST = REPO / "manifests/STAGE45F_CORELAB_VANDA_SCHEMA_SHADOW_AUDIT_V1.json"
CAPSULE_TEMPLATE = REPO / "manifests/STAGE45E_THREE_ROUTE_REPRODUCIBILITY_CAPSULE_TEMPLATE_V1.json"
BRANCH_COMPATIBILITY = REPO / "manifests/STAGE45F_CORELAB_BRANCH_HANDOFF_COMPATIBILITY_PI0137.json"
SYNTHETIC_TOKEN = "STAGE45F_CORELAB_C10_SYNTHETIC_REHEARSAL_PASS"
REAL_CORELAB_TOKEN = "STAGE45F_CORELAB_THREE_ROUTE_EVIDENCE_CONVERGENCE_READY_FOR_PI"
ZERO = "0" * 64
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch", "transformers", "gr00t", "jax", "tensorflow", "numpy",
    "pandas", "subprocess", "isaacgym", "omni",
}
REASON_CODES = {
    "MISSING_REQUIRED_ARM", "DUPLICATE_ARM_IDENTITY", "ARM_AXIS_MISMATCH",
    "CHECKPOINT_TREE_SHA_MISMATCH", "ROW_BUNDLE_SHA_MISMATCH",
    "UPSTREAM_ARTIFACT_SHA_MISMATCH", "INVENTORY_IMMUTABILITY_VIOLATION",
    "CAMPAIGN_ROOT_ESCAPE", "UNSAFE_PATH_ESCAPE",
    "EXECUTION_LINEAGE_IDENTITY_MISMATCH", "ROUTE_SCIENTIFIC_CONTRACT_DRIFT",
    "CALL_COUNTER_BUDGET_MISMATCH", "TRAIN24_ROW_CONTRACT_DRIFT",
    "NONFINITE_EVIDENCE", "LOCKED_SPLIT_CONTAMINATION",
    "PERFORMANCE_SELECTED_LINEAGE_REPLACEMENT", "UNSUPPORTED_SCHEMA_NO_HEURISTICS",
    "CAMPAIGN_IDENTITY_BYTE_DRIFT", "PI_SELECTION_AUTHORITY_MISSING",
    "PI_SELECTION_BINDING_DRIFT", "VALIDATION_GENERATION_ALREADY_ACCESSED",
    "VALIDATION_SELECTION_GENERATION_DRIFT", "HOLDOUT_FORBIDDEN",
}


class PI0137Failure(convergence.ConvergenceError):
    """Fail-closed error carrying a stable machine-readable reason code."""

    def __init__(self, reason_code: str, detail: str):
        if reason_code not in REASON_CODES:
            raise ValueError(f"unregistered PI0137 reason code: {reason_code}")
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


def reject(reason_code: str, detail: str) -> None:
    raise PI0137Failure(reason_code, detail)


def demand(condition: bool, reason_code: str, detail: str) -> None:
    if not condition:
        reject(reason_code, detail)


def _read(path: Path) -> Any:
    try:
        return convergence.read_json(Path(path).resolve())
    except convergence.ConvergenceError as exc:
        reject("UNSAFE_PATH_ESCAPE" if "locked" in str(exc).lower() else "UNSUPPORTED_SCHEMA_NO_HEURISTICS", str(exc))


def _write(path: Path, value: Any) -> None:
    convergence.write_json(Path(path).resolve(), value)


def _sha(value: Any, reason: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        reject(reason, f"{field} is not a lowercase SHA256")
    return value


def _git_sha(value: Any, reason: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        reject(reason, f"{field} is not a lowercase Git SHA")
    return value


def _contains_locked_token(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_locked_token(key) or _contains_locked_token(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_locked_token(item) for item in value)
    if isinstance(value, str):
        normalized = value.lower().replace("_", "-")
        return "validation-e" in normalized or "holdout-f" in normalized or "/holdout/" in normalized
    return False


def guarded_path(root: Path, raw: Any, *, must_exist: bool = True) -> Path:
    demand(isinstance(raw, str) and raw, "UNSAFE_PATH_ESCAPE", "path is missing")
    supplied = Path(raw)
    demand(not supplied.is_symlink(), "UNSAFE_PATH_ESCAPE", f"symlink input rejected: {supplied}")
    supplied_parts = {part.lower().replace("_", "-") for part in supplied.parts}
    demand(not supplied_parts.intersection({"validation-e", "holdout-f", "holdout"}), "LOCKED_SPLIT_CONTAMINATION", f"locked path token: {supplied}")
    resolved_root = root.resolve()
    resolved = supplied.resolve()
    resolved_parts = {part.lower().replace("_", "-") for part in resolved.parts}
    demand(not resolved_parts.intersection({"validation-e", "holdout-f", "holdout"}), "LOCKED_SPLIT_CONTAMINATION", f"resolved locked path token: {resolved}")
    demand(resolved.is_relative_to(resolved_root), "UNSAFE_PATH_ESCAPE", f"path escapes admitted root: {resolved}")
    if must_exist:
        demand(resolved.is_file() and not resolved.is_symlink(), "UNSAFE_PATH_ESCAPE", f"unsafe or missing file: {resolved}")
    return resolved


def hardening_manifest() -> dict[str, Any]:
    value = _read(HARDENING_MANIFEST)
    demand(value.get("schema") == "STAGE45F_CORELAB_C10_VERIFIER_HARDENING_V1", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "hardening manifest schema drift")
    cases = value.get("mutation_cases")
    demand(isinstance(cases, list) and len(cases) == 17, "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "exact 17-case mutation contract required")
    demand({item.get("reason_code") for item in cases}.issubset(REASON_CODES), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "unknown mutation reason code")
    return value


def shadow_audit() -> dict[str, Any]:
    value = _read(SHADOW_MANIFEST)
    demand(value.get("schema") == "STAGE45F_CORELAB_VANDA_SCHEMA_SHADOW_AUDIT_V1", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "shadow-audit schema drift")
    demand(value.get("status") == "PASS_EXPLICIT_ADAPTERS_PREPARED", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "shadow audit is not ready")
    policy = value.get("audit_policy", {})
    demand(policy.get("field_guessing") is False and policy.get("path_label_inference") is False and policy.get("consumer_checks_weakened") is False, "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "heuristic adapter policy detected")
    expected_heads = {
        "LEGACY": "9ad05f1bc7e2efc9f0d88674d569d42ef8b11e62",
        "L1": "186efc2977e2795bebf0dfe97d6f503a672716de",
        "N2": "6d8a7f132e325faea35346b36497ceb76390b2c3",
        "ORCHESTRATION": "5aa7b225da14bbfca053cde8041d6b85d6505a79",
    }
    surfaces = value.get("surfaces", {})
    demand(set(surfaces) == set(expected_heads), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "producer surface set drift")
    for name, head in expected_heads.items():
        demand(surfaces[name].get("head") == head, "EXECUTION_LINEAGE_IDENTITY_MISMATCH", f"{name} live HEAD binding drift")
        _git_sha(head, "EXECUTION_LINEAGE_IDENTITY_MISMATCH", f"{name}.head")
    demand(surfaces["ORCHESTRATION"].get("delta_from_pi0136_head") == "MECHANICAL_OUTPUT_LIFECYCLE_ONLY", "ROUTE_SCIENTIFIC_CONTRACT_DRIFT", "orchestration delta not mechanically classified")
    return {
        "schema": "STAGE45F_CORELAB_VANDA_SCHEMA_SHADOW_AUDIT_RESULT_V1",
        "status": "PASS_EXPLICIT_ADAPTERS_PREPARED",
        "manifest_sha256": convergence.sha256_file(SHADOW_MANIFEST),
        "heads": expected_heads,
        "scientific_semantics_changed": False,
        "field_guessing": False,
        "real_h200_campaign_ingested": False,
    }


def _route_contract(route: str) -> Mapping[str, Any]:
    return convergence.manifest()["h200_campaign_surfaces"]["routes"][route]


def _loss_identity(route: str) -> str:
    return str(_route_contract(route)["loss"])


def validate_synthetic_campaign(campaign_path: Path, identity_path: Path) -> dict[str, Any]:
    """Validate a synthetic campaign without heuristic field recovery."""
    campaign_path = Path(campaign_path).resolve()
    identity_path = Path(identity_path).resolve()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    demand(identity.get("schema") == "STAGE45F_SYNTHETIC_CAMPAIGN_IDENTITY_V1", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "synthetic identity schema drift")
    expected_bytes = _sha(identity.get("campaign_file_sha256"), "CAMPAIGN_IDENTITY_BYTE_DRIFT", "campaign_file_sha256")
    demand(convergence.sha256_file(campaign_path) == expected_bytes, "CAMPAIGN_IDENTITY_BYTE_DRIFT", "same campaign identity has changed bytes")
    value = json.loads(campaign_path.read_text(encoding="utf-8"))
    demand(value.get("schema") == "STAGE45F_SYNTHETIC_H200_CAMPAIGN_V1", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "unsupported campaign schema")
    demand(value.get("synthetic_fixture") is True and value.get("real_h200_campaign_ingested") is False, "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "fixture is not unmistakably synthetic")
    demand(not _contains_locked_token(value), "LOCKED_SPLIT_CONTAMINATION", "Validation-E/Holdout-F token in campaign")
    root = Path(value.get("campaign_root", "")).resolve()
    demand(root.is_dir() and not root.is_symlink(), "CAMPAIGN_ROOT_ESCAPE", "campaign root unavailable/unsafe")
    for field in ("summary", "inventory", "upstream_result"):
        raw_path = Path(str(value.get(field, "")))
        demand(raw_path.resolve().is_relative_to(root), "CAMPAIGN_ROOT_ESCAPE", f"{field} is outside campaign root")
        path = guarded_path(root, value.get(field))
        expected = _sha(value.get(f"{field}_sha256"), "UPSTREAM_ARTIFACT_SHA_MISMATCH", f"{field}_sha256")
        demand(convergence.sha256_file(path) == expected, "UPSTREAM_ARTIFACT_SHA_MISMATCH", f"{field} SHA drift")
    inventory = json.loads(guarded_path(root, value["inventory"]).read_text(encoding="utf-8"))
    demand(inventory.get("schema") == "STAGE45F_SYNTHETIC_INVENTORY_V1" and isinstance(inventory.get("entries"), list), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "inventory schema drift")
    for entry in inventory["entries"]:
        path = guarded_path(root, str(root / str(entry.get("path", ""))))
        demand(convergence.sha256_file(path) == entry.get("sha256") and path.stat().st_size == entry.get("bytes"), "INVENTORY_IMMUTABILITY_VIOLATION", f"inventoried bytes drift: {path}")
    demand(value.get("execution_head") == shadow_audit()["heads"]["ORCHESTRATION"], "EXECUTION_LINEAGE_IDENTITY_MISMATCH", "execution HEAD drift")
    phase_budgets = value.get("phase_budgets")
    for phase in ("E4", "D1", "D2A"):
        for route in convergence.ROUTES:
            demand(phase_budgets.get(phase, {}).get(route) == convergence._expected_phase_budget(phase, route), "CALL_COUNTER_BUDGET_MISMATCH", f"{phase}/{route} counter drift")
    arms = value.get("arms")
    demand(isinstance(arms, list), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "arm list missing")
    labels = [arm.get("arm") for arm in arms if isinstance(arm, Mapping)]
    demand(len(labels) == len(set(labels)), "DUPLICATE_ARM_IDENTITY", "duplicate arm identity")
    expected_labels = list(convergence.manifest()["matrix"]["arms"])
    missing = sorted(set(expected_labels) - set(labels))
    demand(not missing, "MISSING_REQUIRED_ARM", f"missing arms: {missing}")
    demand(set(labels) == set(expected_labels), "ARM_AXIS_MISMATCH", "unexpected arm identity")
    checked = 0
    for arm in arms:
        label = arm["arm"]
        try:
            route, seed, step = convergence._arm_axes(label)
        except convergence.ConvergenceError as exc:
            reject("ARM_AXIS_MISMATCH", str(exc))
        demand((arm.get("route"), arm.get("seed"), arm.get("step")) == (route, seed, step), "ARM_AXIS_MISMATCH", f"{label} axes drift")
        contract = _route_contract(route)
        demand(arm.get("route_branch") == contract["branch"] and arm.get("route_head") == contract["head"], "EXECUTION_LINEAGE_IDENTITY_MISMATCH", f"{label} route lineage drift")
        demand(arm.get("stats_sha256") == contract["stats_sha256"] and arm.get("normalization_identity") == contract["normalization"] and arm.get("loss_identity") == _loss_identity(route), "ROUTE_SCIENTIFIC_CONTRACT_DRIFT", f"{label} scientific contract drift")
        checkpoint = guarded_path(root, arm.get("checkpoint_file"))
        demand(convergence.sha256_file(checkpoint) == arm.get("checkpoint_tree_sha256"), "CHECKPOINT_TREE_SHA_MISMATCH", f"{label} checkpoint tree drift")
        rows_path = guarded_path(root, arm.get("rows_file"))
        demand(convergence.sha256_file(rows_path) == arm.get("rows_sha256"), "ROW_BUNDLE_SHA_MISMATCH", f"{label} row bundle drift")
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        demand(isinstance(rows, list) and len(rows) == 24, "TRAIN24_ROW_CONTRACT_DRIFT", f"{label} row count drift")
        windows = [row.get("window_id") for row in rows]
        targets = [row.get("target_id") for row in rows]
        demand(windows == list(convergence.WINDOW_IDS) and len(set(windows)) == 24, "TRAIN24_ROW_CONTRACT_DRIFT", f"{label} window identity drift")
        demand(targets == [convergence.TARGETS[index // 6] for index in range(24)], "TRAIN24_ROW_CONTRACT_DRIFT", f"{label} target order drift")
        demand(all(row.get("target_xyz_in_model_input") is False for row in rows), "TRAIN24_ROW_CONTRACT_DRIFT", f"{label} target XYZ input drift")
        finite_values = [row.get("synthetic_metric") for row in rows]
        demand(all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in finite_values), "NONFINITE_EVIDENCE", f"{label} non-finite evidence")
        lineage = arm.get("lineage", {})
        demand(lineage.get("performance_selected_retry") is False and lineage.get("duplicate_complete_lineages") == 0, "PERFORMANCE_SELECTED_LINEAGE_REPLACEMENT", f"{label} retry lineage policy drift")
        checked += 1
    return {
        "schema": "STAGE45F_CORELAB_C10_SYNTHETIC_CAMPAIGN_VERIFICATION_V1",
        "status": "PASS",
        "campaign_identity": identity.get("campaign_identity"),
        "campaign_file_sha256": expected_bytes,
        "checked_arms": checked,
        "reason_code": None,
        "real_h200_campaign_ingested": False,
    }


def _synthetic_row(index: int, label: str, offset: float = 0.0) -> dict[str, Any]:
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "window_id": convergence.WINDOW_IDS[index],
        "target_id": convergence.TARGETS[index // 6],
        "checkpoint": {"label": label, "sha256": ZERO},
        "model": {"label": label, "sha256": ZERO},
        "metrics": {"target": {
            "final_target_distance_m": 0.20 + offset,
            "absolute_progress_m": 0.10 - offset,
            "minimum_target_distance_m": 0.15 + offset,
            "endpoint_target_vector_cosine": 0.80 - offset,
            "path_length_m": 0.30 + offset,
            "roughness_position_second_difference_m": {"mean": 0.01 + offset},
            "progress_sign": "POSITIVE",
        }},
        "normalized_metrics": {"active_rmse": 0.2 + offset, "active_mae": 0.1 + offset, "active_max_abs": 0.4 + offset, "raw_active_outside_neg1_pos1_fraction": 0.0},
        "sampler": {"steps_observed": 4, "official_public_sampler_called": False},
        "safety_gate_decision": None,
        "target_xyz_in_model_input": False,
        "finite": True,
        "synthetic_metric": 0.123 + offset,
    }


def _build_evidence_index(root: Path, *, reversed_order: bool = False) -> Path:
    cfg = convergence.manifest()
    frozen = cfg["h200_campaign_surfaces"]["frozen_sources"]
    labels = list(cfg["matrix"]["arms"])
    if reversed_order:
        labels.reverse()
    arms: dict[str, Any] = {}
    for arm_index, label in enumerate(labels):
        route, seed, step = convergence._arm_axes(label)
        path = root / f"rows_{route}_{seed}_{step}.json"
        if not path.exists():
            _write(path, [_synthetic_row(index, label, arm_index * 0.0001) for index in range(24)])
        route_cfg = cfg["h200_campaign_surfaces"]["routes"][route]
        arms[label] = {
            "rows_json": str(path), "rows_sha256": convergence.sha256_file(path),
            "provenance": {
                "route": route, "seed": seed, "step": step, "row_checkpoint_label": label,
                "checkpoint_tree_sha256": ZERO, "evidence_seal_sha256": ZERO,
                "lineage_artifact_sha256": ZERO, "training_result_sha256": ZERO,
                "trainer_state_sha256": ZERO, "hardware_runtime_sha256": ZERO,
                "training_recipe_sha256": ZERO, "route_branch": route_cfg["branch"],
                "route_head": route_cfg["head"], "normalization_identity": route_cfg["normalization"],
                "stats_sha256": route_cfg["stats_sha256"], "stats_payload_sha256": route_cfg["stats_payload_sha256"],
                "hardware": "H200", "decoder_identity": "accepted Stage45C decoder",
                "metric_identity": "accepted Stage44R metrics", "sampler_identity": "accepted E4",
                "raw_before_derived": True, "full_state_checkpoint": True, "completed_step": step,
                "target_xyz_in_model_input": False, "base_tree_sha256": frozen["base_tree_sha256"],
                "full24_tree_sha256": frozen["full24_tree_sha256"], "validation_e_read": False,
                "holdout_f_read": False,
            },
        }
    path = root / ("evidence_index_reversed.json" if reversed_order else "evidence_index.json")
    _write(path, {"schema": "STAGE45F_H200_TRAIN24_EVIDENCE_INDEX_V1", "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False, "arms": arms})
    return path


def _build_lineage_ledger(root: Path) -> Path:
    arms = [{
        "arm": label, "audit_state": "PASS", "accepted_lineage_id": f"SYNTHETIC-{label}",
        "checkpoint_tree_sha256": ZERO, "training_result_sha256": ZERO,
        "trainer_state_sha256": ZERO, "lineage_artifact_sha256": ZERO,
        "duplicate_complete_lineages": 0,
    } for label in convergence.manifest()["matrix"]["arms"]]
    path = root / "lineage_ledger.json"
    _write(path, {"schema": "CORELAB_STAGE45F_LINEAGE_ACCEPTANCE_LEDGER_V1", "status": "PASS", "policy": {"performance_based_retry_selection": False, "duplicate_complete_lineage_requires_pi": True}, "arms": arms, "pass_arm_count": 18, "blocked_arm_count": 0})
    return path


def _build_hardware_index(root: Path) -> Path:
    cfg = convergence.manifest()
    hw_cfg = cfg["n2_cross_hardware"]
    frozen = cfg["h200_campaign_surfaces"]["frozen_sources"]
    labels = [f"N2(seed{seed})@{step}" for seed in convergence.SEEDS for step in convergence.STEPS]
    arms: dict[str, dict[str, Any]] = {hardware: {} for hardware in pi0136.HARDWARES}
    for hw_index, hardware in enumerate(pi0136.HARDWARES):
        for label in labels:
            _, seed, step = convergence._arm_axes(label)
            rows = root / f"hardware_{hardware}_{seed}_{step}.json"
            _write(rows, [_synthetic_row(index, label, hw_index * 0.001) for index in range(24)])
            arms[hardware][label] = {"rows_json": str(rows), "rows_sha256": convergence.sha256_file(rows), "provenance": {
                "route": "N2", "seed": seed, "step": step, "hardware": hardware,
                "route_head": hw_cfg["historical_4080s_head"] if hardware == "RTX4080S" else hw_cfg["h200_head"],
                "stats_sha256": hw_cfg["accepted_stats_file_sha256_by_hardware"][hardware], "stats_payload_sha256": hw_cfg["stats_payload_sha256"],
                "base_tree_sha256": frozen["base_tree_sha256"], "full24_tree_sha256": frozen["full24_tree_sha256"],
                "full_state_checkpoint": True, "completed_step": step, "target_xyz_in_model_input": False,
                "row_checkpoint_label": label, "checkpoint_tree_sha256": ZERO, "evidence_seal_sha256": ZERO,
                "lineage_artifact_sha256": ZERO, "training_result_sha256": ZERO, "trainer_state_sha256": ZERO,
                "training_recipe_identity": {**cfg["h200_campaign_surfaces"]["training_recipe"], "seed": seed},
                "science_source_identity": {key: frozen[key] for key in ("groot_commit", "cosmos_revision", "cosmos_tree_sha256", "modality_sha256", "config_sha256")},
                "device_runtime_identity": {"device": hardware, "synthetic": True},
            }}
    path = root / "hardware_index.json"
    _write(path, {"schema": "STAGE45F_N2_CROSS_HARDWARE_EVIDENCE_INDEX_V1", "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False, "arms": arms})
    return path


def _build_mechanism_index(root: Path, train_packet: Path, hardware_packet: Path) -> Path:
    arms = {}
    for label in convergence.manifest()["matrix"]["arms"]:
        route, seed, step = convergence._arm_axes(label)
        arms[label] = {"route": route, "seed": seed, "step": step, "checkpoint_tree_sha256": ZERO,
            "D1": {"seal_sha256": ZERO, "rows_sha256": ZERO, "row_count": 216, "raw_before_derived": True, "target_xyz_in_model_input": False, "metrics": {"teacher_forced_local_flow_rmse": 0.2, "teacher_forced_local_flow_cosine": 0.8}},
            "D2A": {"seal_sha256": ZERO, "rows_sha256": ZERO, "row_count": 96, "raw_before_derived": True, "target_xyz_in_model_input": False, "metrics": {"tf_to_fr_state_path_divergence_l2": 0.3, "field_sensitivity_l2": 0.2, "teacher_bridge_local_error_l2": 0.1, "off_path_diagnostic_l2": 0.4}}}
    path = root / "mechanism_index.json"
    _write(path, {"schema": "STAGE45F_H200_MECHANISM_EVIDENCE_INDEX_V1", "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False, "train24_decision_packet": str(train_packet), "train24_decision_packet_sha256": convergence.sha256_file(train_packet), "n2_hardware_packet": str(hardware_packet), "n2_hardware_packet_sha256": convergence.sha256_file(hardware_packet), "arms": arms})
    return path


def _build_contract_campaign(root: Path, evidence_index: Path) -> tuple[Path, Path]:
    index = json.loads(evidence_index.read_text(encoding="utf-8"))
    checkpoint = root / "synthetic_checkpoint.bin"
    checkpoint.write_bytes(b"PI0137 synthetic checkpoint identity only\n")
    summary = root / "synthetic_summary.json"
    result = root / "synthetic_upstream_result.json"
    _write(summary, {"schema": "STAGE45F_SYNTHETIC_SUMMARY_V1", "status": "PASS_SYNTHETIC_ONLY"})
    _write(result, {"schema": "STAGE45F_SYNTHETIC_UPSTREAM_RESULT_V1", "status": "PASS_SYNTHETIC_ONLY"})
    inventory = root / "synthetic_inventory.json"
    entries = [{"path": path.name, "sha256": convergence.sha256_file(path), "bytes": path.stat().st_size} for path in (summary, result)]
    _write(inventory, {"schema": "STAGE45F_SYNTHETIC_INVENTORY_V1", "entries": entries})
    arms = []
    for label in convergence.manifest()["matrix"]["arms"]:
        route, seed, step = convergence._arm_axes(label)
        spec = index["arms"][label]
        contract = _route_contract(route)
        arms.append({"arm": label, "route": route, "seed": seed, "step": step,
            "route_branch": contract["branch"], "route_head": contract["head"],
            "normalization_identity": contract["normalization"], "stats_sha256": contract["stats_sha256"],
            "loss_identity": _loss_identity(route), "checkpoint_file": str(checkpoint),
            "checkpoint_tree_sha256": convergence.sha256_file(checkpoint), "rows_file": spec["rows_json"],
            "rows_sha256": spec["rows_sha256"], "lineage": {"performance_selected_retry": False, "duplicate_complete_lineages": 0}})
    campaign = root / "synthetic_campaign.json"
    payload = {"schema": "STAGE45F_SYNTHETIC_H200_CAMPAIGN_V1", "synthetic_fixture": True,
        "real_h200_campaign_ingested": False, "campaign_root": str(root),
        "execution_head": shadow_audit()["heads"]["ORCHESTRATION"], "summary": str(summary),
        "summary_sha256": convergence.sha256_file(summary), "inventory": str(inventory),
        "inventory_sha256": convergence.sha256_file(inventory), "upstream_result": str(result),
        "upstream_result_sha256": convergence.sha256_file(result), "phase_budgets": {
            phase: {route: convergence._expected_phase_budget(phase, route) for route in convergence.ROUTES}
            for phase in ("E4", "D1", "D2A")}, "arms": arms}
    _write(campaign, payload)
    identity = root / "synthetic_campaign_identity.json"
    _write(identity, {"schema": "STAGE45F_SYNTHETIC_CAMPAIGN_IDENTITY_V1", "campaign_identity": "PI0137-SYNTHETIC-CAMPAIGN-1", "campaign_file_sha256": convergence.sha256_file(campaign)})
    return campaign, identity


def _scientific_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: packet[key] for key in ("arm_summaries", "paired_comparisons", "three_seed_descriptive", "failure_taxonomy")}


def run_synthetic_rehearsal(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    demand(not root.exists(), "UNSAFE_PATH_ESCAPE", "synthetic rehearsal output must be fresh")
    root.mkdir(parents=True)
    evidence = _build_evidence_index(root)
    lineage = _build_lineage_ledger(root)
    campaign, identity = _build_contract_campaign(root, evidence)
    campaign_verification = validate_synthetic_campaign(campaign, identity)
    readiness = pi0136.build_readiness(evidence, root / "readiness.json", lineage)
    demand(readiness["counts"]["READY_SEALED"] == 18, "MISSING_REQUIRED_ARM", "synthetic readiness did not close 18/18")
    train_one = convergence.build_packet(evidence, root / "train_packet_1.json")
    train_two = convergence.build_packet(evidence, root / "train_packet_2.json")
    demand(convergence.canonical_sha(train_one) == convergence.canonical_sha(train_two), "INVENTORY_IMMUTABILITY_VIOLATION", "identical ingestion is not idempotent")
    reversed_index = _build_evidence_index(root, reversed_order=True)
    reversed_packet = convergence.build_packet(reversed_index, root / "train_packet_reversed.json")
    demand(convergence.canonical_sha(_scientific_payload(train_one)) == convergence.canonical_sha(_scientific_payload(reversed_packet)), "DUPLICATE_ARM_IDENTITY", "candidate ordering changed aggregate output")
    hardware_index = _build_hardware_index(root)
    hardware = pi0136.compare_n2_hardware(hardware_index, root / "hardware_packet.json")
    mechanism_index = _build_mechanism_index(root, root / "train_packet_1.json", root / "hardware_packet.json")
    mechanism = pi0136.build_mechanism_packet(mechanism_index, root / "mechanism_packet.json")
    neutral = {"schema": "STAGE45F_CORELAB_SYNTHETIC_NEUTRAL_DECISION_PACKET_V1", "status": "PASS_SYNTHETIC_ONLY", "decision_state": "PI_DECISION_REQUIRED", "winner_declared": False, "train_packet_sha256": convergence.sha256_file(root / "train_packet_1.json"), "hardware_packet_sha256": convergence.sha256_file(root / "hardware_packet.json"), "mechanism_packet_sha256": convergence.sha256_file(root / "mechanism_packet.json"), "real_h200_campaign_ingested": False}
    _write(root / "neutral_decision_packet.json", neutral)
    recipe_template = {"schema": "STAGE45F_PI_SELECTION_ARTIFACT_TEMPLATE_V1", "status": "UNBOUND_PI_DECISION_REQUIRED", "selected_route": None, "selected_checkpoint_set": None, "winner_inferred": False, "validation_execution_enabled": False}
    _write(root / "recipe_freeze_input_template.json", recipe_template)
    validation = {"schema": "STAGE45F_VALIDATION_E_SYNTHETIC_PREFLIGHT_V1", "status": "LOCKED_PLAN_ONLY", "accessed": False, "execution_enabled": False, "paths_opened": False, "generation": 1}
    _write(root / "validation_preflight.json", validation)
    mutated = root / "synthetic_campaign_mutated.json"
    changed = campaign.read_text(encoding="utf-8") + "\n"
    mutated.write_text(changed, encoding="utf-8")
    byte_drift_reason = None
    try:
        validate_synthetic_campaign(mutated, identity)
    except PI0137Failure as exc:
        byte_drift_reason = exc.reason_code
    demand(byte_drift_reason == "CAMPAIGN_IDENTITY_BYTE_DRIFT", "CAMPAIGN_IDENTITY_BYTE_DRIFT", "changed bytes were not rejected under same identity")
    artifact = {
        "schema": "STAGE45F_CORELAB_C10_SYNTHETIC_REHEARSAL_V1", "status": "PASS",
        "token": SYNTHETIC_TOKEN, "real_h200_campaign_ingested": False,
        "campaign_verification": campaign_verification, "readiness_counts": readiness["counts"],
        "idempotent_reingestion": True, "candidate_order_invariant": True,
        "same_identity_changed_bytes_rejected": True, "changed_bytes_reason_code": byte_drift_reason,
        "train_decision_state": train_one["decision_state"], "hardware_pair_count": hardware["pair_count"],
        "mechanism_decision_state": mechanism["decision_state"], "neutral_decision_state": neutral["decision_state"],
        "recipe_freeze_template_unbound": True, "validation_preflight": "LOCKED_PLAN_ONLY",
        "output_hashes": {name: convergence.sha256_file(root / name) for name in ("readiness.json", "train_packet_1.json", "hardware_packet.json", "mechanism_packet.json", "neutral_decision_packet.json", "recipe_freeze_input_template.json", "validation_preflight.json")},
        "execution_boundary": {"gpu": 0, "model_construction": 0, "model_forward": 0, "training": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }
    demand(REAL_CORELAB_TOKEN not in json.dumps(artifact, sort_keys=True), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "synthetic artifact emitted real convergence token")
    _write(root / "rehearsal_result.json", artifact)
    artifact["artifact_sha256"] = convergence.sha256_file(root / "rehearsal_result.json")
    return artifact


def generate_post_selection(selection_path: Path, decision_packet_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    selection = _read(selection_path)
    demand(selection.get("schema") == "STAGE45F_PI_SELECTION_ARTIFACT_V1" and selection.get("status") == "PI_SELECTED", "PI_SELECTION_AUTHORITY_MISSING", "exact PI selection artifact required")
    demand(isinstance(selection.get("pi_authority_comment"), int) and selection["pi_authority_comment"] > 0 and isinstance(selection.get("pi_authority_token"), str) and selection["pi_authority_token"], "PI_SELECTION_AUTHORITY_MISSING", "PI comment/token missing")
    packet_path = Path(decision_packet_path).resolve()
    packet = _read(packet_path)
    packet_sha = convergence.sha256_file(packet_path)
    demand(selection.get("source_decision_packet_sha256") == packet_sha, "PI_SELECTION_BINDING_DRIFT", "decision packet SHA drift")
    demand(packet.get("decision_state") == "PI_DECISION_REQUIRED" and packet.get("winner_declared") is False, "PI_SELECTION_BINDING_DRIFT", "source packet is not neutral")
    route = selection.get("selected_route")
    demand(route in convergence.ROUTES, "PI_SELECTION_BINDING_DRIFT", "selected route invalid")
    selected = selection.get("selected_checkpoint_set")
    demand(isinstance(selected, Mapping) and selected, "PI_SELECTION_BINDING_DRIFT", "exact checkpoint set missing")
    summaries = packet.get("arm_summaries", {})
    for label, digest in selected.items():
        try:
            label_route, _, _ = convergence._arm_axes(label)
        except convergence.ConvergenceError as exc:
            reject("PI_SELECTION_BINDING_DRIFT", str(exc))
        demand(label_route == route and label in summaries, "PI_SELECTION_BINDING_DRIFT", f"selected checkpoint not in chosen route/packet: {label}")
        demand(summaries[label].get("checkpoint_tree_sha256") == _sha(digest, "PI_SELECTION_BINDING_DRIFT", f"selected_checkpoint_set.{label}"), "PI_SELECTION_BINDING_DRIFT", f"checkpoint tree drift: {label}")
    for field in ("selection_generation", "selected_checkpoint_tier_or_exact_checkpoint_set", "selection_scope", "validation_generation"):
        demand(field in selection, "PI_SELECTION_AUTHORITY_MISSING", f"selection field missing: {field}")
    route_contract = _route_contract(route)
    result = {"schema": "STAGE45F_POST_SELECTION_RECIPE_FREEZE_HANDOFF_V1", "status": "FROZEN_PLAN_AWAITING_VALIDATION_GATE", "selection_generation": selection["selection_generation"], "source_decision_packet_sha256": packet_sha, "selected_route": route, "selected_checkpoint_tier_or_exact_checkpoint_set": selection["selected_checkpoint_tier_or_exact_checkpoint_set"], "selected_checkpoint_set": dict(selected), "selection_scope": selection["selection_scope"], "pi_authority": {"comment": selection["pi_authority_comment"], "token": selection["pi_authority_token"]}, "recipe_binding": {"route_branch": route_contract["branch"], "route_head": route_contract["head"], "normalization": route_contract["normalization"], "stats_sha256": route_contract["stats_sha256"], "stats_payload_sha256": route_contract["stats_payload_sha256"], "loss": route_contract["loss"], "training_recipe": convergence.manifest()["h200_campaign_surfaces"]["training_recipe"]}, "validation_access_ledger": {"generation": selection["validation_generation"], "accessed": False, "candidate_set_sha256": convergence.canonical_sha(dict(selected)), "selection_generation": selection["selection_generation"]}, "winner_inferred": False, "validation_execution_enabled": False, "validation_e_read": False, "holdout_f_read": False}
    if output_path is not None:
        _write(output_path, result)
    return result


def validation_one_shot_guard(handoff_path: Path, ledger_path: Path) -> dict[str, Any]:
    handoff = _read(handoff_path)
    demand(handoff.get("schema") == "STAGE45F_POST_SELECTION_RECIPE_FREEZE_HANDOFF_V1", "PI_SELECTION_BINDING_DRIFT", "handoff schema drift")
    ledger = _read(ledger_path)
    demand(ledger.get("schema") == "STAGE45F_VALIDATION_E_ACCESS_LEDGER_V1", "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "access ledger schema drift")
    demand(ledger.get("holdout_access_count", 0) == 0 and ledger.get("holdout_requested") is not True, "HOLDOUT_FORBIDDEN", "Holdout-F is never available")
    expected = handoff["validation_access_ledger"]
    demand(ledger.get("generation") == expected["generation"] and ledger.get("selection_generation") == expected["selection_generation"] and ledger.get("candidate_set_sha256") == expected["candidate_set_sha256"], "VALIDATION_SELECTION_GENERATION_DRIFT", "candidate/selection/generation binding drift")
    demand(ledger.get("accessed") is False and ledger.get("completed_access_records", 0) == 0, "VALIDATION_GENERATION_ALREADY_ACCESSED", "generation cannot be called first unseen again")
    return {"schema": "STAGE45F_VALIDATION_E_ONE_SHOT_GUARD_V1", "status": "LOCKED_PLAN_ONLY", "generation": ledger["generation"], "accessed": False, "first_unseen_claim_available_if_future_pi_authorizes": True, "paths_opened": False, "execution_enabled": False, "holdout_f_read": False}


def build_capsule(bindings: Mapping[str, Any]) -> dict[str, Any]:
    template = _read(CAPSULE_TEMPLATE)
    required = set(template["required_reference_groups"])
    demand(set(bindings) == required, "UNSUPPORTED_SCHEMA_NO_HEURISTICS", "capsule reference group set drift")
    for group, value in bindings.items():
        demand(isinstance(value, Mapping) and value, "UPSTREAM_ARTIFACT_SHA_MISMATCH", f"capsule group missing: {group}")
    return {"schema": "STAGE45E_THREE_ROUTE_REPRODUCIBILITY_CAPSULE_V1", "status": "SYNTHETIC_OR_REAL_REFERENCES_BOUND", "references": dict(bindings), "copy_large_checkpoint_bytes": False, "delete_or_rewrite_accepted_evidence": False, "decision_state": "PI_DECISION_REQUIRED", "execution_enabled": False}


def build_branch_handoffs(recipe_handoff: Mapping[str, Any]) -> dict[str, Any]:
    demand(recipe_handoff.get("schema") == "STAGE45F_POST_SELECTION_RECIPE_FREEZE_HANDOFF_V1", "PI_SELECTION_BINDING_DRIFT", "recipe handoff schema drift")
    compatibility = _read(BRANCH_COMPATIBILITY)
    outputs = {name: {"schema": f"STAGE45F_{name}_HANDOFF_V1", "status": "PI_DECISION_REQUIRED", "execution_enabled": False, "selected_recipe_handoff_sha256": convergence.canonical_sha(recipe_handoff)} for name in compatibility["branches"]}
    return {"schema": "STAGE45F_BRANCH_HANDOFF_SET_V1", "status": "PI_DECISION_REQUIRED", "automatic_branch_selection": False, "branches": outputs, "execution_enabled": False}


def static_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    demand(not forbidden, "UNSUPPORTED_SCHEMA_NO_HEURISTICS", f"forbidden top-level imports: {forbidden}")
    hardening = hardening_manifest()
    shadow = shadow_audit()
    branch = _read(BRANCH_COMPATIBILITY)
    demand(all(item.get("execution_enabled") is False and item.get("decision_state") == "PI_DECISION_REQUIRED" for item in branch["branches"].values()), "PI_SELECTION_BINDING_DRIFT", "Stage45F branch surface became executable")
    return {"schema": "STAGE45F_CORELAB_PI0137_STATIC_AUDIT_V1", "status": "PASS", "mutation_case_count": len(hardening["mutation_cases"]), "shadow_audit": shadow, "branch_count": len(branch["branches"]), "real_h200_campaign_ingested": False, "forbidden_top_level_imports": forbidden, "execution_boundary": {"gpu": 0, "model_construction": 0, "model_forward": 0, "training": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "winner_selection": 0}}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("static", "shadow", "rehearsal", "verify-synthetic"), default="static")
    value.add_argument("--input", type=Path)
    value.add_argument("--identity", type=Path)
    value.add_argument("--output-root", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "static":
        result = static_audit()
    elif args.mode == "shadow":
        result = shadow_audit()
    elif args.mode == "rehearsal":
        demand(args.output_root is not None, "UNSAFE_PATH_ESCAPE", "--output-root required")
        result = run_synthetic_rehearsal(args.output_root)
    else:
        demand(args.input is not None and args.identity is not None, "UNSAFE_PATH_ESCAPE", "--input and --identity required")
        result = validate_synthetic_campaign(args.input, args.identity)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PI0137Failure as exc:
        print(json.dumps({"status": "BLOCKED", "reason_code": exc.reason_code, "detail": exc.detail}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
