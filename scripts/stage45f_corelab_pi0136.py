#!/usr/bin/env python3
"""PI0136 CPU/static evidence-convergence and decision-readiness tooling.

The entrypoint validates committed shadow-audit ledgers, builds a fail-closed
18-arm readiness matrix, compares sealed N2 TRAIN24 rows across training
hardware, combines sealed E4/D1/D2A evidence, and exercises recipe-freeze and
Validation-E generation-one governance.  It never imports a model framework,
constructs a model, invokes a scheduler, or opens Validation-E/Holdout-F.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import stage45f_corelab_evidence_convergence as convergence


REPO = Path(__file__).resolve().parents[1]
ROUTE_LEDGER = REPO / "manifests/STAGE45F_CORELAB_ROUTE_HEAD_LEDGER_PI0136.json"
LINEAGE_LEDGER = REPO / "manifests/STAGE45F_CORELAB_LINEAGE_ACCEPTANCE_LEDGER_PI0136.json"
READINESS_SNAPSHOT = REPO / "manifests/STAGE45F_CORELAB_18_ARM_READINESS_PI0136.json"
BRANCH_MAP = REPO / "manifests/STAGE45F_CORELAB_BRANCH_DECISION_MAP_PI0136.json"
ALLOWED_CLASSIFICATIONS = {"MECHANICAL_PROVENANCE_RUNTIME_ONLY", "SCIENTIFIC_CONTRACT_CHANGE"}
ALLOWED_READINESS = {"READY_SEALED", "PENDING_VANDA", "BLOCKED_IDENTITY", "BLOCKED_SCHEMA", "BLOCKED_LINEAGE"}
HARDWARES = ("RTX4080S", "H200")
D1_METRICS = {
    "teacher_forced_local_flow_rmse": "LOWER",
    "teacher_forced_local_flow_cosine": "HIGHER",
}
D2A_METRICS = {
    "tf_to_fr_state_path_divergence_l2": "LOWER",
    "field_sensitivity_l2": "LOWER",
    "teacher_bridge_local_error_l2": "LOWER",
    "off_path_diagnostic_l2": "DESCRIPTIVE",
}
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch", "transformers", "gr00t", "jax", "tensorflow", "numpy",
    "pandas", "subprocess", "isaacgym", "omni",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise convergence.ConvergenceError(message)


def _sha(value: Any, field: str) -> str:
    return convergence._sha(value, field)


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    return convergence._finite(value, field, nullable=nullable)


def _canonical(value: Any) -> str:
    return convergence.canonical_sha(value)


def _read(path: Path) -> Any:
    return convergence.read_json(Path(path).resolve())


def _write(path: Path | None, value: Any) -> None:
    if path is not None:
        convergence.write_json(Path(path).resolve(), value)


def _expected_labels() -> list[str]:
    return list(convergence.manifest()["matrix"]["arms"])


def _lineage_map(path: Path = LINEAGE_LEDGER) -> dict[str, Mapping[str, Any]]:
    value = _read(path)
    require(value.get("schema") == "CORELAB_STAGE45F_LINEAGE_ACCEPTANCE_LEDGER_V1", "lineage ledger schema drift")
    require(value.get("policy", {}).get("performance_based_retry_selection") is False, "performance-selected retry policy drift")
    require(value.get("policy", {}).get("duplicate_complete_lineage_requires_pi") is True, "duplicate-lineage PI gate missing")
    records = value.get("arms")
    require(isinstance(records, list), "lineage ledger arms missing")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        require(isinstance(record, Mapping), "lineage record is not an object")
        label = record.get("arm")
        require(label in _expected_labels() and label not in result, f"lineage arm invalid/duplicate: {label}")
        require(record.get("audit_state") in {"PASS", "BLOCKED"}, f"{label}: audit state must be PASS/BLOCKED")
        require(record.get("duplicate_complete_lineages") == 0, f"{label}: duplicate complete lineages require PI review")
        if record["audit_state"] == "PASS":
            for field in ("checkpoint_tree_sha256", "training_result_sha256", "trainer_state_sha256", "lineage_artifact_sha256"):
                _sha(record.get(field), f"{label}.{field}")
            require(isinstance(record.get("accepted_lineage_id"), str) and record["accepted_lineage_id"], f"{label}: accepted lineage id missing")
        else:
            require(isinstance(record.get("blocker"), str) and record["blocker"], f"{label}: blocked lineage lacks blocker")
        result[str(label)] = record
    require(set(result) == set(_expected_labels()), "lineage ledger is not the exact 18-arm matrix")
    require(value.get("pass_arm_count") == sum(record["audit_state"] == "PASS" for record in records), "lineage PASS count drift")
    require(value.get("blocked_arm_count") == sum(record["audit_state"] == "BLOCKED" for record in records), "lineage BLOCKED count drift")
    return result


def ledger_audit() -> dict[str, Any]:
    cfg = convergence.manifest()
    require(cfg["authority"]["pi_gate"] == "PI_GATE_STATE_0136", "PI0136 manifest authority missing")
    route = _read(ROUTE_LEDGER)
    require(route.get("schema") == "CORELAB_STAGE45F_ROUTE_HEAD_LEDGER_V1", "route-head ledger schema drift")
    require(route.get("scientific_contract_change_detected") is False, "route ledger reports scientific-contract drift")
    surfaces = route.get("surfaces")
    require(isinstance(surfaces, list) and len(surfaces) == 4, "route-head ledger must contain four surfaces")
    expected = {
        **{name: value for name, value in cfg["h200_campaign_surfaces"]["routes"].items()},
        "ORCHESTRATION": cfg["h200_campaign_surfaces"]["orchestration"],
    }
    seen: set[str] = set()
    for record in surfaces:
        name = record.get("route")
        require(name in expected and name not in seen, f"route-head surface invalid/duplicate: {name}")
        seen.add(str(name))
        require(record.get("branch") == expected[name]["branch"], f"{name}: branch drift")
        require(record.get("head_sha") == expected[name]["head"], f"{name}: head drift")
        require(record.get("classification") in ALLOWED_CLASSIFICATIONS, f"{name}: classification drift")
        require(record.get("classification") != "SCIENTIFIC_CONTRACT_CHANGE", f"{name}: scientific-contract change blocks convergence")
        require(isinstance(record.get("changed_files"), list), f"{name}: changed-files list missing")
        require(isinstance(record.get("accepted_completed_lineages"), list), f"{name}: accepted-lineage list missing")
        require(isinstance(record.get("known_failed_roots"), list), f"{name}: failed-root list missing")
    _lineage_map()
    readiness = _read(READINESS_SNAPSHOT)
    require(readiness.get("schema") == "CORELAB_STAGE45F_18_ARM_READINESS_MATRIX_V1", "readiness snapshot schema drift")
    require(set(readiness.get("arms", {})) == set(_expected_labels()), "readiness snapshot arm set drift")
    require(set(readiness["arms"].values()).issubset(ALLOWED_READINESS), "readiness status vocabulary drift")
    branch_map = _read(BRANCH_MAP)
    expected_branches = set(cfg["stage45f_branch_decision_map"]["branches"])
    require(branch_map.get("status") == "LOCKED_PLAN_ONLY", "Stage45F branch map is not locked")
    require(set(branch_map.get("branches", {})) == expected_branches, "Stage45F branch map set drift")
    require(all(item.get("execution_enabled") is False for item in branch_map["branches"].values()), "Stage45F branch execution enabled")
    return {
        "schema": "CORELAB_STAGE45F_PI0136_LEDGER_STATIC_AUDIT_V1",
        "status": "PASS",
        "route_head_ledger_sha256": convergence.sha256_file(ROUTE_LEDGER),
        "lineage_acceptance_ledger_sha256": convergence.sha256_file(LINEAGE_LEDGER),
        "readiness_snapshot_sha256": convergence.sha256_file(READINESS_SNAPSHOT),
        "branch_decision_map_sha256": convergence.sha256_file(BRANCH_MAP),
        "surface_heads": {record["route"]: record["head_sha"] for record in surfaces},
        "scientific_contract_change": False,
        "winner_selection": False,
    }


def build_readiness(
    evidence_index: Path | None,
    output_path: Path | None = None,
    lineage_ledger: Path = LINEAGE_LEDGER,
) -> dict[str, Any]:
    lineages = _lineage_map(lineage_ledger)
    specs: Mapping[str, Any] = {}
    source_sha: str | None = None
    if evidence_index is not None:
        raw = _read(evidence_index)
        require(raw.get("schema") == "STAGE45F_H200_TRAIN24_EVIDENCE_INDEX_V1", "readiness evidence index schema drift")
        require(raw.get("split") == "TRAIN24" and raw.get("validation_e_read") is False and raw.get("holdout_f_read") is False, "readiness evidence split drift")
        specs = raw.get("arms", {})
        require(isinstance(specs, Mapping) and set(specs).issubset(set(_expected_labels())), "readiness evidence contains an unexpected arm")
        source_sha = convergence.sha256_file(evidence_index)
    arms: dict[str, Any] = {}
    for label in _expected_labels():
        if label not in specs:
            arms[label] = {"status": "PENDING_VANDA", "reason": "sealed H200 evidence arm not yet supplied"}
            continue
        try:
            loaded = convergence._load_arm(label, specs[label])
        except convergence.ConvergenceError as exc:
            message = str(exc)
            status = "BLOCKED_IDENTITY" if any(word in message.lower() for word in ("identity", "sha", "tree", "branch", "head", "checkpoint")) else "BLOCKED_SCHEMA"
            arms[label] = {"status": status, "reason": message}
            continue
        lineage = lineages[label]
        if lineage["audit_state"] != "PASS":
            arms[label] = {"status": "BLOCKED_LINEAGE", "reason": lineage["blocker"], "rows_sha256": loaded["rows_sha256"]}
            continue
        provenance = loaded["provenance"]
        for field in ("checkpoint_tree_sha256", "training_result_sha256", "trainer_state_sha256", "lineage_artifact_sha256"):
            require(lineage[field] == provenance[field], f"{label}: readiness/lineage {field} drift")
        arms[label] = {
            "status": "READY_SEALED",
            "checkpoint_tree_sha256": provenance["checkpoint_tree_sha256"],
            "rows_sha256": loaded["rows_sha256"],
            "evidence_seal_sha256": provenance["evidence_seal_sha256"],
            "lineage_artifact_sha256": provenance["lineage_artifact_sha256"],
        }
    counts = {status: sum(record["status"] == status for record in arms.values()) for status in sorted(ALLOWED_READINESS)}
    result = {
        "schema": "CORELAB_STAGE45F_18_ARM_READINESS_MATRIX_V1",
        "status": "READY_SEALED" if counts["READY_SEALED"] == 18 else "PENDING_OR_BLOCKED",
        "source_index_sha256": source_sha,
        "lineage_ledger_sha256": convergence.sha256_file(lineage_ledger),
        "arms": arms,
        "counts": counts,
        "validation_e_read": False,
        "holdout_f_read": False,
        "automatic_winner": False,
    }
    _write(output_path, result)
    return result


def _hardware_arm(index: Mapping[str, Any], hardware: str, label: str) -> dict[str, Any]:
    cfg = convergence.manifest()
    spec = index["arms"][hardware][label]
    require(isinstance(spec, Mapping), f"{hardware}/{label}: arm spec missing")
    route, seed, step = convergence._arm_axes(label)
    require(route == "N2", f"{hardware}/{label}: only N2 route is allowed")
    provenance = spec.get("provenance")
    require(isinstance(provenance, Mapping), f"{hardware}/{label}: provenance missing")
    require(provenance.get("route") == "N2" and provenance.get("seed") == seed and provenance.get("step") == step, f"{hardware}/{label}: axes drift")
    require(provenance.get("hardware") == hardware, f"{hardware}/{label}: hardware identity drift")
    hw = cfg["n2_cross_hardware"]
    expected_head = hw["historical_4080s_head"] if hardware == "RTX4080S" else hw["h200_head"]
    require(provenance.get("route_head") == expected_head, f"{hardware}/{label}: route HEAD drift")
    require(provenance.get("stats_sha256") == hw["accepted_stats_file_sha256_by_hardware"][hardware], f"{hardware}/{label}: stats file drift")
    require(provenance.get("stats_payload_sha256") == hw["stats_payload_sha256"], f"{hardware}/{label}: N2 stats payload drift")
    frozen = cfg["h200_campaign_surfaces"]["frozen_sources"]
    require(provenance.get("base_tree_sha256") == frozen["base_tree_sha256"], f"{hardware}/{label}: BASE tree drift")
    require(provenance.get("full24_tree_sha256") == frozen["full24_tree_sha256"], f"{hardware}/{label}: FULL24 tree drift")
    require(provenance.get("full_state_checkpoint") is True and provenance.get("completed_step") == step, f"{hardware}/{label}: checkpoint/trainer closure drift")
    require(provenance.get("target_xyz_in_model_input") is False, f"{hardware}/{label}: target XYZ provenance drift")
    for field in ("checkpoint_tree_sha256", "evidence_seal_sha256", "lineage_artifact_sha256", "training_result_sha256", "trainer_state_sha256"):
        _sha(provenance.get(field), f"{hardware}/{label}.{field}")
    for field in ("training_recipe_identity", "science_source_identity", "device_runtime_identity"):
        require(isinstance(provenance.get(field), Mapping) and provenance[field], f"{hardware}/{label}: {field} missing")
    recipe = dict(provenance["training_recipe_identity"])
    require(recipe.pop("seed", None) == seed, f"{hardware}/{label}: training seed binding drift")
    require(recipe == cfg["h200_campaign_surfaces"]["training_recipe"], f"{hardware}/{label}: frozen training recipe drift")
    source = provenance["science_source_identity"]
    for key in ("groot_commit", "cosmos_revision", "cosmos_tree_sha256", "modality_sha256", "config_sha256"):
        require(source.get(key) == frozen[key], f"{hardware}/{label}: science source {key} drift")
    rows_path = Path(spec.get("rows_json", "")).resolve()
    require(rows_path.is_file() and not rows_path.is_symlink(), f"{hardware}/{label}: rows unavailable")
    require(convergence.sha256_file(rows_path) == _sha(spec.get("rows_sha256"), f"{hardware}/{label}.rows_sha256"), f"{hardware}/{label}: rows SHA drift")
    rows_payload = _read(rows_path)
    require(isinstance(rows_payload, list) and len(rows_payload) == 24, f"{hardware}/{label}: expected 24 rows")
    compat = dict(provenance)
    compat["row_checkpoint_label"] = provenance.get("row_checkpoint_label", label)
    rows = [convergence._validate_row(label, idx, row, compat) for idx, row in enumerate(rows_payload)]
    return {"label": label, "hardware": hardware, "seed": seed, "step": step, "provenance": dict(provenance), "rows": rows, "rows_sha256": spec["rows_sha256"]}


def _normalized_hardware_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in convergence.NORMALIZED_METRICS:
        deltas = [float(rrow["normalized"][metric] - lrow["normalized"][metric]) for lrow, rrow in zip(left["rows"], right["rows"])]
        result[metric] = convergence._summary(deltas)
    return result


def compare_n2_hardware(index_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    index = _read(index_path)
    require(index.get("schema") == "STAGE45F_N2_CROSS_HARDWARE_EVIDENCE_INDEX_V1", "N2 hardware index schema drift")
    require(index.get("split") == "TRAIN24" and index.get("validation_e_read") is False and index.get("holdout_f_read") is False, "N2 hardware index split drift")
    require(isinstance(index.get("arms"), Mapping) and set(index["arms"]) == set(HARDWARES), "N2 hardware set drift")
    labels = [f"N2(seed{seed})@{step}" for seed in convergence.SEEDS for step in convergence.STEPS]
    loaded = {hardware: {label: _hardware_arm(index, hardware, label) for label in labels} for hardware in HARDWARES}
    pairs: dict[str, Any] = {}
    for label in labels:
        left, right = loaded["RTX4080S"][label], loaded["H200"][label]
        require(_canonical(left["provenance"]["training_recipe_identity"]) == _canonical(right["provenance"]["training_recipe_identity"]), f"{label}: training recipe differs across hardware")
        require(_canonical(left["provenance"]["science_source_identity"]) == _canonical(right["provenance"]["science_source_identity"]), f"{label}: science source differs across hardware")
        pairs[label] = {
            "physical_delta_h200_minus_4080s": convergence._paired(left, right, f"H200_minus_RTX4080S_{label}"),
            "normalized_n2_route_internal_delta_h200_minus_4080s": _normalized_hardware_pair(left, right),
            "checkpoint_trees": {hardware: loaded[hardware][label]["provenance"]["checkpoint_tree_sha256"] for hardware in HARDWARES},
            "device_runtime_identity": {hardware: loaded[hardware][label]["provenance"]["device_runtime_identity"] for hardware in HARDWARES},
        }
    result = {
        "schema": "CORELAB_STAGE45F_N2_CROSS_HARDWARE_ROBUSTNESS_PACKET_V1",
        "status": "PASS_DESCRIPTIVE_REPLICATION_ROBUSTNESS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "source_index_sha256": convergence.sha256_file(index_path),
        "pair_count": 6,
        "pairs": pairs,
        "claim_boundary": "hardware-associated replication stability only; no hardware causality, performance selection, or substitution into the primary H200 matrix",
        "winner_declared": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_boundary": {"gpu": 0, "model_forward": 0, "training": 0, "pbs": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }
    _write(output_path, result)
    return result


def _mechanism_arm(label: str, value: Mapping[str, Any], train_packet: Mapping[str, Any]) -> dict[str, Any]:
    route, seed, step = convergence._arm_axes(label)
    require(value.get("route") == route and value.get("seed") == seed and value.get("step") == step, f"{label}: mechanism axes drift")
    checkpoint_tree = _sha(value.get("checkpoint_tree_sha256"), f"{label}.checkpoint_tree_sha256")
    require(train_packet["arm_summaries"][label]["checkpoint_tree_sha256"] == checkpoint_tree, f"{label}: E4/mechanism checkpoint tree drift")
    result: dict[str, Any] = {"route": route, "seed": seed, "step": step, "checkpoint_tree_sha256": checkpoint_tree}
    for phase, metrics in (("D1", D1_METRICS), ("D2A", D2A_METRICS)):
        record = value.get(phase)
        require(isinstance(record, Mapping), f"{label}: {phase} evidence missing")
        _sha(record.get("seal_sha256"), f"{label}.{phase}.seal_sha256")
        _sha(record.get("rows_sha256"), f"{label}.{phase}.rows_sha256")
        require(record.get("raw_before_derived") is True and record.get("target_xyz_in_model_input") is False, f"{label}: {phase} evidence boundary drift")
        require(record.get("row_count") == (216 if phase == "D1" else 96), f"{label}: {phase} row count drift")
        summary = record.get("metrics")
        require(isinstance(summary, Mapping) and set(summary) == set(metrics), f"{label}: {phase} metric schema drift")
        result[phase] = {
            "seal_sha256": record["seal_sha256"],
            "rows_sha256": record["rows_sha256"],
            "row_count": record["row_count"],
            "metrics": {metric: _finite(summary[metric], f"{label}.{phase}.{metric}", nullable=(metric.endswith("cosine"))) for metric in metrics},
        }
    return result


def _direction(delta: float | None, preference: str) -> str:
    if delta is None:
        return "UNDEFINED"
    if delta == 0.0:
        return "TIED_EXACT"
    if preference == "DESCRIPTIVE":
        return "INCREASED" if delta > 0.0 else "DECREASED"
    improved = delta < 0.0 if preference == "LOWER" else delta > 0.0
    return "IMPROVED_EXACT_SIGN" if improved else "WORSENED_EXACT_SIGN"


def build_mechanism_packet(index_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    value = _read(index_path)
    require(value.get("schema") == "STAGE45F_H200_MECHANISM_EVIDENCE_INDEX_V1", "mechanism index schema drift")
    require(value.get("split") == "TRAIN24" and value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "mechanism index split drift")
    train_path = Path(value.get("train24_decision_packet", "")).resolve()
    require(train_path.is_file() and convergence.sha256_file(train_path) == _sha(value.get("train24_decision_packet_sha256"), "train24_decision_packet_sha256"), "TRAIN24 decision packet binding drift")
    train = _read(train_path)
    require(train.get("schema") == "CORELAB_STAGE45F_THREE_ROUTE_TRAIN24_DECISION_PACKET_V1" and train.get("decision_state") == "PI_DECISION_REQUIRED", "TRAIN24 packet state drift")
    require(train.get("winner_declared") is False and train.get("validation_e_read") is False, "TRAIN24 packet claim boundary drift")
    hardware_path = Path(value.get("n2_hardware_packet", "")).resolve()
    require(hardware_path.is_file() and convergence.sha256_file(hardware_path) == _sha(value.get("n2_hardware_packet_sha256"), "n2_hardware_packet_sha256"), "N2 hardware packet binding drift")
    hardware = _read(hardware_path)
    require(hardware.get("schema") == "CORELAB_STAGE45F_N2_CROSS_HARDWARE_ROBUSTNESS_PACKET_V1" and hardware.get("winner_declared") is False, "N2 hardware packet state drift")
    arms_value = value.get("arms")
    require(isinstance(arms_value, Mapping) and set(arms_value) == set(_expected_labels()), "mechanism index must contain exact 18 arms")
    arms = {label: _mechanism_arm(label, arms_value[label], train) for label in _expected_labels()}
    exposure: dict[str, Any] = {}
    seed_consistency: dict[str, Any] = {}
    for route in convergence.ROUTES:
        route_directions: dict[str, list[str]] = {f"{phase}.{metric}": [] for phase, fields in (("D1", D1_METRICS), ("D2A", D2A_METRICS)) for metric in fields}
        for seed in convergence.SEEDS:
            low = arms[f"{route}(seed{seed})@1000"]
            high = arms[f"{route}(seed{seed})@2000"]
            record: dict[str, Any] = {}
            for phase, metrics in (("D1", D1_METRICS), ("D2A", D2A_METRICS)):
                record[phase] = {}
                for metric, preference in metrics.items():
                    low_value, high_value = low[phase]["metrics"][metric], high[phase]["metrics"][metric]
                    delta = None if low_value is None or high_value is None else float(high_value - low_value)
                    direction = _direction(delta, preference)
                    record[phase][metric] = {"at_1000": low_value, "at_2000": high_value, "delta_2000_minus_1000": delta, "preference": preference, "exact_sign_description": direction}
                    route_directions[f"{phase}.{metric}"].append(direction)
            exposure[f"{route}(seed{seed})"] = record
        seed_consistency[route] = {
            key: {direction: values.count(direction) for direction in sorted(set(values))}
            for key, values in route_directions.items()
        }
    route_readouts = {
        route: {
            "e4_physical_arm_summaries": {
                label: train["arm_summaries"][label]["physical"]
                for label in _expected_labels() if label.startswith(f"{route}(")
            },
            "e4_physical_exposure_2000_minus_1000": train["three_seed_descriptive"]["exposure_2000_minus_1000"][route],
            "d1_d2a_exposure_by_seed": {
                str(seed): exposure[f"{route}(seed{seed})"] for seed in convergence.SEEDS
            },
            "d1_d2a_three_seed_exact_sign_consistency": seed_consistency[route],
            "path_and_roughness_reported_jointly": True,
            "normalized_cross_route_common_scale": False,
        }
        for route in convergence.ROUTES
    }
    result = {
        "schema": "CORELAB_STAGE45F_MECHANISM_CONVERGENCE_PACKET_V1",
        "status": "PASS_DESCRIPTIVE_SEALED_EVIDENCE_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "source_index_sha256": convergence.sha256_file(index_path),
        "train24_decision_packet_sha256": convergence.sha256_file(train_path),
        "n2_hardware_packet_sha256": convergence.sha256_file(hardware_path),
        "arms": arms,
        "route_readouts": route_readouts,
        "exposure_2000_minus_1000": exposure,
        "three_seed_exact_sign_consistency": seed_consistency,
        "e4_physical_outcome": {
            "status": "BOUND_BY_TRAIN24_DECISION_PACKET",
            "three_seed_route_minus_legacy": train["three_seed_descriptive"]["route_minus_legacy"],
            "failure_taxonomy": train["failure_taxonomy"],
        },
        "d1_interpretation": "teacher-forced/local-field diagnostics only",
        "d2a_interpretation": "TF-to-FR degradation and block-substitution attribution; not additive causal decomposition",
        "n2_hardware_interpretation": "replication robustness only; not hardware causality or performance selection",
        "n2_hardware_summary": {
            "status": hardware["status"],
            "pair_count": hardware["pair_count"],
            "claim_boundary": hardware["claim_boundary"],
        },
        "question_contract": {
            "local_field_improving_or_degrading": "reported as within-route @2000-@1000 continuous deltas and exact-sign seed consistency; no threshold",
            "free_running_degradation_relative_to_teacher_forced": "D1 and D2A continuous metrics reported side by side; no posthoc large/small threshold",
            "exposure_help_consistency": "reported per seed and as 0/3..3/3 exact-sign consistency",
            "endpoint_progress_vs_path_roughness": "reported jointly from E4 physical metrics",
            "training_seed_stability": "three seed is the replication axis; descriptive only",
            "n2_cross_hardware_stability": "reported by paired same-seed/same-tier/same-window comparator; no hardware causality",
        },
        "winner_declared": False,
        "p_value_computed": False,
        "posthoc_threshold_used": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_boundary": {"gpu": 0, "model_forward": 0, "training": 0, "pbs": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }
    _write(output_path, result)
    return result


def freeze_recipe(request_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    request = _read(request_path)
    require(request.get("schema") == "CORELAB_STAGE45F_RECIPE_FREEZE_REQUEST_V1", "recipe-freeze request schema drift")
    synthetic = request.get("synthetic_fixture") is True
    require(request.get("status") == ("SYNTHETIC_TEST_ONLY" if synthetic else "PI_SELECTED"), "recipe-freeze selection authority missing")
    selection_id = request.get("selection_generation_id")
    require(isinstance(selection_id, str) and selection_id.strip(), "selection-generation identifier is required")
    if not synthetic:
        require(isinstance(request.get("pi_selection_issue_comment"), int) and request["pi_selection_issue_comment"] > 0, "real recipe freeze requires explicit PI selection comment")
    bindings = request.get("bindings")
    require(isinstance(bindings, Mapping), "recipe-freeze bindings missing")
    required = set(convergence.manifest()["recipe_freeze"]["required_fields"]) - {"selection_generation_id"}
    require(set(bindings) == required, "recipe-freeze binding field set drift")
    route_identity = bindings["route_intervention_identity"]
    require(isinstance(route_identity, Mapping) and route_identity.get("route") in convergence.ROUTES, "selected route identity invalid")
    checkpoint_trees = bindings["checkpoint_trees"]
    require(isinstance(checkpoint_trees, Mapping) and checkpoint_trees, "checkpoint tree map missing")
    for label, digest in checkpoint_trees.items():
        require(isinstance(label, str) and label, "checkpoint label missing")
        _sha(digest, f"checkpoint_trees.{label}")
    for field in ("base_tree", "dataset_tree", "split_lock", "train24_packet_sha256", "evidence_inventory_sha256"):
        _sha(bindings[field], field)
    require(bindings["validation_generation"] == 1, "first Validation-E generation must be 1")
    for field in required - {"checkpoint_trees", "base_tree", "dataset_tree", "split_lock", "train24_packet_sha256", "evidence_inventory_sha256", "validation_generation"}:
        require(isinstance(bindings[field], Mapping) and bindings[field], f"recipe binding {field} must be a nonempty object")
    result = {
        "schema": "CORELAB_STAGE45F_RECIPE_FREEZE_BUNDLE_V1",
        "status": "FROZEN_SYNTHETIC_TEST_ONLY" if synthetic else "FROZEN_AWAITING_VALIDATION_GATE",
        "selection_generation_id": selection_id,
        "pi_selection_issue_comment": request.get("pi_selection_issue_comment"),
        "synthetic_fixture": synthetic,
        "request_sha256": convergence.sha256_file(request_path),
        "bindings": dict(bindings),
        "normalization_recomputed": False,
        "checkpoint_reselected": False,
        "automatic_winner": False,
        "execution_enabled": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    result["canonical_payload_sha256"] = _canonical(result)
    _write(output_path, result)
    return result


def validation_preflight(request_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    request = _read(request_path)
    require(request.get("schema") == "CORELAB_STAGE45F_VALIDATION_E_GENERATION1_PREFLIGHT_REQUEST_V1", "Validation-E preflight request schema drift")
    forbidden_keys = {"validation_path", "validation_dataset", "validation_samples", "holdout_path", "holdout_dataset", "holdout_samples"}
    require(not forbidden_keys.intersection(request), "Validation/Holdout data locator supplied to plan-only preflight")
    bundle_path = Path(request.get("recipe_freeze_bundle", "")).resolve()
    require(bundle_path.is_file() and convergence.sha256_file(bundle_path) == _sha(request.get("recipe_freeze_bundle_sha256"), "recipe_freeze_bundle_sha256"), "recipe-freeze bundle binding drift")
    bundle = _read(bundle_path)
    require(bundle.get("schema") == "CORELAB_STAGE45F_RECIPE_FREEZE_BUNDLE_V1", "recipe-freeze bundle schema drift")
    require(bundle.get("status") in {"FROZEN_SYNTHETIC_TEST_ONLY", "FROZEN_AWAITING_VALIDATION_GATE"}, "recipe is not frozen")
    ledger = request.get("access_ledger")
    require(isinstance(ledger, Mapping), "Validation access ledger missing")
    require(ledger.get("prior_validation_read_count") == 0 and ledger.get("next_generation") == 1, "first unseen Validation-E generation ledger drift")
    require(ledger.get("holdout_read_count") == 0 and ledger.get("post_validation_tuning_count") == 0, "Validation/Holdout access ledger is not pristine")
    require(ledger.get("selection_generation_id") == bundle["selection_generation_id"], "selection-generation binding drift")
    result = {
        "schema": "CORELAB_STAGE45F_VALIDATION_E_GENERATION1_PREFLIGHT_V1",
        "status": "LOCKED_PLAN_ONLY",
        "execution_enabled": False,
        "validation_read": False,
        "holdout_read": False,
        "paths_opened": False,
        "PI_GATE_REQUIRED": True,
        "validation_access_generation": 1,
        "generation_one_is_single_use": True,
        "later_access_generation_minimum": 2,
        "post_validation_tuning_may_not_reuse_generation_one": True,
        "recipe_freeze_bundle_sha256": convergence.sha256_file(bundle_path),
        "selection_generation_id": bundle["selection_generation_id"],
        "metric_schema": ["endpoint_final_distance_m", "progress_m", "path_length_m", "roughness_m"],
        "success_or_winner_threshold": None,
        "pbs_authorized": False,
        "gpu_authorized": False,
    }
    _write(output_path, result)
    return result


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
    require(not forbidden, f"forbidden top-level imports: {forbidden}")
    ledgers = ledger_audit()
    cfg = convergence.manifest()
    require(cfg["pairing_policy"]["contract_frozen_before_final_h200_metrics"] is True, "pre-metric comparison freeze marker missing")
    return {
        "schema": "CORELAB_STAGE45F_PI0136_STATIC_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256": convergence.sha256_file(convergence.MANIFEST_PATH),
        "forbidden_top_level_imports": forbidden,
        "ledgers": ledgers,
        "matrix_arms": 18,
        "primary_pairing": "same seed x same TRAIN24 window x same checkpoint tier",
        "normalized_cross_route_common_scale": False,
        "windows_as_iid_replicates": False,
        "p_value_threshold": None,
        "automatic_winner": False,
        "execution_boundary": {"gpu": 0, "cuda": 0, "model_construction": 0, "model_forward": 0, "training": 0, "backward": 0, "optimizer": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("static", "readiness", "n2-hardware", "mechanism", "freeze-recipe", "validation-preflight", "branch-map"), default="static")
    value.add_argument("--input", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--lineage-ledger", type=Path, default=LINEAGE_LEDGER)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "static":
        result = static_audit()
    elif args.mode == "branch-map":
        result = _read(BRANCH_MAP)
    elif args.mode == "readiness":
        result = build_readiness(args.input, args.output, args.lineage_ledger)
    else:
        require(args.input is not None, f"--input required for {args.mode}")
        functions = {
            "n2-hardware": compare_n2_hardware,
            "mechanism": build_mechanism_packet,
            "freeze-recipe": freeze_recipe,
            "validation-preflight": validation_preflight,
        }
        result = functions[args.mode](args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except convergence.ConvergenceError as exc:
        print(f"CORELAB_STAGE45F_PI0136_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
