#!/usr/bin/env python3
"""CPU-only Stage45E evidence convergence and Stage45F gate preparation.

This entrypoint consumes sealed JSON metadata and already-derived TRAIN24 row
bundles.  It never constructs a model, imports Torch/NumPy/GR00T, opens a
dataset split, or talks to a scheduler.  Cross-route comparisons are made only
in decoded physical units; normalized values remain route-local descriptors.
The packet deliberately ends in ``PI_DECISION_REQUIRED`` and has no winner
selector.  Validation-E/Holdout-F paths are rejected before any file read.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45F_CORELAB_EVIDENCE_CONVERGENCE_V1.json"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch", "transformers", "gr00t", "jax", "tensorflow", "numpy",
    "pandas", "subprocess", "isaacgym", "omni",
}
LOCKED_PATH_PARTS = {"validation-e", "validation_e", "holdout-f", "holdout_f", "holdout"}
ROUTES = ("LEGACY", "N2", "L1")
SEEDS = (2222, 2223, 2224)
STEPS = (1000, 2000)
TARGETS = tuple(
    f"red_target_train_{target}"
    for target in ("a", "b", "c", "d")
)
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TARGETS
    for index in range(6)
)
ARM_RE = re.compile(r"^(LEGACY|N2|L1)\(seed(2222|2223|2224)\)@(1000|2000)$")
PHYSICAL_METRICS = (
    "final_target_distance_m",
    "absolute_progress_m",
    "minimum_target_distance_m",
    "endpoint_target_vector_cosine",
    "path_length_m",
    "roughness_position_second_difference_m",
)
NORMALIZED_METRICS = (
    "active_rmse",
    "active_mae",
    "active_max_abs",
    "raw_active_outside_neg1_pos1_fraction",
)
PHYSICAL_PREFERENCES = {
    "final_target_distance_m": "LOWER",
    "absolute_progress_m": "HIGHER",
    "minimum_target_distance_m": "LOWER",
    "endpoint_target_vector_cosine": "HIGHER",
    "path_length_m": "TRADEOFF_REPORT_JOINTLY",
    "roughness_position_second_difference_m": "TRADEOFF_REPORT_JOINTLY",
}


class ConvergenceError(RuntimeError):
    """Raised when evidence or provenance is not sufficient for convergence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConvergenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    _safe_input_path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON input unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path).resolve()
    _safe_output_path(path)
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _safe_input_path(path: Path) -> None:
    parts = {part.lower() for part in path.parts}
    require(not parts.intersection(LOCKED_PATH_PARTS), f"locked Validation-E/Holdout-F path rejected: {path}")


def _safe_output_path(path: Path) -> None:
    _safe_input_path(path)


def _sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{field} must be a 64-character SHA256")
    require(value == value.lower() and all(char in "0123456789abcdef" for char in value), f"{field} must be lowercase hexadecimal")
    return value


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None:
        require(nullable, f"{field} is null")
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is non-finite")
    return result


def _mapping_path(value: Mapping[str, Any], dotted: str, field: str | None = None) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, Mapping) and part in current, f"{field or dotted} missing")
        current = current[part]
    return current


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "CORELAB_STAGE45F_EVIDENCE_CONVERGENCE_V1", "convergence manifest schema drift")
    require(value.get("status") == "CPU_STATIC_PREP_ONLY", "convergence manifest is not CPU/static-only")
    matrix = value.get("matrix")
    require(isinstance(matrix, Mapping), "matrix contract missing")
    require(matrix.get("routes") == list(ROUTES) and matrix.get("seeds") == list(SEEDS) and matrix.get("checkpoint_steps") == list(STEPS), "matrix axes drift")
    require(matrix.get("window_count") == 24 and matrix.get("split") == "TRAIN24", "TRAIN24 matrix contract drift")
    require(len(matrix.get("arms", [])) == 18, "three-route matrix must contain exactly 18 arms")
    policy = value.get("pairing_policy", {})
    require(policy.get("no_automatic_winner") is True and policy.get("packet_status") == "PI_DECISION_REQUIRED", "winner/decision policy drift")
    routes = value.get("h200_campaign_surfaces", {}).get("routes", {})
    require(routes.get("LEGACY", {}).get("t16_active_scalars") == 190, "LEGACY T16 mask drift")
    require(routes.get("L1", {}).get("loss_denominator") == "binary_action_mask.sum()", "L1 denominator semantics drift")
    require(routes.get("L1", {}).get("continuous_weight_tensor_separate") is True and routes.get("L1", {}).get("t16_active_scalars") == 190, "L1 weight/mask semantics drift")
    require(routes.get("N2", {}).get("xyz_common_half_range_m") == 0.107349471, "N2 XYZ half-range drift")
    require(routes.get("N2", {}).get("robust_centers_unchanged") is True and routes.get("N2", {}).get("non_xyz_equal_full24") is True, "N2 stats contract drift")
    require(routes.get("N2", {}).get("t16_active_scalars") == 190, "N2 T16 mask drift")
    return value


def static_audit() -> dict[str, Any]:
    cfg = manifest()
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
    require("PI_DECISION_REQUIRED" in source and "automatic_winner" in source, "decision boundary markers missing")
    return {
        "schema": "CORELAB_STAGE45F_EVIDENCE_CONVERGENCE_STATIC_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "support_branch": cfg["repository"]["support_branch"],
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "gpu": 0,
        "pbs": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "winner_selection": 0,
    }


def _arm_axes(label: str) -> tuple[str, int, int]:
    match = ARM_RE.fullmatch(label)
    require(match is not None, f"non-canonical arm label: {label}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def _route_contract(route: str) -> Mapping[str, Any]:
    value = manifest()["h200_campaign_surfaces"]["routes"].get(route)
    require(isinstance(value, Mapping), f"route contract missing: {route}")
    return value


def _validate_provenance(label: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    route, seed, step = _arm_axes(label)
    provenance = spec.get("provenance")
    require(isinstance(provenance, Mapping), f"{label}: provenance missing")
    require(provenance.get("route") == route and provenance.get("seed") == seed and provenance.get("step") == step, f"{label}: provenance axes drift")
    contract = _route_contract(route)
    require(provenance.get("route_branch") == contract["branch"], f"{label}: route branch drift")
    require(provenance.get("route_head") == contract["head"], f"{label}: route HEAD drift")
    require(provenance.get("normalization_identity") == contract["normalization"], f"{label}: normalization identity drift")
    require(provenance.get("stats_sha256") == contract["stats_sha256"], f"{label}: stats identity drift")
    require(provenance.get("stats_payload_sha256") == contract["stats_payload_sha256"], f"{label}: stats payload identity drift")
    require(provenance.get("hardware") == "H200", f"{label}: primary matrix is not H200-bound")
    require(provenance.get("validation_e_read") is False and provenance.get("holdout_f_read") is False, f"{label}: locked split read indicated")
    require(provenance.get("raw_before_derived") is True, f"{label}: raw-before-derived contract missing")
    require(provenance.get("full_state_checkpoint") is True, f"{label}: full-state checkpoint closure missing")
    require(provenance.get("completed_step") == step, f"{label}: trainer/checkpoint step drift")
    require(provenance.get("target_xyz_in_model_input") is False, f"{label}: target XYZ provenance drift")
    frozen = manifest()["h200_campaign_surfaces"]["frozen_sources"]
    require(provenance.get("base_tree_sha256") == frozen["base_tree_sha256"], f"{label}: BASE tree drift")
    require(provenance.get("full24_tree_sha256") == frozen["full24_tree_sha256"], f"{label}: FULL24 tree drift")
    for field in (
        "checkpoint_tree_sha256", "evidence_seal_sha256", "lineage_artifact_sha256",
        "training_result_sha256", "trainer_state_sha256", "hardware_runtime_sha256",
        "training_recipe_sha256",
    ):
        _sha(provenance.get(field), f"{label}.provenance.{field}")
    for field in ("decoder_identity", "metric_identity", "sampler_identity"):
        require(isinstance(provenance.get(field), str) and provenance[field].strip(), f"{label}: {field} missing")
    return dict(provenance)


def _validate_row(label: str, index: int, row: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    require(row.get("schema") == "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1", f"{label}:{index}: row schema drift")
    require(row.get("window_id") == WINDOW_IDS[index], f"{label}:{index}: window order drift")
    require(row.get("target_id") in TARGETS, f"{label}:{index}: target identity drift")
    require(row.get("target_xyz_in_model_input") is False, f"{label}:{index}: target XYZ hidden input")
    require(row.get("finite") is True, f"{label}:{index}: finite flag is not true")
    checkpoint = row.get("checkpoint")
    require(isinstance(checkpoint, Mapping), f"{label}:{index}: checkpoint identity missing")
    require(checkpoint.get("label") == provenance["row_checkpoint_label"], f"{label}:{index}: checkpoint label drift")
    require(checkpoint.get("sha256") == provenance["checkpoint_tree_sha256"], f"{label}:{index}: checkpoint tree binding drift")
    model = row.get("model")
    require(isinstance(model, Mapping) and model.get("sha256") == provenance["checkpoint_tree_sha256"], f"{label}:{index}: model/checkpoint binding drift")
    metrics = row.get("metrics")
    require(isinstance(metrics, Mapping), f"{label}:{index}: metrics missing")
    target = metrics.get("target")
    require(isinstance(target, Mapping), f"{label}:{index}: target metrics missing")
    physical_paths = {
        "final_target_distance_m": target.get("final_target_distance_m"),
        "absolute_progress_m": target.get("absolute_progress_m"),
        "minimum_target_distance_m": target.get("minimum_target_distance_m"),
        "endpoint_target_vector_cosine": target.get("endpoint_target_vector_cosine"),
        "path_length_m": target.get("path_length_m"),
        "roughness_position_second_difference_m": _mapping_path(target, "roughness_position_second_difference_m.mean", f"{label}:{index}:roughness"),
    }
    physical = {key: _finite(value, f"{label}:{index}:{key}", nullable=(key == "endpoint_target_vector_cosine")) for key, value in physical_paths.items()}
    sign = target.get("progress_sign")
    require(sign in {"POSITIVE", "ZERO", "NEGATIVE"}, f"{label}:{index}: progress sign drift")
    normalized = row.get("normalized_metrics")
    require(isinstance(normalized, Mapping), f"{label}:{index}: normalized route-local metrics missing")
    normalized_values = {key: _finite(normalized.get(key), f"{label}:{index}:normalized.{key}") for key in NORMALIZED_METRICS}
    sampler = row.get("sampler")
    require(isinstance(sampler, Mapping) and sampler.get("steps_observed") == 4 and sampler.get("official_public_sampler_called") is False, f"{label}:{index}: sampler evidence drift")
    safety = row.get("safety_gate_decision")
    require(safety in {None, "ACCEPT", "WARN"}, f"{label}:{index}: safety-gate decision drift")
    return {"window_id": WINDOW_IDS[index], "target_id": row["target_id"], "physical": physical, "normalized": normalized_values, "progress_sign": sign, "safety_gate_decision": safety, "finite": True, "sampler_steps": 4}


def _load_arm(label: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _validate_provenance(label, spec)
    rows_path_value = spec.get("rows_json")
    require(isinstance(rows_path_value, str) and rows_path_value, f"{label}: rows_json missing")
    rows_path = Path(rows_path_value).resolve()
    expected_rows_sha = _sha(spec.get("rows_sha256"), f"{label}.rows_sha256")
    require(rows_path.is_file() and not rows_path.is_symlink(), f"{label}: rows JSON unavailable")
    require(sha256_file(rows_path) == expected_rows_sha, f"{label}: rows JSON SHA mismatch")
    payload = read_json(rows_path)
    require(isinstance(payload, list) and len(payload) == 24, f"{label}: expected 24 rows")
    rows = [_validate_row(label, index, row, provenance) for index, row in enumerate(payload)]
    require([row["window_id"] for row in rows] == list(WINDOW_IDS), f"{label}: window set/order mismatch")
    require([row["target_id"] for row in rows] == [TARGETS[index // 6] for index in range(24)], f"{label}: target-major order drift")
    return {"label": label, "route": _arm_axes(label)[0], "seed": _arm_axes(label)[1], "step": _arm_axes(label)[2], "provenance": provenance, "rows": rows, "rows_json": str(rows_path), "rows_sha256": expected_rows_sha}


def _summary(values: Sequence[float]) -> dict[str, Any]:
    require(values and all(math.isfinite(float(value)) for value in values), "summary values must be finite/nonempty")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / len(ordered), "median": median, "p95": ordered[p95_index]}


def _optional_summary(values: Sequence[float]) -> dict[str, Any]:
    return _summary(values) if values else {"count": 0, "status": "ALL_UNDEFINED_ACCEPTED"}


def _arm_summary(arm: Mapping[str, Any]) -> dict[str, Any]:
    rows = arm["rows"]
    provenance = arm["provenance"]
    return {
        "label": arm["label"],
        "route": arm["route"],
        "seed": arm["seed"],
        "step": arm["step"],
        "checkpoint_tree_sha256": provenance["checkpoint_tree_sha256"],
        "lineage_artifact_sha256": provenance["lineage_artifact_sha256"],
        "evidence_seal_sha256": provenance["evidence_seal_sha256"],
        "rows_sha256": arm["rows_sha256"],
        "physical": {metric: _optional_summary([row["physical"][metric] for row in rows if row["physical"][metric] is not None]) for metric in PHYSICAL_METRICS},
        "normalized_route_local": {metric: _summary([row["normalized"][metric] for row in rows]) for metric in NORMALIZED_METRICS},
        "progress_sign_counts": {sign: sum(row["progress_sign"] == sign for row in rows) for sign in ("POSITIVE", "ZERO", "NEGATIVE")},
        "safety_gate_counts": {str(decision).lower(): sum(row["safety_gate_decision"] == decision for row in rows) for decision in ("ACCEPT", "WARN", None)},
    }


def _paired(left: Mapping[str, Any], right: Mapping[str, Any], name: str) -> dict[str, Any]:
    require([row["window_id"] for row in left["rows"]] == [row["window_id"] for row in right["rows"]], f"{name}: window pairing drift")
    require([row["target_id"] for row in left["rows"]] == [row["target_id"] for row in right["rows"]], f"{name}: target pairing drift")
    deltas: dict[str, list[float]] = {metric: [] for metric in PHYSICAL_METRICS}
    by_target: dict[str, dict[str, list[float]]] = {target: {metric: [] for metric in PHYSICAL_METRICS} for target in TARGETS}
    transitions: dict[str, int] = {}
    window_rows: list[dict[str, Any]] = []
    for lrow, rrow in zip(left["rows"], right["rows"]):
        delta: dict[str, float | None] = {}
        for metric in PHYSICAL_METRICS:
            lv, rv = lrow["physical"][metric], rrow["physical"][metric]
            delta[metric] = None if lv is None or rv is None else float(rv - lv)
            if delta[metric] is not None:
                deltas[metric].append(float(delta[metric]))
                by_target[lrow["target_id"]][metric].append(float(delta[metric]))
        transition = f"{lrow['progress_sign']}->{rrow['progress_sign']}"
        transitions[transition] = transitions.get(transition, 0) + 1
        window_rows.append({"window_id": lrow["window_id"], "target_id": lrow["target_id"], "physical_delta_right_minus_left": delta, "progress_sign_transition_left_to_right": transition})
    return {
        "name": name,
        "left": left["label"],
        "right": right["label"],
        "window_count": 24,
        "physical_delta_right_minus_left": {metric: _summary(values) for metric, values in deltas.items() if values},
        "physical_delta_exact_sign_counts": {
            metric: {
                "negative": sum(value < 0.0 for value in values),
                "zero": sum(value == 0.0 for value in values),
                "positive": sum(value > 0.0 for value in values),
            }
            for metric, values in deltas.items() if values
        },
        "physical_metric_preferences": PHYSICAL_PREFERENCES,
        "by_target": {target: {metric: _summary(values) for metric, values in metrics.items() if values} for target, metrics in by_target.items()},
        "progress_sign_transition_counts_left_to_right": dict(sorted(transitions.items())),
        "window_rows": window_rows,
        "normalized_cross_route_comparison": False,
        "statistical_claim": "DESCRIPTIVE_PAIRED_ONLY_NO_IID_INFERENCE",
    }


def _three_seed_descriptive(pairs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"route_minus_legacy": {}, "exposure_2000_minus_1000": {}}
    for route in ("N2", "L1"):
        result["route_minus_legacy"][route] = {}
        for step in STEPS:
            records = [pairs[f"{route.lower()}_minus_legacy_seed{seed}_at_{step}"] for seed in SEEDS]
            metrics: dict[str, Any] = {}
            for metric in PHYSICAL_METRICS:
                seed_means = {str(seed): records[index]["physical_delta_right_minus_left"].get(metric, {}).get("mean") for index, seed in enumerate(SEEDS)}
                values = [value for value in seed_means.values() if value is not None]
                metrics[metric] = {
                    "per_seed_mean_delta_right_minus_left": seed_means,
                    "three_seed_descriptive": _optional_summary(values),
                    "seed_mean_exact_sign_counts": {
                        "negative": sum(value < 0.0 for value in values),
                        "zero": sum(value == 0.0 for value in values),
                        "positive": sum(value > 0.0 for value in values),
                        "undefined": 3 - len(values),
                    },
                    "preference": PHYSICAL_PREFERENCES[metric],
                }
            result["route_minus_legacy"][route][str(step)] = metrics
    for route in ROUTES:
        records = [pairs[f"{route.lower()}_exposure_seed{seed}"] for seed in SEEDS]
        metrics = {}
        for metric in PHYSICAL_METRICS:
            seed_means = {str(seed): records[index]["physical_delta_right_minus_left"].get(metric, {}).get("mean") for index, seed in enumerate(SEEDS)}
            values = [value for value in seed_means.values() if value is not None]
            metrics[metric] = {
                "per_seed_mean_delta_2000_minus_1000": seed_means,
                "three_seed_descriptive": _optional_summary(values),
                "seed_mean_exact_sign_counts": {
                    "negative": sum(value < 0.0 for value in values),
                    "zero": sum(value == 0.0 for value in values),
                    "positive": sum(value > 0.0 for value in values),
                    "undefined": 3 - len(values),
                },
                "preference": PHYSICAL_PREFERENCES[metric],
            }
        result["exposure_2000_minus_1000"][route] = metrics
    result["replication_unit"] = "training seed"
    result["window_iid_inference"] = False
    result["p_value_computed"] = False
    return result


def _failure_taxonomy(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "CORELAB_STAGE45F_THREE_ROUTE_FAILURE_TAXONOMY_V1", "arms": {}}
    for label, arm in arms.items():
        rows = arm["rows"]
        result["arms"][label] = {
            "row_count": len(rows),
            "nonfinite": sum(not row["finite"] for row in rows),
            "sampler_step_mismatch": sum(row["sampler_steps"] != 4 for row in rows),
            "target_xyz_hidden_input": 0,
            "positive_progress": sum(row["progress_sign"] == "POSITIVE" for row in rows),
            "zero_progress": sum(row["progress_sign"] == "ZERO" for row in rows),
            "negative_progress": sum(row["progress_sign"] == "NEGATIVE" for row in rows),
            "safety_warn": sum(row["safety_gate_decision"] == "WARN" for row in rows),
            "normalized_clipping_rows": sum(row["normalized"]["raw_active_outside_neg1_pos1_fraction"] > 0.0 for row in rows),
        }
    result["policy"] = "Physical metrics are primary across routes; normalized metrics are route-local; no iid significance or automatic winner."
    return result


def load_index(path: Path) -> dict[str, Any]:
    value = read_json(path)
    cfg = manifest()
    require(value.get("schema") == "STAGE45F_H200_TRAIN24_EVIDENCE_INDEX_V1", "evidence index schema drift")
    require(value.get("split") == "TRAIN24" and value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "evidence index split boundary drift")
    arms_value = value.get("arms")
    require(isinstance(arms_value, Mapping), "evidence index arms missing")
    expected_labels = list(cfg["matrix"]["arms"])
    # JSON object key order is not an evidence identity (and canonical JSON
    # writers commonly sort keys).  The packet iterates the frozen manifest
    # order below, while still failing closed on any missing or extra arm.
    require(set(arms_value.keys()) == set(expected_labels), "evidence index arm set drift")
    arms = {label: _load_arm(label, arms_value[label]) for label in expected_labels}
    require(all(arm["route"] in ROUTES and arm["seed"] in SEEDS and arm["step"] in STEPS for arm in arms.values()), "evidence index arm axes invalid")
    return {"schema": value["schema"], "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False, "arms": arms, "source_index_sha256": sha256_file(path), "source_index_path": str(Path(path).resolve())}


def build_packet(index_path: Path, output_path: Path, campaign_path: Path | None = None) -> dict[str, Any]:
    campaign_verification: dict[str, Any] | None = None
    if campaign_path is not None:
        # Keep Vanda admission on the same public packet path.  A packet must
        # never be built from a V2 handoff that has not passed the exact
        # producer/schema/provenance adapter first.
        campaign_verification = verify_campaign(Path(campaign_path))
    index = load_index(index_path)
    arms = index["arms"]
    summaries = {label: _arm_summary(arm) for label, arm in arms.items()}
    pairs: dict[str, Any] = {}
    for seed in SEEDS:
        for step in STEPS:
            legacy = arms[f"LEGACY(seed{seed})@{step}"]
            for route in ("N2", "L1"):
                right = arms[f"{route}(seed{seed})@{step}"]
                pairs[f"{route.lower()}_minus_legacy_seed{seed}_at_{step}"] = _paired(legacy, right, f"{route}_minus_LEGACY_seed{seed}_at_{step}")
        for route in ROUTES:
            left = arms[f"{route}(seed{seed})@1000"]
            right = arms[f"{route}(seed{seed})@2000"]
            pairs[f"{route.lower()}_exposure_seed{seed}"] = _paired(left, right, f"{route}_exposure_seed{seed}")
    taxonomy = _failure_taxonomy(arms)
    three_seed = _three_seed_descriptive(pairs)
    packet = {
        "schema": "CORELAB_STAGE45F_THREE_ROUTE_TRAIN24_DECISION_PACKET_V1",
        "status": "PASS_DESCRIPTIVE_H200_TRAIN24_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "winner_declared": False,
        "automatic_winner": False,
        "posthoc_threshold_used": False,
        "iid_significance_test_performed": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "split": "TRAIN24",
        "window_count": 24,
        "routes": list(ROUTES),
        "seeds": list(SEEDS),
        "checkpoint_steps": list(STEPS),
        "primary_metric_space": "decoded physical metrics",
        "normalized_metric_space": "route-specific descriptive/mechanistic only",
        "source_index_sha256": index["source_index_sha256"],
        "source_index_path": index["source_index_path"],
        "source_campaign_result": campaign_verification,
        "arm_summaries": summaries,
        "paired_comparisons": pairs,
        "three_seed_descriptive": three_seed,
        "failure_taxonomy": taxonomy,
        "claim_boundary": {
            "same_seed_same_window_pairing": True,
            "same_target_breakdown": True,
            "three_seed_replication_axis": True,
            "windows_are_not_iid_training_replications": True,
            "no_automatic_winner": True,
            "no_validation_generalization_claim": True,
            "historical_4080s_n2_context_excluded_from_primary_h200_matrix": True,
        },
        "execution_boundary": {
            "gpu": 0, "model_construction": 0, "model_forward": 0,
            "training": 0, "backward": 0, "optimizer_steps": 0,
            "pbs": 0, "Validation-E": 0, "Holdout-F": 0,
            "WBC": 0, "simulator": 0, "hardware": 0,
        },
        "next_action": "PI must review descriptive H200 TRAIN24 evidence and issue a separate recipe-freeze/Validation-E gate.",
    }
    write_json(output_path, packet)
    return packet


def _expected_phase_budget(phase: str, route: str) -> dict[str, int]:
    arms = {"E4": {"LEGACY": 4, "L1": 4, "N2": 6}, "D1": {"LEGACY": 4, "L1": 6, "N2": 6}, "D2A": {"LEGACY": 4, "L1": 6, "N2": 6}}[phase][route]
    calls = {"E4": 4, "D1": 9, "D2A": 8}[phase]
    rows = {"E4": 1, "D1": 9, "D2A": 4}[phase]
    return {"arms": arms, "conditioning_passes": arms * 24, "action_head_calls": arms * 24 * calls, "raw_rows": arms * 24 * rows}


def _verify_inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory_path = inventory_path.resolve()
    require(inventory_path.is_relative_to(root), "campaign inventory must be inside campaign root")
    _safe_input_path(inventory_path)
    inventory = read_json(inventory_path)
    require(inventory.get("status") == "PASS" and isinstance(inventory.get("entries"), list), "campaign inventory is not sealed PASS")
    for index, entry in enumerate(inventory["entries"]):
        require(isinstance(entry, Mapping), f"inventory entry {index} is not an object")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, f"inventory path {index} escapes campaign root")
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), f"inventory path {index} escapes root")
        _safe_input_path(path)
        require(path.is_file() and not path.is_symlink(), f"inventory file missing/unsafe: {path}")
        require(sha256_file(path) == entry.get("sha256") and path.stat().st_size == entry.get("bytes"), f"inventory hash/size drift: {path}")
    return {"sha256": sha256_file(inventory_path), "entry_count": len(inventory["entries"])}


def _verify_campaign_v1(campaign_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    value = read_json(campaign_path)
    require(value.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V1", "Vanda final result schema drift")
    require(value.get("status") == "PASS" and value.get("token") == "STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_COMPLETE_FOR_PI_REVIEW", "Vanda campaign is not a completed PASS")
    require(value.get("training") is False and value.get("Validation-E") == 0 and value.get("Holdout-F") == 0, "Vanda final boundary drift")
    require(value.get("WBC", 0) == 0 and value.get("simulator", 0) == 0 and value.get("hardware", 0) == 0 and value.get("merge") is False, "Vanda downstream boundary drift")
    require(value.get("total_model_side_calls") == 8976 and value.get("fresh_training_lineages") == 7, "Vanda campaign total budget/lineage drift")
    root_value = value.get("campaign_root")
    require(isinstance(root_value, str) and root_value, "Vanda campaign root missing")
    root = Path(root_value).resolve()
    _safe_input_path(root)
    require(root.is_dir() and not root.is_symlink(), f"Vanda campaign root unavailable: {root}")
    summary_path = Path(value.get("cpu_summary", "")).resolve()
    inventory_path = Path(value.get("cpu_inventory", "")).resolve()
    require(summary_path.is_relative_to(root), "Vanda CPU summary must be inside campaign root")
    require(inventory_path.is_relative_to(root), "Vanda CPU inventory must be inside campaign root")
    _safe_input_path(summary_path)
    _safe_input_path(inventory_path)
    require(summary_path.is_file() and inventory_path.is_file(), "Vanda CPU summary/inventory unavailable")
    require(sha256_file(summary_path) == value.get("cpu_summary_sha256"), "Vanda CPU summary SHA drift")
    require(sha256_file(inventory_path) == value.get("cpu_inventory_sha256"), "Vanda CPU inventory SHA drift")
    inventory = _verify_inventory(root, inventory_path)
    summary = read_json(summary_path)
    require(summary.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_CPU_EVIDENCE_SUMMARY_V1" and summary.get("status") == "PASS_READONLY_FROM_SEALED_GPU_EVIDENCE", "Vanda CPU summary schema/status drift")
    require(summary.get("descriptive_only") is True and summary.get("no_performance_based_winner") is True, "Vanda summary claim boundary drift")
    require(summary.get("Validation-E") == 0 and summary.get("Holdout-F") == 0, "Vanda summary locked split drift")
    totals_expected = {phase: {key: sum(_expected_phase_budget(phase, route)[key] for route in ROUTES) for key in ("conditioning_passes", "action_head_calls", "raw_rows")} for phase in ("E4", "D1", "D2A")}
    for phase, expected in totals_expected.items():
        expected["total_model_side_calls"] = expected["conditioning_passes"] + expected["action_head_calls"]
    require(summary.get("phase_totals") == totals_expected, "Vanda CPU phase totals drift")
    phase_seals = summary.get("phase_seals")
    require(isinstance(phase_seals, Mapping), "Vanda phase seal map missing")
    checked_phase_seals = 0
    for phase in ("E4", "D1", "D2A"):
        require(isinstance(phase_seals.get(phase), Mapping), f"Vanda {phase} seal map missing")
        for route in ROUTES:
            record = phase_seals[phase].get(route)
            require(isinstance(record, Mapping) and record.get("route") == route and record.get("phase") == phase, f"Vanda {phase}/{route} seal identity drift")
            require(record.get("budget") == _expected_phase_budget(phase, route), f"Vanda {phase}/{route} budget drift")
            seal_path = Path(record.get("path", "")).resolve()
            require(seal_path.is_file() and sha256_file(seal_path) == record.get("sha256"), f"Vanda {phase}/{route} seal hash drift")
            seal = read_json(seal_path)
            require(seal.get("status") == "PASS" and seal.get("route") == route and seal.get("phase") == phase, f"Vanda {phase}/{route} seal status drift")
            require(seal.get("raw_before_derived") is True and seal.get("public_sampler_calls") == 0 and seal.get("get_action_calls") == 0, f"Vanda {phase}/{route} raw/sampler boundary drift")
            require(seal.get("Validation-E") == 0 and seal.get("Holdout-F") == 0 and seal.get("WBC") == 0 and seal.get("simulator") == 0 and seal.get("hardware") == 0, f"Vanda {phase}/{route} forbidden boundary drift")
            inner_path = Path(seal.get("inner_seal", "")).resolve()
            require(inner_path.is_file() and sha256_file(inner_path) == seal.get("inner_seal_sha256"), f"Vanda {phase}/{route} inner seal hash drift")
            require(read_json(inner_path).get("status") == "PASS", f"Vanda {phase}/{route} inner seal is not PASS")
            checked_phase_seals += 1
    result = {
        "schema": "CORELAB_STAGE45F_VANDA_H200_CAMPAIGN_VERIFICATION_V1",
        "status": "PASS_READONLY_PROVENANCE_VERIFIED",
        "campaign_result_path": str(Path(campaign_path).resolve()),
        "campaign_result_sha256": sha256_file(campaign_path),
        "campaign_root": str(root),
        "cpu_summary_sha256": sha256_file(summary_path),
        "cpu_inventory_sha256": inventory["sha256"],
        "checked_phase_seals": checked_phase_seals,
        "expected_phase_seals": 9,
        "route_matrix": "LEGACY/N2/L1 x seeds 2222/2223/2224 x @1000/@2000",
        "automatic_winner": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_boundary": {"gpu": 0, "model_forward": 0, "training": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def verify_campaign(campaign_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Verify a Vanda final result through the real C10 entrypoint.

    PI0145 explicitly requires the producer's V2 document to be adapted at
    admission time.  Dispatching here (rather than asking callers to invoke a
    helper manually) keeps the actual ``verify-campaign`` and downstream
    packet entrypoint on one fail-closed path.  V1 remains readable for
    historical compatibility; V2 is handled by the typed producer adapter.
    """

    value = read_json(campaign_path)
    schema = value.get("schema")
    if schema == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V2":
        from stage45f_corelab_pi0145_v2_adapter import verify_v2_campaign

        return verify_v2_campaign(campaign_path, output_path)
    require(schema == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V1", "unsupported Vanda final result schema; no heuristic relabeling")
    return _verify_campaign_v1(campaign_path, output_path)


def recipe_freeze_plan() -> dict[str, Any]:
    cfg = manifest()
    return {
        "schema": "CORELAB_STAGE45F_RECIPE_FREEZE_PLAN_V1",
        "status": "LOCKED_PLAN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "candidate_must_be_pi_selected": True,
        "selection_source_split": "TRAIN24",
        "required_fields": cfg["recipe_freeze"]["required_fields"],
        "validation_generation": 1,
        "requires_train24_packet_sha256": True,
        "requires_evidence_inventory_sha256": True,
        "automatic_selection": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execute_supported": False,
        "next_gate": "new PI gate must select exactly one candidate and freeze all identities before any Validation-E read",
    }


def validation_plan() -> dict[str, Any]:
    return {
        "schema": "CORELAB_STAGE45F_VALIDATION_E_ADMISSION_PLAN_V1",
        "status": "LOCKED_PLAN_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "validation_e_read": False,
        "holdout_f_read": False,
        "paths_may_be_opened": False,
        "read_authorized": False,
        "requires_new_pi_gate": True,
        "requires_recipe_freeze_before_read": True,
        "requires_empty_prior_access_ledger_for_first_unseen_claim": True,
        "no_post_validation_tuning_under_same_selection_decision": True,
        "execution_supported": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("static", "verify-campaign", "verify-vanda-v2-synthetic", "build-packet", "recipe-freeze-plan", "validation-plan"), default="static")
    value.add_argument("--input", type=Path)
    value.add_argument("--campaign", type=Path, help="optional Vanda final result to admit before packet construction")
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "static":
        print(json.dumps(static_audit(), indent=2, sort_keys=True, allow_nan=False)); return 0
    if args.mode == "recipe-freeze-plan":
        print(json.dumps(recipe_freeze_plan(), indent=2, sort_keys=True, allow_nan=False)); return 0
    if args.mode == "validation-plan":
        print(json.dumps(validation_plan(), indent=2, sort_keys=True, allow_nan=False)); return 0
    require(args.input is not None, f"--input required for {args.mode}")
    if args.mode == "build-packet":
        require(args.output is not None, "--output required for build-packet")
        print(json.dumps(build_packet(args.input, args.output, args.campaign), indent=2, sort_keys=True, allow_nan=False)); return 0
    if args.mode == "verify-vanda-v2-synthetic":
        from stage45f_corelab_pi0145_v2_adapter import verify_synthetic_v2_fixture

        result = verify_synthetic_v2_fixture(args.input)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0
    result = verify_campaign(args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConvergenceError as exc:
        print(f"CORELAB_STAGE45F_EVIDENCE_CONVERGENCE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
