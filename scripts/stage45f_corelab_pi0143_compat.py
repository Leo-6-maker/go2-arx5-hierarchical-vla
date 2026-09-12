#!/usr/bin/env python3
"""CPU-only Stage45F C10 compatibility adapters (PI0143 B0).

This module consumes sealed metadata and raw/derived row descriptions.  It
never imports a model framework and never fabricates producer evidence.  All
source-schema conversion is explicit and records the source schema, source
SHA, and converter identity in the canonical row.
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
MANIFEST_PATH = REPO / "manifests/STAGE45F_CORELAB_PI0143_C10_COMPATIBILITY_V1.json"
ROUTES = ("LEGACY", "N2", "L1")
SEEDS = (2222, 2223, 2224)
STEPS = (1000, 2000)
PHYSICAL = ("final_target_distance_m", "absolute_progress_m", "minimum_target_distance_m", "endpoint_target_vector_cosine", "path_length_m", "roughness_position_second_difference_m")
SOURCE_CONVERTERS = {
    "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1": "STAGE45F_C10_ROW_CONVERTER_STAGE45C_V1",
    "CORELAB_STAGE45E_LEGACY_E4_ROW_V1": "STAGE45F_C10_ROW_CONVERTER_LEGACY_E4_V1",
    "CORELAB_STAGE45E_LEGACY_D1_ROW_V1": "STAGE45F_C10_ROW_CONVERTER_LEGACY_D1_V1",
    "CORELAB_STAGE45E_LEGACY_D2A_ROW_V1": "STAGE45F_C10_ROW_CONVERTER_LEGACY_D2A_V1",
}
FORBIDDEN_IMPORTS = {"torch", "numpy", "transformers", "gr00t", "subprocess", "isaacgym", "omni"}


class CompatibilityError(convergence.ConvergenceError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def _sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and value == value.lower() and all(char in "0123456789abcdef" for char in value), f"{field} must be lowercase SHA256")
    return value


def _git_sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 40 and value == value.lower() and all(char in "0123456789abcdef" for char in value), f"{field} must be lowercase Git SHA")
    return value


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None:
        require(nullable, f"{field} is null")
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"{field} is non-finite/non-numeric")
    return float(value)


def _read(path: Path) -> Any:
    try:
        return convergence.read_json(Path(path).resolve())
    except convergence.ConvergenceError as exc:
        raise CompatibilityError(str(exc)) from exc


def _write(path: Path, value: Any) -> None:
    convergence.write_json(Path(path).resolve(), value)


def manifest() -> dict[str, Any]:
    value = _read(MANIFEST_PATH)
    require(value.get("schema") == "STAGE45F_CORELAB_PI0143_C10_COMPATIBILITY_V1", "PI0143 compatibility manifest schema drift")
    require(value.get("authority", {}).get("pi_gate") == "PI_GATE_STATE_0143", "PI0143 authority drift")
    require(value.get("execution_boundary", {}).get("gpu") == 0 and value.get("execution_boundary", {}).get("model_forward") == 0, "compatibility execution boundary drift")
    return value


def _route_contract(route: str) -> Mapping[str, Any]:
    value = convergence.manifest()["h200_campaign_surfaces"]["routes"].get(route)
    require(isinstance(value, Mapping), f"route contract missing: {route}")
    return value


def canonicalize_row(row: Mapping[str, Any], source_schema: str, source_sha256: str, converter_identity: str | None = None) -> dict[str, Any]:
    """Convert one explicitly permitted source row; never guess fields."""
    require(source_schema in SOURCE_CONVERTERS, f"unsupported source schema: {source_schema}")
    _sha(source_sha256, "source_sha256")
    converter = converter_identity or SOURCE_CONVERTERS[source_schema]
    require(converter == SOURCE_CONVERTERS[source_schema], "converter identity is not the registered explicit adapter")
    require(isinstance(row.get("window_id"), str) and row["window_id"], "source row window_id missing")
    target_id = row.get("target_id")
    if target_id is None and isinstance(row.get("identity"), Mapping):
        target_id = row["identity"].get("target_id")
    require(isinstance(target_id, str) and target_id in convergence.TARGETS, "source row target_id missing")
    physical: dict[str, float | None] = {}
    target = row.get("metrics", {}).get("target", {}) if isinstance(row.get("metrics"), Mapping) else {}
    if source_schema in {"CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1", "CORELAB_STAGE45E_LEGACY_E4_ROW_V1"}:
        roughness = target.get("roughness_position_second_difference_m")
        if isinstance(roughness, Mapping):
            roughness = roughness.get("mean")
        for field in PHYSICAL:
            physical[field] = _finite(target.get(field) if field != "roughness_position_second_difference_m" else roughness, f"source.{field}", nullable=(field == "endpoint_target_vector_cosine"))
        progress_sign = target.get("progress_sign")
        require(progress_sign in {"POSITIVE", "ZERO", "NEGATIVE"}, "source progress sign drift")
    else:
        physical = {field: None for field in PHYSICAL}
        progress_sign = None
    normalized_source = row.get("normalized_metrics")
    if not isinstance(normalized_source, Mapping):
        normalized_source = row.get("normalized_local_flow")
    normalized = dict(normalized_source) if isinstance(normalized_source, Mapping) else {}
    if source_schema in {"CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1", "CORELAB_STAGE45E_LEGACY_E4_ROW_V1"}:
        require(set(("active_rmse", "active_mae", "active_max_abs", "raw_active_outside_neg1_pos1_fraction")) <= set(normalized), "source normalized metric fields missing")
    require(row.get("target_xyz_in_model_input", row.get("identity", {}).get("target_xyz_in_model_input", False)) is False, "target XYZ hidden input marked true")
    return {
        "schema": "STAGE45F_C10_CANONICAL_TRAIN24_ROW_V1",
        "window_id": row["window_id"],
        "target_id": target_id,
        "physical": physical,
        "normalized_route_local": normalized,
        "progress_sign": progress_sign,
        "finite": row.get("finite", True) is True,
        "target_xyz_in_model_input": False,
        "source": {"schema": source_schema, "sha256": source_sha256, "converter_identity": converter},
    }


def canonicalize_rows(rows: Sequence[Mapping[str, Any]], source_schema: str, source_sha256: str, converter_identity: str | None = None) -> dict[str, Any]:
    require(len(rows) == 24, "canonical TRAIN24 row count must be 24")
    canonical = [canonicalize_row(row, source_schema, source_sha256, converter_identity) for row in rows]
    require([row["window_id"] for row in canonical] == list(convergence.WINDOW_IDS), "canonical TRAIN24 window descriptor/order drift")
    require([row["target_id"] for row in canonical] == [convergence.TARGETS[index // 6] for index in range(24)], "canonical TRAIN24 target order drift")
    require(all(row["finite"] for row in canonical), "canonical row finite flag drift")
    return {"schema": "STAGE45F_C10_CANONICAL_TRAIN24_ROW_BUNDLE_V1", "row_count": 24, "source_schema": source_schema, "source_sha256": source_sha256, "converter_identity": converter_identity or SOURCE_CONVERTERS[source_schema], "rows": canonical, "target_xyz_in_model_input": False, "primary_physical_metrics_preserved": True}


def validate_lineage(record: Mapping[str, Any]) -> dict[str, Any]:
    source_kind = record.get("source_kind")
    variants = manifest()["lineage_variants"]
    require(source_kind in variants, "unsupported lineage source kind")
    variant = variants[source_kind]
    require(record.get("producer_fabrication") is False, "producer-side lineage fabrication")
    _git_sha(record.get("training_head"), "training_head")
    for field in ("evaluation_head", "adapter_head", "orchestration_head"):
        _git_sha(record.get(field), field)
    if variant.get("training_result_required"):
        require(isinstance(record.get("training_result"), Mapping) and record["training_result"].get("schema"), "standard lineage training result missing")
    else:
        require(record.get("training_result") is None and isinstance(record.get("recovered_input_v2_sha256"), str), "recovered lineage fabricated or missing V2 input")
        _sha(record["recovered_input_v2_sha256"], "recovered_input_v2_sha256")
        require(record.get("primary_retry_substitution") is False, "recovered primary substituted by retry")
    _sha(record.get("checkpoint_tree_sha256"), "checkpoint_tree_sha256")
    return {"source_kind": source_kind, "training_result_present": isinstance(record.get("training_result"), Mapping), "recovered_primary": source_kind == "RECOVERED_PRIMARY", "producer_fabrication": False}


def validate_route_identities(record: Mapping[str, Any], route: str) -> dict[str, Any]:
    contract = _route_contract(route)
    require(record.get("route") == route, "route identity drift")
    for field in ("training_head", "evaluation_head", "adapter_head", "orchestration_head"):
        _git_sha(record.get(field), field)
    require(record.get("route_branch") == contract["branch"], "route branch drift")
    require(record.get("normalization_identity") == contract["normalization"], "route normalization drift")
    require(record.get("loss_identity") == contract["loss"], "route loss identity drift")
    return {"route": route, "route_branch": contract["branch"], "training_head": record["training_head"], "evaluation_head": record["evaluation_head"], "adapter_head": record["adapter_head"], "orchestration_head": record["orchestration_head"]}


def validate_external_diagnostic_handoff(handoff_path: Path, expected_root: Path | None = None) -> dict[str, Any]:
    value = _read(handoff_path)
    require(value.get("schema") == "STAGE45E_EXTERNAL_DIAGNOSTIC_HANDOFF_V1", "external diagnostic handoff schema drift")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "external handoff locked split drift")
    artifacts = value.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "external artifact inventory missing")
    checked = []
    for item in artifacts:
        require(isinstance(item, Mapping) and isinstance(item.get("path"), str), "external artifact entry malformed")
        path = Path(item["path"]).resolve()
        if expected_root is not None:
            require(path.is_relative_to(expected_root.resolve()) is False, "external handoff unexpectedly nested in campaign root")
        require(path.is_file() and not path.is_symlink(), f"external artifact unavailable: {path}")
        require(convergence.sha256_file(path) == _sha(item.get("sha256"), "external artifact sha256"), f"external artifact SHA drift: {path}")
        checked.append({"path": str(path), "sha256": item["sha256"], "schema": item.get("schema")})
    return {"schema": "STAGE45F_C10_EXTERNAL_DIAGNOSTIC_HANDOFF_VERIFIED_V1", "status": "PASS_READONLY", "artifact_count": len(checked), "artifacts": checked}


def validate_campaign_accounting(campaign: Mapping[str, Any]) -> dict[str, Any]:
    history = campaign.get("campaign_history")
    closure = campaign.get("cpu_closure_execution")
    require(isinstance(history, Mapping) and isinstance(closure, Mapping), "campaign history/CPU closure separation missing")
    require(history.get("training") is True and closure.get("training") is False, "training history was collapsed into CPU closure")
    accepted = campaign.get("accepted_unique_scientific_calls")
    attempted = campaign.get("all_attempted_model_calls")
    failed = campaign.get("failed_attempt_model_calls")
    require(isinstance(accepted, int) and isinstance(attempted, int) and isinstance(failed, int) and accepted >= 0 and attempted >= accepted and failed >= 0, "accepted/attempted accounting malformed")
    require(campaign.get("reused_readonly_model_calls") == 0 and accepted <= 8976, "campaign call accounting boundary drift")
    return {"accepted_unique_scientific_calls": accepted, "all_attempted_model_calls": attempted, "failed_attempt_model_calls": failed, "reused_readonly_model_calls": 0, "campaign_history_training": True, "cpu_closure_training": False}


def validate_window_noise_sampler(contract: Mapping[str, Any]) -> dict[str, Any]:
    require(contract.get("split") == "TRAIN24" and contract.get("window_count") == 24, "TRAIN24 descriptor drift")
    require(contract.get("window_ids") == list(convergence.WINDOW_IDS), "TRAIN24 window IDs/order drift")
    noise = contract.get("noise_sha256_by_window")
    require(isinstance(noise, Mapping) and list(noise) == list(convergence.WINDOW_IDS), "per-window noise identity missing/order drift")
    for window, digest in noise.items():
        _sha(digest, f"noise_sha256_by_window.{window}")
    sampler = contract.get("sampler")
    require(isinstance(sampler, Mapping) and sampler.get("name") and sampler.get("steps_observed") == 4 and sampler.get("official_public_sampler_called") is False, "sampler identity drift")
    require(contract.get("target_xyz_in_model_input") is False, "target XYZ input drift")
    return {"split": "TRAIN24", "window_count": 24, "noise_identity_bound": True, "sampler_identity": dict(sampler), "target_xyz_in_model_input": False}


def decoder_semantic_equivalence(payload: Mapping[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == "STAGE45F_C10_DECODER_SEMANTIC_EQUIVALENCE_INPUT_V1", "decoder equivalence input schema drift")
    _sha(payload.get("stats_payload_sha256"), "stats_payload_sha256")
    left, right = payload.get("stage45c_v2"), payload.get("stage45e_with_stats")
    require(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) and left, "decoder equivalence arrays missing")
    max_abs = 0.0
    for index, (lrow, rrow) in enumerate(zip(left, right)):
        require(isinstance(lrow, list) and isinstance(rrow, list) and len(lrow) == len(rrow), f"decoder equivalence shape drift at {index}")
        for col, (lv, rv) in enumerate(zip(lrow, rrow)):
            _finite(lv, f"stage45c[{index}][{col}]")
            _finite(rv, f"stage45e[{index}][{col}]")
            max_abs = max(max_abs, abs(float(lv) - float(rv)))
    require(max_abs <= float(payload.get("max_abs_tolerance", 1e-12)), "decoder/metric numerical semantic mismatch")
    return {"schema": "STAGE45F_C10_DECODER_SEMANTIC_EQUIVALENCE_RESULT_V1", "status": "PASS", "max_abs": max_abs, "stats_payload_sha256": payload["stats_payload_sha256"], "primary_physical_metrics_preserved": True}


def reconstruct_l1_secondary(raw_rows: Sequence[Mapping[str, Any]], train_descriptor: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    require(train_descriptor.get("split") == "TRAIN24" and train_descriptor.get("source_kind") == "SEALED_TRAIN_DESCRIPTOR", "immutable TRAIN descriptor missing")
    require(stats.get("normalization") == "FULL24_RUNTIME_V2", "L1 reconstruction stats route drift")
    _sha(stats.get("payload_sha256"), "stats.payload_sha256")
    require(len(raw_rows) == 24, "L1 raw secondary row count drift")
    reconstructed = []
    for index, raw in enumerate(raw_rows):
        require(raw.get("schema") == "VANDA_STAGE45E_L1_RAW_PREDICTION_ARTIFACT_V1", "L1 raw artifact schema drift")
        require(raw.get("window_id") == convergence.WINDOW_IDS[index], "L1 raw window drift")
        source = raw.get("normalized_secondary_raw")
        require(isinstance(source, Mapping), "missing sealed raw normalized secondary values; no invention allowed")
        normalized = {key: _finite(source.get(key), f"raw.normalized_secondary_raw.{key}") for key in ("active_rmse", "active_mae", "active_max_abs", "raw_active_outside_neg1_pos1_fraction")}
        reconstructed.append({"window_id": raw["window_id"], "normalized_route_local": normalized, "physical": None, "source_kind": "RECONSTRUCTED_SECONDARY_DIAGNOSTIC", "primary_physical_metrics_source": "SEALED_HISTORICAL_EVALUATION"})
    return {"schema": "STAGE45F_C10_L1_RECONSTRUCTED_SECONDARY_DIAGNOSTIC_V1", "status": "PASS_CPU_ONLY", "rows": reconstructed, "primary_physical_metrics_replaced": False, "source_stats_payload_sha256": stats["payload_sha256"], "validation_e_read": False, "holdout_f_read": False}


def l1_vs_n2_paired_delta(arms: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    deltas = {}
    for seed in SEEDS:
        for step in STEPS:
            l1 = arms.get(f"L1(seed{seed})@{step}")
            n2 = arms.get(f"N2(seed{seed})@{step}")
            require(isinstance(l1, Sequence) and isinstance(n2, Sequence) and len(l1) == len(n2) == 24, f"L1/N2 paired arm missing: seed{seed}@{step}")
            require([row.get("window_id") for row in l1] == [row.get("window_id") for row in n2], "L1/N2 window pairing drift")
            rows = []
            for left, right in zip(l1, n2):
                rows.append({"window_id": left["window_id"], "target_id": left.get("target_id"), "physical_delta_n2_minus_l1": {metric: _finite(right.get("physical", {}).get(metric), f"n2.{metric}", nullable=True) - _finite(left.get("physical", {}).get(metric), f"l1.{metric}", nullable=True) if right.get("physical", {}).get(metric) is not None and left.get("physical", {}).get(metric) is not None else None for metric in PHYSICAL}})
            deltas[f"seed{seed}@{step}"] = rows
    return {"schema": "STAGE45F_L1_VS_N2_PAIRED_PHYSICAL_DELTA_V1", "status": "PASS_DESCRIPTIVE_ONLY", "pairing": "same seed x same tier x same TRAIN24 window", "deltas": deltas, "winner_declared": False, "normalized_cross_route_comparison": False}


def strict_recipe_freeze(request: Mapping[str, Any]) -> dict[str, Any]:
    require(request.get("schema") == "STAGE45F_C10_RECIPE_FREEZE_REQUEST_V1", "recipe freeze request schema drift")
    require(request.get("route") in ROUTES and request.get("checkpoint_step") in STEPS, "recipe freeze must select one route and one tier")
    require(request.get("seeds") == list(SEEDS), "recipe freeze requires exactly seeds 2222/2223/2224")
    require(isinstance(request.get("checkpoint_trees"), Mapping) and set(request["checkpoint_trees"]) == {f"{request['route']}(seed{seed})@{request['checkpoint_step']}" for seed in SEEDS}, "recipe freeze checkpoint set must be exact three-seed single tier")
    for label, digest in request["checkpoint_trees"].items():
        _sha(digest, f"checkpoint_trees.{label}")
    require(request.get("best_seed_selection") is False and request.get("mixed_tier") is False and request.get("automatic_winner") is False, "recipe freeze selection policy drift")
    require(isinstance(request.get("pi_selection_comment"), int) and request["pi_selection_comment"] > 0, "recipe freeze requires explicit PI selection")
    return {"schema": "STAGE45F_C10_RECIPE_FREEZE_V1", "status": "PI_SELECTED_PLAN_ONLY", "route": request["route"], "checkpoint_step": request["checkpoint_step"], "seeds": list(SEEDS), "checkpoint_trees": dict(request["checkpoint_trees"]), "automatic_winner": False, "validation_execution_enabled": False, "validation_e_read": False, "holdout_f_read": False}


def static_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_IMPORTS))
    require(not forbidden, f"forbidden top-level imports: {forbidden}")
    cfg = manifest()
    return {"schema": "STAGE45F_CORELAB_PI0143_C10_COMPATIBILITY_STATIC_AUDIT_V1", "status": "PASS", "manifest_sha256": convergence.sha256_file(MANIFEST_PATH), "converter_count": len(cfg["canonical_row_adapters"]), "lineage_variants": len(cfg["lineage_variants"]), "forbidden_top_level_imports": forbidden, "execution_boundary": cfg["execution_boundary"]}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("static", "accounting", "freeze"), default="static")
    value.add_argument("--input", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "static":
        result = static_audit()
    elif args.mode == "accounting":
        require(args.input is not None, "--input required")
        result = validate_campaign_accounting(_read(args.input))
    else:
        require(args.input is not None, "--input required")
        result = strict_recipe_freeze(_read(args.input))
    if args.output:
        _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompatibilityError as exc:
        print(f"STAGE45F_CORELAB_PI0143_COMPATIBILITY_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
