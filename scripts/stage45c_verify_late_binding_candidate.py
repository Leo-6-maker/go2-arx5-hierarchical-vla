#!/usr/bin/env python3
"""Verify the pathless Stage45C Corelab late-binding candidate.

This verifier is deliberately standard-library only.  It checks identities,
ordering, ABI, sampler/decode/metric declarations, and the zero-exposure
boundary without opening any runtime, checkpoint, dataset, or model artifact.
It is a candidate verifier, not the final Stage45C execution admission gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "STAGE45C_TRAIN24_LATE_BINDING_CANDIDATE_V1.json"

SCHEMA = "CORELAB_STAGE45C_TRAIN24_LATE_BINDING_CANDIDATE_V1"
STATUS = "LATE_BINDING_CANDIDATE_NOT_FROZEN"
PREP_SHA = "4ff83c03d1071560c99070882ebd86b69c6c92f3"
GOLDEN_HEAD = "8197e4cd591b9c44cf9f43576d3020e9e6ff5f24"
PROMOTION_SHA = "a6f803e257ceeb0434027d8a35e32a7b8c66baf8226234bd6299cebf417446f9"
INTEGRITY_SHA = "a672e39b45e025f9a7681c1a1a45d807dd9be2a4641e54db60b036252bb59539"
FULL24_1000 = "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c"
FULL24_2000 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_SPLIT = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
STRUCTURED_1000 = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
STRUCTURED_STATS = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
RUNTIME_STATS = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_PAYLOAD = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_SOURCE = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
DECODER = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"
METRICS = "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4"

WINDOWS = tuple(
    f"stage22_expert_red_target_train_{target}:expert-{index:04d}"
    for target in ("a", "b", "c", "d")
    for index in range(6)
)


class CandidateError(ValueError):
    """Raised when a late-binding candidate violates a frozen predicate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def _equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_strings(value: Any, prefix: str = "root") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_walk_strings(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        found.append((prefix, value))
    return found


def _reject_node_local_paths(manifest: Mapping[str, Any]) -> None:
    for location, value in _walk_strings(manifest):
        lowered = value.lower()
        _require("/scratch/" not in lowered, f"node-local scratch path at {location}")
        _require("/var/cache/" not in lowered, f"node-local cache path at {location}")
        _require("/home/" not in lowered, f"node-local home path at {location}")


def _validate_authorities(manifest: Mapping[str, Any]) -> None:
    authorities = manifest.get("accepted_authorities")
    _require(isinstance(authorities, Mapping), "accepted_authorities is required")
    _equal(authorities["golden_release_promotion_manifest"]["sha256"], PROMOTION_SHA, "promotion authority")
    _equal(authorities["checkpoint_integrity_delta_audit"]["sha256"], INTEGRITY_SHA, "integrity authority")
    _equal(authorities["full24_checkpoint_1000"]["tree_sha256"], FULL24_1000, "FULL24@1000 authority")
    _equal(authorities["full24_checkpoint_2000"]["tree_sha256"], FULL24_2000, "FULL24@2000 authority")


def _validate_cohort(manifest: Mapping[str, Any]) -> None:
    cohort = manifest.get("full24_train24")
    _require(isinstance(cohort, Mapping), "full24_train24 is required")
    _equal(cohort.get("dataset_id"), "FULL24_EXISTING_TRAIN", "cohort dataset")
    _equal(cohort.get("split"), "TRAIN", "cohort split")
    _equal(cohort.get("tree_sha256"), FULL24_TREE, "FULL24 tree")
    _equal(cohort.get("split_lock_sha256"), FULL24_SPLIT, "FULL24 split lock")
    _equal(cohort.get("source_manifest_sha256"), "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566", "FULL24 source manifest")
    _equal(cohort.get("regular_file_count"), 85, "FULL24 file count")
    _equal(cohort.get("total_bytes"), 919661, "FULL24 byte count")
    _equal(cohort.get("window_count"), 24, "FULL24 window count")
    _equal(cohort.get("frame_count"), 960, "FULL24 frame count")
    _equal(cohort.get("window_order"), "target_major_then_snapshot_major", "FULL24 order")
    _equal(cohort.get("window_ids"), list(WINDOWS), "FULL24 window IDs")


def _validate_contract(manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("structured_action_contract")
    _require(isinstance(contract, Mapping), "structured_action_contract is required")
    expected = {
        "schema": "StructuredActionV1",
        "geometry": "K16",
        "timing": "T16",
        "horizon": "H40",
        "raw_shape": [1, 40, 132],
        "decoded_shape": [40, 10],
        "state_v1_shape": [10],
        "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
        "tcp_semantic_id": "x5_link6_gripper_center",
        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "target_xyz_in_model_input": False,
        "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
        "decoder_source_sha256": DECODER,
    }
    for key, value in expected.items():
        _equal(contract.get(key), value, f"decoder contract {key}")


def _validate_runtime_and_checkpoints(manifest: Mapping[str, Any]) -> None:
    base = manifest.get("base")
    _require(isinstance(base, Mapping), "base identity is required")
    _equal(base.get("tree_sha256"), BASE_TREE, "BASE tree")
    _equal(base.get("index_sha256"), "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59", "BASE index")
    _equal(base.get("processor_config_sha256"), "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be", "BASE processor config")
    _equal(base.get("statistics_sha256"), "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9", "BASE statistics")

    runtime = manifest.get("runtime_v2")
    _require(isinstance(runtime, Mapping), "runtime_v2 is required")
    _equal(runtime.get("stats_sha256"), RUNTIME_STATS, "runtime V2 stats")
    _equal(runtime.get("payload_sha256"), RUNTIME_PAYLOAD, "runtime V2 payload")
    _equal(runtime.get("protocol_revision"), "RUNTIME_PARQUET_PROVENANCE_V2", "runtime protocol")
    _equal(runtime.get("path"), None, "runtime path")
    _equal(runtime.get("locator"), "LATE_BOUND_FROM_GOLDEN_RELEASE_V2", "runtime locator")

    groot = manifest.get("groot_runtime")
    _require(isinstance(groot, Mapping), "groot_runtime is required")
    _equal(groot.get("groot_source_commit_sha"), GROOT_SOURCE, "GR00T source")
    _equal(groot.get("cosmos_revision"), COSMOS_REVISION, "Cosmos revision")
    _equal(groot.get("cosmos_materialized_tree_sha256"), COSMOS_TREE, "Cosmos tree")
    _equal(groot.get("modality_sha256"), MODALITY, "modality")
    _equal(groot.get("training_config_sha256"), CONFIG, "runtime config")
    _equal(groot.get("path"), None, "GR00T runtime path")
    _equal(groot.get("runtime_manifest", {}).get("sha256"), PROMOTION_SHA, "runtime manifest hash")
    _equal(groot.get("runtime_manifest", {}).get("status"), "ACCEPTED_READ_ONLY", "runtime manifest status")

    checkpoints = manifest.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "checkpoints is required")
    _equal(checkpoints.get("integrity_delta_audit_sha256"), INTEGRITY_SHA, "checkpoint audit")
    entries = checkpoints.get("entries")
    _require(isinstance(entries, list) and len(entries) == 4, "four checkpoint entries are required")
    _equal([entry.get("label") for entry in entries], ["BASE", "STRUCTURED@1000", "FULL24@1000", "FULL24@2000"], "checkpoint order")
    _equal(entries[0].get("tree_sha256"), BASE_TREE, "BASE checkpoint")
    _equal(entries[1].get("tree_sha256"), STRUCTURED_1000, "STRUCTURED@1000 checkpoint")
    _equal(entries[2].get("tree_sha256"), FULL24_1000, "FULL24@1000 checkpoint")
    _equal(entries[3].get("tree_sha256"), FULL24_2000, "FULL24@2000 checkpoint")


def _validate_historical(manifest: Mapping[str, Any]) -> None:
    historical = manifest.get("historical_comparator")
    _require(isinstance(historical, Mapping), "historical_comparator is required")
    _equal(historical.get("expected_tree_sha256"), STRUCTURED_1000, "historical tree")
    _equal(historical.get("expected_file_count"), 19, "historical file count")
    _equal(historical.get("expected_total_bytes"), 20889596357, "historical bytes")
    _equal(historical.get("expected_structured_action_stats_sha256"), STRUCTURED_STATS, "historical stats")
    _equal(historical.get("runtime_materialization_status"), "PENDING_VANDA_RECOVERY", "historical materialization")
    _equal(historical.get("materialization_path"), None, "historical materialization path")
    _equal(historical.get("substitution_allowed"), False, "historical substitution")


def _validate_sampler(manifest: Mapping[str, Any]) -> None:
    sampler = manifest.get("sampler_noise_protocol")
    _require(isinstance(sampler, Mapping), "sampler_noise_protocol is required")
    expected = {
        "development_seed": 4242,
        "noise_generation": "Golden_Release_runtime_RNG_seeded_from_development_seed",
        "seed_scope": "fixed_seed_4242_for_all_checkpoint_passes; exact_RNG_construction_late_bound_to_Golden_Release",
        "noise_shape": [1, 40, 132],
        "same_noise_across_checkpoints": True,
        "sampler": "source-mapped-four-step-Euler-action-head-graph",
        "official_public_sampler_called": False,
        "steps": 4,
        "dt": 0.25,
        "timestep_ids": [0, 250, 500, 750],
        "checkpoint_order": ["BASE", "STRUCTURED@1000", "FULL24@1000", "FULL24@2000"],
        "windows_per_checkpoint": 24,
        "planned_forwards_total": 96,
        "adaptive_extension": False,
        "path": None,
    }
    for key, value in expected.items():
        _equal(sampler.get(key), value, f"sampler {key}")


def _validate_metrics(manifest: Mapping[str, Any]) -> None:
    metrics = manifest.get("metric_protocol")
    _require(isinstance(metrics, Mapping), "metric_protocol is required")
    _equal(metrics.get("semantics_source"), "ACCEPTED_STAGE44R", "metric semantics source")
    _equal(metrics.get("trajectory_source_sha256"), METRICS, "trajectory metric source")
    _equal(metrics.get("comparison_source_sha256"), METRICS, "comparison metric source")
    _equal(metrics.get("progress_tolerance_m"), 1e-9, "progress tolerance")
    _equal(metrics.get("task_success_threshold"), None, "task threshold")
    _equal(metrics.get("safety_gate_allowed"), ["ACCEPT", "WARN"], "SafetyGate outcomes")
    _require(isinstance(metrics.get("target_metrics"), list) and "absolute_progress_m" in metrics["target_metrics"], "target progress metric missing")
    _require(isinstance(metrics.get("comparison_metrics"), list) and "endpoint_error_m" in metrics["comparison_metrics"], "endpoint metric missing")
    _equal(metrics.get("path"), None, "metric path")


def _validate_boundary(manifest: Mapping[str, Any]) -> None:
    locked = manifest.get("locked_inputs")
    _require(isinstance(locked, Mapping), "locked_inputs is required")
    for key in ("Validation-E", "Holdout-F", "WBC", "simulator", "hardware", "new_data"):
        _equal(locked.get(key), "LOCKED", f"locked input {key}")
    boundary = manifest.get("execution_boundary")
    _require(isinstance(boundary, Mapping), "execution_boundary is required")
    zeroes = ("gr00t_forwards", "pbs_submissions", "optimizer_steps", "wbc_steps", "simulator_steps")
    for key in zeroes:
        _equal(boundary.get(key), 0, f"execution zero {key}")
    falses = ("model_constructed", "model_forward", "gpu", "h200", "training", "checkpoint_recovery_or_transfer", "validation_e_read", "holdout_f_read", "hardware_execution")
    for key in falses:
        _equal(boundary.get(key), False, f"execution false {key}")
    integration = manifest.get("integration")
    _require(isinstance(integration, Mapping), "integration is required")
    _equal(integration.get("base_sha"), PREP_SHA, "integration base")
    _equal(integration.get("canonical_stage45c_branch_touched"), False, "canonical branch mutation")
    _equal(integration.get("merge_performed"), False, "merge state")
    _equal(integration.get("final_protocol_freeze_issued"), False, "final freeze state")

    evidence = manifest.get("output_evidence_schema")
    _require(isinstance(evidence, Mapping), "output_evidence_schema is required")
    _equal(evidence.get("per_window_schema"), "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1", "per-window schema")
    _equal(evidence.get("aggregate_schema"), "CORELAB_STAGE45C_TRAIN24_EVALUATION_AGGREGATE_V1", "aggregate schema")
    _equal(evidence.get("raw_predictions_preserved_separately"), True, "raw evidence retention")
    _equal(evidence.get("output_root"), None, "output root")


def validate_candidate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a parsed candidate and return a compact CPU-only summary."""

    _equal(manifest.get("schema"), SCHEMA, "schema")
    _equal(manifest.get("stage"), "Stage45C", "stage")
    _equal(manifest.get("pi_gate"), "PI_GATE_STATE_0089", "PI gate")
    _equal(manifest.get("status"), STATUS, "candidate status")
    _equal(manifest.get("branch"), "stage45c/corelab-train24-evaluator-prep", "support branch")
    _equal(manifest.get("source_prep_sha"), PREP_SHA, "prep SHA")
    _equal(manifest.get("source_golden_head"), GOLDEN_HEAD, "Golden HEAD")
    _equal(manifest.get("scientific_contract_changed"), False, "scientific contract")
    _equal(manifest.get("canonical_branch_untouched"), True, "canonical branch untouched")
    _validate_authorities(manifest)
    _validate_cohort(manifest)
    _validate_contract(manifest)
    _validate_runtime_and_checkpoints(manifest)
    _validate_historical(manifest)
    _validate_sampler(manifest)
    _validate_metrics(manifest)
    _validate_boundary(manifest)
    _reject_node_local_paths(manifest)
    _equal(manifest.get("output_evidence_schema", {}).get("output_root"), None, "output root")
    _equal(manifest.get("return_token"), "CORELAB_STAGE45C_LATE_BINDING_CANDIDATE_READY", "return token")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "branch": manifest["branch"],
        "cohort_window_count": 24,
        "checkpoint_count": 4,
        "planned_forwards": 96,
        "gr00t_forwards_observed": 0,
        "gpu": False,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc_steps": 0,
        "hardware_execution": False,
        "historical_materialization": "PENDING_VANDA_RECOVERY",
        "final_protocol_freeze_issued": False,
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"cannot read candidate JSON: {exc}") from exc
    _require(isinstance(document, Mapping), "candidate JSON root must be an object")
    summary = validate_candidate(document)
    summary["manifest_sha256"] = _sha256(path)
    summary["manifest"] = str(path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        summary = load_and_validate(args.manifest)
    except CandidateError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
