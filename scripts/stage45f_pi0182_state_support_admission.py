#!/usr/bin/env python3
"""PI0182 CPU-only admission for the MULTI_VISITED20 training package.

This audit consumes the already sealed PI0181/Stage43 support pack and the
existing frozen runtime manifests.  It does not read Validation-E or Holdout-F,
construct a model, import GR00T, use CUDA, run a simulator/WBC, or execute a
training step.  The only row transformation is a provenance overlay that
classifies q0 as INITIAL_SUPPORT and q1-q3 as AUTONOMOUS_VISITED.  The actual
Stage38 structured-training label boundary is exercised through
``StructuredStateSupportAdapterV1`` with the pre-registered statistics file.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from go2arx5_vla import stage38_structured_training as structured_training
from go2arx5_vla.stage38_structured_training import (
    STRUCTURED_TRAINING_FORBIDDEN_MODEL_INPUT_FIELDS,
    STRUCTURED_TRAINING_MODEL_INPUT_FIELDS,
    StructuredActionNormalizer,
    StructuredActionTargetLeakageError,
    StructuredStateSupportAdapterV1,
)
from go2arx5_vla.stage40_structured_decode import (
    EXPECTED_STRUCTURED_STATS_SHA256,
    load_structured_statistics,
)
from go2arx5_vla.structured_action import (
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    STRUCTURED_ACTIVE_DENSE_DIM,
    STRUCTURED_TIMING_MODE,
    structured_action_mask,
)


PI_GATE = 5665436865
PR_POINTER = 5665439699
ACCEPTED_HEAD = "df827767a144913d80bbba6f30fef4fa3ad3bb49"
PACKAGE_ID = "MULTI_VISITED20_V1"
PI0181_ROOT = REPO / "reports/stage45f_pi0181_visited_state_support_20260914"
DEFAULT_OUTPUT_ROOT = REPO / "reports/stage45f_pi0182_state_support_training_admission_20260914"

CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
Q_ORDER = (0, 1, 2, 3)
ORIGIN_BY_Q = {0: "INITIAL_SUPPORT", 1: "AUTONOMOUS_VISITED", 2: "AUTONOMOUS_VISITED", 3: "AUTONOMOUS_VISITED"}

FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_RUNTIME_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_RUNTIME_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_PROCESSOR_STATISTICS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
TRAINING_CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
DECODER_SOURCE_SHA256 = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"
TRAIN_SEED = 6164
UPDATE_COUNT = 256
VISITED_ROW_COUNT = 20
FULL24_SLOT_COUNT = 24


class PI0182Error(RuntimeError):
    """Fail-closed PI0182 admission error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0182Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


def array_sha(value: Any, *, dtype: Any | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"required JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_rows() -> list[dict[str, Any]]:
    rows_path = PI0181_ROOT / "visited_rows.jsonl"
    require(rows_path.is_file(), f"PI0181 rows file missing: {rows_path}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == VISITED_ROW_COUNT, f"expected {VISITED_ROW_COUNT} PI0181 rows, got {len(rows)}")
    expected_ids = [f"{condition}-q{q}" for condition in CONDITION_ORDER for q in Q_ORDER]
    require([row.get("row_id") for row in rows] == expected_ids, "PI0181 row order/identity drift")
    for row in rows:
        require(row.get("condition_id") in CONDITION_ORDER, f"unknown condition: {row.get('condition_id')}")
        q = int(row.get("q_index", -1))
        require(q in Q_ORDER, f"unknown q index: {q}")
        require(row.get("model_input_fields") == list(STRUCTURED_TRAINING_MODEL_INPUT_FIELDS), f"model input whitelist drift in {row['row_id']}")
        require(row.get("model_input_excludes"), f"missing exclusion list in {row['row_id']}")
    return rows


def classify_rows(rows: Sequence[Mapping[str, Any]], output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    by_origin = Counter()
    by_condition = Counter()
    for source in rows:
        row = dict(source)
        q = int(row["q_index"])
        origin = ORIGIN_BY_Q[q]
        row["origin_class"] = origin
        row["origin_rule"] = "q0=INITIAL_SUPPORT; q1-q3=AUTONOMOUS_VISITED"
        row["classification_source"] = "PI0182_DERIVED_IMMUTABLE_OVERLAY"
        classified.append(row)
        by_origin[origin] += 1
        by_condition[(str(row["condition_id"]), origin)] += 1
    require(by_origin == Counter({"INITIAL_SUPPORT": 5, "AUTONOMOUS_VISITED": 15}), f"origin classification drift: {by_origin}")
    output_path = output_root / "origin_classified_rows.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in classified:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return classified, {
        "path": str(output_path.relative_to(output_root)),
        "sha256": sha256_file(output_path),
        "row_count": len(classified),
        "origin_counts": dict(sorted(by_origin.items())),
        "condition_origin_counts": {
            f"{condition}:{origin}": by_condition[(condition, origin)]
            for condition in CONDITION_ORDER
            for origin in ("INITIAL_SUPPORT", "AUTONOMOUS_VISITED")
        },
        "source_rows_sha256": sha256_file(PI0181_ROOT / "visited_rows.jsonl"),
        "source_payloads_regenerated": False,
    }


def read_frozen_runtime_bindings() -> dict[str, Any]:
    mapping_path = REPO / "manifests/STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1.json"
    mapping = read_json(mapping_path)
    entries = {entry["label"]: entry for entry in mapping["checkpoint_runtime_mapping"]}
    base = entries["BASE"]
    structured = entries["STRUCTURED@1000"]
    for label, entry in entries.items():
        require(entry.get("processor_config_sha256") == PROCESSOR_CONFIG_SHA256, f"{label} processor config SHA drift")
        require(entry.get("processor_statistics_sha256") == BASE_PROCESSOR_STATISTICS_SHA256, f"{label} processor statistics SHA drift")
    require(base["decode_statistics"]["sha256"] == BASE_PROCESSOR_STATISTICS_SHA256, "BASE processor stats binding drift")
    require(structured["decode_statistics"]["sha256"] == EXPECTED_STRUCTURED_STATS_SHA256, "structured action stats binding drift")
    require(mapping["model_input_contract"]["target_xyz_in_model_input"] is False, "target input boundary drift")
    require(mapping["model_input_contract"]["state_v1_shape"] == [10], "StateV1 shape drift")
    pi0154 = read_json(REPO / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json")
    shared = pi0154["shared_runtime_identity"]
    require(shared["modality_sha256"] == MODALITY_SHA256, "modality SHA drift")
    require(shared["processor_config_sha256"] == PROCESSOR_CONFIG_SHA256, "PI0154 processor config SHA drift")
    require(shared["base_processor_statistics_sha256"] == BASE_PROCESSOR_STATISTICS_SHA256, "PI0154 processor stats SHA drift")
    require(shared["full24_runtime_stats_sha256"] == FULL24_RUNTIME_STATS_SHA256, "FULL24 runtime stats SHA drift")
    require(shared["full24_runtime_stats_payload_sha256"] == FULL24_RUNTIME_STATS_PAYLOAD_SHA256, "FULL24 runtime stats payload SHA drift")
    full24_design = read_json(REPO / "manifests/STAGE45A_FULL24_ADAPTATION_DESIGN_V1.json")
    require(full24_design["data_ladder"]["level_1"]["canonical_tree_sha256"] == FULL24_TREE_SHA256, "FULL24 tree binding drift")
    return {
        "base_processor_config_sha256": PROCESSOR_CONFIG_SHA256,
        "base_processor_statistics_sha256": BASE_PROCESSOR_STATISTICS_SHA256,
        "structured_action_statistics_sha256": EXPECTED_STRUCTURED_STATS_SHA256,
        "full24_runtime_statistics_sha256": FULL24_RUNTIME_STATS_SHA256,
        "full24_runtime_statistics_payload_sha256": FULL24_RUNTIME_STATS_PAYLOAD_SHA256,
        "modality_sha256": MODALITY_SHA256,
        "training_config_sha256": TRAINING_CONFIG_SHA256,
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "full24_tree_sha256": FULL24_TREE_SHA256,
        "source_mapping_sha256": sha256_file(mapping_path),
        "processor_recompute_allowed": False,
        "s20_statistics_recompute_allowed": False,
        "s20_statistics_source": "EXACT_STAGE38_STRUCTURED_ACTION_STATISTICS_ONLY",
    }


def exposure_report(rows: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    schedule_path = PI0181_ROOT / "s20_schedule.json"
    schedule = read_json(schedule_path)
    require(sha256_file(schedule_path) == "defd516cc2922d34ea5fcafef35b0c44288d8e78c4791c66cc51c7c7cd402e5e", "PI0181 S20 schedule SHA drift")
    require(schedule.get("seed") == TRAIN_SEED and schedule.get("update_count") == UPDATE_COUNT, "S20 schedule seed/update drift")
    require(len(schedule.get("E", [])) == UPDATE_COUNT and len(schedule.get("S20", [])) == UPDATE_COUNT, "S20 schedule length drift")
    require(all(slot["source"] == "FULL24" for slot in schedule["E"]), "E schedule is not FULL24-only")
    require(sum(slot["source"] == "FULL24" for slot in schedule["S20"]) == 128, "S20 FULL24 exposure drift")
    require(sum(slot["source"] == "VISITED20" for slot in schedule["S20"]) == 128, "S20 VISITED20 exposure drift")
    row_map = {str(row["row_id"]): row for row in rows}
    counts = Counter()
    row_counts = Counter()
    for slot in schedule["S20"]:
        if slot["source"] != "VISITED20":
            continue
        row_id = str(slot["visited_row_id"])
        require(row_id in row_map, f"S20 references unknown visited row: {row_id}")
        row = row_map[row_id]
        key = (str(row["condition_id"]), int(row["q_index"]), str(row["origin_class"]))
        counts[key] += 1
        row_counts[row_id] += 1
    require(set(row_counts) == set(row_map), "S20 does not expose every visited row")
    require(all(count in (6, 7) for count in row_counts.values()), f"S20 per-row exposure is not 6/7: {dict(row_counts)}")
    require(sum(row_counts.values()) == 128, "S20 exposure total drift")
    by_condition = defaultdict(int)
    by_origin = defaultdict(int)
    by_q = defaultdict(int)
    per_row = {}
    for row_id, count in sorted(row_counts.items()):
        row = row_map[row_id]
        per_row[row_id] = {
            "condition_id": row["condition_id"],
            "q_index": int(row["q_index"]),
            "origin_class": row["origin_class"],
            "exposure_count": int(count),
        }
        by_condition[str(row["condition_id"])] += count
        by_origin[str(row["origin_class"])] += count
        by_q[str(int(row["q_index"]))] += count
    report = {
        "schema": "CORELAB_PI0182_S20_EXPOSURE_COUNT_REPORT_V1",
        "status": "PASS",
        "schedule_sha256": sha256_file(schedule_path),
        "seed": TRAIN_SEED,
        "update_count": UPDATE_COUNT,
        "E": {"FULL24": 256, "VISITED20": 0},
        "S20": {"FULL24": 128, "VISITED20": 128},
        "per_condition_q_origin": {
            f"{condition}:q{q}:{ORIGIN_BY_Q[q]}": int(counts[(condition, q, ORIGIN_BY_Q[q])])
            for condition in CONDITION_ORDER
            for q in Q_ORDER
        },
        "by_condition": dict(sorted(by_condition.items())),
        "by_origin": dict(sorted(by_origin.items())),
        "by_q": dict(sorted(by_q.items())),
        "per_row": per_row,
        "s20_stats_recomputed": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "training_executed": False,
    }
    path = output_root / "s20_exposure_counts.json"
    write_json(path, report)
    report["sha256"] = sha256_file(path)
    return report


def find_stats_path(argument: Path | None) -> Path:
    candidates = []
    if argument is not None:
        candidates.append(Path(argument))
    candidates.append(Path("/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/experiment_cfg/structured_action_statistics.json"))
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            return path
    raise PI0182Error("exact Stage38 structured_action_statistics.json is unavailable; no substitute is allowed")


def target_leakage_and_adapter_audit(
    rows: Sequence[Mapping[str, Any]],
    stats_path: Path,
    runtime_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    lower, upper, mask, observed_stats_sha = load_structured_statistics(stats_path)
    require(observed_stats_sha == EXPECTED_STRUCTURED_STATS_SHA256, "structured stats SHA mismatch")
    frozen_metadata = {
        "schema": "PI0182_FROZEN_STAGE38_STRUCTURED_STATS_BINDING_V1",
        "selected_lower": lower.tolist(),
        "selected_upper": upper.tolist(),
        "active_mask": mask.tolist(),
        "timing_mode": STRUCTURED_TIMING_MODE,
        "action_horizon": STRUCTURED_ACTION_HORIZON,
        "universal_action_dim": STRUCTURED_ACTION_DIM,
        "active_scalar_count": int(np.count_nonzero(mask)),
        "source": "EXACT_STAGE38_STRUCTURED_ACTION_STATISTICS",
        "statistics_recomputed": False,
    }
    normalizer = StructuredActionNormalizer(frozen_metadata)
    adapter = StructuredStateSupportAdapterV1(
        normalizer=normalizer,
        processor_config_sha256=str(runtime_bindings["base_processor_config_sha256"]),
        processor_statistics_sha256=str(runtime_bindings["base_processor_statistics_sha256"]),
        structured_statistics_sha256=str(runtime_bindings["structured_action_statistics_sha256"]),
        modality_sha256=str(runtime_bindings["modality_sha256"]),
    )
    require(adapter.model_input_fields == STRUCTURED_TRAINING_MODEL_INPUT_FIELDS, "adapter whitelist drift")
    accepted = 0
    negative_cases: list[dict[str, Any]] = []
    output_keys: set[str] | None = None
    original_stats_function = structured_training._structured_stats

    def forbidden_stats_recompute(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("PI0182 S20 attempted to recompute structured statistics")

    # The guard makes the no-recompute claim executable against the actual
    # imported adapter module rather than relying only on a manifest statement.
    structured_training._structured_stats = forbidden_stats_recompute
    try:
        for row in rows:
            row_id = str(row["row_id"])
            visited_path = PI0181_ROOT / str(row["payload_files"]["visited_row_npz"])
            expert_path = PI0181_ROOT / str(row["payload_files"]["expert_label_npz"])
            with np.load(visited_path, allow_pickle=False) as visited, np.load(expert_path, allow_pickle=False) as expert:
                inputs = {
                    "body_rgb": np.asarray(visited["body_rgb"]),
                    "wrist_rgb": np.asarray(visited["wrist_rgb"]),
                    "state_v1": np.asarray(visited["state_v1"]),
                    "language": str(row["language"]),
                }
                expert_action = np.asarray(expert["expert_action_structured_v1"], dtype=np.float64)
                timestamps = np.asarray(expert["source_timestamps"], dtype=np.float64)
            datapoint = adapter.build_datapoint(
                inputs,
                expert_action=expert_action,
                source_timestamps=timestamps,
                chunk_id=row_id,
                observation_timestamp=float(row["observation_timestamp_s"]),
                provenance={"origin_class": row["origin_class"], "source_row_id": row_id},
            )
            expected_keys = {"body_rgb", "wrist_rgb", "state", "language", "action", "action_mask"}
            require(set(datapoint) == expected_keys, f"adapter output key drift in {row_id}")
            require(np.asarray(datapoint["action"]).shape == (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM), f"adapter action shape drift in {row_id}")
            require(np.asarray(datapoint["action_mask"]).shape == (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM), f"adapter mask shape drift in {row_id}")
            require(np.array_equal(np.asarray(datapoint["action_mask"]), mask.astype(np.float32)), f"adapter mask drift in {row_id}")
            require(np.isfinite(np.asarray(datapoint["action"])).all(), f"adapter action is non-finite in {row_id}")
            require(not np.any(np.asarray(datapoint["action"])[mask == 0.0]), f"inactive action values are nonzero in {row_id}")
            output_keys = set(datapoint)
            accepted += 1

            for forbidden_name, forbidden_value in (
                ("target_world_m", np.asarray(row["target_world_m"], dtype=np.float64)),
                ("historical_decoded", np.zeros((STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTIVE_DENSE_DIM), dtype=np.float64)),
            ):
                poisoned = dict(inputs)
                poisoned[forbidden_name] = forbidden_value
                try:
                    adapter.build_datapoint(
                        poisoned,
                        expert_action=expert_action,
                        source_timestamps=timestamps,
                        chunk_id=f"{row_id}:{forbidden_name}",
                    )
                except StructuredActionTargetLeakageError as exc:
                    negative_cases.append({"row_id": row_id, "field": forbidden_name, "status": "PASS", "error": str(exc)})
                else:
                    raise PI0182Error(f"target leakage negative test unexpectedly accepted {forbidden_name} in {row_id}")
    finally:
        structured_training._structured_stats = original_stats_function
    require(accepted == VISITED_ROW_COUNT, f"adapter accepted {accepted}/{VISITED_ROW_COUNT} rows")
    require(len(negative_cases) == VISITED_ROW_COUNT * 2, "target leakage negative test count drift")
    source_text = (REPO / "src/go2arx5_vla/stage38_structured_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    helper_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == "StructuredStateSupportAdapterV1"
    }
    require(helper_names == {"StructuredStateSupportAdapterV1"}, "actual structured adapter source not present")
    adapter_source_sha = sha256_file(REPO / "src/go2arx5_vla/stage38_structured_training.py")
    codec_source_sha = sha256_file(REPO / "src/go2arx5_vla/structured_action.py")
    result = {
        "schema": "CORELAB_PI0182_STRUCTURED_TRAINING_ADAPTER_AUDIT_V1",
        "status": "PASS",
        "adapter_path": "src/go2arx5_vla/stage38_structured_training.py",
        "adapter_class": "StructuredStateSupportAdapterV1",
        "adapter_source_sha256": adapter_source_sha,
        "codec_source_sha256": codec_source_sha,
        "model_input_fields": list(adapter.model_input_fields),
        "forbidden_model_input_fields": sorted(STRUCTURED_TRAINING_FORBIDDEN_MODEL_INPUT_FIELDS),
        "adapter_output_keys": sorted(output_keys or set()),
        "rows_admitted": accepted,
        "stats_path": str(stats_path),
        "stats_sha256": observed_stats_sha,
        "action_mask_sha256_float32": array_sha(mask, dtype=np.float32),
        "active_scalar_count": int(np.count_nonzero(mask)),
        "s20_statistics_recomputed": False,
        "statistics_guard": "_structured_stats replaced with fail-closed sentinel during all 20 adapter admissions",
        "negative_test": {
            "status": "PASS",
            "case_count": len(negative_cases),
            "cases": negative_cases,
        },
        "official_launch_binding": {
            "factory_path": "src/go2arx5_vla/stage38_structured_training.py:install_structured_dataset_factory",
            "frozen_stats_env": "STAGE45F_FROZEN_STRUCTURED_STATS_PATH + STAGE45F_FROZEN_STRUCTURED_STATS_SHA256",
            "model_constructed": False,
            "processor_invoked": False,
            "late_bound_remaining": ["PI-selected Legacy/L1 base checkpoint", "H200 allocation/start"],
        },
    }
    return result


def build_package(
    output_root: Path,
    rows_overlay: Mapping[str, Any],
    exposure: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any],
    adapter_audit: Mapping[str, Any],
) -> dict[str, Any]:
    package = {
        "schema": "CORELAB_PI0182_STATE_SUPPORT_TRAINING_ADMISSION_PACKAGE_V1",
        "status": "READY",
        "package_id": PACKAGE_ID,
        "authority": {
            "issue_11_comment": PI_GATE,
            "pr_75_pointer": PR_POINTER,
            "accepted_head": ACCEPTED_HEAD,
        },
        "base_checkpoint": {
            "selection": "PI_LATE_BOUND",
            "allowed_values": ["Legacy", "L1"],
            "scientific_selection_executed": False,
        },
        "arms": {
            "E": {"updates": 256, "FULL24": 256, "STATE_SUPPORT20": 0},
            "S20": {"updates": 256, "FULL24": 128, "STATE_SUPPORT20": 128},
        },
        "schedule": {
            "seed": TRAIN_SEED,
            "schedule_sha256": exposure["schedule_sha256"],
            "exposure_report_sha256": exposure["sha256"],
            "pairing": "E and S20 share every retained FULL24 position; S20 substitutes deterministic VISITED20 on odd positions",
            "per_condition_q_origin": exposure["per_condition_q_origin"],
        },
        "row_origin_overlay": rows_overlay,
        "input_contract": {
            "model_input_fields": list(STRUCTURED_TRAINING_MODEL_INPUT_FIELDS),
            "target_xyz_in_model_input": False,
            "historical_policy_output_in_model_input": False,
            "expert_action_role": "supervision_only",
            "target_and_telemetry_role": "metadata_only",
        },
        "frozen_identity": dict(runtime_bindings),
        "structured_training_adapter": {
            "path": adapter_audit["adapter_path"],
            "class": adapter_audit["adapter_class"],
            "source_sha256": adapter_audit["adapter_source_sha256"],
            "codec_source_sha256": adapter_audit["codec_source_sha256"],
            "admission_status": adapter_audit["status"],
            "negative_target_leakage_test": adapter_audit["negative_test"]["status"],
            "s20_stats_recomputed": False,
            "official_factory_frozen_stats_switch": adapter_audit["official_launch_binding"]["frozen_stats_env"],
        },
        "runspec": {
            "optimizer": "AdamW",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "schedule": "cosine",
            "warmup_updates": 10,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "precision": "bf16",
            "trainable_modules": ["projector", "diffusion"],
            "frozen_modules": ["visual", "llm", "vlln"],
            "loss": "masked StructuredActionV1 flow matching",
            "timing": "T16",
            "horizon": 40,
            "action_dim": 132,
            "rng_seed": TRAIN_SEED,
            "target_excluded_from_model_input": True,
        },
        "late_bound_remaining": ["PI-selected Legacy/L1 base checkpoint", "H200 allocation and launch"],
        "training_executed": False,
        "model_constructed": False,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "gpu_execution": 0,
        "h200_submissions": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "simulator_steps": 0,
        "wbc_steps": 0,
    }
    path = output_root / "training_package" / "MULTI_VISITED20_V1_ADMISSION.json"
    write_json(path, package)
    package["package_sha256"] = sha256_file(path)
    package["path"] = str(path.relative_to(output_root))
    return package


def run(output_root: Path, stats_path: Path | None) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    require(not output_root.exists() or not any(output_root.iterdir()), f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    require(ACCEPTED_HEAD == "df827767a144913d80bbba6f30fef4fa3ad3bb49", "PI0182 accepted HEAD constant drift")
    rows = read_rows()
    classified, overlay = classify_rows(rows, output_root)
    runtime_bindings = read_frozen_runtime_bindings()
    exposure = exposure_report(classified, output_root)
    exact_stats_path = find_stats_path(stats_path)
    adapter_audit = target_leakage_and_adapter_audit(classified, exact_stats_path, runtime_bindings)
    package = build_package(output_root, overlay, exposure, runtime_bindings, adapter_audit)
    binding_path = output_root / "stats_processor_binding.json"
    write_json(binding_path, {
        "schema": "CORELAB_PI0182_STATS_PROCESSOR_BINDING_V1",
        "status": "PASS",
        "runtime_bindings": runtime_bindings,
        "adapter_audit": adapter_audit,
        "stats_path": str(exact_stats_path),
        "stats_sha256": sha256_file(exact_stats_path),
        "s20_statistics_recomputed": False,
    })
    leakage_path = output_root / "target_leakage_negative_test.json"
    write_json(leakage_path, adapter_audit["negative_test"])
    result = {
        "schema": "CORELAB_PI0182_STATE_SUPPORT_TRAINING_ADMISSION_RESULT_V1",
        "token": "CORELAB_PI0182_STATE_SUPPORT_TRAINING_ADMISSION_PACKAGE — READY",
        "status": "READY",
        "authority": {"issue_11_comment": PI_GATE, "pr_75_pointer": PR_POINTER, "accepted_head": ACCEPTED_HEAD},
        "origin_classification": overlay,
        "adapter_audit": {
            "status": adapter_audit["status"],
            "rows_admitted": adapter_audit["rows_admitted"],
            "adapter_source_sha256": adapter_audit["adapter_source_sha256"],
            "negative_test": adapter_audit["negative_test"],
            "s20_statistics_recomputed": False,
        },
        "frozen_identity": runtime_bindings,
        "exposure_report": {"path": "s20_exposure_counts.json", "sha256": exposure["sha256"], "by_condition": exposure["by_condition"], "by_origin": exposure["by_origin"], "by_q": exposure["by_q"]},
        "package": {"path": package["path"], "sha256": package["package_sha256"], "package_id": PACKAGE_ID},
        "late_bound_remaining": ["PI-selected Legacy/L1 base checkpoint", "H200 allocation and launch"],
        "hard_zeroes": {
            "validation_e_read": 0,
            "holdout_f_read": 0,
            "new_model_calls": 0,
            "model_constructed": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "gpu_execution": 0,
            "h200_submissions": 0,
            "simulator_steps": 0,
            "wbc_steps": 0,
            "hardware_steps": 0,
        },
    }
    write_json(output_root / "pi0182_result.json", result)
    write_json(output_root / "README.md", {
        "summary": "CPU/static PI0182 admission for MULTI_VISITED20_V1; no training or runtime execution occurred.",
        "origin_rule": "q0=INITIAL_SUPPORT; q1-q3=AUTONOMOUS_VISITED",
        "model_inputs": list(STRUCTURED_TRAINING_MODEL_INPUT_FIELDS),
        "s20_stats_recomputed": False,
        "remaining": result["late_bound_remaining"],
        "hard_zeroes": result["hard_zeroes"],
        "artifacts": ["pi0182_result.json", "origin_classified_rows.jsonl", "s20_exposure_counts.json", "stats_processor_binding.json", "target_leakage_negative_test.json", "training_package/MULTI_VISITED20_V1_ADMISSION.json"],
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--structured-stats", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args.output_root, args.structured_stats)
    except (PI0182Error, OSError, ValueError, KeyError, json.JSONDecodeError, AssertionError) as exc:
        print(json.dumps({"schema": "CORELAB_PI0182_STATE_SUPPORT_TRAINING_ADMISSION_RESULT_V1", "status": "BLOCKED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
