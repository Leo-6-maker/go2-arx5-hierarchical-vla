#!/usr/bin/env python3
"""Evaluate sealed Stage45E N2 checkpoints against the accepted FULL24 baseline.

This is the post-training Corelab evaluator authorized by PI0122.  It is
deliberately a small runtime adapter: the legacy FULL24@1000/@2000 outputs are
read from the sealed raw-evidence mirror and are never re-inferred, while the
two N2 checkpoints are sampled on exactly the same TRAIN24 windows and fixed
noise bank.  Raw arrays are written and hashed before any decode, and all
derived metrics use the accepted Stage44R/Stage45C CPU metric functions.

The default/static path does not import Torch or GR00T.  GPU execution is an
explicit, path-bound mode and has no training, backward, optimizer, scheduler,
checkpoint-write, WBC, simulator, Validation-E, or Holdout-F path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import stage45e_n2_statistics as n2  # noqa: E402


STAGE = "Stage45E"
PI_GATE = "PI_GATE_STATE_0122"
BRANCH = "stage45e/n2-balanced-xyz-normalization"
RETURN_TOKEN = "STAGE45E_N2_CORELAB_4080_TRAIN24_COMPLETE_FOR_PI_REVIEW"
EVAL_TOKEN = "STAGE45E_N2_CORELAB_TRAIN24_EVALUATION_COMPLETE_FOR_PI_REVIEW"
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
EXPECTED_FULL24_1000_TREE = "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c"
EXPECTED_FULL24_2000_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
EXPECTED_RUNTIME_STATS_SHA = n2.SOURCE_STATS_SHA256
EXPECTED_RUNTIME_STATS_PAYLOAD_SHA = n2.SOURCE_PAYLOAD_SHA256
EXPECTED_N2_HALF_RANGE_M = n2.HALF_RANGE_M
HORIZON = 40
UNIVERSAL_DIM = 132
PHYSICAL_DIM = 10
STATE_DIM = 10
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
PHYSICAL_SHAPE = (HORIZON, PHYSICAL_DIM)
TIMING_MODE = "T16"
TIMESTEP_IDS = (0, 250, 500, 750)
SAMPLER_STEPS = 4
SAMPLER_DT = 0.25
DEVELOPMENT_SEED = 4242
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
TARGET_ORDER = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TARGET_ORDER
    for index in range(6)
)
CHECKPOINT_ORDER = ("FULL24@1000", "FULL24@2000", "N2@1000", "N2@2000")
N2_CHECKPOINT_ORDER = ("N2@1000", "N2@2000")
LEGACY_CHECKPOINT_ORDER = ("FULL24@1000", "FULL24@2000")
LEGACY_TREE = {
    "FULL24@1000": EXPECTED_FULL24_1000_TREE,
    "FULL24@2000": EXPECTED_FULL24_2000_TREE,
}


_EXECUTION_BOUNDARY: dict[str, Any] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_boundary() -> None:
    global _EXECUTION_BOUNDARY
    _EXECUTION_BOUNDARY = {
        "gpu": False,
        "model_construction": 0,
        "model_forward_windows": 0,
        "action_head_dit_forward_calls": 0,
        "conditioning_passes": 0,
        "training": False,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "pbs_submissions": 0,
        "wbc_steps": 0,
        "simulator_steps": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "hardware": False,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _equal(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    _require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode
    _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"tree root is unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            _require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            _require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree file: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def _gpu_identity(expected_uuid: str) -> dict[str, Any]:
    import subprocess

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    _require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-1000:]}")
    parsed: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        _equal(len(fields), 5, "nvidia-smi field count")
        parsed.append({
            "index": fields[0],
            "name": fields[1],
            "uuid": fields[2],
            "memory_total_mib": int(fields[3]),
            "memory_used_mib_at_start": int(fields[4]),
        })
    selected = [item for item in parsed if item["uuid"] == expected_uuid]
    _equal(len(selected), 1, "selected GPU UUID cardinality")
    gpu = selected[0]
    _require("RTX 4080 SUPER" in gpu["name"], f"selected GPU is not RTX 4080S: {gpu}")
    _require(gpu["memory_total_mib"] >= 30000, f"selected GPU is not the preferred 32GB card: {gpu}")
    gpu["host"] = subprocess.check_output(["hostname"], text=True).strip()
    gpu["query_sha256"] = sha256_bytes(query.stdout.encode("utf-8"))
    return gpu


def _prepare_environment(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any]) -> None:
    cache_root = output_root / "cache"
    tmp_root = output_root / "tmp"
    cache_root.mkdir(parents=True, exist_ok=False)
    tmp_root.mkdir(parents=True, exist_ok=False)
    hub_root = cache_root / "huggingface" / "hub"
    hub_root.mkdir(parents=True, exist_ok=False)
    xdg_root = cache_root / "xdg"
    xdg_root.mkdir(parents=True, exist_ok=False)
    (hub_root / "models--nvidia--Cosmos-Reason2-2B").symlink_to(paths["cosmos"], target_is_directory=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
        "TMPDIR": str(tmp_root),
        "TMP": str(tmp_root),
        "TEMP": str(tmp_root),
        "XDG_CACHE_HOME": str(xdg_root),
        "HF_HOME": str(cache_root / "huggingface"),
        "TRANSFORMERS_CACHE": str(hub_root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "STAGE45E_N2_STATS_ARTIFACT": str(paths["n2_stats"]),
    })


def _verify_external(paths: Mapping[str, Path], training_root: Path) -> dict[str, Any]:
    training_result = read_json(training_root / "stage45e_n2_training_result.json")
    _equal(training_result.get("status"), "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW", "training status")
    _equal(training_result.get("starting_checkpoint"), "BASE", "starting checkpoint")
    _equal(training_result.get("base_tree_sha256"), EXPECTED_BASE_TREE, "training BASE identity")
    _equal(training_result.get("full24_tree_sha256"), EXPECTED_FULL24_TREE, "training FULL24 identity")
    _equal(training_result.get("seed"), 2222, "training seed")
    _equal(training_result.get("max_steps"), 2000, "training step budget")
    _equal(training_result.get("save_steps"), [1000, 2000], "training checkpoint cadence")
    for step in (1000, 2000):
        checkpoint = training_root / f"checkpoint-{step}"
        _require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"N2 checkpoint is unavailable: {checkpoint}")
        state = read_json(checkpoint / "trainer_state.json")
        _equal(state.get("global_step"), step, f"N2 checkpoint global step {step}")
        _require(all((checkpoint / name).is_file() for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")), f"N2 checkpoint state incomplete at {step}")
    n2_doc = read_json(paths["n2_stats"])
    verified_n2 = n2.verify_n2_document(n2_doc, source=read_json(paths["runtime_stats"]))
    _equal(float(n2_doc["intervention"]["common_half_range_m"]), EXPECTED_N2_HALF_RANGE_M, "N2 half-range")
    sidecar = read_json(training_root / "experiment_cfg" / "structured_action_statistics.json")
    _equal(sidecar.get("statistics"), n2_doc.get("statistics"), "N2 training sidecar statistics")
    _equal(sha256_file(paths["runtime_stats"]), EXPECTED_RUNTIME_STATS_SHA, "runtime-V2 statistics SHA")
    _equal(sha256_file(paths["modality"]), EXPECTED_MODALITY_SHA, "modality SHA")
    _equal(sha256_file(paths["config"]), EXPECTED_CONFIG_SHA, "training config SHA")
    _equal(_git(paths["groot"], "rev-parse", "HEAD"), EXPECTED_GROOT_COMMIT, "GR00T source commit")
    cosmos_refs = paths["cosmos"] / "refs" / "main"
    _equal(cosmos_refs.read_text(encoding="utf-8").strip(), EXPECTED_COSMOS_REVISION, "Cosmos revision")
    dataset_identity = n2.validate_dataset_identity(paths["dataset"])
    _equal(dataset_identity["tree_sha256"], EXPECTED_FULL24_TREE, "FULL24 dataset tree")
    return {
        "training_result_sha256": sha256_file(training_root / "stage45e_n2_training_result.json"),
        "checkpoint_1000_tree_sha256": tree_digest(tree_entries(training_root / "checkpoint-1000", "stage45e/checkpoint-1000")),
        "checkpoint_2000_tree_sha256": tree_digest(tree_entries(training_root / "checkpoint-2000", "stage45e/checkpoint-2000")),
        "n2_stats_sha256": sha256_file(paths["n2_stats"]),
        "n2_stats_payload_sha256": n2_doc.get("statistics_payload_sha256"),
        "n2_verification": verified_n2,
        "runtime_stats_sha256": sha256_file(paths["runtime_stats"]),
        "runtime_stats_payload_sha256": n2_doc.get("source_runtime_v2", {}).get("payload_sha256"),
        "modality_sha256": sha256_file(paths["modality"]),
        "config_sha256": sha256_file(paths["config"]),
        "groot_commit": _git(paths["groot"], "rev-parse", "HEAD"),
        "cosmos_revision": cosmos_refs.read_text(encoding="utf-8").strip(),
        "full24": dataset_identity,
    }


def _verify_sealed_baseline(root: Path, snapshots_path: Path) -> dict[str, Any]:
    preflight = read_json(root / "stage45c_runtime_preflight.json")
    snapshots = preflight.get("snapshots")
    _require(isinstance(snapshots, list) and len(snapshots) == 24, "sealed baseline must contain 24 snapshots")
    observed_ids = [str(item.get("window_id")) for item in snapshots]
    _equal(observed_ids, list(WINDOW_IDS), "sealed TRAIN24 window order")
    _equal([str(item.get("target_xyz_in_model_input")) for item in snapshots], ["False"] * 24, "target exclusion witness")
    _require(all(item.get("target_excluded_from_model_input") is True for item in snapshots), "target exclusion is not sealed")
    if snapshots_path != root / "stage45c_runtime_preflight.json":
        _equal(sha256_file(snapshots_path), sha256_file(root / "stage45c_runtime_preflight.json"), "snapshot preflight identity")
    raw_root = root / "raw"
    aggregates: dict[str, Any] = {}
    for label in LEGACY_CHECKPOINT_ORDER:
        directory = raw_root / label.replace("@", "_at_")
        _require(directory.is_dir(), f"sealed raw directory missing: {directory}")
        files = sorted(directory.glob("*.npy"))
        _equal([item.name for item in files], [f"{index:02d}.npy" for index in range(24)], f"sealed raw file order {label}")
        aggregate_path = root / f"aggregate_{label.replace('@', '_at_')}.json"
        aggregate = read_json(aggregate_path)
        _equal(aggregate.get("window_count"), 24, f"sealed aggregate row count {label}")
        _equal(aggregate.get("window_ids"), list(WINDOW_IDS), f"sealed aggregate window order {label}")
        _equal(aggregate.get("checkpoint", {}).get("label"), label, f"sealed aggregate checkpoint label {label}")
        _equal(aggregate.get("checkpoint", {}).get("sha256"), LEGACY_TREE[label], f"sealed aggregate checkpoint tree {label}")
        aggregates[label] = aggregate
    noise_manifest = read_json(root / "sampler_noise_manifest.json")
    _equal(noise_manifest.get("development_seed"), DEVELOPMENT_SEED, "sealed noise seed")
    _equal(noise_manifest.get("window_ids"), list(WINDOW_IDS), "sealed noise order")
    _equal(noise_manifest.get("noise_shape"), list(RAW_SHAPE), "sealed noise shape")
    _equal(noise_manifest.get("same_noise_across_checkpoints"), True, "sealed same-noise contract")
    raw_manifest = root / "raw_predictions.jsonl"
    _require(raw_manifest.is_file(), "sealed raw prediction manifest is missing")
    return {
        "root": str(root),
        "preflight_sha256": sha256_file(root / "stage45c_runtime_preflight.json"),
        "snapshots": snapshots,
        "aggregates": aggregates,
        "noise_manifest": noise_manifest,
        "noise_manifest_sha256": sha256_file(root / "sampler_noise_manifest.json"),
        "raw_predictions_manifest_sha256": sha256_file(raw_manifest),
        "aggregate_sha256": {label: sha256_file(root / f"aggregate_{label.replace('@', '_at_')}.json") for label in LEGACY_CHECKPOINT_ORDER},
    }


def _decode_with_stats(raw: Any, state: Any, stats_document: Mapping[str, Any], *, stats_sha256: str, payload_sha256: str, chunk_id: str, timestamps: Any, observation_timestamp: float) -> tuple[Any, dict[str, Any]]:
    """The accepted Stage40 inverse-normalization/materialization law."""

    import numpy as np
    from go2arx5_vla.structured_action import decode_structured_action, materialize_structured_action, structured_action_contract_summary, structured_action_mask
    from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows

    statistics = stats_document.get("statistics", stats_document)
    lower = np.asarray(stats_document.get("selected_lower", statistics.get("selected_lower")), dtype=np.float64)
    upper = np.asarray(stats_document.get("selected_upper", statistics.get("selected_upper")), dtype=np.float64)
    mask = np.asarray(stats_document.get("active_mask", statistics.get("active_mask")), dtype=np.float64)
    expected_mask = structured_action_mask(TIMING_MODE)
    _require(lower.shape == (UNIVERSAL_DIM,) and upper.shape == (UNIVERSAL_DIM,), "stats bounds must be [132]")
    _require(mask.shape == expected_mask.shape and np.array_equal(mask, expected_mask), "stats mask is not frozen T16")
    raw_np = np.asarray(raw, dtype=np.float64)
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    timestamps_np = np.asarray(timestamps, dtype=np.float64)
    _require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), "raw prediction shape/finite contract")
    _require(state_np.shape == (STATE_DIM,) and np.isfinite(state_np).all(), "state shape/finite contract")
    _require(timestamps_np.shape == (HORIZON,) and np.isfinite(timestamps_np).all(), "timestamp shape/finite contract")
    normalized = np.clip(raw_np[0], -1.0, 1.0)
    values = (normalized + 1.0) * 0.5 * (upper[None, :] - lower[None, :]) + lower[None, :]
    values[mask == 0.0] = 0.0
    _require(np.isfinite(values).all(), "inverse-normalized values are nonfinite")
    intent = decode_structured_action(
        values,
        anchor_state_v1=state_np.astype(np.float64),
        source_timestamps=timestamps_np,
        source_duration_s=1.9500000000000002,
        timing_mode=TIMING_MODE,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        observation_timestamp=float(observation_timestamp),
        intent_id=chunk_id,
        provenance={
            "stage45e_decoder_bridge": "Stage40_inverse_normalization_and_materialization",
            "stats_route": "N2_STRUCTURED_STATS" if stats_sha256 != EXPECTED_RUNTIME_STATS_SHA else "FULL24_RUNTIME_V2_STATS",
            "structured_stats_sha256": stats_sha256,
            "statistics_payload_sha256": payload_sha256,
            "no_dense_decode_fallback": True,
        },
    )
    chunk = materialize_structured_action(values, intent, source_timestamps=timestamps_np)
    rows = rotation_matrix_to_rot6d_rows(quat_to_matrix(np.asarray(chunk.eef_orientation, dtype=np.float64)))
    physical = np.concatenate([np.asarray(chunk.eef_position, dtype=np.float64), rows, np.asarray(chunk.gripper, dtype=np.float64).reshape(HORIZON, 1)], axis=1)
    _require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), "decoded physical shape/finite contract")
    return physical, {
        "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
        "decoder_bridge": "Stage40_inverse_normalization_and_materialization",
        "stats_route": "N2_STRUCTURED_STATS" if stats_sha256 != EXPECTED_RUNTIME_STATS_SHA else "FULL24_RUNTIME_V2_STATS",
        "stats_sha256": stats_sha256,
        "statistics_payload_sha256": payload_sha256,
        "raw_shape": list(RAW_SHAPE),
        "physical_shape": list(PHYSICAL_SHAPE),
        "raw_array_sha256": sha256_bytes(np.asarray(raw, dtype=np.float32).tobytes()),
        "contract": structured_action_contract_summary(TIMING_MODE),
    }


def _normalized_expert(expert: Any, state: Any, timestamps: Any, stats_document: Mapping[str, Any], window_id: str) -> Any:
    import numpy as np
    from go2arx5_vla.stage38_structured_training import StructuredActionNormalizer
    from go2arx5_vla.structured_action import encode_structured_action

    label = encode_structured_action(
        np.asarray(expert, dtype=np.float64),
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(state, dtype=np.float64),
        timing_mode=TIMING_MODE,
        chunk_id=window_id,
        source_frame=SOURCE_FRAME,
        tcp_semantic_id=TCP_SEMANTIC_ID,
        observation_timestamp=float(np.asarray(timestamps, dtype=np.float64)[0]),
    )
    statistics = stats_document.get("statistics", stats_document)
    return StructuredActionNormalizer(statistics, clip_outliers=True).normalize(label)


def _normalized_metrics(raw: Any, expert: Any, state: Any, timestamps: Any, stats_document: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.structured_action import structured_action_mask

    raw_np = np.asarray(raw, dtype=np.float32)
    expert_norm = _normalized_expert(expert, state, timestamps, stats_document, "normalized-metric")
    mask = structured_action_mask(TIMING_MODE).astype(bool)
    values = raw_np[0].astype(np.float64)
    expected = np.asarray(expert_norm, dtype=np.float64)
    active_values = values[mask]
    active_expected = expected[mask]
    delta = active_values - active_expected
    _require(np.isfinite(delta).all(), "normalized metric delta is nonfinite")
    return {
        "active_scalar_count": int(mask.sum()),
        "active_rmse": float(np.sqrt(np.mean(delta * delta))),
        "active_mae": float(np.mean(np.abs(delta))),
        "active_max_abs": float(np.max(np.abs(delta))),
        "raw_active_min": float(np.min(active_values)),
        "raw_active_max": float(np.max(active_values)),
        "raw_active_outside_neg1_pos1_count": int(np.count_nonzero((active_values < -1.0) | (active_values > 1.0))),
        "raw_active_outside_neg1_pos1_fraction": float(np.mean((active_values < -1.0) | (active_values > 1.0))),
        "expert_active_clipped_count": int(np.count_nonzero(np.abs(active_expected) >= 1.0 - 1.0e-7)),
    }


def _evaluation_row(raw: Any, physical: Any, details: Mapping[str, Any], normalized: Mapping[str, Any], snapshot: Mapping[str, Any], checkpoint: Mapping[str, Any], raw_path: Path, raw_file_sha: str, runtime_label: str) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import array_sha256, comparison_metrics

    raw_np = np.asarray(raw, dtype=np.float32)
    physical_np = np.asarray(physical, dtype=np.float64)
    expert = np.asarray(snapshot["expert_physical_action"], dtype=np.float64)
    metrics = comparison_metrics(
        physical_np,
        expert,
        snapshot["anchor_world_m"],
        snapshot["target_world_m"],
        prefix=f"{snapshot['window_id']}.prediction",
    )
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "window_id": snapshot["window_id"],
        "target_id": snapshot["target_id"],
        "raw_prediction_artifact": {
            "path": str(raw_path),
            "sha256": raw_file_sha,
            "array_sha256": array_sha256(raw_np),
            "shape": list(raw_np.shape),
        },
        "checkpoint": {"label": checkpoint["label"], "sha256": checkpoint["tree_sha256"]},
        "model": {"label": checkpoint["label"], "sha256": checkpoint["tree_sha256"]},
        "model_runtime_identity": {"sha256": runtime_label, "label": "Golden_Release_V2_GR00T_runtime"},
        "decoded_action": {
            "array_sha256": array_sha256(physical_np),
            "shape": list(physical_np.shape),
            "decoder_details": dict(details),
        },
        "normalized_metrics": dict(normalized),
        "metrics": metrics,
        "safety_gate_decision": None,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "target_xyz_in_model_input": False,
        "finite": True,
    }


def _snapshot_lookup(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _equal([str(item.get("window_id")) for item in records], list(WINDOW_IDS), "snapshot window order")
    normalized: list[dict[str, Any]] = []
    for item in records:
        for key in ("state_v1", "expert_physical_action", "anchor_world_m", "target_world_m", "source_timestamps"):
            _require(key in item, f"sealed snapshot field missing: {key}")
        normalized.append(dict(item))
    return normalized


def _load_runtime_processor(paths: Mapping[str, Path]) -> tuple[Any, Any, Mapping[str, Any]]:
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor

    return _build_runtime_processor(
        groot_source=paths["groot"],
        base_checkpoint=paths["base_model"],
        dataset_dir=paths["dataset"],
        modality_path=paths["modality"],
        cosmos_snapshot=paths["cosmos"] / "snapshots" / EXPECTED_COSMOS_REVISION,
    )


def _make_windows(processor: Any, embodiment: Any, modality_configs: Mapping[str, Any], dataset_dir: Path, snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from stage45c_h200_eval_runtime_adapter import _prepare_train24_windows

    return _prepare_train24_windows(processor, embodiment, modality_configs, dataset_dir, snapshots)


def _make_noise_bank(torch: Any, sealed_noise: Mapping[str, Any]) -> list[Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DEVELOPMENT_SEED)
    bank: list[Any] = []
    observed_hashes = list(sealed_noise.get("noise_sha256_by_window", []))
    _equal(len(observed_hashes), 24, "sealed noise hash count")
    for index in range(24):
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
        digest = sha256_bytes(noise.numpy().tobytes())
        _equal(digest, observed_hashes[index], f"fixed noise identity {index}")
        bank.append(noise)
    return bank


def _checkpoint_record(path: Path, label: str) -> dict[str, Any]:
    prefix = f"stage45e/{path.name}"
    tree = tree_digest(tree_entries(path, prefix))
    return {"label": label, "tree_sha256": tree, "path": str(path)}


def _aggregate_core(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema", "window_count", "window_ids", "checkpoint", "model",
        "raw_prediction_array_sha256", "decoded_action_array_sha256",
        "target_metrics", "comparison_metrics", "safety_gate_decision_counts",
        "raw_predictions_preserved_separately", "threshold_used", "finite",
    )
    return {key: aggregate.get(key) for key in keys}


def _compare_json(left: Any, right: Any, *, tolerance: float = 1.0e-12, path: str = "root") -> list[dict[str, Any]]:
    import math

    mismatches: list[dict[str, Any]] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                mismatches.append({"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)})
            else:
                mismatches.extend(_compare_json(left[key], right[key], tolerance=tolerance, path=f"{path}.{key}"))
        return mismatches
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            mismatches.append({"path": path, "left_length": len(left), "right_length": len(right)})
            return mismatches
        for index, (a, b) in enumerate(zip(left, right)):
            mismatches.extend(_compare_json(a, b, tolerance=tolerance, path=f"{path}[{index}]"))
        return mismatches
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
            mismatches.append({"path": path, "left": left, "right": right})
    elif left != right:
        mismatches.append({"path": path, "left": left, "right": right})
    return mismatches


def _read_legacy_rows(sealed: Mapping[str, Any], runtime_stats: Mapping[str, Any], output_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    reproduction: dict[str, Any] = {}
    snapshots = sealed["snapshots"]
    for label in LEGACY_CHECKPOINT_ORDER:
        rows: list[dict[str, Any]] = []
        for index, snapshot in enumerate(snapshots):
            source_path = Path(sealed["root"]) / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
            raw = np.load(source_path, allow_pickle=False)
            raw_file_sha = sha256_file(source_path)
            physical, details = _decode_with_stats(
                raw,
                snapshot["state_v1"],
                runtime_stats,
                stats_sha256=EXPECTED_RUNTIME_STATS_SHA,
                payload_sha256=EXPECTED_RUNTIME_STATS_PAYLOAD_SHA,
                chunk_id=str(snapshot["window_id"]),
                timestamps=snapshot["source_timestamps"],
                observation_timestamp=float(snapshot["observation_timestamp"]),
            )
            normalized = _normalized_metrics(raw, snapshot["expert_physical_action"], snapshot["state_v1"], snapshot["source_timestamps"], runtime_stats)
            row = _evaluation_row(
                raw,
                physical,
                details,
                normalized,
                snapshot,
                {"label": label, "tree_sha256": LEGACY_TREE[label]},
                source_path.relative_to(output_root) if source_path.is_relative_to(output_root) else source_path,
                raw_file_sha,
                "8306b76f327b260310a22a5d22551ebf299366f8fea4124bcbfbdab87e807666",
            )
            rows.append(row)
        rows_by_label[label] = rows
        aggregate = aggregate_evaluation_rows(rows)
        sealed_aggregate = sealed["aggregates"][label]
        mismatches = _compare_json(_aggregate_core(aggregate), _aggregate_core(sealed_aggregate))
        reproduction[label] = {
            "status": "PASS" if not mismatches else "BLOCKED",
            "computed_aggregate_sha256": sha256_bytes((json.dumps(_jsonable(aggregate), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()),
            "sealed_aggregate_sha256": sha256_file(Path(sealed["root"]) / f"aggregate_{label.replace('@', '_at_')}.json"),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
        }
        _require(not mismatches, f"sealed aggregate reproduction failed for {label}")
        write_json(output_root / f"legacy_{label.replace('@', '_at_')}_aggregate_reproduced.json", aggregate)
        write_json(output_root / f"legacy_{label.replace('@', '_at_')}_rows.json", rows)
    return rows_by_label, reproduction


def _run_n2_inference(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any], snapshots: Sequence[Mapping[str, Any]], sealed_noise: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from stage38_structured_offline_evaluation import _load_structured_model
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    _EXECUTION_BOUNDARY["gpu"] = True
    processor, embodiment, modality_configs = _load_runtime_processor(paths)
    windows = _make_windows(processor, embodiment, modality_configs, paths["dataset"], snapshots)
    _equal(len(windows), 24, "TRAIN24 prepared window count")
    noise_bank = _make_noise_bank(torch, sealed_noise)
    runtime_stats = read_json(paths["runtime_stats"])
    n2_stats = read_json(paths["n2_stats"])
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, Any] = {}
    raw_manifest = output_root / "n2_raw_predictions.jsonl"
    raw_manifest.write_text("", encoding="utf-8")
    for step in (1000, 2000):
        label = f"N2@{step}"
        checkpoint_path = paths["training_root"] / f"checkpoint-{step}"
        checkpoint = _checkpoint_record(checkpoint_path, label)
        expected_tree = paths.get(f"expected_n2_{step}_tree")
        if expected_tree:
            _equal(checkpoint["tree_sha256"], expected_tree, f"N2 checkpoint tree {step}")
        model = _load_structured_model(checkpoint_path, "cuda:0")
        _EXECUTION_BOUNDARY["model_construction"] = int(_EXECUTION_BOUNDARY["model_construction"]) + 1
        model.eval()
        hook_counter = {"count": 0}

        def count_action_head(*_args: Any, **_kwargs: Any) -> None:
            hook_counter["count"] += 1
            _EXECUTION_BOUNDARY["action_head_dit_forward_calls"] = int(_EXECUTION_BOUNDARY["action_head_dit_forward_calls"]) + 1

        hook = model.action_head.model.register_forward_hook(count_action_head)
        rows: list[dict[str, Any]] = []
        try:
            for index, window in enumerate(windows):
                with torch.inference_mode():
                    raw, sampler_calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
                _EXECUTION_BOUNDARY["model_forward_windows"] = int(_EXECUTION_BOUNDARY["model_forward_windows"]) + 1
                _EXECUTION_BOUNDARY["conditioning_passes"] = int(_EXECUTION_BOUNDARY["conditioning_passes"]) + 1
                _equal(sampler_calls, SAMPLER_STEPS, f"N2 sampler call count {label}/{index}")
                raw_np = raw.detach().cpu().numpy().astype(np.float32, copy=False)
                raw_path = output_root / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, raw_np, allow_pickle=False)
                raw_file_sha = sha256_file(raw_path)
                raw_array_sha = sha256_bytes(np.ascontiguousarray(raw_np).tobytes())
                append_jsonl(raw_manifest, {
                    "checkpoint": checkpoint,
                    "window_id": window["window_id"],
                    "path": str(raw_path.relative_to(output_root)),
                    "file_sha256": raw_file_sha,
                    "array_sha256": raw_array_sha,
                    "shape": list(raw_np.shape),
                    "noise_sha256": sealed_noise["noise_sha256_by_window"][index],
                    "sampler_calls_observed": sampler_calls,
                    "decoded": False,
                })
                physical, details = _decode_with_stats(
                    raw_np,
                    window["snapshot"]["state_v1"],
                    n2_stats,
                    stats_sha256=sha256_file(paths["n2_stats"]),
                    payload_sha256=str(n2_stats.get("statistics_payload_sha256")),
                    chunk_id=str(window["window_id"]),
                    timestamps=window["snapshot"]["source_timestamps"],
                    observation_timestamp=float(window["snapshot"]["observation_timestamp"]),
                )
                normalized = _normalized_metrics(raw_np, window["snapshot"]["expert_physical_action"], window["snapshot"]["state_v1"], window["snapshot"]["source_timestamps"], n2_stats)
                row = _evaluation_row(
                    raw_np,
                    physical,
                    details,
                    normalized,
                    window["snapshot"],
                    checkpoint,
                    raw_path.relative_to(output_root),
                    raw_file_sha,
                    EXPECTED_GROOT_COMMIT,
                )
                row["noise_sha256"] = sealed_noise["noise_sha256_by_window"][index]
                row["sampler"] = {
                    "name": "source-mapped-four-step-Euler-action-head-graph",
                    "timestep_ids": list(TIMESTEP_IDS),
                    "steps_observed": sampler_calls,
                    "official_public_sampler_called": False,
                }
                rows.append(row)
                append_jsonl(raw_manifest, {
                    "checkpoint": checkpoint,
                    "window_id": window["window_id"],
                    "decoded": True,
                    "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                    "metric_semantics_source": "ACCEPTED_STAGE44R",
                })
        finally:
            hook.remove()
            del model
            torch.cuda.empty_cache()
        _equal(hook_counter["count"], 24 * SAMPLER_STEPS, f"N2 action-head count {label}")
        _equal(len(rows), 24, f"N2 row count {label}")
        aggregate = aggregate_evaluation_rows(rows)
        aggregate["runtime_observed"] = {
            "checkpoint_window_forwards": 24,
            "action_head_dit_forward_calls": hook_counter["count"],
            "conditioning_passes": 24,
            "raw_prediction_count": 24,
            "reference_updates": 0,
            "wbc_steps": 0,
        }
        write_json(output_root / f"aggregate_{label.replace('@', '_at_')}.json", aggregate)
        write_json(output_root / f"{label.replace('@', '_at_')}_rows.json", rows)
        rows_by_label[label] = rows
        aggregates[label] = aggregate
    return rows_by_label, aggregates, {
        "raw_predictions_manifest_sha256": sha256_file(raw_manifest),
        "action_head_dit_forward_calls": int(_EXECUTION_BOUNDARY["action_head_dit_forward_calls"]),
        "conditioning_passes": int(_EXECUTION_BOUNDARY["conditioning_passes"]),
        "model_forward_windows": int(_EXECUTION_BOUNDARY["model_forward_windows"]),
    }


def _paired_metrics(rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import numeric_summary

    scalar = (
        "position_rmse_m", "position_max_abs_m", "endpoint_error_m", "prediction_path_length_m",
        "expert_path_length_m", "eef_rot6d_rmse", "eef_rot6d_max_abs", "gripper_rmse",
        "gripper_max_abs", "full_action_rmse_mixed_units",
    )
    target_scalar = (
        "initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m",
        "absolute_progress_m", "best_progress_m", "path_length_m", "endpoint_target_vector_cosine",
    )

    def pair(left_label: str, right_label: str, name: str) -> dict[str, Any]:
        left = rows_by_label[left_label]
        right = rows_by_label[right_label]
        _equal([row["window_id"] for row in left], [row["window_id"] for row in right], f"paired order {name}")
        rows: list[dict[str, Any]] = []
        for lrow, rrow in zip(left, right):
            lm = lrow["metrics"]
            rm = rrow["metrics"]
            lt = lm["target"]
            rt = rm["target"]
            delta = {field: float(rm[field] - lm[field]) for field in scalar}
            target_delta = {field: (None if rt.get(field) is None or lt.get(field) is None else float(rt[field] - lt[field])) for field in target_scalar}
            rows.append({
                "window_id": lrow["window_id"],
                "target_id": lrow.get("target_id"),
                "left": left_label,
                "right": right_label,
                "physical_metric_delta_right_minus_left": delta,
                "target_metric_delta_right_minus_left": target_delta,
                "progress_sign_left": lt.get("progress_sign"),
                "progress_sign_right": rt.get("progress_sign"),
                "normalized_active_rmse_left": lrow["normalized_metrics"]["active_rmse"],
                "normalized_active_rmse_right": rrow["normalized_metrics"]["active_rmse"],
                "normalized_active_rmse_delta_right_minus_left": float(rrow["normalized_metrics"]["active_rmse"] - lrow["normalized_metrics"]["active_rmse"]),
            })
        aggregate: dict[str, Any] = {"name": name, "left": left_label, "right": right_label, "window_count": len(rows)}
        for field in scalar:
            aggregate.setdefault("physical_metric_delta", {})[field] = numeric_summary(row["physical_metric_delta_right_minus_left"][field] for row in rows)
        for field in target_scalar:
            values = [row["target_metric_delta_right_minus_left"][field] for row in rows if row["target_metric_delta_right_minus_left"][field] is not None]
            aggregate.setdefault("target_metric_delta", {})[field] = numeric_summary(values) if values else None
        aggregate["normalized_active_rmse_delta"] = numeric_summary(row["normalized_active_rmse_delta_right_minus_left"] for row in rows)
        aggregate["progress_sign_transition_counts"] = {
            f"{left_sign}->{right_sign}": int(sum(row["progress_sign_left"] == left_sign and row["progress_sign_right"] == right_sign for row in rows))
            for left_sign in ("POSITIVE", "ZERO", "NEGATIVE")
            for right_sign in ("POSITIVE", "ZERO", "NEGATIVE")
        }
        return {"summary": aggregate, "rows": rows}

    pairs = {
        "coverage_legacy_to_n2_at_1000": pair("FULL24@1000", "N2@1000", "coverage_legacy_to_n2_at_1000"),
        "coverage_legacy_to_n2_at_2000": pair("FULL24@2000", "N2@2000", "coverage_legacy_to_n2_at_2000"),
        "exposure_n2_1000_to_2000": pair("N2@1000", "N2@2000", "exposure_n2_1000_to_2000"),
        "legacy_exposure_1000_to_2000": pair("FULL24@1000", "FULL24@2000", "legacy_exposure_1000_to_2000"),
    }
    return pairs, {
        "schema": "CORELAB_STAGE45E_N2_TRAIN24_PAIRED_ANALYSIS_V1",
        "comparison_policy": "right_minus_left; descriptive; no automatic winner",
        "pairs": {name: value["summary"] for name, value in pairs.items()},
    }, pairs


def _per_target(rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    from go2arx5_vla.evaluation.trajectory_metrics import numeric_summary

    result: dict[str, Any] = {"schema": "CORELAB_STAGE45E_N2_TRAIN24_PER_TARGET_V1", "targets": {}}
    for target in TARGET_ORDER:
        result["targets"][target] = {}
        for label, rows in rows_by_label.items():
            selected = [row for row in rows if row.get("target_id") == target]
            _equal(len(selected), 6, f"per-target row count {label}/{target}")
            result["targets"][target][label] = {
                "window_ids": [row["window_id"] for row in selected],
                "physical": {
                    field: numeric_summary(float(row["metrics"][field]) for row in selected)
                    for field in ("position_rmse_m", "endpoint_error_m", "full_action_rmse_mixed_units")
                },
                "target": {
                    field: numeric_summary(float(row["metrics"]["target"][field]) for row in selected)
                    for field in ("initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m")
                },
                "normalized": {
                    field: numeric_summary(float(row["normalized_metrics"][field]) for row in selected)
                    for field in ("active_rmse", "active_mae", "active_max_abs", "raw_active_outside_neg1_pos1_fraction")
                },
                "progress_sign_counts": {
                    sign: int(sum(row["metrics"]["target"]["progress_sign"] == sign for row in selected))
                    for sign in ("POSITIVE", "ZERO", "NEGATIVE")
                },
            }
    return result


def _clipping_and_gain(runtime_stats: Mapping[str, Any], n2_stats: Mapping[str, Any], rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.structured_action import structured_action_mask

    mask = structured_action_mask(TIMING_MODE).astype(bool)
    def route(doc: Mapping[str, Any], labels: Sequence[str]) -> dict[str, Any]:
        statistics = doc.get("statistics", doc)
        lower = np.asarray(doc.get("selected_lower", statistics.get("selected_lower")), dtype=np.float64)
        upper = np.asarray(doc.get("selected_upper", statistics.get("selected_upper")), dtype=np.float64)
        gain = (upper - lower) * 0.5
        xyz_mask = np.zeros_like(mask, dtype=bool)
        xyz_mask[:, :3] = mask[:, :3]
        active_dimensions = mask.any(axis=0)
        active_gain = gain[active_dimensions]
        return {
            "stats_selected_lower_sha256": sha256_bytes(lower.tobytes()),
            "stats_selected_upper_sha256": sha256_bytes(upper.tobytes()),
            "active_scalar_count": int(mask.sum()),
            "active_clipped_gain_m_per_unit_summary": {
                "count": int(active_gain.size),
                "min": float(np.min(active_gain)),
                "max": float(np.max(active_gain)),
                "mean": float(np.mean(active_gain)),
            },
            "xyz_active_scalar_count_by_axis": [int(np.count_nonzero(xyz_mask[:, axis])) for axis in range(3)],
            "xyz_gain_m_per_normalized_unit": [float(value) for value in gain[:3]],
            "xyz_common_half_range_m": float((upper[0] - lower[0]) * 0.5),
            "checkpoint_raw_clipping": {
                label: {
                    "active_outside_neg1_pos1_count_total": int(sum(row["normalized_metrics"]["raw_active_outside_neg1_pos1_count"] for row in rows_by_label[label])),
                    "active_scalar_total": int(24 * int(mask.sum())),
                }
                for label in labels
            },
        }
    return {
        "schema": "CORELAB_STAGE45E_N2_CLIPPING_GAIN_V1",
        "full24_runtime_v2": route(runtime_stats, LEGACY_CHECKPOINT_ORDER),
        "n2_balanced_xyz": route(n2_stats, N2_CHECKPOINT_ORDER),
        "interpretation": "N2 changes only selected XYZ inverse-normalization gain; metric reports are descriptive and do not declare a winner.",
    }


def _failure_taxonomy(rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]], pairs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "CORELAB_STAGE45E_N2_TRAIN24_FAILURE_TAXONOMY_V1", "cohort_count": 24, "checkpoints": {}}
    for label, rows in rows_by_label.items():
        categories = {
            "nonfinite": int(sum(not row.get("finite", False) for row in rows)),
            "sampler_step_mismatch": int(sum(row.get("sampler", {}).get("steps_observed") != SAMPLER_STEPS for row in rows)),
            "target_progress_positive": int(sum(row["metrics"]["target"]["progress_sign"] == "POSITIVE" for row in rows)),
            "target_progress_zero": int(sum(row["metrics"]["target"]["progress_sign"] == "ZERO" for row in rows)),
            "target_progress_negative": int(sum(row["metrics"]["target"]["progress_sign"] == "NEGATIVE" for row in rows)),
            "normalized_clipping_rows": int(sum(row["normalized_metrics"]["raw_active_outside_neg1_pos1_count"] > 0 for row in rows)),
        }
        result["checkpoints"][label] = categories
    paired_outliers: dict[str, Any] = {}
    for name, value in pairs.items():
        rows = value["rows"]
        ranked = sorted(rows, key=lambda row: abs(float(row["target_metric_delta_right_minus_left"]["absolute_progress_m"])), reverse=True)
        paired_outliers[name] = [
            {
                "window_id": row["window_id"],
                "target_id": row.get("target_id"),
                "absolute_progress_delta_m": row["target_metric_delta_right_minus_left"]["absolute_progress_m"],
                "final_distance_delta_m": row["target_metric_delta_right_minus_left"]["final_target_distance_m"],
                "normalized_rmse_delta": row["normalized_active_rmse_delta_right_minus_left"],
                "progress_sign_left": row["progress_sign_left"],
                "progress_sign_right": row["progress_sign_right"],
            }
            for row in ranked[:5]
        ]
    result["paired_outliers_top5_by_abs_progress_delta"] = paired_outliers
    result["failure_policy"] = "No binary task-success threshold; finite/sampler integrity and continuous target-progress signs only."
    return result


def _static_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden_top = sorted(name for name in imports if name.split(".")[0] in {"torch", "gr00t", "transformers", "isaacgym", "omni"})
    forbidden_calls = {"get_action", "backward", "optimizer.step", "scheduler.step", "save_pretrained", "save_model", "save_checkpoint", "qsub"}
    call_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "step"}:
                    call_hits.append(name)
            elif isinstance(node.func, ast.Name) and node.func.id == "qsub":
                call_hits.append("qsub")
    # The runtime adapter deliberately uses the direct action-head graph and
    # never calls the public get_action/backward/optimizer APIs.  Mentions of
    # locked resources in prose and evidence schemas are not execution.
    locked_hits = sorted(set(call_hits).intersection(forbidden_calls))
    return {
        "schema": "CORELAB_STAGE45E_N2_TRAIN24_EVAL_STATIC_AUDIT_V1",
        "status": "PASS" if not forbidden_top and not locked_hits else "BLOCKED",
        "top_level_forbidden_imports": forbidden_top,
        "forbidden_runtime_call_hits": locked_hits,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "gpu": False,
        "wbc_steps": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _reset_boundary()
    if args.mode == "static":
        return _static_audit()
    paths = {
        "base_model": Path(args.base_model).resolve(),
        "dataset": Path(args.dataset).resolve(),
        "groot": Path(args.groot).resolve(),
        "cosmos": Path(args.cosmos).resolve(),
        "modality": Path(args.modality).resolve(),
        "config": Path(args.config).resolve(),
        "runtime_stats": Path(args.runtime_stats).resolve(),
        "n2_stats": Path(args.n2_stats).resolve(),
        "training_root": Path(args.training_root).resolve(),
    }
    for name, path in paths.items():
        _require(path.exists() and (not path.is_symlink() or name == "cosmos"), f"required path unavailable: {name}={path}")
    baseline_root = Path(args.sealed_baseline).resolve()
    snapshots_path = Path(args.snapshot_preflight).resolve()
    output_root = Path(args.output_root).resolve()
    _require(not output_root.exists() and not output_root.is_symlink(), f"output root must be new: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    _require(shutil.disk_usage(output_root).free >= 8 * 1024**3, "at least 8 GiB free disk is required for evaluation evidence")
    identities = _verify_external(paths, paths["training_root"])
    sealed = _verify_sealed_baseline(baseline_root, snapshots_path)
    snapshots = _snapshot_lookup(read_json(snapshots_path)["snapshots"])
    gpu = _gpu_identity(args.gpu_uuid)
    _prepare_environment(paths, output_root, gpu)
    write_json(output_root / "stage45e_n2_train24_eval_admission.json", {
        "schema": "CORELAB_STAGE45E_N2_TRAIN24_EVAL_ADMISSION_V1",
        "status": "PASS",
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "branch": BRANCH,
        "repository_head": _git(REPO, "rev-parse", "HEAD"),
        "training_root": str(paths["training_root"]),
        "sealed_baseline_root": str(baseline_root),
        "identities": identities,
        "sealed_baseline": {key: value for key, value in sealed.items() if key != "snapshots" and key != "aggregates"},
        "gpu": gpu,
        "execution_boundary": dict(_EXECUTION_BOUNDARY),
    })
    runtime_stats = read_json(paths["runtime_stats"])
    legacy_rows, reproduction = _read_legacy_rows(sealed, runtime_stats, output_root)
    write_json(output_root / "sealed_aggregate_reproduction.json", {
        "schema": "CORELAB_STAGE45E_N2_SEALED_AGGREGATE_REPRODUCTION_V1",
        "status": "PASS" if all(item["status"] == "PASS" for item in reproduction.values()) else "BLOCKED",
        "checkpoints": reproduction,
        "raw_first": True,
        "cpu_only": True,
    })
    n2_rows, n2_aggregates, runtime = _run_n2_inference(paths, output_root, gpu, snapshots, sealed["noise_manifest"])
    rows_by_label: dict[str, Sequence[Mapping[str, Any]]] = {**legacy_rows, **n2_rows}
    all_aggregates: dict[str, Any] = {}
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    for label, rows in rows_by_label.items():
        if label in n2_aggregates:
            all_aggregates[label] = n2_aggregates[label]
        else:
            all_aggregates[label] = aggregate_evaluation_rows(rows)
    write_json(output_root / "all_checkpoint_aggregates.json", all_aggregates)
    pair_data, paired_summary, pair_details = _paired_metrics(rows_by_label)
    write_json(output_root / "paired_comparisons.json", pair_data)
    write_json(output_root / "paired_summary.json", paired_summary)
    per_target = _per_target(rows_by_label)
    write_json(output_root / "per_target_summary.json", per_target)
    n2_stats = read_json(paths["n2_stats"])
    clipping = _clipping_and_gain(runtime_stats, n2_stats, rows_by_label)
    write_json(output_root / "clipping_and_inverse_normalization_gain.json", clipping)
    taxonomy = _failure_taxonomy(rows_by_label, pair_details)
    write_json(output_root / "failure_taxonomy.json", taxonomy)
    aggregate_hashes = {label: sha256_file(output_root / f"aggregate_{label.replace('@', '_at_')}.json") for label in N2_CHECKPOINT_ORDER}
    seal = {
        "schema": "CORELAB_STAGE45E_N2_TRAIN24_EVALUATION_SEAL_V1",
        "status": "PASS",
        "token": EVAL_TOKEN,
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "branch": BRANCH,
        "repository_head": _git(REPO, "rev-parse", "HEAD"),
        "training_result_sha256": identities["training_result_sha256"],
        "n2_checkpoint_tree_sha256": {
            "N2@1000": identities["checkpoint_1000_tree_sha256"],
            "N2@2000": identities["checkpoint_2000_tree_sha256"],
        },
        "legacy_read_only_baselines": {
            label: {
                "tree_sha256": LEGACY_TREE[label],
                "sealed_aggregate_sha256": sealed["aggregate_sha256"][label],
            }
            for label in LEGACY_CHECKPOINT_ORDER
        },
        "stats_routes": {
            "FULL24_RUNTIME_V2": {"sha256": EXPECTED_RUNTIME_STATS_SHA, "payload_sha256": EXPECTED_RUNTIME_STATS_PAYLOAD_SHA},
            "N2_BALANCED_XYZ": {"sha256": identities["n2_stats_sha256"], "payload_sha256": identities["n2_stats_payload_sha256"]},
        },
        "cohort": {"name": "FULL24_EXISTING_TRAIN", "split": "TRAIN", "window_count": 24, "window_order": "target_major_then_snapshot_major"},
        "sampler": {"seed": DEVELOPMENT_SEED, "noise_manifest_sha256": sealed["noise_manifest_sha256"], "steps": SAMPLER_STEPS, "dt": SAMPLER_DT, "timestep_ids": list(TIMESTEP_IDS), "same_noise_across_checkpoints": True, "official_public_sampler_called": False},
        "runtime_counters": {**runtime, "reference_updates": 0, "wbc_steps": 0, "training": False, "backward_steps": 0, "optimizer_steps": 0, "scheduler_steps": 0, "checkpoint_writes": 0},
        "evidence_artifacts": {
            "n2_raw_predictions_manifest_sha256": runtime["raw_predictions_manifest_sha256"],
            "n2_aggregate_sha256": aggregate_hashes,
            "sealed_aggregate_reproduction_sha256": sha256_file(output_root / "sealed_aggregate_reproduction.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "per_target_summary_sha256": sha256_file(output_root / "per_target_summary.json"),
            "clipping_gain_sha256": sha256_file(output_root / "clipping_and_inverse_normalization_gain.json"),
            "failure_taxonomy_sha256": sha256_file(output_root / "failure_taxonomy.json"),
        },
        "scientific_semantics": {
            "metric_source": "ACCEPTED_STAGE44R",
            "decoder": "Stage40_inverse_normalization_and_materialization",
            "target_xyz_in_model_input": False,
            "automatic_winner": False,
            "threshold_used": None,
        },
        "gpu": gpu,
        "execution_boundary": {**_EXECUTION_BOUNDARY, "training": False, "validation_e_read": False, "holdout_f_read": False, "wbc_steps": 0, "simulator_steps": 0, "hardware": False, "pbs_submissions": 0},
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage45e_n2_train24_eval_seal.json", seal)
    result = {
        "schema": "CORELAB_STAGE45E_N2_4080_TRAIN24_COMPLETE_V1",
        "status": RETURN_TOKEN,
        "return_token": RETURN_TOKEN,
        "training": read_json(paths["training_root"] / "stage45e_n2_training_result.json"),
        "evaluation": seal,
        "paired_summary": paired_summary,
        "output_root": str(output_root),
        "forbidden_action_counters": {
            "new_training_after_seal": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
            "PBS/qsub": 0,
        },
    }
    write_json(output_root / "stage45e_n2_corelab_complete.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "eval"), default="static")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--sealed-baseline", type=Path)
    parser.add_argument("--snapshot-preflight", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--groot", type=Path)
    parser.add_argument("--cosmos", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime-stats", type=Path)
    parser.add_argument("--n2-stats", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root is not None else None
    try:
        result = run(args)
    except Exception as exc:  # pragma: no cover - fail-closed CLI boundary
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "stage45e_n2_train24_eval_failure.json", {
                "schema": "CORELAB_STAGE45E_N2_TRAIN24_EVAL_FAILURE_V1",
                "status": "BLOCKED",
                "stage": STAGE,
                "pi_gate": PI_GATE,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "execution_boundary": dict(_EXECUTION_BOUNDARY),
                "retry_or_substitution": False,
                "timestamp_utc": now_utc(),
            }, overwrite=True)
        print(f"STAGE45E_N2_TRAIN24_EVAL_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") == RETURN_TOKEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
