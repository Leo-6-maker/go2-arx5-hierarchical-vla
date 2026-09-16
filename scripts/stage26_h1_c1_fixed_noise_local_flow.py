#!/usr/bin/env python3
"""Prepare or execute the Stage26 H1 fixed-noise local-flow diagnostic.

The default modes are standard-library-only preparation and self-test modes.
They do not import Torch, construct a model, read checkpoint tensors, open a
CUDA device, run a forward pass, train, submit PBS, or execute WBC.

The separately authorized --execute mode is intentionally narrow. It reuses
the official Stage22/23 processor and checkpoint loading path, then
source-maps only the teacher-forced action-head operations from the pinned
N1.7 training forward. It supplies the same canonical CPU-generated noise,
the same fixed t, and the same processed visual/state conditioning to BASE
and O1@200. It never calls the official free-running sampler, get_action or
the complete training forward; those would answer a different question.

This file is an experiment runner, not a scientific interpretation gate. It
reports paired measurements and descriptive aggregates; it does not assign a
quantitative PASS threshold to local-flow quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from statistics import geometric_mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence


STAGE26_SCHEMA = "STAGE26_H1_C1_FIXED_NOISE_LOCAL_FLOW_V1"
STAGE23_SCHEMA = "STAGE23_PHASE_C_FLOW_FIELD_DIAGNOSTIC_PROTOCOL_V1"
STAGE26_BASE_SHA = "4005a4dd1ab6c82807985ef856d340fa50acaa6c"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
EXPECTED_TINY1_FILE_COUNT = 13
EXPECTED_TINY1_TOTAL_BYTES = 152958
EXPECTED_STAGE22_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
EXPECTED_STAGE22_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
EXPECTED_BASE_PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
EXPECTED_BASE_STATISTICS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
EXPECTED_MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_BASE_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
EXPECTED_BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_BASE_TREE_PREFIX = "payload/GR00T-N1.7-3B"
EXPECTED_O1_INDEX_SHA256 = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
EXPECTED_O1_SEAL_SHA256 = "7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806"
EXPECTED_O1_SEAL_PATH = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-stage23-o1-phase-a-20260903/"
    "extracted/phase_a_seal_20260903/checkpoint_seal/checkpoint_file_seal.tsv"
)
EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST = (
    "experiment_cfg/conf.yaml",
    "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
)
EXPECTED_BASE_CHECKPOINT = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"
)
EXPECTED_O1_CHECKPOINT = Path(
    "/scratch/e1192548/stage23_o1_20260903/runs/597966.hopper-m-02/candidate_200/checkpoint-200"
)
EXPECTED_DATASET = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/tiny1_h2stats_lerobot_v2"
)
EXPECTED_SNAPSHOT_ROOT = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage22/expert_dataset_v1/snapshots"
)
EXPECTED_SNAPSHOT_RELATIVE_PATH = Path(
    "stage22_expert_red_target_train_a/expert-0000.json"
)
EXPECTED_MODALITY_PATH = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py"
)
EXPECTED_STAGE10_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10")
EXPECTED_COSMOS_CACHE_RELATIVE = Path(
    "cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"
)

HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
INACTIVE_DIM = UNIVERSAL_DIM - ACTIVE_DIM
NOISE_SEEDS = (2223, 2224, 2225)
T_QUANTILES = (0.10, 0.50, 0.90)
ACTION_PERIOD_S = 0.05
TRAIN_SEED = 2222
MAX_CHECKPOINTS = 2
MAX_PROBES = 9
MAX_FLOW_EVALUATIONS = MAX_CHECKPOINTS * MAX_PROBES
TARGET_SOURCE_EPISODE = "stage22_expert_red_target_train_a"
EXPECTED_SOURCE_SNAPSHOT_STEMS = tuple(f"expert-{index:04d}" for index in range(6))

# These are source locations and operation anchors, not a duplicated
# scientific implementation of the official model. The runner's direct
# action-head function below must stay aligned with these anchors.
SOURCE_EQUIVALENCE_MAP: dict[str, dict[str, Any]] = {
    "checkpoint_load": {
        "path": "gr00t/model/gr00t_n1d7/setup.py",
        "lines": [75, 89],
        "anchor": "AutoModel.from_pretrained(... start_from_checkpoint ..., load_bf16=...)",
        "runner_use": "AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)",
    },
    "inference_dtype": {
        "path": "scripts/stage22_profile_evaluation.py",
        "lines": [358, 364],
        "anchor": "model.eval(); model.to(device=args.device, dtype=torch.bfloat16)",
        "runner_use": "exact Stage22/23 inference cast before both paired probes",
    },
    "backbone_conditioning": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [598, 599],
        "anchor": "backbone_outputs = self.backbone(backbone_inputs)",
        "runner_use": "one eval backbone pass per checkpoint, then process_backbone_output",
    },
    "state_encoder": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [209, 217],
        "anchor": "state reshape followed by self.state_encoder",
        "runner_use": "direct teacher-forced action-head path",
    },
    "action_encoder_and_time": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [237, 239],
        "anchor": "t_discretized then self.action_encoder(noisy_trajectory, ...)",
        "runner_use": "fixed noisy_action and fixed t_discretized; no official resampling",
    },
    "fusion_and_dit": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [241, 270],
        "anchor": "position embedding, concat, self.model(... return_all_hidden_states=True)",
        "runner_use": "directly mirrored with the same alternate/non-alternate branch",
    },
    "velocity_output": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [272, 278],
        "anchor": "action_decoder then pred[:, -actions.shape[1]:]",
        "runner_use": "raw predicted velocity before decode, loss, or sampler",
    },
    "free_running_exclusion": {
        "path": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "lines": [325, 443],
        "anchor": "get_action_with_features free-running Euler loop",
        "runner_use": "must not be called by C1",
    },
}


def _jsonable(value: Any) -> Any:
    """Convert common runtime values without importing numerical packages."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_regular(path: Path, label: str) -> None:
    mode = os.lstat(path).st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")


def _regular_file_entries(root: Path) -> list[dict[str, Any]]:
    """Return a deterministic regular-file manifest and reject symlinks."""

    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"root must not be a symlink: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"directory is missing: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                raise ValueError(f"tree contains symlinked directory: {child}")
            if not stat.S_ISDIR(os.lstat(child).st_mode):
                raise ValueError(f"tree contains non-directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            _assert_regular(child, "tree entry")
            relative = child.relative_to(root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "bytes": int(child.stat().st_size),
                    "sha256": _sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def _tiny_tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(
            f"{entry['path']}\0{entry['sha256']}\0{entry['bytes']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _stage22_tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    """Reproduce the Stage22 handoff tree digest for optional evidence."""

    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(
            f"payload/lerobot_train/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    _assert_regular(path, "JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_tiny_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Verify TINY1 identity before any official loader or model read."""

    entries = _regular_file_entries(dataset_dir)
    tree_sha = _tiny_tree_digest(entries)
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    if len(entries) != EXPECTED_TINY1_FILE_COUNT:
        raise ValueError(f"TINY1 file count mismatch: {len(entries)}")
    if total_bytes != EXPECTED_TINY1_TOTAL_BYTES:
        raise ValueError(f"TINY1 byte count mismatch: {total_bytes}")
    if tree_sha != EXPECTED_TINY1_TREE_SHA256:
        raise ValueError(f"TINY1 tree SHA mismatch: {tree_sha}")

    meta = dataset_dir / "meta"
    stats_path = meta / "stats.json"
    relative_stats_path = meta / "relative_stats.json"
    if _sha256_file(stats_path) != EXPECTED_STAGE22_STATS_SHA256:
        raise ValueError("TINY1 stats.json does not match frozen Stage22 stats")
    if _sha256_file(relative_stats_path) != EXPECTED_STAGE22_RELATIVE_STATS_SHA256:
        raise ValueError("TINY1 relative_stats.json does not match frozen Stage22 stats")

    episode_lines = [
        line
        for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episode_lines) != 1:
        raise ValueError(f"TINY1 must contain exactly one episode row, got {len(episode_lines)}")
    episode = json.loads(episode_lines[0])
    if str(episode.get("split")) != "train":
        raise ValueError("TINY1 episode is not in the train split")
    if int(episode.get("episode_index", -1)) != 0:
        raise ValueError("TINY1 episode index is not zero")
    if int(episode.get("length", -1)) != HORIZON:
        raise ValueError("TINY1 episode length is not H40")
    source_episode_id = str(episode.get("source_episode_id", ""))
    if source_episode_id != f"{TARGET_SOURCE_EPISODE}:expert-0000":
        raise ValueError(f"unexpected TINY1 source_episode_id: {source_episode_id!r}")

    split_lock = _read_json(meta / "stage23_tiny_split_lock.json")
    if split_lock.get("selected_source_episode_ids") != [source_episode_id]:
        raise ValueError("TINY1 split lock does not bind the first Stage22 source episode")
    if split_lock.get("validation_target_ids") or split_lock.get("final_test_target_ids"):
        raise ValueError("TINY1 split lock contains non-training targets")
    if (
        split_lock.get("holdout_e_accessed") is not False
        or split_lock.get("holdout_f_accessed") is not False
    ):
        raise ValueError("TINY1 split lock does not prove holdout exclusion")

    return {
        "path": str(dataset_dir),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": tree_sha,
        "stage22_tree_digest": _stage22_tree_digest(entries),
        "stats_json_sha256": _sha256_file(stats_path),
        "relative_stats_json_sha256": _sha256_file(relative_stats_path),
        "episode": {
            "episode_index": 0,
            "length": HORIZON,
            "split": "train",
            "source_episode_id": source_episode_id,
        },
        "validation_e_read": False,
        "holdout_f_read": False,
        "symlinks_rejected": True,
    }


def _verify_snapshot_identity(snapshot_root: Path) -> tuple[dict[str, Any], Path]:
    """Bind only the six train-a source snapshots; never scan holdout dirs."""

    root = Path(snapshot_root)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"snapshot root is unavailable or symlinked: {root}")
    train_dir = root / TARGET_SOURCE_EPISODE
    if train_dir.is_symlink() or not train_dir.is_dir():
        raise FileNotFoundError(f"source train snapshot directory is unavailable: {train_dir}")

    observed: list[str] = []
    for entry in sorted(os.scandir(train_dir), key=lambda item: item.name):
        path = Path(entry.path)
        if entry.is_symlink():
            raise ValueError(f"source snapshot directory contains symlink: {path}")
        if not entry.is_file() or path.suffix != ".json":
            raise ValueError(f"unexpected source snapshot entry: {path}")
        observed.append(path.stem)
    if tuple(observed) != EXPECTED_SOURCE_SNAPSHOT_STEMS:
        raise ValueError(f"source snapshot order mismatch: {observed}")

    selected = train_dir / "expert-0000.json"
    payload = _read_json(selected)
    if not isinstance(payload, Mapping):
        raise ValueError("selected source snapshot is not a JSON object")
    example_id = str(payload.get("example_id", ""))
    raw_episode_id = str(payload.get("raw_episode_id", ""))
    if raw_episode_id != TARGET_SOURCE_EPISODE:
        raise ValueError(f"selected source snapshot raw_episode_id mismatch: {raw_episode_id!r}")
    if example_id != f"{TARGET_SOURCE_EPISODE}:expert-0000":
        raise ValueError(f"selected source snapshot example_id mismatch: {example_id!r}")
    timestamps = payload.get("future_timestamps")
    desired = payload.get("future_desired_eef_9d")
    gripper = payload.get("future_gripper")
    if not isinstance(timestamps, list) or len(timestamps) != HORIZON:
        raise ValueError("selected source snapshot does not contain H40 timestamps")
    if not isinstance(desired, list) or len(desired) != HORIZON:
        raise ValueError("selected source snapshot does not contain H40 EEF labels")
    if not isinstance(gripper, list) or len(gripper) != HORIZON:
        raise ValueError("selected source snapshot does not contain H40 gripper labels")
    if payload.get("target_excluded_from_model_input") is not True:
        raise ValueError("selected source snapshot target exclusion is not explicit")
    if any(not math.isfinite(float(value)) for value in timestamps):
        raise ValueError("selected source snapshot timestamps are non-finite")
    if any(not math.isfinite(float(value)) for row in desired for value in row):
        raise ValueError("selected source snapshot EEF labels are non-finite")
    if any(not math.isfinite(float(value)) for row in gripper for value in row):
        raise ValueError("selected source snapshot gripper labels are non-finite")

    sha256 = _sha256_file(selected)
    return (
        {
            "root": str(root),
            "selected_relative_path": EXPECTED_SNAPSHOT_RELATIVE_PATH.as_posix(),
            "selected_path": str(selected),
            "selected_sha256": sha256,
            "source_episode_id": raw_episode_id,
            "example_id": example_id,
            "snapshot_count_in_bound_train_directory": len(observed),
            "snapshot_order": observed,
            "horizon": HORIZON,
            "target_excluded_from_model_input": True,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        selected,
    )


def _load_stage23_protocol() -> Any:
    """Import the existing static protocol only."""

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir.parent) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent))
    return importlib.import_module("scripts.stage23_phase_c_flow_field_diagnostic")


def _static_probe_contract() -> dict[str, Any]:
    stage23 = _load_stage23_protocol()
    plan = stage23.protocol_plan()
    stage23.assert_protocol_static(plan)
    probes = stage23.build_probe_grid()
    if len(probes) != MAX_PROBES:
        raise AssertionError(f"expected {MAX_PROBES} probes, got {len(probes)}")
    expected_t = {
        0.10: (0.06776241796562847, 67),
        0.50: (0.36966943557751086, 369),
        0.90: (0.7837719744658148, 783),
    }
    for probe in probes:
        t_value, t_discretized = expected_t[float(probe["t_quantile"])]
        if not math.isclose(float(probe["t_value"]), t_value, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"probe t drifted: {probe}")
        if int(probe["t_discretized"]) != t_discretized:
            raise AssertionError(f"probe discrete t drifted: {probe}")
        if probe["noise_shape"] != [1, HORIZON, UNIVERSAL_DIM]:
            raise AssertionError("probe noise shape drifted")
    return {
        "schema": STAGE23_SCHEMA,
        "probe_count": len(probes),
        "probes": probes,
        "mask_partition": stage23.expected_mask_partition(),
        "stage23_assert_protocol_static": True,
    }


def _source_map_static() -> dict[str, Any]:
    required = {
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
            "process_backbone_output",
            "self.state_encoder",
            "self.action_encoder",
            "return_all_hidden_states=True",
            "self.action_decoder",
            "pred[:, -actions.shape[1]",
        ),
        "gr00t/model/gr00t_n1d7/setup.py": ("AutoModel.from_pretrained", "load_bf16"),
    }
    return {
        "status": "PASS",
        "map": copy.deepcopy(SOURCE_EQUIVALENCE_MAP),
        "required_anchors": {path: list(anchors) for path, anchors in required.items()},
        "official_graph": "pinned GR00T N1.7 Stage23 source at EXPECTED_GROOT_SHA",
        "pre_sampler": True,
        "complete_model_forward_called": False,
        "free_running_sampler_called": False,
    }


def protocol_plan() -> dict[str, Any]:
    """Return the complete non-executing Stage26 runner contract."""

    probe_contract = _static_probe_contract()
    return {
        "schema": STAGE26_SCHEMA,
        "stage": "Stage26 H1 C1 preparation",
        "status": "PREPARED_NOT_EXECUTED",
        "base_sha": STAGE26_BASE_SHA,
        "stage23_c1_protocol_reused": True,
        "fixed_identity": {
            "dataset_name": "TINY1_H2STATS",
            "dataset_path": str(EXPECTED_DATASET),
            "dataset_tree_sha256": EXPECTED_TINY1_TREE_SHA256,
            "dataset_file_count": EXPECTED_TINY1_FILE_COUNT,
            "dataset_total_bytes": EXPECTED_TINY1_TOTAL_BYTES,
            "selected_snapshot_relative_path": EXPECTED_SNAPSHOT_RELATIVE_PATH.as_posix(),
            "selected_snapshot_sha256": "RUNTIME_BOUND_FROM_EXACT_SOURCE_SNAPSHOT",
            "base_checkpoint": str(EXPECTED_BASE_CHECKPOINT),
            "base_index_sha256": EXPECTED_BASE_INDEX_SHA256,
            "base_tree_sha256": EXPECTED_BASE_TREE_SHA256,
            "base_tree_prefix": EXPECTED_BASE_TREE_PREFIX,
            "o1_checkpoint": str(EXPECTED_O1_CHECKPOINT),
            "o1_index_sha256": EXPECTED_O1_INDEX_SHA256,
            "o1_seal_path": str(EXPECTED_O1_SEAL_PATH),
            "o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
            "o1_supplemental_metadata_allowlist": list(
                EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST
            ),
            "groot_sha": EXPECTED_GROOT_SHA,
            "modality_sha256": EXPECTED_MODALITY_SHA256,
            "stage22_stats_sha256": EXPECTED_STAGE22_STATS_SHA256,
            "stage22_relative_stats_sha256": EXPECTED_STAGE22_RELATIVE_STATS_SHA256,
            "base_processor_config_sha256": EXPECTED_BASE_PROCESSOR_CONFIG_SHA256,
            "base_statistics_sha256": EXPECTED_BASE_STATISTICS_SHA256,
            "cosmos_revision": EXPECTED_COSMOS_REVISION,
            "cosmos_tree_sha256": EXPECTED_COSMOS_TREE_SHA256,
        },
        "fixed_inputs": {
            "sample": "first ordered snapshot in red_target_train_a",
            "horizon": HORIZON,
            "universal_action_shape": [1, HORIZON, UNIVERSAL_DIM],
            "physical_action_shape": [1, HORIZON, ACTIVE_DIM],
            "action_period_s": ACTION_PERIOD_S,
            "action_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
            "train_seed_context": TRAIN_SEED,
            "target_excluded_from_model_input": True,
            "processor_path": "official LeRobotEpisodeLoader -> extract_step_data -> Gr00tN1d7Processor -> collator",
        },
        "flow_equations": {
            "noise_generation": "torch.Generator(device='cpu').manual_seed(seed); torch.randn([1,40,132], dtype=torch.float32, device='cpu')",
            "noisy_action": "x_t=(1-t)*noise+t*expert_action",
            "target_velocity": "v_target=expert_action-noise",
            "t_discretized": "(t*1000).long()",
            "loss_mask": "first 10 active physical dimensions only; inactive 122 remain in input but mask zero",
            "pairing": "one canonical float32 CPU noise tensor and one t reused for BASE and O1 before model dtype/device transfer",
        },
        "probe_contract": probe_contract,
        "source_equivalence_map": _source_map_static(),
        "runtime_precision": {
            "load": "AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True)",
            "eval": True,
            "device_cast": "model.to(device=args.device, dtype=torch.bfloat16)",
            "basis": "Stage22/23 pinned profile evaluator and Stage20S inference path",
            "no_precision_override_beyond_pinned_path": True,
        },
        "execution_budget": {
            "checkpoints": ["BASE", "O1@200"],
            "maximum_checkpoints": MAX_CHECKPOINTS,
            "probes_per_checkpoint": MAX_PROBES,
            "maximum_c1_flow_evaluations": MAX_FLOW_EVALUATIONS,
            "backbone_passes_per_checkpoint": 1,
        },
        "readiness": {
            "h1_c1_execution_ready": True,
            "runtime_identity_revalidation_required": True,
            "execution_started": False,
        },
        "output_schema": {
            "per_probe": [
                "noise_sha256",
                "noisy_action_sha256",
                "target_velocity_sha256",
                "raw_predicted_velocity_sha256",
                "active_masked_flow_mse",
                "inactive_prediction_energy_mean",
                "inactive_target_energy_mean",
                "implied_universal_action_sha256",
                "active_model_action_sha256",
                "decoded_physical_action_sha256",
                "decoded_action_profile",
                "safety_gate",
                "finite_flags",
            ],
            "paired_aggregates": [
                "base_mse",
                "o1_mse",
                "o1_over_base_ratio",
                "improved_count_over_9",
                "median_ratio",
                "geometric_mean_ratio",
            ],
            "raw_velocity_hashes": "required for all 18 rows",
            "descriptive_only": True,
            "no_new_scientific_pass_threshold": True,
        },
        "execution_boundary": {
            "model_construction": False,
            "model_forward": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "iterative_sampler": False,
            "pbs_submission": False,
            "wbc_execution": False,
            "hardware_execution": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "inactive_noise_intervention": False,
            "c2": False,
        },
        "stop_rules": [
            "fail closed on any identity, source SHA, processor, shape, finiteness, or dtype/device precondition",
            "stop after at most 18 direct teacher-forced flow probes",
            "do not call official free-running sampler or complete model forward",
            "do not run training, backward, optimizer, PBS, WBC, hardware, validation e, holdout f, or C2",
            "return paired evidence to GPT/PI; do not select H2 or interpret scientific outcome in runner",
        ],
    }


def assert_protocol_static(plan: Mapping[str, Any] | None = None) -> None:
    candidate = protocol_plan() if plan is None else plan
    if candidate.get("schema") != STAGE26_SCHEMA:
        raise AssertionError("Stage26 schema drifted")
    if candidate.get("status") != "PREPARED_NOT_EXECUTED":
        raise AssertionError("Stage26 preparation must remain non-executing")
    boundary = candidate.get("execution_boundary", {})
    if any(bool(value) for value in boundary.values()):
        raise AssertionError("Stage26 preparation contains an enabled execution boundary")
    if candidate.get("base_sha") != STAGE26_BASE_SHA:
        raise AssertionError("Stage26 base SHA drifted")
    fixed_identity = candidate.get("fixed_identity", {})
    expected_identity = {
        "base_index_sha256": EXPECTED_BASE_INDEX_SHA256,
        "base_tree_sha256": EXPECTED_BASE_TREE_SHA256,
        "base_tree_prefix": EXPECTED_BASE_TREE_PREFIX,
        "o1_index_sha256": EXPECTED_O1_INDEX_SHA256,
        "o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
        "o1_supplemental_metadata_allowlist": list(
            EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST
        ),
    }
    for key, expected_value in expected_identity.items():
        if fixed_identity.get(key) != expected_value:
            raise AssertionError(f"Stage26 fixed identity drifted: {key}")
    budget = candidate.get("execution_budget", {})
    if budget.get("maximum_c1_flow_evaluations") != MAX_FLOW_EVALUATIONS:
        raise AssertionError("Stage26 flow-evaluation budget drifted")
    readiness = candidate.get("readiness", {})
    if readiness.get("h1_c1_execution_ready") is not True:
        raise AssertionError("Stage26 H1 C1 readiness was not explicitly established")
    if readiness.get("runtime_identity_revalidation_required") is not True:
        raise AssertionError("Stage26 runtime identity revalidation gate drifted")
    if readiness.get("execution_started") is not False:
        raise AssertionError("Stage26 preparation must remain non-executing")
    probes = candidate.get("probe_contract", {}).get("probes", [])
    if len(probes) != MAX_PROBES:
        raise AssertionError("Stage26 requires exactly nine probes")
    mask = candidate.get("probe_contract", {}).get("mask_partition", {})
    if mask.get("active_dimensions") != ACTIVE_DIM or mask.get("inactive_dimensions") != INACTIVE_DIM:
        raise AssertionError("Stage26 active/inactive partition drifted")
    if mask.get("inactive_dimensions_contribute_to_loss") is not False:
        raise AssertionError("Stage26 inactive mask contract drifted")
    output = candidate.get("output_schema", {})
    if output.get("no_new_scientific_pass_threshold") is not True:
        raise AssertionError("Stage26 must not invent a scientific pass threshold")
    if candidate.get("runtime_precision", {}).get("device_cast") != "model.to(device=args.device, dtype=torch.bfloat16)":
        raise AssertionError("Stage26 inference precision path drifted")
def _sha256_tensor_float32(value: Any) -> str:
    """Hash a tensor after the declared float32 canonicalization boundary."""

    tensor = value.detach().to(device="cpu")
    if str(tensor.dtype) != "torch.float32":
        tensor = tensor.float()
    return _sha256_bytes(tensor.contiguous().numpy().tobytes())


def _tensor_to_numpy_float32(value: Any) -> Any:
    tensor = value.detach().to(device="cpu")
    return tensor.float().contiguous().numpy().astype("float32", copy=False)


def _finite_tensor(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all().item())


def _build_runtime_processor(
    *,
    groot_source: Path,
    base_checkpoint: Path,
    dataset_dir: Path,
    modality_path: Path,
    cosmos_snapshot: Path,
    expected_processor_config_sha256: str = EXPECTED_BASE_PROCESSOR_CONFIG_SHA256,
    expected_statistics_sha256: str = EXPECTED_BASE_STATISTICS_SHA256,
) -> tuple[Any, Any, Any]:
    """Build the exact custom processor used by the Stage22 training path."""

    if _sha256_file(modality_path) != EXPECTED_MODALITY_SHA256:
        raise ValueError("modality config SHA256 does not match the Stage22 frozen payload")
    if str(groot_source) not in sys.path:
        sys.path.insert(0, str(groot_source))
    sys.path.insert(0, str(modality_path.parent))

    modality_module_name = f"stage26_modality_{_sha256_file(modality_path)[:12]}"
    spec = importlib.util.spec_from_file_location(modality_module_name, modality_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load modality config: {modality_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modality_module_name] = module
    spec.loader.exec_module(module)
    modality = module.GO2_ARX5_MODALITY_CONFIG

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_config_path = base_checkpoint / "processor_config.json"
    statistics_path = base_checkpoint / "statistics.json"
    if _sha256_file(processor_config_path) != expected_processor_config_sha256:
        raise ValueError("BASE processor_config.json SHA256 mismatch")
    if _sha256_file(statistics_path) != expected_statistics_sha256:
        raise ValueError("BASE statistics.json SHA256 mismatch")
    flat_stats_path = dataset_dir / "meta" / "stats.json"
    relative_stats_path = dataset_dir / "meta" / "relative_stats.json"
    processor_config = _read_json(processor_config_path)
    model_statistics = _read_json(statistics_path)
    flat_stats = _read_json(flat_stats_path)
    relative_stats = _read_json(relative_stats_path)

    def split_stats(values: Mapping[str, Sequence[float]]) -> dict[str, dict[str, list[float]]]:
        keys = ("min", "max", "mean", "std", "q01", "q99")
        return {
            "eef_9d": {key: list(values[key][:9]) for key in keys},
            "gripper_position": {key: list(values[key][9:10]) for key in keys},
        }

    custom_statistics = {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "state": split_stats(flat_stats["observation.state"]),
            "action": split_stats(flat_stats["action"]),
            "relative_action": {"eef_9d": relative_stats["eef_9d"]},
        }
    }
    kwargs = copy.deepcopy(processor_config["processor_kwargs"])
    kwargs["modality_configs"][EmbodimentTag.NEW_EMBODIMENT.value] = modality
    kwargs["statistics"] = {**model_statistics, **custom_statistics}
    kwargs["model_name"] = str(cosmos_snapshot)
    kwargs["model_type"] = "qwen"
    kwargs["transformers_loading_kwargs"] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    processor = Gr00tN1d7Processor(**kwargs)
    processor.eval()
    return processor, EmbodimentTag.NEW_EMBODIMENT, modality


def _verify_cosmos_cache(cache_root: Path) -> dict[str, Any]:
    """Verify the pinned cache tree using the Stage22 supplement algorithm."""

    root = Path(cache_root)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Cosmos cache root unavailable: {root}")
    entries: list[dict[str, Any]] = []
    digest_cache: dict[str, str] = {}

    def file_sha(path: Path) -> str:
        key = str(path)
        if key not in digest_cache:
            digest_cache[key] = _sha256_file(path)
        return digest_cache[key]

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(os.scandir(directory), key=lambda value: value.name):
            path = Path(item.path)
            rel = (relative / item.name).as_posix()
            mode = os.lstat(path).st_mode
            if stat.S_ISDIR(mode):
                entries.append(
                    {"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None}
                )
                visit(path, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(path)
                resolved = path.resolve(strict=True)
                try:
                    resolved_rel = resolved.relative_to(root.resolve(strict=True)).as_posix()
                except ValueError as exc:
                    raise ValueError(f"Cosmos cache symlink escapes root: {path}") from exc
                if not resolved.is_file():
                    raise ValueError(f"Cosmos cache symlink does not resolve to a file: {path}")
                entries.append(
                    {
                        "path": rel,
                        "kind": "symlink",
                        "target": target,
                        "resolved_path": resolved_rel,
                        "size_bytes": int(resolved.stat().st_size),
                        "sha256": file_sha(resolved),
                    }
                )
            elif stat.S_ISREG(mode):
                entries.append(
                    {
                        "path": rel,
                        "kind": "regular_file",
                        "size_bytes": int(path.stat().st_size),
                        "sha256": file_sha(path),
                    }
                )
            else:
                raise ValueError(f"unsupported Cosmos cache entry: {path}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: value["path"]):
        line = "\t".join(
            (
                str(entry["kind"]),
                str(entry["path"]),
                str(entry.get("size_bytes", 0)),
                str(entry.get("sha256") or "-"),
                str(entry.get("target") or "-"),
                str(entry.get("resolved_path") or "-"),
            )
        )
        digest.update((line + "\n").encode("utf-8"))
    observed = digest.hexdigest()
    if observed != EXPECTED_COSMOS_TREE_SHA256:
        raise ValueError(f"Cosmos cache tree SHA mismatch: {observed}")
    refs_main = root / "refs" / "main"
    if (
        not refs_main.is_file()
        or refs_main.read_text(encoding="utf-8").strip() != EXPECTED_COSMOS_REVISION
    ):
        raise ValueError("Cosmos cache refs/main does not match the pinned revision")
    snapshot = root / "snapshots" / EXPECTED_COSMOS_REVISION
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("pinned Cosmos snapshot directory is unavailable or symlinked")
    return {
        "cache_root": str(root),
        "tree_sha256": observed,
        "revision": EXPECTED_COSMOS_REVISION,
        "refs_main": str(refs_main),
        "snapshot": str(snapshot),
        "entry_count": len(entries),
    }


def _prepare_official_sample(
    *,
    processor: Any,
    embodiment: Any,
    modality_configs: dict[str, Any],
    dataset_dir: Path,
) -> dict[str, Any]:
    """Run only the official loader/processor boundary for one train sample."""

    import torch

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType

    loader = LeRobotEpisodeLoader(dataset_dir, modality_configs)
    if len(loader) != 1 or loader.episode_lengths != [HORIZON]:
        raise ValueError(f"TINY1 official loader identity drifted: {loader.episode_lengths}")
    episode = loader[0]
    step_data = extract_step_data(
        episode,
        0,
        modality_configs,
        embodiment,
        allow_padding=False,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])
    collated = processor.collator([processed])
    model_inputs = collated["inputs"]
    required = {"action", "action_mask", "state", "embodiment_id"}
    missing = sorted(required.difference(model_inputs))
    if missing:
        raise ValueError(f"official processor output is missing: {missing}")
    action = model_inputs["action"]
    if tuple(int(value) for value in action.shape) != (1, HORIZON, UNIVERSAL_DIM):
        raise ValueError(f"official normalized action shape drifted: {tuple(action.shape)}")
    if not _finite_tensor(action):
        raise ValueError("official normalized action is non-finite")
    action_mask = model_inputs["action_mask"]
    if tuple(int(value) for value in action_mask.shape) != (1, HORIZON, UNIVERSAL_DIM):
        raise ValueError(f"official action mask shape drifted: {tuple(action_mask.shape)}")
    if not bool((action_mask[..., :ACTIVE_DIM] == 1).all().item()):
        raise ValueError("official active action mask is not one")
    if not bool((action_mask[..., ACTIVE_DIM:] == 0).all().item()):
        raise ValueError("official inactive action mask is not zero")

    action_cpu = action.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "loader": loader,
        "episode": episode,
        "step_data": step_data,
        "processed": processed,
        "collated": collated,
        "model_inputs": model_inputs,
        "expert_action_cpu": action_cpu,
        "expert_action_sha256": _sha256_tensor_float32(action_cpu),
        "action_mask_sha256": _sha256_tensor_float32(action_mask),
    }


def _canonical_probe_inputs(
    stage23: Any, expert_action_cpu: Any, probe: Mapping[str, Any]
) -> dict[str, Any]:
    """Generate one canonical float32 CPU probe, once for both checkpoints."""

    fixed = stage23.build_fixed_flow_inputs(
        expert_action_cpu,
        noise_seed=int(probe["noise_seed"]),
        t_value=float(probe["t_value"]),
    )
    for key in ("noise", "noisy_action", "target_velocity", "action_mask", "t"):
        value = fixed[key]
        if str(value.device) != "cpu" or str(value.dtype) != "torch.float32":
            raise RuntimeError(f"canonical probe {key} did not remain CPU float32")
    return fixed


def _copy_to_model_dtype(value: Any, *, device: Any, dtype: Any) -> Any:
    import torch

    if torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _teacher_forced_local_flow(
    *,
    action_head: Any,
    backbone_output: Any,
    action_input: Any,
    noisy_action: Any,
    t_discretized: Any,
    on_before_direct_probe_forward: Callable[[], None] | None = None,
) -> Any:
    """Mirror only the pinned training graph after fixed inputs are supplied.

    Source mapping:
      * lines 209-217: state reshape and state encoder;
      * lines 237-239: action encoder receives fixed noisy action/time;
      * lines 241-270: optional position embedding, fusion, and DiT;
      * lines 272-278: decoder and action-slice velocity output.

    There is deliberately no call to the complete model forward and no call
    to the action-head free-running sampler here.
    """

    import torch

    if tuple(int(value) for value in noisy_action.shape)[1:] != (HORIZON, UNIVERSAL_DIM):
        raise ValueError(f"noisy action shape drifted: {tuple(noisy_action.shape)}")
    action_head.set_frozen_modules_to_eval_mode()
    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    # Official gr00t_n1d7.py lines 209-217.
    if int(action_input.state.shape[1]) != int(action_head.config.state_history_length):
        raise ValueError("state history length differs from the pinned action-head config")
    state = action_input.state.view(action_input.state.shape[0], 1, -1)
    state_features = action_head.state_encoder(state, embodiment_id)

    # Official gr00t_n1d7.py lines 237-239, with fixed C1 inputs.
    action_features = action_head.action_encoder(noisy_action, t_discretized, embodiment_id)

    # Official gr00t_n1d7.py lines 241-270.
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs
    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask
    if on_before_direct_probe_forward is not None:
        on_before_direct_probe_forward()
    if action_head.config.use_alternate_vl_dit:
        model_output, _ = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output, _ = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )

    # Official gr00t_n1d7.py lines 272-278.
    pred = action_head.action_decoder(model_output, embodiment_id)
    predicted_velocity = pred[:, -noisy_action.shape[1] :]
    if tuple(int(value) for value in predicted_velocity.shape) != (1, HORIZON, UNIVERSAL_DIM):
        raise ValueError(f"predicted velocity shape drifted: {tuple(predicted_velocity.shape)}")
    if not _finite_tensor(predicted_velocity):
        raise FloatingPointError("predicted velocity contains NaN or Inf")
    return predicted_velocity


def _flow_metrics(stage23: Any, predicted: Any, fixed: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    predicted_f32 = predicted.detach().to(device="cpu", dtype=torch.float32)
    metrics = stage23.masked_flow_metrics(
        predicted_f32,
        fixed["target_velocity"],
        fixed["action_mask"],
    )
    return {
        **_jsonable(metrics),
        "predicted_dtype_at_metric_boundary": str(predicted_f32.dtype),
        "target_dtype": str(fixed["target_velocity"].dtype),
    }


def _load_stage25_identity_module() -> Any:
    """Load the dependency-free Stage25 identity gates without model imports."""

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir.parent) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent))
    return importlib.import_module("scripts.stage25_o1_parameter_delta_audit")


def _checkpoint_preflight(
    label: str,
    checkpoint: Path,
    *,
    o1_seal: Path | None = None,
) -> dict[str, Any]:
    """Verify index plus full checkpoint bytes before model construction.

    BASE reuses the Stage25 canonical full-tree digest. O1 reuses the
    Stage25 immutable seal and two-layer scope verifier, including the exact
    five supplemental historical byte/SHA bindings. These checks deliberately
    happen before any model, Torch tensor backend, or CUDA import.
    """

    expected_index = {
        "BASE": EXPECTED_BASE_INDEX_SHA256,
        "O1@200": EXPECTED_O1_INDEX_SHA256,
    }[label]
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FileNotFoundError(
            f"{label} checkpoint directory is unavailable or symlinked: {checkpoint}"
        )
    index_paths = sorted(checkpoint.glob("*.safetensors.index.json"))
    if len(index_paths) != 1:
        raise ValueError(f"{label} checkpoint must contain exactly one safetensors index")
    index_path = index_paths[0]
    _assert_regular(index_path, f"{label} checkpoint index")
    observed_index = _sha256_file(index_path)
    if observed_index != expected_index:
        raise ValueError(f"{label} checkpoint index SHA mismatch: {observed_index}")
    index = _read_json(index_path)
    if not isinstance(index, Mapping) or not isinstance(index.get("weight_map"), Mapping):
        raise ValueError(f"{label} checkpoint index has no weight_map")
    referenced_shards = sorted({str(value) for value in index["weight_map"].values()})
    if not referenced_shards or any(
        Path(name).is_absolute() or ".." in Path(name).parts
        for name in referenced_shards
    ):
        raise ValueError(f"{label} checkpoint index has unsafe shard references")
    for shard_name in referenced_shards:
        shard_path = checkpoint / Path(shard_name)
        _assert_regular(shard_path, f"{label} referenced shard")

    stage25 = _load_stage25_identity_module()
    if (
        stage25.EXPECTED_BASE_INDEX_SHA256 != EXPECTED_BASE_INDEX_SHA256
        or stage25.EXPECTED_BASE_TREE_SHA256 != EXPECTED_BASE_TREE_SHA256
        or stage25.EXPECTED_O1_INDEX_SHA256 != EXPECTED_O1_INDEX_SHA256
        or stage25.EXPECTED_O1_SEAL_SHA256 != EXPECTED_O1_SEAL_SHA256
        or tuple(stage25.SUPPLEMENTAL_METADATA_ALLOWLIST)
        != EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST
    ):
        raise RuntimeError("Stage25 identity constants drifted from Stage26 contract")
    identity: dict[str, Any] = {
        "label": label,
        "path": str(checkpoint),
        "index_path": str(index_path),
        "index_sha256": observed_index,
        "index_key_count": len(index["weight_map"]),
        "referenced_shards": referenced_shards,
        "tensor_reads_before_model_load": False,
        "full_byte_identity_verified_before_model_load": False,
    }
    if label == "BASE":
        entries = stage25.walk_regular_files(checkpoint)
        observed_tree = stage25.stage24_tree_sha256(entries, EXPECTED_BASE_TREE_PREFIX)
        if observed_tree != EXPECTED_BASE_TREE_SHA256:
            raise ValueError(f"BASE canonical tree SHA mismatch: {observed_tree}")
        referenced_shard_set = set(referenced_shards)
        referenced_shard_manifest = [
            {
                "path": entry["path"],
                "bytes": int(entry["bytes"]),
                "sha256": entry["sha256"],
            }
            for entry in entries
            if entry["path"] in referenced_shard_set
        ]
        if {entry["path"] for entry in referenced_shard_manifest} != referenced_shard_set:
            raise ValueError("BASE canonical tree does not contain every indexed shard")
        identity.update(
            {
                "tree_sha256": observed_tree,
                "tree_prefix": EXPECTED_BASE_TREE_PREFIX,
                "tree_file_count": len(entries),
                "tree_total_bytes": sum(int(entry["bytes"]) for entry in entries),
                "referenced_shard_manifest": referenced_shard_manifest,
                "identity_gate": "STAGE25_BASE_CANONICAL_TREE",
                "full_byte_identity_verified_before_model_load": True,
            }
        )
    else:
        if o1_seal is None:
            raise ValueError("O1 preflight requires the immutable Stage23 checkpoint seal path")
        o1_seal = Path(o1_seal)
        if o1_seal.is_symlink() or not o1_seal.is_file():
            raise FileNotFoundError(
                f"O1 immutable checkpoint seal is unavailable or symlinked: {o1_seal}"
            )
        seal_result = stage25._verify_o1_seal_against_checkpoint(checkpoint, o1_seal)
        if not seal_result["pass"]:
            errors = seal_result.get("errors") or ["O1 two-layer provenance mismatch"]
            raise ValueError("O1 two-layer provenance mismatch: " + "; ".join(errors[:4]))
        scope = seal_result.get("scope") or {}
        supplemental = scope.get("allowed_supplemental_metadata") or {}
        identity.update(
            {
                "o1_seal_path": str(o1_seal),
                "o1_seal_sha256": seal_result["observed_o1_seal_sha256"],
                "o1_seal_expected_sha256": EXPECTED_O1_SEAL_SHA256,
                "o1_seal_verified": bool(seal_result["verified"]),
                "o1_sealed_core_file_count": seal_result["expected_file_count"],
                "o1_actual_file_count": seal_result["actual_file_count"],
                "o1_supplemental_metadata_manifest_sha256": supplemental.get(
                    "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256"
                ),
                "o1_supplemental_history_status": supplemental.get(
                    "SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS"
                ),
                "o1_supplemental_history_verified": bool(
                    supplemental.get("SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS")
                ),
                "o1_two_layer_scope": scope,
                "identity_gate": "STAGE25_O1_IMMUTABLE_CORE_PLUS_FIVE_HISTORICAL_SUPPLEMENTALS",
                "full_byte_identity_verified_before_model_load": True,
            }
        )
    return identity


def _profile_result(
    *,
    decoded: Any,
    snapshot: Any,
    model_checkpoint: Path,
    source_label: str,
) -> dict[str, Any]:
    """Decode/profile without WBC using the existing project contracts."""

    import numpy as np

    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.metrics.action_profile import chunk_profile
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    decoded_array = np.asarray(decoded, dtype=np.float32)
    if decoded_array.shape != (1, HORIZON, ACTIVE_DIM) or not np.isfinite(decoded_array).all():
        raise ValueError(f"decoded physical action shape/finite contract failed: {decoded_array.shape}")
    state = snapshot.state_v1.as_array()
    provider = GR00TTrajectoryProvider(
        decoded_array,
        config=GR00TProviderConfig(
            source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
            tcp_semantic_id="x5_link6_gripper_center",
            action_sample_period_s=ACTION_PERIOD_S,
            timestamp_provenance="TEST_ONLY_STAGE22_SIMULATION",
            source="GR00T_N1.7_STAGE26_H1_C1",
        ),
        point_timestamps=snapshot.future_timestamps.copy(),
        chunk_id=f"{source_label}:{snapshot.example_id}",
        observation_timestamp=float(snapshot.observation_timestamp),
        provenance={
            "stage": "Stage26",
            "model_checkpoint": str(model_checkpoint),
            "source_snapshot": snapshot.example_id,
            "trajectory_correction": False,
            "wbc_executed": False,
        },
    )
    prediction = provider.next_chunk(float(snapshot.observation_timestamp))
    if prediction is None:
        raise RuntimeError("GR00T provider emitted no C1 profile chunk")
    matrices = rot6d_rows_to_matrix(snapshot.future_desired_eef_9d[:, 3:9])
    label = TrajectoryChunkV1(
        chunk_id=f"expert-label:{snapshot.example_id}",
        source="STAGE22_SAME_SCENE_EXPERT_LABEL",
        observation_timestamp=float(snapshot.observation_timestamp),
        point_timestamps=snapshot.future_timestamps.copy(),
        eef_position=snapshot.future_desired_eef_9d[:, :3].copy(),
        eef_orientation=matrix_to_quat(matrices),
        eef_rot6d=snapshot.future_desired_eef_9d[:, 3:9].copy(),
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        validity_start=float(snapshot.future_timestamps[0]),
        validity_end=float(snapshot.future_timestamps[-1]),
        gripper=snapshot.future_gripper.reshape(-1).copy(),
        provenance={"stage": "Stage22", "target_excluded_from_model_input": True},
    )
    achieved_position = state[:3]
    state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    pred_profile = chunk_profile(
        prediction,
        achieved_position=achieved_position,
        achieved_quaternion=state_quat,
        provenance={"profile_role": "model_implied_action", "source_label": source_label},
    )
    label_profile = chunk_profile(label, provenance={"profile_role": "expert_label"})
    gate_result = TrajectorySafetyGateV1().evaluate(
        prediction,
        now=float(snapshot.observation_timestamp),
        achieved_position=achieved_position,
        achieved_orientation=state_quat,
    )
    label_length = float(label_profile["path_length_m"])
    label_speed = float(label_profile["linear_speed_mps"]["mean"] or 0.0)
    label_max_speed = float(label_profile["linear_speed_mps"]["max"] or 0.0)
    pred_length = float(pred_profile["path_length_m"])
    pred_speed = float(pred_profile["linear_speed_mps"]["mean"] or 0.0)
    pred_max_speed = float(pred_profile["linear_speed_mps"]["max"] or 0.0)
    return {
        "decoded_shape": list(decoded_array.shape),
        "decoded_physical_action_sha256": _sha256_bytes(
            np.ascontiguousarray(decoded_array).tobytes()
        ),
        "decoded_physical_action": decoded_array,
        "prediction_profile": pred_profile,
        "expert_label_profile": label_profile,
        "profile_ratios": {
            "path_length_ratio_to_label": None
            if label_length <= 1e-12
            else pred_length / label_length,
            "mean_speed_ratio_to_label": None
            if label_speed <= 1e-12
            else pred_speed / label_speed,
            "max_speed_ratio_to_label": None
            if label_max_speed <= 1e-12
            else pred_max_speed / label_max_speed,
        },
        "safety_gate": gate_result.as_dict(),
        "source_frame": prediction.source_frame,
        "tcp_semantic_id": prediction.tcp_semantic_id,
        "action_period_s": ACTION_PERIOD_S,
        "action_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
        "wbc_executed": False,
    }


def _run_one_checkpoint(
    *,
    checkpoint_label: str,
    checkpoint: Path,
    groot_source: Path,
    prepared: Mapping[str, Any],
    processor: Any,
    embodiment: Any,
    stage23: Any,
    probes: Sequence[Mapping[str, Any]],
    fixed_probes: Sequence[Mapping[str, Any]],
    device: str,
    snapshot: Any,
    on_before_direct_probe_forward: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    on_record: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Load one checkpoint and perform exactly nine direct C1 probes."""

    import numpy as np
    import torch

    if not str(device).startswith("cuda"):
        raise RuntimeError("Stage26 execution requires an explicitly selected CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage26 execution requested CUDA but CUDA is unavailable")
    sys.path.insert(0, str(groot_source))
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    # Model construction is permitted only in --execute, after identity gates.
    # The dtype cast is the exact established Stage22/23 inference path.
    torch.manual_seed(TRAIN_SEED)
    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    model_inputs = prepared["model_inputs"]
    backbone_inputs, action_inputs = model.prepare_input(dict(model_inputs))
    with torch.inference_mode():
        backbone_output = model.backbone(backbone_inputs)
        processed_backbone = model.action_head.process_backbone_output(backbone_output)

    records: list[dict[str, Any]] = []
    model_dtype = next(iter(model.parameters())).dtype
    if len(fixed_probes) != len(probes):
        raise ValueError("fixed probe count does not match the frozen probe contract")
    for probe, fixed in zip(probes, fixed_probes):
        if int(fixed["noise_seed"]) != int(probe["noise_seed"]):
            raise ValueError(f"fixed noise seed drifted for {probe['probe_id']}")
        if int(fixed["t_discretized"].reshape(-1)[0].item()) != int(probe["t_discretized"]):
            raise ValueError(f"fixed t bucket drifted for {probe['probe_id']}")
        model_device = processed_backbone.backbone_features.device
        noisy_action = _copy_to_model_dtype(
            fixed["noisy_action"],
            device=model_device,
            dtype=model_dtype,
        )
        t_discretized = fixed["t_discretized"].to(device=model_device)
        start = time.perf_counter()
        with torch.inference_mode():
            predicted_velocity = _teacher_forced_local_flow(
                action_head=model.action_head,
                backbone_output=processed_backbone,
                action_input=action_inputs,
                noisy_action=noisy_action,
                t_discretized=t_discretized,
                on_before_direct_probe_forward=(
                    None
                    if on_before_direct_probe_forward is None
                    else lambda: on_before_direct_probe_forward(checkpoint_label, probe)
                ),
            )
        elapsed = time.perf_counter() - start

        metrics = _flow_metrics(stage23, predicted_velocity, fixed)
        predicted_cpu = predicted_velocity.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        implied = stage23.implied_action_from_velocity(
            fixed["noise"], predicted_cpu
        )
        active = stage23.extract_active_action_for_official_decode(implied)
        batched_state = {
            "eef_9d": np.asarray(
                snapshot.state_v1.as_array()[None, None, :9], dtype=np.float32
            ),
            "gripper_position": np.asarray(
                snapshot.state_v1.as_array()[None, None, 9:], dtype=np.float32
            ),
        }
        decoded_groups = processor.decode_action(
            _tensor_to_numpy_float32(implied),
            embodiment,
            batched_state,
        )
        decoded = np.concatenate(
            [
                np.asarray(decoded_groups["eef_9d"]),
                np.asarray(decoded_groups["gripper_position"]),
            ],
            axis=-1,
        ).astype(np.float32)
        profile = _profile_result(
            decoded=decoded,
            snapshot=snapshot,
            model_checkpoint=checkpoint,
            source_label=checkpoint_label,
        )
        decoded_hash = profile["decoded_physical_action_sha256"]
        profile_public = dict(profile)
        profile_public.pop("decoded_physical_action", None)
        row = {
            "checkpoint": checkpoint_label,
            "checkpoint_path": str(checkpoint),
            "probe_id": str(probe["probe_id"]),
            "noise_seed": int(probe["noise_seed"]),
            "t_quantile": float(probe["t_quantile"]),
            "t_value": float(probe["t_value"]),
            "t_discretized": int(probe["t_discretized"]),
            "noise_shape": [1, HORIZON, UNIVERSAL_DIM],
            "noise_sha256": _sha256_tensor_float32(fixed["noise"]),
            "noisy_action_sha256": _sha256_tensor_float32(fixed["noisy_action"]),
            "target_velocity_sha256": _sha256_tensor_float32(fixed["target_velocity"]),
            "model_noisy_action_dtype": str(noisy_action.dtype),
            "model_noisy_action_sha256": _sha256_tensor_float32(noisy_action),
            "raw_predicted_velocity_sha256": _sha256_tensor_float32(predicted_cpu),
            "raw_predicted_velocity_shape": list(predicted_cpu.shape),
            "raw_predicted_velocity_dtype_at_hash_boundary": str(predicted_cpu.dtype),
            "implied_universal_action_sha256": _sha256_tensor_float32(implied),
            "active_model_action_sha256": _sha256_tensor_float32(active),
            "flow_metrics": metrics,
            "profile": profile_public,
            "finite_flags": {
                "canonical_noise": _finite_tensor(fixed["noise"]),
                "canonical_noisy_action": _finite_tensor(fixed["noisy_action"]),
                "canonical_target_velocity": _finite_tensor(fixed["target_velocity"]),
                "raw_predicted_velocity": _finite_tensor(predicted_cpu),
                "implied_universal_action": _finite_tensor(implied),
                "decoded_physical_action": bool(np.isfinite(decoded).all()),
            },
            "elapsed_s": float(elapsed),
            "forward_scope": "teacher_forced_action_head_only_after_one_shared_backbone_pass",
            "iterative_sampler": False,
            "training": False,
            "backward": False,
            "wbc_executed": False,
            "decoded_physical_action_sha256": decoded_hash,
        }
        records.append(row)
        if on_record is not None:
            on_record(row)
    del model
    torch.cuda.empty_cache()
    return records


def _paired_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_probe: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_probe.setdefault(str(row["probe_id"]), {})[str(row["checkpoint"])] = row
    expected_probe_ids = {
        str(item["probe_id"]) for item in _static_probe_contract()["probes"]
    }
    if set(by_probe) != expected_probe_ids:
        raise ValueError("paired aggregate probe set is not exactly the nine frozen probes")

    paired: list[dict[str, Any]] = []
    ratios: list[float] = []
    improved = 0
    for probe_id in sorted(by_probe):
        pair = by_probe[probe_id]
        if set(pair) != {"BASE", "O1@200"}:
            raise ValueError(f"incomplete checkpoint pair for {probe_id}: {sorted(pair)}")
        base_mse = float(pair["BASE"]["flow_metrics"]["active_masked_mse"])
        o1_mse = float(pair["O1@200"]["flow_metrics"]["active_masked_mse"])
        ratio = None if base_mse <= 0.0 else o1_mse / base_mse
        if ratio is not None and math.isfinite(ratio):
            ratios.append(ratio)
        if o1_mse < base_mse:
            improved += 1
        paired.append(
            {
                "probe_id": probe_id,
                "base_active_masked_flow_mse": base_mse,
                "o1_active_masked_flow_mse": o1_mse,
                "o1_over_base_ratio": ratio,
                "o1_improved": bool(o1_mse < base_mse),
                "base_noise_sha256": pair["BASE"]["noise_sha256"],
                "o1_noise_sha256": pair["O1@200"]["noise_sha256"],
                "same_canonical_noise": pair["BASE"]["noise_sha256"]
                == pair["O1@200"]["noise_sha256"],
                "base_t_discretized": pair["BASE"]["t_discretized"],
                "o1_t_discretized": pair["O1@200"]["t_discretized"],
                "same_t": pair["BASE"]["t_discretized"]
                == pair["O1@200"]["t_discretized"],
                "safety_gate_transition": {
                    "base": pair["BASE"]["profile"]["safety_gate"]["decision"],
                    "o1": pair["O1@200"]["profile"]["safety_gate"]["decision"],
                },
                "profile_ratio_transition": {
                    "base": pair["BASE"]["profile"]["profile_ratios"],
                    "o1": pair["O1@200"]["profile"]["profile_ratios"],
                },
            }
        )

    geomean = None
    if ratios:
        geomean = 0.0 if any(value == 0.0 for value in ratios) else float(geometric_mean(ratios))
    return {
        "probe_count": len(paired),
        "paired_by_probe": paired,
        "base_mse": [item["base_active_masked_flow_mse"] for item in paired],
        "o1_mse": [item["o1_active_masked_flow_mse"] for item in paired],
        "o1_over_base_ratio": [item["o1_over_base_ratio"] for item in paired],
        "improved_count_over_9": improved,
        "median_ratio": None if not ratios else float(median(ratios)),
        "geometric_mean_ratio": geomean,
        "descriptive_only": True,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
    }


def _load_snapshot(path: Path) -> Any:
    if str(Path(__file__).resolve().parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from go2arx5_vla.data.stage22_expert import load_snapshot

    return load_snapshot(path)


def _verify_groot_source(groot_source: Path) -> str:
    if groot_source.is_symlink() or not groot_source.is_dir():
        raise FileNotFoundError(f"GR00T source directory is unavailable or symlinked: {groot_source}")
    observed = subprocess.check_output(
        ["git", "-C", str(groot_source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed != EXPECTED_GROOT_SHA:
        raise ValueError(f"pinned GR00T source SHA mismatch: {observed}")
    required = {
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
            "process_backbone_output",
            "self.state_encoder",
            "self.action_encoder",
            "return_all_hidden_states=True",
            "self.action_decoder",
            "pred[:, -actions.shape[1]",
        ),
        "gr00t/model/gr00t_n1d7/setup.py": (
            "AutoModel.from_pretrained",
            "load_bf16",
        ),
    }
    for relative, anchors in required.items():
        path = groot_source / relative
        _assert_regular(path, "GR00T source file")
        source_text = path.read_text(encoding="utf-8")
        for anchor in anchors:
            if anchor not in source_text:
                raise ValueError(f"pinned source anchor missing: {relative}: {anchor}")
    return observed


def _decode_physical_action(processor: Any, embodiment: Any, universal: Any, snapshot: Any) -> Any:
    import numpy as np

    state = snapshot.state_v1.as_array()
    decoded_groups = processor.decode_action(
        np.asarray(universal, dtype=np.float32),
        embodiment,
        {
            "eef_9d": np.asarray(state[None, None, :9], dtype=np.float32),
            "gripper_position": np.asarray(state[None, None, 9:], dtype=np.float32),
        },
    )
    decoded = np.concatenate(
        [
            np.asarray(decoded_groups["eef_9d"]),
            np.asarray(decoded_groups["gripper_position"]),
        ],
        axis=-1,
    ).astype(np.float32)
    if decoded.shape != (1, HORIZON, ACTIVE_DIM) or not np.isfinite(decoded).all():
        raise ValueError(f"decoded physical action shape/finite contract failed: {decoded.shape}")
    return decoded


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the conditional Hopper-only forward audit after all gates."""

    if args.device is None or not str(args.device).startswith("cuda"):
        raise RuntimeError("execute requires --device cuda:<index>")
    if args.base_checkpoint.resolve() == args.o1_checkpoint.resolve():
        raise ValueError("BASE and O1 checkpoint paths must be distinct")

    # These checks are deliberately ordered before the first model import.
    base_identity = _checkpoint_preflight("BASE", args.base_checkpoint)
    o1_identity = _checkpoint_preflight(
        "O1@200",
        args.o1_checkpoint,
        o1_seal=args.o1_seal,
    )
    dataset_identity = _verify_tiny_dataset(args.dataset_dir)
    snapshot_identity, snapshot_path = _verify_snapshot_identity(args.snapshot_root)
    source_repo_sha = _verify_groot_source(args.groot_source)
    cosmos_identity = _verify_cosmos_cache(args.cosmos_cache_root)
    stage23 = _load_stage23_protocol()
    probe_contract = _static_probe_contract()
    stage23.assert_protocol_static(stage23.protocol_plan())

    snapshot = _load_snapshot(snapshot_path)
    if snapshot.example_id != f"{TARGET_SOURCE_EPISODE}:expert-0000":
        raise ValueError("loaded source snapshot example_id changed after static identity check")
    if snapshot.source_frame != "NATIVE_WBC_RELATIVE_REFERENCE":
        raise ValueError(f"unexpected source frame: {snapshot.source_frame}")
    if snapshot.tcp_semantic_id != "x5_link6_gripper_center":
        raise ValueError(f"unexpected TCP semantic ID: {snapshot.tcp_semantic_id}")

    processor, embodiment, modality_configs = _build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=args.base_checkpoint,
        dataset_dir=args.dataset_dir,
        modality_path=args.modality_config,
        cosmos_snapshot=Path(cosmos_identity["snapshot"]),
    )
    prepared = _prepare_official_sample(
        processor=processor,
        embodiment=embodiment,
        modality_configs=copy.deepcopy(modality_configs),
        dataset_dir=args.dataset_dir,
    )
    # Bind the official normalized sample to the exact source snapshot label
    # before constructing either checkpoint model.
    expert_decoded = _decode_physical_action(
        processor,
        embodiment,
        _tensor_to_numpy_float32(prepared["expert_action_cpu"]),
        snapshot,
    )
    import numpy as np

    expected_physical = np.asarray(snapshot.physical_action, dtype=np.float32)[None, ...]
    if not np.allclose(expert_decoded, expected_physical, atol=1e-5, rtol=0.0):
        raise ValueError("TINY1 processor action does not round-trip to the source snapshot label")

    probes = probe_contract["probes"]
    # Generate the nine CPU float32 probes exactly once. The same tensor
    # objects are passed to both checkpoint runs; only the declared model
    # dtype/device transfer occurs at the action-head boundary.
    fixed_probes = tuple(
        _canonical_probe_inputs(stage23, prepared["expert_action_cpu"], probe)
        for probe in probes
    )
    records: list[dict[str, Any]] = []
    records.extend(
        _run_one_checkpoint(
            checkpoint_label="BASE",
            checkpoint=args.base_checkpoint,
            groot_source=args.groot_source,
            prepared=prepared,
            processor=processor,
            embodiment=embodiment,
            stage23=stage23,
            probes=probes,
            fixed_probes=fixed_probes,
            device=str(args.device),
            snapshot=snapshot,
        )
    )
    records.extend(
        _run_one_checkpoint(
            checkpoint_label="O1@200",
            checkpoint=args.o1_checkpoint,
            groot_source=args.groot_source,
            prepared=prepared,
            processor=processor,
            embodiment=embodiment,
            stage23=stage23,
            probes=probes,
            fixed_probes=fixed_probes,
            device=str(args.device),
            snapshot=snapshot,
        )
    )
    if len(records) != MAX_FLOW_EVALUATIONS:
        raise RuntimeError(f"C1 flow evaluation count drifted: {len(records)}")

    aggregates = _paired_aggregates(records)
    raw_hash_manifest = [
        {
            "checkpoint": item["checkpoint"],
            "probe_id": item["probe_id"],
            "raw_predicted_velocity_sha256": item["raw_predicted_velocity_sha256"],
        }
        for item in records
    ]
    decoded_hash_manifest = [
        {
            "checkpoint": item["checkpoint"],
            "probe_id": item["probe_id"],
            "decoded_physical_action_sha256": item["decoded_physical_action_sha256"],
        }
        for item in records
    ]
    result = {
        "schema": STAGE26_SCHEMA,
        "status": "PASS_EXECUTION",
        "stage": "Stage26 H1 C1",
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "identity": {
            "base_sha": STAGE26_BASE_SHA,
            "groot_sha": source_repo_sha,
            "base_checkpoint": base_identity,
            "o1_checkpoint": o1_identity,
            "dataset": dataset_identity,
            "selected_snapshot": snapshot_identity,
            "cosmos": cosmos_identity,
            "modality_sha256": _sha256_file(args.modality_config),
        },
        "sample": {
            "example_id": snapshot.example_id,
            "raw_episode_id": snapshot.raw_episode_id,
            "expert_action_sha256": prepared["expert_action_sha256"],
            "action_mask_sha256": prepared["action_mask_sha256"],
            "roundtrip_max_abs": float(np.max(np.abs(expert_decoded - expected_physical))),
            "horizon": HORIZON,
            "universal_action_shape": [1, HORIZON, UNIVERSAL_DIM],
            "physical_action_shape": [1, HORIZON, ACTIVE_DIM],
            "target_excluded_from_model_input": True,
        },
        "probe_contract": probe_contract,
        "records": records,
        "paired_aggregates": aggregates,
        "raw_velocity_hashes": raw_hash_manifest,
        "decoded_action_hashes": decoded_hash_manifest,
        "execution": {
            "flow_evaluation_count": len(records),
            "maximum_flow_evaluation_count": MAX_FLOW_EVALUATIONS,
            "complete_model_forward_called": False,
            "backbone_conditioning_forward_count": 2,
            "direct_teacher_forced_flow_forward_count": len(records),
            "teacher_forced_action_head_path": True,
            "free_running_sampler_called": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "pbs_submitted": False,
            "wbc_executed": False,
            "hardware_executed": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "c2_started": False,
            "device": str(args.device),
            "torch_dtype_cast": "torch.bfloat16",
        },
        "claim_boundary": "paired teacher-forced local-flow evidence only; no task-performance or hardware claim",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "stage26_h1_c1_result.json"
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result["result_json"] = str(output_path)
    result["result_json_sha256"] = _sha256_file(output_path)
    hashes_path = args.output_dir / "stage26_h1_c1_hash_manifest.json"
    hashes_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "schema": STAGE26_SCHEMA,
                    "raw_velocity_hashes": raw_hash_manifest,
                    "decoded_action_hashes": decoded_hash_manifest,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result["hash_manifest"] = str(hashes_path)
    result["hash_manifest_sha256"] = _sha256_file(hashes_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="print the frozen non-executing plan")
    mode.add_argument("--self-test", action="store_true", help="run standard-library static checks")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run the conditional Hopper-only C1 forward audit",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=EXPECTED_BASE_CHECKPOINT)
    parser.add_argument("--o1-checkpoint", type=Path, default=EXPECTED_O1_CHECKPOINT)
    parser.add_argument("--o1-seal", type=Path, default=EXPECTED_O1_SEAL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=EXPECTED_DATASET)
    parser.add_argument("--snapshot-root", type=Path, default=EXPECTED_SNAPSHOT_ROOT)
    parser.add_argument("--modality-config", type=Path, default=EXPECTED_MODALITY_PATH)
    parser.add_argument("--groot-source", type=Path, default=None)
    parser.add_argument("--stage10-root", type=Path, default=EXPECTED_STAGE10_ROOT)
    parser.add_argument("--cosmos-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/stage26_h1_c1"))
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.groot_source is None:
        args.groot_source = args.stage10_root / "src" / "isaac-gr00t"
    if args.cosmos_cache_root is None:
        args.cosmos_cache_root = args.stage10_root / EXPECTED_COSMOS_CACHE_RELATIVE
    if args.plan_only or (not args.self_test and not args.execute):
        plan = protocol_plan()
        assert_protocol_static(plan)
        print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
        return 0
    if args.self_test:
        plan = protocol_plan()
        assert_protocol_static(plan)
        _source_map_static()
        print(
            json.dumps(
                {
                    "schema": STAGE26_SCHEMA,
                    "status": "SELF_TEST_PASS",
                    "probe_count": MAX_PROBES,
                    "flow_evaluation_budget": MAX_FLOW_EVALUATIONS,
                },
                indent=2,
            )
        )
        return 0
    if args.execute:
        try:
            result = execute(args)
        except Exception as exc:
            summary = str(exc).splitlines()[0][:300]
            failure = {
                "schema": STAGE26_SCHEMA,
                "status": "STOP_OR_BLOCKED",
                "error_type": type(exc).__name__,
                "error_summary": summary,
                "scientific_result_available": False,
                "model_forward_scope": "not_claimed",
                "training": False,
                "pbs_submitted": False,
                "wbc_executed": False,
            }
            print(json.dumps(failure, indent=2, sort_keys=True))
            return 2
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
