#!/usr/bin/env python3
"""Prepare and (only when explicitly authorized) run Stage28 H2.

The default and ``--preflight`` paths are CPU/static paths.  They verify the
exact BASE tree, the locked TINY1 fixture, the pinned GR00T source and the
Cosmos cache before any model construction.  ``--train`` and ``--evaluate``
are execution modes for the separately authorized Hopper wrapper; Corelab
does not invoke either mode.

Stage28 is deliberately a new recipe, not a continuation of O1@200:

    fresh exact BASE -> 2000 updates, cosine, explicit warmup_steps=10

The training path keeps ``save_only_model=False`` and requires the stock
trainer state at checkpoint-1000 and checkpoint-2000.  After training, the
evaluation path reuses the Stage26 teacher-forced action-head graph and the
same nine CPU-float32 fixed probes for BASE and H2@2000.  It never calls the
free-running sampler or ``get_action``.

This module has no torch/transformers/safetensors import at module load.  The
execution-only imports occur after the identity and storage gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from statistics import geometric_mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence


STAGE28_SCHEMA = "STAGE28_H2_FRESH_BASE_2000_FIXED_WARMUP10_V1"
STAGE28_BASE_SHA = "9d0cf7c33b2d89ffee86168befe65978115564aa"
EXPECTED_STAGE26_BASE_SHA = "4005a4dd1ab6c82807985ef856d340fa50acaa6c"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_BASE_INDEX_SHA256 = (
    "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
)
EXPECTED_BASE_TREE_SHA256 = (
    "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
)
EXPECTED_BASE_TREE_PREFIX = "payload/GR00T-N1.7-3B"
EXPECTED_BASE_PROCESSOR_CONFIG_SHA256 = (
    "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
)
EXPECTED_BASE_STATISTICS_SHA256 = (
    "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
)
EXPECTED_TINY1_TREE_SHA256 = (
    "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
)
EXPECTED_TINY1_FILE_COUNT = 13
EXPECTED_TINY1_TOTAL_BYTES = 152958
EXPECTED_STAGE22_STATS_SHA256 = (
    "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
)
EXPECTED_STAGE22_RELATIVE_STATS_SHA256 = (
    "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
)
EXPECTED_MODALITY_SHA256 = (
    "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
)
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA256 = (
    "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
)
EXPECTED_TRAINING_SEED = 2222
EXPECTED_H2_STEPS = 2000
EXPECTED_SAVE_STEPS = 1000
EXPECTED_WARMUP_STEPS = 10
EXPECTED_WARMUP_RATIO_DISABLED = 0.0
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_WEIGHT_DECAY = 1e-5
EXPECTED_MAX_GRAD_NORM = 1.0
EXPECTED_OPTIMIZER = "adamw_torch"
EXPECTED_SCHEDULER = "cosine"
EXPECTED_BATCH_SIZE = 1
EXPECTED_GRADIENT_ACCUMULATION = 1
EXPECTED_HORIZON = 40
EXPECTED_ACTIVE_DIM = 10
EXPECTED_UNIVERSAL_DIM = 132
EXPECTED_PROBE_COUNT = 9
EXPECTED_MAX_FLOW_EVALUATIONS = 18
EXPECTED_STAGE26_MANIFEST_ROW_COUNT = 18
EXPECTED_TRAINABLE_PARAMETERS = 1_419_078_784
EXPECTED_STAGE26_HASH_MANIFEST_SHA256 = (
    "eb33c81b68bb8bd49ce0ed9b0b818fdbb29546b49a2d79a3ee62845e172afb7b"
)
EXPECTED_H2_CHECKPOINT_TREE_SHA256 = (
    "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
)
EXPECTED_H2_CHECKPOINT_SEAL_SHA256 = (
    "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
)
EXPECTED_H2_TRAINING_SOURCE_SHA = (
    "a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0"
)
EXPECTED_H2_TRAINING_RUNNER_SHA256 = (
    "d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4"
)
EXPECTED_STAGE26_PROBE_IDS = tuple(
    f"c1_seed{seed}_tq{quantile:02d}"
    for seed in (2223, 2224, 2225)
    for quantile in (10, 50, 90)
)

DEFAULT_BASE_MODEL_PATH = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"
)
DEFAULT_DATASET_PATH = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/"
    "action_pipeline_20260903_p0/tiny1_h2stats_lerobot_v2"
)
DEFAULT_STAGE10_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10")
DEFAULT_MODALITY_PATH = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/"
    "payload/modality/go2_arx5_modality.py"
)
DEFAULT_SNAPSHOT_ROOT = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage22/expert_dataset_v1/snapshots"
)
DEFAULT_COSMOS_CACHE_ROOT = DEFAULT_STAGE10_ROOT / "cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"
DEFAULT_STAGE26_HASH_MANIFEST = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage26/h1_c1/"
    "stage26_h1_c1_hash_manifest.json"
)

STATE_FILES_REQUIRED = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)
MODEL_INDEX_NAME = "model.safetensors.index.json"
CHECKPOINT_NAMES = ("checkpoint-1000", "checkpoint-2000")
# This token is retained only for the historical training/preparation path.
# Stage28 recovery evaluation has a separate current-gate token below.
HOPPER_AUTHORIZATION_TOKEN = "PI_GATE_STATE_0007"
EVALUATION_PI_GATE_ID = "PI_GATE_STATE_0008"
EVALUATION_AUTHORIZATION_TOKEN = EVALUATION_PI_GATE_ID
EXPECTED_EVALUATION_PBS_PROJECT = "CFP03-CF-004"
EVALUATION_NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE28_FIXED_PROBE_RESULT"
BASE_HASH_MISMATCH_EVIDENCE_NAME = "stage28_h2_base_hash_mismatch.json"

# The reserve is an operational guard, not a scientific input.  It covers the
# two retained model copies, the final model copy, full Adam moments for the
# selected trainable set, a write-time model copy, metadata, and headroom.
GIB = 1024**3
MIN_STORAGE_RESERVE_BYTES = 80 * GIB


class Stage28Error(RuntimeError):
    """Raised for any Stage28 identity, contract, or artifact failure."""


def _sha256_file(path: Path) -> str:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise Stage28Error(f"missing required file: {path}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise Stage28Error(f"required path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Stage28Error(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise Stage28Error(f"{label} is not a regular file: {path}")


def _resolve_non_symlink(path: Path, label: str) -> Path:
    """Resolve an input only after rejecting a symlink at the named root."""

    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Stage28Error(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(mode):
        raise Stage28Error(f"{label} root must not be a symlink: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise Stage28Error(f"{label} cannot be resolved: {path}") from exc


def _read_json(path: Path, label: str = "JSON evidence") -> dict[str, Any]:
    _assert_regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage28Error(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Stage28Error(f"{label} must be a JSON object: {path}")
    return value


def _regular_entries(root: Path) -> list[dict[str, Any]]:
    """Hash a tree while rejecting every symlink and special file."""

    root = Path(root)
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise Stage28Error(f"tree root is missing: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise Stage28Error(f"tree root is not a non-symlink directory: {root}")

    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise Stage28Error(f"tree contains an unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            _assert_regular(child, "tree file")
            relative = child.relative_to(root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "bytes": int(child.stat().st_size),
                    "sha256": _sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]], prefix: str) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(
            f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    path = Path(value)
    if not value or "\x00" in value or path.is_absolute() or ".." in path.parts:
        raise Stage28Error(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage28Error(f"cannot resolve git HEAD for {repo}") from exc


def _git_status(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain=v1"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage28Error(f"cannot inspect git worktree for {repo}") from exc


def _git_show(repo: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage28Error(
            f"cannot read pinned source object {revision}:{relative_path}"
        ) from exc


def _require_anchors(payload: bytes, anchors: Sequence[str], label: str) -> dict[str, bool]:
    text = payload.decode("utf-8")
    checks = {anchor: anchor in text for anchor in anchors}
    missing = [anchor for anchor, present in checks.items() if not present]
    if missing:
        raise Stage28Error(f"{label} missing source anchors: {missing}")
    return checks


def _verify_groot_source(groot_source: Path) -> dict[str, Any]:
    groot_source = _resolve_non_symlink(groot_source, "GR00T source")
    observed_head = _git_head(groot_source)
    if observed_head != EXPECTED_GROOT_SHA:
        raise Stage28Error(
            f"GR00T source SHA mismatch: {observed_head} != {EXPECTED_GROOT_SHA}"
        )
    # The imported source is the working tree, not an abstract Git object.
    # Documentation/notebook edits are irrelevant, but a modified gr00t/
    # source tree would invalidate the pinned runtime graph.
    source_status = _git_status(groot_source)
    source_code_changes = [
        line for line in source_status.splitlines() if line and line[3:].startswith("gr00t/")
    ]
    if source_code_changes:
        raise Stage28Error(
            "GR00T gr00t/ working tree is dirty: "
            + "; ".join(source_code_changes[:5])
        )
    source_anchors = {
        "gr00t/configs/training/training_config.py": (
            'lr_scheduler_type: str = "cosine"',
            "warmup_ratio: float = 0.05",
            "warmup_steps: int = 0",
            "save_only_model: bool = False",
            "warmup_steps",
            "check_resume_compatibility",
        ),
        "gr00t/experiment/experiment.py": (
            "max_steps=config.training.max_steps",
            "lr_scheduler_type=config.training.lr_scheduler_type",
            "warmup_ratio=config.training.warmup_ratio",
            "warmup_steps=config.training.warmup_steps",
            "save_only_model=config.training.save_only_model",
            "trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)",
            "trainer.save_model()",
            "ignore_data_skip=True",
        ),
        "gr00t/model/gr00t_n1d7/setup.py": (
            "start_from_checkpoint",
            "load_bf16=self.config.model.load_bf16",
            "tune_projector=self.config.model.tune_projector",
            "tune_diffusion_model=self.config.model.tune_diffusion_model",
        ),
    }
    files: dict[str, Any] = {}
    for relative_path, anchors in source_anchors.items():
        payload = _git_show(groot_source, EXPECTED_GROOT_SHA, relative_path)
        files[relative_path] = {
            "sha256": _sha256_bytes(payload),
            "anchors": _require_anchors(payload, anchors, relative_path),
        }
    return {
        "repository": str(groot_source),
        "observed_head": observed_head,
        "expected_head": EXPECTED_GROOT_SHA,
        "files": files,
    }


def _verify_index_and_shards(root: Path) -> dict[str, Any]:
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    if len(index_paths) != 1:
        raise Stage28Error(
            f"checkpoint must contain exactly one safetensors index: {index_paths}"
        )
    index_path = index_paths[0]
    _assert_regular(index_path, "checkpoint index")
    index_sha = _sha256_file(index_path)
    index = _read_json(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise Stage28Error("safetensors index has no non-empty weight_map")
    shard_names = sorted({str(value) for value in weight_map.values()})
    if any(
        not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or Path(name).as_posix() != name
        for name in shard_names
    ):
        raise Stage28Error("safetensors index contains an unsafe shard reference")
    shard_manifest: list[dict[str, Any]] = []
    for shard_name in shard_names:
        shard_path = root / shard_name
        _assert_regular(shard_path, f"indexed shard {shard_name}")
        shard_manifest.append(
            {
                "path": shard_name,
                "bytes": int(shard_path.stat().st_size),
                "sha256": _sha256_file(shard_path),
            }
        )
    return {
        "index_path": index_path.name,
        "index_sha256": index_sha,
        "index_key_count": len(weight_map),
        "referenced_shards": shard_manifest,
    }


def _verify_base_checkpoint(base_checkpoint: Path) -> dict[str, Any]:
    """Verify the immutable BASE identity before model construction."""

    base_checkpoint = _resolve_non_symlink(base_checkpoint, "BASE checkpoint")
    entries = _regular_entries(base_checkpoint)
    index = _verify_index_and_shards(base_checkpoint)
    if index["index_sha256"] != EXPECTED_BASE_INDEX_SHA256:
        raise Stage28Error(
            f"BASE index SHA mismatch: {index['index_sha256']} != "
            f"{EXPECTED_BASE_INDEX_SHA256}"
        )
    tree_sha = _tree_digest(entries, EXPECTED_BASE_TREE_PREFIX)
    if tree_sha != EXPECTED_BASE_TREE_SHA256:
        raise Stage28Error(f"BASE canonical tree SHA mismatch: {tree_sha}")
    by_path = {str(entry["path"]): entry for entry in entries}
    for relative, expected in (
        ("processor_config.json", EXPECTED_BASE_PROCESSOR_CONFIG_SHA256),
        ("statistics.json", EXPECTED_BASE_STATISTICS_SHA256),
    ):
        observed = by_path.get(relative, {}).get("sha256")
        if observed != expected:
            raise Stage28Error(
                f"BASE {relative} SHA mismatch: {observed} != {expected}"
            )
    indexed_names = {str(item["path"]) for item in index["referenced_shards"]}
    observed_names = indexed_names & set(by_path)
    if observed_names != indexed_names:
        raise Stage28Error("BASE tree is missing an indexed shard")
    return {
        "path": str(base_checkpoint),
        "tree_sha256": tree_sha,
        "tree_prefix": EXPECTED_BASE_TREE_PREFIX,
        "tree_file_count": len(entries),
        "tree_total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "index": index,
        "processor_config_sha256": EXPECTED_BASE_PROCESSOR_CONFIG_SHA256,
        "statistics_sha256": EXPECTED_BASE_STATISTICS_SHA256,
        "symlinks_rejected": True,
    }


def _load_stage26_module() -> Any:
    """Load Stage26 helpers without importing runtime ML packages at import time."""

    script_path = Path(__file__).with_name("stage26_h1_c1_fixed_noise_local_flow.py")
    spec = importlib.util.spec_from_file_location("stage26_h1_c1_for_stage28", script_path)
    if spec is None or spec.loader is None:
        raise Stage28Error(f"cannot load Stage26 helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_tiny_dataset(dataset_dir: Path) -> dict[str, Any]:
    stage26 = _load_stage26_module()
    dataset_dir = _resolve_non_symlink(dataset_dir, "TINY1 dataset")
    identity = stage26._verify_tiny_dataset(dataset_dir)
    if identity["tree_sha256"] != EXPECTED_TINY1_TREE_SHA256:
        raise Stage28Error("TINY1 tree identity drifted")
    if identity["file_count"] != EXPECTED_TINY1_FILE_COUNT:
        raise Stage28Error("TINY1 file count drifted")
    if identity["total_bytes"] != EXPECTED_TINY1_TOTAL_BYTES:
        raise Stage28Error("TINY1 byte count drifted")
    return identity


def _verify_cosmos_cache(cache_root: Path) -> dict[str, Any]:
    stage26 = _load_stage26_module()
    cache_root = _resolve_non_symlink(cache_root, "Cosmos cache")
    identity = stage26._verify_cosmos_cache(cache_root)
    if identity["tree_sha256"] != EXPECTED_COSMOS_TREE_SHA256:
        raise Stage28Error("Cosmos cache tree identity drifted")
    if identity["revision"] != EXPECTED_COSMOS_REVISION:
        raise Stage28Error("Cosmos cache revision drifted")
    return identity


def _assert_path_not_inside(path: Path, immutable: Path, label: str) -> None:
    try:
        path.resolve().relative_to(immutable.resolve())
    except ValueError:
        return
    raise Stage28Error(f"{label} must be outside immutable input: {path}")


def storage_reservation(base_tree_bytes: int) -> dict[str, int]:
    """Return the conservative free-space reserve for full-state checkpoints."""

    if base_tree_bytes < 1:
        raise Stage28Error("BASE tree byte count must be positive")
    retained_model = 2 * int(base_tree_bytes)
    final_model = int(base_tree_bytes)
    write_scratch = int(base_tree_bytes)
    adam_moments = 2 * EXPECTED_TRAINABLE_PARAMETERS * 12
    metadata_and_margin = 10 * GIB
    calculated = retained_model + final_model + write_scratch + adam_moments + metadata_and_margin
    required = max(MIN_STORAGE_RESERVE_BYTES, calculated)
    return {
        "retained_model_checkpoint_bytes": retained_model,
        "final_model_copy_bytes": final_model,
        "write_scratch_bytes": write_scratch,
        "adam_full_state_reserve_bytes": adam_moments,
        "metadata_and_margin_bytes": metadata_and_margin,
        "calculated_required_free_bytes": calculated,
        "minimum_required_free_bytes": required,
    }


def _ensure_output_root(output_root: Path) -> None:
    output_root = Path(output_root)
    if output_root.exists():
        mode = output_root.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise Stage28Error(f"Stage28 output root is unsafe: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    allowed = {"stage28_preflight.json"}
    extras = []
    for path in output_root.iterdir():
        if path.name in allowed:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise Stage28Error(f"Stage28 preflight file is unsafe: {path}")
        else:
            extras.append(path.name)
    extras.sort()
    if extras:
        raise Stage28Error(
            "Stage28 output root must be new or contain only stage28_preflight.json: "
            f"{extras}"
        )


def _storage_preflight(
    output_root: Path,
    *,
    base_tree_bytes: int,
    immutable_inputs: Sequence[Path],
) -> dict[str, Any]:
    output_root = Path(output_root)
    for immutable in immutable_inputs:
        _assert_path_not_inside(output_root, immutable, "Stage28 output root")
    parent = output_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise Stage28Error(f"Stage28 output parent is not a real directory: {parent}")
    _ensure_output_root(output_root)
    usage = shutil.disk_usage(parent)
    reserve = storage_reservation(base_tree_bytes)
    passed = int(usage.free) >= reserve["minimum_required_free_bytes"]
    result = {
        "status": "PASS" if passed else "FAIL",
        "filesystem_path": str(parent),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "reserve": reserve,
    }
    if not passed:
        raise Stage28Error(
            "full-state checkpoint storage preflight failed: "
            f"free={usage.free} < required={reserve['minimum_required_free_bytes']}"
        )
    return result


def _verify_modality(modality_path: Path) -> str:
    observed = _sha256_file(Path(modality_path).resolve(strict=True))
    if observed != EXPECTED_MODALITY_SHA256:
        raise Stage28Error(f"modality SHA mismatch: {observed}")
    return observed


def _probe_contract() -> dict[str, Any]:
    stage26 = _load_stage26_module()
    plan = stage26.protocol_plan()
    stage26.assert_protocol_static(plan)
    probes = plan["probe_contract"]["probes"]
    if len(probes) != EXPECTED_PROBE_COUNT:
        raise Stage28Error(f"Stage26 probe count changed: {len(probes)}")
    return {
        "source_schema": plan["schema"],
        "source_base_sha": plan["base_sha"],
        "probe_count": len(probes),
        "probes": probes,
        "mask_partition": plan["probe_contract"]["mask_partition"],
        "pairing": "one canonical CPU-float32 noise/t tensor reused for BASE and H2",
    }


def _cosine_learning_rate(
    global_step: int,
    *,
    total_steps: int = EXPECTED_H2_STEPS,
    warmup_steps: int = EXPECTED_WARMUP_STEPS,
) -> float:
    """Mirror the pinned Trainer log convention used by Stage27."""

    if global_step < 1 or total_steps < 1 or warmup_steps < 0:
        raise Stage28Error("invalid schedule dimensions")
    scheduler_step = global_step - 1
    if scheduler_step < warmup_steps:
        factor = scheduler_step / max(1, warmup_steps)
    else:
        progress = (scheduler_step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress)))
    return float(EXPECTED_LEARNING_RATE * factor)


def _training_contract() -> dict[str, Any]:
    return {
        "initialization": "fresh exact BASE; no O1 warm-start",
        "dataset": "TINY1_H2STATS",
        "horizon": EXPECTED_HORIZON,
        "physical_action_dim": EXPECTED_ACTIVE_DIM,
        "universal_action_dim": EXPECTED_UNIVERSAL_DIM,
        "action_abi": "dense relative EEF XYZ_ROT6D plus absolute gripper",
        "active_action_columns": "0:10",
        "inactive_action_columns": "10:132 remain input; loss mask zero",
        "seed": EXPECTED_TRAINING_SEED,
        "batch_size": EXPECTED_BATCH_SIZE,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_visual": False,
        "tune_llm": False,
        "tune_vlln": False,
        "load_bf16": True,
        "backbone_trainable_params_fp32": False,
        "optimizer": EXPECTED_OPTIMIZER,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "max_grad_norm": EXPECTED_MAX_GRAD_NORM,
        "lr_scheduler_type": EXPECTED_SCHEDULER,
        "max_steps": EXPECTED_H2_STEPS,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "warmup_ratio": EXPECTED_WARMUP_RATIO_DISABLED,
        "cosine_denominator_steps": EXPECTED_H2_STEPS - EXPECTED_WARMUP_STEPS,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "save_steps": EXPECTED_SAVE_STEPS,
        "save_total_limit": 2,
        "save_only_model": False,
        "resume_from_checkpoint": False,
        "local_files_only": True,
        "wandb": False,
    }


def protocol_plan() -> dict[str, Any]:
    """Return the complete Stage28 preparation/handoff contract."""

    contract = _training_contract()
    return {
        "schema": STAGE28_SCHEMA,
        "stage": "Stage28 H2 fresh BASE 2000 fixed-warmup10",
        "status": "PREPARED_NOT_EXECUTED",
        "stage28_base_sha": STAGE28_BASE_SHA,
        "stage27_accepted_sha": STAGE28_BASE_SHA,
        "stage26_probe_base_sha": EXPECTED_STAGE26_BASE_SHA,
        "scientific_question": (
            "whether one coherent larger-budget fresh adaptation recipe can "
            "substantially improve the same TINY1 local flow field"
        ),
        "claim_boundary": (
            "new larger-budget fine-tune recipe; not O1+1800, not a pure "
            "extra-exposure intervention, and not a task/generalization/WBC claim"
        ),
        "fixed_identities": {
            "base_model_path": str(DEFAULT_BASE_MODEL_PATH),
            "base_index_sha256": EXPECTED_BASE_INDEX_SHA256,
            "base_tree_sha256": EXPECTED_BASE_TREE_SHA256,
            "base_tree_prefix": EXPECTED_BASE_TREE_PREFIX,
            "dataset": "TINY1_H2STATS",
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "dataset_tree_sha256": EXPECTED_TINY1_TREE_SHA256,
            "dataset_file_count": EXPECTED_TINY1_FILE_COUNT,
            "dataset_total_bytes": EXPECTED_TINY1_TOTAL_BYTES,
            "modality_sha256": EXPECTED_MODALITY_SHA256,
            "groot_sha": EXPECTED_GROOT_SHA,
            "cosmos_revision": EXPECTED_COSMOS_REVISION,
            "cosmos_tree_sha256": EXPECTED_COSMOS_TREE_SHA256,
        },
        "training_contract": contract,
        "schedule": {
            "selected_warmup_steps": EXPECTED_WARMUP_STEPS,
            "warmup_ratio_disabled": EXPECTED_WARMUP_RATIO_DISABLED,
            "cosine_denominator_steps": EXPECTED_H2_STEPS - EXPECTED_WARMUP_STEPS,
            "selected_rows": [
                {
                    "global_step": step,
                    "learning_rate": _cosine_learning_rate(step),
                }
                for step in (1, 2, 10, 11, 12, 100, 200, 1000, 1999, 2000)
            ],
        },
        "checkpoint_policy": {
            "save_only_model": False,
            "save_steps": EXPECTED_SAVE_STEPS,
            "save_total_limit": 2,
            "required_checkpoint_directories": list(CHECKPOINT_NAMES),
            "required_state_files_per_checkpoint": list(STATE_FILES_REQUIRED),
            "fp16_scaler": "not_applicable_bf16_true_fp16_false",
            "final_checkpoint": "checkpoint-2000",
            "final_seal": "stage28_h2_checkpoint_2000_seal.json",
            "exact_resume_claim": False,
        },
        "probe_contract": _probe_contract(),
        "execution_budget": {
            "training_runs": 1,
            "gpu_class": "1xH200",
            "checkpoints_evaluated": ["BASE", "H2@2000"],
            "probes_per_checkpoint": EXPECTED_PROBE_COUNT,
            "maximum_direct_flow_evaluations": EXPECTED_MAX_FLOW_EVALUATIONS,
            "free_running_sampler": False,
        },
        "storage_policy": {
            "reserve_formula": "two retained models + final model + write scratch + full Adam state reserve + margin",
            "minimum_free_bytes": MIN_STORAGE_RESERVE_BYTES,
            "runtime_preflight_required_before_model_construction": True,
        },
        "conditional_handoff": {
            "pi_gate_id": "PI_GATE_STATE_0007",
            "token": HOPPER_AUTHORIZATION_TOKEN,
            "h2_execution_ready": True,
            "storage_preflight": "HOPPER_RUNTIME_REQUIRED",
            "scientific_contract_changed": False,
            "downstream_authorization_still_valid": True,
            "live_head_must_equal_returned_sha": True,
        },
        "evaluation_recovery": {
            "pi_gate_id": EVALUATION_PI_GATE_ID,
            "token": EVALUATION_AUTHORIZATION_TOKEN,
            "evaluation_only": True,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "h2_checkpoint_tree_sha256": EXPECTED_H2_CHECKPOINT_TREE_SHA256,
            "h2_checkpoint_seal_sha256": EXPECTED_H2_CHECKPOINT_SEAL_SHA256,
            "historical_training_source_sha": EXPECTED_H2_TRAINING_SOURCE_SHA,
            "historical_training_runner_sha256": EXPECTED_H2_TRAINING_RUNNER_SHA256,
            "maximum_direct_flow_evaluations": EXPECTED_MAX_FLOW_EVALUATIONS,
            "next_gate": EVALUATION_NEXT_GATE,
        },
        "preparation_execution_boundary": {
            "model_construction": False,
            "model_forward": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "gpu": False,
            "pbs_submission": False,
            "free_running_sampler": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "c2": False,
            "wbc": False,
            "hardware": False,
        },
        "next_gate": EVALUATION_NEXT_GATE,
    }


def assert_protocol_static(plan: Mapping[str, Any] | None = None) -> None:
    candidate = protocol_plan() if plan is None else plan
    if candidate.get("schema") != STAGE28_SCHEMA:
        raise Stage28Error("Stage28 schema drifted")
    if candidate.get("status") != "PREPARED_NOT_EXECUTED":
        raise Stage28Error("Corelab Stage28 plan must remain non-executing")
    if candidate.get("stage28_base_sha") != STAGE28_BASE_SHA:
        raise Stage28Error("Stage28 base SHA drifted")
    if candidate.get("stage27_accepted_sha") != STAGE28_BASE_SHA:
        raise Stage28Error("Stage28 is not based on accepted Stage27 HEAD")
    contract = candidate.get("training_contract", {})
    expected_contract = _training_contract()
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise Stage28Error(f"H2 training contract drifted: {key}")
    if candidate.get("fixed_identities", {}).get("base_index_sha256") != EXPECTED_BASE_INDEX_SHA256:
        raise Stage28Error("BASE index identity drifted")
    if candidate.get("fixed_identities", {}).get("base_tree_sha256") != EXPECTED_BASE_TREE_SHA256:
        raise Stage28Error("BASE tree identity drifted")
    if candidate.get("fixed_identities", {}).get("dataset_tree_sha256") != EXPECTED_TINY1_TREE_SHA256:
        raise Stage28Error("TINY1 identity drifted")
    probes = candidate.get("probe_contract", {}).get("probes", [])
    if len(probes) != EXPECTED_PROBE_COUNT:
        raise Stage28Error("Stage28 must retain exactly nine Stage26 probes")
    budget = candidate.get("execution_budget", {})
    if budget.get("training_runs") != 1:
        raise Stage28Error("Stage28 training repeat budget drifted")
    if budget.get("maximum_direct_flow_evaluations") != EXPECTED_MAX_FLOW_EVALUATIONS:
        raise Stage28Error("Stage28 flow-evaluation budget drifted")
    boundary = candidate.get("preparation_execution_boundary", {})
    if any(bool(value) for value in boundary.values()):
        raise Stage28Error("Stage28 preparation plan enables a forbidden operation")
    handoff = candidate.get("conditional_handoff", {})
    if handoff.get("h2_execution_ready") is not True:
        raise Stage28Error("Stage28 H2 execution readiness is not explicit")
    if handoff.get("storage_preflight") != "HOPPER_RUNTIME_REQUIRED":
        raise Stage28Error("Stage28 storage gate is not runtime-bound")
    if handoff.get("scientific_contract_changed") is not False:
        raise Stage28Error("Stage28 scientific contract changed")
    if handoff.get("downstream_authorization_still_valid") is not True:
        raise Stage28Error("Stage28 downstream token was not preserved")
    recovery = candidate.get("evaluation_recovery", {})
    expected_recovery = {
        "pi_gate_id": EVALUATION_PI_GATE_ID,
        "token": EVALUATION_AUTHORIZATION_TOKEN,
        "evaluation_only": True,
        "training": False,
        "backward": False,
        "optimizer_step": False,
        "scheduler_step": False,
        "resume": False,
        "checkpoint_write": False,
        "h2_checkpoint_tree_sha256": EXPECTED_H2_CHECKPOINT_TREE_SHA256,
        "h2_checkpoint_seal_sha256": EXPECTED_H2_CHECKPOINT_SEAL_SHA256,
        "historical_training_source_sha": EXPECTED_H2_TRAINING_SOURCE_SHA,
        "historical_training_runner_sha256": EXPECTED_H2_TRAINING_RUNNER_SHA256,
        "maximum_direct_flow_evaluations": EXPECTED_MAX_FLOW_EVALUATIONS,
        "next_gate": EVALUATION_NEXT_GATE,
    }
    for key, expected in expected_recovery.items():
        if recovery.get(key) != expected:
            raise Stage28Error(f"Stage28 evaluation recovery contract drifted: {key}")


def _load_modality(path: Path, stage10_root: Path) -> Any:
    source_root = Path(stage10_root) / "src/isaac-gr00t"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module_name = f"stage28_modality_{_sha256_file(path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Stage28Error(f"cannot load modality module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_training_config(args: argparse.Namespace) -> Any:
    """Build the official config only after the runtime identity gates."""

    _load_modality(args.modality_config, args.stage10_root)
    source_root = Path(args.groot_source).resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(args.dataset_dir)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # These values mirror the accepted Stage23 action-head adaptation policy;
    # the only intentional recipe changes are the selected 2000-step budget,
    # explicit warmup_steps=10, and full-state checkpoint retention.
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_vlln = False
    config.model.load_bf16 = True
    config.model.backbone_trainable_params_fp32 = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.use_relative_action = True
    config.model.action_horizon = EXPECTED_HORIZON
    config.model.state_dropout_prob = 0.0
    config.model.color_jitter_params = None
    config.model.random_rotation_angle = None
    config.model.image_crop_size = None
    config.model.image_target_size = None
    config.model.shortest_image_edge = 96
    config.model.crop_fraction = 1.0
    config.model.use_albumentations_transforms = True

    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(args.base_model_path)
    config.training.transformers_local_files_only = True
    config.training.transformers_trust_remote_code = True
    config.training.optim = EXPECTED_OPTIMIZER
    config.training.global_batch_size = EXPECTED_BATCH_SIZE
    config.training.per_gpu_batch_size = None
    config.training.gradient_accumulation_steps = EXPECTED_GRADIENT_ACCUMULATION
    config.training.dataloader_num_workers = 0
    config.training.learning_rate = EXPECTED_LEARNING_RATE
    config.training.output_dir = str(args.training_output_dir)
    config.training.save_steps = EXPECTED_SAVE_STEPS
    config.training.save_total_limit = 2
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.max_steps = EXPECTED_H2_STEPS
    config.training.weight_decay = EXPECTED_WEIGHT_DECAY
    config.training.max_grad_norm = EXPECTED_MAX_GRAD_NORM
    config.training.lr_scheduler_type = EXPECTED_SCHEDULER
    config.training.warmup_steps = EXPECTED_WARMUP_STEPS
    config.training.warmup_ratio = EXPECTED_WARMUP_RATIO_DISABLED
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    config.training.gradient_checkpointing = False
    config.training.logging_steps = 1
    config.training.shard_size = 256
    config.training.episode_sampling_rate = 1.0
    config.training.num_shards_per_epoch = 1
    config.training.save_only_model = False
    config.training.resume_from_checkpoint = False
    config.training.skip_weight_loading = False
    config.training.upload_checkpoints = False
    config.data.seed = EXPECTED_TRAINING_SEED
    return config


def _config_summary(config: Any) -> dict[str, Any]:
    return {
        "start_from_checkpoint": str(config.training.start_from_checkpoint),
        "max_steps": int(config.training.max_steps),
        "save_steps": int(config.training.save_steps),
        "save_total_limit": int(config.training.save_total_limit),
        "warmup_steps": int(config.training.warmup_steps),
        "warmup_ratio": float(config.training.warmup_ratio),
        "lr_scheduler_type": str(config.training.lr_scheduler_type),
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "max_grad_norm": float(config.training.max_grad_norm),
        "optim": str(config.training.optim),
        "global_batch_size": int(config.training.global_batch_size),
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "bf16": bool(config.training.bf16),
        "fp16": bool(config.training.fp16),
        "tf32": bool(config.training.tf32),
        "save_only_model": bool(config.training.save_only_model),
        "resume_from_checkpoint": bool(config.training.resume_from_checkpoint),
        "tune_projector": bool(config.model.tune_projector),
        "tune_diffusion_model": bool(config.model.tune_diffusion_model),
        "tune_visual": bool(config.model.tune_visual),
        "tune_llm": bool(config.model.tune_llm),
        "tune_vlln": bool(config.model.tune_vlln),
        "load_bf16": bool(config.model.load_bf16),
        "backbone_trainable_params_fp32": bool(config.model.backbone_trainable_params_fp32),
        "seed": EXPECTED_TRAINING_SEED,
    }


def _assert_config_contract(summary: Mapping[str, Any], base_path: Path, output_path: Path) -> None:
    expected = _training_contract()
    checks = {
        "max_steps": expected["max_steps"],
        "save_steps": expected["save_steps"],
        "save_total_limit": expected["save_total_limit"],
        "warmup_steps": expected["warmup_steps"],
        "warmup_ratio": expected["warmup_ratio"],
        "lr_scheduler_type": expected["lr_scheduler_type"],
        "learning_rate": expected["learning_rate"],
        "weight_decay": expected["weight_decay"],
        "max_grad_norm": expected["max_grad_norm"],
        "optim": expected["optimizer"],
        "global_batch_size": expected["batch_size"],
        "gradient_accumulation_steps": expected["gradient_accumulation_steps"],
        "bf16": expected["bf16"],
        "fp16": expected["fp16"],
        "tf32": expected["tf32"],
        "save_only_model": expected["save_only_model"],
        "resume_from_checkpoint": expected["resume_from_checkpoint"],
        "tune_projector": expected["tune_projector"],
        "tune_diffusion_model": expected["tune_diffusion_model"],
        "tune_visual": expected["tune_visual"],
        "tune_llm": expected["tune_llm"],
        "tune_vlln": expected["tune_vlln"],
        "load_bf16": expected["load_bf16"],
        "backbone_trainable_params_fp32": expected["backbone_trainable_params_fp32"],
    }
    for key, value in checks.items():
        if summary.get(key) != value:
            raise Stage28Error(f"resolved training config mismatch: {key}")
    if summary.get("start_from_checkpoint") != str(base_path):
        raise Stage28Error("training does not start from the exact preflighted BASE path")
    if summary.get("start_from_checkpoint") == str(output_path):
        raise Stage28Error("training output path aliases the BASE path")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run all runtime identity and storage checks without model construction."""

    base = _verify_base_checkpoint(args.base_model_path)
    dataset = _verify_tiny_dataset(args.dataset_dir)
    modality_sha = _verify_modality(args.modality_config)
    source = _verify_groot_source(args.groot_source)
    cosmos = _verify_cosmos_cache(args.cosmos_cache_root)
    project_repo = _resolve_non_symlink(args.project_repo, "Stage28 project checkout")
    project_head = _git_head(project_repo)
    project_status = _git_status(project_repo)
    if project_status:
        raise Stage28Error("Stage28 project checkout is dirty")
    if args.expected_project_sha and project_head != args.expected_project_sha:
        raise Stage28Error(
            f"Stage28 checkout SHA mismatch: {project_head} != {args.expected_project_sha}"
        )
    output_root_arg = Path(args.output_root)
    storage = _storage_preflight(
        output_root_arg,
        base_tree_bytes=int(base["tree_total_bytes"]),
        immutable_inputs=(Path(args.base_model_path), Path(args.dataset_dir)),
    )
    return {
        "schema": STAGE28_SCHEMA,
        "status": "PASS",
        "stage": "Stage28 runtime preflight",
        "stage28_git_sha": project_head,
        "expected_stage28_git_sha": args.expected_project_sha,
        "stage28_worktree_clean": True,
        "base": base,
        "dataset": dataset,
        "modality_sha256": modality_sha,
        "groot": source,
        "cosmos": cosmos,
        "output_root": str(output_root_arg.resolve()),
        "storage": storage,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "backward": False,
        "optimizer_step": False,
        "gpu": False,
        "pbs_submitted": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "scientific_contract_changed": False,
        "downstream_authorization_still_valid": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    result = _preflight(args)
    output_root = _resolve_non_symlink(args.output_root, "Stage28 output root")
    _write_json(output_root / "stage28_preflight.json", result)
    return result


def _require_token(args: argparse.Namespace) -> None:
    if args.authorization_token != HOPPER_AUTHORIZATION_TOKEN:
        raise Stage28Error(
            "execution mode requires the exact PI_GATE_STATE_0007 authorization token"
        )


def _require_evaluation_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Bind recovery evaluation to the current PI gate and sealed artifacts."""

    if args.authorization_token != EVALUATION_AUTHORIZATION_TOKEN:
        raise Stage28Error(
            "evaluation-only recovery requires the exact PI_GATE_STATE_0008 token"
        )
    expected = {
        "expected_h2_checkpoint_tree_sha256": EXPECTED_H2_CHECKPOINT_TREE_SHA256,
        "expected_h2_checkpoint_seal_sha256": EXPECTED_H2_CHECKPOINT_SEAL_SHA256,
        "training_source_sha": EXPECTED_H2_TRAINING_SOURCE_SHA,
        "training_runner_sha256": EXPECTED_H2_TRAINING_RUNNER_SHA256,
    }
    for argument, expected_value in expected.items():
        observed = getattr(args, argument, None)
        if observed != expected_value:
            raise Stage28Error(
                f"evaluation provenance binding mismatch for {argument}: "
                f"{observed!r} != {expected_value}"
            )

    evaluation_runner_sha256 = _sha256_file(Path(__file__).resolve())
    expected_runner_sha256 = getattr(args, "expected_evaluation_runner_sha256", None)
    if expected_runner_sha256 != evaluation_runner_sha256:
        raise Stage28Error(
            "evaluation runner SHA mismatch: "
            f"{evaluation_runner_sha256} != {expected_runner_sha256}"
        )
    evaluation_launcher_sha256 = getattr(args, "evaluation_launcher_sha256", None)
    if (
        not isinstance(evaluation_launcher_sha256, str)
        or len(evaluation_launcher_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evaluation_launcher_sha256)
    ):
        raise Stage28Error("evaluation launcher SHA is not a lowercase SHA256 digest")
    pbs_job_id = getattr(args, "pbs_job_id", None)
    if (
        not isinstance(pbs_job_id, str)
        or not pbs_job_id.strip()
        or any(character.isspace() for character in pbs_job_id)
    ):
        raise Stage28Error("evaluation PBS job ID is missing or malformed")
    pbs_project = getattr(args, "pbs_project", None)
    if pbs_project != EXPECTED_EVALUATION_PBS_PROJECT:
        raise Stage28Error(
            "evaluation PBS project mismatch: "
            f"{pbs_project!r} != {EXPECTED_EVALUATION_PBS_PROJECT}"
        )
    return {
        "pi_gate": EVALUATION_PI_GATE_ID,
        "training_source_sha": EXPECTED_H2_TRAINING_SOURCE_SHA,
        "training_runner_sha256": EXPECTED_H2_TRAINING_RUNNER_SHA256,
        "evaluation_runner_sha256": evaluation_runner_sha256,
        "evaluation_launcher_sha256": evaluation_launcher_sha256,
        "pbs_job_id": pbs_job_id,
        "pbs_project": pbs_project,
    }


def _load_and_revalidate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_non_symlink(args.output_root, "Stage28 output root")
    report_path = output_root / "stage28_preflight.json"
    report = _read_json(report_path, "Stage28 preflight report")
    if report.get("schema") != STAGE28_SCHEMA or report.get("status") != "PASS":
        raise Stage28Error("Stage28 preflight report is not a PASS for this schema")
    if report.get("output_root") != str(output_root):
        raise Stage28Error("Stage28 preflight output root binding changed")
    current = _preflight(args)
    for key in (
        "base",
        "dataset",
        "modality_sha256",
        "groot",
        "cosmos",
        "stage28_git_sha",
        "expected_stage28_git_sha",
        "stage28_worktree_clean",
    ):
        if current.get(key) != report.get(key):
            raise Stage28Error(f"Stage28 preflight changed before execution: {key}")
    current_storage = current.get("storage") or {}
    report_storage = report.get("storage") or {}
    for key in ("status", "filesystem_path", "reserve"):
        if current_storage.get(key) != report_storage.get(key):
            raise Stage28Error(f"Stage28 storage preflight binding changed: {key}")
    if current_storage.get("status") != "PASS":
        raise Stage28Error("Stage28 storage preflight is not PASS")
    return current


def _checkpoint_state_files(relative_paths: Iterable[str]) -> list[str]:
    return sorted(
        path
        for path in relative_paths
        if Path(path).name in STATE_FILES_REQUIRED
    )


def _inspect_checkpoint(checkpoint: Path, *, expected_step: int) -> dict[str, Any]:
    checkpoint = _resolve_non_symlink(checkpoint, "H2 checkpoint")
    entries = _regular_entries(checkpoint)
    index = _verify_index_and_shards(checkpoint)
    paths = {str(entry["path"]) for entry in entries}
    missing_state = sorted(set(STATE_FILES_REQUIRED) - paths)
    if missing_state:
        raise Stage28Error(
            f"{checkpoint.name} is missing full Trainer state files: {missing_state}"
        )
    trainer_state = _read_json(checkpoint / "trainer_state.json", "trainer state")
    if trainer_state.get("global_step") != expected_step:
        raise Stage28Error(
            f"{checkpoint.name} global_step mismatch: {trainer_state.get('global_step')}"
        )
    if trainer_state.get("max_steps") != EXPECTED_H2_STEPS:
        raise Stage28Error(f"{checkpoint.name} trainer max_steps is not 2000")
    return {
        "path": str(checkpoint),
        "name": checkpoint.name,
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "state_files": _checkpoint_state_files(paths),
        "index": index,
        "tree_sha256": _tree_digest(entries, f"stage28/{checkpoint.name}"),
        "entries": entries,
    }


def _seal_checkpoint(checkpoint_info: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    entries = checkpoint_info.get("entries")
    if not isinstance(entries, list):
        raise Stage28Error("checkpoint inventory has no entries to seal")
    seal = {
        "schema": "STAGE28_H2_FINAL_CHECKPOINT_SEAL_V1",
        "status": "PASS",
        "checkpoint_name": "checkpoint-2000",
        "checkpoint_path": checkpoint_info["path"],
        "tree_sha256": checkpoint_info["tree_sha256"],
        "file_count": checkpoint_info["file_count"],
        "total_bytes": checkpoint_info["total_bytes"],
        "entries": entries,
        "scientific_contract_changed": False,
    }
    seal_path = Path(output_root) / "stage28_h2_checkpoint_2000_seal.json"
    _write_json(seal_path, seal)
    return {
        "path": str(seal_path),
        "sha256": _sha256_file(seal_path),
        "tree_sha256": checkpoint_info["tree_sha256"],
        "file_count": checkpoint_info["file_count"],
        "total_bytes": checkpoint_info["total_bytes"],
    }


def _verify_final_checkpoint_artifacts(
    output_root: Path,
    *,
    expected_checkpoint_tree_sha256: str | None = None,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    training_output = Path(output_root) / "h2_training"
    if training_output.is_symlink() or not training_output.is_dir():
        raise Stage28Error(f"H2 training output is missing or symlinked: {training_output}")
    checkpoint_dirs = sorted(
        path.name
        for path in training_output.iterdir()
        if path.name.startswith("checkpoint-")
    )
    if checkpoint_dirs != list(CHECKPOINT_NAMES):
        raise Stage28Error(
            f"unexpected checkpoint ladder: {checkpoint_dirs} != {list(CHECKPOINT_NAMES)}"
        )
    checkpoint_1000 = _inspect_checkpoint(
        training_output / "checkpoint-1000", expected_step=1000
    )
    checkpoint_2000 = _inspect_checkpoint(
        training_output / "checkpoint-2000", expected_step=2000
    )
    if (
        expected_checkpoint_tree_sha256 is not None
        and checkpoint_2000["tree_sha256"] != expected_checkpoint_tree_sha256
    ):
        raise Stage28Error(
            "H2 checkpoint-2000 external tree SHA mismatch: "
            f"{checkpoint_2000['tree_sha256']} != {expected_checkpoint_tree_sha256}"
        )
    seal_path = Path(output_root) / "stage28_h2_checkpoint_2000_seal.json"
    observed_seal_sha256 = _sha256_file(seal_path)
    if expected_seal_sha256 is not None and observed_seal_sha256 != expected_seal_sha256:
        raise Stage28Error(
            "H2 checkpoint-2000 external seal SHA mismatch: "
            f"{observed_seal_sha256} != {expected_seal_sha256}"
        )
    seal = _read_json(seal_path, "Stage28 final checkpoint seal")
    if seal.get("schema") != "STAGE28_H2_FINAL_CHECKPOINT_SEAL_V1" or seal.get("status") != "PASS":
        raise Stage28Error("Stage28 final checkpoint seal is not a PASS")
    if (
        expected_checkpoint_tree_sha256 is not None
        and seal.get("tree_sha256") != expected_checkpoint_tree_sha256
    ):
        raise Stage28Error("H2 checkpoint-2000 seal tree is not externally bound")
    if seal.get("tree_sha256") != checkpoint_2000["tree_sha256"]:
        raise Stage28Error("Stage28 final checkpoint tree changed after sealing")
    sealed_entries = seal.get("entries")
    if sealed_entries != checkpoint_2000["entries"]:
        raise Stage28Error("Stage28 final checkpoint file bytes changed after sealing")
    return {
        "checkpoint-1000": {key: value for key, value in checkpoint_1000.items() if key != "entries"},
        "checkpoint-2000": {key: value for key, value in checkpoint_2000.items() if key != "entries"},
        "seal": {
            "path": str(seal_path),
            "sha256": observed_seal_sha256,
            "tree_sha256": seal["tree_sha256"],
            "file_count": seal["file_count"],
            "total_bytes": seal["total_bytes"],
        },
    }


def execute_training(args: argparse.Namespace) -> dict[str, Any]:
    _require_token(args)
    preflight = _load_and_revalidate_preflight(args)
    output_root = _resolve_non_symlink(args.output_root, "Stage28 output root")
    training_output = output_root / "h2_training"
    if training_output.exists():
        raise Stage28Error(f"refusing to reuse H2 training output: {training_output}")
    training_output.mkdir(parents=True, exist_ok=False)
    args.training_output_dir = training_output

    config = _build_training_config(args)
    config_summary = _config_summary(config)
    _assert_config_contract(config_summary, Path(args.base_model_path).resolve(), training_output)
    _write_json(
        output_root / "stage28_training_config.json",
        {
            "schema": STAGE28_SCHEMA,
            "status": "BOUND_BEFORE_TRAINING",
            "config": config_summary,
            "preflight_base_tree_sha256": preflight["base"]["tree_sha256"],
            "scientific_contract_changed": False,
        },
    )

    started = time.time()
    try:
        # This is the sole model/training entry point.  It is unreachable from
        # Corelab's plan/self-test/preflight modes.
        from gr00t.experiment.experiment import run

        run(config)
    except Exception as exc:
        _write_json(
            output_root / "stage28_h2_execution_manifest.json",
            {
                "schema": STAGE28_SCHEMA,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error_summary": str(exc).splitlines()[0][:500],
                "training_started": True,
                "scientific_contract_changed": False,
                "downstream_authorization_still_valid": True,
            },
        )
        raise

    # The official Trainer should have emitted both full-state checkpoints.
    # Do not turn a model-only or partial output into a successful handoff.
    checkpoint_1000 = _inspect_checkpoint(training_output / "checkpoint-1000", expected_step=1000)
    checkpoint_2000 = _inspect_checkpoint(training_output / "checkpoint-2000", expected_step=2000)
    seal = _seal_checkpoint(checkpoint_2000, output_root)
    checkpoints = _verify_final_checkpoint_artifacts(output_root)
    manifest = {
        "schema": STAGE28_SCHEMA,
        "status": "PASS",
        "stage": "Stage28 H2 fresh BASE 2000 fixed-warmup10 training",
        "recipe": "H2_FRESH_BASE_2000_FIXED_WARMUP10",
        "training_started": True,
        "training_completed": True,
        "elapsed_s": time.time() - started,
        "preflight": preflight,
        "config": config_summary,
        "checkpoint-1000": checkpoints["checkpoint-1000"],
        "checkpoint-2000": checkpoints["checkpoint-2000"],
        "final_seal": seal,
        "exact_resume_claim": False,
        "model_construction": True,
        "training": True,
        "backward": True,
        "optimizer_step": True,
        "gpu": True,
        "pbs_submitted": True,
        "validation_e_read": False,
        "holdout_f_read": False,
        "free_running_sampler": False,
        "c2": False,
        "wbc": False,
        "hardware": False,
        "scientific_contract_changed": False,
        "downstream_authorization_still_valid": True,
        "next_action": "run exactly one BASE/H2@2000 fixed-probe evaluation, then stop",
    }
    _write_json(output_root / "stage28_h2_execution_manifest.json", manifest)
    return manifest


def _load_stage26_hash_manifest(path: Path) -> dict[str, Any]:
    path = _resolve_non_symlink(path, "Stage26 hash manifest")
    observed = _sha256_file(path)
    if observed != EXPECTED_STAGE26_HASH_MANIFEST_SHA256:
        raise Stage28Error(
            f"Stage26 hash manifest SHA mismatch: {observed} != "
            f"{EXPECTED_STAGE26_HASH_MANIFEST_SHA256}"
        )
    payload = _read_json(path, "Stage26 hash manifest")
    if payload.get("schema") != "STAGE26_H1_C1_FIXED_NOISE_LOCAL_FLOW_V1":
        raise Stage28Error("Stage26 hash manifest schema mismatch")
    rows = payload.get("raw_velocity_hashes")
    if not isinstance(rows, list):
        raise Stage28Error("Stage26 raw velocity hashes must be a list")
    if len(rows) != EXPECTED_STAGE26_MANIFEST_ROW_COUNT:
        raise Stage28Error(
            "Stage26 hash manifest must contain exactly eighteen rows"
        )
    base_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Stage28Error("Stage26 hash manifest row is not an object")
        if row.get("checkpoint") == "BASE":
            base_rows.append(row)
        elif row.get("checkpoint") != "O1@200":
            raise Stage28Error("Stage26 hash manifest has an unknown checkpoint")
    if len(base_rows) != EXPECTED_PROBE_COUNT:
        raise Stage28Error(
            "Stage26 BASE hash manifest does not contain exactly nine BASE rows"
        )
    normalized: dict[str, str] = {}
    for row in base_rows:
        probe_id = row.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            raise Stage28Error("Stage26 BASE probe ID is invalid")
        digest = row.get("raw_predicted_velocity_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Stage28Error("Stage26 BASE raw hash is invalid")
        if probe_id in normalized:
            raise Stage28Error("Stage26 BASE hash manifest has duplicate probes")
        normalized[probe_id] = digest
    observed_ids = set(normalized)
    expected_ids = set(EXPECTED_STAGE26_PROBE_IDS)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        foreign = sorted(observed_ids - expected_ids)
        raise Stage28Error(
            "Stage26 BASE hash manifest probe set mismatch: "
            f"missing={missing}, foreign={foreign}"
        )
    return {"path": str(path), "sha256": observed, "raw_velocity_hashes": normalized}


def _h2_flow_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_probe_ids: set[str] = set()
    for probe in _probe_contract()["probes"]:
        expected_probe_ids.add(str(probe["probe_id"]))
    by_probe: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        probe_id = str(record.get("probe_id"))
        checkpoint = str(record.get("checkpoint"))
        by_probe.setdefault(probe_id, {})[checkpoint] = record
    if set(by_probe) != expected_probe_ids:
        raise Stage28Error("H2 paired evidence does not contain exactly nine probes")
    paired: list[dict[str, Any]] = []
    ratios: list[float] = []
    improved = 0
    for probe_id in sorted(expected_probe_ids):
        pair = by_probe[probe_id]
        if set(pair) != {"BASE", "H2@2000"}:
            raise Stage28Error(f"incomplete BASE/H2 pair for {probe_id}: {sorted(pair)}")
        base_mse = float(pair["BASE"]["flow_metrics"]["active_masked_mse"])
        h2_mse = float(pair["H2@2000"]["flow_metrics"]["active_masked_mse"])
        if not math.isfinite(base_mse) or not math.isfinite(h2_mse) or base_mse <= 0.0:
            raise Stage28Error(f"non-finite or non-positive flow MSE for {probe_id}")
        ratio = h2_mse / base_mse
        ratios.append(ratio)
        if h2_mse < base_mse:
            improved += 1
        paired.append(
            {
                "probe_id": probe_id,
                "base_mse": base_mse,
                "h2_mse": h2_mse,
                "h2_over_base_ratio": ratio,
                "improved": h2_mse < base_mse,
            }
        )
    return {
        "paired": paired,
        "improved_count_over_9": improved,
        "median_ratio": median(ratios),
        "geometric_mean_ratio": geometric_mean(ratios),
        "descriptive_only": True,
        "no_scientific_pass_threshold": True,
    }


def _base_hash_reproduction(
    records: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    observed = {
        str(row["probe_id"]): str(row["raw_predicted_velocity_sha256"])
        for row in records
        if row.get("checkpoint") == "BASE"
    }
    expected = dict(baseline["raw_velocity_hashes"])
    differences = [
        {
            "probe_id": probe_id,
            "expected": expected.get(probe_id),
            "observed": observed.get(probe_id),
        }
        for probe_id in sorted(set(expected) | set(observed))
        if expected.get(probe_id) != observed.get(probe_id)
    ]
    return {
        "status": (
            "MATCH"
            if len(records) == EXPECTED_PROBE_COUNT and not differences
            else "MISMATCH_REPORTED"
        ),
        "stage26_manifest_sha256": baseline["sha256"],
        "expected_record_count": EXPECTED_PROBE_COUNT,
        "observed_record_count": len(records),
        "differences": differences,
        "normalization_or_repair_applied": False,
    }


def _raw_velocity_hashes(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": row["checkpoint"],
            "probe_id": row["probe_id"],
            "raw_predicted_velocity_sha256": row["raw_predicted_velocity_sha256"],
        }
        for row in records
    ]


def _historical_training_provenance() -> dict[str, Any]:
    return {
        "job": "600695.hopper-m-02",
        "git_sha": EXPECTED_H2_TRAINING_SOURCE_SHA,
        "runner_sha256": EXPECTED_H2_TRAINING_RUNNER_SHA256,
        "retrained": False,
    }


def _run_h2_after_base_hash_gate(
    *,
    base_records: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    on_mismatch: Callable[[Mapping[str, Any]], None],
    h2_runner: Callable[[], list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run H2 only after exact BASE reproduction has passed."""

    reproduction = _base_hash_reproduction(base_records, baseline)
    if reproduction["status"] != "MATCH":
        on_mismatch(reproduction)
        raise Stage28Error(
            "Stage28 BASE hash reproduction mismatch; H2 evaluations=0"
        )
    return reproduction, h2_runner()


def _write_base_hash_mismatch_evidence(
    *,
    output_root: Path,
    project_head: str,
    evaluation_provenance: Mapping[str, Any],
    training_artifacts: Mapping[str, Any],
    base_identity: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    modality_sha: str,
    source_identity: Mapping[str, Any],
    cosmos_identity: Mapping[str, Any],
    snapshot_identity: Mapping[str, Any],
    baseline: Mapping[str, Any],
    probe_contract: Mapping[str, Any],
    base_records: Sequence[Mapping[str, Any]],
    base_hash_reproduction: Mapping[str, Any],
) -> Path:
    raw_hashes = _raw_velocity_hashes(base_records)
    evidence = {
        "schema": STAGE28_SCHEMA,
        "status": "STOP_OR_BLOCKED",
        "stage": "Stage28 H2 BASE/H2@2000 fixed local-flow evaluation",
        "reason": "BASE_HASH_REPRODUCTION_MISMATCH",
        "identity": {
            "stage28_git_sha": project_head,
            "evaluation_provenance": dict(evaluation_provenance),
            "historical_training_provenance": _historical_training_provenance(),
            "base": dict(base_identity),
            "h2_training_artifacts": dict(training_artifacts),
            "dataset": dict(dataset_identity),
            "modality_sha256": modality_sha,
            "groot": dict(source_identity),
            "cosmos": dict(cosmos_identity),
            "selected_snapshot": dict(snapshot_identity),
            "stage26_baseline": dict(baseline),
        },
        "probe_contract": {
            **dict(probe_contract),
            "same_fixed_probes_reused_for": ["BASE", "H2@2000"],
        },
        "base_records": list(base_records),
        "h2_records": [],
        "base_hash_reproduction": dict(base_hash_reproduction),
        "raw_velocity_hashes": raw_hashes,
        "execution": {
            "evaluation_only": True,
            "training_run_count": 0,
            "historical_training_run_count": 1,
            "base_flow_evaluation_count": len(base_records),
            "h2_flow_evaluation_count": 0,
            "flow_evaluation_count": len(base_records),
            "maximum_flow_evaluation_count": EXPECTED_MAX_FLOW_EVALUATIONS,
            "h2_evaluations_started": False,
            "complete_model_forward_called": False,
            "teacher_forced_action_head_path": True,
            "free_running_sampler_called": False,
            "get_action_called": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "pbs_submitted": True,
            "pbs_job_id": evaluation_provenance["pbs_job_id"],
            "pbs_project": evaluation_provenance["pbs_project"],
            "validation_e_read": False,
            "holdout_f_read": False,
            "c2_started": False,
            "wbc_executed": False,
            "hardware_executed": False,
        },
        "scientific_contract_changed": False,
        "downstream_authorization_still_valid": True,
        "claim_boundary": "BASE control evidence only; H2 comparison was withheld",
        "next_gate": EVALUATION_NEXT_GATE,
    }
    evidence_path = output_root / BASE_HASH_MISMATCH_EVIDENCE_NAME
    _write_json(evidence_path, _jsonable(evidence))
    _write_json(
        output_root / "stage28_h2_raw_velocity_hashes.json",
        {"schema": STAGE28_SCHEMA, "raw_velocity_hashes": raw_hashes},
    )
    return evidence_path


def evaluate_fixed_probes(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_provenance = _require_evaluation_provenance(args)
    project_repo = _resolve_non_symlink(args.project_repo, "Stage28 project checkout")
    project_head = _git_head(project_repo)
    if args.expected_project_sha and project_head != args.expected_project_sha:
        raise Stage28Error(
            f"Stage28 checkout SHA mismatch: {project_head} != {args.expected_project_sha}"
        )
    if _git_status(project_repo):
        raise Stage28Error("Stage28 project checkout is dirty")
    output_root = _resolve_non_symlink(args.output_root, "Stage28 output root")
    execution_manifest = _read_json(
        output_root / "stage28_h2_execution_manifest.json",
        "Stage28 execution manifest",
    )
    if execution_manifest.get("schema") != STAGE28_SCHEMA or execution_manifest.get("status") != "PASS":
        raise Stage28Error("Stage28 training execution is not a PASS")
    training_artifacts = _verify_final_checkpoint_artifacts(
        output_root,
        expected_checkpoint_tree_sha256=args.expected_h2_checkpoint_tree_sha256,
        expected_seal_sha256=args.expected_h2_checkpoint_seal_sha256,
    )
    base_identity = _verify_base_checkpoint(args.base_model_path)
    dataset_identity = _verify_tiny_dataset(args.dataset_dir)
    modality_sha = _verify_modality(args.modality_config)
    source_identity = _verify_groot_source(args.groot_source)
    cosmos_identity = _verify_cosmos_cache(args.cosmos_cache_root)
    baseline = _load_stage26_hash_manifest(args.stage26_hash_manifest)
    stage26 = _load_stage26_module()
    if str(args.device) != "cuda:0":
        raise Stage28Error("Stage28 evaluation is frozen to one explicit CUDA device: cuda:0")
    if not args.device.startswith("cuda"):
        raise Stage28Error("Stage28 evaluation requires CUDA")

    # Stage26 performs all runtime imports and model construction below this
    # point.  Its input processor and direct-flow helper are the frozen graph.
    stage23 = stage26._load_stage23_protocol()
    probe_contract = stage26._static_probe_contract()
    stage23.assert_protocol_static(stage23.protocol_plan())
    snapshot_identity, snapshot_path = stage26._verify_snapshot_identity(args.snapshot_root)
    snapshot = stage26._load_snapshot(snapshot_path)
    processor, embodiment, modality_configs = stage26._build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=args.base_model_path,
        dataset_dir=args.dataset_dir,
        modality_path=args.modality_config,
        cosmos_snapshot=Path(cosmos_identity["snapshot"]),
    )
    prepared = stage26._prepare_official_sample(
        processor=processor,
        embodiment=embodiment,
        modality_configs=modality_configs,
        dataset_dir=args.dataset_dir,
    )
    fixed_probes = tuple(
        stage26._canonical_probe_inputs(stage23, prepared["expert_action_cpu"], probe)
        for probe in probe_contract["probes"]
    )
    base_records = stage26._run_one_checkpoint(
        checkpoint_label="BASE",
        checkpoint=args.base_model_path,
        groot_source=args.groot_source,
        prepared=prepared,
        processor=processor,
        embodiment=embodiment,
        stage23=stage23,
        probes=probe_contract["probes"],
        fixed_probes=fixed_probes,
        device=args.device,
        snapshot=snapshot,
    )

    def run_h2_records() -> list[dict[str, Any]]:
        h2_checkpoint = output_root / "h2_training" / "checkpoint-2000"
        return stage26._run_one_checkpoint(
            checkpoint_label="H2@2000",
            checkpoint=h2_checkpoint,
            groot_source=args.groot_source,
            prepared=prepared,
            processor=processor,
            embodiment=embodiment,
            stage23=stage23,
            probes=probe_contract["probes"],
            fixed_probes=fixed_probes,
            device=args.device,
            snapshot=snapshot,
        )

    def write_base_mismatch_evidence(
        reproduction: Mapping[str, Any],
    ) -> None:
        _write_base_hash_mismatch_evidence(
            output_root=output_root,
            project_head=project_head,
            evaluation_provenance=evaluation_provenance,
            training_artifacts=training_artifacts,
            base_identity=base_identity,
            dataset_identity=dataset_identity,
            modality_sha=modality_sha,
            source_identity=source_identity,
            cosmos_identity=cosmos_identity,
            snapshot_identity=snapshot_identity,
            baseline=baseline,
            probe_contract=probe_contract,
            base_records=base_records,
            base_hash_reproduction=reproduction,
        )

    base_hash_reproduction, h2_records = _run_h2_after_base_hash_gate(
        base_records=base_records,
        baseline=baseline,
        on_mismatch=write_base_mismatch_evidence,
        h2_runner=run_h2_records,
    )
    records = [*base_records, *h2_records]
    if len(records) != EXPECTED_MAX_FLOW_EVALUATIONS:
        raise Stage28Error(f"Stage28 flow evaluation count drifted: {len(records)}")
    aggregates = _h2_flow_aggregates(records)
    raw_hashes = _raw_velocity_hashes(records)
    result = {
        "schema": STAGE28_SCHEMA,
        "status": "PASS_EXECUTION",
        "stage": "Stage28 H2 BASE/H2@2000 fixed local-flow evaluation",
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "identity": {
            "stage28_git_sha": project_head,
            "evaluation_provenance": evaluation_provenance,
            "historical_training_provenance": _historical_training_provenance(),
            "base": base_identity,
            "h2_training_artifacts": training_artifacts,
            "dataset": dataset_identity,
            "modality_sha256": modality_sha,
            "groot": source_identity,
            "cosmos": cosmos_identity,
            "selected_snapshot": snapshot_identity,
            "stage26_baseline": baseline,
        },
        "probe_contract": {
            **probe_contract,
            "same_fixed_probes_reused_for": ["BASE", "H2@2000"],
        },
        "records": records,
        "paired_aggregates": aggregates,
        "base_hash_reproduction": base_hash_reproduction,
        "raw_velocity_hashes": raw_hashes,
        "execution": {
            "evaluation_only": True,
            "training_run_count": 0,
            "historical_training_run_count": 1,
            "base_flow_evaluation_count": len(base_records),
            "h2_flow_evaluation_count": len(h2_records),
            "flow_evaluation_count": len(records),
            "maximum_flow_evaluation_count": EXPECTED_MAX_FLOW_EVALUATIONS,
            "h2_evaluations_started": True,
            "complete_model_forward_called": False,
            "teacher_forced_action_head_path": True,
            "free_running_sampler_called": False,
            "get_action_called": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "pbs_submitted": True,
            "pbs_job_id": evaluation_provenance["pbs_job_id"],
            "pbs_project": evaluation_provenance["pbs_project"],
            "validation_e_read": False,
            "holdout_f_read": False,
            "c2_started": False,
            "wbc_executed": False,
            "hardware_executed": False,
            "device": args.device,
        },
        "scientific_contract_changed": False,
        "downstream_authorization_still_valid": True,
        "claim_boundary": (
            "paired teacher-forced local-flow evidence on the single TINY1 training "
            "sample; no sampler, task, generalization, WBC, or hardware claim"
        ),
        "next_gate": EVALUATION_NEXT_GATE,
    }
    _write_json(output_root / "stage28_h2_local_flow_result.json", _jsonable(result))
    _write_json(
        output_root / "stage28_h2_raw_velocity_hashes.json",
        {"schema": STAGE28_SCHEMA, "raw_velocity_hashes": raw_hashes},
    )
    return result


def _jsonable(value: Any) -> Any:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--project-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-project-sha", default=None)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--stage10-root", type=Path, default=DEFAULT_STAGE10_ROOT)
    parser.add_argument("--groot-source", type=Path, default=None)
    parser.add_argument("--modality-config", type=Path, default=DEFAULT_MODALITY_PATH)
    parser.add_argument("--cosmos-cache-root", type=Path, default=None)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--stage26-hash-manifest", type=Path, default=DEFAULT_STAGE26_HASH_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/stage28_h2"))
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-h2-checkpoint-tree-sha256", default=None)
    parser.add_argument("--expected-h2-checkpoint-seal-sha256", default=None)
    parser.add_argument("--training-source-sha", default=None)
    parser.add_argument("--training-runner-sha256", default=None)
    parser.add_argument("--expected-evaluation-runner-sha256", default=None)
    parser.add_argument("--evaluation-launcher-sha256", default=None)
    parser.add_argument("--pbs-job-id", default=None)
    parser.add_argument("--pbs-project", default=None)
    parser.add_argument("--training-output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.groot_source is None:
        args.groot_source = args.stage10_root / "src/isaac-gr00t"
    if args.cosmos_cache_root is None:
        args.cosmos_cache_root = args.stage10_root / "cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"

    try:
        if args.plan_only or not any(
            (args.self_test, args.preflight, args.train, args.evaluate)
        ):
            plan = protocol_plan()
            assert_protocol_static(plan)
            print(json.dumps(_jsonable(plan), indent=2, sort_keys=True))
            return 0
        if args.self_test:
            plan = protocol_plan()
            assert_protocol_static(plan)
            print(
                json.dumps(
                    {
                        "schema": STAGE28_SCHEMA,
                        "status": "SELF_TEST_PASS",
                        "training_started": False,
                        "probe_count": EXPECTED_PROBE_COUNT,
                        "maximum_flow_evaluations": EXPECTED_MAX_FLOW_EVALUATIONS,
                    },
                    indent=2,
                )
            )
            return 0
        if args.preflight:
            result = run_preflight(args)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return 0
        if args.train:
            result = execute_training(args)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return 0
        if args.evaluate:
            result = evaluate_fixed_probes(args)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return 0
    except Exception as exc:
        failure = {
            "schema": STAGE28_SCHEMA,
            "status": "STOP_OR_BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "training_started": False,
            "model_construction": False,
            "model_forward": False,
            "gpu": False,
            "pbs_submitted": False,
            "scientific_contract_changed": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    raise AssertionError("unreachable Stage28 mode")


if __name__ == "__main__":
    raise SystemExit(main())
