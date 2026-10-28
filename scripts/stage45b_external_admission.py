"""Shared fail-closed checks for the PI0087 external Stage45B admission seal."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping

from stage45b_groot_source_snapshot import canonical_sha256, sha256_file, verify_snapshot
from stage45b_video_decoder_runtime import (
    EXPECTED_FFMPEG_RUNTIME_SCHEMA,
    EXPECTED_VIDEO_SCHEMA,
    EXPECTED_VIDEO_TOKEN,
    verify_runtime_artifact,
)


PI_GATE = "PI_GATE_STATE_0087"
SCHEMA = "VANDA_STAGE45B_PI0087_EXTERNAL_ADMISSION_BUNDLE_V1"
STATUS = "VANDA_STAGE45B_PI0087_FINAL_ADMISSION_QUALIFIED"
BRANCH = "stage45b/full24-fresh-base-execution"
STAGE = "Stage45B"
SOURCE_HEAD_PLACEHOLDER = "SOURCE_HEAD_BOUND_AFTER_FINAL_COMMIT"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_FILE_COUNT = 85
FULL24_BYTES = 919661
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
RUNTIME_STATS_SCHEMA = "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2"
RUNTIME_PROTOCOL_REVISION = "RUNTIME_PARQUET_PROVENANCE_V2"
RUNTIME_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
B0_STATS_SHA256 = "49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d"
B0_PAYLOAD_SHA256 = "c31c818121ac01a402fc70e577db225ffb6f5d110649aef9dada94338c8a5032"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_TREE = "43617c11aa6c4ea1be47594882d797255c1a575b"
GROOT_SNAPSHOT_SCHEMA = "STAGE45B_GROOT_SOURCE_SNAPSHOT_V2"
GROOT_ARCHIVE_SHA256 = "705681fece08cb8a7c1963d4531464c90b4b031f2ab5aa2c8e25e74676310939"
GROOT_MANIFEST_FILE_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA256 = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
PRODUCTION_RELEASE_HELPER_PATH = "scripts/stage45b_production_release.py"
PRODUCTION_RELEASE_SCHEMA = "STAGE45B_B2_TRAINING_RELEASE_V2"
PRODUCTION_RELEASE_TOKEN = "VANDA_STAGE45B_B2_PRODUCTION_TRAINING_RELEASE_V2_QUALIFIED"
VIDEO_RUNTIME_HELPER_PATH = "scripts/stage45b_video_decoder_runtime.py"
VIDEO_RUNTIME_QUALIFICATION_TOKEN = "VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED"
RUNTIME_DATASET_QUALIFICATION_SHA256 = "8ec632bfb41af09464b7d6e546a2a7427d5c72035c475d048e6b7e3e360b4f1a"
FFMPEG_RUNTIME_ARTIFACT_ID = "9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"
FFMPEG_RUNTIME_MANIFEST_SHA256 = "6160d579b98dc5ee4b114ada4a4e00e88ee13df6ed1fc3d308b5879d67027d2c"


def read_json(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"required JSON is unavailable: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _regular_file(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _relative_execution_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"execution file path is not repository-relative: {value}")
    return path


def _require_equal(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if mapping.get(key) != expected:
        raise RuntimeError(f"{label}.{key} mismatch: {mapping.get(key)!r}")


def _expected_contract() -> dict[str, Any]:
    return {
        "active_loss_dimensions": 10,
        "body_rgb_shape": [96, 96, 3],
        "d1_decoder_changed": False,
        "decoded_physical_action_shape": [40, 10],
        "geometry": "K16",
        "horizon": 40,
        "language": "move the end effector to the red target",
        "masked_padding_dimensions": 122,
        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "safety_gate_changed": False,
        "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
        "state_v1_shape": [10],
        "structured_action_schema": "StructuredActionV1",
        "target_xyz_in_model_input": False,
        "tcp_semantic_id": "x5_link6_gripper_center",
        "timing": "T16",
        "universal_action_shape": [1, 40, 132],
        "wbc_changed": False,
        "wrist_rgb_shape": [96, 96, 3],
    }


def _expected_recipe() -> dict[str, Any]:
    return {
        "bf16": True,
        "dataloader_num_workers": 0,
        "eval_strategy": "no",
        "fp16": False,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "max_steps": 2000,
        "num_gpus": 1,
        "optimizer": "adamw_torch",
        "per_device_train_batch_size": 1,
        "resume_from_checkpoint": False,
        "save_only_model": False,
        "save_steps": [1000, 2000],
        "save_total_limit": 2,
        "seed": 2222,
        "tf32": True,
        "tune_diffusion_model": True,
        "tune_llm": False,
        "tune_projector": True,
        "tune_visual": False,
        "tune_vlln": False,
        "use_percentiles": True,
        "use_relative_action": True,
        "warmup_ratio": 0.0,
        "warmup_steps": 10,
        "weight_decay": 0.00001,
    }


def assert_bundle(bundle: Mapping[str, Any], *, repo: Path, expected_head: str, bundle_path: Path) -> dict[str, Any]:
    if bundle.get("schema") != SCHEMA:
        raise RuntimeError("PI0087 external admission schema mismatch")
    if bundle.get("status") != STATUS or bundle.get("pi_gate") != PI_GATE or bundle.get("stage") != STAGE:
        raise RuntimeError("PI0087 external admission status/gate mismatch")
    repository = bundle.get("repository")
    if not isinstance(repository, Mapping) or repository.get("branch") != BRANCH:
        raise RuntimeError("Stage45B branch binding mismatch")
    source_head = str(repository.get("source_head", ""))
    if len(source_head) != 40 or source_head == SOURCE_HEAD_PLACEHOLDER or source_head != expected_head:
        raise RuntimeError("external admission does not bind the exact final Stage45B HEAD")
    if _git(repo, "branch", "--show-current") != BRANCH:
        raise RuntimeError("current checkout is not the Stage45B execution branch")
    if _git(repo, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("current checkout HEAD differs from external admission")
    if _git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("Stage45B checkout is dirty")
    if bundle_path.is_symlink():
        raise RuntimeError("external admission bundle must not be a symlink")
    if bundle_path.lstat().st_mode & 0o222:
        raise RuntimeError("external admission bundle must be immutable")

    files = bundle.get("execution_files")
    if not isinstance(files, Mapping):
        raise RuntimeError("external admission execution file binding is missing")
    required_files = (
        "runner",
        "pbs",
        "premodel_runtime_smoke",
        "groot_source_snapshot_helper",
        "external_admission_helper",
        "production_release_helper",
        "runtime_qualification",
        "video_decoder_runtime",
    )
    for key in required_files:
        item = files.get(key)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"missing execution file binding: {key}")
        relative = _relative_execution_path(str(item.get("path", "")))
        path = repo / relative
        _regular_file(path, f"execution file {key}")
        observed = sha256_file(path)
        if observed != item.get("sha256"):
            raise RuntimeError(f"execution file hash mismatch: {key}")

    video_runtime = bundle.get("video_decoder_runtime")
    if not isinstance(video_runtime, Mapping):
        raise RuntimeError("native video decoder runtime qualification binding is missing")
    for key, expected in (
        ("schema", EXPECTED_VIDEO_SCHEMA),
        ("token", EXPECTED_VIDEO_TOKEN),
        ("status", "PASS"),
        ("pi_gate", PI_GATE),
        ("ffmpeg_runtime_schema", EXPECTED_FFMPEG_RUNTIME_SCHEMA),
        ("ffmpeg_runtime_artifact_id", FFMPEG_RUNTIME_ARTIFACT_ID),
        ("ffmpeg_runtime_manifest_sha256", FFMPEG_RUNTIME_MANIFEST_SHA256),
        ("runtime_dataset_qualification_sha256", RUNTIME_DATASET_QUALIFICATION_SHA256),
    ):
        _require_equal(video_runtime, key, expected, "video_decoder_runtime")
    video_evidence_path = Path(str(video_runtime.get("qualification_path", "")))
    _regular_file(video_evidence_path, "native video decoder qualification evidence")
    if video_evidence_path.lstat().st_mode & 0o222:
        raise RuntimeError("native video decoder qualification evidence must be immutable")
    if sha256_file(video_evidence_path) != video_runtime.get("qualification_sha256"):
        raise RuntimeError("native video decoder qualification evidence hash mismatch")
    video_evidence = read_json(video_evidence_path)
    if video_evidence.get("schema") != EXPECTED_VIDEO_SCHEMA or video_evidence.get("status") != "PASS" or video_evidence.get("token") != EXPECTED_VIDEO_TOKEN:
        raise RuntimeError("native video decoder qualification evidence is not PASS")
    if video_evidence.get("execution_head") != expected_head or video_evidence.get("branch") != BRANCH:
        raise RuntimeError("native video decoder qualification is bound to a different Stage45B checkout")
    artifact = video_evidence.get("ffmpeg_runtime")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("native video decoder qualification has no FFmpeg artifact identity")
    if artifact.get("artifact_id") != FFMPEG_RUNTIME_ARTIFACT_ID or artifact.get("manifest_sha256") != FFMPEG_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("native video decoder qualification FFmpeg identity changed")
    verify_runtime_artifact(Path(str(artifact.get("root", ""))), expected_id=FFMPEG_RUNTIME_ARTIFACT_ID)
    exposure = video_evidence.get("scientific_exposure")
    if not isinstance(exposure, Mapping):
        raise RuntimeError("native video decoder qualification scientific boundary is missing")
    for key in ("model_constructed", "forward", "backward", "optimizer", "trainer", "checkpoint_creation", "pbs_submissions", "validation_e_read", "holdout_f_read", "wbc", "simulator", "hardware"):
        if exposure.get(key) not in (False, 0):
            raise RuntimeError(f"native video decoder qualification crossed boundary: {key}")
    video_sweep = video_evidence.get("video_sweep")
    if not isinstance(video_sweep, Mapping) or video_sweep.get("video_file_count") != 48 or video_sweep.get("total_frames") != 1920:
        raise RuntimeError("native video decoder qualification did not cover FULL24 video frames")
    if video_sweep.get("files_per_modality") != {"external_rgb": 24, "wrist_rgb": 24}:
        raise RuntimeError("native video decoder qualification did not cover both RGB modalities")
    production_loader = video_evidence.get("production_loader")
    if not isinstance(production_loader, Mapping):
        raise RuntimeError("production loader qualification is missing")
    loader = production_loader.get("dataloader", {})
    if loader.get("num_workers") != 0 or loader.get("worker_subprocesses_before") or loader.get("worker_subprocesses_after"):
        raise RuntimeError("production loader worker semantics changed")
    if production_loader.get("structured_action_label_count") != 24:
        raise RuntimeError("production loader did not retain FULL24 StructuredAction labels")

    snapshot = bundle.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("GR00T source snapshot binding is missing")
    for key, expected in (
        ("schema", GROOT_SNAPSHOT_SCHEMA),
        ("origin_repository", "NVIDIA/Isaac-GR00T"),
        ("origin_commit", GROOT_COMMIT),
        ("origin_tree", GROOT_TREE),
        ("source_route", "HISTORICAL_LOCAL"),
    ):
        _require_equal(snapshot, key, expected, "source_snapshot")
    for key in (
        "archive_path",
        "manifest_path",
        "archive_sha256",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "payload_tree_sha256",
    ):
        if not isinstance(snapshot.get(key), str) or not snapshot[key]:
            raise RuntimeError(f"source_snapshot.{key} is missing")
    for key, expected in (
        ("archive_sha256", GROOT_ARCHIVE_SHA256),
        ("manifest_file_sha256", GROOT_MANIFEST_FILE_SHA256),
        ("manifest_payload_sha256", GROOT_MANIFEST_PAYLOAD_SHA256),
        ("payload_tree_sha256", GROOT_PAYLOAD_TREE_SHA256),
    ):
        _require_equal(snapshot, key, expected, "source_snapshot")

    dataset = bundle.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("FULL24 identity binding is missing")
    for key, expected in (
        ("tree_sha256", FULL24_TREE_SHA256),
        ("regular_file_count", FULL24_FILE_COUNT),
        ("total_bytes", FULL24_BYTES),
        ("episode_count", 24),
        ("frame_count", 960),
        ("window_count", 24),
        ("role", "TRAIN_ONLY"),
        ("new_data_generation", False),
        ("validation_e_read", False),
        ("holdout_f_read", False),
        ("simulator_read", False),
    ):
        _require_equal(dataset, key, expected, "dataset")

    stats = bundle.get("structured_stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError("runtime-parquet V2 stats binding is missing")
    for key, expected in (
        ("schema", RUNTIME_STATS_SCHEMA),
        ("protocol_revision", RUNTIME_PROTOCOL_REVISION),
        ("sha256", RUNTIME_STATS_SHA256),
        ("payload_sha256", RUNTIME_PAYLOAD_SHA256),
        ("source", "SEALED_FULL24_TRAIN_PARQUET"),
        ("label_count", 24),
    ):
        _require_equal(stats, key, expected, "structured_stats")

    historical = bundle.get("historical_b0_stats")
    if not isinstance(historical, Mapping):
        raise RuntimeError("historical B0 binding is missing")
    for key, expected in (
        ("sha256", B0_STATS_SHA256),
        ("payload_sha256", B0_PAYLOAD_SHA256),
        ("execution_stats_authority", False),
        ("role", "IMMUTABLE_HISTORICAL_PROVENANCE_ONLY"),
    ):
        _require_equal(historical, key, expected, "historical_b0_stats")

    lineage = bundle.get("model_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("model lineage binding is missing")
    for key, expected in (
        ("base_tree_sha256", BASE_TREE_SHA256),
        ("cosmos_revision", COSMOS_REVISION),
        ("cosmos_tree_sha256", COSMOS_TREE_SHA256),
        ("modality_sha256", MODALITY_SHA256),
        ("config_sha256", CONFIG_SHA256),
        ("groot_source_commit", GROOT_COMMIT),
        ("groot_source_tree", GROOT_TREE),
    ):
        _require_equal(lineage, key, expected, "model_lineage")

    if bundle.get("intervention") != "B_FRESH_BASE_FULL24":
        raise RuntimeError("non-primary intervention is forbidden")
    if bundle.get("contract") != _expected_contract():
        raise RuntimeError("frozen Stage45B contract changed")
    if bundle.get("recipe") != _expected_recipe():
        raise RuntimeError("frozen Stage45B training recipe changed")
    anchor = bundle.get("label_anchor_reconciliation")
    expected_anchor = {
        "enabled": True,
        "mode": "LABEL_ONLY_STATE_EEF_ANCHOR_TO_PHYSICAL_ACTION_POSE0",
        "normal_codec_position_tolerance_m": 1e-09,
        "normal_codec_rotation_tolerance_rad": 1e-09,
        "observation_state_untouched": True,
        "physical_action_untouched": True,
        "provenance_position_tolerance_m": 1e-06,
        "provenance_rotation_tolerance_rad": 1e-06,
        "state_gripper_preserved": True,
    }
    if anchor != expected_anchor:
        raise RuntimeError("label-anchor reconciliation changed")
    forbidden = bundle.get("forbidden")
    if not isinstance(forbidden, Mapping):
        raise RuntimeError("forbidden execution branches are missing")
    for key in ("validation_e_read", "holdout_f_read", "stage43_rerun", "vanda_a40", "wbc_or_simulator"):
        if forbidden.get(key) is not True:
            raise RuntimeError(f"forbidden branch is not locked: {key}")

    lock = bundle.get("scientific_lock")
    if not isinstance(lock, Mapping) or not isinstance(lock.get("projection"), Mapping):
        raise RuntimeError("tracked scientific projection lock is missing")
    if canonical_sha256(lock["projection"]) != lock.get("projection_sha256"):
        raise RuntimeError("tracked scientific projection hash mismatch")
    tracked = lock.get("tracked_manifest")
    if not isinstance(tracked, Mapping):
        raise RuntimeError("tracked manifest lock is missing")
    tracked_path = Path(str(tracked.get("path", ""))).resolve()
    _regular_file(tracked_path, "tracked execution manifest")
    if sha256_file(tracked_path) != tracked.get("sha256"):
        raise RuntimeError("tracked execution manifest hash mismatch")
    if tracked.get("sha256") != repository.get("tracked_manifest_sha256"):
        raise RuntimeError("tracked manifest hash binding mismatch")
    training_config = bundle.get("training_config")
    if not isinstance(training_config, Mapping) or training_config.get("sha256") != CONFIG_SHA256:
        raise RuntimeError("training config identity binding is missing or changed")

    boundary = bundle.get("execution_boundary")
    if not isinstance(boundary, Mapping):
        raise RuntimeError("execution boundary is missing")
    for key in ("model_constructed", "gr00t_forwards", "training_steps", "optimizer_steps", "backward_steps", "pbs_submissions"):
        if boundary.get(key) not in {False, 0}:
            raise RuntimeError(f"pre-admission execution boundary is non-zero: {key}")
    tokens = bundle.get("required_tokens")
    expected_tokens = {
        "VANDA_STAGE45B_GROOT_CANONICAL_SOURCE_RECOVERED",
        "VANDA_STAGE45B_GROOT_SOURCE_SNAPSHOT_V2_SEALED",
        "VANDA_STAGE45B_GROOT_GITLESS_RUNTIME_QUALIFIED_V2",
        STATUS,
        VIDEO_RUNTIME_QUALIFICATION_TOKEN,
        PRODUCTION_RELEASE_TOKEN,
    }
    if not isinstance(tokens, list) or not expected_tokens.issubset(set(tokens)):
        raise RuntimeError("required PI0087 admission tokens are incomplete")
    return dict(bundle)


_PATH_KEYS = {
    "path",
    "path_template",
    "corelab_path",
    "h200_path_template",
    "corelab_source_path",
    "producer_path",
    "source_path_template",
    "groot_source_path_template",
    "base_path_template",
    "cosmos_path_template",
    "modality_path_template",
}


def _without_runtime_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_paths(item)
            for key, item in value.items()
            if str(key) not in _PATH_KEYS
        }
    if isinstance(value, list):
        return [_without_runtime_paths(item) for item in value]
    return value


def scientific_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stage",
        "intervention",
        "contract",
        "dataset",
        "structured_stats",
        "historical_b0_stats",
        "recipe",
        "trainable_modules",
        "label_anchor_reconciliation",
        "forbidden",
        "predecessors",
        "checkpoint_policy",
        "runtime_requirements",
        "model_lineage",
    )
    return _without_runtime_paths({key: manifest[key] for key in keys if key in manifest})


def _immutable_file_identity(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    mode = path.lstat().st_mode
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_bundle(
    *,
    repo: Path,
    output: Path,
    expected_head: str,
    snapshot_archive: Path,
    snapshot_manifest: Path,
    dataset: Path,
    base_model: Path,
    cosmos: Path,
    modality: Path,
    config: Path,
    stats: Path,
    video_runtime_evidence: Path,
    ffmpeg_runtime_root: Path,
) -> dict[str, Any]:
    if len(expected_head) != 40 or any(character not in "0123456789abcdef" for character in expected_head):
        raise RuntimeError("expected execution HEAD is not a lowercase commit SHA")
    manifest_path = repo / "manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json"
    manifest_identity = _immutable_file_identity(manifest_path, "tracked execution manifest")
    if manifest_identity["sha256"] != "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566":
        raise RuntimeError("tracked Stage45B manifest identity changed")
    manifest = read_json(manifest_path)
    projection = scientific_projection(manifest)
    execution_paths = {
        "runner": "scripts/stage45b_full24_fresh_base_train.py",
        "pbs": "hpc/pbs/stage45b_full24_fresh_base.pbs",
        "premodel_runtime_smoke": "scripts/stage45b_premodel_runtime_smoke.py",
        "groot_source_snapshot_helper": "scripts/stage45b_groot_source_snapshot.py",
        "external_admission_helper": "scripts/stage45b_external_admission.py",
        "production_release_helper": PRODUCTION_RELEASE_HELPER_PATH,
        "runtime_qualification": "scripts/stage45b_full24_runtime_parquet_qualification.py",
        "video_decoder_runtime": VIDEO_RUNTIME_HELPER_PATH,
    }
    execution_files = {
        key: {"path": path, "sha256": sha256_file(repo / path)}
        for key, path in execution_paths.items()
    }
    snapshot_archive = snapshot_archive.resolve()
    snapshot_manifest_path = snapshot_manifest.resolve()
    snapshot_verification = verify_snapshot(snapshot_archive, snapshot_manifest_path)
    if snapshot_verification["source_route"] != "HISTORICAL_LOCAL":
        raise RuntimeError("PI0087 final admission requires the recovered historical GR00T route")
    for key, expected in (
        ("archive_sha256", GROOT_ARCHIVE_SHA256),
        ("manifest_file_sha256", GROOT_MANIFEST_FILE_SHA256),
        ("manifest_payload_sha256", GROOT_MANIFEST_PAYLOAD_SHA256),
        ("payload_tree_sha256", GROOT_PAYLOAD_TREE_SHA256),
    ):
        if snapshot_verification[key] != expected:
            raise RuntimeError(f"sealed GR00T snapshot identity changed: {key}")
    snapshot_manifest = read_json(snapshot_manifest_path)
    snapshot = {
        "schema": snapshot_verification["schema"],
        "origin_repository": snapshot_verification["origin_repository"],
        "origin_commit": snapshot_verification["origin_commit"],
        "origin_tree": snapshot_verification["origin_tree"],
        "source_route": snapshot_verification["source_route"],
        "archive_path": str(snapshot_archive),
        "manifest_path": str(snapshot_manifest_path.resolve()),
        "archive_sha256": snapshot_verification["archive_sha256"],
        "manifest_file_sha256": snapshot_verification["manifest_file_sha256"],
        "manifest_payload_sha256": snapshot_verification["manifest_payload_sha256"],
        "payload_tree_sha256": snapshot_verification["payload_tree_sha256"],
        "tracked_entry_count": snapshot_verification["tracked_entry_count"],
        "regular_file_count": snapshot_verification["regular_file_count"],
        "symlink_count": snapshot_verification["symlink_count"],
        "gitlink_count": snapshot_verification["gitlink_count"],
    }
    stats_identity = _immutable_file_identity(stats, "runtime-parquet V2 stats")
    if stats_identity["sha256"] != RUNTIME_STATS_SHA256:
        raise RuntimeError("runtime-parquet V2 stats identity changed")
    stats_document = read_json(stats)
    if stats_document.get("statistics_payload_sha256") != RUNTIME_PAYLOAD_SHA256:
        raise RuntimeError("runtime-parquet V2 stats payload changed")
    _regular_file(video_runtime_evidence, "native video decoder qualification evidence")
    if video_runtime_evidence.lstat().st_mode & 0o222:
        raise RuntimeError("native video decoder qualification evidence must be immutable")
    video_evidence = read_json(video_runtime_evidence)
    if (
        video_evidence.get("schema") != EXPECTED_VIDEO_SCHEMA
        or video_evidence.get("status") != "PASS"
        or video_evidence.get("token") != EXPECTED_VIDEO_TOKEN
        or video_evidence.get("pi_gate") != PI_GATE
    ):
        raise RuntimeError("native video decoder qualification is not a PI0087 PASS")
    if video_evidence.get("execution_head") != expected_head or video_evidence.get("branch") != BRANCH:
        raise RuntimeError("native video decoder qualification is bound to a different checkout")
    artifact_identity = verify_runtime_artifact(
        ffmpeg_runtime_root, expected_id=FFMPEG_RUNTIME_ARTIFACT_ID
    )
    if artifact_identity["manifest_sha256"] != FFMPEG_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("FFmpeg runtime manifest identity changed")
    evidence_artifact = video_evidence.get("ffmpeg_runtime")
    if (
        not isinstance(evidence_artifact, Mapping)
        or evidence_artifact.get("artifact_id") != FFMPEG_RUNTIME_ARTIFACT_ID
        or evidence_artifact.get("manifest_sha256") != FFMPEG_RUNTIME_MANIFEST_SHA256
    ):
        raise RuntimeError("video evidence does not bind the sealed FFmpeg runtime")
    if Path(str(evidence_artifact.get("root", ""))).resolve() != ffmpeg_runtime_root.resolve():
        raise RuntimeError("video evidence FFmpeg path differs from bundle input")
    runtime_dataset_evidence = video_evidence.get("scientific_inputs", {}).get(
        "runtime_dataset_qualification", {}
    )
    if runtime_dataset_evidence.get("sha256") != RUNTIME_DATASET_QUALIFICATION_SHA256:
        raise RuntimeError("video evidence does not bind the existing 24/24 qualification")
    video_runtime_record = {
        "schema": EXPECTED_VIDEO_SCHEMA,
        "status": "PASS",
        "token": EXPECTED_VIDEO_TOKEN,
        "pi_gate": PI_GATE,
        "qualification_path": str(video_runtime_evidence.resolve()),
        "qualification_sha256": sha256_file(video_runtime_evidence),
        "ffmpeg_runtime_schema": EXPECTED_FFMPEG_RUNTIME_SCHEMA,
        "ffmpeg_runtime_root": str(ffmpeg_runtime_root.resolve()),
        "ffmpeg_runtime_artifact_id": artifact_identity["artifact_id"],
        "ffmpeg_runtime_manifest_sha256": artifact_identity["manifest_sha256"],
        "runtime_dataset_qualification_sha256": RUNTIME_DATASET_QUALIFICATION_SHA256,
        "torch_version": video_evidence.get("torch_runtime", {}).get("version"),
        "torchcodec_version": video_evidence.get("native_torchcodec", {}).get("version"),
        "torch_package_tree_sha256": video_evidence.get("torch_runtime", {}).get("package_identity", {}).get("tree_sha256"),
        "torchcodec_package_tree_sha256": video_evidence.get("native_torchcodec", {}).get("package_identity", {}).get("tree_sha256"),
        "video_file_count": video_evidence.get("video_sweep", {}).get("video_file_count"),
        "decoded_frame_count": video_evidence.get("video_sweep", {}).get("total_frames"),
        "production_loader_batches": video_evidence.get("production_loader", {}).get("dataloader", {}).get("batches_consumed"),
    }
    dataset_document = dict(manifest["dataset"])
    dataset_document.update(
        {
            "tree_sha256": FULL24_TREE_SHA256,
            "regular_file_count": FULL24_FILE_COUNT,
            "total_bytes": FULL24_BYTES,
            "runtime_path": str(dataset.resolve()),
            "video_file_count": 48,
            "video_frame_count": 1920,
            "video_modalities": ["external_rgb", "wrist_rgb"],
        }
    )
    bundle = {
        "schema": SCHEMA,
        "status": STATUS,
        "pi_gate": PI_GATE,
        "stage": STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intervention": "B_FRESH_BASE_FULL24",
        "repository": {
            "branch": BRANCH,
            "source_head": expected_head,
            "source_head_role": "EXACT_EXTERNAL_POST_COMMIT_EXECUTION_HEAD",
            "tracked_manifest_sha256": manifest_identity["sha256"],
        },
        "execution_files": execution_files,
        "video_decoder_runtime": video_runtime_record,
        "source_snapshot": snapshot,
        "dataset": dataset_document,
        "structured_stats": {
            **dict(manifest["structured_stats"]),
            "path": str(stats.resolve()),
            "sha256": stats_identity["sha256"],
            "payload_sha256": RUNTIME_PAYLOAD_SHA256,
        },
        "historical_b0_stats": dict(manifest["historical_b0_stats"]),
        "model_lineage": {
            "base_tree_sha256": BASE_TREE_SHA256,
            "cosmos_revision": COSMOS_REVISION,
            "cosmos_tree_sha256": COSMOS_TREE_SHA256,
            "modality_sha256": MODALITY_SHA256,
            "config_sha256": CONFIG_SHA256,
            "groot_source_commit": GROOT_COMMIT,
            "groot_source_tree": GROOT_TREE,
            "base_model_path": str(base_model.resolve()),
            "cosmos_root": str(cosmos.resolve()),
            "modality_path": str(modality.resolve()),
        },
        "contract": dict(manifest["contract"]),
        "recipe": dict(manifest["recipe"]),
        "label_anchor_reconciliation": dict(manifest["label_anchor_reconciliation"]),
        "forbidden": dict(manifest["forbidden"]),
        "training_config": {
            "path": str(config.resolve()),
            "sha256": CONFIG_SHA256,
        },
        "runtime_paths": {
            "base_model": str(base_model.resolve()),
            "dataset": str(dataset.resolve()),
            "cosmos_cache": str(cosmos.resolve()),
            "modality_config": str(modality.resolve()),
            "training_config": str(config.resolve()),
            "structured_stats_artifact": str(stats.resolve()),
            "video_runtime_qualification": str(video_runtime_evidence.resolve()),
            "ffmpeg_runtime_root": str(ffmpeg_runtime_root.resolve()),
        },
        "scientific_lock": {
            "tracked_manifest": manifest_identity,
            "projection_definition": "manifest science/protocol fields with runtime path keys removed",
            "projection": projection,
            "projection_sha256": canonical_sha256(projection),
        },
        "execution_boundary": {
            "model_constructed": False,
            "gr00t_forwards": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_steps": 0,
            "simulator": False,
            "hardware": False,
        },
        "required_tokens": [
            "VANDA_STAGE45B_GROOT_CANONICAL_SOURCE_RECOVERED",
            "VANDA_STAGE45B_GROOT_SOURCE_SNAPSHOT_V2_SEALED",
            "VANDA_STAGE45B_GROOT_GITLESS_RUNTIME_QUALIFIED_V2",
            STATUS,
            VIDEO_RUNTIME_QUALIFICATION_TOKEN,
            PRODUCTION_RELEASE_TOKEN,
        ],
        "runtime_requirements": {
            "base_model_identity": BASE_TREE_SHA256,
            "full24_identity": FULL24_TREE_SHA256,
            "groot_source_snapshot_identity": GROOT_PAYLOAD_TREE_SHA256,
            "ffmpeg_runtime_identity": FFMPEG_RUNTIME_ARTIFACT_ID,
        },
        "next_gate": "GPT_PI_REVIEW_AFTER_STAGE45B_B2_PI0087_RETURN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"external admission output already exists: {output}")
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o444)
    return bundle


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--cosmos", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--video-runtime-evidence", type=Path)
    parser.add_argument("--ffmpeg-runtime-root", type=Path)
    args = parser.parse_args()
    if not args.build:
        parser.error("--build is required")
    required = (
        args.repo,
        args.output,
        args.expected_head,
        args.snapshot_archive,
        args.snapshot_manifest,
        args.dataset,
        args.base_model,
        args.cosmos,
        args.modality,
        args.config,
        args.stats,
        args.video_runtime_evidence,
        args.ffmpeg_runtime_root,
    )
    if any(value is None for value in required):
        parser.error("all build inputs are required")
    bundle = build_bundle(
        repo=args.repo.resolve(),
        output=args.output.resolve(),
        expected_head=args.expected_head,
        snapshot_archive=args.snapshot_archive.resolve(),
        snapshot_manifest=args.snapshot_manifest.resolve(),
        dataset=args.dataset.resolve(),
        base_model=args.base_model.resolve(),
        cosmos=args.cosmos.resolve(),
        modality=args.modality.resolve(),
        config=args.config.resolve(),
        stats=args.stats.resolve(),
        video_runtime_evidence=args.video_runtime_evidence.resolve(),
        ffmpeg_runtime_root=args.ffmpeg_runtime_root.resolve(),
    )
    print(json.dumps({"status": "PASS", "schema": SCHEMA, "path": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
