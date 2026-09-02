#!/usr/bin/env python3
"""Run the Stage23 non-training action-pipeline learnability diagnostic.

The script is deliberately split into two independent parts:

* a CPU/source audit of the official GR00T action transform, normalization,
  universal padding, and loss-mask semantics; and
* a train-only TINY1/TINY6 fixture check which optionally re-runs the official
  LeRobotEpisodeLoader -> ShardedSingleStepDataset -> Gr00tN1d7Processor path.

It never performs a model forward, backward pass, optimizer step, PBS submit,
validation/final-test read, WBC execution, or trajectory correction.  The
input dataset is expected to be the already materialized Stage22 train-only
fixture.  The derived tiny fixtures are node-local and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from go2arx5_vla.data import load_snapshot
from go2arx5_vla.diagnostics import (
    action_contract_summary,
    action_mask_partition,
    build_universal_action_and_mask,
    label_as_prediction_metrics,
    relative_eef_roundtrip,
)


DEFAULT_SOURCE_DATASET = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/native_expert/lerobot_train"
)
DEFAULT_SNAPSHOT_ROOT = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage22/expert_dataset_v1/snapshots"
)
DEFAULT_OUTPUT = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903"
)
DEFAULT_GR00T_SOURCE = Path("/home/liuyu-nfs/Projects/isaac-gr00t")
DEFAULT_STAGE10_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10")

TRAIN_A = "red_target_train_a"
TINY1 = "TINY1"
TINY6 = "TINY6"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_manifest(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    excluded = {str(item) for item in exclude}
    rows: list[tuple[str, str, int]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        file_hash = _sha256_file(path)
        size = path.stat().st_size
        rows.append((relative, file_hash, size))
        total_bytes += size
    payload = "".join(f"{name}\0{file_hash}\0{size}\n" for name, file_hash, size in rows)
    return {
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": {name: {"sha256": file_hash, "bytes": size} for name, file_hash, size in rows},
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    """Convert official-loader numpy scalars/arrays before writing the audit."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for(text: str, needle: str) -> int | None:
    for index, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return index
    return None


def audit_official_source(gr00t_source: Path, repo_root: Path) -> dict[str, Any]:
    """Check immutable source markers without importing or modifying GR00T."""

    source_files = {
        "project_modality": repo_root / "scripts" / "stage20s_groot_modality.py",
        "state_action_processor": gr00t_source / "gr00t/data/state_action/state_action_processor.py",
        "pose": gr00t_source / "gr00t/data/state_action/pose.py",
        "action_chunking": gr00t_source / "gr00t/data/state_action/action_chunking.py",
        "n1d7_processing": gr00t_source / "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
        "n1d7_model": gr00t_source / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "stats": gr00t_source / "gr00t/data/stats.py",
    }
    markers = {
        "project_modality": {
            "action_horizon_40": "SIM_ACTION_HORIZON = 40",
            "action_indices_0_to_39": "delta_indices=list(range(SIM_ACTION_HORIZON))",
            "eef_relative": "rep=ActionRepresentation.RELATIVE",
            "eef_action_type": "type=ActionType.EEF",
            "eef_xyz_rot6d": "format=ActionFormat.XYZ_ROT6D",
            "eef_state_reference": 'state_key="eef_9d"',
            "gripper_absolute": "rep=ActionRepresentation.ABSOLUTE",
            "test_period": "SIM_ACTION_SAMPLE_PERIOD_S = 0.05",
        },
        "state_action_processor": {
            "reference_is_last_state": "reference_state = state[state_key][-1]",
            "relative_conversion": "self._convert_to_relative_action",
            "relative_stats_required": '"relative_action" not in self.statistics[embodiment_tag]',
            "relative_stats_selected": 'self.statistics[embodiment_tag]["relative_action"][key]',
            "absolute_reconstruction": "self._convert_to_absolute_action",
        },
        "pose": {
            "row_rot6d_decode": "first two rows of rotation matrix flattened",
            "row_rot6d_reshape": "rot6d = rot6d.reshape(2, 3)",
            "relative_pose_order": "T_other^{-1} * T_self",
        },
        "action_chunking": {
            "one_reference_all_points": "All poses are transformed to be relative to the reference frame.",
            "not_frame_deltas": "delta action chunking",
            "absolute_composition": "T_absolute = T_ref @ T_relative",
        },
        "n1d7_processing": {
            "physical_action_concatenation": "normalized_actions",
            "padding_mask_zero": "action_mask[:, action_dim:] = 0",
            "horizon_mask_zero": "action_mask[action_horizon:] = 0",
            "padded_action_width": "self.max_action_dim - normalized_actions.shape[1]",
        },
        "n1d7_model": {
            "masked_mse": 'F.mse_loss(pred_actions, velocity, reduction="none") * action_mask',
            "masked_denominator": "action_mask.sum() + 1e-6",
        },
        "stats": {
            "relative_action_loader": "class RelativeActionLoader",
            "reference_state_last": "last_state = state_data[state_ind]",
            "relative_stats_generation": "generate_rel_stats",
        },
    }
    file_results: dict[str, Any] = {}
    all_present = True
    for key, path in source_files.items():
        exists = path.is_file()
        text = path.read_text(encoding="utf-8") if exists else ""
        marker_results = {}
        for marker, needle in markers[key].items():
            line = _line_for(text, needle)
            marker_results[marker] = {
                "present": line is not None,
                "line": line,
                "needle": needle,
            }
            all_present = all_present and line is not None
        file_results[key] = {
            "path": str(path),
            "exists": exists,
            "sha256": _sha256_file(path) if exists else None,
            "markers": marker_results,
        }
    return {
        "schema": "STAGE23_OFFICIAL_SOURCE_ACTION_CONTRACT_AUDIT_V1",
        "status": "PASS" if all_present else "SOURCE_MARKER_AUDIT_INCOMPLETE",
        "gr00t_source": str(gr00t_source),
        "gr00t_repo_sha": _git_sha(gr00t_source),
        "files": file_results,
        "interpretation": {
            "absolute_to_relative": "T_reference^-1 @ T_absolute_target",
            "reference": "last state[eef_9d] sample in the observation window",
            "rotation": "GR00T first two rotation rows, flattened row-major",
            "gripper": "absolute and separate from the EEF WBC action",
            "normalization": "EEF uses generated relative_action[eef_9d] stats; gripper uses absolute action stats; default path is min/max normalization",
            "timestamps": "ordered action delta_indices 0..39 and preserved source timestamps; timestamps are not a model input tensor",
            "sample_period": "0.05 s is TEST_ONLY_STAGE22_SIMULATION metadata, not a production model contract and not a model input",
            "loss": "only active physical 10 dimensions contribute; 122 universal padding dimensions are masked out",
        },
    }


def _episode_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"episode row is not an object: {path}")
            rows.append(row)
    return rows


def _validate_source_train_split(source_dir: Path) -> list[dict[str, Any]]:
    info = _read_json(source_dir / "meta" / "info.json")
    lock = _read_json(source_dir / "meta" / "stage22_split_lock.json")
    if lock.get("dataset_role") != "TRAIN" or lock.get("training_allowed") is not True:
        raise ValueError("Stage23 input must be the Stage22 train-only fixture")
    if lock.get("final_test_in_training") is not False:
        raise ValueError("Stage23 input split lock does not exclude final test")
    rows = _episode_rows(source_dir)
    train_a = [
        row
        for row in rows
        if str(row.get("source_episode_id", "")).startswith("stage22_expert_red_target_train_a:")
    ]
    train_a.sort(key=lambda row: (str(row.get("source_episode_id")), int(row.get("episode_index", -1))))
    expected_ids = [f"stage22_expert_{TRAIN_A}:expert-{index:04d}" for index in range(6)]
    observed_ids = [str(row.get("source_episode_id")) for row in train_a]
    if observed_ids != expected_ids:
        raise ValueError(f"TINY1/TINY6 source order mismatch: {observed_ids}")
    if any(int(row.get("length", -1)) != 40 for row in train_a):
        raise ValueError("all Stage23 tiny source episodes must be H=40")
    for row in rows:
        source_id = str(row.get("source_episode_id", ""))
        if "holdout_e" in source_id or "holdout_f" in source_id:
            # The full Stage22 train fixture should not contain these at all.
            raise ValueError("Stage23 input unexpectedly contains holdout data")
    return [{"source": "Stage22_train_fixture", "info_total_episodes": info.get("total_episodes"), **row} for row in train_a]


def _copy_jsonl_rows(source: Path, destination: Path, rows: list[dict[str, Any]]) -> None:
    destination.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _create_tiny_fixture(
    source_dir: Path,
    output_dir: Path,
    *,
    name: str,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty tiny fixture: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data" / "chunk-000"
    video_root = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    selected_count = 1 if name == TINY1 else 6
    selected = source_rows[:selected_count]
    source_ids = [str(row["source_episode_id"]) for row in selected]
    source_indices = [int(row["episode_index"]) for row in selected]
    source_info = _read_json(source_dir / "meta" / "info.json")
    info = dict(source_info)
    info.update(
        {
            "total_episodes": selected_count,
            "total_frames": selected_count * 40,
            "total_videos": selected_count * 2,
            "total_chunks": 1,
            "splits": {"train": f"0:{selected_count}", "validation": f"{selected_count}:{selected_count}"},
        }
    )
    _write_json(meta_dir / "info.json", info)
    for filename in ("modality.json", "tasks.jsonl", "stats.json", "relative_stats.json"):
        source_path = source_dir / "meta" / filename
        if source_path.is_file():
            shutil.copy2(source_path, meta_dir / filename)

    episode_rows = []
    for new_index, source_row in enumerate(selected):
        episode_row = dict(source_row)
        episode_row["episode_index"] = new_index
        episode_row["split"] = "train"
        episode_rows.append(episode_row)
        source_index = int(source_row["episode_index"])
        source_parquet = source_dir / "data" / "chunk-000" / f"episode_{source_index:06d}.parquet"
        destination_parquet = data_dir / f"episode_{new_index:06d}.parquet"
        if not source_parquet.is_file():
            raise FileNotFoundError(source_parquet)
        shutil.copy2(source_parquet, destination_parquet)
        for video_key in ("observation.images.external_rgb", "observation.images.wrist_rgb"):
            source_video = (
                source_dir
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{source_index:06d}.mp4"
            )
            destination_video = (
                video_root / video_key / f"episode_{new_index:06d}.mp4"
            )
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            destination_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, destination_video)
    _copy_jsonl_rows(source_dir / "meta" / "episodes.jsonl", meta_dir / "episodes.jsonl", episode_rows)

    split_lock = {
        "schema": "STAGE23_TINY_TRAIN_SPLIT_LOCK_V1",
        "source_dataset": str(source_dir),
        "dataset_role": f"{name}_TRAIN_ONLY",
        "selected_target_id": TRAIN_A,
        "selected_source_episode_ids": source_ids,
        "selected_source_episode_indices": source_indices,
        "training_allowed": True,
        "validation_target_ids": [],
        "final_test_target_ids": [],
        "holdout_e_accessed": False,
        "holdout_f_accessed": False,
        "selection_policy": "lexicographic source snapshot order within red_target_train_a",
        "action_horizon": 40,
        "physical_action_dim": 10,
        "universal_action_dim": 132,
        "action_sample_period_s": 0.05,
        "action_sample_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
        "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
    }
    _write_json(meta_dir / "stage23_tiny_split_lock.json", split_lock)
    result = {
        "schema": "STAGE23_TINY_FIXTURE_V1",
        "name": name,
        "status": "MATERIALIZED_FROM_STAGE22_LEROBOT_V2_TRAIN_FIXTURE",
        "dataset_dir": str(output_dir),
        "source_dataset_dir": str(source_dir),
        "source_episode_ids": source_ids,
        "source_episode_indices": source_indices,
        "episode_count": selected_count,
        "frame_count": selected_count * 40,
        "horizon": 40,
        "physical_action_shape": [40, 10],
        "desired_label_only": True,
        "achieved_training_label": False,
        "target_in_model_input": False,
        "training_allowed": True,
        "validation_read": False,
        "final_test_read": False,
        "source_tree": _tree_manifest(source_dir),
    }
    _write_json(output_dir / "stage23_tiny_fixture_manifest.json", result)
    # The Stage19 qualification writer expects the builder manifest used to
    # create a LeRobot-v2 dataset.  Keep this manifest tiny-fixture-specific
    # and derived from the frozen train-only source; it is not a claim that
    # the source dataset was rebuilt or that any holdout was read.
    _write_json(
        meta_dir / "stage19_dataset_manifest.json",
        {
            "schema": "STAGE23_TINY_STAGE19_BUILDER_MANIFEST_V1",
            "status": "DERIVED_TRAIN_ONLY_TINY_FIXTURE",
            "dataset_role": f"{name}_TRAIN_ONLY",
            "source_dataset_dir": str(source_dir),
            "source_episode_ids": source_ids,
            "source_episode_indices": source_indices,
            "episode_count": selected_count,
            "frame_count": selected_count * 40,
            "action_horizon": 40,
            "physical_action_dim": 10,
            "desired_labels_only": True,
            "achieved_training_label": False,
            "holdout_e_accessed": False,
            "holdout_f_accessed": False,
            "source_tree": _tree_manifest(source_dir),
        },
    )
    result["tiny_tree"] = _tree_manifest(output_dir)
    return result


def _read_parquet_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        actions = np.stack(frame["action"].to_numpy(), axis=0).astype(np.float64)
        return timestamps, actions
    except ImportError:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        payload = table.to_pydict()
        timestamps = np.asarray(payload["timestamp"], dtype=np.float64)
        actions = np.asarray(payload["action"], dtype=np.float64)
        return timestamps, actions


def _materialized_label_checks(
    tiny_dir: Path,
    snapshots: list[Any],
) -> dict[str, Any]:
    rows = []
    for index, snapshot in enumerate(snapshots):
        path = tiny_dir / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        timestamps, actions = _read_parquet_arrays(path)
        expected = snapshot.physical_action
        if timestamps.shape != (40,) or actions.shape != (40, 10):
            raise ValueError(f"unexpected materialized shape in {path}: {timestamps.shape}, {actions.shape}")
        error = np.abs(actions - expected)
        timestamp_delta = timestamps - snapshot.future_timestamps
        rows.append(
            {
                "episode_index": index,
                "source_episode_id": snapshot.example_id,
                "action_max_abs_error": float(np.max(error)),
                "action_p95_abs_error": float(np.percentile(error, 95.0)),
                "timestamp_max_abs_error": float(np.max(np.abs(timestamp_delta))),
                "timestamp_period_median_s": float(np.median(np.diff(timestamps))),
                "timestamps_strictly_increasing": bool(np.all(np.diff(timestamps) > 0.0)),
            }
        )
    return {
        "status": "PASS" if all(row["action_max_abs_error"] <= 1e-6 for row in rows) else "FAIL",
        "rows": rows,
        "max_action_abs_error": float(max(row["action_max_abs_error"] for row in rows)),
        "max_timestamp_abs_error": float(max(row["timestamp_max_abs_error"] for row in rows)),
        "all_timestamps_strictly_increasing": all(row["timestamps_strictly_increasing"] for row in rows),
    }


def _official_qualification(tiny_dir: Path, stage10_root: Path) -> dict[str, Any]:
    """Run only the official loader/collator/processor path, never a model."""

    required = [
        stage10_root / "runtime" / "venv" / "bin" / "python",
        stage10_root / "src" / "isaac-gr00t",
        stage10_root / "assets" / "GR00T-N1.7-3B",
    ]
    if not all(path.exists() for path in required):
        return {
            "status": "OFFICIAL_LOADER_BLOCKED_STAGE10_RUNTIME_UNAVAILABLE",
            "model_forward": False,
            "training_or_finetuning": False,
            "gpu_used": False,
            "missing": [str(path) for path in required if not path.exists()],
        }
    try:
        cosmos_cache = (
            stage10_root
            / "cache"
            / "huggingface"
            / "hub"
            / "models--nvidia--Cosmos-Reason2-2B"
        )
        revision_file = cosmos_cache / "refs" / "main"
        revision = revision_file.read_text(encoding="utf-8").strip()
        cosmos_snapshot = cosmos_cache / "snapshots" / revision
        required_cosmos_files = (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if not revision or not cosmos_snapshot.is_dir() or not all(
            (cosmos_snapshot / name).is_file() for name in required_cosmos_files
        ):
            raise FileNotFoundError(
                "frozen Stage10 Cosmos snapshot is incomplete or unresolved"
            )
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from stage19_official_dataloader import (
            _generate_official_stats,
            _qualify,
            _write_qualified_manifest,
        )

        _generate_official_stats(tiny_dir, stage10_root)
        result = _qualify(
            tiny_dir,
            stage10_root,
            transformers_loading_kwargs={
                "trust_remote_code": True,
                "local_files_only": True,
            },
            model_name=cosmos_snapshot,
        )
        qualified_manifest = _write_qualified_manifest(tiny_dir, result)
        result = dict(result)
        result.update(
            {
                "status": "PASS",
                "qualified_manifest_path": str(qualified_manifest),
                "model_forward": False,
                "training_or_finetuning": False,
                "gpu_used": False,
                "processor_resolution": {
                    "model_name": "nvidia/Cosmos-Reason2-2B",
                    "local_files_only": True,
                    "offline_mode": True,
                    "resolved_snapshot_revision": revision,
                    "resolved_snapshot_path": str(cosmos_snapshot),
                    "required_files_present": list(required_cosmos_files),
                },
                "official_path_scope": "LeRobotEpisodeLoader -> ShardedSingleStepDataset -> Gr00tN1d7Processor only",
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - depends on node-local runtime
        return {
            "status": "OFFICIAL_LOADER_FAILED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:240],
            "model_forward": False,
            "training_or_finetuning": False,
            "gpu_used": False,
        }


def _snapshot_metrics(snapshot_root: Path, tiny_name: str) -> dict[str, Any]:
    directory = snapshot_root / f"stage22_expert_{TRAIN_A}"
    snapshot_paths = sorted(directory.glob("*.json"))
    if [path.stem for path in snapshot_paths] != [f"expert-{index:04d}" for index in range(6)]:
        raise ValueError(f"unexpected train_a snapshot order below {directory}")
    snapshots = [load_snapshot(path) for path in snapshot_paths]
    selected = snapshots[:1] if tiny_name == TINY1 else snapshots
    identity_rows = []
    relative_rows = []
    for snapshot in selected:
        identity_rows.append(
            label_as_prediction_metrics(
                physical_action=snapshot.physical_action,
                timestamps=snapshot.future_timestamps,
                state_v1=snapshot.state_v1.as_array(),
                chunk_id=snapshot.example_id,
                source_frame=snapshot.source_frame,
                tcp_semantic_id=snapshot.tcp_semantic_id,
                observation_timestamp=snapshot.observation_timestamp,
            )
        )
        relative_rows.append(
            relative_eef_roundtrip(
                snapshot.future_desired_eef_9d,
                snapshot.state_v1.as_array()[:9],
            )
        )
    identity_pass = all(row["identity_pass"] for row in identity_rows)
    max_position_error = max(row["prediction_error"]["position_m"]["max"] for row in identity_rows)
    max_rotation_error = max(row["prediction_error"]["rotation_rad"]["max"] for row in identity_rows)
    return {
        "schema": "STAGE23_LABEL_AS_PREDICTION_AGGREGATE_V1",
        "tiny_name": tiny_name,
        "source_snapshot_directory": str(directory),
        "source_snapshot_json_sha256": {
            path.name: _sha256_file(path) for path in snapshot_paths[: len(selected)]
        },
        "selected_snapshot_ids": [snapshot.example_id for snapshot in selected],
        "selected_count": len(selected),
        "horizon": 40,
        "identity_pass": identity_pass,
        "classification": (
            "LABEL_AS_PREDICTION_PASS"
            if identity_pass
            else "STAGE23_EVALUATOR_OR_ACTION_CONTRACT_BUG"
        ),
        "max_prediction_position_error_m": float(max_position_error),
        "max_prediction_rotation_error_rad": float(max_rotation_error),
        "max_relative_roundtrip_position_error_m": float(
            max(row["max_position_error_m"] for row in relative_rows)
        ),
        "max_relative_roundtrip_rotation_error_rad": float(
            max(row["max_rotation_error_rad"] for row in relative_rows)
        ),
        "rows": identity_rows,
        "relative_roundtrip_rows": relative_rows,
        "all_future_timestamps_strictly_increasing": all(
            bool(np.all(np.diff(snapshot.future_timestamps) > 0.0)) for snapshot in selected
        ),
        "future_timestamp_period_median_s": [
            float(np.median(np.diff(snapshot.future_timestamps))) for snapshot in selected
        ],
        "target_excluded_from_model_input": all(
            snapshot.target_excluded_from_model_input for snapshot in selected
        ),
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = args.source_dataset.expanduser().resolve()
    snapshot_root = args.snapshot_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Stage22 train fixture not found: {source_dir}")
    if not snapshot_root.is_dir():
        raise FileNotFoundError(f"Stage22 snapshot root not found: {snapshot_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Stage23 output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = _validate_source_train_split(source_dir)
    source_tree = _tree_manifest(source_dir)
    tiny1_dir = output_root / "tiny1_lerobot_v2"
    tiny6_dir = output_root / "tiny6_lerobot_v2"
    tiny1 = _create_tiny_fixture(source_dir, tiny1_dir, name=TINY1, source_rows=source_rows)
    tiny6 = _create_tiny_fixture(source_dir, tiny6_dir, name=TINY6, source_rows=source_rows)

    snapshots_all = [
        load_snapshot(snapshot_root / f"stage22_expert_{TRAIN_A}" / f"expert-{index:04d}.json")
        for index in range(6)
    ]
    snapshots_tiny1 = snapshots_all[:1]
    snapshots_tiny6 = snapshots_all
    tiny1_label = _materialized_label_checks(tiny1_dir, snapshots_tiny1)
    tiny6_label = _materialized_label_checks(tiny6_dir, snapshots_tiny6)

    # This is a purely numerical construction of the official padded action
    # shape. It does not call a model or the GPU runtime.
    universal, mask = build_universal_action_and_mask(snapshots_tiny6[0].physical_action)
    mask_partition = action_mask_partition(mask)
    contract = action_contract_summary()
    static_source = audit_official_source(args.gr00t_source.expanduser().resolve(), repo_root)

    official = {TINY1: {"status": "SKIPPED"}, TINY6: {"status": "SKIPPED"}}
    if args.run_official:
        official = {
            TINY1: _official_qualification(tiny1_dir, args.stage10_root.expanduser().resolve()),
            TINY6: _official_qualification(tiny6_dir, args.stage10_root.expanduser().resolve()),
        }

    label_metrics = {
        TINY1: _snapshot_metrics(snapshot_root, TINY1),
        TINY6: _snapshot_metrics(snapshot_root, TINY6),
    }
    tiny_results = {
        TINY1: {
            "fixture": tiny1,
            "materialized_label_check": tiny1_label,
            "label_as_prediction": label_metrics[TINY1],
            "official_qualification": official[TINY1],
            "final_tree": _tree_manifest(tiny1_dir),
        },
        TINY6: {
            "fixture": tiny6,
            "materialized_label_check": tiny6_label,
            "label_as_prediction": label_metrics[TINY6],
            "official_qualification": official[TINY6],
            "final_tree": _tree_manifest(tiny6_dir),
        },
    }
    official_pass = all(
        value["official_qualification"].get("status") == "PASS" for value in tiny_results.values()
    ) if args.run_official else False
    label_pass = all(
        value["label_as_prediction"]["identity_pass"] for value in tiny_results.values()
    )
    materialized_pass = all(
        value["materialized_label_check"]["status"] == "PASS" for value in tiny_results.values()
    )
    result = {
        "schema": "STAGE23_ACTION_PIPELINE_DIAGNOSTIC_V1",
        "stage": "Stage23",
        "status": "PASS" if label_pass and materialized_pass and (official_pass or not args.run_official) and static_source["status"] == "PASS" else "DEVIATION",
        "classification": (
            "STAGE23_ACTION_PIPELINE_AND_EVALUATOR_SANITY_PASS"
            if label_pass and materialized_pass and static_source["status"] == "PASS" and (official_pass or not args.run_official)
            else (
                "STAGE23_OFFICIAL_LOADER_OR_PROCESSOR_BLOCKED"
                if label_pass and materialized_pass and static_source["status"] == "PASS" and args.run_official
                else "STAGE23_EVALUATOR_OR_ACTION_CONTRACT_BUG"
            )
        ),
        "stage22_input": {
            "source_dataset": str(source_dir),
            "source_tree": source_tree,
            "source_episode_count": len(_episode_rows(source_dir)),
            "selected_target": TRAIN_A,
            "selected_source_snapshot_count": 6,
            "validation_e_accessed": False,
            "final_test_f_accessed": False,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "gpu_used": False,
            "model_forward": False,
            "backward": False,
            "optimizer_step": False,
            "pbs_submitted": False,
            "hardware_control": False,
            "wbc_executed": False,
        },
        "official_source_audit": static_source,
        "contract": contract,
        "universal_shape_audit": {
            "physical_input_shape": list(snapshots_tiny6[0].physical_action.shape),
            "universal_shape": list(universal.shape),
            "mask_shape": list(mask.shape),
            "mask_partition": mask_partition,
            "physical_loss_dimensions": 10,
            "padding_loss_dimensions": 0,
            "padding_width": 122,
            "padding_contributes_to_loss": False,
            "construction_correction": False,
        },
        "tiny_fixtures": tiny_results,
        "future_hopper_protocol": {
            "status": "PREPARED_ONLY_NOT_AUTHORIZED",
            "O1": "TINY1 deliberate overfit; not executed",
            "O6": "TINY6 conditional diagnostic; not executed",
            "recipe": {
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_visual": False,
                "tune_llm": False,
                "tune_vlln": False,
                "action_horizon": 40,
            },
            "selection_thresholds": "proposal only; must be frozen before any future Hopper run",
        },
        "no_training_assertions": {
            "stage23_training_executed": False,
            "base_only_inference_executed": False,
            "validation_or_final_test_read": False,
            "model_selection_performed": False,
            "checkpoint_created_or_changed": False,
            "trajectory_correction": False,
        },
    }
    output_path = output_root / "stage23_action_pipeline_diagnostic.json"
    _write_json(output_path, result)
    result["output_path"] = str(output_path)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gr00t-source", type=Path, default=DEFAULT_GR00T_SOURCE)
    parser.add_argument("--stage10-root", type=Path, default=DEFAULT_STAGE10_ROOT)
    parser.add_argument(
        "--run-official",
        action="store_true",
        help="run CPU-only official loader/collator/processor qualification for TINY1/TINY6",
    )
    args = parser.parse_args()
    run_diagnostic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
