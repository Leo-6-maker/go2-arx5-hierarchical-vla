"""Materialize the Stage19 on-disk GR00T/LeRobot v2 fixture.

The builder is deliberately independent of the optional ``lerobot`` Python
package.  It writes the v2 parquet/video layout consumed by the official
Isaac-GR00T ``LeRobotEpisodeLoader`` and keeps the richer robot provenance in
explicit sidecar metadata.  It never edits a source episode and never writes a
dataset into Git.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .quality import EpisodeQualityResult
from .readiness import (
    CalibrationBundleV1,
    RawEpisodeV1,
    SynchronizationResultV1,
    _jsonable,
)


STATE_DIM = 10
ACTION_DIM = 10
DEFAULT_HORIZON = 40
BODY_VIDEO_KEY = "observation.images.external_rgb"
WRIST_VIDEO_KEY = "observation.images.wrist_rgb"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    repo = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class MaterializationConfigV1:
    """Reproducible fixture settings; production timing remains separate."""

    output_dir: Path
    horizon: int = DEFAULT_HORIZON
    sample_stream: str = "body"
    dataset_name: str = "go2_arx5_stage19_d3_target"
    overwrite: bool = False
    video_codec: str = "mp4v"
    train_episode_count: int | None = None
    split_seed: int = 1901

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError("horizon must be positive")
        if not self.sample_stream:
            raise ValueError("sample_stream is required")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        if len(self.video_codec) != 4:
            raise ValueError("video_codec must be a four-character OpenCV codec")
        if self.train_episode_count is not None and int(self.train_episode_count) < 1:
            raise ValueError("train_episode_count must be positive when provided")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "MaterializationConfigV1",
            "output_dir": str(self.output_dir),
            "horizon": int(self.horizon),
            "sample_stream": self.sample_stream,
            "dataset_name": self.dataset_name,
            "video_codec": self.video_codec,
            "train_episode_count": self.train_episode_count,
            "split_seed": int(self.split_seed),
            "horizon_provenance": "TEST_ONLY_D3_FIXTURE",
            "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        }


@dataclass(frozen=True)
class MaterializationInputV1:
    raw: RawEpisodeV1
    synchronization: SynchronizationResultV1
    calibration: CalibrationBundleV1
    quality: EpisodeQualityResult | None = None


@dataclass(frozen=True)
class MaterializedDatasetV1:
    output_dir: Path
    dataset_name: str
    episode_count: int
    frame_count: int
    sample_rate_hz: float | None
    horizon: int
    config_sha256: str
    manifest_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "MaterializedDatasetV1",
            "output_dir": str(self.output_dir),
            "dataset_name": self.dataset_name,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "sample_rate_hz": self.sample_rate_hz,
            "horizon": self.horizon,
            "config_sha256": self.config_sha256,
            "manifest_path": str(self.manifest_path),
        }


def _require_safe_output_path(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if resolved == Path("/") or len(resolved.parts) < 4:
        raise ValueError(f"refusing to overwrite broad output path: {resolved}")


def _read_bgr(path: str | Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode camera frame: {path}")
    return image


def _write_video(
    output_path: Path,
    frame_paths: Sequence[str],
    *,
    nominal_fps: float,
    codec: str,
) -> dict[str, Any]:
    import cv2

    if not frame_paths:
        raise ValueError(f"cannot write empty video: {output_path}")
    first = _read_bgr(frame_paths[0])
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(nominal_fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")
    try:
        for frame_path in frame_paths:
            frame = _read_bgr(frame_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"camera frame shape changed in {output_path}: "
                    f"expected {(height, width)}, got {frame.shape[:2]}"
                )
            writer.write(frame)
    finally:
        writer.release()
    return {
        "path": str(output_path),
        "frame_count": len(frame_paths),
        "shape": [int(height), int(width), 3],
        "nominal_fps_hz": float(nominal_fps),
        "sha256": _sha256_file(output_path),
    }


def _feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError(f"expected [N,{STATE_DIM}] values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("dataset statistics cannot contain non-finite values")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _task_instruction(raw: RawEpisodeV1) -> str:
    for key in ("instruction", "task_description", "description", "name"):
        value = raw.task.get(key)
        if value:
            return str(value)
    return f"execute trajectory {raw.trajectory_id}"


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _effective_sample_rate(sync: SynchronizationResultV1, sample_stream: str) -> float | None:
    rates = dict(sync.effective_rates_hz)
    if sample_stream in rates and rates[sample_stream] is not None:
        return float(rates[sample_stream])
    timestamps = np.asarray(sync.sample_timestamps, dtype=np.float64)
    if timestamps.size > 1:
        return float((timestamps.size - 1) / (timestamps[-1] - timestamps[0]))
    return None


def _episode_split(index: int, count: int, train_count: int | None) -> str:
    if train_count is None:
        train_count = count
    return "train" if index < int(train_count) else "validation"


def materialize_lerobot_v2(
    inputs: Sequence[MaterializationInputV1],
    config: MaterializationConfigV1,
) -> MaterializedDatasetV1:
    """Write a real LeRobot-v2/GR00T dataset fixture and provenance sidecars."""

    if not inputs:
        raise ValueError("at least one episode is required")
    output = Path(config.output_dir).expanduser()
    _require_safe_output_path(output)
    if output.exists():
        if not config.overwrite:
            raise FileExistsError(f"output exists; pass overwrite explicitly: {output}")
        shutil.rmtree(output)

    chunk_size = 1000
    data_root = output / "data"
    meta_dir = output / "meta"
    data_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.as_dict()
    config_sha = _canonical_hash(config_dict)
    task_to_index: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    action_timestamp_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    video_rows: list[dict[str, Any]] = []
    sample_rates: list[float] = []

    for episode_index, item in enumerate(inputs):
        raw = item.raw
        sync = item.synchronization
        if sync.sample_stream != config.sample_stream:
            raise ValueError(
                f"episode {raw.episode_id} uses sample stream {sync.sample_stream!r}, "
                f"expected {config.sample_stream!r}"
            )
        sample_count = sync.sample_count
        if sample_count < int(config.horizon):
            raise ValueError(
                f"episode {raw.episode_id} has {sample_count} synchronized samples, "
                f"need at least {config.horizon}"
            )
        sample_timestamps = np.asarray(sync.sample_timestamps, dtype=np.float64)
        state_indices = np.asarray(sync.indices["state"], dtype=np.int64)
        desired_indices = np.asarray(sync.indices["desired"], dtype=np.int64)
        achieved_indices = np.asarray(sync.indices["achieved"], dtype=np.int64)
        if any(array.shape != (sample_count,) for array in (state_indices, desired_indices, achieved_indices)):
            raise ValueError(f"synchronization indices are incomplete for {raw.episode_id}")

        state = np.concatenate(
            [raw.state_eef_9d[state_indices], raw.state_gripper[state_indices]], axis=1
        ).astype(np.float32)
        desired = np.concatenate(
            [raw.desired_eef_9d[desired_indices], raw.desired_gripper[desired_indices]], axis=1
        ).astype(np.float32)
        if raw.achieved_gripper is None:
            achieved = raw.achieved_eef_9d[achieved_indices].astype(np.float32)
        else:
            achieved = np.concatenate(
                [raw.achieved_eef_9d[achieved_indices], raw.achieved_gripper[achieved_indices]],
                axis=1,
            ).astype(np.float32)
        if not np.isfinite(state).all() or not np.isfinite(desired).all() or not np.isfinite(achieved).all():
            raise ValueError(f"non-finite materialized state/action in {raw.episode_id}")
        state_rows.append(state)
        action_rows.append(desired)

        instruction = _task_instruction(raw)
        if instruction not in task_to_index:
            task_to_index[instruction] = len(task_to_index)
            task_rows.append({"task_index": task_to_index[instruction], "task": instruction})
        task_index = task_to_index[instruction]
        table_rows = {
            "timestamp": sample_timestamps,
            "task_index": np.full(sample_count, task_index, dtype=np.int64),
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in desired],
        }
        import pandas as pd

        episode_chunk = episode_index // chunk_size
        data_dir = data_root / f"chunk-{episode_chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(table_rows).to_parquet(parquet_path, index=False)

        sample_rate = _effective_sample_rate(sync, config.sample_stream)
        nominal_fps = float(sample_rate or 1.0)
        sample_rates.extend([] if sample_rate is None else [sample_rate])
        camera_outputs: dict[str, Any] = {}
        for source_name, video_key in (("body", BODY_VIDEO_KEY), ("wrist", WRIST_VIDEO_KEY)):
            stream = raw.camera_streams.get(source_name)
            if stream is None:
                raise ValueError(f"required camera {source_name!r} absent in {raw.episode_id}")
            camera_indices = np.asarray(sync.indices[f"camera:{source_name}"], dtype=np.int64)
            frame_paths = [stream.frame_paths[int(index)] for index in camera_indices]
            video_path = (
                output
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            camera_outputs[source_name] = _write_video(
                video_path,
                frame_paths,
                nominal_fps=nominal_fps,
                codec=config.video_codec,
            )
        video_rows.append({"episode_index": episode_index, "cameras": camera_outputs})

        valid_count = sample_count - int(config.horizon) + 1
        for sample_index in range(valid_count):
            future_slice = slice(sample_index, sample_index + int(config.horizon))
            future_desired_indices = desired_indices[future_slice]
            future_achieved_indices = achieved_indices[future_slice]
            future_desired = np.concatenate(
                [raw.desired_eef_9d[future_desired_indices], raw.desired_gripper[future_desired_indices]],
                axis=1,
            )
            if raw.achieved_gripper is None:
                future_achieved = raw.achieved_eef_9d[future_achieved_indices]
            else:
                future_achieved = np.concatenate(
                    [
                        raw.achieved_eef_9d[future_achieved_indices],
                        raw.achieved_gripper[future_achieved_indices],
                    ],
                    axis=1,
                )
            action_timestamp_rows.append(
                {
                    "schema": "Stage19ActionTimestampRecordV1",
                    "dataset_episode_index": episode_index,
                    "dataset_frame_index": sample_index,
                    "raw_episode_id": raw.episode_id,
                    "observation_timestamp": float(sample_timestamps[sample_index]),
                    "future_action_point_timestamps": sample_timestamps[future_slice].tolist(),
                    "future_source_desired_indices": future_desired_indices.tolist(),
                    "future_desired_action_shape": list(future_desired.shape),
                    "future_desired_action_sha256": _array_sha256(future_desired),
                    "future_achieved_audit_shape": list(future_achieved.shape),
                    "future_achieved_audit_sha256": _array_sha256(future_achieved),
                    "label_source": "desired",
                    "achieved_role": "audit_only_not_training_label",
                    "achieved_gripper_present": raw.achieved_gripper is not None,
                    "timestamp_authority": "synchronization_sample_timestamps",
                }
            )

        source_json = raw.metadata.get("source_json")
        provenance_rows.append(
            {
                "schema": "Stage19EpisodeProvenanceV1",
                "dataset_episode_index": episode_index,
                "episode_index": episode_index,
                "source_episode_id": raw.episode_id,
                "source_trajectory_id": raw.trajectory_id,
                "source_json": source_json,
                "source_json_sha256": raw.metadata.get("source_json_sha256"),
                "sample_stream": config.sample_stream,
                "sample_count": sample_count,
                "valid_future_window_count": valid_count,
                "sample_timestamps": sample_timestamps.tolist(),
                "source_native_indices": {
                    "state": state_indices.tolist(),
                    "desired": desired_indices.tolist(),
                    "achieved": achieved_indices.tolist(),
                },
                "declared_rates_hz": dict(sync.declared_rates_hz),
                "effective_rates_hz": dict(sync.effective_rates_hz),
                "synchronization_metrics": dict(sync.metrics),
                "synchronization_warnings": list(sync.warnings),
                "calibration_bundle_id": raw.calibration_bundle_id,
                "source_frame": raw.source_frame,
                "tcp_semantic_id": raw.tcp_semantic_id,
                "quality_decision": None if item.quality is None else item.quality.decision.value,
                "quality_issue_codes": [] if item.quality is None else [issue.code for issue in item.quality.issues],
                "desired_label": True,
                "achieved_eef_retained_for_audit": True,
                "raw_whole_body_streams_retained": bool(raw.raw_whole_body_streams),
                "source_timestamps_preserved": True,
            }
        )
        calibration_rows.append(
            {
                "dataset_episode_index": episode_index,
                "raw_episode_id": raw.episode_id,
                "calibration": item.calibration.as_dict(),
            }
        )
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": sample_count,
                "source_episode_id": raw.episode_id,
                "sample_stream": config.sample_stream,
                "split": _episode_split(episode_index, len(inputs), config.train_episode_count),
            }
        )

    state_values = np.concatenate(state_rows, axis=0)
    action_values = np.concatenate(action_rows, axis=0)
    if state_values.shape[-1] != STATE_DIM or action_values.shape[-1] != ACTION_DIM:
        raise AssertionError("materialized state/action ABI is not 10D")

    total_frames = int(state_values.shape[0])
    train_count = config.train_episode_count or len(inputs)
    if train_count > len(inputs):
        raise ValueError("train_episode_count cannot exceed episode count")
    fps = float(np.median(sample_rates)) if sample_rates else None
    info = {
        "codebase_version": "v2.1",
        "robot_type": "go2_arx5_d3_stage19_proxy",
        "total_episodes": len(inputs),
        "total_frames": total_frames,
        "total_tasks": len(task_rows),
        "total_videos": len(inputs) * 2,
        "total_chunks": (len(inputs) + chunk_size - 1) // chunk_size,
        "chunks_size": chunk_size,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "splits": {
            "train": f"0:{train_count}",
            "validation": f"{train_count}:{len(inputs)}",
        },
        "features": {
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            BODY_VIDEO_KEY: {
                "dtype": "video",
                "shape": video_rows[0]["cameras"]["body"]["shape"],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": config.video_codec},
            },
            WRIST_VIDEO_KEY: {
                "dtype": "video",
                "shape": video_rows[0]["cameras"]["wrist"]["shape"],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": config.video_codec},
            },
        },
        "stage19_timing": {
            "sample_stream": config.sample_stream,
            "source_timestamp_authority": "D3_SOURCE_TIMESTAMPS",
            "declared_camera_rate_is_not_authority": True,
            "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        },
        "stage19_abi": {
            "state": "StateV1 = EEF XYZ 3 + GR00T_FIRST_TWO_ROWS rot6d 6 + gripper 1",
            "action": "future desired EEF [H,9] + future desired gripper [H,1]",
            "achieved_action_materialized": False,
        },
    }
    modality = {
        "state": {
            "eef_9d": {"start": 0, "end": 9, "original_key": "observation.state"},
            "gripper_position": {"start": 9, "end": 10, "original_key": "observation.state"},
        },
        "action": {
            "eef_9d": {"start": 0, "end": 9, "original_key": "action"},
            "gripper_position": {"start": 9, "end": 10, "original_key": "action"},
        },
        "video": {
            "external_rgb": {"original_key": BODY_VIDEO_KEY},
            "wrist_rgb": {"original_key": WRIST_VIDEO_KEY},
        },
        "annotation": {"language.task_description": {"original_key": "task_index"}},
    }

    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "modality.json", modality)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(meta_dir / "tasks.jsonl", task_rows)
    _write_json(meta_dir / "stats.json", {
        "observation.state": _feature_stats(state_values),
        "action": _feature_stats(action_values),
        "__stage19_stats_provenance__": {
            "source": "Stage19 materializer desired/action and current/state rows",
            "desired_labels": True,
            "achieved_labels_used_for_stats": False,
            "stats_complete": True,
        },
    })
    _write_jsonl(meta_dir / "stage19_episode_provenance.jsonl", provenance_rows)
    _write_jsonl(meta_dir / "stage19_action_timestamps.jsonl", action_timestamp_rows)
    _write_jsonl(meta_dir / "stage19_calibration_bundles.jsonl", calibration_rows)
    _write_json(meta_dir / "stage19_video_manifest.json", {
        "schema": "Stage19VideoManifestV1",
        "source_frames_encoded_without_source_rewrite": True,
        "videos": video_rows,
    })

    manifest = {
        "schema": "STAGE19_LEROBOT_MATERIALIZED_DATASET_V1",
        "status": "NODE_LOCAL_TEST_FIXTURE",
        "dataset_name": config.dataset_name,
        "format": "LeRobot v2.1 flavor consumed by official GR00T loader",
        "output_dir": str(output),
        "episode_count": len(inputs),
        "frame_count": total_frames,
        "sample_rate_hz": fps,
        "horizon": int(config.horizon),
        "horizon_provenance": "TEST_ONLY_D3_FIXTURE",
        "config": config_dict,
        "config_sha256": config_sha,
        "builder_git_sha": _git_sha(),
        "source_episode_ids": [item.raw.episode_id for item in inputs],
        "source_episode_sha256": [item.raw.metadata.get("source_json_sha256") for item in inputs],
        "source_timestamps_preserved": True,
        "future_action_timestamps_sidecar": str(meta_dir / "stage19_action_timestamps.jsonl"),
        "achieved_eef_sidecar_role": "audit_only_not_training_label",
        "calibration_sidecar": str(meta_dir / "stage19_calibration_bundles.jsonl"),
        "official_statistics_generation": "RUN_BY_STAGE19_OFFICIAL_QUALIFICATION",
        "official_loader": "gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader",
        "official_dataloader": "gr00t.data.dataset.sharded_single_step_dataset.ShardedSingleStepDataset",
        "native_lerobot_python_package": "OPTIONAL_NOT_REQUIRED_BY_GR00T_V2_LOADER",
        "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        "real_robot_calibration": False,
        "hardware_control": False,
        "training_or_finetuning": False,
        "file_manifest": {},
    }
    # Hash every generated artifact except the manifest that is about to be
    # written, so the node-local artifact is independently auditable.
    file_manifest: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            file_manifest[str(path.relative_to(output))] = _sha256_file(path)
    manifest["file_manifest"] = file_manifest
    manifest_path = meta_dir / "stage19_dataset_manifest.json"
    _write_json(manifest_path, manifest)

    return MaterializedDatasetV1(
        output_dir=output,
        dataset_name=config.dataset_name,
        episode_count=len(inputs),
        frame_count=total_frames,
        sample_rate_hz=fps,
        horizon=int(config.horizon),
        config_sha256=config_sha,
        manifest_path=manifest_path,
    )
