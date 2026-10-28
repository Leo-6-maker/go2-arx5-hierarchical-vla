#!/usr/bin/env python3
"""CPU-only materialization and verification of the Stage45D TRAIN24 closure.

The historical ``SameSceneExpertSnapshotV1`` JSON replicas are incomplete:
only the six train-a witness files survive.  This module creates a separate,
explicit closure from the already accepted FULL24 TRAIN parquet/video tree.
It never writes the dataset, labels, checkpoints, or model outputs.  The
result is a descriptor/observation-frame materialization with an audit that
binds every descriptor to one parquet episode, both accepted video streams,
the frozen language task, and the non-target model-input boundary.

The module is intentionally CPU-only.  It has no Torch, GR00T, CUDA, PBS, or
model import, and its verifier is called before the H200 child imports the
model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


HORIZON = 40
STATE_DIM = 10
PHYSICAL_DIM = 10
RGB_SHAPE = (96, 96, 3)
TARGET_ORDER = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
TARGET_POSITIONS = {
    "red_target_train_a": (0.30, 0.10, 0.58),
    "red_target_train_b": (0.30, -0.10, 0.58),
    "red_target_train_c": (0.30, 0.05, 0.64),
    "red_target_train_d": (0.30, -0.05, 0.63),
}
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
CALIBRATION_BUNDLE_ID = "STAGE22_NATIVE_SIM_CALIBRATION_BUNDLE_V1"
NATIVE_CONTROL_DT_S = 0.019999999552965164
DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
CLOSURE_SCHEMA = "STAGE45D_D1_TRAIN24_DESCRIPTOR_CLOSURE_V1"
CLOSURE_MANIFEST_NAME = "STAGE45D_D1_TRAIN24_DESCRIPTOR_CLOSURE_V1.json"
SOURCE_PREFLIGHT_SCHEMA = "STAGE45C_RUNTIME_PREFLIGHT_V1"


class ClosureError(RuntimeError):
    """Raised when the descriptor closure cannot be proven exact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ClosureError(f"JSON is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"JSON cannot be parsed: {path}") from exc


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ClosureError(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def regular_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ClosureError(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def tree_digest(root: Path, prefix: str) -> tuple[str, int, int]:
    """Use the same path/bytes/SHA law as the accepted dataset verifier."""

    regular_directory(root, "FULL24 dataset")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"dataset directory is unsafe: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"dataset file is unsafe: {child}")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    entries.sort(key=lambda item: str(item["path"]))
    digest = hashlib.sha256()
    for item in entries:
        digest.update(f"{item['path']}\0{item['bytes']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest(), len(entries), sum(int(item["bytes"]) for item in entries)


def _array_rows(frame: Any, column: str, width: int, *, numpy: Any) -> Any:
    require(column in frame.columns, f"canonical parquet column is missing: {column}")
    try:
        values = numpy.stack([numpy.asarray(value, dtype=numpy.float64).reshape(-1) for value in frame[column]])
    except (TypeError, ValueError) as exc:
        raise ClosureError(f"canonical parquet column is not numeric: {column}") from exc
    require(values.shape == (HORIZON, width), f"canonical parquet shape drift for {column}: {values.shape}")
    require(bool(numpy.isfinite(values).all()), f"canonical parquet column is non-finite: {column}")
    return values


def read_parquet_projection(path: Path) -> dict[str, Any]:
    """Read only the accepted state/action/timestamp projection on CPU."""

    regular_file(path, "canonical parquet")
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(path, columns=["timestamp", "action", "observation.state"])
    require(len(frame) == HORIZON, f"canonical parquet row count drift: {path}")
    timestamps = np.asarray(frame["timestamp"].to_numpy(), dtype=np.float64)
    require(timestamps.shape == (HORIZON,) and bool(np.isfinite(timestamps).all()), f"timestamp projection drift: {path}")
    require(float(np.max(np.abs(np.diff(timestamps) - 0.05))) <= 1e-9, f"timestamp interval drift: {path}")
    action = _array_rows(frame, "action", PHYSICAL_DIM, numpy=np)
    state = _array_rows(frame, "observation.state", STATE_DIM, numpy=np)
    require(bool(np.array_equal(state, np.broadcast_to(state[0], state.shape))), f"state is not repeated: {path}")
    require(float(np.linalg.norm(state[0, :3] - action[0, :3])) <= 1e-6, f"state/action anchor drift: {path}")
    require(float(np.max(np.abs(action[:, 9] - 0.5))) <= 1e-6, f"action gripper drift: {path}")
    require(float(np.max(np.abs(state[:, 9] - 0.5))) <= 1e-6, f"state gripper drift: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "state_v1": state[0].tolist(),
        "expert_physical_action": action.tolist(),
        "source_timestamps": timestamps.tolist(),
        "observation_timestamp": float(timestamps[0]),
    }


def decode_first_rgb(path: Path) -> Any:
    """Decode frame zero with the CPU FFmpeg backend exposed by OpenCV."""

    regular_file(path, "accepted RGB video")
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        ok, bgr = capture.read()
    finally:
        capture.release()
    require(frame_count == HORIZON, f"RGB video frame count drift: {path}: {frame_count}")
    require(ok and bgr is not None, f"RGB video frame zero cannot be decoded: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    require(tuple(int(value) for value in rgb.shape) == RGB_SHAPE, f"RGB video shape drift: {path}: {rgb.shape}")
    require(rgb.dtype == np.uint8 and bool(np.isfinite(rgb).all()), f"RGB video dtype/finite drift: {path}")
    return np.ascontiguousarray(rgb)


def write_ppm(path: Path, rgb: Any) -> None:
    import numpy as np

    array = np.asarray(rgb)
    require(tuple(int(value) for value in array.shape) == RGB_SHAPE and array.dtype == np.uint8, f"PPM source shape/dtype drift: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{RGB_SHAPE[1]} {RGB_SHAPE[0]}\n255\n".encode("ascii")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(header + np.ascontiguousarray(array).tobytes())
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_ppm(path: Path) -> Any:
    import numpy as np

    regular_file(path, "materialized RGB frame")
    header = f"P6\n{RGB_SHAPE[1]} {RGB_SHAPE[0]}\n255\n".encode("ascii")
    data = path.read_bytes()
    require(data.startswith(header), f"materialized RGB frame header drift: {path}")
    payload = data[len(header):]
    expected_bytes = RGB_SHAPE[0] * RGB_SHAPE[1] * RGB_SHAPE[2]
    require(len(payload) == expected_bytes, f"materialized RGB frame byte count drift: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(RGB_SHAPE).copy()


def _window_id(index: int) -> str:
    target = TARGET_ORDER[index // 6]
    return f"stage22_expert_{target}:expert-{index % 6:04d}"


def _target_id(index: int) -> str:
    return TARGET_ORDER[index // 6]


def _video_path(dataset_root: Path, index: int, kind: str) -> Path:
    return dataset_root / "videos" / "chunk-000" / f"observation.images.{kind}" / f"episode_{index:06d}.mp4"


def _descriptor_payload(
    *,
    index: int,
    projection: Mapping[str, Any],
    body_frame: Path,
    wrist_frame: Path,
    body_video: Path,
    wrist_video: Path,
) -> dict[str, Any]:
    import numpy as np

    target_id = _target_id(index)
    window_id = _window_id(index)
    target_world = np.asarray(TARGET_POSITIONS[target_id], dtype=np.float64)
    state = np.asarray(projection["state_v1"], dtype=np.float64)
    action = np.asarray(projection["expert_physical_action"], dtype=np.float64)
    timestamps = list(projection["source_timestamps"])
    anchor = target_world - action[-1, :3]
    achieved_world = anchor + state[:3]
    return {
        "schema": "SameSceneExpertSnapshotV1",
        "example_id": window_id,
        "raw_episode_id": f"stage22_expert_{target_id}",
        "simulator_instance_id": "STAGE22_FULL24_PARQUET_MATERIALIZED_DESCRIPTOR_V1",
        "observation_timestamp": float(projection["observation_timestamp"]),
        "body_frame_path": str(body_frame.resolve()),
        "wrist_frame_path": str(wrist_frame.resolve()),
        "state_v1": {
            "schema": "StateV1",
            "layout": ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"],
            "rot6d_layout": "GR00T_FIRST_TWO_ROWS",
            "values": list(projection["state_v1"]),
        },
        "future_timestamps": timestamps,
        "future_desired_eef_9d": action[:, :9].tolist(),
        "future_gripper": action[:, 9:10].tolist(),
        "target_position_world_m": list(TARGET_POSITIONS[target_id]),
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "calibration_bundle_id": CALIBRATION_BUNDLE_ID,
        "target_id": target_id,
        "target_excluded_from_model_input": True,
        "native_control_dt_s": NATIVE_CONTROL_DT_S,
        "raw_telemetry": {
            "materialization_source": "accepted_FULL24_TRAIN_parquet_projection",
            "achieved_position_world_m": achieved_world.tolist(),
            "telemetry_is_not_historical_execution_authority": True,
        },
        "provenance": {
            "stage": "Stage22",
            "task_id": "SIM_VISUAL_REACH_V0",
            "language": LANGUAGE,
            "source": "deterministic_FULL24_TRAIN_parquet_video_materialization",
            "materialization_schema": CLOSURE_SCHEMA,
            "materialization_authority": "CANONICAL_FULL24_TRAIN_PARQUET_AND_VIDEO",
            "canonical_parquet_path": str(projection["path"]),
            "canonical_parquet_sha256": str(projection["sha256"]),
            "body_video_path": str(body_video.resolve()),
            "body_video_sha256": sha256_file(body_video),
            "wrist_video_path": str(wrist_video.resolve()),
            "wrist_video_sha256": sha256_file(wrist_video),
            "observation_frame_index": 0,
            "historical_source_json_execution_authority": False,
            "target_excluded_from_model_input": True,
            "target_xyz_in_model_input": False,
            "new_scientific_data_generated": False,
        },
    }


def _source_projection_hashes(projection: Mapping[str, Any]) -> dict[str, str]:
    return {
        "state_v1": sha256_bytes(canonical_json(projection["state_v1"])),
        "source_timestamps": sha256_bytes(canonical_json(projection["source_timestamps"])),
        "expert_physical_action": sha256_bytes(canonical_json(projection["expert_physical_action"])),
        "language": sha256_bytes(canonical_json(LANGUAGE)),
    }


def _load_source_preflight(path: Path, dataset_root: Path) -> dict[str, Any]:
    regular_file(path, "Stage45C source preflight")
    document = read_json(path)
    require(document.get("status") == "PASS", "source preflight is not PASS")
    source = document.get("canonical_label_source")
    require(isinstance(source, Mapping), "source preflight canonical label source is absent")
    require(source.get("kind") == "SEALED_FULL24_TRAIN_PARQUET", "source preflight label source drift")
    require(source.get("root") == str(dataset_root.resolve()), "source preflight dataset root drift")
    require(source.get("tree_sha256") == DATASET_TREE_SHA256, "source preflight dataset tree drift")
    require(source.get("label_records") == 24 and source.get("parquet_files_read") == 24, "source preflight record count drift")
    require(source.get("target_xyz_in_model_input") is False, "source preflight target exclusion drift")
    rows = document.get("snapshots")
    require(isinstance(rows, list) and len(rows) == 24, "source preflight snapshot closure is not all-24")
    return document


def materialize(
    *,
    dataset_root: Path,
    snapshot_root: Path,
    source_preflight: Path,
    historical_snapshot_root: Path,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    snapshot_root = snapshot_root.resolve()
    source_preflight = source_preflight.resolve()
    historical_snapshot_root = historical_snapshot_root.resolve()
    regular_directory(dataset_root, "FULL24 dataset root")
    regular_directory(historical_snapshot_root, "historical witness root")
    require(not snapshot_root.exists(), f"descriptor closure root must be fresh: {snapshot_root}")
    dataset_digest, dataset_file_count, dataset_bytes = tree_digest(dataset_root, "payload/lerobot_train")
    require(dataset_digest == DATASET_TREE_SHA256, "FULL24 dataset tree does not match accepted identity")
    source_document = _load_source_preflight(source_preflight, dataset_root)
    source_rows = source_document["snapshots"]

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    regular_file(episodes_path, "FULL24 episodes metadata")
    regular_file(tasks_path, "FULL24 tasks metadata")
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(episodes) == 24 and len(tasks) == 1, "FULL24 metadata count drift")
    require(tasks[0].get("task_index") == 0 and tasks[0].get("task") == LANGUAGE, "FULL24 language metadata drift")

    snapshot_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    equality_rows: list[dict[str, Any]] = []
    historical_rows: list[dict[str, Any]] = []
    for index in range(24):
        window_id = _window_id(index)
        target_id = _target_id(index)
        expected_source_id = window_id
        episode = episodes[index]
        require(episode.get("episode_index") == index, f"episode index drift: {index}")
        require(episode.get("source_episode_id") == expected_source_id, f"episode source ID drift: {index}")
        require(episode.get("tasks") == [LANGUAGE] and episode.get("length") == HORIZON and episode.get("split") == "train", f"episode metadata drift: {index}")

        parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        projection = read_parquet_projection(parquet_path)
        source_row = source_rows[index]
        require(source_row.get("window_id") == window_id, f"source preflight window order drift: {index}")
        require(source_row.get("canonical_parquet_path") == projection["path"], f"source preflight parquet path drift: {index}")
        require(source_row.get("canonical_parquet_sha256") == projection["sha256"], f"source preflight parquet SHA drift: {index}")
        for key in ("state_v1", "source_timestamps", "expert_physical_action"):
            require(source_row.get(key) == projection[key], f"source preflight {key} drift: {index}")
        require(source_row.get("target_id") == target_id and source_row.get("target_xyz_in_model_input") is False and source_row.get("target_excluded_from_model_input") is True, f"source preflight target boundary drift: {index}")

        body_video = _video_path(dataset_root, index, "external_rgb")
        wrist_video = _video_path(dataset_root, index, "wrist_rgb")
        body_rgb = decode_first_rgb(body_video)
        wrist_rgb = decode_first_rgb(wrist_video)
        body_frame = snapshot_root / "frames" / target_id / f"expert-{index % 6:04d}_body.ppm"
        wrist_frame = snapshot_root / "frames" / target_id / f"expert-{index % 6:04d}_wrist.ppm"
        write_ppm(body_frame, body_rgb)
        write_ppm(wrist_frame, wrist_rgb)
        require(bool(np_equal(read_ppm(body_frame), body_rgb)), f"body RGB round-trip drift: {index}")
        require(bool(np_equal(read_ppm(wrist_frame), wrist_rgb)), f"wrist RGB round-trip drift: {index}")

        descriptor = _descriptor_payload(
            index=index,
            projection=projection,
            body_frame=body_frame,
            wrist_frame=wrist_frame,
            body_video=body_video,
            wrist_video=wrist_video,
        )
        descriptor_path = snapshot_root / f"stage22_expert_{target_id}" / f"expert-{index % 6:04d}.json"
        write_json_durable(descriptor_path, descriptor)
        descriptor_reloaded = read_json(descriptor_path)
        require(descriptor_reloaded == descriptor, f"descriptor durable round-trip drift: {index}")
        require(descriptor_reloaded.get("provenance", {}).get("language") == LANGUAGE, f"descriptor language drift: {index}")
        require(descriptor_reloaded.get("target_excluded_from_model_input") is True, f"descriptor target exclusion drift: {index}")
        require(descriptor_reloaded.get("provenance", {}).get("target_xyz_in_model_input") is False, f"descriptor target XYZ boundary drift: {index}")

        historical_path = historical_snapshot_root / f"stage22_expert_{target_id}" / f"expert-{index % 6:04d}.json"
        if historical_path.is_file() and not historical_path.is_symlink():
            historical_rows.append({"window_id": window_id, "status": "PRESENT_NOT_EXECUTION_AUTHORITY", "path": str(historical_path), "sha256": sha256_file(historical_path)})
        else:
            historical_rows.append({"window_id": window_id, "status": "UNAVAILABLE", "path": str(historical_path), "execution_authority": False})

        equality_rows.append({
            "window_id": window_id,
            "descriptor_relative_path": str(descriptor_path.relative_to(snapshot_root)),
            "descriptor_sha256": sha256_file(descriptor_path),
            "canonical_parquet_path": projection["path"],
            "canonical_parquet_sha256": projection["sha256"],
            "body_video_path": str(body_video.resolve()),
            "body_video_sha256": sha256_file(body_video),
            "body_frame_relative_path": str(body_frame.relative_to(snapshot_root)),
            "body_frame_sha256": sha256_file(body_frame),
            "wrist_video_path": str(wrist_video.resolve()),
            "wrist_video_sha256": sha256_file(wrist_video),
            "wrist_frame_relative_path": str(wrist_frame.relative_to(snapshot_root)),
            "wrist_frame_sha256": sha256_file(wrist_frame),
            "rgb_shape": list(RGB_SHAPE),
            "rgb_frame_index": 0,
            "source_projection_hashes": _source_projection_hashes(projection),
            "language": LANGUAGE,
            "target_id": target_id,
            "target_xyz_in_model_input": False,
            "target_excluded_from_model_input": True,
        })

    manifest = {
        "schema": CLOSURE_SCHEMA,
        "status": "PASS",
        "snapshot_root": str(snapshot_root),
        "materialization": {
            "kind": "DETERMINISTIC_FROM_ACCEPTED_FULL24_TRAIN_PARQUET_AND_VIDEO",
            "scientific_data_generated": False,
            "model_execution": False,
            "training": False,
            "Validation-E": False,
            "Holdout-F": False,
            "historical_json_execution_authority": False,
        },
        "dataset": {
            "root": str(dataset_root),
            "tree_sha256": dataset_digest,
            "regular_file_count": dataset_file_count,
            "total_bytes": dataset_bytes,
            "window_count": 24,
            "frame_count": 960,
            "split": "TRAIN_ONLY",
            "window_order": "target-major then snapshot-major",
        },
        "source_preflight": {
            "path": str(source_preflight),
            "sha256": sha256_file(source_preflight),
            "schema": SOURCE_PREFLIGHT_SCHEMA,
            "status": "PASS",
            "authority": "accepted_stage45c_parquet_projection",
        },
        "historical_witness": {
            "root": str(historical_snapshot_root),
            "execution_authority": False,
            "present_count": sum(item["status"] == "PRESENT_NOT_EXECUTION_AUTHORITY" for item in historical_rows),
            "unavailable_count": sum(item["status"] == "UNAVAILABLE" for item in historical_rows),
            "records": historical_rows,
        },
        "equality_audit": {
            "status": "PASS",
            "all_24": True,
            "exact_fields": ["body_rgb", "wrist_rgb", "state_v1", "source_timestamps", "expert_physical_action", "language"],
            "rgb_decoder": "OpenCV CPU FFmpeg backend; frame index 0; RGB uint8 NHWC",
            "rgb_max_abs_error": 0,
            "state_v1_exact_equal": True,
            "source_timestamps_exact_equal": True,
            "expert_physical_action_exact_equal": True,
            "language_exact_equal": True,
            "target_excluded_from_model_input": True,
            "target_xyz_in_model_input": False,
        },
        "ordered_records": equality_rows,
    }
    write_json_durable(snapshot_root / CLOSURE_MANIFEST_NAME, manifest)
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "PASS",
        "snapshot_root": str(snapshot_root),
        "manifest_path": str(snapshot_root / CLOSURE_MANIFEST_NAME),
        "manifest_sha256": sha256_file(snapshot_root / CLOSURE_MANIFEST_NAME),
        "descriptor_count": len(equality_rows),
        "rgb_stream_count": 2 * len(equality_rows),
        "historical_present_count": manifest["historical_witness"]["present_count"],
        "historical_unavailable_count": manifest["historical_witness"]["unavailable_count"],
        "equality_audit": manifest["equality_audit"],
    }


def np_equal(left: Any, right: Any) -> Any:
    import numpy as np

    return np.array_equal(left, right)


def verify(
    *,
    dataset_root: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    """Fail closed on the closure and its source projection before model import."""

    dataset_root = dataset_root.resolve()
    snapshot_root = snapshot_root.resolve()
    regular_directory(dataset_root, "FULL24 dataset root")
    regular_directory(snapshot_root, "TRAIN24 descriptor closure root")
    manifest_path = snapshot_root / CLOSURE_MANIFEST_NAME
    regular_file(manifest_path, "TRAIN24 descriptor closure manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == CLOSURE_SCHEMA and manifest.get("status") == "PASS", "descriptor closure manifest schema/status drift")
    require(manifest.get("snapshot_root") == str(snapshot_root), "descriptor closure root identity drift")
    dataset_digest, dataset_file_count, dataset_bytes = tree_digest(dataset_root, "payload/lerobot_train")
    dataset_identity = manifest.get("dataset")
    require(isinstance(dataset_identity, Mapping), "descriptor closure dataset identity missing")
    require(dataset_identity.get("root") == str(dataset_root), "descriptor closure dataset root drift")
    require(dataset_identity.get("tree_sha256") == DATASET_TREE_SHA256 == dataset_digest, "descriptor closure dataset tree mismatch")
    require(dataset_identity.get("regular_file_count") == dataset_file_count and dataset_identity.get("total_bytes") == dataset_bytes, "descriptor closure dataset size identity mismatch")
    source_identity = manifest.get("source_preflight")
    require(isinstance(source_identity, Mapping), "descriptor closure source preflight identity missing")
    source_path = Path(str(source_identity.get("path"))).resolve()
    regular_file(source_path, "descriptor closure source preflight")
    require(sha256_file(source_path) == source_identity.get("sha256"), "descriptor closure source preflight SHA mismatch")
    source_document = _load_source_preflight(source_path, dataset_root)
    source_rows = source_document["snapshots"]
    historical = manifest.get("historical_witness")
    require(isinstance(historical, Mapping) and historical.get("execution_authority") is False, "historical witness authority drift")
    records = manifest.get("ordered_records")
    require(isinstance(records, list) and len(records) == 24, "descriptor closure record count drift")
    checked: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        require(isinstance(record, Mapping), f"descriptor closure record is not an object: {index}")
        window_id = _window_id(index)
        target_id = _target_id(index)
        require(record.get("window_id") == window_id and record.get("target_id") == target_id, f"descriptor closure order drift: {index}")
        descriptor_path = snapshot_root / str(record.get("descriptor_relative_path"))
        regular_file(descriptor_path, f"descriptor {window_id}")
        require(sha256_file(descriptor_path) == record.get("descriptor_sha256"), f"descriptor SHA mismatch: {window_id}")
        descriptor = read_json(descriptor_path)
        require(descriptor.get("schema") == "SameSceneExpertSnapshotV1", f"descriptor schema drift: {window_id}")
        require(descriptor.get("example_id") == window_id and descriptor.get("target_id") == target_id, f"descriptor identity drift: {window_id}")
        require(descriptor.get("target_excluded_from_model_input") is True, f"descriptor target exclusion drift: {window_id}")
        provenance = descriptor.get("provenance")
        require(isinstance(provenance, Mapping) and provenance.get("language") == LANGUAGE and provenance.get("target_xyz_in_model_input") is False, f"descriptor provenance boundary drift: {window_id}")
        body_frame = Path(str(descriptor.get("body_frame_path"))).resolve()
        wrist_frame = Path(str(descriptor.get("wrist_frame_path"))).resolve()
        require(body_frame.is_relative_to(snapshot_root) and wrist_frame.is_relative_to(snapshot_root), f"descriptor frame escapes closure root: {window_id}")
        regular_file(body_frame, f"body frame {window_id}")
        regular_file(wrist_frame, f"wrist frame {window_id}")
        require(sha256_file(body_frame) == record.get("body_frame_sha256"), f"body frame SHA mismatch: {window_id}")
        require(sha256_file(wrist_frame) == record.get("wrist_frame_sha256"), f"wrist frame SHA mismatch: {window_id}")
        body_video = Path(str(record.get("body_video_path"))).resolve()
        wrist_video = Path(str(record.get("wrist_video_path"))).resolve()
        require(body_video == _video_path(dataset_root, index, "external_rgb").resolve() and wrist_video == _video_path(dataset_root, index, "wrist_rgb").resolve(), f"descriptor video source drift: {window_id}")
        require(sha256_file(body_video) == record.get("body_video_sha256"), f"body video SHA mismatch: {window_id}")
        require(sha256_file(wrist_video) == record.get("wrist_video_sha256"), f"wrist video SHA mismatch: {window_id}")
        body_rgb = decode_first_rgb(body_video)
        wrist_rgb = decode_first_rgb(wrist_video)
        require(np_equal(read_ppm(body_frame), body_rgb) and np_equal(read_ppm(wrist_frame), wrist_rgb), f"descriptor RGB does not equal accepted video frame zero: {window_id}")

        projection = read_parquet_projection(Path(str(record.get("canonical_parquet_path"))))
        require(projection["path"] == record.get("canonical_parquet_path") and projection["sha256"] == record.get("canonical_parquet_sha256"), f"descriptor parquet source drift: {window_id}")
        source_row = source_rows[index]
        require(source_row.get("canonical_parquet_path") == projection["path"] and source_row.get("canonical_parquet_sha256") == projection["sha256"], f"descriptor/source parquet binding drift: {window_id}")
        state = descriptor.get("state_v1", {}).get("values")
        desired = descriptor.get("future_desired_eef_9d")
        gripper = descriptor.get("future_gripper")
        timestamps = descriptor.get("future_timestamps")
        action = [list(row) + list(gripper[row_index]) for row_index, row in enumerate(desired or [])]
        require(state == projection["state_v1"] and timestamps == projection["source_timestamps"] and action == projection["expert_physical_action"], f"descriptor numeric projection drift: {window_id}")
        require(state == source_row.get("state_v1") and timestamps == source_row.get("source_timestamps") and action == source_row.get("expert_physical_action"), f"descriptor/source preflight equality drift: {window_id}")
        checked.append({"window_id": window_id, "state_v1": True, "timestamps": True, "expert_physical_action": True, "language": True, "body_rgb": True, "wrist_rgb": True})
    audit = manifest.get("equality_audit")
    require(isinstance(audit, Mapping) and audit.get("status") == "PASS" and audit.get("all_24") is True and audit.get("target_xyz_in_model_input") is False, "descriptor closure equality audit is not PASS")
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "descriptor_count": len(checked),
        "rgb_stream_count": 2 * len(checked),
        "all_24_exact": True,
        "target_xyz_in_model_input": False,
        "historical_execution_authority": False,
    }


def static_guard() -> dict[str, Any]:
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".", 1)[0])
    forbidden = sorted(top_level.intersection({"torch", "transformers", "gr00t", "cuda"}))
    return {"schema": "STAGE45D_D1_TRAIN24_DESCRIPTOR_CLOSURE_STATIC_GUARD_V1", "status": "PASS" if not forbidden else "FAIL", "top_level_forbidden_imports": forbidden, "cpu_only": True, "model_execution": False, "training": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--materialize", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--static-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--source-preflight", type=Path)
    parser.add_argument("--historical-snapshot-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.static_only:
            result = static_guard()
        elif args.materialize:
            require(args.dataset_root is not None and args.snapshot_root is not None and args.source_preflight is not None and args.historical_snapshot_root is not None, "materialize requires all four paths")
            result = materialize(dataset_root=args.dataset_root, snapshot_root=args.snapshot_root, source_preflight=args.source_preflight, historical_snapshot_root=args.historical_snapshot_root)
        else:
            require(args.dataset_root is not None and args.snapshot_root is not None, "verify requires dataset-root and snapshot-root")
            result = verify(dataset_root=args.dataset_root, snapshot_root=args.snapshot_root)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"STAGE45D_D1_TRAIN24_DESCRIPTOR_CLOSURE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
