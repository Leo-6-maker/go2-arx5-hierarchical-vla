#!/usr/bin/env python3
"""PI0087 native TorchCodec/FFmpeg production-path qualification.

This qualification is deliberately stronger than importing ``torchcodec``.  It
checks the immutable FFmpeg payload, the exact Python/Torch/TorchCodec runtime,
the dynamic linker bindings, every FULL24 video frame in both RGB modalities,
and the same Stage38-D shard/cache/DataLoader path used by the trainer.  It
does not construct a GR00T model, enter a trainer, run a forward/backward pass,
or submit PBS work.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import traceback
from typing import Any, Iterable, Mapping


BRANCH = "stage45b/full24-fresh-base-execution"
EXPECTED_REAL_PYTHON = "/app1/Hopper/shared/apps/python/3.12.3/bin/python3.12"
EXPECTED_TORCH = "2.9.0+cu128"
EXPECTED_CUDA_BUILD = "12.8"
EXPECTED_TORCHCODEC = "0.8.0"
EXPECTED_FFMPEG_VERSION = "6.1.1"
EXPECTED_FFMPEG_RUNTIME_SCHEMA = "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1"
EXPECTED_VIDEO_TOKEN = "VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED"
EXPECTED_VIDEO_SCHEMA = "VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED_V1"
EXPECTED_DATASET_TOKEN = "VANDA_STAGE45B_FULL24_RUNTIME_DATASET_QUALIFIED_V2"
EXPECTED_FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
EXPECTED_FULL24_FILES = 85
EXPECTED_FULL24_BYTES = 919661
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
EXPECTED_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
EXPECTED_STATS_SCHEMA = "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2"
EXPECTED_PROTOCOL = "RUNTIME_PARQUET_PROVENANCE_V2"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_GROOT_TREE = "43617c11aa6c4ea1be47594882d797255c1a575b"

FFMPEG_FILES = {
    "libavutil.so.58.29.100": {
        "soname": "libavutil.so.58",
        "sha256": "baa4c9ad311e62076357b4f88e4283d5b4766a91af01d198895df2bb0dd2b4af",
    },
    "libavcodec.so.60.31.102": {
        "soname": "libavcodec.so.60",
        "sha256": "bd1c04b19953ddc41c03557940225233ea1cd62749035f6c1a038019b6a4d8d1",
    },
    "libavformat.so.60.16.100": {
        "soname": "libavformat.so.60",
        "sha256": "ae8e2385e65792f71fbf19c629a22e748204c60015c6084b703590c21599dd4a",
    },
    "libavdevice.so.60.3.100": {
        "soname": "libavdevice.so.60",
        "sha256": "2aa576664396d22773ca2875d099e05d35f21583d2ec2c495cbedf23e8d8e50b",
    },
    "libavfilter.so.9.12.100": {
        "soname": "libavfilter.so.9",
        "sha256": "b01b101cfc98e1a0568cc9f67f4ddf541ebdeb4179146160359b777d60aacb40",
    },
    "libswscale.so.7.5.100": {
        "soname": "libswscale.so.7",
        "sha256": "fb15835e00eb7eec5cc9efb516221e9cb68c806cb868201f62f6dde509e5304b",
    },
    "libswresample.so.4.12.100": {
        "soname": "libswresample.so.4",
        "sha256": "f0733d8cdcfbb0c2818bb9ebdb9d98e9439a126134eb7fa2b3f5d8ffdf278e85",
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = root.resolve()
    mode = root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            if child.is_symlink() or not stat.S_ISREG(mode):
                raise RuntimeError(f"unsafe regular tree entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def artifact_entries(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for top_name in ("lib", "compat"):
        top = root / top_name
        if top.is_symlink() or not top.is_dir():
            raise RuntimeError(f"FFmpeg artifact component is unavailable: {top}")
        for child in sorted(top.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(child)
                resolved = child.resolve(strict=True)
                if not resolved.is_file() or not resolved.resolve().is_relative_to(root):
                    raise RuntimeError(f"FFmpeg artifact symlink escapes root: {child}")
                entries.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": target,
                        "bytes": 0,
                        "sha256": None,
                    }
                )
            elif stat.S_ISREG(mode):
                entries.append(
                    {
                        "path": relative,
                        "kind": "regular_file",
                        "target": None,
                        "bytes": int(child.stat().st_size),
                        "sha256": sha256_file(child),
                    }
                )
            else:
                raise RuntimeError(f"unsupported FFmpeg artifact entry: {child}")
    return sorted(entries, key=lambda item: str(item["path"]))


def artifact_tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            "\t".join(
                (
                    str(entry["kind"]),
                    str(entry["path"]),
                    str(entry["bytes"]),
                    str(entry.get("sha256") or "-"),
                    str(entry.get("target") or "-"),
                )
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def _readelf_soname(path: Path) -> str:
    result = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"readelf failed for {path}: {result.stderr.strip()}")
    match = re.search(r"SONAME\).*\[(.*?)\]", result.stdout)
    if not match:
        raise RuntimeError(f"missing SONAME for {path}")
    return match.group(1)


def verify_runtime_artifact(root: Path, expected_id: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"sealed FFmpeg runtime root is unavailable: {root}")
    manifest_path = root / "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("sealed FFmpeg runtime manifest is unavailable")
    if manifest_path.stat().st_mode & 0o222:
        raise RuntimeError("sealed FFmpeg runtime manifest is writable")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != EXPECTED_FFMPEG_RUNTIME_SCHEMA:
        raise RuntimeError("FFmpeg runtime schema mismatch")
    if manifest.get("status") != "SEALED_IMMUTABLE":
        raise RuntimeError("FFmpeg runtime is not sealed immutable")
    actual_entries = artifact_entries(root)
    actual_id = artifact_tree_digest(actual_entries)
    if manifest.get("artifact_id") != actual_id or manifest.get("payload_tree_sha256") != actual_id:
        raise RuntimeError("FFmpeg runtime payload digest mismatch")
    if expected_id is not None and actual_id != expected_id:
        raise RuntimeError(f"FFmpeg runtime identity mismatch: {actual_id}")
    if not root.name.endswith(actual_id):
        raise RuntimeError("FFmpeg runtime path is not content addressed")
    if manifest.get("payload_entries") != actual_entries:
        raise RuntimeError("FFmpeg runtime payload entry manifest mismatch")
    if manifest.get("ffmpeg", {}).get("version") != EXPECTED_FFMPEG_VERSION:
        raise RuntimeError("FFmpeg version mismatch")
    if manifest.get("ffmpeg", {}).get("system_install_modified") is not False:
        raise RuntimeError("system FFmpeg modification is not explicitly false")
    observed_libraries: dict[str, Any] = {}
    lib_root = root / "lib"
    for filename, expected in FFMPEG_FILES.items():
        path = lib_root / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"sealed FFmpeg binary is unavailable: {path}")
        observed_sha = sha256_file(path)
        observed_soname = _readelf_soname(path)
        if observed_sha != expected["sha256"] or observed_soname != expected["soname"]:
            raise RuntimeError(f"sealed FFmpeg binary identity mismatch: {filename}")
        if path.stat().st_mode & 0o222:
            raise RuntimeError(f"sealed FFmpeg binary is writable: {path}")
        observed_libraries[filename] = {
            "path": str(path),
            "sha256": observed_sha,
            "soname": observed_soname,
            "bytes": int(path.stat().st_size),
        }
    anchor = manifest.get("python_static_executable_symbol_anchor")
    anchor_path = root / "compat" / "libpython3.12.so.1.0"
    if not isinstance(anchor, Mapping) or anchor.get("path") != "compat/libpython3.12.so.1.0":
        raise RuntimeError("static Python dynamic-link anchor metadata is missing")
    if anchor.get("sha256") != sha256_file(anchor_path) or _readelf_soname(anchor_path) != "libpython3.12.so.1.0":
        raise RuntimeError("static Python dynamic-link anchor identity mismatch")
    return {
        "schema": manifest["schema"],
        "root": str(root),
        "artifact_id": actual_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "payload_tree_sha256": actual_id,
        "payload_entry_count": len(actual_entries),
        "libraries": observed_libraries,
        "python_static_executable_symbol_anchor": dict(anchor),
        "ffmpeg_version": EXPECTED_FFMPEG_VERSION,
    }


def package_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"package root is unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                raise RuntimeError(f"package tree contains a symlink directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            if child.is_symlink() or not stat.S_ISREG(mode):
                raise RuntimeError(f"package tree contains an unsafe file: {child}")
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    entries.sort(key=lambda item: str(item["path"]))
    return {
        "root": str(root),
        "tree_sha256": tree_digest(entries),
        "regular_file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
    }


def _ldd_bindings(path: Path, runtime_root: Path) -> dict[str, Any]:
    result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ldd failed for {path}: {result.stderr.strip()}")
    bindings: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "lib" not in stripped:
            continue
        name = stripped.split()[0]
        if not (name.startswith("libav") or name.startswith("libsw") or name.startswith("libtorch") or name.startswith("libc10") or name.startswith("libpython")):
            continue
        if "not found" in stripped:
            raise RuntimeError(f"unresolved native dependency for {path}: {stripped}")
        match = re.search(r"=>\s+(\S+)", stripped)
        resolved = match.group(1) if match else (stripped.split()[0] if stripped.startswith("/") else None)
        if resolved is None or not resolved.startswith("/"):
            raise RuntimeError(f"could not parse native dependency for {path}: {stripped}")
        resolved_path = Path(resolved).resolve()
        if name.startswith(("libav", "libsw")) and not resolved_path.is_relative_to(runtime_root / "lib"):
            raise RuntimeError(f"{name} resolved outside sealed FFmpeg runtime: {resolved_path}")
        bindings.append({"name": name, "resolved_path": str(resolved_path), "line": stripped})
    return {"path": str(path), "returncode": result.returncode, "bindings": bindings, "raw": result.stdout}


def native_binding_record(runtime: Mapping[str, Any], torchcodec_root: Path) -> dict[str, Any]:
    runtime_root = Path(str(runtime["root"])).resolve()
    native_paths = [
        torchcodec_root / "libtorchcodec_core6.so",
        torchcodec_root / "libtorchcodec_custom_ops6.so",
        torchcodec_root / "libtorchcodec_pybind_ops6.so",
    ]
    ldd_records = []
    for path in native_paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"TorchCodec native library is unavailable: {path}")
        ctypes.CDLL(str(path), mode=os.RTLD_GLOBAL)
        ldd_records.append(_ldd_bindings(path, runtime_root))
    return {
        "native_libraries": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
            for path in native_paths
        ],
        "ldd": ldd_records,
    }


def mapped_libav_paths(runtime_root: Path) -> list[str]:
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        if "/libav" not in line and "/libsw" not in line:
            continue
        fields = line.split()
        for field in reversed(fields):
            if field.startswith("/") and ("/libav" in field or "/libsw" in field):
                candidate = Path(field).resolve()
                if not candidate.is_relative_to(runtime_root / "lib"):
                    raise RuntimeError(f"mapped libav/libsw library is outside sealed runtime: {candidate}")
                paths.add(str(candidate))
                break
    if not paths:
        raise RuntimeError("no sealed libav/libsw libraries were observed in process maps")
    return sorted(paths)


def verify_environment(runtime_root: Path) -> dict[str, Any]:
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [item for item in ld.split(":") if item]
    expected_prefix = [str(runtime_root / "lib"), str(runtime_root / "compat")]
    if parts[:2] != expected_prefix:
        raise RuntimeError(
            "LD_LIBRARY_PATH must start with sealed FFmpeg lib then compatibility anchor: "
            f"{parts[:2]!r}"
        )
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"offline/runtime environment mismatch: {name}")
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError("ambient PYTHONPATH is not allowed")
    if os.environ.get("STAGE45B_LABEL_ANCHOR_RECONCILIATION") != "1":
        raise RuntimeError("Stage45B label-anchor reconciliation is not enabled")
    return {"ld_library_path": ld, "parts": parts, "offline_environment": expected}


def verify_dataset_qualification(path: Path, stats_path: Path) -> dict[str, Any]:
    document = read_json(path)
    if document.get("status") != EXPECTED_DATASET_TOKEN or document.get("return_token") != EXPECTED_DATASET_TOKEN:
        raise RuntimeError("existing 24/24 runtime-dataset qualification token is missing")
    if document.get("runtime_dataset_label_count") != 24:
        raise RuntimeError("existing runtime-dataset qualification is not FULL24")
    if document.get("protocol_revision") != EXPECTED_PROTOCOL:
        raise RuntimeError("existing runtime-dataset qualification protocol changed")
    boundary = document.get("execution_boundary", {})
    for key in ("gpu_or_simulator_execution", "model_constructed", "gr00t_forwards", "wbc_steps", "training_steps", "backward_steps", "optimizer_steps", "pbs_submissions", "validation_e_read", "holdout_f_read"):
        if boundary.get(key) not in (False, 0):
            raise RuntimeError(f"existing runtime-dataset qualification crossed boundary: {key}")
    stats_identity = document.get("stats_artifact")
    if not isinstance(stats_identity, Mapping) or stats_identity.get("sha256") != EXPECTED_STATS_SHA:
        raise RuntimeError("existing runtime-dataset qualification does not bind V2 stats")
    if sha256_file(stats_path) != EXPECTED_STATS_SHA:
        raise RuntimeError("runtime-parquet V2 stats identity changed")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema": document.get("schema"),
        "status": document["status"],
        "label_count": document["runtime_dataset_label_count"],
        "stats_sha256": stats_identity["sha256"],
        "execution_boundary": boundary,
    }


def _frame_digest(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sweep_all_videos(dataset: Path, runtime_root: Path) -> dict[str, Any]:
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    modalities = {
        "external_rgb": sorted((dataset / "videos" / "chunk-000" / "observation.images.external_rgb").glob("episode_*.mp4")),
        "wrist_rgb": sorted((dataset / "videos" / "chunk-000" / "observation.images.wrist_rgb").glob("episode_*.mp4")),
    }
    records: list[dict[str, Any]] = []
    for modality, paths in modalities.items():
        if len(paths) != 24:
            raise RuntimeError(f"{modality} video cohort is not 24 files: {len(paths)}")
        for path in paths:
            decoder = VideoDecoder(
                str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
            )
            length = len(decoder)
            if length != 40:
                raise RuntimeError(f"video length mismatch for {path}: {length}")
            metadata = decoder.metadata
            packet = decoder.get_frames_at(indices=range(length))
            frames = packet.data.numpy()
            pts = packet.pts_seconds.numpy()
            if tuple(frames.shape) != (40, 96, 96, 3):
                raise RuntimeError(f"video shape mismatch for {path}: {frames.shape}")
            if not np.isfinite(frames.astype(np.float32)).all() or not np.isfinite(pts).all():
                raise RuntimeError(f"non-finite decoded video data: {path}")
            if abs(float(pts[0])) > 1e-9 or abs(float(pts[-1]) - 1.95) > 1e-9:
                raise RuntimeError(f"video timestamp endpoint mismatch: {path}")
            if np.max(np.abs(np.diff(pts) - 0.05)) > 1e-9:
                raise RuntimeError(f"video timestamp grid mismatch: {path}")
            records.append(
                {
                    "modality": modality,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "frame_count": length,
                    "frame_shape": list(frames.shape),
                    "frame_dtype": str(frames.dtype),
                    "frames_sha256": _frame_digest(frames),
                    "pts_sha256": _frame_digest(pts),
                    "pts_first": float(pts[0]),
                    "pts_last": float(pts[-1]),
                    "fps": float(metadata.average_fps),
                    "codec": str(metadata.codec),
                    "decoder": "torchcodec.decoders.VideoDecoder",
                    "decoder_kwargs": {
                        "device": "cpu",
                        "dimension_order": "NHWC",
                        "num_ffmpeg_threads": 0,
                    },
                }
            )
            del frames, pts, packet, decoder
    if len(records) != 48 or {item["modality"] for item in records} != {"external_rgb", "wrist_rgb"}:
        raise RuntimeError("FULL24 video sweep did not cover both RGB modalities")
    return {
        "status": "PASS",
        "video_file_count": len(records),
        "files_per_modality": {name: sum(item["modality"] == name for item in records) for name in modalities},
        "frames_per_file": 40,
        "total_frames": sum(int(item["frame_count"]) for item in records),
        "records": records,
        "mapped_libav_paths_after_sweep": mapped_libav_paths(runtime_root),
    }


def _batch_shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, Mapping):
        return {str(key): _batch_shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_batch_shape(item) for item in value[:4]]
    return type(value).__name__


def production_loader_qualification(
    *, repo: Path, base: Path, dataset: Path, groot: Path, cosmos: Path, config_path: Path,
    qualification_root: Path, expected_stats: Mapping[str, Any], runtime_root: Path,
) -> dict[str, Any]:
    """Run the official pipeline dataset factory and trainer-equivalent DataLoader."""
    qualification_root.mkdir(parents=True, exist_ok=False)
    save_cfg_dir = qualification_root / "experiment_cfg"
    save_cfg_dir.mkdir()

    from stage45b_full24_fresh_base_train import _bind_cosmos_processor, _load_config
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory

    _bind_cosmos_processor(cosmos / "snapshots" / EXPECTED_COSMOS_REVISION)
    install_structured_dataset_factory()
    config = _load_config(
        config_path,
        output_root=qualification_root,
        base_model=base,
        dataset=dataset,
    )
    if config.training.dataloader_num_workers != 0:
        raise RuntimeError("frozen Stage45B DataLoader worker count changed")
    if config.training.global_batch_size != 1 or config.training.num_gpus != 1:
        raise RuntimeError("frozen Stage45B batch semantics changed")

    import gr00t.data.dataset.lerobot_episode_loader as loader_module

    decode_calls: list[dict[str, Any]] = []
    executor = None
    original_decode = loader_module.get_frames_by_indices

    def traced_decode(video_path: str, indices: Any, decoder_kwargs: dict | None = None) -> Any:
        index_list = [int(item) for item in indices]
        value = original_decode(video_path, index_list, decoder_kwargs=decoder_kwargs)
        decode_calls.append(
            {
                "path": str(Path(video_path).resolve()),
                "indices": index_list,
                "count": len(index_list),
                "decoder_kwargs": dict(decoder_kwargs or {}),
                "decoded_shape": list(value.shape),
                "decoded_dtype": str(value.dtype),
            }
        )
        return value

    loader_module.get_frames_by_indices = traced_decode
    try:
        pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir)
        train_dataset, eval_dataset = pipeline._create_dataset(save_cfg_dir)
        if eval_dataset is not None:
            raise RuntimeError("Stage45B production dataset unexpectedly constructed evaluation data")
        datasets = getattr(train_dataset, "datasets", None)
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise RuntimeError("production factory did not construct exactly one dataset")
        dataset_impl = datasets[0]
        if len(dataset_impl) != 1 or int(dataset_impl.get_shard_length(0)) != 24:
            raise RuntimeError("frozen FULL24 shard layout is not one 24-sample shard")
        scheduled = []
        for shard in dataset_impl.sharded_episodes:
            for episode_index, step_indices in shard:
                for step_index in step_indices:
                    scheduled.append((int(episode_index), int(step_index)))
        if sorted(scheduled) != [(index, 0) for index in range(24)]:
            raise RuntimeError("production shard schedule does not cover exactly FULL24 step-0 windows")
        if len(getattr(train_dataset, "structured_action_labels", [])) != 24:
            raise RuntimeError("production factory did not retain 24 StructuredAction labels")
        if getattr(train_dataset, "structured_action_statistics", None) != expected_stats:
            raise RuntimeError("production StructuredAction statistics differ from sealed V2 stats")

        import torch

        batch_size = int(config.training.global_batch_size // config.training.num_gpus)
        pin_memory = bool(config.training.dataloader_pin_memory)
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=pipeline.processor.collator,
            num_workers=int(config.training.dataloader_num_workers),
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        children_before = [child.pid for child in multiprocessing.active_children()]
        iterator = iter(data_loader)
        batches = []
        for _ in range(24):
            batches.append(_batch_shape(next(iterator)))
        children_after = [child.pid for child in multiprocessing.active_children()]
        executor = getattr(train_dataset, "_executor", None)
        executor_workers = getattr(executor, "_max_workers", None)
        worker_id = getattr(train_dataset, "worker_id", None)
        mixture_worker_count = getattr(train_dataset, "num_workers", None)
        if children_after:
            raise RuntimeError(f"DataLoader unexpectedly spawned worker subprocesses: {children_after}")
        if worker_id != 0 or mixture_worker_count != 1 or executor_workers != 1:
            raise RuntimeError(
                f"production shard/cache worker semantics changed: worker_id={worker_id}, "
                f"mixture_num_workers={mixture_worker_count}, executor_workers={executor_workers}"
            )
    finally:
        if executor is None:
            candidate = locals().get("train_dataset")
            executor = getattr(candidate, "_executor", None) if candidate is not None else None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        loader_module.get_frames_by_indices = original_decode

    expected_paths = {
        str(path.resolve())
        for path in (dataset / "videos" / "chunk-000").glob("observation.images.*_rgb/episode_*.mp4")
    }
    observed_paths = {str(item["path"]) for item in decode_calls}
    if observed_paths != expected_paths:
        raise RuntimeError(
            f"production loader did not reference every RGB video: "
            f"observed={len(observed_paths)} expected={len(expected_paths)}"
        )
    if not decode_calls or any(item["indices"] != list(range(40)) for item in decode_calls):
        raise RuntimeError("production loader did not decode the exact 0..39 timestamp indices")
    if any(tuple(item["decoded_shape"]) != (40, 96, 96, 3) for item in decode_calls):
        raise RuntimeError("production loader decoded an unexpected RGB shape")
    if children_before or children_after:
        raise RuntimeError("production DataLoader subprocess behavior is not zero-worker")
    return {
        "status": "PASS",
        "dataset_factory": "go2arx5_vla.stage38_structured_training.install_structured_dataset_factory",
        "official_pipeline_dataset_path": "gr00t.model.gr00t_n1d7.setup.Gr00tN1d7Pipeline._create_dataset",
        "dataloader": {
            "batch_size": batch_size,
            "num_workers": int(config.training.dataloader_num_workers),
            "pin_memory": pin_memory,
            "persistent_workers": False,
            "worker_subprocesses_before": children_before,
            "worker_subprocesses_after": children_after,
            "batches_consumed": len(batches),
            "batch_examples": batches[:2],
        },
        "shard_cache": {
            "dataset_count": 1,
            "shard_count": len(dataset_impl),
            "shard_lengths": [int(item) for item in dataset_impl.shard_lengths],
            "scheduled_window_count": len(scheduled),
            "scheduled_windows": scheduled,
            "mixture_worker_id": int(worker_id),
            "mixture_num_workers": int(mixture_worker_count),
            "thread_pool_max_workers": int(executor_workers),
            "background_cache_path": "ShardedMixtureDataset.__iter__ -> cache_next_shard -> StructuredShardedSingleStepDataset.get_shard",
        },
        "production_decode_calls": decode_calls,
        "structured_action_label_count": len(train_dataset.structured_action_labels),
        "scientific_exposure": {
            "model_constructed": False,
            "forward": False,
            "backward": False,
            "optimizer": False,
            "trainer": False,
            "checkpoint_creation": False,
            "pbs_submissions": 0,
            "new_data": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
        },
    }


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    base = Path(args.base_model).resolve()
    dataset = Path(args.dataset).resolve()
    groot = Path(args.groot_source).resolve()
    cosmos = Path(args.cosmos).resolve()
    modality = Path(args.modality).resolve()
    config_path = Path(args.config).resolve()
    stats_path = Path(args.stats).resolve()
    dataset_qualification_path = Path(args.dataset_qualification).resolve()
    runtime_root = Path(args.ffmpeg_runtime_root).resolve()
    evidence_path = Path(args.evidence).resolve()
    qualification_root = Path(args.qualification_root).resolve()
    requested_python = Path(args.requested_python).resolve()
    if git(repo, "branch", "--show-current") != BRANCH:
        raise RuntimeError("Stage45B execution branch mismatch")
    expected_head = args.expected_head
    if git(repo, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("Stage45B execution HEAD mismatch")
    if git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("Stage45B execution checkout is dirty")
    if Path(sys.executable).resolve() != requested_python or str(requested_python) != EXPECTED_REAL_PYTHON:
        raise RuntimeError(
            f"exact Python mismatch: requested={requested_python}, actual={Path(sys.executable).resolve()}"
        )
    if not evidence_path.parent.is_dir() or evidence_path.exists() or evidence_path.is_symlink():
        raise RuntimeError("video qualification evidence path must be new in an existing evidence root")
    if qualification_root.exists() or qualification_root.is_symlink():
        raise RuntimeError("video qualification root must be absent before qualification")
    if not groot.is_dir() or (groot / ".git").exists():
        raise RuntimeError("hostile gitless GR00T source root is not gitless")

    bundle_document = None
    if args.bundle is not None:
        sys.path.insert(0, str(repo / "scripts"))
        sys.path.insert(0, str(repo / "src"))
        from stage45b_external_admission import assert_bundle

        bundle_path = Path(args.bundle).resolve()
        bundle_document = read_json(bundle_path)
        assert_bundle(bundle_document, repo=repo, expected_head=expected_head, bundle_path=bundle_path)
        if args.release is not None:
            if not args.expected_release_sha256:
                raise RuntimeError("Release V2 SHA is required with a release argument")
            from stage45b_production_release import assert_release

            release_path = Path(args.release).resolve()
            assert_release(
                read_json(release_path),
                repo=repo,
                expected_head=expected_head,
                release_path=release_path,
                expected_release_sha256=args.expected_release_sha256,
                bundle_path=bundle_path,
            )

    artifact = verify_runtime_artifact(runtime_root, expected_id=args.expected_ffmpeg_runtime_id)
    environment = verify_environment(runtime_root)
    dataset_qualification = verify_dataset_qualification(dataset_qualification_path, stats_path)
    if tree_digest(regular_tree_entries(base, "payload/GR00T-N1.7-3B")) != EXPECTED_BASE_TREE:
        raise RuntimeError("BASE identity mismatch")
    full_entries = regular_tree_entries(dataset, "payload/lerobot_train")
    full_tree = tree_digest(full_entries)
    full_bytes = sum(int(item["bytes"]) for item in full_entries)
    if full_tree != EXPECTED_FULL24_TREE or len(full_entries) != EXPECTED_FULL24_FILES or full_bytes != EXPECTED_FULL24_BYTES:
        raise RuntimeError("FULL24 identity mismatch")
    if sha256_file(modality) != EXPECTED_MODALITY_SHA or sha256_file(config_path) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("modality/config identity mismatch")
    stats_document = read_json(stats_path)
    if sha256_file(stats_path) != EXPECTED_STATS_SHA or stats_document.get("schema") != EXPECTED_STATS_SCHEMA or stats_document.get("protocol_revision") != EXPECTED_PROTOCOL or stats_document.get("statistics_payload_sha256") != EXPECTED_STATS_PAYLOAD_SHA:
        raise RuntimeError("runtime-parquet V2 stats identity mismatch")
    if stats_document.get("canonical_label_source", {}).get("tree_sha256") != EXPECTED_FULL24_TREE:
        raise RuntimeError("runtime-parquet V2 stats source mismatch")

    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(groot))
    from stage45b_groot_source_snapshot import verify_extracted
    from stage45b_premodel_runtime_smoke import import_chain

    snapshot_check = verify_extracted(
        groot,
        Path(args.groot_manifest).resolve(),
        expected_manifest_file_sha256=args.groot_manifest_sha256,
        expected_manifest_payload_sha256=args.groot_manifest_payload_sha256,
        expected_payload_tree_sha256=args.groot_payload_tree_sha256,
    )
    if snapshot_check.get("origin_commit") != EXPECTED_GROOT_COMMIT or snapshot_check.get("origin_tree") != EXPECTED_GROOT_TREE:
        raise RuntimeError("GR00T snapshot origin identity mismatch")
    import_results = import_chain(repo, groot)

    import torch
    torch_version = str(torch.__version__)
    cuda_build = str(torch.version.cuda)
    if torch_version != EXPECTED_TORCH or cuda_build != EXPECTED_CUDA_BUILD:
        raise RuntimeError(f"Torch identity mismatch: {torch_version} / {cuda_build}")
    torch_root = Path(torch.__file__).resolve().parent
    torch_identity = package_identity(torch_root)
    import torchcodec
    from torchcodec._core.ops import _get_json_ffmpeg_library_versions

    torchcodec_version = __import__("importlib.metadata", fromlist=["version"]).version("torchcodec")
    if str(torchcodec_version) != EXPECTED_TORCHCODEC:
        raise RuntimeError(f"TorchCodec version mismatch: {torchcodec_version}")
    torchcodec_root = Path(torchcodec.__file__).resolve().parent
    if torchcodec_root != Path(args.torchcodec_package_root).resolve():
        raise RuntimeError("TorchCodec imported outside the exact venv package root")
    torchcodec_identity = package_identity(torchcodec_root)
    native_bindings = native_binding_record(artifact, torchcodec_root)
    ffmpeg_versions = json.loads(_get_json_ffmpeg_library_versions())
    if ffmpeg_versions.get("ffmpeg_version") != EXPECTED_FFMPEG_VERSION or ffmpeg_versions.get("libavcodec") != [60, 31, 102] or ffmpeg_versions.get("libavformat") != [60, 16, 100] or ffmpeg_versions.get("libavutil") != [58, 29, 100] or ffmpeg_versions.get("libavfilter") != [9, 12, 100]:
        raise RuntimeError(f"TorchCodec native FFmpeg versions mismatch: {ffmpeg_versions}")
    maps_after_native_load = mapped_libav_paths(runtime_root)
    video_sweep = sweep_all_videos(dataset, runtime_root)
    loader_root = qualification_root / "production_loader"
    loader_qualification = production_loader_qualification(
        repo=repo,
        base=base,
        dataset=dataset,
        groot=groot,
        cosmos=cosmos,
        config_path=config_path,
        qualification_root=loader_root,
        expected_stats=stats_document.get("statistics", {}),
        runtime_root=runtime_root,
    )
    final_maps = mapped_libav_paths(runtime_root)
    return {
        "schema": EXPECTED_VIDEO_SCHEMA,
        "status": "PASS",
        "token": EXPECTED_VIDEO_TOKEN,
        "pi_gate": "PI_GATE_STATE_0087",
        "stage": "Stage45B",
        "timestamp_utc": now_utc(),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "execution_head": expected_head,
        "branch": BRANCH,
        "requested_python": str(Path(args.requested_python)),
        "python_realpath": str(Path(sys.executable).resolve()),
        "runtime": {
            "requested_interpreter": str(Path(args.requested_python)),
            "sys_executable_realpath": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch_version": torch_version,
            "torch_cuda_build": cuda_build,
        },
        "runtime_environment": environment,
        "ffmpeg_runtime": artifact,
        "native_torchcodec": {
            "package_root": str(torchcodec_root),
            "package_identity": torchcodec_identity,
            "version": str(torchcodec_version),
            "ffmpeg_library_versions": ffmpeg_versions,
            "bindings": native_bindings,
            "mapped_libav_paths_after_native_load": maps_after_native_load,
            "mapped_libav_paths_after_full_qualification": final_maps,
        },
        "torch_runtime": {
            "version": torch_version,
            "cuda_build": cuda_build,
            "file": str(Path(torch.__file__).resolve()),
            "package_identity": torch_identity,
        },
        "groot": {
            "source_root": str(groot),
            "git_directory_absent": not (groot / ".git").exists(),
            "snapshot": snapshot_check,
            "imports": import_results,
            "all_imports_from_sealed_source": True,
        },
        "scientific_inputs": {
            "base_tree_sha256": EXPECTED_BASE_TREE,
            "full24_tree_sha256": full_tree,
            "full24_regular_file_count": len(full_entries),
            "full24_total_bytes": full_bytes,
            "runtime_stats_sha256": EXPECTED_STATS_SHA,
            "runtime_stats_payload_sha256": EXPECTED_STATS_PAYLOAD_SHA,
            "runtime_dataset_qualification": dataset_qualification,
            "cosmos_revision": EXPECTED_COSMOS_REVISION,
            "cosmos_tree_sha256": EXPECTED_COSMOS_TREE,
            "modality_sha256": EXPECTED_MODALITY_SHA,
            "config_sha256": EXPECTED_CONFIG_SHA,
        },
        "video_sweep": video_sweep,
        "production_loader": loader_qualification,
        "scientific_exposure": {
            "model_constructed": False,
            "forward": False,
            "backward": False,
            "optimizer": False,
            "trainer": False,
            "checkpoint_creation": False,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
            "new_data": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--requested-python", required=True)
    parser.add_argument("--torchcodec-package-root", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--groot-source", type=Path, required=True)
    parser.add_argument("--groot-manifest", type=Path, required=True)
    parser.add_argument("--groot-manifest-sha256", required=True)
    parser.add_argument("--groot-manifest-payload-sha256", required=True)
    parser.add_argument("--groot-payload-tree-sha256", required=True)
    parser.add_argument("--cosmos", type=Path, required=True)
    parser.add_argument("--modality", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--dataset-qualification", type=Path, required=True)
    parser.add_argument("--ffmpeg-runtime-root", type=Path, required=True)
    parser.add_argument("--expected-ffmpeg-runtime-id", required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--expected-release-sha256")
    args = parser.parse_args()
    try:
        result = qualify(args)
    except Exception as exc:
        failure = {
            "schema": EXPECTED_VIDEO_SCHEMA,
            "status": "FAIL",
            "token": None,
            "pi_gate": "PI_GATE_STATE_0087",
            "timestamp_utc": now_utc(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "scientific_exposure": {
                "model_constructed": False,
                "forward": False,
                "backward": False,
                "optimizer": False,
                "trainer": False,
                "checkpoint_creation": False,
                "pbs_submissions": 0,
            },
        }
        try:
            write_json(Path(args.evidence).resolve(), failure)
        except Exception:
            pass
        print(f"STAGE45B_VIDEO_RUNTIME_BLOCKED: {failure['error']}", file=sys.stderr)
        return 2
    write_json(Path(args.evidence).resolve(), result)
    print(EXPECTED_VIDEO_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
