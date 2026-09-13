#!/usr/bin/env python3
"""PI0163 source/compute runtime import-only and CPU media probe.

Dynamic imports are deliberately performed only from ``main``. This module
checks the exact interpreter, FFmpeg shared libraries, TorchCodec CPU decode,
and offline Hugging Face resolution. It never constructs a GR00T model,
performs a model forward, reads Validation-E payloads, or trains.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "STAGE45F_PI0163_RUNTIME_IMPORT_PROBE_V1"
EXPECTED_COSMOS_REPO = "nvidia/Cosmos-Reason2-2B"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


class ProbeError(RuntimeError):
    """Raised when the actual runtime cannot be verified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"fresh probe record required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def loaded_runtime_maps(root: Path) -> list[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return []
    prefix = str(root.resolve())
    return sorted({line.rsplit(" ", 1)[-1].strip() for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines() if prefix in line})


def run_probe(*, mode: str, ffmpeg_root: Path, video: Path, cosmos_root: Path, record: Path, expected_cosmos_revision: str) -> dict[str, Any]:
    require(mode in {"source", "compute"}, f"unsupported probe mode: {mode}")
    ffmpeg_root = ffmpeg_root.resolve(strict=True)
    require((ffmpeg_root / "lib").is_dir() and (ffmpeg_root / "compat").is_dir(), "FFmpeg lib/compat roots are unavailable")
    video = video.resolve(strict=True)
    require(video.is_file() and not video.is_symlink(), f"regular probe video required: {video}")
    cosmos_root = cosmos_root.resolve(strict=True)
    require(cosmos_root.is_dir() and not cosmos_root.is_symlink(), f"Cosmos root unavailable: {cosmos_root}")
    require(expected_cosmos_revision == EXPECTED_COSMOS_REVISION, "Cosmos revision is not allowlisted")
    require(all(os.environ.get(key) == value for key, value in EXPECTED_OFFLINE.items()), f"offline environment drift: {EXPECTED_OFFLINE}")
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE must be enabled")
    library_path = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    require(str(ffmpeg_root / "lib") in library_path, "FFmpeg lib is absent from LD_LIBRARY_PATH")
    require(str(ffmpeg_root / "compat") in library_path, "FFmpeg compat is absent from LD_LIBRARY_PATH")

    library_names = (
        "libavutil.so.58.29.100",
        "libswresample.so.4.12.100",
        "libswscale.so.7.5.100",
        "libavcodec.so.60.31.102",
        "libavformat.so.60.16.100",
        "libavfilter.so.9.12.100",
        "libavdevice.so.60.3.100",
    )
    loaded_libraries: list[dict[str, Any]] = []
    for name in library_names:
        path = ffmpeg_root / "lib" / name
        require(path.is_file() and not path.is_symlink(), f"FFmpeg library missing: {path}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        loaded_libraries.append({"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)})

    torch = importlib.import_module("torch")
    torchcodec = importlib.import_module("torchcodec")
    decoder_module = importlib.import_module("torchcodec.decoders")
    video_decoder = getattr(decoder_module, "VideoDecoder")
    decoder = video_decoder(str(video), device="cpu")
    frame = decoder[0]
    frame_data = getattr(frame, "data", frame)
    shape = [int(value) for value in getattr(frame_data, "shape", ())]
    require(shape, "TorchCodec returned a frame without shape")

    hub = importlib.import_module("huggingface_hub")
    resolved = hub.hf_hub_download(
        repo_id=EXPECTED_COSMOS_REPO,
        filename="config.json",
        revision=expected_cosmos_revision,
        local_files_only=True,
    )
    resolved_path = Path(str(resolved)).resolve(strict=True)
    expected_snapshot = (cosmos_root / "snapshots" / expected_cosmos_revision).resolve(strict=True)
    require(resolved_path.is_relative_to(expected_snapshot), f"offline resolver escaped accepted Cosmos snapshot: {resolved_path}")

    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "mode": mode,
        "timestamp_utc": now_utc(),
        "python": {"executable": sys.executable, "resolved_executable": str(Path(sys.executable).resolve()), "version": sys.version},
        "torch": {"module": str(Path(str(torch.__file__)).resolve()), "version": str(torch.__version__)},
        "torchcodec": {"module": str(Path(str(torchcodec.__file__)).resolve())},
        "ffmpeg_root": str(ffmpeg_root),
        "ffmpeg_libraries": loaded_libraries,
        "loaded_runtime_maps": loaded_runtime_maps(ffmpeg_root),
        "video": {"path": str(video), "sha256": sha256_file(video), "frame_count": int(len(decoder)), "first_frame_shape": shape, "first_frame_dtype": str(getattr(frame_data, "dtype", "unknown"))},
        "cosmos": {"repo_id": EXPECTED_COSMOS_REPO, "revision": expected_cosmos_revision, "root": str(cosmos_root), "resolved_config": str(resolved_path)},
        "offline_environment": {key: os.environ.get(key) for key in sorted(EXPECTED_OFFLINE)},
        "model_construction": False,
        "model_forward": 0,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "network_download": False,
    }
    write_json_durable(record, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "compute"), required=True)
    parser.add_argument("--ffmpeg-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--cosmos-revision", default=EXPECTED_COSMOS_REVISION)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(
            mode=args.mode,
            ffmpeg_root=args.ffmpeg_root,
            video=args.video,
            cosmos_root=args.cosmos_root,
            record=args.record,
            expected_cosmos_revision=args.cosmos_revision,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "mode": args.mode,
            "timestamp_utc": now_utc(),
            "error": f"{type(exc).__name__}: {exc}",
            "model_construction": False,
            "model_forward": 0,
            "training": False,
            "validation_e_read": False,
            "holdout_f_read": False,
        }
        try:
            if not args.record.exists() and not args.record.is_symlink():
                write_json_durable(args.record, failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 92


if __name__ == "__main__":
    raise SystemExit(main())
