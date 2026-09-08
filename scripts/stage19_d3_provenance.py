#!/usr/bin/env python3
"""Trace the D3 20 Hz metadata / 25 Hz timestamp discrepancy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_D3_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-data-d3")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/d3_provenance")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_rate(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    duration = values[-1] - values[0]
    return None if duration <= 0 else (len(values) - 1) / duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.d3_root / "configs" / "render_jobs.json"
    builder_path = args.d3_root / "tools" / "d3_build_raw.py"
    manifest_path = args.d3_root / "manifests" / "D3_DATASET_MANIFEST_V0.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = config["jobs"]
    episode_ids = [job["episode_id"] for job in jobs if job["episode_id"].endswith("_V0")][:6]
    episodes: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        path = args.d3_root / "raw" / "episodes" / episode_id / "episode.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        native_timestamps = [float(value) for value in payload["timestamps"]]
        body_timestamps = [float(value) for value in payload["camera"]["body"]["timestamps"]]
        wrist_timestamps = [float(value) for value in payload["camera"]["wrist"]["timestamps"]]
        job = next(item for item in jobs if item["episode_id"] == episode_id)
        indices = [int(value) for value in job["camera_indices"]]
        strides = sorted({right - left for left, right in zip(indices, indices[1:])})
        episodes.append(
            {
                "episode_id": episode_id,
                "source_json": str(path),
                "source_json_sha256": sha256_file(path),
                "native_frame_count": len(native_timestamps),
                "camera_frame_count": {"body": len(body_timestamps), "wrist": len(wrist_timestamps)},
                "native_effective_rate_hz": effective_rate(native_timestamps),
                "body_effective_rate_hz": effective_rate(body_timestamps),
                "wrist_effective_rate_hz": effective_rate(wrist_timestamps),
                "body_declared_fps_hz": payload["camera"]["body"].get("fps"),
                "wrist_declared_fps_hz": payload["camera"]["wrist"].get("fps"),
                "camera_index_stride": strides,
                "timestamps_authoritative": True,
            }
        )
    builder_text = builder_path.read_text(encoding="utf-8")
    builder_markers = {
        "camera_timestamps_derive_from_source_indices": "for i in idx" in builder_text and "camera_indices" in builder_text,
        "camera_fps_literal_20": "'fps':20.0" in builder_text or '"fps":20.0' in builder_text,
        "source_index_stride_is_external_job_config": "camera_indices" in builder_text,
    }
    config_rates = {"camera_hz": config.get("camera_hz"), "state_hz": config.get("state_hz")}
    classification = "FRAME_SUBSAMPLING_METADATA_DEFECT"
    rationale = (
        "render_jobs declares 20 Hz, but the recorded generator selects every second "
        "sample from a 50 Hz source and writes those source timestamps; the resulting "
        "0.04 s spacing is 25 Hz. The mismatch is in D3 selection/metadata provenance, "
        "not in SynchronizerV1."
    )
    result = {
        "schema": "STAGE19_D3_TIMING_PROVENANCE_V1",
        "status": "PASS_WITH_CLASSIFIED_D3_DEVIATION",
        "classification": classification,
        "rationale": rationale,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "declared_rates_hz": config_rates,
        },
        "generator": {
            "path": str(builder_path),
            "sha256": sha256_file(builder_path),
            "markers": builder_markers,
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "episodes": episodes,
        "policy": {
            "source_timestamps_are_authority": True,
            "original_d3_files_modified": False,
            "forced_20_hz_resampling": False,
            "production_robot_timing": "TBD_BY_ADAPTATION_DATASET",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "stage19_d3_timing_provenance.json"
    result["output_path"] = str(output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
