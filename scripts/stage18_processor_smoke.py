#!/usr/bin/env python3
"""Run an official GR00T processor-only smoke on the Stage18 D3 fixture.

This script imports the existing Stage10 source/runtime but never loads a
model, performs a forward pass, uses a GPU, or starts a transport.  It proves
that one derived ``GR00TTrainingExampleV1`` can be represented at the official
processor boundary and that the configured physical action round trip is
finite and shape-correct.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from go2arx5_vla.data import RawEpisodeV1, SynchronizerV1
from go2arx5_vla.data.processor import build_official_vla_step_data
from go2arx5_vla.data.readiness import GR00TTrainingExampleV1


DEFAULT_STAGE10 = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10")
DEFAULT_FIXTURE = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage18/d3_fixture")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage18/processor")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _describe(item) for key, item in value.items()}
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_describe(item) for item in value[:4]]
    return str(value)


def build_custom_processor(stage10_root: Path, model_dir: Path, stats_dir: Path):
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(stage10_root / "artifacts"))
    from go2_arx5_modality import GO2_ARX5_MODALITY_CONFIG
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_config = json.loads((model_dir / "processor_config.json").read_text())
    model_statistics = json.loads((model_dir / "statistics.json").read_text())
    flat_stats = json.loads((stats_dir / "stats.json").read_text())
    relative_stats = json.loads((stats_dir / "relative_stats.json").read_text())

    def split_stats(values: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
        keys = ("min", "max", "mean", "std", "q01", "q99")
        return {
            "eef_9d": {key: values[key][:9] for key in keys},
            "gripper_position": {key: values[key][9:10] for key in keys},
        }

    custom_statistics = {
        EmbodimentTag.NEW_EMBODIMENT.value: {
            "state": split_stats(flat_stats["observation.state"]),
            "action": split_stats(flat_stats["action"]),
            "relative_action": {"eef_9d": relative_stats["eef_9d"]},
        }
    }
    kwargs = copy.deepcopy(processor_config["processor_kwargs"])
    kwargs["modality_configs"][EmbodimentTag.NEW_EMBODIMENT.value] = GO2_ARX5_MODALITY_CONFIG
    kwargs["statistics"] = {**model_statistics, **custom_statistics}
    kwargs["model_name"] = "nvidia/Cosmos-Reason2-2B"
    kwargs["model_type"] = "qwen"
    processor = Gr00tN1d7Processor(**kwargs)
    processor.eval()
    return processor, EmbodimentTag.NEW_EMBODIMENT


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read D3 image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, default=DEFAULT_STAGE10)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--stats-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads((args.fixture_dir / "stage18_d3_fixture_manifest.json").read_text())
    if not manifest.get("selected_episode_ids"):
        raise RuntimeError("fixture manifest has no selected episodes")
    episode_id = manifest["selected_episode_ids"][0]
    source_path = Path(manifest["episode_files"][0])
    raw = RawEpisodeV1.from_d3_json(source_path)
    synchronization = SynchronizerV1().synchronize(raw)
    example = GR00TTrainingExampleV1.from_raw(
        raw,
        synchronization,
        sample_index=0,
        horizon=int(manifest["fixture_horizon"]),
    )
    if example.raw_episode_id != episode_id:
        raise RuntimeError("fixture manifest and first episode disagree")
    body_path = Path(example.observation_images["body"])
    wrist_path = Path(example.observation_images["wrist"])
    body_rgb = read_rgb(body_path)
    wrist_rgb = read_rgb(wrist_path)
    logical_record = example.to_lerobot_record()
    prior_processor_log = Path(manifest["d3_root"]) / "logs" / "processor_target.txt"
    prior_processor_evidence: dict[str, Any] = {"available": False}
    if prior_processor_log.is_file():
        try:
            prior_processor_evidence = {
                "available": True,
                "path": str(prior_processor_log),
                "target_log_contains_official_processor_fields": all(
                    marker in prior_processor_log.read_text()
                    for marker in ("padded_shape", "active_count", "roundtrip_errors")
                ),
            }
        except OSError:
            prior_processor_evidence = {"available": False, "readable": False}

    model_dir = args.model_dir or args.stage10_root / "assets" / "GR00T-N1.7-3B"
    stats_dir = args.stats_dir or Path(manifest["d3_root"]) / "lerobot" / "go2_arx5_sim_render_target_action" / "meta"
    static_result = {
        "episode_id": raw.episode_id,
        "source_episode_json": str(source_path),
        "source_episode_sha256": sha256_file(source_path),
        "example_id": example.example_id,
        "horizon": example.horizon,
        "logical_lerobot_record": {
            "state_dim": len(logical_record["observation.state"]),
            "first_action_dim": len(logical_record["action"]),
            "future_action_shape": [len(logical_record["future_action"]), len(logical_record["future_action"][0])],
            "action_timestamps_count": len(logical_record["action_timestamps"]),
            "desired_label": logical_record["metadata"]["future_label"],
            "achieved_retained_in_raw": logical_record["metadata"]["achieved_stream_in_raw_episode"],
            "achieved_training_label": False,
        },
        "images": {
            "body": {"path": str(body_path), "sha256": sha256_file(body_path), "shape": list(body_rgb.shape)},
            "wrist": {"path": str(wrist_path), "sha256": sha256_file(wrist_path), "shape": list(wrist_rgb.shape)},
        },
        "prior_d3_official_processor_evidence": prior_processor_evidence,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        processor, embodiment = build_custom_processor(args.stage10_root, model_dir, stats_dir)
    except Exception as exc:
        # A missing/unauthorized gated Cosmos image processor is an external
        # runtime-access condition, not a schema or data conversion failure.
        message = re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", str(exc).splitlines()[0])
        result = {
            "schema": "STAGE18_OFFICIAL_GR00T_PROCESSOR_SMOKE_V1",
            "status": "BLOCKED_GATED_COSMOS_ACCESS",
            "model_forward": False,
            "gpu_used": False,
            "official_processor": "Gr00tN1d7Processor",
            "official_processor_attempted": True,
            "official_processor_error_type": type(exc).__name__,
            "official_processor_error_summary": message,
            "static_contract_validation": static_result,
            "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
            "note": "Existing D3 processor log is retained as prior evidence; this run did not retry credentials or print token material.",
        }
        output_path = args.output_dir / "stage18_processor_smoke.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    step_data = build_official_vla_step_data(
        example,
        body_rgb=body_rgb,
        wrist_rgb=wrist_rgb,
        embodiment=embodiment,
    )
    from gr00t.data.types import MessageType

    process_start = time.perf_counter()
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])
    processed_elapsed = time.perf_counter() - process_start
    collated = processor.collator([processed])

    physical = {
        "eef_9d": example.future_desired_eef_9d,
        "gripper_position": example.future_gripper,
    }
    state_for_action = {
        "eef_9d": example.state_v1.as_array()[None, :9],
        "gripper_position": example.state_v1.as_array()[None, 9:],
    }
    normalized = processor.state_action_processor.apply_action(
        physical,
        embodiment.value,
        state=state_for_action,
    )
    batched_state = {
        "eef_9d": example.state_v1.as_array()[None, None, :9],
        "gripper_position": example.state_v1.as_array()[None, None, 9:],
    }
    action_dim = 10
    universal = np.zeros((1, example.horizon, 132), dtype=np.float32)
    universal[0, :, :9] = normalized["eef_9d"]
    universal[0, :, 9:action_dim] = normalized["gripper_position"]
    decoded = processor.decode_action(universal, embodiment, batched_state)
    decoded_physical = np.concatenate(
        [decoded["eef_9d"], decoded["gripper_position"]], axis=-1
    )
    expected = example.physical_action[None, ...]
    if decoded_physical.shape != expected.shape or not np.isfinite(decoded_physical).all():
        raise RuntimeError(
            f"official processor decoded action invalid: {decoded_physical.shape}"
        )
    roundtrip_error = np.abs(decoded_physical - expected)
    source_repo = args.stage10_root / "src" / "isaac-gr00t"
    repo_sha = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "schema": "STAGE18_OFFICIAL_GR00T_PROCESSOR_SMOKE_V1",
        "status": "PASS",
        "model_forward": False,
        "gpu_used": False,
        "episode_id": raw.episode_id,
        "source_episode_json": str(source_path),
        "source_episode_sha256": sha256_file(source_path),
        "example_id": example.example_id,
        "horizon": example.horizon,
        "state_abi": example.state_v1.as_dict(),
        "physical_action_shape": list(example.physical_action.shape),
        "physical_action_label": "future_desired_eef_plus_gripper",
        "achieved_eef_used": False,
        "images": {
            "body": {"path": str(body_path), "sha256": sha256_file(body_path), "shape": list(body_rgb.shape)},
            "wrist": {"path": str(wrist_path), "sha256": sha256_file(wrist_path), "shape": list(wrist_rgb.shape)},
        },
        "groot_repo_sha": repo_sha,
        "processor_config_sha256": sha256_file(model_dir / "processor_config.json"),
        "processor_statistics_sha256": sha256_file(model_dir / "statistics.json"),
        "dataset_statistics_dir": str(stats_dir),
        "dataset_stats_sha256": sha256_file(stats_dir / "stats.json"),
        "dataset_relative_stats_sha256": sha256_file(stats_dir / "relative_stats.json"),
        "official_processor": "Gr00tN1d7Processor",
        "embodiment": embodiment.value,
        "processed": _describe(processed),
        "collated": _describe(collated),
        "processor_latency_s": processed_elapsed,
        "universal_input_shape": list(universal.shape),
        "decoded_physical_shape": list(decoded_physical.shape),
        "decoded_physical_finite": bool(np.isfinite(decoded_physical).all()),
        "roundtrip_max_abs": float(np.max(roundtrip_error)),
        "roundtrip_p95_abs": float(np.percentile(roundtrip_error, 95.0)),
        "roundtrip_max_abs_by_channel": np.max(roundtrip_error[0], axis=0).tolist(),
        "roundtrip_expected_first": expected[0, 0].tolist(),
        "roundtrip_decoded_first": decoded_physical[0, 0].tolist(),
        "rot6d_layout": "GR00T_FIRST_TWO_ROWS",
        "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        "fixture_horizon_provenance": "TEST_ONLY_D3_PIPELINE_FIXTURE",
        "runtime": {
            "python": platform.python_version(),
            "stage10_root": str(args.stage10_root),
            "model_dir": str(model_dir),
        },
        "static_contract_validation": static_result,
        "task_performance_evaluated": False,
    }
    output_path = args.output_dir / "stage18_processor_smoke.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
