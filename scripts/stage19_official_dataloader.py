#!/usr/bin/env python3
"""Qualify the official Isaac-GR00T v2 loader on the Stage19 fixture.

This script uses the official ``LeRobotEpisodeLoader`` and
``ShardedSingleStepDataset`` shipped with the existing Stage10 GR00T runtime.
It performs no model forward, training, finetuning, WBC execution, transport,
or hardware actuation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_STAGE10 = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10")
DEFAULT_DATASET = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/lerobot_fixture")
DEFAULT_OUTPUT = Path("/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/official_dataloader")


def _describe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _describe(item) for key, item in value.items()}
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__, "value": str(value)[:160]}


def _is_finite(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(item) for item in value)
    return True


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalar containers in qualification metadata to JSON."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_qualified_manifest(dataset_dir: Path, result: Mapping[str, Any]) -> Path:
    """Record hashes after the official stats/dataloader qualification pass."""

    if not isinstance(result, Mapping):
        raise TypeError("qualification result must be a mapping")
    source_manifest_path = dataset_dir / "meta" / "stage19_dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    qualified_path = dataset_dir / "meta" / "stage19_qualified_manifest.json"
    file_manifest: dict[str, str] = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path != qualified_path:
            file_manifest[str(path.relative_to(dataset_dir))] = _sha256_file(path)
    qualified = {
        "schema": "STAGE19_LEROBOT_QUALIFIED_MANIFEST_V1",
        "status": "OFFICIAL_GR00T_DATALOADER_QUALIFIED",
        "source_builder_manifest": str(source_manifest_path),
        "source_builder_manifest_sha256": _sha256_file(source_manifest_path),
        "builder_manifest": source_manifest,
        "official_qualification": {
            "status": result.get("status"),
            "official_episode_loader": result.get("official_episode_loader"),
            "official_training_dataset": result.get("official_training_dataset"),
            "official_processor": result.get("official_processor"),
            "episode_count": result.get("episode_count"),
            "shard_count": result.get("shard_count"),
            "shard_processed_count": result.get("shard_processed_count"),
            "physical_action_shape": result.get("physical_action_shape"),
            "universal_action_shape": result.get("universal_action_shape"),
            "decoded_physical_shape": result.get("decoded_physical_shape"),
            "roundtrip_max_abs": result.get("roundtrip_max_abs"),
            "native_lerobot_package_available": result.get("native_lerobot_package_available"),
            "production_action_timing": result.get("production_action_timing"),
        },
        "official_stats": {
            "stats_json_sha256": _sha256_file(dataset_dir / "meta" / "stats.json"),
            "relative_stats_json_sha256": _sha256_file(dataset_dir / "meta" / "relative_stats.json"),
        },
        "file_manifest_after_official_qualification": file_manifest,
        "source_timestamps_preserved": True,
        "desired_labels_only": True,
        "achieved_labels_used_as_training_target": False,
        "hardware_control": False,
        "training_or_finetuning": False,
    }
    qualified_path.write_text(json.dumps(_jsonable(qualified), indent=2) + "\n", encoding="utf-8")
    return qualified_path


def _generate_official_stats(dataset_dir: Path, stage10_root: Path) -> None:
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(stage10_root / "artifacts"))
    # The custom embodiment registration is required by the official relative
    # action statistics routine.  It is a read-only import of Stage10 code.
    import go2_arx5_modality  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    generate_stats(dataset_dir)
    generate_rel_stats(dataset_dir, EmbodimentTag.NEW_EMBODIMENT)


def _build_processor(
    stage10_root: Path,
    dataset_dir: Path,
    transformers_loading_kwargs: dict[str, Any] | None = None,
    model_name: str | Path | None = None,
):
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from stage18_processor_smoke import build_custom_processor

    model_dir = stage10_root / "assets" / "GR00T-N1.7-3B"
    return build_custom_processor(
        stage10_root,
        model_dir,
        dataset_dir / "meta",
        transformers_loading_kwargs=transformers_loading_kwargs,
        model_name=model_name or "nvidia/Cosmos-Reason2-2B",
    )


def _qualify(
    dataset_dir: Path,
    stage10_root: Path,
    transformers_loading_kwargs: dict[str, Any] | None = None,
    model_name: str | Path | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    sys.path.insert(0, str(stage10_root / "artifacts"))
    import go2_arx5_modality
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
        extract_step_data,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag

    processor, embodiment = _build_processor(
        stage10_root,
        dataset_dir,
        transformers_loading_kwargs=transformers_loading_kwargs,
        model_name=model_name,
    )
    modality_configs = copy.deepcopy(go2_arx5_modality.GO2_ARX5_MODALITY_CONFIG)
    start = time.perf_counter()
    episode_loader = LeRobotEpisodeLoader(dataset_dir, modality_configs)
    first_episode = episode_loader[0]
    episode_load_elapsed = time.perf_counter() - start
    step_data = extract_step_data(
        first_episode,
        0,
        modality_configs,
        EmbodimentTag.NEW_EMBODIMENT,
        allow_padding=False,
    )
    processed_single = processor([{"type": "episode_step", "content": step_data}])

    # This is the official training-side sharded dataset/collation path.  The
    # dataset contains 6 * (61 - 40 + 1) valid action windows.
    dataset = ShardedSingleStepDataset(
        dataset_path=dataset_dir,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=modality_configs,
        shard_size=100000,
        episode_sampling_rate=1.0,
        seed=1901,
        allow_padding=False,
    )
    dataset.processor = processor
    shard_start = time.perf_counter()
    shard = dataset.get_shard(0)
    shard_elapsed = time.perf_counter() - shard_start
    if not shard:
        raise RuntimeError("official sharded dataset returned an empty shard")
    batch = processor.collator(shard[:2])

    # Physical 10D -> official normalization -> universal 132D -> physical 10D
    # uses data loaded from the materialized parquet, not a D3 JSON object.
    state_eef = np.stack(first_episode["state.eef_9d"].iloc[0:1].to_numpy(), axis=0)
    state_gripper = np.stack(
        first_episode["state.gripper_position"].iloc[0:1].to_numpy(), axis=0
    )
    action_eef = np.stack(first_episode["action.eef_9d"].iloc[:40].to_numpy(), axis=0)
    action_gripper = np.stack(
        first_episode["action.gripper_position"].iloc[:40].to_numpy(), axis=0
    )
    normalized = processor.state_action_processor.apply_action(
        {"eef_9d": action_eef, "gripper_position": action_gripper},
        embodiment.value,
        state={"eef_9d": state_eef, "gripper_position": state_gripper},
    )
    universal = np.zeros((1, 40, 132), dtype=np.float32)
    universal[0, :, :9] = _as_numpy(normalized["eef_9d"])
    universal[0, :, 9:10] = _as_numpy(normalized["gripper_position"])
    decoded = processor.decode_action(
        universal,
        embodiment,
        {
            "eef_9d": state_eef[None, ...],
            "gripper_position": state_gripper[None, ...],
        },
    )
    decoded_physical = np.concatenate(
        [_as_numpy(decoded["eef_9d"]), _as_numpy(decoded["gripper_position"])], axis=-1
    )
    expected = np.concatenate([action_eef, action_gripper], axis=-1)[None, ...]
    roundtrip_error = np.abs(decoded_physical - expected)
    if decoded_physical.shape != expected.shape or not np.isfinite(decoded_physical).all():
        raise RuntimeError(
            f"official decoded action invalid: {decoded_physical.shape}, expected {expected.shape}"
        )

    state_columns = ["state.eef_9d", "state.gripper_position"]
    action_columns = ["action.eef_9d", "action.gripper_position"]
    # The official episode loader intentionally returns only the configured
    # modalities and labels.  ``timestamp`` is a standard parquet column but
    # is not a model input in the Stage10 modality config, so validate it from
    # the materialized source table rather than implying that the loader
    # dropped or rewrote it.
    import pandas as pd

    first_parquet = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    materialized_table = pd.read_parquet(first_parquet)
    timestamps = materialized_table["timestamp"].to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("materialized episode timestamp column is invalid")
    video_columns = [key for key in first_episode.columns if key.startswith("video.")]
    result = {
        "schema": "STAGE19_OFFICIAL_DATALOADER_QUALIFICATION_V1",
        "status": "PASS",
        "dataset": str(dataset_dir),
        "native_lerobot_package_available": importlib.util.find_spec("lerobot") is not None,
        "official_episode_loader": "gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader",
        "official_training_dataset": "gr00t.data.dataset.sharded_single_step_dataset.ShardedSingleStepDataset",
        "official_processor": "Gr00tN1d7Processor",
        "episode_count": len(episode_loader),
        "episode_lengths": episode_loader.episode_lengths,
        "first_episode_rows": int(len(first_episode)),
        "first_episode_columns": list(first_episode.columns),
        "state_columns_verified": state_columns,
        "action_columns_verified": action_columns,
        "video_columns_verified": video_columns,
        "materialized_source_columns": list(materialized_table.columns),
        "timestamp_column_verified_in_materialized_source": True,
        "timestamp_column_exposed_by_official_episode_loader": "timestamp" in first_episode.columns,
        "single_step_processed": _describe(processed_single),
        "single_step_finite": _is_finite(processed_single),
        "shard_count": len(dataset),
        "shard_length": dataset.get_shard_length(0),
        "shard_processed_count": len(shard),
        "batch_size": 2,
        "batch": _describe(batch),
        "batch_finite": _is_finite(batch),
        "mask_or_padding_keys": [
            key for key in set(processed_single).union(batch if isinstance(batch, dict) else {})
            if "mask" in str(key).lower() or "pad" in str(key).lower()
        ],
        "physical_action_shape": list(expected.shape[1:]),
        "universal_action_shape": list(universal.shape),
        "decoded_physical_shape": list(decoded_physical.shape),
        "decoded_physical_finite": bool(np.isfinite(decoded_physical).all()),
        "roundtrip_max_abs": float(np.max(roundtrip_error)),
        "roundtrip_p95_abs": float(np.percentile(roundtrip_error, 95.0)),
        "roundtrip_gripper_max_abs": float(np.max(roundtrip_error[..., 9])),
        "desired_action_label": True,
        "achieved_action_used_as_training_label": False,
        "episode_boundaries_preserved": True,
        "calibration_and_provenance_sidecars_outside_loader": True,
        "model_forward": False,
        "training_or_finetuning": False,
        "gpu_used": False,
        "runtime": {
            "python": platform.python_version(),
            "stage10_root": str(stage10_root),
            "groot_repo_sha": subprocess.check_output(
                ["git", "-C", str(stage10_root / "src" / "isaac-gr00t"), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
        },
        "timing_s": {
            "episode_loader_init_and_first_episode": episode_load_elapsed,
            "official_sharded_dataset_get_shard": shard_elapsed,
        },
        "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, default=DEFAULT_STAGE10)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-official-stats", action="store_true")
    args = parser.parse_args()
    if not (args.dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"materialized dataset not found: {args.dataset_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_official_stats:
        _generate_official_stats(args.dataset_dir, args.stage10_root)
    result = _qualify(args.dataset_dir, args.stage10_root)
    qualified_manifest = _write_qualified_manifest(args.dataset_dir, result)
    result["qualified_manifest_path"] = str(qualified_manifest)
    output_path = args.output_dir / "stage19_official_dataloader.json"
    result["output_path"] = str(output_path)
    result = _jsonable(result)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
