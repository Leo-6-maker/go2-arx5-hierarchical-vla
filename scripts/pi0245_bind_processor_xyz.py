#!/usr/bin/env python3
"""Bind the pinned action processor's effective XYZ norm without model weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("groot-source", "modality", "base-model", "dataset-meta", "r2-records", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.groot_source), "rev-parse", "HEAD"], text=True).strip() == "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
    sys.path.insert(0, str(args.groot_source))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    spec = importlib.util.spec_from_file_location("pi0245_pinned_modality", args.modality)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modality = getattr(module, "GO2_ARX5_MODALITY_CONFIG", None) or module.SIM_VISUAL_REACH_MODALITY_CONFIG
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    config_file = args.base_model / "processor_config.json"
    model_stats_file = args.base_model / "statistics.json"
    flat_file = args.dataset_meta / "stats.json"
    relative_file = args.dataset_meta / "relative_stats.json"
    kwargs = copy.deepcopy(json.loads(config_file.read_text())["processor_kwargs"])
    kwargs["modality_configs"][tag] = modality
    flat = json.loads(flat_file.read_text())
    relative = json.loads(relative_file.read_text())
    keys = ("min", "max", "mean", "std", "q01", "q99")
    split = lambda values, begin, end: {key: values[key][begin:end] for key in keys}
    stats = {
        **json.loads(model_stats_file.read_text()),
        tag: {
            "state": {"eef_9d": split(flat["observation.state"], 0, 9), "gripper_position": split(flat["observation.state"], 9, 10)},
            "action": {"eef_9d": split(flat["action"], 0, 9), "gripper_position": split(flat["action"], 9, 10)},
            "relative_action": {"eef_9d": relative["eef_9d"]},
        },
    }
    processor = StateActionProcessor(
        modality_configs=kwargs["modality_configs"], statistics=stats,
        use_percentiles=kwargs["use_percentiles"],
        clip_outliers=kwargs["clip_outliers"],
        apply_sincos_state_encoding=kwargs["apply_sincos_state_encoding"],
        use_relative_action=kwargs["use_relative_action"],
    )
    processor.eval()
    params = processor.norm_params[tag]["action"]["eef_9d"]
    state_params = processor.norm_params[tag]["state"]["eef_9d"]
    lo = np.asarray(params["min"], dtype=np.float64)
    hi = np.asarray(params["max"], dtype=np.float64)
    assert lo.shape == hi.shape == (40, 9)
    assert np.array_equal(lo, np.asarray(relative["eef_9d"]["min"]))
    assert np.array_equal(hi, np.asarray(relative["eef_9d"]["max"]))
    assert kwargs["use_relative_action"] is True
    assert "eef_9d" not in (modality["action"].mean_std_embedding_keys or [])

    rows = [json.loads(line) for line in args.r2_records.read_text().splitlines() if line.strip()]
    assert len(rows) == 16
    row = rows[0]
    tensor_path = Path(row["tensor_path"])
    assert sha(tensor_path) == row["tensor_sha256"]
    with np.load(tensor_path, allow_pickle=False) as npz:
        raw = npz["raw"]
        state = npz["state"].reshape(-1)
        expected = npz["decoded"]
    assert raw.shape == (1, 40, 132) and state.shape == (10,) and expected.shape == (1, 40, 10)
    decoded = processor.unapply_action(
        {"eef_9d": raw[:, :, :9], "gripper_position": raw[:, :, 9:10]}, tag,
        {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
    )
    actual = np.concatenate([decoded["eef_9d"], decoded["gripper_position"]], axis=-1)
    parity = float(np.max(np.abs(actual - expected)))
    assert parity < 1e-5, parity
    result = {
        "schema": "PI0245_P0_EFFECTIVE_PROCESSOR_XYZ_BINDING_V1",
        "groot_commit": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
        "source_sha256": sha(Path(sys.modules[StateActionProcessor.__module__].__file__)),
        "processor_config_sha256": sha(config_file), "model_statistics_sha256": sha(model_stats_file),
        "flat_stats_sha256": sha(flat_file), "relative_stats_sha256": sha(relative_file),
        "modality_sha256": sha(args.modality), "records_sha256": sha(args.r2_records),
        "tensor_sha256": row["tensor_sha256"], "source_episode_id": row["source_episode_id"],
        "use_relative_action": kwargs["use_relative_action"], "use_percentiles": kwargs["use_percentiles"],
        "clip_outliers": kwargs["clip_outliers"], "eef_meanstd": False,
        "norm_param_minmax_from_relative_override": True,
        "scale_xyz_m": ((hi[:, :3] - lo[:, :3]) / 2.0).tolist(),
        "offset_xyz_m": ((hi[:, :3] + lo[:, :3]) / 2.0).tolist(),
        "shape": [40, 3], "native_decode_max_abs_error": parity,
        "state_min": np.asarray(state_params["min"], dtype=np.float64).tolist(),
        "state_max": np.asarray(state_params["max"], dtype=np.float64).tolist(),
        "state_norm_shape": list(np.asarray(state_params["min"]).shape),
        "model_weight_loads": 0, "model_forwards": 0,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "native_decode_max_abs_error": parity}))


if __name__ == "__main__":
    main()
