"""Export only admitted ARX5 episodes; fit statistics exclusively on TRAIN."""
from __future__ import annotations

import argparse
import resource
from pathlib import Path
import time

import numpy as np

from native_dataset import DOMAIN, OFFICIAL_SHA, TAG, pose10, read, sha, validate_samples, write

STATS_RULE = "TRAIN_ONLY_ALL_FULL_H40_EQUAL_WINDOW_TIMESTEP_WEIGHT_MINMAX_V1"


def statistics(values, axis):
    return {key: np.asarray(value).tolist() for key, value in {
        "min": values.min(axis=axis), "max": values.max(axis=axis), "mean": values.mean(axis=axis),
        "std": values.std(axis=axis), "q01": np.quantile(values, .01, axis=axis, method="linear"),
        "q99": np.quantile(values, .99, axis=axis, method="linear")}.items()}


def export(root, admission_path, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior native export; choose a new derived destination")
    admission = read(admission_path)
    # An actual admission receipt is required; this script does not infer success.
    candidates = admission["episodes"]
    train_groups = {r["group"] for r in candidates if r["split"] == "TRAIN"}
    dev_groups = {r["group"] for r in candidates if r["split"] == "DEV"}
    train_seeds = {r["seed"] for r in candidates if r["split"] == "TRAIN"}
    dev_seeds = {r["seed"] for r in candidates if r["split"] == "DEV"}
    if train_groups & dev_groups or train_seeds & dev_seeds:
        raise ValueError("Episode/layout group and seed split leakage")
    output.mkdir(parents=True)
    records, states, absolute, relatives = [], [], [], []
    for row in candidates:
        if not row["admitted"]:
            continue
        if row["split"] not in ("TRAIN", "DEV"):
            raise ValueError("Commissioning/pilot/legacy data cannot enter formal ARX5 export")
        folder = root / "episodes" / row["id"]
        result = read(folder / "RESULT.json")
        if result["domain"] != DOMAIN or result["status"] != "VALID_COMPLETE":
            raise ValueError("Wrong domain or incomplete raw episode")
        if "samples_sha256" in row and row["samples_sha256"] != sha(folder / "samples.json"):
            raise ValueError("Admission source hash changed")
        samples = read(folder / "samples.json")
        validate_samples(samples, folder)
        references = {ref["ref_id"]: ref for ref in read(folder / "references.json")}
        for index, sample in enumerate(samples[:640]):
            command = sample["supervision"]
            reference = references[command["ref_id"]]
            if reference["committed_prefix_count"] != 8 or command["point_index"] != index % 8:
                raise ValueError("Reference prefix not actually committed")
            if not np.array_equal(command["committed"], reference["planned_H40"][command["point_index"]]) or command["committed_gripper"] != reference["gripper_H40"][command["point_index"]]:
                raise ValueError("Installed command diverges from registered reference")
        anchors = np.asarray([s["policy_obs"]["actual_tcp"] for s in samples[:601]])
        desired = np.asarray([s["supervision"]["committed"] for s in samples[:640]])
        grips = np.asarray([s["supervision"]["committed_gripper"] for s in samples[:640]])
        actual = pose10(anchors, np.asarray([s["policy_obs"]["actual_gripper"] for s in samples[:601]]))
        commands = pose10(desired, grips)
        future = np.arange(601)[:, None] + np.arange(40)
        relative = np.linalg.inv(anchors)[:, None] @ desired[future]
        relative9 = pose10(relative, grips[future])[..., :9]
        instructions = {s["policy_obs"]["instruction"] for s in samples}
        if len(instructions) != 1:
            raise ValueError("Instruction identity changed within episode")
        paths = np.asarray([[s["policy_obs"][camera]["path"] for camera in ("body_rgb", "wrist_rgb")] for s in samples[:601]])
        hashes = np.asarray([[s["policy_obs"][camera]["sha256"] for camera in ("body_rgb", "wrist_rgb")] for s in samples[:601]])
        path = output / "episodes" / (row["id"] + ".npz")
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, actual_state10=actual, committed_absolute10=commands, rgb_paths=paths,
                            rgb_sha256=hashes, valid_window=np.ones(601, dtype=bool), anchor_time_s=np.arange(601) / 20,
                            offset_s=np.arange(40) / 20, instruction=np.asarray(next(iter(instructions))),
                            label_representation=np.asarray("ABSOLUTE_COMMITTED_EEF10"))
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["committed_absolute10"], commands) or saved["valid_window"].shape != (601,):
                raise ValueError("Export readback mismatch")
        records.append({"id": row["id"], "seed": row["seed"], "group": row["group"], "split": row["split"],
                        "file": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size,
                        "raw_root": str(folder.resolve()), "windows": 601, "source_sha": result["source_sha"],
                        "samples_sha256": sha(folder / "samples.json"), "physics_sha256": sha(folder / "physics.npz"),
                        "raw_FILE_MANIFEST_sha256": sha(folder / "FILE_MANIFEST.json")})
        if row["split"] == "TRAIN":
            states.append(actual)
            absolute.append(commands[future])
            relatives.append(relative9)
    if not states:
        raise ValueError("No admitted TRAIN episode; statistics cannot be invented")
    state_values, action_values, relative_values = np.concatenate(states), np.concatenate(absolute), np.concatenate(relatives)
    fitted = {TAG: {"state": {"eef_9d": statistics(state_values[:, :9], 0), "gripper_position": statistics(state_values[:, 9:], 0)},
                    "action": {"eef_9d": statistics(action_values[..., :9], (0, 1)), "gripper_position": statistics(action_values[..., 9:], (0, 1))},
                    "relative_action": {"eef_9d": statistics(relative_values, (0, 1))}}}
    write(output / "statistics.json", fitted)
    write(output / "STATISTICS_PROVENANCE.json", {"rule": STATS_RULE, "TRAIN_ids": [r["id"] for r in records if r["split"] == "TRAIN"],
          "state_anchor_rows": len(state_values), "action_window_timestep_rows": int(np.prod(action_values.shape[:2])),
          "overlap_weighting": "Each full H40 window and each of its40 committed points have equal weight; interior commands intentionally repeated",
          "DEV_PILOT_failed_legacy_in_fit": False, "statistics_sha256": sha(output / "statistics.json"),
          "use_percentiles": False, "clip_outliers": False, "stock_inverse_clip": "Measured during qualification; no processor monkeypatch"})
    manifest = {"domain": DOMAIN, "official_source_sha": OFFICIAL_SHA, "label_representation": "ABSOLUTE_COMMITTED_EEF10",
                "relative_conversion": "Official use_relative_action=True once, anchored at actual observation TCP",
                "episodes": records, "statistics_sha256": sha(output / "statistics.json"), "statistics_rule": STATS_RULE,
                "admission_sha256": sha(admission_path), "TRAIN_windows": sum(r["windows"] for r in records if r["split"] == "TRAIN"),
                "DEV_windows": sum(r["windows"] for r in records if r["split"] == "DEV"), "RAW_DATA_READY": True, "NATIVE_DATALOADER_PASS": "NOT_RUN"}
    write(output / "DATASET_MANIFEST.json", manifest)
    write(output / "modality.json", {"video": ["external_rgb", "wrist_rgb"], "state": {"eef_9d": 9, "gripper_position": 1},
          "action": {"eef_9d": "RELATIVE_EEF_XYZ_ROW_ROT6D", "gripper_position": "ABSOLUTE_CLOSED0_OPEN1"},
          "action_delta_indices": list(range(40)), "max_state_dim": 132, "max_action_dim": 132, "input_raw_action": "absolute_committed",
          "valid_dimensions": 10, "joint_state_policy_input": False})
    write(output / "SPLIT.json", {split: [r["id"] for r in records if r["split"] == split] for split in ("TRAIN", "DEV")})
    write(output / "task_map.json", {"T1": "Place the red cube in the target region and release it.", "instruction_source": "Original episode policy_obs, not substituted"})
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    result = export(args.root, args.admission, args.output)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(Path(args.output) / "EXPORT_RECEIPT.json", {"status": "RAW_DATA_READY", "TRAIN_windows": result["TRAIN_windows"], "DEV_windows": result["DEV_windows"], "elapsed_s": time.monotonic() - begin, "CPU_s": usage.ru_utime + usage.ru_stime, "model_loads": 0, "physics_steps": 0})


if __name__ == "__main__":
    main()
