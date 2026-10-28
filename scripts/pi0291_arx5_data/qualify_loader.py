"""Actual fixed official processor/collator CPU smoke, never a network forward."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from native_dataset import DatasetLoader, TAG, OFFICIAL_SHA, build_processor, collate_project, matrix_from10, pose10, read, sha, validate_samples, write


def inventory(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): inventory(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [inventory(item) for item in value]
    return value


def physical_errors(expected, actual):
    expected_matrix, actual_matrix = matrix_from10(expected), matrix_from10(actual)
    delta = expected_matrix[..., :3, :3] @ np.swapaxes(actual_matrix[..., :3, :3], -1, -2)
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return {"XYZ_m": float(np.max(np.abs(expected[..., :3] - actual[..., :3]))),
            "SO3_rad": float(np.max(np.arccos(cosine))), "gripper": float(np.max(np.abs(expected[..., 9] - actual[..., 9])))}


def maxima(left, right):
    return {key: max(left[key], right[key]) for key in left}


def codec_all_windows(dataset_root, processor, splits=("TRAIN", "DEV")):
    root = Path(dataset_root)
    records = read(root / "DATASET_MANIFEST.json")["episodes"]
    from gr00t.data.types import ActionType, ActionFormat
    from gr00t.data.embodiment_tags import EmbodimentTag
    sap = processor.state_action_processor
    reports = {}
    for split in splits:
        normalized_errors = {"XYZ_m": 0., "SO3_rad": 0., "gripper": 0.}
        raw_errors = normalized_errors.copy()
        relative_max_error, preclip, inverseclip, windows = 0., 0, 0, 0
        state_preclip = 0
        for row in records:
            if row["split"] != split:
                continue
            with np.load(root / row["file"], allow_pickle=False) as saved:
                states, commands = saved["actual_state10"], saved["committed_absolute10"]
            for anchor in range(601):
                state, target = states[anchor], commands[anchor:anchor + 40]
                reference = matrix_from10(state)
                expected_rel = np.linalg.inv(reference) @ matrix_from10(target)
                independent_rel = pose10(expected_rel, target[:, 9])
                relative = sap._convert_to_relative_action(target[:, :9], state[:9], ActionType.EEF, ActionFormat.XYZ_ROT6D)
                relative_max_error = max(relative_max_error, float(np.max(np.abs(relative - independent_rel[:, :9]))))
                absolute_raw = sap._convert_to_absolute_action(relative, state[:9], ActionType.EEF, ActionFormat.XYZ_ROT6D)
                raw_roundtrip = np.concatenate((absolute_raw, target[:, 9:]), axis=-1)
                raw_errors = maxima(raw_errors, physical_errors(target, raw_roundtrip))
                state_dict = {"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}
                action_dict = {"eef_9d": target[:, :9], "gripper_position": target[:, 9:]}
                normalized = sap.apply_action(action_dict, TAG, state=state_dict)
                normalized_state = sap.apply_state(state_dict, TAG)
                state_preclip += sum(int(np.count_nonzero(np.abs(v) > 1)) for v in normalized_state.values())
                packed = np.concatenate([normalized[k] for k in ("eef_9d", "gripper_position")], axis=-1)
                preclip += int(np.count_nonzero(np.abs(packed) > 1))
                inverseclip += int(np.count_nonzero(np.abs(packed.astype(np.float32)) > 1))
                universal = np.pad(packed.astype(np.float32), ((0, 0), (0, 122)))[None]
                decoded = processor.decode_action(universal, EmbodimentTag.NEW_EMBODIMENT,
                    {key: value[None] for key, value in state_dict.items()})
                decoded_physical = np.concatenate([decoded[k] for k in ("eef_9d", "gripper_position")], axis=-1)[0]
                normalized_errors = maxima(normalized_errors, physical_errors(target, decoded_physical))
                windows += 1
        reports[split] = {"full_H40_windows_actually_checked": windows, "raw_official_pose_roundtrip": raw_errors,
                          "official_normalized_float32_132D_decode": normalized_errors, "independent_SE3_vs_official_relative_max_abs": relative_max_error,
                          "action_preclip_components": preclip, "stock_inverse_clip_components_float32": inverseclip,
                          "state_preclip_components": state_preclip, "encode_clip_components": 0,
                          "DEV_refit": False, "input_actions": "absolute committed; never pre-relative"}
        if split == "TRAIN" and windows:
            if normalized_errors["XYZ_m"] > 1e-5 or normalized_errors["SO3_rad"] > 1e-4 or normalized_errors["gripper"] > 1e-5 or relative_max_error > 1e-8:
                raise ValueError("TRAIN official codec strict gate failed: " + json.dumps(reports[split]))
    if "TRAIN" in reports and not reports["TRAIN"]["full_H40_windows_actually_checked"]:
        raise ValueError("No TRAIN H40 passed official encode/decode")
    return reports


def check_batch(batch, count):
    inputs = batch["inputs"]
    action, mask = inputs["action"].numpy(), inputs["action_mask"].numpy()
    state = inputs["state"].numpy()
    if action.shape != (count, 40, 132) or mask.shape != action.shape:
        raise ValueError("Official action/padding shape mismatch")
    if not np.all(mask[..., :10] == 1) or not np.all(mask[..., 10:] == 0):
        raise ValueError("Official route must have exactly10 valid EEF/gripper dimensions")
    if state.shape != (count, 1, 132) or not np.all(state[..., 10:] == 0):
        raise ValueError("Official state padding mismatch")
    for value in inputs.values():
        if hasattr(value, "numpy") and not np.isfinite(value.numpy()).all():
            raise ValueError("Nonfinite official batch")
    return inventory(batch)


def negative_gates(root, manifest):
    first = manifest["episodes"][0]
    samples = read(Path(first["raw_root"]) / "samples.json")
    mutations = {}
    for key in ("object_pose", "teacher_stage", "success"):
        altered = copy.deepcopy(samples)
        altered[0]["policy_obs"][key] = True
        mutations["policy_" + key] = altered
    altered = copy.deepcopy(samples)
    altered[0]["causal_history"]["previous_committed"] = 1
    mutations["future_history"] = altered
    altered = copy.deepcopy(samples)
    altered[1]["policy_obs"]["timestamp_s"] = 0.
    mutations["unordered_time"] = altered
    results = {}
    for name, altered in mutations.items():
        try:
            validate_samples(altered)
        except ValueError as error:
            results[name] = {"rejected": True, "reason": str(error)}
        else:
            raise ValueError("Negative oracle/causality gate not rejected: " + name)
    return results


def qualify(root, source, cosmos, output):
    root = Path(root)
    manifest = read(root / "DATASET_MANIFEST.json")
    train = [row for row in manifest["episodes"] if row["split"] == "TRAIN"]
    if len(train) < 2:
        raise ValueError("Actual mixed batch requires at least two different admitted TRAIN episodes")
    processor = build_processor(source, cosmos, read(root / "statistics.json"))
    offsets = [0, 80, 160, 320, 480, 600]
    indices = offsets + [601 + value for value in offsets]
    dataset = DatasetLoader(root, source, cosmos, "TRAIN", indices)
    dataset.processor = processor
    single = processor.collator([dataset[0]["processed"]])
    single_receipt = check_batch(single, 1)
    mixed_receipt = check_batch(processor.collator([dataset[1]["processed"], dataset[7]["processed"]]), 2)
    # Text-only diagnostics reuse real observations, never rewrite raw language.
    value = dataset[0]["processed"]
    short, long = copy.deepcopy(value), copy.deepcopy(value)
    short["vlm_content"]["text"] += " Short."
    long["vlm_content"]["text"] += " This longer instruction is a CPU padding diagnostic only and is not new task data."
    variable_text = check_batch(processor.collator([short, long]), 2)
    import torch
    from torch.utils.data import DataLoader
    worker_receipts = {}
    for workers in (0, 2):
        loader = DataLoader(dataset, batch_size=2, num_workers=workers, shuffle=False,
                            collate_fn=collate_project, multiprocessing_context="spawn" if workers else None,
                            persistent_workers=False)
        rows = []
        for batch in loader:
            collated = processor.collator([item["processed"] for item in batch])
            proof = check_batch(collated, len(batch))
            rows.append({"identities": [item["identity"] for item in batch], "batch": proof})
        worker_receipts[str(workers)] = rows
    if worker_receipts["0"] != worker_receipts["2"]:
        raise ValueError("Two data-worker batches differ from deterministic single-worker data")
    codec = codec_all_windows(root, processor)
    negatives = negative_gates(root, manifest)
    forbidden = [name for name in sys.modules if name.startswith("gr00t") and ("modeling_" in name or ".backbone." in name or ".action_head." in name)]
    if forbidden or torch.cuda.is_initialized():
        raise PermissionError("Unexpected neural module import or CUDA initialization")
    result = {"status": "NATIVE_DATALOADER_PASS", "RAW_DATA_READY": True, "CODEC_READY": True, "NATIVE_DATALOADER_PASS": True,
              "official_source_sha": OFFICIAL_SHA, "dataset_manifest_sha256": sha(root / "DATASET_MANIFEST.json"),
              "statistics_sha256": sha(root / "statistics.json"), "statistics_provenance_sha256": sha(root / "STATISTICS_PROVENANCE.json"),
              "single_actual_sample": single_receipt, "mixed_two_episode_batch": mixed_receipt,
              "actual_multiple_stage_anchor_indices": offsets, "variable_instruction_padding_diagnostic": variable_text,
              "raw_episode_language_modified": False, "worker_batch_checks": worker_receipts, "two_workers_repeat_equal": True,
              "codec": codec, "oracle_and_causality_negatives": negatives, "official_processor": type(processor).__name__,
              "official_collator": type(processor.collator).__name__, "valid_physical_dims": 10, "native_padded_dims": 132,
              "official_state_mask": "Not emitted by fixed upstream; raw valid flags enforced before processing, padded state dims verified zero",
              "policy_joint_or_privileged_inputs": False, "model_loads": 0, "neural_forward": 0, "training_steps": 0,
              "GPU_used": False, "physics_integrations": 0, "joint_state_is_execution_evidence_only": True}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--cosmos-processor-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        result = qualify(args.dataset_root, args.official_source, args.cosmos_processor_cache, args.output)
    except Exception as error:
        result = {"status": "DATA_READY_PROCESSOR_BLOCKED", "error_type": type(error).__name__, "error": str(error),
                  "RAW_DATA_READY": (Path(args.dataset_root) / "DATASET_MANIFEST.json").is_file(), "NATIVE_DATALOADER_PASS": False,
                  "model_loads": 0, "neural_forward": 0, "physics_integrations": 0}
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(elapsed_s=time.monotonic() - start, CPU_self_s=own.ru_utime + own.ru_stime, CPU_children_s=children.ru_utime + children.ru_stime,
                  CPU_total_s=own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "CPU_total_s", "elapsed_s")}))
    if result["status"] != "NATIVE_DATALOADER_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
