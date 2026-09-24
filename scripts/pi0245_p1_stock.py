#!/usr/bin/env python3
"""PI0245 fixed bridge4 stock sampler; one query per first-valid cell."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

SOURCES = (
    "pi0227_persistent_train_a_j03_yp015_s03_yp050_r00_r00_227072:obs-0033",
    "pi0227_persistent_train_b_j00_nominal_s00_c0_r02_r02_227182:obs-0033",
    "pi0227_persistent_train_c_j01_xp015_s04_ym050_r03_r03_227399:obs-0013",
    "pi0227_persistent_train_d_j03_yp015_s00_c0_r02_r02_227602:obs-0046",
)
RECORD_SHA = "4c502ab316ba9b4b800bfc743c9c5ecb1668a14d8b998f4c7a0f101cf5ed0d76"
NOISES = ("N0", "N1", "N2", "N0_roll11")
PATH_OVERLAY_KEYS = frozenset({
    "code_root", "groot_source", "stage10_root", "cosmos_snapshot",
    "processor_base_model", "train_dataset", "modality_config", "output_root",
    "r0_root", "noise_root", "model_base_root", "p1_runner",
})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_npz(path: Path, **arrays: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def durable_torch(path: Path, value) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_tensor_path(root: Path, recorded_path: str) -> Path:
    recorded = Path(recorded_path)
    assert recorded.name and recorded.parent.name == "cells" and ".." not in recorded.parts, f"unexpected sealed tensor path: {recorded_path}"
    cell_root = (root / "screen/step-10000/cells").resolve()
    resolved = cell_root / recorded.name
    assert resolved.resolve().parent == cell_root, f"tensor escaped screen cell root: {recorded_path}"
    return resolved


def resolve_path_overlay(card: dict, overlay_path: Path | None) -> dict[str, str]:
    paths = dict(card["paths"])
    if overlay_path is None:
        return paths
    overlay = json.loads(overlay_path.read_text())
    assert overlay["schema"] == "PI0245_PATH_OVERLAY_V1"
    assert overlay["source_run_card_sha256"] == card["_sha256"]
    overrides = overlay["paths"]
    assert set(overrides) <= PATH_OVERLAY_KEYS, f"non-path override: {set(overrides) - PATH_OVERLAY_KEYS}"
    assert all(isinstance(value, str) and value for value in overrides.values())
    paths.update(overrides)
    return paths


def sources(root: Path) -> dict[str, dict]:
    records_path = root / "screen/step-10000/records.jsonl"
    assert sha(records_path) == RECORD_SHA
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 16 and len({row["source_episode_id"] for row in rows}) == 16
    selected = {row["source_episode_id"]: row for row in rows if row["source_episode_id"] in SOURCES}
    assert set(selected) == set(SOURCES)
    for row in selected.values():
        row["recorded_tensor_path"] = row["tensor_path"]
        path = resolve_tensor_path(root, row["tensor_path"])
        row["tensor_path"] = str(path)
        assert sha(path) == row["tensor_sha256"]
        with np.load(path, allow_pickle=False) as tensor:
            assert tensor["raw"].shape == (1, 40, 132)
            assert tensor["decoded"].shape == (1, 40, 10)
            assert tensor["expert"].shape == (40, 10)
            assert tensor["action_mask"].shape == (40, 132)
            assert tensor["state"].shape == (10,)
            assert np.isfinite(tensor["expert"]).all()
    return selected


def noise_bank(root: Path) -> dict[str, np.ndarray]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "PI0245_P1_FIXED_NOISE_BANK_V1"
    assert tuple(row["name"] for row in manifest["rows"]) == NOISES
    result = {}
    for row in manifest["rows"]:
        path = root / f"{row['name']}.npz"
        assert sha(path) == row["file_sha256"]
        with np.load(path, allow_pickle=False) as payload:
            array = payload["float32"].copy()
            assert array.shape == (1, 40, 132) and np.isfinite(array).all()
            assert hashlib.sha256(array.tobytes()).hexdigest() == row["float32_bytes_sha256"]
            assert hashlib.sha256(payload["bfloat16_bits"].tobytes()).hexdigest() == row["bfloat16_bytes_sha256"]
        result[row["name"]] = array
    assert np.array_equal(result["N0_roll11"], np.roll(result["N0"], 11, axis=1))
    return result


def verify_model_files(model_root: Path) -> dict:
    restore_path = model_root / "pi0245_inference_restore.json"
    restore = json.loads(restore_path.read_text())
    assert restore["optimizer_restored"] is False
    checked = []
    for entry in restore["files"]:
        path = model_root / entry["path"]
        assert path.is_file() and sha(path) == entry["sha256"], f"model source drift: {path}"
        checked.append(entry["path"])
    assert "config.json" in checked and "model.safetensors.index.json" in checked
    assert sum(name.endswith(".safetensors") for name in checked) >= 2
    return {"manifest_sha256": sha(restore_path), "files_sha256_verified": len(checked), "revision": restore["revision"]}


def encode_expert(processor, embodiment, expert: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, float]:
    groups = processor.state_action_processor.apply_action(
        {"eef_9d": expert[:, :9].copy(), "gripper_position": expert[:, 9:10].copy()},
        embodiment.value,
        state={"eef_9d": state[None, :9].copy(), "gripper_position": state[None, 9:10].copy()},
    )
    active = np.concatenate([groups["eef_9d"], groups["gripper_position"]], axis=-1).astype(np.float32)
    universal = np.zeros((1, 40, 132), dtype=np.float32)
    universal[0, :, :active.shape[-1]] = active
    decoded = processor.decode_action(
        universal, embodiment,
        {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]},
    )
    roundtrip = np.concatenate([decoded["eef_9d"], decoded["gripper_position"]], axis=-1)[0]
    return active, float(np.max(np.abs(roundtrip - expert)))


def one(args: argparse.Namespace, row: dict, noise_name: str, fixed_noise: np.ndarray,
        *, paths: dict[str, str], model, model_files, processor, embodiment, bridge, selector, stage43) -> None:
    import torch
    from gr00t.data.types import MessageType, VLAStepData
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    import stage21_same_scene_groot_inference as stage21

    source = row["source_episode_id"]
    model_name = args.model_name
    cell = args.output_root / "cells" / f"{model_name}__{source.split(':', 1)[0]}__{noise_name}"
    if cell.exists():
        record_path = cell / "record.json"
        assert record_path.is_file(), f"incomplete prior attempt: {cell}"
        record = json.loads(record_path.read_text())
        assert record["source_episode_id"] == source and record["noise_name"] == noise_name and record["model_name"] == model_name
        assert sha(cell / "tensors.npz") == record["tensor_sha256"]
        assert sha(cell / "condition.pt") == record["condition_sha256"]
        assert sha(cell / "action_inputs.pt") == record["action_inputs_sha256"]
        assert sha(cell / "backbone_features.pt") == record["backbone_features_sha256"]
        print(json.dumps({"status": "REUSED_FIRST_VALID", "cell": str(cell)}), flush=True)
        return

    with np.load(row["tensor_path"], allow_pickle=False) as prior:
        body = prior["body_rgb"].copy()
        wrist = prior["wrist_rgb"].copy()
        state = prior["state"].copy()
        expert = prior["expert"].copy()
        mask = prior["action_mask"].copy()
    expert_normalized, expert_roundtrip_max_abs = encode_expert(processor, embodiment, expert, state)
    assert expert_roundtrip_max_abs <= 1e-5, f"physical expert cannot roundtrip through bound processor: {expert_roundtrip_max_abs}"
    assert body.dtype == wrist.dtype == np.uint8
    assert mask.shape == (40, 132) and np.all(mask[:, :10] == 1)
    content = VLAStepData(
        images={"external_rgb": body[None], "wrist_rgb": wrist[None]},
        states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]},
        actions={}, text=stage43.LANGUAGE, embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    policy = bridge._policy_inputs(processor.collator([processed]))
    policy_inventory = bridge._leaf_inventory(policy)
    forbidden = [key for key in policy_inventory if any(token in key.lower() for token in bridge.FORBIDDEN_INPUT_TOKENS)]
    assert not forbidden, f"forbidden model input: {forbidden}"
    condition_identity = bridge._canonical_hash(policy_inventory)
    for sibling in (args.output_root / "cells").glob(f"*{source.split(':', 1)[0]}__*"):
        prior_path = sibling / "record.json"
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text())
            assert prior["condition_identity"] == condition_identity, f"paired condition changed for {source}"
    cell.mkdir(parents=True, exist_ok=False)
    durable_torch(cell / "condition.pt", policy)
    attempted = {"source_episode_id": source, "model_name": model_name, "noise_name": noise_name, **model_files,
                 "recorded_tensor_path": row["recorded_tensor_path"], "resolved_tensor_path": row["tensor_path"],
                 "condition_sha256": sha(cell / "condition.pt"), "condition_identity": condition_identity,
                 "processor_config_sha256": sha(Path(paths["processor_base_model"]) / "processor_config.json"),
                 "relative_stats_sha256": sha(Path(paths["train_dataset"]) / "meta/relative_stats.json"),
                 "source_tensor_sha256": row["tensor_sha256"], "runner_sha256": sha(Path(__file__).resolve()),
                 "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                 "model_parameter_dtype": str(next(model.parameters()).dtype),
                 "model_device": str(next(model.parameters()).device)}
    durable_json(cell / "attempt.json", attempted)
    ledger = args.output_root / "queries.jsonl"
    previous = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.is_file() else []
    assert len(previous) < 32, "PI0245 P1 32-query budget exhausted"
    key = f"{model_name}__{source}__{noise_name}"
    assert key not in {record["cell_key"] for record in previous}, f"P1 cell already attempted: {key}"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ordinal": len(previous) + 1, "cell_key": key, **attempted}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "ATTEMPTED_STOCK_QUERY", **attempted}), flush=True)
    stock = bridge._stock_query(model, policy, stage21, fixed_noise)
    raw = np.asarray(stock["raw"], dtype=np.float32)
    injected = torch.from_numpy(fixed_noise).to(torch.bfloat16).float().numpy()
    assert np.array_equal(stock["noise"], fixed_noise), "requested fixed noise differs"
    assert np.array_equal(stock["trace"][0]["x"], injected), "actual initial BF16 noise differs"
    trace_x = np.stack([item["x"] for item in stock["trace"]])
    trace_v = np.stack([item["v"] for item in stock["trace"]])
    trace_t = np.stack([item["t"] for item in stock["trace"]])
    if raw.shape != (1, 40, 132) or not np.isfinite(raw).all():
        durable_npz(cell / "invalid_output.npz", raw=raw, expert=expert, action_mask=mask, state=state,
                    noise_source_float32=fixed_noise, noise_initial=injected, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
        durable_json(cell / "record.json", {**attempted, "schema": "PI0245_P1_INVALID_STOCK_CELL_V1",
                                             "status": "INVALID_RAW_SHAPE_OR_NONFINITE", "raw_shape": list(raw.shape),
                                             "raw_all_finite": bool(np.isfinite(raw).all()),
                                             "tensor_sha256": sha(cell / "invalid_output.npz"),
                                             "target_xyz_in_model_input": False})
        raise RuntimeError("P1 native stock output is illegal; preserve cell and stop")
    decoded = bridge._decode_physical(processor, embodiment, raw, state)
    if decoded.shape != (1, 40, 10) or not np.isfinite(decoded).all():
        durable_npz(cell / "invalid_output.npz", raw=raw, decoded=decoded, expert=expert, action_mask=mask, state=state,
                    noise_source_float32=fixed_noise, noise_initial=injected, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
        durable_json(cell / "record.json", {**attempted, "schema": "PI0245_P1_INVALID_STOCK_CELL_V1",
                                             "status": "INVALID_DECODED_SHAPE_OR_NONFINITE", "decoded_shape": list(decoded.shape),
                                             "decoded_all_finite": bool(np.isfinite(decoded).all()),
                                             "tensor_sha256": sha(cell / "invalid_output.npz"),
                                             "target_xyz_in_model_input": False})
        raise RuntimeError("P1 decoded stock output is illegal; preserve cell and stop")
    state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    gate = TrajectorySafetyGateV1().evaluate(
        selector._build_chunk(decoded, chunk_id=f"pi0245-{model_name}-{source}-{noise_name}"),
        now=0.0, achieved_position=state[:3], achieved_orientation=state_quat,
    )
    metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
    e = decoded[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
    e2 = np.sum(e * e, axis=1)
    delta_e = np.linalg.norm(np.diff(e, axis=0), axis=1) / .05
    raw_jump = np.linalg.norm(np.diff(raw[0, :, :3], axis=0), axis=1)
    physical_jump = np.linalg.norm(np.diff(decoded[0, :, :3], axis=0), axis=1)
    durable_torch(cell / "action_inputs.pt", stock["action_inputs"])
    durable_torch(cell / "backbone_features.pt", {"raw": stock["backbone_raw"], "processed_once": stock["backbone_once"]})
    durable_npz(cell / "tensors.npz", raw=raw, decoded=decoded, expert=expert, action_mask=mask, state=state,
                expert_normalized=expert_normalized,
                noise_source_float32=fixed_noise, noise_initial=injected, trace_x=trace_x, trace_v=trace_v, trace_t=trace_t)
    normalized_error = raw[0, :, :10].astype(np.float64) - expert_normalized.astype(np.float64)
    record = {**attempted, "schema": "PI0245_P1_STOCK_CELL_V1", "tensor_sha256": sha(cell / "tensors.npz"),
              "action_inputs_sha256": sha(cell / "action_inputs.pt"),
              "backbone_features_sha256": sha(cell / "backbone_features.pt"),
              "cell_key": key,
              "source_episode_id": source, "target_family": row["target_family"], "phase": row["phase"],
              "gate_decision": gate.decision.value, "gate": gate.as_dict(), "metrics": metrics,
              "noise_sha256": hashlib.sha256(injected.tobytes()).hexdigest(), "sampler_steps": len(stock["trace"]),
              "raw_xyz_peak_segment": int(np.argmax(raw_jump)), "physical_xyz_peak_segment": int(np.argmax(physical_jump)),
              "physical_max_jump_m": float(physical_jump.max()), "physical_residual_delta_max_mps": float(delta_e.max()),
              "feature_provenance": stock["feature_provenance"],
              "model_get_action_calls": 1, "model_backbone_forwards": 1,
              "action_head_denoising_forwards": len(stock["trace"]),
              "tail35_39_pooled_sse_fraction": float(e2[35:].sum() / e2.sum()) if e2.sum() > 0 else 0.0,
              "normalized_active10_rmse": float(np.sqrt(np.mean(normalized_error**2))),
              "normalized_xyz_rmse": float(np.sqrt(np.mean(normalized_error[:, :3]**2))),
              "expert_processor_roundtrip_max_abs": expert_roundtrip_max_abs,
              "raw_xyz_clipped_scalars": int(np.sum(np.abs(raw[0, :, :3]) > 1)), "target_xyz_in_model_input": False}
    durable_json(cell / "record.json", record)
    print(json.dumps({"status": "PASS_FIRST_VALID_CELL", "cell": str(cell), "gate": gate.decision.value, "xyz_rmse_m": metrics["xyz_rmse_m"]}), flush=True)


def close_p1(args: argparse.Namespace) -> None:
    from collections import Counter

    expected = [f"{model}__{source}__{noise}" for model in ("R0", "R2") for source in SOURCES for noise in NOISES]
    ledger = [json.loads(line) for line in (args.output_root / "queries.jsonl").read_text().splitlines() if line.strip()]
    assert [row["cell_key"] for row in ledger] == expected, "P1 attempted-query identity/order mismatch"
    records = []
    observed = Counter()
    for key in expected:
        source_key = key.split("__", 2)[1]
        cell_dir = args.output_root / "cells" / key.replace(source_key, source_key.split(":", 1)[0])
        record_path, tensor_path = cell_dir / "record.json", cell_dir / "tensors.npz"
        record = json.loads(record_path.read_text())
        assert record["cell_key"] == key and record["schema"] == "PI0245_P1_STOCK_CELL_V1"
        assert sha(tensor_path) == record["tensor_sha256"]
        assert sha(cell_dir / "condition.pt") == record["condition_sha256"]
        assert record["target_xyz_in_model_input"] is False
        observed.update({name: int(record[name]) for name in ("model_get_action_calls", "model_backbone_forwards", "action_head_denoising_forwards")})
        with np.load(tensor_path, allow_pickle=False) as data:
            assert data["raw"].shape == (1, 40, 132) and np.isfinite(data["raw"]).all()
            assert data["decoded"].shape == (1, 40, 10) and np.isfinite(data["decoded"]).all()
            assert data["trace_x"].shape[0] == record["sampler_steps"]
            assert data["trace_v"].shape[0] == record["sampler_steps"]
            assert data["trace_t"].shape[0] == record["sampler_steps"]
        records.append({"cell_key": key, "record_sha256": sha(record_path), "tensor_sha256": record["tensor_sha256"],
                        "gate_decision": record["gate_decision"], "metrics": record["metrics"],
                        "physical_max_jump_m": record["physical_max_jump_m"],
                        "physical_residual_delta_max_mps": record["physical_residual_delta_max_mps"],
                        "raw_xyz_peak_segment": record["raw_xyz_peak_segment"],
                        "physical_xyz_peak_segment": record["physical_xyz_peak_segment"]})
    summaries = {}
    for model in ("R0", "R2"):
        group = [json.loads((args.output_root / "cells" / key.replace(key.split("__", 2)[1], key.split("__", 2)[1].split(":", 1)[0]) / "record.json").read_text())
                 for key in expected if key.startswith(model + "__")]
        summaries[model] = {"count": len(group), "gate_counts": dict(Counter(row["gate_decision"] for row in group)),
                            "median_xyz_rmse_m": float(np.median([row["metrics"]["xyz_rmse_m"] for row in group])),
                            "median_so3_rmse_rad": float(np.median([row["metrics"]["so3_rmse_rad"] for row in group])),
                            "median_max_linear_speed_mps": float(np.median([row["metrics"]["max_linear_speed_mps"] for row in group])),
                            "median_physical_residual_delta_max_mps": float(np.median([row["physical_residual_delta_max_mps"] for row in group]))}
        summaries[model]["by_noise"] = {
            noise: {"gate_counts": dict(Counter(row["gate_decision"] for row in group if row["noise_name"] == noise)),
                    "median_xyz_rmse_m": float(np.median([row["metrics"]["xyz_rmse_m"] for row in group if row["noise_name"] == noise])),
                    "median_physical_max_jump_m": float(np.median([row["physical_max_jump_m"] for row in group if row["noise_name"] == noise])),
                    "median_physical_residual_delta_max_mps": float(np.median([row["physical_residual_delta_max_mps"] for row in group if row["noise_name"] == noise]))}
            for noise in NOISES}
    closure = {"schema": "PI0245_P1_CLOSURE_V1", "status": "PASS_COMPLETE_32_QUERY_MATRIX",
               "attempted_queries": len(ledger), "completed_cells": len(records), "query_order": expected,
               "observed_calls": dict(observed),
               "summaries": summaries, "cells": records, "runtime_binding_sha256": sha(args.output_root / "runtime_binding.json"),
               "noise_bank_manifest_sha256": sha(args.noise_root / "manifest.json"),
               "R0_records_sha256": sha(args.r0_root / "screen/step-10000/records.jsonl"),
               "selected_checkpoint": "NONE"}
    path = args.output_root / "p1_closure.json"
    if path.is_file():
        assert json.loads(path.read_text()) == closure, "P1 first-valid closure changed"
    else:
        durable_json(path, closure)
    print(json.dumps({"status": closure["status"], "queries": len(ledger), "summaries": summaries}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-root", type=Path, required=True)
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--path-overlay", type=Path)
    parser.add_argument("--p0-binding", type=Path)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-name", choices=("R0", "R2"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-index", type=int, choices=range(4), required=True)
    parser.add_argument("--noise-name", choices=NOISES, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--bind-preflight", action="store_true")
    parser.add_argument("--model-preflight", action="store_true")
    parser.add_argument("--label-preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--close-p1", action="store_true")
    args = parser.parse_args()
    card = json.loads(args.run_card.read_text())
    card["_sha256"] = sha(args.run_card)
    paths = resolve_path_overlay(card, args.path_overlay)
    for key, attribute in (("r0_root", "r0_root"), ("noise_root", "noise_root"),
                           ("output_root", "output_root")):
        if key in paths:
            setattr(args, attribute, Path(paths[key]))
    if "model_base_root" in paths:
        args.model_root = Path(paths["model_base_root"]) / args.model_name
    if args.close_p1:
        close_p1(args)
        return
    rows = sources(args.r0_root)
    noises = noise_bank(args.noise_root)
    if args.preflight:
        print(json.dumps({"status": "PASS_P1_CPU_INPUT_PREFLIGHT", "source_ids": list(SOURCES), "noise_names": list(noises)}))
        return
    if args.model_preflight:
        print(json.dumps({"status": "PASS_P1_MODEL_FILE_PREFLIGHT", **verify_model_files(args.model_root)}))
        return
    code_root = Path(paths["code_root"])
    sys.path[:0] = [str(code_root / "scripts"), str(code_root / "src"), str(paths["groot_source"])]
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import stage40_b0_model_control as stage40
    cosmos_binding = stage40._bind_local_cosmos_processor(Path(paths["stage10_root"]), Path(paths["cosmos_snapshot"]))
    import stage21_same_scene_groot_inference as stage21
    processor, embodiment = stage21._build_processor(
        Path(paths["stage10_root"]), Path(paths["processor_base_model"]),
        Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"]),
    )
    if args.bind_preflight:
        print(json.dumps({"status": "PASS_P1_EXACT_RUNTIME_PROCESSOR_BINDING", "embodiment": str(embodiment)}))
        return
    if args.label_preflight:
        rows = sources(args.r0_root)
        checks = {}
        for source_id, row in rows.items():
            with np.load(row["tensor_path"], allow_pickle=False) as payload:
                _, error = encode_expert(processor, embodiment, payload["expert"], payload["state"])
                checks[source_id] = error
        assert max(checks.values()) <= 1e-5, checks
        print(json.dumps({"status": "PASS_P1_EXPERT_LABEL_PROCESSOR_ROUNDTRIP", "count": len(checks), "max_abs_by_source": checks}))
        return
    import torch
    from transformers import AutoModel
    import pi0243_staged_learning_to_sim as stage43
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import stage21_same_scene_groot_inference as stage21
    model_files = verify_model_files(args.model_root)
    runtime_files = [Path(__file__).resolve(), Path(stage21.__file__).resolve(), Path(bridge.__file__).resolve(),
                     Path(selector.__file__).resolve(), Path(stage43.__file__).resolve(),
                     Path(stage40.__file__).resolve(), Path(inspect.getfile(type(processor))).resolve()]
    modules = []
    for path in runtime_files:
        modules.append({"path": str(path), "sha256": sha(path)})
    p0_binding_path = args.p0_binding or (args.output_root.parent / "p0/effective_processor_xyz_v2.json")
    runtime_binding = {"schema": "PI0245_P1_RUNTIME_BINDING_V1", "run_card_sha256": sha(args.run_card),
                       "path_overlay_sha256": sha(args.path_overlay) if args.path_overlay else None,
                       "effective_paths": paths,
                       "r0_records_sha256": sha(args.r0_root / "screen/step-10000/records.jsonl"),
                       "noise_manifest_sha256": sha(args.noise_root / "manifest.json"),
                       "p0_effective_processor_xyz_sha256": sha(p0_binding_path),
                       "model_restore_manifests": {name: sha(args.model_root.parent / name / "pi0245_inference_restore.json") for name in ("R0", "R2")},
                       "cosmos_binding": cosmos_binding,
                       "modality_config_sha256": sha(Path(paths["modality_config"])),
                       "relative_stats_sha256": sha(Path(paths["train_dataset"]) / "meta/relative_stats.json"),
                       "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                       "python": sys.version, "torch": torch.__version__, "modules": modules}
    binding_path = args.output_root / "runtime_binding.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    if binding_path.is_file():
        assert json.loads(binding_path.read_text()) == runtime_binding, "P1 runtime binding drift"
    else:
        durable_json(binding_path, runtime_binding)
    model = AutoModel.from_pretrained(args.model_root, local_files_only=True, trust_remote_code=True)
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    cells = [(source, name) for source in SOURCES for name in NOISES] if args.all else [(SOURCES[args.source_index], args.noise_name)]
    for source, name in cells:
        one(args, rows[source], name, noises[name], model=model, model_files=model_files, processor=processor,
            paths=paths, embodiment=embodiment, bridge=bridge, selector=selector, stage43=stage43)
    if args.all:
        ledger_path = args.output_root / "queries.jsonl"
        if ledger_path.is_file() and len([line for line in ledger_path.read_text().splitlines() if line.strip()]) == 32:
            close_p1(args)


if __name__ == "__main__":
    main()
