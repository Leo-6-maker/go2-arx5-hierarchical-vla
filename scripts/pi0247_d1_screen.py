#!/usr/bin/env python3
"""PI0247 D1: frozen-weight stock-step and structural-padding comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(stage_root: Path, code_root: Path) -> tuple[dict, dict]:
    import torch

    # ponytail: all scientific inputs come from the sealed P1/S0 artifacts; no new condition builder.
    sys.path[:0] = [str(stage_root), str(code_root / "scripts"), str(code_root / "src")]
    import pi0245_p1_stock as p1
    import pi0246_paired_screen as screen

    rows, paths, cards = screen.inputs(stage_root)
    screen.bind_offline_runtime(paths)
    sys.path.insert(0, paths["groot_source"])
    import stage21_same_scene_groot_inference as stage21
    import stage40_b0_model_control as stage40

    stage40._bind_local_cosmos_processor(Path(paths["stage10_root"]), Path(paths["cosmos_snapshot"]))
    processor, embodiment = stage21._build_processor(
        Path(paths["stage10_root"]), Path(paths["processor_base_model"]),
        Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"]),
    )

    padding_rows = []
    for row in rows:
        with np.load(row["resolved_tensor_path"], allow_pickle=False) as data:
            expert, state, mask = data["expert"].copy(), data["state"].copy(), data["action_mask"].copy()
        assert mask.shape == (40, 132) and np.all(mask[:, :10] == 1)
        inactive_mask_nonzero = int(np.count_nonzero(mask[:, 10:]))
        encoded_active, roundtrip = p1.encode_expert(processor, embodiment, expert, state)
        assert encoded_active.shape == (40, 10) and np.isfinite(encoded_active).all()
        encoded = np.zeros((1, 40, 132), dtype=np.float32)
        encoded[0, :, :10] = encoded_active
        inactive_encoded_nonzero = int(np.count_nonzero(encoded[..., 10:]))
        padding_rows.append({
            "source_episode_id": row["source_episode_id"],
            "source_tensor_sha256": row["tensor_sha256"],
            "active_mask_ones": int(np.count_nonzero(mask[:, :10])),
            "inactive_mask_nonzero": inactive_mask_nonzero,
            "inactive_encoded_nonzero": inactive_encoded_nonzero,
            "expert_roundtrip_max_abs": roundtrip,
        })

    noise_root = Path(paths["noise_root"])
    bank = p1.noise_bank(noise_root)
    noise_manifest_sha = p1.sha(noise_root / "manifest.json")
    p1_root = stage_root / "p1_corelab"
    closure_sha = p1.sha(p1_root / "p1_closure.json")
    runtime_binding_sha = p1.sha(p1_root / "runtime_binding.json")
    torch_noise = {
        name: torch.from_numpy(bank[name]).to(dtype=torch.bfloat16).float().numpy()
        for name in ("N0", "N1")
    }
    reused = []
    source_cells = {}
    for source in p1.SOURCES:
        source_cells[source] = {}
        base = source.split(":", 1)[0]
        condition_identity = None
        for noise_name in ("N0", "N1"):
            cell = p1_root / "cells" / f"R0__{base}__{noise_name}"
            record_path, tensor_path = cell / "record.json", cell / "tensors.npz"
            record = read_json(record_path)
            assert record["schema"] == "PI0245_P1_STOCK_CELL_V1"
            assert record["cell_key"] == f"R0__{source}__{noise_name}"
            assert record["source_episode_id"] == source and record["model_name"] == "R0"
            assert record["noise_name"] == noise_name and record["sampler_steps"] == 4
            assert record["target_xyz_in_model_input"] is False
            attempt = read_json(cell / "attempt.json")
            assert attempt["source_episode_id"] == source and attempt["model_name"] == "R0"
            assert attempt["noise_name"] == noise_name
            assert attempt["condition_sha256"] == record["condition_sha256"]
            assert attempt["source_tensor_sha256"] == record["source_tensor_sha256"]
            assert p1.sha(tensor_path) == record["tensor_sha256"]
            assert p1.sha(cell / "condition.pt") == record["condition_sha256"]
            assert p1.sha(cell / "action_inputs.pt") == record["action_inputs_sha256"]
            assert p1.sha(cell / "backbone_features.pt") == record["backbone_features_sha256"]
            if condition_identity is None:
                condition_identity = record["condition_identity"]
            assert condition_identity == record["condition_identity"]
            with np.load(tensor_path, allow_pickle=False) as data:
                assert np.array_equal(data["noise_source_float32"], bank[noise_name])
                assert np.array_equal(data["noise_initial"], torch_noise[noise_name])
                assert np.array_equal(data["trace_x"][0], torch_noise[noise_name])
                assert np.all(data["action_mask"][:, :10] == 1)
                assert np.count_nonzero(data["action_mask"][:, 10:]) == 0
                assert np.count_nonzero(data["expert_normalized"][..., 10:]) == 0
            source_cells[source][noise_name] = {
                "cell_path": str(cell),
                "record_sha256": p1.sha(record_path),
                "attempt_sha256": p1.sha(cell / "attempt.json"),
                "tensor_sha256": record["tensor_sha256"],
                "condition_sha256": record["condition_sha256"],
                "condition_identity": record["condition_identity"],
                "noise_source_sha256": hashlib.sha256(bank[noise_name].tobytes()).hexdigest(),
                "noise_initial_sha256": hashlib.sha256(torch_noise[noise_name].tobytes()).hexdigest(),
            }
            reused.append({
                "model": "R0", "runtime": "S4", "source_episode_id": source, "noise": noise_name,
                "cell_path": str(cell), "record_sha256": p1.sha(record_path),
                "tensor_sha256": record["tensor_sha256"], "condition_sha256": record["condition_sha256"],
                "condition_identity": record["condition_identity"],
            })

    assert len(rows) == 16 and len(padding_rows) == 16
    m_applicable = all(
        row["inactive_mask_nonzero"] == 0 and row["inactive_encoded_nonzero"] == 0
        for row in padding_rows
    )
    facts = {
        "schema": "PI0247_D1_STATIC_PREFLIGHT_V1",
        "stage_root": str(stage_root),
        "code_root": str(code_root),
        "cohort_sha256": screen.COHORT_SHA,
        "source_records_sha256": screen.RECORD_SHA,
        "p1_closure_sha256": closure_sha,
        "runtime_binding_sha256": runtime_binding_sha,
        "source_code_files_sha256": {
            "p1_runner": sha(Path(p1.__file__).resolve()),
            "paired_screen": sha(Path(screen.__file__).resolve()),
            "bridge": sha(Path(code_root) / "scripts/pi0242_train_infer_bridge.py"),
            "screen_metrics": sha(Path(code_root) / "scripts/pi0243_staged_learning_to_sim.py"),
            "selector": sha(Path(code_root) / "scripts/pi0228_train_calib_checkpoint_selection.py"),
            "stage21": sha(Path(stage21.__file__).resolve()),
            "stage40": sha(Path(stage40.__file__).resolve()),
        },
        "noise_manifest_sha256": noise_manifest_sha,
        "noise_files_sha256": {name: p1.sha(noise_root / f"{name}.npz") for name in ("N0", "N1")},
        "noise_source_tensor_sha256": {
            name: hashlib.sha256(bank[name].tobytes()).hexdigest() for name in ("N0", "N1")
        },
        "noise_initial_bfloat16_sha256": {
            name: hashlib.sha256(torch_noise[name].tobytes()).hexdigest() for name in ("N0", "N1")
        },
        "bridge4_sources": list(p1.SOURCES),
        "reused_r0_s4_cells": reused,
        "padding_proof": {
            "source": "all 16 fixed TRAIN windows, existing action_mask plus bound P1 processor encode_expert",
            "inactive_dims": "10:132",
            "structural_zero_padding_proven": m_applicable,
            "rows": padding_rows,
        },
        "m_applicable": m_applicable,
        "paths": {key: paths[key] for key in (
            "code_root", "groot_source", "stage10_root", "cosmos_snapshot", "processor_base_model",
            "train_dataset", "modality_config", "r0_root", "noise_root", "model_base_root",
        )},
    }
    context = {
        "p1": p1, "screen": screen, "paths": paths, "cards": cards, "rows": rows,
        "noise": bank, "noise_root": noise_root, "p1_root": p1_root, "source_cells": source_cells,
        "processor": processor, "embodiment": embodiment,
    }
    return facts, context


def self_check() -> None:
    import torch

    for dtype in (torch.float32, torch.bfloat16):
        for steps in (4, 16):
            epsilon = (torch.arange(40 * 132, dtype=torch.float32).reshape(1, 40, 132) / 997).to(dtype)
            x = epsilon.clone()
            active_reference = x[..., :10].clone()
            for j in range(steps):
                inactive = (1.0 - j / steps) * epsilon[..., 10:]
                x = torch.cat((x[..., :10], inactive), dim=-1)
                assert torch.equal(x[..., :10], active_reference)
                assert torch.equal(x[..., 10:], inactive)
                v = torch.cat((torch.ones_like(x[..., :10]), -epsilon[..., 10:]), dim=-1)
                x = x + v / steps
                active_reference = active_reference + 1.0 / steps
            scored = torch.cat((x[..., :10], torch.zeros_like(x[..., 10:])), dim=-1)
            assert torch.equal(scored[..., :10], x[..., :10])
            assert torch.count_nonzero(scored[..., 10:]) == 0


def validate_card(card_path: Path, sidecar: Path) -> tuple[dict, str]:
    card = read_json(card_path)
    card_sha = sha(card_path)
    assert sidecar.read_text(encoding="ascii").strip() == card_sha
    assert card["schema"] == "PI0247_D1_RUN_CARD_V1"
    assert sha(Path(__file__).resolve()) == card["runner_sha256"]
    return card, card_sha


def run_one(
    *,
    args: argparse.Namespace,
    card: dict,
    card_sha: str,
    context: dict,
    model_name: str,
    runtime: dict,
    source: str,
    noise_name: str,
    model: Any,
    model_files: dict,
    bridge: Any,
    selector: Any,
    stage43: Any,
    gate: Any,
) -> dict:
    import torch

    p1 = context["p1"]
    p1_root = context["p1_root"]
    source_info = context["source_cells"][source][noise_name]
    parent_cell = Path(source_info["cell_path"])
    parent = read_json(parent_cell / "record.json")
    policy = torch.load(parent_cell / "condition.pt", map_location="cpu", weights_only=False)
    identity = bridge._canonical_hash(bridge._leaf_inventory(policy))
    assert identity == parent["condition_identity"] == source_info["condition_identity"]
    with np.load(parent_cell / "tensors.npz", allow_pickle=False) as data:
        state, expert, mask = data["state"].copy(), data["expert"].copy(), data["action_mask"].copy()
    fixed_noise = context["noise"][noise_name]
    n = int(runtime["steps"])
    variant = runtime["id"]
    cell_key = f"{model_name}__{variant}__{source.split(':', 1)[0]}__{noise_name}"
    cell = args.output_root / "cells" / cell_key
    if cell.exists():
        record_path = cell / "record.json"
        assert record_path.is_file(), f"incomplete preserved attempt; refusing duplicate query: {cell}"
        record = read_json(record_path)
        assert record["cell_key"] == cell_key and record["run_card_sha256"] == card_sha
        tensor_name = "tensors.npz" if record["status"] == "VALID" else "invalid_output.npz"
        assert p1.sha(cell / tensor_name) == record["tensor_sha256"]
        assert p1.sha(cell / "attempt.json") == record["attempt_sha256"]
        print(json.dumps({"status": "REUSED_D1_FIRST_VALID", "cell": str(cell)}), flush=True)
        return record

    cell.mkdir(parents=True, exist_ok=False)
    attempt = {
        "schema": "PI0247_D1_ATTEMPT_V1",
        "cell_key": cell_key,
        "run_card_sha256": card_sha,
        "runner_sha256": card["runner_sha256"],
        "model_name": model_name,
        "model_manifest_sha256": model_files["manifest_sha256"],
        "model_revision": model_files["revision"],
        "runtime": variant,
        "steps": n,
        "source_episode_id": source,
        "source_tensor_sha256": parent["source_tensor_sha256"],
        "condition_sha256": parent["condition_sha256"],
        "condition_identity": identity,
        "condition_source_cell": str(parent_cell),
        "noise_name": noise_name,
        "noise_file_sha256": context["facts"]["noise_files_sha256"][noise_name],
        "noise_source_sha256": hashlib.sha256(fixed_noise.tobytes()).hexdigest(),
        "target_xyz_in_model_input": False,
        "status": "MODEL_CALL_PENDING",
    }
    p1.durable_json(cell / "attempt.json", attempt)
    print(json.dumps({"status": "ATTEMPTED_D1_CELL", **attempt}), flush=True)

    head = model.action_head
    original_steps = int(head.num_inference_timesteps)
    assert original_steps == 4, f"unexpected bound stock sampler steps: {original_steps}"
    head.num_inference_timesteps = n
    m_trace = {"x_before": [], "x_effective": []}
    handles = []
    if runtime["inactive"] == "linear_noise_to_zero":
        epsilon = torch.from_numpy(np.asarray(fixed_noise, dtype=np.float32)).to(
            device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
        )
        def set_inactive_path(_module: Any, inputs: tuple[Any, ...]):
            step = len(m_trace["x_effective"])
            assert step < n and len(inputs) >= 1
            x = inputs[0]
            assert tuple(x.shape) == (1, 40, 132) and tuple(epsilon.shape) == tuple(x.shape)
            path = (1.0 - step / n) * epsilon[..., 10:]
            effective = torch.cat((x[..., :10], path), dim=-1)
            assert torch.equal(effective[..., :10], x[..., :10])
            m_trace["x_before"].append(x.detach().to(device="cpu", dtype=torch.float32).numpy().copy())
            m_trace["x_effective"].append(effective.detach().to(device="cpu", dtype=torch.float32).numpy().copy())
            return (effective, *inputs[1:])
        handles.append(head.action_encoder.register_forward_pre_hook(set_inactive_path))
    else:
        epsilon = None

    try:
        stock = bridge._stock_query(model, policy, __import__("stage21_same_scene_groot_inference"), fixed_noise)
    finally:
        for handle in handles:
            handle.remove()
        head.num_inference_timesteps = original_steps

    raw_native = np.asarray(stock["raw"], dtype=np.float32)
    noise_initial = stock["trace"][0]["x"]
    trace_x = np.stack([item["x"] for item in stock["trace"]])
    trace_v = np.stack([item["v"] for item in stock["trace"]])
    trace_t = np.stack([item["t"] for item in stock["trace"]])
    assert len(stock["trace"]) == n and raw_native.shape == (1, 40, 132)
    assert np.array_equal(stock["noise"], fixed_noise)
    torch_initial = torch.from_numpy(fixed_noise).to(
        device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
    ).to(device="cpu", dtype=torch.float32).numpy()
    assert np.array_equal(noise_initial, torch_initial)
    if epsilon is not None:
        trace_x_before = np.stack(m_trace["x_before"])
        trace_x_effective = np.stack(m_trace["x_effective"])
        assert len(trace_x_effective) == n and np.array_equal(trace_x, trace_x_effective)
        assert np.array_equal(trace_x[0], torch_initial)
        for step in range(n):
            assert np.array_equal(trace_x[step, ..., :10], trace_x_before[step, ..., :10])
            expected_path = ((1.0 - step / n) * epsilon[..., 10:]).to(
                device="cpu", dtype=torch.float32
            ).numpy()
            assert np.array_equal(trace_x[step, ..., 10:], expected_path)
        raw_scored = np.concatenate((raw_native[..., :10], np.zeros_like(raw_native[..., 10:])), axis=-1)
    else:
        trace_x_before = trace_x.copy()
        trace_x_effective = trace_x.copy()
        raw_scored = raw_native.copy()
    assert np.array_equal(raw_scored[..., :10], raw_native[..., :10])
    if not np.isfinite(raw_native).all():
        p1.durable_npz(
            cell / "invalid_output.npz", raw_native=raw_native, raw_scored=raw_scored,
            noise_source_float32=fixed_noise, noise_initial=noise_initial,
            trace_x_before=trace_x_before, trace_x=trace_x_effective, trace_v=trace_v, trace_t=trace_t,
        )
        record = {**attempt, "status": "INVALID_RAW_NONFINITE",
                  "attempt_sha256": p1.sha(cell / "attempt.json"), "gate_decision": "UNSCORABLE",
                  "metrics": None, "tensor_sha256": p1.sha(cell / "invalid_output.npz")}
        p1.durable_json(cell / "record.json", record)
        return record

    try:
        decoded = bridge._decode_physical(context["processor"], context["embodiment"], raw_scored, state)
        assert decoded.shape == (1, 40, 10) and np.isfinite(decoded).all()
    except Exception as exc:
        p1.durable_npz(
            cell / "invalid_output.npz", raw_native=raw_native, raw_scored=raw_scored, expert=expert,
            state=state, action_mask=mask, noise_source_float32=fixed_noise, noise_initial=noise_initial,
            trace_x_before=trace_x_before, trace_x=trace_x_effective, trace_v=trace_v, trace_t=trace_t,
        )
        record = {**attempt, "status": "INVALID_DECODE", "error": repr(exc),
                  "attempt_sha256": p1.sha(cell / "attempt.json"), "gate_decision": "UNSCORABLE",
                  "metrics": None, "tensor_sha256": p1.sha(cell / "invalid_output.npz")}
        p1.durable_json(cell / "record.json", record)
        return record

    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix
    state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    gate_result = gate.evaluate(
        selector._build_chunk(decoded, chunk_id=f"pi0247-{variant}-{model_name}-{source}-{noise_name}"),
        now=0.0, achieved_position=state[:3], achieved_orientation=state_quat,
    )
    metrics = stage43._screen_metrics(decoded[0].astype(np.float64), expert.astype(np.float64), None)
    residual = decoded[0, :, :3].astype(np.float64) - expert[:, :3].astype(np.float64)
    squared = np.sum(residual * residual, axis=1)
    metrics["physical_residual_delta_max_mps"] = float(np.linalg.norm(np.diff(residual, axis=0), axis=1).max() / .05)
    metrics["tail35_39_pooled_sse_fraction"] = float(squared[35:].sum() / squared.sum()) if squared.sum() else 0.0
    inactive_rms = np.sqrt(np.mean(trace_x_effective[..., 10:] ** 2, axis=(1, 2, 3)))
    active_rms = np.sqrt(np.mean(trace_x_effective[..., :10] ** 2, axis=(1, 2, 3)))
    expected_inactive_paths = [
        ((1.0 - step / n) * epsilon[..., 10:]).to(device="cpu", dtype=torch.float32).numpy()
        for step in range(n)
    ] if epsilon is not None else None
    expected_inactive = [
        float(np.sqrt(np.mean(value ** 2))) for value in expected_inactive_paths
    ] if expected_inactive_paths is not None else inactive_rms.tolist()
    p1.durable_npz(
        cell / "tensors.npz", raw_native=raw_native, raw_scored=raw_scored, decoded=decoded,
        expert=expert, state=state, action_mask=mask, noise_source_float32=fixed_noise,
        noise_initial=noise_initial, trace_x_before=trace_x_before, trace_x=trace_x_effective,
        trace_v=trace_v, trace_t=trace_t, tau_left=np.arange(n, dtype=np.float64) / n,
        active_rms=active_rms, inactive_rms=inactive_rms,
        expected_inactive_rms=np.asarray(expected_inactive, dtype=np.float64),
    )
    record = {
        **attempt,
        "status": "VALID",
        "attempt_sha256": p1.sha(cell / "attempt.json"),
        "tensor_sha256": p1.sha(cell / "tensors.npz"),
        "condition_identity_recomputed": identity,
        "action_head_config": stock["action_head_config"],
        "sampler_steps": len(stock["trace"]),
        "head_denoising_forwards": len(stock["trace"]),
        "backbone_forwards": 1,
        "get_action_calls": 1,
        "physics": {"horizon": 40, "dt_s": 0.05, "active_dims": "0:10", "inactive_dims": "10:132"},
        "m_path": {
            "applicable": epsilon is not None,
            "definition": "(1 - j/N) * epsilon_inactive at each native action_encoder input",
            "active_input_unchanged": True,
            "final_inactive_raw_scored_zero": epsilon is not None,
            "max_inactive_input_roundoff": float(np.max(np.abs(
                trace_x_effective[..., 10:] - np.stack(expected_inactive_paths)
            ))) if epsilon is not None else None,
        },
        "trace_active_rms": active_rms.tolist(),
        "trace_inactive_rms": inactive_rms.tolist(),
        "trace_expected_inactive_rms": np.asarray(expected_inactive).tolist(),
        "metrics": metrics,
        "gate_decision": gate_result.decision.value,
        "gate": gate_result.as_dict(),
        "target_xyz_in_model_input": False,
    }
    p1.durable_json(cell / "record.json", record)
    print(json.dumps({"status": "PASS_FIRST_VALID_D1_CELL", "cell": str(cell),
                      "gate": record["gate_decision"], "xyz_rmse_m": metrics["xyz_rmse_m"]}), flush=True)
    return record


def collect_record(path: Path, p1: Any) -> dict:
    record = read_json(path)
    if record["status"] == "VALID":
        assert p1.sha(path.parent / "tensors.npz") == record["tensor_sha256"]
    else:
        assert p1.sha(path.parent / "invalid_output.npz") == record["tensor_sha256"]
    assert p1.sha(path.parent / "attempt.json") == record["attempt_sha256"]
    return record


def summarize(args: argparse.Namespace, card: dict, card_sha: str, context: dict) -> dict:
    p1 = context["p1"]
    cells = []
    new_records = {}
    for weight in card["weights"]:
        model = weight["id"]
        for source in card["bridge4_sources"]:
            base = source.split(":", 1)[0]
            for noise in ("N0", "N1"):
                for runtime in card["runtimes"]:
                    variant = runtime["id"]
                    if model == "R0" and variant == "S4":
                        old = context["source_cells"][source][noise]
                        old_record = read_json(Path(old["cell_path"]) / "record.json")
                        assert old_record["tensor_sha256"] == old["tensor_sha256"]
                        record = old_record
                        key = f"{model}__{variant}__{base}__{noise}"
                        record = {
                            "cell_key": key, "status": "REUSED_VALID", "model_name": model,
                            "runtime": variant, "source_episode_id": source, "noise_name": noise,
                            "metrics": record["metrics"], "gate_decision": record["gate_decision"],
                            "gate": record["gate"], "sampler_steps": record["sampler_steps"],
                            "record_sha256": old["record_sha256"], "tensor_sha256": old["tensor_sha256"],
                            "source_cell": old["cell_path"],
                        }
                    else:
                        key = f"{model}__{variant}__{base}__{noise}"
                        record = collect_record(args.output_root / "cells" / key / "record.json", p1)
                        new_records[key] = record
                        record = {**record, "record_sha256": p1.sha(args.output_root / "cells" / key / "record.json")}
                    cells.append(record)

    if len(cells) != len(card["weights"]) * len(card["bridge4_sources"]) * 2 * len(card["runtimes"]):
        raise RuntimeError("D1 matrix incomplete")
    effect_rows = []
    comparisons = (("M4", "S4"), ("M16", "S16"), ("S16", "S4"), ("M16", "M4"))
    by_key = {
        (row["model_name"], row["runtime"], row["source_episode_id"], row["noise_name"]): row
        for row in cells
    }
    for weight in card["weights"]:
        model = weight["id"]
        for candidate, control in comparisons:
            if candidate not in {row["id"] for row in card["runtimes"]} or control not in {row["id"] for row in card["runtimes"]}:
                continue
            for source in card["bridge4_sources"]:
                for noise in ("N0", "N1"):
                    a = by_key[(model, candidate, source, noise)]
                    b = by_key[(model, control, source, noise)]
                    effect_rows.append({
                        "model": model, "contrast": f"{candidate}-{control}",
                        "source_episode_id": source, "noise": noise,
                        "xyz_rmse_delta_m": (
                            float(a["metrics"]["xyz_rmse_m"] - b["metrics"]["xyz_rmse_m"])
                            if a.get("metrics") and b.get("metrics") else None
                        ),
                        "so3_rmse_delta_rad": (
                            float(a["metrics"]["so3_rmse_rad"] - b["metrics"]["so3_rmse_rad"])
                            if a.get("metrics") and b.get("metrics") else None
                        ),
                        "physical_residual_delta_max_delta_mps": (
                            float(a["metrics"].get("physical_residual_delta_max_mps", 0.0)
                                  - b["metrics"].get("physical_residual_delta_max_mps", 0.0))
                            if a.get("metrics") and b.get("metrics") else None
                        ),
                        "candidate_gate": a["gate_decision"], "control_gate": b["gate_decision"],
                    })

    selection_rows = []
    p_rows = [row for row in cells if row["model_name"] == "P2000"]
    p_s4 = [row for row in p_rows if row["runtime"] == "S4"]
    base_metrics = np.asarray([row["metrics"]["xyz_rmse_m"] for row in p_s4 if row.get("metrics")], dtype=np.float64)
    baseline_invalid = len(p_s4) != 8 or len(base_metrics) != 8
    for runtime in card["runtimes"]:
        if runtime["id"] == "S4":
            continue
        candidate = [row for row in p_rows if row["runtime"] == runtime["id"]]
        valid = [row for row in candidate if row.get("metrics")]
        values = np.asarray([row["metrics"]["xyz_rmse_m"] for row in valid], dtype=np.float64)
        base_rejects = sum(row["gate_decision"] == "REJECT" for row in p_s4)
        rejects = sum(row["gate_decision"] == "REJECT" for row in candidate)
        invalid = len(candidate) - len(valid)
        lower_count = sum(
            a["metrics"]["xyz_rmse_m"] < b["metrics"]["xyz_rmse_m"]
            for a, b in zip(candidate, p_s4) if a.get("metrics") and b.get("metrics")
        )
        summary = {
            "runtime": runtime["id"], "invalid_count": invalid, "reject_count": rejects,
            "s4_reject_count": base_rejects,
            "median_xyz_rmse_m": float(np.median(values)) if values.size else None,
            "s4_median_xyz_rmse_m": float(np.median(base_metrics)) if base_metrics.size else None,
            "p95_xyz_rmse_m": float(np.quantile(values, .95, method="linear")) if values.size else None,
            "s4_p95_xyz_rmse_m": float(np.quantile(base_metrics, .95, method="linear")) if base_metrics.size else None,
            "paired_xyz_lower_count_of_8": lower_count,
            "median_so3_rmse_rad": float(np.median([row["metrics"]["so3_rmse_rad"] for row in valid])) if valid else None,
        }
        summary["eligible"] = bool(
            not baseline_invalid and invalid == 0 and len(valid) == 8 and rejects <= base_rejects
            and summary["median_xyz_rmse_m"] is not None and summary["s4_median_xyz_rmse_m"] is not None
            and summary["median_xyz_rmse_m"] <= .9 * summary["s4_median_xyz_rmse_m"]
            and lower_count >= 6
            and summary["p95_xyz_rmse_m"] is not None and summary["s4_p95_xyz_rmse_m"] is not None
            and summary["p95_xyz_rmse_m"] <= summary["s4_p95_xyz_rmse_m"]
            and summary["median_so3_rmse_rad"] is not None and summary["median_so3_rmse_rad"] <= .050
        )
        selection_rows.append(summary)
    eligible = [row for row in selection_rows if row["eligible"]]
    rank = {row["id"]: row for row in card["runtimes"]}
    if eligible:
        winner = min(eligible, key=lambda row: (
            row["median_xyz_rmse_m"], row["p95_xyz_rmse_m"], row["reject_count"],
            rank[row["runtime"]]["steps"], rank[row["runtime"]]["inactive"] != "stock",
            row["runtime"],
        ))
        r_star, route = winner["runtime"], "SUPPORTED_RUNTIME_GAIN_IN_D1"
    else:
        r_star, route = "S4", "NO_SUPPORTED_RUNTIME_GAIN_IN_D1"

    return {
        "schema": "PI0247_D1_MECHANISM_REPORT_V1",
        "run_card_sha256": card_sha,
        "release_commit": card["release_commit"],
        "execution_parent": card["execution_parent"],
        "weights": card["weights"],
        "bridge4_sources": card["bridge4_sources"],
        "noise_identity": card["noise_identity"],
        "runtime_matrix": card["runtimes"],
        "counts": {
            "logical_cells": len(cells),
            "new_scored_queries": sum(row["status"] != "REUSED_VALID" for row in cells),
            "reused_valid_cells": sum(row["status"] == "REUSED_VALID" for row in cells),
            "head_denoising_forwards": sum(int(row.get("head_denoising_forwards", row.get("action_head_denoising_forwards", 0))) for row in cells),
        },
        "padding_proof": card["padding_proof"],
        "factor_effects": effect_rows,
        "p_weight_routing": {"criterion": card["r_star_rule"], "candidates": selection_rows,
                             "r_star": r_star, "route": route},
        "cells": cells,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run(args: argparse.Namespace) -> None:
    card, card_sha = validate_card(args.card, args.card_sha)
    code_root = Path(card["paths"]["code_root"])
    facts, context = prepare(Path(card["paths"]["stage_root"]), code_root)
    context["facts"] = facts
    assert canonical_sha(facts) == card["static_preflight_canonical_sha256"]
    expected_runtimes = [
        {"id": "S4", "steps": 4, "inactive": "stock"},
        {"id": "M4", "steps": 4, "inactive": "linear_noise_to_zero"},
        {"id": "S16", "steps": 16, "inactive": "stock"},
        {"id": "M16", "steps": 16, "inactive": "linear_noise_to_zero"},
    ] if facts["m_applicable"] else [
        {"id": "S4", "steps": 4, "inactive": "stock"},
        {"id": "S16", "steps": 16, "inactive": "stock"},
    ]
    assert card["release_commit"] == "50e66efe6ccfbe814d04caf12499493821be7957"
    assert [weight["id"] for weight in card["weights"]] == ["R0", "P2000"]
    assert card["bridge4_sources"] == facts["bridge4_sources"]
    assert card["runtimes"] == expected_runtimes
    assert card["padding_proof"] == facts["padding_proof"]
    assert card["noise_identity"] == {
        "manifest_sha256": facts["noise_manifest_sha256"],
        "files_sha256": facts["noise_files_sha256"],
        "source_tensor_sha256": facts["noise_source_tensor_sha256"],
        "initial_bfloat16_sha256": facts["noise_initial_bfloat16_sha256"],
    }
    assert card["resource"]["device"] == "cuda:0"
    assert str(Path(card["paths"]["output_root"])) == str(args.output_root)
    p1, screen = context["p1"], context["screen"]
    rows, paths, cards = context["rows"], context["paths"], context["cards"]
    from transformers import AutoModel
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0242_train_infer_bridge as bridge
    import pi0243_staged_learning_to_sim as stage43
    model_info = {}
    for weight in card["weights"]:
        model_root = Path(weight["model_root"])
        model_files = p1.verify_model_files(model_root)
        assert model_files["manifest_sha256"] == weight["manifest_sha256"]
        assert model_files["revision"] == weight["revision"]
        restore = read_json(model_root / "pi0245_inference_restore.json")
        if weight["id"] == "P2000":
            assert restore["checkpoint_step"] == 2000 and weight["family"] in restore["prefix"]
        else:
            assert restore["prefix"].endswith("checkpoint-10000")
        model_info[weight["id"]] = (model_root, model_files)

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_new = []
    for weight in card["weights"]:
        model_name = weight["id"]
        model_root, model_files = model_info[model_name]
        model = AutoModel.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
        model.eval().to(device=card["resource"]["device"], dtype=__import__("torch").bfloat16)
        assert int(model.action_head.num_inference_timesteps) == 4
        for runtime in card["runtimes"]:
            if model_name == "R0" and runtime["id"] == "S4":
                continue
            for source in card["bridge4_sources"]:
                for noise_name in ("N0", "N1"):
                    record = run_one(
                        args=args, card=card, card_sha=card_sha, context=context,
                        model_name=model_name, runtime=runtime, source=source, noise_name=noise_name,
                        model=model, model_files=model_files, bridge=bridge, selector=selector,
                        stage43=stage43, gate=TrajectorySafetyGateV1(),
                    )
                    all_new.append(record)
        del model
        __import__("torch").cuda.empty_cache()
    with (args.output_root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_new:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = summarize(args, card, card_sha, context)
    p1.durable_json(args.output_root / "pi0247_d1_mechanism_report.json", report)
    print(json.dumps({"status": "D1_MATRIX_COMPLETE", "r_star": report["p_weight_routing"]["r_star"],
                      "route": report["p_weight_routing"]["route"], "report": str(args.output_root / "pi0247_d1_mechanism_report.json")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--card-sha", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("PI0247 D1 CPU M-path self-check: PASS")
        return
    if args.preflight:
        assert args.stage_root and args.code_root and args.preflight_output
        facts, _ = prepare(args.stage_root, args.code_root)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "STATIC_PREFLIGHT_PASS", "m_applicable": facts["m_applicable"],
                          "padding_rows": len(facts["padding_proof"]["rows"]),
                          "reused_r0_s4": len(facts["reused_r0_s4_cells"]),
                          "preflight_sha256": canonical_sha(facts)}), flush=True)
        return
    assert args.card and args.card_sha and args.output_root
    run(args)


if __name__ == "__main__":
    main()
