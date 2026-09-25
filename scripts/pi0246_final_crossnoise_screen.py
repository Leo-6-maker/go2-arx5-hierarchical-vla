#!/usr/bin/env python3
"""Run the authorized C/T-final × bridge4 × sealed-noise stock diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import pi0245_p1_stock as p1

MODELS = ("C2000", "T2000")


def paths_for(stage_root: Path, output_root: Path) -> tuple[dict, dict, dict]:
    card_path = stage_root / "r0/pi0243_run_card.json"
    overlay_path = stage_root / "pi0246_p1_path_overlay_corelab_20260924.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["_sha256"] = p1.sha(card_path)
    paths = p1.resolve_path_overlay(card, overlay_path)
    paths["output_root"] = str(output_root)
    return card, paths, {"run_card": card_path, "overlay": overlay_path}


def expected_keys() -> list[str]:
    return [f"{model}__{source}__{noise}" for model in MODELS for source in p1.SOURCES for noise in p1.NOISES]


def close(root: Path, rows: dict[str, dict]) -> dict:
    keys = expected_keys()
    ledger_path = root / "queries.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if [row.get("cell_key") for row in ledger] != keys:
        raise RuntimeError("cross-noise query ledger is not the exact frozen 32-cell order")
    cells = []
    records = []
    for key in keys:
        model_name, source, noise_name = key.split("__", 2)
        cell = root / "cells" / f"{model_name}__{source.split(':', 1)[0]}__{noise_name}"
        record_path, tensor_path = cell / "record.json", cell / "tensors.npz"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema") != "PI0245_P1_STOCK_CELL_V1" or record.get("cell_key") != key:
            raise RuntimeError(f"unexpected or invalid first-valid cell: {key}")
        if record.get("model_name") != model_name or record.get("source_episode_id") != source or record.get("noise_name") != noise_name:
            raise RuntimeError(f"cell identity mismatch: {key}")
        if record.get("target_xyz_in_model_input") is not False:
            raise RuntimeError(f"target XYZ entered model input: {key}")
        if record.get("source_tensor_sha256") != rows[source]["tensor_sha256"]:
            raise RuntimeError(f"source tensor identity mismatch: {key}")
        if (record.get("model_get_action_calls"), record.get("model_backbone_forwards")) != (1, 1):
            raise RuntimeError(f"unexpected policy/backbone call count: {key}")
        if record.get("gate_decision") not in {"ACCEPT", "WARN", "REJECT"}:
            raise RuntimeError(f"unknown gate decision: {key}")
        for name, expected in (("tensors.npz", record.get("tensor_sha256")),
                               ("condition.pt", record.get("condition_sha256")),
                               ("action_inputs.pt", record.get("action_inputs_sha256")),
                               ("backbone_features.pt", record.get("backbone_features_sha256"))):
            if not expected or p1.sha(cell / name) != expected:
                raise RuntimeError(f"artifact hash mismatch: {key}/{name}")
        with np.load(tensor_path, allow_pickle=False) as data:
            if data["raw"].shape != (1, 40, 132) or data["decoded"].shape != (1, 40, 10):
                raise RuntimeError(f"tensor shape mismatch: {key}")
            if any(not np.isfinite(data[name]).all() for name in data.files):
                raise RuntimeError(f"nonfinite tensor: {key}")
        item = {"cell_key": key, "record_sha256": p1.sha(record_path),
                "tensor_sha256": record["tensor_sha256"], "condition_sha256": record["condition_sha256"],
                "gate_decision": record["gate_decision"], "noise_name": noise_name,
                "model_name": model_name, "source_episode_id": source, "metrics": record["metrics"],
                "physical_residual_delta_max_mps": record["physical_residual_delta_max_mps"],
                "physical_max_jump_m": record["physical_max_jump_m"]}
        records.append(record)
        cells.append(item)
    summaries = {}
    for model_name in MODELS:
        for noise_name in p1.NOISES:
            group = [row for row in records if row["model_name"] == model_name and row["noise_name"] == noise_name]
            xyz = np.asarray([row["metrics"]["xyz_rmse_m"] for row in group], dtype=np.float64)
            so3 = np.asarray([row["metrics"]["so3_rmse_rad"] for row in group], dtype=np.float64)
            summaries[f"{model_name}/{noise_name}"] = {
                "cells": len(group), "gate_counts": dict(Counter(row["gate_decision"] for row in group)),
                "median_xyz_rmse_m": float(np.median(xyz)), "p95_xyz_rmse_m": float(np.quantile(xyz, .95, method="linear")),
                "median_so3_rmse_rad": float(np.median(so3)),
            }
    closure = {"schema": "PI0246_C_T_FINAL_CROSSNOISE_32_V1", "status": "PASS_COMPLETE_32_CELLS",
               "records": len(records), "model_queries": 32, "models": list(MODELS),
               "sources": list(p1.SOURCES), "noise_names": list(p1.NOISES),
               "runtime_binding_sha256": p1.sha(root / "runtime_binding.json"),
               "queries_sha256": p1.sha(ledger_path), "summaries": summaries, "cells": cells}
    closure_path = root / "crossnoise_closure.json"
    if closure_path.exists():
        if json.loads(closure_path.read_text(encoding="utf-8")) != closure:
            raise RuntimeError("first-valid cross-noise closure differs")
    else:
        p1.durable_json(closure_path, closure)
    return closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--c2000-model-root", type=Path, required=True)
    parser.add_argument("--t2000-model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.stage_root = args.stage_root.resolve(strict=True)
    args.noise_root = args.noise_root.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    model_roots = {"C2000": args.c2000_model_root.resolve(strict=True),
                   "T2000": args.t2000_model_root.resolve(strict=True)}
    card, paths, card_paths = paths_for(args.stage_root, args.output_root)
    r0_root = Path(paths["r0_root"]).resolve(strict=True)
    rows = p1.sources(r0_root)
    noises = p1.noise_bank(args.noise_root)
    model_files = {name: p1.verify_model_files(model_roots[name]) for name in MODELS}
    if set(noises) != set(p1.NOISES) or set(rows) != set(p1.SOURCES):
        raise RuntimeError("sealed bridge4/noise identities or ordering differ")
    sys.path[:0] = [str(Path(paths["code_root"]) / "scripts"), str(Path(paths["code_root"]) / "src"), paths["groot_source"]]
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import stage40_b0_model_control as stage40
    cosmos_binding = stage40._bind_local_cosmos_processor(Path(paths["stage10_root"]), Path(paths["cosmos_snapshot"]))
    import stage21_same_scene_groot_inference as stage21
    processor, embodiment = stage21._build_processor(
        Path(paths["stage10_root"]), Path(paths["processor_base_model"]),
        Path(paths["train_dataset"]) / "meta", Path(paths["modality_config"]),
    )
    import pi0242_train_infer_bridge as bridge
    import pi0228_train_calib_checkpoint_selection as selector
    import pi0243_staged_learning_to_sim as stage43
    import torch
    from transformers import AutoModel
    if args.preflight:
        print(json.dumps({"status": "PASS_CROSSNOISE_CPU_PREFLIGHT", "cells": len(expected_keys()),
                          "source_ids": list(p1.SOURCES), "noise_names": list(noises),
                          "model_files": model_files, "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
                          "cosmos_binding": cosmos_binding, "model_weights_loaded": 0, "model_queries": 0}, sort_keys=True))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime_paths = {**paths, "output_root": str(args.output_root),
                     "run_card": str(card_paths["run_card"]), "path_overlay": str(card_paths["overlay"]),
                     "noise_root": str(args.noise_root), "model_roots": {k: str(v) for k, v in model_roots.items()}}
    runtime_files = [Path(__file__).resolve(), Path(p1.__file__).resolve(), Path(stage21.__file__).resolve(),
                     Path(bridge.__file__).resolve(), Path(selector.__file__).resolve(), Path(stage43.__file__).resolve(),
                     Path(stage40.__file__).resolve()]
    binding = {"schema": "PI0246_C_T_FINAL_CROSSNOISE_RUNTIME_V1", "stage_run_card_sha256": p1.sha(card_paths["run_card"]),
               "path_overlay_sha256": p1.sha(card_paths["overlay"]), "effective_paths": runtime_paths,
               "r0_records_sha256": p1.sha(r0_root / "screen/step-10000/records.jsonl"),
               "noise_manifest_sha256": p1.sha(args.noise_root / "manifest.json"),
               "models": {name: {**model_files[name], "restore_manifest_sha256": p1.sha(model_roots[name] / "pi0245_inference_restore.json")} for name in MODELS},
               "cosmos_binding": cosmos_binding, "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
               "python": sys.version, "torch": torch.__version__, "modules": [{"path": str(path), "sha256": p1.sha(path)} for path in runtime_files]}
    binding_path = args.output_root / "runtime_binding.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise RuntimeError("cross-noise runtime binding drift")
    else:
        p1.durable_json(binding_path, binding)
    for model_name in MODELS:
        model = AutoModel.from_pretrained(model_roots[model_name], local_files_only=True, trust_remote_code=True)
        model.eval().to(device=args.device, dtype=torch.bfloat16)
        model_files_for_query = {**model_files[model_name], "checkpoint_step": 2000,
                                 "family": json.loads((model_roots[model_name] / "pi0245_inference_restore.json").read_text())["family"]}
        one_args = SimpleNamespace(output_root=args.output_root, model_name=model_name, device=args.device)
        for source in p1.SOURCES:
            for noise_name in p1.NOISES:
                p1.one(one_args, rows[source], noise_name, noises[noise_name], paths=paths, model=model,
                       model_files=model_files_for_query, processor=processor, embodiment=embodiment,
                       bridge=bridge, selector=selector, stage43=stage43)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(close(args.output_root, rows), sort_keys=True))


if __name__ == "__main__":
    main()
