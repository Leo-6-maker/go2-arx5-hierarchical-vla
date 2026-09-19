#!/usr/bin/env python3
"""Run the fixed PI0232 baseline + four-candidate TRAIN-CALIB evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


STEPS = (1000, 2500, 5000, 7500, 10000)
FRESH_ROOT = Path("/scratch/e1192548/pi0228_h200_large_adapt_v1_623593.hopper-m-02")
BASELINE = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/checkpoint-1000")
CALIB = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_calib_dataset")
TRAIN = Path("/scratch/e1192548/pi0227_large_adapt_v1_inputs/train_dataset")
STAGE10 = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902")
GROOT = STAGE10 / "src" / "isaac-gr00t"
MODALITY = Path("/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py")
COSMOS = Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561")
STAGE38 = Path("/scratch/e1192548/pi0228_h200_prep_20260919/scripts/stage38_structured_train.py")
SELECTOR = Path("/scratch/e1192548/pi0228_h200_prep_20260919/scripts/pi0228_train_calib_checkpoint_selection.py")
SEAL = Path("/scratch/e1192548/pi0228_h200_prep_20260919/evidence/pi0228_train_calib_seal.json")
AUTHORITY = Path("/scratch/e1192548/pi0228_h200_prep_20260919/evidence/pi0228_train_calib_target_authority.json")
EXPECTED_FORWARD_COUNT = 1088


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_cache(cache_root: Path) -> dict[str, object]:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(GROOT))
    spec = importlib.util.spec_from_file_location("pi0232_stage38", STAGE38)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage38 binding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"cache": module.bind_cosmos_cache(cache_root), "processor": module.bind_cosmos_processor()}


def run_profile(step: int, model_path: Path, output: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(SELECTOR),
        "--mode", "evaluate",
        "--output", str(output),
        "--stage10-root", str(STAGE10),
        "--base-model", str(BASELINE),
        "--model-path", str(model_path),
        "--train-dataset", str(TRAIN),
        "--calib-dataset", str(CALIB),
        "--modality-config", str(MODALITY),
        "--cosmos-snapshot", str(COSMOS),
        "--seal", str(SEAL),
        "--target-authority", str(AUTHORITY),
        "--checkpoint-step", str(step),
        "--device", "cuda:0",
        "--seed", "22301",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    (output.parent / f"profile-{step}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output.parent / f"profile-{step}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"profile {step} failed with {result.returncode}: {result.stderr[-1000:]}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    required = {
        "schema": "PI0228_TRAIN_CALIB_PROFILE_V1",
        "checkpoint_step": step,
        "forward_count": EXPECTED_FORWARD_COUNT,
        "expected_forward_count": EXPECTED_FORWARD_COUNT,
        "action_contract": "official_dense_relative_H40",
        "action_dt_s": 0.05,
        "seed": 22301,
        "target_xyz_in_model_input": False,
        "wbc_execution": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"profile {step} admission mismatch for {key}: {payload.get(key)!r}")
    if payload.get("safety_gate_counts", {}).get("REJECT", 1) != 0:
        raise RuntimeError(f"profile {step} has SafetyGate REJECT")
    return {"step": step, "path": str(output), "sha256": sha256_file(output), "payload": payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    if admission.get("status") != "PASS":
        raise RuntimeError("evaluator admission is not PASS")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"evaluation output root is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    binding = bind_cache(args.cache_root)
    profiles = []
    for step in STEPS:
        model_path = BASELINE if step == 1000 else FRESH_ROOT / "training" / f"checkpoint-{step}"
        output = args.output_root / f"profile-{step}.json"
        profiles.append(run_profile(step, model_path, output))
    payload = {
        "schema": "PI0232_TRAIN_CALIB_EVALUATION_RUN_V1",
        "status": "PASS",
        "admission": {"path": str(args.admission), "sha256": sha256_file(args.admission)},
        "runtime_binding": binding,
        "evaluation_order": list(STEPS),
        "expected_forward_count_per_profile": EXPECTED_FORWARD_COUNT,
        "profiles": [{k: v for k, v in item.items() if k != "payload"} for item in profiles],
        "safety_gate_reject_total": sum(item["payload"]["safety_gate_counts"]["REJECT"] for item in profiles),
        "wbc_execution": False,
        "target_xyz_in_model_input": False,
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
    }
    output = args.output_root / "pi0232_train_calib_evaluation_run.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
