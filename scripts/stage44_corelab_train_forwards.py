#!/usr/bin/env python3
"""Execute the bounded Stage44 A3 TRAIN-only GR00T forward cohort.

The runner loads the accepted STRUCTURED@1000 model and processor once, then
performs exactly one real ``model.get_action`` call for each of the 24 sealed
Stage22 TRAIN expert snapshots.  Each model-facing NPZ contains only
``body_rgb``, ``wrist_rgb``, and ``state``.  The target coordinate remains a
separate provenance field and is never written to those input NPZs.

This file intentionally does not import IsaacGym, native WBC code, or any
controller.  It is a model-forward diagnostic, not a rollout or training
job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

import stage44_corelab_evidence as evidence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _set_model_environment(stage10_root: Path, cosmos_snapshot: Path, gpu_uuid: str) -> Dict[str, str]:
    # Keep the model process independent from any native IsaacGym library path
    # that may be present in a user's shell.  All cache locations are the
    # already-materialized Stage10/Stage40 inputs; no download or install is
    # permitted here.
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    snapshot = Path(cosmos_snapshot).resolve(strict=True)
    cache_root = snapshot.parents[2]
    os.environ["HF_HUB_CACHE"] = str(cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root)
    os.environ["HF_HOME"] = str(cache_root.parent)
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu_uuid),
        "HF_HUB_CACHE": str(cache_root),
        "TRANSFORMERS_CACHE": str(cache_root),
        "HF_HOME": str(cache_root.parent),
        "offline": "true",
    }


def _ppm_rgb(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    cursor = 0

    def token() -> bytes:
        nonlocal cursor
        while cursor < len(data) and data[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            return token()
        start = cursor
        while cursor < len(data) and data[cursor] not in b" \t\r\n":
            cursor += 1
        return data[start:cursor]

    _require(token() == b"P6", "input frame is not P6: %s" % path)
    width = int(token()); height = int(token()); maximum = int(token())
    _require((width, height, maximum) == (96, 96, 255), "input frame ABI mismatch: %s" % path)
    while cursor < len(data) and data[cursor] in b" \t\r\n":
        cursor += 1
    payload = data[cursor:]
    _require(len(payload) == width * height * 3, "input frame byte count mismatch: %s" % path)
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()


def _snapshot(snapshot_path: Path) -> Dict[str, Any]:
    payload = evidence.read_json(snapshot_path)
    state = np.asarray(payload["state_v1"]["values"], dtype=np.float32).reshape(-1)
    _require(state.shape == (10,) and bool(np.isfinite(state).all()), "snapshot StateV1 is invalid: %s" % snapshot_path)
    body_path = Path(str(payload["body_frame_path"])).resolve(strict=True)
    wrist_path = Path(str(payload["wrist_frame_path"])).resolve(strict=True)
    body = _ppm_rgb(body_path); wrist = _ppm_rgb(wrist_path)
    return {
        "payload": payload,
        "example_id": str(payload["example_id"]),
        "target_id": str(payload["target_id"]),
        "target_position_world_m": list(payload["target_position_world_m"]),
        "observation_timestamp_s": float(payload["observation_timestamp"]),
        "state": state,
        "body": body,
        "wrist": wrist,
        "body_path": body_path,
        "wrist_path": wrist_path,
    }


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _install_forward_counter(model: Any, counter: Dict[str, int]) -> None:
    original = model.get_action

    def counted_get_action(*args: Any, **kwargs: Any) -> Any:
        counter["calls"] += 1
        return original(*args, **kwargs)

    # AutoModel is a normal Python object; assigning this bound closure makes
    # the serialized counter depend on the actual runtime call path instead of
    # a declared loop budget.
    setattr(model, "get_action", counted_get_action)


def _load_runtime(bundle: Mapping[str, Any], args: argparse.Namespace):
    artifacts = bundle["artifacts"]
    stage10_root = Path(artifacts["stage10_root"]["path"]).resolve(strict=True)
    cosmos_snapshot = Path(artifacts["cosmos_snapshot"]["path"]).resolve(strict=True)
    _set_model_environment(stage10_root, cosmos_snapshot, args.gpu_uuid)
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    if str(stage10_root / "src" / "isaac-gr00t") not in sys.path:
        sys.path.insert(0, str(stage10_root / "src" / "isaac-gr00t"))
    import stage21_same_scene_groot_inference as inference
    import torch
    from stage40_b0_model_control import _bind_local_cosmos_processor

    _bind_local_cosmos_processor(stage10_root, cosmos_snapshot)
    processor, embodiment = inference._build_processor(
        stage10_root,
        Path(artifacts["base_model_path"]["path"]).resolve(strict=True),
        Path(artifacts["tiny1_dataset"]["path"]).resolve(strict=True) / "meta",
        Path(artifacts["modality"]["path"]).resolve(strict=True),
    )
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    model_path = Path(artifacts["structured_checkpoint"]["path"]).resolve(strict=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    return inference, torch, model, processor, embodiment


def run(args: argparse.Namespace) -> Dict[str, Any]:
    bundle_path = Path(args.bundle).resolve(strict=True)
    bundle = evidence.read_json(bundle_path, canonical=True)
    _require(bundle.get("schema") == "CORELAB_STAGE44_EXECUTION_BUNDLE_V1", "wrong Stage44 bundle schema")
    _require(bundle.get("pi_gate") == evidence.PI_GATE and bundle.get("train_only_inputs", {}).get("snapshot_count") == evidence.SNAPSHOT_COUNT, "Stage44 bundle contract drift")
    _require(bundle.get("budgets", {}).get("a3_gr00t_forward_count") == evidence.SNAPSHOT_COUNT and bundle.get("budgets", {}).get("wbc_forward_count") == 0, "Stage44 budget drift")
    _require(bundle.get("forbidden_operations") == evidence.forbidden_operations(), "Stage44 forbidden-operation map drift")
    runner = Path(__file__).resolve(strict=True)
    _require(bundle.get("runner", {}).get("sha256") == evidence.sha256_file(runner), "A3 runner hash is not the frozen bundle runner")
    output_root = Path(args.output_root).resolve()
    _require(not output_root.exists(), "A3 output root already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    rows_a1 = evidence.read_json(Path(args.pre_root).resolve(strict=True) / "a1_expert_label_audit.json", canonical=True)["rows"]
    _require(len(rows_a1) == evidence.SNAPSHOT_COUNT, "A1 does not contain 24 labels")
    expected_ids = list(evidence._expected_snapshot_ids())
    _require([row["example_id"] for row in rows_a1] == expected_ids, "A1 example order drift")

    inference, torch, model, processor, embodiment = _load_runtime(bundle, args)
    from go2arx5_vla.stage40_structured_decode import decode_raw_structured_action

    call_counter = {"calls": 0}
    _install_forward_counter(model, call_counter)
    all_rows: List[Dict[str, Any]] = []
    all_finite = True
    for index, row in enumerate(rows_a1):
        snapshot = _snapshot(Path(row["snapshot_path"]).resolve(strict=True))
        example_id = snapshot["example_id"]
        _require(example_id == expected_ids[index], "A3 input order drift: %s" % example_id)
        input_stem = "sample-%04d" % index
        input_path = output_root / "inputs" / (input_stem + ".npz")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately omit target_position_world_m.  The task target is kept
        # only in the result row as label provenance.
        np.savez_compressed(input_path, body_rgb=snapshot["body"], wrist_rgb=snapshot["wrist"], state=snapshot["state"])
        input_sha = evidence.sha256_file(input_path)
        observation = inference._load_observation(input_path)
        _require(observation.get("target_position_world_m") is None, "A3 observation unexpectedly loaded a target")
        before = call_counter["calls"]
        _set_seed(evidence.DEV_SEED, torch)
        raw, latency, gpu = inference._run_one_raw(
            model, processor, embodiment, observation,
            language=evidence.LANGUAGE, seed=evidence.DEV_SEED, device=args.device,
        )
        after = call_counter["calls"]
        _require(after - before == 1, "A3 runtime did not observe exactly one get_action call: %s" % example_id)
        raw = np.asarray(raw, dtype=np.float32)
        _require(raw.shape == (1, 40, 132) and bool(np.isfinite(raw).all()), "A3 raw output ABI/finiteness failure: %s" % example_id)
        decoded_chunk, decoded, decoder_details = decode_raw_structured_action(
            raw,
            snapshot["state"],
            stats_path=Path(bundle["artifacts"]["structured_stats"]["path"]).resolve(strict=True),
            source_timestamps=np.asarray(snapshot["payload"]["future_timestamps"], dtype=np.float64),
            observation_timestamp=float(snapshot["observation_timestamp_s"]),
            chunk_id="stage44-%s" % input_stem,
            source_frame=evidence.SOURCE_FRAME,
            tcp_semantic_id=evidence.TCP_SEMANTIC_ID,
        )
        del decoded_chunk
        decoded = np.asarray(decoded, dtype=np.float64)
        _require(decoded.shape == (40, 10) and bool(np.isfinite(decoded).all()), "A3 decoded output ABI/finiteness failure: %s" % example_id)
        artifact_path = output_root / "outputs" / (input_stem + ".npz")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact_path, raw_universal_action=raw, decoded_physical_action=decoded, input_state=snapshot["state"], body_rgb=snapshot["body"], wrist_rgb=snapshot["wrist"])
        artifact_sha = evidence.sha256_file(artifact_path)
        all_finite = all_finite and bool(np.isfinite(raw).all() and np.isfinite(decoded).all())
        all_rows.append({
            "index": index,
            "example_id": example_id,
            "status": "PASS",
            "target_id": snapshot["target_id"],
            "target_position_world_m": snapshot["target_position_world_m"],
            # Preserve the sealed source path for replay/audit traceability;
            # the example id remains a separate immutable lineage key.
            "snapshot_path": str(Path(row["snapshot_path"]).resolve(strict=True)),
            "input_npz_path": str(input_path),
            "input_npz_sha256": input_sha,
            "model_input_keys": list(evidence.MODEL_INPUT_KEYS),
            "target_xyz_in_model_input": False,
            "body_shape": list(snapshot["body"].shape),
            "wrist_shape": list(snapshot["wrist"].shape),
            "body_sha256": evidence.sha256_array(snapshot["body"]),
            "wrist_sha256": evidence.sha256_array(snapshot["wrist"]),
            "state_v1_sha256": evidence.sha256_array(snapshot["state"]),
            "state_v1": snapshot["state"].tolist(),
            "raw_shape": list(raw.shape),
            "decoded_shape": list(decoded.shape),
            "raw_action_sha256": evidence.sha256_array(raw),
            "decoded_action_sha256": evidence.sha256_array(decoded),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "observation_timestamp_s": snapshot["observation_timestamp_s"],
            "language": evidence.LANGUAGE,
            "seed": evidence.DEV_SEED,
            "forward_count_before": before,
            "forward_count_after": after,
            "forward_count_delta_observed": after - before,
            "latency_s": float(latency),
            "gpu": gpu,
            "decoder": decoder_details,
            "finite": True,
            "simulation_only": True,
            "hardware_control": False,
        })
    _require(call_counter["calls"] == evidence.SNAPSHOT_COUNT, "A3 observed forward total mismatch")
    manifest = {
        "schema": "CORELAB_STAGE44_A3_TRAIN_ONLY_GR00T_FORWARD_MANIFEST_V1",
        "stage": evidence.STAGE_ID,
        "pi_gate": evidence.PI_GATE,
        "status": "PASS" if all_finite else "FAIL",
        "bundle_sha256": evidence.sha256_file(bundle_path),
        "bundle": str(bundle_path),
        "repository_head": bundle.get("repository", {}).get("head"),
        "model_input_keys": list(evidence.MODEL_INPUT_KEYS),
        "target_xyz_in_model_input": False,
        "language": evidence.LANGUAGE,
        "seed": evidence.DEV_SEED,
        "gr00t_forward_count_observed": call_counter["calls"],
        "wbc_forward_count_observed": 0,
        "reference_update_count_observed": 0,
        "training_steps_observed": 0,
        "backward_steps_observed": 0,
        "optimizer_steps_observed": 0,
        "all_outputs_finite": all_finite,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu_uuid_binding": args.gpu_uuid,
            "shared_gpu_allowed": True,
            "environment": {key: value for key, value in os.environ.items() if key in {"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "WANDB_MODE"}},
        },
        "rows": all_rows,
        "forbidden_operations": evidence.forbidden_operations(),
        "claim_boundary": "24 real TRAIN-only GR00T forwards on sealed Stage22 observations; no WBC, training, validation, holdout, or task-success claim",
    }
    evidence.write_json(output_root / "a3_train_forwards_manifest.json", manifest)
    print(json.dumps({"stage": evidence.STAGE_ID, "status": manifest["status"], "gr00t_forward_count_observed": manifest["gr00t_forward_count_observed"], "output_root": str(output_root), "manifest_sha256": evidence.sha256_file(output_root / "a3_train_forwards_manifest.json")}, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pre-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-uuid", default="GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
