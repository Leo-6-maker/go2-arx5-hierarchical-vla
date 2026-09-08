#!/usr/bin/env python3
"""Run the bounded Stage42 archived replay and modality probes.

This runner is the only Stage42 command that constructs the frozen GR00T
runtime.  It performs exactly three archived replays followed, only when all
three replay hashes match, by the six pre-registered component swaps.  It
never constructs IsaacGym/WBC, writes checkpoints, reads a new dataset split,
or changes the model/processor/precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import stage42_feedback_causality as evidence  # noqa: E402


PROBE_IDS = ("REPLAY-S0", "REPLAY-S5", "REPLAY-S10", "CF1", "CF2", "CF3", "CF4", "CF5", "CF6")
REPLAY_IDS = PROBE_IDS[:3]
CF_IDS = PROBE_IDS[3:]
LANGUAGE = evidence.LANGUAGE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _set_model_environment(stage10_root: Path, cosmos_snapshot: Path) -> Dict[str, str]:
    # Match the accepted Stage41 model-worker isolation: the native IsaacGym
    # library path must not leak into the Python 3.12 model process.
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    cache_root = Path(cosmos_snapshot).resolve().parents[2]
    os.environ["HF_HUB_CACHE"] = str(cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root)
    os.environ["HF_HOME"] = str(cache_root.parent)
    return {
        "HF_HUB_CACHE": str(cache_root),
        "TRANSFORMERS_CACHE": str(cache_root),
        "HF_HOME": str(cache_root.parent),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return evidence.read_json(path)


def _load_source_archive(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    frozen = bundle.get("frozen_inputs", {})
    root = Path(str(frozen.get("source_run_root", ""))).resolve(strict=True)
    records = frozen.get("selected_observations", {})
    _require(set(records) == set(evidence.SELECTED_IDS), "frozen selected-observation set drift")
    output: Dict[str, Dict[str, Any]] = {}
    for request_id in evidence.SELECTED_IDS:
        record = records[request_id]
        path = root / str(record.get("request_relative_path"))
        _require(path.is_file() and not path.is_symlink(), "archived request NPZ missing: %s" % request_id)
        _require(evidence.sha256_file(path) == str(record.get("request_file_sha256")), "archived request NPZ hash mismatch: %s" % request_id)
        with np.load(path, allow_pickle=False) as data:
            required = ("body_rgb", "wrist_rgb", "state", "target_position_world_m")
            _require(all(key in data.files for key in required), "archived request ABI mismatch: %s" % request_id)
            body = np.asarray(data["body_rgb"], dtype=np.uint8)
            wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
            state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
            target = np.asarray(data["target_position_world_m"], dtype=np.float64).reshape(-1)
        evidence._finite_array(body, (96, 96, 3), request_id + ".body", np.uint8)
        evidence._finite_array(wrist, (96, 96, 3), request_id + ".wrist", np.uint8)
        evidence._finite_array(state, (10,), request_id + ".state", np.float32)
        evidence._finite_array(target, (3,), request_id + ".target", np.float64)
        expected_components = record.get("components", {})
        _require(evidence.sha256_array(body) == expected_components.get("body_sha256"), "archived body component hash mismatch: %s" % request_id)
        _require(evidence.sha256_array(wrist) == expected_components.get("wrist_sha256"), "archived wrist component hash mismatch: %s" % request_id)
        _require(evidence.sha256_array(state) == expected_components.get("state_v1_sha256"), "archived StateV1 component hash mismatch: %s" % request_id)
        output[request_id] = {
            "request_id": request_id,
            "source_path": str(path),
            "request_file_sha256": str(record["request_file_sha256"]),
            "body": body,
            "wrist": wrist,
            "state": state,
            "target": target,
            "timestamp_s": float(record["observation_timestamp_s"]),
            "component_hashes": {
                "body_sha256": evidence.sha256_array(body),
                "wrist_sha256": evidence.sha256_array(wrist),
                "state_v1_sha256": evidence.sha256_array(state),
            },
        }
    return output


def _write_hybrid(path: Path, body: np.ndarray, wrist: np.ndarray, state: np.ndarray, target: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, body_rgb=body, wrist_rgb=wrist, state=state, target_position_world_m=target)


def _load_expected_raw_hash(bundle: Mapping[str, Any], request_id: str) -> str:
    expected = bundle.get("frozen_inputs", {}).get("accepted_raw_action_sha256", {})
    value = expected.get(request_id)
    _require(isinstance(value, str) and len(value) == 64, "accepted raw hash is missing: %s" % request_id)
    return value


def _state_orientation_quaternion(state: np.ndarray) -> np.ndarray:
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    return np.asarray(matrix_to_quat(rot6d_rows_to_matrix(np.asarray(state[3:9], dtype=np.float64))), dtype=np.float64)


def _build_safety_gate(chunk: Any, state: np.ndarray, timestamp_s: float) -> Dict[str, Any]:
    from go2arx5_vla.safety import TrajectorySafetyGateV1

    gate = TrajectorySafetyGateV1()
    result = gate.evaluate(
        chunk,
        now=float(timestamp_s),
        achieved_position=np.asarray(state[:3], dtype=np.float64),
        achieved_orientation=_state_orientation_quaternion(state),
    )
    return result.as_dict()


def _delta_summary(current: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(current) - np.asarray(baseline)
    return {
        "l2": float(np.linalg.norm(delta.reshape(-1).astype(np.float64))),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
    }


def _make_task_inputs(archive: Mapping[str, Mapping[str, Any]], output_root: Path) -> List[Dict[str, Any]]:
    s0, s5, s10 = archive["stage41-q0"], archive["stage41-q5"], archive["stage41-q10"]
    definitions = [
        {"probe_id": "REPLAY-S0", "kind": "exact_replay", "source_observation_id": "stage41-q0", "baseline_id": "stage41-q0", "input_path": s0["source_path"]},
        {"probe_id": "REPLAY-S5", "kind": "exact_replay", "source_observation_id": "stage41-q5", "baseline_id": "stage41-q5", "input_path": s5["source_path"]},
        {"probe_id": "REPLAY-S10", "kind": "exact_replay", "source_observation_id": "stage41-q10", "baseline_id": "stage41-q10", "input_path": s10["source_path"]},
        {"probe_id": "CF1", "kind": "counterfactual", "source_observation_id": "stage41-q0+state:stage41-q10", "baseline_id": "stage41-q0", "body": s0["body"], "wrist": s0["wrist"], "state": s10["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
        {"probe_id": "CF2", "kind": "counterfactual", "source_observation_id": "stage41-q10+state:stage41-q0", "baseline_id": "stage41-q0", "body": s10["body"], "wrist": s10["wrist"], "state": s0["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
        {"probe_id": "CF3", "kind": "counterfactual", "source_observation_id": "body:stage41-q10+stage41-q0", "baseline_id": "stage41-q0", "body": s10["body"], "wrist": s0["wrist"], "state": s0["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
        {"probe_id": "CF4", "kind": "counterfactual", "source_observation_id": "wrist:stage41-q10+stage41-q0", "baseline_id": "stage41-q0", "body": s0["body"], "wrist": s10["wrist"], "state": s0["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
        {"probe_id": "CF5", "kind": "counterfactual", "source_observation_id": "stage41-q5+state:stage41-q0", "baseline_id": "stage41-q0", "body": s5["body"], "wrist": s5["wrist"], "state": s0["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
        {"probe_id": "CF6", "kind": "counterfactual", "source_observation_id": "state:stage41-q5+stage41-q0", "baseline_id": "stage41-q0", "body": s0["body"], "wrist": s0["wrist"], "state": s5["state"], "target": s0["target"], "timestamp_s": s0["timestamp_s"]},
    ]
    tasks: List[Dict[str, Any]] = []
    for item in definitions:
        task = dict(item)
        if item["kind"] == "counterfactual":
            input_path = output_root / "inputs" / (str(item["probe_id"]) + ".npz")
            _write_hybrid(input_path, item["body"], item["wrist"], item["state"], item["target"])
            task["input_path"] = str(input_path)
            task["component_hashes"] = {
                "body_sha256": evidence.sha256_array(item["body"]),
                "wrist_sha256": evidence.sha256_array(item["wrist"]),
                "state_v1_sha256": evidence.sha256_array(item["state"]),
            }
            task.pop("body", None)
            task.pop("wrist", None)
            task.pop("state", None)
            task.pop("target", None)
        else:
            selected = archive[str(item["source_observation_id"])]
            task["component_hashes"] = dict(selected["component_hashes"])
            task["timestamp_s"] = float(selected["timestamp_s"])
        tasks.append(task)
    _require([item["probe_id"] for item in tasks] == list(PROBE_IDS), "probe task order drift")
    return tasks


def _load_model(args: argparse.Namespace):
    _set_model_environment(args.stage10_root, args.cosmos_snapshot)
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    import stage21_same_scene_groot_inference as inference
    import torch
    from stage40_b0_model_control import _bind_local_cosmos_processor

    _bind_local_cosmos_processor(args.stage10_root, args.cosmos_snapshot)
    processor, embodiment = inference._build_processor(args.stage10_root, args.base_model_path, args.dataset_dir / "meta", args.modality_config)
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path, local_files_only=bool(args.local_files_only), trust_remote_code=True)
    model.eval()
    model.to(device=args.device, dtype=torch.bfloat16)
    return inference, model, processor, embodiment, torch


def run(args: argparse.Namespace) -> Dict[str, Any]:
    bundle = _load_json(args.bundle)
    evidence._verify_bundle(bundle)
    bundle_sha = evidence.sha256_bytes(evidence.canonical_json_bytes(bundle))
    output_root = Path(args.output_dir).resolve()
    _require(not output_root.exists(), "probe output directory already exists")
    output_root.mkdir(parents=True)
    archive = _load_source_archive(bundle)
    tasks = _make_task_inputs(archive, output_root)
    inference, model, processor, embodiment, torch = _load_model(args)
    from go2arx5_vla.stage40_structured_decode import decode_raw_structured_action

    results: List[Dict[str, Any]] = []
    replay_pass = True
    for index, task in enumerate(tasks):
        input_path = Path(task["input_path"])
        request = {
            "schema": "CORELAB_STAGE42_GR00T_PROBE_REQUEST_V1",
            "probe_id": task["probe_id"],
            "kind": task["kind"],
            "source_observation_id": task["source_observation_id"],
            "input_path": str(input_path),
            "input_file_sha256": evidence.sha256_file(input_path),
            "component_hashes": task["component_hashes"],
            "language": LANGUAGE,
            "seed": evidence.DEV_SEED,
            "action_contract": "STRUCTURED_ACTION_V1",
            "simulation_only": True,
            "hardware_control": False,
            "wbc_execution": False,
            "reference_update": False,
        }
        evidence.write_canonical_json(output_root / "requests" / (task["probe_id"] + ".json"), request)
        observation = inference._load_observation(input_path)
        raw, latency, gpu = inference._run_one_raw(
            model, processor, embodiment, observation, language=LANGUAGE, seed=evidence.DEV_SEED, device=args.device
        )
        raw = np.asarray(raw, dtype=np.float32)
        chunk, decoded, decoder_details = decode_raw_structured_action(
            raw,
            observation["state"],
            stats_path=args.structured_stats,
            source_timestamps=np.arange(40, dtype=np.float64) * evidence.ACTION_DT_S,
            observation_timestamp=float(task["timestamp_s"]),
            chunk_id=task["probe_id"],
            source_frame=evidence.SOURCE_FRAME,
            tcp_semantic_id=evidence.TCP_SEMANTIC_ID,
        )
        decoded = np.asarray(decoded, dtype=np.float64)
        _require(raw.shape == (1, 40, 132) and decoded.shape == (40, 10), "probe action ABI mismatch")
        _require(bool(np.isfinite(raw).all() and np.isfinite(decoded).all()), "probe action is non-finite")
        raw_sha = evidence.sha256_array(raw)
        decoded_sha = evidence.sha256_array(decoded)
        if task["kind"] == "exact_replay":
            expected_hash = _load_expected_raw_hash(bundle, str(task["source_observation_id"]))
            exact = raw_sha == expected_hash
            replay_pass = replay_pass and exact
        else:
            exact = None
        baseline_result = next((item for item in results if item["probe_id"] == task["baseline_id"]), None)
        raw_delta = None if baseline_result is None else _delta_summary(raw, np.load(Path(baseline_result["artifact_path"]), allow_pickle=False)["raw_universal_action"])
        decoded_delta = None if baseline_result is None else _delta_summary(decoded, np.load(Path(baseline_result["artifact_path"]), allow_pickle=False)["decoded_physical_action"])
        gate = _build_safety_gate(chunk, observation["state"], float(task["timestamp_s"]))
        artifact_path = output_root / "outputs" / (task["probe_id"] + ".npz")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            raw_universal_action=raw,
            decoded_physical_action=decoded,
            input_state=observation["state"],
            body_rgb=observation["body_rgb"],
            wrist_rgb=observation["wrist_rgb"],
        )
        response = {
            "schema": "CORELAB_STAGE42_GR00T_PROBE_RESPONSE_V1",
            "status": "PASS",
            "probe_id": task["probe_id"],
            "action_contract": "STRUCTURED_ACTION_V1",
            "raw_shape": list(raw.shape),
            "decoded_shape": list(decoded.shape),
            "raw_action_sha256": raw_sha,
            "decoded_action_sha256": decoded_sha,
            "latency_s": float(latency),
            "gpu": gpu,
            "seed": evidence.DEV_SEED,
            "simulation_only": True,
            "hardware_control": False,
            "wbc_execution": False,
            "reference_update": False,
        }
        evidence.write_canonical_json(output_root / "responses" / (task["probe_id"] + ".json"), response)
        results.append({
            "probe_id": task["probe_id"],
            "kind": task["kind"],
            "source_observation_id": task["source_observation_id"],
            "baseline_id": task["baseline_id"],
            "input_path": str(input_path),
            "input_file_sha256": request["input_file_sha256"],
            "component_hashes": task["component_hashes"],
            "forward_count_before": index,
            "forward_count_after": index + 1,
            "forward_count_delta": 1,
            "raw_action_sha256": raw_sha,
            "decoded_action_sha256": decoded_sha,
            "accepted_stage41_raw_sha256": _load_expected_raw_hash(bundle, str(task["source_observation_id"])) if task["kind"] == "exact_replay" else None,
            "exact_replay_hash_match": exact,
            "raw_shape": list(raw.shape),
            "decoded_shape": list(decoded.shape),
            "finite": True,
            "latency_s": float(latency),
            "gpu": gpu,
            "safety_gate": gate,
            "raw_delta_vs_baseline": raw_delta,
            "decoded_delta_vs_baseline": decoded_delta,
            "artifact_path": str(artifact_path),
            "request_path": str(output_root / "requests" / (task["probe_id"] + ".json")),
            "response_path": str(output_root / "responses" / (task["probe_id"] + ".json")),
            "decoder_details": decoder_details,
        })
        if task["kind"] == "exact_replay" and not exact:
            break
    counterfactual_pass = replay_pass and len(results) == len(PROBE_IDS)
    if replay_pass and len(results) == len(PROBE_IDS):
        # Controlled swap classifications use exact byte identity first.  The
        # named probes isolate one component; CF2/CF5 remain supporting
        # combined-vision characterizations and are not silently promoted to
        # single-modality claims.
        by_id = {item["probe_id"]: item for item in results}
        sensitivity = {}
        for modality, probe_id in (("proprioception", "CF1"), ("body_vision", "CF3"), ("wrist_vision", "CF4")):
            item = by_id[probe_id]
            changed = item["raw_action_sha256"] != by_id["REPLAY-S0"]["raw_action_sha256"] or item["decoded_action_sha256"] != by_id["REPLAY-S0"]["decoded_action_sha256"]
            sensitivity[modality] = {
                "probe_id": probe_id,
                "classification": "SENSITIVITY_DETECTED" if changed else "NO_DETECTABLE_SENSITIVITY_IN_THIS_PROBE",
                "output_bytes_changed": bool(changed),
                "raw_max_abs_delta": item["raw_delta_vs_baseline"]["max_abs"],
                "decoded_max_abs_delta": item["decoded_delta_vs_baseline"]["max_abs"],
                "raw_l2": item["raw_delta_vs_baseline"]["l2"],
                "decoded_l2": item["decoded_delta_vs_baseline"]["l2"],
            }
    else:
        sensitivity = {}
    result = {
        "schema": "CORELAB_STAGE42_PROBE_RESULT_V1",
        "status": "PASS" if counterfactual_pass else "BLOCKED",
        "pi_gate": evidence.PI_GATE,
        "bundle_sha256": bundle_sha,
        "seed": evidence.DEV_SEED,
        "language": LANGUAGE,
        "tasks": results,
        "replay_pass": bool(replay_pass and len(results) >= 3 and all(item.get("exact_replay_hash_match") is True for item in results[:3])),
        "replay_hashes_exact": bool(replay_pass and len(results) >= 3 and all(item.get("exact_replay_hash_match") is True for item in results[:3])),
        "counterfactual_pass": bool(counterfactual_pass),
        "sensitivity": sensitivity,
        "counters": {
            "gr00t_forward_count": len(results),
            "replay_forward_count": min(len(results), 3),
            "counterfactual_forward_count": max(0, len(results) - 3),
            "wbc_step_count": 0,
            "reference_update_count": 0,
        },
        "runtime": {"python": platform.python_version(), "device": str(args.device), "model_path": str(args.model_path)},
        "forbidden_operations": evidence.expected_forbidden_operations(),
    }
    evidence.write_canonical_json(output_root / "probe_result.json", result)
    evidence.write_canonical_json(output_root / "runtime_metadata.json", {
        "schema": "CORELAB_STAGE42_PROBE_RUNTIME_METADATA_V1",
        "python": platform.python_version(),
        "device": str(args.device),
        "model_path": str(args.model_path),
        "base_model_path": str(args.base_model_path),
        "stage10_root": str(args.stage10_root),
        "structured_stats": str(args.structured_stats),
        "seed": evidence.DEV_SEED,
        "simulation_only": True,
        "hardware_control": False,
        "wbc_execution": False,
        "reference_update": False,
        "forbidden_operations": evidence.expected_forbidden_operations(),
    })
    print(json.dumps({"status": result["status"], "output_dir": str(output_root), "forwards": len(results), "replay_pass": result["replay_pass"], "counterfactual_pass": result["counterfactual_pass"]}, sort_keys=True))
    return result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--cosmos-snapshot", type=Path, required=True)
    parser.add_argument("--structured-stats", type=Path, required=True)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.model_python is not None:
        _require(Path(sys.executable).resolve() == Path(args.model_python).resolve(), "runner must execute under the supplied model Python")
    for path in (args.bundle, args.stage10_root, args.base_model_path, args.model_path, args.dataset_dir, args.modality_config, args.cosmos_snapshot, args.structured_stats):
        if not Path(path).exists():
            parser.error("missing required path: %s" % path)
    result = run(args)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
