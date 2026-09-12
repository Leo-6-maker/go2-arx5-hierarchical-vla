#!/usr/bin/env python3
"""PI0149: one bounded FULL24@2000 Corelab engineering smoke.

The default path is a path/identity/device preflight and never imports Torch,
Transformers, GR00T, or CUDA.  ``--execute`` is deliberately late-bound: after
the preflight passes it imports the already reviewed Stage45C/Stage38 runtime,
constructs exactly one model, prepares one TRAIN window, and runs the frozen
four-step direct action-head graph.  This is an engineering qualification, not
a scientific evaluation and not a free-running robot rollout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "STAGE45F_CORELAB_PI0149_FULL24_SINGLE_WINDOW_ENGINEERING_SMOKE_V1"
PI_GATE = "PI_GATE_STATE_0149"
PI_COMMENT = 5646608846
RETURN_TOKEN = "STAGE45F_CORELAB_PI0149_FULL24_SINGLE_WINDOW_ENGINEERING_SMOKE_COMPLETE_FOR_PI_REVIEW"
BLOCKED_TOKEN = "STAGE45F_CORELAB_PI0149_RUNTIME_PREFLIGHT_BLOCKED"
ATTEMPT_FAILED_TOKEN = "STAGE45F_CORELAB_PI0149_MODEL_ATTEMPT_FAILED_STOPPED"

HOST = "corelab-4080s"
GPU_UUID = "GPU-845348f1-1511-dc19-0877-cc97658e3edd"
RUNTIME_INTERPRETER = "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/runtime/venv/bin/python"
CHECKPOINT_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
DATASET_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
BASE_INDEX = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
WINDOW_ID = "stage22_expert_red_target_train_a:expert-0000"
SNAPSHOT_SHA = "74af1af85e21c6e3a5e91e5b4b695d080f0f6b7f2204d012310d6457334106f1"
NOISE_SEED = 4242
TIMESTEPS = (0, 250, 500, 750)
RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
MAX_CONDITIONING = 1
MAX_ACTION_HEAD = 4
MIN_FREE_MIB = 14 * 1024


class SmokeError(RuntimeError):
    """A fail-closed preflight or single-attempt error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON unavailable or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, Mapping), f"JSON root is not an object: {path}")
    return value


def _run(argv: Sequence[str], *, env: Mapping[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=timeout, env=dict(env) if env is not None else None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": list(argv), "returncode": None, "stdout": "", "stderr": str(exc)}
    return {"argv": list(argv), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 7:
            continue
        try:
            rows.append({
                "index": int(values[0]), "name": values[1], "uuid": values[2],
                "memory_total_mib": int(float(values[3])), "memory_used_mib": int(float(values[4])),
                "memory_free_mib": int(float(values[5])), "driver_version": values[6],
            })
        except ValueError:
            continue
    return rows


def _parse_process_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4:
            continue
        try:
            rows.append({"uuid": values[0], "pid": int(values[1]), "process_name": values[2], "used_memory_mib": int(float(values[3]))})
        except ValueError:
            continue
    return rows


def device_preflight() -> dict[str, Any]:
    inventory = _run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    processes = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    result = {
        "host": platform.node(),
        "inventory_command": inventory,
        "process_command": processes,
        "gpus": _parse_gpu_rows(inventory["stdout"]) if inventory["returncode"] == 0 else [],
        "processes": _parse_process_rows(processes["stdout"]) if processes["returncode"] == 0 else [],
    }
    _require(result["host"] == HOST, f"wrong host: {result['host']}")
    _require(inventory["returncode"] == 0 and processes["returncode"] == 0, "nvidia-smi query failed")
    selected = [row for row in result["gpus"] if row.get("uuid") == GPU_UUID]
    _require(len(selected) == 1, "selected GPU UUID was not observed exactly once")
    _require(selected[0]["memory_free_mib"] >= MIN_FREE_MIB, f"selected GPU has insufficient free VRAM: {selected[0]['memory_free_mib']} MiB")
    busy = [row for row in result["processes"] if row.get("uuid") == GPU_UUID]
    _require(not busy, f"selected GPU has compute process(es): {busy}")
    result["selected"] = selected[0]
    result["selected_processes"] = busy
    return result


def static_guard() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top.intersection({"torch", "transformers", "gr00t", "cuda", "numpy"}))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train", "qsub"}:
                forbidden_calls.append(name)
    return {"top_level_forbidden_imports": forbidden_imports, "forbidden_call_hits": sorted(set(forbidden_calls)), "status": "PASS" if not forbidden_imports and not forbidden_calls else "FAIL"}


def _binding_paths(binding: Mapping[str, Any]) -> dict[str, Path]:
    paths = binding.get("paths")
    _require(isinstance(paths, Mapping), "binding.paths must be an object")
    required = ("base_checkpoint", "full24_2000_checkpoint", "full24_dataset", "snapshot_root", "groot_source", "cosmos_root", "modality_config", "full24_runtime_statistics", "output_root")
    result: dict[str, Path] = {}
    for key in required:
        value = paths.get(key)
        _require(isinstance(value, str) and value, f"missing binding path: {key}")
        result[key] = Path(value).expanduser()
    return result


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(binding.get("schema") == "STAGE45F_CORELAB_PI0149_RUNTIME_BINDING_V1", "binding schema drift")
    _require(binding.get("pi_gate") == PI_GATE and binding.get("issue_11_comment") == PI_COMMENT, "PI authority drift")
    _require(binding.get("gpu_uuid") == GPU_UUID and binding.get("host") == HOST, "device binding drift")
    _require(binding.get("interpreter") == RUNTIME_INTERPRETER, "runtime interpreter drift")
    execution = binding.get("execution")
    _require(isinstance(execution, Mapping), "execution contract missing")
    _require(execution.get("conditioning_passes") == 1 and execution.get("action_head_calls") == 4, "model call budget drift")
    _require(execution.get("target_xyz_in_model_input") is False and execution.get("public_sampler_calls") == 0, "input/sampler boundary drift")
    noise = binding.get("noise")
    _require(isinstance(noise, Mapping) and noise.get("seed") == NOISE_SEED and noise.get("status") == "NEW_ENGINEERING_PROBE_NOISE", "noise binding is not the declared engineering probe")
    paths = _binding_paths(binding)
    return {"binding": binding, "paths": paths}


def preflight(binding: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_binding(binding)
    paths = checked["paths"]
    static = static_guard()
    _require(static["status"] == "PASS", "wrapper static guard failed")
    for key, path in paths.items():
        if key == "output_root":
            _require(not path.exists(), f"output_root is not fresh: {path}")
        elif key in {"base_checkpoint", "full24_2000_checkpoint", "full24_dataset", "snapshot_root", "groot_source", "cosmos_root"}:
            _require(path.is_dir() and not path.is_symlink(), f"required directory unavailable: {key}={path}")
        else:
            _require(path.is_file() and not path.is_symlink(), f"required file unavailable: {key}={path}")
    import stage45c_h200_eval_runtime_adapter as frozen
    import stage26_h1_c1_fixed_noise_local_flow as stage26

    checkpoint_entries = frozen.tree_entries(paths["full24_2000_checkpoint"], "checkpoint-2000")
    checkpoint_tree = frozen.tree_digest(checkpoint_entries)
    _require(checkpoint_tree == CHECKPOINT_TREE, f"FULL24@2000 tree mismatch: {checkpoint_tree}")
    base_identity = stage26._checkpoint_preflight("BASE", paths["base_checkpoint"])
    _require(base_identity.get("tree_sha256") == BASE_TREE and base_identity.get("index_sha256") == BASE_INDEX, "BASE identity mismatch")
    dataset_identity = frozen._verify_dataset(paths["full24_dataset"])
    _require(dataset_identity.get("tree_sha256") == DATASET_TREE, "FULL24 TRAIN tree mismatch")
    stats_identity = frozen._verify_runtime_stats(paths["full24_runtime_statistics"])
    _require(stats_identity.get("sha256") == STATS_SHA and stats_identity.get("payload_sha256") == STATS_PAYLOAD_SHA, "runtime-V2 stats identity mismatch")
    _require(_sha256(paths["modality_config"]) == MODALITY_SHA, "modality SHA mismatch")
    snapshot_records = frozen._verify_snapshots(paths["snapshot_root"])
    first = snapshot_records[0]
    _require(first["window_id"] == WINDOW_ID and first["sha256"] == SNAPSHOT_SHA, "selected snapshot identity mismatch")
    head = _run(["git", "-C", str(paths["groot_source"]), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(paths["groot_source"]), "status", "--porcelain=v1"])
    _require(head["returncode"] == 0 and head["stdout"].strip() == GROOT_COMMIT, "GR00T source commit mismatch")
    _require(status["returncode"] == 0 and not status["stdout"].strip(), "GR00T source checkout is dirty")
    cosmos_tree, cosmos_entries = frozen.cosmos_tree_digest(paths["cosmos_root"])
    _require(cosmos_tree == COSMOS_TREE and (paths["cosmos_root"] / "refs/main").read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos identity mismatch")
    device = device_preflight()
    return {
        "status": "PASS",
        "static_guard": static,
        "device": device,
        "runtime": {"interpreter": sys.executable, "python": sys.version.split()[0]},
        "checkpoint": {"tree_sha256": checkpoint_tree, "regular_file_count": len(checkpoint_entries), "total_bytes": sum(int(x["bytes"]) for x in checkpoint_entries)},
        "base": {"tree_sha256": base_identity["tree_sha256"], "index_sha256": base_identity["index_sha256"]},
        "dataset": dataset_identity,
        "stats": stats_identity,
        "snapshot": {"window_id": first["window_id"], "sha256": first["sha256"], "count_verified": len(snapshot_records)},
        "groot": {"commit": head["stdout"].strip(), "clean": True},
        "cosmos": {"tree_sha256": cosmos_tree, "entry_count": cosmos_entries, "revision": COSMOS_REVISION},
        "model_calls": {"conditioning": 0, "action_head": 0},
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def execute(binding: Mapping[str, Any]) -> dict[str, Any]:
    pre = preflight(binding)
    paths = _binding_paths(binding)
    output = paths["output_root"]
    output.mkdir(parents=True, exist_ok=False)
    attempt_started = time.time()
    counters: dict[str, Any] = {"conditioning": 0, "action_head": 0}
    _write_json(output / "attempt_started.json", {"schema": "STAGE45F_PI0149_ATTEMPT_V1", "started_unix": attempt_started, "host": HOST, "gpu_uuid": GPU_UUID, "model_load_started": True})
    # Bind the selected UUID before importing torch.  The process must expose
    # exactly one logical CUDA device; no fallback or post-import rebinding.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": GPU_UUID})
    model = None
    hook = None
    try:
        import torch
        import numpy as np
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        import stage38_structured_offline_evaluation as stage38
        import stage45c_h200_eval_runtime_adapter as frozen

        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "bound CUDA device is not available")
        _require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080 SUPER", "bound GPU model drift")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(NOISE_SEED)
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
        _require(bool(torch.isfinite(noise).all().item()), "engineering noise is nonfinite")
        noise_sha = hashlib.sha256(noise.numpy().tobytes()).hexdigest()
        _write_json(output / "noise_manifest.json", {"schema": "STAGE45F_PI0149_NEW_ENGINEERING_PROBE_NOISE_V1", "status": "NEW_ENGINEERING_PROBE_NOISE", "seed": NOISE_SEED, "shape": list(RAW_SHAPE), "sha256": noise_sha, "reference_equivalence": "NOT_CLAIMED"})

        runtime_processor, embodiment, modality_configs = stage26._build_runtime_processor(
            groot_source=paths["groot_source"], base_checkpoint=paths["base_checkpoint"], dataset_dir=paths["full24_dataset"], modality_path=paths["modality_config"], cosmos_snapshot=paths["cosmos_root"] / "snapshots" / COSMOS_REVISION,
        )
        windows = frozen._prepare_train24_windows(runtime_processor, embodiment, modality_configs, paths["full24_dataset"], [frozen._verify_snapshots(paths["snapshot_root"])[0]])
        model = stage38._load_structured_model(paths["full24_2000_checkpoint"], "cuda:0")
        action_counter = {"count": 0}
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: action_counter.__setitem__("count", action_counter["count"] + 1))
        try:
            with torch.inference_mode():
                _, action_input, processed, state_features = frozen._prepare_model_conditioning(model, windows[0]["model_inputs"])
                counters["conditioning"] = 1
                current = noise.to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                for timestep in TIMESTEPS:
                    velocity = frozen._offline_step(model.action_head, processed, action_input, state_features, current, timestep)
                    current = current + 0.25 * velocity
            counters["action_head"] = int(action_counter["count"])
            _require(counters["action_head"] == MAX_ACTION_HEAD, f"action-head call count drift: {counters['action_head']}")
            raw = current.detach().to(device="cpu", dtype=torch.float32).contiguous()
            _require(tuple(int(x) for x in raw.shape) == RAW_SHAPE and bool(torch.isfinite(raw).all().item()), "raw output shape/finite contract failed")
            raw_np = raw.numpy().astype("float32", copy=False)
            raw_path = output / "raw_universal_action.npy"
            np.save(raw_path, raw_np, allow_pickle=False)
            physical, decode = frozen._decode_v2(raw_np, windows[0]["snapshot"]["state_v1"], paths["full24_runtime_statistics"], windows[0]["snapshot"]["source_timestamps"], windows[0]["snapshot"]["observation_timestamp"], WINDOW_ID)
            physical_np = np.asarray(physical, dtype="float64")
            _require(tuple(int(x) for x in physical_np.shape) == PHYSICAL_SHAPE and np.isfinite(physical_np).all(), "decoded physical ABI/finite contract failed")
            peak = {"allocated_mib": int(torch.cuda.max_memory_allocated() // (1024 * 1024)), "reserved_mib": int(torch.cuda.max_memory_reserved() // (1024 * 1024))}
            _write_json(output / "smoke_report.json", {"schema": SCHEMA, "status": "PASS", "token": RETURN_TOKEN, "gate": {"name": PI_GATE, "issue_11_comment": PI_COMMENT}, "attempt_started": True, "runtime": pre["runtime"], "device": pre["device"], "checkpoint": {"label": "FULL24@2000", "tree_sha256": CHECKPOINT_TREE}, "window_id": WINDOW_ID, "model_input_keys": ["body_rgb", "wrist_rgb", "state"], "target_xyz_in_model_input": False, "sampler": {"name": "source-mapped-four-step-Euler-action-head-graph", "timestep_ids": list(TIMESTEPS), "dt": 0.25, "public_sampler_calls": 0}, "counters": counters, "raw": {"path": str(raw_path), "sha256": _sha256(raw_path), "shape": list(raw_np.shape), "dtype": str(raw_np.dtype)}, "decoded": {"shape": list(physical_np.shape), "finite": True, "stats_sha256": STATS_SHA, "stats_payload_sha256": STATS_PAYLOAD_SHA, "decoder": decode}, "noise": {"seed": NOISE_SEED, "sha256": noise_sha, "status": "NEW_ENGINEERING_PROBE_NOISE", "reference_equivalence": "NOT_CLAIMED"}, "telemetry": {"peak_memory_allocated_mib": peak["allocated_mib"], "peak_memory_reserved_mib": peak["reserved_mib"], "elapsed_seconds": time.time() - attempt_started}, "forbidden": {"training": 0, "backward": 0, "optimizer_step": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "toy_cuda_calls": 0}})
            return _read_json(output / "smoke_report.json")
        finally:
            if hook is not None:
                hook.remove()
            if model is not None:
                del model
            torch.cuda.empty_cache()
    except Exception as exc:
        failure = {"schema": SCHEMA, "status": "FAILED_STOPPED", "token": ATTEMPT_FAILED_TOKEN, "gate": {"name": PI_GATE, "issue_11_comment": PI_COMMENT}, "attempt_started": True, "failure": {"type": type(exc).__name__, "message": str(exc)}, "counters": counters, "model_calls_unknown": bool(model is not None and counters["conditioning"] == 0), "preflight": pre, "forbidden": {"training": 0, "backward": 0, "optimizer_step": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "toy_cuda_calls": 0}}
        _write_json(output / "failure_seal.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    binding = _read_json(args.binding)
    try:
        result = execute(binding) if args.execute else preflight(binding)
        result = {"schema": SCHEMA, "status": result.get("status", "PASS"), "result": result, "model_forward": result.get("counters", {}).get("conditioning", 0) + result.get("counters", {}).get("action_head", 0)}
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "BLOCKED_PREFLIGHT" if not args.execute else "FAILED_STOPPED", "error": {"type": type(exc).__name__, "message": str(exc)}, "model_forward": 0}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in {"PASS", "FAILED_STOPPED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
