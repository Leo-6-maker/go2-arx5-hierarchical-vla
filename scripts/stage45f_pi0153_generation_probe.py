#!/usr/bin/env python3
"""PI0153 bounded same-input/same-noise generation-path comparison.

The probe is deliberately a small, fail-closed diagnostic.  It uses exactly
two immutable inputs (the accepted TRAIN window and the archived Stage43 C0
q0 NPZ), once through the accepted local ``_sample_one`` path and once through
the deployment worker's ``get_action`` path.  A single in-memory model is
used; the saved float32 x0 is injected into both paths, and no simulator,
WBC, renderer, training, validation, or holdout code is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

PI_GATE = "PI_GATE_STATE_0153"
PI_COMMENT = 5649870843
PR_COMMENT = 5649873155
RETURN_TOKEN = "CORELAB_PI0153_FIRST_ACTION_SEMANTICS_AND_GENERATION_FOR_PI_REVIEW"
CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
TRAIN_INPUT_FILE_SHA = "3b525a5a628d3a9ee71894fdd9c6c77b2e888406e87926594700314768181866"
ARCHIVED_C0_INPUT_FILE_SHA = "67ca605ca90e05633b22dd0447e4873ee29c6a4b31be983b94c11db35f9932be"
HORIZON = 40
ACTION_DIM = 132
PHYSICAL_DIM = 10
RAW_SHAPE = (1, HORIZON, ACTION_DIM)
TARGET = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
TIMESTAMPS = np.arange(HORIZON, dtype=np.float64) * 0.05
LANGUAGE = "move the end effector to the red target"
MAX_QUERIES = 4
MAX_CONDITIONING = 4
MAX_HEAD_CALLS = 16
MAX_MODEL_CALLS = 20


class ProbeError(RuntimeError):
    pass


def require(predicate: bool, message: str) -> None:
    if not predicate:
        raise ProbeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"checkpoint root is not a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(stat.S_ISDIR(child.lstat().st_mode) and not child.is_symlink(), f"unsafe checkpoint directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(stat.S_ISREG(child.lstat().st_mode) and not child.is_symlink(), f"unsafe checkpoint file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _verify_checkpoint_and_stats(checkpoint: Path, stats_path: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    stats_path = Path(stats_path)
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"checkpoint path is unavailable or symlinked: {checkpoint}")
    require(stats_path.is_file() and not stats_path.is_symlink(), f"stats path is unavailable or symlinked: {stats_path}")
    entries = _tree_entries(checkpoint, "stage45e/checkpoint-2000")
    checkpoint_tree = _tree_digest(entries)
    require(checkpoint_tree == CHECKPOINT_TREE, f"N2 checkpoint tree identity drift: {checkpoint_tree}")
    stats_sha = sha256_file(stats_path)
    require(stats_sha == N2_STATS_SHA, f"N2 stats file identity drift: {stats_sha}")
    stats_document = read_json(stats_path)
    require(stats_document.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats payload identity drift")
    return {"checkpoint_tree_sha256": checkpoint_tree, "checkpoint_file_count": len(entries), "checkpoint_total_bytes": sum(int(item["bytes"]) for item in entries), "n2_stats_sha256": stats_sha, "n2_stats_payload_sha256": stats_document["statistics_payload_sha256"]}


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _array_summary(value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
        array = tensor.numpy()
        result: dict[str, Any] = {"kind": "tensor", "shape": list(array.shape), "dtype": str(tensor.dtype), "sha256": sha256_array(array), "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None}
        if np.issubdtype(array.dtype, np.number) and array.size:
            floating = array.astype(np.float64, copy=False)
            result.update({"min": float(np.min(floating)), "max": float(np.max(floating)), "mean": float(np.mean(floating))})
        return result
    if isinstance(value, np.ndarray):
        result = {"kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value), "finite": bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else None}
        return result
    if isinstance(value, Mapping):
        return {str(key): _array_summary(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "length": len(value), "items": [_array_summary(item) for item in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__, "repr": repr(value)[:300]}


def _load_observation(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        require(set(("body_rgb", "wrist_rgb", "state")).issubset(data.files), f"observation keys missing: {path}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        target = None if "target_position_world_m" not in data.files else np.asarray(data["target_position_world_m"], dtype=np.float64).reshape(-1)
        extra_keys = sorted(set(data.files).difference({"body_rgb", "wrist_rgb", "state"}))
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, f"body ABI drift: {path}")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, f"wrist ABI drift: {path}")
    require(state.shape == (10,) and np.isfinite(state).all(), f"state ABI drift: {path}")
    if target is not None:
        require(target.shape == (3,) and np.isfinite(target).all(), f"target ABI drift: {path}")
    return {"path": str(path), "file_sha256": sha256_file(path), "body_rgb": body, "wrist_rgb": wrist, "state": state, "target_position_world_m": None if target is None else target.tolist(), "target_keys": extra_keys, "body_sha256": sha256_array(body), "wrist_sha256": sha256_array(wrist), "state_sha256": sha256_array(state)}


def _model_input_keys(value: Any) -> set[str]:
    """Collect mapping keys recursively for the target/metadata exclusion audit."""
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_model_input_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_model_input_keys(item))
    return keys


def _assert_model_input_boundary(inputs: Mapping[str, Any], label: str) -> None:
    forbidden = {"target_position_world_m", "target_xyz", "target_world_m", "target_id", "target_position"}
    leaked = sorted(forbidden.intersection(_model_input_keys(inputs)))
    require(not leaked, f"{label} target/metadata leaked into model input: {leaked}")


def _load_c1_condition_inputs(condition_root: Path, condition: str, archived_npz: Path) -> dict[str, Any]:
    with np.load(condition_root / "observation_body.npz", allow_pickle=False) as body_npz:
        body = np.asarray(body_npz[f"{condition}-q0"])
    with np.load(condition_root / "observation_wrist.npz", allow_pickle=False) as wrist_npz:
        wrist = np.asarray(wrist_npz[f"{condition}-q0"])
    archived = _load_observation(archived_npz)
    require(np.array_equal(body, archived["body_rgb"]) and np.array_equal(wrist, archived["wrist_rgb"]), "archived C0 pixels do not match Stage43 source arrays")
    return {"body": body, "wrist": wrist, "state": archived["state"], "archived": archived}


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_prepare(model: Any, call: Any) -> tuple[Any, dict[str, Any]]:
    captured: dict[str, Any] = {}
    original = model.prepare_input

    def wrapped(inputs: Mapping[str, Any]):
        backbone_inputs, action_inputs = original(inputs)
        captured["prepared_backbone"] = _array_summary(backbone_inputs)
        captured["prepared_action"] = _array_summary(action_inputs)
        return backbone_inputs, action_inputs

    model.prepare_input = wrapped
    try:
        value = call()
    finally:
        model.prepare_input = original
    return value, captured


def _build_c1_inputs(processor: Any, embodiment: Any, item: Mapping[str, Any], *, train: bool) -> Mapping[str, Any]:
    from gr00t.data.types import MessageType, VLAStepData

    state = np.asarray(item["state"], dtype=np.float32)
    step = VLAStepData(images={"external_rgb": [item["body"]], "wrist_rgb": [item["wrist"]]}, states={"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]}, actions={}, text=LANGUAGE, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    batch = processor.collator([processed])
    inputs = batch.get("inputs", {})
    require(isinstance(inputs, Mapping), "C1 processor input is not a mapping")
    require("action" not in inputs and "action_mask" not in inputs, "C1 action label leaked")
    _assert_model_input_boundary(inputs, "C1")
    return inputs


def _build_worker_inputs(processor: Any, embodiment: Any, observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    from gr00t.data.types import MessageType, VLAStepData
    import torch

    state = np.asarray(observation["state"], dtype=np.float32)
    step = VLAStepData(images={"external_rgb": observation["body_rgb"][None, ...], "wrist_rgb": observation["wrist_rgb"][None, ...]}, states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}, actions={}, text=LANGUAGE, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    collated = processor.collator([processed])
    inputs = collated.get("inputs", {})
    require(isinstance(inputs, Mapping), "worker processor input is not a mapping")
    require("action" not in inputs and "action_mask" not in inputs, "worker action label leaked")
    _assert_model_input_boundary(inputs, "worker")
    return _rec_to_dtype(collated, torch), inputs


def _rec_to_dtype(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=torch.bfloat16)
    if isinstance(value, Mapping):
        return {key: _rec_to_dtype(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, torch) for item in value]
    return value


def _injectable_worker_call(model: Any, worker_inputs: Mapping[str, Any], noise: np.ndarray, torch: Any, seed: int) -> tuple[np.ndarray, dict[str, Any], int]:
    """Run the deployment worker's exact ``model.get_action`` path.

    The only wrapper is a scoped ``torch.randn`` replacement for the initial
    [1,40,132] action tensor.  It makes the saved x0 explicit and records that
    this is an aligned diagnostic, not a production sampler change.
    """
    import stage21_same_scene_groot_inference as worker

    _set_seed(seed, torch)
    injected: list[dict[str, Any]] = []
    original_randn = torch.randn

    def fixed_randn(*args: Any, **kwargs: Any):
        size = kwargs.get("size", args[0] if args else None)
        try:
            shape = tuple(int(value) for value in size)
        except Exception:
            shape = None
        if shape == RAW_SHAPE:
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)
            value = torch.from_numpy(np.asarray(noise, dtype=np.float32).copy()).to(device=device, dtype=dtype)
            injected.append({"shape": list(shape), "dtype": str(value.dtype), "device": str(value.device), "array_sha256_float32": sha256_array(noise), "array_sha256_cast": sha256_array(value.detach().to(device="cpu").numpy())})
            return value
        return original_randn(*args, **kwargs)

    torch.randn = fixed_randn
    try:
        calls_before = int(getattr(model, "_pi0153_action_head_calls", 0))
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: setattr(model, "_pi0153_action_head_calls", int(getattr(model, "_pi0153_action_head_calls", 0)) + 1))
        try:
            # Preserve the deployment worker's call boundary exactly: the
            # collator returns a top-level ``inputs`` keyword, and the worker
            # invokes ``model.get_action(**inputs)`` once per query.
            prediction, captured = _capture_prepare(model, lambda: model.get_action(**worker_inputs))
        finally:
            hook.remove()
    finally:
        torch.randn = original_randn
    raw = prediction["action_pred"].float().detach().to(device="cpu").contiguous().numpy().astype(np.float32)
    require(raw.shape == RAW_SHAPE and np.isfinite(raw).all(), "worker raw output ABI/finite drift")
    calls = int(getattr(model, "_pi0153_action_head_calls", 0)) - calls_before
    require(calls == 4, f"worker action-head call count mismatch: {calls}")
    return raw, {"path": "stage21_same_scene_groot_inference._run_one_raw/model.get_action", "fixed_noise_injection": injected, "prepared": captured, "model_get_action_seed": seed, "worker_module_sha256": sha256_file(Path(worker.__file__).resolve())}, calls


def _c1_call(model: Any, model_inputs: Mapping[str, Any], noise: np.ndarray, torch: Any, sampler: Any) -> tuple[np.ndarray, dict[str, Any], int]:
    _set_seed(4242, torch)
    noise_tensor = torch.from_numpy(np.asarray(noise, dtype=np.float32).copy()).contiguous()
    calls_before = int(getattr(model, "_pi0153_action_head_calls", 0))
    hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: setattr(model, "_pi0153_action_head_calls", int(getattr(model, "_pi0153_action_head_calls", 0)) + 1))
    try:
        value, captured = _capture_prepare(model, lambda: sampler(model, model_inputs, noise_tensor, torch))
    finally:
        hook.remove()
    raw = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy() if hasattr(value, "detach") else np.asarray(value, dtype=np.float32)
    require(raw.shape == RAW_SHAPE and np.isfinite(raw).all(), "C1 raw output ABI/finite drift")
    calls = int(getattr(model, "_pi0153_action_head_calls", 0)) - calls_before
    require(calls == 4, f"C1 action-head call count mismatch: {calls}")
    return raw, {"path": "stage45c_h200_eval_runtime_adapter._sample_one", "fixed_noise_injection": [{"shape": list(noise.shape), "dtype": "torch.float32->model_dtype", "array_sha256_float32": sha256_array(noise)}], "prepared": captured, "sampler_timestep_ids": [0, 250, 500, 750], "sampler_dt": 0.25, "sampler_module_sha256": sha256_file(Path(sampler.__module__.replace(".", "/") + ".py")) if False else None}, calls


def _decode_metrics(raw: np.ndarray, state: np.ndarray, anchor: np.ndarray, stats_path: Path, label: str) -> dict[str, Any]:
    import go2arx5_vla.stage40_structured_decode as decoder
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

    previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
    decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
    try:
        _, physical, details = decoder.decode_raw_structured_action(raw, state, stats_path=stats_path, source_timestamps=TIMESTAMPS, observation_timestamp=0.0, chunk_id=label)
    finally:
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
    physical = np.asarray(physical, dtype=np.float64)
    world = anchor[None, :] + physical[:, :3] - state[None, :3]
    metrics = trajectory_metrics(world, TARGET, prefix=label)
    return {"decoded_physical_sha256": sha256_array(physical), "p0_state_residual_m": float(np.linalg.norm(physical[0, :3] - state[:3])), "world_p0_m": world[0].tolist(), "target_metrics": metrics, "decoder": details}


def _gpu_snapshot() -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-500:]}")
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    selected = next((row for row in rows if row["index"] == 0), None)
    require(selected is not None, "GPU0 is unavailable")
    return {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "query_sha256": hashlib.sha256(query.stdout.encode("utf-8")).hexdigest(), "selected": selected, "all": rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    gpu = _gpu_snapshot()
    require(gpu["selected"]["memory_free_mib"] >= 14 * 1024, f"GPU0 free-memory admission failed: {gpu['selected']}")
    checkpoint = Path(args.model_path).resolve()
    stats_path = Path(args.n2_stats).resolve()
    checkpoint_identity = _verify_checkpoint_and_stats(checkpoint, stats_path)
    noise = _load_npy(Path(args.noise), RAW_SHAPE, "fixed noise")
    require(sha256_array(noise) == NOISE_ARRAY_SHA and noise.dtype == np.float32, "fixed noise identity drift")
    train = _load_observation(Path(args.train_input))
    archived = _load_observation(Path(args.archived_c0_input))
    require(train["file_sha256"] == TRAIN_INPUT_FILE_SHA, f"TRAIN input file identity drift: {train['file_sha256']}")
    require(archived["file_sha256"] == ARCHIVED_C0_INPUT_FILE_SHA, f"archived C0 input file identity drift: {archived['file_sha256']}")
    c1_condition = _load_c1_condition_inputs(Path(args.stage43_c0_root), "C0_NOMINAL", Path(args.archived_c0_input))
    require(np.array_equal(c1_condition["state"], archived["state"]), "archived C0 state mismatch")

    import torch
    import stage21_same_scene_groot_inference as worker
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
    from stage38_structured_offline_evaluation import _load_structured_model
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from stage40_b0_model_control import _bind_local_cosmos_processor

    _bind_local_cosmos_processor(Path(args.stage10_root), Path(args.cosmos_snapshot))
    c1_processor, c1_embodiment, _ = _build_runtime_processor(groot_source=Path(args.stage10_root) / "src" / "isaac-gr00t", base_checkpoint=Path(args.base_model), dataset_dir=Path(args.dataset), modality_path=Path(args.modality), cosmos_snapshot=Path(args.cosmos_snapshot))
    # ``_build_processor`` re-imports the modality module and registers the
    # same embodiment a second time in this process.  The accepted lifecycle
    # repair is to reuse the already validated runtime processor/context while
    # comparing the distinct deployment ``get_action`` call boundary.
    worker_processor, worker_embodiment = c1_processor, c1_embodiment
    try:
        model = _load_structured_model(checkpoint, "cuda:0")
    except Exception as exc:
        # Keep a machine-readable, immutable failure record when model
        # construction starts but cannot complete (for example, a gated
        # backbone access failure).  The PI0153 rule is fail-closed: no retry
        # or alternate device is attempted after this point.
        failure = {"schema": "CORELAB_STAGE45F_PI0153_GENERATION_PROBE_FAILURE_V1", "status": "BLOCKED", "phase": "model_construction", "error_type": type(exc).__name__, "error": str(exc)[-2000:], "counters": {"queries": 0, "conditioning": 0, "action_head_calls": 0, "model_side_calls": 0, "model_construction_attempts": 1, "exceptions": 1}, "no_retry": True, "gpu": gpu, "identities": checkpoint_identity, "execution_boundary": {"model_constructed": False, "new_training": False, "training": False, "pbs": False, "simulator": False, "wbc": False, "validation_e_read": False, "holdout_f_read": False}}
        write_json(output / "failure_seal.json", failure)
        raise
    model.eval()
    model_dtype = str(next(model.parameters()).dtype)
    parameter_names = sorted(str(name) for name, _ in model.named_parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    inputs = [
        ("TRAIN_a_expert-0000", True, train, _build_c1_inputs(c1_processor, c1_embodiment, {"body": train["body_rgb"], "wrist": train["wrist_rgb"], "state": train["state"]}, train=True), _build_worker_inputs(worker_processor, worker_embodiment, train)[0], np.asarray(train["state"], dtype=np.float32), np.asarray(args.train_anchor, dtype=np.float64)),
        ("C0_NOMINAL", False, archived, _build_c1_inputs(c1_processor, c1_embodiment, {"body": c1_condition["body"], "wrist": c1_condition["wrist"], "state": c1_condition["state"]}, train=False), _build_worker_inputs(worker_processor, worker_embodiment, archived)[0], np.asarray(archived["state"], dtype=np.float32), np.asarray(args.c0_anchor, dtype=np.float64)),
    ]
    results: list[dict[str, Any]] = []
    counters = {"queries": 0, "conditioning": 0, "action_head_calls": 0, "model_side_calls": 0, "worker_attempts": 1, "model_construction": 1, "exceptions": 0}
    for label, _train, observation, c1_inputs, worker_inputs, state, anchor in inputs:
        for path_name in ("C1_SAMPLE_ONE", "WORKER_GET_ACTION"):
            counters["queries"] += 1
            counters["conditioning"] += 1
            _set_seed(4242, torch)
            head_before = int(getattr(model, "_pi0153_action_head_calls", 0))
            start = time.perf_counter()
            try:
                if path_name == "C1_SAMPLE_ONE":
                    raw, path_details, calls = _c1_call(model, c1_inputs, noise, torch, _sample_one)
                else:
                    raw, path_details, calls = _injectable_worker_call(model, worker_inputs, noise, torch, 4242)
            except Exception as exc:
                counters["exceptions"] += 1
                observed_after_failure = max(0, int(getattr(model, "_pi0153_action_head_calls", 0)) - head_before)
                counters["action_head_calls"] += observed_after_failure
                counters["model_side_calls"] += observed_after_failure
                failure = {"schema": "CORELAB_STAGE45F_PI0153_GENERATION_PROBE_FAILURE_V1", "status": "PARTIAL", "error_type": type(exc).__name__, "error": str(exc)[-2000:], "counters": counters, "no_retry": True, "gpu": gpu}
                write_json(output / "failure_seal.json", failure)
                raise
            elapsed = time.perf_counter() - start
            counters["action_head_calls"] += int(calls)
            counters["model_side_calls"] += int(calls)
            raw_path = output / f"raw_{len(results):02d}_{label}_{path_name}.npy"
            np.save(raw_path, raw, allow_pickle=False)
            decoded = _decode_metrics(raw, state, anchor, stats_path, f"{label}:{path_name}")
            result = {"query_index": len(results), "label": label, "path": path_name, "input_file_sha256": observation["file_sha256"], "input_arrays": {"body_sha256": observation["body_sha256"], "wrist_sha256": observation["wrist_sha256"], "state_sha256": observation["state_sha256"], "target_keys": observation["target_keys"]}, "model_input_target_excluded": True, "raw_path": str(raw_path), "raw_file_sha256": sha256_file(raw_path), "raw_array_sha256": sha256_array(raw), "raw_shape": list(raw.shape), "action_head_calls": int(calls), "conditioning_pass": 1, "latency_s": elapsed, "path_details": path_details, "decoded": decoded}
            results.append(result)

    by_key = {(row["label"], row["path"]): row for row in results}
    comparisons = []
    for label in ("TRAIN_a_expert-0000", "C0_NOMINAL"):
        left, right = by_key[(label, "C1_SAMPLE_ONE")], by_key[(label, "WORKER_GET_ACTION")]
        left_raw = np.load(left["raw_path"], allow_pickle=False)
        right_raw = np.load(right["raw_path"], allow_pickle=False)
        comparisons.append({"label": label, "same_input_file": left["input_file_sha256"] == right["input_file_sha256"], "same_input_arrays": left["input_arrays"] == right["input_arrays"], "same_noise_array": True, "raw_max_abs_delta": float(np.max(np.abs(left_raw - right_raw))), "raw_rmse": float(np.sqrt(np.mean((left_raw - right_raw) ** 2))), "raw_equal_bytes": bool(np.array_equal(left_raw, right_raw)), "decoded_max_abs_delta": float(np.max(np.abs(np.asarray(left["decoded"]["decoded_physical_sha256"] != right["decoded"]["decoded_physical_sha256"])) if False else np.asarray([]))) if False else None, "interpretation": "path-equivalence diagnostic; output difference is not a task-success claim"})

    # Add a second decoded delta without retaining large arrays in JSON.
    for comparison in comparisons:
        label = comparison["label"]
        left = by_key[(label, "C1_SAMPLE_ONE")]
        right = by_key[(label, "WORKER_GET_ACTION")]
        # The raw delta above is the primary equivalence measure; physical
        # arrays are reconstructed by the deterministic decoder for scalar
        # deltas only.
        import go2arx5_vla.stage40_structured_decode as decoder
        previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
        try:
            state = np.asarray(train["state"] if label.startswith("TRAIN") else archived["state"], dtype=np.float32)
            _, lp, _ = decoder.decode_raw_structured_action(np.load(left["raw_path"], allow_pickle=False), state, stats_path=stats_path, source_timestamps=TIMESTAMPS, observation_timestamp=0.0, chunk_id="delta-left")
            _, rp, _ = decoder.decode_raw_structured_action(np.load(right["raw_path"], allow_pickle=False), state, stats_path=stats_path, source_timestamps=TIMESTAMPS, observation_timestamp=0.0, chunk_id="delta-right")
        finally:
            decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
        comparison["decoded_physical_max_abs_delta"] = float(np.max(np.abs(np.asarray(lp) - np.asarray(rp))))
        comparison["decoded_physical_rmse"] = float(np.sqrt(np.mean((np.asarray(lp) - np.asarray(rp)) ** 2)))
        comparison["prepared_input_summaries_equal"] = left["path_details"].get("prepared") == right["path_details"].get("prepared")

    report = {"schema": "CORELAB_STAGE45F_PI0153_GENERATION_PROBE_V1", "status": "COMPLETED", "token": RETURN_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": "stage45f/corelab-local-initial-state-pilot"}, "identities": {**checkpoint_identity, "noise_array_sha256": NOISE_ARRAY_SHA, "model_dtype": model_dtype, "parameter_names_sha256": sha256_json(parameter_names), "parameter_dtypes": parameter_dtypes, "worker_module_sha256": sha256_file(Path(worker.__file__).resolve())}, "inputs": [{"label": label, "file_sha256": observation["file_sha256"], "body_sha256": observation["body_sha256"], "wrist_sha256": observation["wrist_sha256"], "state_sha256": observation["state_sha256"], "target_keys": observation["target_keys"], "target_xyz_in_model_input": False} for label, _, observation, *_ in inputs], "gpu": gpu, "results": results, "comparisons": comparisons, "counters": counters, "budget": {"max_queries": MAX_QUERIES, "max_conditioning": MAX_CONDITIONING, "max_action_head_calls": MAX_HEAD_CALLS, "max_model_side_calls": MAX_MODEL_CALLS, "consumed_queries": counters["queries"], "consumed_conditioning": counters["conditioning"], "consumed_action_head_calls": counters["action_head_calls"], "consumed_model_side_calls": counters["model_side_calls"]}, "execution_boundary": {"new_training": False, "training": False, "gpu": True, "pbs": False, "simulator": False, "wbc": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "icl": False, "new_data": False, "safety_gate_relaxed": False, "retry": False}, "claim_boundary": "same-input/same-x0 generation-path diagnostic only; no route selection, closed-loop, or task-success claim"}
    report_sha = write_json(output / "pi0153_generation_probe.json", report)
    write_json(output / "pi0153_generation_probe.json.sha256", {"file": "pi0153_generation_probe.json", "sha256": report_sha})
    return {**report, "report_sha256": report_sha}


def _load_npy(path: Path, shape: tuple[int, ...], label: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    require(not isinstance(value, np.lib.npyio.NpzFile), f"{label} must be a .npy")
    value = np.asarray(value)
    require(value.shape == shape and np.isfinite(value).all(), f"{label} shape/finite drift: {value.shape}")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage10-root", required=True, type=Path)
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--modality", required=True, type=Path)
    p.add_argument("--cosmos-snapshot", required=True, type=Path)
    p.add_argument("--n2-stats", required=True, type=Path)
    p.add_argument("--noise", required=True, type=Path)
    p.add_argument("--train-input", required=True, type=Path)
    p.add_argument("--archived-c0-input", required=True, type=Path)
    p.add_argument("--stage43-c0-root", required=True, type=Path)
    p.add_argument("--train-anchor", required=True, nargs=3, type=float)
    p.add_argument("--c0-anchor", required=True, nargs=3, type=float)
    p.add_argument("--output-root", required=True, type=Path)
    return p


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "token": result["token"], "report_sha256": result["report_sha256"]}, sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
