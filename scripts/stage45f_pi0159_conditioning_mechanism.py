#!/usr/bin/env python3
"""PI0159 bounded teacher-forced/local-field conditioning diagnosis.

The runner keeps the scientific question deliberately small: one sealed N2
seed2223@2000 policy, one native TRAIN expert target, and four observation
conditions (TRAIN, state-only C0, vision-only C0, and full C0).  At three
reviewed flow times it evaluates the source-mapped teacher-forced action-head
graph.  It never calls the public/free-running sampler, WBC, a simulator, or
training.  Raw tensors are written before any derived metric is calculated.

Torch/GR00T/Transformers are imported only after identity, input-boundary,
GPU, and attempt-admission checks.  This file intentionally has no top-level
runtime-model import so that CPU/static tests can audit the gate without a
model side effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for _path in (SCRIPTS, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PI_GATE = "PI_GATE_STATE_0159"
PI_COMMENT = 5651409238
PR_COMMENT = 5651409688
RETURN_TOKEN = "CORELAB_PI0159_CONDITIONING_MECHANISM_FOR_PI_REVIEW"
ADMISSION_TOKEN = "CORELAB_PI0159_CONDITIONING_ADMISSION_RECORDED"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
STARTING_HEAD = "e14b5b587eafc8c2e50aa5da1375aa2428cbffaa"

CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
BASE_PROCESSOR_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_STATISTICS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
NOISE_FILE_SHA = "83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922"
TRAIN_INPUT_FILE_SHA = "3b525a5a628d3a9ee71894fdd9c6c77b2e888406e87926594700314768181866"
C0_INPUT_FILE_SHA = "67ca605ca90e05633b22dd0447e4873ee29c6a4b31be983b94c11db35f9932be"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
AUTHORIZED_GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"

HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
PHYSICAL_DIM = 10
RAW_SHAPE = (1, HORIZON, ACTION_DIM)
TIMESTAMPS = np.arange(HORIZON, dtype=np.float64) * 0.05
LANGUAGE = "move the end effector to the red target"
T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
T_BUCKETS = (67, 369, 783)
MAX_CONDITIONING = 4
MAX_HEAD_CALLS = 12
MAX_MODEL_SIDE_CALLS = 16
MAX_MODEL_CONSTRUCTION = 1
TRAIN_LABEL_ID = "stage22_expert_red_target_train_a:expert-0000"


class ProbeError(RuntimeError):
    """Fail-closed PI0159 predicate failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value)).tobytes())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_text(path: Path, value: str) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return sha256_file(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"checkpoint root unavailable: {root}")
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
    return sorted(entries, key=lambda item: item["path"])


def _tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _verify_checkpoint(checkpoint: Path, stats_path: Path) -> dict[str, Any]:
    require(Path(stats_path).is_file() and not Path(stats_path).is_symlink(), f"stats unavailable: {stats_path}")
    entries = _tree_entries(checkpoint, "stage45e/checkpoint-2000")
    tree = _tree_digest(entries)
    require(tree == CHECKPOINT_TREE, f"N2 checkpoint tree drift: {tree}")
    stats_sha = sha256_file(stats_path)
    require(stats_sha == N2_STATS_SHA, f"N2 stats file drift: {stats_sha}")
    stats = read_json(stats_path)
    require(stats.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats payload drift")
    selected = stats.get("statistics", {})
    require(selected.get("action_horizon") == HORIZON and selected.get("universal_action_dim") == ACTION_DIM, "N2 stats ABI drift")
    require(selected.get("active_scalar_count") == 190 and selected.get("timing_mode") == "T16", "N2 stats active contract drift")
    return {"checkpoint_tree_sha256": tree, "checkpoint_file_count": len(entries), "checkpoint_total_bytes": sum(int(e["bytes"]) for e in entries), "checkpoint_path": str(Path(checkpoint).resolve()), "stats_sha256": stats_sha, "stats_payload_sha256": stats["statistics_payload_sha256"], "stats_path": str(Path(stats_path).resolve())}


def _verify_base(base: Path) -> dict[str, Any]:
    base = Path(base).resolve()
    require(base.is_dir() and not base.is_symlink(), f"BASE processor root unavailable: {base}")
    config = base / "processor_config.json"
    statistics = base / "statistics.json"
    require(config.is_file() and not config.is_symlink(), f"BASE processor config unavailable: {config}")
    require(statistics.is_file() and not statistics.is_symlink(), f"BASE processor stats unavailable: {statistics}")
    config_sha = sha256_file(config)
    stats_sha = sha256_file(statistics)
    require(config_sha == BASE_PROCESSOR_CONFIG_SHA, f"BASE processor config drift: {config_sha}")
    require(stats_sha == BASE_STATISTICS_SHA, f"BASE processor statistics drift: {stats_sha}")
    return {"base_processor_root": str(base), "base_processor_config_sha256": config_sha, "base_statistics_sha256": stats_sha}


def _load_npy(path: Path) -> np.ndarray:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"fixed x0 unavailable: {path}")
    require(sha256_file(path) == NOISE_FILE_SHA, "fixed x0 file identity drift")
    value = np.asarray(np.load(path, allow_pickle=False))
    require(value.shape == RAW_SHAPE and value.dtype == np.float32 and np.isfinite(value).all(), f"fixed x0 ABI drift: {value.shape} {value.dtype}")
    require(sha256_array(value) == NOISE_ARRAY_SHA, "fixed x0 array identity drift")
    return value


def _load_observation(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} unavailable: {path}")
    actual = sha256_file(path)
    require(actual == expected_sha, f"{label} file identity drift: {actual}")
    with np.load(path, allow_pickle=False) as data:
        require({"body_rgb", "wrist_rgb", "state"}.issubset(data.files), f"{label} keys drift: {data.files}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        target = None if "target_position_world_m" not in data.files else np.asarray(data["target_position_world_m"], dtype=np.float64).reshape(-1)
        extra = sorted(set(data.files).difference({"body_rgb", "wrist_rgb", "state"}))
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, f"{label} body ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, f"{label} wrist ABI drift")
    require(state.shape == (STATE_DIM,) and np.isfinite(state).all(), f"{label} state ABI drift")
    if target is not None:
        require(target.shape == (3,) and np.isfinite(target).all(), f"{label} target ABI drift")
    return {"label": label, "path": str(path.resolve()), "file_sha256": actual, "body_rgb": body, "wrist_rgb": wrist, "state": state, "body_sha256": sha256_array(body), "wrist_sha256": sha256_array(wrist), "state_sha256": sha256_array(state), "target_keys": extra, "target_position_world_m": None if target is None else target.tolist()}


def _gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-500:]}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    selected = next((row for row in rows if row["uuid"] == AUTHORIZED_GPU_UUID), None)
    require(selected is not None, f"authorized UUID unavailable: {AUTHORIZED_GPU_UUID}")
    require(selected["memory_free_mib"] >= 14 * 1024, f"GPU free memory below gate: {selected}")
    return {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "query_sha256": sha256_bytes(result.stdout.encode()), "authorized_uuid": AUTHORIZED_GPU_UUID, "selected": selected, "all": rows}


def _load_old_probe() -> Any:
    path = SCRIPTS / "stage45f_pi0153_generation_probe.py"
    spec = importlib.util.spec_from_file_location("pi0159_previous_probe", path)
    require(spec is not None and spec.loader is not None, f"cannot load C1 input helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_input_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_model_input_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_model_input_keys(item))
    return keys


def _assert_boundary(inputs: Mapping[str, Any], label: str) -> None:
    forbidden = {"target_position_world_m", "target_xyz", "target_world_m", "target_id", "target_position", "expert", "label", "future_desired_eef_9d"}
    leaked = sorted(forbidden.intersection(_model_input_keys(inputs)))
    require(not leaked, f"{label} target/metadata/label leaked: {leaked}")


def _delta(value: np.ndarray) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(x))), "rmse": float(np.sqrt(np.mean(x * x))), "count": int(x.size), "finite": bool(np.isfinite(x).all())}


def _field_blocks(value: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    return {"active_190": _delta(x[mask]), "xyz": _delta(x[:, :3]), "rot6d": _delta(x[:, 3:9]), "gripper": _delta(x[:, 10]), "timing": _delta(x[:, 9]), "padding": _delta(x[:, 11:])}


def _tail_metrics(physical: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(physical, dtype=np.float64)[:, :3]
    tail = xyz[20:]
    second = np.diff(tail, n=2, axis=0)
    return {"tail_index": [20, 39], "tail_start_xyz": tail[0].tolist(), "tail_end_xyz": tail[-1].tolist(), "tail_displacement_m": float(np.linalg.norm(tail[-1] - tail[0])), "tail_path_length_m": float(np.sum(np.linalg.norm(np.diff(tail, axis=0), axis=1))), "tail_roughness_mean": float(np.mean(np.linalg.norm(second, axis=1))) if len(second) else 0.0}


def _tail_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    return {"tail_xyz": _delta(np.asarray(candidate, dtype=np.float64)[20:, :3] - np.asarray(baseline, dtype=np.float64)[20:, :3]), "tail_full_physical": _delta(np.asarray(candidate, dtype=np.float64)[20:] - np.asarray(baseline, dtype=np.float64)[20:])}


def _save_raw(path: Path, *, x0: np.ndarray, x_t: np.ndarray, v_target: np.ndarray, v_hat: np.ndarray, x1_hat: np.ndarray, active_mask: np.ndarray, state: np.ndarray, t_value: float, t_bucket: int) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite raw evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x0_float32=np.asarray(x0, dtype=np.float32), x_t_float32=np.asarray(x_t, dtype=np.float32), v_target_float32=np.asarray(v_target, dtype=np.float32), v_hat_float32=np.asarray(v_hat, dtype=np.float32), x1_hat_float32=np.asarray(x1_hat, dtype=np.float32), active_mask_float64=np.asarray(active_mask, dtype=np.float64), state_v1_float32=np.asarray(state, dtype=np.float32), t_value=np.asarray([t_value], dtype=np.float64), t_bucket=np.asarray([t_bucket], dtype=np.int64))
    return sha256_file(path)


def _state_audit(processor: Any, embodiment: Any, state: np.ndarray, inputs: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Audit processor round-trip and clipping without changing the input."""
    tag = getattr(embodiment, "value", str(embodiment))
    sap = processor.state_action_processor
    raw = {"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]}
    applied = sap.apply_state(raw, tag)
    reconstructed_groups = sap.unapply_state(applied, tag)
    reconstructed = np.concatenate([np.asarray(reconstructed_groups["eef_9d"]), np.asarray(reconstructed_groups["gripper_position"])], axis=-1).reshape(-1)
    processed = inputs.get("state")
    require(processed is not None, f"{label}: processed state missing")
    processed_np = processed.detach().to(device="cpu").float().contiguous().numpy().reshape(-1)
    expected = np.concatenate([np.asarray(applied["eef_9d"]), np.asarray(applied["gripper_position"])], axis=-1).reshape(-1)
    require(processed_np.size >= STATE_DIM and np.allclose(processed_np[:STATE_DIM], expected, rtol=0.0, atol=1e-6), f"{label}: processed state mismatch")
    clipped = {}
    for key in ("eef_9d", "gripper_position"):
        params = sap.norm_params[tag]["state"][key]
        lo = np.asarray(params["min"], dtype=np.float64).reshape(-1)
        hi = np.asarray(params["max"], dtype=np.float64).reshape(-1)
        source = raw[key].reshape(-1).astype(np.float64)
        pre = (source - lo) / np.maximum(hi - lo, 1e-8) * 2.0 - 1.0
        clipped[key] = {"preclip_out_of_range_count": int(np.count_nonzero((pre < -1.0) | (pre > 1.0))), "preclip_min": float(np.min(pre)), "preclip_max": float(np.max(pre)), "dimensions": np.flatnonzero((pre < -1.0) | (pre > 1.0)).tolist()}
    roundtrip = reconstructed - state.astype(np.float64)
    return {"label": label, "state_v1_sha256": sha256_array(state), "processed_state_shape": list(processed_np.shape), "processed_state_dtype": str(processed.dtype), "processed_state_prefix_sha256": sha256_array(processed_np[:STATE_DIM].astype(np.float32)), "processed_state_padding_nonzero": int(np.count_nonzero(np.abs(processed_np[STATE_DIM:]) > 0.0)), "clipping": clipped, "roundtrip_raw_state_max_abs": float(np.max(np.abs(roundtrip))), "roundtrip_raw_state_rmse": float(np.sqrt(np.mean(roundtrip * roundtrip))), "information_loss_observed": bool(np.max(np.abs(roundtrip)) > 1e-5)}


def _attempt_lock(control_root: Path, output_root: Path, gpu: Mapping[str, Any], repository_head: str) -> tuple[Path, dict[str, Any]]:
    control_root = Path(control_root).resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    lock = control_root / "stage45f_pi0159_attempt.lock.json"
    payload = {"schema": "CORELAB_STAGE45F_PI0159_ATTEMPT_LOCK_V1", "status": "STARTED", "token": RETURN_TOKEN, "output_root": str(output_root), "branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "repository_head": repository_head, "gpu_uuid": gpu["selected"]["uuid"], "created_unix": time.time()}
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ProbeError(f"PI0159 attempt already consumed: {lock}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return lock, payload


def _finish_lock(lock: Path, payload: Mapping[str, Any], **updates: Any) -> None:
    value = dict(payload)
    value.update(updates)
    Path(lock).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _make_svg(paths: Mapping[str, np.ndarray], output: Path) -> str:
    all_xy = np.concatenate([np.asarray(value)[:, :2] for value in paths.values()], axis=0)
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="520" viewBox="0 0 860 520">', '<rect width="100%" height="100%" fill="white"/>', '<text x="20" y="24" font-family="sans-serif" font-size="16">PI0159 implied endpoint tail (fixed T anchor)</text>']
    for idx, (label, value) in enumerate(paths.items()):
        pts = [f"{50 + 740 * (float(p[0]) - lo[0]) / span[0]:.2f},{460 - 390 * (float(p[1]) - lo[1]) / span[1]:.2f}" for p in np.asarray(value)[:, :2]]
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.85"/>')
        lines.append(f'<text x="610" y="{48 + idx * 18}" fill="{color}" font-family="sans-serif" font-size="11">{label}</text>')
    lines.append("</svg>")
    return write_text(output, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    head = str(args.repository_head).lower()
    require(len(head) == 40 and all(c in "0123456789abcdef" for c in head), "repository HEAD must be a concrete 40-hex SHA")

    gpu = _gpu_snapshot()
    identities = {**_verify_checkpoint(Path(args.model_path), Path(args.n2_stats)), **_verify_base(Path(args.base_model))}
    x0 = _load_npy(Path(args.noise))
    train = _load_observation(Path(args.train_input), TRAIN_INPUT_FILE_SHA, "canonical TRAIN/T")
    c0 = _load_observation(Path(args.archived_c0_input), C0_INPUT_FILE_SHA, "archived C0/C")
    stats_doc = read_json(Path(args.n2_stats))
    native_path = Path(args.native_a).resolve()
    require(native_path.is_file() and not native_path.is_symlink(), f"native expert snapshot unavailable: {native_path}")
    require(read_json(native_path).get("example_id") == TRAIN_LABEL_ID, "native expert identity drift")
    native_sha = sha256_file(native_path)

    lock, lock_payload = _attempt_lock(Path(args.control_root), output, gpu, head)
    counters = {"queries": 0, "conditioning_entered": 0, "conditioning_completed": 0, "action_head_calls": 0, "model_side_calls": 0, "exceptions": 0, "model_construction_attempts": 0}
    admission = {"schema": "CORELAB_STAGE45F_PI0159_CONDITIONING_ADMISSION_V1", "status": "ADMITTED_FOR_BOUNDED_EXECUTION", "token": ADMISSION_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "execution_head": head}, "fixed_identities": {**identities, "noise_array_sha256": NOISE_ARRAY_SHA, "noise_file_sha256": NOISE_FILE_SHA, "train_input_sha256": TRAIN_INPUT_FILE_SHA, "c0_input_sha256": C0_INPUT_FILE_SHA, "native_expert_snapshot_sha256": native_sha, "native_expert_snapshot": str(native_path), "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE, "modality_sha256": MODALITY_SHA}, "protocol": {"conditions": ["TRAIN_REFERENCE", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN", "C0_FULL"], "flow_t_values": list(T_VALUES), "flow_t_buckets": list(T_BUCKETS), "flow_equation": "x_t=(1-t)x0+t*x1; v_target=x1-x0; x1_hat=x_t+(1-t)*v_hat", "expert_target_fixed_across_conditions": True, "target_xyz_in_model_input": False, "raw_first": True}, "budget": {"conditioning_cap": MAX_CONDITIONING, "action_head_cap": MAX_HEAD_CALLS, "model_side_cap": MAX_MODEL_SIDE_CALLS, "model_construction_cap": MAX_MODEL_CONSTRUCTION}, "execution_boundary": {"model_forward": 0, "training": False, "backward": False, "optimizer_step": 0, "public_sampler": False, "free_running_inference": False, "icl": False, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "new_data": False, "safety_gate_relaxed": False}}
    admission_sha = write_json(output / "admission.json", admission)

    old = None
    model = None
    try:
        old = _load_old_probe()
        cosmos_binding = old._bind_local_cosmos_runtime(Path(args.stage10_root), Path(args.cosmos_snapshot))
        import torch
        from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor, _load_snapshot, _teacher_forced_local_flow
        from stage38_structured_offline_evaluation import _load_structured_model, _structured_label_and_normalizer
        import stage45c_h200_eval_runtime_adapter as runtime
        from go2arx5_vla.structured_action import structured_action_mask
        processor, embodiment, _ = _build_runtime_processor(groot_source=Path(args.stage10_root) / "src" / "isaac-gr00t", base_checkpoint=Path(args.base_model), dataset_dir=Path(args.dataset), modality_path=Path(args.modality), cosmos_snapshot=Path(args.cosmos_snapshot))

        snapshot = _load_snapshot(native_path)
        expert = np.asarray(snapshot.physical_action, dtype=np.float64)[None, ...]
        _, _, x1 = _structured_label_and_normalizer(expert, snapshot, stats_doc)
        x1 = np.asarray(x1, dtype=np.float32)[None, ...]
        require(x1.shape == RAW_SHAPE and np.isfinite(x1).all(), f"expert normalized target ABI drift: {x1.shape}")
        mask = np.asarray(structured_action_mask("T16"), dtype=bool)
        require(mask.shape == (HORIZON, ACTION_DIM) and int(mask.sum()) == 190, "T16 active mask drift")

        condition_specs = {
            "TRAIN_REFERENCE": (train["body_rgb"], train["wrist_rgb"], train["state"], "TRAIN/T"),
            "STATE_C0_VISION_TRAIN": (train["body_rgb"], train["wrist_rgb"], c0["state"], "V_T,S_C"),
            "VISION_C0_STATE_TRAIN": (c0["body_rgb"], c0["wrist_rgb"], train["state"], "V_C,S_T"),
            "C0_FULL": (c0["body_rgb"], c0["wrist_rgb"], c0["state"], "C0/C"),
        }
        inputs_by_condition: dict[str, Mapping[str, Any]] = {}
        audit_by_condition: dict[str, Any] = {}
        for label, (body, wrist, state, source_label) in condition_specs.items():
            inputs = old._build_c1_inputs(processor, embodiment, {"body": body, "wrist": wrist, "state": state}, train=True)
            _assert_boundary(inputs, label)
            inputs_by_condition[label] = inputs
            audit_by_condition[label] = old._state_audit(processor, embodiment, np.asarray(state, dtype=np.float32), inputs, label)

        model = _load_structured_model(Path(args.model_path), f"cuda:{gpu['selected']['index']}")
        counters["model_construction_attempts"] = 1
        model.eval()
        model_dtype = str(next(model.parameters()).dtype)
        require(model_dtype == "torch.bfloat16", f"model dtype drift: {model_dtype}")
        hook_calls = {"count": 0}
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_calls.__setitem__("count", hook_calls["count"] + 1))
        raw_rows: list[dict[str, Any]] = []
        decoded_rows: list[dict[str, Any]] = []
        try:
            for condition_index, (condition, (_, _, state, source_label)) in enumerate(condition_specs.items()):
                require(counters["conditioning_completed"] < MAX_CONDITIONING, "PI0159 conditioning budget exhausted")
                counters["queries"] += 1
                counters["conditioning_entered"] += 1
                with torch.inference_mode():
                    _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, inputs_by_condition[condition])
                counters["conditioning_completed"] += 1
                counters["model_side_calls"] += 1
                require(counters["model_side_calls"] <= MAX_MODEL_SIDE_CALLS, "PI0159 model-side budget exhausted after conditioning")
                for t_index, (t_value, bucket) in enumerate(zip(T_VALUES, T_BUCKETS)):
                    require(counters["action_head_calls"] < MAX_HEAD_CALLS, "PI0159 action-head budget exhausted")
                    x_t = ((1.0 - t_value) * x0 + t_value * x1).astype(np.float32)
                    v_target = (x1 - x0).astype(np.float32)
                    model_x_t = torch.from_numpy(x_t).to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                    t_tensor = torch.full((1,), int(bucket), device=processed.backbone_features.device, dtype=torch.long)
                    before_hook = hook_calls["count"]
                    with torch.inference_mode():
                        v_hat_tensor = _teacher_forced_local_flow(action_head=model.action_head, backbone_output=processed, action_input=action_input, noisy_action=model_x_t, t_discretized=t_tensor)
                    v_hat = np.asarray(v_hat_tensor.detach().to(device="cpu").float().contiguous().numpy(), dtype=np.float32)
                    observed = hook_calls["count"] - before_hook
                    require(observed == 1, f"{condition} t={bucket}: action-head hook count drift {observed}")
                    counters["action_head_calls"] += observed
                    counters["model_side_calls"] += observed
                    require(v_hat.shape == RAW_SHAPE and np.isfinite(v_hat).all(), f"{condition} t={bucket}: local field ABI/finite drift")
                    x1_hat = (x_t + (1.0 - t_value) * v_hat).astype(np.float32)
                    raw_path = output / "raw" / f"{condition_index:02d}_{condition}_t{bucket:03d}.npz"
                    raw_sha = _save_raw(raw_path, x0=x0, x_t=x_t, v_target=v_target, v_hat=v_hat, x1_hat=x1_hat, active_mask=mask, state=np.asarray(state, dtype=np.float32), t_value=t_value, t_bucket=bucket)
                    raw_rows.append({"condition": condition, "source_label": source_label, "t_value": t_value, "t_bucket": bucket, "state_sha256": sha256_array(state), "raw_path": str(raw_path), "raw_file_sha256": raw_sha, "v_hat_array_sha256": sha256_array(v_hat), "x1_hat_array_sha256": sha256_array(x1_hat), "action_head_calls": observed})
        finally:
            hook.remove()

        # The model loop is complete.  Everything below is CPU-derived from
        # the saved arrays and the frozen decoder bridge.
        physical_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
        prior_t_by_condition: dict[str, dict[int, np.ndarray]] = {}
        for row in raw_rows:
            with np.load(row["raw_path"], allow_pickle=False) as raw_file:
                x1_hat = np.asarray(raw_file["x1_hat_float32"], dtype=np.float32)
                state = np.asarray(raw_file["state_v1_float32"], dtype=np.float32)
            previous_sha, previous_payload = runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256
            runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = N2_STATS_SHA, N2_STATS_PAYLOAD_SHA
            try:
                physical_actual, decode_actual = runtime._decode_v2(x1_hat, state, Path(args.n2_stats), TIMESTAMPS, float(snapshot.observation_timestamp), f"PI0159:{row['condition']}:t{row['t_bucket']}:actual")
                physical_train, decode_train = runtime._decode_v2(x1_hat, train["state"], Path(args.n2_stats), TIMESTAMPS, float(snapshot.observation_timestamp), f"PI0159:{row['condition']}:t{row['t_bucket']}:standardized-T")
            finally:
                runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = previous_sha, previous_payload
            physical_by_condition.setdefault(row["condition"], {})[int(row["t_bucket"])] = {"actual": np.asarray(physical_actual, dtype=np.float64), "standardized_T": np.asarray(physical_train, dtype=np.float64), "decode_actual": decode_actual, "decode_standardized_T": decode_train}
            prior_t_by_condition.setdefault(row["condition"], {})[int(row["t_bucket"])] = x1_hat

        baseline_label = "TRAIN_REFERENCE"
        condition_rows = []
        difference_rows = []
        for condition, (_, _, state, source_label) in condition_specs.items():
            for t_value, bucket in zip(T_VALUES, T_BUCKETS):
                raw_row = next(item for item in raw_rows if item["condition"] == condition and item["t_bucket"] == bucket)
                with np.load(raw_row["raw_path"], allow_pickle=False) as raw_file:
                    v_hat = np.asarray(raw_file["v_hat_float32"], dtype=np.float64)
                    v_target = np.asarray(raw_file["v_target_float32"], dtype=np.float64)
                baseline_raw_path = next(item["raw_path"] for item in raw_rows if item["condition"] == baseline_label and item["t_bucket"] == bucket)
                with np.load(baseline_raw_path, allow_pickle=False) as base_file:
                    baseline_v = np.asarray(base_file["v_hat_float32"], dtype=np.float64)
                physical = physical_by_condition[condition][bucket]["standardized_T"]
                base_physical = physical_by_condition[baseline_label][bucket]["standardized_T"]
                local_error = _field_blocks(v_hat - v_target, mask)
                active_delta = _field_blocks(v_hat - baseline_v, mask)
                tail = _tail_metrics(physical)
                tail_diff = _tail_delta(physical, base_physical)
                condition_rows.append({"condition": condition, "source_label": source_label, "t_value": t_value, "t_bucket": bucket, "local_field_active_rmse": local_error["active_190"]["rmse"], "local_field_active_max_abs": local_error["active_190"]["max_abs"], "local_field_xyz_rmse": local_error["xyz"]["rmse"], "active_delta_vs_train_rmse": active_delta["active_190"]["rmse"], "active_delta_vs_train_max_abs": active_delta["active_190"]["max_abs"], "tail_path_length_m": tail["tail_path_length_m"], "tail_roughness_mean": tail["tail_roughness_mean"], "tail_displacement_m": tail["tail_displacement_m"], "tail_delta_xyz_rmse": tail_diff["tail_xyz"]["rmse"], "tail_delta_xyz_max_abs": tail_diff["tail_xyz"]["max_abs"], "processor_information_loss": audit_by_condition[condition]["information_loss_observed"], "processor_clip_dimensions": audit_by_condition[condition]["clipping"], "raw_file_sha256": raw_row["raw_file_sha256"]})
                difference_rows.append({"condition": condition, "t_bucket": bucket, "local_field_error_vs_teacher": local_error, "active_output_delta_vs_train": active_delta, "tail_delta_vs_train_standardized_anchor": tail_diff, "physical_actual_anchor_tail": _tail_metrics(physical_by_condition[condition][bucket]["actual"]), "physical_standardized_T_anchor_tail": tail})

        # Re-read the sealed C0 path to preserve the already accepted prefix /
        # tail observation.  This is CPU-only and does not create a new query.
        c0_prior = read_json(Path(args.prior_c0_analysis)) if args.prior_c0_analysis else None
        prior_reference = {"status": "NOT_PROVIDED" if c0_prior is None else "REUSED_SEALED_CPU_EVIDENCE", "path": None if c0_prior is None else str(Path(args.prior_c0_analysis).resolve()), "sha256": None if c0_prior is None else sha256_file(Path(args.prior_c0_analysis)), "content": c0_prior}

        states_nontrain = [row for row in condition_rows if row["condition"] != baseline_label]
        state_rows = [row for row in states_nontrain if row["condition"] == "STATE_C0_VISION_TRAIN"]
        vision_rows = [row for row in states_nontrain if row["condition"] == "VISION_C0_STATE_TRAIN"]
        state_effect = float(np.mean([row["active_delta_vs_train_rmse"] for row in state_rows]))
        vision_effect = float(np.mean([row["active_delta_vs_train_rmse"] for row in vision_rows]))
        tail_effect_state = float(np.mean([row["tail_delta_xyz_rmse"] for row in state_rows]))
        tail_effect_vision = float(np.mean([row["tail_delta_xyz_rmse"] for row in vision_rows]))
        if state_effect > 0.0 and vision_effect > 0.0:
            classification = "STATE_AND_VISION_REACH_LOCAL_FIELD"
            sentence = "Both the state-only and vision-only substitutions changed the teacher-forced active local field at every sampled flow time; the standardized-T decode separates this from decoder-anchor p0 mechanics."
        elif state_effect > 0.0:
            classification = "STATE_DOMINANT_LOCAL_FIELD_EFFECT"
            sentence = "The state-only substitution changed the active local field while the vision-only response was null or below exact float comparison; interpretation remains limited to this fixed teacher-forced probe."
        elif vision_effect > 0.0:
            classification = "VISION_DOMINANT_LOCAL_FIELD_EFFECT"
            sentence = "The vision-only substitution changed the active local field while the state-only response was null or below exact float comparison; interpretation remains limited to this fixed teacher-forced probe."
        else:
            classification = "NO_DETECTABLE_LOCAL_FIELD_EFFECT"
            sentence = "Neither single-modality substitution changed the active teacher-forced local field at the sampled flow times."

        output_hashes = {}
        output_hashes["conditions_csv_sha256"] = write_csv(output / "conditions.csv", condition_rows, ["condition", "source_label", "t_value", "t_bucket", "local_field_active_rmse", "local_field_active_max_abs", "local_field_xyz_rmse", "active_delta_vs_train_rmse", "active_delta_vs_train_max_abs", "tail_path_length_m", "tail_roughness_mean", "tail_displacement_m", "tail_delta_xyz_rmse", "tail_delta_xyz_max_abs", "processor_information_loss", "processor_clip_dimensions", "raw_file_sha256"])
        output_hashes["condition_differences_json_sha256"] = write_json(output / "condition_differences.json", difference_rows)
        output_hashes["state_audit_json_sha256"] = write_json(output / "processed_state_audit.json", audit_by_condition)
        output_hashes["raw_manifest_json_sha256"] = write_json(output / "raw_manifest.json", raw_rows)
        output_hashes["prior_c0_reference_json_sha256"] = write_json(output / "prior_c0_reference.json", prior_reference)
        output_hashes["trajectory_svg_sha256"] = _make_svg({condition: physical_by_condition[condition][T_BUCKETS[-1]]["standardized_T"] for condition in condition_specs}, output / "trajectory_plot.svg")
        conclusion = f"""# PI0159 conditioning-mechanism result

Status: **COMPLETED**

Science classification: **{classification}**

{sentence}

## Fixed diagnostic boundary

N2 seed2223@2000, the sealed float32 x0, original N2 runtime-V2 statistics,
native TRAIN-a expert label, H40/T16 and bf16 precision were fixed. The local
field used `x_t=(1-t)x0+t*x1`, `v_target=x1-x0`, and the exact implied endpoint
`x1_hat=x_t+(1-t)*v_hat`. Four conditioning passes and twelve action-head
calls were executed; raw arrays were saved before all CPU decoding and metrics.
No public sampler, training, WBC, simulator or closed-loop call occurred.

State clipping is reported per condition. Standardized-T physical tails are
the primary cross-condition view; actual-anchor tails are retained only to
show decoder mechanics. The previously sealed C0 prefix/tail evidence is
referenced without rerun. This packet does not establish task success,
world-frame TRAIN metrics, visited-state behavior, or generalization.

## Single recommendation

Authorize one fixed-budget independent-expert / visited-state local-field
supervision diagnostic before any ICL or training decision; preserve the same
raw-first ABI and compare teacher-forced local-field error on states sampled
after policy execution. Do not start that intervention automatically.
"""
        output_hashes["one_page_conclusion_sha256"] = write_text(output / "one_page_conclusion.md", conclusion)
        report = {"schema": "CORELAB_STAGE45F_PI0159_CONDITIONING_MECHANISM_RESULT_V1", "status": "COMPLETED", "token": RETURN_TOKEN, "science_classification": classification, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "execution_head": head}, "identities": {**identities, "noise_array_sha256": NOISE_ARRAY_SHA, "noise_file_sha256": NOISE_FILE_SHA, "train_input_sha256": TRAIN_INPUT_FILE_SHA, "c0_input_sha256": C0_INPUT_FILE_SHA, "native_expert_snapshot_sha256": native_sha, "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE, "modality_sha256": MODALITY_SHA, "model_dtype": model_dtype, "cosmos_binding": cosmos_binding}, "conditions": condition_rows, "condition_differences": difference_rows, "processed_state_audit": audit_by_condition, "prior_c0_reference": prior_reference, "flow": {"t_values": list(T_VALUES), "t_buckets": list(T_BUCKETS), "equations": ["x_t=(1-t)x0+t*x1", "v_target=x1-x0", "x1_hat=x_t+(1-t)*v_hat"], "active_mask_shape": [HORIZON, ACTION_DIM], "active_scalars": 190}, "outputs": output_hashes, "counters": {**counters, "raw_rows": len(raw_rows), "cumulative_new_model_side_calls": counters["model_side_calls"]}, "budget": {"conditioning_cap": MAX_CONDITIONING, "action_head_cap": MAX_HEAD_CALLS, "model_side_cap": MAX_MODEL_SIDE_CALLS, "conditioning_used": counters["conditioning_completed"], "action_head_used": counters["action_head_calls"], "model_side_used": counters["model_side_calls"]}, "effect_summary": {"state_mean_active_delta_rmse": state_effect, "vision_mean_active_delta_rmse": vision_effect, "state_mean_tail_delta_xyz_rmse": tail_effect_state, "vision_mean_tail_delta_xyz_rmse": tail_effect_vision}, "execution_boundary": {"model_forward": counters["model_side_calls"], "model_construction": counters["model_construction_attempts"], "training": False, "backward": False, "optimizer_step": 0, "public_sampler": False, "free_running_inference": False, "icl": False, "new_data": False, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "safety_gate_relaxed": False, "retry": False}, "claim_boundary": "fixed-policy teacher-forced conditioning and tail characterization only; no task-success, WBC, closed-loop, world-frame TRAIN, ICL, training, or generalization claim", "raw_first": True}
        report_sha = write_json(output / "pi0159_conditioning_mechanism.json", report)
        output_hashes["report_sha256"] = report_sha
        evidence_sha = write_json(output / "evidence_hashes.json", {**output_hashes, "admission_sha256": admission_sha, "raw_sha256_by_row": {f"{row['condition']}:t{row['t_bucket']}": row["raw_file_sha256"] for row in raw_rows}})
        _finish_lock(lock, lock_payload, status="COMPLETED", completed_unix=time.time(), counters=report["counters"], report_sha256=report_sha, evidence_hashes_sha256=evidence_sha)
        return {**report, "report_sha256": report_sha, "evidence_hashes_sha256": evidence_sha, "admission_sha256": admission_sha}
    except Exception as exc:
        counters["exceptions"] += 1
        if lock.exists():
            try:
                _finish_lock(lock, lock_payload, status="BLOCKED_OR_FAILED", failed_unix=time.time(), counters=counters, error_type=type(exc).__name__, error=str(exc)[-3000:])
            except Exception:
                pass
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-head", required=True)
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
    p.add_argument("--native-a", required=True, type=Path)
    p.add_argument("--prior-c0-analysis", type=Path)
    p.add_argument("--control-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    return p


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "science_classification": result["science_classification"], "token": result["token"], "report_sha256": result["report_sha256"], "counters": result["counters"]}, sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
