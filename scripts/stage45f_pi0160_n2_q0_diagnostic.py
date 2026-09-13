#!/usr/bin/env python3
"""PI0160 fixed-N2 q0 diagnostics for the matched expert campaign.

The native expert campaign is intentionally a separate process.  This runner
consumes only its sealed q0 snapshots, constructs the already accepted N2
seed2223@2000 model once, and records one E4 free-running query plus three
teacher-forced/local-flow queries per qualified condition.  Every raw tensor is
written before CPU decoding.  GR00T output is never passed to WBC, and no
simulator is imported here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
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
for _path in (SCRIPTS, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PI_GATE = "PI_GATE_STATE_0160"
PI_COMMENT = 5651703057
PR_COMMENT = 5651704759
RETURN_TOKEN = "CORELAB_PI0160_MATCHED_RESET_EXPERT_FOR_PI_REVIEW"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
STARTING_HEAD = "f2d115811051ce0b1cd3c177e5b6c32bf1efce64"

CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
NOISE_FILE_SHA = "83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922"
BASE_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_STATS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"

HORIZON = 40
ACTION_DIM = 132
PHYSICAL_DIM = 10
STATE_DIM = 10
TIMESTAMPS = np.arange(HORIZON, dtype=np.float64) * 0.05
T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
T_BUCKETS = (67, 369, 783)
TIMESTEP_IDS = (0, 250, 500, 750)
SAMPLER_DT = 0.25
CONDITION_ORDER = (
    "C0_NOMINAL",
    "C1_X_PLUS_5CM",
    "C2_X_MINUS_5CM",
    "C3_Y_PLUS_5CM",
    "C4_Y_MINUS_5CM",
)
RAW_SHAPE = (1, HORIZON, ACTION_DIM)
PHYSICAL_SHAPE = (HORIZON, PHYSICAL_DIM)


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value)).tobytes())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
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


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
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


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    return {"hostname": platform.node(), "query_sha256": sha256_bytes(result.stdout.encode()), "query": result.stdout, "rows": rows}


def select_gpu(snapshot: Mapping[str, Any], requested: str | None) -> dict[str, Any]:
    candidates = [row for row in snapshot["rows"] if "4080 SUPER" in str(row["name"]) and int(row["memory_total_mib"]) >= 30000 and int(row["memory_free_mib"]) >= 14 * 1024]
    if requested:
        candidates = [row for row in candidates if str(row["uuid"]) == str(requested)]
    require(candidates, f"no eligible 32GB RTX 4080 SUPER: {snapshot['rows']}")
    candidates.sort(key=lambda row: (-int(row["memory_free_mib"]), int(row["index"])))
    return dict(candidates[0])


def verify_fixed_inputs(args: argparse.Namespace, expert_summary: Mapping[str, Any]) -> dict[str, Any]:
    require(expert_summary.get("status") == "COMPLETED", f"expert campaign is not complete: {expert_summary.get('status')}")
    require(list(item.get("condition_id") for item in expert_summary.get("conditions", [])) == list(CONDITION_ORDER), "expert condition order drift")
    require(all(item.get("status") == "PASS" for item in expert_summary["conditions"]), "model diagnostics require five qualified expert conditions")
    checkpoint = Path(args.model_path).resolve()
    stats = Path(args.n2_stats).resolve()
    noise = Path(args.noise).resolve()
    entries = tree_entries(checkpoint, "stage45e/checkpoint-2000")
    checkpoint_tree = tree_digest(entries)
    require(checkpoint_tree == CHECKPOINT_TREE, f"N2 checkpoint tree drift: {checkpoint_tree}")
    require(stats.is_file() and not stats.is_symlink(), f"N2 stats unavailable: {stats}")
    stats_sha = sha256_file(stats)
    require(stats_sha == N2_STATS_SHA, f"N2 stats file drift: {stats_sha}")
    stats_doc = read_json(stats)
    require(stats_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats payload drift")
    statistics = stats_doc.get("statistics", stats_doc)
    require(int(statistics.get("action_horizon")) == HORIZON and int(statistics.get("universal_action_dim")) == ACTION_DIM, "N2 action ABI drift")
    require(int(statistics.get("active_scalar_count")) == 190 and statistics.get("timing_mode") == "T16", "N2 structured ABI drift")
    require(noise.is_file() and not noise.is_symlink() and sha256_file(noise) == NOISE_FILE_SHA, "fixed x0 file identity drift")
    x0 = np.asarray(np.load(noise, allow_pickle=False))
    require(x0.shape == RAW_SHAPE and x0.dtype == np.float32 and np.isfinite(x0).all() and sha256_array(x0) == NOISE_ARRAY_SHA, "fixed x0 array identity drift")
    base = Path(args.base_model).resolve()
    require((base / "processor_config.json").is_file() and sha256_file(base / "processor_config.json") == BASE_CONFIG_SHA, "BASE processor config drift")
    require((base / "statistics.json").is_file() and sha256_file(base / "statistics.json") == BASE_STATS_SHA, "BASE statistics drift")
    modality = Path(args.modality).resolve()
    require(modality.is_file() and sha256_file(modality) == MODALITY_SHA, "modality identity drift")
    return {"checkpoint_path": str(checkpoint), "checkpoint_tree_sha256": checkpoint_tree, "checkpoint_file_count": len(entries), "checkpoint_total_bytes": int(sum(int(item["bytes"]) for item in entries)), "n2_stats_path": str(stats), "n2_stats_sha256": stats_sha, "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "noise_path": str(noise), "noise_file_sha256": NOISE_FILE_SHA, "noise_array_sha256": NOISE_ARRAY_SHA, "base_model_path": str(base), "base_processor_config_sha256": BASE_CONFIG_SHA, "base_statistics_sha256": BASE_STATS_SHA, "modality_path": str(modality), "modality_sha256": MODALITY_SHA}


def load_expert_q0(expert_root: Path, condition: str) -> dict[str, Any]:
    path = Path(expert_root) / "conditions" / condition / "snapshots" / "q0.npz"
    meta_path = Path(expert_root) / "conditions" / condition / "condition_metadata.json"
    require(path.is_file() and not path.is_symlink(), f"missing expert q0: {path}")
    require(meta_path.is_file() and read_json(meta_path).get("status") == "PASS", f"expert condition not PASS: {condition}")
    with np.load(path, allow_pickle=False) as data:
        required = {"body_rgb", "wrist_rgb", "state_v1", "expert_physical_local", "expert_timestamps", "eef_world_m", "eef_local_m", "anchor_world_m", "target_world_m"}
        require(required.issubset(set(data.files)), f"expert q0 keys drift for {condition}: {data.files}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state_v1"], dtype=np.float32).reshape(-1)
        expert = np.asarray(data["expert_physical_local"], dtype=np.float64)
        timestamps = np.asarray(data["expert_timestamps"], dtype=np.float64).reshape(-1)
        eef_world = np.asarray(data["eef_world_m"], dtype=np.float64).reshape(-1)
        eef_local = np.asarray(data["eef_local_m"], dtype=np.float64).reshape(-1)
        anchor = np.asarray(data["anchor_world_m"], dtype=np.float64).reshape(-1)
        target = np.asarray(data["target_world_m"], dtype=np.float64).reshape(-1)
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, f"body ABI drift for {condition}")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, f"wrist ABI drift for {condition}")
    require(state.shape == (STATE_DIM,) and expert.shape == PHYSICAL_SHAPE and timestamps.shape == (HORIZON,), f"q0 ABI drift for {condition}")
    require(eef_world.shape == eef_local.shape == anchor.shape == target.shape == (3,), f"q0 pose ABI drift for {condition}")
    require(np.isfinite(body).all() and np.isfinite(wrist).all() and np.isfinite(state).all() and np.isfinite(expert).all() and np.isfinite(timestamps).all(), f"q0 nonfinite for {condition}")
    require(np.all(np.diff(timestamps) > 0.0) and np.allclose(eef_local, state[:3], atol=1e-5, rtol=0.0), f"q0 local anchor mismatch for {condition}")
    return {"condition": condition, "path": str(path.resolve()), "file_sha256": sha256_file(path), "body": body, "wrist": wrist, "state": state, "expert": expert, "timestamps": timestamps, "eef_world": eef_world, "eef_local": eef_local, "anchor": anchor, "target": target}


def model_input_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(model_input_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(model_input_keys(item))
    return keys


def assert_input_boundary(inputs: Mapping[str, Any], label: str) -> None:
    forbidden = {"target_position_world_m", "target_xyz", "target_world_m", "target_id", "target_position", "expert", "label", "future_desired_eef_9d"}
    leaked = sorted(forbidden.intersection(model_input_keys(inputs)))
    require(not leaked, f"{label} target/metadata/label leaked into model input: {leaked}")


def field_delta(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    return {"rmse": float(np.sqrt(np.mean(value * value))), "max_abs": float(np.max(np.abs(value))), "count": int(value.size), "finite": bool(np.isfinite(value).all())}


def field_blocks(value: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == RAW_SHAPE:
        value = value[0]
    require(value.shape == (HORIZON, ACTION_DIM), f"field shape drift: {value.shape}")
    return {"active_190": field_delta(value[mask]), "xyz": field_delta(value[:, :3]), "rot6d": field_delta(value[:, 3:9]), "timing": field_delta(value[:, 9]), "gripper": field_delta(value[:, 10]), "padding": field_delta(value[:, 11:])}


def physical_metrics(candidate: np.ndarray, expert: np.ndarray, state: np.ndarray, target: np.ndarray, anchor: np.ndarray, label: str) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    xyz = candidate[:, :3]
    expert_xyz = expert[:, :3]
    state_xyz = np.asarray(state[:3], dtype=np.float64)
    require(candidate.shape == expert.shape == PHYSICAL_SHAPE and np.isfinite(candidate).all(), f"{label} physical ABI/finite drift")
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    # Actions are local to the native reference frame while target metrics are
    # world-frame.  Convert only for target distances; residual/path metrics
    # remain local-frame quantities.
    xyz_world = xyz + anchor[None, :]
    expert_xyz_world = expert_xyz + anchor[None, :]
    state_xyz_world = state_xyz + anchor
    distances = np.linalg.norm(xyz_world - target[None, :], axis=1)
    expert_distances = np.linalg.norm(expert_xyz_world - target[None, :], axis=1)
    start_residual = float(np.linalg.norm(xyz[0] - state_xyz))
    path = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    roughness = float(np.mean(np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1))) if len(xyz) > 2 else 0.0
    speed = np.diff(xyz, axis=0) / 0.05
    d0 = float(np.linalg.norm(state_xyz_world - target))
    return {"label": label, "endpoint_error_to_expert_m": float(np.linalg.norm(candidate[-1, :3] - expert[-1, :3])), "xyz_rmse_to_expert_m": float(np.sqrt(np.mean((xyz - expert_xyz) ** 2))), "physical_rmse_to_expert": float(np.sqrt(np.mean((candidate - expert) ** 2))), "initial_target_error_m": d0, "minimum_target_error_m": float(np.min(distances)), "final_target_error_m": float(distances[-1]), "absolute_progress_m": float(d0 - distances[-1]), "best_progress_m": float(d0 - np.min(distances)), "progress_sign": "POSITIVE" if d0 > distances[-1] else ("ZERO" if np.isclose(d0, distances[-1]) else "NEGATIVE"), "expert_final_target_error_m": float(expert_distances[-1]), "expert_progress_m": float(d0 - expert_distances[-1]), "start_residual_m": start_residual, "path_length_m": path, "roughness_m": roughness, "speed_rms_mps": float(np.sqrt(np.mean(speed * speed))) if len(speed) else 0.0, "clipping_separate_from_model": True, "metric_frame": "world_target_local_action"}


def label_clipping(expert: np.ndarray, stats_doc: Mapping[str, Any]) -> dict[str, Any]:
    statistics = stats_doc.get("statistics", stats_doc)
    lower = np.asarray(statistics.get("selected_lower", stats_doc.get("selected_lower")), dtype=np.float64)
    upper = np.asarray(statistics.get("selected_upper", stats_doc.get("selected_upper")), dtype=np.float64)
    require(lower.shape == (ACTION_DIM,) and upper.shape == (ACTION_DIM,), "N2 selected bounds unavailable")
    from go2arx5_vla.structured_action import encode_structured_action
    # The caller supplies the encoded label separately; this helper records
    # bounds only.  It intentionally does not clip physical expert values.
    return {"selected_lower_sha256": sha256_array(lower), "selected_upper_sha256": sha256_array(upper), "selected_lower_min": float(np.min(lower)), "selected_upper_max": float(np.max(upper)), "physical_expert_untouched": True}


def decode_v2(runtime: Any, raw: np.ndarray, state: np.ndarray, stats_path: Path, observation_timestamp: float, chunk_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    old_sha, old_payload = runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256
    runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = N2_STATS_SHA, N2_STATS_PAYLOAD_SHA
    try:
        physical, details = runtime._decode_v2(raw, state, stats_path, TIMESTAMPS, observation_timestamp, chunk_id)
    finally:
        runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = old_sha, old_payload
    physical = np.asarray(physical, dtype=np.float64)
    require(physical.shape == PHYSICAL_SHAPE and np.isfinite(physical).all(), f"{chunk_id} decoded physical ABI drift")
    return physical, details


def save_raw(path: Path, **arrays: Any) -> str:
    require(not path.exists(), f"refusing to overwrite raw evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def make_svg(paths: Mapping[str, np.ndarray], path: Path) -> str:
    all_xy = np.concatenate([np.asarray(value)[:, :2] for value in paths.values()], axis=0)
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">', '<rect width="100%" height="100%" fill="white"/>', '<text x="20" y="24" font-family="sans-serif" font-size="16">PI0160 matched q0 decoded trajectories (local XY)</text>']
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for idx, (label, value) in enumerate(paths.items()):
        points = [f"{55 + 760 * (float(p[0]) - lo[0]) / span[0]:.2f},{500 - 420 * (float(p[1]) - lo[1]) / span[1]:.2f}" for p in np.asarray(value)[:, :2]]
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        lines.append(f'<text x="650" y="{48 + idx * 17}" fill="{color}" font-family="sans-serif" font-size="11">{label}</text>')
    lines.append("</svg>\n")
    return write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    require(str(args.repository_head).lower() == STARTING_HEAD, f"repository HEAD drift: {args.repository_head}")
    expert_root = Path(args.expert_root).resolve()
    expert_summary = read_json(expert_root / "pi0160_matched_expert_result.json")
    identities = verify_fixed_inputs(args, expert_summary)
    snapshot = gpu_snapshot()
    gpu = select_gpu(snapshot, args.gpu_uuid)
    q0 = {condition: load_expert_q0(expert_root, condition) for condition in CONDITION_ORDER}
    stats_doc = read_json(Path(args.n2_stats))
    x0 = np.asarray(np.load(Path(args.noise), allow_pickle=False), dtype=np.float32)
    admission = {"schema": "CORELAB_PI0160_MODEL_ADMISSION_V1", "status": "ADMITTED_FOR_DIRECT_EXECUTION", "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "head": STARTING_HEAD}, "expert_root": str(expert_root), "expert_result_sha256": sha256_file(expert_root / "pi0160_matched_expert_result.json"), "fixed_identities": identities, "gpu": {"selected": gpu, "snapshot": snapshot}, "protocol": {"condition_order": list(CONDITION_ORDER), "fr": {"sampler": "E4", "timesteps": list(TIMESTEP_IDS), "dt": SAMPLER_DT, "per_condition_heads": 4}, "teacher": {"t_values": list(T_VALUES), "t_buckets": list(T_BUCKETS), "flow": ["x_t=(1-t)x0+t*x1", "v_target=x1-x0", "x1_hat=x_t+(1-t)v_hat"], "per_condition_heads": 3}, "target_xyz_in_model_input": False, "gr00t_output_sent_to_wbc": False, "raw_first": True}, "budget": {"conditioning_cap": 10, "action_head_cap": 35, "model_side_cap": 45, "model_construction_cap": 1}, "execution_boundary": {"gr00t_forward": 45, "model_forward": 45, "training": False, "backward": False, "optimizer_step": 0, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "icl": False, "validation_e_read": False, "holdout_f_read": False, "gr00t_output_sent_to_wbc": False}
    }
    admission_sha = write_json(output / "model_admission.json", admission)
    old_probe = load_module(SCRIPTS / "stage45f_pi0153_generation_probe.py", "pi0160_input_probe")
    flow = load_module(SCRIPTS / "stage26_h1_c1_fixed_noise_local_flow.py", "pi0160_flow_helpers")
    runtime = load_module(SCRIPTS / "stage45c_h200_eval_runtime_adapter.py", "pi0160_runtime_adapter")
    cosmos_binding = old_probe._bind_local_cosmos_runtime(Path(args.stage10_root), Path(args.cosmos_snapshot))
    from stage40_b0_model_control import _bind_local_cosmos_processor
    from stage38_structured_offline_evaluation import _load_structured_model
    processor_binding = _bind_local_cosmos_processor(Path(args.stage10_root), Path(args.cosmos_snapshot))
    processor, embodiment, _ = flow._build_runtime_processor(groot_source=Path(args.stage10_root) / "src" / "isaac-gr00t", base_checkpoint=Path(args.base_model), dataset_dir=Path(args.dataset), modality_path=Path(args.modality), cosmos_snapshot=Path(args.cosmos_snapshot))
    from go2arx5_vla.structured_action import encode_structured_action, structured_action_mask
    from go2arx5_vla.stage38_structured_training import StructuredActionNormalizer
    mask = np.asarray(structured_action_mask("T16"), dtype=bool)
    require(mask.shape == (HORIZON, ACTION_DIM) and int(mask.sum()) == 190, "T16 mask drift")
    normalizer = StructuredActionNormalizer(stats_doc.get("statistics", stats_doc), clip_outliers=True)
    inputs_by_condition: dict[str, Mapping[str, Any]] = {}
    x1_by_condition: dict[str, np.ndarray] = {}
    label_by_condition: dict[str, Any] = {}
    clipping: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        item = q0[condition]
        inputs = old_probe._build_c1_inputs(processor, embodiment, {"body": item["body"], "wrist": item["wrist"], "state": item["state"]}, train=False)
        assert_input_boundary(inputs, condition)
        inputs_by_condition[condition] = inputs
        label = encode_structured_action(item["expert"], item["timestamps"], item["state"], timing_mode="T16", chunk_id=f"PI0160:{condition}:expert", source_frame="NATIVE_WBC_RELATIVE_REFERENCE", tcp_semantic_id="x5_link6_gripper_center", observation_timestamp=float(item["timestamps"][0]), provenance={"stage": "PI0160", "source": "independent_same_scene_expert"})
        x1_by_condition[condition] = np.asarray(normalizer.normalize(label), dtype=np.float32)[None, ...]
        require(x1_by_condition[condition].shape == RAW_SHAPE and np.isfinite(x1_by_condition[condition]).all(), f"{condition} normalized expert label drift")
        label_by_condition[condition] = label
        clipping[condition] = {"active_label_scalars": int(np.count_nonzero(label.mask)), "active_mask_shape": list(label.mask.shape), "normalized_clipped_scalar_count": int(np.count_nonzero(np.abs(x1_by_condition[condition][0][mask]) >= 1.0)), "label_codec_roundtrip_input_untouched": True, "stats_route": N2_STATS_SHA}
    model = _load_structured_model(Path(args.model_path), "cuda:0")
    model.eval()
    model_dtype = str(next(model.parameters()).dtype)
    require(model_dtype == "torch.bfloat16", f"model dtype drift: {model_dtype}")
    import torch
    hook_calls = {"count": 0}
    hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_calls.__setitem__("count", hook_calls["count"] + 1))
    counters = {"model_construction_attempts": 1, "conditioning_entered": 0, "conditioning_completed": 0, "action_head_calls": 0, "model_side_calls": 0, "fr_queries": 0, "teacher_queries": 0, "exceptions": 0}
    raw_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []
    paths: dict[str, np.ndarray] = {}
    try:
        for condition in CONDITION_ORDER:
            item = q0[condition]
            # FR: fixed x0, exactly four accepted sampler heads.
            counters["fr_queries"] += 1
            counters["conditioning_entered"] += 1
            before = hook_calls["count"]
            with torch.inference_mode():
                raw, heads = runtime._sample_one(model, inputs_by_condition[condition], torch.from_numpy(x0), torch)
            observed = hook_calls["count"] - before
            counters["conditioning_completed"] += 1
            counters["action_head_calls"] += int(observed)
            counters["model_side_calls"] += 1 + int(observed)
            require(heads == 4 and observed == 4, f"{condition} FR head count drift: {heads}/{observed}")
            raw_np = np.asarray(raw.detach().cpu().float().numpy(), dtype=np.float32)
            raw_path = output / "raw" / f"{condition}_fr.npz"
            raw_sha = save_raw(raw_path, raw_universal_action=raw_np, x0_float32=x0, state_v1_float32=item["state"])
            physical, decode_details = decode_v2(runtime, raw_np, item["state"], Path(args.n2_stats), float(item["timestamps"][0]), f"PI0160:{condition}:FR")
            metrics = physical_metrics(physical, item["expert"], item["state"], item["target"], item["anchor"], f"{condition}:FR")
            row = {"condition": condition, "mode": "FR", "t_bucket": "", "raw_path": str(raw_path), "raw_file_sha256": raw_sha, "action_head_calls": int(observed), "conditioning_calls": 1, **metrics, "decode_stats_sha256": decode_details.get("stats_sha256"), "clip": clipping[condition]}
            raw_rows.append(row)
            decoded_rows.append(row)
            paths[f"{condition}:FR"] = physical
            # Matched teacher/local field uses the same observation but the
            # condition's own expert label only in x_t/v_target, never in
            # model-visible conditioning.
            counters["teacher_queries"] += 1
            counters["conditioning_entered"] += 1
            with torch.inference_mode():
                _, action_input, processed, _state_features = runtime._prepare_model_conditioning(model, inputs_by_condition[condition])
            counters["conditioning_completed"] += 1
            counters["model_side_calls"] += 1
            v_target = (x1_by_condition[condition] - x0).astype(np.float32)
            for t_value, bucket in zip(T_VALUES, T_BUCKETS):
                x_t = ((1.0 - t_value) * x0 + t_value * x1_by_condition[condition]).astype(np.float32)
                model_x_t = torch.from_numpy(x_t).to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                t_tensor = torch.full((1,), int(bucket), device=processed.backbone_features.device, dtype=torch.long)
                before = hook_calls["count"]
                with torch.inference_mode():
                    v_hat_tensor = flow._teacher_forced_local_flow(action_head=model.action_head, backbone_output=processed, action_input=action_input, noisy_action=model_x_t, t_discretized=t_tensor)
                observed = hook_calls["count"] - before
                counters["action_head_calls"] += int(observed)
                counters["model_side_calls"] += int(observed)
                require(observed == 1, f"{condition} teacher t={bucket} head count drift: {observed}")
                v_hat = np.asarray(v_hat_tensor.detach().cpu().float().numpy(), dtype=np.float32)
                x1_hat = (x_t + (1.0 - t_value) * v_hat).astype(np.float32)
                raw_path = output / "raw" / f"{condition}_teacher_t{bucket:03d}.npz"
                raw_sha = save_raw(raw_path, x0_float32=x0, x_t_float32=x_t, v_target_float32=v_target, v_hat_float32=v_hat, x1_hat_float32=x1_hat, active_mask_float64=mask.astype(np.float64), state_v1_float32=item["state"], t_value=np.asarray([t_value], dtype=np.float64), t_bucket=np.asarray([bucket], dtype=np.int64))
                physical, decode_details = decode_v2(runtime, x1_hat, item["state"], Path(args.n2_stats), float(item["timestamps"][0]), f"PI0160:{condition}:T{bucket}")
                zero_physical, _zero_details = decode_v2(runtime, x_t, item["state"], Path(args.n2_stats), float(item["timestamps"][0]), f"PI0160:{condition}:ZERO{bucket}")
                local_field_error = field_blocks(v_hat - v_target, mask)
                zero_field_error = field_blocks(x_t - x1_by_condition[condition], mask)
                metrics = physical_metrics(physical, item["expert"], item["state"], item["target"], item["anchor"], f"{condition}:T{bucket}")
                metrics.update({"teacher_field_active_rmse": local_field_error["active_190"]["rmse"], "teacher_field_active_max_abs": local_field_error["active_190"]["max_abs"], "teacher_field_xyz_rmse": local_field_error["xyz"]["rmse"], "zero_field_active_rmse": zero_field_error["active_190"]["rmse"], "zero_field_xyz_rmse": zero_field_error["xyz"]["rmse"], "implied_endpoint_vs_zero_xyz_rmse": float(np.sqrt(np.mean((physical[:, :3] - zero_physical[:, :3]) ** 2)))} )
                row = {"condition": condition, "mode": "MATCHED_LOCAL_FIELD", "t_bucket": int(bucket), "t_value": float(t_value), "raw_path": str(raw_path), "raw_file_sha256": raw_sha, "action_head_calls": int(observed), "conditioning_calls": 0, **metrics, "decode_stats_sha256": decode_details.get("stats_sha256"), "clip": clipping[condition]}
                raw_rows.append(row)
                decoded_rows.append(row)
                if bucket == T_BUCKETS[-1]:
                    paths[f"{condition}:T{bucket}"] = physical
            require(counters["model_side_calls"] <= 45 and counters["conditioning_completed"] <= 10 and counters["action_head_calls"] <= 35, "PI0160 model budget exceeded")
    except Exception:
        counters["exceptions"] += 1
        raise
    finally:
        hook.remove()
    require(counters["conditioning_completed"] == 10 and counters["action_head_calls"] == 35 and counters["model_side_calls"] == 45, f"PI0160 model counters drift: {counters}")
    fields = ["condition", "mode", "t_bucket", "t_value", "raw_path", "raw_file_sha256", "action_head_calls", "conditioning_calls", "endpoint_error_to_expert_m", "xyz_rmse_to_expert_m", "physical_rmse_to_expert", "initial_target_error_m", "minimum_target_error_m", "final_target_error_m", "absolute_progress_m", "best_progress_m", "progress_sign", "expert_final_target_error_m", "expert_progress_m", "start_residual_m", "path_length_m", "roughness_m", "speed_rms_mps", "teacher_field_active_rmse", "teacher_field_active_max_abs", "teacher_field_xyz_rmse", "zero_field_active_rmse", "zero_field_xyz_rmse", "implied_endpoint_vs_zero_xyz_rmse", "decode_stats_sha256", "clip"]
    table_sha = write_csv(output / "matched_expert_model_table.csv", decoded_rows, fields)
    raw_manifest_sha = write_json(output / "raw_manifest.json", [{"condition": row["condition"], "mode": row["mode"], "t_bucket": row.get("t_bucket", ""), "raw_path": row["raw_path"], "raw_file_sha256": row["raw_file_sha256"]} for row in raw_rows])
    clipping_sha = write_json(output / "clipping_and_label_quality.json", clipping)
    trajectory_sha = make_svg(paths, output / "trajectory_plot.svg")
    fr_rows = [row for row in decoded_rows if row["mode"] == "FR"]
    teacher_rows = [row for row in decoded_rows if row["mode"] == "MATCHED_LOCAL_FIELD"]
    mean_fr_progress = float(np.mean([row["absolute_progress_m"] for row in fr_rows]))
    mean_teacher_progress = float(np.mean([row["absolute_progress_m"] for row in teacher_rows]))
    conclusion = f"""# PI0160 matched-reset expert/model result

Status: **COMPLETED**

Five fixed initial conditions were first run through the independent native
smoothstep expert (245 post-S0 ticks per condition).  The fixed N2
seed2223@2000 policy then received only each condition's q0 body RGB, wrist RGB,
StateV1 and language.  For each q0 one E4 free-running query and three
teacher-forced local-field probes were executed.  GR00T output was never sent
to WBC; this packet is not a learned-policy closed loop or task-success claim.

The exact flow algebra was `x_t=(1-t)x0+t*x1`, `v_target=x1-x0`, and
`x1_hat=x_t+(1-t)*v_hat`.  Raw arrays were sealed before decoding.  Mean FR
world-target progress from the measured q0 EEF was **{mean_fr_progress:.6f} m**;
the corresponding teacher-probe mean (repeated per t) was
**{mean_teacher_progress:.6f} m**.  These are descriptive condition-level
metrics only; no threshold or winner is inferred.  Label normalization/clipping
is reported separately from model error.

## Counters

`model_construction=1`, `conditioning=10`, `action_head=35`,
`model_side=45`; training/backward/optimizer=0.  Expert-side counters and
SafetyGate decisions remain in the sealed `pi0160_matched_expert_result.json`.

## Single recommendation

Authorize one fixed-budget visited-state expert/local-field supervision
intervention using these matched q0/after-execution pairs before ICL or broad
training; do not change seed, noise, decoder, or SafetyGate automatically.
"""
    conclusion_sha = write_text(output / "one_page_conclusion.md", conclusion)
    evidence = {"model_admission_sha256": admission_sha, "table_sha256": table_sha, "raw_manifest_sha256": raw_manifest_sha, "clipping_sha256": clipping_sha, "trajectory_svg_sha256": trajectory_sha, "conclusion_sha256": conclusion_sha, "raw_rows": {f"{row['condition']}:{row['mode']}:t{row.get('t_bucket', '')}": row["raw_file_sha256"] for row in raw_rows}}
    evidence_sha = write_json(output / "evidence_hashes.json", evidence)
    report = {"schema": "CORELAB_PI0160_MATCHED_RESET_EXPERT_MODEL_RESULT_V1", "status": "COMPLETED", "token": RETURN_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "head": STARTING_HEAD}, "fixed_identities": identities, "expert_root": str(expert_root), "expert_result_sha256": sha256_file(expert_root / "pi0160_matched_expert_result.json"), "runtime": {"model_dtype": model_dtype, "cosmos_binding": cosmos_binding, "processor_binding": processor_binding, "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE}, "conditions": decoded_rows, "clipping_and_label_quality": clipping, "outputs": evidence, "counters": counters, "expert_counters": expert_summary.get("counters", {}), "execution_boundary": {"gr00t_forward": 45, "model_forward": 45, "training": False, "backward": False, "optimizer_step": 0, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "icl": False, "validation_e_read": False, "holdout_f_read": False, "gr00t_output_sent_to_wbc": False, "safety_gate_relaxed": False}, "claim_boundary": "matched independent-expert reference and fixed-N2 q0 FR/local-field characterization; no model-to-WBC closed-loop or task-success claim", "recommendation": "Use matched visited-state expert/local-field supervision as the single next intervention candidate."}
    report_sha = write_json(output / "pi0160_matched_reset_expert_model_result.json", report)
    return {**report, "report_sha256": report_sha, "evidence_hashes_sha256": evidence_sha}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--expert-root", required=True, type=Path)
    parser.add_argument("--stage10-root", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--modality", required=True, type=Path)
    parser.add_argument("--cosmos-snapshot", required=True, type=Path)
    parser.add_argument("--n2-stats", required=True, type=Path)
    parser.add_argument("--noise", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-uuid", default=None)
    return parser


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "token": result["token"], "report_sha256": result["report_sha256"], "counters": result["counters"]}, sort_keys=True), flush=True)
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
