#!/usr/bin/env python3
"""PI0157 fixed-conditioning dependence probe.

This runner is intentionally narrower than the preceding path-parity probes.
It reuses the accepted C1 sampler and the sealed N2 seed2223@2000 checkpoint,
then asks whether the decoded action changes when the four native TRAIN
observations, the complete StateV1, or the two camera pairs are changed.  The
T/C baseline raw arrays are immutable and are referenced rather than rerun.

The module has no top-level Torch/GR00T import.  CPU/static identity checks run
before a model is constructed; raw arrays are written immediately after each
query and all decoding/plots happen only after the model loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


PI_GATE = "PI_GATE_STATE_0157"
PI_COMMENT = 5651062080
PR_COMMENT = 5651064217
RETURN_TOKEN = "CORELAB_PI0157_CONDITIONING_DEPENDENCE_FOR_PI_REVIEW"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
STARTING_HEAD = "f4fdb01e5071b530d6997ec2d0a17666cdeacddb"
# Kept as an alias for downstream static readers of the accepted PI0157
# starting identity.  The executable commit necessarily advances when this
# runner and its manifest are committed, so execution verifies a concrete
# 40-hex HEAD and records both identities instead of rejecting its own commit.
EXPECTED_HEAD = STARTING_HEAD

CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
BASE_PROCESSOR_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_STATISTICS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
NOISE_FILE_SHA = "83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922"
TRAIN_INPUT_FILE_SHA = "3b525a5a628d3a9ee71894fdd9c6c77b2e888406e87926594700314768181866"
C0_INPUT_FILE_SHA = "67ca605ca90e05633b22dd0447e4873ee29c6a4b31be983b94c11db35f9932be"
TRAIN_RAW_ARRAY_SHA = "2c4fab5f17b8d8d6621278be210849de2fffe44e6107d8bea9dfb250dcc6c47c"
TRAIN_RAW_FILE_SHA = "cd05fdc101cdd2f54b173a98f72d9bf80d4143ddeda703d618659988ccd1b7c2"
C0_RAW_ARRAY_SHA = "e58e5d31a50a506e6840fd6b08aca2d1214d836c5ce279add53b016afc71f690"
C0_RAW_FILE_SHA = "64a7c832ff7e290575ca8ad52c735b16009884a2af9bc00b2ee2d18301c4e834"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
AUTHORIZED_GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
LANGUAGE = "move the end effector to the red target"
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
PHYSICAL_DIM = 10
RAW_SHAPE = (1, HORIZON, ACTION_DIM)
TIMESTAMPS = np.arange(HORIZON, dtype=np.float64) * 0.05
TARGET_WORLD = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
# The four native TRAIN snapshots intentionally cover four distinct target
# fixtures.  These coordinates are metadata-only and are never passed to the
# model; binding them here prevents a mislabeled snapshot from entering the
# paired expert comparison while preserving the pre-registered target order.
NATIVE_TARGETS = {
    "a": np.asarray([0.30, 0.10, 0.58], dtype=np.float64),
    "b": np.asarray([0.30, -0.10, 0.58], dtype=np.float64),
    "c": np.asarray([0.30, 0.05, 0.64], dtype=np.float64),
    "d": np.asarray([0.30, -0.05, 0.63], dtype=np.float64),
}
C0_MEASURED_EEF_WORLD = np.asarray([0.3318595886230469, -0.11061844229698181, 0.655968964099884], dtype=np.float64)
MAX_QUERIES = 10
MAX_CONDITIONING = 10
MAX_HEAD_CALLS = 40
MAX_MODEL_SIDE_CALLS = 50
NOMINAL_QUERIES = 8
NOMINAL_HEAD_CALLS = 32
NOMINAL_MODEL_SIDE_CALLS = 40
PRIOR_CLOSED_DIAGNOSTIC = {"queries": 5, "conditioning": 5, "action_head_calls": 16, "model_side_calls": 21}


class ProbeError(RuntimeError):
    """Fail-closed PI0157 predicate failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_text(path: Path, text: str) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _array_summary(value: Any) -> dict[str, Any]:
    """Small, dtype-aware summaries; Torch is imported only on execution."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu")
        numeric = tensor.float().contiguous().numpy()
        raw_bytes = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        result: dict[str, Any] = {
            "kind": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "raw_dtype_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "numeric_float32_sha256": sha256_array(numeric),
            "finite": bool(np.isfinite(numeric).all()),
        }
        if numeric.size:
            result.update({"min": float(np.min(numeric)), "max": float(np.max(numeric)), "mean": float(np.mean(numeric))})
        return result
    if isinstance(value, np.ndarray):
        result = {"kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value)}
        if np.issubdtype(value.dtype, np.number):
            result["finite"] = bool(np.isfinite(value).all())
            if value.size:
                result.update({"min": float(np.min(value)), "max": float(np.max(value)), "mean": float(np.mean(value))})
        return result
    if isinstance(value, Mapping):
        return {str(k): _array_summary(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "length": len(value), "items": [_array_summary(v) for v in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__, "repr": repr(value)[:300]}


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


def _assert_model_input_boundary(inputs: Mapping[str, Any], label: str) -> None:
    forbidden = {"target_position_world_m", "target_xyz", "target_world_m", "target_id", "target_position", "expert", "label", "future_desired_eef_9d"}
    leaked = sorted(forbidden.intersection(_model_input_keys(inputs)))
    require(not leaked, f"{label} target/metadata/label leaked into model input: {leaked}")


def _load_npy(path: Path, shape: tuple[int, ...], expected_array_sha: str, expected_file_sha: str | None, label: str) -> np.ndarray:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    if expected_file_sha is not None:
        actual_file_sha = sha256_file(path)
        require(actual_file_sha == expected_file_sha, f"{label} file identity drift: {actual_file_sha}")
    value = np.load(path, allow_pickle=False)
    require(not isinstance(value, np.lib.npyio.NpzFile), f"{label} must be .npy")
    value = np.asarray(value)
    require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), f"{label} ABI drift: {value.shape} {value.dtype}")
    actual_array_sha = sha256_array(value)
    require(actual_array_sha == expected_array_sha, f"{label} array identity drift: {actual_array_sha}")
    return value


def _load_observation(path: Path, expected_file_sha: str, label: str) -> dict[str, Any]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    actual_file_sha = sha256_file(path)
    require(actual_file_sha == expected_file_sha, f"{label} file identity drift: {actual_file_sha}")
    with np.load(path, allow_pickle=False) as data:
        require({"body_rgb", "wrist_rgb", "state"}.issubset(data.files), f"{label} keys missing: {data.files}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        extras = sorted(set(data.files).difference({"body_rgb", "wrist_rgb", "state"}))
        target = None if "target_position_world_m" not in data.files else np.asarray(data["target_position_world_m"], dtype=np.float64).reshape(-1)
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, f"{label} body ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, f"{label} wrist ABI drift")
    require(state.shape == (STATE_DIM,) and np.isfinite(state).all(), f"{label} state ABI drift")
    if target is not None:
        require(target.shape == (3,) and np.isfinite(target).all(), f"{label} target ABI drift")
    return {
        "label": label,
        "path": str(path),
        "file_sha256": actual_file_sha,
        "body_rgb": body,
        "wrist_rgb": wrist,
        "state": state,
        "body_sha256": sha256_array(body),
        "wrist_sha256": sha256_array(wrist),
        "state_sha256": sha256_array(state),
        "target_keys": extras,
        "target_position_world_m": None if target is None else target.tolist(),
    }


def _ppm_rgb(path: Path) -> np.ndarray:
    """Read a binary P6 PPM without Pillow/image/video dependencies."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"PPM is not a regular file: {path}")
    payload = path.read_bytes()
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(payload) and payload[index] in b" \t\r\n":
            index += 1
        require(index < len(payload), f"PPM header truncated: {path}")
        if payload[index:index + 1] == b"#":
            end = payload.find(b"\n", index)
            index = len(payload) if end < 0 else end + 1
            continue
        end = index
        while end < len(payload) and payload[end] not in b" \t\r\n":
            end += 1
        tokens.append(payload[index:end])
        index = end
    require(tokens[0] == b"P6" and tokens[1:4] == [b"96", b"96", b"255"], f"PPM ABI drift: {path} header={tokens[:4]}")
    while index < len(payload) and payload[index] in b" \t\r\n":
        index += 1
    raw = payload[index:]
    require(len(raw) == 96 * 96 * 3, f"PPM payload size drift: {path} {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(96, 96, 3).copy()


def _load_native_snapshot(path: Path, expected_id: str) -> dict[str, Any]:
    path = Path(path).resolve()
    doc = read_json(path)
    require(doc.get("example_id") == expected_id, f"native example id drift: {path}")
    state_doc = doc.get("state_v1", {})
    state = np.asarray(state_doc.get("values"), dtype=np.float32)
    require(state.shape == (STATE_DIM,) and np.isfinite(state).all(), f"native StateV1 drift: {path}")
    require(state_doc.get("layout") == ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"], f"native StateV1 layout drift: {path}")
    require(doc.get("language") is None or doc.get("language") == LANGUAGE, f"native language drift: {path}")
    require(doc.get("target_excluded_from_model_input") is True, f"native target exclusion marker missing: {path}")
    target = np.asarray(doc.get("target_position_world_m"), dtype=np.float64)
    require(target.shape == (3,) and np.isfinite(target).all(), f"native target identity drift: {path}")
    suffix = expected_id.split("_train_", 1)[-1].split(":", 1)[0]
    require(suffix in NATIVE_TARGETS and np.array_equal(target, NATIVE_TARGETS[suffix]), f"native target identity drift: {path}")
    body_path = Path(doc.get("body_frame_path", "")).resolve()
    wrist_path = Path(doc.get("wrist_frame_path", "")).resolve()
    require(body_path.parent == path.parent and wrist_path.parent == path.parent, f"native frame path escaped snapshot: {path}")
    body = _ppm_rgb(body_path)
    wrist = _ppm_rgb(wrist_path)
    provenance = doc.get("provenance", {})
    if provenance.get("body_frame_sha256"):
        require(sha256_file(body_path) == provenance["body_frame_sha256"], f"native body file hash drift: {body_path}")
    if provenance.get("wrist_frame_sha256"):
        require(sha256_file(wrist_path) == provenance["wrist_frame_sha256"], f"native wrist file hash drift: {wrist_path}")
    expert_xyz = np.asarray(doc.get("future_desired_eef_9d"), dtype=np.float32)
    expert_gripper = np.asarray(doc.get("future_gripper"), dtype=np.float32)
    timestamps = np.asarray(doc.get("future_timestamps"), dtype=np.float64)
    require(expert_xyz.shape == (HORIZON, 9) and np.isfinite(expert_xyz).all(), f"native expert pose shape drift: {path}")
    require(expert_gripper.shape == (HORIZON, 1) and np.isfinite(expert_gripper).all(), f"native expert gripper shape drift: {path}")
    require(timestamps.shape == (HORIZON,) and np.allclose(timestamps, TIMESTAMPS, rtol=0.0, atol=1e-9), f"native timestamps drift: {path}")
    expert = np.concatenate([expert_xyz, expert_gripper], axis=1).astype(np.float64)
    return {
        "label": expected_id,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "episode_id": doc.get("raw_episode_id"),
        "body_path": str(body_path),
        "wrist_path": str(wrist_path),
        "body_file_sha256": sha256_file(body_path),
        "wrist_file_sha256": sha256_file(wrist_path),
        "body": body,
        "wrist": wrist,
        "state": state,
        "state_sha256": sha256_array(state),
        "body_sha256": sha256_array(body),
        "wrist_sha256": sha256_array(wrist),
        "expert": expert,
        "expert_sha256": sha256_array(expert),
        "target_id": doc.get("target_id"),
        "target_world_m": target.tolist(),
        "language": doc.get("language", LANGUAGE),
    }


def _gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-500:]}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    selected = next((row for row in rows if row["uuid"] == AUTHORIZED_GPU_UUID), None)
    require(selected is not None, f"authorized GPU UUID unavailable: {AUTHORIZED_GPU_UUID}")
    require(selected["memory_free_mib"] >= 14 * 1024, f"GPU admission requires >=14GiB free: {selected}")
    return {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "query_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(), "authorized_uuid": AUTHORIZED_GPU_UUID, "selected": selected, "all": rows}


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"checkpoint root unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(stat.S_ISDIR(child.lstat().st_mode) and not child.is_symlink(), f"checkpoint directory symlink/unsafe: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(stat.S_ISREG(child.lstat().st_mode) and not child.is_symlink(), f"checkpoint file symlink/unsafe: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: item["path"])


def _tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def _verify_checkpoint(checkpoint: Path, stats_path: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    stats_path = Path(stats_path).resolve()
    entries = _tree_entries(checkpoint, "stage45e/checkpoint-2000")
    tree = _tree_digest(entries)
    require(tree == CHECKPOINT_TREE, f"N2 checkpoint tree identity drift: {tree}")
    stats_sha = sha256_file(stats_path)
    require(stats_sha == N2_STATS_SHA, f"N2 stats identity drift: {stats_sha}")
    stats = read_json(stats_path)
    require(stats.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats payload identity drift")
    return {"checkpoint_tree_sha256": tree, "checkpoint_file_count": len(entries), "checkpoint_total_bytes": sum(int(e["bytes"]) for e in entries), "stats_sha256": stats_sha, "stats_payload_sha256": stats["statistics_payload_sha256"], "checkpoint_path": str(checkpoint), "stats_path": str(stats_path)}


def _verify_base_processor(base_checkpoint: Path) -> dict[str, Any]:
    """Bind the immutable BASE processor metadata used to build inference inputs."""
    base_checkpoint = Path(base_checkpoint).resolve()
    require(base_checkpoint.is_dir() and not base_checkpoint.is_symlink(), f"BASE processor root unavailable: {base_checkpoint}")
    processor_config = base_checkpoint / "processor_config.json"
    statistics = base_checkpoint / "statistics.json"
    require(processor_config.is_file() and not processor_config.is_symlink(), f"BASE processor_config unavailable: {processor_config}")
    require(statistics.is_file() and not statistics.is_symlink(), f"BASE statistics unavailable: {statistics}")
    processor_sha = sha256_file(processor_config)
    statistics_sha = sha256_file(statistics)
    require(processor_sha == BASE_PROCESSOR_CONFIG_SHA, f"BASE processor_config identity drift: {processor_sha}")
    require(statistics_sha == BASE_STATISTICS_SHA, f"BASE statistics identity drift: {statistics_sha}")
    return {
        "base_processor_root": str(base_checkpoint),
        "base_processor_config_sha256": processor_sha,
        "base_statistics_sha256": statistics_sha,
    }


def _load_previous_probe():
    path = SCRIPTS / "stage45f_pi0153_generation_probe.py"
    spec = importlib.util.spec_from_file_location("pi0157_previous_probe", path)
    require(spec is not None and spec.loader is not None, f"cannot load accepted sampler helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu").float().contiguous().numpy()
    return np.asarray(value)


def _state_audit(processor: Any, embodiment: Any, state: np.ndarray, inputs: Mapping[str, Any], label: str) -> dict[str, Any]:
    tag = getattr(embodiment, "value", str(embodiment))
    raw_groups = {"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]}
    sap = processor.state_action_processor
    applied = sap.apply_state(raw_groups, tag)
    reconstructed_groups = sap.unapply_state(applied, tag)
    reconstructed = np.concatenate([np.asarray(reconstructed_groups["eef_9d"]), np.asarray(reconstructed_groups["gripper_position"])], axis=-1).reshape(-1)
    processed_expected = np.concatenate([np.asarray(applied["eef_9d"]), np.asarray(applied["gripper_position"])], axis=-1).reshape(-1)
    state_tensor = inputs.get("state")
    require(state_tensor is not None, f"{label}: processed state tensor missing")
    processed_full = _tensor_numpy(state_tensor).reshape(-1)
    require(processed_full.size >= STATE_DIM, f"{label}: processed state width drift: {processed_full.shape}")
    prefix = processed_full[:STATE_DIM]
    require(np.allclose(prefix, processed_expected, rtol=0.0, atol=1e-6), f"{label}: processor state prefix mismatch")
    stats_source: dict[str, Any] = {}
    norm_ranges: dict[str, Any] = {}
    clipped: dict[str, Any] = {}
    for key in ("eef_9d", "gripper_position"):
        params = sap.norm_params[tag]["state"][key]
        stats_source[key] = {name: np.asarray(getattr(sap, "statistics")[tag]["state"][key].get(name), dtype=np.float64).reshape(-1).tolist() for name in ("min", "max", "mean", "std", "q01", "q99") if name in getattr(sap, "statistics")[tag]["state"][key]}
        minimum = np.asarray(params["min"], dtype=np.float64).reshape(-1)
        maximum = np.asarray(params["max"], dtype=np.float64).reshape(-1)
        source = raw_groups[key].reshape(-1).astype(np.float64)
        denom = np.maximum(maximum - minimum, 1e-8)
        preclip = (source - minimum) / denom * 2.0 - 1.0
        clipped[key] = {"preclip_out_of_range_count": int(np.count_nonzero((preclip < -1.0) | (preclip > 1.0))), "preclip_min": float(np.min(preclip)), "preclip_max": float(np.max(preclip))}
        norm_ranges[key] = {"min": minimum.tolist(), "max": maximum.tolist(), "zero_range_dims": np.flatnonzero((maximum - minimum) <= 1e-8).tolist()}
    roundtrip_delta = reconstructed - state.astype(np.float64)
    information_loss = bool(np.max(np.abs(roundtrip_delta)) > 1e-5)
    return {
        "label": label,
        "state_v1_sha256": sha256_array(state),
        "state_v1_values": state.astype(np.float64).tolist(),
        "processed_state_shape": list(processed_full.shape),
        "processed_state_dtype": str(getattr(state_tensor, "dtype", "unknown")),
        "processed_state_prefix_sha256": sha256_array(prefix.astype(np.float32)),
        "processed_state_full_sha256": sha256_array(processed_full.astype(np.float32)),
        "processed_state_padding_count": int(max(0, processed_full.size - STATE_DIM)),
        "processed_state_padding_nonzero": int(np.count_nonzero(np.abs(processed_full[STATE_DIM:]) > 0.0)),
        "normalization_source": "processor.state_action_processor.statistics[NEW_EMBODIMENT].state",
        "normalization_use_percentiles": bool(getattr(sap, "use_percentiles", False)),
        "clip_outliers": bool(getattr(sap, "clip_outliers", False)),
        "normalization_ranges": norm_ranges,
        "stats_source": stats_source,
        "clipping": clipped,
        "roundtrip_raw_state_max_abs": float(np.max(np.abs(roundtrip_delta))),
        "roundtrip_raw_state_rmse": float(np.sqrt(np.mean(roundtrip_delta * roundtrip_delta))),
        "information_loss_observed": information_loss,
        "information_loss_interpretation": "clipping/saturation or non-reversible transform observed" if information_loss else "no scalar loss in processor round-trip within 1e-5",
    }


def _delta_summary(delta: Any) -> dict[str, Any]:
    value = np.asarray(delta, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(value))), "rmse": float(np.sqrt(np.mean(value * value))), "count": int(value.size), "equal_bytes": bool(np.array_equal(value, np.zeros_like(value)))}


def _raw_blocks(delta: np.ndarray, active_mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(delta, dtype=np.float64)
    return {"active_190": _delta_summary(value[0][active_mask]), "xyz": _delta_summary(value[0][:, :3]), "rot6d": _delta_summary(value[0][:, 3:9]), "timing": _delta_summary(value[0][:, 9]), "gripper": _delta_summary(value[0][:, 10]), "padding": _delta_summary(value[0][:, 11:]), "full": _delta_summary(value)}


def _expert_metrics(prediction: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(expert, dtype=np.float64)
    delta = pred - ref
    def length(x: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1)))
    second_pred = np.linalg.norm(np.diff(pred[:, :3], n=2, axis=0), axis=1)
    second_ref = np.linalg.norm(np.diff(ref[:, :3], n=2, axis=0), axis=1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(delta[:, :3] ** 2))),
        "position_max_abs_m": float(np.max(np.abs(delta[:, :3]))),
        "endpoint_error_m": float(np.linalg.norm(delta[-1, :3])),
        "eef_rot6d_rmse": float(np.sqrt(np.mean(delta[:, 3:9] ** 2))),
        "gripper_rmse": float(np.sqrt(np.mean(delta[:, 9] ** 2))),
        "prediction_path_length_m": length(pred),
        "expert_path_length_m": length(ref),
        "prediction_roughness_mean": float(np.mean(second_pred)),
        "expert_roughness_mean": float(np.mean(second_ref)),
        "world_target_metrics_status": "UNVERIFIED",
        "world_target_metrics_reason": "TRAIN source frame is relative/reference-derived; this packet does not promote it to measured world EEF",
    }


def _trajectory_metrics(positions: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    distances = np.linalg.norm(np.asarray(positions, dtype=np.float64) - target[None, :], axis=1)
    displacement = np.asarray(positions, dtype=np.float64)[-1] - positions[0]
    target_vector = target - positions[0]
    denom = np.linalg.norm(displacement) * np.linalg.norm(target_vector)
    return {
        "initial_target_distance_m": float(distances[0]),
        "minimum_target_distance_m": float(np.min(distances)),
        "minimum_target_index": int(np.argmin(distances)),
        "final_target_distance_m": float(distances[-1]),
        "absolute_progress_m": float(distances[0] - distances[-1]),
        "best_progress_m": float(distances[0] - np.min(distances)),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "endpoint_target_vector_cosine": None if denom <= 1e-12 else float(np.dot(displacement, target_vector) / denom),
        "sample_count": int(len(positions)),
    }


def _decode(raw: np.ndarray, state: np.ndarray, stats_path: Path, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    import go2arx5_vla.stage40_structured_decode as decoder
    previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
    decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
    try:
        _, physical, details = decoder.decode_raw_structured_action(raw, state, stats_path=stats_path, source_timestamps=TIMESTAMPS, observation_timestamp=0.0, chunk_id=label)
    finally:
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
    value = np.asarray(physical, dtype=np.float64)
    require(value.shape == (HORIZON, PHYSICAL_DIM) and np.isfinite(value).all(), f"{label}: decoded physical ABI drift")
    return value, details


def _make_svg(paths: Mapping[str, np.ndarray], path: Path) -> str:
    width, height = 860, 520
    all_xyz = np.concatenate([np.asarray(v)[:, :3] for v in paths.values()], axis=0)
    lo = np.min(all_xyz[:, :2], axis=0)
    hi = np.max(all_xyz[:, :2], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#7f7f7f", "#8c564b", "#e377c2", "#bcbd22"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="20" y="24" font-family="sans-serif" font-size="16">PI0157 decoded XYZ (standardized T anchor diagnostic)</text>']
    for index, (label, values) in enumerate(paths.items()):
        points = []
        for point in np.asarray(values)[:, :2]:
            x = 50 + 740 * (float(point[0]) - lo[0]) / span[0]
            y = 460 - 390 * (float(point[1]) - lo[1]) / span[1]
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[index % len(colors)]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.4" opacity="0.85"/>')
        lines.append(f'<text x="610" y="{48 + 17 * index}" fill="{color}" font-family="sans-serif" font-size="11">{label}</text>')
    tx = 50 + 740 * (TARGET_WORLD[0] - lo[0]) / span[0]
    ty = 460 - 390 * (TARGET_WORLD[1] - lo[1]) / span[1]
    lines.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" fill="#000"/><text x="{tx + 8:.2f}" y="{ty - 8:.2f}" font-family="sans-serif" font-size="11">target XY (world marker; not used for cross-condition scores)</text>')
    lines.append("</svg>")
    return write_text(path, "\n".join(lines) + "\n")


def _acquire_attempt(control_root: Path, output_root: Path, gpu: Mapping[str, Any], repository_head: str) -> tuple[Path, dict[str, Any]]:
    control_root = Path(control_root).resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    lock = control_root / "stage45f_pi0157_attempt.lock.json"
    payload = {"schema": "CORELAB_STAGE45F_PI0157_ATTEMPT_LOCK_V1", "status": "STARTED", "token": RETURN_TOKEN, "output_root": str(output_root), "branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "repository_head": repository_head, "gpu_uuid": gpu["selected"]["uuid"], "created_unix": time.time()}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise ProbeError(f"PI0157 attempt already consumed: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return lock, payload


def _finish_attempt(lock: Path, payload: Mapping[str, Any], **updates: Any) -> None:
    value = dict(payload)
    value.update(updates)
    Path(lock).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _condition_item(label: str, body: np.ndarray, wrist: np.ndarray, state: np.ndarray, *, source: str, source_label: str, expert: np.ndarray | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    require(body.shape == (96, 96, 3) and wrist.shape == (96, 96, 3) and state.shape == (STATE_DIM,), f"{label}: condition ABI drift")
    return {"label": label, "body": np.asarray(body, dtype=np.uint8), "wrist": np.asarray(wrist, dtype=np.uint8), "state": np.asarray(state, dtype=np.float32), "source": source, "source_label": source_label, "expert": expert, "metadata": dict(metadata or {})}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    execution_head = str(args.repository_head).lower()
    require(len(execution_head) == 40 and all(char in "0123456789abcdef" for char in execution_head), f"repository HEAD must be a concrete 40-hex SHA: {args.repository_head}")

    gpu = _gpu_snapshot()
    checkpoint = Path(args.model_path).resolve()
    base_checkpoint = Path(args.base_model).resolve()
    stats_path = Path(args.n2_stats).resolve()
    identities = _verify_checkpoint(checkpoint, stats_path)
    base_identity = _verify_base_processor(base_checkpoint)
    noise = _load_npy(Path(args.noise), RAW_SHAPE, NOISE_ARRAY_SHA, NOISE_FILE_SHA, "fixed x0")
    train_npz = _load_observation(Path(args.train_input), TRAIN_INPUT_FILE_SHA, "canonical TRAIN/T")
    c0_npz = _load_observation(Path(args.archived_c0_input), C0_INPUT_FILE_SHA, "archived C0/C")

    natives = {}
    native_args = {"a": args.native_a, "b": args.native_b, "c": args.native_c, "d": args.native_d}
    for suffix, native_path in native_args.items():
        natives[suffix] = _load_native_snapshot(Path(native_path), f"stage22_expert_red_target_train_{suffix}:expert-0000")
    native_a_matches_t = bool(np.array_equal(natives["a"]["body"], train_npz["body_rgb"]) and np.array_equal(natives["a"]["wrist"], train_npz["wrist_rgb"]) and np.array_equal(natives["a"]["state"], train_npz["state"]))

    # This is the only point at which a new attempt is admitted.  All identity
    # and metadata exclusion checks below occur before model construction.
    lock, lock_payload = _acquire_attempt(Path(args.control_root), output, gpu, execution_head)
    counters = {"queries": 0, "conditioning": 0, "conditioning_completed": 0, "conditioning_unknown": 0, "action_head_calls": 0, "model_side_calls": 0, "exceptions": 0, "model_construction_attempts": 0}
    admission = {
        "schema": "CORELAB_STAGE45F_PI0157_CONDITIONING_ADMISSION_V1",
        "status": "ADMITTED_FOR_BOUNDED_EXECUTION",
        "token": "CORELAB_PI0157_CONDITIONING_ADMISSION_RECORDED",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {"branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "execution_head": execution_head, "head": execution_head},
        "runtime": {"gpu": gpu, "python": sys.executable, "no_model_forward_before_admission": True},
        "raw_first": True,
        "fixed_identity": {**identities, **base_identity, "noise_array_sha256": NOISE_ARRAY_SHA, "noise_file_sha256": NOISE_FILE_SHA, "train_file_sha256": TRAIN_INPUT_FILE_SHA, "c0_file_sha256": C0_INPUT_FILE_SHA, "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE, "modality_sha256": MODALITY_SHA},
        "input_order": ["N1_NATIVE_a", "N2_NATIVE_b", "N3_NATIVE_c", "N4_NATIVE_d", "X1_VT_SC", "X2_VC_ST", "Z1_CONST128_ST", "Z2_CONST128_SC"],
        "native_a_matches_canonical_T": native_a_matches_t,
        "budget": {"nominal_queries": NOMINAL_QUERIES, "nominal_action_head_calls": NOMINAL_HEAD_CALLS, "nominal_model_side_calls": NOMINAL_MODEL_SIDE_CALLS, "max_queries_including_mechanical": MAX_QUERIES, "max_action_head_calls": MAX_HEAD_CALLS, "max_model_side_calls": MAX_MODEL_SIDE_CALLS, "prior_closed_diagnostic": PRIOR_CLOSED_DIAGNOSTIC},
        "forbidden": {"training": False, "icl": False, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "safety_gate_relaxed": False},
    }
    admission_sha = write_json(output / "admission.json", admission)

    old = _load_previous_probe()
    # Offline nested-Cosmos binding is the accepted PI0156 mechanical helper;
    # it is called only after the CPU gate and never changes credentials.  The
    # imported C1 helper encloses the complete query in torch.inference_mode().
    cosmos_binding = None
    processor_binding = None
    model = None
    try:
        cosmos_binding = old._bind_local_cosmos_runtime(Path(args.stage10_root), Path(args.cosmos_snapshot))
        import torch
        from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
        from stage38_structured_offline_evaluation import _load_structured_model
        from stage45c_h200_eval_runtime_adapter import _sample_one
        from go2arx5_vla.structured_action import structured_action_mask

        processor_binding = __import__("stage40_b0_model_control", fromlist=["_bind_local_cosmos_processor"])._bind_local_cosmos_processor(Path(args.stage10_root), Path(args.cosmos_snapshot))
        processor, embodiment, _ = _build_runtime_processor(groot_source=Path(args.stage10_root) / "src" / "isaac-gr00t", base_checkpoint=base_checkpoint, dataset_dir=Path(args.dataset), modality_path=Path(args.modality), cosmos_snapshot=Path(args.cosmos_snapshot))

        # Build every condition with the real processor before any forward.
        items = {
            "T_BASELINE": _condition_item("T_BASELINE", train_npz["body_rgb"], train_npz["wrist_rgb"], train_npz["state"], source="sealed_PI0156_raw", source_label="TRAIN/T"),
            "C_BASELINE": _condition_item("C_BASELINE", c0_npz["body_rgb"], c0_npz["wrist_rgb"], c0_npz["state"], source="sealed_PI0156_raw", source_label="C0/C"),
            "N1_NATIVE_a": _condition_item("N1_NATIVE_a", natives["a"]["body"], natives["a"]["wrist"], natives["a"]["state"], source="Stage22_native_expert", source_label=natives["a"]["label"], expert=natives["a"]["expert"], metadata={"target_id": natives["a"]["target_id"]}),
            "N2_NATIVE_b": _condition_item("N2_NATIVE_b", natives["b"]["body"], natives["b"]["wrist"], natives["b"]["state"], source="Stage22_native_expert", source_label=natives["b"]["label"], expert=natives["b"]["expert"], metadata={"target_id": natives["b"]["target_id"]}),
            "N3_NATIVE_c": _condition_item("N3_NATIVE_c", natives["c"]["body"], natives["c"]["wrist"], natives["c"]["state"], source="Stage22_native_expert", source_label=natives["c"]["label"], expert=natives["c"]["expert"], metadata={"target_id": natives["c"]["target_id"]}),
            "N4_NATIVE_d": _condition_item("N4_NATIVE_d", natives["d"]["body"], natives["d"]["wrist"], natives["d"]["state"], source="Stage22_native_expert", source_label=natives["d"]["label"], expert=natives["d"]["expert"], metadata={"target_id": natives["d"]["target_id"]}),
            "X1_VT_SC": _condition_item("X1_VT_SC", train_npz["body_rgb"], train_npz["wrist_rgb"], c0_npz["state"], source="counterfactual", source_label="V_T,S_C"),
            "X2_VC_ST": _condition_item("X2_VC_ST", c0_npz["body_rgb"], c0_npz["wrist_rgb"], train_npz["state"], source="counterfactual", source_label="V_C,S_T"),
            "Z1_CONST128_ST": _condition_item("Z1_CONST128_ST", np.full((96, 96, 3), 128, dtype=np.uint8), np.full((96, 96, 3), 128, dtype=np.uint8), train_npz["state"], source="fixed_constant128_ood", source_label="CONST128,S_T"),
            "Z2_CONST128_SC": _condition_item("Z2_CONST128_SC", np.full((96, 96, 3), 128, dtype=np.uint8), np.full((96, 96, 3), 128, dtype=np.uint8), c0_npz["state"], source="fixed_constant128_ood", source_label="CONST128,S_C"),
        }
        input_records: dict[str, dict[str, Any]] = {}
        for label, item in items.items():
            model_inputs = old._build_c1_inputs(processor, embodiment, {"body": item["body"], "wrist": item["wrist"], "state": item["state"]}, train=True)
            _assert_model_input_boundary(model_inputs, label)
            state_audit = _state_audit(processor, embodiment, item["state"], model_inputs, label)
            input_records[label] = {"label": label, "source": item["source"], "source_label": item["source_label"], "body_sha256": sha256_array(item["body"]), "wrist_sha256": sha256_array(item["wrist"]), "state_sha256": sha256_array(item["state"]), "input_keys": sorted(_model_input_keys(model_inputs)), "target_xyz_in_model_input": False, "input_summary": old._array_summary(model_inputs), "state_audit": state_audit, "model_inputs": model_inputs}

        active_mask = np.asarray(structured_action_mask("T16"), dtype=bool)
        require(active_mask.shape == RAW_SHAPE[1:] and int(np.count_nonzero(active_mask)) == 190, "StructuredAction active mask drift")
        try:
            model = _load_structured_model(checkpoint, f"cuda:{gpu['selected']['index']}")
            counters["model_construction_attempts"] = 1
        except Exception as exc:
            counters["exceptions"] += 1
            failure = {"schema": "CORELAB_STAGE45F_PI0157_FAILURE_SEAL_V1", "status": "BLOCKED", "phase": "model_construction", "error_type": type(exc).__name__, "error": str(exc)[-3000:], "counters": counters, "gpu": gpu, "no_retry": True, "execution_boundary": {"training": False, "icl": False, "wbc": False, "simulator": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False}}
            failure_sha = write_json(output / "failure_seal.json", failure)
            _finish_attempt(lock, lock_payload, status="BLOCKED_MODEL_CONSTRUCTION_FAILURE", counters=counters, failure_sha256=failure_sha)
            raise
        model.eval()
        model_dtype = str(next(model.parameters()).dtype)
        require(model_dtype == "torch.bfloat16", f"model dtype drift: {model_dtype}")
        query_labels = ["N1_NATIVE_a", "N2_NATIVE_b", "N3_NATIVE_c", "N4_NATIVE_d", "X1_VT_SC", "X2_VC_ST", "Z1_CONST128_ST", "Z2_CONST128_SC"]
        raw_records: list[dict[str, Any]] = []
        # The canonical T/C arrays are immutable references.  N1 reuses T only
        # when all three actual native arrays match, otherwise it consumes its
        # registered query slot and remains a proper materialization contrast.
        reused = {
            "T_BASELINE": Path(args.raw_t),
            "C_BASELINE": Path(args.raw_c),
        }
        if native_a_matches_t:
            reused["N1_NATIVE_a"] = Path(args.raw_t)
        for label, raw_path in reused.items():
            if label == "T_BASELINE":
                raw = _load_npy(raw_path, RAW_SHAPE, TRAIN_RAW_ARRAY_SHA, TRAIN_RAW_FILE_SHA, label)
            elif label == "C_BASELINE":
                raw = _load_npy(raw_path, RAW_SHAPE, C0_RAW_ARRAY_SHA, C0_RAW_FILE_SHA, label)
            else:
                raw = _load_npy(raw_path, RAW_SHAPE, TRAIN_RAW_ARRAY_SHA, TRAIN_RAW_FILE_SHA, label)
            raw_records.append({"label": label, "status": "REUSED_SEALED_RAW", "raw_path": str(raw_path), "raw_file_sha256": sha256_file(raw_path), "raw_array_sha256": sha256_array(raw), "raw": raw, "query": False, "conditioning": 0, "action_head_calls": 0})

        for label in query_labels:
            if label == "N1_NATIVE_a" and native_a_matches_t:
                continue
            require(counters["queries"] < MAX_QUERIES, "PI0157 query budget exhausted")
            require(counters["model_side_calls"] + 5 <= MAX_MODEL_SIDE_CALLS, "PI0157 model-side budget exhausted before query")
            counters["queries"] += 1
            counters["conditioning"] += 1
            counters["model_side_calls"] += 1
            start = time.perf_counter()
            try:
                raw, details, calls = old._c1_call(model, input_records[label]["model_inputs"], noise.copy(), torch, _sample_one)
            except Exception as exc:
                counters["exceptions"] += 1
                counters["conditioning_unknown"] += 1
                failure = {"schema": "CORELAB_STAGE45F_PI0157_FAILURE_SEAL_V1", "status": "PARTIAL", "phase": "model_query", "condition": label, "error_type": type(exc).__name__, "error": str(exc)[-3000:], "counters": counters, "gpu": gpu, "no_retry": True, "execution_boundary": {"training": False, "icl": False, "wbc": False, "simulator": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False}}
                write_json(output / "failure_seal.json", failure)
                _finish_attempt(lock, lock_payload, status="PARTIAL_MODEL_QUERY_FAILURE", counters=counters, failure_sha256=sha256_file(output / "failure_seal.json"))
                raise
            elapsed = time.perf_counter() - start
            counters["conditioning_completed"] += 1
            counters["action_head_calls"] += int(calls)
            counters["model_side_calls"] += int(calls)
            require(int(calls) == 4, f"{label}: sampler head count drift: {calls}")
            require(counters["action_head_calls"] <= MAX_HEAD_CALLS and counters["model_side_calls"] <= MAX_MODEL_SIDE_CALLS, "PI0157 budget exceeded after query")
            raw_path = output / f"raw_{len(raw_records):02d}_{label}.npy"
            np.save(raw_path, np.asarray(raw, dtype=np.float32), allow_pickle=False)
            raw_disk = np.asarray(np.load(raw_path, allow_pickle=False), dtype=np.float32)
            require(np.array_equal(raw_disk, raw), f"{label}: raw immediate round-trip mismatch")
            raw_records.append({"label": label, "status": "NEW_QUERY_COMPLETED", "raw_path": str(raw_path), "raw_file_sha256": sha256_file(raw_path), "raw_array_sha256": sha256_array(raw_disk), "raw": raw_disk, "query": True, "conditioning": 1, "action_head_calls": int(calls), "latency_s": elapsed, "path_details": details})

        # CPU-only decoding starts after the model loop and never triggers a
        # model call.  Each result gets an actual-anchor and a fixed-T-anchor
        # view so p0 movement cannot masquerade as learned state response.
        records_by_label = {row["label"]: row for row in raw_records}
        decoded: dict[str, dict[str, Any]] = {}
        for label, row in records_by_label.items():
            item = items[label]
            actual, details_actual = _decode(row["raw"], item["state"], stats_path, f"PI0157:{label}:actual")
            standardized, details_standardized = _decode(row["raw"], train_npz["state"], stats_path, f"PI0157:{label}:standardized-T")
            actual_path = output / f"decoded_actual_{label}.npy"
            standardized_path = output / f"decoded_standardized_T_{label}.npy"
            np.save(actual_path, actual, allow_pickle=False)
            np.save(standardized_path, standardized, allow_pickle=False)
            decoded[label] = {"actual": actual, "standardized_T": standardized, "actual_sha256": sha256_array(actual), "standardized_T_sha256": sha256_array(standardized), "actual_path": str(actual_path), "standardized_T_path": str(standardized_path), "decoder_actual": details_actual, "decoder_standardized_T": details_standardized, "p0_state_residual_m": float(np.linalg.norm(actual[0, :3] - item["state"][:3]))}

        t_raw = records_by_label["T_BASELINE"]["raw"]
        t_std = decoded["T_BASELINE"]["standardized_T"]
        condition_rows = []
        diff_rows = []
        for label, item in items.items():
            row = records_by_label[label]
            d_raw = row["raw"] - t_raw
            d_std = decoded[label]["standardized_T"] - t_std
            condition_rows.append({"condition": label, "source": item["source"], "source_label": item["source_label"], "body_sha256": input_records[label]["body_sha256"], "wrist_sha256": input_records[label]["wrist_sha256"], "state_sha256": input_records[label]["state_sha256"], "status": row["status"], "query": row["query"], "conditioning": row["conditioning"], "action_head_calls": row["action_head_calls"], "raw_array_sha256": row["raw_array_sha256"], "decoded_actual_sha256": decoded[label]["actual_sha256"], "decoded_standardized_T_sha256": decoded[label]["standardized_T_sha256"], "p0_state_residual_m": decoded[label]["p0_state_residual_m"], "raw_active190_max_abs": _raw_blocks(d_raw, active_mask)["active_190"]["max_abs"], "raw_active190_rmse": _raw_blocks(d_raw, active_mask)["active_190"]["rmse"], "standardized_T_xyz_max_abs": _delta_summary(d_std[:, :3])["max_abs"], "standardized_T_xyz_rmse": _delta_summary(d_std[:, :3])["rmse"], "processor_information_loss": input_records[label]["state_audit"]["information_loss_observed"]})
            diff_rows.append({"condition": label, "comparison_to": "T_BASELINE", "raw_blocks": _raw_blocks(d_raw, active_mask), "standardized_T_physical": {"xyz": _delta_summary(d_std[:, :3]), "rot6d": _delta_summary(d_std[:, 3:9]), "gripper": _delta_summary(d_std[:, 9:10]), "full": _delta_summary(d_std)}, "actual_anchor_p0_delta_m": _delta_summary(decoded[label]["actual"][:1, :3] - decoded["T_BASELINE"]["actual"][:1, :3])})

        expert_rows = []
        for suffix, label in (("a", "N1_NATIVE_a"), ("b", "N2_NATIVE_b"), ("c", "N3_NATIVE_c"), ("d", "N4_NATIVE_d")):
            metrics = _expert_metrics(decoded[label]["actual"], natives[suffix]["expert"])
            expert_rows.append({"condition": label, "native_source": natives[suffix]["label"], "expert_sha256": natives[suffix]["expert_sha256"], **metrics})

        c0_world = C0_MEASURED_EEF_WORLD[None, :] + decoded["C_BASELINE"]["actual"][:, :3] - c0_npz["state"][None, :3]
        c0_prefix = _trajectory_metrics(c0_world[:20], TARGET_WORLD)
        c0_tail = _trajectory_metrics(c0_world[20:], TARGET_WORLD)
        c0_indices = {str(index): {"decoded_physical_xyz": decoded["C_BASELINE"]["actual"][index, :3].tolist(), "world_xyz_measured_bridge": c0_world[index].tolist(), "timestamp_s": float(TIMESTAMPS[index]), "target_distance_m": float(np.linalg.norm(c0_world[index] - TARGET_WORLD))} for index in (0, 10, 20, 39)}

        state_audits = {label: input_records[label]["state_audit"] for label in items}
        raw_manifest = [{key: value for key, value in row.items() if key != "raw"} for row in raw_records]
        path_effect_state = _raw_blocks(records_by_label["X1_VT_SC"]["raw"] - t_raw, active_mask)
        path_effect_vision = _raw_blocks(records_by_label["X2_VC_ST"]["raw"] - t_raw, active_mask)
        path_effect_devision = _raw_blocks(records_by_label["Z1_CONST128_ST"]["raw"] - t_raw, active_mask)
        state_nonzero = path_effect_state["active_190"]["max_abs"] > 0.0
        vision_nonzero = path_effect_vision["active_190"]["max_abs"] > 0.0
        devision_nonzero = path_effect_devision["active_190"]["max_abs"] > 0.0
        if state_nonzero and vision_nonzero and devision_nonzero:
            science = "SUPPORTED"
            science_sentence = "The fixed policy is sensitive to both complete StateV1 substitution and paired camera substitution under a common x0; constant-128 inputs also change the active output."
        elif state_nonzero or vision_nonzero or devision_nonzero:
            science = "MIXED"
            science_sentence = "At least one authorized conditioning perturbation changes the active output, but the complete modality pattern is not uniformly supported."
        else:
            science = "INCONCLUSIVE"
            science_sentence = "No exact active-output change was observed in the registered conditioning comparisons."

        outputs = {}
        outputs["conditions_csv_sha256"] = write_csv(output / "conditions.csv", condition_rows, ["condition", "source", "source_label", "body_sha256", "wrist_sha256", "state_sha256", "status", "query", "conditioning", "action_head_calls", "raw_array_sha256", "decoded_actual_sha256", "decoded_standardized_T_sha256", "p0_state_residual_m", "raw_active190_max_abs", "raw_active190_rmse", "standardized_T_xyz_max_abs", "standardized_T_xyz_rmse", "processor_information_loss"])
        outputs["expert_comparison_csv_sha256"] = write_csv(output / "expert_comparison.csv", expert_rows, ["condition", "native_source", "expert_sha256", "position_rmse_m", "position_max_abs_m", "endpoint_error_m", "eef_rot6d_rmse", "gripper_rmse", "prediction_path_length_m", "expert_path_length_m", "prediction_roughness_mean", "expert_roughness_mean", "world_target_metrics_status", "world_target_metrics_reason"])
        outputs["difference_json_sha256"] = write_json(output / "condition_differences.json", diff_rows)
        outputs["state_audit_json_sha256"] = write_json(output / "processed_state_audit.json", state_audits)
        outputs["c0_prefix_tail_json_sha256"] = write_json(output / "c0_prefix_tail_analysis.json", {"condition": "C0_BASELINE", "measured_eef_bridge_world_m": C0_MEASURED_EEF_WORLD.tolist(), "target_world_m": TARGET_WORLD.tolist(), "indices": c0_indices, "overall": _trajectory_metrics(c0_world, TARGET_WORLD), "prefix_index_0_19": c0_prefix, "tail_index_20_39": c0_tail, "world_metrics_status": "QUALIFIED_C0_MEASURED_BRIDGE"})
        outputs["trajectory_svg_sha256"] = _make_svg({label: decoded[label]["standardized_T"] for label in ("T_BASELINE", "C_BASELINE", "N1_NATIVE_a", "N2_NATIVE_b", "N3_NATIVE_c", "N4_NATIVE_d", "X1_VT_SC", "X2_VC_ST", "Z1_CONST128_ST", "Z2_CONST128_SC")}, output / "trajectory_plot.svg")

        conclusion = f"""# PI0157 conditioning-dependence result\n\nStatus: **COMPLETED**\n\nScience classification: **{science}**\n\n{science_sentence}\n\n## Fixed boundary\n\nThe accepted C1 sampler, N2 seed2223@2000, original processor statistics, fixed float32 x0, language, H40/T16 and bf16 model were held constant. T/C raw arrays and N1 when byte-identical were reused; no baseline or worker parity query was repeated. All raw arrays were saved before CPU decoding. Standardized-T physical views are diagnostic only and separate decoder-anchor mechanics from input sensitivity.\n\nTRAIN target/world progress remains **UNVERIFIED**. The C0 measured bridge retains the accepted Stage43 interpretation: its fixed decoded path approaches the target in the prefix and departs in the tail; this is policy output, not WBC execution.\n\n## Single recommendation\n\nAuthorize one fixed-budget teacher-forced/local-field diagnostic on the same four native TRAIN inputs, with StateV1 and paired-image conditioning recorded separately from decoded p0/relative action. Do not start training, ICL, WBC, or a new closed loop from this packet alone.\n"""
        outputs["one_page_conclusion_sha256"] = write_text(output / "one_page_conclusion.md", conclusion)
        report = {
            "schema": "CORELAB_STAGE45F_PI0157_CONDITIONING_DEPENDENCE_RESULT_V1",
            "status": "COMPLETED",
            "token": RETURN_TOKEN,
            "science_classification": science,
            "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
            "repository": {"branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "execution_head": execution_head, "head": execution_head},
            "identities": {**identities, **base_identity, "noise_array_sha256": NOISE_ARRAY_SHA, "noise_file_sha256": NOISE_FILE_SHA, "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE, "modality_sha256": MODALITY_SHA, "model_dtype": model_dtype, "cosmos_binding": cosmos_binding, "processor_binding": processor_binding},
            "native_input_identity": [{key: value for key, value in item.items() if key not in {"body", "wrist", "state", "expert"}} for item in natives.values()],
            "canonical_inputs": {key: {k: value for k, value in val.items() if k not in {"body_rgb", "wrist_rgb", "state"}} for key, val in (("T", train_npz), ("C", c0_npz))},
            "native_a_matches_canonical_T": native_a_matches_t,
            "input_records": [{key: value for key, value in input_records[label].items() if key != "model_inputs"} for label in items],
            "raw_records": raw_manifest,
            "conditions": condition_rows,
            "condition_differences": diff_rows,
            "expert_comparison": expert_rows,
            "c0_prefix_tail": {"path": str(output / "c0_prefix_tail_analysis.json"), "sha256": outputs["c0_prefix_tail_json_sha256"]},
            "path_effects": {"state_swap_X1_vs_T": path_effect_state, "vision_swap_X2_vs_T": path_effect_vision, "constant128_Z1_vs_T": path_effect_devision, "interpretation": "raw active and standardized-T physical changes are descriptive; actual-anchor p0 changes are not learned state response"},
            "outputs": outputs,
            "counters": {**counters, "cumulative_with_prior_closed_diagnostic": {"queries": PRIOR_CLOSED_DIAGNOSTIC["queries"] + counters["queries"], "conditioning": PRIOR_CLOSED_DIAGNOSTIC["conditioning"] + counters["conditioning"], "action_head_calls": PRIOR_CLOSED_DIAGNOSTIC["action_head_calls"] + counters["action_head_calls"], "model_side_calls": PRIOR_CLOSED_DIAGNOSTIC["model_side_calls"] + counters["model_side_calls"]}},
            "budget": {"nominal_queries": NOMINAL_QUERIES, "max_queries_including_mechanical": MAX_QUERIES, "max_conditioning": MAX_CONDITIONING, "max_action_head_calls": MAX_HEAD_CALLS, "max_model_side_calls": MAX_MODEL_SIDE_CALLS, "queries_consumed": counters["queries"], "conditioning_consumed": counters["conditioning"], "action_head_calls_consumed": counters["action_head_calls"], "model_side_calls_consumed": counters["model_side_calls"]},
            "execution_boundary": {"model_forward": counters["model_side_calls"], "training": False, "backward": False, "optimizer_step": 0, "icl": False, "new_data": False, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "safety_gate_relaxed": False, "retry": False},
            "claim_boundary": "conditioning sensitivity and processor/materialization characterization only; TRAIN world target metrics remain UNVERIFIED; no task-success, WBC, closed-loop, or generalization claim",
            "raw_first": True,
        }
        report_sha = write_json(output / "pi0157_conditioning_dependence.json", report)
        write_json(output / "evidence_hashes.json", {"report_sha256": report_sha, **outputs, "admission_sha256": admission_sha, "raw_array_sha256_by_condition": {row["label"]: row["raw_array_sha256"] for row in raw_records}})
        _finish_attempt(lock, lock_payload, status="COMPLETED", completed_unix=time.time(), counters=report["counters"], report_sha256=report_sha)
        return {**report, "report_sha256": report_sha, "admission_sha256": admission_sha}
    except Exception:
        if lock.exists():
            current = read_json(lock)
            if current.get("status") == "STARTED":
                _finish_attempt(lock, lock_payload, status="BLOCKED_OR_FAILED", failed_unix=time.time(), counters=counters)
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-head", required=True)
    p.add_argument("--stage10-root", required=True, type=Path)
    p.add_argument("--base-model", required=True, type=Path, help="SHA-bound BASE processor/model metadata root; model weights come from --model-path")
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--modality", required=True, type=Path)
    p.add_argument("--cosmos-snapshot", required=True, type=Path)
    p.add_argument("--n2-stats", required=True, type=Path)
    p.add_argument("--noise", required=True, type=Path)
    p.add_argument("--train-input", required=True, type=Path)
    p.add_argument("--archived-c0-input", required=True, type=Path)
    p.add_argument("--raw-t", required=True, type=Path)
    p.add_argument("--raw-c", required=True, type=Path)
    p.add_argument("--native-a", required=True, type=Path)
    p.add_argument("--native-b", required=True, type=Path)
    p.add_argument("--native-c", required=True, type=Path)
    p.add_argument("--native-d", required=True, type=Path)
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
