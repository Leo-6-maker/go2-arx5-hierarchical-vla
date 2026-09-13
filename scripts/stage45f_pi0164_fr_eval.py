#!/usr/bin/env python3
"""PI0164 frozen free-running evaluation for the E/S pilot.

The evaluator is deliberately separate from training.  It consumes the
accepted N2 seed-2223@2000 raw baseline (R1@2000) and the two newly written
checkpoint-256 arms, then runs the unchanged Stage45C four-step action-head
path with fixed noise.  Every new raw array is persisted and hashed before
decoding.  No WBC, simulator, Validation-E, Holdout-F, or training path is
reachable from this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


PI_GATE = "PI_GATE_STATE_0164"
PI_COMMENT = 5653126447
PR_COMMENT = 5653131645
RETURN_TOKEN = "CORELAB_PI0164_CONTROLLED_SUPERVISION_PILOT_FOR_PI_REVIEW"
CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_MODEL_DIR = "models--nvidia--Cosmos-Reason2-2B"
N2_X0_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
N2_X0_FILE_SHA = "83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922"
Q0_NPZ_SHA = "e0376e138bb1610a557164d3afffa3da1d7d0446a3b28406374360bbe7c441ad"
Q0_JSON_SHA = "a4d53a962af98b074b8a9f4a88a2b238e15254db362b2be06b2134d454c67e06"
RAW_SHAPE = (1, 40, 132)
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
TIMESTEP_IDS = (0, 250, 500, 750)
SAMPLER_STEPS = 4
SAMPLER_DT = 0.25
TARGET_ORDER = (
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
)
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TARGET_ORDER
    for index in range(6)
)
LANGUAGE = "move the end effector to the red target"
FFMPEG6_LIB = "/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env/public_native/python_env/lib"


class EvalError(RuntimeError):
    """Fail-closed evaluation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return sha256_file(path)


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"tree root unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(child.is_dir() and not child.is_symlink(), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(child.is_file() and not child.is_symlink(), f"unsafe tree file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _cosmos_model_root(root: Path) -> Path:
    """Resolve either a HF hub cache root or its model-specific directory."""
    root = Path(root).resolve()
    nested = root / COSMOS_MODEL_DIR
    if nested.is_dir() and (nested / "refs" / "main").is_file():
        return nested
    return root


def gpu_snapshot() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-1000:]}")
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    return {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "query_sha256": sha256_bytes(query.stdout.encode("utf-8")), "query": query.stdout, "rows": rows}


def select_gpu(snapshot: Mapping[str, Any], requested_uuid: str | None) -> dict[str, Any]:
    candidates = [row for row in snapshot["rows"] if "4080 SUPER" in str(row["name"]) and int(row["memory_total_mib"]) >= 30000]
    if requested_uuid:
        candidates = [row for row in candidates if str(row["uuid"]) == str(requested_uuid)]
    require(candidates, f"no RTX 4080 SUPER candidate: {snapshot['rows']}")
    candidates.sort(key=lambda row: (-int(row["memory_free_mib"]), int(row["index"])))
    selected = dict(candidates[0])
    require(int(selected["memory_free_mib"]) >= 12000, f"insufficient free VRAM for exact FR graph: {selected}")
    return selected


def prepare_environment(root: Path, gpu: Mapping[str, Any], cosmos: Path, n2_stats: Path) -> None:
    cache = root / "cache"
    tmp = root / "tmp"
    cache.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    hub = cache / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    (cache / "xdg").mkdir(parents=True, exist_ok=False)
    (hub / COSMOS_MODEL_DIR).symlink_to(_cosmos_model_root(cosmos), target_is_directory=True)
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    venv_root = Path(sys.executable).resolve().parents[1]
    torch_lib = venv_root / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
    env_lib = venv_root / "lib"
    ld_parts = [FFMPEG6_LIB]
    if torch_lib.is_dir():
        ld_parts.append(str(torch_lib))
    if env_lib.is_dir():
        ld_parts.append(str(env_lib))
    if old_ld:
        ld_parts.append(old_ld)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
        "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(cache / "xdg"), "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(hub), "TRANSFORMERS_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": os.pathsep.join(ld_parts),
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "STAGE45E_N2_STATS_ARTIFACT": str(n2_stats),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })


def load_q0(npz_path: Path, json_path: Path) -> dict[str, Any]:
    import numpy as np

    require(sha256_file(npz_path) == Q0_NPZ_SHA, "q0 NPZ identity drift")
    require(sha256_file(json_path) == Q0_JSON_SHA, "q0 JSON identity drift")
    document = read_json(json_path)
    require(document.get("condition_id") == "C1_X_PLUS_5CM", "q0 condition drift")
    require(document.get("expert_chunk_source") == "STAGE22_SAME_SCENE_EXPERT", "q0 source drift")
    require(document.get("target_excluded_from_model_input") is True, "q0 target exclusion witness missing")
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"body_rgb", "wrist_rgb", "state_v1", "expert_physical_local", "expert_timestamps", "eef_local_m", "anchor_world_m", "target_world_m"}
        require(required.issubset(set(data.files)), f"q0 fields drift: {data.files}")
        body = np.asarray(data["body_rgb"]); wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state_v1"], dtype=np.float32).reshape(-1)
        expert = np.asarray(data["expert_physical_local"], dtype=np.float64)
        timestamps = np.asarray(data["expert_timestamps"], dtype=np.float64).reshape(-1)
        anchor = np.asarray(data["anchor_world_m"], dtype=np.float64).reshape(-1)
        target = np.asarray(data["target_world_m"], dtype=np.float64).reshape(-1)
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, "q0 body ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, "q0 wrist ABI drift")
    require(state.shape == (10,) and expert.shape == (40, 10) and timestamps.shape == (40,), "q0 numeric ABI drift")
    require(anchor.shape == target.shape == (3,), "q0 pose ABI drift")
    require(np.isfinite(state).all() and np.isfinite(expert).all() and np.isfinite(timestamps).all(), "q0 nonfinite")
    require(np.all(np.diff(timestamps) > 0.0), "q0 timestamps are not increasing")
    require(np.allclose(np.asarray(data["eef_local_m"], dtype=np.float64).reshape(-1), state[:3], atol=1e-5, rtol=0.0), "q0 EEF/StateV1 mismatch")
    return {"body": body, "wrist": wrist, "state": state, "expert": expert, "timestamps": timestamps, "anchor": anchor, "target": target, "metadata": document, "npz_path": str(npz_path), "json_path": str(json_path), "npz_sha256": Q0_NPZ_SHA, "json_sha256": Q0_JSON_SHA}


def load_x0(path: Path) -> Any:
    import numpy as np

    require(path.is_file() and not path.is_symlink(), f"fixed eval x0 unavailable: {path}")
    require(sha256_file(path) == N2_X0_FILE_SHA, "fixed eval x0 file SHA drift")
    value = np.load(path, allow_pickle=False)
    require(value.shape == RAW_SHAPE and value.dtype == np.float32 and np.isfinite(value).all(), "fixed eval x0 ABI drift")
    require(sha256_bytes(np.ascontiguousarray(value).tobytes()) == N2_X0_ARRAY_SHA, "fixed eval x0 array SHA drift")
    return value


def load_snapshots(sealed_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight = read_json(sealed_root / "stage45c_runtime_preflight.json")
    snapshots = preflight.get("snapshots")
    require(isinstance(snapshots, list) and len(snapshots) == 24, "sealed baseline snapshots must contain 24 windows")
    require([str(item.get("window_id")) for item in snapshots] == list(WINDOW_IDS), "sealed window order drift")
    for item in snapshots:
        require(item.get("target_excluded_from_model_input") is True, "sealed target exclusion witness missing")
        for key in ("state_v1", "expert_physical_action", "anchor_world_m", "target_world_m", "source_timestamps", "observation_timestamp"):
            require(key in item, f"sealed snapshot field missing: {key}")
    noise_manifest = read_json(sealed_root / "sampler_noise_manifest.json")
    require(noise_manifest.get("window_ids") == list(WINDOW_IDS), "sealed noise order drift")
    require(noise_manifest.get("noise_shape") == list(RAW_SHAPE), "sealed noise shape drift")
    require(noise_manifest.get("same_noise_across_checkpoints") is True, "sealed same-noise contract missing")
    return [dict(item) for item in snapshots], noise_manifest


def _safety_gate(physical: Any, state: Any, timestamps: Any, chunk_id: str) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.diagnostics.stage23_action_pipeline import _chunk_from_physical
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    state_np = np.asarray(state, dtype=np.float64).reshape(-1)
    chunk = _chunk_from_physical(
        np.asarray(physical, dtype=np.float64), np.asarray(timestamps, dtype=np.float64),
        chunk_id=chunk_id, source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center", observation_timestamp=float(np.asarray(timestamps)[0]),
        state=state_np,
    )
    result = TrajectorySafetyGateV1().evaluate(
        chunk, now=float(np.asarray(timestamps)[0]), achieved_position=state_np[:3],
        achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state_np[3:9])),
    )
    return result.as_dict()


def make_row(evaluator: Any, raw: Any, raw_path: Path, raw_sha: str, snapshot: Mapping[str, Any], checkpoint: Mapping[str, Any], stats: Mapping[str, Any], *, runtime_sha: str = GROOT_COMMIT) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.trajectory_metrics import array_sha256

    physical, details = evaluator._decode_with_stats(
        raw, snapshot["state_v1"], stats, stats_sha256=N2_STATS_SHA, payload_sha256=N2_STATS_PAYLOAD_SHA,
        chunk_id=str(snapshot["window_id"]), timestamps=snapshot["source_timestamps"], observation_timestamp=float(snapshot["observation_timestamp"]),
    )
    normalized = evaluator._normalized_metrics(raw, snapshot["expert_physical_action"], snapshot["state_v1"], snapshot["source_timestamps"], stats)
    row = evaluator._evaluation_row(
        raw, physical, details, normalized, snapshot, checkpoint, raw_path, raw_sha, runtime_sha,
    )
    gate = _safety_gate(physical, snapshot["state_v1"], snapshot["source_timestamps"], str(snapshot["window_id"]))
    row["safety_gate_decision"] = gate.get("decision")
    row["safety_gate"] = gate
    row["raw_prediction_artifact"]["array_sha256"] = array_sha256(np.asarray(raw, dtype=np.float32))
    return row


def _q0_snapshot(q0: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "window_id": "PI0163_C1_X_PLUS_5CM:q0",
        "target_id": "C1_X_PLUS_5CM",
        "state_v1": q0["state"], "expert_physical_action": q0["expert"],
        "anchor_world_m": q0["anchor"], "target_world_m": q0["target"],
        "source_timestamps": q0["timestamps"], "observation_timestamp": float(q0["timestamps"][0]),
    }


def _load_baseline(evaluator: Any, baseline_dir: Path, q0_raw: Path, q0: Mapping[str, Any], snapshots: Sequence[Mapping[str, Any]], stats: Mapping[str, Any], output: Path) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    files = sorted(baseline_dir.glob("*.npy"))
    require([item.name for item in files] == [f"{index:02d}.npy" for index in range(24)], "baseline raw file order/coverage drift")
    checkpoint = {"label": "N2(seed2223)@2000", "tree_sha256": CHECKPOINT_TREE}
    rows: list[dict[str, Any]] = []
    for index, (path, snapshot) in enumerate(zip(files, snapshots)):
        raw = np.load(path, allow_pickle=False)
        require(raw.shape == RAW_SHAPE and raw.dtype == np.float32 and np.isfinite(raw).all(), f"baseline raw ABI drift at {index}")
        rows.append(make_row(evaluator, raw, path, sha256_file(path), snapshot, checkpoint, stats))
    aggregate = aggregate_evaluation_rows(rows)
    q_raw = np.load(q0_raw, allow_pickle=False)["raw_universal_action"]
    q_snapshot = _q0_snapshot(q0)
    q_row = make_row(evaluator, q_raw, q0_raw, sha256_file(q0_raw), q_snapshot, checkpoint, stats)
    write_json(output / "baseline_FULL24_rows.json", rows)
    write_json(output / "baseline_FULL24_aggregate.json", aggregate)
    write_json(output / "baseline_q0_row.json", q_row)
    return {"checkpoint": checkpoint, "rows": rows, "aggregate": aggregate, "q0": q_row, "raw_reused": {"source_dir": str(baseline_dir), "q0_source": str(q0_raw), "window_count": 24}}


def _eval_arm(evaluator: Any, torch: Any, load_model: Any, model_path: Path, label: str, output: Path, windows: Sequence[Mapping[str, Any]], noise_bank: Sequence[Any], q0: Mapping[str, Any], q0_x0: Any, processor: Any, embodiment: Any, stats: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from stage45f_pi0153_generation_probe import _build_c1_inputs

    tree = tree_digest(tree_entries(model_path, f"stage45f_pi0164/{label}"))
    model = load_model(model_path, "cuda:0")
    model.eval()
    hook_counter = {"count": 0}
    hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_counter.__setitem__("count", hook_counter["count"] + 1))
    raw_manifest = output / f"{label}_raw_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    q_row: dict[str, Any] | None = None
    try:
        for index, window in enumerate(windows):
            with torch.inference_mode():
                raw, calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
            raw_np = raw.detach().cpu().numpy().astype(np.float32, copy=False)
            raw_path = output / "raw" / label / f"{index:02d}.npy"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, raw_np, allow_pickle=False)
            raw_sha = sha256_file(raw_path)
            with raw_manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"window_id": window["window_id"], "checkpoint": label, "path": str(raw_path.relative_to(output)), "file_sha256": raw_sha, "array_sha256": sha256_bytes(np.ascontiguousarray(raw_np).tobytes()), "sampler_calls_observed": calls, "decoded": False}, sort_keys=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            require(calls == SAMPLER_STEPS, f"{label} sampler calls drift at {index}")
            row = make_row(evaluator, raw_np, raw_path.relative_to(output), raw_sha, window["snapshot"], {"label": label, "tree_sha256": tree}, stats)
            rows.append(row)
            with raw_manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"window_id": window["window_id"], "checkpoint": label, "decoded": True, "decoded_array_sha256": row["decoded_action"]["array_sha256"]}, sort_keys=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())
        q_inputs = _build_c1_inputs(processor, embodiment, {"body": q0["body"], "wrist": q0["wrist"], "state": q0["state"]}, train=False)
        with torch.inference_mode():
            q_raw, q_calls = _sample_one(model, q_inputs, torch.from_numpy(np.asarray(q0_x0, dtype=np.float32)), torch)
        q_np = q_raw.detach().cpu().numpy().astype(np.float32, copy=False)
        q_path = output / "raw" / label / "q0.npy"
        np.save(q_path, q_np, allow_pickle=False)
        q_sha = sha256_file(q_path)
        q_row = make_row(evaluator, q_np, q_path.relative_to(output), q_sha, _q0_snapshot(q0), {"label": label, "tree_sha256": tree}, stats)
        require(q_calls == SAMPLER_STEPS, f"{label} q0 sampler calls drift")
    finally:
        hook.remove()
        del model
        torch.cuda.empty_cache()
    require(hook_counter["count"] == 25 * SAMPLER_STEPS, f"{label} action-head count drift: {hook_counter['count']}")
    aggregate = aggregate_evaluation_rows(rows)
    write_json(output / f"{label}_FULL24_rows.json", rows)
    write_json(output / f"{label}_FULL24_aggregate.json", aggregate)
    write_json(output / f"{label}_q0_row.json", q_row)
    return {"label": label, "checkpoint_tree_sha256": tree, "rows": rows, "aggregate": aggregate, "q0": q_row, "model_construction": 1, "conditioning_passes": 25, "action_head_calls": hook_counter["count"], "query_count": 25, "raw_manifest_sha256": sha256_file(raw_manifest)}


def _mean(mapping: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def write_summary(output: Path, baseline: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    def pack(item: Mapping[str, Any]) -> dict[str, Any]:
        aggregate = item["aggregate"]
        return {"full24": {"endpoint_error_mean_m": _mean(aggregate, "comparison_metrics", "endpoint_error_m", "mean"), "xyz_rmse_mean_m": _mean(aggregate, "comparison_metrics", "position_rmse_m", "mean"), "path_length_mean_m": _mean(aggregate, "comparison_metrics", "prediction_path_length_m", "mean"), "roughness_mean": _mean(aggregate, "comparison_metrics", "prediction_roughness_position_second_difference_m", "mean", "mean"), "safety_gate_decisions": aggregate.get("safety_gate_decision_counts")}, "q0": {"endpoint_error_m": _mean(item["q0"], "metrics", "endpoint_error_m"), "final_target_distance_m": _mean(item["q0"], "metrics", "target", "final_target_distance_m"), "absolute_progress_m": _mean(item["q0"], "metrics", "target", "absolute_progress_m"), "path_length_m": _mean(item["q0"], "metrics", "target", "path_length_m"), "safety_gate_decision": item["q0"].get("safety_gate_decision")}}

    values = {"baseline": pack(baseline), **{name: pack(item) for name, item in arms.items()}}
    base_ep = values["baseline"]["full24"]["endpoint_error_mean_m"]
    for name in arms:
        values[name]["old_task_delta_vs_baseline"] = {"endpoint_error_mean_m": None if base_ep is None else values[name]["full24"]["endpoint_error_mean_m"] - base_ep}
    summary = {"schema": "CORELAB_PI0164_CONTROLLED_SUPERVISION_SUMMARY_V1", "status": "COMPLETED", "provenance": dict(provenance), "metrics": values, "claim_boundary": "local 256-update supervision pilot; FULL24 and one qualified C1 q0 only; no initial-state generalization or WBC claim", "recommendation": "If S improves q0 without materially degrading FULL24, next run should use multiple independently synchronized visited-state expert windows under a new PI gate; do not infer broad generalization from this single-Q pilot."}
    write_json(output / "pilot_summary.json", summary)
    lines = ["<svg xmlns='http://www.w3.org/2000/svg' width='900' height='440' viewBox='0 0 900 440'>", "<rect width='100%' height='100%' fill='white'/>", "<text x='24' y='30' font-family='sans-serif' font-size='18'>PI0164 frozen FR comparison (mean FULL24 endpoint error)</text>"]
    labels = ["N2 baseline", "E@256", "S@256"]
    names = ["baseline", "E", "S"]
    scale = 2600.0
    for index, (label, name) in enumerate(zip(labels, names)):
        value = values[name]["full24"]["endpoint_error_mean_m"] or 0.0
        width = max(1.0, value * scale)
        y = 70 + index * 95
        lines.append(f"<text x='24' y='{y+24}' font-family='sans-serif' font-size='14'>{label}</text>")
        lines.append(f"<rect x='150' y='{y}' width='{width:.2f}' height='34' fill='#4c78a8'/>")
        lines.append(f"<text x='{160+width:.2f}' y='{y+23}' font-family='sans-serif' font-size='13'>{value:.6f} m</text>")
    lines.append("<text x='24' y='390' font-family='sans-serif' font-size='12'>Raw-first, same E4/noise; bars are descriptive and not a winner threshold.</text>")
    lines.append("</svg>\n")
    (output / "pilot_summary.svg").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    output = Path(args.output_root).resolve()
    require(not output.exists() and not output.is_symlink(), f"evaluation root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot_records, noise_manifest = load_snapshots(Path(args.sealed_root).resolve())
    q0 = load_q0(Path(args.q0_npz).resolve(), Path(args.q0_json).resolve())
    x0 = load_x0(Path(args.eval_x0).resolve())
    gpu_snapshot_value = gpu_snapshot()
    gpu = select_gpu(gpu_snapshot_value, args.gpu_uuid)
    prepare_environment(output, gpu, Path(args.cosmos).resolve(), Path(args.n2_stats).resolve())
    scripts = Path(args.groot).resolve()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    if str(Path(args.repo_src).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(args.repo_src).resolve()))
    from stage45e_n2_train24_eval import _load_runtime_processor
    from stage45b_full24_fresh_base_train import _bind_cosmos_processor
    cosmos_model = _cosmos_model_root(Path(args.cosmos).resolve())
    _bind_cosmos_processor(cosmos_model / "snapshots" / COSMOS_REVISION)
    paths = {"groot": Path(args.groot).resolve(), "base_model": Path(args.base_model).resolve(), "dataset": Path(args.dataset).resolve(), "modality": Path(args.modality).resolve(), "cosmos": cosmos_model}
    processor, embodiment, modality_configs = _load_runtime_processor(paths)
    windows = __import__("stage45e_n2_train24_eval")._make_windows(processor, embodiment, modality_configs, Path(args.dataset).resolve(), snapshot_records)
    require(len(windows) == 24, "prepared TRAIN24 window count drift")
    import torch
    evaluator = __import__("stage45e_n2_train24_eval")
    noise_bank = evaluator._make_noise_bank(torch, noise_manifest)
    stats = read_json(Path(args.n2_stats).resolve())
    baseline = _load_baseline(evaluator, Path(args.baseline_raw_dir).resolve(), Path(args.baseline_q0_raw).resolve(), q0, snapshot_records, stats, output)
    load_model = __import__("stage38_structured_offline_evaluation")._load_structured_model
    arms: dict[str, Any] = {}
    for label, path in (("E", Path(args.e_checkpoint).resolve()), ("S", Path(args.s_checkpoint).resolve())):
        arms[label] = _eval_arm(evaluator, torch, load_model, path, label, output, windows, noise_bank, q0, x0, processor, embodiment, stats)
    provenance = {"authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "gpu": gpu, "gpu_snapshot": gpu_snapshot_value, "n2_checkpoint_tree_sha256": CHECKPOINT_TREE, "n2_stats_sha256": sha256_file(Path(args.n2_stats).resolve()), "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "fixed_eval_x0_array_sha256": N2_X0_ARRAY_SHA, "fixed_eval_x0_file_sha256": N2_X0_FILE_SHA, "full24_tree_sha256": FULL24_TREE, "cosmos_revision": COSMOS_REVISION, "groot_commit": GROOT_COMMIT, "baseline_raw_reused": True, "new_model_queries": 50, "new_conditioning_passes": 50, "new_action_head_calls": 200, "new_model_constructions": 2, "training": False, "backward": False, "optimizer_steps": 0, "wbc": 0, "simulator": 0, "hardware_control": False, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "ICL": 0, "timestamp_utc": now_utc()}
    summary = write_summary(output, baseline, arms, provenance)
    result = {"schema": "CORELAB_PI0164_CONTROLLED_SUPERVISION_EVAL_RESULT_V1", "status": "COMPLETED", "return_token": RETURN_TOKEN, "summary_sha256": sha256_file(output / "pilot_summary.json"), "summary": summary, "provenance": provenance, "baseline": {"aggregate_sha256": sha256_file(output / "baseline_FULL24_aggregate.json"), "q0_row_sha256": sha256_file(output / "baseline_q0_row.json")}, "arms": {name: {"aggregate_sha256": sha256_file(output / f"{name}_FULL24_aggregate.json"), "q0_row_sha256": sha256_file(output / f"{name}_q0_row.json"), "model_constructions": value["model_construction"], "query_count": value["query_count"], "conditioning_passes": value["conditioning_passes"], "action_head_calls": value["action_head_calls"]} for name, value in arms.items()}, "execution_boundary": {"gpu": True, "model_construction": 2, "model_forward_queries": 50, "conditioning_passes": 50, "action_head_calls": 200, "training": False, "backward": False, "optimizer_steps": 0, "wbc": 0, "simulator": 0, "hardware": False, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "ICL": 0}}
    write_json(output / "evaluation_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--baseline-raw-dir", type=Path, required=True)
    parser.add_argument("--baseline-q0-raw", type=Path, required=True)
    parser.add_argument("--q0-npz", type=Path, required=True)
    parser.add_argument("--q0-json", type=Path, required=True)
    parser.add_argument("--eval-x0", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--groot", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--cosmos", type=Path, required=True)
    parser.add_argument("--modality", type=Path, required=True)
    parser.add_argument("--n2-stats", type=Path, required=True)
    parser.add_argument("--e-checkpoint", type=Path, required=True)
    parser.add_argument("--s-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    try:
        print(json.dumps(jsonable(run(args)), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"PI0164_EVAL_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
