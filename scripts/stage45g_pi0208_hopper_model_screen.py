#!/usr/bin/env python3
"""PI0208 G1 model-side screen on the frozen FULL24/support surfaces.

This is a thin driver around the accepted Stage45C sampler and Stage40
decoder.  It writes each raw [1,40,132] array durably before decoding it and
never calls WBC, a simulator, or a training API.
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
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45c_h200_eval_runtime_adapter as stage45c  # noqa: E402
import stage45f_pi0183_matched_training as training  # noqa: E402


EXPECTED_HEAD = os.environ.get("PI0208_EXPECTED_HEAD", "")
GROOT_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA256 = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
STRUCTURED_STATS_SHA256 = "430cfc259357259791e069e564fc4b5754cbeb144b720c5289c032fe62d80e3d"
STRUCTURED_SOURCE_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
ROWS_SHA256 = "ab08414ac5cdda3cfadf1a08e5a85ccc3ec96fec74b32b6bf34bb5625f84f876"
VISITED_TREE_SHA256 = "f5ccefcb17594d563dce24aae3e476b7c56d2e6e75304a7358b4ec887aafb416"
EXPERT_TREE_SHA256 = "9b77dbddaf7b7fca6aacf55b0f68b202b398c8f1aa1dd0f0437d5dc18398be13"
LANGUAGE = "move the end effector to the red target"
HORIZON = 40
RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
SCREEN_SEED = 4242
SURFACES = ("FULL24", "STATE_SUPPORT20")
ARM_ORDER = ("E256", "C80", "S25", "S50")
ARM_SPECS = {
    "E256": {
        "label": "E@256",
        "tree": "f91f4fd3cf3c440127c57d49cdd2dde52ce6a1294bd17a05710fba1026b9361b",
        "prefix": "617089/E/checkpoint-256",
        "step": 256,
        "arm": "E",
    },
    "C80": {
        "label": "C80@80",
        "tree": "b7f3190a64d291491cf4317c88525db24a01c35742c824f66f3da8a62bead7bf",
        "prefix": "pi0205_C80/checkpoint-80",
        "step": 80,
        "arm": "C80",
    },
    "S25": {
        "label": "S25@80",
        "tree": "93b8cd12dd1aa34ae1be91aa0e39baeacae0e189c72ea83d48a764a93c796492",
        "prefix": "pi0205_S25/checkpoint-80",
        "step": 80,
        "arm": "S25",
    },
    "S50": {
        "label": "S50@80",
        "tree": "3c2e8178474c16f9fe405edb63df275f1c95b48aec593b60cd4f67fecf3e1a3e",
        "prefix": "pi0205_S50/checkpoint-80",
        "step": 80,
        "arm": "S50",
    },
}


class ScreenError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ScreenError(f"JSON unavailable: {path}") from exc
    require(stat.S_ISREG(mode), f"JSON is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> str:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return sha256_file(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_npy(path: Path, value: Any) -> None:
    import numpy as np

    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite raw evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _regular(path: Path, label: str, directory: bool = False) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ScreenError(f"{label} unavailable: {path}") from exc
    require((stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)) and not stat.S_ISLNK(mode), f"{label} is unsafe: {path}")
    return path.resolve()


def _command(argv: Sequence[str]) -> tuple[int, str, str]:
    result = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def checkout_identity(expected_head: str) -> dict[str, Any]:
    def git(*args: str) -> str:
        code, stdout, stderr = _command(["git", "-C", str(REPO), *args])
        require(code == 0, f"git failed: {' '.join(args)}: {stderr[-400:]}")
        return stdout.strip()

    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    require(expected_head and head == expected_head, f"checkout HEAD drift: {head} != {expected_head}")
    require(branch == "stage45g/e256-state-support-dose-response", f"branch drift: {branch}")
    require(not status, "checkout is dirty")
    return {"root": str(REPO), "branch": branch, "head": head, "clean": True}


def gpu_identity() -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required")
    query = "index,name,uuid,memory.total,memory.free,memory.used"
    code, stdout, stderr = _command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    require(code == 0, f"nvidia-smi failed: {stderr[-400:]}")
    rows = []
    for line in stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, f"malformed GPU row: {line}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_mib": int(fields[3]), "free_mib": int(fields[4]), "used_mib": int(fields[5])})
    require(len(rows) == 1, f"expected one visible GPU: {rows!r}")
    row = rows[0]
    require(any(name in row["name"].upper() for name in ("H100", "H200")), f"unsupported Hopper GPU: {row['name']}")
    require(row["free_mib"] >= 12288, f"insufficient free VRAM: {row['free_mib']} MiB")
    code, apps, _ = _command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    return {"schema": "LOCALPC_PI0208_HOPPER_G1_GPU_ADMISSION_V1", "host": os.uname().nodename, "pbs_jobid": os.environ["PBS_JOBID"], "selected": row, "visible_gpu_count": 1, "free_vram_requirement_mib": 12288, "foreign_compute_processes": apps.splitlines()[1:] if code == 0 and apps.strip() else [], "nvidia_smi_gpu_stdout_sha256": sha256_bytes((stdout + "\n").encode())}


def _verify_structured_stats(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    require(observed == STRUCTURED_STATS_SHA256, f"structured stats SHA drift: {observed}")
    document = read_json(path)
    stats = document.get("statistics", document)
    require(document.get("schema") == "STAGE38_D_STRUCTURED_TRAINING_SIDE_CAR_V1", "structured stats schema drift")
    require(stats.get("source_file_sha256") == STRUCTURED_SOURCE_SHA256, "structured stats source provenance drift")
    require(stats.get("statistics_recomputed") is False, "structured stats were recomputed")
    return {"path": str(path), "sha256": observed, "source_file_sha256": STRUCTURED_SOURCE_SHA256, "statistics_recomputed": False}


def _verify_training_record(path: Path, checkpoint: Path, key: str) -> dict[str, Any]:
    result = read_json(path)
    expected_arm = ARM_SPECS[key]["arm"]
    require(result.get("status") == "PASS" and result.get("arm") == expected_arm, f"{key} training result is not PASS")
    step = int(result.get("global_step", -1))
    require(step == ARM_SPECS[key]["step"], f"{key} global step drift")
    require(str(Path(result.get("final_checkpoint", "")).resolve()) == str(checkpoint.resolve()), f"{key} checkpoint path drift")
    require(int(result.get("model_forward", -1)) == step and int(result.get("backward", -1)) == step and int(result.get("optimizer_steps", -1)) == step, f"{key} training counters drift")
    if key != "E256":
        require(result.get("schedule_sha256") == "03634d5adec0366f1f8550f8cb10f48cc0096dc6e2d69f4d9de4a90c10376bb7", f"{key} schedule drift")
        require(result.get("identities", {}).get("base_checkpoint", {}).get("tree_sha256") == ARM_SPECS["E256"]["tree"], f"{key} base identity drift")
    else:
        require(result.get("identities", {}).get("base_checkpoint", {}).get("tree_sha256") == "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16", "E base identity drift")
    return {"path": str(path), "sha256": sha256_file(path), "schema": result.get("schema"), "arm": expected_arm, "global_step": step, "forward": int(result["model_forward"]), "backward": int(result["backward"]), "optimizer_steps": int(result["optimizer_steps"])}


def _verify_checkpoint(key: str, path: Path) -> dict[str, Any]:
    spec = ARM_SPECS[key]
    entries_tree, entries = training.tree_digest(path, spec["prefix"])
    require(len(entries) == 19, f"{key} checkpoint file count drift: {len(entries)}")
    require(entries_tree == spec["tree"], f"{key} checkpoint tree drift: {entries_tree}")
    stats = _verify_structured_stats(path / "experiment_cfg" / "structured_action_statistics.json")
    state = read_json(path / "trainer_state.json")
    require(int(state.get("global_step", -1)) == spec["step"], f"{key} trainer state step drift")
    return {"label": spec["label"], "path": str(path.resolve()), "tree_sha256": entries_tree, "tree_prefix": spec["prefix"], "file_count": len(entries), "total_bytes": sum(int(item["bytes"]) for item in entries), "global_step": spec["step"], "structured_stats": stats}


def audit_inputs(args: argparse.Namespace, gpu: Mapping[str, Any]) -> dict[str, Any]:
    checkout = checkout_identity(args.expected_head)
    dataset = training.validate_dataset(args.dataset)
    require(dataset["tree_sha256"] == DATASET_TREE_SHA256, "FULL24 dataset identity drift")
    rows, rows_identity = training.load_rows(args.rows, args.support_root)
    require(rows_identity["sha256"] == ROWS_SHA256 and rows_identity["row_count"] == 20, "support row identity drift")
    visited = training.validate_pack(args.support_root, "visited_pack_manifest.json", "CORELAB_PI0181_VISITED_STATE_PACK_V1", VISITED_TREE_SHA256)
    expert = training.validate_pack(args.support_root, "expert_label_manifest.json", "CORELAB_PI0181_EXPERT_LABEL_PACK_V1", EXPERT_TREE_SHA256)
    manifest = _regular(args.groot_manifest, "GR00T manifest")
    require(sha256_file(manifest) == GROOT_MANIFEST_SHA256, "GR00T manifest SHA drift")
    modality = _regular(args.modality, "modality")
    config = _regular(args.config, "training config")
    require(sha256_file(modality) == MODALITY_SHA256, "modality SHA drift")
    require(sha256_file(config) == CONFIG_SHA256, "training config SHA drift")
    stats = _regular(args.stats, "FULL24 runtime stats")
    runtime_stats = stage45c._verify_runtime_stats(stats)
    require(runtime_stats["sha256"] == STATS_SHA256, "FULL24 stats identity drift")
    cosmos_root = _regular(args.cosmos_root, "Cosmos cache", True)
    refs = _regular(cosmos_root / "refs" / "main", "Cosmos refs/main")
    require(refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    _regular(cosmos_root / "snapshots" / COSMOS_REVISION, "Cosmos pinned snapshot", True)
    cosmos_tree, cosmos_entries = stage45c.cosmos_tree_digest(cosmos_root)
    require(cosmos_tree == COSMOS_TREE_SHA256, f"Cosmos tree drift: {cosmos_tree}")
    from stage45b_groot_source_snapshot import verify_extracted

    groot = verify_extracted(args.groot_source, manifest, expected_manifest_file_sha256=GROOT_MANIFEST_SHA256, expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA256, expected_payload_tree_sha256=GROOT_PAYLOAD_TREE_SHA256)
    checkpoints = {}
    records = {}
    for key in ARM_ORDER:
        checkpoint = _regular(getattr(args, key.lower()), f"{key} checkpoint", True)
        checkpoints[key] = _verify_checkpoint(key, checkpoint)
        result_path = checkpoint.parent / "training_result.json"
        invocation_path = checkpoint.parent / "training_invocation.json"
        records[key] = {"result": _verify_training_record(result_path, checkpoint, key), "invocation": {"path": str(invocation_path), "sha256": sha256_file(_regular(invocation_path, f"{key} training invocation"))}}
    descriptor = stage45c._verify_train24_descriptor_closure(args.snapshot_root, args.dataset)
    return {"checkout": checkout, "gpu": dict(gpu), "dataset": dataset, "support_rows": rows_identity, "support_packs": {"visited": visited, "expert": expert}, "groot": groot, "groot_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)}, "modality": {"path": str(modality), "sha256": sha256_file(modality)}, "config": {"path": str(config), "sha256": sha256_file(config)}, "full24_runtime_stats": runtime_stats, "cosmos": {"root": str(cosmos_root), "revision": COSMOS_REVISION, "tree_sha256": cosmos_tree, "entry_count": cosmos_entries}, "checkpoints": checkpoints, "training_records": records, "descriptor_closure": descriptor, "screen_seed": SCREEN_SEED}


def _model_inputs(processor: Any, embodiment: Any, body: Any, wrist: Any, state: Any, label: str) -> Mapping[str, Any]:
    import torch
    from gr00t.data.types import MessageType, VLAStepData

    content = VLAStepData(images={"external_rgb": body[None, ...], "wrist_rgb": wrist[None, ...]}, states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}, actions={}, text=LANGUAGE, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    batch = processor.collator([processed])
    inputs = batch.get("inputs", {})
    require(isinstance(inputs, Mapping), f"model input mapping missing: {label}")
    require("action" not in inputs and "action_mask" not in inputs, f"action leaked into model input: {label}")
    for key in ("target_position_world_m", "target_xyz", "target_world_m"):
        require(key not in inputs, f"target leaked into model input: {label}")
    if inputs.get("state") is not None:
        require(bool(torch.isfinite(inputs["state"]).all().item()), f"processed state is nonfinite: {label}")
    return inputs


def support_windows(processor: Any, embodiment: Any, rows: Mapping[str, Mapping[str, Any]], support_root: Path) -> list[dict[str, Any]]:
    import numpy as np

    windows = []
    for row in rows.values():
        row_id = str(row["row_id"])
        visited = support_root / str(row["payload_files"]["visited_row_npz"])
        expert_path = support_root / str(row["payload_files"]["expert_label_npz"])
        with np.load(visited, allow_pickle=False) as archive:
            body = np.asarray(archive["body_rgb"], dtype=np.uint8)
            wrist = np.asarray(archive["wrist_rgb"], dtype=np.uint8)
            state = np.asarray(archive["state_v1"], dtype=np.float32).reshape(-1)
            anchor = np.asarray(archive["anchor_world_m"], dtype=np.float64).reshape(-1)
            target = np.asarray(archive["target_world_m"], dtype=np.float64).reshape(-1)
        with np.load(expert_path, allow_pickle=False) as archive:
            expert = np.asarray(archive["expert_action_structured_v1"], dtype=np.float64)
            timestamps = np.asarray(archive["source_timestamps"], dtype=np.float64)
        require(body.shape == (96, 96, 3) and wrist.shape == (96, 96, 3) and state.shape == (10,), f"support input shape drift: {row_id}")
        require(expert.shape == PHYSICAL_SHAPE and timestamps.shape == (HORIZON,), f"support label shape drift: {row_id}")
        require(np.isfinite(state).all() and np.isfinite(expert).all() and np.isfinite(timestamps).all(), f"support input nonfinite: {row_id}")
        require(sha256_bytes(np.ascontiguousarray(body).tobytes()) == row["body_sha256"], f"support body SHA drift: {row_id}")
        require(sha256_bytes(np.ascontiguousarray(wrist).tobytes()) == row["wrist_sha256"], f"support wrist SHA drift: {row_id}")
        require(sha256_bytes(np.ascontiguousarray(expert.astype(np.float32)).tobytes()) == row["expert_action_sha256"], f"support expert SHA drift: {row_id}")
        require(np.allclose(state, np.asarray(row["state_v1"], dtype=np.float32), atol=0.0, rtol=0.0), f"support state binding drift: {row_id}")
        require(np.allclose(anchor, np.asarray(row["anchor_world_m"], dtype=np.float64), atol=1e-7, rtol=0.0), f"support anchor binding drift: {row_id}")
        require(np.allclose(target, np.asarray(row["target_world_m"], dtype=np.float64), atol=1e-7, rtol=0.0), f"support target binding drift: {row_id}")
        observation_timestamp = float(row["observation_timestamp_s"])
        require(abs(float(timestamps[0]) - observation_timestamp) <= 1e-7 and np.allclose(np.diff(timestamps), 0.05, atol=1e-8, rtol=0.0), f"support timestamp drift: {row_id}")
        windows.append({"surface": "STATE_SUPPORT20", "index": len(windows), "window_id": row_id, "model_inputs": _model_inputs(processor, embodiment, body, wrist, state, row_id), "snapshot": {"window_id": row_id, "target_world_m": target.tolist(), "state_v1": state.tolist(), "expert_physical_action": expert.tolist(), "source_timestamps": timestamps.tolist(), "observation_timestamp": observation_timestamp, "anchor_world_m": anchor.tolist(), "target_excluded_from_model_input": True}, "support_origin_class": row["origin_class"]})
    require(len(windows) == 20, f"support window count drift: {len(windows)}")
    return windows


def _noise_bank(torch: Any, count: int) -> list[Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SCREEN_SEED)
    bank = []
    for _ in range(count):
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
        require(bool(torch.isfinite(noise).all().item()), "noise is nonfinite")
        bank.append(noise)
    return bank


def _profile_gate(physical: Any, snapshot: Mapping[str, Any], chunk_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.metrics.action_profile import chunk_profile
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix

    state = np.asarray(snapshot["state_v1"], dtype=np.float64)
    timestamps = np.asarray(snapshot["source_timestamps"], dtype=np.float64)
    provider = GR00TTrajectoryProvider(physical, config=GR00TProviderConfig(source_frame=stage45c.SOURCE_FRAME, tcp_semantic_id=stage45c.TCP_SEMANTIC_ID, timestamp_provenance="FROZEN_STAGE45C_TRAIN24_OR_SUPPORT_LABEL_GRID", source="PI0208_G1_MODEL_SCREEN"), point_timestamps=timestamps, chunk_id=chunk_id, observation_timestamp=float(snapshot["observation_timestamp"]), provenance={"trajectory_correction": False, "target_excluded_from_model_input": True})
    chunk = provider.next_chunk(float(snapshot["observation_timestamp"]))
    require(chunk is not None, f"provider emitted no chunk: {chunk_id}")
    achieved_quaternion = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
    gate = TrajectorySafetyGateV1().evaluate(chunk, now=float(snapshot["observation_timestamp"]), achieved_position=state[:3], achieved_orientation=achieved_quaternion)
    profile = chunk_profile(chunk, achieved_position=state[:3], achieved_quaternion=achieved_quaternion, provenance={"surface": snapshot.get("surface", "unknown"), "trajectory_correction": False})
    return profile, gate.as_dict()


def _mean(rows: Sequence[Mapping[str, Any]], getter) -> float:
    import numpy as np

    values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
    require(values.size and np.isfinite(values).all(), "aggregate values are empty/nonfinite")
    return float(np.mean(values))


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    by_surface = {}
    for surface in SURFACES:
        cohort = [row for row in rows if row["surface"] == surface]
        require(len(cohort) == (24 if surface == "FULL24" else 20), f"{surface} row count drift")
        by_surface[surface] = {
            "count": len(cohort),
            "mean_trajectory_error_m": _mean(cohort, lambda row: row["metrics"]["position_rmse_m"]),
            "mean_endpoint_error_m": _mean(cohort, lambda row: row["metrics"]["endpoint_error_m"]),
            "mean_absolute_progress_m": _mean(cohort, lambda row: row["metrics"]["target"]["absolute_progress_m"]),
            "mean_final_target_distance_m": _mean(cohort, lambda row: row["metrics"]["target"]["final_target_distance_m"]),
            "mean_path_length_m": _mean(cohort, lambda row: row["metrics"]["target"]["path_length_m"]),
            "mean_roughness_m": _mean(cohort, lambda row: row["metrics"]["target"]["roughness_position_second_difference_m"]["mean"]),
            "mean_linear_speed_mps": _mean(cohort, lambda row: row["profile"]["linear_speed_mps"]["mean"]),
            "peak_linear_speed_mps": float(max(float(row["profile"]["linear_speed_mps"]["max"]) for row in cohort)),
            "mean_angular_speed_radps": _mean(cohort, lambda row: row["profile"]["angular_speed_radps"]["mean"]),
            "peak_angular_speed_radps": float(max(float(row["profile"]["angular_speed_radps"]["max"]) for row in cohort)),
            "safety_gate_counts": {decision: int(sum(row["safety_gate_decision"] == decision for row in cohort)) for decision in ("ACCEPT", "WARN", "REJECT")},
            "finite_count": int(sum(bool(row["finite"]) for row in cohort)),
            "abi_valid_count": int(sum(bool(row["abi_valid"]) for row in cohort)),
        }
    return {"surfaces": by_surface}


def screen_rule(aggregates: Mapping[str, Mapping[str, Any]], rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    c80 = aggregates["C80"]["surfaces"]
    s25 = aggregates["S25"]["surfaces"]
    c80_support = [row for row in rows_by_arm["C80"] if row["surface"] == "STATE_SUPPORT20"]
    s25_support = [row for row in rows_by_arm["S25"] if row["surface"] == "STATE_SUPPORT20"]
    c80_by_id = {row["window_id"]: row for row in c80_support}
    endpoint_wins = sum(float(row["metrics"]["endpoint_error_m"]) < float(c80_by_id[row["window_id"]]["metrics"]["endpoint_error_m"]) for row in s25_support)
    conditions = {
        "support_mean_trajectory_error_lower": s25["STATE_SUPPORT20"]["mean_trajectory_error_m"] < c80["STATE_SUPPORT20"]["mean_trajectory_error_m"],
        "support_mean_endpoint_error_lower": s25["STATE_SUPPORT20"]["mean_endpoint_error_m"] < c80["STATE_SUPPORT20"]["mean_endpoint_error_m"],
        "support_endpoint_wins_at_least_12_of_20": endpoint_wins >= 12,
        "full24_mean_trajectory_error_within_1_10x": s25["FULL24"]["mean_trajectory_error_m"] <= 1.10 * c80["FULL24"]["mean_trajectory_error_m"],
        "full24_mean_endpoint_error_within_1_10x": s25["FULL24"]["mean_endpoint_error_m"] <= 1.10 * c80["FULL24"]["mean_endpoint_error_m"],
        "no_new_nonfinite_abi_or_reject": s25["FULL24"]["finite_count"] == 24 and s25["STATE_SUPPORT20"]["finite_count"] == 20 and s25["FULL24"]["abi_valid_count"] == 24 and s25["STATE_SUPPORT20"]["abi_valid_count"] == 20 and sum(s25[surface]["safety_gate_counts"]["REJECT"] for surface in SURFACES) <= sum(c80[surface]["safety_gate_counts"]["REJECT"] for surface in SURFACES),
    }
    return {"classification": "S25_MODEL_SCREEN_POSITIVE" if all(conditions.values()) else "S25_MODEL_SCREEN_NOT_QUALIFIED", "conditions": conditions, "support_endpoint_wins": endpoint_wins, "support_endpoint_total": len(s25_support)}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    output = args.output.resolve()
    require(not output.exists() and not output.is_symlink(), f"screen output must be fresh: {output}")
    gpu = gpu_identity()
    audit = audit_inputs(args, gpu)
    env = training.prepare_runtime(output, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=False)
    write_json(output / "pre_model_audit.json", {"audit": audit, "runtime_environment": env, "forbidden_science": {"Validation-E": 0, "Holdout-F": 0, "training": 0, "backward": 0, "optimizer_steps": 0, "WBC": 0, "simulator": 0, "hardware": 0}})
    require(not any(name == "torch" or name.startswith(("torch.", "transformers", "gr00t", "isaacgym")) for name in sys.modules), "model runtime imported before pre-model seal")

    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
    runtime_processor, embodiment, modality_configs = _build_runtime_processor(groot_source=args.groot_source.resolve(), base_checkpoint=args.e256.resolve(), dataset_dir=args.dataset.resolve(), modality_path=args.modality.resolve(), cosmos_snapshot=args.cosmos_root.resolve() / "snapshots" / COSMOS_REVISION)
    full_windows = stage45c._prepare_train24_windows(runtime_processor, embodiment, modality_configs, args.dataset.resolve(), audit["descriptor_closure"]["snapshots"])
    require(len(full_windows) == 24, "FULL24 window materialization drift")
    for index, window in enumerate(full_windows):
        window["surface"] = "FULL24"
        window["index"] = index
    support = training.load_rows(args.rows, args.support_root)[0]
    support_windows_list = support_windows(runtime_processor, embodiment, support, args.support_root.resolve())
    windows = full_windows + support_windows_list
    require(len(windows) == 44, "screen window count drift")
    write_json(output / "screen_manifest.json", {"schema": "LOCALPC_PI0208_HOPPER_G1_SCREEN_MANIFEST_V1", "status": "SEALED_BEFORE_MODEL_LOAD", "checkpoint_order": [ARM_SPECS[key]["label"] for key in ARM_ORDER], "surface_order": list(SURFACES), "window_ids": {surface: [window["window_id"] for window in windows if window["surface"] == surface] for surface in SURFACES}, "seed": SCREEN_SEED, "noise_shape": list(RAW_SHAPE), "same_noise_across_checkpoints": True, "sampler": {"timesteps": list(stage45c.TIMESTEP_IDS), "dt": stage45c.SAMPLER_DT, "carried_state": True, "official_public_sampler_called": False}, "decoder_stats_sha256": STATS_SHA256, "target_xyz_in_model_input": False, "maximum_unique_outputs": 176})
    torch, load_structured_model = stage45c._load_torch_runtime()
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(any(name in torch.cuda.get_device_name(torch.device("cuda:0")).upper() for name in ("H100", "H200")), "runtime CUDA device is not H100/H200")
    noise_bank = _noise_bank(torch, len(windows))
    noise_hashes = [sha256_bytes(noise.detach().cpu().numpy().tobytes()) for noise in noise_bank]
    write_json(output / "sampler_noise_manifest.json", {"schema": "LOCALPC_PI0208_G1_NOISE_BANK_V1", "seed": SCREEN_SEED, "shape": list(RAW_SHAPE), "dtype": "float32", "window_ids": [window["window_id"] for window in windows], "noise_sha256": noise_hashes, "same_noise_across_checkpoints": True})
    raw_manifest = output / "raw_predictions.jsonl"
    derived_manifest = output / "derived_rows.jsonl"
    rows_by_arm: dict[str, list[dict[str, Any]]] = {key: [] for key in ARM_ORDER}
    aggregates: dict[str, Any] = {}
    counters = {"model_construction": 0, "conditioning_passes": 0, "action_head_calls": 0, "model_forward": 0, "raw_prediction_count": 0, "derived_row_count": 0}
    runtime_sha = sha256_bytes(json.dumps({"head": audit["checkout"]["head"], "decoder": stage45c.DECODER_SOURCE_SHA256, "metrics": stage45c.METRICS_SOURCE_SHA256, "stats": STATS_SHA256}, sort_keys=True).encode())
    checkpoint_paths = {"E256": args.e256.resolve(), "C80": args.c80.resolve(), "S25": args.s25.resolve(), "S50": args.s50.resolve()}
    for key in ARM_ORDER:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = load_structured_model(checkpoint_paths[key], "cuda:0")
        model.eval()
        counters["model_construction"] += 1
        hook_count = {"count": 0}
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_count.__setitem__("count", hook_count["count"] + 1))
        arm_rows = []
        try:
            for index, window in enumerate(windows):
                with torch.inference_mode():
                    raw, action_calls = stage45c._sample_one(model, window["model_inputs"], noise_bank[index], torch)
                counters["conditioning_passes"] += 1
                counters["action_head_calls"] += action_calls
                counters["model_forward"] += action_calls
                counters["raw_prediction_count"] += 1
                raw_np = raw.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
                require(raw_np.shape == RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output invalid: {key}/{window['window_id']}")
                raw_path = output / "raw" / key / f"{window['surface'].lower()}-{index:02d}.npy"
                write_npy(raw_path, raw_np)
                raw_file_sha = sha256_file(raw_path)
                raw_array_sha = sha256_bytes(raw_np.tobytes())
                append_jsonl(raw_manifest, {"schema": "LOCALPC_PI0208_G1_RAW_V1", "checkpoint": ARM_SPECS[key], "surface": window["surface"], "window_id": window["window_id"], "path": str(raw_path.relative_to(output)), "file_sha256": raw_file_sha, "array_sha256": raw_array_sha, "shape": list(raw_np.shape), "dtype": "float32", "noise_sha256": noise_hashes[index], "sampler_calls": action_calls, "decoded": False})
                snapshot = dict(window["snapshot"])
                snapshot["surface"] = window["surface"]
                physical, details = stage45c._decode_v2(raw_np, snapshot["state_v1"], args.stats.resolve(), snapshot["source_timestamps"], float(snapshot["observation_timestamp"]), f"{key}:{window['window_id']}")
                require(np.asarray(physical).shape == PHYSICAL_SHAPE and bool(np.isfinite(physical).all()), f"decoded output invalid: {key}/{window['window_id']}")
                row = stage45c._evaluation_row(raw_np, physical, details, {"snapshot": snapshot, "window_id": window["window_id"]}, {"label": ARM_SPECS[key]["label"], "tree_sha256": ARM_SPECS[key]["tree"]}, runtime_sha, str(raw_path.relative_to(output)), raw_file_sha)
                decoded_np = np.asarray(physical, dtype=np.float64)
                decoded_path = output / "decoded" / key / f"{window['surface'].lower()}-{index:02d}.npy"
                write_npy(decoded_path, decoded_np)
                profile, gate = _profile_gate(decoded_np, snapshot, f"{key}:{window['window_id']}")
                row.update({"surface": window["surface"], "support_origin_class": window.get("support_origin_class"), "decoded_artifact": {"path": str(decoded_path.relative_to(output)), "file_sha256": sha256_file(decoded_path), "array_sha256": sha256_bytes(np.ascontiguousarray(decoded_np).tobytes()), "shape": list(decoded_np.shape)}, "profile": profile, "safety_gate": gate, "safety_gate_decision": gate["decision"], "finite": True, "abi_valid": True, "target_xyz_in_model_input": False, "trajectory_correction_applied": False})
                row_path = output / "rows" / key / f"{window['surface'].lower()}-{index:02d}.json"
                write_json(row_path, row)
                append_jsonl(derived_manifest, {"schema": "LOCALPC_PI0208_G1_DERIVED_V1", "checkpoint": ARM_SPECS[key], "surface": window["surface"], "window_id": window["window_id"], "raw_path": str(raw_path.relative_to(output)), "decoded_path": str(decoded_path.relative_to(output)), "row_path": str(row_path.relative_to(output)), "decoded": True, "decoded_array_sha256": row["decoded_action"]["array_sha256"], "metric_semantics_source": "go2arx5_vla.evaluation.trajectory_metrics.comparison_metrics"})
                append_jsonl(raw_manifest, {"schema": "LOCALPC_PI0208_G1_RAW_REFERENCE_V1", "checkpoint": ARM_SPECS[key], "surface": window["surface"], "window_id": window["window_id"], "raw_path": str(raw_path.relative_to(output)), "decoded": True, "decoded_array_sha256": row["decoded_action"]["array_sha256"]})
                counters["derived_row_count"] += 1
                arm_rows.append(row)
        finally:
            hook.remove()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        require(hook_count["count"] == len(windows) * len(stage45c.TIMESTEP_IDS), f"action-head count drift: {key}")
        require(len(arm_rows) == len(windows), f"row count drift: {key}")
        rows_by_arm[key] = arm_rows
        aggregates[key] = {"checkpoint": ARM_SPECS[key], "summary": summarize(arm_rows), "action_head_calls": hook_count["count"], "conditioning_passes": len(windows), "model_side_calls": len(windows) + hook_count["count"], "model_load_and_screen_elapsed_s": time.perf_counter() - started, "peak_memory_allocated_bytes": peak_allocated, "peak_memory_reserved_bytes": peak_reserved}
        del model
        torch.cuda.empty_cache()
    require(counters["model_construction"] == 4 and counters["conditioning_passes"] == 176 and counters["action_head_calls"] == 704 and counters["model_forward"] == 704 and counters["raw_prediction_count"] == 176 and counters["derived_row_count"] == 176, f"screen counter drift: {counters}")
    result = {"schema": "LOCALPC_PI0208_HOPPER_G1_MODEL_SCREEN_V1", "status": "PASS", "checkpoint_order": [ARM_SPECS[key]["label"] for key in ARM_ORDER], "surface_counts": {surface: len([window for window in windows if window["surface"] == surface]) for surface in SURFACES}, "aggregates": aggregates, "counters": counters, "gpu": gpu, "runtime": {"host": os.uname().nodename, "python": sys.version, "cuda": str(torch.version.cuda), "runtime_sha256": runtime_sha, "environment": env}, "scientific_zeroes": {"Validation-E": 0, "Holdout-F": 0, "training": 0, "backward": 0, "optimizer_steps": 0, "WBC": 0, "simulator": 0, "hardware": 0}, "screen_rule": screen_rule({key: aggregates[key]["summary"] for key in ARM_ORDER}, rows_by_arm), "rows_by_arm": {key: [{"surface": row["surface"], "window_id": row["window_id"], "endpoint_error_m": row["metrics"]["endpoint_error_m"], "trajectory_error_m": row["metrics"]["position_rmse_m"], "safety_gate_decision": row["safety_gate_decision"]} for row in rows] for key, rows in rows_by_arm.items()}}
    write_json(output / "screen_result.json", result)
    result["terminal_token"] = "LOCALPC_PI0208_HOPPER_MODEL_SCREEN — " + ("S25_HOPPER_SCREEN_QUALIFIED" if result["screen_rule"]["classification"] == "S25_MODEL_SCREEN_POSITIVE" else "S25_HOPPER_SCREEN_NOT_QUALIFIED")
    write_json(output / "screen_seal.json", result)
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"evidence_inventory.json"}:
            inventory.append({"path": str(path.relative_to(output).as_posix()), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    write_json(output / "evidence_inventory.json", {"schema": "LOCALPC_PI0208_G1_EVIDENCE_INVENTORY_V1", "entries": inventory, "entry_count": len(inventory)})
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-head", default=EXPECTED_HEAD)
    p.add_argument("--e256", type=Path, required=True)
    p.add_argument("--c80", type=Path, required=True)
    p.add_argument("--s25", type=Path, required=True)
    p.add_argument("--s50", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--support-root", type=Path, required=True)
    p.add_argument("--groot-source", type=Path, required=True)
    p.add_argument("--groot-manifest", type=Path, required=True)
    p.add_argument("--cosmos-root", type=Path, required=True)
    p.add_argument("--modality", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--ffmpeg-root", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        result = execute(args)
        print(json.dumps({"status": result["status"], "terminal_token": result["terminal_token"], "screen_rule": result["screen_rule"], "counters": result["counters"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema": "LOCALPC_PI0208_HOPPER_G1_FAILURE_V1", "status": "BLOCKED_MATERIAL", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
