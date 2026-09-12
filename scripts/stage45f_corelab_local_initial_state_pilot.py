#!/usr/bin/env python3
"""PI0152 bounded Corelab local-policy characterization.

The default/audit and preflight paths are CPU/static.  ``execute`` is the one
explicit model attempts authorized by PI0152: one fixed N2 seed2223@2000
checkpoint, five sealed Stage43 q0 observations plus one sealed TRAIN24
reference, common seed-4242 noise, and the accepted four-step T16 action-head
graph.  All node-local paths are supplied late; no path is guessed by the
repository manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

STAGE = "Stage45F"
PI_GATE = "PI_GATE_STATE_0152"
PI_COMMENT = 5647237688
PR_POINTER_COMMENT = 5647244579
RETURN_TOKEN = "CORELAB_PI0152_POLICY_CHARACTERIZATION_FOR_PI_REVIEW"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
BASE_HEAD = "9fef9064979192b66161e91ed0e7d0ab505b0364"
LOCAL_SEED = 2223
TRAIN_STEPS = 2000
CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
TRAINING_RESULT_SHA = "f6c5c37bf3f9e86eee026f49599262d0894471f033bdb3d59662219d2f6466bd"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
# UUIDs recorded by the accepted Corelab RTX inventory.  Runtime selection is
# always made from live nvidia-smi output and remains UUID-bound.
GPU_ALLOWLIST = {
    "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad": {
        "host": "corelab-4080s", "name_substring": "RTX 4080 SUPER", "min_total_mib": 30000, "role": "preferred_32gb"
    },
    "GPU-845348f1-1511-dc19-0877-cc97658e3edd": {
        "host": "corelab-4080s", "name_substring": "RTX 4080 SUPER", "min_total_mib": 15000, "role": "fallback_16gb"
    },
}
PREFERRED_GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
MIN_FREE_MIB = 14 * 1024
MIN_OUTPUT_FREE_BYTES = 8 * 1024**3
HORIZON = 40
UNIVERSAL_DIM = 132
STATE_DIM = 10
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
TIMESTEP_IDS = (0, 250, 500, 750)
SAMPLER_DT = 0.25
DEVELOPMENT_SEED = 4242
LANGUAGE = "move the end effector to the red target"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
TARGET_WORLD_M = (0.30, 0.10, 0.58)
CONDITIONS = ("C0_NOMINAL", "C1_X_PLUS_5CM", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM")
TRAIN_WINDOW_ID = "stage22_expert_red_target_train_a:expert-0000"
MANIFEST_PATH = REPO / "manifests/STAGE45F_CORELAB_LOCAL_INITIAL_STATE_PILOT_V1.json"
EXPECTED_NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
EXPECTED_NOISE_FILE_SHA = "83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922"
MAX_NEW_ATTEMPTS = 2
DEFAULT_ATTEMPT_NAME = "recovery-01"


class PilotError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"required regular JSON missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot parse JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return sha256_file(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def _safe_dir(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def _safe_file(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def _manifest() -> Mapping[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "CORELAB_STAGE45F_LOCAL_INITIAL_STATE_PILOT_V1", "pilot manifest schema drift")
    require(value.get("authority", {}).get("pi_gate") == PI_GATE, "pilot authority drift")
    require(value.get("authority", {}).get("issue_comment") == PI_COMMENT, "pilot issue authority drift")
    require(value.get("repository", {}).get("branch") == BRANCH, "pilot branch drift")
    require(value.get("local_reference", {}).get("checkpoint_tree_sha256") == CHECKPOINT_TREE, "pilot checkpoint drift")
    require(value.get("sampler", {}).get("timestep_ids") == list(TIMESTEP_IDS), "pilot sampler drift")
    return value


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    names = ("stage43_root", "n2_root", "n2_stats", "runtime_stats", "base_model", "dataset", "groot", "cosmos", "modality", "config", "snapshot_preflight")
    result: dict[str, Path] = {}
    for name in names:
        value = getattr(args, name, None)
        require(value is not None, f"missing late-bound path --{name.replace('_', '-')}")
        result[name] = Path(value).resolve()
    return result


def _train_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the accepted Stage44 TRAIN pixel snapshot late-bound.

    The Stage45C video materializer is intentionally not used as a source of
    truth here: PI0152 permits consuming a complete, source-bound canonical
    pixel artifact when the video dependency is broken.  Both paths therefore
    remain explicit execution arguments rather than repository defaults.
    """
    npz_value = getattr(args, "train_input_npz", None)
    manifest_value = getattr(args, "train_input_manifest", None)
    require(npz_value is not None, "missing late-bound path --train-input-npz")
    require(manifest_value is not None, "missing late-bound path --train-input-manifest")
    return Path(npz_value).resolve(), Path(manifest_value).resolve()


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()


def _parse_gpu_rows(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, f"malformed nvidia-smi row: {line!r}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    return rows


def _select_gpu_row(rows: Sequence[Mapping[str, Any]], host: str, expected_uuid: str | None = None) -> dict[str, Any]:
    candidates = []
    for row in rows:
        policy = GPU_ALLOWLIST.get(str(row.get("uuid")))
        if policy is None or policy["host"] != host:
            continue
        if policy["name_substring"] not in str(row.get("name")):
            continue
        if int(row.get("memory_total_mib", 0)) < int(policy["min_total_mib"]):
            continue
        if int(row.get("memory_free_mib", 0)) < MIN_FREE_MIB:
            continue
        candidates.append({**dict(row), "allowlist_role": policy["role"]})
    if expected_uuid is not None:
        selected = [row for row in candidates if row["uuid"] == expected_uuid]
        require(len(selected) == 1, f"requested UUID is not an eligible live allowlist device: {expected_uuid!r}; observed={list(rows)!r}")
        return selected[0]
    require(candidates, f"no eligible live allowlist device: host={host!r}, observed={list(rows)!r}")
    candidates.sort(key=lambda row: (row["allowlist_role"] != "preferred_32gb", -row["memory_free_mib"], row["index"]))
    return candidates[0]


def _gpu_process_observation() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    require(query.returncode == 0, f"nvidia-smi process query failed: {query.stderr[-500:]}")
    processes = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 4, f"malformed process query row: {line!r}")
        pid = int(fields[1])
        owner = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True, check=False)
        require(owner.returncode == 0 and owner.stdout.strip(), f"process owner query failed for pid {pid}")
        processes.append({"gpu_uuid": fields[0], "pid": pid, "process_name": fields[2], "used_memory_mib": int(fields[3]), "owner": owner.stdout.strip()})
    return {"query_sha256": sha256_bytes(query.stdout.encode("utf-8")), "processes": processes, "query_ok": True}


def _gpu_observation(expected_uuid: str | None = None) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-500:]}")
    rows = _parse_gpu_rows(query.stdout)
    host = subprocess.check_output(["hostname"], text=True).strip()
    row = _select_gpu_row(rows, host, expected_uuid)
    row = {**row, "host": host, "all_gpu_rows": rows, "process_observation": _gpu_process_observation()}
    row["query_sha256"] = sha256_bytes(query.stdout.encode("utf-8"))
    row["admission_free_mib_min"] = MIN_FREE_MIB
    row["admission_pass"] = row["memory_free_mib"] >= MIN_FREE_MIB
    require(row["admission_pass"], f"pinned GPU has insufficient free memory: {row}")
    return row


def _stage43_audit(root: Path) -> dict[str, Any]:
    import numpy as np
    import stage43_corelab_evidence as ev

    _safe_dir(root, "Stage43 root")
    bundle_path = root / "stage43_execution_bundle.json"
    bundle = read_json(bundle_path)
    ev.validate_stage43_bundle(bundle)
    ev.verify_stage43_bundle_files(bundle, repo_root=REPO)
    bundle_sha = sha256_file(bundle_path)
    conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_root = root / "conditions" / condition
        _safe_dir(condition_root, condition)
        manifest = read_json(condition_root / "condition_manifest.json")
        ev.verify_condition_manifest(manifest, condition_root)
        result = read_json(condition_root / "condition_result.json")
        require(result.get("status") == "PASS" and result.get("condition_id") == condition, f"Stage43 condition result is not PASS: {condition}")
        metrics = result.get("metrics")
        require(isinstance(metrics, Mapping) and metrics.get("target_world_m") == list(TARGET_WORLD_M), f"Stage43 target metric drift: {condition}")
        state_doc = read_json(condition_root / "state_v1.json")
        observations = state_doc.get("observations")
        require(isinstance(observations, list) and len(observations) == 4, f"Stage43 observation count drift: {condition}")
        q0 = next((item for item in observations if item.get("request_id") == f"{condition}-q0"), None)
        require(isinstance(q0, Mapping), f"Stage43 q0 observation missing: {condition}")
        state = np.asarray(q0.get("state_v1"), dtype=np.float32)
        require(state.shape == (STATE_DIM,) and np.isfinite(state).all(), f"Stage43 q0 StateV1 invalid: {condition}")
        body_path = condition_root / "observation_body.npz"
        wrist_path = condition_root / "observation_wrist.npz"
        with np.load(body_path, allow_pickle=False) as body_npz, np.load(wrist_path, allow_pickle=False) as wrist_npz:
            body = np.asarray(body_npz[f"{condition}-q0"])
            wrist = np.asarray(wrist_npz[f"{condition}-q0"])
        require(body.shape == (96, 96, 3) and body.dtype == np.uint8, f"Stage43 body RGB ABI drift: {condition}")
        require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, f"Stage43 wrist RGB ABI drift: {condition}")
        require(np.isfinite(body.astype(np.float32)).all() and np.isfinite(wrist.astype(np.float32)).all(), f"Stage43 RGB nonfinite: {condition}")
        request = q0.get("request", {})
        require(request.get("language") == LANGUAGE and request.get("target_position_world_m") == list(TARGET_WORLD_M), f"Stage43 task contract drift: {condition}")
        conditions.append({
            "condition_id": condition,
            "condition_manifest_sha256": sha256_file(condition_root / "condition_manifest.json"),
            "q0": {
                "request_id": q0["request_id"],
                "body_sha256": sha256_file(body_path),
                "wrist_sha256": sha256_file(wrist_path),
                "body_array_sha256": sha256_bytes(body.tobytes()),
                "wrist_array_sha256": sha256_bytes(wrist.tobytes()),
                "state_v1_sha256": str(q0["state_v1_sha256"]),
                "state_v1": state.astype(float).tolist(),
                "observation_timestamp": float(q0["observation_timestamp"]),
            },
            "target_world_m": list(TARGET_WORLD_M),
            "anchor_world_m": metrics.get("initial_eef_world_m"),
            "task_metrics_source": "Stage43 condition_result.metrics",
            "target_xyz_in_model_input": False,
        })
    return {
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "bundle_schema": bundle.get("schema"),
        "repository_head": bundle.get("repository", {}).get("head"),
        "condition_order": list(CONDITIONS),
        "conditions": conditions,
        "target_world_m": list(TARGET_WORLD_M),
        "target_xyz_in_model_input": False,
    }


def _runtime_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    import stage45e_n2_gpu_sprint as sprint
    import stage45e_n2_statistics as n2

    args = argparse.Namespace(
        base_model=paths["base_model"], dataset=paths["dataset"], groot=paths["groot"], cosmos=paths["cosmos"],
        modality=paths["modality"], config=paths["config"], runtime_stats=paths["runtime_stats"], n2_stats=paths["n2_stats"],
    )
    identities = sprint._verify_core_identities(paths)
    n2_root = sprint._verify_n2_checkpoint_root(paths["n2_root"], LOCAL_SEED, paths["n2_stats"])
    require(n2_root["result_sha256"] == TRAINING_RESULT_SHA, f"N2 training result SHA drift: {n2_root['result_sha256']}")
    require(n2_root["checkpoints"]["2000"]["tree_sha256"] == CHECKPOINT_TREE, "N2 @2000 checkpoint tree drift")
    require(_git_head(paths["groot"]) == GROOT_COMMIT, "GR00T source commit drift")
    cosmos_ref = paths["cosmos"] / "refs" / "main"
    _safe_file(cosmos_ref, "Cosmos refs/main")
    require(cosmos_ref.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    runtime_doc = read_json(paths["runtime_stats"])
    n2_doc = read_json(paths["n2_stats"])
    require(sha256_file(paths["n2_stats"]) == N2_STATS_SHA, "N2 stats SHA drift")
    require(n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats payload drift")
    return {
        "base_tree_sha256": identities["base_tree_sha256"],
        "full24_dataset": {key: identities["full24"].get(key) for key in ("tree_sha256", "window_count", "frame_count", "total_bytes")},
        "runtime_stats": {"outer_sha256": identities["runtime_stats"]["outer_sha256"], "payload_sha256": identities["runtime_stats"]["payload_sha256"]},
        "n2_stats_sha256": sha256_file(paths["n2_stats"]),
        "n2_stats_payload_sha256": n2_doc.get("statistics_payload_sha256"),
        "n2_training_result_sha256": n2_root["result_sha256"],
        "n2_checkpoint_1000_tree_sha256": n2_root["checkpoints"]["1000"]["tree_sha256"],
        "n2_checkpoint_2000_tree_sha256": n2_root["checkpoints"]["2000"]["tree_sha256"],
        "n2_sidecar_sha256": n2_root["sidecar_sha256"],
        "groot_commit": _git_head(paths["groot"]),
        "cosmos_revision": cosmos_ref.read_text(encoding="utf-8").strip(),
        "cosmos_tree_expected": COSMOS_TREE,
        "modality_sha256": sha256_file(paths["modality"]),
        "config_sha256": sha256_file(paths["config"]),
        "runtime_stats_file_sha256": sha256_file(paths["runtime_stats"]),
        "runtime_stats_payload_sha256": runtime_doc.get("statistics_payload_sha256"),
    }


def _train_snapshot_audit(paths: Mapping[str, Path]) -> dict[str, Any]:
    import stage45e_n2_train24_eval as ev

    preflight = read_json(paths["snapshot_preflight"])
    require(preflight.get("status") == "PASS", "sealed Stage45C preflight is not PASS")
    snapshots = ev._snapshot_lookup(preflight.get("snapshots", []))
    train = next((item for item in snapshots if item.get("window_id") == TRAIN_WINDOW_ID), None)
    require(isinstance(train, Mapping), "sealed TRAIN reference snapshot is missing")
    require(train.get("target_xyz_in_model_input") in (False, None) or preflight.get("canonical_label_source", {}).get("target_xyz_in_model_input") is False, "TRAIN target channel is not excluded")
    return {
        "snapshot_preflight_path": str(paths["snapshot_preflight"]),
        "snapshot_preflight_sha256": sha256_file(paths["snapshot_preflight"]),
        "window_id": TRAIN_WINDOW_ID,
        "state_v1_sha256": sha256_bytes(json.dumps(train["state_v1"], separators=(",", ":")).encode("utf-8")),
        "target_id": train.get("target_id"),
        "target_world_m": train.get("target_world_m"),
        "anchor_world_m": train.get("anchor_world_m"),
        "target_xyz_in_model_input": False,
    }


def collect_audit(args: argparse.Namespace) -> dict[str, Any]:
    _manifest()
    require(git("branch", "--show-current") == BRANCH, "pilot branch mismatch")
    require(git("rev-parse", "HEAD") != "", "pilot HEAD missing")
    paths = _paths(args)
    stage43 = _stage43_audit(paths["stage43_root"])
    runtime = _runtime_audit(paths)
    train = _train_snapshot_audit(paths)
    return {
        "schema": "CORELAB_STAGE45F_PI0152_LOCAL_INITIAL_STATE_AUDIT_V1",
        "status": "PASS",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_pointer_comment": PR_POINTER_COMMENT},
        "repository": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "base_head": BASE_HEAD, "worktree_clean": git("status", "--porcelain=v1") == ""},
        "stage43": stage43,
        "runtime": runtime,
        "train_reference": train,
        "budget": {"observations": 6, "conditioning_passes": 6, "action_head_calls": 24, "model_forward_windows": 6},
        "model_forward": 0,
        "gpu": False,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": False,
        "simulator": False,
        "hardware": False,
    }


def preflight(args: argparse.Namespace, *, output: Path | None = None) -> dict[str, Any]:
    audit = collect_audit(args)
    gpu = _gpu_observation(args.gpu_uuid)
    paths = _paths(args)
    output_root = Path(args.output_root).resolve() if args.output_root else REPO / "stage45f_pi0152_output"
    output_parent = output_root.parent
    _safe_dir(output_parent, "output parent")
    dataset_free_bytes = shutil.disk_usage(paths["dataset"]).free
    output_free_bytes = shutil.disk_usage(output_parent).free
    require(output_free_bytes >= MIN_OUTPUT_FREE_BYTES, f"insufficient output/cache free storage: {output_free_bytes}")
    value = {
        "schema": "CORELAB_STAGE45F_PI0152_LOCAL_INITIAL_STATE_PREFLIGHT_V1",
        "status": "PASS",
        "audit": audit,
        "gpu": gpu,
        "storage": {
            "output_parent": str(output_parent),
            "output_mount_free_bytes": output_free_bytes,
            "dataset_mount_free_bytes": dataset_free_bytes,
            "minimum_output_free_bytes": MIN_OUTPUT_FREE_BYTES,
            "estimated_new_payload_bytes": 512 * 1024 * 1024,
        },
        "selected_gpu_uuid": gpu["uuid"],
        "selected_gpu_index": gpu["index"],
        "model_attempt_started": False,
        "model_construction": 0,
        "conditioning_passes": 0,
        "action_head_calls": 0,
        "gr00t_forwards": 0,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "wbc": False,
        "simulator": False,
        "hardware": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if output is not None:
        write_json(output, value)
    return value


def _prepare_env(paths: Mapping[str, Path], output: Path, gpu_uuid: str, ffmpeg_runtime: Path) -> dict[str, Any]:
    """Prepare an isolated process environment for the recovery generation.

    Only PATH/LD_LIBRARY_PATH are changed, and only for this worker.  The
    system FFmpeg/driver and other users' environments are never modified.
    """
    ffmpeg_runtime = Path(ffmpeg_runtime).resolve()
    _safe_dir(ffmpeg_runtime, "FFmpeg runtime root")
    ffmpeg_bin = ffmpeg_runtime / "bin" / "ffmpeg"
    ffprobe_bin = ffmpeg_runtime / "bin" / "ffprobe"
    ffmpeg_lib = ffmpeg_runtime / "lib"
    _safe_file(ffmpeg_bin, "isolated FFmpeg binary")
    _safe_file(ffprobe_bin, "isolated FFprobe binary")
    _safe_dir(ffmpeg_lib, "isolated FFmpeg library directory")
    version = subprocess.run([str(ffmpeg_bin), "-version"], capture_output=True, text=True, check=False)
    require(version.returncode == 0, f"isolated FFmpeg version probe failed: {version.stderr[-500:]}")
    first_line = version.stdout.splitlines()[0] if version.stdout.splitlines() else ""
    match = re.search(r"ffmpeg version (\d+)", first_line)
    require(match is not None, f"cannot parse isolated FFmpeg version: {first_line!r}")
    major = int(match.group(1))
    require(4 <= major <= 7, f"isolated FFmpeg major is incompatible with torchcodec 0.8.0: {major}")
    cache = output / "cache"
    tmp = output / "tmp"
    cache.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    hub = cache / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    (cache / "xdg").mkdir(parents=True, exist_ok=False)
    cosmos_link = hub / "models--nvidia--Cosmos-Reason2-2B"
    cosmos_link.symlink_to(paths["cosmos"], target_is_directory=True)
    old_path = os.environ.get("PATH", "")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    path_value = str(ffmpeg_runtime / "bin") + (os.pathsep + old_path if old_path else "")
    ld_value = str(ffmpeg_lib) + (os.pathsep + old_ld if old_ld else "")
    os.environ.update({
        # UUID binding remaps the selected physical card to cuda:0 inside the
        # worker; never infer the device from a mutable ordinal.
        "CUDA_VISIBLE_DEVICES": str(gpu_uuid),
        "NVIDIA_VISIBLE_DEVICES": str(gpu_uuid),
        "PATH": path_value,
        "LD_LIBRARY_PATH": ld_value,
        "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "TRANSFORMERS_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "STAGE45E_N2_STATS_ARTIFACT": str(paths["n2_stats"]),
    })
    return {
        "root": str(ffmpeg_runtime),
        "ffmpeg": str(ffmpeg_bin),
        "ffprobe": str(ffprobe_bin),
        "lib_dir": str(ffmpeg_lib),
        "version_first_line": first_line,
        "major": major,
        "path_prefix": str(ffmpeg_runtime / "bin"),
        "ld_library_path_prefix": str(ffmpeg_lib),
    }


def _stage43_model_inputs(processor: Any, embodiment: Any, condition: Mapping[str, Any], condition_root: Path) -> Mapping[str, Any]:
    import numpy as np
    from gr00t.data.types import MessageType, VLAStepData

    condition_id = str(condition["condition_id"])
    q0 = condition["q0"]
    with np.load(condition_root / "observation_body.npz", allow_pickle=False) as body_npz, np.load(condition_root / "observation_wrist.npz", allow_pickle=False) as wrist_npz:
        body = np.asarray(body_npz[f"{condition_id}-q0"])
        wrist = np.asarray(wrist_npz[f"{condition_id}-q0"])
    state = np.asarray(q0["state_v1"], dtype=np.float32)
    step = VLAStepData(
        images={"external_rgb": [body], "wrist_rgb": [wrist]},
        states={"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]},
        actions={},
        text=LANGUAGE,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    batch = processor.collator([processed])
    model_inputs = batch.get("inputs", {})
    require(isinstance(model_inputs, Mapping), f"Stage43 model input is not a mapping: {condition_id}")
    require("action" not in model_inputs and "action_mask" not in model_inputs, f"Stage43 action label leaked: {condition_id}")
    for forbidden in ("target_position_world_m", "target_xyz", "target_world_m"):
        require(forbidden not in model_inputs, f"Stage43 target leaked into model input: {condition_id}")
    return model_inputs


def _read_ppm_rgb(path: Path) -> Any:
    """Read a canonical P6 RGB frame without introducing an image library."""
    import numpy as np

    _safe_file(path, "canonical RGB frame")
    payload = path.read_bytes()
    cursor = 0

    def token() -> bytes:
        nonlocal cursor
        while cursor < len(payload) and payload[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(payload) and payload[cursor] == ord("#"):
            while cursor < len(payload) and payload[cursor] not in b"\r\n":
                cursor += 1
            return token()
        start = cursor
        while cursor < len(payload) and payload[cursor] not in b" \t\r\n":
            cursor += 1
        return payload[start:cursor]

    require(token() == b"P6", f"canonical RGB frame is not P6: {path}")
    width, height, maximum = int(token()), int(token()), int(token())
    require((width, height, maximum) == (96, 96, 255), f"canonical RGB frame ABI drift: {path}")
    while cursor < len(payload) and payload[cursor] in b" \t\r\n":
        cursor += 1
    raw = payload[cursor:]
    require(len(raw) == width * height * 3, f"canonical RGB frame byte count drift: {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _video_frame_probe(dataset: Path, ffmpeg_runtime: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Probe frame zero under the repaired decoder, never loading a model.

    The accepted Stage44 NPZ is the canonical input when the materialized
    LeRobot video is not byte-equivalent.  A mismatch is recorded explicitly,
    not silently treated as a successful video replay.
    """
    import numpy as np

    result: dict[str, Any] = {
        "decoder": "torchcodec.decoders.VideoDecoder",
        "runtime_root": str(Path(ffmpeg_runtime).resolve()),
        "frame_index": 0,
        "video_paths": {},
        "video_frame_matches_canonical": {},
        "status": "SKIPPED",
    }
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as exc:
        result.update({"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)[-2000:]})
        return result
    video_root = Path(dataset) / "videos" / "chunk-000"
    for key, relative in (
        ("body_rgb", Path("observation.images.external_rgb") / "episode_000000.mp4"),
        ("wrist_rgb", Path("observation.images.wrist_rgb") / "episode_000000.mp4"),
    ):
        path = video_root / relative
        if not path.is_file():
            result["video_paths"][key] = {"path": str(path), "status": "UNAVAILABLE"}
            continue
        try:
            decoder = VideoDecoder(str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0)
            packet = decoder.get_frames_at([0])
            frames = packet.data.detach().cpu().numpy()
            require(tuple(frames.shape) == (1, 96, 96, 3) and frames.dtype == np.uint8, f"decoded video frame ABI drift: {path}")
            frame = np.asarray(frames[0])
            actual_sha = sha256_bytes(frame.tobytes())
            expected_sha = str(expected[f"{key}_sha256"])
            result["video_paths"][key] = {
                "path": str(path), "file_sha256": sha256_file(path), "frame_shape": list(frame.shape),
                "frame_dtype": str(frame.dtype), "frame_sha256": actual_sha,
            }
            result["video_frame_matches_canonical"][key] = actual_sha == expected_sha
        except Exception as exc:
            result["video_paths"][key] = {"path": str(path), "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)[-2000:]}
            result["video_frame_matches_canonical"][key] = False
    result["status"] = "PASS" if all(key in result["video_paths"] and result["video_paths"][key].get("status") != "ERROR" for key in ("body_rgb", "wrist_rgb")) else "PARTIAL"
    result["canonical_input_consumer"] = "accepted_stage44_npz_when_video_frame_not_identical"
    return result


def qualify_train_pixels(paths: Mapping[str, Path], output: Path, ffmpeg_runtime: Path, train_input_npz: Path, train_input_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prove the fixed TRAIN pixel identity before any GR00T import.

    The accepted Stage44 input NPZ is source-bound by its manifest and exact
    array digests.  If the repaired video decoder yields different pixels, the
    canonical NPZ is consumed and the mismatch is preserved in evidence.
    """
    import numpy as np

    _safe_file(train_input_npz, "accepted TRAIN input NPZ")
    _safe_file(train_input_manifest, "accepted TRAIN input manifest")
    manifest = read_json(train_input_manifest)
    rows = manifest.get("rows", [])
    row = next((item for item in rows if item.get("example_id") == TRAIN_WINDOW_ID), None)
    require(isinstance(row, Mapping), f"accepted Stage44 TRAIN input row missing: {TRAIN_WINDOW_ID}")
    expected_file_sha = str(row.get("input_npz_sha256"))
    require(sha256_file(train_input_npz) == expected_file_sha, "accepted TRAIN input NPZ SHA drift")
    with np.load(train_input_npz, allow_pickle=False) as data:
        require(set(data.files) == {"body_rgb", "wrist_rgb", "state"}, "accepted TRAIN input key drift")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"])
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, "accepted TRAIN body RGB ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, "accepted TRAIN wrist RGB ABI drift")
    require(state.shape == (10,) and state.dtype == np.float32 and np.isfinite(state).all(), "accepted TRAIN StateV1 ABI drift")
    body_sha = sha256_bytes(body.tobytes()); wrist_sha = sha256_bytes(wrist.tobytes()); state_sha = sha256_bytes(state.tobytes())
    require(body_sha == str(row.get("body_sha256")), "accepted TRAIN body pixel SHA drift")
    require(wrist_sha == str(row.get("wrist_sha256")), "accepted TRAIN wrist pixel SHA drift")
    require(state_sha == str(row.get("state_v1_sha256")), "accepted TRAIN StateV1 SHA drift")
    require(row.get("target_xyz_in_model_input") is False, "accepted TRAIN target channel is not excluded")

    source_snapshot: dict[str, Any] = {"status": "UNAVAILABLE"}
    snapshot_path_value = row.get("snapshot_path")
    if snapshot_path_value:
        snapshot_path = Path(str(snapshot_path_value))
        if snapshot_path.is_file():
            snapshot = read_json(snapshot_path)
            source_snapshot = {"path": str(snapshot_path), "sha256": sha256_file(snapshot_path), "status": "PASS"}
            for key, array, field in (("body_rgb", body, "body_frame_path"), ("wrist_rgb", wrist, "wrist_frame_path")):
                source_path_value = snapshot.get(field)
                if source_path_value and Path(str(source_path_value)).is_file():
                    source_array = _read_ppm_rgb(Path(str(source_path_value)))
                    source_snapshot[f"{key}_path"] = str(Path(str(source_path_value)))
                    source_snapshot[f"{key}_file_sha256"] = sha256_file(Path(str(source_path_value)))
                    source_snapshot[f"{key}_array_sha256"] = sha256_bytes(source_array.tobytes())
                    source_snapshot[f"{key}_matches_canonical"] = bool(np.array_equal(source_array, array))
                    require(source_snapshot[f"{key}_matches_canonical"], f"canonical snapshot {key} pixels do not match accepted NPZ")
                else:
                    source_snapshot[f"{key}_status"] = "UNAVAILABLE"
        else:
            source_snapshot = {"path": str(snapshot_path), "status": "UNAVAILABLE"}

    expected = {"body_rgb_sha256": body_sha, "wrist_rgb_sha256": wrist_sha}
    video_probe = _video_frame_probe(paths["dataset"], ffmpeg_runtime, expected)
    qualification = {
        "schema": "CORELAB_STAGE45F_PI0152_TRAIN_PIXEL_QUALIFICATION_V1",
        "status": "PASS",
        "window_id": TRAIN_WINDOW_ID,
        "accepted_manifest_path": str(train_input_manifest),
        "accepted_manifest_sha256": sha256_file(train_input_manifest),
        "input_npz_path": str(train_input_npz),
        "input_npz_sha256": sha256_file(train_input_npz),
        "array_identity": {
            "body_rgb": {"shape": list(body.shape), "dtype": str(body.dtype), "sha256": body_sha},
            "wrist_rgb": {"shape": list(wrist.shape), "dtype": str(wrist.dtype), "sha256": wrist_sha},
            "state": {"shape": list(state.shape), "dtype": str(state.dtype), "sha256": state_sha},
        },
        "manifest_row_identity": {"body_sha256": row.get("body_sha256"), "wrist_sha256": row.get("wrist_sha256"), "state_v1_sha256": row.get("state_v1_sha256"), "target_xyz_in_model_input": False},
        "source_snapshot": source_snapshot,
        "video_probe": video_probe,
        "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
        "target_xyz_in_model_input": False,
        "consumer": "accepted_stage44_input_npz",
        "model_forward": 0,
        "gpu": False,
    }
    write_json(output / "train_pixel_qualification.json", qualification)
    return qualification, {"body": body, "wrist": wrist, "state": state}


def _train_model_inputs(processor: Any, embodiment: Any, arrays: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build the TRAIN reference from the source-bound canonical pixel arrays."""
    from gr00t.data.types import MessageType, VLAStepData

    step = VLAStepData(
        images={"external_rgb": [arrays["body"]], "wrist_rgb": [arrays["wrist"]]},
        states={"eef_9d": arrays["state"][:9][None, :], "gripper_position": arrays["state"][9:10][None, :]},
        actions={}, text=LANGUAGE, embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    batch = processor.collator([processed])
    model_inputs = batch.get("inputs", {})
    require(isinstance(model_inputs, Mapping), "canonical TRAIN model input is not a mapping")
    require("action" not in model_inputs and "action_mask" not in model_inputs, "canonical TRAIN action label leaked")
    for forbidden in ("target_position_world_m", "target_xyz", "target_world_m"):
        require(forbidden not in model_inputs, f"canonical TRAIN target leaked into model input: {forbidden}")
    return model_inputs


def _make_observation_snapshot(condition: Mapping[str, Any]) -> dict[str, Any]:
    q0 = condition["q0"]
    return {
        "window_id": f"{condition['condition_id']}:q0",
        "state_v1": list(q0["state_v1"]),
        "source_timestamps": [0.05 * index for index in range(HORIZON)],
        "observation_timestamp": float(q0["observation_timestamp"]),
        "anchor_world_m": list(condition["anchor_world_m"]),
        "target_world_m": list(TARGET_WORLD_M),
        "target_id": "red_target_stage43_shared",
        "target_xyz_in_model_input": False,
    }


def _write_npy(path: Path, array: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _attempt_lock(control_root: Path, *, repository_head: str, output_root: Path, gpu: Mapping[str, Any], noise_sha256: str, attempt_name: str = DEFAULT_ATTEMPT_NAME) -> tuple[Path, dict[str, Any]]:
    """Atomically consume one explicitly named PI0152 recovery attempt.

    A STARTED lock is intentionally retained after crashes.  This prevents an
    exception/OOM from being silently converted into a retry or card switch;
    a second lock can only be selected explicitly within PI0152's two-attempt
    ceiling.
    """
    require(attempt_name in ("recovery-01", "recovery-02"), f"invalid PI0152 attempt name: {attempt_name}")
    control_root = Path(control_root).resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    lock_path = control_root / f"stage45f_pi0152_{attempt_name}_attempt.lock.json"
    attempt = {
        "schema": "CORELAB_STAGE45F_PI0152_ATTEMPT_LOCK_V1",
        "attempt_name": attempt_name,
        "attempt_id": f"pi0152-{attempt_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        "status": "STARTED",
        "repository_head": repository_head,
        "output_root": str(output_root),
        "selected_gpu_uuid": gpu.get("uuid"),
        "selected_gpu_index": gpu.get("index"),
        "noise_sha256": noise_sha256,
        "created_at_utc": now_utc(),
        "retry": False,
        "card_switch": False,
    }
    payload = canonical_json(attempt) + b"\n"
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PilotError(f"ATTEMPT_ALREADY_CONSUMED: {lock_path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return lock_path, attempt


def _finish_attempt(lock_path: Path, attempt: Mapping[str, Any], *, status: str, counters: Mapping[str, Any], failure: Mapping[str, Any] | None = None) -> None:
    value = dict(attempt)
    value.update({"status": status, "counters": dict(counters), "finished_at_utc": now_utc()})
    if failure is not None:
        value["failure"] = dict(failure)
    write_json(lock_path, value, overwrite=True)


def _noise_artifact(torch: Any, output: Path, source: Path) -> tuple[Any, dict[str, Any]]:
    """Copy and validate the accepted PI0151 noise byte-for-byte."""
    import numpy as np

    source = Path(source).resolve()
    _safe_file(source, "saved common noise source")
    source_file_sha = sha256_file(source)
    require(source_file_sha == EXPECTED_NOISE_FILE_SHA, f"saved common noise file SHA drift: {source_file_sha}")
    loaded = np.load(source, allow_pickle=False)
    require(not isinstance(loaded, np.lib.npyio.NpzFile), "saved common noise must be a single .npy array")
    array = np.asarray(loaded)
    require(array.shape == RAW_SHAPE and array.dtype == np.float32 and np.isfinite(array).all(), "saved common noise ABI/finite drift")
    array_sha = sha256_bytes(array.tobytes())
    require(array_sha == EXPECTED_NOISE_ARRAY_SHA, f"saved common noise array SHA drift: {array_sha}")
    destination = output / "noise" / "common_x0.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"refusing to overwrite noise evidence: {destination}")
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    file_sha = sha256_file(destination)
    require(file_sha == source_file_sha == EXPECTED_NOISE_FILE_SHA, "copied noise file SHA drift")
    noise = torch.from_numpy(array.copy()).contiguous()
    require(tuple(int(v) for v in noise.shape) == RAW_SHAPE and bool(torch.isfinite(noise).all().item()), "common noise invalid")
    metadata = {
        "schema": "CORELAB_STAGE45F_PI0152_FIXED_NOISE_REUSE_V1",
        "seed": DEVELOPMENT_SEED,
        "shape": list(RAW_SHAPE),
        "dtype": "float32",
        "array_sha256": array_sha,
        "file_sha256": file_sha,
        "source_path": str(source),
        "source_file_sha256": source_file_sha,
        "source_array_sha256": array_sha,
        "source_generation": "PI0151 sealed common_x0.npy; copied byte-for-byte",
        "same_noise_for_all_observations": True,
    }
    write_json(output / "noise" / "common_x0.json", metadata)
    return noise, metadata


def _write_trajectory_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    import numpy as np

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    series = []
    for idx, row in enumerate(rows):
        decoded_path = Path(row["decoded_path"])
        if not decoded_path.is_absolute():
            decoded_path = Path(path).parent / decoded_path
        decoded = np.load(decoded_path, allow_pickle=False)
        anchor = np.asarray(row["anchor_world_m"], dtype=float)
        world = decoded[:, :3] + anchor[None, :]
        series.append((str(row["label"]), world, colors[idx % len(colors)]))
    width, height = 1000, 700
    pad = 70
    all_points = np.concatenate([item[1] for item in series], axis=0)
    xmin, xmax = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    ymin, ymax = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    xr = max(xmax - xmin, 1.0e-6)
    yr = max(ymax - ymin, 1.0e-6)
    def xy(point: Any) -> tuple[float, float]:
        x = pad + (float(point[0]) - xmin) / xr * (width - 2 * pad)
        y = height - pad - (float(point[1]) - ymin) / yr * (height - 2 * pad)
        return x, y
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="70" y="28" font-family="sans-serif" font-size="18">PI0151 decoded EEF XY trajectories (world frame)</text>']
    target = np.asarray(TARGET_WORLD_M, dtype=float)
    tx, ty = xy(target)
    out.append(f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="7" fill="#e41a1c"/><text x="{tx+10:.2f}" y="{ty-8:.2f}" font-family="sans-serif" font-size="13">red target</text>')
    for label, points, color in series:
        coords = " ".join(f"{xy(point)[0]:.2f},{xy(point)[1]:.2f}" for point in points)
        sx, sy = xy(points[0])
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        out.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="3" fill="{color}"/>')
    out.extend([f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>', f'<text x="{width/2:.0f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">world x (m)</text>', f'<text x="18" y="{height/2:.0f}" transform="rotate(-90 18 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="13">world y (m)</text>'])
    for index, (label, _, color) in enumerate(series):
        y = 55 + index * 20
        out.append(f'<line x1="760" y1="{y}" x2="785" y2="{y}" stroke="{color}" stroke-width="3"/><text x="792" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    out.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return sha256_file(path)


def _analyze_raw_records(raw_records: Sequence[Mapping[str, Any]], output: Path, n2_doc: Mapping[str, Any], ev: Any, np: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """CPU-only raw->decoded->metrics phase; never calls the model."""
    from go2arx5_vla.evaluation.trajectory_metrics import comparison_metrics, trajectory_metrics

    rows: list[dict[str, Any]] = []
    for item in raw_records:
        raw_path = Path(item["raw_path"])
        raw_np = np.load(raw_path, allow_pickle=False)
        require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), f"raw output invalid during CPU analysis: {item['label']}")
        snapshot = item["snapshot"]
        physical, decode_details = ev._decode_with_stats(
            raw_np, snapshot["state_v1"], n2_doc,
            stats_sha256=N2_STATS_SHA, payload_sha256=N2_STATS_PAYLOAD_SHA,
            chunk_id=str(snapshot["window_id"]), timestamps=snapshot["source_timestamps"],
            observation_timestamp=float(snapshot["observation_timestamp"]),
        )
        physical = np.asarray(physical, dtype=np.float32)
        require(physical.shape == (HORIZON, 10) and np.isfinite(physical).all(), f"decoded output invalid: {item['label']}")
        decoded_path = output / "decoded" / f"{item['index']:02d}_{item['label']}.npy"
        decoded_file_sha = _write_npy(decoded_path, physical)
        anchor = np.asarray(snapshot["anchor_world_m"], dtype=np.float64)
        target = np.asarray(snapshot["target_world_m"], dtype=np.float64)
        world = np.asarray(physical, dtype=np.float64)[:, :3] + anchor[None, :]
        target_metrics = trajectory_metrics(world, target, prefix=f"{item['label']}.target")
        measured_d0 = float(np.linalg.norm(anchor - target))
        measured_final = float(np.linalg.norm(world[-1] - target))
        measured = {
            "measured_initial_eef_world_m": anchor.tolist(),
            "measured_initial_target_distance_m": measured_d0,
            "measured_final_target_distance_m": measured_final,
            "progress_H40_m": measured_d0 - measured_final,
            "start_jump_m": float(np.linalg.norm(world[0] - anchor)),
            "predicted_first_point_world_m": world[0].tolist(),
            "predicted_final_point_world_m": world[-1].tolist(),
        }
        row: dict[str, Any] = {
            "index": int(item["index"]), "label": item["label"], "kind": item["kind"], "window_id": snapshot["window_id"],
            "raw_path": str(raw_path.relative_to(output)) if raw_path.is_relative_to(output) else str(raw_path), "raw_sha256": item["raw_sha256"],
            "decoded_path": str(decoded_path.relative_to(output)), "decoded_file_sha256": decoded_file_sha,
            "decoded_array_sha256": sha256_bytes(np.asarray(physical, dtype=np.float64).tobytes()),
            "noise_sha256": item["noise_sha256"], "decode": decode_details, "target_metrics": target_metrics,
            "measured_start": measured, "anchor_world_m": anchor.tolist(), "target_world_m": target.tolist(),
            "target_xyz_in_model_input": False, "finite": True,
        }
        if item["kind"] == "TRAIN24_REFERENCE":
            row["train24_comparison_metrics"] = comparison_metrics(
                np.asarray(physical, dtype=np.float64), np.asarray(snapshot["expert_physical_action"], dtype=np.float64),
                anchor, target, prefix=f"{item['label']}.comparison",
            )
        rows.append(row)

    table_path = output / "metrics_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["index", "label", "kind", "window_id", "raw_sha256", "decoded_file_sha256", "canonical_initial_target_distance_m", "canonical_minimum_target_distance_m", "canonical_final_target_distance_m", "canonical_absolute_progress_m", "canonical_best_progress_m", "canonical_progress_sign", "canonical_path_length_m", "canonical_endpoint_cosine", "measured_initial_target_distance_m", "measured_final_target_distance_m", "progress_H40_m", "start_jump_m"]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            metrics = row["target_metrics"]
            measured = row["measured_start"]
            writer.writerow({
                "index": row["index"], "label": row["label"], "kind": row["kind"], "window_id": row["window_id"],
                "raw_sha256": row["raw_sha256"], "decoded_file_sha256": row["decoded_file_sha256"],
                "canonical_initial_target_distance_m": metrics["initial_target_distance_m"], "canonical_minimum_target_distance_m": metrics["minimum_target_distance_m"],
                "canonical_final_target_distance_m": metrics["final_target_distance_m"], "canonical_absolute_progress_m": metrics["absolute_progress_m"],
                "canonical_best_progress_m": metrics["best_progress_m"], "canonical_progress_sign": metrics["progress_sign"], "canonical_path_length_m": metrics["path_length_m"],
                "canonical_endpoint_cosine": metrics["endpoint_target_vector_cosine"], "measured_initial_target_distance_m": measured["measured_initial_target_distance_m"],
                "measured_final_target_distance_m": measured["measured_final_target_distance_m"], "progress_H40_m": measured["progress_H40_m"], "start_jump_m": measured["start_jump_m"],
            })
    plot_sha = _write_trajectory_svg(output / "trajectory_plot.svg", rows)
    stage_rows = [row for row in rows if row["kind"] == "STAGE43_Q0"]
    negative = sum(row["measured_start"]["progress_H40_m"] < -1e-9 for row in stage_rows)
    positive = sum(row["measured_start"]["progress_H40_m"] > 1e-9 for row in stage_rows)
    if positive == len(stage_rows):
        recommendation = "申请该 checkpoint 的有界短闭环资格测试；本轮仍未执行 WBC。"
    else:
        recommendation = "优先申请同预算 independent-expert 监督/条件动作生成对照，区分首条计划错误与状态访问监督缺失；不自动启动。"
    conclusion = output / "one_page_conclusion.md"
    lines = [
        "# PI0152 local-policy initial-state model diagnostic",
        "",
        "本报告仅覆盖固定 N2 seed2223@2000、已复用的 seed-4242 noise 和六个已封存 observation 的离线 GR00T decoded EEF 输出；没有 WBC/simulator/hardware。",
        "",
        f"- Stage43 measured-start progress: positive={positive}, negative={negative}, total={len(stage_rows)}.",
        f"- TRAIN reference: `{rows[0]['label']}`；其 expert comparison 仅作训练样本参考，不等同于泛化验证。",
        "- canonical metrics 保持 Stage44R ABI；`progress_H40_m` 从实际 measured initial EEF 到预测 H40 末点计算，`start_jump_m` 为实际初态到预测首点的距离。",
        "",
        "## Interpretation",
        "",
        "本轮只回答模型侧首条计划是否随相容初态变化，以及物理目标距离是否改善；不把 action difference 解释为有效适应，不宣称泛化或闭环成功。",
        "",
        f"## Recommended next experiment\n\n{recommendation}",
    ]
    conclusion.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, {"metrics_table_sha256": sha256_file(table_path), "trajectory_plot_sha256": plot_sha, "conclusion_sha256": sha256_file(conclusion)}


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    require(getattr(args, "ffmpeg_runtime", None) is not None, "--ffmpeg-runtime is required for PI0152 execute")
    require(getattr(args, "noise_source", None) is not None, "--noise-source is required for PI0152 execute")
    paths = _paths(args)
    train_input_npz, train_input_manifest = _train_input_paths(args)
    output = Path(args.output_root).resolve()
    attempt_name = str(getattr(args, "attempt_name", DEFAULT_ATTEMPT_NAME))
    require(attempt_name in ("recovery-01", "recovery-02"), f"invalid PI0152 attempt name: {attempt_name}")
    control_root = Path(args.control_root).resolve() if args.control_root else output.parent / "stage45f_pi0152_control"
    lock_candidate = control_root / f"stage45f_pi0152_{attempt_name}_attempt.lock.json"
    require(not lock_candidate.exists(), f"ATTEMPT_ALREADY_CONSUMED: {lock_candidate}")
    admission = preflight(args)
    require(not output.exists(), f"pilot output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "admission.json", admission)
    env_info = _prepare_env(paths, output, str(admission["selected_gpu_uuid"]), Path(args.ffmpeg_runtime))
    write_json(output / "isolated_runtime.json", env_info)
    counters: dict[str, Any] = {
        "model_attempt_started": False, "model_construction": 0,
        "conditioning_passes_attempted": 0, "conditioning_passes_completed": 0, "conditioning_passes": 0,
        "action_head_calls_entered": 0, "action_head_calls_completed": 0, "action_head_calls": 0,
        "action_head_calls_unknown": 0, "model_forward_windows": 0, "gr00t_forwards": 0,
        "exceptions": 0, "training": False, "backward": False, "optimizer_steps": 0,
        "wbc": False, "simulator": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False,
    }
    _write_boundary = lambda: write_json(output / "execution_boundary.json", counters, overwrite=True)
    _write_boundary()
    lock_path: Path | None = None
    attempt_record: dict[str, Any] | None = None
    raw_manifest: Path | None = None
    try:
        # torch is imported only after device admission and output isolation;
        # noise is sealed before any GR00T/model module or weight is imported.
        import torch

        common_noise, noise_meta = _noise_artifact(torch, output, Path(args.noise_source))
        train_pixel_qualification, train_arrays = qualify_train_pixels(
            paths, output, Path(args.ffmpeg_runtime), train_input_npz, train_input_manifest,
        )
        lock_path, attempt_record = _attempt_lock(
            control_root, repository_head=git("rev-parse", "HEAD"), output_root=output,
            gpu=admission["gpu"], noise_sha256=noise_meta["array_sha256"], attempt_name=attempt_name,
        )
        counters["model_attempt_started"] = True
        write_json(output / "attempt_start.json", {**attempt_record, "noise": noise_meta})
        _write_boundary()

        import stage45e_n2_train24_eval as ev
        from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor
        from stage38_structured_offline_evaluation import _load_structured_model
        from stage45c_h200_eval_runtime_adapter import _sample_one
        from stage45b_full24_fresh_base_train import _bind_cosmos_processor

        _bind_cosmos_processor(paths["cosmos"] / "snapshots" / COSMOS_REVISION)
        processor, embodiment, modality_configs = _build_runtime_processor(
            groot_source=paths["groot"], base_checkpoint=paths["base_model"], dataset_dir=paths["dataset"],
            modality_path=paths["modality"], cosmos_snapshot=paths["cosmos"] / "snapshots" / COSMOS_REVISION,
        )
        stage43_audit = admission["audit"]["stage43"]
        observations: list[dict[str, Any]] = []
        preflight_doc = read_json(paths["snapshot_preflight"])
        snapshots = ev._snapshot_lookup(preflight_doc["snapshots"])
        train_snapshot = next(item for item in snapshots if item.get("window_id") == TRAIN_WINDOW_ID)
        train_state = np.asarray(train_snapshot["state_v1"], dtype=np.float32)
        require(train_state.shape == (STATE_DIM,) and np.array_equal(train_state, train_arrays["state"]), "canonical TRAIN StateV1 disagrees with sealed snapshot")
        # The frozen execution order is TRAIN reference, then C0..C4.
        observations.append({"label": "TRAIN_a_expert-0000", "model_inputs": _train_model_inputs(processor, embodiment, train_arrays), "snapshot": train_snapshot, "kind": "TRAIN24_REFERENCE"})
        for condition in CONDITIONS:
            record = next(item for item in stage43_audit["conditions"] if item["condition_id"] == condition)
            model_inputs = _stage43_model_inputs(processor, embodiment, record, paths["stage43_root"] / "conditions" / condition)
            observations.append({"label": condition, "model_inputs": model_inputs, "snapshot": _make_observation_snapshot(record), "kind": "STAGE43_Q0"})
        require(len(observations) == 6, "pilot observation count drift")

        # Dynamic model import/weight loading begins the single authorized attempt.
        counters["model_construction"] = 1
        counters["gr00t_forwards"] = 0
        _write_boundary()
        model = _load_structured_model(paths["n2_root"] / "checkpoint-2000", "cuda:0")
        model.eval()
        hook_counter = {"count": 0}
        handle = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_counter.__setitem__("count", hook_counter["count"] + 1))
        n2_doc = read_json(paths["n2_stats"])
        raw_manifest = output / "raw_predictions.jsonl"
        raw_manifest.write_text("", encoding="utf-8")
        raw_records: list[dict[str, Any]] = []
        try:
            for index, item in enumerate(observations):
                counters["conditioning_passes_attempted"] += 1
                counters["model_forward_windows"] += 1
                _write_boundary()
                entered_before = hook_counter["count"]
                try:
                    with torch.inference_mode():
                        raw, calls = _sample_one(model, item["model_inputs"], common_noise.clone(), torch)
                except Exception:
                    counters["exceptions"] += 1
                    entered = hook_counter["count"] - entered_before
                    counters["action_head_calls_entered"] += max(0, entered)
                    counters["action_head_calls_unknown"] += max(0, 4 - entered)
                    _write_boundary()
                    raise
                entered = hook_counter["count"] - entered_before
                counters["action_head_calls_entered"] += int(entered)
                counters["action_head_calls_completed"] += int(calls)
                counters["action_head_calls"] += int(calls)
                counters["conditioning_passes_completed"] += 1
                counters["conditioning_passes"] = counters["conditioning_passes_completed"]
                counters["gr00t_forwards"] += int(calls)
                require(entered == int(calls), f"action-head hook/call count mismatch: entered={entered}, returned={calls}")
                raw_np = raw.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), f"raw output invalid: {item['label']}")
                raw_path = output / "raw" / f"{index:02d}_{item['label']}.npy"
                raw_sha = _write_npy(raw_path, raw_np)
                with raw_manifest.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"index": index, "label": item["label"], "kind": item["kind"], "raw_path": str(raw_path.relative_to(output)), "raw_sha256": raw_sha, "array_sha256": sha256_bytes(raw_np.tobytes()), "noise_sha256": noise_meta["array_sha256"], "sampler_timestep_ids": list(TIMESTEP_IDS), "action_head_calls": int(calls)}, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                raw_records.append({"index": index, "label": item["label"], "kind": item["kind"], "snapshot": item["snapshot"], "raw_path": raw_path, "raw_sha256": raw_sha, "noise_sha256": noise_meta["array_sha256"]})
                _write_boundary()
        finally:
            handle.remove()
        require(counters["conditioning_passes"] == 6, "conditioning budget drift")
        require(counters["action_head_calls"] == 24 and hook_counter["count"] == 24 and counters["action_head_calls_entered"] == 24, "action-head budget drift")
        require(counters["gr00t_forwards"] == 24, "GR00T forward counter drift")
        _write_boundary()
        # CPU-only analysis starts only after all six raw arrays have been
        # fsynced.  A decoder/report error therefore cannot consume another
        # model call.
        rows, artifacts = _analyze_raw_records(raw_records, output, n2_doc, ev, np)
        report = {"schema": "CORELAB_STAGE45F_LOCAL_INITIAL_STATE_PILOT_RESULT_V3", "status": "COMPLETED", "token": RETURN_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_pointer_comment": PR_POINTER_COMMENT}, "repository": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "base_head": BASE_HEAD}, "local_reference": {"seed": LOCAL_SEED, "checkpoint_tree_sha256": CHECKPOINT_TREE, "training_result_sha256": TRAINING_RESULT_SHA, "n2_stats_sha256": N2_STATS_SHA, "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA}, "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_start": admission["gpu"], "gpu_uuid": admission["selected_gpu_uuid"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "isolated_ffmpeg": env_info}, "inputs": {"stage43_bundle_sha256": admission["audit"]["stage43"]["bundle_sha256"], "stage43_condition_count": 5, "train_window_id": TRAIN_WINDOW_ID, "train_input_npz_sha256": train_pixel_qualification["input_npz_sha256"], "train_input_manifest_sha256": train_pixel_qualification["accepted_manifest_sha256"], "train_pixel_qualification_sha256": sha256_file(output / "train_pixel_qualification.json"), "common_noise_seed": DEVELOPMENT_SEED, "common_noise_array_sha256": noise_meta["array_sha256"], "common_noise_file_sha256": noise_meta["file_sha256"], "target_xyz_in_model_input": False}, "counters": counters, "rows": rows, "artifacts": {"raw_predictions_manifest_sha256": sha256_file(raw_manifest), **artifacts}, "execution_boundary": {"training": False, "backward": False, "optimizer_steps": 0, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "wbc": False, "simulator": False, "hardware": False, "icl": False, "merge": False}, "claim_boundary": "decoded model-side initial-state diagnostic only; no closed-loop success or route selection claim", "timestamp_utc": now_utc(), "attempt_name": attempt_name}
        report_sha = write_json(output / "pilot_result.json", report)
        write_json(output / "pilot_result.json.sha256", {"sha256": report_sha, "file": "pilot_result.json"})
        report["report_sha256"] = report_sha
        if lock_path is not None and attempt_record is not None:
            _finish_attempt(lock_path, attempt_record, status="COMPLETED", counters=counters)
        return report
    except Exception as exc:
        counters["exceptions"] += 1
        failure = {"schema": "CORELAB_STAGE45F_LOCAL_INITIAL_STATE_PILOT_FAILURE_SEAL_V2", "status": "PARTIAL" if counters.get("model_attempt_started") else "BLOCKED", "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT}, "repository_head": git("rev-parse", "HEAD"), "counters": counters, "error_type": type(exc).__name__, "error": str(exc)[-2000:], "traceback_tail": traceback.format_exc()[-4000:], "retry": False, "card_switch": False, "training": False, "wbc": False, "simulator": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "timestamp_utc": now_utc()}
        write_json(output / "failure_seal.json", failure, overwrite=True)
        if lock_path is not None and attempt_record is not None:
            _finish_attempt(lock_path, attempt_record, status="FAILED_OR_PARTIAL", counters=counters, failure=failure)
        raise


def _analyze_existing_raw(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild C1 derived evidence from an already sealed raw attempt.

    This mode is deliberately CPU-only and never imports Torch/GR00T.  It is
    used after a report/serialization failure so the six model outputs remain
    immutable and no model call is repeated.
    """
    import numpy as np

    paths = _paths(args)
    raw_root = Path(args.raw_root).resolve()
    _safe_dir(raw_root, "sealed raw attempt root")
    raw_manifest = raw_root / "raw_predictions.jsonl"
    _safe_file(raw_manifest, "sealed raw prediction manifest")
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"CPU analysis output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)

    stage43 = _stage43_audit(paths["stage43_root"])
    preflight_doc = read_json(paths["snapshot_preflight"])
    import stage45e_n2_train24_eval as ev
    snapshots = ev._snapshot_lookup(preflight_doc["snapshots"])
    snapshot_by_id = {item["window_id"]: item for item in snapshots}
    snapshot_by_label = {"TRAIN_a_expert-0000": snapshot_by_id[TRAIN_WINDOW_ID]}
    for condition in stage43["conditions"]:
        snapshot_by_label[condition["condition_id"]] = _make_observation_snapshot(condition)

    raw_records: list[dict[str, Any]] = []
    lines = [line for line in raw_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == 6, f"sealed raw manifest row count drift: {len(lines)}")
    seen_indices: set[int] = set()
    for raw_line in lines:
        item = json.loads(raw_line)
        index = int(item["index"])
        label = str(item["label"])
        require(index not in seen_indices and 0 <= index < 6, f"sealed raw manifest index drift: {index}")
        seen_indices.add(index)
        raw_path = Path(str(item["raw_path"]))
        if not raw_path.is_absolute():
            raw_path = raw_root / raw_path
        raw_path = raw_path.resolve()
        _safe_file(raw_path, f"sealed raw array {label}")
        require(sha256_file(raw_path) == str(item["raw_sha256"]), f"sealed raw SHA drift: {label}")
        snapshot = snapshot_by_label.get(label)
        require(snapshot is not None, f"sealed raw label is not recognized: {label}")
        raw_records.append({"index": index, "label": label, "kind": item["kind"], "snapshot": snapshot, "raw_path": raw_path, "raw_sha256": item["raw_sha256"], "noise_sha256": item["noise_sha256"]})
    raw_records.sort(key=lambda item: item["index"])
    require([item["index"] for item in raw_records] == list(range(6)), "sealed raw manifest order drift")
    n2_doc = read_json(paths["n2_stats"])
    rows, artifacts = _analyze_raw_records(raw_records, output, n2_doc, ev, np)
    result = {
        "schema": "CORELAB_STAGE45F_PI0152_CPU_RAW_REANALYSIS_V1",
        "status": "COMPLETED",
        "token": RETURN_TOKEN,
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_pointer_comment": PR_POINTER_COMMENT},
        "repository": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD")},
        "source_attempt_root": str(raw_root),
        "source_raw_manifest_sha256": sha256_file(raw_manifest),
        "source_raw_rows": 6,
        "rows": rows,
        "artifacts": artifacts,
        "counters": {"model_forward": 0, "conditioning_passes": 0, "action_head_calls": 0, "training": False, "gpu": False, "wbc": False, "simulator": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False},
        "claim_boundary": "CPU-only reconstruction of sealed PI0152 raw model outputs; no new inference and no closed-loop claim",
        "timestamp_utc": now_utc(),
    }
    result_sha = write_json(output / "cpu_reanalysis.json", result)
    write_json(output / "cpu_reanalysis.json.sha256", {"sha256": result_sha, "file": "cpu_reanalysis.json"})
    result["result_sha256"] = result_sha
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "preflight", "execute", "analyze-raw"), required=True)
    parser.add_argument("--stage43-root", type=Path, required=True)
    parser.add_argument("--n2-root", type=Path, required=True)
    parser.add_argument("--n2-stats", type=Path, required=True)
    parser.add_argument("--runtime-stats", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--groot", type=Path, required=True)
    parser.add_argument("--cosmos", type=Path, required=True)
    parser.add_argument("--modality", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-preflight", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg-runtime", type=Path)
    parser.add_argument("--noise-source", type=Path)
    parser.add_argument("--train-input-npz", type=Path)
    parser.add_argument("--train-input-manifest", type=Path)
    parser.add_argument("--attempt-name", choices=("recovery-01", "recovery-02"), default=DEFAULT_ATTEMPT_NAME)
    parser.add_argument("--raw-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "audit":
        value = collect_audit(args)
        if args.output_root:
            output = Path(args.output_root).resolve()
            require(not output.exists(), f"audit output root must be fresh: {output}")
            output.mkdir(parents=True, exist_ok=False)
            digest = write_json(output / "audit.json", value)
            value["audit_sha256"] = digest
            write_json(output / "audit.json", value, overwrite=True)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.mode == "preflight":
        output = Path(args.output_root).resolve() if args.output_root else None
        if output is not None:
            require(not output.exists(), f"preflight output root must be fresh: {output}")
            output.mkdir(parents=True, exist_ok=False)
            preflight_path = output / "preflight.json"
        else:
            preflight_path = None
        value = preflight(args, output=preflight_path)
        print(json.dumps(value, sort_keys=True))
        return 0
    require(args.output_root is not None, "--output-root is required for execute")
    if args.mode == "analyze-raw":
        require(args.raw_root is not None, "--raw-root is required for analyze-raw")
        value = _analyze_existing_raw(args)
        print(json.dumps({"status": value.get("status"), "token": value.get("token"), "rows": len(value.get("rows", [])), "output_root": str(Path(args.output_root).resolve())}, sort_keys=True))
        return 0
    value = _execute(args)
    print(json.dumps({"status": value.get("status"), "token": value.get("token"), "counters": value.get("counters"), "output_root": str(Path(args.output_root).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
