#!/usr/bin/env python3
"""Execute the PI0154 matched Validation-E transfer test.

The entrypoint has a standard-library-only pre-model section.  It verifies the
committed recipe freeze, the late-bound admission, the observed H200 and all
artifact identities before importing the model runtime.  Only then does it
load one checkpoint at a time and run the frozen direct action-head E4 graph
on the six original ``red_target_holdout_e`` snapshots.

There is no training child, public sampler, adaptive retry, simulator, WBC,
or Holdout-F path in this program.  Raw arrays are durably written before any
decode or metric computation.  A failure seal records the actual phase and
call counters rather than claiming that model execution did not start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
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

import stage45f_vanda_validation_e_admission as admission  # noqa: E402


SCHEMA = "STAGE45F_PI0154_VALIDATION_E_GEN1_EXECUTION_V1"
SEAL_SCHEMA = "STAGE45F_PI0154_VALIDATION_E_GEN1_SEAL_V1"
FAILURE_SCHEMA = "STAGE45F_PI0154_VALIDATION_E_FAILURE_SEAL_V1"
EXPECTED_BRANCH = admission.EXPECTED_BRANCH
EXPECTED_GATE = admission.EXPECTED_GATE
EXPECTED_HEAD_BASE = admission.EXPECTED_BASE_HEAD
TARGET_ID = admission.TARGET_ID
WINDOW_IDS = admission.EXPECTED_SNAPSHOT_IDS
T_VALUES = tuple(admission.EXPECTED_T)
TIMESTEP_IDS = tuple(admission.EXPECTED_BUCKETS)
HORIZON = 40
UNIVERSAL_DIM = 132
PHYSICAL_DIM = 10
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
PHYSICAL_SHAPE = (HORIZON, PHYSICAL_DIM)
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
TIMING_MODE = "T16"
MODEL_DTYPE = "bfloat16"
MIN_VRAM_BYTES = 80 * 1024**3


class ValidationExecutionError(RuntimeError):
    """Raised when PI0154 cannot continue within its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationExecutionError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return jsonable(value.item())
    return value


def write_json_durable(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if not overwrite:
        require(not path.exists() and not path.is_symlink(), f"record must be fresh: {path}")
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


def append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_npy_durable(path: Path, value: Any) -> None:
    import numpy as np

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


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-800:]}")
    return completed.stdout.strip()


def static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top_imports.intersection({"torch", "transformers", "gr00t", "cuda", "numpy", "cv2"}))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train"}:
                forbidden_calls.append(name)
    markers = {
        "recipe_freeze": "STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1" in source,
        "six_windows": "EXPECTED_SNAPSHOT_IDS" in source and "len(windows) == 6" in source,
        "e4_schedule": "TIMESTEP_IDS" in source and "T_VALUES" in source,
        "raw_before_decode": False,
        "durable_raw": "write_npy_durable" in source,
        "failure_phase_and_counters": '"phase": phase' in source and '"actual_counters": counters' in source,
        "no_training": '"training": False' in source,
        "holdout_f_locked": '"holdout_f_read": False' in source,
    }
    # Keep this structural check independent of the model runtime.  The raw
    # manifest append must occur before the call to the decoder in source.
    markers["raw_before_decode"] = '"decoded": False' in source and "physical, details = runtime._decode_v2" in source and source.index('"decoded": False') < source.index("physical, details = runtime._decode_v2")
    return {
        "schema": "STAGE45F_PI0154_EXECUTION_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "model_imported": False,
        "model_forward": 0,
        "training": False,
        "pbs": 0,
        "h200": 0,
    }


def observe_checkout(expected_head: str) -> dict[str, Any]:
    root = run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    require(root == str(REPO), f"repository root drift: {root}")
    require(branch == EXPECTED_BRANCH, f"branch drift: {branch}")
    require(head == expected_head, f"execution HEAD drift: {head} != {expected_head}")
    require(not status, "execution checkout is dirty, including untracked files")
    require(head != EXPECTED_HEAD_BASE, "execution HEAD is still the pre-validation support HEAD")
    return {"observed_repo_root": root, "observed_branch": branch, "observed_head": head, "observed_clean_worktree": True}


def observe_gpu() -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "PBS_JOBID is required")
    raw = run_text(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_vram_mib": int(fields[3]), "used_vram_mib": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == 0, f"expected exactly one visible GPU at index 0: {rows}")
    require("H200" in rows[0]["name"].upper(), f"admitted GPU is not H200: {rows[0]['name']}")
    require(rows[0]["total_vram_mib"] * 1024**2 >= MIN_VRAM_BYTES, "H200 VRAM is below 80 GiB")
    return {
        "schema": "STAGE45F_PI0154_OBSERVED_H200_V1",
        "host": socket.gethostname(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "selected_device": "cuda:0",
        "gpu_name": rows[0]["name"],
        "gpu_uuid": rows[0]["uuid"],
        "total_vram_bytes": rows[0]["total_vram_mib"] * 1024**2,
        "memory_used_mib_at_admission": rows[0]["used_vram_mib"],
        "visible_gpu_count": 1,
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_stdout_sha256": sha256_bytes((raw + "\n").encode("utf-8")),
    }


def observed_runtime(gpu: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    require(all(os.environ.get(key) == value for key, value in expected.items()), f"offline environment drift: {expected}")
    return {
        "schema": "STAGE45F_PI0154_OBSERVED_H200_RUNTIME_V1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "host": gpu["host"],
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "selected_device": "cuda:0",
        "offline_environment": expected,
        "precision": MODEL_DTYPE,
        "numpy_imported_before_model": False,
    }


def _regular(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def _regular_dir(path: Path, label: str) -> Path:
    require(path.is_dir() and not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = _regular_dir(root, "checkpoint")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            require(not child.is_symlink() and stat.S_ISDIR(child.lstat().st_mode), f"checkpoint contains unsafe directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            require(not child.is_symlink() and stat.S_ISREG(child.lstat().st_mode), f"checkpoint contains unsafe file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: item["path"])


def tree_sha256(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    entries = tree_entries(root, prefix)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest(), entries


def _frame_ppm(path: Path) -> Any:
    """Read the native P6 frame without a model/video runtime dependency."""

    import numpy as np

    path = _regular(path, "RGB frame")
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        try:
            import cv2
        except ImportError as exc:
            raise ValidationExecutionError(f"non-PPM E frame needs cv2: {path}") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(image is not None, f"could not decode E frame: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index:index + 1] == b"#":
            newline = data.find(b"\n", index)
            require(newline >= 0, f"malformed PPM comment: {path}")
            index = newline + 1
            continue
        end = index
        while end < len(data) and data[end] not in b" \t\r\n":
            end += 1
        require(end > index, f"malformed PPM header: {path}")
        tokens.append(data[index:end])
        index = end
    require(tokens[0] == b"P6", f"unsupported PPM encoding: {path}")
    width, height, maxval = (int(tokens[1]), int(tokens[2]), int(tokens[3]))
    require((width, height, maxval) == (96, 96, 255), f"E frame geometry/bit depth drift: {path}: {(width, height, maxval)}")
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    payload = data[index:]
    require(len(payload) == width * height * 3, f"PPM payload length drift: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()


def _anchor_from_snapshot(payload: Mapping[str, Any]) -> list[float]:
    for container_name in ("raw_telemetry", "provenance"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            for key in ("anchor_world_m", "native_anchor_world_m"):
                value = container.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    return [float(item) for item in value]
    raise ValidationExecutionError(f"snapshot lacks explicit anchor_world_m: {payload.get('example_id')}")


def validate_snapshot_records(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    input_binding = binding.get("validation_input")
    require(isinstance(input_binding, Mapping), "validation input binding missing")
    records = input_binding.get("snapshot_records")
    require(isinstance(records, list) and len(records) == 6, "admitted snapshot record count drift")
    windows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        require(isinstance(record, Mapping), f"snapshot record {index} is not an object")
        expected_id = WINDOW_IDS[index]
        require(record.get("snapshot_id") == expected_id, f"snapshot order drift at {index}")
        descriptor_path = _regular(Path(str(record["descriptor_path"])), "snapshot descriptor")
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        require(payload.get("example_id") == expected_id and payload.get("target_id") == TARGET_ID, f"snapshot identity drift: {expected_id}")
        require(payload.get("target_excluded_from_model_input") is True, f"target exclusion drift: {expected_id}")
        require(payload.get("source_frame") == SOURCE_FRAME and payload.get("tcp_semantic_id") == TCP_SEMANTIC_ID, f"frame/TCP drift: {expected_id}")
        timestamps = np.asarray(payload.get("future_timestamps"), dtype=np.float64)
        desired = np.asarray(payload.get("future_desired_eef_9d"), dtype=np.float64)
        gripper = np.asarray(payload.get("future_gripper"), dtype=np.float64)
        state_payload = payload.get("state_v1")
        state = np.asarray(state_payload.get("values") if isinstance(state_payload, Mapping) else None, dtype=np.float64)
        require(timestamps.shape == (HORIZON,) and desired.shape == (HORIZON, 9) and gripper.shape == (HORIZON, 1) and state.shape == (10,), f"snapshot array shape drift: {expected_id}")
        require(np.isfinite(timestamps).all() and np.isfinite(desired).all() and np.isfinite(gripper).all() and np.isfinite(state).all(), f"snapshot contains non-finite values: {expected_id}")
        require(np.all(np.diff(timestamps) > 0.0), f"snapshot timestamps are not increasing: {expected_id}")
        body = _regular(Path(str(payload.get("body_frame_path"))), "E body frame")
        wrist = _regular(Path(str(payload.get("wrist_frame_path"))), "E wrist frame")
        require(sha256_file(descriptor_path) == record.get("descriptor_sha256"), f"descriptor SHA drift: {expected_id}")
        require(sha256_file(body) == record.get("body_frame_sha256"), f"body frame SHA drift: {expected_id}")
        require(sha256_file(wrist) == record.get("wrist_frame_sha256"), f"wrist frame SHA drift: {expected_id}")
        anchor = _anchor_from_snapshot(payload)
        target = payload.get("target_position_world_m")
        require(isinstance(target, list) and len(target) == 3 and np.isfinite(np.asarray(target, dtype=np.float64)).all(), f"target metadata drift: {expected_id}")
        physical = np.concatenate([desired, gripper], axis=1)
        windows.append({
            "window_id": expected_id,
            "descriptor_path": str(descriptor_path),
            "descriptor_sha256": record["descriptor_sha256"],
            "body_frame_path": str(body),
            "body_frame_sha256": record["body_frame_sha256"],
            "wrist_frame_path": str(wrist),
            "wrist_frame_sha256": record["wrist_frame_sha256"],
            "state_v1": state.astype(np.float32),
            "source_timestamps": timestamps,
            "observation_timestamp": float(payload["observation_timestamp"]),
            "expert_physical_action": physical,
            "anchor_world_m": np.asarray(anchor, dtype=np.float64),
            "target_world_m": np.asarray(target, dtype=np.float64),
            "source_frame": SOURCE_FRAME,
            "tcp_semantic_id": TCP_SEMANTIC_ID,
            "target_excluded_from_model_input": True,
            "payload": payload,
        })
    return windows


def validate_artifacts(config: Mapping[str, Any], binding: Mapping[str, Any], expected_head: str) -> dict[str, Any]:
    """Verify paths and hashes without importing Torch/GR00T."""

    import ast

    require(not admission.model_imports_present(), "model-side module already imported before execution admission")
    freeze = admission.load_json(admission.FREEZE_PATH)
    admission._validate_freeze(freeze, expected_code_commit=expected_head)
    require(binding.get("execution_head") == expected_head, "admission HEAD differs from execution HEAD")
    require(binding.get("execution_branch") == EXPECTED_BRANCH, "admission branch drift")
    require(binding.get("pi_gate") == EXPECTED_GATE and binding.get("pr_pointer") == admission.EXPECTED_PR_POINTER, "admission authority drift")
    require(binding.get("status") == "ADMITTED_BEFORE_MODEL_IMPORT", "admission status is not pre-model")
    require(binding.get("model_imported") is False and binding.get("model_construction") is False and binding.get("model_forward") is False and binding.get("training") is False, "admission already crossed model boundary")
    stats = _regular(Path(str(config["stats"])), "runtime stats")
    require(sha256_file(stats) == admission.EXPECTED_STATS_SHA, "runtime stats SHA drift")
    base_model = _regular_dir(Path(str(config["base_model"])), "BASE model")
    dataset = _regular_dir(Path(str(config["dataset"])), "FULL24 dataset")
    groot = _regular_dir(Path(str(config["groot_source"])), "GR00T source")
    cosmos = _regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache")
    modality = _regular(Path(str(config["modality"])), "modality config")
    require(sha256_file(modality) == "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031", "modality SHA drift")
    base_config = _regular(base_model / "processor_config.json", "BASE processor config")
    base_stats = _regular(base_model / "statistics.json", "BASE processor statistics")
    require(sha256_file(base_config) == "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be", "BASE processor config SHA drift")
    require(sha256_file(base_stats) == "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9", "BASE processor statistics SHA drift")
    groot_head = run_text(["git", "-C", str(groot), "rev-parse", "HEAD"])
    require(groot_head == "51d4c89f72fda44cbf77285c6a8114b52676b8a1", "GR00T source HEAD drift")
    source_manifest = _regular(Path(str(config["groot_manifest"])), "GR00T source manifest")
    require(sha256_file(source_manifest) == "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11", "GR00T source manifest SHA drift")
    checkpoint_config = config.get("checkpoints")
    require(isinstance(checkpoint_config, Mapping), "checkpoint path map missing")
    arms = binding.get("checkpoint_arms")
    require(isinstance(arms, list) and len(arms) == 6, "admission checkpoint arm count drift")
    checkpoint_audit: list[dict[str, Any]] = []
    for arm in arms:
        require(isinstance(arm, Mapping), "checkpoint arm is not an object")
        label = str(arm["label"])
        path_value = checkpoint_config.get(label)
        require(isinstance(path_value, str), f"checkpoint path missing: {label}")
        path = _regular_dir(Path(path_value), f"checkpoint {label}")
        observed, entries = tree_sha256(path, f"{label}/")
        require(observed == arm["tree_sha256"], f"checkpoint tree SHA drift: {label}: {observed}")
        state_path = _regular(path / "trainer_state.json", f"trainer state {label}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        require(int(state.get("global_step")) == 2000, f"checkpoint tier drift: {label}")
        checkpoint_audit.append({"label": label, "route": arm["route"], "seed": arm["seed"], "tier": arm["tier"], "path": str(path), "tree_sha256": observed, "file_count": len(entries), "total_bytes": sum(int(item["bytes"]) for item in entries), "trainer_state_sha256": sha256_file(state_path)})
    snapshot_lock = _regular(REPO / "manifests/STAGE22_VALIDATION_SNAPSHOT_LOCK_V1.json", "snapshot lock")
    split_lock = _regular(REPO / "manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json", "split lock")
    require(sha256_file(snapshot_lock) == admission.EXPECTED_SNAPSHOT_LOCK_SHA, "snapshot lock SHA drift")
    require(sha256_file(split_lock) == admission.EXPECTED_SPLIT_LOCK_SHA, "split lock SHA drift")
    windows = validate_snapshot_records(binding)
    require(len(windows) == 6 and [item["window_id"] for item in windows] == list(WINDOW_IDS), "six E windows are not complete/in order")
    return {
        "schema": "STAGE45F_PI0154_PRE_MODEL_ARTIFACT_AUDIT_V1",
        "status": "PASS",
        "head": expected_head,
        "paths": {"base_model": str(base_model), "dataset": str(dataset), "groot_source": str(groot), "cosmos_root": str(cosmos), "modality": str(modality), "stats": str(stats), "groot_manifest": str(source_manifest)},
        "checkpoint_audit": checkpoint_audit,
        "checkpoint_count": len(checkpoint_audit),
        "snapshot_count": len(windows),
        "snapshot_ids": list(WINDOW_IDS),
        "target_xyz_in_model_input": False,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "holdout_f_read": False,
        "validation_e_read": True,
        "model_imported": False,
        "model_forward": 0,
        "training": False,
    }


def _prepare_windows(processor: Any, embodiment: Any, windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    from go2arx5_vla.data.processor import build_official_vla_step_data
    from go2arx5_vla.data.stage22_expert import load_snapshot, snapshot_to_training_example
    from gr00t.data.types import MessageType

    prepared: list[dict[str, Any]] = []
    for window in windows:
        snapshot = load_snapshot(Path(str(window["descriptor_path"])))
        body = _frame_ppm(Path(str(window["body_frame_path"])))
        wrist = _frame_ppm(Path(str(window["wrist_frame_path"])))
        require(tuple(body.shape) == (96, 96, 3) and tuple(wrist.shape) == (96, 96, 3), f"E frame shape drift: {window['window_id']}")
        example = snapshot_to_training_example(snapshot)
        content = build_official_vla_step_data(example, body_rgb=body, wrist_rgb=wrist, embodiment=embodiment)
        processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
        batch = processor.collator([processed])
        model_inputs = batch.get("inputs", {})
        require(isinstance(model_inputs, Mapping), f"model inputs are not a mapping: {window['window_id']}")
        require("action" not in model_inputs and "action_mask" not in model_inputs, f"expert action leaked into E model input: {window['window_id']}")
        for forbidden in ("target_position_world_m", "target_xyz", "target_world_m"):
            require(forbidden not in model_inputs, f"target numeric input leaked into E model input: {window['window_id']}")
        state = model_inputs.get("state")
        if state is not None:
            require(bool(__import__("torch").isfinite(state).all().item()), f"processed state is nonfinite: {window['window_id']}")
        prepared.append({**dict(window), "snapshot": snapshot, "model_inputs": model_inputs, "logical_model_input_keys": ["body_rgb", "wrist_rgb", "state"], "target_xyz_in_model_input": False})
    return prepared


def _make_noise_bank(torch: Any) -> tuple[list[Any], list[str]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(admission.EXPECTED_NOISE_SEED)
    bank: list[Any] = []
    hashes: list[str] = []
    for _ in WINDOW_IDS:
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
        require(tuple(int(value) for value in noise.shape) == RAW_SHAPE and bool(torch.isfinite(noise).all().item()), "noise bank shape/finite drift")
        bank.append(noise)
        hashes.append(sha256_bytes(noise.numpy().tobytes()))
    return bank, hashes


def _inventory(root: Path, excluded: set[str]) -> tuple[Path, str, int]:
    entries: list[dict[str, Any]] = []
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if relative in excluded or path.is_symlink():
                continue
            require(path.is_file(), f"evidence inventory found non-file: {path}")
            entries.append({"path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    inventory = root / "evidence_inventory.json"
    write_json_durable(inventory, {"schema": "STAGE45F_PI0154_EVIDENCE_INVENTORY_V1", "entries": entries, "entry_count": len(entries), "excluded": sorted(excluded)})
    return inventory, sha256_file(inventory), len(entries)


def _metric_row(runtime: Any, raw_np: Any, physical: Any, details: Mapping[str, Any], window: Mapping[str, Any], checkpoint: Mapping[str, Any], runtime_sha: str, raw_path: str, raw_sha: str) -> dict[str, Any]:
    row = runtime._evaluation_row(raw_np, physical, details, window, checkpoint, runtime_sha, raw_path, raw_sha)
    row["window_id"] = window["window_id"]
    row["checkpoint"] = dict(checkpoint)
    row["target_xyz_in_model_input"] = False
    row["source_frame"] = SOURCE_FRAME
    row["tcp_semantic_id"] = TCP_SEMANTIC_ID
    return row


def paired_summary(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = {str(arm["label"]): arm for arm in arms}
    output: list[dict[str, Any]] = []
    by_seed: dict[str, list[dict[str, Any]]] = {str(seed): [] for seed in (2222, 2223, 2224)}
    for seed in (2222, 2223, 2224):
        legacy_label = by_label["Legacy(seed%d)@2000" % seed]["label"]
        l1_label = by_label["L1(seed%d)@2000" % seed]["label"]
        legacy_rows = list(rows_by_arm[legacy_label])
        l1_rows = list(rows_by_arm[l1_label])
        require([row["window_id"] for row in legacy_rows] == [row["window_id"] for row in l1_rows], f"paired window order drift for seed {seed}")
        for legacy, l1 in zip(legacy_rows, l1_rows):
            legacy_target = legacy["metrics"]["target"]
            l1_target = l1["metrics"]["target"]
            record = {
                "seed": seed,
                "window_id": legacy["window_id"],
                "comparison": "L1_minus_Legacy_same_tier_same_window_noise",
                "final_target_distance_delta_m": float(l1_target["final_target_distance_m"] - legacy_target["final_target_distance_m"]),
                "progress_delta_m": float(l1_target["absolute_progress_m"] - legacy_target["absolute_progress_m"]),
                "minimum_target_distance_delta_m": float(l1_target["minimum_target_distance_m"] - legacy_target["minimum_target_distance_m"]),
                "path_length_delta_m": float(l1_target["path_length_m"] - legacy_target["path_length_m"]),
                "absolute_roughness_delta_m": float(l1["metrics"]["prediction_roughness_position_second_difference_m"]["mean"] - legacy["metrics"]["prediction_roughness_position_second_difference_m"]["mean"]),
                "roughness_ratio_l1": l1["metrics"].get("roughness_ratio"),
                "roughness_ratio_legacy": legacy["metrics"].get("roughness_ratio"),
                "endpoint_target_vector_cosine_l1": l1_target.get("endpoint_target_vector_cosine"),
                "endpoint_target_vector_cosine_legacy": legacy_target.get("endpoint_target_vector_cosine"),
                "progress_sign_l1": l1_target["progress_sign"],
                "progress_sign_legacy": legacy_target["progress_sign"],
            }
            output.append(record)
            by_seed[str(seed)].append(record)
    return {
        "schema": "STAGE45F_PI0154_L1_MINUS_LEGACY_PAIRED_SUMMARY_V1",
        "pair_count": len(output),
        "paired_by_seed": by_seed,
        "paired_by_window": output,
        "descriptive_only": True,
        "iid_or_p_value_claim": False,
        "linked_endpoint_boundary": "final_distance_and_progress_are linked through common initial distance",
    }


def execute(config_path: Path) -> int:
    phase = "START"
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_construction_attempts": 0,
        "model_construction_completed": 0,
        "conditioning_attempts": 0,
        "conditioning_completed": 0,
        "action_head_attempts": 0,
        "action_head_completed": 0,
        "model_forward_completed": 0,
        "raw_prediction_count": 0,
        "derived_row_count": 0,
        "raw_manifest_records": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "failed_model_calls": 0,
    }
    current: dict[str, Any] = {"arm": None, "window_id": None, "step": None, "checkpoint": None}
    output_root: Path | None = None
    try:
        config = admission.load_json(config_path)
        require(isinstance(config, Mapping), "execution config must be an object")
        expected_head = str(config["expected_head"])
        output_root = Path(str(config["output_root"])).resolve()
        admission_path = _regular(Path(str(config["admission_path"])), "admission")
        binding = admission.load_json(admission_path)
        require(not output_root.is_symlink() and output_root.is_dir(), "admission output root is unavailable")
        require(Path(str(binding.get("output_root"))).resolve() == output_root, "output root does not match admission")
        for forbidden in ("raw_predictions.jsonl", "derived_rows.jsonl", "stage45f_validation_e_seal.json", "stage45f_validation_e_result.json"):
            require(not (output_root / forbidden).exists(), f"execution output is not fresh for {forbidden}")
        guard = static_guard()
        require(guard["status"] == "PASS", "execution static guard failed")
        checkout = observe_checkout(expected_head)
        gpu = observe_gpu()
        runtime_identity = observed_runtime(gpu)
        phase = "PRE_MODEL_ARTIFACT_AUDIT"
        artifact_audit = validate_artifacts(config, binding, expected_head)
        windows_meta = validate_snapshot_records(binding)
        write_json_durable(output_root / "stage45f_execution_static_guard.json", guard)
        write_json_durable(output_root / "stage45f_pre_model_artifact_audit.json", artifact_audit)
        write_json_durable(output_root / "stage45f_observed_runtime.json", {"checkout": checkout, "gpu": gpu, "runtime": runtime_identity})
        phase = "MODEL_IMPORT_AND_PROCESSOR_CONSTRUCTION"
        counters["model_import_started"] = True
        # Model/runtime imports begin only after the complete CPU admission.
        import numpy as np
        import torch
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

        runtime_processor, embodiment, _modality_configs = stage26._build_runtime_processor(
            groot_source=Path(str(config["groot_source"])).resolve(),
            base_checkpoint=Path(str(config["base_model"])).resolve(),
            dataset_dir=Path(str(config["dataset"])).resolve(),
            modality_path=Path(str(config["modality"])).resolve(),
            cosmos_snapshot=Path(str(config["cosmos_root"])).resolve() / "snapshots" / "9ce19a195e423419c349abfc86fd07178b230561",
        )
        windows = _prepare_windows(runtime_processor, embodiment, windows_meta)
        require(len(windows) == 6, "prepared E window count drift")
        noise_bank, noise_hashes = _make_noise_bank(torch)
        write_json_durable(output_root / "sampler_noise_manifest.json", {
            "schema": "STAGE45F_PI0154_CPU_FLOAT32_NOISE_BANK_V1",
            "seed": admission.EXPECTED_NOISE_SEED,
            "device": "cpu",
            "dtype": "float32",
            "shape": list(RAW_SHAPE),
            "window_ids": list(WINDOW_IDS),
            "noise_sha256_by_window": noise_hashes,
            "same_noise_across_all_six_checkpoint_arms": True,
        })
        raw_manifest = output_root / "raw_predictions.jsonl"
        derived_manifest = output_root / "derived_rows.jsonl"
        arms = binding["checkpoint_arms"]
        checkpoint_paths = config["checkpoints"]
        rows_by_arm: dict[str, list[dict[str, Any]]] = {str(arm["label"]): [] for arm in arms}
        runtime_sha = canonical_sha({"binding": binding, "artifact_audit": artifact_audit, "runtime": runtime_identity})
        for arm in arms:
            label = str(arm["label"])
            current.update({"arm": label, "checkpoint": label, "window_id": None, "step": None})
            phase = "MODEL_LOAD"
            counters["model_loads_attempted"] += 1
            model = runtime._load_torch_runtime()[1](Path(str(checkpoint_paths[label])).resolve(), "cuda:0")
            counters["model_loads_completed"] += 1
            counters["model_construction_attempts"] += 1
            counters["model_construction_completed"] += 1
            model.eval()
            action_rows: list[dict[str, Any]] = []
            try:
                for index, window in enumerate(windows):
                    current["window_id"] = window["window_id"]
                    current["step"] = None
                    phase = "CONDITIONING"
                    counters["conditioning_attempts"] += 1
                    _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                    counters["conditioning_completed"] += 1
                    current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                    with torch.inference_mode():
                        for step, timestep in enumerate(TIMESTEP_IDS):
                            current["step"] = int(step)
                            phase = "ACTION_HEAD"
                            counters["action_head_attempts"] += 1
                            try:
                                velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                                counters["action_head_completed"] += 1
                                counters["model_forward_completed"] += 1
                            except Exception:
                                counters["failed_model_calls"] += 1
                                raise
                            current_action = current_action + 0.25 * velocity
                    phase = "RAW_SEAL"
                    raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                    require(raw_np.shape == RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output shape/finite drift: {label}:{window['window_id']}")
                    raw_path = output_root / "raw" / label.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.npy"
                    write_npy_durable(raw_path, raw_np)
                    raw_file_sha = sha256_file(raw_path)
                    raw_array_sha = sha256_bytes(np.ascontiguousarray(raw_np).tobytes())
                    append_jsonl_durable(raw_manifest, {
                        "schema": "STAGE45F_PI0154_RAW_PREDICTION_V1",
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "path": str(raw_path.relative_to(output_root)),
                        "file_sha256": raw_file_sha,
                        "array_sha256": raw_array_sha,
                        "shape": list(raw_np.shape),
                        "dtype": str(raw_np.dtype),
                        "noise_sha256": noise_hashes[index],
                        "timestep_ids": list(TIMESTEP_IDS),
                        "action_head_calls_observed": len(TIMESTEP_IDS),
                        "decoded": False,
                    })
                    counters["raw_prediction_count"] += 1
                    counters["raw_manifest_records"] += 1
                    phase = "DECODE_AND_DERIVED_METRICS"
                    physical, details = runtime._decode_v2(raw_np, window["state_v1"], Path(str(config["stats"])).resolve(), window["source_timestamps"], float(window["observation_timestamp"]), window["window_id"] + ":" + label)
                    row = _metric_row(runtime, raw_np, physical, details, window, {"label": label, "tree_sha256": arm["tree_sha256"]}, runtime_sha, str(raw_path.relative_to(output_root)), raw_file_sha)
                    row["route"] = arm["route"]
                    row["seed"] = arm["seed"]
                    row["tier"] = arm["tier"]
                    row["noise_sha256"] = noise_hashes[index]
                    row["sampler"] = {
                        "name": "direct-source-mapped-action-head-E4",
                        "t_values": list(T_VALUES),
                        "timestep_ids": list(TIMESTEP_IDS),
                        "dt": 0.25,
                        "carried_state": True,
                        "reset_between_stages": False,
                        "official_public_sampler_called": False,
                        "get_action_calls": 0,
                    }
                    row["stats"] = {"sha256": admission.EXPECTED_STATS_SHA, "payload_sha256": admission.EXPECTED_STATS_PAYLOAD_SHA, "route": "FULL24_RUNTIME_V2"}
                    write_json_durable(output_root / "derived" / label.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.json", row)
                    append_jsonl_durable(derived_manifest, {
                        "schema": "STAGE45F_PI0154_DERIVED_EVIDENCE_REFERENCE_V1",
                        "checkpoint": dict(arm),
                        "window_id": window["window_id"],
                        "raw_path": str(raw_path.relative_to(output_root)),
                        "raw_file_sha256": raw_file_sha,
                        "derived_path": str((output_root / "derived" / label.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.json").relative_to(output_root)),
                        "decoded": True,
                        "decoded_array_sha256": row["decoded_action"]["array_sha256"],
                    })
                    action_rows.append(row)
                    counters["derived_row_count"] += 1
                require(len(action_rows) == 6, f"per-arm row count drift: {label}")
            finally:
                del model
                gc = __import__("gc")
                gc.collect()
                torch.cuda.empty_cache()
            rows_by_arm[label] = action_rows
            aggregate = aggregate_evaluation_rows(action_rows)
            aggregate["route"] = arm["route"]
            aggregate["seed"] = arm["seed"]
            aggregate["tier"] = arm["tier"]
            aggregate["checkpoint_tree_sha256"] = arm["tree_sha256"]
            write_json_durable(output_root / f"aggregate_{label.replace('(', '_').replace(')', '').replace('@', '_at_')}.json", aggregate)
        require(counters["model_loads_completed"] == 6, "model load count drift")
        require(counters["conditioning_completed"] == 36, "conditioning count drift")
        require(counters["action_head_completed"] == 144, "action-head count drift")
        require(counters["model_forward_completed"] == 144, "model-forward count drift")
        require(counters["raw_prediction_count"] == 36 and counters["derived_row_count"] == 36, "raw/derived row count drift")
        phase = "CPU_PAIRED_SUMMARY"
        paired = paired_summary(rows_by_arm, arms)
        write_json_durable(output_root / "paired_summary.json", paired)
        cpu_summary = {
            "schema": "STAGE45F_PI0154_VALIDATION_E_CPU_SUMMARY_V1",
            "status": "PASS_READONLY_DERIVED_FROM_SEALED_RAW",
            "checkpoint_order": [str(arm["label"]) for arm in arms],
            "window_ids": list(WINDOW_IDS),
            "window_count_per_arm": 6,
            "arms": 6,
            "paired_summary": paired,
            "aggregate_files": [f"aggregate_{str(arm['label']).replace('(', '_').replace(')', '').replace('@', '_at_')}.json" for arm in arms],
            "raw_predictions": str(raw_manifest),
            "derived_rows": str(derived_manifest),
            "descriptive_only": True,
            "no_iid_claim": True,
        }
        write_json_durable(output_root / "cpu_summary.json", cpu_summary)
        phase = "SEAL"
        telemetry = {
            "schema": "STAGE45F_PI0154_H200_TELEMETRY_V1",
            "gpu": gpu,
            "runtime": runtime_identity,
            "counters": counters,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device("cuda:0"))),
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
        }
        write_json_durable(output_root / "telemetry.json", telemetry)
        excluded = {"evidence_inventory.json", "stage45f_validation_e_seal.json", "stage45f_validation_e_result.json"}
        inventory_path, inventory_sha, inventory_count = _inventory(output_root, excluded)
        seal = {
            "schema": SEAL_SCHEMA,
            "status": "PASS",
            "token": "VANDA_PI0154_VALIDATION_E_GEN1_FOR_PI_REVIEW",
            "pi_gate": EXPECTED_GATE,
            "execution_head": expected_head,
            "branch": EXPECTED_BRANCH,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "gpu": gpu,
            "runtime": runtime_identity,
            "admission_path": str(admission_path),
            "admission_sha256": sha256_file(admission_path),
            "checkpoint_arms": [dict(arm) for arm in arms],
            "input_access_ledger": binding["validation_input"]["access_ledger"],
            "input_access_ledger_sha256": binding["validation_input"]["access_ledger_sha256"],
            "snapshot_ids": list(WINDOW_IDS),
            "snapshot_count": 6,
            "checkpoint_count": 6,
            "noise_manifest": str(output_root / "sampler_noise_manifest.json"),
            "noise_manifest_sha256": sha256_file(output_root / "sampler_noise_manifest.json"),
            "call_budget": {"nominal": admission.EXPECTED_CALLS, "max_recovery": 210},
            "actual_counters": counters,
            "raw_manifest": str(raw_manifest),
            "raw_manifest_sha256": sha256_file(raw_manifest),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "paired_summary": str(output_root / "paired_summary.json"),
            "paired_summary_sha256": sha256_file(output_root / "paired_summary.json"),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
            "automatic_retry": False,
            "scientific_contract_changed": False,
        }
        write_json_durable(output_root / "stage45f_validation_e_seal.json", seal)
        result = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "token": "VANDA_PI0154_VALIDATION_E_GEN1_FOR_PI_REVIEW",
            "execution_head": expected_head,
            "branch": EXPECTED_BRANCH,
            "seal_path": str(output_root / "stage45f_validation_e_seal.json"),
            "seal_sha256": sha256_file(output_root / "stage45f_validation_e_seal.json"),
            "cpu_summary": str(output_root / "cpu_summary.json"),
            "cpu_summary_sha256": sha256_file(output_root / "cpu_summary.json"),
            "actual_counters": counters,
            "checkpoint_count": 6,
            "window_count": 6,
            "raw_output_count": 36,
            "input_access_ledger_sha256": binding["validation_input"]["access_ledger_sha256"],
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "training": False,
            "holdout_f_read": False,
            "automatic_retry": False,
        }
        write_json_durable(output_root / "stage45f_validation_e_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if output_root is not None:
            try:
                failure = {
                    "schema": FAILURE_SCHEMA,
                    "status": "PARTIAL" if counters["model_import_started"] else "BLOCKED",
                    "pi_gate": EXPECTED_GATE,
                    "execution_head": current.get("checkpoint") or None,
                    "branch": EXPECTED_BRANCH,
                    "pbs_jobid": os.environ.get("PBS_JOBID"),
                    "phase": phase,
                    "current": current,
                    "actual_counters": counters,
                    "model_import_started": bool(counters["model_import_started"]),
                    "model_execution_started": bool(counters["model_import_started"]),
                    "training": False,
                    "backward": False,
                    "optimizer_steps": 0,
                    "checkpoint_writes": 0,
                    "validation_e_read": True,
                    "holdout_f_read": False,
                    "automatic_retry": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp_utc": now_utc(),
                }
                write_json_durable(output_root / "stage45f_validation_e_failure.json", failure)
            except Exception:
                pass
        print(f"VANDA_PI0154_VALIDATION_E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def plan() -> dict[str, Any]:
    guard = static_guard()
    freeze = admission.static_plan()
    return {
        "schema": SCHEMA,
        "status": "PASS_STATIC_NOT_EXECUTED" if guard["status"] == "PASS" else "FAIL",
        "static_guard": guard,
        "recipe_freeze": freeze,
        "matrix": {"arms": 6, "windows_per_arm": 6, "queries": 36},
        "call_budget": {"nominal": admission.EXPECTED_CALLS, "max_recovery": 210},
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "pbs": 0,
        "h200": 0,
        "return_token": "VANDA_PI0154_VALIDATION_E_GEN1_FOR_PI_REVIEW",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan(), indent=2, sort_keys=True))
            return 0
        require(args.config is not None, "--execute requires --config")
        return execute(args.config.resolve(strict=True))
    except Exception as exc:
        print(f"VANDA_PI0154_VALIDATION_E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
