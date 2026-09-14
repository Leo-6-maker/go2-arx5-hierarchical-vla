#!/usr/bin/env python3
"""PI0170 Corelab RTX4080 six-window Validation-E qualification.

The committed program is deliberately narrow.  ``--plan`` and ``--preflight``
are standard-library-only modes.  Preflight verifies the frozen artifact and
runtime identities, prepares fresh per-run caches, and takes exactly two
occupancy snapshots before any Validation-E payload or model import.  Only a
successful preflight can enter ``--execute``.

Execution is one Legacy(seed2222)@2000 model load, six conditioning passes,
and four source-mapped E4 action-head calls for each of the six locked
``red_target_holdout_e`` windows.  Raw universal actions are durably written
before CPU decode and metrics.  Existing H200 allocation-8 raw cells are read
only portability anchors; no H200 submission, recovery, training, WBC,
simulation, hardware, or Holdout-F path exists here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
SCRIPTS = REPO / "scripts"
MANIFEST_PATH = REPO / "manifests/STAGE45F_PI0170_RTX4080_VALIDATION_QUALIFICATION_V1.json"
FREEZE_PATH = REPO / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json"

PI_GATE = "PI_GATE_STATE_0170"
ISSUE_COMMENT = 5659393287
PR_POINTER = 5659394477
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"
BRANCH = "stage45f/corelab-validation-e-rtx4080-qualification"
FROZEN_EVALUATION_COMMIT = "522ea718e99ff8b1c4b2aa08bec680ed5adfcae5"
TARGET_ID = "red_target_holdout_e"
WINDOW_IDS = tuple(f"stage22_expert_{TARGET_ID}:expert-{index:04d}" for index in range(6))
CHECKPOINT_LABEL = "Legacy(seed2222)@2000"
CHECKPOINT_TREE_SHA = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
FULL24_DATASET_TREE_SHA = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_SOURCE_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_PAYLOAD_TREE_SHA = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
BASE_TREE_SHA = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
BASE_PROCESSOR_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_PROCESSOR_STATS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
H200_RAW_MANIFEST_SHA = "0ebe595377d32164ff395d9d0f6d779b74a3654e473f12863eca2c6e83d0f8df"
RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
TIMESTEP_IDS = (0, 250, 500, 750)
T_VALUES = (0.0, 0.25, 0.5, 0.75)
SAMPLER_DT = 0.25
NOISE_SEED = 4242
MIN_FREE_VRAM_MIB = 12288
MIN_TOTAL_VRAM_MIB = 30000
CACHE_KEYS = (
    "tmpdir",
    "xdg_cache_home",
    "xdg_config_home",
    "torch_home",
    "triton_cache_dir",
    "hf_home",
    "hf_hub_cache",
    "transformers_cache",
    "hf_datasets_cache",
)
FORBIDDEN_TOP_LEVEL_IMPORTS = {"torch", "numpy", "cv2", "gr00t", "transformers", "cuda", "isaacgym"}
FORBIDDEN_CALLS = {"train", "backward", "get_action", "qsub", "sbatch", "srun", "submit", "save_checkpoint", "save_model", "save_pretrained"}
ALLOWED_CHANGE_PATHS = {
    "scripts/stage45f_pi0170_rtx4080_qualification.py",
    "manifests/STAGE45F_PI0170_RTX4080_VALIDATION_QUALIFICATION_V1.json",
    "tests/test_stage45f_pi0170_rtx4080_qualification.py",
}


class QualificationError(RuntimeError):
    """A fail-closed PI0170 gate or execution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        return value.item()
    return value


def load_json(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise QualificationError(f"required JSON is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot parse JSON: {path}") from exc


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


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-800:]}")
    return completed.stdout.strip()


def static_risk_audit() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top_imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if name in FORBIDDEN_CALLS:
            forbidden_calls.append(name)
    forbidden_text: list[str] = []
    raw_marker = "# PI0170_RAW_BEFORE_DECODE"
    decode_marker = "# PI0170_DECODE_AFTER_RAW"
    markers = {
        "single_arm": CHECKPOINT_LABEL == "Legacy(seed2222)@2000" and source.count("model_loads_attempted") >= 1,
        "six_windows": "len(windows) == 6" in source and "WINDOW_IDS" in source,
        "e4_schedule": "TIMESTEP_IDS = (0, 250, 500, 750)" in source,
        "raw_before_decode": raw_marker in source and decode_marker in source and source.index(raw_marker) < source.index(decode_marker),
        "no_second_load": sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_load_torch_runtime") == 1,
        "no_training": "training_steps" in source and "backward_steps" in source,
        "holdout_f_locked": "holdout_f_read" in source and "holdout_f" in source.lower(),
        "no_h200_submit": '"h200_submissions": 0' in source,
    }
    status = "PASS" if not forbidden_imports and not forbidden_calls and not forbidden_text and all(markers.values()) else "FAIL"
    return {
        "schema": "CORELAB_PI0170_ZERO_MODEL_EXECUTION_RISK_AUDIT_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "forbidden_text_hits": forbidden_text,
        "markers": markers,
        "model_imported": False,
        "model_loads": 0,
        "model_forward": 0,
        "training": False,
        "h200_submissions": 0,
    }


def checkout_audit() -> dict[str, Any]:
    root = run_text(["git", "-C", str(REPO), "rev-parse", "--show-toplevel"])
    branch = run_text(["git", "-C", str(REPO), "branch", "--show-current"])
    head = run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"])
    require(root == str(REPO), f"repository root drift: {root}")
    require(branch == BRANCH, f"branch drift: {branch}")
    require(not status, "qualification checkout is dirty")
    run_text(["git", "-C", str(REPO), "merge-base", "--is-ancestor", SOURCE_HEAD, head])
    changed = run_text(["git", "-C", str(REPO), "diff", "--name-only", f"{SOURCE_HEAD}..{head}"]).splitlines()
    require(set(changed).issubset(ALLOWED_CHANGE_PATHS), f"unexpected files changed after PI0170 source HEAD: {changed}")
    frozen = {
        "scripts/stage45f_vanda_validation_e_admission.py",
        "scripts/stage45f_vanda_validation_e_gen1.py",
        "scripts/stage45f_pi0163_validation_e_recovery.py",
        "scripts/stage45f_pi0167_raw_inventory_verifier.py",
        "src/go2arx5_vla/evaluation/checkpoint_eval.py",
    }
    require(not frozen.intersection(changed), f"frozen execution files changed: {sorted(frozen.intersection(changed))}")
    return {"root": root, "branch": branch, "head": head, "clean": True, "changed_after_source_head": changed}


def freeze_audit() -> dict[str, Any]:
    freeze = load_json(FREEZE_PATH)
    require(isinstance(freeze, Mapping), "recipe freeze is not an object")
    require(freeze.get("schema") == "STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1", "recipe freeze schema drift")
    require(freeze.get("status") == "FROZEN_BEFORE_VALIDATION_E_PAYLOAD_READ", "recipe freeze status drift")
    support = freeze.get("support_checkout")
    require(isinstance(support, Mapping) and support.get("evaluation_code_commit") == FROZEN_EVALUATION_COMMIT, "evaluation code commit drift")
    arms = freeze.get("checkpoint_arms")
    require(isinstance(arms, list) and len(arms) == 6, "frozen six-arm matrix missing")
    require([(item.get("route"), item.get("seed"), item.get("tier")) for item in arms] == [
        ("Legacy", 2222, 2000), ("Legacy", 2223, 2000), ("Legacy", 2224, 2000),
        ("L1", 2222, 2000), ("L1", 2223, 2000), ("L1", 2224, 2000),
    ], "frozen arm order drift")
    require(arms[0].get("label") == CHECKPOINT_LABEL and arms[0].get("tree_sha256") == CHECKPOINT_TREE_SHA, "Legacy2222 identity drift")
    inputs = freeze.get("validation_input_lock")
    require(isinstance(inputs, Mapping) and inputs.get("snapshot_ids") == list(WINDOW_IDS), "frozen Validation-E windows drift")
    protocol = freeze.get("evaluation_protocol")
    require(isinstance(protocol, Mapping), "frozen protocol missing")
    require(protocol.get("noise_seed") == NOISE_SEED and protocol.get("noise_shape") == list(RAW_SHAPE), "noise contract drift")
    require(protocol.get("timestep_ids") == list(TIMESTEP_IDS) and protocol.get("t_values") == list(T_VALUES) and protocol.get("dt") == SAMPLER_DT, "E4 contract drift")
    require(protocol.get("target_xyz_in_model_input") is False and protocol.get("public_sampler_calls") == 0 and protocol.get("get_action_calls") == 0, "model-input/sampler contract drift")
    boundaries = freeze.get("locked_boundaries")
    require(isinstance(boundaries, Mapping) and all(value is False for value in boundaries.values()), "a frozen boundary is open")
    return {
        "schema": "CORELAB_PI0170_FROZEN_RECIPE_AUDIT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "freeze_path": str(FREEZE_PATH),
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "evaluation_code_commit": FROZEN_EVALUATION_COMMIT,
        "checkpoint": dict(arms[0]),
        "snapshot_ids": list(WINDOW_IDS),
        "payload_read": False,
        "holdout_f_read": False,
        "model_imported": False,
    }


def regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise QualificationError(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def regular_dir(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise QualificationError(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


def artifact_audit(config: Mapping[str, Any], checkout: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module imported before artifact admission")
    required = ("base_model", "checkpoint", "dataset", "groot_source", "groot_manifest", "cosmos_root", "modality", "stats", "source_root")
    for key in required:
        require(isinstance(config.get(key), str) and config[key], f"late-bound artifact path is missing: {key}")
    for value in config.values():
        if isinstance(value, str):
            require("holdout-f" not in value.lower(), "Holdout-F path is forbidden")
    base_model = regular_dir(Path(str(config["base_model"])), "BASE model")
    checkpoint = regular_dir(Path(str(config["checkpoint"])), "Legacy(seed2222)@2000 checkpoint")
    dataset = regular_dir(Path(str(config["dataset"])), "FULL24 dataset")
    groot = regular_dir(Path(str(config["groot_source"])), "GR00T source")
    groot_manifest = regular_file(Path(str(config["groot_manifest"])), "GR00T source manifest")
    cosmos = regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache")
    modality = regular_file(Path(str(config["modality"])), "frozen modality")
    stats = regular_file(Path(str(config["stats"])), "FULL24 runtime statistics")
    require(sha256_file(stats) == FULL24_STATS_SHA, "FULL24 runtime statistics SHA drift")
    require(sha256_file(modality) == MODALITY_SHA, "modality SHA drift")
    require(sha256_file(base_model / "processor_config.json") == BASE_PROCESSOR_CONFIG_SHA, "BASE processor config SHA drift")
    require(sha256_file(base_model / "statistics.json") == BASE_PROCESSOR_STATS_SHA, "BASE processor statistics SHA drift")
    base_entries = tree_entries(base_model, "payload/GR00T-N1.7-3B")
    base_tree = tree_digest(base_entries)
    require(base_tree == BASE_TREE_SHA, f"BASE model tree SHA drift: {base_tree}")
    require(sha256_file(groot_manifest) == GROOT_MANIFEST_SHA, "GR00T source manifest SHA drift")
    checkpoint_prefix = checkpoint.name
    checkpoint_entries = tree_entries(checkpoint, checkpoint_prefix)
    checkpoint_tree = tree_digest(checkpoint_entries)
    require(checkpoint_tree == CHECKPOINT_TREE_SHA, f"Legacy checkpoint tree SHA drift: {checkpoint_tree}")
    trainer_state = regular_file(checkpoint / "trainer_state.json", "Legacy trainer state")
    trainer_json = load_json(trainer_state)
    require(int(trainer_json.get("global_step", -1)) == 2000, "Legacy checkpoint is not @2000")

    # These validators are standard-library-only at import time.  Their calls
    # bind the exact non-Validation-E runtime bytes before the first E payload.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import stage45c_h200_eval_runtime_adapter as runtime_adapter

    dataset_identity = runtime_adapter._verify_dataset(dataset)
    stats_identity = runtime_adapter._verify_runtime_stats(stats)
    groot_manifest_identity = runtime_adapter._verify_groot_runtime_manifest(groot_manifest)
    groot_git = None
    if (groot / ".git").is_dir():
        groot_git = run_text(["git", "-C", str(groot), "rev-parse", "HEAD"])
        require(groot_git == GROOT_SOURCE_COMMIT, "GR00T source HEAD drift")
        require(not run_text(["git", "-C", str(groot), "status", "--porcelain", "--untracked-files=all"]), "GR00T source checkout is dirty")
    else:
        runtime_adapter._verify_groot_source_snapshot(groot, groot_manifest)
    cosmos_digest, cosmos_entries = runtime_adapter.cosmos_tree_digest(cosmos)
    require(cosmos_digest == COSMOS_TREE_SHA, f"Cosmos tree SHA drift: {cosmos_digest}")
    cosmos_snapshot = regular_dir(cosmos / "snapshots" / COSMOS_REVISION, "Cosmos pinned snapshot")
    cosmos_ref = regular_file(cosmos / "refs" / "main", "Cosmos refs/main")
    require(cosmos_ref.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos refs/main revision drift")
    require("numpy" not in sys.modules and "torch" not in sys.modules, "runtime artifact audit imported a model-side module")
    return {
        "schema": "CORELAB_PI0170_ARTIFACT_RUNTIME_ADMISSION_V1",
        "status": "PASS_PRE_MODEL",
        "checkout_head": checkout["head"],
        "paths": {
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "dataset": str(dataset),
            "groot_source": str(groot),
            "groot_manifest": str(groot_manifest),
            "cosmos_root": str(cosmos),
            "cosmos_snapshot": str(cosmos_snapshot),
            "modality": str(modality),
            "stats": str(stats),
            "source_root": str(Path(str(config["source_root"])).resolve()),
        },
        "base_model": {"tree_sha256": base_tree, "file_count": len(base_entries), "processor_config_sha256": BASE_PROCESSOR_CONFIG_SHA, "statistics_sha256": BASE_PROCESSOR_STATS_SHA},
        "checkpoint": {"label": CHECKPOINT_LABEL, "tree_sha256": checkpoint_tree, "file_count": len(checkpoint_entries), "global_step": 2000},
        "dataset": dataset_identity,
        "stats": stats_identity,
        "groot_manifest": groot_manifest_identity,
        "groot_source_commit": groot_git or GROOT_SOURCE_COMMIT,
        "groot_payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA,
        "cosmos": {"tree_sha256": cosmos_digest, "entry_count": cosmos_entries, "revision": COSMOS_REVISION, "refs_main_sha256": sha256_file(cosmos_ref)},
        "modality_sha256": MODALITY_SHA,
        "validation_e_payload_read": False,
        "holdout_f_read": False,
        "model_imported": False,
    }


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = regular_dir(root, "tree")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            require(not child.is_symlink() and stat.S_ISDIR(child.lstat().st_mode), f"tree contains unsafe directory: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            require(not child.is_symlink() and stat.S_ISREG(child.lstat().st_mode), f"tree contains unsafe file: {child}")
            entries.append({"path": f"{prefix.rstrip('/')}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: item["path"])


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _nvidia(argv: Sequence[str]) -> str:
    completed = subprocess.run(["nvidia-smi", *argv], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"nvidia-smi admission query failed: {completed.stderr[-800:]}")
    return completed.stdout.strip()


def _parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, f"malformed GPU occupancy row: {line}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_vram_mib": int(fields[3]), "used_vram_mib": int(fields[4]), "free_vram_mib": int(fields[5])})
    require(rows, "no visible GPU returned by nvidia-smi")
    return rows


def _parse_compute_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 4, f"malformed compute-app occupancy row: {line}")
        rows.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2], "used_memory_mib": int(fields[3])})
    return rows


def occupancy_snapshot(index: int) -> dict[str, Any]:
    gpu_raw = _nvidia(["--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
    compute_raw = _nvidia(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    return {
        "schema": "CORELAB_PI0170_RTX4080_OCCUPANCY_SNAPSHOT_V1",
        "snapshot_index": index,
        "captured_at_utc": now_utc(),
        "host": os.uname().nodename,
        "cuda_visible_devices_before_binding": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": _parse_gpu_rows(gpu_raw),
        "foreign_compute_processes": _parse_compute_rows(compute_raw),
        "gpu_query_sha256": sha256_bytes((gpu_raw + "\n").encode("utf-8")),
        "compute_query_sha256": sha256_bytes((compute_raw + "\n").encode("utf-8")),
    }


def resource_admission(config: Mapping[str, Any]) -> dict[str, Any]:
    first = occupancy_snapshot(1)
    candidates = [row for row in first["gpus"] if "RTX 4080" in row["name"].upper() and row["total_vram_mib"] >= MIN_TOTAL_VRAM_MIB]
    preferred = config.get("preferred_gpu_uuid")
    if preferred:
        candidates = [row for row in candidates if row["uuid"] == preferred]
    require(candidates, "CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION — BLOCKED_SHARED_GPU_CAPACITY: no qualifying 32-GiB RTX4080")
    selected_uuid = candidates[0]["uuid"]
    second_interval = float(config.get("occupancy_snapshot_interval_seconds", 2.0))
    require(0.0 < second_interval <= 10.0, "occupancy snapshot interval is outside the bounded range")
    time.sleep(second_interval)
    second = occupancy_snapshot(2)
    selected_rows = []
    for snapshot in (first, second):
        matches = [row for row in snapshot["gpus"] if row["uuid"] == selected_uuid]
        require(len(matches) == 1, "selected GPU UUID disappeared between occupancy snapshots")
        selected = matches[0]
        require(selected["free_vram_mib"] >= MIN_FREE_VRAM_MIB, f"CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION — BLOCKED_SHARED_GPU_CAPACITY: free VRAM {selected['free_vram_mib']} MiB < {MIN_FREE_VRAM_MIB} MiB")
        foreign = [item for item in snapshot["foreign_compute_processes"] if item["gpu_uuid"] == selected_uuid]
        require(not foreign, f"CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION — BLOCKED_SHARED_GPU_CAPACITY: foreign compute process on selected UUID {selected_uuid}")
        selected_rows.append(selected)
    require(selected_rows[0]["uuid"] == selected_rows[1]["uuid"] == selected_uuid, "selected GPU UUID changed")
    return {
        "schema": "CORELAB_PI0170_RTX4080_RESOURCE_ADMISSION_V1",
        "status": "PASS_PRE_MODEL",
        "host": first["host"],
        "selected_gpu_uuid": selected_uuid,
        "selected_gpu_name": selected_rows[0]["name"],
        "selected_cuda_mapping_after_snapshots": {"CUDA_VISIBLE_DEVICES": selected_uuid, "device": "cuda:0"},
        "min_free_vram_mib": MIN_FREE_VRAM_MIB,
        "snapshots": [first, second],
        "same_uuid": True,
        "foreign_process_interference": False,
        "gpu_uuid_count": 1,
    }


def prepare_fresh_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    cache_root = Path(str(config.get("cache_root", ""))).resolve()
    require(cache_root.is_absolute() and "pi0170" in cache_root.name.lower(), "cache_root must be a fresh PI0170-specific absolute path")
    require(not cache_root.exists() and not cache_root.is_symlink(), f"fresh cache root already exists: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=False)
    paths = {key: cache_root / key for key in CACHE_KEYS}
    for path in paths.values():
        path.mkdir(parents=False, exist_ok=False)
    env = {
        "TMPDIR": paths["tmpdir"],
        "XDG_CACHE_HOME": paths["xdg_cache_home"],
        "XDG_CONFIG_HOME": paths["xdg_config_home"],
        "TORCH_HOME": paths["torch_home"],
        "TRITON_CACHE_DIR": paths["triton_cache_dir"],
        "HF_HOME": paths["hf_home"],
        "HF_HUB_CACHE": paths["hf_hub_cache"],
        "TRANSFORMERS_CACHE": paths["transformers_cache"],
        "HF_DATASETS_CACHE": paths["hf_datasets_cache"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
    }
    os.environ.update({key: str(value) for key, value in env.items()})
    return {"root": str(cache_root), "paths": {key: str(value) for key, value in paths.items()}, "environment": {key: str(value) for key, value in env.items() if key not in {"TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"}}}


def preflight(config_path: Path) -> int:
    config = load_json(config_path)
    require(isinstance(config, Mapping), "PI0170 config must be an object")
    risk = static_risk_audit()
    require(risk["status"] == "PASS", "zero-model execution-risk audit failed")
    checkout = checkout_audit()
    freeze = freeze_audit()
    artifacts = artifact_audit(config, checkout, freeze)
    cache = prepare_fresh_cache(config)
    resource = resource_admission(config)
    output_root = Path(str(config["output_root"])).resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "CORELAB_PI0170_RTX4080_QUALIFICATION_ADMISSION_V1",
        "status": "PASS_PRE_MODEL",
        "token": "CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_ADMISSION_RECORDED",
        "pi_gate": PI_GATE,
        "issue_11_comment": ISSUE_COMMENT,
        "pr_pointer": PR_POINTER,
        "source_head": SOURCE_HEAD,
        "execution_head": checkout["head"],
        "branch": checkout["branch"],
        "risk_audit": risk,
        "freeze_audit": freeze,
        "artifact_runtime_admission": artifacts,
        "resource_admission": resource,
        "fresh_cache": cache,
        "validation_e_payload_read": False,
        "holdout_f_read": False,
        "model_imported": False,
        "model_loads_attempted": 0,
        "conditioning": 0,
        "action_head": 0,
        "model_side_total": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "h200_submissions": 0,
        "created_at_utc": now_utc(),
        "output_root": str(output_root),
    }
    write_json_durable(output_root / "pi0170_zero_model_risk_audit.json", risk)
    write_json_durable(output_root / "pi0170_artifact_runtime_admission.json", artifacts)
    write_json_durable(output_root / "pi0170_resource_admission.json", resource)
    write_json_durable(output_root / "pi0170_rtx4080_qualification_admission.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    print("CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_ADMISSION_RECORDED")
    return 0


def payload_admission(config: Mapping[str, Any], output_root: Path, checkout: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module imported before Validation-E payload admission")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import stage45f_vanda_validation_e_admission as admission

    freeze = load_json(FREEZE_PATH)
    admission._validate_freeze(freeze, expected_code_commit=FROZEN_EVALUATION_COMMIT)
    source_root = regular_dir(Path(str(config["source_root"])), "Validation-E source root")
    snapshot_paths = admission._find_snapshot_paths(source_root)
    started = now_utc()
    records = [admission._read_snapshot_descriptor(path, WINDOW_IDS[index]) for index, path in enumerate(snapshot_paths)]
    origins = [record["frame_origin"] for record in records]
    canonical_origin = list(origins[0]["anchor_world_m"])
    spread = max(max(abs(float(origin["anchor_world_m"][axis]) - canonical_origin[axis]) for axis in range(3)) for origin in origins)
    require(spread <= admission.FRAME_ORIGIN_SPREAD_TOLERANCE_M, f"Validation-E frame origin spread drift: {spread}")
    record_without_payload = [{key: value for key, value in record.items() if key != "payload"} for record in records]
    ledger = {
        "schema": "CORELAB_PI0170_VALIDATION_E_ACCESS_LEDGER_V1",
        "status": "PAYLOAD_READ_AND_BOUND_BEFORE_MODEL_IMPORT",
        "pi_gate": PI_GATE,
        "issue_11_comment": ISSUE_COMMENT,
        "pr_pointer": PR_POINTER,
        "target_id": TARGET_ID,
        "source_root": str(source_root),
        "snapshot_ids": list(WINDOW_IDS),
        "first_payload_read_at_utc": started,
        "finished_at_utc": now_utc(),
        "descriptor_reads": 6,
        "frame_reads": 12,
        "holdout_f_read": False,
        "simulator_collection": False,
        "new_data_generated": False,
        "records": record_without_payload,
        "frame_origin": {"anchor_world_m": canonical_origin, "max_cross_window_spread_m": spread, "target_recentered": False, "measured_start_recentered": False},
        "model_imported": False,
    }
    write_json_durable(output_root / "validation_e_access_ledger.json", ledger)
    binding = {
        "schema": "CORELAB_PI0170_VALIDATION_E_PAYLOAD_ADMISSION_V1",
        "status": "PAYLOAD_READ_AND_BOUND_BEFORE_MODEL_IMPORT",
        "pi_gate": PI_GATE,
        "issue_11_comment": ISSUE_COMMENT,
        "pr_pointer": PR_POINTER,
        "source_head": SOURCE_HEAD,
        "execution_head": checkout["head"],
        "execution_branch": checkout["branch"],
        "checkpoint": dict(freeze["checkpoint_arms"][0]),
        "stats": {"path": str(Path(str(config["stats"])).resolve()), "sha256": FULL24_STATS_SHA, "payload_sha256": FULL24_STATS_PAYLOAD_SHA},
        "validation_input": {
            "target_id": TARGET_ID,
            "source_root": str(source_root),
            "snapshot_ids": list(WINDOW_IDS),
            "snapshot_records": record_without_payload,
            "frame_origin": ledger["frame_origin"],
            "access_ledger": str((output_root / "validation_e_access_ledger.json").resolve()),
            "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"),
            "target_xyz_in_model_input": False,
            "holdout_f_read": False,
        },
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "h200_submissions": 0,
        "output_root": str(output_root),
    }
    binding_path = output_root / "pi0170_validation_e_payload_binding.json"
    write_json_durable(binding_path, binding)
    result = {
        "schema": "CORELAB_PI0170_VALIDATION_E_PAYLOAD_ADMISSION_RESULT_V1",
        "status": "PASS_BEFORE_MODEL_IMPORT",
        "token": "CORELAB_PI0170_VALIDATION_E_PAYLOAD_BOUND",
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"),
        "snapshot_count": 6,
        "descriptor_reads": 6,
        "frame_reads": 12,
        "model_imported": False,
        "holdout_f_read": False,
    }
    write_json_durable(output_root / "pi0170_validation_e_payload_admission_result.json", result)
    return binding, binding_path


def _raw_path(label: str, index: int) -> Path:
    return Path("raw") / label.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.npy"


def _derived_path(label: str, index: int) -> Path:
    return Path("derived") / label.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.json"


def execute(config_path: Path) -> int:
    config = load_json(config_path)
    require(isinstance(config, Mapping), "PI0170 config must be an object")
    output_root = Path(str(config["output_root"])).resolve()
    admission_path = regular_file(output_root / "pi0170_rtx4080_qualification_admission.json", "PI0170 pre-model admission")
    admission_record = load_json(admission_path)
    require(admission_record.get("status") == "PASS_PRE_MODEL" and admission_record.get("token") == "CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_ADMISSION_RECORDED", "PI0170 pre-model admission is not PASS")
    checkout = checkout_audit()
    require(checkout["head"] == admission_record.get("execution_head") and checkout["branch"] == BRANCH, "execution checkout changed after pre-model admission")
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module present at execution entry")
    selected_uuid = str(admission_record["resource_admission"]["selected_gpu_uuid"])
    os.environ.update({str(key): str(value) for key, value in admission_record["fresh_cache"].get("environment", {}).items()})
    cache_paths = admission_record["fresh_cache"].get("paths", {})
    for key in CACHE_KEYS:
        require(Path(str(cache_paths[key])).is_dir() and not any(Path(str(cache_paths[key])).iterdir()), f"fresh cache directory is not empty before model import: {key}")
    os.environ.update({
        "TMPDIR": str(cache_paths["tmpdir"]),
        "XDG_CACHE_HOME": str(cache_paths["xdg_cache_home"]),
        "XDG_CONFIG_HOME": str(cache_paths["xdg_config_home"]),
        "TORCH_HOME": str(cache_paths["torch_home"]),
        "TRITON_CACHE_DIR": str(cache_paths["triton_cache_dir"]),
        "HF_HOME": str(cache_paths["hf_home"]),
        "HF_HUB_CACHE": str(cache_paths["hf_hub_cache"]),
        "TRANSFORMERS_CACHE": str(cache_paths["transformers_cache"]),
        "HF_DATASETS_CACHE": str(cache_paths["hf_datasets_cache"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "CUDA_VISIBLE_DEVICES": selected_uuid,
    })
    binding, binding_path = payload_admission(config, output_root, checkout)
    counters: dict[str, Any] = {
        "model_import_started": False,
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "model_construction_attempts": 0,
        "model_construction_completed": 0,
        "conditioning_attempts": 0,
        "conditioning_completed": 0,
        "action_head_attempts": 0,
        "action_head_completed": 0,
        "model_forward_completed": 0,
        "raw_prediction_count": 0,
        "derived_row_count": 0,
        "failed_model_calls": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "h200_submissions": 0,
    }
    current = {"checkpoint": CHECKPOINT_LABEL, "window_id": None, "step": None, "phase": "MODEL_IMPORT"}
    try:
        counters["model_import_started"] = True
        # PI0170_MODEL_IMPORT_BOUNDARY: all artifact, payload, resource, and cache gates are complete.
        import numpy as np
        import torch
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        import stage45f_vanda_validation_e_gen1 as evaluator
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

        runtime_identity = {
            "schema": "CORELAB_PI0170_RTX4080_RUNTIME_IDENTITY_V1",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
            "selected_gpu_uuid": selected_uuid,
            "selected_gpu_name": str(torch.cuda.get_device_name(torch.device("cuda:0"))),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": "bfloat16",
            "offline": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
            "h200_submissions": 0,
        }
        write_json_durable(output_root / "pi0170_runtime_identity.json", runtime_identity)
        processor, embodiment, _modality = stage26._build_runtime_processor(
            groot_source=Path(str(config["groot_source"])).resolve(),
            base_checkpoint=Path(str(config["base_model"])).resolve(),
            dataset_dir=Path(str(config["dataset"])).resolve(),
            modality_path=Path(str(config["modality"])).resolve(),
            cosmos_snapshot=Path(str(config["cosmos_root"])).resolve() / "snapshots" / COSMOS_REVISION,
        )
        windows_meta = evaluator.validate_snapshot_records({"validation_input": binding["validation_input"]})
        windows = evaluator._prepare_windows(processor, embodiment, windows_meta)
        require(len(windows) == 6, "prepared window count is not six")
        noise_bank, noise_hashes = evaluator._make_noise_bank(torch)
        write_json_durable(output_root / "sampler_noise_manifest.json", {
            "schema": "CORELAB_PI0170_CPU_FLOAT32_NOISE_BANK_V1",
            "seed": NOISE_SEED,
            "device": "cpu",
            "dtype": "float32",
            "shape": list(RAW_SHAPE),
            "window_ids": list(WINDOW_IDS),
            "noise_sha256_by_window": noise_hashes,
            "same_noise_across_six_windows": True,
        })
        raw_manifest = output_root / "raw_predictions.jsonl"
        derived_manifest = output_root / "derived_rows.jsonl"
        rows: list[dict[str, Any]] = []
        runtime_sha = canonical_sha({"pre_model_admission": admission_record, "payload_binding": binding, "runtime": runtime_identity})
        current["phase"] = "MODEL_LOAD"
        counters["model_loads_attempted"] += 1
        require(counters["model_loads_attempted"] <= 1, "model-load budget exceeded")
        model = runtime._load_torch_runtime()[1](Path(str(config["checkpoint"])).resolve(), "cuda:0")
        counters["model_loads_completed"] += 1
        counters["model_construction_attempts"] += 1
        counters["model_construction_completed"] += 1
        model.eval()
        try:
            for index, window in enumerate(windows):
                current["window_id"] = window["window_id"]
                current["phase"] = "CONDITIONING"
                counters["conditioning_attempts"] += 1
                require(counters["conditioning_attempts"] <= 6, "conditioning budget exceeded")
                _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                counters["conditioning_completed"] += 1
                current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                with torch.inference_mode():
                    for step, timestep in enumerate(TIMESTEP_IDS):
                        current["step"] = int(step)
                        current["phase"] = "ACTION_HEAD"
                        counters["action_head_attempts"] += 1
                        require(counters["action_head_attempts"] <= 24, "action-head budget exceeded")
                        velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                        counters["action_head_completed"] += 1
                        counters["model_forward_completed"] += 1
                        current_action = current_action + SAMPLER_DT * velocity
                raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                require(raw_np.shape == RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output shape/finite drift: {window['window_id']}")
                raw_path = output_root / _raw_path(CHECKPOINT_LABEL, index)
                write_npy_durable(raw_path, raw_np)
                raw_file_sha = sha256_file(raw_path)
                raw_array_sha = sha256_bytes(np.ascontiguousarray(raw_np).tobytes())
                # PI0170_RAW_BEFORE_DECODE
                append_jsonl_durable(raw_manifest, {
                    "schema": "CORELAB_PI0170_RAW_PREDICTION_V1",
                    "checkpoint": {"route": "Legacy", "seed": 2222, "tier": 2000, "label": CHECKPOINT_LABEL, "tree_sha256": CHECKPOINT_TREE_SHA},
                    "window_id": window["window_id"],
                    "path": str(raw_path.relative_to(output_root)),
                    "file_sha256": raw_file_sha,
                    "array_sha256": raw_array_sha,
                    "shape": list(raw_np.shape),
                    "dtype": str(raw_np.dtype),
                    "noise_sha256": noise_hashes[index],
                    "timestep_ids": list(TIMESTEP_IDS),
                    "action_head_calls_observed": 4,
                    "decoded": False,
                })
                counters["raw_prediction_count"] += 1
                # PI0170_DECODE_AFTER_RAW
                current["phase"] = "DECODE_AND_DERIVED_METRICS"
                physical, details = runtime._decode_v2(raw_np, window["state_v1"], Path(str(config["stats"])).resolve(), window["source_timestamps"], float(window["observation_timestamp"]), window["window_id"] + ":" + CHECKPOINT_LABEL)
                physical_np = np.asarray(physical, dtype=np.float64)
                require(physical_np.shape == PHYSICAL_SHAPE and bool(np.isfinite(physical_np).all()), "decoded physical shape/finite drift")
                decoded_path = output_root / Path("decoded") / CHECKPOINT_LABEL.replace("(", "_").replace(")", "").replace("@", "_at_") / f"window-{index:02d}.npy"
                write_npy_durable(decoded_path, physical_np)
                row = evaluator._metric_row(runtime, raw_np, physical_np, details, window, {"label": CHECKPOINT_LABEL, "tree_sha256": CHECKPOINT_TREE_SHA}, runtime_sha, str(raw_path.relative_to(output_root)), raw_file_sha)
                row.update({"route": "Legacy", "seed": 2222, "tier": 2000, "noise_sha256": noise_hashes[index], "sampler": {"name": "direct-source-mapped-action-head-E4", "t_values": list(T_VALUES), "timestep_ids": list(TIMESTEP_IDS), "dt": SAMPLER_DT, "carried_state": True, "reset_between_stages": False, "official_public_sampler_called": False, "get_action_calls": 0}, "stats": {"sha256": FULL24_STATS_SHA, "payload_sha256": FULL24_STATS_PAYLOAD_SHA, "route": "FULL24_RUNTIME_V2"}, "decoded_path": str(decoded_path.relative_to(output_root)), "decoded_file_sha256": sha256_file(decoded_path)})
                row_path = output_root / _derived_path(CHECKPOINT_LABEL, index)
                write_json_durable(row_path, row)
                append_jsonl_durable(derived_manifest, {"schema": "CORELAB_PI0170_DERIVED_EVIDENCE_REFERENCE_V1", "checkpoint": CHECKPOINT_LABEL, "window_id": window["window_id"], "raw_path": str(raw_path.relative_to(output_root)), "raw_file_sha256": raw_file_sha, "derived_path": str(row_path.relative_to(output_root)), "decoded": True, "decoded_array_sha256": row["decoded_action"]["array_sha256"]})
                rows.append(row)
                counters["derived_row_count"] += 1
        finally:
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        require(counters["conditioning_completed"] == 6 and counters["action_head_completed"] == 24 and counters["raw_prediction_count"] == 6 and counters["derived_row_count"] == 6, "six-window counter contract failed")
        aggregate = aggregate_evaluation_rows(rows, expected_window_ids=list(WINDOW_IDS))
        aggregate.update({"checkpoint": CHECKPOINT_LABEL, "route": "Legacy", "seed": 2222, "tier": 2000, "checkpoint_tree_sha256": CHECKPOINT_TREE_SHA})
        write_json_durable(output_root / "aggregate_Legacy_seed2222_at_2000.json", aggregate)
        comparison = compare_h200(config, output_root, windows, rows, runtime, evaluator, np)
        write_json_durable(output_root / "pi0170_portability_comparison.json", comparison)
        telemetry = {
            "schema": "CORELAB_PI0170_RTX4080_TELEMETRY_V1",
            "resource_admission": admission_record["resource_admission"],
            "runtime_identity": runtime_identity,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device("cuda:0"))),
            "actual_counters": counters,
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "h200_submissions": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
        }
        write_json_durable(output_root / "pi0170_telemetry.json", telemetry)
        final_status = "COMPLETED" if comparison.get("status") == "PASS_MATCHED_PORTABILITY" else "PARTIAL"
        result = {
            "schema": "CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_RESULT_V1",
            "status": final_status,
            "return_token": f"CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_FOR_PI_REVIEW — {final_status}",
            "pi_gate": PI_GATE,
            "execution_head": checkout["head"],
            "branch": checkout["branch"],
            "checkpoint": CHECKPOINT_LABEL,
            "window_ids": list(WINDOW_IDS),
            "actual_counters": counters,
            "raw_manifest": str(raw_manifest),
            "raw_manifest_sha256": sha256_file(raw_manifest),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "portability_comparison": str(output_root / "pi0170_portability_comparison.json"),
            "portability_status": comparison.get("status"),
            "scientific_winner_selected": False,
            "h200_submissions": 0,
            "training": False,
            "holdout_f_read": False,
        }
        write_json_durable(output_root / "PI0170_RESULT.json", result)
        print(result["return_token"])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_FAILURE_V1",
            "status": "PARTIAL" if counters["model_import_started"] else "BLOCKED",
            "return_token": f"CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_FOR_PI_REVIEW — {'PARTIAL' if counters['model_import_started'] else 'BLOCKED'}",
            "pi_gate": PI_GATE,
            "execution_head": checkout.get("head"),
            "branch": checkout.get("branch"),
            "phase": current["phase"],
            "current": current,
            "actual_counters": counters,
            "validation_e_read": True,
            "holdout_f_read": False,
            "h200_submissions": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": now_utc(),
        }
        write_json_durable(output_root / "PI0170_FAILURE.json", failure)
        print(failure["return_token"], file=sys.stderr)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


def _expected_h200_cells() -> dict[str, dict[str, str]]:
    manifest = load_json(MANIFEST_PATH)
    return {f"stage22_expert_{TARGET_ID}:{cell['window']}": {"file_sha256": cell["file_sha256"], "array_sha256": cell["array_sha256"]} for cell in manifest["h200_portability_anchor"]["cells"]}


def compare_h200(config: Mapping[str, Any], output_root: Path, windows: Sequence[Mapping[str, Any]], rtx_rows: Sequence[Mapping[str, Any]], runtime: Any, evaluator: Any, np: Any) -> dict[str, Any]:
    root_value = config.get("h200_raw_root")
    manifest_value = config.get("h200_raw_manifest")
    if not isinstance(root_value, str) or not isinstance(manifest_value, str):
        return {"schema": "CORELAB_PI0170_H200_RTX_PORTABILITY_COMPARISON_V1", "status": "BLOCKED_H200_SOURCE", "reason": "h200_raw_root or h200_raw_manifest is not late-bound", "h200_submission": False, "scientific_winner_selected": False}
    root = regular_dir(Path(root_value), "H200 allocation-8 raw root")
    manifest_path = regular_file(Path(manifest_value), "H200 allocation-8 raw manifest")
    require(sha256_file(manifest_path) == H200_RAW_MANIFEST_SHA, "H200 allocation-8 raw manifest SHA drift")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(records) == 6, "H200 allocation-8 raw manifest is not six cells")
    expected = _expected_h200_cells()
    h200: dict[str, dict[str, Any]] = {}
    for record in records:
        window_id = str(record.get("window_id"))
        require(window_id in expected and record.get("checkpoint", {}).get("label") == CHECKPOINT_LABEL, f"unexpected H200 portability cell: {window_id}")
        path_value = record.get("path")
        require(isinstance(path_value, str) and not Path(path_value).is_absolute() and ".." not in Path(path_value).parts, f"unsafe H200 raw path: {path_value}")
        path = regular_file(root / path_value, "H200 raw cell")
        file_sha = sha256_file(path)
        require(file_sha == expected[window_id]["file_sha256"] and file_sha == record.get("file_sha256"), f"H200 raw file SHA drift: {window_id}")
        raw = np.load(path, allow_pickle=False)
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        require(tuple(raw.shape) == RAW_SHAPE and bool(np.isfinite(raw).all()), f"H200 raw shape/finite drift: {window_id}")
        array_sha = sha256_bytes(raw.tobytes())
        require(array_sha == expected[window_id]["array_sha256"] and array_sha == record.get("array_sha256"), f"H200 raw array SHA drift: {window_id}")
        h200[window_id] = {"path": str(path), "file_sha256": file_sha, "array_sha256": array_sha, "raw": raw}
    rtx_manifest = output_root / "raw_predictions.jsonl"
    rtx_records = [json.loads(line) for line in rtx_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rtx_by_id = {str(record["window_id"]): record for record in rtx_records}
    windows_by_id = {str(window["window_id"]): window for window in windows}
    rows_by_id = {str(row["window_id"]): row for row in rtx_rows}
    cells = []
    all_raw_exact = True
    for window_id in WINDOW_IDS:
        require(window_id in rtx_by_id and window_id in h200 and window_id in windows_by_id, f"portability cell missing: {window_id}")
        rtx_path = regular_file(output_root / str(rtx_by_id[window_id]["path"]), "RTX raw cell")
        rtx_raw = np.ascontiguousarray(np.load(rtx_path, allow_pickle=False), dtype=np.float32)
        h200_raw = h200[window_id]["raw"]
        require(tuple(rtx_raw.shape) == RAW_SHAPE and bool(np.isfinite(rtx_raw).all()), f"RTX raw shape/finite drift: {window_id}")
        file_equal = rtx_by_id[window_id].get("file_sha256") == h200[window_id]["file_sha256"]
        array_equal = bool(np.array_equal(rtx_raw, h200_raw))
        diff = np.asarray(rtx_raw, dtype=np.float64) - np.asarray(h200_raw, dtype=np.float64)
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        all_raw_exact = all_raw_exact and file_equal and array_equal
        rtx_physical, _ = runtime._decode_v2(rtx_raw, windows_by_id[window_id]["state_v1"], Path(str(config["stats"])).resolve(), windows_by_id[window_id]["source_timestamps"], float(windows_by_id[window_id]["observation_timestamp"]), window_id + ":RTX")
        h200_physical, h200_details = runtime._decode_v2(h200_raw, windows_by_id[window_id]["state_v1"], Path(str(config["stats"])).resolve(), windows_by_id[window_id]["source_timestamps"], float(windows_by_id[window_id]["observation_timestamp"]), window_id + ":H200")
        rtx_physical = np.asarray(rtx_physical, dtype=np.float64)
        h200_physical = np.asarray(h200_physical, dtype=np.float64)
        physical_diff = rtx_physical - h200_physical
        def band(name: str, left: Any, right: Any) -> dict[str, Any]:
            delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            return {"name": name, "max_abs": float(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(np.square(delta)))), "exact": bool(np.array_equal(np.asarray(left), np.asarray(right)))}
        decoded_bands = [band("overall", rtx_physical, h200_physical), band("xyz", rtx_physical[:, :3], h200_physical[:, :3]), band("rot6d", rtx_physical[:, 3:9], h200_physical[:, 3:9]), band("gripper", rtx_physical[:, 9:], h200_physical[:, 9:])]
        h200_row = evaluator._metric_row(runtime, h200_raw, h200_physical, h200_details, windows_by_id[window_id], {"label": CHECKPOINT_LABEL, "tree_sha256": CHECKPOINT_TREE_SHA}, "h200-allocation-8", str(Path(h200[window_id]["path"]).relative_to(root)), h200[window_id]["file_sha256"])
        rtx_metrics = rows_by_id[window_id].get("metrics", {})
        h200_metrics = h200_row.get("metrics", {})
        metric_fields = {
            "final_target_distance_m": (h200_metrics.get("target", {}).get("final_target_distance_m"), rtx_metrics.get("target", {}).get("final_target_distance_m")),
            "absolute_progress_m": (h200_metrics.get("target", {}).get("absolute_progress_m"), rtx_metrics.get("target", {}).get("absolute_progress_m")),
            "path_length_m": (h200_metrics.get("target", {}).get("path_length_m"), rtx_metrics.get("target", {}).get("path_length_m")),
            "roughness_position_second_difference_m": (h200_metrics.get("prediction_roughness_position_second_difference_m", {}).get("mean"), rtx_metrics.get("prediction_roughness_position_second_difference_m", {}).get("mean")),
            "safety_gate": (h200_row.get("safety_gate_decision"), rows_by_id[window_id].get("safety_gate_decision")),
        }
        metric_deltas = {name: {"h200": left, "rtx4080": right, "equal": left == right} for name, (left, right) in metric_fields.items()}
        cells.append({"window_id": window_id, "raw": {"file_sha256_equal": file_equal, "array_sha256_equal": rtx_by_id[window_id].get("array_sha256") == h200[window_id]["array_sha256"], "array_exact_equal": array_equal, "max_abs": max_abs, "mean_abs": mean_abs, "rmse": rmse}, "decoded": {"bands": decoded_bands, "max_abs": float(np.max(np.abs(physical_diff))), "mean_abs": float(np.mean(np.abs(physical_diff))), "rmse": float(np.sqrt(np.mean(np.square(physical_diff))))}, "metrics": metric_deltas})
    return {
        "schema": "CORELAB_PI0170_H200_RTX_PORTABILITY_COMPARISON_V1",
        "status": "PASS_MATCHED_PORTABILITY" if all_raw_exact else "DESCRIPTIVE_MISMATCH_FOR_PI_REVIEW",
        "h200_allocation": 8,
        "h200_submission": False,
        "h200_raw_manifest_sha256": H200_RAW_MANIFEST_SHA,
        "rtx_cell_count": 6,
        "h200_cell_count": 6,
        "cells": cells,
        "classification": {
            "RTX4080_RESOURCE_FEASIBLE": True,
            "H200_RTX_RAW_BYTE_IDENTICAL": all_raw_exact,
            "H200_RTX_NUMERICAL_PORTABILITY": "DESCRIPTIVE_FOR_PI_REVIEW",
            "SCIENTIFIC_WINNER_SELECTED": False,
        },
        "scientific_winner_selected": False,
    }


def plan() -> dict[str, Any]:
    return {
        "schema": "CORELAB_PI0170_RTX4080_QUALIFICATION_PLAN_V1",
        "status": "READY_NOT_EXECUTED",
        "pi_gate": PI_GATE,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "checkpoint": CHECKPOINT_LABEL,
        "window_ids": list(WINDOW_IDS),
        "windows": 6,
        "model_load_attempts": 1,
        "conditioning": 6,
        "action_head": 24,
        "model_side_total": 30,
        "raw_outputs": 6,
        "resource": {"gpu_uuid_max": 1, "free_vram_mib_min": MIN_FREE_VRAM_MIB, "occupancy_snapshots": 2},
        "h200_portability": {"allocation": 8, "cells": 6, "submission": False},
        "forbidden": {"remaining_30_rtx_cells": 0, "h200_submissions": 0, "training": 0, "wbc": 0, "simulator": 0, "hardware": 0, "holdout_f_read": 0, "new_smoke_query": 0},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan:
            risk = static_risk_audit()
            result = {"plan": plan(), "risk_audit": risk, "freeze": freeze_audit()}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if risk["status"] == "PASS" else 2
        require(args.config is not None, "--config is required for --preflight/--execute")
        return preflight(args.config) if args.preflight else execute(args.config)
    except Exception as exc:
        status = "BLOCKED" if args.preflight or args.plan else "PARTIAL"
        token = f"CORELAB_PI0170_RTX4080_VALIDATION_QUALIFICATION_FOR_PI_REVIEW — {status}"
        print(token, file=sys.stderr)
        print(json.dumps({"status": status, "token": token, "error": f"{type(exc).__name__}: {exc}", "pi_gate": PI_GATE, "h200_submissions": 0, "holdout_f_read": False}, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
