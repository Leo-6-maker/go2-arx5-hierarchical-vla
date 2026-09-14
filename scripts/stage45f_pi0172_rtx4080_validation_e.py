#!/usr/bin/env python3
"""PI0172 Corelab RTX4080 Validation-E generation and CPU sealing.

This entrypoint is intentionally fail-closed.  Its standard-library-only
preflight binds the exact six frozen checkpoints, runtime artifacts, source
manifest, branch, and RTX resource admission before Validation-E payload or
model imports.  Execution then reuses valid raw cells and evaluates only
missing cells on the same admitted GPU UUID.  Raw arrays are sealed before
decode; all decode, metrics, aggregation, and portability work is CPU-side.

No H200 submission, training, WBC, simulator, hardware, Holdout-F, probe,
or automatic scientific selection is present in this program.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MANIFEST_PATH = REPO / "manifests/STAGE45F_PI0172_RTX4080_VALIDATION_E_V1.json"
FREEZE_PATH = REPO / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json"

PI_GATE = "PI_GATE_STATE_0172"
ISSUE_COMMENT = 5659769556
PR_POINTER = 5659771060
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"
BRANCH = "stage45f/corelab-validation-e-rtx4080-qualification"
FROZEN_EVALUATION_COMMIT = "522ea718e99ff8b1c4b2aa08bec680ed5adfcae5"
TARGET_ID = "red_target_holdout_e"
WINDOW_IDS = tuple(f"stage22_expert_{TARGET_ID}:expert-{index:04d}" for index in range(6))

ARM_SPECS = (
    {"route": "Legacy", "seed": 2222, "tier": 2000, "label": "Legacy(seed2222)@2000", "tree_sha256": "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"},
    {"route": "Legacy", "seed": 2223, "tier": 2000, "label": "Legacy(seed2223)@2000", "tree_sha256": "4895df0a0d5adf7e9010bce70fe2aaf447bbb7e4d4a28a179316a502d666592b"},
    {"route": "Legacy", "seed": 2224, "tier": 2000, "label": "Legacy(seed2224)@2000", "tree_sha256": "fc4dc36959a44398aed73b0b8f949dcbbfe42d1d15157e87667c17850384d2e9"},
    {"route": "L1", "seed": 2222, "tier": 2000, "label": "L1(seed2222)@2000", "tree_sha256": "53293d0ae3555c00acb7595b6c6dd19ed92450668344123cdc46efd5b69fcd49"},
    {"route": "L1", "seed": 2223, "tier": 2000, "label": "L1(seed2223)@2000", "tree_sha256": "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16"},
    {"route": "L1", "seed": 2224, "tier": 2000, "label": "L1(seed2224)@2000", "tree_sha256": "3c25b67036fcf3af037e3ad98d7a76b97c1933cd54445dd984436dbac5951b92"},
)
ARM_BY_LABEL = {item["label"]: item for item in ARM_SPECS}

FULL24_DATASET_TREE_SHA = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_SOURCE_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_PAYLOAD_TREE_SHA = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
GROOT_MANIFEST_SHA = "96b883e56b54b963bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
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
MODEL_LOAD_ATTEMPTS_MAX = 8
CONDITIONING_ATTEMPTS_MAX = 40
ACTION_HEAD_ATTEMPTS_MAX = 160
MODEL_SIDE_ATTEMPTS_MAX = 200

CACHE_KEYS = (
    "tmpdir", "xdg_cache_home", "xdg_config_home", "torch_home",
    "triton_cache_dir", "hf_home", "hf_hub_cache", "transformers_cache",
    "hf_datasets_cache",
)
FORBIDDEN_TOP_LEVEL_IMPORTS = {"torch", "numpy", "cv2", "gr00t", "transformers", "cuda", "isaacgym"}
FORBIDDEN_CALLS = {"train", "backward", "get_action", "qsub", "sbatch", "srun", "submit", "save_checkpoint", "save_model", "save_pretrained"}
ALLOWED_CHANGE_PATHS = {
    "scripts/stage45f_pi0170_rtx4080_qualification.py",
    "manifests/STAGE45F_PI0170_RTX4080_VALIDATION_QUALIFICATION_V1.json",
    "tests/test_stage45f_pi0170_rtx4080_qualification.py",
    "scripts/stage45f_pi0172_rtx4080_validation_e.py",
    "manifests/STAGE45F_PI0172_RTX4080_VALIDATION_E_V1.json",
    "tests/test_stage45f_pi0172_rtx4080_validation_e.py",
}


class PI0172Error(RuntimeError):
    """Raised when the PI0172 frozen gate cannot continue."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0172Error(message)


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
        raise PI0172Error(f"required JSON is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PI0172Error(f"cannot parse JSON: {path}") from exc


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


def write_if_absent(path: Path, value: Any) -> None:
    if path.exists():
        return
    write_json_durable(path, value)


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-800:]}")
    return completed.stdout.strip()


def regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PI0172Error(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def regular_dir(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PI0172Error(f"{label} is unavailable: {path}") from exc
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


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


def static_risk_audit() -> dict[str, Any]:
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
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                forbidden_calls.append(name)
    markers = {
        "six_frozen_arms": len(ARM_SPECS) == 6 and "ARM_SPECS" in source,
        "six_windows": len(WINDOW_IDS) == 6 and "WINDOW_IDS" in source,
        "e4_schedule": "TIMESTEP_IDS = (0, 250, 500, 750)" in source,
        "raw_before_decode": "PI0172_RAW_BEFORE_DECODE" in source and "PI0172_DECODE_AFTER_RAW" in source and source.index("PI0172_RAW_BEFORE_DECODE") < source.index("PI0172_DECODE_AFTER_RAW"),
        "ledger_before_model_load": "execution_ledger.jsonl" in source and "MODEL_LOAD" in source,
        "same_uuid": "same_uuid" in source and "CUDA_VISIBLE_DEVICES" in source,
        "no_h200_submit": '"h200_submissions": 0' in source,
        "no_training": '"training": False' in source and '"optimizer_steps": 0' in source,
        "holdout_f_locked": "holdout_f_read" in source and "Holdout-F" in source,
        "remaining_30_continuation": "remaining_30_rtx_cells" in source and "36" in source,
    }
    status = "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL"
    return {
        "schema": "CORELAB_PI0172_ZERO_MODEL_EXECUTION_RISK_AUDIT_V1",
        "status": status,
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "model_imported": False,
        "model_loads": 0,
        "model_forward": 0,
        "training": False,
        "h200_submissions": 0,
        "holdout_f_read": False,
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
    require(set(changed).issubset(ALLOWED_CHANGE_PATHS), f"unexpected files changed after PI0172 source HEAD: {changed}")
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
    require([(item.get("route"), item.get("seed"), item.get("tier")) for item in arms] == [(item["route"], item["seed"], item["tier"]) for item in ARM_SPECS], "frozen arm order drift")
    for expected, observed in zip(ARM_SPECS, arms):
        require(observed.get("label") == expected["label"] and observed.get("tree_sha256") == expected["tree_sha256"], f"frozen arm identity drift: {expected['label']}")
    inputs = freeze.get("validation_input_lock")
    require(isinstance(inputs, Mapping) and inputs.get("snapshot_ids") == list(WINDOW_IDS), "frozen Validation-E windows drift")
    protocol = freeze.get("evaluation_protocol")
    require(isinstance(protocol, Mapping), "frozen protocol missing")
    require(protocol.get("noise_seed") == NOISE_SEED and protocol.get("noise_shape") == list(RAW_SHAPE), "noise contract drift")
    require(protocol.get("timestep_ids") == list(TIMESTEP_IDS) and protocol.get("t_values") == list(T_VALUES) and protocol.get("dt") == SAMPLER_DT, "E4 contract drift")
    require(protocol.get("target_xyz_in_model_input") is False and protocol.get("public_sampler_calls") == 0 and protocol.get("get_action_calls") == 0, "model-input/sampler contract drift")
    boundaries = freeze.get("locked_boundaries")
    require(isinstance(boundaries, Mapping) and all(value is False for value in boundaries.values()), "a frozen boundary is open")
    return {"schema": "CORELAB_PI0172_FROZEN_RECIPE_AUDIT_V1", "status": "PASS_STATIC_NOT_EXECUTED", "freeze_path": str(FREEZE_PATH), "freeze_sha256": sha256_file(FREEZE_PATH), "evaluation_code_commit": FROZEN_EVALUATION_COMMIT, "checkpoint_arms": [dict(item) for item in ARM_SPECS], "snapshot_ids": list(WINDOW_IDS), "payload_read": False, "holdout_f_read": False, "model_imported": False}


def checkpoint_hash_prefix(arm: Mapping[str, Any], checkpoint: Path) -> str:
    route = str(arm["route"])
    seed = int(arm["seed"])
    if route == "Legacy" and seed == 2222:
        return checkpoint.name
    if route == "Legacy" and seed in {2223, 2224}:
        return f"stage45e_legacy/{checkpoint.name}"
    if route == "L1" and seed in {2222, 2223, 2224}:
        return f"stage45e_l1/{checkpoint.name}"
    raise PI0172Error(f"unsupported checkpoint identity namespace: {route}@{seed}")


def artifact_audit(config: Mapping[str, Any], checkout: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module imported before artifact admission")
    required = ("base_model", "checkpoints", "dataset", "groot_source", "groot_manifest", "cosmos_root", "modality", "stats", "source_root")
    for key in required:
        require(key in config and config[key], f"late-bound artifact path is missing: {key}")
    for value in config.values():
        if isinstance(value, str):
            require("holdout-f" not in value.lower(), "Holdout-F path is forbidden")
    checkpoints = config["checkpoints"]
    require(isinstance(checkpoints, Mapping), "six-checkpoint path map is missing")
    base_model = regular_dir(Path(str(config["base_model"])), "BASE model")
    dataset = regular_dir(Path(str(config["dataset"])), "FULL24 dataset")
    groot = regular_dir(Path(str(config["groot_source"])), "GR00T source")
    groot_manifest_path = Path(str(config["groot_manifest"]))
    if not groot_manifest_path.exists():
        raise PI0172Error(f"CORELAB_PI0172_ARTIFACT_RELAY_REQUIRED: exact GR00T manifest unavailable: {groot_manifest_path}; expected SHA256={GROOT_MANIFEST_SHA}")
    groot_manifest = regular_file(groot_manifest_path, "GR00T source manifest")
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
    checkpoint_audit: list[dict[str, Any]] = []
    for arm in ARM_SPECS:
        path_value = checkpoints.get(arm["label"])
        require(isinstance(path_value, str) and path_value, f"checkpoint path missing: {arm['label']}")
        checkpoint = regular_dir(Path(path_value), f"checkpoint {arm['label']}")
        entries = tree_entries(checkpoint, checkpoint_hash_prefix(arm, checkpoint))
        observed = tree_digest(entries)
        require(observed == arm["tree_sha256"], f"checkpoint tree SHA drift: {arm['label']}: {observed}")
        trainer_state = regular_file(checkpoint / "trainer_state.json", f"trainer state {arm['label']}")
        trainer_json = load_json(trainer_state)
        require(int(trainer_json.get("global_step", -1)) == 2000, f"checkpoint tier drift: {arm['label']}")
        checkpoint_audit.append({**dict(arm), "path": str(checkpoint), "hash_prefix": checkpoint_hash_prefix(arm, checkpoint), "file_count": len(entries), "total_bytes": sum(int(item["bytes"]) for item in entries), "trainer_state_sha256": sha256_file(trainer_state)})
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import stage45c_h200_eval_runtime_adapter as runtime_adapter

    dataset_identity = runtime_adapter._verify_dataset(dataset)
    stats_identity = runtime_adapter._verify_runtime_stats(stats)
    groot_manifest_identity = runtime_adapter._verify_groot_runtime_manifest(groot_manifest)
    if (groot / ".git").is_dir():
        groot_git = run_text(["git", "-C", str(groot), "rev-parse", "HEAD"])
        require(groot_git == GROOT_SOURCE_COMMIT, "GR00T source HEAD drift")
        require(not run_text(["git", "-C", str(groot), "status", "--porcelain", "--untracked-files=all"]), "GR00T source checkout is dirty")
    else:
        runtime_adapter._verify_groot_source_snapshot(groot, groot_manifest)
        groot_git = GROOT_SOURCE_COMMIT
    cosmos_digest, cosmos_entries = runtime_adapter.cosmos_tree_digest(cosmos)
    require(cosmos_digest == COSMOS_TREE_SHA, f"Cosmos tree SHA drift: {cosmos_digest}")
    cosmos_snapshot = regular_dir(cosmos / "snapshots" / COSMOS_REVISION, "Cosmos pinned snapshot")
    cosmos_ref = regular_file(cosmos / "refs" / "main", "Cosmos refs/main")
    require(cosmos_ref.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos refs/main revision drift")
    require("numpy" not in sys.modules and "torch" not in sys.modules, "runtime artifact audit imported a model-side module")
    return {
        "schema": "CORELAB_PI0172_ARTIFACT_RUNTIME_ADMISSION_V1", "status": "PASS_PRE_MODEL", "checkout_head": checkout["head"],
        "paths": {"base_model": str(base_model), "checkpoints": {item["label"]: item["path"] for item in checkpoint_audit}, "dataset": str(dataset), "groot_source": str(groot), "groot_manifest": str(groot_manifest), "cosmos_root": str(cosmos), "cosmos_snapshot": str(cosmos_snapshot), "modality": str(modality), "stats": str(stats), "source_root": str(Path(str(config["source_root"])).resolve())},
        "base_model": {"tree_sha256": base_tree, "file_count": len(base_entries), "processor_config_sha256": BASE_PROCESSOR_CONFIG_SHA, "statistics_sha256": BASE_PROCESSOR_STATS_SHA},
        "checkpoint_audit": checkpoint_audit, "checkpoint_count": len(checkpoint_audit), "dataset": dataset_identity, "stats": stats_identity, "groot_manifest": groot_manifest_identity, "groot_source_commit": groot_git, "groot_payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA,
        "cosmos": {"tree_sha256": cosmos_digest, "entry_count": cosmos_entries, "revision": COSMOS_REVISION, "refs_main_sha256": sha256_file(cosmos_ref)}, "modality_sha256": MODALITY_SHA, "validation_e_payload_read": False, "holdout_f_read": False, "model_imported": False,
    }


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
    return {"schema": "CORELAB_PI0172_RTX4080_OCCUPANCY_SNAPSHOT_V1", "snapshot_index": index, "captured_at_utc": now_utc(), "host": os.uname().nodename, "gpus": _parse_gpu_rows(gpu_raw), "foreign_compute_processes": _parse_compute_rows(compute_raw), "gpu_query_sha256": sha256_bytes((gpu_raw + "\n").encode()), "compute_query_sha256": sha256_bytes((compute_raw + "\n").encode())}


def resource_admission(config: Mapping[str, Any], *, expected_uuid: str | None = None, purpose: str = "preflight") -> dict[str, Any]:
    first = occupancy_snapshot(1)
    candidates = [row for row in first["gpus"] if "RTX 4080" in row["name"].upper() and row["total_vram_mib"] >= MIN_TOTAL_VRAM_MIB]
    preferred = expected_uuid or config.get("preferred_gpu_uuid")
    if preferred:
        candidates = [row for row in candidates if row["uuid"] == preferred]
    require(candidates, f"BLOCKED_SHARED_GPU_CAPACITY: no qualifying 32-GiB RTX4080 for {purpose}")
    selected_uuid = candidates[0]["uuid"]
    interval = float(config.get("occupancy_snapshot_interval_seconds", 2.0))
    require(0.0 < interval <= 10.0, "occupancy snapshot interval is outside the bounded range")
    time.sleep(interval)
    second = occupancy_snapshot(2)
    selected_rows = []
    for snapshot in (first, second):
        matches = [row for row in snapshot["gpus"] if row["uuid"] == selected_uuid]
        require(len(matches) == 1, f"BLOCKED_SHARED_GPU_CAPACITY: selected GPU UUID disappeared during {purpose}")
        selected = matches[0]
        require(selected["free_vram_mib"] >= MIN_FREE_VRAM_MIB, f"BLOCKED_SHARED_GPU_CAPACITY: free VRAM {selected['free_vram_mib']} MiB < {MIN_FREE_VRAM_MIB} MiB during {purpose}")
        foreign = [item for item in snapshot["foreign_compute_processes"] if item["gpu_uuid"] == selected_uuid and int(item["pid"]) != os.getpid()]
        require(not foreign, f"BLOCKED_SHARED_GPU_CAPACITY: foreign compute process on {selected_uuid} during {purpose}")
        selected_rows.append(selected)
    require(selected_rows[0]["uuid"] == selected_rows[1]["uuid"] == selected_uuid, "selected GPU UUID changed")
    return {"schema": "CORELAB_PI0172_RTX4080_RESOURCE_ADMISSION_V1", "status": "PASS_PRE_MODEL" if purpose == "preflight" else "PASS_MODEL_RESTART", "purpose": purpose, "host": first["host"], "selected_gpu_uuid": selected_uuid, "selected_gpu_name": selected_rows[0]["name"], "selected_cuda_mapping": {"CUDA_VISIBLE_DEVICES": selected_uuid, "device": "cuda:0"}, "min_free_vram_mib": MIN_FREE_VRAM_MIB, "snapshots": [first, second], "same_uuid": True, "foreign_process_interference": False, "gpu_uuid_count": 1}


def prepare_fresh_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    cache_root = Path(str(config.get("cache_root", ""))).resolve()
    require(cache_root.is_absolute() and "pi0172" in cache_root.name.lower(), "cache_root must be a fresh PI0172-specific absolute path")
    require(not cache_root.exists() and not cache_root.is_symlink(), f"fresh cache root already exists: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=False)
    paths = {key: cache_root / key for key in CACHE_KEYS}
    for path in paths.values():
        path.mkdir(parents=False, exist_ok=False)
    env = {"TMPDIR": paths["tmpdir"], "XDG_CACHE_HOME": paths["xdg_cache_home"], "XDG_CONFIG_HOME": paths["xdg_config_home"], "TORCH_HOME": paths["torch_home"], "TRITON_CACHE_DIR": paths["triton_cache_dir"], "HF_HOME": paths["hf_home"], "HF_HUB_CACHE": paths["hf_hub_cache"], "TRANSFORMERS_CACHE": paths["transformers_cache"], "HF_DATASETS_CACHE": paths["hf_datasets_cache"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "GROOT_HF_LOCAL_FIRST": "1", "GROOT_PATCH_MISTRAL": "1"}
    os.environ.update({key: str(value) for key, value in env.items()})
    return {"root": str(cache_root), "paths": {key: str(value) for key, value in paths.items()}, "environment": {key: str(value) for key, value in env.items() if key not in CACHE_KEYS}}


def _admission_module() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import stage45f_vanda_validation_e_admission as admission
    return admission


def preflight(config_path: Path) -> int:
    config = load_json(config_path)
    require(isinstance(config, Mapping), "PI0172 config must be an object")
    risk = static_risk_audit()
    require(risk["status"] == "PASS", "zero-model execution-risk audit failed")
    checkout = checkout_audit()
    expected_head = str(config.get("expected_head", checkout["head"]))
    require(checkout["head"] == expected_head, "preflight expected_head does not match checkout")
    freeze = freeze_audit()
    artifacts = artifact_audit(config, checkout, freeze)
    cache = prepare_fresh_cache(config)
    resource_record = resource_admission(config, purpose="preflight")
    output_root = Path(str(config["output_root"])).resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    record = {"schema": "CORELAB_PI0172_RTX4080_VALIDATION_E_ADMISSION_V1", "status": "PASS_PRE_MODEL", "token": "CORELAB_PI0172_RTX4080_VALIDATION_E_ADMISSION_RECORDED", "pi_gate": PI_GATE, "issue_11_comment": ISSUE_COMMENT, "pr_pointer": PR_POINTER, "source_head": SOURCE_HEAD, "execution_head": checkout["head"], "branch": checkout["branch"], "risk_audit": risk, "freeze_audit": freeze, "artifact_runtime_admission": artifacts, "resource_admission": resource_record, "fresh_cache": cache, "budgets": {"model_load_attempts_max": MODEL_LOAD_ATTEMPTS_MAX, "conditioning_max": CONDITIONING_ATTEMPTS_MAX, "action_head_max": ACTION_HEAD_ATTEMPTS_MAX, "model_side_max": MODEL_SIDE_ATTEMPTS_MAX, "unique_raw_outputs": 36}, "validation_e_payload_read": False, "holdout_f_read": False, "model_imported": False, "training": False, "h200_submissions": 0, "created_at_utc": now_utc(), "output_root": str(output_root)}
    write_json_durable(output_root / "pi0172_zero_model_risk_audit.json", risk)
    write_json_durable(output_root / "pi0172_artifact_runtime_admission.json", artifacts)
    write_json_durable(output_root / "pi0172_resource_admission.json", resource_record)
    write_json_durable(output_root / "pi0172_validation_e_admission.json", record)
    append_jsonl_durable(output_root / "execution_ledger.jsonl", {"schema": "CORELAB_PI0172_EXECUTION_LEDGER_V1", "sequence": 1, "event": "PRE_MODEL_ADMISSION", "phase": "ADMISSION", "timestamp_utc": now_utc(), "execution_head": checkout["head"], "selected_gpu_uuid": resource_record["selected_gpu_uuid"], "model_imported": False, "holdout_f_read": False})
    print(json.dumps(record, indent=2, sort_keys=True))
    print("CORELAB_PI0172_RTX4080_VALIDATION_E_ADMISSION_RECORDED")
    return 0


def payload_admission(config: Mapping[str, Any], output_root: Path, checkout: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    binding_path = output_root / "pi0172_validation_e_payload_binding.json"
    if binding_path.exists():
        binding = load_json(binding_path)
        require(binding.get("pi_gate") == PI_GATE and binding.get("execution_head") == checkout["head"] and binding.get("execution_branch") == BRANCH, "existing payload binding authority drift")
        return binding, binding_path
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module imported before Validation-E payload admission")
    admission = _admission_module()
    freeze = load_json(FREEZE_PATH)
    admission._validate_freeze(freeze, expected_code_commit=FROZEN_EVALUATION_COMMIT)
    source_root = regular_dir(Path(str(config["source_root"])), "Validation-E source root")
    snapshot_paths = admission._find_snapshot_paths(source_root)
    require(len(snapshot_paths) == 6, "Validation-E snapshot descriptor count is not six")
    records = [admission._read_snapshot_descriptor(path, WINDOW_IDS[index]) for index, path in enumerate(snapshot_paths)]
    origins = [record["frame_origin"] for record in records]
    canonical_origin = list(origins[0]["anchor_world_m"])
    spread = max(max(abs(float(origin["anchor_world_m"][axis]) - canonical_origin[axis]) for axis in range(3)) for origin in origins)
    require(spread <= admission.FRAME_ORIGIN_SPREAD_TOLERANCE_M, f"Validation-E frame origin spread drift: {spread}")
    record_without_payload = [{key: value for key, value in record.items() if key != "payload"} for record in records]
    ledger = {"schema": "CORELAB_PI0172_VALIDATION_E_ACCESS_LEDGER_V1", "status": "PAYLOAD_READ_AND_BOUND_BEFORE_MODEL_IMPORT", "pi_gate": PI_GATE, "issue_11_comment": ISSUE_COMMENT, "pr_pointer": PR_POINTER, "target_id": TARGET_ID, "source_root": str(source_root), "snapshot_ids": list(WINDOW_IDS), "first_payload_read_at_utc": now_utc(), "finished_at_utc": now_utc(), "descriptor_reads": 6, "frame_reads": 12, "holdout_f_read": False, "simulator_collection": False, "new_data_generated": False, "records": record_without_payload, "frame_origin": {"anchor_world_m": canonical_origin, "max_cross_window_spread_m": spread, "target_recentered": False, "measured_start_recentered": False}, "model_imported": False}
    write_json_durable(output_root / "validation_e_access_ledger.json", ledger)
    binding = {"schema": "CORELAB_PI0172_VALIDATION_E_PAYLOAD_ADMISSION_V1", "status": "PAYLOAD_READ_AND_BOUND_BEFORE_MODEL_IMPORT", "pi_gate": PI_GATE, "issue_11_comment": ISSUE_COMMENT, "pr_pointer": PR_POINTER, "source_head": SOURCE_HEAD, "execution_head": checkout["head"], "execution_branch": checkout["branch"], "checkpoint_arms": [dict(item) for item in ARM_SPECS], "stats": {"path": str(Path(str(config["stats"])).resolve()), "sha256": FULL24_STATS_SHA, "payload_sha256": FULL24_STATS_PAYLOAD_SHA}, "validation_input": {"target_id": TARGET_ID, "source_root": str(source_root), "snapshot_ids": list(WINDOW_IDS), "snapshot_records": record_without_payload, "frame_origin": ledger["frame_origin"], "access_ledger": str((output_root / "validation_e_access_ledger.json").resolve()), "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"), "target_xyz_in_model_input": False, "holdout_f_read": False}, "model_imported": False, "model_construction": False, "model_forward": False, "training": False, "backward": False, "optimizer_steps": 0, "checkpoint_write": False, "h200_submissions": 0, "output_root": str(output_root)}
    write_json_durable(binding_path, binding)
    write_json_durable(output_root / "pi0172_validation_e_payload_admission_result.json", {"schema": "CORELAB_PI0172_VALIDATION_E_PAYLOAD_ADMISSION_RESULT_V1", "status": "PASS_BEFORE_MODEL_IMPORT", "token": "CORELAB_PI0172_VALIDATION_E_PAYLOAD_BOUND", "binding_path": str(binding_path), "binding_sha256": sha256_file(binding_path), "access_ledger_sha256": sha256_file(output_root / "validation_e_access_ledger.json"), "snapshot_count": 6, "descriptor_reads": 6, "frame_reads": 12, "model_imported": False, "holdout_f_read": False})
    return binding, binding_path


def _safe_label(label: str) -> str:
    return label.replace("(", "_").replace(")", "").replace("@", "_at_")


def _raw_path(label: str, index: int) -> Path:
    return Path("raw") / _safe_label(label) / f"window-{index:02d}.npy"


def _derived_path(label: str, index: int) -> Path:
    return Path("derived") / _safe_label(label) / f"window-{index:02d}.json"


def _decoded_path(label: str, index: int) -> Path:
    return Path("decoded") / _safe_label(label) / f"window-{index:02d}.npy"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, Mapping), f"JSONL record is not an object: {path}")
            records.append(dict(value))
    return records


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str), f"{label} path is not a string")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe {label} path: {value}")
    path = root / relative
    require(path.resolve().parent == path.parent.resolve() or root.resolve() in path.resolve().parents, f"{label} path escapes output root: {value}")
    return path


def _rss_bytes() -> int:
    try:
        value = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in value.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1))


def _torch_memory() -> dict[str, Any]:
    module = sys.modules.get("torch")
    if module is None:
        return {"torch_max_memory_allocated": None, "torch_max_memory_reserved": None}
    try:
        device = module.device("cuda:0")
        return {"torch_max_memory_allocated": int(module.cuda.max_memory_allocated(device)), "torch_max_memory_reserved": int(module.cuda.max_memory_reserved(device))}
    except Exception:
        return {"torch_max_memory_allocated": None, "torch_max_memory_reserved": None}


class ExecutionLedger:
    """Durable append-only phase ledger with resume-safe sequence numbers."""

    def __init__(self, path: Path, selected_uuid: str):
        self.path = path
        self.selected_uuid = selected_uuid
        self.active: dict[int, tuple[str, float]] = {}
        existing = _jsonl(path)
        self.sequence = max([int(item.get("sequence", 0)) for item in existing] or [0])
        self.emit("LEDGER_OPEN", phase="ADMISSION", selected_gpu_uuid=selected_uuid, resumed=bool(existing))

    def emit(self, event: str, *, phase: str, **details: Any) -> int:
        self.sequence += 1
        record = {"schema": "CORELAB_PI0172_EXECUTION_LEDGER_V1", "sequence": self.sequence, "event": event, "phase": phase, "timestamp_utc": now_utc(), "monotonic_s": time.monotonic(), "rss_bytes": _rss_bytes(), "selected_gpu_uuid": self.selected_uuid, "model_imported": any(name in sys.modules for name in ("torch", "numpy")), "holdout_f_read": False, **jsonable(details)}
        append_jsonl_durable(self.path, record)
        return self.sequence

    def begin(self, event: str, *, phase: str, **details: Any) -> int:
        token = self.emit(event, phase=phase, status="BEGIN", **details)
        self.active[token] = (event, time.monotonic())
        return token

    def end(self, token: int, *, status: str = "END", **details: Any) -> int:
        event, started = self.active.pop(token)
        return self.emit(event, phase=str(details.pop("phase", event)), status=status, elapsed_s=time.monotonic() - started, torch_memory=_torch_memory(), **details)

    def count_begin(self, event: str) -> int:
        return sum(1 for item in _jsonl(self.path) if item.get("event") == event and item.get("status") == "BEGIN")


def _set_runtime_environment(admission_record: Mapping[str, Any]) -> None:
    cache = admission_record["fresh_cache"]
    os.environ.update({str(key): str(value) for key, value in cache.get("environment", {}).items()})
    for key, value in cache["paths"].items():
        require(Path(str(value)).is_dir(), f"fresh cache directory is unavailable: {key}")
        os.environ[str(key).upper()] = str(value)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "GROOT_HF_LOCAL_FIRST": "1", "GROOT_PATCH_MISTRAL": "1", "CUDA_VISIBLE_DEVICES": str(admission_record["resource_admission"]["selected_gpu_uuid"])})


def _validate_raw_records(output_root: Path, records: Sequence[Mapping[str, Any]], np: Any) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        checkpoint = record.get("checkpoint")
        require(isinstance(checkpoint, Mapping), "raw record checkpoint identity is missing")
        label = str(checkpoint.get("label"))
        require(label in ARM_BY_LABEL, f"raw record has unknown checkpoint: {label}")
        window_id = str(record.get("window_id"))
        require(window_id in WINDOW_IDS, f"raw record has unknown window: {window_id}")
        key = (label, window_id)
        require(key not in index, f"duplicate raw cell: {label}/{window_id}")
        path = regular_file(_safe_relative(output_root, record.get("path"), "raw"), "raw cell")
        require(sha256_file(path) == record.get("file_sha256"), f"raw file SHA drift: {label}/{window_id}")
        raw = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.float32)
        require(tuple(raw.shape) == RAW_SHAPE and bool(np.isfinite(raw).all()), f"raw shape/finite drift: {label}/{window_id}")
        require(sha256_bytes(raw.tobytes()) == record.get("array_sha256"), f"raw array SHA drift: {label}/{window_id}")
        require(record.get("action_head_calls_observed") == 4 and record.get("decoded") is False, f"raw execution contract drift: {label}/{window_id}")
        require(record.get("tree_sha256") == ARM_BY_LABEL[label]["tree_sha256"], f"raw checkpoint tree identity drift: {label}")
        index[key] = {**dict(record), "raw": raw, "path_abs": str(path)}
    return index


def _validate_derived_records(output_root: Path, records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        label = str(record.get("checkpoint"))
        window_id = str(record.get("window_id"))
        key = (label, window_id)
        require(label in ARM_BY_LABEL and window_id in WINDOW_IDS and key not in index, f"derived identity drift: {label}/{window_id}")
        path = regular_file(_safe_relative(output_root, record.get("derived_path"), "derived"), "derived row")
        require(sha256_file(path) == record.get("derived_file_sha256"), f"derived file SHA drift: {label}/{window_id}")
        row = load_json(path)
        require(isinstance(row, Mapping) and row.get("window_id") == window_id and row.get("checkpoint", {}).get("label") == label, f"derived row content drift: {label}/{window_id}")
        index[key] = {**dict(record), "row": row, "path_abs": str(path)}
    return index


def _write_raw(output_root: Path, raw_manifest: Path, arm: Mapping[str, Any], window: Mapping[str, Any], raw_np: Any, noise_sha: str, ledger: ExecutionLedger, np: Any) -> dict[str, Any]:
    label = str(arm["label"])
    index = WINDOW_IDS.index(str(window["window_id"]))
    require(tuple(raw_np.shape) == RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output shape/finite drift: {label}/{window['window_id']}")
    raw_path = output_root / _raw_path(label, index)
    require(not raw_path.exists(), f"raw cell already exists and was not admitted for reuse: {label}/{window['window_id']}")
    raw_np = np.ascontiguousarray(raw_np, dtype=np.float32)
    from_numpy_path = raw_path
    temporary = from_numpy_path.with_name(f".{from_numpy_path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.save(handle, raw_np, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, from_numpy_path)
    file_sha = sha256_file(from_numpy_path)
    array_sha = sha256_bytes(raw_np.tobytes())
    record = {"schema": "CORELAB_PI0172_RAW_PREDICTION_V1", "checkpoint": {"route": arm["route"], "seed": arm["seed"], "tier": arm["tier"], "label": label, "tree_sha256": arm["tree_sha256"]}, "tree_sha256": arm["tree_sha256"], "window_id": window["window_id"], "path": str(from_numpy_path.relative_to(output_root)), "file_sha256": file_sha, "array_sha256": array_sha, "shape": list(raw_np.shape), "dtype": str(raw_np.dtype), "noise_sha256": noise_sha, "timestep_ids": list(TIMESTEP_IDS), "action_head_calls_observed": 4, "decoded": False}
    # PI0172_RAW_BEFORE_DECODE
    append_jsonl_durable(raw_manifest, record)
    ledger.emit("RAW_SEALED", phase="RAW_SEAL", arm=label, window_id=window["window_id"], path=record["path"], file_sha256=file_sha, array_sha256=array_sha, shape=list(raw_np.shape), finite=True)
    return {**record, "raw": raw_np, "path_abs": str(from_numpy_path)}


def _decode_one(output_root: Path, derived_manifest: Path, arm: Mapping[str, Any], window: Mapping[str, Any], raw_record: Mapping[str, Any], runtime: Any, evaluator: Any, np: Any, config: Mapping[str, Any], runtime_sha: str, noise_sha: str, ledger: ExecutionLedger) -> dict[str, Any]:
    label = str(arm["label"])
    index = WINDOW_IDS.index(str(window["window_id"]))
    physical, details = runtime._decode_v2(raw_record["raw"], window["state_v1"], Path(str(config["stats"])).resolve(), window["source_timestamps"], float(window["observation_timestamp"]), window["window_id"] + ":" + label)
    physical_np = np.asarray(physical, dtype=np.float64)
    require(tuple(physical_np.shape) == PHYSICAL_SHAPE and bool(np.isfinite(physical_np).all()), f"decoded physical shape/finite drift: {label}/{window['window_id']}")
    decoded_path = output_root / _decoded_path(label, index)
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = decoded_path.with_name(f".{decoded_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, physical_np, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, decoded_path)
    checkpoint = {"route": arm["route"], "seed": arm["seed"], "tier": arm["tier"], "label": label, "tree_sha256": arm["tree_sha256"]}
    row = evaluator._metric_row(runtime, raw_record["raw"], physical_np, details, window, checkpoint, runtime_sha, str(raw_record["path"]), str(raw_record["file_sha256"]))
    row.update({"route": arm["route"], "seed": arm["seed"], "tier": arm["tier"], "noise_sha256": noise_sha, "sampler": {"name": "direct-source-mapped-action-head-E4", "t_values": list(T_VALUES), "timestep_ids": list(TIMESTEP_IDS), "dt": SAMPLER_DT, "carried_state": True, "reset_between_stages": False, "official_public_sampler_called": False, "get_action_calls": 0}, "stats": {"sha256": FULL24_STATS_SHA, "payload_sha256": FULL24_STATS_PAYLOAD_SHA, "route": "FULL24_RUNTIME_V2"}, "decoded_path": str(decoded_path.relative_to(output_root)), "decoded_file_sha256": sha256_file(decoded_path)})
    row_path = output_root / _derived_path(label, index)
    write_json_durable(row_path, row)
    record = {"schema": "CORELAB_PI0172_DERIVED_EVIDENCE_REFERENCE_V1", "checkpoint": label, "window_id": window["window_id"], "raw_path": raw_record["path"], "raw_file_sha256": raw_record["file_sha256"], "derived_path": str(row_path.relative_to(output_root)), "derived_file_sha256": sha256_file(row_path), "decoded": True, "decoded_array_sha256": row["decoded_action"]["array_sha256"]}
    append_jsonl_durable(derived_manifest, record)
    ledger.emit("DECODE_DERIVED", phase="CPU_POSTHOC", arm=label, window_id=window["window_id"], derived_path=record["derived_path"], derived_file_sha256=record["derived_file_sha256"], decoded_array_sha256=record["decoded_array_sha256"])
    return {**record, "row": row, "path_abs": str(row_path)}


def _expected_h200_cells() -> dict[str, dict[str, str]]:
    manifest = load_json(MANIFEST_PATH)
    return {f"stage22_expert_{TARGET_ID}:{cell['window']}": {"file_sha256": cell["file_sha256"], "array_sha256": cell["array_sha256"]} for cell in manifest["h200_portability_anchor"]["cells"]}


def compare_h200(config: Mapping[str, Any], output_root: Path, windows: Sequence[Mapping[str, Any]], rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]], runtime: Any, evaluator: Any, np: Any) -> dict[str, Any]:
    root_value = config.get("h200_raw_root")
    manifest_value = config.get("h200_raw_manifest")
    if not isinstance(root_value, str) or not isinstance(manifest_value, str):
        return {"schema": "CORELAB_PI0172_H200_RTX_PORTABILITY_APPENDIX_V1", "status": "BLOCKED_H200_SOURCE", "reason": "h200_raw_root or h200_raw_manifest is not late-bound", "h200_submission": False, "scientific_winner_selected": False}
    root = regular_dir(Path(root_value), "H200 allocation-8 raw root")
    manifest_path = regular_file(Path(manifest_value), "H200 allocation-8 raw manifest")
    require(sha256_file(manifest_path) == H200_RAW_MANIFEST_SHA, "H200 allocation-8 raw manifest SHA drift")
    records = _jsonl(manifest_path)
    require(len(records) == 6, "H200 allocation-8 raw manifest is not six cells")
    expected = _expected_h200_cells()
    h200: dict[str, dict[str, Any]] = {}
    for record in records:
        window_id = str(record.get("window_id"))
        require(window_id in expected and record.get("checkpoint", {}).get("label") == "Legacy(seed2222)@2000", f"unexpected H200 portability cell: {window_id}")
        path = regular_file(_safe_relative(root, record.get("path"), "H200 raw"), "H200 raw cell")
        file_sha = sha256_file(path)
        require(file_sha == expected[window_id]["file_sha256"] and file_sha == record.get("file_sha256"), f"H200 raw file SHA drift: {window_id}")
        raw = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.float32)
        require(tuple(raw.shape) == RAW_SHAPE and bool(np.isfinite(raw).all()), f"H200 raw shape/finite drift: {window_id}")
        array_sha = sha256_bytes(raw.tobytes())
        require(array_sha == expected[window_id]["array_sha256"] and array_sha == record.get("array_sha256"), f"H200 raw array SHA drift: {window_id}")
        h200[window_id] = {"raw": raw, "file_sha256": file_sha, "array_sha256": array_sha, "path": str(path)}
    windows_by_id = {str(window["window_id"]): window for window in windows}
    cells = []
    for window_id in WINDOW_IDS:
        key = ("Legacy(seed2222)@2000", window_id)
        require(key in rows_by_key and window_id in h200 and window_id in windows_by_id, f"portability cell missing: {window_id}")
        rtx_record = rows_by_key[key]
        rtx_raw = np.ascontiguousarray(rtx_record["raw"], dtype=np.float32)
        h200_raw = h200[window_id]["raw"]
        diff = np.asarray(rtx_raw, dtype=np.float64) - np.asarray(h200_raw, dtype=np.float64)
        rtx_physical, _ = runtime._decode_v2(rtx_raw, windows_by_id[window_id]["state_v1"], Path(str(config["stats"])).resolve(), windows_by_id[window_id]["source_timestamps"], float(windows_by_id[window_id]["observation_timestamp"]), window_id + ":RTX4080")
        h200_physical, _ = runtime._decode_v2(h200_raw, windows_by_id[window_id]["state_v1"], Path(str(config["stats"])).resolve(), windows_by_id[window_id]["source_timestamps"], float(windows_by_id[window_id]["observation_timestamp"]), window_id + ":H200")
        physical_diff = np.asarray(rtx_physical, dtype=np.float64) - np.asarray(h200_physical, dtype=np.float64)
        def band(name: str, left: Any, right: Any) -> dict[str, Any]:
            delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            return {"name": name, "max_abs": float(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(np.square(delta)))), "exact": bool(np.array_equal(np.asarray(left), np.asarray(right)))}
        cells.append({"window_id": window_id, "raw": {"file_sha256_equal": bool(rtx_record["file_sha256"] == h200[window_id]["file_sha256"]), "array_sha256_equal": bool(rtx_record["array_sha256"] == h200[window_id]["array_sha256"]), "array_exact_equal": bool(np.array_equal(rtx_raw, h200_raw)), "max_abs": float(np.max(np.abs(diff))), "mean_abs": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(np.square(diff))))}, "decoded": {"bands": [band("overall", rtx_physical, h200_physical), band("xyz", np.asarray(rtx_physical)[:, :3], np.asarray(h200_physical)[:, :3]), band("rot6d", np.asarray(rtx_physical)[:, 3:9], np.asarray(h200_physical)[:, 3:9]), band("gripper", np.asarray(rtx_physical)[:, 9:], np.asarray(h200_physical)[:, 9:])], "max_abs": float(np.max(np.abs(physical_diff))), "mean_abs": float(np.mean(np.abs(physical_diff))), "rmse": float(np.sqrt(np.mean(np.square(physical_diff))))}})
    return {"schema": "CORELAB_PI0172_H200_RTX_PORTABILITY_APPENDIX_V1", "status": "DESCRIPTIVE_COMPLETE", "h200_allocation": 8, "h200_submission": False, "h200_raw_manifest_sha256": H200_RAW_MANIFEST_SHA, "rtx_cell_count": 6, "h200_cell_count": 6, "cells": cells, "raw_byte_identity_is_not_a_gate": True, "scientific_winner_selected": False}


def _noise_manifest(output_root: Path, evaluator: Any, torch: Any) -> tuple[list[Any], list[str]]:
    noise_bank, noise_hashes = evaluator._make_noise_bank(torch)
    write_if_absent(output_root / "sampler_noise_manifest.json", {"schema": "CORELAB_PI0172_CPU_FLOAT32_NOISE_BANK_V1", "seed": NOISE_SEED, "device": "cpu", "dtype": "float32", "shape": list(RAW_SHAPE), "window_ids": list(WINDOW_IDS), "noise_sha256_by_window": noise_hashes, "same_noise_across_all_six_checkpoint_arms": True})
    return noise_bank, noise_hashes


def execute(config_path: Path) -> int:
    config = load_json(config_path)
    require(isinstance(config, Mapping), "PI0172 config must be an object")
    output_root = Path(str(config["output_root"])).resolve()
    admission_path = regular_file(output_root / "pi0172_validation_e_admission.json", "PI0172 pre-model admission")
    admission_record = load_json(admission_path)
    require(admission_record.get("status") == "PASS_PRE_MODEL" and admission_record.get("token") == "CORELAB_PI0172_RTX4080_VALIDATION_E_ADMISSION_RECORDED", "PI0172 pre-model admission is not PASS")
    checkout = checkout_audit()
    require(checkout["head"] == admission_record.get("execution_head") and checkout["branch"] == BRANCH, "execution checkout changed after pre-model admission")
    require("numpy" not in sys.modules and "torch" not in sys.modules, "model-side module present at execution entry")
    selected_uuid = str(admission_record["resource_admission"]["selected_gpu_uuid"])
    _set_runtime_environment(admission_record)
    binding, _ = payload_admission(config, output_root, checkout)
    ledger = ExecutionLedger(output_root / "execution_ledger.jsonl", selected_uuid)
    current: dict[str, Any] = {"phase": "MODEL_RUNTIME_IMPORT", "arm": None, "window_id": None, "step": None}
    model_import_started = False
    try:
        model_import_started = True
        # PI0172_MODEL_IMPORT_BOUNDARY: artifact, payload, resource, and cache gates are complete.
        import numpy as np
        import torch
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        import stage45f_vanda_validation_e_gen1 as evaluator
        from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

        runtime_identity = {"schema": "CORELAB_PI0172_RTX4080_RUNTIME_IDENTITY_V1", "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version, "torch_version": str(torch.__version__), "torch_cuda_version": str(getattr(torch.version, "cuda", None)), "selected_gpu_uuid": selected_uuid, "selected_gpu_name": str(torch.cuda.get_device_name(torch.device("cuda:0"))), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": "bfloat16", "offline": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}, "h200_submissions": 0}
        write_if_absent(output_root / "pi0172_runtime_identity.json", runtime_identity)
        processor_token = ledger.begin("PROCESSOR_BUILD", phase="PROCESSOR_BUILD")
        processor, embodiment, _modality = stage26._build_runtime_processor(groot_source=Path(str(config["groot_source"])).resolve(), base_checkpoint=Path(str(config["base_model"])).resolve(), dataset_dir=Path(str(config["dataset"])).resolve(), modality_path=Path(str(config["modality"])).resolve(), cosmos_snapshot=Path(str(config["cosmos_root"])).resolve() / "snapshots" / COSMOS_REVISION)
        ledger.end(processor_token, phase="PROCESSOR_BUILD")
        windows_meta = evaluator.validate_snapshot_records({"validation_input": binding["validation_input"]})
        window_token = ledger.begin("WINDOW_PREPARATION", phase="WINDOW_PREPARATION", window_count=6)
        windows = evaluator._prepare_windows(processor, embodiment, windows_meta)
        require(len(windows) == 6 and [item["window_id"] for item in windows] == list(WINDOW_IDS), "prepared E window count/order drift")
        ledger.end(window_token, phase="WINDOW_PREPARATION", window_count=len(windows))
        noise_bank, noise_hashes = _noise_manifest(output_root, evaluator, torch)
        raw_manifest = output_root / "raw_predictions.jsonl"
        derived_manifest = output_root / "derived_rows.jsonl"
        raw_by_key = _validate_raw_records(output_root, _jsonl(raw_manifest), np)
        derived_by_key = _validate_derived_records(output_root, _jsonl(derived_manifest))
        runtime_sha = canonical_sha({"pre_model_admission": admission_record, "payload_binding": binding, "runtime": runtime_identity})
        checkpoint_paths = config["checkpoints"]
        for arm in ARM_SPECS:
            label = str(arm["label"])
            current.update({"phase": "ARM", "arm": label, "window_id": None, "step": None})
            missing = [index for index, window_id in enumerate(WINDOW_IDS) if (label, window_id) not in raw_by_key]
            model = None
            if missing:
                attempt_number = ledger.count_begin("MODEL_LOAD") + 1
                require(attempt_number <= MODEL_LOAD_ATTEMPTS_MAX, f"PI0172 model-load recovery ceiling exceeded: {attempt_number} > {MODEL_LOAD_ATTEMPTS_MAX}")
                restart_resource = resource_admission(config, expected_uuid=selected_uuid, purpose=f"model_load_{attempt_number}")
                ledger.emit("RESOURCE_ADMISSION", phase="MODEL_LOAD", arm=label, attempt=attempt_number, resource_admission=restart_resource)
                model_token = ledger.begin("MODEL_LOAD", phase="MODEL_LOAD", arm=label, checkpoint=label, checkpoint_path=str(Path(str(checkpoint_paths[label])).resolve()), attempt=attempt_number, resource_snapshots=restart_resource["snapshots"], free_vram_mib=[item["free_vram_mib"] for item in restart_resource["snapshots"]], checkpoint_timestamp_utc=now_utc())
                try:
                    model = runtime._load_torch_runtime()[1](Path(str(checkpoint_paths[label])).resolve(), "cuda:0")
                    model.eval()
                    ledger.end(model_token, phase="MODEL_LOAD", checkpoint=label, load_status="COMPLETED")
                except Exception as exc:
                    ledger.end(model_token, status="ERROR", phase="MODEL_LOAD", checkpoint=label, error=f"{type(exc).__name__}: {exc}")
                    if "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower():
                        raise PI0172Error("CORELAB_PI0172_RESOURCE_MODEL_MISMATCH_FOR_PI_REVIEW: model load failed after exact >=12 GiB admission; no recipe/device fallback is allowed") from exc
                    raise
                try:
                    for index in missing:
                        window = windows[index]
                        current.update({"phase": "CONDITIONING", "window_id": window["window_id"], "step": None})
                        conditioning_attempt = ledger.count_begin("CONDITIONING") + 1
                        require(conditioning_attempt <= CONDITIONING_ATTEMPTS_MAX, f"PI0172 conditioning recovery ceiling exceeded: {conditioning_attempt} > {CONDITIONING_ATTEMPTS_MAX}")
                        condition_token = ledger.begin("CONDITIONING", phase="CONDITIONING", arm=label, window_id=window["window_id"], attempt=conditioning_attempt)
                        try:
                            _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                            ledger.end(condition_token, phase="CONDITIONING", arm=label, window_id=window["window_id"], status_detail="COMPLETED")
                        except Exception as exc:
                            ledger.end(condition_token, status="ERROR", phase="CONDITIONING", arm=label, window_id=window["window_id"], error=f"{type(exc).__name__}: {exc}")
                            if "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower():
                                raise PI0172Error("CORELAB_PI0172_RESOURCE_MODEL_MISMATCH_FOR_PI_REVIEW: conditioning failed after exact admission") from exc
                            raise
                        current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                        with torch.inference_mode():
                            for step, timestep in enumerate(TIMESTEP_IDS):
                                current["phase"] = "ACTION_HEAD"
                                current["step"] = int(step)
                                action_attempt = ledger.count_begin("ACTION_HEAD") + 1
                                require(action_attempt <= ACTION_HEAD_ATTEMPTS_MAX, f"PI0172 action-head recovery ceiling exceeded: {action_attempt} > {ACTION_HEAD_ATTEMPTS_MAX}")
                                action_token = ledger.begin("ACTION_HEAD", phase="ACTION_HEAD", arm=label, window_id=window["window_id"], step=int(step), timestep_id=int(timestep), attempt=action_attempt)
                                try:
                                    velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                                    current_action = current_action + SAMPLER_DT * velocity
                                    ledger.end(action_token, phase="ACTION_HEAD", arm=label, window_id=window["window_id"], step=int(step), timestep_id=int(timestep), status_detail="COMPLETED")
                                except Exception as exc:
                                    ledger.end(action_token, status="ERROR", phase="ACTION_HEAD", arm=label, window_id=window["window_id"], step=int(step), timestep_id=int(timestep), error=f"{type(exc).__name__}: {exc}")
                                    if "out of memory" in str(exc).lower() or "cuda error" in str(exc).lower():
                                        raise PI0172Error("CORELAB_PI0172_RESOURCE_MODEL_MISMATCH_FOR_PI_REVIEW: action head failed after exact admission") from exc
                                    raise
                        raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                        raw_record = _write_raw(output_root, raw_manifest, arm, window, raw_np, noise_hashes[index], ledger, np)
                        raw_by_key[(label, window["window_id"])] = raw_record
                finally:
                    del model
                    model = None
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            for index, window in enumerate(windows):
                key = (label, window["window_id"])
                require(key in raw_by_key, f"raw cell missing after arm execution: {label}/{window['window_id']}")
                if key not in derived_by_key:
                    derived_by_key[key] = _decode_one(output_root, derived_manifest, arm, window, raw_by_key[key], runtime, evaluator, np, config, runtime_sha, noise_hashes[index], ledger)
            rows = [derived_by_key[(label, window_id)]["row"] for window_id in WINDOW_IDS]
            require(len(rows) == 6, f"arm row count drift: {label}")
            aggregate = aggregate_evaluation_rows(rows, expected_window_ids=list(WINDOW_IDS))
            aggregate.update({"checkpoint": label, "route": arm["route"], "seed": arm["seed"], "tier": arm["tier"], "checkpoint_tree_sha256": arm["tree_sha256"]})
            aggregate_path = output_root / f"aggregate_{_safe_label(label)}.json"
            write_if_absent(aggregate_path, aggregate)
            ledger.emit("ARM_AGGREGATE", phase="CPU_POSTHOC", arm=label, row_count=len(rows), aggregate_path=str(aggregate_path.relative_to(output_root)), aggregate_file_sha256=sha256_file(aggregate_path))
        require(len(raw_by_key) == 36, f"36/36 RTX raw matrix incomplete: {len(raw_by_key)}")
        require(len(derived_by_key) == 36, f"36/36 RTX derived matrix incomplete: {len(derived_by_key)}")
        rows_by_arm = {label: [derived_by_key[(label, window_id)]["row"] for window_id in WINDOW_IDS] for label in ARM_BY_LABEL}
        paired = evaluator.paired_summary(rows_by_arm, list(ARM_SPECS))
        paired_path = output_root / "pi0172_l1_minus_legacy_paired_summary.json"
        write_if_absent(paired_path, paired)
        aggregates = {label: load_json(output_root / f"aggregate_{_safe_label(label)}.json") for label in ARM_BY_LABEL}
        target_summary = {"schema": "CORELAB_PI0172_TARGET_PROGRESS_SAFETY_SUMMARY_V1", "arm_count": 6, "arms": [{"checkpoint": label, "target_metrics": aggregates[label].get("metrics", {}).get("target"), "safety_gate": aggregates[label].get("safety_gate_decision")} for label in ARM_BY_LABEL], "scientific_winner_selected": False}
        write_if_absent(output_root / "pi0172_target_progress_safety_summary.json", target_summary)
        rows_by_key = {(label, window_id): {**raw_by_key[(label, window_id)], "row": derived_by_key[(label, window_id)]["row"]} for label in ARM_BY_LABEL for window_id in WINDOW_IDS}
        portability = compare_h200(config, output_root, windows, rows_by_key, runtime, evaluator, np)
        write_if_absent(output_root / "pi0172_h200_rtx_portability_appendix.json", portability)
        ledger_counts = {"model_load_attempts": ledger.count_begin("MODEL_LOAD"), "conditioning_attempts": ledger.count_begin("CONDITIONING"), "action_head_attempts": ledger.count_begin("ACTION_HEAD"), "model_side_attempts": ledger.count_begin("CONDITIONING") + ledger.count_begin("ACTION_HEAD"), "model_loads_completed": sum(1 for item in _jsonl(ledger.path) if item.get("event") == "MODEL_LOAD" and item.get("status") == "END" and item.get("load_status") == "COMPLETED"), "conditioning_completed": sum(1 for item in _jsonl(ledger.path) if item.get("event") == "CONDITIONING" and item.get("status") == "END" and item.get("status_detail") == "COMPLETED"), "action_head_completed": sum(1 for item in _jsonl(ledger.path) if item.get("event") == "ACTION_HEAD" and item.get("status") == "END" and item.get("status_detail") == "COMPLETED"), "raw_count": len(raw_by_key), "derived_count": len(derived_by_key)}
        write_json_durable(output_root / "pi0172_raw_inventory.json", {"schema": "CORELAB_PI0172_RAW_INVENTORY_V1", "cell_count": len(raw_by_key), "cells": [{key: value for key, value in record.items() if key != "raw" and key != "path_abs"} for _, record in sorted(raw_by_key.items())]})
        write_json_durable(output_root / "pi0172_derived_inventory.json", {"schema": "CORELAB_PI0172_DERIVED_INVENTORY_V1", "cell_count": len(derived_by_key), "cells": [{key: value for key, value in record.items() if key != "row" and key != "path_abs"} for _, record in sorted(derived_by_key.items())]})
        status = "COMPLETED" if portability.get("status") == "DESCRIPTIVE_COMPLETE" else "PARTIAL"
        result = {"schema": "CORELAB_PI0172_RTX4080_VALIDATION_E_RESULT_V1", "status": status, "return_token": f"CORELAB_PI0172_RTX4080_VALIDATION_E_FOR_PI_REVIEW — {status}", "science_hold": "CORELAB_PI0172_SCIENCE_REVIEW_HOLD", "pi_gate": PI_GATE, "execution_head": checkout["head"], "branch": checkout["branch"], "gpu_uuid": selected_uuid, "checkpoint_arms": [dict(item) for item in ARM_SPECS], "window_ids": list(WINDOW_IDS), "qualification": {"legacy_seed2222_cells": 6, "objective_integrity": True}, "rtx_matrix": {"raw_cells": len(raw_by_key), "derived_cells": len(derived_by_key), "same_generation_reuse": True, "same_uuid": True}, "actual_counters": ledger_counts, "raw_inventory": "pi0172_raw_inventory.json", "derived_inventory": "pi0172_derived_inventory.json", "paired_summary": "pi0172_l1_minus_legacy_paired_summary.json", "target_progress_safety_summary": "pi0172_target_progress_safety_summary.json", "execution_ledger": "execution_ledger.jsonl", "portability_appendix": "pi0172_h200_rtx_portability_appendix.json", "portability_status": portability.get("status"), "training": False, "wbc": False, "simulator": False, "hardware": False, "holdout_f_read": False, "h200_submissions": 0, "scientific_winner_selected": False}
        write_json_durable(output_root / "PI0172_RESULT.json", result)
        ledger.emit("FINAL_SEAL", phase="FINAL_SEAL", status=status, result_path="PI0172_RESULT.json", result_file_sha256=sha256_file(output_root / "PI0172_RESULT.json"), raw_count=len(raw_by_key), derived_count=len(derived_by_key), science_hold="CORELAB_PI0172_SCIENCE_REVIEW_HOLD")
        print(result["return_token"])
        print("CORELAB_PI0172_SCIENCE_REVIEW_HOLD")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if status == "COMPLETED" else 2
    except Exception as exc:
        status = "PARTIAL" if model_import_started or ledger.count_begin("MODEL_LOAD") else "BLOCKED"
        token = f"CORELAB_PI0172_RTX4080_VALIDATION_E_FOR_PI_REVIEW — {status}"
        failure = {"schema": "CORELAB_PI0172_RTX4080_VALIDATION_E_FAILURE_V1", "status": status, "return_token": token, "pi_gate": PI_GATE, "execution_head": checkout.get("head"), "branch": checkout.get("branch"), "phase": current["phase"], "current": current, "error": f"{type(exc).__name__}: {exc}", "holdout_f_read": False, "h200_submissions": 0, "training": False, "science_hold": "CORELAB_PI0172_SCIENCE_REVIEW_HOLD" if status == "PARTIAL" else None, "timestamp_utc": now_utc()}
        failure_path = output_root / "PI0172_FAILURE.json"
        if not failure_path.exists():
            write_json_durable(failure_path, failure)
        print(token, file=sys.stderr)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


def plan() -> dict[str, Any]:
    return {"schema": "CORELAB_PI0172_RTX4080_VALIDATION_E_PLAN_V1", "status": "READY_NOT_EXECUTED", "pi_gate": PI_GATE, "issue_11_comment": ISSUE_COMMENT, "pr_pointer": PR_POINTER, "source_head": SOURCE_HEAD, "branch": BRANCH, "checkpoint_arms": [dict(item) for item in ARM_SPECS], "window_ids": list(WINDOW_IDS), "qualification": {"checkpoint": "Legacy(seed2222)@2000", "windows": 6, "conditioning": 6, "action_head": 24, "model_side_total": 30, "raw_outputs": 6}, "full_generation_nominal": {"model_loads": 6, "conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36, "derived_outputs": 36}, "recovery_ceiling": {"model_load_attempts": MODEL_LOAD_ATTEMPTS_MAX, "conditioning_attempts": CONDITIONING_ATTEMPTS_MAX, "action_head_attempts": ACTION_HEAD_ATTEMPTS_MAX, "model_side_attempts": MODEL_SIDE_ATTEMPTS_MAX}, "resource": {"gpu_uuid_max": 1, "free_vram_mib_min": MIN_FREE_VRAM_MIB, "occupancy_snapshots_before_each_model_load": 2, "same_uuid_for_all_cells": True}, "h200_portability": {"allocation": 8, "cells": 6, "descriptive_only": True, "submission": False}, "forbidden": {"remaining_30_rtx_cells": 0, "h200_submissions": 0, "training": 0, "wbc": 0, "simulator": 0, "hardware": 0, "holdout_f_read": 0, "new_probe": 0, "new_smoke_query": 0, "merge": 0}, "review_stop": "CORELAB_PI0172_SCIENCE_REVIEW_HOLD"}


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
        status = "BLOCKED"
        token = f"CORELAB_PI0172_RTX4080_VALIDATION_E_FOR_PI_REVIEW — {status}"
        print(token, file=sys.stderr)
        print(json.dumps({"status": status, "token": token, "error": f"{type(exc).__name__}: {exc}", "pi_gate": PI_GATE, "h200_submissions": 0, "holdout_f_read": False}, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
